#!/usr/bin/env python3
"""Read-only audit of the group objectness/no-object logit and its inference use.

Three things happen here, and nothing else:

1. a minimal gradient check on the *same* 21st logit that `group_instance_loss`
   supervises, with one Hungarian-matched and one unmatched query, recording the
   direction training pushes that logit;
2. the CE-consistent score `P(thing) = sum_{c<20} softmax(21 logits)[c]` versus
   the legacy `sigmoid(raw_noobject_logit)` score, computed on the same
   checkpoints without touching them;
3. a re-evaluation of G0/G1 at step 3000 and step 6000 on the same 8 unseen
   windows and the same recorded training windows, under both score
   conventions, with the 0.5 / 0.5 / 50 thresholds unchanged.

No training, no checkpoint writes, no threshold tuning, no decoder swap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.models.group_locusgs import group_instance_loss  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from tokengs.models.ssst_contracts import NO_OBJECT_CLASS  # noqa: E402
from tokengs.models.ssst_loss import hungarian_match  # noqa: E402
from scripts.group_locusgs_eval import (  # noqa: E402
    MASK_THRESHOLD,
    MIN_PRED_PIXELS,
    OBJECTNESS_THRESHOLD,
    build_val_entries,
    forward_group,
)
from scripts.object_locusgs_eval import gt_instances, instance_metrics  # noqa: E402
from scripts.eval_group_locusgs import (  # noqa: E402
    build_train_entries,
    load_model,
    load_run,
)


# --------------------------------------------------------------------------- #
# 1. minimal gradient check
# --------------------------------------------------------------------------- #
def minimal_gradient_check(device="cpu") -> dict:
    """One matched + one unmatched query: which way does training move z_20?"""
    torch.manual_seed(0)
    queries, num_classes = 100, 20
    class_logits = (torch.randn(1, queries, num_classes + 1, device=device) * 0.10)
    mask_logits = torch.randn(1, queries, 1, 6, 6, device=device) * 0.10
    gt_class = torch.tensor([5], device=device)
    gt_mask = (torch.rand(1, 1, 6, 6, device=device) > 0.5).float()
    rows, cols = hungarian_match(class_logits[0], mask_logits[0], gt_class, gt_mask)
    matched_row = int(rows[0])
    unmatched_row = int(next(i for i in range(queries) if i != matched_row))

    class_logits = class_logits.detach().clone().requires_grad_(True)
    out = group_instance_loss(class_logits, mask_logits, [gt_class], [gt_mask])
    out["loss"].backward()
    grad = class_logits.grad[0, :, NO_OBJECT_CLASS]
    raw0 = class_logits.detach()[0, :, NO_OBJECT_CLASS]

    # three plain gradient-descent steps on the same objective (toy, no data)
    tuned = class_logits.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([tuned], lr=1.0)
    for _ in range(3):
        optimizer.zero_grad()
        loss = group_instance_loss(tuned, mask_logits, [gt_class], [gt_mask])["loss"]
        loss.backward()
        optimizer.step()
    moved = tuned.detach()[0, :, NO_OBJECT_CLASS]
    return {
        "matched_row": matched_row,
        "unmatched_row": unmatched_row,
        "matched_query": {
            "gradient_of_no_object_logit": float(grad[matched_row]),
            "training_wants": "decrease" if float(grad[matched_row]) > 0 else "increase",
            "logit_before": float(raw0[matched_row]),
            "logit_after_3_sgd_steps": float(moved[matched_row]),
        },
        "unmatched_query": {
            "gradient_of_no_object_logit": float(grad[unmatched_row]),
            "training_wants": "decrease" if float(grad[unmatched_row]) > 0 else "increase",
            "logit_before": float(raw0[unmatched_row]),
            "logit_after_3_sgd_steps": float(moved[unmatched_row]),
            "cross_entropy_target": NO_OBJECT_CLASS,
        },
        "conclusion": (
            "the 21st logit IS the no-object logit: CE pushes it DOWN for matched "
            "queries and UP for unmatched queries, so sigmoid(z20) is P(no-object), "
            "not objectness; the CE-consistent object score is "
            "P(thing) = sum_{c<20} softmax(21 logits)[c] = 1 - softmax(...)[20]"
        ),
    }


# --------------------------------------------------------------------------- #
# 2. two score conventions, identical gates
# --------------------------------------------------------------------------- #
def score_table(forward):
    raw = forward["group"]["objectness"][0].float()
    class_logits = forward["group"]["class_logits"][0].float()
    joint = torch.cat([class_logits, raw.unsqueeze(-1)], dim=-1)
    legacy = torch.sigmoid(raw)
    thing = 1.0 - torch.softmax(joint, dim=-1)[..., NO_OBJECT_CLASS]
    return {"raw": raw, "legacy": legacy, "thing": thing}


def predictions(forward, view, *, score_mode: str, scores=None):
    """Same gate order as the frozen reader, only the score definition changes."""
    table = score_table(forward) if scores is None else scores
    scores = table[score_mode]
    mass = forward["masks"]["group_mass"][0, view].float()
    class_logits = forward["group"]["class_logits"][0].float()
    out = []
    for group_index in range(mass.shape[0]):
        score = float(scores[group_index])
        if score < OBJECTNESS_THRESHOLD:
            continue
        mask = (mass[group_index] > MASK_THRESHOLD).cpu().numpy()
        area = int(mask.sum())
        if area < MIN_PRED_PIXELS:
            continue
        out.append({
            "group": group_index,
            "class": int(class_logits[group_index].argmax()),
            "mask": mask,
            "area": area,
            "score": score,
            "raw": float(forward["group"]["objectness"][0, group_index]),
            "legacy": float(table["legacy"][group_index]),
            "thing": float(table["thing"][group_index]),
        })
    return out


def query_instance_iou_matrix(forward, view, semantic_gt, instance_gt):
    """[Q, n_gt] IoU of every group mask (mask>0.5) against every visible GT instance."""
    mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
    instances = list(gt_instances(semantic_gt, instance_gt, view).values())
    matrix = np.zeros((mass.shape[0], len(instances)), dtype=np.float64)
    for group_index in range(mass.shape[0]):
        mask = mass[group_index] > MASK_THRESHOLD
        for index, truth in enumerate(instances):
            union = int((mask | truth).sum())
            if union:
                matrix[group_index, index] = int((mask & truth).sum()) / union
    return matrix


def query_best_iou(forward, view, semantic_gt, instance_gt, *, min_area=MIN_PRED_PIXELS):
    """GT-assisted per-query best IoU (mask>0.5, area>=min_area, no score gate)."""
    mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
    matrix = query_instance_iou_matrix(forward, view, semantic_gt, instance_gt)
    areas = (mass > MASK_THRESHOLD).reshape(mass.shape[0], -1).sum(axis=1)
    masked = matrix.copy()
    if masked.size:
        masked[areas < min_area] = 0.0
    per_query = masked.max(axis=1) if masked.size else np.zeros(mass.shape[0])
    return per_query, areas, matrix, masked


def evaluate(forward, semantic_gt, instance_gt, view):
    raw_instances = gt_instances(semantic_gt, instance_gt, view)
    result = {"gt_visible": len(raw_instances)}
    table = score_table(forward)
    for name, mode in (("legacy_sigmoid", "legacy"), ("ce_thing_prob", "thing")):
        preds = predictions(forward, view, score_mode=mode, scores=table)
        metrics = instance_metrics(preds, raw_instances)
        result[name] = {
            "n_pass": len(preds),
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            "ap50": metrics["ap50"], "n_gt": metrics["n_gt"],
            "buckets": metrics["buckets"],
        }
    best_iou, best_area, matrix, masked = query_best_iou(
        forward, view, semantic_gt, instance_gt
    )
    scores = table
    matched = best_iou >= 0.5
    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return None
        return {"mean": float(values.mean()), "median": float(np.median(values)),
                "min": float(values.min()), "max": float(values.max())}
    result["queries"] = {
        "n_matched_iou50": int(matched.sum()),
        "legacy_score_matched": stats(scores["legacy"].cpu().numpy()[matched]),
        "legacy_score_unmatched": stats(scores["legacy"].cpu().numpy()[~matched]),
        "thing_score_matched": stats(scores["thing"].cpu().numpy()[matched]),
        "thing_score_unmatched": stats(scores["thing"].cpu().numpy()[~matched]),
        "raw_noobject_matched": stats(scores["raw"].cpu().numpy()[matched]),
        "raw_noobject_unmatched": stats(scores["raw"].cpu().numpy()[~matched]),
    }
    result["best_any_group_iou_no_score_gate"] = {
        # GT-side ceiling: for every visible GT instance, the best IoU over the
        # 100 group masks (no score gate), mask>0.5 and area>=50.
        "per_gt_mean_best_iou": float(masked.max(axis=0).mean()) if masked.size else 0.0,
        "per_gt_recall50": float((masked.max(axis=0) >= 0.5).mean()) if masked.size else 0.0,
        "per_query_mean_best_iou": float(best_iou.mean()) if best_iou.size else 0.0,
        "n_gt_visible": len(raw_instances),
    }
    result["best_any_group_iou_mask_only"] = {
        "per_gt_mean_best_iou": float(matrix.max(axis=0).mean()) if matrix.size else 0.0,
        "per_gt_recall50": float((matrix.max(axis=0) >= 0.5).mean()) if matrix.size else 0.0,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--runs", nargs="*", default=[
        "g0_step3000=workspace_group_locusgs/arm_g0/ckpt_step3000",
        "g0_step6000=workspace_group_locusgs/arm_g0/ckpt_step6000",
        "g1_step3000=workspace_group_locusgs/arm_g1/ckpt_step3000",
        "g1_step6000=workspace_group_locusgs/arm_g1/ckpt_step6000",
    ])
    parser.add_argument("--out", default="workspace_group_locusgs/audit_scores.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-windows", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    report: dict = {
        "scope": "read-only audit; no training, no checkpoint writes, thresholds unchanged",
        "thresholds": {"objectness": OBJECTNESS_THRESHOLD, "mask": MASK_THRESHOLD,
                       "min_area": MIN_PRED_PIXELS},
        "minimal_gradient_check": minimal_gradient_check(),
        "runs": {},
    }

    for spec in args.runs:
        name, directory = load_run(spec)
        opt, model, arm, step = load_model(directory, args.preset, args.seed, device)
        entries = build_val_entries(opt, split, device)
        train_entries = build_train_entries(opt, split, plan, device, args.train_windows)
        run = {"checkpoint": str(directory), "arm": arm, "step": step,
               "unseen": {}, "training": {}}
        for kind, selection in (("unseen", entries), ("training", train_entries)):
            rows = {}
            for entry in selection:
                with torch.no_grad():
                    forward = forward_group(model, entry["batch"], opt)
                    semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
                    instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
                views = {}
                for view in range(semantic_gt.shape[0]):
                    views[str(view)] = evaluate(forward, semantic_gt, instance_gt, view)
                rows[entry["scene"]] = views
                print(f"[audit] {name} {kind} {entry['scene']}", flush=True)
            run[kind] = rows
        report["runs"][name] = run

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    summarise(report)
    print(f"[audit] wrote {out}")
    return 0


def summarise(report: dict) -> None:
    print("\n[audit] gradient check:")
    grad = report["minimal_gradient_check"]
    for key in ("matched_query", "unmatched_query"):
        info = grad[key]
        print(f"  {key:16s} z20 gradient {info['gradient_of_no_object_logit']:+.4f} "
              f"-> training wants {info['training_wants']:8s} "
              f"({info['logit_before']:+.4f} -> {info['logit_after_3_sgd_steps']:+.4f})")
    for name, run in report["runs"].items():
        for kind in ("unseen", "training"):
            legacy = {"n": 0, "tp": 0, "fp": 0, "fn": 0, "ap": [], "gt": 0}
            thing = {"n": 0, "tp": 0, "fp": 0, "fn": 0, "ap": [], "gt": 0}
            best = []
            best_mask_only = []
            for scene, views in run[kind].items():
                del scene
                for view, metrics in views.items():
                    if int(view) < 2:
                        continue
                    legacy["n"] += metrics["legacy_sigmoid"]["n_pass"]
                    legacy["tp"] += metrics["legacy_sigmoid"]["tp"]
                    legacy["fp"] += metrics["legacy_sigmoid"]["fp"]
                    legacy["fn"] += metrics["legacy_sigmoid"]["fn"]
                    legacy["ap"].append(metrics["legacy_sigmoid"]["ap50"])
                    legacy["gt"] += metrics["legacy_sigmoid"]["n_gt"]
                    thing["n"] += metrics["ce_thing_prob"]["n_pass"]
                    thing["tp"] += metrics["ce_thing_prob"]["tp"]
                    thing["fp"] += metrics["ce_thing_prob"]["fp"]
                    thing["fn"] += metrics["ce_thing_prob"]["fn"]
                    thing["ap"].append(metrics["ce_thing_prob"]["ap50"])
                    thing["gt"] += metrics["ce_thing_prob"]["n_gt"]
                    best.append(
                        metrics["best_any_group_iou_no_score_gate"]["per_gt_mean_best_iou"]
                    )
                    best_mask_only.append(
                        metrics["best_any_group_iou_mask_only"]["per_gt_mean_best_iou"]
                    )
            print(f"\n[audit] {name} {kind} (novel views): gt {legacy['gt']}")
            for tag, values in (("legacy sigmoid(z20)", legacy), ("CE P(thing)", thing)):
                print(f"   {tag:20s} pass {values['n']:4d} TP {values['tp']:3d} FP {values['fp']:3d} "
                      f"FN {values['fn']:3d} AP50 {np.mean(values['ap']):.4f}")
            print(f"   best-any-group IoU per GT (no score gate): area>=50 {np.mean(best):.3f} "
                  f"| mask-only {np.mean(best_mask_only):.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
