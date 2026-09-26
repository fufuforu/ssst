#!/usr/bin/env python3
"""Read-only TP-ranking diagnostic for the group GT-free read-out.

For each of the 8 fixed unseen windows and the novel views 2/3, the frozen
GT-free reader (P(thing) >= 0.5, thing class, mask > 0.5, area >= 50 px) yields
the view's candidate set.  Candidates are ordered by P(thing) descending and the
**existing** AP50 greedy matching (IoU >= 0.5, one GT per candidate) decides which
are TP.  We report each TP's 1-based within-view rank, the median TP rank, the
number of TPs at rank <= 1 / <= 3 / <= 5 and the candidate totals.  The greedy
loop is a faithful copy of ``object_locusgs_eval.instance_metrics`` and is
cross-checked against it (TP/FP/FN must match exactly) so the diagnostic cannot
silently use a different rule.

Read-only: no training, no optimizer step, no checkpoint write.
"""
from __future__ import annotations

import argparse
import hashlib
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
from tokengs.options import config_defaults  # noqa: E402
from scripts.group_eval_v2 import build_val_entries, forward_group, group_predictions_v2  # noqa: E402
from scripts.object_locusgs_eval import gt_instances, instance_metrics  # noqa: E402


def file_identity(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": path.stat().st_mtime}


def greedy_with_ranks(predictions, instances):
    """Same greedy rule as instance_metrics, additionally returning TP ranks."""
    masks = list(instances.values())
    ious = []
    for prediction in predictions:
        row = {}
        for index, mask in enumerate(masks):
            union = int((prediction["mask"] | mask).sum())
            if union:
                row[index] = int((prediction["mask"] & mask).sum()) / union
        ious.append(row)
    scores = [prediction["score"] for prediction in predictions]
    order = (np.argsort(-np.asarray(scores, dtype=np.float64))
             if scores else np.zeros(0, dtype=np.int64))
    used, tp, fp, tp_ranks = set(), 0, 0, []
    for position in order:
        best, best_index = 0.0, None
        for index, value in ious[position].items():
            if index not in used and value > best:
                best, best_index = value, index
        if best_index is not None and best >= 0.5:
            used.add(best_index)
            tp += 1
            rank = int(np.flatnonzero(order == position)[0]) + 1
            tp_ranks.append({"rank": rank, "group": int(predictions[position]["group"]),
                             "score": float(scores[position]), "iou": float(best),
                             "gt_index": int(best_index)})
        else:
            fp += 1
    fn = len(masks) - len(used)
    return {"tp": tp, "fp": fp, "fn": fn, "tp_ranks": tp_ranks,
            "n_pred": len(predictions), "n_gt": len(masks)}


def load(directory: str, preset: str, seed: int, device, *, recipe: bool, g0plus: bool):
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm="g0", group_bg_supervision=g0plus,
        group_recipe=recipe, group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(directory) / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True,
    )
    model.eval()
    return opt, model


def run_one(label, directory, opt, model, entries, device):
    per_view, tps = [], []
    for entry in entries:
        with torch.no_grad():
            forward = forward_group(model, entry["batch"], opt)
        semantic = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
        instance = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
        for view in (2, 3):
            predictions, _ = group_predictions_v2(forward, view)
            instances = gt_instances(semantic, instance, view)
            ref = instance_metrics(predictions, instances)
            mine = greedy_with_ranks(predictions, instances)
            if (ref["tp"], ref["fp"], ref["fn"]) != (mine["tp"], mine["fp"], mine["fn"]):
                raise SystemExit(
                    f"{label} {entry['scene']} view {view}: greedy reproduction "
                    f"{(mine['tp'], mine['fp'], mine['fn'])} != instance_metrics "
                    f"{(ref['tp'], ref['fp'], ref['fn'])}"
                )
            per_view.append({
                "scene": entry["scene"], "view": view,
                "frame_id": int(entry["novel"][view - 2]),
                "n_candidates": mine["n_pred"], "n_gt": mine["n_gt"],
                "tp": mine["tp"], "fp": mine["fp"], "fn": mine["fn"],
                "tp_ranks": mine["tp_ranks"],
            })
            for item in mine["tp_ranks"]:
                tps.append({**item, "scene": entry["scene"], "view": view})
    ranks = [t["rank"] for t in tps]
    scores = [t["score"] for t in tps]
    summary = {
        "label": label, "checkpoint": directory,
        "n_candidates_total": int(sum(v["n_candidates"] for v in per_view)),
        "tp": len(tps), "fp": int(sum(v["fp"] for v in per_view)),
        "fn": int(sum(v["fn"] for v in per_view)),
        "tp_median_rank": float(np.median(ranks)) if ranks else None,
        "tp_rank_le_1": int(sum(1 for r in ranks if r <= 1)),
        "tp_rank_le_3": int(sum(1 for r in ranks if r <= 3)),
        "tp_rank_le_5": int(sum(1 for r in ranks if r <= 5)),
        "tp_score_min": float(min(scores)) if scores else None,
        "tp_score_median": float(np.median(scores)) if scores else None,
        "per_view": per_view, "tps": tps,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/recipe_v2/tp_rank_diag.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recipe-v1", default="workspace_group_plus/recipe_v1/run/ckpt_step6000")
    parser.add_argument("--g0plus", default="workspace_group_plus/arm_g0plus/ckpt_step6000")
    parser.add_argument("--recipe-v2", default=None,
                        help="optional recipe_v2 checkpoint to include in the same table")
    args = parser.parse_args()

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    specs = [("g0plus_step6000", args.g0plus, {"recipe": False, "g0plus": True}),
             ("recipe_v1_step6000", args.recipe_v1, {"recipe": True, "g0plus": False})]
    if args.recipe_v2:
        specs.append(("recipe_v2_step6000", args.recipe_v2,
                      {"recipe": True, "g0plus": False}))

    identities_before = {label: file_identity(Path(directory) / "model.pt")
                         for label, directory, _ in specs}
    out = {"scope": "8 unseen windows, novel views 2/3; GT-free read-out unchanged",
           "reader": "P(thing)>=0.5, thing class, mask>0.5, area>=50; greedy AP50 match",
           "variants": {}}
    for label, directory, flags in specs:
        opt, model = load(directory, args.preset, args.seed, device, **flags)
        entries = build_val_entries(opt, split, device)
        summary = run_one(label, directory, opt, model, entries, device)
        out["variants"][label] = summary
        print(f"[rank] {label}: cand {summary['n_candidates_total']} TP {summary['tp']} "
              f"FP {summary['fp']} FN {summary['fn']} median-rank "
              f"{summary['tp_median_rank']} <=1/3/5 "
              f"{summary['tp_rank_le_1']}/{summary['tp_rank_le_3']}/"
              f"{summary['tp_rank_le_5']}", flush=True)
        del model
        torch.cuda.empty_cache()
    out["identity_after"] = {label: file_identity(Path(directory) / "model.pt")
                             for label, directory, _ in specs}
    out["checkpoints_unchanged"] = out["identity_after"] == identities_before
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[rank] wrote {args.out} unchanged={out['checkpoints_unchanged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
