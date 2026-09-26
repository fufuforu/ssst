#!/usr/bin/env python3
"""Step 1 (read-only): group-mask formation diagnosis and pre-registered branch.

Reports the `GroupQueryHead.forward_full` wiring, runs a one-window smoke with
assertions and cross-checks against the recorded G0+ numbers, then measures

  (3) fragmentation of the 100 group masks (union of the top-1/2/3 hard masks
      versus GT), and
  (4) a token-granularity oracle whose token->instance assignment uses only the
      two context frames' GT (novel GT only scores it).

The pre-registered branch decision uses only the 8 unseen scenes' novel views
(55 instance records).  Nothing is trained, no checkpoint is written, no
threshold is changed.
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

from tokengs.models.input_types import ModelInputDecoder  # noqa: E402
from scripts.audit_bg_counterfactual import variant_metrics  # noqa: E402
from scripts.audit_mask_failure import hard_metrics, load_model  # noqa: E402
from scripts.eval_group_locusgs import build_train_entries  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    MASK_THRESHOLD,
    MIN_PRED_PIXELS,
    build_val_entries,
    forward_group,
)
from scripts.object_locusgs_eval import gt_instances  # noqa: E402

CHECKPOINTS = {
    "g0plus_step6000": "workspace_group_plus/arm_g0plus/ckpt_step6000",
    "g0_step6000": "workspace_group_locusgs/arm_g0/ckpt_step6000",
}
EXPECTED = {
    "all_views": {"tp": 6, "fp": 140, "fn": 104, "records": 110, "ap50": 0.0984},
    "novel": {"tp": 3, "fp": 71, "fn": 52, "records": 55},
}
AP50_TOL = 0.005
ORACLE_FEASIBLE_COUNT = 28          # of 55 novel records
FRAG_MEAN_DELTA = 0.08
FRAG_COUNT = 14                     # of 55 novel records with delta >= 0.10


def code_facts() -> dict:
    """Static description of the current group head (checked against the file)."""
    head = Path("tokengs/models/group_locusgs.py").read_text(encoding="utf-8")
    return {
        "file": "tokengs/models/group_locusgs.py",
        "class": "GroupQueryHead.forward_full",
        "query_token_interaction_layers": 1,
        "interaction_detail": "one nn.MultiheadAttention cross-attention "
                              "(self.cross_attn, 4 heads) from the 100 learnable queries "
                              "to the token features",
        "query_query_communication": False,
        "query_post_attention": "query_norm(queries + attended) then + mlp(...), both "
                                "per-query; no self-attention among queries",
        "spatial_encoding_location": "added to the *token* features "
                                     "(token_proj(token_norm(tokens)) + spatial_proj("
                                     "build_anchor_encoding(anchors, radii))) before the "
                                     "cross-attention and before the similarity",
        "final_logits": "similarity = (token_features . query_features) * logit_scale, "
                        "then concatenated with the shared scalar background_bias -> "
                        "slot_logits[B,T,101]; objectness/class heads read query_features",
        "assertions_in_source": {
            "one_cross_attn_call": head.count("self.cross_attn(") == 1,
            "no_self_attn_module": "self_attn" not in head,
            "spatial_proj_before_similarity": head.index("self.spatial_proj(") <
                                              head.index("similarity = ("),
            "background_bias_expand": "self.background_bias.expand" in head,
        },
    }


def checkpoint_identity() -> dict:
    identity = {}
    for name, directory in CHECKPOINTS.items():
        path = Path(directory) / "model.pt"
        payload = torch.load(Path(directory) / "train_state.pt", map_location="cpu",
                             weights_only=False)
        identity[name] = {
            "dir": directory,
            "arm": payload["meta"].get("arm"),
            "step": int(payload["step"]),
            "plan_sha256": payload["meta"].get("plan_sha256"),
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_mtime": path.stat().st_mtime,
        }
    return identity


def smoke(model, opt, entry, device) -> dict:
    """One checkpoint x one window with the pre-registered smoke assertions."""
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        repeat = forward_group(model, batch, opt)
    slot_logits = forward["group"]["slot_logits"]
    slot_prob = forward["group"]["slot_prob"]
    num_tokens = int(forward["output"]["gaussians"].shape[1] // 64)
    result = {
        "window": {"scene": entry["scene"], "context": entry["context"],
                   "novel": entry["novel"]},
        "tokens": num_tokens,
        "slot_logits_shape": list(slot_logits.shape),
        "slot_logits_dtype": str(slot_logits.dtype),
        "slot_logits_finite": bool(torch.isfinite(slot_logits).all()),
        "prob_sum_max_error": float((slot_prob.sum(-1) - 1).abs().max()),
        "conservation_error": float(
            (forward["masks"]["group_mass"].sum(2) + forward["masks"]["background_mass"][:, :, 0]
             - forward["masks"]["alpha"][:, :, 0]).abs().max()
        ),
        "rgb_repeat_max_diff": float(
            (forward["output"]["render"]["images_pred"]
             - repeat["output"]["render"]["images_pred"]).abs().max()
        ),
        "depth_repeat_max_diff": float(
            (forward["output"]["render"]["depths_pred"]
             - repeat["output"]["render"]["depths_pred"]).abs().max()
        ),
    }
    checks = {
        "tokens_is_1024": num_tokens == 1024,
        "slot_logits_shape_ok": list(slot_logits.shape) == [1, 1024, 101],
        "fp32": slot_logits.dtype == torch.float32,
        "finite": result["slot_logits_finite"],
        "prob_sums_to_one": result["prob_sum_max_error"] <= 1e-5,
        "conservation_le_2e-6": result["conservation_error"] <= 2e-6,
        "rgb_repeat_identical": result["rgb_repeat_max_diff"] == 0.0,
        "depth_repeat_identical": result["depth_repeat_max_diff"] == 0.0,
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


def crosscheck(model, opt, val_entries) -> dict:
    """Reproduce the recorded G0+ counts on the 8 fixed windows."""
    out = {}
    for tag, views in (("all_views", (0, 1, 2, 3)), ("novel", (2, 3))):
        tp = fp = fn = 0
        records = 0
        ap = []
        for entry in val_entries:
            semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
            instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
            with torch.no_grad():
                forward = forward_group(model, entry["batch"], opt)
            metrics = variant_metrics(forward, semantic_gt, instance_gt, views)
            tp += metrics["tp"]
            fp += metrics["fp"]
            fn += metrics["fn"]
            ap.append(metrics["ap50"])
            for view in views:
                records += int(metrics["views"][view]["n_gt"])
        out[tag] = {"tp": tp, "fp": fp, "fn": fn, "records": records,
                    "ap50_mean_over_scenes": float(np.mean(ap))}
    return out


def fragmentation(mass_view, truth, other_truths) -> dict:
    """Top-1/2/3 hard-mask union IoU for one GT instance in one view."""
    masks, ious = [], []
    for group_index in range(mass_view.shape[0]):
        hard = mass_view[group_index] > MASK_THRESHOLD
        if int(hard.sum()) < MIN_PRED_PIXELS:
            continue
        intersection = int((hard & truth).sum())
        union = int((hard | truth).sum())
        masks.append((group_index, hard))
        ious.append(intersection / max(1, union))
    if not masks:
        return {"iou1": 0.0, "iou2": 0.0, "iou3": 0.0, "delta": 0.0,
                "ranked_groups": [], "top1_area_over_gt": None,
                "union_intrusion_other_gt": None}
    order = sorted(range(len(masks)), key=lambda i: (-ious[i], masks[i][0]))
    unions = {}
    for k in (1, 2, 3):
        union_mask = np.zeros_like(truth)
        for position in order[:k]:
            union_mask |= masks[position][1]
        unions[k] = union_mask
    iou_k = {k: int((unions[k] & truth).sum()) / max(1, int((unions[k] | truth).sum()))
             for k in (1, 2, 3)}
    top1 = masks[order[0]][1]
    intrusion = 0
    for other in other_truths:
        overlap = int((unions[3] & other).sum())
        intrusion += overlap
    return {
        "iou1": iou_k[1], "iou2": iou_k[2], "iou3": iou_k[3],
        "delta": max(iou_k.values()) - iou_k[1],
        "ranked_groups": [int(masks[position][0]) for position in order[:3]],
        "ranked_ious": [float(ious[position]) for position in order[:3]],
        "top1_area_over_gt": float(int(top1.sum()) / max(1, int(truth.sum()))),
        "union3_area_over_gt": float(int(unions[3].sum()) / max(1, int(truth.sum()))),
        "union3_intrusion_other_gt_fraction": float(
            intrusion / max(1, int(unions[3].sum()))
        ),
    }


def token_oracle(model, opt, entry, *, device, return_masks: bool = False) -> dict:
    """Context-GT token assignment -> novel per-instance masks."""
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        gaussians = forward["output"]["gaussians"]
        alpha = forward["masks"]["alpha"]
        semantic = batch["semantic_label_all"].long()
        instance = batch["instance_label_all"].long()
        num_gaussians = gaussians.shape[1]
        num_tokens = num_gaussians // 64
        token_of = torch.arange(num_gaussians, device=device) // 64
        onehot = torch.zeros(1, num_gaussians, num_tokens, device=device)
        onehot[0, torch.arange(num_gaussians, device=device), token_of] = 1.0
        contribution = torch.zeros(num_tokens, device=device)
        context_keys = set()
        per_instance = {}
        for view in (0, 1):
            rendered = model.gs.render_feature_channels(
                gaussians, onehot, batch["cam_view_all"][:, view:view + 1],
                intrinsics=batch["intrinsics_all"][:, view:view + 1],
            )["images_pred"][0, 0]                       # [T,H,W]
            sem, ins = semantic[0, view], instance[0, view]
            valid = ((sem >= 2) & (sem < 20) & (ins > 0)
                     & (alpha[0, view, 0] > 0.5))
            keys = (torch.unique(((sem + 1) * 1000 + ins)[valid])
                    if valid.any() else torch.empty(0, dtype=torch.long, device=device))
            for key in keys.tolist():
                mask = valid & (((sem + 1) * 1000 + ins) == key)
                mass = rendered[:, mask].sum(-1)          # [T]
                per_instance[int(key)] = per_instance.get(int(key), torch.zeros_like(mass)) + mass
                context_keys.add(int(key))
            contribution += rendered.sum(dim=(1, 2))
            del rendered
        keys_sorted = sorted(per_instance)
        if keys_sorted:
            stacked = torch.stack([per_instance[k] for k in keys_sorted], dim=-1)  # [T,K]
            best = stacked.argmax(dim=-1)
            has = stacked.max(dim=-1).values > 0
            assignment = torch.zeros(num_tokens, len(keys_sorted) + 1, device=device)
            assignment[torch.arange(num_tokens, device=device)[has], best[has]] = 1.0
            assignment[~has, -1] = 1.0
        else:
            assignment = torch.zeros(num_tokens, 1, device=device)
            assignment[:, 0] = 1.0
        gaussian_assignment = assignment[token_of].unsqueeze(0)   # [1, N, K+1]
        rendered = model.gs.render_feature_channels(
            gaussians, gaussian_assignment, batch["cam_view_all"],
            intrinsics=batch["intrinsics_all"],
        )["images_pred"]                                      # [1,V,K+1,H,W]
    rows = []
    semantic_gt = semantic[0].cpu().numpy()
    instance_gt = instance[0].cpu().numpy()
    for view in (2, 3):
        sem, ins = semantic_gt[view], instance_gt[view]
        key_map = (sem + 1) * 1000 + ins
        visible = (sem >= 2) & (sem < 20) & (ins > 0)
        for key in np.unique(key_map[visible]).tolist():
            truth = visible & (key_map == key)
            context_visible = int(key) in context_keys
            iou = 0.0
            pred_area = 0
            if context_visible:
                channel = keys_sorted.index(int(key))
                mask = rendered[0, view, channel].cpu().numpy() > MASK_THRESHOLD
                pred_area = int(mask.sum())
                if pred_area >= MIN_PRED_PIXELS:
                    intersection = int((mask & truth).sum())
                    iou = intersection / max(1, int((mask | truth).sum()))
                else:
                    iou = 0.0
            rows.append({"scene": entry["scene"], "view": view, "key": int(key),
                         "gt_area": int(truth.sum()), "context_visible": context_visible,
                         "oracle_iou": float(iou), "oracle_pred_area": pred_area})
    rest_fraction = float(assignment[:, -1].mean()) if assignment.shape[1] else 1.0
    result = {"rows": rows, "rest_fraction": rest_fraction,
              "context_instances": len(keys_sorted)}
    if return_masks:
        result["masks"] = rendered.detach()
        result["keys_sorted"] = [int(k) for k in keys_sorted]
    return result


def resolve_branch(records_novel, oracle_rows) -> dict:
    """Pre-registered decision on novel-only (55) records."""
    deltas = [r["delta"] for r in records_novel]
    oracle_iou = {**{(r["scene"], r["view"], r["key"]): r for r in oracle_rows}}
    oracle_scores = []
    for record in records_novel:
        row = oracle_iou.get((record["scene"], record["view"], record["key"]))
        oracle_scores.append(0.0 if row is None else row["oracle_iou"])
    feasible = sum(1 for value in oracle_scores if value >= 0.5)
    frag_mean = float(np.mean(deltas)) if deltas else 0.0
    frag_count = sum(1 for value in deltas if value >= 0.10)
    decision = {
        "records_novel": len(records_novel),
        "oracle_iou_ge_0.5": feasible,
        "oracle_feasible": feasible >= ORACLE_FEASIBLE_COUNT,
        "fragmentation_mean_delta": frag_mean,
        "fragmentation_count_delta_ge_0.10": frag_count,
        "fragmentation_significant": frag_mean >= FRAG_MEAN_DELTA and frag_count >= FRAG_COUNT,
    }
    if not decision["oracle_feasible"]:
        decision["branch"] = "STOP_ORACLE_INFEASIBLE"
    elif decision["fragmentation_significant"]:
        decision["branch"] = "A_query_refiner"
    else:
        decision["branch"] = "B_relative_position"
    # context_visible subset statistics for the stop report
    visible_rows = [r for r in oracle_rows if r["context_visible"]]
    decision["context_visible_instances"] = len(visible_rows)
    decision["context_visible_oracle_iou_ge_0.5"] = sum(
        1 for r in visible_rows if r["oracle_iou"] >= 0.5
    )
    decision["context_visible_rate"] = (
        decision["context_visible_oracle_iou_ge_0.5"] / len(visible_rows)
        if visible_rows else None
    )
    if not decision["oracle_feasible"]:
        if not visible_rows:
            decision["stop_reason"] = "cannot determine: no context-visible instances"
        elif decision["context_visible_rate"] >= 0.5:
            decision["stop_reason"] = ("context-visibility limitation, token ceiling "
                                       "cannot be determined")
        else:
            decision["stop_reason"] = "oracle read-out insufficient under this protocol"
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--instances", default="group_plus/mask_failure/manifest.json")
    parser.add_argument("--out-dir", default="group_plus/routing_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.instances).read_text(encoding="utf-8"))["selected"]

    report = {
        "scope": "Step 1 read-only group-mask formation diagnosis",
        "code_facts": code_facts(),
        "checkpoints_before": checkpoint_identity(),
        "plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
    }
    opt_plus, model_plus, _, _, _ = load_model(
        CHECKPOINTS["g0plus_step6000"], args.preset, args.seed, device
    )
    opt_g0, model_g0, _, _, _ = load_model(
        CHECKPOINTS["g0_step6000"], args.preset, args.seed, device
    )
    val_entries_plus = build_val_entries(opt_plus, split, device)
    val_entries_g0 = build_val_entries(opt_g0, split, device)
    train_entries_plus = build_train_entries(opt_plus, split, plan, device, 8)
    train_entries_g0 = build_train_entries(opt_g0, split, plan, device, 8)

    # ---- 1.2 smoke + cross-check ---- #
    report["smoke"] = smoke(model_plus, opt_plus, val_entries_plus[0], device)
    print("[routing] smoke:", json.dumps(report["smoke"]["checks"]), flush=True)
    if not report["smoke"]["passed"]:
        (out_dir / "step1_smoke_failure.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8"
        )
        print("[routing] SMOKE FAILED - stop", flush=True)
        return 1
    report["crosscheck"] = crosscheck(model_plus, opt_plus, val_entries_plus)
    expected_ok = (
        report["crosscheck"]["all_views"]["tp"] == EXPECTED["all_views"]["tp"]
        and report["crosscheck"]["all_views"]["fp"] == EXPECTED["all_views"]["fp"]
        and report["crosscheck"]["all_views"]["fn"] == EXPECTED["all_views"]["fn"]
        and report["crosscheck"]["all_views"]["records"] == EXPECTED["all_views"]["records"]
        and abs(report["crosscheck"]["all_views"]["ap50_mean_over_scenes"]
                - EXPECTED["all_views"]["ap50"]) <= AP50_TOL
        and report["crosscheck"]["novel"]["tp"] == EXPECTED["novel"]["tp"]
        and report["crosscheck"]["novel"]["fp"] == EXPECTED["novel"]["fp"]
        and report["crosscheck"]["novel"]["fn"] == EXPECTED["novel"]["fn"]
        and report["crosscheck"]["novel"]["records"] == EXPECTED["novel"]["records"]
    )
    report["crosscheck"]["matches_recorded"] = bool(expected_ok)
    print("[routing] crosscheck:", json.dumps(report["crosscheck"]), flush=True)
    if not expected_ok:
        (out_dir / "step1_crosscheck_failure.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8"
        )
        print("[routing] CROSS-CHECK FAILED - fix the harness before deciding", flush=True)
        return 1
    if args.smoke_only:
        (out_dir / "step1_smoke.json").write_text(json.dumps(report, indent=1),
                                                  encoding="utf-8")
        print("[routing] smoke-only run finished", flush=True)
        return 0

    # ---- 1.3 fragmentation + 1.4 oracle ---- #
    fragmentation_rows = {"g0plus_step6000": [], "g0_step6000": []}
    oracle_rows = {"g0plus_step6000": [], "g0_step6000": []}
    for name, model, opt, val_entries, train_entries in (
        ("g0plus_step6000", model_plus, opt_plus, val_entries_plus, train_entries_plus),
        ("g0_step6000", model_g0, opt_g0, val_entries_g0, train_entries_g0),
    ):
        for group, entries in (("unseen", val_entries), ("training", train_entries)):
            for entry in entries:
                semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
                instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
                with torch.no_grad():
                    forward = forward_group(model, entry["batch"], opt)
                for view in (2, 3):
                    sem, ins = semantic_gt[view], instance_gt[view]
                    key_map = (sem + 1) * 1000 + ins
                    visible = (sem >= 2) & (sem < 20) & (ins > 0)
                    keys = np.unique(key_map[visible]).tolist()
                    for key in keys:
                        truth = visible & (key_map == key)
                        others = [visible & (key_map == other) for other in keys
                                  if other != key]
                        result = fragmentation(
                            forward["masks"]["group_mass"][0, view].float().cpu().numpy(),
                            truth, others,
                        )
                        result.update({"scene": entry["scene"], "view": view,
                                       "key": int(key), "group": group,
                                       "gt_area": int(truth.sum())})
                        fragmentation_rows[name].append(result)
                if not args.skip_oracle:
                    oracle = token_oracle(model, opt, entry, device=device)
                    for row in oracle["rows"]:
                        row["group"] = group
                    oracle_rows[name].extend(oracle["rows"])
                    print(f"[routing] oracle {name} {entry['scene']} rest "
                          f"{oracle['rest_fraction']:.3f} "
                          f"context_instances {oracle['context_instances']}", flush=True)
                del forward
    report["fragmentation"] = fragmentation_rows
    report["oracle"] = oracle_rows
    novel_records = [r for r in fragmentation_rows["g0plus_step6000"]
                     if r["group"] == "unseen"]
    report["branch"] = resolve_branch(novel_records, oracle_rows["g0plus_step6000"])
    report["checkpoints_after"] = checkpoint_identity()
    report["checkpoint_files_unchanged"] = {
        name: (report["checkpoints_before"][name]["model_sha256"]
               == report["checkpoints_after"][name]["model_sha256"]
               and report["checkpoints_before"][name]["model_mtime"]
               == report["checkpoints_after"][name]["model_mtime"])
        for name in CHECKPOINTS
    }
    (out_dir / "step1_summary.json").write_text(json.dumps(report, indent=1),
                                                encoding="utf-8")
    write_csv(out_dir / "fragmentation.csv", fragmentation_rows)
    write_csv(out_dir / "oracle.csv", oracle_rows)
    print("[routing] branch decision:", json.dumps(report["branch"], indent=1), flush=True)
    print(f"[routing] wrote {out_dir}")
    return 0


def write_csv(path: Path, groups: dict) -> None:
    rows = []
    for name, records in groups.items():
        for record in records:
            flat = {"variant": name}
            flat.update({k: v for k, v in record.items()
                         if isinstance(v, (int, float, str, bool)) or v is None})
            rows.append(flat)
    if not rows:
        return
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
