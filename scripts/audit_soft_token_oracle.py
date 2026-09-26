#!/usr/bin/env python3
"""Read-only diagnostic: per-scene soft token->instance fit with frozen G0+.

Phase 0 reproduces the Step 1 hard majority-vote oracle line by line; phase 1 fits
one temporary Z[T,K] per scene with Adam (the *only* trainable variable; every
model parameter has requires_grad=False and never receives a gradient); phase 3
pairs the step-500 soft oracle with Step 1 on the fixed 55 novel records; phase 4
applies the pre-registered B1/B2/B3 decision.

This is a diagnostic fit, not feed-forward training and not a GT-free instance
segmentation result: the soft assignment is fitted on the two context frames' GT
and scored on novel frames.  No checkpoint, Z, A or dense per-token map is saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models.input_types import ModelInputDecoder  # noqa: E402
from scripts.audit_group_routing import token_oracle  # noqa: E402
from scripts.audit_mask_failure import hard_metrics, load_model  # noqa: E402
from scripts.group_eval_v2 import build_val_entries, forward_group  # noqa: E402

CHECKPOINT = "workspace_group_plus/arm_g0plus/ckpt_step6000"
PRESET = "train_siu3r_group_logs_locusgs_ab"          # replaced below if needed
PRESET = "train_siu3r_group_locusgs_ab"
SIZE_SPLIT = 3000
STEPS = 500
LR = 0.05
LOG_STEPS = (0, 50, 100, 200, 300, 400, 500)
B1_BAR = 28
B2_BAR = 44
PURITY_THRESHOLD = 0.50


def file_identity(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": path.stat().st_mtime, "size": path.stat().st_size}


def checkpoint_identity() -> dict:
    directory = Path(CHECKPOINT)
    state = directory / "train_state.pt"
    out = {"model": file_identity(directory / "model.pt")}
    if state.is_file():
        out["train_state"] = file_identity(state)
    return out


def context_instances(semantic, instance, views=(0, 1), *, alpha):
    """Context thing instances with valid pixels, sorted by packed instance id."""
    per_view = []
    keys = set()
    for view in views:
        sem, ins = semantic[0, view], instance[0, view]
        valid = (sem >= 2) & (sem < 20) & (ins > 0) & (alpha[0, view, 0] > 0.5)
        packed = (sem + 1) * 1000 + ins
        per_view.append((valid, packed))
        if valid.any():
            keys |= set(int(k) for k in torch.unique(packed[valid]).tolist())
    return sorted(keys), per_view


def majority_vote_assignment(model, opt, entry, device):
    """Phase-0 context GT majority vote (reuses audit_group_routing.token_oracle)."""
    oracle = token_oracle(model, opt, entry, device=device)
    return oracle


def token_contribution_stats(model, opt, entry, keys, device):
    """Per-token contribution mass to each instance and to all valid annotated px."""
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
        per_instance = {int(k): torch.zeros(num_tokens, device=device) for k in keys}
        annotated_total = torch.zeros(num_tokens, device=device)
        single_best = torch.zeros(num_tokens, device=device)
        for view in (0, 1):
            rendered = model.gs.render_feature_channels(
                gaussians, onehot, batch["cam_view_all"][:, view:view + 1],
                intrinsics=batch["intrinsics_all"][:, view:view + 1],
            )["images_pred"][0, 0]
            sem, ins = semantic[0, view], instance[0, view]
            annotated = (sem != 255) & (alpha[0, view, 0] > 0.5)
            if annotated.any():
                annotated_total += rendered[:, annotated].sum(-1)
            for key in keys:
                mask = (annotated
                        & (((sem + 1) * 1000 + ins) == int(key))
                        & (sem >= 2) & (sem < 20) & (ins > 0))
                if mask.any():
                    per_instance[int(key)] += rendered[:, mask].sum(-1)
            del rendered
        if keys:
            stacked = torch.stack([per_instance[int(k)] for k in keys], dim=-1)
            single_best = stacked.max(dim=-1).values
        return {"per_instance": per_instance, "annotated_total": annotated_total,
                "single_best": single_best, "alpha": alpha}


def init_Z(num_tokens, keys, votes, device):
    """Z init from the phase-0 majority vote: 4.0 on the token's column, else 0."""
    columns = {int(key): index for index, key in enumerate(keys)}
    Z = torch.zeros(num_tokens, len(keys) + 1, device=device)
    for token, key in enumerate(votes):
        column = columns.get(int(key), len(keys)) if key is not None else len(keys)
        Z[token, column] = 4.0
    return Z


def render_assignment(model, gaussians, A, cam_view, intrinsics):
    """Render K channels from a [T,K] token assignment (same compositor as RGB)."""
    num_gaussians = gaussians.shape[1]
    token_of = torch.arange(num_gaussians, device=A.device) // 64
    features = A[token_of].unsqueeze(0)                      # [1,N,K]
    rendered = model.gs.render_feature_channels(
        gaussians, features, cam_view, intrinsics=intrinsics
    )
    return rendered["images_pred"], rendered["alphas_pred"]


def fit_scene(model, opt, entry, device, *, steps: int = STEPS, log_steps=LOG_STEPS,
              seed: int = 42):
    """Per-scene Adam fit of Z; returns records and the final A (not stored)."""
    torch.manual_seed(seed)
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        gaussians = forward["output"]["gaussians"].detach()
        alpha_fixed = forward["masks"]["alpha"].detach()
        rgb_fixed = forward["output"]["render"]["images_pred"].detach()
        depth_fixed = forward["output"]["render"]["depths_pred"].detach()
    semantic = batch["semantic_label_all"].long()
    instance = batch["instance_label_all"].long()
    keys, _ = context_instances(semantic, instance, alpha=alpha_fixed)
    oracle = majority_vote_assignment(model, opt, entry, device)
    del oracle
    # per-token majority vote is recovered from the phase-0 contribution statistics
    stats = token_contribution_stats(model, opt, entry, keys, device)
    if keys:
        stacked = torch.stack([stats["per_instance"][int(k)] for k in keys], dim=-1)
        best = stacked.argmax(dim=-1)
        has = stacked.max(dim=-1).values > 0
        token_votes = [keys[int(best[t])] if bool(has[t]) else None
                       for t in range(stacked.shape[0])]
    else:
        token_votes = [None] * 1024
    num_tokens = gaussians.shape[1] // 64
    Z = init_Z(num_tokens, keys, token_votes, device)
    Z.requires_grad_(True)
    optimizer = torch.optim.Adam([Z], lr=LR, weight_decay=0)
    context_view = batch["cam_view_all"][:, :2]
    context_intr = batch["intrinsics_all"][:, :2]
    valid_pixels = (semantic[0, :2] != 255) & (alpha_fixed[0, :2, 0] > 0.5)   # [2,H,W]
    stuff = ((semantic[0, :2] == 0) | (semantic[0, :2] == 1)) & valid_pixels
    thing = ((semantic[0, :2] >= 2) & (semantic[0, :2] < 20)
             & (instance[0, :2] > 0) & valid_pixels)
    gt_masks = {}
    for key in keys:
        packed = (semantic[0, :2] + 1) * 1000 + instance[0, :2]
        gt_masks[int(key)] = valid_pixels & (packed == int(key)) & (
            (semantic[0, :2] >= 2) & (semantic[0, :2] < 20) & (instance[0, :2] > 0)
        )
    records = []
    final_A = None
    for step in range(steps + 1):
        if step in log_steps:
            with torch.no_grad():
                A = torch.softmax(Z.detach(), dim=-1)
                context_mass, _ = render_assignment(
                    model, gaussians, A, context_view, context_intr
                )
                losses = scene_loss(context_mass, gt_masks, keys, alpha_fixed,
                                    valid_pixels, stuff, thing)
                context_hard = []
                for index, key in enumerate(keys):
                    for view in (0, 1):
                        if not bool(gt_masks[int(key)][view].any()):
                            continue
                        metrics = hard_metrics(
                            context_mass[0, view, index].cpu(), gt_masks[int(key)][view].cpu()
                        )
                        context_hard.append(metrics["iou"])
                record = {
                    "step": step, "loss": float(losses["total"]),
                    "loss_instance": float(losses["instance"]),
                    "loss_rest": float(losses["rest"]),
                    "context_hard_iou_mean": float(np.mean(context_hard)) if context_hard else 0.0,
                    "entropy": float(-(A.clamp_min(1e-8).log() * A).sum(-1).mean()),
                    "rest_fraction": float(A[:, -1].mean()),
                    "n_instances": len(keys),
                    "stuff_pixels": int(stuff.sum()), "thing_pixels": int(thing.sum()),
                }
            records.append(record)
        if step == steps:
            with torch.no_grad():
                final_A = torch.softmax(Z.detach(), dim=-1)
            break
        optimizer.zero_grad()
        A = torch.softmax(Z, dim=-1)
        context_mass, _ = render_assignment(model, gaussians, A, context_view, context_intr)
        losses = scene_loss(context_mass, gt_masks, keys, alpha_fixed,
                            valid_pixels, stuff, thing)
        if not math.isfinite(float(losses["total"])):
            raise SystemExit(f"non-finite loss at step {step} in scene {entry['scene']}")
        losses["total"].backward()
        if Z.grad is None or not torch.isfinite(Z.grad).all():
            raise SystemExit(f"bad Z gradient at step {step} in scene {entry['scene']}")
        if step == 0:
            grad_norm_first = float(Z.grad.norm())
        optimizer.step()
    return {
        "scene": entry["scene"], "keys": [int(k) for k in keys],
        "records": records, "final_A": final_A, "alpha": alpha_fixed,
        "gaussians": gaussians, "rgb": rgb_fixed, "depth": depth_fixed,
        "semantic": semantic, "instance": instance, "gt_masks": gt_masks,
        "stuff": stuff, "thing": thing, "valid_pixels": valid_pixels,
        "stats": stats,
        "grad_norm_first_step": grad_norm_first,
    }


def scene_loss(context_mass, gt_masks, keys, alpha_fixed, valid_pixels, stuff, thing):
    """L_instance (5*mean BCE + 5*mean Dice) + 1.0 * L_rest (class balanced)."""
    zero = context_mass.sum() * 0.0
    bce_terms, dice_terms = [], []
    for index, key in enumerate(keys):
        for view in (0, 1):
            mask = gt_masks[int(key)][view]
            if not bool(mask.any()):
                continue
            probability = context_mass[0, view, index].clamp(1e-5, 1 - 1e-5)
            logit = torch.logit(probability)
            bce_terms.append(F.binary_cross_entropy_with_logits(logit[mask], mask[mask].float()))
            intersection = (probability * mask.float()).sum()
            dice_terms.append(1.0 - (2.0 * intersection + 1.0) / (
                probability.sum() + mask.float().sum() + 1.0
            ))
    instance = zero
    if bce_terms:
        instance = 5.0 * torch.stack(bce_terms).mean() + 5.0 * torch.stack(dice_terms).mean()
    rest_mass = context_mass[0, :2, keys.__len__()] if context_mass.shape[2] > len(keys) else None
    rest = zero
    if rest_mass is not None:
        p_rest = (rest_mass / alpha_fixed[0, :2, 0].detach().clamp_min(0.5)).clamp(1e-5, 1 - 1e-5)
        terms = []
        if bool(stuff.any()):
            terms.append(0.5 * F.binary_cross_entropy(p_rest[stuff], torch.ones_like(p_rest[stuff])))
        if bool(thing.any()):
            terms.append(0.5 * F.binary_cross_entropy(p_rest[thing], torch.zeros_like(p_rest[thing])))
        if terms:
            rest = torch.stack(terms).sum()
    return {"total": instance + 1.0 * rest, "instance": instance, "rest": rest}


def iou_with_gate(mass, truth, *, min_area=50):
    mask = (mass > 0.5).cpu().numpy()
    area = int(mask.sum())
    if area < min_area:
        return 0.0, area
    truth_np = truth.cpu().numpy()
    intersection = int((mask & truth_np).sum())
    union = int((mask | truth_np).sum())
    return (intersection / max(1, union)), area


def evaluate_scene(result, model, opt, entry, device):
    """Step-500 read-out: novel soft IoU, merged context IoU, area statistics."""
    A = result["final_A"]
    gaussians = result["gaussians"]
    batch = entry["batch"]
    semantic, instance = result["semantic"], result["instance"]
    with torch.no_grad():
        rendered, _ = render_assignment(
            model, gaussians, A, batch["cam_view_all"], batch["intrinsics_all"]
        )
    keys = result["keys"]
    index_of = {int(key): index for index, key in enumerate(keys)}
    context_iou = {}
    for key in keys:
        index = index_of[int(key)]
        views = []
        for view in (0, 1):
            sem, ins = semantic[0, view], instance[0, view]
            truth = ((sem >= 2) & (sem < 20) & (ins > 0)
                     & (((sem + 1) * 1000 + ins) == int(key)))
            if not bool(truth.any()):
                continue
            data = {"truth": truth.cpu(), "mass": rendered[0, view, index].cpu()}
            views.append(data)
        if not views:
            context_iou[int(key)] = {"iou": 0.0, "both_below_area": True,
                                     "n_counted_views": 0}
            continue
        counted = []
        for data in views:
            mask = (data["mass"] > 0.5).numpy()
            if int(mask.sum()) >= 50:
                counted.append((mask, data["truth"].numpy()))
        if not counted:
            context_iou[int(key)] = {"iou": 0.0, "both_below_area": True,
                                     "n_counted_views": 0}
            continue
        merged_pred = np.zeros_like(counted[0][0])
        merged_truth = np.zeros_like(counted[0][1])
        for mask, truth in counted:
            merged_pred |= mask
            merged_truth |= truth
        intersection = int((merged_pred & merged_truth).sum())
        union = int((merged_pred | merged_truth).sum())
        context_iou[int(key)] = {
            "iou": intersection / max(1, union),
            "both_below_area": False,
            "n_counted_views": len(counted),
        }
    novel_rows = []
    for view in (2, 3):
        sem, ins = semantic[0, view].cpu().numpy(), instance[0, view].cpu().numpy()
        packed = (sem + 1) * 1000 + ins
        visible = (sem >= 2) & (sem < 20) & (ins > 0)
        for key in sorted(set(int(k) for k in np.unique(packed[visible]))):
            truth = visible & (packed == key)
            index = index_of.get(int(key))
            soft_iou, soft_area = (0.0, 0)
            if index is not None:
                soft_iou, soft_area = iou_with_gate(rendered[0, view, index], torch.from_numpy(truth))
            context_areas = []
            for ctx_view in (0, 1):
                ctx_truth = ((semantic[0, ctx_view].cpu().numpy() + 1) * 1000
                             + instance[0, ctx_view].cpu().numpy()) == key
                ctx_truth &= ((semantic[0, ctx_view].cpu().numpy() >= 2)
                              & (semantic[0, ctx_view].cpu().numpy() < 20)
                              & (instance[0, ctx_view].cpu().numpy() > 0))
                context_areas.append(int(ctx_truth.sum()))
            record = {
                "scene": entry["scene"], "view": view,
                "frame_id": int(entry["novel"][view - 2]), "key": int(key),
                "novel_gt_area": int(truth.sum()),
                "context_gt_areas": context_areas,
                "size_bucket": "small" if int(truth.sum()) < SIZE_SPLIT else "large",
                "soft_iou": float(soft_iou), "soft_pred_area": int(soft_area),
                "soft_pass": bool(soft_iou >= 0.5),
                "has_context_column": index is not None,
                "context_iou": context_iou.get(int(key), {}).get("iou", 0.0),
                "context_both_below_area": context_iou.get(int(key), {}).get(
                    "both_below_area", True),
                "area_ratio_mean_context_over_novel": (
                    float(np.mean(context_areas)) / int(truth.sum())
                    if int(truth.sum()) > 0 else None
                ),
            }
            novel_rows.append(record)
    return novel_rows, rendered


def purity_filtered_oracle(result, model, entry, device, keys):
    """Report-only (a): keep the majority vote only for tokens with purity >= 0.50."""
    stats = result["stats"]
    total = stats["annotated_total"].clamp_min(1e-12)
    purity = stats["single_best"] / total
    if keys:
        stacked = torch.stack([stats["per_instance"][int(k)] for k in keys], dim=-1)
        best = stacked.argmax(dim=-1)
        has = (stacked.max(dim=-1).values > 0) & (purity >= PURITY_THRESHOLD)
    else:
        best = torch.zeros(result["gaussians"].shape[1] // 64, dtype=torch.long,
                           device=device)
        has = torch.zeros_like(best, dtype=torch.bool)
    num_tokens = best.shape[0]
    A = torch.zeros(num_tokens, len(keys) + 1, device=device)
    if keys:
        A[torch.arange(num_tokens, device=device)[has], best[has]] = 1.0
    A[~has, -1] = 1.0
    with torch.no_grad():
        rendered, _ = render_assignment(
            model, result["gaussians"], A, entry["batch"]["cam_view_all"],
            entry["batch"]["intrinsics_all"],
        )
    rows = []
    index_of = {int(key): index for index, key in enumerate(keys)}
    for view in (2, 3):
        sem = result["semantic"][0, view].cpu().numpy()
        ins = result["instance"][0, view].cpu().numpy()
        packed = (sem + 1) * 1000 + ins
        visible = (sem >= 2) & (sem < 20) & (ins > 0)
        for key in sorted(set(int(k) for k in np.unique(packed[visible]))):
            truth = visible & (packed == key)
            index = index_of.get(int(key))
            iou, area = 0.0, 0
            if index is not None:
                iou, area = iou_with_gate(rendered[0, view, index], torch.from_numpy(truth))
            rows.append({"scene": entry["scene"], "view": view, "key": int(key),
                         "purity_oracle_iou": float(iou), "purity_oracle_area": int(area),
                         "purity_pass": bool(iou >= 0.5)})
    return rows, float(A[:, -1].mean())


def phase0_reproduction(model, opt, entries, device, recorded_csv: Path):
    """Recompute the Step-1 hard oracle and align it line by line."""
    recorded = {}
    with recorded_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "g0plus_step6000" or row["group"] != "unseen":
                continue
            recorded[(row["scene"], int(row["view"]), int(row["key"]))] = row
    ours = {}
    for entry in entries:
        oracle = token_oracle(model, opt, entry, device=device)
        for row in oracle["rows"]:
            ours[(entry["scene"], int(row["view"]), int(row["key"]))] = row
    max_gap = 0.0
    mismatches = []
    for key, row in recorded.items():
        if key not in ours:
            mismatches.append({"key": key, "reason": "missing in recomputation"})
            continue
        gap = abs(float(row["oracle_iou"]) - float(ours[key]["oracle_iou"]))
        max_gap = max(max_gap, gap)
        if gap > 1e-5:
            mismatches.append({"key": key, "recorded": float(row["oracle_iou"]),
                               "recomputed": float(ours[key]["oracle_iou"])})
    pass_count = sum(1 for row in ours.values() if float(row["oracle_iou"]) >= 0.5)
    sizes = {"small": [0, 0], "large": [0, 0]}
    for key, row in ours.items():
        bucket = "small" if int(row["gt_area"]) < SIZE_SPLIT else "large"
        sizes[bucket][1] += 1
        if float(row["oracle_iou"]) >= 0.5:
            sizes[bucket][0] += 1
    result = {
        "recorded_rows": len(recorded), "recomputed_rows": len(ours),
        "max_abs_iou_gap": max_gap, "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:5],
        "oracle_pass_ge_0.5": pass_count,
        "size_strata": {"small_lt3000": sizes["small"], "large_ge3000": sizes["large"]},
        "expected": {"pass": 22, "rows": 55, "small": [2, 20], "large": [20, 35],
                     "max_gap": 1e-5},
    }
    result["reproduced"] = (
        result["recorded_rows"] == 55 and result["recomputed_rows"] == 55
        and result["oracle_pass_ge_0.5"] == 22
        and result["size_strata"]["small_lt3000"] == [2, 20]
        and result["size_strata"]["large_ge3000"] == [20, 35]
        and max_gap <= 1e-5
    )
    return result


def smoke_scene(model, opt, entry, device, *, steps: int = 10) -> dict:
    """10-step smoke on one scene with the pre-registered assertions."""
    prior = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = fit_scene(model, opt, entry, device, steps=steps, log_steps=(0, steps))
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
    A = result["final_A"]
    checks = {
        "A_shape": list(A.shape) == [1024, len(result["keys"]) + 1],
        "A_fp32": A.dtype == torch.float32,
        "A_finite": bool(torch.isfinite(A).all()),
        "A_rows_sum_one": float((A.sum(-1) - 1).abs().max()) <= 1e-6,
        "grads_absent_on_model": all(p.grad is None for p in model.parameters()),
        "params_unchanged": all(
            torch.equal(prior[name], p.detach()) for name, p in model.named_parameters()
        ),
        "loss_decreased": (
            result["records"][0]["loss"] - result["records"][-1]["loss"] >= 1e-5
        ),
        "loss_finite": all(math.isfinite(r["loss"]) for r in result["records"]),
        "Z_grad_finite_nonzero": (math.isfinite(result["grad_norm_first_step"])
                                  and result["grad_norm_first_step"] > 0),
        "alpha_conservation": None,
        "rgb_unchanged": None,
    }
    alpha = forward["masks"]["alpha"]
    conservation = float(
        (forward["masks"]["group_mass"].sum(2) + forward["masks"]["background_mass"][:, :, 0]
         - alpha[:, :, 0]).abs().max()
    )
    checks["alpha_conservation"] = conservation <= 2e-6
    checks["rgb_unchanged"] = float(
        (forward["output"]["render"]["images_pred"] - result["rgb"]).abs().max()
    ) == 0.0
    checks["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return {"scene": entry["scene"], "checks": checks,
            "records": result["records"], "conservation_error": conservation,
            "grad_norm_first_step": result["grad_norm_first_step"],
            "keys": result["keys"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default=PRESET)
    parser.add_argument("--recorded", default="group_plus/routing_v1/oracle.csv")
    parser.add_argument("--out-dir", default="group_plus/routing_v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--phase0-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    before = checkpoint_identity()
    opt, model, arm, step, meta = load_model(CHECKPOINT, args.preset, args.seed, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    entries = build_val_entries(opt, split, device, scenes=args.scenes)
    report = {
        "scope": "read-only per-scene soft token->instance fit on frozen G0+; "
                 "diagnostic only, not feed-forward and not GT-free",
        "checkpoint": CHECKPOINT, "arm": arm, "step": step,
        "checkpoint_before": before,
        "hyperparameters": {"steps": args.steps, "lr": LR, "optimizer": "Adam",
                            "weight_decay": 0, "seed": args.seed,
                            "loss": "L_instance(5*meanBCE+5*meanDice) + 1.0*L_rest",
                            "gates": {"mask": 0.5, "area": 50}},
        "preset": args.preset,
    }
    print("[soft] phase 0 reproduction ...", flush=True)
    report["step1_reproduction"] = phase0_reproduction(
        model, opt, entries, device, Path(args.recorded)
    )
    print(json.dumps(report["step1_reproduction"], indent=1), flush=True)
    (out_dir / "step1_reproduction.json").write_text(
        json.dumps(report["step1_reproduction"], indent=1), encoding="utf-8"
    )
    if not report["step1_reproduction"]["reproduced"]:
        (out_dir / "phase0_failure.json").write_text(json.dumps(report, indent=1),
                                                     encoding="utf-8")
        print("[soft] phase 0 NOT reproduced - stop before any Z optimisation", flush=True)
        return 1
    if args.phase0_only:
        return 0

    # ---- phase 2 smoke on the first scene (by scene id) with context instances --- #
    def has_context_things(candidate) -> bool:
        sem = candidate["batch"]["semantic_label_all"][0, :2].long()
        ins = candidate["batch"]["instance_label_all"][0, :2].long()
        return bool(((sem >= 2) & (sem < 20) & (ins > 0)).any())

    smoke_entry = next(
        (e for e in sorted(entries, key=lambda e: e["scene"]) if has_context_things(e)), None
    )
    if smoke_entry is None:
        print("[soft] no scene with context instances - stop", flush=True)
        return 1
    report["smoke"] = smoke_scene(model, opt, smoke_entry, device, steps=10)
    print("[soft] smoke checks:", json.dumps(report["smoke"]["checks"]), flush=True)
    (out_dir / "smoke.json").write_text(json.dumps(report["smoke"], indent=1),
                                        encoding="utf-8")
    if not report["smoke"]["checks"]["passed"]:
        (out_dir / "smoke_failure.json").write_text(json.dumps(report, indent=1),
                                                    encoding="utf-8")
        print("[soft] SMOKE FAILED - no full run", flush=True)
        return 1
    if args.smoke_only:
        return 0

    # ---- phase 1: 500 steps per scene (fresh Z each) ---- #
    recorded_rows = {}
    with Path(args.recorded).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "g0plus_step6000" and row["group"] == "unseen":
                recorded_rows[(row["scene"], int(row["view"]), int(row["key"]))] = row
    per_scene = {}
    all_novel_rows = []
    purity_rows_all = []
    for entry in entries:
        print(f"[soft] fitting {entry['scene']} ...", flush=True)
        result = fit_scene(model, opt, entry, device, steps=args.steps)
        novel_rows, rendered = evaluate_scene(result, model, opt, entry, device)
        for row in novel_rows:
            step1 = recorded_rows.get((row["scene"], row["view"], row["key"]))
            row["step1_oracle_iou"] = float(step1["oracle_iou"]) if step1 else None
            row["step1_pass"] = bool(step1 and float(step1["oracle_iou"]) >= 0.5)
            # report-only (b): same A and mask>0.5 without the area filter
            index = result["keys"].index(row["key"]) if row["key"] in result["keys"] else None
            row["soft_iou_no_area_gate"] = None
            if index is not None:
                mass = rendered[0, row["view"], index].cpu().numpy()
                sem = result["semantic"][0, row["view"]].cpu().numpy()
                ins = result["instance"][0, row["view"]].cpu().numpy()
                truth = ((sem >= 2) & (sem < 20) & (ins > 0)
                         & (((sem + 1) * 1000 + ins) == row["key"]))
                mask = mass > 0.5
                intersection = int((mask & truth).sum())
                union = int((mask | truth).sum())
                row["soft_iou_no_area_gate"] = intersection / max(1, union)
            all_novel_rows.append(row)
        purity_rows, rest_fraction = purity_filtered_oracle(
            result, model, entry, device, result["keys"]
        )
        for row in purity_rows:
            step1 = recorded_rows.get((row["scene"], row["view"], row["key"]))
            row["step1_oracle_iou"] = float(step1["oracle_iou"]) if step1 else None
        purity_rows_all.extend(purity_rows)
        per_scene[entry["scene"]] = {
            "keys": result["keys"],
            "final_A_cpu": result["final_A"].detach().cpu(),
            "records": result["records"],
            "rest_fraction_step500": float(result["final_A"][:, -1].mean()),
            "purity_rest_fraction": rest_fraction,
            "novel_rows": novel_rows,
            "purity_rows": purity_rows,
        }
        del rendered

    # ---- phase 3/4: pairing, report-only diagnostics and the decision ---- #
    novel_pass_soft = sum(1 for r in all_novel_rows if r["soft_pass"])
    novel_pass_step1 = sum(1 for r in all_novel_rows if r["step1_pass"])
    context_pass = sum(1 for r in all_novel_rows if r["context_iou"] >= 0.5)
    buckets = {}
    for name, predicate in (("small_lt3000", lambda r: r["size_bucket"] == "small"),
                            ("large_ge3000", lambda r: r["size_bucket"] == "large")):
        subset = [r for r in all_novel_rows if predicate(r)]
        buckets[name] = {
            "n": len(subset),
            "step1_pass": sum(1 for r in subset if r["step1_pass"]),
            "soft_pass": sum(1 for r in subset if r["soft_pass"]),
            "context_pass": sum(1 for r in subset if r["context_iou"] >= 0.5),
        }
    purity_pass = sum(1 for r in purity_rows_all if r["purity_pass"])
    no_area_cross = sum(
        1 for r in all_novel_rows
        if not r["soft_pass"] and (r["soft_iou_no_area_gate"] or 0.0) >= 0.5
    )
    decision = {
        "records": len(all_novel_rows),
        "soft_novel_pass": novel_pass_soft, "soft_bar": B1_BAR,
        "context_pass": context_pass, "context_bar": B2_BAR,
        "step1_pass": novel_pass_step1,
        "buckets": buckets,
        "report_only": {
            "purity_filtered_oracle_pass": purity_pass,
            "purity_filtered_rest_fraction_mean": float(np.mean(
                [payload["purity_rest_fraction"] for payload in per_scene.values()]
            )),
            "no_area_gate_crossings": no_area_cross,
        },
    }
    if novel_pass_soft >= B1_BAR:
        decision["branch"] = "B1_capacity_sufficient"
    elif context_pass >= B2_BAR:
        decision["branch"] = "B2_cross_view_gap"
    else:
        decision["branch"] = "B3_representation_footprint_limited"
    per_scene_serialisable = {
        scene: {k: v for k, v in payload.items() if k != "final_A_cpu"}
        for scene, payload in per_scene.items()
    }
    (out_dir / "per_scene.json").write_text(
        json.dumps(per_scene_serialisable, indent=1), encoding="utf-8"
    )
    report["novel_rows"] = all_novel_rows
    report["decision"] = decision
    report["checkpoint_after"] = checkpoint_identity()
    report["checkpoint_unchanged"] = report["checkpoint_before"] == report["checkpoint_after"]
    (out_dir / "manifest.json").write_text(json.dumps(report["hyperparameters"], indent=1),
                                           encoding="utf-8")
    write_csv(out_dir / "per_instance.csv", all_novel_rows,
              extra=[("purity_oracle_iou", { (r["scene"], r["view"], r["key"]): r["purity_oracle_iou"]
                                             for r in purity_rows_all })])
    write_csv(out_dir / "purity_diagnostic.csv", purity_rows_all)
    figures = make_figures(out_dir, per_scene, all_novel_rows, model, opt, entries, device)
    decision["figures"] = figures
    print("[soft] decision:", json.dumps(decision, indent=1), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=str),
                                          encoding="utf-8")
    print(f"[soft] wrote {out_dir}")
    return 0


def write_csv(path: Path, rows, extra=None) -> None:
    if not rows:
        return
    extra_map = dict(extra or [])
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items()
                if isinstance(v, (int, float, str, bool)) or v is None}
        for name, mapping in extra_map.items():
            flat[name] = mapping.get((row["scene"], row["view"], row["key"]))
        flat_rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in flat_rows for k in r}))
        writer.writeheader()
        writer.writerows(flat_rows)


def make_figures(out_dir: Path, per_scene, novel_rows, model, opt, entries, device):
    """Four representative figures chosen by the fixed rules."""
    by_key = {(r["scene"], r["view"], r["key"]): r for r in novel_rows}
    paths = []
    matrix = {(s, r["view"], r["key"]): payload for s, payload in per_scene.items()
              for r in payload["novel_rows"]}
    small_gain = [r for r in novel_rows if r["size_bucket"] == "small"
                  and not r["step1_pass"] and r["soft_pass"]]
    if small_gain:
        first = max(small_gain, key=lambda r: r["soft_iou"])
    else:
        small = [r for r in novel_rows if r["size_bucket"] == "small"]
        first = max(small, key=lambda r: (r["soft_iou"] - (r["step1_oracle_iou"] or 0.0)),
                    default=None)
    large_fail = [r for r in novel_rows if r["size_bucket"] == "large" and not r["soft_pass"]]
    second = min(large_fail, key=lambda r: r["soft_iou"]) if large_fail else None
    third = next((r for r in novel_rows if r["context_iou"] >= 0.5 and not r["soft_pass"]), None)
    fourth = next((r for r in novel_rows
                   if not r["soft_pass"] and not r["step1_pass"]), None)
    for name, row in (("small_soft_success", first), ("large_still_failing", second),
                      ("context_ok_novel_fail", third), ("both_fail", fourth)):
        if row is None:
            continue
        try:
            path = draw_case(out_dir, name, row, per_scene[row["scene"]], entries,
                             device, model, opt)
            paths.append(str(path))
        except Exception as error:  # noqa: BLE001
            print(f"[soft] figure {name} failed: {error}", flush=True)
    return paths


def draw_case(out_dir: Path, name: str, row, scene_payload, entries, device, model, opt):
    entry = next(e for e in entries if e["scene"] == row["scene"])
    batch = entry["batch"]
    result_A = scene_payload.get("final_A_cpu")
    if result_A is None:
        return None
    keys = scene_payload["keys"]
    A = result_A.to(device)
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        soft_mass, _ = render_assignment(
            model, forward["output"]["gaussians"], A, batch["cam_view_all"],
            batch["intrinsics_all"],
        )
        hard = token_oracle(model, opt, entry, device=device, return_masks=True)
    semantic = batch["semantic_label_all"][0].long().cpu().numpy()
    instance = batch["instance_label_all"][0].long().cpu().numpy()
    packed_all = (semantic + 1) * 1000 + instance
    visible_all = ((semantic >= 2) & (semantic < 20) & (instance > 0))
    key = int(row["key"])
    ctx_view, novel_view = 0, int(row["view"])
    ctx_truth = visible_all[ctx_view] & (packed_all[ctx_view] == key)
    ctx_index = keys.index(key) if key in keys else None
    ctx_soft = (soft_mass[0, ctx_view, ctx_index].cpu().numpy() > 0.5
                if ctx_index is not None else np.zeros_like(ctx_truth))
    novel_truth = visible_all[novel_view] & (packed_all[novel_view] == key)
    soft_mask = (soft_mass[0, novel_view, ctx_index].cpu().numpy() > 0.5
                 if ctx_index is not None else np.zeros_like(novel_truth))
    hard_mask = np.zeros_like(novel_truth)
    hard_keys = hard.get("keys_sorted", [])
    if key in hard_keys:
        hard_mask = hard["masks"][0, novel_view, hard_keys.index(key)].cpu().numpy() > 0.5
    rgb = batch["images_all"][0, novel_view].float().cpu().numpy()

    def panel(mask, colour=(1.0, 1.0, 1.0)):
        out = np.zeros(mask.shape + (3,))
        out[mask] = colour
        return out.transpose(2, 0, 1)

    ctx_rgb = batch["images_all"][0, ctx_view].float().cpu().numpy()
    error = np.zeros(novel_truth.shape + (3,))
    error[novel_truth & (~soft_mask)] = [1.0, 0.0, 0.0]
    error[(~novel_truth) & soft_mask] = [0.0, 0.6, 1.0]
    panels = [ctx_rgb, panel(ctx_truth, (0.2, 1.0, 0.2)), panel(ctx_soft),
              rgb, panel(novel_truth, (0.2, 1.0, 0.2)), panel(hard_mask, (1.0, 0.8, 0.2)),
              panel(soft_mask), error.transpose(2, 0, 1)]
    tile = np.concatenate(panels, axis=2)
    image = Image.fromarray((np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
    image = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), f"{row['scene']} frame {row['frame_id']} id {key} | ctx GT | ctx soft | "
                      f"novel RGB | novel GT | hard oracle (IoU {row['step1_oracle_iou']}) | "
                      f"soft oracle (IoU {row['soft_iou']:.3f}) | error", fill=(255, 255, 0))
    path = out_dir / f"{name}_{row['scene']}_id{key}.png"
    image.save(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
