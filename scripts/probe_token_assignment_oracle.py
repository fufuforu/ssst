#!/usr/bin/env python3
"""structure_probe_v2 §3: per-sample token-assignment oracle on the fixed four frames.

Thin wrapper that reuses, without re-implementing:

* ``scripts/probe_group_capacity.sample_batch`` / ``load_g0plus`` for the fixed
  sample frames and the frozen G0+ step6000 checkpoint;
* ``scripts/audit_soft_token_oracle``'s ``context_instances``,
  ``token_contribution_stats``, ``init_Z``, ``render_assignment`` and
  ``scene_loss`` (the same Adam fit, loss and Gaussian compositor as the verified
  phase-0 oracle).  The old CLI's window selection is *not* used: the four frames
  come from ``sample.json`` and the sentinel keys are asserted.

Only the temporary per-scene assignment Z [1024, K+1] is optimised; every model
parameter and every Gaussian stays frozen.  The result is a **GT-assisted,
per-scene fit - not feed-forward and not an official AP/PQ**.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from scripts.audit_mask_failure import hard_metrics  # noqa: E402
from scripts.audit_soft_token_oracle import (  # noqa: E402
    context_instances,
    init_Z,
    render_assignment,
    scene_loss,
    token_contribution_stats,
)
from scripts.group_eval_v2 import forward_group  # noqa: E402
from scripts.probe_group_capacity import (  # noqa: E402
    load_g0plus,
    sample_batch,
)

G0PLUS = "workspace_group_plus/arm_g0plus/ckpt_step6000"
MIN_AREA = 50
REQUIRED_SAMPLE_SHA = "98a4d35d97eea33169d2fb4f7346ed7c128c0385a385e223ab36732690bc83df"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(assignment, gaussians, batch, keys, gt_masks, sentinels, model, views=4):
    """Render the assignment on all four views and score every instance column."""
    with torch.no_grad():
        mass, alpha = render_assignment(
            model, gaussians, assignment,
            batch["cam_view_all"], batch["intrinsics_all"],
        )
        per_instance, per_view = {}, {}
        for view in range(views):
            per_view[str(view)] = {}
            for index, key in enumerate(keys):
                hard = (mass[0, view, index] > 0.5)
                area = int(hard.sum())
                truth = gt_masks.get(int(key))
                has_truth = truth is not None and bool(truth[view].any())
                iou = None
                if has_truth and area >= MIN_AREA:
                    iou = float(hard_metrics(hard.cpu(), truth[view].cpu())["iou"])
                row = {"area": area, "iou": iou,
                       "passes_area": area >= MIN_AREA,
                       "gt_pixels": int(truth[view].sum()) if has_truth else 0}
                per_view[str(view)][str(int(key))] = row
                if str(key) not in per_instance:
                    per_instance[str(int(key))] = {}
                per_instance[str(int(key))][str(view)] = row
        # mass is [B, V, K+1, H, W]; the K instance columns plus the rest column
        # (i.e. all K+1 columns) must reproduce the rendered alpha of every view.
        rest_mass = mass[0, :, len(keys)]
        conservation = float((mass[0].sum(1) - alpha[0, :, 0]).abs().max())
        rest_stats = {"mean": float(rest_mass.mean()),
                      "mean_on_stuff": None, "mean_on_thing": None}
    return {"per_view": per_view, "per_instance": per_instance,
            "conservation_max_abs_error": conservation, "rest": rest_stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/structure_probe_v2/oracle_pair.json")
    parser.add_argument("--sample", default="group_plus/structure_probe_v1/sample.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--log-steps", type=int, nargs="*",
                        default=[0, 50, 100, 200, 300, 400, 500])
    parser.add_argument("--smoke", action="store_true",
                        help="10 steps only, no full JSON")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    checks = {
        "sample_sha_matches_required": sample["sha256"] == REQUIRED_SAMPLE_SHA,
        "sample_frames": sample["frames"] == [209, 253, 215, 247],
        "sentinel_keys": [s["packed_key"] for s in sample["sentinels"]] == [18032, 20030],
        "sentinel_classes": [s["internal_class"] for s in sample["sentinels"]] == [17, 19],
    }
    if not all(checks.values()):
        raise SystemExit(f"sample assertions failed: {checks}")

    checkpoint_sha_before = sha256_file(Path(G0PLUS) / "model.pt")
    checkpoint_mtime_before = (Path(G0PLUS) / "model.pt").stat().st_mtime
    steps = 10 if args.smoke else args.steps
    log_steps = [s for s in args.log_steps if s <= steps] or [0]
    if steps not in log_steps:
        log_steps = sorted(set(log_steps) | {steps})

    opt, model = load_g0plus(args.preset, args.seed, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    batch = sample_batch(opt, split, sample, device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        gaussians = forward["output"]["gaussians"].detach()
        alpha_fixed = forward["masks"]["alpha"].detach()
        rgb_fixed = forward["output"]["render"]["images_pred"].detach().clone()
        depth_fixed = forward["output"]["render"]["depths_pred"].detach().clone()
        semantic = batch["semantic_label_all"].long()
        instance = batch["instance_label_all"].long()
        keys, _ = context_instances(semantic, instance, alpha=alpha_fixed)
        stats = token_contribution_stats(model, opt,
                                         {"batch": batch, "scene": sample["scene"]},
                                         [int(k) for k in keys], device)
    sentinel_keys = [int(s["packed_key"]) for s in sample["sentinels"]]
    if not set(sentinel_keys) <= set(int(k) for k in keys):
        raise SystemExit(f"sentinels {sentinel_keys} are not in the context instances "
                         f"{[int(k) for k in keys]}")
    stacked = torch.stack([stats["per_instance"][int(k)] for k in keys], dim=-1)
    best = stacked.argmax(dim=-1)
    has = stacked.max(dim=-1).values > 0
    token_votes = [keys[int(best[t])] if bool(has[t]) else None
                   for t in range(stacked.shape[0])]
    num_tokens = gaussians.shape[1] // 64
    Z = init_Z(num_tokens, keys, token_votes, device)
    Z.requires_grad_(True)
    optimizer = torch.optim.Adam([Z], lr=args.lr, weight_decay=0)

    valid_pixels = (semantic[0, :2] != 255) & (alpha_fixed[0, :2, 0] > 0.5)
    stuff = ((semantic[0, :2] == 0) | (semantic[0, :2] == 1)) & valid_pixels
    thing = ((semantic[0, :2] >= 2) & (semantic[0, :2] < 20)
             & (instance[0, :2] > 0) & valid_pixels)
    gt_masks = {}
    for key in keys:
        packed = (semantic[0] + 1) * 1000 + instance[0]
        gt_masks[int(key)] = valid_pixels.new_zeros((4,) + valid_pixels.shape[1:])
        for view in range(4):
            gt_masks[int(key)][view] = (
                (semantic[0, view] >= 2) & (semantic[0, view] < 20)
                & (instance[0, view] > 0) & (packed[view] == int(key))
                & (alpha_fixed[0, view, 0] > 0.5))

    curve = []
    assignment_final = None
    grad_norm_first = None
    for step in range(steps + 1):
        if step in log_steps:
            with torch.no_grad():
                A = torch.softmax(Z.detach(), dim=-1)
                context_mass, _ = render_assignment(
                    model, gaussians, A, batch["cam_view_all"][:, :2],
                    batch["intrinsics_all"][:, :2])
                losses = scene_loss(context_mass, gt_masks, keys, alpha_fixed,
                                    valid_pixels, stuff, thing)
            scored = evaluate(A, gaussians, batch, keys, gt_masks, sentinel_keys,
                              model)
            row = {"step": step, "loss": float(losses["total"]),
                   "loss_instance": float(losses["instance"]),
                   "loss_rest": float(losses["rest"]),
                   "assignment_entropy": float(
                       -(A.clamp_min(1e-8).log() * A).sum(-1).mean()),
                   "rest_fraction": float(A[:, -1].mean()),
                   "n_instances": len(keys),
                   "conservation_max_abs_error": scored["conservation_max_abs_error"],
                   "sentinels": {str(k): {
                       v: scored["per_instance"][str(k)][v] for v in
                       ("0", "1", "2", "3")} for k in sentinel_keys},
                   "sentinels_context_best_iou": {
                       str(k): max([scored["per_instance"][str(k)][v]["iou"] or 0.0
                                    for v in ("0", "1")]) for k in sentinel_keys},
                   "sentinels_novel_best_iou": {
                       str(k): max([scored["per_instance"][str(k)][v]["iou"] or 0.0
                                    for v in ("2", "3")]) for k in sentinel_keys},
                   "all_instances_context_best_iou": {
                       str(int(k)): max([scored["per_instance"][str(int(k))][v]["iou"] or 0.0
                                         for v in ("0", "1")]) for k in keys},
                   "all_instances_novel_best_iou": {
                       str(int(k)): max([scored["per_instance"][str(int(k))][v]["iou"] or 0.0
                                         for v in ("2", "3")]) for k in keys}}
            curve.append(row)
            print(f"[oracle] step {step} loss {row['loss']:.4f} "
                  f"ctx {row['sentinels_context_best_iou']} "
                  f"novel {row['sentinels_novel_best_iou']} "
                  f"conserve {row['conservation_max_abs_error']:.1e}", flush=True)
            assignment_final = A
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        A = torch.softmax(Z, dim=-1)
        context_mass, _ = render_assignment(model, gaussians, A,
                                            batch["cam_view_all"][:, :2],
                                            batch["intrinsics_all"][:, :2])
        losses = scene_loss(context_mass, gt_masks, keys, alpha_fixed,
                            valid_pixels, stuff, thing)
        if not math.isfinite(float(losses["total"])):
            raise SystemExit(f"non-finite loss at step {step}")
        losses["total"].backward()
        if Z.grad is None or not torch.isfinite(Z.grad).all():
            raise SystemExit(f"non-finite Z gradient at step {step}")
        if step == 0:
            grad_norm_first = float(Z.grad.norm())
            if grad_norm_first <= 0:
                raise SystemExit("Z gradient is zero at step 0; the oracle is not learning")
        optimizer.step()

    with torch.no_grad():
        forward_after = forward_group(model, batch, opt)
    invariance = {
        "rgb_max_abs_diff": float((rgb_fixed
                                   - forward_after["output"]["render"]["images_pred"]
                                   ).abs().max()),
        "depth_max_abs_diff": float((depth_fixed
                                     - forward_after["output"]["render"]["depths_pred"]
                                     ).abs().max()),
        "gaussians_max_abs_diff": float((gaussians
                                         - forward_after["output"]["gaussians"]).abs().max()),
    }
    checkpoint_sha_after = sha256_file(Path(G0PLUS) / "model.pt")
    checkpoint_mtime_after = (Path(G0PLUS) / "model.pt").stat().st_mtime
    last = curve[-1]
    oracle_pair_pass = all(last["sentinels_context_best_iou"][str(k)] >= 0.5
                           for k in sentinel_keys) and \
        all(last["sentinels_novel_best_iou"][str(k)] >= 0.5 for k in sentinel_keys)
    payload = {
        "scope": "GT-assisted per-scene token-assignment oracle; NOT feed-forward and "
                 "NOT an official AP/PQ result",
        "sample": {k: sample[k] for k in ("scene", "frames", "context", "novel", "sha256")},
        "sample_checks": checks,
        "config": {"lr": args.lr, "weight_decay": 0, "steps": steps, "seed": args.seed,
                   "optimizer_variable": "Z [1024, K+1] only",
                   "loss": "scene_loss: 5*mean BCE (GT-positive pixels only) + 5*mean Dice "
                           "(whole image) + 1.0*class-balanced rest BCE - the previous "
                           "oracle's existing implementation, not a full-pixel BCE"},
        "keys": [int(k) for k in keys],
        "sentinel_keys": sentinel_keys,
        "sentinels": sample["sentinels"],
        "n_instances": len(keys),
        "grad_norm_first_step": grad_norm_first,
        "curve": curve,
        "invariance": invariance,
        "checkpoint": {"path": G0PLUS, "sha256_before": checkpoint_sha_before,
                       "sha256_after": checkpoint_sha_after,
                       "mtime_before": checkpoint_mtime_before,
                       "mtime_after": checkpoint_mtime_after,
                       "unchanged": (checkpoint_sha_before == checkpoint_sha_after
                                     and checkpoint_mtime_before == checkpoint_mtime_after)},
        "oracle_pair_pass": bool(oracle_pair_pass),
        "pass_rule": "both sentinels raw IoU >= 0.5 in at least one context view AND in at "
                     "least one novel view",
        "smoke": bool(args.smoke),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    # The final A is tiny ([1024, K+1], ~20 kB) and is the documented
    # initialisation for the per-Gaussian follow-up; Z itself is never stored.
    if not args.smoke and assignment_final is not None:
        a_path = out.parent / "oracle_final_A.pt"
        torch.save({"A": assignment_final.detach().cpu(),
                    "keys": [int(k) for k in keys],
                    "sample_sha256": sample["sha256"],
                    "steps": steps, "seed": args.seed}, a_path)
        payload["final_A_path"] = str(a_path)
        payload["final_A_bytes"] = a_path.stat().st_size
        out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(f"[oracle] final A saved to {a_path} ({a_path.stat().st_size} bytes)", flush=True)
    print(json.dumps({"oracle_pair_pass": payload["oracle_pair_pass"],
                      "invariance": invariance,
                      "checkpoint_unchanged": payload["checkpoint"]["unchanged"],
                      "n_instances": len(keys)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
