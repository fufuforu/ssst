#!/usr/bin/env python3
"""structure_probe_v2 §4.B: per-Gaussian assignment oracle (diagnostic only).

Used only if the token-level oracle fails.  Same fixed sample, same frozen G0+
Gaussians and the same loss/GT as the token oracle, but the optimised variable is
one assignment row per *Gaussian* instead of one per token: Z is [N, K+1] with
N = 65536, initialised by replicating the token oracle's final A over each
token's 64 Gaussians.  Adam lr 0.01, weight_decay 0, at most 300 steps.

Only a small summary is stored - never the Z tensor.
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
    scene_loss,
    token_contribution_stats,
)
from scripts.group_eval_v2 import forward_group  # noqa: E402
from scripts.probe_group_capacity import load_g0plus, sample_batch  # noqa: E402

G0PLUS = "workspace_group_plus/arm_g0plus/ckpt_step6000"
MIN_AREA = 50
REQUIRED_SAMPLE_SHA = "98a4d35d97eea33169d2fb4f7346ed7c128c0385a385e223ab36732690bc83df"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_per_gaussian(model, gaussians, A, cam_view, intrinsics):
    rendered = model.gs.render_feature_channels(gaussians, A, cam_view,
                                                intrinsics=intrinsics)
    return rendered["images_pred"], rendered["alphas_pred"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/structure_probe_v2/"
                                        "per_gaussian_oracle.json")
    parser.add_argument("--oracle", default="group_plus/structure_probe_v2/oracle_pair.json",
                        help="token oracle JSON, used only for the initialisation rule")
    parser.add_argument("--sample", default="group_plus/structure_probe_v1/sample.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--log-steps", type=int, nargs="*", default=[0, 50, 150, 300])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    if sample["sha256"] != REQUIRED_SAMPLE_SHA:
        raise SystemExit("sample SHA mismatch")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    opt, model = load_g0plus(args.preset, args.seed, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    batch = sample_batch(opt, split, sample, device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        gaussians = forward["output"]["gaussians"].detach()
        alpha_fixed = forward["masks"]["alpha"].detach()
        semantic = batch["semantic_label_all"].long()
        instance = batch["instance_label_all"].long()
        keys, _ = context_instances(semantic, instance, alpha=alpha_fixed)
        stats = token_contribution_stats(model, opt,
                                         {"batch": batch, "scene": sample["scene"]},
                                         [int(k) for k in keys], device)
    stacked = torch.stack([stats["per_instance"][int(k)] for k in keys], dim=-1)
    best = stacked.argmax(dim=-1)
    has = stacked.max(dim=-1).values > 0
    token_votes = [keys[int(best[t])] if bool(has[t]) else None
                   for t in range(stacked.shape[0])]
    num_tokens = gaussians.shape[1] // 64
    # Initialise from the token oracle's **final** A (saved by the token run),
    # replicated over each token's 64 Gaussians; fall back to the token oracle's
    # own initialisation only if that file is unavailable.
    a_path = Path(args.oracle).parent / "oracle_final_A.pt"
    if a_path.is_file():
        saved = torch.load(a_path, map_location="cpu", weights_only=False)
        if [int(k) for k in saved["keys"]] != [int(k) for k in keys]:
            raise SystemExit("token oracle A was fitted with different instance keys")
        A_init = saved["A"].to(device).repeat_interleave(64, dim=0)
        init_source = f"token oracle final A from {a_path}"
    else:
        Z_token = init_Z(num_tokens, keys, token_votes, device)
        A_init = torch.softmax(Z_token, dim=-1).repeat_interleave(64, dim=0)
        del Z_token
        init_source = "token oracle initialisation (final A file missing)"
    Z = torch.log(A_init.clamp_min(1e-6))
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

    steps = 10 if args.smoke else args.steps
    curve = []
    for step in range(steps + 1):
        if step in args.log_steps or step == steps:
            with torch.no_grad():
                A = torch.softmax(Z.detach(), dim=-1)
                mass, alpha = render_per_gaussian(model, gaussians, A.unsqueeze(0),
                                                  batch["cam_view_all"], batch["intrinsics_all"])
                conservation = float((mass[0].sum(1) - alpha[0, :, 0]).abs().max())
                per_view = {}
                for view in range(4):
                    per_view[str(view)] = {}
                    for index, key in enumerate(keys):
                        hard = mass[0, view, index] > 0.5
                        area = int(hard.sum())
                        truth = gt_masks[int(key)][view]
                        iou = None
                        if bool(truth.any()) and area >= MIN_AREA:
                            iou = float(hard_metrics(hard.cpu(), truth.cpu())["iou"])
                        per_view[str(view)][str(int(key))] = {"area": area, "iou": iou}
            sent = [int(s["packed_key"]) for s in sample["sentinels"]]
            curve.append({
                "step": step, "conservation_max_abs_error": conservation,
                "sentinels_context_best_iou": {
                    str(k): max([per_view[v][str(k)]["iou"] or 0.0 for v in ("0", "1")])
                    for k in sent},
                "sentinels_novel_best_iou": {
                    str(k): max([per_view[v][str(k)]["iou"] or 0.0 for v in ("2", "3")])
                    for k in sent},
                "all_instances_context_best_iou_mean": float(np.mean(
                    [max([per_view[v][str(int(k))]["iou"] or 0.0 for v in ("0", "1")])
                     for k in keys])),
            })
            print(f"[pg] step {step} ctx {curve[-1]['sentinels_context_best_iou']} "
                  f"novel {curve[-1]['sentinels_novel_best_iou']} "
                  f"conserve {conservation:.1e}", flush=True)
        if step == steps or args.smoke:
            break
        optimizer.zero_grad(set_to_none=True)
        A = torch.softmax(Z, dim=-1)
        mass, _ = render_per_gaussian(model, gaussians, A.unsqueeze(0),
                                      batch["cam_view_all"][:, :2],
                                      batch["intrinsics_all"][:, :2])
        losses = scene_loss(mass, gt_masks, keys, alpha_fixed, valid_pixels, stuff, thing)
        if not math.isfinite(float(losses["total"])):
            raise SystemExit(f"non-finite loss at step {step}")
        losses["total"].backward()
        if Z.grad is None or not torch.isfinite(Z.grad).all():
            raise SystemExit(f"non-finite Z gradient at step {step}")
        optimizer.step()

    last = curve[-1]
    sent = [int(s["packed_key"]) for s in sample["sentinels"]]
    per_gaussian_pass = all(last["sentinels_context_best_iou"][str(k)] >= 0.5 for k in sent) \
        and all(last["sentinels_novel_best_iou"][str(k)] >= 0.5 for k in sent)
    payload = {
        "scope": "GT-assisted per-Gaussian assignment oracle; diagnostic only, NOT "
                 "feed-forward and NOT an official metric",
        "sample_sha256": sample["sha256"],
        "config": {"lr": args.lr, "weight_decay": 0, "steps": steps, "seed": args.seed,
                   "variable": "Z [N=65536, K+1]",
        "init": init_source},
        "keys": [int(k) for k in keys], "n_instances": len(keys),
        "curve": curve, "per_gaussian_pass": bool(per_gaussian_pass),
        "checkpoint": {"sha256_before": sha256_file(Path(G0PLUS) / "model.pt")},
        "smoke": bool(args.smoke),
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"per_gaussian_pass": payload["per_gaussian_pass"],
                      "last": last}, indent=1, default=str)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
