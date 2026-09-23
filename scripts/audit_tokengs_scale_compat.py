#!/usr/bin/env python3
"""Read-only scale-compatibility audit: TokenGS training recipe vs SIU3R ScanNet.

Quantifies, in the frame the plain TokenGS baseline actually consumes:
  * processed GT surface depth (camera z) per context view
  * camera baseline / trajectory range of the fixed 2+2 sample
  * the initial Gaussian cloud's camera-coordinate z and its *actual* pixel
    coverage, and the ratio against the scene's depth
No training, no model changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models.activations import ClipActivationHead  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

TRAIN_ROOT = Path("/space/mawb/SIU3R/data/scannet/train")


def q(x, probs=(0.05, 0.10, 0.50, 0.90, 0.95)):
    a = np.asarray(x, dtype=np.float64)
    return {f"p{int(p*100)}": float(np.percentile(a, p * 100)) for p in probs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--scenes", type=int, default=80, help="multi-scene surface statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out",
                        default="/space/mawb/ssst/workspace_recon_diag/plain_tokengs_baseline/scale_audit.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        img_size=(256, 256), batch_size=1, num_workers=0, seed=args.seed,
    )
    provider = SIU3RProcessedProvider(opt, root=str(TRAIN_ROOT), subset="all", training=True, rank=0)
    names = [s.name for s in provider.dataset.sample_list]

    report: dict = {"scene_scale": float(provider.scene_scale),
                    "camera_scale_method": opt.camera_scale_method,
                    "gaussian_z_offset": float(opt.gaussian_z_offset),
                    "img_size": list(opt.img_size)}

    # ---- initial Gaussian cloud (exactly the model's zero-init head output) --- #
    torch.manual_seed(int(opt.seed))
    head = ClipActivationHead(opt)
    tokens = torch.zeros(1, int(opt.num_gs_tokens), int(opt.enc_embed_dim))
    with torch.no_grad():
        g = head(tokens)[0]                      # [N*P, 14]
    pos = g[:, 0:3].clone()
    pos[:, 2] += float(opt.gaussian_z_offset)
    scale = g[:, 4:7]
    report["initial_cloud"] = {
        "xyz_z": q(pos[:, 2]), "xyz_abs_max": float(pos.abs().max()),
        "scale": q(scale), "opacity": q(g[:, 3]), "rgb": q(g[:, 11:14].mean(-1)),
        "radius_world": float(scale.mean()),
    }

    # ---- fixed sample -------------------------------------------------------- #
    idx = names.index(args.scene)
    provider.pair_rng.seed(int(opt.seed))
    sample = provider[idx]
    frames = [int(x) for x in sample["frame_ids"]]
    ctx = frames[: int(opt.num_input_views)]
    intr = sample["intrinsics_all"].numpy()
    cam_view = sample["cam_view_all"].numpy()
    zs, baselines = [], []
    for v, frame in enumerate(frames):
        d = np.asarray(Image.open(TRAIN_ROOT / args.scene / "depth" / f"{frame}.png")).astype(np.float32) / 1000.0
        valid = d > 1e-6
        zs.append(0.15 * d[valid])
    zs_all = np.concatenate(zs)
    centers = np.stack([np.linalg.inv(cam_view[v].T)[:3, 3] for v in range(len(frames))])
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            baselines.append(float(np.linalg.norm(centers[i] - centers[j])))

    # pixel coverage of the initial cloud in the first context view
    c2w = np.linalg.inv(cam_view[0].T)
    w2c = np.linalg.inv(c2w)
    fx, fy, cx, cy = intr[0]
    pw = pos.numpy()
    pc = pw @ w2c[:3, :3].T + w2c[:3, 3]
    front = pc[:, 2] > float(opt.znear)
    zc = np.where(front, pc[:, 2], np.nan)
    u = fx * pc[:, 0] / np.clip(pc[:, 2], 1e-6, None) + cx
    v_ = fy * pc[:, 1] / np.clip(pc[:, 2], 1e-6, None) + cy
    H, W = int(opt.img_size[0]), int(opt.img_size[1])
    inside = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H) & front
    r_px = float(scale.mean()) * fx / max(float(np.nanmedian(zc)), 1e-6)
    report["fixed_sample"] = {
        "scene": args.scene, "context_frames": ctx, "frames": frames,
        "processed_surface_depth": q(zs_all),
        "camera_baseline_pairs": q(baselines),
        "cam_z_of_context_views": [float(centers[i, 2]) for i in range(len(frames))],
        "initial_cloud": {
            "frac_in_frame": float(inside.mean()),
            "frac_in_front": float(front.mean()),
            "median_z_cam": float(np.nanmedian(zc)),
            "u_range": [float(np.nanmin(u[front])), float(np.nanmax(u[front]))],
            "v_range": [float(np.nanmin(v_[front])), float(np.nanmax(v_[front]))],
            "gaussian_radius_px": r_px,
            "covered_pixel_frac_estimate": float(min(1.0, np.pi * r_px ** 2 / (H * W))),
        },
        "compat_ratio_initial_z_over_surface_p50":
            float(np.median(pos[:, 2].numpy()) / np.median(zs_all)),
        "compat_ratio_initial_z_over_surface_p90":
            float(np.median(pos[:, 2].numpy()) / np.percentile(zs_all, 90)),
    }

    # ---- multi-scene surface depth (same 80-scene sample as the other audits) - #
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(names), size=min(args.scenes, len(names)), replace=False).tolist())
    per_scene = []
    for i in idxs:
        provider.pair_rng.seed(args.seed + 1000 * i)
        s = provider[i]
        scene = names[i]
        ds = []
        for v, frame in enumerate([int(x) for x in s["frame_ids"]][: int(opt.num_input_views)]):
            d = np.asarray(Image.open(TRAIN_ROOT / scene / "depth" / f"{frame}.png")).astype(np.float32) / 1000.0
            ds.append(0.15 * d[d > 1e-6])
        ds = np.concatenate(ds)
        per_scene.append({"scene": scene, "p50": float(np.median(ds)), "p90": float(np.percentile(ds, 90))})
    p50 = np.array([p["p50"] for p in per_scene])
    report["multi_scene"] = {
        "n_scenes": len(per_scene),
        "surface_p50_of_scenes": q(p50),
        "scene_median_depth_min": float(p50.min()), "scene_median_depth_max": float(p50.max()),
        "z_offset_over_scene_p50": {k: float(float(opt.gaussian_z_offset) / v)
                                    for k, v in q(p50).items()},
    }

    print(f"[scale] scene_scale={provider.scene_scale} camera_scale_method={opt.camera_scale_method} "
          f"gaussian_z_offset={opt.gaussian_z_offset}")
    print(f"[scale] initial cloud: z {q(pos[:, 2])} | scale mean {float(scale.mean()):.5f} | opacity mean {float(g[:,3].mean()):.4f}")
    print(f"[scale] processed GT surface depth (fixed sample): {q(zs_all)}")
    print(f"[scale] camera baseline (fixed sample, processed units): {q(baselines)}")
    ic = report["fixed_sample"]["initial_cloud"]
    print(f"[scale] initial cloud in view0: in-front {ic['frac_in_front']:.4f} in-frame {ic['frac_in_frame']:.4f} "
          f"| median z_cam {ic['median_z_cam']:.4f} | radius {ic['gaussian_radius_px']:.2f} px "
          f"| covered pixel frac ~{ic['covered_pixel_frac_estimate']:.4f}")
    print(f"[scale] MISMATCH: initial z / surface p50 = {report['fixed_sample']['compat_ratio_initial_z_over_surface_p50']:.2f}x ; "
          f"over p90 = {report['fixed_sample']['compat_ratio_initial_z_over_surface_p90']:.2f}x")
    print(f"[scale] multi-scene ({len(per_scene)} scenes) surface p50 of scenes: {q(p50)} ; "
          f"z_offset/p50 ratio {report['multi_scene']['z_offset_over_scene_p50']}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[wrote] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
