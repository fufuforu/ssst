#!/usr/bin/env python3
"""Read-only per-scene LocusGS spatial-scale audit (fresh anchors, no updates).

For a fixed-seed sample of training scenes and two pairs each it reports, in the
frame the model actually sees (provider normalization, scene_scale=0.15):
  * context depth statistics and camera-trajectory range
  * back-projected visible surface range
  * fresh-anchor -> patch-ray nearest distance, per-query unclamped-ray counts,
    zero-support ratio, and the radii that would give a typical query >= 256
    unclamped rays and >= 95 % of queries >= 1 unclamped ray.
Nothing is trained or written outside the audit directory.
"""
from __future__ import annotations

import argparse
import csv
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
from tokengs.models.canonical_recon_models import patch_plucker_rays  # noqa: E402
from tokengs.models.locusgs_recon import plucker_point_distance  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

ROOT = Path("/space/mawb/SIU3R/data/scannet/train")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=80)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma0", type=float, default=0.1)
    parser.add_argument("--r0", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="/space/mawb/ssst/workspace_recon_diag/locusgs_spatial_scale")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["train_siu3r_locusgs_inferred_v2_gamma_calibrated"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        num_views=4, num_input_views=2, img_size=(256, 256), batch_size=1,
        num_workers=0, seed=args.seed,
    )
    provider = SIU3RProcessedProvider(opt, root=str(ROOT), subset="all", training=True, rank=0)
    names = sorted(s.name for s in provider.dataset.sample_list)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(names), size=min(args.scenes, len(names)), replace=False).tolist())

    # the exact fresh anchor initialization of the model (LocusGSAnchorDecoder.__init__)
    gen = torch.Generator(device="cpu").manual_seed(int(opt.seed) + 23)
    extent, center_z = float(opt.locusgs_anchor_init_extent), float(opt.locusgs_anchor_init_center_z)
    mu = torch.empty(int(opt.num_gs_tokens), 3).uniform_(-extent, extent, generator=gen)
    mu[:, 2] = mu[:, 2] + center_z
    mu = mu.to(device)

    c = float(np.sqrt(40.0)) * float(args.sigma0)
    print(f"[scale] scenes={len(chosen)} pairs/scene={args.pairs} sigma0={args.sigma0} r0={args.r0} c={c:.4f}")
    print(f"[scale] anchors={mu.shape[0]} fresh init range x[{float(mu[:,0].min()):.3f},{float(mu[:,0].max()):.3f}] "
          f"z[{float(mu[:,2].min()):.3f},{float(mu[:,2].max()):.3f}]")

    rows = []
    for n, scene_idx in enumerate(chosen):
        scene = names[scene_idx]
        for pair in range(args.pairs):
            provider.pair_rng.seed(args.seed + 1000 * scene_idx + pair)
            sample = provider[scene_idx]
            frames = [int(x) for x in sample["frame_ids"]]
            ctx = frames[: int(opt.num_input_views)]
            rays_o = sample["rays_os"].to(device)      # [V,3,H,W]
            rays_d = sample["rays_ds"].to(device)
            intr = sample["intrinsics_all"].to(device)  # [V,4]
            cam_view = sample["cam_view_all"].to(device)  # [V,4,4] = w2c^T
            with torch.no_grad():
                moment, direction = patch_plucker_rays(
                    rays_o[None, : int(opt.num_input_views)],
                    rays_d[None, : int(opt.num_input_views)],
                    patch_size=int(opt.patch_size),
                )
                D = plucker_point_distance(mu.unsqueeze(0), moment, direction)[0]   # [N,P]
                threshold = c * args.r0
                unclamped = (D < threshold).sum(-1).float()                # [N]
                d_sorted, _ = torch.sort(D, dim=-1)
                k = 256
                d_256 = d_sorted[:, min(k, d_sorted.shape[1] - 1)]
                r_256 = float(d_256.median()) / c
                r_95 = float(torch.quantile(D.amin(-1), 0.95)) / c
                d_nearest = D.amin(-1)

            # camera trajectory range (normalized frame) and depth statistics
            centers = rays_o[: int(opt.num_input_views), :, 0, 0]          # [Vc,3]
            traj = float(torch.cdist(centers, centers).max())
            depths = []
            pts_canon = []
            for v, frame in enumerate(frames[: int(opt.num_input_views)]):
                d_m = torch.from_numpy(
                    np.asarray(Image.open(ROOT / scene / "depth" / f"{frame}.png")).astype(np.float32) / 1000.0
                ).to(device) * 0.15
                depths.append(d_m[d_m > 1e-6])
                H, W = d_m.shape
                fx, fy, cx, cy = [float(x) for x in intr[v]]
                uu, vv = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
                z = d_m
                valid = z > 1e-6
                pc = torch.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], -1)[valid]
                c2w = torch.inverse(cam_view[v].T)
                pts_canon.append(pc @ c2w[:3, :3].T + c2w[:3, 3])
            depth_all = torch.cat(depths)
            pts = torch.cat(pts_canon)
            extent_vec = pts.max(0).values - pts.min(0).values
            rows.append({
                "scene": scene, "pair": pair, "frames": frames,
                "depth_norm_p10": float(torch.quantile(depth_all, 0.10)),
                "depth_norm_p50": float(torch.quantile(depth_all, 0.50)),
                "depth_norm_p90": float(torch.quantile(depth_all, 0.90)),
                "surface_range": float(torch.linalg.norm(extent_vec)),
                "surface_z_p50": float(torch.quantile(pts[:, 2], 0.50)),
                "surface_z_p90": float(torch.quantile(pts[:, 2], 0.90)),
                "trajectory_range": traj,
                "D_nearest_p50": float(torch.quantile(d_nearest, 0.50)),
                "D_nearest_p95": float(torch.quantile(d_nearest, 0.95)),
                "unclamped_at_r0_median": float(unclamped.median()),
                "unclamped_at_r0_mean": float(unclamped.mean()),
                "zero_support_ratio": float((unclamped == 0).float().mean()),
                "r_for_256_rays": r_256,
                "r_for_95pct_support": r_95,
                "r_required_max_of_two": max(r_256, r_95),
            })
        if (n + 1) % 20 == 0:
            print(f"[scale] processed {n + 1}/{len(chosen)} scenes", flush=True)

    # ---- aggregates -------------------------------------------------------- #
    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    def stats(a):
        return {"p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)), "mean": float(a.mean()), "std": float(a.std()),
                "p90_over_p10": float(np.percentile(a, 90) / max(np.percentile(a, 10), 1e-9))}

    per_scene: dict[str, list] = {}
    for r in rows:
        per_scene.setdefault(r["scene"], []).append(r)
    within = {}
    for key in ("r_for_256_rays", "r_for_95pct_support", "depth_norm_p50", "trajectory_range", "surface_range"):
        stds = [np.std([x[key] for x in v]) for v in per_scene.values() if len(v) > 1]
        within[key] = float(np.mean(stds)) if stds else 0.0
    summary = {
        "n_scenes": len(per_scene), "n_pairs": len(rows),
        "r0": args.r0, "sigma0": args.sigma0, "c": c,
        "threshold_at_r0": c * args.r0,
        "depth_norm_p50": stats(col("depth_norm_p50")),
        "surface_range": stats(col("surface_range")),
        "trajectory_range": stats(col("trajectory_range")),
        "D_nearest_p50": stats(col("D_nearest_p50")),
        "unclamped_at_r0_median": stats(col("unclamped_at_r0_median")),
        "zero_support_ratio": stats(col("zero_support_ratio")),
        "r_for_256_rays": stats(col("r_for_256_rays")),
        "r_for_95pct_support": stats(col("r_for_95pct_support")),
        "r_required_max": stats(col("r_required_max_of_two")),
        "within_scene_std": within,
        "between_scene_std": {k: float(np.std([np.mean([x[k] for x in v]) for v in per_scene.values()]))
                              for k in ("r_for_256_rays", "r_for_95pct_support", "depth_norm_p50",
                                        "trajectory_range", "surface_range")},
    }
    for a, b in (("r_for_256_rays", "depth_norm_p50"), ("r_for_256_rays", "trajectory_range"),
                 ("r_for_256_rays", "surface_range"), ("r_for_95pct_support", "depth_norm_p50"),
                 ("zero_support_ratio", "trajectory_range")):
        x, y = col(a), col(b)
        summary[f"corr({a},{b})"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None

    with open(out_dir / "per_scene.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"{'quantity':>28} | {'p10':>8} {'p50':>8} {'p90':>8} {'p90/p10':>8}")
    for key in ("depth_norm_p50", "surface_range", "trajectory_range", "D_nearest_p50",
                "unclamped_at_r0_median", "zero_support_ratio", "r_for_256_rays",
                "r_for_95pct_support", "r_required_max"):
        s = summary[key]
        print(f"{key:>28} | {s['p10']:>8.4f} {s['p50']:>8.4f} {s['p90']:>8.4f} {s['p90_over_p10']:>8.3f}")
    print()
    for k, v in summary.items():
        if k.startswith("corr("):
            print(f"  {k} = {v:.3f}" if v is not None else f"  {k} = n/a")
    print(f"[scale] wrote {out_dir/'per_scene.csv'} and {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
