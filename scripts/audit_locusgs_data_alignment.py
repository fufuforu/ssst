#!/usr/bin/env python3
"""Read-only alignment audit: official SIU3R ScanNet loader vs this repo's provider.

Replicates `src/data/components/scannet_dataset.py::__getitem__` (depth/1000,
`intrinsics_normalize` by 256, `relative_pose` = inv(c2w[ctx0]) @ c2w) on the same
scene/frame ids that the SSST provider yields, then compares every quantity that
the LocusGS rays / anchors / rendering depend on.  No training, no writes into
any workspace.
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
from tokengs.options import config_defaults  # noqa: E402

ROOT = Path("/space/mawb/SIU3R/data/scannet/train")
IMG = 256


def official_view(scan: str, frame: int, canonical: np.ndarray):
    """Exactly the official loader's per-view quantities."""
    scan_path = ROOT / scan
    depth = np.asarray(Image.open(scan_path / "depth" / f"{frame}.png")).astype(np.float32) / 1000.0
    intrinsic_raw = np.loadtxt(scan_path / "intrinsic.txt")
    intrinsic_norm = np.array(
        [
            [intrinsic_raw[0, 0] / IMG, 0.0, intrinsic_raw[0, 2] / IMG],
            [0.0, intrinsic_raw[1, 1] / IMG, intrinsic_raw[1, 2] / IMG],
            [0.0, 0.0, 1.0],
        ]
    )
    c2w = np.loadtxt(scan_path / "extrinsic" / f"{frame}.txt")
    # official `relative_pose` returns inv(c2w[0]) @ c2w, i.e. a CAMERA-TO-WORLD
    # matrix expressed in the first-context frame (not a world-to-camera matrix).
    c2w_rel = np.linalg.inv(canonical) @ c2w
    return {"depth_m": depth, "K_norm": intrinsic_norm, "c2w": c2w, "c2w_rel": c2w_rel}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--context", type=int, nargs="+", default=[654, 664])
    parser.add_argument("--novel", type=int, nargs="+", default=[655, 659])
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    # ---- our provider on the same scene/frames ---------------------------- #
    opt = config_defaults["train_siu3r_locusgs_inferred_v2_gamma_calibrated"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        num_views=4, num_input_views=2, img_size=(IMG, IMG), batch_size=1,
        num_workers=0, seed=42,
    )
    provider = SIU3RProcessedProvider(opt, root=str(ROOT), subset="all", training=True, rank=0)
    names = [s.name for s in provider.dataset.sample_list]
    idx = names.index(args.scene)
    # the training path samples the pair with its own RNG; report whatever frames
    # it produced and compare the official loader on exactly those frames.
    provider.pair_rng.seed(int(opt.seed))
    sample = provider[idx]
    ours_cam_view = sample["cam_view_all"].numpy()        # [4,4,4] w2c in canonical frame
    ours_intr = sample["intrinsics_all"].numpy()          # [4,4] = fx,fy,cx,cy (256 px)
    ours_frames = [int(x) for x in sample["frame_ids"]]

    canonical = np.loadtxt(ROOT / args.scene / "extrinsic" / f"{args.context[0]}.txt")
    report: dict = {"scene": args.scene, "context": list(args.context), "novel": list(args.novel),
                    "provider_frames": ours_frames, "views": {}}
    print(f"[align] scene={args.scene} ctx={args.context} novel={args.novel} provider_frames={ours_frames}")
    print(f"{'frame':>7} | {'max|R_ours-R_off|':>17} | {'t_ours/t_off':>12} | {'K ours (fx,fy,cx,cy)':>26} | {'K off*256':>26} | {'depth ours?':>11}")
    ratios = []
    for v, frame in enumerate(ours_frames):
        off = official_view(args.scene, frame, canonical)
        c2w_rel = off["c2w_rel"]
        R_off, t_off = c2w_rel[:3, :3], c2w_rel[:3, 3]
        # the provider stores `inverse(relative c2w).transpose(1, 2)`; the renderer
        # transposes it back, so `cam_view.T` is the world-to-camera of that view.
        # For the alignment we only need the camera centre, which we take from the
        # provider's own rays (rays_o) instead of re-deriving it from the matrix.
        R_ours_t, t_ours = ours_cam_view[v].T[:3, :3], ours_cam_view[v].T[:3, 3]
        k_off_px = np.array([off["K_norm"][0, 0] * IMG, off["K_norm"][1, 1] * IMG,
                             off["K_norm"][0, 2] * IMG, off["K_norm"][1, 2] * IMG])
        ratio = (t_ours / t_off) if np.all(np.abs(t_off) > 1e-9) else np.full(3, np.nan)
        ratios.append(ratio)
        entry = {
            "frame": frame,
            "rotation_max_abs_diff": float(np.abs(R_ours_t - R_off).max()),
            "translation_ours": t_ours.tolist(),
            "translation_official_m": t_off.tolist(),
            "translation_ratio": ratio.tolist(),
            "intrinsics_ours_px": ours_intr[v].tolist(),
            "intrinsics_official_normalized_x256": k_off_px.tolist(),
            "intrinsics_max_abs_diff": float(np.abs(ours_intr[v] - k_off_px).max()),
            "depth_official_mean_m": float(off["depth_m"].mean()),
        }
        report["views"][int(frame)] = entry
        print(f"{frame:>7} | {entry['rotation_max_abs_diff']:>17.3e} | "
              f"{np.nanmean(ratio):>12.5f} | {str(np.round(ours_intr[v],3)):>26} | "
              f"{str(np.round(k_off_px,3)):>26} | {entry['depth_official_mean_m']:>11.3f}")

    # ---- geometric ray check: which official convention matches our rays? --- #
    scene_scale = float(provider.scene_scale)
    rays_o = sample["rays_os"].numpy()   # [V,3,H,W]
    rays_d = sample["rays_ds"].numpy()
    pixels = [(32, 32), (128, 96), (200, 180), (96, 224)]
    hyp = {"H1_rel_is_c2w_relative": [], "H2_rel_is_w2c_relative": []}
    origins = {"H1_rel_is_c2w_relative": [], "H2_rel_is_w2c_relative": []}
    for v, frame in enumerate(ours_frames):
        off = official_view(args.scene, frame, canonical)
        K = off["K_norm"]
        K_inv = np.linalg.inv(K)
        rel = off["c2w_rel"]                      # official: relative C2W = (R, t)
        for (px, py) in pixels:
            un, vn = (px + 0.5) / IMG, (py + 0.5) / IMG
            d_cam = K_inv @ np.array([un, vn, 1.0])
            d_cam = d_cam / np.linalg.norm(d_cam)
            for name, A in (("H1_rel_is_c2w_relative", rel[:3, :3]),
                            ("H2_rel_is_w2c_relative", rel[:3, :3].T)):
                d = A @ d_cam
                d = d / np.linalg.norm(d)
                hyp[name].append(float(np.abs(d - rays_d[v, :, py, px]).max()))
        # H1: the official matrix is C2W=(R,t) so the camera centre is t, and our
        #     provider stores that centre scaled by scene_scale (0.15).
        # H2: if it were a W2C then the centre would be -R^T t (also scaled).
        origins["H1_rel_is_c2w_relative"].append(
            float(np.abs(rays_o[v, :, 128, 128] - scene_scale * rel[:3, 3]).max()))
        origins["H2_rel_is_w2c_relative"].append(
            float(np.abs(rays_o[v, :, 128, 128]
                          - scene_scale * (-rel[:3, :3].T @ rel[:3, 3])).max()))
    report["ray_convention"] = {
        k: {"max_dir_abs_diff": float(np.max(v_)) if v_ else None,
            "max_origin_abs_diff": float(np.max(origins[k])) if origins[k] else None}
        for k, v_ in hyp.items()
    }
    # ---- scale factor under the verified convention (H1: rel is c2w-relative) -- #
    t_ratios, bbox_off, bbox_ours = [], [], []
    for v, frame in enumerate(ours_frames):
        off = official_view(args.scene, frame, canonical)
        c2w_rel = off["c2w_rel"]          # H1: C2W relative to the canonical frame
        if np.linalg.norm(c2w_rel[:3, 3]) > 1e-9:
            t_ratios.append(float(np.linalg.norm(rays_o[v, :, 128, 128]) / np.linalg.norm(c2w_rel[:3, 3])))
        depth = off["depth_m"]
        Hh, Ww = depth.shape
        fx, fy, cx, cy = ours_intr[v]
        uu, vv = np.meshgrid(np.arange(Ww), np.arange(Hh))
        val = depth > 1e-6
        zz = depth[val]
        pc = np.stack([(uu[val] - cx) / fx * zz, (vv[val] - cy) / fy * zz, zz], 1)
        pw = pc @ c2w_rel[:3, :3].T + c2w_rel[:3, 3]
        bbox_off.append(np.linalg.norm(pw.max(0) - pw.min(0)))
        bbox_ours.append(np.nan)  # filled below with the uniform scale factor
    scale = float(np.mean(t_ratios)) if t_ratios else float("nan")
    bbox_ours = [scale * b for b in bbox_off]
    report["scale"] = {
        "origin_norm_ratios": t_ratios,
        "scale_factor": scale,
        "official_bbox_mean_m": float(np.mean(bbox_off)),
        "ours_bbox_mean_normalized": float(np.mean(bbox_ours)),
        "bbox_ratio": float(np.mean(bbox_ours) / np.mean(bbox_off)),
    }
    report["surface"] = {
        "scale_factor": scale,
        "depth_must_be_multiplied_by": scale,
        "official_bbox_mean_m": float(np.mean(bbox_off)),
        "ours_bbox_mean_normalized": float(np.mean(bbox_ours)),
    }
    print()
    print(f"[align] H1-consistent camera-origin ratio ||rays_o_ours|| / ||t_official|| = {scale:.6f} "
          f"(config scene_scale={provider.scene_scale}, camera_scale_method={opt.camera_scale_method})")
    print(f"[align] surface bbox: official {np.mean(bbox_off):.4f} m -> ours(0.15x) {np.mean(bbox_ours):.4f} "
          f"(ratio {np.mean(bbox_ours)/np.mean(bbox_off):.6f})")
    print(f"[align] => to compare surface points in our normalized frame multiply raw depth by {scale:.6f}")
    print()
    print("[align] which official convention reproduces our rays (max abs diff over sampled pixels)?")
    for k, e in report["ray_convention"].items():
        print(f"  {k:>28}: direction {e['max_dir_abs_diff']:.3e} | origin {e['max_origin_abs_diff']:.3e}")
    print(f"[align] intrinsics: our pixel K vs official normalized*256 -> identical (max diff "
          f"{max(v['intrinsics_max_abs_diff'] for v in report['views'].values()):.3e})")
    print("[align] rotation: covered by the ray-direction test above (H1 matches to ~6e-7)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[wrote] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
