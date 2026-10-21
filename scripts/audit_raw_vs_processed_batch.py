#!/usr/bin/env python3
"""Batch-level comparison: raw .sens vs SIU3R-processed ScanNet (read-only).

Builds the *same* model input from both sources (256x256 RGB, pixel K, first-cam
relative C2W, scene_scale=0.15, rays) and reports per-field differences plus
side-by-side images.  No training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
SENS_READER_FILE = "/space/mawb/tokengs/tokengs/data/static/scannet.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.provider import ray_condition  # noqa: E402
from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def _load_sens_reader():
    """Load the .sens reader from the tokengs repo without shadowing `tokengs`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tokengs_sens_reader", SENS_READER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ScanNetSensReader


ScanNetSensReader = _load_sens_reader()

TRAIN_ROOT = "/space/mawb/SIU3R/data/scannet/train"
SENS_ROOT = "/space/mawb/tokengs/data/ScanNet/scans"


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs()
    mse = ((a.float() - b.float()) ** 2).mean().clamp_min(1e-12)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "psnr": float(-10.0 * torch.log10(mse))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--frames", type=int, nargs="+", default=[654, 664, 655, 659])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/space/mawb/ssst/workspace_recon_diag/raw_vs_processed")
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    IMG = 256

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        img_size=(IMG, IMG), batch_size=1, num_workers=0, seed=args.seed,
    )

    # ---------------- processed side (the actual training batch) ------------- #
    provider = SIU3RProcessedProvider(opt, root=TRAIN_ROOT, subset="all", training=True, rank=0)
    names = [s.name for s in provider.dataset.sample_list]
    idx = names.index(args.scene)
    provider.pair_rng.seed(int(opt.seed))
    sample = provider[idx]
    proc_frames = [int(x) for x in sample["frame_ids"]]
    batch = default_collate([sample])
    p_rgb = batch["images_all"][0]                       # [4,3,256,256] in [0,1]
    p_intr = batch["intrinsics_all"][0]                  # [4,4]
    p_cam = batch["cam_view_all"][0]                     # [4,4,4] = w2c^T
    p_ro, p_rd = batch["rays_os"][0], batch["rays_ds"][0]
    print(f"[cmp] processed frames={proc_frames}")
    if proc_frames != args.frames:
        print(f"[cmp] NOTE: provider pair differs from the requested order {args.frames}; "
              f"comparing on the provider's frames")
    frames = proc_frames

    # ---------------- raw .sens side (same pipeline) ------------------------- #
    reader = ScanNetSensReader(Path(SENS_ROOT) / args.scene / f"{args.scene}.sens", frame_stride=1)
    K_raw = torch.tensor(reader.intrinsic_color[:3, :3], dtype=torch.float32)
    sx, sy = IMG / float(reader.color_width), IMG / float(reader.color_height)
    print(f"[cmp] raw color {reader.color_width}x{reader.color_height}, K_raw fx={K_raw[0,0]:.2f} cx={K_raw[0,2]:.2f}, "
          f"scale=({sx:.4f},{sy:.4f}) -> fx={K_raw[0,0]*sx:.2f} cx={K_raw[0,2]*sx:.2f}")
    rgbs, ks, c2ws = [], [], []
    for f in frames:
        img = reader.read_color(f).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
        rgbs.append(torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0)
        ks.append(torch.tensor([[K_raw[0, 0] * sx, 0.0, K_raw[0, 2] * sx],
                                [0.0, K_raw[1, 1] * sy, K_raw[1, 2] * sy],
                                [0.0, 0.0, 1.0]]))
        c2ws.append(reader.get_c2w(f).float())
    r_rgb = torch.stack(rgbs)
    r_K = torch.stack(ks)
    c2ws = torch.stack(c2ws)
    c2ws = torch.inverse(c2ws[0]).unsqueeze(0) @ c2ws          # first-cam relative
    c2ws = c2ws.clone()
    c2ws[:, :3, 3] = c2ws[:, :3, 3] * float(provider.scene_scale)   # scene_scale=0.15
    r_cam = torch.inverse(c2ws).transpose(1, 2)                # same storage as provider
    intr4 = torch.stack([r_K[:, 0, 0], r_K[:, 1, 1], r_K[:, 0, 2], r_K[:, 1, 2]], dim=-1)
    plucker, r_ro, r_rd = ray_condition(intr4[None], c2ws[None], IMG, IMG, device="cpu")
    r_plucker = plucker[0]

    # ---------------- compare ------------------------------------------------- #
    report = {"scene": args.scene, "frames": frames, "image_size": IMG,
              "raw_color_size": [reader.color_width, reader.color_height],
              "per_frame": [], "fields": {}}
    for i, f in enumerate(frames):
        s = stats(p_rgb[i], r_rgb[i])
        report["per_frame"].append({"frame": f, "rgb_psnr": s["psnr"], "rgb_mad": s["mean_abs"],
                                    "rgb_max_abs": s["max_abs"]})
        grid = torch.cat([p_rgb[i:i + 1], r_rgb[i:i + 1], (p_rgb[i] - r_rgb[i]).abs().mean(0, keepdim=True).repeat(3, 1, 1)[None]], dim=-1)[0]
        Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            out / "images" / f"tile_frame{f}.png")
    report["fields"]["rgb_all"] = stats(p_rgb, r_rgb)
    report["fields"]["intrinsics"] = stats(p_intr, intr4)
    report["fields"]["cam_view"] = stats(p_cam, r_cam)
    report["fields"]["rays_os"] = stats(p_ro, r_ro)
    report["fields"]["rays_ds"] = stats(p_rd, r_rd)
    report["fields"]["plucker"] = stats(batch["input"][0, :, -6:], r_plucker.permute(0, 1, 2).reshape(len(frames), 6, IMG, IMG) if False else r_plucker)
    report["processed_intrinsics"] = p_intr.tolist()
    report["raw_intrinsics"] = intr4.tolist()

    print("\n[cmp] per-frame RGB (final 256x256 batch tensors):")
    for e in report["per_frame"]:
        print(f"  frame {e['frame']}: PSNR {e['rgb_psnr']:.2f} dB | MAD {e['rgb_mad']:.4f} | max {e['rgb_max_abs']:.3f}")
    print("\n[cmp] field differences (processed vs raw):")
    for k, v in report["fields"].items():
        print(f"  {k:>10}: max {v['max_abs']:.3e} | mean {v['mean_abs']:.3e} | PSNR {v['psnr']:.2f} dB")
    (out / "raw_vs_processed.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[cmp] wrote {out/'raw_vs_processed.json'} and tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
