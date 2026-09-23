#!/usr/bin/env python3
"""Final-batch comparison: raw ScanNet `.sens` vs SIU3R-processed ScanNet.

Both groups are pushed through the *real* TokenGS ``Provider``/``ImageTransform``
pipeline, so the numbers reported here are exactly the tensors a training step
would consume.  Nothing is trained and nothing is written outside ``--out-dir``.

For the raw group the RGB is read at native 1296x968 and the shared
``ImageTransform`` performs the centre-crop-to-fill + resize and the matching
K update; the raw K is never hand-scaled.  For the processed group the RGB/K
come from the released SIU3R tree (already 256x256) and the ``ImageTransform``
is the identity (256 -> 256, no crop).

Self-consistency per group is checked by re-deriving the rays from that group's
own pixel K and relative C2W and comparing them with the rays the provider
emitted, and (raw only) by re-deriving the batch K from the native colour K and
the transform that produced the 256x256 RGB.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.provider import ray_condition  # noqa: E402
from tokengs.data.scannet_raw_recon import (  # noqa: E402
    DEFAULT_SCANS_ROOT,
    ScanNetRawReconProvider,
)
from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.data.datafield import DF_FRAME_IDS, DF_SCENE_NAME  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

TRAIN_ROOT = "/space/mawb/SIU3R/data/scannet/train"
IMG = 256


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    mse = ((a - b) ** 2).mean().clamp_min(1e-12)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "psnr": float(-10.0 * torch.log10(mse)),
    }


def build_opt(seed: int):
    return config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        img_size=(IMG, IMG),
        batch_size=1,
        num_workers=0,
        seed=seed,
    )


def extract(batch: dict, num_views: int) -> dict:
    """Pull the tensors a training step consumes out of a collated batch."""
    return {
        "rgb": batch["images_all"][0].clone(),
        "masks": batch["masks_all"][0].clone(),
        "intrinsics": batch["intrinsics_all"][0].clone(),
        "cam_view": batch["cam_view_all"][0].clone(),
        "c2w_input": batch["cam_to_world_input"][0].clone(),
        "rays_os": batch["rays_os"][0].clone(),
        "rays_ds": batch["rays_ds"][0].clone(),
        "plucker": batch["input"][0, :, -6:].clone().permute(0, 2, 3, 1),
        "frame_ids": [int(x) for x in batch[DF_FRAME_IDS][0]],
        "scene": batch[DF_SCENE_NAME][0],
        "num_views": num_views,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--context", type=int, nargs=2, default=[654, 664])
    parser.add_argument("--novel", type=int, nargs=2, default=[655, 659])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scans-root", default=str(DEFAULT_SCANS_ROOT))
    parser.add_argument(
        "--out-dir", default="/space/mawb/ssst/workspace_recon_diag/raw_vs_processed_v2"
    )
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames = [*args.context, *args.novel]

    opt = build_opt(args.seed)
    num_views = int(opt.num_views)

    # ---------------- processed (SIU3R) group -------------------------------- #
    proc_provider = SIU3RProcessedProvider(
        opt, root=TRAIN_ROOT, subset="all", training=True, rank=0
    )
    names = [s.name for s in proc_provider.dataset.sample_list]
    proc_idx = names.index(args.scene)
    proc_provider.pair_rng.seed(int(opt.seed))
    proc_batch = default_collate([proc_provider[proc_idx]])
    proc = extract(proc_batch, num_views)

    # ---------------- raw .sens group --------------------------------------- #
    raw_provider = ScanNetRawReconProvider(
        opt,
        root=args.scans_root,
        scene=args.scene,
        context_frame_ids=tuple(args.context),
        novel_frame_ids=tuple(args.novel),
        training=True,
    )
    raw_batch = default_collate([raw_provider[0]])
    raw = extract(raw_batch, num_views)

    if proc["frame_ids"] != frames:
        raise RuntimeError(
            f"processed order {proc['frame_ids']} != requested {frames}; the SIU3R "
            "pair sampling did not reproduce the fixed record"
        )
    if raw["frame_ids"] != frames:
        raise RuntimeError(f"raw order {raw['frame_ids']} != requested {frames}")

    report: dict = {
        "scene": args.scene,
        "context": args.context,
        "novel": args.novel,
        "frame_order": frames,
        "image_size": IMG,
        "note": (
            "raw RGB is native .sens resolution; the shared ImageTransform does the "
            "centre-crop-to-fill + resize and the matching K update. Neither K is "
            "hand-adjusted."
        ),
        "fields": {},
        "per_frame": [],
        "self_consistency": {},
    }

    # ---------------- per-frame supervised-image difference ----------------- #
    for i, f in enumerate(frames):
        s = stats(proc["rgb"][i], raw["rgb"][i])
        report["per_frame"].append(
            {"frame": f, "rgb_psnr": s["psnr"], "rgb_mad": s["mean_abs"],
             "rgb_max_abs": s["max_abs"]}
        )
        diff = (proc["rgb"][i] - raw["rgb"][i]).abs().mean(0, keepdim=True).repeat(3, 1, 1)
        tile = torch.cat([proc["rgb"][i], raw["rgb"][i], diff.clamp(0, 1)], dim=-1)
        Image.fromarray((tile.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            out / "images" / f"tile_frame{f}.png"
        )

    # a 2x2 sheet of every supervised (GT) frame, processed over raw
    sheet = torch.cat(
        [
            torch.cat([proc["rgb"][0], proc["rgb"][1]], dim=-1),
            torch.cat([proc["rgb"][2], proc["rgb"][3]], dim=-1),
        ],
        dim=-2,
    )
    sheet_raw = torch.cat(
        [
            torch.cat([raw["rgb"][0], raw["rgb"][1]], dim=-1),
            torch.cat([raw["rgb"][2], raw["rgb"][3]], dim=-1),
        ],
        dim=-2,
    )
    side = torch.cat([sheet, sheet_raw], dim=-1)
    Image.fromarray((side.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
        out / "images" / "supervision_processed_vs_raw.png"
    )
    # canonical side-by-side sheet: each row = processed | raw for the 4 frames
    cols = []
    for i in range(num_views):
        cols.append(torch.cat([proc["rgb"][i], raw["rgb"][i]], dim=-1))
    grid = torch.cat([torch.cat(cols[0:2], dim=-2), torch.cat(cols[2:4], dim=-2)], dim=-1)
    Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
        out / "images" / "side_by_side_256.png"
    )

    # ---------------- field differences ------------------------------------- #
    for key in ("rgb", "intrinsics", "cam_view", "rays_os", "rays_ds", "plucker"):
        report["fields"][key] = stats(proc[key], raw[key])

    report["processed_intrinsics"] = proc["intrinsics"].tolist()
    report["raw_intrinsics"] = raw["intrinsics"].tolist()
    report["processed_cam_view"] = proc["cam_view"].tolist()
    report["raw_cam_view"] = raw["cam_view"].tolist()

    # ---------------- self-consistency -------------------------------------- #
    # 1) rays re-derived from the group's own K and relative C2W must equal the
    #    rays the provider emitted (K, pose and rays agree inside each group).
    sc: dict = {}
    for name, group in (("processed", proc), ("raw", raw)):
        c2w = torch.inverse(group["cam_view"].transpose(1, 2))
        plucker, ro, rd = ray_condition(
            group["intrinsics"][None], c2w[None], IMG, IMG, device="cpu"
        )
        sc[name] = {
            "rays_os_rederived_vs_emitted": stats(group["rays_os"], ro),
            "rays_ds_rederived_vs_emitted": stats(group["rays_ds"], rd),
            "plucker_rederived_vs_emitted": stats(
                group["plucker"], plucker.permute(0, 2, 3, 1)
            ),
        }
        # pixel K must equal the K implied by the pipeline geometry
        fx, fy, cx, cy = (float(x) for x in group["intrinsics"][0])
        sc[name]["intrinsics_vec"] = [fx, fy, cx, cy]
        sc[name]["fov_x_deg"] = float(2 * np.degrees(np.arctan(IMG / (2 * fx))))
        sc[name]["fov_y_deg"] = float(2 * np.degrees(np.arctan(IMG / (2 * fy))))

    # 2) raw only: the batch K must be reproducible from the native colour K and
    #    the exact crop+resize the RGB went through.
    reader = raw_provider.dataset.reader
    nat = [float(x) for x in reader.intrinsics]
    ori_h, ori_w = reader.color_height, reader.color_width
    crop_ratio = min(ori_h / IMG, ori_w / IMG)
    new_h = int(IMG * crop_ratio)
    new_w = int(IMG * crop_ratio)
    shift = ((new_w - ori_w) / 2, (new_h - ori_h) / 2)
    scale = (IMG / new_w, IMG / new_h)
    exp = [
        nat[0] * scale[0],
        nat[1] * scale[1],
        (nat[2] + shift[0]) * scale[0],
        (nat[3] + shift[1]) * scale[1],
    ]
    sc["raw"]["native_k"] = nat
    sc["raw"]["native_size_hw"] = [ori_h, ori_w]
    sc["raw"]["transform"] = {"crop_hw": [new_h, new_w], "shift": list(shift), "scale": list(scale)}
    sc["raw"]["expected_batch_k"] = exp
    sc["raw"]["batch_k_matches_transform"] = bool(
        np.allclose(exp, sc["raw"]["intrinsics_vec"], atol=1e-4)
    )
    report["self_consistency"] = sc

    # ---------------- printing ---------------------------------------------- #
    print(f"[cmp] scene={args.scene} frames={frames} order_ok=True")
    print("[cmp] per-frame supervised RGB (final 256x256 tensors), processed vs raw:")
    for e in report["per_frame"]:
        print(f"  frame {e['frame']}: PSNR {e['rgb_psnr']:.2f} dB | "
              f"MAD {e['rgb_mad']:.4f} | max {e['rgb_max_abs']:.3f}")
    print("[cmp] field differences (processed vs raw):")
    for k, v in report["fields"].items():
        print(f"  {k:>10}: max {v['max_abs']:.3e} | mean {v['mean_abs']:.3e} | PSNR {v['psnr']:.2f} dB")
    print("[cmp] pixel K  processed:", [round(x, 3) for x in sc["processed"]["intrinsics_vec"]])
    print("[cmp] pixel K  raw      :", [round(x, 3) for x in sc["raw"]["intrinsics_vec"]])
    print("[cmp] ray self-consistency (rederived vs emitted, max_abs):")
    for name in ("processed", "raw"):
        c = sc[name]
        print(f"  {name}: rays_o {c['rays_os_rederived_vs_emitted']['max_abs']:.2e} | "
              f"rays_d {c['rays_ds_rederived_vs_emitted']['max_abs']:.2e} | "
              f"plucker {c['plucker_rederived_vs_emitted']['max_abs']:.2e}")
    print(f"[cmp] raw batch K reproduces the crop+resize transform: "
          f"{sc['raw']['batch_k_matches_transform']}")

    (out / "raw_vs_processed_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[cmp] wrote {out/'raw_vs_processed_v2.json'} and tiles under {out/'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
