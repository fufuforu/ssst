#!/usr/bin/env python3
"""Map SIU3R-processed ScanNet frame ids to raw .sens frame ids (read-only).

Uses the existing `ScanNetSensReader` (frame_stride=1) and compares downsampled
RGB plus the camera-to-world matrix of each candidate raw frame against the
processed `color/<id>.jpg` / `extrinsic/<id>.txt` files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SENS_REPO = "/space/mawb/tokengs"
if SENS_REPO not in sys.path:
    sys.path.insert(0, SENS_REPO)

from tokengs.data.static.scannet import ScanNetSensReader  # noqa: E402


def rgb_diff(a: Image.Image, b: Image.Image, size: int = 64) -> tuple[float, float]:
    """Return (PSNR, mean abs diff) between two images resized to size x size."""
    aa = np.asarray(a.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    bb = np.asarray(b.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    mse = float(np.mean((aa - bb) ** 2))
    return (-10.0 * np.log10(max(mse, 1e-12))), float(np.mean(np.abs(aa - bb)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--processed-root", default="/space/mawb/SIU3R/data/scannet/train")
    parser.add_argument("--sens-root", default="/space/mawb/tokengs/data/ScanNet/scans")
    parser.add_argument("--frames", type=int, nargs="+", default=[654, 664, 655, 659])
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    sens_path = Path(args.sens_root) / args.scene / f"{args.scene}.sens"
    reader = ScanNetSensReader(sens_path, frame_stride=1)
    print(f"[map] {sens_path.name}: num_raw_frames={reader.num_raw_frames} "
          f"invalid_pose={len(reader.invalid_pose_frame_ids)} usable(stride=1)={len(reader.frame_ids)}")
    proc = Path(args.processed_root) / args.scene
    report = {"scene": args.scene, "num_raw_frames": reader.num_raw_frames,
              "invalid_pose_frames": len(reader.invalid_pose_frame_ids),
              "usable_frames": len(reader.frame_ids), "matches": {}}

    for pid in args.frames:
        p_img = proc / "color" / f"{pid}.jpg"
        p_ext = proc / "extrinsic" / f"{pid}.txt"
        if not p_img.is_file():
            print(f"[map] processed {pid}: MISSING {p_img}")
            continue
        target = Image.open(p_img)
        tgt_c2w = np.loadtxt(p_ext) if p_ext.is_file() else None
        rows = []
        for cand in range(pid - args.window, pid + args.window + 1):
            if not (0 <= cand < reader.num_raw_frames):
                continue
            try:
                raw = reader.read_color(cand)
                psnr, mad = rgb_diff(target, raw)
                c2w = reader.get_c2w(cand).numpy() if tgt_c2w is not None else None
                rot_err = float(np.abs(c2w[:3, :3] - tgt_c2w[:3, :3]).max()) if c2w is not None else float("nan")
                tr_err = float(np.abs(c2w[:3, 3] - tgt_c2w[:3, 3]).max()) if c2w is not None else float("nan")
            except Exception as error:  # frame failed to decode
                psnr, mad, rot_err, tr_err = float("-inf"), float("nan"), float("nan"), float("nan")
                print(f"[map]   candidate {cand}: decode failed ({type(error).__name__})")
            rows.append({"raw_frame": cand, "rgb_psnr": psnr, "rgb_mad": mad,
                         "c2w_rot_maxdiff": rot_err, "c2w_trans_maxdiff": tr_err})
        best = max(rows, key=lambda r: r["rgb_psnr"])
        report["matches"][str(pid)] = {"best": best, "candidates": rows}
        print(f"[map] processed {pid} -> best raw {best['raw_frame']} "
              f"(rgb PSNR {best['rgb_psnr']:.2f} dB, MAD {best['rgb_mad']:.4f}, "
              f"c2w rot {best['c2w_rot_maxdiff']:.2e}, trans {best['c2w_trans_maxdiff']:.2e})")
        top = sorted(rows, key=lambda r: -r["rgb_psnr"])[:3]
        print("       top3: " + " | ".join(f"{r['raw_frame']}:{r['rgb_psnr']:.1f}dB" for r in top))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[wrote] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
