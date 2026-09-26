#!/usr/bin/env python3
"""Representative GT | prediction panels for the val_pair reconstruction eval.

Read-only: picks the best / median / worst scene directories by the SIU3R
evaluator's own per-scene mean PSNR and renders one context and one novel view
as `GT rgb | pred rgb | GT depth | pred depth`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def depth_vis(path: Path, vmax: float) -> np.ndarray:
    d = np.asarray(Image.open(path)).astype(np.float32) / 1000.0
    x = np.clip(d / vmax, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    out = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    out[d <= 0] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="best47500")
    args = ap.parse_args()
    root = Path(args.eval_root) / "predictions" / "official_predictions"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    scored = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rs = d / "render_scores.json"
        if rs.is_file():
            items = json.loads(rs.read_text(encoding="utf-8"))
            scored.append((float(np.mean([i["psnr"] for i in items])), d))
    scored.sort(key=lambda t: t[0])
    picks = [("worst", scored[0]), ("median", scored[len(scored) // 2]), ("best", scored[-1])]

    for label, (psnr, scene_dir) in picks:
        scene = scene_dir.name.split("_context")[0]
        frames = sorted(int(p.stem.split("_")[-1]) for p in (scene_dir / "rgb").glob("*.png"))
        for pos, idx in enumerate((0, -1)):        # one context view, one novel view
            f = frames[idx]
            tiles = [np.asarray(Image.open(scene_dir / "rgb_gt" / f"{scene}_{f}.png").convert("RGB")),
                     np.asarray(Image.open(scene_dir / "rgb" / f"{scene}_{f}.png").convert("RGB")),
                     depth_vis(scene_dir / "depth_gt" / f"{scene}_{f}.png", 5.0),
                     depth_vis(scene_dir / "depth" / f"{scene}_{f}.png", 5.0)]
            panel = np.concatenate(tiles, axis=1)
            img = Image.fromarray(panel); d = ImageDraw.Draw(img)
            for j, t in enumerate(["GT rgb", "pred rgb", "GT depth", "pred depth"]):
                d.text((j * 256 + 4, 4), t, fill=(255, 255, 0))
            kind = "context" if pos == 0 else "novel"
            d.text((4, 18), f"{args.tag} {label} {scene_dir.name} frame {f} ({kind}) "
                            f"scene-mean PSNR {psnr:.2f}", fill=(0, 255, 255))
            img.save(out / f"{args.tag}_{label}_{scene_dir.name}_{kind}_f{f}.png")
            print(f"[fig] {label:>6} {scene_dir.name} f{f} ({kind}) scene PSNR {psnr:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
