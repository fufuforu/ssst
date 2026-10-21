#!/usr/bin/env python3
"""Summarise the paired processed-vs-raw plain-TokenGS reconstruction A/B.

Reads ``<root>/<arm>/<arm>_rows.json`` for the processed and raw arms (either
flat or nested under ``--subdir``) and prints a per-arm table plus the
context/novel PSNR gain over each arm's **own** grey-image GT baseline.  Writes
``summary.json`` next to the rows it consumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path, arm: str, subdir: str | None) -> dict | None:
    candidates = []
    if subdir:
        candidates.append(root / subdir / arm / f"{arm}_rows.json")
    candidates.append(root / arm / f"{arm}_rows.json")
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/space/mawb/ssst/workspace_recon_diag/plain_ab")
    parser.add_argument("--subdir", default="preset")
    parser.add_argument("--label", default="preset lr=1e-4 warmup=1000 total=4000")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root)

    summary: dict = {"label": args.label, "root": str(root), "subdir": args.subdir, "arms": {}}
    for arm in ("processed", "raw"):
        data = load(root, arm, args.subdir)
        if data is None:
            print(f"[sum] missing rows for arm={arm}")
            continue
        rows = data["rows"]
        first, last = rows[0], rows[-1]
        mid = next((r for r in rows if r["step"] == 2000), None)
        summary["arms"][arm] = {
            "pair": data.get("pair"),
            "frames": data.get("frames"),
            "first": first,
            "mid": mid,
            "final": last,
            "ctx_gain_first": first["ctx_psnr"] - first["ctx_grey_psnr"],
            "ctx_gain_final": last["ctx_psnr"] - last["ctx_grey_psnr"],
            "novel_gain_first": first["novel_psnr"] - first["novel_grey_psnr"],
            "novel_gain_final": last["novel_psnr"] - last["novel_grey_psnr"],
            "ctx_gain_best": max(r["ctx_psnr"] - r["ctx_grey_psnr"] for r in rows),
            "novel_gain_best": max(r["novel_psnr"] - r["novel_grey_psnr"] for r in rows),
            "best_ctx_psnr": max(r["ctx_psnr"] for r in rows),
            "best_novel_psnr": max(r["novel_psnr"] for r in rows),
            "rows": rows,
        }

    print(f"[sum] {args.label}")
    header = (
        f"{'arm':>10} {'step':>5} {'loss':>8} {'ctxP':>7} {'novP':>7} "
        f"{'ctxGain':>8} {'novGain':>8} {'ctxSSIM':>8} {'novSSIM':>8} {'alpha':>6}"
    )
    print(header)
    for arm, a in summary["arms"].items():
        points = [a["first"], a["mid"], a["final"]]
        for r in points:
            if r is None:
                continue
            cg = r["ctx_psnr"] - r["ctx_grey_psnr"]
            ng = r["novel_psnr"] - r["novel_grey_psnr"]
            print(f"{arm:>10} {r['step']:>5} {r['loss']:>8.4f} {r['ctx_psnr']:>7.2f} "
                  f"{r['novel_psnr']:>7.2f} {cg:>+8.2f} {ng:>+8.2f} "
                  f"{r['ctx_ssim']:>8.3f} {r['novel_ssim']:>8.3f} {r['alpha_mean']:>6.3f}")
        print(f"{arm:>10}  best  ctx {a['best_ctx_psnr']:.2f} (gain {a['ctx_gain_best']:+.2f}) | "
              f"novel {a['best_novel_psnr']:.2f} (gain {a['novel_gain_best']:+.2f})")

    out = Path(args.out) if args.out else (root / args.subdir / "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sum] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
