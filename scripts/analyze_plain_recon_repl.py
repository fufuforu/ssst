#!/usr/bin/env python3
"""Aggregate the plain-TokenGS processed-vs-raw replication study.

Reads ``<root>/<rep>/<arm>/<arm>_rows.json`` for every replicate directory and
reports, per arm, the best and final PSNR against that arm's own grey baseline,
plus the paired (processed - raw) difference per replicate and the same-seed
repeat spread that measures pure run-to-run non-determinism.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load(root: Path, rep: str, arm: str) -> dict | None:
    path = root / rep / arm / f"{arm}_rows.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    best = max(rows, key=lambda r: r["ctx_psnr"] - r["ctx_grey_psnr"])
    peak_nov = max(rows, key=lambda r: r["novel_psnr"] - r["novel_grey_psnr"])
    final = rows[-1]
    return {
        "best_ctx": best["ctx_psnr"], "best_ctx_gain": best["ctx_psnr"] - best["ctx_grey_psnr"],
        "best_ctx_step": best["step"], "best_ctx_ssim": best["ctx_ssim"],
        "best_novel": peak_nov["novel_psnr"],
        "best_novel_gain": peak_nov["novel_psnr"] - peak_nov["novel_grey_psnr"],
        "best_novel_step": peak_nov["step"],
        "grey_ctx": final["ctx_grey_psnr"], "grey_novel": final["novel_grey_psnr"],
        "final_ctx": final["ctx_psnr"], "final_ctx_gain": final["ctx_psnr"] - final["ctx_grey_psnr"],
        "final_novel": final["novel_psnr"], "final_alpha": final["alpha_mean"],
        "final_ctx_ssim": final["ctx_ssim"], "final_novel_ssim": final["novel_ssim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/space/mawb/ssst/workspace_recon_diag/plain_ab/repl")
    parser.add_argument("--reps", nargs="+",
                        default=["seed42", "seed42b", "seed43", "seed44", "seed45", "seed46"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root)

    table: dict[str, dict] = {}
    for rep in args.reps:
        for arm in ("processed", "raw"):
            rec = load(root, rep, arm)
            if rec:
                table.setdefault(rep, {})[arm] = rec

    print(f"{'replicate':>10} {'arm':>10} {'greyC':>6} {'bestC':>7} {'dCtxB':>7} {'@step':>6} "
          f"{'bestN':>7} {'dNovB':>7} {'bestSS':>7} {'finC':>7} {'dCtxF':>7} {'alphaF':>7}")
    for rep, arms in table.items():
        for arm, r in arms.items():
            print(f"{rep:>10} {arm:>10} {r['grey_ctx']:>6.2f} {r['best_ctx']:>7.2f} "
                  f"{r['best_ctx_gain']:>+7.2f} {r['best_ctx_step']:>6} {r['best_novel']:>7.2f} "
                  f"{r['best_novel_gain']:>+7.2f} {r['best_ctx_ssim']:>7.3f} {r['final_ctx']:>7.2f} "
                  f"{r['final_ctx_gain']:>+7.2f} {r['final_alpha']:>7.3f}")

    print()
    for metric in ("best_ctx_gain", "final_ctx_gain", "best_ctx_ssim"):
        print(f"--- {metric} ---")
        for arm in ("processed", "raw"):
            vals = [arms[arm][metric] for arms in table.values() if arm in arms]
            if vals:
                print(f"  {arm:>10}: n={len(vals)} mean={st.mean(vals):+.4f} "
                      f"sd={st.pstdev(vals):.4f} min={min(vals):+.4f} max={max(vals):+.4f}")
        paired = []
        for rep, arms in table.items():
            if "processed" in arms and "raw" in arms:
                paired.append(arms["raw"][metric] - arms["processed"][metric])
        if paired:
            print(f"  paired raw-processed (n={len(paired)}): "
                  f"mean={st.mean(paired):+.4f} values={[round(v, 3) for v in paired]}")

    if "seed42" in table and "seed42b" in table:
        print()
        print("--- same-seed (42) repeat spread = pure run-to-run non-determinism ---")
        for arm in ("processed", "raw"):
            a, b = table["seed42"].get(arm), table["seed42b"].get(arm)
            if a and b:
                print(f"  {arm:>10}: best_ctx {a['best_ctx']:.3f} vs {b['best_ctx']:.3f} "
                      f"(diff {abs(a['best_ctx'] - b['best_ctx']):.3f}) | "
                      f"final_ctx {a['final_ctx']:.3f} vs {b['final_ctx']:.3f} "
                      f"(diff {abs(a['final_ctx'] - b['final_ctx']):.3f})")

    out = Path(args.out) if args.out else root / "repl_summary.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\n[repl] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
