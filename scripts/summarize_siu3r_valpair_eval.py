#!/usr/bin/env python3
"""Summarise the official SIU3R val_pair reconstruction evaluation.

Read-only: consumes the adapter's `eval_report.json`, the pinned evaluator's
`official_evaluator_result.json` and the evaluator's own per-scene
`render_scores.json` / `depth_scores.json`, and reports

  * the official aggregate (image + depth quality only; semantics is N/A),
  * frame-id / GT-alignment and depth-range evidence from the adapter report,
  * an explicitly separated **diagnostic** split of the evaluator's own per-item
    scores into the 2 context vs the 4 novel views.

The novel/context split is derived from the evaluator's per-item outputs, not a
second metric implementation, and must never be quoted as the official number.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.eval_root)
    base = root / "predictions" if (root / "predictions" / "eval_report.json").is_file() else root
    preds = base / "official_predictions"

    adapter = json.loads((base / "eval_report.json").read_text(encoding="utf-8"))
    official = json.loads((root / "official_evaluator_result.json").read_text(encoding="utf-8"))

    kind_of = {}          # (scene, frame_id) -> "context" | "novel"
    ctx_pairs = set()
    for rec in adapter["predictions"]:
        scene = rec["scene"]
        ctx_pairs.add((scene, tuple(rec["context_ids"])))
        for d in rec["depth_rows"]:
            kind_of[(scene, int(d["frame_id"]))] = d["kind"]

    img = defaultdict(lambda: {"psnr": [], "ssim": [], "lpips": []})
    dep = defaultdict(lambda: {"absrel": [], "rmse": []})
    for scene_dir in sorted(p for p in preds.iterdir() if p.is_dir()):
        scene = scene_dir.name.split("_context")[0]
        rs = scene_dir / "render_scores.json"
        ds = scene_dir / "depth_scores.json"
        if rs.is_file():
            for item in json.loads(rs.read_text(encoding="utf-8")):
                frame = int(Path(item["item"]).stem.split("_")[-1])
                k = kind_of.get((scene, frame), "unknown")
                for key in ("psnr", "ssim", "lpips"):
                    img[k][key].append(item[key])
                    img["all"][key].append(item[key])
        if ds.is_file():
            for item in json.loads(ds.read_text(encoding="utf-8")):
                frame = int(Path(item["item"]).stem.split("_")[-1])
                k = kind_of.get((scene, frame), "unknown")
                for key in ("absrel", "rmse"):
                    dep[k][key].append(item[key])
                    dep["all"][key].append(item[key])

    depth_rows = [d for rec in adapter["predictions"] for d in rec["depth_rows"]]
    summary = {
        "checkpoint_dir": adapter["checkpoint_dir"],
        "official_evaluator": {
            "used": official.get("official_evaluator_used"),
            "siu3r_repo": official.get("siu3r_repo"),
            "siu3r_commit": official.get("siu3r_commit"),
            "result": official.get("result"),
            "metrics_returned": sorted((official.get("result") or {}).keys()),
            "semantic_instance_metrics": "N/A - reconstruction-only run emits no semantic/instance maps",
        },
        "adapter": {
            "records_exported": adapter["records"],
            "scenes_exported": len({r["scene"] for r in adapter["predictions"]}),
            "scene_dirs_on_disk": len([p for p in preds.iterdir() if p.is_dir()]),
            "unique_context_pairs": len(ctx_pairs),
            "reconstruction_only": adapter["reconstruction_only"],
            "model_type": adapter["config"]["model_type"],
            "gates": {
                "batch_frame_ids_match_manifest": all(
                    set(r["batch_frame_ids"]) == set(r["target_ids"])
                    and r["batch_frame_ids"][:2] == r["context_ids"]
                    and sorted(r["novel_ids"]) == sorted(set(r["target_ids"]) - set(r["context_ids"]))
                    for r in adapter["predictions"]),
                "all_outputs_finite": all(r["all_outputs_finite"] for r in adapter["predictions"]),
                "depth_unit_scale": adapter["predictions"][0]["depth_unit_scale"],
            },
            "depth_coverage": {
                "pred_nonzero_frac_mean": mean([d["pred_depth_m_nonzero_frac"] for d in depth_rows]),
                "gt_valid_frac_mean": mean([d["gt_depth_valid_frac"] for d in depth_rows]),
                "pred_depth_m_range": [min(d["pred_depth_m_min"] for d in depth_rows),
                                       max(d["pred_depth_m_max"] for d in depth_rows)],
                "gt_depth_m_range": [min(d["gt_depth_m_min"] for d in depth_rows if d["gt_depth_m_min"]),
                                     max(d["gt_depth_m_max"] for d in depth_rows if d["gt_depth_m_max"])],
                "gt_over_pred_median_mean": mean([d["gt_over_pred_median"] for d in depth_rows]),
            },
        },
        "diagnostic_split_of_official_per_item_scores": {
            "note": "derived from the SIU3R evaluator's own render_scores.json / depth_scores.json; "
                    "NOT an official aggregate and NOT a replacement for it",
            "n_items": {k: len(v["psnr"]) for k, v in img.items()},
            "image": {k: {kk: mean(vv) for kk, vv in v.items()} for k, v in img.items()},
            "depth": {k: {kk: mean(vv) for kk, vv in v.items()} for k, v in dep.items()},
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["official_evaluator"], indent=2))
    print(json.dumps(summary["diagnostic_split_of_official_per_item_scores"]["image"], indent=2))
    print(json.dumps(summary["diagnostic_split_of_official_per_item_scores"]["depth"], indent=2))
    print(f"[sum] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
