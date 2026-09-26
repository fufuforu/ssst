#!/usr/bin/env python3
"""Join several corrected evaluations into one paired table with the five gates.

Each arm is passed as ``label=path/to/eval_dir``.  Records are aligned by
``(scene, view, frame_id, key)``; the table reports the routing_v1 IoU1 mean, the
>=0.5 count, GT-free TP/FP/FN and AP50, and the recomputed novel PSNR/SSIM, plus
the five pre-registered gate booleans for every arm that is not the baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BASELINE = "g0plus"
GATES = {
    "novel_psnr_ge_18.957": lambda m: m["novel_psnr"] >= 18.957,
    "iou_mean_ge_0.2311": lambda m: m["mean_iou1"] >= 0.2311,
    "iou_ge05_ge_6": lambda m: m["iou1_ge_0.5"] >= 6,
    "gt_free_tp_ge_6": lambda m: m["tp"] >= 6,
    "ap50_ge_0.1675": lambda m: m["ap50"] >= 0.1675,
}


def load_arm(directory: Path):
    rows = list(csv.DictReader((directory / "eval_per_instance.csv").open(encoding="utf-8")))
    summary = json.loads((directory / "eval_summary.json").read_text(encoding="utf-8"))
    table = {(r["scene"], str(r["view"]), str(r["frame_id"]), str(r["key"])): r
             for r in rows}
    if len(table) != len(rows):
        raise SystemExit(f"{directory}: duplicate (scene,view,frame,key)")
    metrics = {
        "n": len(rows),
        "mean_iou1": float(np.mean([float(r["iou1"]) for r in rows])),
        "iou1_ge_0.5": sum(1 for r in rows if float(r["iou1"]) >= 0.5),
        "tp": summary["novel_only_55"]["gt_free"]["tp"],
        "fp": summary["novel_only_55"]["gt_free"]["fp"],
        "fn": summary["novel_only_55"]["gt_free"]["fn"],
        "ap50": summary["novel_only_55"]["gt_free"]["ap50"],
        "novel_psnr": summary["gate_recomputed"]["novel_psnr"],
        "novel_ssim": summary["gate_recomputed"]["novel_ssim"],
        "checkpoint": summary["checkpoint"],
        "head_mode": summary["effective_config"]["head_mode"],
    }
    metrics["paired_iou1_ge_0.5_delta"] = metrics["iou1_ge_0.5"]
    return table, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", help="label=eval_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    arms = {}
    for item in args.arms:
        label, directory = item.split("=", 1)
        arms[label] = load_arm(Path(directory))
    baseline_rows = arms[BASELINE][0] if BASELINE in arms else None
    table = {}
    for label, (rows, metrics) in arms.items():
        entry = dict(metrics)
        if baseline_rows is not None:
            missing = [k for k in rows if k not in baseline_rows]
            extra = [k for k in baseline_rows if k not in rows]
            entry["alignment"] = {"missing_vs_baseline": len(missing),
                                  "extra_vs_baseline": len(extra)}
            entry["mean_iou1_delta_vs_baseline"] = (
                metrics["mean_iou1"]
                - arms[BASELINE][1]["mean_iou1"])
        if label != BASELINE:
            entry["gates"] = {name: bool(fn(metrics)) for name, fn in GATES.items()}
            entry["gates"]["all_five"] = all(entry["gates"].values())
        table[label] = entry
    payload = {"baseline": BASELINE, "gates": table}
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    if args.csv:
        labels = list(arms)
        keys = sorted(arms[labels[0]][0])
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scene", "view", "frame_id", "key"] + [f"{l}_iou1" for l in labels])
            for key in keys:
                writer.writerow(list(key) + [
                    (arms[l][0].get(key, {}) or {}).get("iou1", "") for l in labels])
    for label, entry in table.items():
        print(label, {k: entry[k] for k in ("n", "mean_iou1", "iou1_ge_0.5", "tp", "fp",
                                            "fn", "ap50", "novel_psnr", "novel_ssim")},
              entry.get("gates", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
