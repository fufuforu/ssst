#!/usr/bin/env python3
"""Three-way pairing table: G0+ / recipe_v1 / recipe_v2 on the same 55 novel records.

The per-record best-over-groups numbers are produced by the existing
`scripts/eval_recipe_v1.py` run for each arm (which itself calls
`audit_group_routing.fragmentation`).  This script only *joins* those outputs by
`(scene, view, key)`, asserts the 55-record alignment, and adds the paired
differences and the pre-registered acceptance booleans for recipe_v2.  It does
not recompute any metric.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("group_plus/recipe_v2")
BASELINE = Path("group_plus/recipe_v1/baseline.json")
V1 = Path("group_plus/recipe_v1")
V2 = Path("group_plus/recipe_v2")


def load(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = {}
    for row in rows:
        key = (row["scene"], str(row["view"]), str(row["key"]))
        if key in out:
            raise SystemExit(f"duplicate key {key} in {path}")
        out[key] = row
    return out


def summarise(prefix, rows):
    iou = np.array([float(r[f"{prefix}_iou1"]) for r in rows])
    return {"n": len(rows), "mean_iou1": float(iou.mean()),
            "iou1_ge_0.5": int((iou >= 0.5).sum())}


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    v1_rows, v2_rows = load(V1 / "eval_per_instance.csv"), load(V2 / "eval_per_instance.csv")
    if len(v1_rows) != 55 or len(v2_rows) != 55:
        raise SystemExit(f"expected 55 records, got v1={len(v1_rows)} v2={len(v2_rows)}")
    if set(v1_rows) != set(v2_rows):
        raise SystemExit("recipe_v1 and recipe_v2 record keys differ")
    keys = sorted(v2_rows, key=lambda k: (k[0], int(k[1]), int(k[2])))
    for key in keys:
        if abs(float(v2_rows[key]["baseline_iou1"])
               - float(v1_rows[key]["baseline_iou1"])) > 1e-12:
            raise SystemExit(f"G0+ baseline column differs between arms at {key}")

    rows = []
    for key in keys:
        a, b = v1_rows[key], v2_rows[key]
        rows.append({
            "scene": key[0], "view": key[1], "key": key[2],
            "frame_id": b["frame_id"], "gt_area": int(b["gt_area"]),
            "bucket": b["bucket"],
            "g0plus_iou1": float(b["baseline_iou1"]),
            "recipe_v1_iou1": float(a["iou1"]),
            "recipe_v2_iou1": float(b["iou1"]),
            "d_v2_minus_v1": float(b["iou1"]) - float(a["iou1"]),
            "d_v2_minus_g0plus": float(b["iou1"]) - float(b["baseline_iou1"]),
        })
    with (OUT / "eval_three_way.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = {name: summarise(name, rows) for name in ("g0plus", "recipe_v1", "recipe_v2")}
    table["recipe_v2_vs_v1_delta"] = table["recipe_v2"]["mean_iou1"] - table["recipe_v1"]["mean_iou1"]
    table["recipe_v2_vs_g0plus_delta"] = (table["recipe_v2"]["mean_iou1"]
                                          - table["g0plus"]["mean_iou1"])

    buckets = {}
    for bucket, predicate in (("small_lt3000", lambda r: r["bucket"] == "small"),
                              ("large_ge3000", lambda r: r["bucket"] == "large")):
        subset = [r for r in rows if predicate(r)]
        buckets[bucket] = {
            "n": len(subset),
            **{name: {"mean_iou1": float(np.mean([r[f"{name}_iou1"] for r in subset])),
                      "iou1_ge_0.5": int(sum(1 for r in subset if r[f"{name}_iou1"] >= 0.5))}
               for name in ("g0plus", "recipe_v1", "recipe_v2")},
        }
    per_scene = {}
    for scene in sorted({r["scene"] for r in rows}):
        subset = [r for r in rows if r["scene"] == scene]
        per_scene[scene] = {
            "n": len(subset),
            **{name: {"mean_iou1": float(np.mean([r[f"{name}_iou1"] for r in subset])),
                      "iou1_ge_0.5": int(sum(1 for r in subset if r[f"{name}_iou1"] >= 0.5))}
               for name in ("g0plus", "recipe_v1", "recipe_v2")},
            "d_v2_minus_v1": float(np.mean([r["d_v2_minus_v1"] for r in subset])),
        }

    v2_summary = json.loads((V2 / "eval_summary.json").read_text(encoding="utf-8"))
    v1_summary = json.loads((V1 / "eval_summary.json").read_text(encoding="utf-8"))
    payload = {
        "scope": "8 unseen windows, novel views 2/3, 55 records; development set only",
        "main_metric": "routing_v1 fragmentation IoU1, joined by (scene, view, key)",
        "alignment": {"records": len(rows), "unique": len(set(v2_rows)),
                      "v1_v2_keys_identical": True, "baseline_column_identical": True},
        "best_over_groups": table,
        "buckets": buckets,
        "per_scene": per_scene,
        "gt_free_and_gate": {
            "g0plus": {"tp": baseline["gt_free"]["tp"], "fp": baseline["gt_free"]["fp"],
                       "fn": baseline["gt_free"]["fn"], "ap50": baseline["gt_free"]["ap50"],
                       "novel_psnr": baseline["gate"]["novel_psnr"]},
            "recipe_v1": {**v1_summary["novel_only_55"]["gt_free"],
                          "novel_psnr": v1_summary["gate"]["novel_psnr"],
                          "novel_ssim": v1_summary["gate"]["novel_ssim"]},
            "recipe_v2": {**v2_summary["novel_only_55"]["gt_free"],
                          "novel_psnr": v2_summary["gate"]["novel_psnr"],
                          "novel_ssim": v2_summary["gate"]["novel_ssim"]},
        },
        "acceptance_v2_vs_g0plus": v2_summary["acceptance"],
        "acceptance_v2_vs_v1": {
            "iou_mean": {"v1": table["recipe_v1"]["mean_iou1"],
                         "v2": table["recipe_v2"]["mean_iou1"],
                         "delta": table["recipe_v2"]["mean_iou1"]
                                  - table["recipe_v1"]["mean_iou1"]},
            "iou_ge_0.5": {"v1": table["recipe_v1"]["iou1_ge_0.5"],
                           "v2": table["recipe_v2"]["iou1_ge_0.5"],
                           "delta": table["recipe_v2"]["iou1_ge_0.5"]
                                    - table["recipe_v1"]["iou1_ge_0.5"]},
        },
    }
    (OUT / "summary_three_way.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({"best_over_groups": table, "buckets": buckets,
                      "gt_free_and_gate": payload["gt_free_and_gate"],
                      "acceptance": payload["acceptance_v2_vs_g0plus"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
