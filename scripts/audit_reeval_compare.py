#!/usr/bin/env python3
"""Compare the corrected re-evaluation against the published numbers.

Reads the three corrected arms under ``group_plus/implementation_audit_v1/`` and
the originally published JSON, joins the 55 novel records by
``(scene, view, frame_id, key)``, and asserts the reproduction tolerances from
the brief: TP/FP/FN and the >=0.5 count must match exactly; IoU/AP within 1e-6;
PSNR/SSIM within 1e-4.  Any larger difference is reported with the exact record
so it can be traced - it is never silently widened.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

AUDIT = Path("group_plus/implementation_audit_v1")
PUBLISHED = {
    "g0plus": Path("group_plus/recipe_v1/baseline.json"),
    "recipe_v1": Path("group_plus/recipe_v1/eval_summary.json"),
    "recipe_v2": Path("group_plus/recipe_v2/eval_summary.json"),
}
TOL_IOU = 1e-6
TOL_AP = 1e-6
TOL_PSNR = 1e-4


def published_numbers(label):
    payload = json.loads(PUBLISHED[label].read_text(encoding="utf-8"))
    if label == "g0plus":
        return {"mean_iou1": payload["main_metric"]["best_over_groups_iou_mean"],
                "ge05": payload["main_metric"]["iou_ge_0.5_count"],
                "gt_free": {k: payload["gt_free"][k] for k in ("tp", "fp", "fn")},
                "ap50": payload["gt_free"]["ap50"],
                "novel_psnr": payload["gate"]["novel_psnr"],
                "novel_ssim": payload["gate"]["novel_ssim"]}
    return {"mean_iou1": payload["novel_only_55"]["mean_iou1"],
            "ge05": payload["novel_only_55"]["iou1_ge_0.5"],
            "gt_free": {k: payload["novel_only_55"]["gt_free"][k] for k in ("tp", "fp", "fn")},
            "ap50": payload["novel_only_55"]["gt_free"]["ap50"],
            "novel_psnr": payload["gate"]["novel_psnr"],
            "novel_ssim": payload["gate"]["novel_ssim"]}


def rows_of(label):
    path = AUDIT / f"eval_{label}" / "eval_per_instance.csv"
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = (row["scene"], str(row["view"]), str(row["frame_id"]), str(row["key"]))
        if key in out:
            raise SystemExit(f"duplicate key {key} in {path}")
        out[key] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(AUDIT / "reeval_comparison.json"))
    args = parser.parse_args()
    result = {"tolerances": {"iou": TOL_IOU, "ap": TOL_AP, "psnr": TOL_PSNR,
                             "counts": "exact"},
              "arms": {}, "record_alignment": {}, "failures": []}
    arm_rows = {}
    for label in PUBLISHED:
        rows = rows_of(label)
        arm_rows[label] = rows
        summary = json.loads((AUDIT / f"eval_{label}" / "eval_summary.json").read_text())
        new = {
            "mean_iou1": summary["novel_only_55"]["mean_iou1"],
            "ge05": summary["novel_only_55"]["iou1_ge_0.5"],
            "gt_free": {k: summary["novel_only_55"]["gt_free"][k] for k in ("tp", "fp", "fn")},
            "ap50": summary["novel_only_55"]["gt_free"]["ap50"],
            "novel_psnr": summary["gate_recomputed"]["novel_psnr"],
            "novel_ssim": summary["gate_recomputed"]["novel_ssim"],
            "n_records": summary["novel_only_55"]["n"],
        }
        old = published_numbers(label)
        entry = {
            "published": old, "recomputed": new,
            "records_new": len(rows),
            "key_format": "(scene, view, frame_id, key)",
            "iou_abs_diff": abs(new["mean_iou1"] - old["mean_iou1"]),
            "ap_abs_diff": abs(new["ap50"] - old["ap50"]),
            "psnr_abs_diff": abs(new["novel_psnr"] - old["novel_psnr"]),
            "ssim_abs_diff": abs(new["novel_ssim"] - old["novel_ssim"]),
            "ge05_identical": new["ge05"] == old["ge05"],
            "gt_free_identical": new["gt_free"] == old["gt_free"],
        }
        entry["passes"] = (
            entry["iou_abs_diff"] <= TOL_IOU and entry["ap_abs_diff"] <= TOL_AP
            and entry["psnr_abs_diff"] <= TOL_PSNR
            and entry["ssim_abs_diff"] <= TOL_PSNR
            and entry["ge05_identical"] and entry["gt_free_identical"]
            and new["n_records"] == 55
        )
        if not entry["passes"]:
            result["failures"].append(
                f"{label}: diff iou {entry['iou_abs_diff']:.3e} ap {entry['ap_abs_diff']:.3e} "
                f"psnr {entry['psnr_abs_diff']:.3e} ssim {entry['ssim_abs_diff']:.3e} "
                f"ge05 {entry['ge05_identical']} gtfree {entry['gt_free_identical']}"
            )
        result["arms"][label] = entry
    # alignment across arms
    base = set(arm_rows["g0plus"])
    for label, rows in arm_rows.items():
        result["record_alignment"][label] = {
            "n": len(rows),
            "same_keys_as_g0plus": set(rows) == base,
            "unique": len(rows) == len(set(rows)),
        }
        if set(rows) != base:
            result["failures"].append(f"{label}: record keys differ from g0plus "
                                      f"({len(set(rows) ^ base)} symmetric difference)")
    result["all_passed"] = not result["failures"]
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("iou_abs_diff", "ap_abs_diff",
                                               "psnr_abs_diff", "ssim_abs_diff",
                                               "ge05_identical", "gt_free_identical",
                                               "passes")}
                      for k, v in result["arms"].items()}, indent=1))
    print(f"all_passed={result['all_passed']}")
    for failure in result["failures"]:
        print("FAIL:", failure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
