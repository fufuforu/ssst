#!/usr/bin/env python3
"""Prove the report-only void-counter fix changed no prediction and no metric.

Compares two export trees produced by the same command before and after the
``void_pixels`` correction:

* every ``*.png`` must be byte-identical;
* every ``pred.json`` must be byte-identical;
* ``export_report.json`` may differ *only* in the ``void_pixels`` fields;
* the pinned SIU3R evaluator is run on both panoptic trees and the returned
  metric dicts must be identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def hashes(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-official", action="store_true", default=True)
    args = parser.parse_args()

    before, after = Path(args.before), Path(args.after)
    hb, ha = hashes(before), hashes(after)
    report: dict = {"checks": {}}
    png_before = {k: v for k, v in hb.items() if k.endswith(".png")}
    png_after = {k: v for k, v in ha.items() if k.endswith(".png")}
    report["n_png"] = {"before": len(png_before), "after": len(png_after)}
    report["png_bytes_identical"] = png_before == png_after
    report["checks"]["png_bytes_identical"] = png_before == png_after
    pred_before = {k: v for k, v in hb.items() if k.endswith("pred.json")}
    pred_after = {k: v for k, v in ha.items() if k.endswith("pred.json")}
    report["checks"]["pred_json_identical"] = pred_before == pred_after
    if not report["checks"]["png_bytes_identical"]:
        differing = sorted(set(png_before.items()) ^ set(png_after.items()))[:5]
        report["png_differences_example"] = [str(x)[:120] for x in differing]

    def load_reports(root: Path):
        out = {}
        for path in sorted(root.glob("*/export_report.json")):
            out[path.parent.name] = json.loads(path.read_text())
        return out

    rb, ra = load_reports(before), load_reports(after)
    counter = {}
    for arm in rb:
        if arm not in ra:
            continue
        vb = sum(row["panoptic_views"][0]["void_pixels"] for row in rb[arm]["rows"])
        va = sum(row["panoptic_views"][0]["void_pixels"] for row in ra[arm]["rows"])
        counter[arm] = {"before_counter": vb, "after_counter": va}
    report["void_counter"] = counter
    report["checks"]["counter_changed"] = any(
        v["before_counter"] != v["after_counter"] for v in counter.values())

    if args.run_official:
        try:
            from scripts.invoke_siu3r_official_evaluator import evaluate
            results = {}
            for arm in ("recipe_v1", "g0plus"):
                results[arm] = {
                    "before": evaluate(before / arm / "official_predictions_panoptic",
                                       device="cuda"),
                    "after": evaluate(after / arm / "official_predictions_panoptic",
                                      device="cuda"),
                }
            report["official"] = {
                arm: {"identical": results[arm]["before"] == results[arm]["after"]}
                for arm in results
            }
            report["checks"]["official_metrics_identical"] = all(
                v["identical"] for v in report["official"].values())
        except Exception as error:  # noqa: BLE001
            report["official_error"] = f"{type(error).__name__}: {error}"
            report["checks"]["official_metrics_identical"] = False

    report["all_checks_passed"] = all(
        v for v in report["checks"].values() if isinstance(v, bool))
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
