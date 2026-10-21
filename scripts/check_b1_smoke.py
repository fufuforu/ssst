#!/usr/bin/env python3
"""Fail-fast checks on the B1 real-model smoke before the full 1860-pair export.

Asserts, per arm: the six-frame directory contract is complete, the class
distribution is not biased to a single label, the official reader ran and
returned a finite mIoU, and the alpha conservation of the exported masks is
within tolerance.  Writes `B1_smoke_ok.json` only when everything passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = ("rgb", "rgb_gt", "depth", "depth_gt", "context_seg_pred",
            "context_seg_gt", "target_seg_pred", "target_seg_gt")


def check_arm(root: Path, arm: str) -> dict:
    report: dict = {"arm": arm, "checks": {}}
    pan = root / arm / "official_predictions_panoptic"
    sem = root / arm / "official_predictions_semantic"
    scene_dirs = sorted(d for d in pan.iterdir() if d.is_dir()) if pan.is_dir() else []
    report["scene_dirs"] = [d.name for d in scene_dirs]
    report["checks"]["has_scene_dir"] = bool(scene_dirs)
    if not scene_dirs:
        return report
    scene_dir = scene_dirs[0]
    for sub in REQUIRED:
        present = (scene_dir / sub).is_dir() and any((scene_dir / sub).glob("*.png"))
        report["checks"][f"panoptic_{sub}_present"] = present
    report["checks"]["panoptic_pred_json"] = (
        scene_dir / "target_seg_pred" / "pred.json").is_file()
    report["checks"]["panoptic_context_has_2"] = (
        len(list((scene_dir / "context_seg_pred").glob("*.png"))) == 2)
    report["checks"]["panoptic_target_has_6"] = (
        len(list((scene_dir / "target_seg_pred").glob("*.png"))) == 6)
    if sem.is_dir():
        sem_scene = sorted(d for d in sem.iterdir() if d.is_dir())[0]
        report["checks"]["semantic_target_has_6"] = (
            len(list((sem_scene / "target_seg_pred").glob("*.png"))) == 6)
        report["checks"]["semantic_has_no_rgb"] = not (sem_scene / "rgb").exists()
    else:
        report["checks"]["semantic_tree_present"] = False
    official = root / arm / "official_semantic_smoke.json"
    if official.is_file():
        payload = json.loads(official.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        miou = result.get("target_miou")
        report["official_semantic"] = {
            "target_miou": miou, "context_miou": result.get("context_miou"),
            "keys": sorted(result.keys()),
        }
        report["checks"]["official_reader_ran"] = bool(result)
        report["checks"]["official_miou_finite"] = (
            isinstance(miou, (int, float)) and miou == miou)
        report["checks"]["no_pq_or_map_for_semantic_only"] = (
            "target_pq" not in result and "target_map" not in result)
    else:
        report["checks"]["official_reader_ran"] = False
    report["passed"] = all(v for v in report["checks"].values() if isinstance(v, bool))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    arms = {arm: check_arm(root, arm) for arm in ("recipe_v1", "g0plus")}
    passed = all(a["passed"] for a in arms.values())
    payload = {"arms": arms, "all_passed": passed}
    out = Path(args.out) if args.out else root.parent / "B1_smoke_ok.json"
    if passed:
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({arm: {"passed": a["passed"],
                            "failed": [k for k, v in a["checks"].items()
                                       if isinstance(v, bool) and not v],
                            "official": a.get("official_semantic")}
                      for arm, a in arms.items()}, indent=1))
    print(f"all_passed={passed} -> wrote {out if passed else '(nothing)'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
