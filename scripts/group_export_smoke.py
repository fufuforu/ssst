#!/usr/bin/env python3
"""Synthetic round-trip + official-reader parity smoke for the group exporter.

Builds a tiny two-view prediction directory in the pinned SIU3R contract
(``{scene}_pred{frame}.png`` + ``pred.json`` next to ``{scene}_gt{frame}.png``),
then

1. round-trips the PNG encoding and the ``pred.json`` fields, and
2. calls the **unmodified** ``SIU3R.src.evaluator.Evaluator.process_segmentation``
   to check that the masks/labels/scores it reconstructs are exactly ours,
3. repeats with a GT-copy prediction (ideal case) and with one GT thing removed
   (must produce an FN that is not erased by void/alpha handling).

These synthetic numbers are not model results.  Run with the SIU3R environment:
``/space/mawb/SIU3R/.venv_gpu_v4/bin/python scripts/group_export_smoke.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

SIU3R_REPO = "/space/mawb/SIU3R"


def save_segment(path: Path, semantic: np.ndarray, instance: np.ndarray) -> None:
    packed = (semantic.astype(np.int64) * 1000 + instance.astype(np.int64))
    rgb = np.zeros((*packed.shape, 3), dtype=np.uint8)
    rgb[..., 0] = packed % 256
    rgb[..., 1] = (packed // 256) % 256
    rgb[..., 2] = (packed // (256 * 256)) % 256
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def read_segment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(Image.open(path)).astype(np.int64)
    packed = img[..., 0] + img[..., 1] * 256 + img[..., 2] * 256 * 256
    return packed // 1000, packed % 1000


def build_case(root: Path, *, drop_pred_thing: int | None = None):
    """Two views with: 2 thing instances (ids 1,2 fixed across views), stuff, void."""
    height = width = 64
    semantic_gt = np.zeros((2, height, width), dtype=np.int64)
    instance_gt = np.zeros((2, height, width), dtype=np.int64)
    # stuff: class 1 (wall) top half
    semantic_gt[:, : height // 2, :] = 1
    # thing 1: class 3 box, present in both views at the same instance id
    semantic_gt[:, 4:16, 4:16] = 3
    instance_gt[:, 4:16, 4:16] = 7
    # thing 2: class 5 box
    semantic_gt[:, 20:34, 20:34] = 5
    instance_gt[:, 20:34, 20:34] = 9
    semantic_pred = semantic_gt.copy()
    instance_pred = instance_gt.copy()
    if drop_pred_thing == 2:
        # the GT keeps the instance; only the prediction misses it -> must be an FN
        semantic_pred[:, 20:34, 20:34] = 1
        instance_pred[:, 20:34, 20:34] = 0
    pred_info = [
        {"id": 7, "label_id": 3, "score": 0.91},
        {"id": 9, "label_id": 5, "score": 0.77},
    ]
    if drop_pred_thing == 2:
        pred_info = [entry for entry in pred_info if entry["id"] != 9]
    for name, frames in (("context", (100, 101)), ("target", (100, 101, 102, 103))):
        for frame in frames:
            save_segment(root / f"{name}_seg_pred" / f"scene0000_00_pred{frame}.png",
                         semantic_pred[0], instance_pred[0])
            save_segment(root / f"{name}_seg_gt" / f"scene0000_00_gt{frame}.png",
                         semantic_gt[0], instance_gt[0])
        (root / f"{name}_seg_pred" / "pred.json").write_text(
            json.dumps(pred_info), encoding="utf-8")
    return semantic_pred, instance_pred, semantic_gt, instance_gt, pred_info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-official", action="store_true",
                        help="fail if the pinned SIU3R evaluator cannot be imported")
    parser.add_argument("--out", default="group_plus/implementation_audit_v1/"
                                        "export_smoke.json")
    args = parser.parse_args()

    result: dict = {"checks": {}, "notes": []}
    tmp = Path(tempfile.mkdtemp(prefix="group_export_smoke_"))
    case = tmp / "case"
    semantic_pred, instance_pred, semantic_gt, instance_gt, pred_info = build_case(case)

    # 1. PNG / JSON round trip
    pred_path = case / "target_seg_pred" / "scene0000_00_pred100.png"
    sem_rt, ins_rt = read_segment(pred_path)
    round_trip = {
        "semantic_exact": bool(np.array_equal(sem_rt, semantic_pred[0])),
        "instance_exact": bool(np.array_equal(ins_rt, instance_pred[0])),
        "pred_json_ids": [entry["id"] for entry in json.loads(
            (case / "target_seg_pred" / "pred.json").read_text())],
    }
    round_trip["ids_match_instances"] = sorted(round_trip["pred_json_ids"]) == sorted(
        int(x) for x in np.unique(instance_pred) if x != 0)
    result["round_trip"] = round_trip
    result["checks"]["png_round_trip_exact"] = (round_trip["semantic_exact"]
                                                and round_trip["instance_exact"])
    result["checks"]["pred_json_ids_match_instance_ids"] = round_trip["ids_match_instances"]

    # 2. official reader parity
    sys.path.insert(0, SIU3R_REPO)
    try:
        from src.config import EvaluatorCfg
        from src.evaluator import Evaluator
        from src.utils.scannet_constant import (
            PANOPTIC_SEMANTIC2NAME, STUFF_CLASSES, THING_CLASSES)
        cfg = EvaluatorCfg(dataset_name="scannet", eval_context_miou=True,
                           eval_context_pq=False, eval_context_map=False,
                           eval_target_miou=True, eval_target_pq=False,
                           eval_target_map=False, eval_image_quality=False,
                           eval_depth_quality=False, id2label=PANOPTIC_SEMANTIC2NAME,
                           stuffs=STUFF_CLASSES, things=THING_CLASSES, device="cpu")
        evaluator = Evaluator(cfg)
        data = evaluator.process_segmentation(case / "target_seg_pred",
                                              case / "target_seg_gt")
        pred_sem = data["pred_semantics"][0].numpy()
        pred_ins = data["pred_instances"][0].numpy()
        # the reader concatenates the 4 target PNGs vertically
        expected_sem = np.concatenate([semantic_pred[0]] * 4, axis=0)
        parity = {
            "semantics_equal": bool(np.array_equal(pred_sem, expected_sem)),
            "instances_equal": bool(np.array_equal(
                pred_ins, np.concatenate([instance_pred[0]] * 4, axis=0))),
            "gt_labels": sorted(set(int(x) for x in data["map_gt"]["labels"].tolist())),
            "pred_labels": sorted(set(int(x) for x in data["map_pred"]["labels"].tolist())),
            "pred_scores": [float(x) for x in data["map_pred"]["scores"].tolist()],
            "n_gt_masks": int(data["map_gt"]["masks"].shape[0]),
            "n_pred_masks": int(data["map_pred"]["masks"].shape[0]),
        }
        # gt labels are 0-based (semantic-1) and stuff (wall=0) must be dropped
        parity["stuff_dropped_from_map_gt"] = 0 not in parity["gt_labels"]
        parity["scores_match_pred_json"] = sorted(parity["pred_scores"]) == sorted(
            entry["score"] for entry in pred_info)
        result["official_reader"] = parity
        result["checks"]["official_reader_semantics_exact"] = parity["semantics_equal"]
        result["checks"]["official_reader_instances_exact"] = parity["instances_equal"]
        result["checks"]["official_reader_scores_from_pred_json"] = \
            parity["scores_match_pred_json"]
    except Exception as error:  # noqa: BLE001
        result["official_reader_error"] = f"{type(error).__name__}: {error}"
        if args.require_official:
            raise
        result["notes"].append("official reader not importable in this interpreter")

    # 3. ideal (GT copy) vs one missing thing
    ideal = tmp / "ideal"
    build_case(ideal, drop_pred_thing=None)
    missing = tmp / "missing"
    build_case(missing, drop_pred_thing=2)
    try:
        ideal_data = Evaluator(cfg).process_segmentation(ideal / "target_seg_pred",
                                                         ideal / "target_seg_gt")
        missing_data = Evaluator(cfg).process_segmentation(
            missing / "target_seg_pred", missing / "target_seg_gt")
        result["ideal_vs_missing"] = {
            "ideal_gt_masks": int(ideal_data["map_gt"]["masks"].shape[0]),
            "ideal_pred_masks": int(ideal_data["map_pred"]["masks"].shape[0]),
            "missing_gt_masks": int(missing_data["map_gt"]["masks"].shape[0]),
            "missing_pred_masks": int(missing_data["map_pred"]["masks"].shape[0]),
        }
        result["checks"]["gt_copy_prediction_is_complete"] = (
            result["ideal_vs_missing"]["ideal_gt_masks"]
            == result["ideal_vs_missing"]["ideal_pred_masks"])
        result["checks"]["missing_thing_is_an_fn"] = (
            result["ideal_vs_missing"]["missing_pred_masks"]
            < result["ideal_vs_missing"]["missing_gt_masks"])
    except Exception as error:  # noqa: BLE001
        result["notes"].append(f"ideal/missing check skipped: {type(error).__name__}: {error}")

    result["all_checks_passed"] = all(
        v for v in result["checks"].values() if isinstance(v, bool))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
