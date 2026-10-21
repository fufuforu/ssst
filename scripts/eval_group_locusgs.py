#!/usr/bin/env python3
"""Final development evaluation for the G0/G1 group experiment.

    python scripts/eval_group_locusgs.py \
        --run g0=workspace_group_locusgs/arm_g0/ckpt_step6000 \
        --run g1=workspace_group_locusgs/arm_g1/ckpt_step6000 \
        --out workspace_group_locusgs/eval_step6000.json \
        --images workspace_group_locusgs/images

Writes per-scene rows, per-arm means, the paired G1-G0 deltas, the GT-assisted
"best over all queries" diagnostic (clearly separated from the GT-free result)
and RGB / semantic / instance-mask tiles.  Development split only: this is not
an SIU3R official evaluation, and the model uses GT camera poses for its rays
while SIU3R is unposed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.models.ssst_contracts import SEMANTIC_CLASS_COUNT  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.group_locusgs_eval import (  # noqa: E402
    MASK_THRESHOLD,
    MIN_PRED_PIXELS,
    OBJECTNESS_THRESHOLD,
    build_val_entries,
    evaluate_group_entry,
    forward_group,
    group_view_predictions,
    summarise_group,
)
from scripts.object_locusgs_eval import gt_instances  # noqa: E402
from scripts.object_locusgs_eval import gt_scene_scale, move  # noqa: E402
from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402


def build_train_entries(opt, split, plan, device, count):
    """One training window per scene, same protocol as the unseen windows."""
    entries = []
    used = set()
    stride = max(1, len(plan["entries"]) // max(1, count))
    for position in range(0, len(plan["entries"]), stride):
        item = plan["entries"][position]
        if item["scene"] in used:
            continue
        used.add(item["scene"])
        root = Path(split["train_root"])
        provider = SIU3RProcessedProvider(
            opt, root=str(root), subset=[item["scene"]], training=True, rank=0
        )
        provider.pin_pair(
            scene_id=item["scene"], context_frame_ids=item["context"],
            novel_frame_ids=item["novel"], pair_iou=item["pair_iou"],
        )
        batch = move(default_collate([provider[0]]), device)
        entries.append({
            "scene": item["scene"], "root": str(root), "batch": batch,
            "context": list(item["context"]), "novel": list(item["novel"]),
            "scale": gt_scene_scale(root / item["scene"], item["context"] + item["novel"]),
            "kind": "train",
        })
        if len(entries) >= count:
            break
    return entries


def load_run(spec: str):
    name, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"--run expects name=checkpoint-dir, got {spec!r}")
    return name, Path(path)


def load_model(directory: Path, preset: str, seed: int, device):
    payload = torch.load(directory / "train_state.pt", map_location="cpu", weights_only=False)
    arm = str(payload["meta"].get("arm", "g0"))
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm=arm, batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    model.eval()
    return opt, model, arm, int(payload["step"])


def best_any_query_diagnostic(forward, semantic_gt, instance_gt, view) -> dict:
    """GT-assisted upper bound: best IoU over *all* 100 groups (diagnostic only)."""
    mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
    instances = gt_instances(semantic_gt, instance_gt, view)
    per_instance = []
    for mask in instances.values():
        best = 0.0
        best_area = 0
        for group_index in range(mass.shape[0]):
            prediction = mass[group_index] > MASK_THRESHOLD
            area = int(prediction.sum())
            if area < MIN_PRED_PIXELS:
                continue
            union = int((prediction | mask).sum())
            if not union:
                continue
            iou = int((prediction & mask).sum()) / union
            if iou > best:
                best, best_area = iou, area
        per_instance.append({"area": int(mask.sum()), "best_any_query_iou": float(best),
                             "pred_area": best_area})
    return {
        "per_instance": per_instance,
        "mean_best_iou": float(np.mean([p["best_any_query_iou"] for p in per_instance]))
        if per_instance else 0.0,
        "recall50_any_query": float(np.mean(
            [p["best_any_query_iou"] >= 0.5 for p in per_instance]
        )) if per_instance else 0.0,
        "note": "GT-assisted diagnostic; never used as a GT-free score",
    }


def semantic_palette() -> np.ndarray:
    rng = np.random.default_rng(7)
    palette = rng.integers(40, 255, (SEMANTIC_CLASS_COUNT + 1, 3), dtype=np.uint8)
    palette[0] = np.array([60, 60, 60], dtype=np.uint8)
    return palette


def mask_tile(gt_semantic, pred_semantic, gt_instances_map, pred_instances_map, palette):
    """GT semantic | predicted semantic | GT instances | GT-free predicted instances."""
    def colour_semantic(labels):
        return palette[np.clip(labels, 0, SEMANTIC_CLASS_COUNT)][..., :3]

    def colour_instances(labels):
        out = np.zeros(labels.shape + (3,), dtype=np.uint8)
        for value in np.unique(labels):
            if value <= 0:
                continue
            colour = palette[(int(value) * 7) % len(palette)]
            out[labels == value] = colour
        return out

    return np.concatenate([
        colour_semantic(gt_semantic),
        colour_semantic(pred_semantic),
        colour_instances(gt_instances_map),
        colour_instances(pred_instances_map),
    ], axis=1)


def rgb_tile(gt, prediction):
    error = np.abs(prediction - gt).mean(axis=0, keepdims=True).repeat(3, axis=0)
    tile = np.concatenate([gt, np.clip(prediction, 0, 1), np.clip(error * 3.0, 0, 1)], axis=2)
    return (np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--out", default="workspace_group_locusgs/eval.json")
    parser.add_argument("--images", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json",
                        help="pre-registered plan used for the training-window control")
    parser.add_argument("--train-windows", type=int, default=8,
                        help="distinct training windows evaluated under the same GT-free rule "
                             "(0 disables the training-window control)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    runs = [load_run(spec) for spec in args.run]
    palette = semantic_palette()
    results: dict = {
        "scope": "32 train / 8 unseen development split; NOT SIU3R official",
        "note": "model uses GT camera poses for rays; SIU3R is unposed",
        "gt_free_rule": {"objectness": OBJECTNESS_THRESHOLD, "mask": MASK_THRESHOLD,
                         "min_area": MIN_PRED_PIXELS, "class_source": "group semantic head"},
        "runs": {}, "scenes": {},
    }

    for name, directory in runs:
        opt, model, arm, step = load_model(directory, args.preset, args.seed, device)
        entries = build_val_entries(opt, split, device, scenes=args.scenes)
        rows = []
        for entry in entries:
            print(f"[eval] {name} {entry['scene']}", flush=True)
            row = evaluate_group_entry(model, entry, opt,
                                       include_diagnostics=not args.no_diagnostics)
            with torch.no_grad():
                forward = forward_group(model, entry["batch"], opt)
            semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
            instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
            for view in (2, 3):
                metrics = row["views"][view]
                metrics["best_any_query_diagnostic"] = best_any_query_diagnostic(
                    forward, semantic_gt, instance_gt, view
                )
            rows.append(row)
            if args.images:
                _write_images(Path(args.images), entry, row, forward, semantic_gt,
                              instance_gt, palette, name, opt)
        summary = summarise_group(rows)
        summary["purity"] = _mean_of(rows, ("purity", "token_purity_mean"))
        summary["purity_gs_mean"] = _mean_of(rows, ("purity", "gs_purity_mean"))
        summary["purity_contributing_gs_mean"] = _mean_of(
            rows, ("purity_contributing", "gs_purity_mean")
        )
        summary["token_group_purity"] = _mean_of(
            rows, ("token_group_purity", "mean_dominant_group_probability")
        )
        summary["generated_groups"] = _mean_of(rows, ("token_group_purity", "generated_groups"))
        summary["locality_all_p50_over_scale"] = _mean_of(rows, ("locality", "all_p50_over_scale"))
        summary["locality_contributing_p50_over_scale"] = _mean_of(
            rows, ("locality", "contrib_p50_over_scale")
        )
        summary["locality_collapse_fraction"] = _mean_of(rows, ("locality", "collapse_fraction"))
        summary["diagnostic_best_any_query_iou"] = float(np.mean([
            row["views"][view]["best_any_query_diagnostic"]["mean_best_iou"]
            for row in rows for view in (2, 3)
        ]))
        summary["diagnostic_best_any_query_recall50"] = float(np.mean([
            row["views"][view]["best_any_query_diagnostic"]["recall50_any_query"]
            for row in rows for view in (2, 3)
        ]))
        results["runs"][name] = {
            "checkpoint": str(directory), "arm": arm, "step": step, "summary": summary,
        }
        for entry, row in zip(entries, rows):
            results["scenes"].setdefault(entry["scene"], {})[name] = {
                k: v for k, v in row.items() if not k.startswith("_")
            }
        if args.train_windows > 0:
            train_entries = build_train_entries(opt, split, plan, device, args.train_windows)
            train_rows = []
            for entry in train_entries:
                print(f"[eval] {name} TRAIN {entry['scene']}", flush=True)
                train_rows.append(evaluate_group_entry(model, entry, opt))
            train_summary = summarise_group(train_rows)
            results["runs"][name]["training_windows"] = {
                "scenes": [entry["scene"] for entry in train_entries],
                "summary": train_summary,
                "unseen_minus_train_fingerprint": "see runs[name].summary for the unseen side",
            }
            results["runs"][name]["train_vs_unseen"] = {
                key: float(train_summary[key] - summary[key])
                for key in ("ctx_psnr", "novel_psnr", "sem_miou", "novel_ap50",
                            "novel_recall50", "novel_tp", "novel_fp", "novel_fn")
            }

    if {"g0", "g1"} <= {name for name, _ in runs}:
        paired = {}
        for scene, per_run in results["scenes"].items():
            if "g0" not in per_run or "g1" not in per_run:
                continue
            a, b = per_run["g0"], per_run["g1"]
            paired[scene] = {
                "novel_psnr_delta": b["novel_psnr"] - a["novel_psnr"],
                "ctx_psnr_delta": b["ctx_psnr"] - a["ctx_psnr"],
                "novel_ssim_delta": b["novel_ssim"] - a["novel_ssim"],
                "sem_miou_delta": b["sem_miou"] - a["sem_miou"],
                "novel_ap50_delta": (
                    float(np.mean([v["ap50"] for v in b["views"].values()]))
                    - float(np.mean([v["ap50"] for v in a["views"].values()]))
                ),
            }
        results["paired"] = {
            "per_scene": paired,
            "novel_psnr_delta_mean": float(np.mean(
                [v["novel_psnr_delta"] for v in paired.values()])),
            "ctx_psnr_delta_mean": float(np.mean(
                [v["ctx_psnr_delta"] for v in paired.values()])),
            "novel_ssim_delta_mean": float(np.mean(
                [v["novel_ssim_delta"] for v in paired.values()])),
            "sem_miou_delta_mean": float(np.mean(
                [v["sem_miou_delta"] for v in paired.values()])),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, payload in results["runs"].items():
        summary = payload["summary"]
        print(f"[eval] {name}: ctx {summary['ctx_psnr']:.2f} novel {summary['novel_psnr']:.2f} "
              f"ssim {summary['novel_ssim']:.3f} mIoU {summary['sem_miou']:.3f} "
              f"AP50 novel {summary['novel_ap50']:.4f} TP/FP/FN {summary['novel_tp']}/"
              f"{summary['novel_fp']}/{summary['novel_fn']} | recall {summary['novel_recall50']:.3f} "
              f"| diag best-any-query IoU {summary['diagnostic_best_any_query_iou']:.3f}")
    print(f"[eval] wrote {out_path}")
    return 0


def _mean_of(rows, path) -> float | None:
    values = []
    for row in rows:
        value = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _write_images(image_dir, entry, row, forward, semantic_gt, instance_gt, palette, name, opt):
    image_dir.mkdir(parents=True, exist_ok=True)
    view = 2  # first novel record
    gt = entry["batch"]["images_all"][0].float().cpu().numpy()
    prediction = forward["output"]["render"]["images_pred"][0].float().cpu().numpy()
    Image.fromarray(rgb_tile(gt[view], prediction[view])).save(
        image_dir / f"{entry['scene']}_{name}_rgb.png"
    )
    pred_semantic = forward["semantic_prob"][0, view].argmax(dim=0).float().cpu().numpy().astype(np.int64)
    predictions = group_view_predictions(forward, view)
    pred_instances = np.zeros(pred_semantic.shape, dtype=np.int64)
    for index, item in enumerate(predictions, start=1):
        pred_instances[item["mask"]] = index
    gt_map = np.zeros(pred_semantic.shape, dtype=np.int64)
    for index, mask in enumerate(gt_instances(semantic_gt, instance_gt, view).values(), start=1):
        gt_map[mask] = index
    tile = mask_tile(semantic_gt[view], pred_semantic, gt_map, pred_instances, palette)
    Image.fromarray(tile).save(image_dir / f"{entry['scene']}_{name}_masks.png")
    row["_instance_visual"] = {
        "view": view,
        "gt_instances": int(gt_map.max()),
        "predicted_instances": int(len(predictions)),
        "tp": row["views"][view]["tp"] if view in row["views"] else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
