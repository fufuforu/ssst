#!/usr/bin/env python3
"""Evaluation CLI with the unified P(thing) criterion (works for G0 and G0+).

    python scripts/eval_group_plus.py \
        --run g0=workspace_group_locusgs/arm_g0/ckpt_step6000 \
        --run g0plus=workspace_group_plus/arm_g0plus/ckpt_step6000 \
        --out ... --images ...

GT-free rule (identical for every run): ``P(thing) = sum_{c=2..19} softmax(21)``
``>= 0.5`` and the predicted 20-class label must be a thing class, then
``mask > 0.5`` and predicted area ``>= 50 px``.  ``P(foreground)`` is reported
only as a diagnostic.  Development split only - not an SIU3R official metric.
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
from tokengs.options import config_defaults  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    MASK_THRESHOLD,
    MIN_PRED_PIXELS,
    SCORE_THRESHOLD,
    build_val_entries,
    evaluate_entry_v2,
    group_predictions_v2,
    summarise_v2,
)
from scripts.eval_group_locusgs import build_train_entries, load_run  # noqa: E402


def load_group_run(spec: str, preset: str, seed: int, device):
    name, directory = load_run(spec)
    payload = torch.load(directory / "train_state.pt", map_location="cpu", weights_only=False)
    arm = str(payload["meta"].get("arm", "g0"))
    opt = config_defaults[preset].evolve(
        seed=seed,
        group_arm="g1" if arm == "g1" else "g0",
        group_bg_supervision=arm == "g0plus",
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    model.eval()
    return name, directory, arm, int(payload["step"]), opt, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--out", required=True)
    parser.add_argument("--images", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--train-windows", type=int, default=8)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds to wait before loading (quick smoke only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.delay:
        import time

        time.sleep(args.delay)

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    report: dict = {
        "scope": "32 train / 8 unseen development split; NOT SIU3R official",
        "gt_free_rule": {
            "score": "P(thing) = sum_{c=2..19} softmax(21 logits)",
            "score_threshold": SCORE_THRESHOLD,
            "class": "argmax of the 20 group class logits must be a thing class (2..19)",
            "mask_threshold": MASK_THRESHOLD,
            "min_area": MIN_PRED_PIXELS,
            "diagnostic_score": "P(foreground) = 1 - softmax(21)[20] (reported only)",
        },
        "note": "model uses GT camera poses for rays; SIU3R is unposed",
        "runs": {}, "scenes": {},
    }

    for spec in args.run:
        name, directory, arm, step, opt, model = load_group_run(spec, args.preset,
                                                                args.seed, device)
        entries = build_val_entries(opt, split, device, scenes=args.scenes)
        rows = []
        for entry in entries:
            print(f"[eval+] {name} {entry['scene']}", flush=True)
            row = evaluate_entry_v2(model, entry, opt, include_diagnostics=args.diagnostics)
            rows.append(row)
            if args.images:
                write_images(Path(args.images), name, entry, row)
        summary = summarise_v2(rows)
        if args.diagnostics:
            summary["purity_gs_mean"] = float(np.mean([r["purity"]["gs_purity_mean"] for r in rows]))
            summary["purity_contributing_gs_mean"] = float(
                np.mean([r["purity_contributing"]["gs_purity_mean"] for r in rows])
            )
        run = {"checkpoint": str(directory), "arm": arm, "step": step, "summary": summary}
        if args.train_windows > 0:
            train_entries = build_train_entries(opt, split, plan, device, args.train_windows)
            train_rows = []
            for entry in train_entries:
                print(f"[eval+] {name} TRAIN {entry['scene']}", flush=True)
                train_rows.append(evaluate_entry_v2(model, entry, opt))
            run["training_windows"] = {
                "scenes": [e["scene"] for e in train_entries],
                "summary": summarise_v2(train_rows),
            }
        report["runs"][name] = run
        for entry, row in zip(entries, rows):
            report["scenes"].setdefault(entry["scene"], {})[name] = {
                k: v for k, v in row.items() if not k.startswith("_")
            }

    names = list(report["runs"])
    if len(names) == 2:
        a, b = names
        paired = {}
        for scene, per_run in report["scenes"].items():
            if a not in per_run or b not in per_run:
                continue
            left, right = per_run[a], per_run[b]
            paired[scene] = {
                "novel_psnr_delta": right["novel_psnr"] - left["novel_psnr"],
                "ctx_psnr_delta": right["ctx_psnr"] - left["ctx_psnr"],
                "novel_ssim_delta": right["novel_ssim"] - left["novel_ssim"],
                "sem_miou_delta": right["sem_miou"] - left["sem_miou"],
                "novel_ap50_delta": (
                    float(np.mean([v["ap50"] for v in right["views"].values()]))
                    - float(np.mean([v["ap50"] for v in left["views"].values()]))
                ),
            }
        report["paired"] = {
            "runs": [a, b],
            "per_scene": paired,
            **{
                f"{key}_mean": float(np.mean([v[key] for v in paired.values()]))
                for key in ("novel_psnr_delta", "ctx_psnr_delta", "novel_ssim_delta",
                            "sem_miou_delta")
            },
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, run in report["runs"].items():
        s = run["summary"]
        print(f"[eval+] {name} (step {run['step']}, arm {run['arm']}): "
              f"ctx {s['ctx_psnr']:.2f} novel {s['novel_psnr']:.2f} ssim {s['novel_ssim']:.3f} "
              f"mIoU {s['sem_miou']:.3f} | AP50 {s['novel_ap50']:.4f} "
              f"TP/FP/FN {s['novel_tp']}/{s['novel_fp']}/{s['novel_fn']} "
              f"| bg mass stuff/thing {s['background_stuff_mass_mean']:.3f}/"
              f"{s['background_thing_mass_mean']:.3f} "
              f"| active groups {s['group_usage_active']:.1f} "
              f"| best-over-groups {s['novel_best_over_groups_iou']:.3f}")
        if "training_windows" in run:
            t = run["training_windows"]["summary"]
            print(f"    training windows {run['training_windows']['scenes']}: "
                  f"psnr {t['novel_psnr']:.2f} mIoU {t['sem_miou']:.3f} "
                  f"AP50 {t['novel_ap50']:.4f} TP/FP/FN {t['novel_tp']}/{t['novel_fp']}/"
                  f"{t['novel_fn']} best-over-groups {t['novel_best_over_groups_iou']:.3f}")
    print(f"[eval+] wrote {out_path}")
    return 0


def write_images(image_dir: Path, name: str, entry, row) -> None:
    """GT | RGB render | GT thing/stuff | background mass | GT-free instances | error."""
    image_dir.mkdir(parents=True, exist_ok=True)
    forward = row["_forward"]
    gt_images = entry["batch"]["images_all"][0].float().cpu().numpy()
    prediction = row["_render"]
    semantic_gt = row["_semantic_gt"]
    instance_gt = row["_instance_gt"]
    alpha = forward["masks"]["alpha"][0, :, 0].float().cpu().numpy()
    background = forward["masks"]["background_mass"][0, :, 0].float().cpu().numpy()
    p_bg = background / np.clip(alpha, 0.5, None)
    rows = []
    for view in (0, 2):
        predictions, _ = group_predictions_v2(forward, view)
        instance_map = np.zeros(prediction[view].shape[-2:], dtype=np.int64)
        for index, item in enumerate(predictions, start=1):
            instance_map[item["mask"]] = index
        gt_thing_stuff = np.full(prediction[view].shape[-2:], 0.5, dtype=np.float64)
        gt_thing_stuff[(semantic_gt[view] == 0) | (semantic_gt[view] == 1)] = 1.0
        gt_thing_stuff[(semantic_gt[view] >= 2) & (semantic_gt[view] < 20)] = 0.0
        gt_thing_stuff[semantic_gt[view] == 255] = 0.5
        error = np.abs(prediction[view] - gt_images[view]).mean(axis=0, keepdims=True)
        tiles = [
            gt_images[view],
            np.clip(prediction[view], 0, 1),
            np.repeat(gt_thing_stuff[None], 3, axis=0),
            np.repeat(np.clip(p_bg[view], 0, 1)[None], 3, axis=0),
            np.repeat((instance_map > 0)[None].astype(np.float64), 3, axis=0),
            np.clip(error * 3.0, 0, 1).repeat(3, axis=0),
        ]
        rows.append(np.concatenate(tiles, axis=2))
    tile = np.concatenate(rows, axis=1)
    Image.fromarray((np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)).save(
        image_dir / f"{entry['scene']}_{name}.png"
    )


if __name__ == "__main__":
    raise SystemExit(main())
