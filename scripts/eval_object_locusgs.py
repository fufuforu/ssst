#!/usr/bin/env python3
"""Development-set evaluation CLI for the object-aware LocusGS A/B experiment.

    python scripts/eval_object_locusgs.py \
        --run a=workspace_object_locusgs/arm_a/ckpt_step6000 \
        --run b=workspace_object_locusgs/arm_b/ckpt_step6000 \
        --out object_locusgs/eval_report.json --images workspace_object_locusgs/images

Reports the fixed 32/8 development numbers only: it is *not* an SIU3R official
evaluation (the official 1860-pair test set is never touched) and the model uses
GT camera poses to build rays, whereas SIU3R is an unposed setting.
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
from scripts.object_locusgs_eval import (  # noqa: E402
    build_val_entries,
    evaluate_entry,
    summarise,
)


def load_run(spec: str):
    name, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"--run expects name=path, got {spec!r}")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="name=checkpoint-dir (arm inferred from the checkpoint meta)")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_object_locusgs_ab")
    parser.add_argument("--out", default="object_locusgs/eval_report.json")
    parser.add_argument("--images", default=None)
    parser.add_argument("--purity", action="store_true",
                        help="also compute the GT-assisted token/GS purity diagnostic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    runs = [load_run(spec) for spec in args.run]

    results: dict = {"scope": "32 train / 8 unseen development split; not SIU3R official",
                     "note": "model uses GT camera poses for rays; SIU3R is unposed",
                     "runs": {}, "scenes": {}}
    for name, directory in runs:
        payload = torch.load(directory / "train_state.pt", map_location="cpu", weights_only=False)
        arm = str(payload["meta"].get("arm", "a"))
        opt = config_defaults[args.preset].evolve(
            seed=args.seed, object_arm=arm, batch_size=1, num_workers=0,
            num_input_views=2, num_views=4,
            dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        )
        model = model_registry[opt.model_type](opt).to(device)
        model.load_state_dict(
            torch.load(directory / "model.pt", map_location="cpu", weights_only=False)["model"],
            strict=True,
        )
        model.eval()
        entries = build_val_entries(opt, split, device, scenes=args.scenes)
        rows = []
        for entry in entries:
            print(f"[eval] {name} {entry['scene']}", flush=True)
            rows.append(evaluate_entry(model, entry, opt, include_purity=args.purity))
        summary = summarise(rows)
        results["runs"][name] = {
            "checkpoint": str(directory),
            "arm": arm,
            "step": int(payload["step"]),
            "summary": summary,
            "config": {
                "sem_weight": opt.object_sem_loss_weight,
                "inst_weight": opt.object_inst_loss_weight,
                "relation_enabled": arm == "b",
            },
        }
        for entry, row in zip(entries, rows):
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            results["scenes"].setdefault(entry["scene"], {})[name] = clean
        if args.images:
            image_dir = Path(args.images)
            image_dir.mkdir(parents=True, exist_ok=True)
            _write_tiles(name, entries, rows, image_dir, opt)

    if len(runs) == 2 and {"a", "b"} <= {name for name, _ in runs}:
        paired = {}
        for scene, per_run in results["scenes"].items():
            if "a" not in per_run or "b" not in per_run:
                continue
            a, b = per_run["a"], per_run["b"]
            paired[scene] = {
                "novel_psnr_delta": b["novel_psnr"] - a["novel_psnr"],
                "ctx_psnr_delta": b["ctx_psnr"] - a["ctx_psnr"],
                "novel_ssim_delta": b["novel_ssim"] - a["novel_ssim"],
                "sem_miou_delta": b["sem_miou"] - a["sem_miou"],
                "alpha_gap_delta": (b.get("alpha_gap_attrs", 0.0) - a.get("alpha_gap_attrs", 0.0)),
                "embedding_same_delta": (
                    (b["embedding_similarity"]["same"] or 0.0)
                    - (a["embedding_similarity"]["same"] or 0.0)
                ),
            }
        ap_delta = [
            float(np.mean([v["ap50"] for v in results["scenes"][scene]["b"]["readout"]["novel"].values()]))
            - float(np.mean([v["ap50"] for v in results["scenes"][scene]["a"]["readout"]["novel"].values()]))
            for scene in paired
        ]
        results["paired"] = {
            "per_scene": paired,
            "novel_ap50_delta_mean": float(np.mean(ap_delta)) if ap_delta else 0.0,
            "novel_psnr_delta_mean": float(np.mean([v["novel_psnr_delta"] for v in paired.values()])),
            "novel_ssim_delta_mean": float(np.mean([v["novel_ssim_delta"] for v in paired.values()])),
            "sem_miou_delta_mean": float(np.mean([v["sem_miou_delta"] for v in paired.values()])),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, payload in results["runs"].items():
        summary = payload["summary"]
        print(f"[eval] {name}: ctx {summary['ctx_psnr']:.2f} novel {summary['novel_psnr']:.2f} "
              f"ssim {summary['ctx_ssim']:.3f}/{summary['novel_ssim']:.3f} "
              f"mIoU {summary['sem_miou']:.3f} AP50 {summary['novel_ap50']:.3f} "
              f"TP/FP/FN {summary['novel_tp']}/{summary['novel_fp']}/{summary['novel_fn']}")
    print(f"[eval] wrote {out_path}")
    return 0


def _write_tiles(name, entries, rows, image_dir: Path, opt) -> None:
    for entry, row in zip(entries, rows):
        tile = _build_tile(entry, row, opt)
        Image.fromarray(tile).save(image_dir / f"{entry['scene']}_{name}.png")


def _build_tile(entry, row, opt) -> np.ndarray:
    """GT | prediction | error, stacked for the first context and first novel view."""
    gt = entry["batch"]["images_all"][0].float().cpu().numpy()
    prediction = row["_render"]
    n_in = int(opt.num_input_views)
    rows = []
    for view in (0, n_in):
        error = np.abs(prediction[view] - gt[view]).mean(axis=0, keepdims=True).repeat(3, axis=0)
        rows.append(np.concatenate(
            [gt[view], np.clip(prediction[view], 0, 1), np.clip(error * 3.0, 0, 1)], axis=2
        ))
    tile = np.concatenate(rows, axis=1)
    return (np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
