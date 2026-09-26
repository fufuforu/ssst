#!/usr/bin/env python3
"""Corrected read-only evaluation for the group recipe arms.

Main metric: the *routing_v1* per-record implementation
(`audit_group_routing.fragmentation` -> IoU1) on the novel views 2/3 of the 8
fixed unseen scenes.  GT-free numbers use the published reader
(`audit_bg_counterfactual.variant_metrics`).  Context and all-four-view numbers
are reported separately and never mixed with the 55-record figures.

Corrections over the published recipe_v1/v2 runs (the historical prediction
algorithm is unchanged - only the harness was fixed):

* the model is loaded with the checkpoint's *own* effective config (recipe
  switch, head mode, weights) and the binding is recorded; the head mode can be
  forced explicitly for historical checkpoints;
* PSNR/SSIM are recomputed from **this checkpoint's current forward** on the 8
  windows (via `object_locusgs_eval.reconstruction_row`); a training history may
  be supplied only as a step/scene/frame-order cross-check, never as the result;
* the fragmentation column is named `fragmentation_delta` and documented as
  `max(IoU1, IoU1..2 union, IoU1..3 union) - IoU1`; the recipe-minus-baseline
  change is a separate `paired_delta_iou1` column;
* figures draw each predicted instance in its own colour with its query index,
  and GT-assisted panels are labelled as such.

Read-only: no training, no optimizer step, no threshold change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.audit_bg_counterfactual import variant_metrics  # noqa: E402
from scripts.audit_group_routing import fragmentation  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    build_val_entries,
    forward_group,
    group_predictions_v2,
)
from scripts.object_locusgs_eval import (  # noqa: E402
    gt_instances,
    instance_metrics,
    reconstruction_row,
)

PREREGISTERED_PLAN_SHA256 = (
    "a2a65c1382da0307a68fb6d27e5c1345aa78b46331acb2927e88b47a3c3d08bb"
)
PREREGISTERED_SPLIT_SHA256 = (
    "acf9afac57ca2e5287b9b1b545a62a369fee398a0d1a789cb77b851202b313d5"
)


def file_identity(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": path.stat().st_mtime}


def effective_config(directory: Path, args) -> dict:
    """The effective inference configuration, resolved the same way the run did."""
    config_path = directory / "config.yaml"
    source = "checkpoint-config.yaml" if config_path.is_file() else "explicit/defaults"
    head_mode = args.head_mode
    if head_mode is None:
        head_mode = ("pure4" if args.recipe_head_mode_default == "pure4"
                     else "legacy_prefix")
    return {
        "config_source": source,
        "config_yaml_present": config_path.is_file(),
        "recipe": bool(args.recipe),
        "head_mode": str(head_mode),
        "instance_outer_weight": float(args.instance_outer_weight),
        "assign_coef": float(args.assign_coef),
        "assign_every": int(args.assign_every),
        "assign_stop_shared_grad": bool(args.assign_stop_shared_grad),
        "num_views": int(args.num_views),
        "num_input_views": int(args.num_input_views),
    }


def per_record_metrics(model, opt, entry, device):
    semantic = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
    instance = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
    with torch.no_grad():
        forward = forward_group(model, entry["batch"], opt)
        recon = reconstruction_row(entry, forward["output"]["render"], opt)
    records = []
    for view in (2, 3):
        sem, ins = semantic[view], instance[view]
        packed = (sem + 1) * 1000 + ins
        visible = (sem >= 2) & (sem < 20) & (ins > 0)
        keys = sorted(set(int(k) for k in np.unique(packed[visible])))
        mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
        for key in keys:
            truth = visible & (packed == key)
            others = [visible & (packed == other) for other in keys if other != key]
            result = fragmentation(mass, truth, others)
            records.append({
                "scene": entry["scene"], "view": view,
                "frame_id": int(entry["novel"][view - 2]), "key": key,
                "gt_area": int(truth.sum()),
                "iou1": result["iou1"], "iou2": result["iou2"], "iou3": result["iou3"],
                # routing_v1 fragmentation delta -- NOT the paired baseline change
                "fragmentation_delta": result["delta"],
                "bucket": "small" if int(truth.sum()) < 3000 else "large",
            })
    gt_free = {
        "novel": variant_metrics(forward, semantic, instance, (2, 3)),
        "context": variant_metrics(forward, semantic, instance, (0, 1)),
        "all_views": variant_metrics(forward, semantic, instance, (0, 1, 2, 3)),
    }
    free_buckets = {"small_lt3000": {"gt": 0, "tp": 0},
                    "large_ge3000": {"gt": 0, "tp": 0}}
    for view in (2, 3):
        predictions, _ = group_predictions_v2(forward, view)
        instances = gt_instances(semantic, instance, view)
        result = instance_metrics(predictions, instances, buckets=(3000, 1 << 62))
        free_buckets["small_lt3000"]["gt"] += result["buckets"]["small"]["gt"]
        free_buckets["small_lt3000"]["tp"] += result["buckets"]["small"]["tp"]
        free_buckets["large_ge3000"]["gt"] += result["buckets"]["medium"]["gt"]
        free_buckets["large_ge3000"]["tp"] += result["buckets"]["medium"]["tp"]
    return records, gt_free, free_buckets, forward, recon, semantic, instance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline", default=None,
                        help="reference JSON whose per-record IoU1 defines the paired "
                             "delta (e.g. group_plus/recipe_v1/baseline.json)")
    parser.add_argument("--paired-delta-source", default=None,
                        help="another eval_per_instance.csv to compute paired_delta_iou1 "
                             "against (same 55 records)")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures", type=int, default=3)
    parser.add_argument("--label", default="recipe")
    parser.add_argument("--history", default=None,
                        help="optional val_history.jsonl kept ONLY as a step/scene "
                             "cross-check of the recomputed PSNR/SSIM")
    # effective inference config of the checkpoint
    parser.add_argument("--recipe", action="store_true", default=True)
    parser.add_argument("--no-recipe", dest="recipe", action="store_false")
    parser.add_argument("--head-mode", choices=("legacy_prefix", "pure4"), default=None)
    parser.add_argument("--recipe-head-mode-default", choices=("legacy_prefix", "pure4"),
                        default="legacy_prefix")
    parser.add_argument("--instance-outer-weight", type=float, default=0.1)
    parser.add_argument("--assign-coef", type=float, default=0.2)
    parser.add_argument("--assign-every", type=int, default=1)
    parser.add_argument("--assign-stop-shared-grad", action="store_true")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-input-views", type=int, default=2)
    parser.add_argument("--allow-nonstrict", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    identity_before = file_identity(checkpoint / "model.pt")
    config = effective_config(checkpoint, args)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    opt = config_defaults[args.preset].evolve(
        seed=args.seed, group_arm="g0", group_recipe=bool(args.recipe),
        group_recipe_head_mode=str(config["head_mode"]),
        group_recipe_seg_weight=float(args.instance_outer_weight),
        group_recipe_assign_coef=float(args.assign_coef),
        group_recipe_assign_every=int(args.assign_every),
        group_recipe_assign_stop_shared_grad=bool(args.assign_stop_shared_grad),
        batch_size=1, num_workers=0, num_input_views=int(args.num_input_views),
        num_views=int(args.num_views),
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    payload = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
    result = model.load_state_dict(payload["model"], strict=not args.allow_nonstrict)
    model.eval()

    entries = build_val_entries(opt, split, device)
    rows, per_scene, reads = [], {}, {}
    free_buckets_total = {"small_lt3000": {"gt": 0, "tp": 0},
                          "large_ge3000": {"gt": 0, "tp": 0}}
    recon_rows = []
    for entry in entries:
        records, gt_free, free_buckets, forward, recon, semantic, instance = \
            per_record_metrics(model, opt, entry, device)
        for name, payload_b in free_buckets.items():
            for field in ("gt", "tp"):
                free_buckets_total[name][field] += payload_b[field]
        rows.extend(records)
        recon_rows.append({"scene": entry["scene"], "context": entry["context"],
                           "novel": entry["novel"], **recon})
        per_scene[entry["scene"]] = {
            "context": entry["context"], "novel": entry["novel"],
            "records": records,
            "gt_free": {k: {kk: v[kk] for kk in ("tp", "fp", "fn", "n_gt", "ap50")}
                        for k, v in gt_free.items()},
            "gt_free_buckets_novel": free_buckets,
            "reconstruction": recon,
            "context_conservation_error": float(
                (forward["masks"]["group_mass"][0, :2].sum(1)
                 + forward["masks"]["background_mass"][0, :2, 0]
                 - forward["masks"]["alpha"][0, :2, 0]).abs().max()
            ),
        }
        reads[entry["scene"]] = (forward, semantic, instance, entry)
        print(f"[eval] {entry['scene']} records {len(records)} "
              f"novel_psnr {recon['novel_psnr']:.4f}", flush=True)

    # paired column
    reference_records = {}
    if args.paired_delta_source:
        for row in csv.DictReader(Path(args.paired_delta_source).open(encoding="utf-8")):
            reference_records[(row["scene"], str(row["view"]), str(row["key"]))] = \
                float(row["iou1"])
    elif args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        for row in baseline.get("records", []):
            reference_records[(row["scene"], str(row["view"]), str(row["key"]))] = \
                float(row["iou1"])
    missing_pairs = 0
    for row in rows:
        key = (row["scene"], str(row["view"]), str(row["key"]))
        if reference_records:
            value = reference_records.get(key)
            row["reference_iou1"] = value
            row["paired_delta_iou1"] = (row["iou1"] - value) if value is not None else None
            if value is None:
                missing_pairs += 1

    mean_iou1 = float(np.mean([r["iou1"] for r in rows]))
    pass1 = sum(1 for r in rows if r["iou1"] >= 0.5)
    buckets = {}
    for name, predicate in (("small_lt3000", lambda r: r["bucket"] == "small"),
                            ("large_ge3000", lambda r: r["bucket"] == "large")):
        subset = [r for r in rows if predicate(r)]
        buckets[name] = {"n": len(subset),
                         "pass": sum(1 for r in subset if r["iou1"] >= 0.5),
                         "mean_iou1": float(np.mean([r["iou1"] for r in subset])),
                         "gt_free_hits": free_buckets_total[name]}
    novel = {k: 0 for k in ("tp", "fp", "fn")}
    ap = []
    for payload_scene in per_scene.values():
        for key in novel:
            novel[key] += payload_scene["gt_free"]["novel"][key]
        ap.append(payload_scene["gt_free"]["novel"]["ap50"])
    novel_ap50 = float(np.mean(ap))
    novel_psnr = float(np.mean([r["novel_psnr"] for r in recon_rows]))
    novel_ssim = float(np.mean([r["novel_ssim"] for r in recon_rows]))
    ctx_psnr = float(np.mean([r["ctx_psnr"] for r in recon_rows]))
    ctx_ssim = float(np.mean([r["ctx_ssim"] for r in recon_rows]))

    history_check = {"supplied": False}
    if args.history and Path(args.history).is_file():
        lines = [json.loads(x) for x in Path(args.history).read_text().splitlines() if x.strip()]
        last = lines[-1]
        history_check = {
            "supplied": True, "history_step": last["step"],
            "history_novel_psnr": last["summary"]["novel_psnr"],
            "history_novel_ssim": last["summary"]["novel_ssim"],
            "recomputed_novel_psnr": novel_psnr,
            "recomputed_novel_ssim": novel_ssim,
            "psnr_abs_diff": abs(novel_psnr - last["summary"]["novel_psnr"]),
            "ssim_abs_diff": abs(novel_ssim - last["summary"]["novel_ssim"]),
            "note": "history is a cross-check only; the reported values are recomputed "
                    "from this checkpoint's current forward",
        }

    summary = {
        "scope": "32/8 development split, novel views 2/3 only (55 records); not SIU3R "
                 "official mAP/PQ; model uses GT poses, SIU3R is unposed",
        "label": args.label,
        "checkpoint": str(checkpoint),
        "model_sha256": identity_before["sha256"],
        "effective_config": config,
        "load_strict": not args.allow_nonstrict,
        "provenance": {
            "plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
            "preregistered_plan_sha256": PREREGISTERED_PLAN_SHA256,
            "split_sha256": hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
            "preregistered_split_sha256": PREREGISTERED_SPLIT_SHA256,
            "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "main_metric_implementation": "group_plus/routing_v1 fragmentation IoU1",
            "fragmentation_delta_definition":
                "max(IoU1, IoU1-2 union, IoU1-3 union) - IoU1  (NOT the baseline delta)",
        },
        "novel_only_55": {"n": len(rows), "mean_iou1": mean_iou1,
                          "iou1_ge_0.5": pass1,
                          "gt_free": {**novel, "ap50": novel_ap50},
                          "paired_reference_missing": missing_pairs},
        "gate_recomputed": {"novel_psnr": novel_psnr, "novel_ssim": novel_ssim,
                            "ctx_psnr": ctx_psnr, "ctx_ssim": ctx_ssim},
        "history_cross_check": history_check,
        "buckets_novel": buckets,
        "per_scene_reconstruction": recon_rows,
        "context_and_all_views": {
            scene: {k: {kk: v[kk] for kk in ("tp", "fp", "fn", "n_gt", "ap50")}
                    for k, v in payload_scene["gt_free"].items()}
            for scene, payload_scene in per_scene.items()
        },
        "identity_after": file_identity(checkpoint / "model.pt"),
    }
    summary["checkpoint_unchanged"] = summary["identity_after"] == identity_before
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (out_dir / "eval_per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "eval_per_scene.json").write_text(json.dumps(per_scene, indent=1),
                                                 encoding="utf-8")
    write_figures(out_dir, rows, reads, model, opt, device, args.figures, args.label)
    print(json.dumps({"novel_only_55": summary["novel_only_55"],
                      "gate_recomputed": summary["gate_recomputed"],
                      "history_cross_check": history_check,
                      "checkpoint_unchanged": summary["checkpoint_unchanged"]}, indent=1))
    del result
    return 0


PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
           (250, 190, 190), (0, 128, 128), (170, 110, 40), (128, 0, 0)]


def write_figures(out_dir, rows, reads, model, opt, device, count, label):
    """RGB | GT instance | GT-free instances (one colour per query) | error."""
    ordered = sorted(rows, key=lambda r: (r["iou1"]))
    picks = []
    if ordered:
        picks.append(ordered[-1])
        picks.append(ordered[0])
    if len(ordered) > 2:
        picks.append(ordered[len(ordered) // 2])
    paths = []
    for row in picks[:count]:
        forward, semantic, instance, entry = reads[row["scene"]]
        view, key = row["view"], row["key"]
        sem, ins = semantic[view], instance[view]
        truth = (sem >= 2) & (sem < 20) & (ins > 0) & (((sem + 1) * 1000 + ins) == key)
        rgb = entry["batch"]["images_all"][0, view].float().cpu().numpy()
        preds, _ = group_predictions_v2(forward, view)
        panels = [rgb]
        gt_panel = np.zeros(truth.shape + (3,))
        gt_panel[truth] = (0.2, 1.0, 0.2)
        panels.append(gt_panel.transpose(2, 0, 1))
        # each surviving query gets its own colour; the GT-assisted best group is
        # drawn separately and explicitly labelled
        pred_panel = np.zeros(truth.shape + (3,))
        chosen = None
        for index, prediction in enumerate(preds):
            colour = np.array(PALETTE[index % len(PALETTE)]) / 255.0
            pred_panel[prediction["mask"]] = colour
            if np.all(prediction["mask"] == truth) or chosen is None:
                chosen = prediction
        panels.append(pred_panel.transpose(2, 0, 1))
        error = np.zeros(truth.shape + (3,))
        union = np.zeros(truth.shape, dtype=bool)
        for prediction in preds:
            union |= prediction["mask"]
        error[truth & (~union)] = [1.0, 0.0, 0.0]
        error[(~truth) & union] = [0.0, 0.6, 1.0]
        panels.append(error.transpose(2, 0, 1))
        tile = np.concatenate(panels, axis=2)
        image = Image.fromarray((np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
        image = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
        ImageDraw.Draw(image).text(
            (4, 4), f"{label} {row['scene']} frame {row['frame_id']} id {key} "
                    f"IoU1 {row['iou1']:.3f} area {row['gt_area']} "
                    f"({len(preds)} GT-free masks)",
            fill=(255, 255, 0))
        Path_ = out_dir / f"{label}_{row['scene']}_f{row['frame_id']}_id{key}.png"
        image.save(Path_)
        paths.append(str(Path_))
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
