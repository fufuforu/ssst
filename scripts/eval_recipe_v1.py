#!/usr/bin/env python3
"""Fixed evaluation for the recipe-v1 single arm (pre-registered acceptance).

Main metric uses the *routing_v1* per-record implementation
(`audit_group_routing.fragmentation` -> IoU1) on the novel views 2/3 of the 8
fixed unseen scenes, aligned by (scene, view, key) with
`group_plus/recipe_v1/baseline.json`.  GT-free numbers use the same reader as the
published G0+ evaluation (`audit_bg_counterfactual.variant_metrics`).  Context and
all-four-view numbers are reported separately and never mixed with the 55-record
figures.  Read-only: no training, no threshold change.
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
from scripts.audit_mask_failure import hard_metrics, load_model  # noqa: E402
from scripts.group_eval_v2 import build_val_entries, forward_group, group_predictions_v2  # noqa: E402


def file_identity(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": path.stat().st_mtime}


def per_record_metrics(model, opt, entry, device):
    """routing_v1-style IoU1 per novel record plus the GT-free reader output."""
    semantic = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
    instance = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
    with torch.no_grad():
        forward = forward_group(model, entry["batch"], opt)
    records, best_masks = [], {}
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
                "delta": result["delta"],
                "bucket": "small" if int(truth.sum()) < 3000 else "large",
            })
            best_group = result["ranked_groups"][0] if result["ranked_groups"] else None
            best_masks[(view, key)] = (best_group, mass, truth)
    gt_free = {
        "novel": variant_metrics(forward, semantic, instance, (2, 3)),
        "context": variant_metrics(forward, semantic, instance, (0, 1)),
        "all_views": variant_metrics(forward, semantic, instance, (0, 1, 2, 3)),
    }
    return records, gt_free, forward, semantic, instance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", default="group_plus/recipe_v1/baseline.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--out-dir", default="group_plus/recipe_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    identity_before = file_identity(checkpoint / "model.pt")
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    opt = config_defaults[args.preset].evolve(
        seed=args.seed, group_arm="g0", group_recipe=True,
        group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scanned".replace("scanned", "scannet")},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    model.eval()
    entries = build_val_entries(opt, split, device)
    rows, per_scene, reads = [], {}, {}
    for entry in entries:
        records, gt_free, forward, semantic, instance = per_record_metrics(
            model, opt, entry, device
        )
        rows.extend(records)
        per_scene[entry["scene"]] = {
            "context": entry["context"], "novel": entry["novel"],
            "records": records,
            "gt_free": {k: {kk: v[kk] for kk in ("tp", "fp", "fn", "n_gt", "ap50", "n_pred")}
                        for k, v in gt_free.items()},
            "context_conservation_error": float(
                (forward["masks"]["group_mass"][0, :2].sum(0)
                 + forward["masks"]["background_mass"][0, :2, 0]
                 - forward["masks"]["alpha"][0, :2, 0]).abs().max()
            ),
        }
        reads[entry["scene"]] = (forward, semantic, instance, entry)
        print(f"[recipe-eval] {entry['scene']} records {len(records)}", flush=True)
    baseline_records = {(r["scene"], r["view"], r["key"]): r for r in baseline["records"]}
    for row in rows:
        base = baseline_records.get((row["scene"], row["view"], row["key"]))
        row["baseline_iou1"] = base["iou1"] if base else None
    mean_iou1 = float(np.mean([r["iou1"] for r in rows]))
    pass1 = sum(1 for r in rows if r["iou1"] >= 0.5)
    base_iou1 = baseline["main_metric"]["best_over_groups_iou_mean"]
    base_pass1 = baseline["main_metric"]["iou_ge_0.5_count"]
    buckets = {}
    for name, predicate in (("small_lt3000", lambda r: r["bucket"] == "small"),
                            ("large_ge3000", lambda r: r["bucket"] == "large")):
        subset = [r for r in rows if predicate(r)]
        buckets[name] = {"n": len(subset),
                         "recipe_pass": sum(1 for r in subset if r["iou1"] >= 0.5),
                         "recipe_mean_iou1": float(np.mean([r["iou1"] for r in subset]))}
    novel = {k: 0 for k in ("tp", "fp", "fn")}
    ap = []
    for payload in per_scene.values():
        for key in novel:
            novel[key] += payload["gt_free"]["novel"][key]
        ap.append(payload["gt_free"]["novel"]["ap50"])
    novel_ap50 = float(np.mean(ap))
    psnr = float(np.mean([
        per_scene[s]["gt_free"]["novel"].get("psnr", 0.0) for s in per_scene
    ])) if False else None
    # novel PSNR from the training-time eval history (same 8 windows/checkpoint)
    history = Path("workspace_group_plus/recipe_v1/run/val_history.jsonl")
    psnr_novel = ssim_novel = None
    if history.is_file():
        last = json.loads(history.read_text(encoding="utf-8").strip().splitlines()[-1])
        psnr_novel = last["summary"]["novel_psnr"]
        ssim_novel = last["summary"]["novel_ssim"]
        ctx_psnr = last["summary"]["ctx_psnr"]
    else:
        ctx_psnr = None
    acceptance = {
        "mask_iou_mean": {"value": mean_iou1, "baseline": base_iou1,
                          "delta": mean_iou1 - base_iou1,
                          "required_delta": 0.05,
                          "passed": mean_iou1 - base_iou1 >= 0.05},
        "mask_pass_count": {"value": pass1, "baseline": base_pass1,
                            "delta": pass1 - base_pass1, "required_delta": 3,
                            "passed": pass1 - base_pass1 >= 3},
        "gt_free_tp": {"value": novel["tp"], "baseline": baseline["gt_free"]["tp"],
                       "delta": novel["tp"] - baseline["gt_free"]["tp"],
                       "required_delta": 3,
                       "passed": novel["tp"] - baseline["gt_free"]["tp"] >= 3,
                       "fp": novel["fp"], "fp_baseline": baseline["gt_free"]["fp"],
                       "fn": novel["fn"]},
        "gt_free_ap50": {"value": novel_ap50,
                         "baseline": baseline["gt_free"]["ap50"],
                         "delta": novel_ap50 - baseline["gt_free"]["ap50"],
                         "required_delta": 0.03,
                         "passed": novel_ap50 - baseline["gt_free"]["ap50"] >= 0.03},
        "gate_novel_psnr": {"value": psnr_novel, "baseline": baseline["gate"]["novel_psnr"],
                            "delta": (psnr_novel - baseline["gate"]["novel_psnr"])
                            if psnr_novel is not None else None,
                            "allowed_drop": 0.20,
                            "passed": (psnr_novel is not None
                                       and psnr_novel - baseline["gate"]["novel_psnr"] >= -0.20)},
    }
    acceptance["overall_passed"] = all(
        v["passed"] for k, v in acceptance.items() if isinstance(v, dict)
    )
    summary = {
        "scope": "32/8 development split, novel views 2/3 only (55 records); "
                 "not SIU3R official mAP/PQ; model uses GT poses, SIU3R is unposed",
        "checkpoint": str(checkpoint),
        "model_sha256": identity_before["sha256"],
        "main_metric_implementation": "group_plus/routing_v1 per-record fragmentation "
                                      "(IoU1), aligned by (scene, view, key)",
        "novel_only_55": {"mean_iou1": mean_iou1, "iou1_ge_0.5": pass1,
                          "gt_free": {**novel, "ap50": novel_ap50}},
        "context_and_all_views": {
            scene: {k: {kk: v[kk] for kk in ("tp", "fp", "fn", "n_gt", "ap50")}
                    for k, v in payload["gt_free"].items()}
            for scene, payload in per_scene.items()
        },
        "buckets_novel": buckets,
        "gate": {"novel_psnr": psnr_novel, "novel_ssim": ssim_novel, "ctx_psnr": ctx_psnr},
        "acceptance": acceptance,
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
    write_figures(out_dir, rows, reads, model, opt, device, args.figures)
    print(json.dumps(summary["novel_only_55"], indent=1))
    print(json.dumps(acceptance, indent=1))
    return 0


def write_figures(out_dir, rows, reads, model, opt, device, count):
    """RGB | GT instance | GT-free mask | error for a few fixed records."""
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
        mask = np.zeros(truth.shape, dtype=bool)
        for prediction in preds:
            mask |= prediction["mask"]
        panels = [rgb]
        for array, colour in ((truth, (0.2, 1.0, 0.2)), (mask, (1.0, 1.0, 1.0))):
            panel = np.zeros(truth.shape + (3,))
            panel[array] = colour
            panels.append(panel.transpose(2, 0, 1))
        error = np.zeros(truth.shape + (3,))
        error[truth & (~mask)] = [1.0, 0.0, 0.0]
        error[(~truth) & mask] = [0.0, 0.6, 1.0]
        panels.append(error.transpose(2, 0, 1))
        tile = np.concatenate(panels, axis=2)
        image = Image.fromarray((np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
        image = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
        ImageDraw.Draw(image).text(
            (4, 4), f"{row['scene']} frame {row['frame_id']} id {key} "
                    f"IoU1 {row['iou1']:.3f} (baseline {row['baseline_iou1']:.3f}) "
                    f"area {row['gt_area']}",
            fill=(255, 255, 0),
        )
        path = out_dir / f"recipe_{row['scene']}_f{row['frame_id']}_id{key}.png"
        image.save(path)
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
