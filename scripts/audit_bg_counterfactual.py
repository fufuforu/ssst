#!/usr/bin/env python3
"""Read-only counterfactual: renormalise G0+'s token->slot mass over the 100 groups.

For the same Gaussians, cameras and compositing weights, the original 101-way
assignment ``A[t, q]`` is replaced by

    A_cf[t, q]  = A[t, q] / sum_{j<100} A[t, j]      (q = 0..99)
    A_cf[t, bg] = 0

computed directly as ``softmax(slot_logits[t, :100])`` (numerically stable when the
background probability is close to 1).  Only *how the existing 100 groups share
the mass that previously went to the background slot* changes: the group-vs-group
relative logits, the query class/score logits, the Gaussians, the cameras and the
RGB/depth rendering are untouched.

No training, no optimizer step, no checkpoint write, no threshold change.  GT is
used only for the separately labelled best-over-groups diagnostic, never to pick
a group for a GT-free prediction.  This is a post-hoc probe of the *current*
checkpoint's mass flow, not an argument about retraining dynamics.
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
from tokengs.models.input_types import ModelInputDecoder  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.audit_mask_failure import (  # noqa: E402
    best_over_groups_iou,
    hard_metrics,
    soft_dice,
)
from scripts.eval_group_locusgs import build_train_entries  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    MASK_THRESHOLD,
    MIN_PRED_PIXELS,
    build_val_entries,
    forward_group,
    group_predictions_v2,
    group_score_table,
)
from scripts.object_locusgs_eval import gt_instances, instance_metrics  # noqa: E402

CHECKPOINTS = {
    "g0plus_step6000": "workspace_group_plus/arm_g0plus/ckpt_step6000",
    "g0_step6000": "workspace_group_locusgs/arm_g0/ckpt_step6000",
}
EQUIVALENCE_TOL = 1e-4          # |softmax(100 logits) - conditional renorm|
ANOMALY_SUM = 1e-3              # sum of the 100 group probabilities below this


def load_model(directory: str, preset: str, seed: int, device):
    payload = torch.load(Path(directory) / "train_state.pt", map_location="cpu",
                         weights_only=False)
    arm = str(payload["meta"].get("arm", "g0"))
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm="g1" if arm == "g1" else "g0",
        group_bg_supervision=arm == "g0plus", batch_size=1, num_workers=0,
        num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(directory) / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True,
    )
    model.eval()
    return opt, model, arm, int(payload["step"])


def counterfactual_assignment(slot_logits: torch.Tensor, slot_prob: torch.Tensor):
    """Return ``A_cf`` [B,T,101] (background column zeroed) plus diagnostics."""
    group_logits = slot_logits[..., :100]
    softmax_form = torch.softmax(group_logits, dim=-1)
    group_prob = slot_prob[..., :100]
    conditional = group_prob / group_prob.sum(-1, keepdim=True).clamp_min(1e-12)
    normal = group_prob.sum(-1) > ANOMALY_SUM
    gap = (softmax_form - conditional).abs().amax(dim=-1)
    anomalies = int((~normal).sum())
    return {
        "assignment": torch.cat(
            [softmax_form, torch.zeros_like(group_prob[..., :1])], dim=-1
        ),
        "anomalous_tokens": anomalies,
        "tokens": int(normal.numel()),
        "max_gap_softmax_vs_conditional": float(gap[normal].max()) if normal.any() else None,
        "max_gap_all_tokens": float(gap.max()),
    }


def region_mass(mass, truth):
    return float(mass[truth].mean()) if bool(truth.any()) else None


def variant_metrics(forward_like, semantic_gt, instance_gt, views, *, key=None):
    """GT-free read-out metrics for one variant (original or counterfactual)."""
    metrics = {"views": {}}
    tp = fp = fn = 0
    ap = []
    gt_total = 0
    for view in views:
        predictions, gate_counts = group_predictions_v2(forward_like, view)
        instances = gt_instances(semantic_gt, instance_gt, view)
        if key is not None:
            instances = {k: v for k, v in instances.items() if k == key}
        result = instance_metrics(predictions, instances)
        metrics["views"][view] = {
            "n_pred": result["n_pred"], "tp": result["tp"], "fp": result["fp"],
            "fn": result["fn"], "ap50": result["ap50"], "n_gt": result["n_gt"],
            "buckets": result["buckets"], "gate_counts": gate_counts,
        }
        tp += result["tp"]
        fp += result["fp"]
        fn += result["fn"]
        gt_total += result["n_gt"]
        if result["n_gt"]:
            ap.append(result["ap50"])
    metrics.update({"tp": tp, "fp": fp, "fn": fn, "n_gt": gt_total,
                    "ap50": float(np.mean(ap)) if ap else 0.0})
    return metrics


def instance_report(original_forward, cf_forward, semantic_gt, instance_gt, key, views):
    """Per view: mass split, best-over-groups mask quality and GT-free outcome."""
    rows = []
    for view in views:
        sem, ins = semantic_gt[view], instance_gt[view]
        thing = (sem >= 2) & (sem < 20) & (ins > 0) & ((sem + 1) * 1000 + ins == key)
        stuff = ((sem == 0) | (sem == 1))
        thing_t = torch.from_numpy(thing)
        stuff_t = torch.from_numpy(stuff)
        valid = torch.from_numpy(sem != 255)
        row = {"view": view, "kind": "context" if view < 2 else "novel",
               "gt_thing_area": int(thing.sum()), "gt_stuff_area": int(stuff.sum())}
        for tag, forward in (("original", original_forward), ("counterfactual", cf_forward)):
            mass = forward["masks"]["group_mass"][0, view].float().cpu()
            background = forward["masks"]["background_mass"][0, view, 0].float().cpu()
            alpha = forward["masks"]["alpha"][0, view, 0].float().cpu()
            row[f"{tag}_conservation_error"] = float(
                (mass.sum(0) + background - alpha).abs().max()
            )
            row[f"{tag}_background_mass_in_gt_thing"] = region_mass(background, thing_t)
            row[f"{tag}_group_mass_in_gt_thing"] = region_mass(mass.sum(0), thing_t)
            row[f"{tag}_background_mass_in_gt_stuff"] = region_mass(background, stuff_t)
            row[f"{tag}_group_mass_in_gt_stuff"] = region_mass(mass.sum(0), stuff_t)
            if thing.any():
                iou = best_over_groups_iou(mass, thing_t)
                best_group, best_metrics = None, None
                best = 0.0
                for group_index in range(mass.shape[0]):
                    m = hard_metrics(mass[group_index], thing_t)
                    if m["pred_area"] >= MIN_PRED_PIXELS and m["iou"] > best:
                        best, best_group, best_metrics = m["iou"], group_index, m
                row[f"{tag}_best_iou"] = best
                row[f"{tag}_best_group"] = best_group
                if best_metrics is not None:
                    best_metrics = dict(best_metrics)
                    best_metrics["soft_dice"] = soft_dice(
                        mass[best_group], thing_t, valid
                    )
                    for name, value in best_metrics.items():
                        row[f"{tag}_best_{name}"] = value
            predictions, _ = group_predictions_v2(forward, view)
            row[f"{tag}_n_gt_free_predictions"] = len(predictions)
            row[f"{tag}_gt_free_detected"] = any(
                int((p["mask"] & thing).sum()) / max(1, int((p["mask"] | thing).sum())) >= 0.5
                for p in predictions
            ) if thing.any() else None
            row[f"{tag}_gt_free_fp"] = sum(
                1 for p in predictions
                if not thing.any() or int((p["mask"] & thing).sum())
                / max(1, int((p["mask"] | thing).sum())) < 0.5
            )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--instances", default="group_plus/mask_failure/manifest.json")
    parser.add_argument("--out-dir", default="workspace_group_plus/bg_counterfactual")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.instances).read_text(encoding="utf-8"))["selected"]

    identities = {}
    for name, directory in CHECKPOINTS.items():
        path = Path(directory) / "model.pt"
        identities[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": path.stat().st_mtime,
        }
    opt_plus, model_plus, _, step_plus = load_model(
        CHECKPOINTS["g0plus_step6000"], args.preset, args.seed, device
    )
    opt_g0, model_g0, _, step_g0 = load_model(
        CHECKPOINTS["g0_step6000"], args.preset, args.seed, device
    )

    report = {
        "scope": "read-only counterfactual renormalising the 101-way assignment over "
                 "the 100 groups (background mass = 0); no training, no checkpoint write",
        "checkpoints": identities,
        "counterfactual_rule": "A_cf[t,:100] = softmax(slot_logits[t,:100]); "
                               "A_cf[t,bg] = 0; equivalent to A[t,q]/sum_j A[t,j]",
        "instances": {}, "validation_scenes": {}, "checks": {},
    }

    # ---------------- 9 fixed training instances ---------------- #
    entries = build_train_entries(opt_plus, split, plan, device, 8)
    entry_by_tuple = {(e["scene"], tuple(e["context"]), tuple(e["novel"])): e
                      for e in entries}
    anomaly_total = 0
    tokens_total = 0
    max_gap = 0.0
    for item in selection:
        tuple_key = (item["scene"], tuple(item["context"]), tuple(item["novel"]))
        if tuple_key not in entry_by_tuple:
            continue
        entry = entry_by_tuple[tuple_key]
        key = item["key"]
        with torch.no_grad():
            forward = forward_group(model_plus, entry["batch"], opt_plus)
            cf = counterfactual_assignment(
                forward["group"]["slot_logits"], forward["group"]["slot_prob"]
            )
            mask_decoder = ModelInputDecoder(
                cam_view=entry["batch"]["cam_view_all"],
                intrinsics=entry["batch"]["intrinsics_all"],
            )
            rendered_cf = model_plus.render_group_masks(
                forward["output"]["gaussians"], cf["assignment"], mask_decoder
            )
        forward_cf = {
            "group": forward["group"],
            "masks": rendered_cf,
        }
        anomaly_total += cf["anomalous_tokens"]
        tokens_total += cf["tokens"]
        if cf["max_gap_softmax_vs_conditional"] is not None:
            max_gap = max(max_gap, cf["max_gap_softmax_vs_conditional"])
        semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
        rows = instance_report(forward, forward_cf, semantic_gt, instance_gt, key,
                               range(int(opt_plus.num_views)))
        report["instances"][str(key)] = {
            "scene": item["scene"], "bucket": item["bucket"], "role": item["role"],
            "context": list(entry["context"]), "novel": list(entry["novel"]),
            "provenance": item["provenance"], "rows": rows,
        }
        del forward, cf, rendered_cf, forward_cf

    report["checks"]["assignment_equivalence"] = {
        "max_gap_softmax_vs_conditional_normalisation": max_gap,
        "tolerance": EQUIVALENCE_TOL,
        "anomalous_tokens_total": anomaly_total,
        "tokens_total": tokens_total,
        "handling": "the softmax over the 100 group logits is used for every token; it is "
                    "equivalent to the conditional normalisation wherever the group mass "
                    "is not degenerate",
    }

    # ---------------- 8 fixed validation windows (unseen) ---------------- #
    val_entries = build_val_entries(opt_plus, split, device)
    for entry in val_entries:
        semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
        views = range(int(opt_plus.num_views))
        scene = {"context": entry["context"], "novel": entry["novel"], "variants": {}}
        for tag, model, opt in (("g0plus_original", model_plus, opt_plus),
                                ("g0_original", model_g0, opt_g0)):
            with torch.no_grad():
                forward = forward_group(model, entry["batch"], opt)
            scene["variants"][tag] = variant_metrics(
                forward, semantic_gt, instance_gt, views
            )
            if tag == "g0plus_original":
                with torch.no_grad():
                    cf = counterfactual_assignment(
                        forward["group"]["slot_logits"], forward["group"]["slot_prob"]
                    )
                    mask_decoder = ModelInputDecoder(
                        cam_view=entry["batch"]["cam_view_all"],
                        intrinsics=entry["batch"]["intrinsics_all"],
                    )
                    rendered_cf = model.render_group_masks(
                        forward["output"]["gaussians"], cf["assignment"], mask_decoder
                    )
                    alpha_original = forward["masks"]["alpha"]
                    group_original = forward["masks"]["group_mass"]
                    background_original = forward["masks"]["background_mass"]
                    conservation_original = float(
                        (group_original.sum(2) + background_original[:, :, 0]
                         - alpha_original[:, :, 0]).abs().max()
                    )
                    conservation_cf = float(
                        (rendered_cf["group_mass"].sum(2) - alpha_original[:, :, 0])
                        .abs().max()
                    )
                    alpha_gap = float(
                        (rendered_cf["alpha"] - alpha_original).abs().max()
                    )
                    repeat = forward_group(model, entry["batch"], opt)
                scene["variants"]["g0plus_counterfactual"] = variant_metrics(
                    {"group": forward["group"], "masks": rendered_cf},
                    semantic_gt, instance_gt, views,
                )
                scene["checks"] = {
                    "conservation_original": conservation_original,
                    "conservation_counterfactual": conservation_cf,
                    "alpha_identical": alpha_gap,
                    "rgb_identical_on_repeat_forward": float(
                        (repeat["output"]["render"]["images_pred"]
                         - forward["output"]["render"]["images_pred"]).abs().max()
                    ),
                    "depth_identical_on_repeat_forward": float(
                        (repeat["output"]["render"]["depths_pred"]
                         - forward["output"]["render"]["depths_pred"]).abs().max()
                    ),
                    "anomalous_tokens": cf["anomalous_tokens"],
                    "tokens": cf["tokens"],
                }
                del cf, rendered_cf, repeat
            del forward
        report["validation_scenes"][entry["scene"]] = scene

    # ---------------- aggregate summary ---------------- #
    cross_up = cross_down = 0
    tp_gain = tp_loss = 0
    area_ratio_original, area_ratio_cf = [], []
    fp_original = fp_cf = 0
    for key, item in report["instances"].items():
        for row in item["rows"]:
            if row["kind"] != "novel":
                continue
            original_iou = row.get("original_best_iou")
            cf_iou = row.get("counterfactual_best_iou")
            if original_iou is not None and cf_iou is not None:
                cross_up += int(original_iou < 0.5 <= cf_iou)
                cross_down += int(cf_iou < 0.5 <= original_iou)
            detected_original = bool(row.get("original_gt_free_detected"))
            detected_cf = bool(row.get("counterfactual_gt_free_detected"))
            tp_gain += int(detected_cf and not detected_original)
            tp_loss += int(detected_original and not detected_cf)
            fp_original += int(row.get("original_gt_free_fp") or 0)
            fp_cf += int(row.get("counterfactual_gt_free_fp") or 0)
            for tag, store in (("original", area_ratio_original),
                               ("counterfactual", area_ratio_cf)):
                gt_area = row.get("gt_thing_area") or 0
                pred_area = row.get(f"{tag}_best_pred_area")
                if gt_area and pred_area is not None:
                    store.append(pred_area / gt_area)
    report["summary"] = {
        "fixed_training_instances": {
            "n_instances": len(report["instances"]),
            "novel_view_checks": cross_up + cross_down,
            "best_iou_crossed_0.5_up": cross_up,
            "best_iou_crossed_0.5_down": cross_down,
            "gt_free_tp_gained": tp_gain,
            "gt_free_tp_lost": tp_loss,
            "gt_free_fp_original": fp_original,
            "gt_free_fp_counterfactual": fp_cf,
            "mean_pred_over_gt_area_original": float(np.mean(area_ratio_original))
            if area_ratio_original else None,
            "mean_pred_over_gt_area_counterfactual": float(np.mean(area_ratio_cf))
            if area_ratio_cf else None,
        },
        "validation_scenes": {
            scene: {
                name: {k: metrics[k] for k in ("tp", "fp", "fn", "n_gt", "ap50")}
                for name, metrics in payload["variants"].items()
            }
            for scene, payload in report["validation_scenes"].items()
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_csv(out_dir / "per_instance.csv", report)
    figures = write_figures(out_dir, entries, report)
    print(json.dumps(report["summary"], indent=1))
    print("checks:", json.dumps(report["checks"], indent=1))
    print("figures:", figures)
    print(f"[cf] wrote {out_dir}")
    return 0


def write_csv(path: Path, report: dict) -> None:
    rows = []
    for key, item in report["instances"].items():
        for row in item["rows"]:
            flat = {"instance_key": key, "scene": item["scene"], "bucket": item["bucket"],
                    "role": item["role"], "view": row["view"], "kind": row["kind"]}
            for name, value in row.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    flat[name] = value
            rows.append(flat)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(out_dir: Path, entries, report: dict):
    """Three small comparison figures: diversion failure, existing TP, stuff region."""
    entry_by_tuple = {(e["scene"], tuple(e["context"]), tuple(e["novel"])): e
                      for e in entries}
    paths = []
    items = list(report["instances"].items())
    def novel_metric(item, name):
        values = [row.get(name) for row in item[1]["rows"] if row["kind"] == "novel"]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    diversion = max(
        items, key=lambda it: novel_metric(it, "original_background_mass_in_gt_thing") or 0.0
    )
    detected = next(
        (it for it in items if any(row.get("original_gt_free_detected")
                                   for row in it[1]["rows"] if row["kind"] == "novel")),
        None,
    )
    checks = [("diversion_failure", diversion)]
    if detected is not None:
        checks.append(("existing_true_positive", detected))
    for name, (key, item) in checks:
        tuple_key = (item["scene"], tuple(item["context"]), tuple(item["novel"]))
        entry = entry_by_tuple.get(tuple_key)
        if entry is None:
            continue
        view = 2
        with torch.no_grad():
            opt_plus, model_plus, _, _ = load_model(
                CHECKPOINTS["g0plus_step6000"], "train_siu3r_group_locusgs_ab", 42,
                torch.device("cuda"),
            )
            forward = forward_group(model_plus, entry["batch"], opt_plus)
            cf = counterfactual_assignment(
                forward["group"]["slot_logits"], forward["group"]["slot_prob"]
            )
            rendered_cf = model_plus.render_group_masks(
                forward["output"]["gaussians"], cf["assignment"],
                ModelInputDecoder(cam_view=entry["batch"]["cam_view_all"],
                                  intrinsics=entry["batch"]["intrinsics_all"]),
            )
        semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
        sem, ins = semantic_gt[view], instance_gt[view]
        truth = (sem >= 2) & (sem < 20) & (ins > 0) & ((sem + 1) * 1000 + ins == key)
        rgb = entry["batch"]["images_all"][0, view].float().cpu().numpy()
        mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
        mass_cf = rendered_cf["group_mass"][0, view].float().cpu().numpy()
        best_a, best_a_iou = None, 0.0
        best_b, best_b_iou = None, 0.0
        truth_t = torch.from_numpy(truth)
        for group_index in range(mass.shape[0]):
            for tag, source in (("a", mass), ("b", mass_cf)):
                metrics = hard_metrics(torch.from_numpy(source[group_index]), truth_t)
                if metrics["pred_area"] >= MIN_PRED_PIXELS:
                    if tag == "a" and metrics["iou"] > best_a_iou:
                        best_a, best_a_iou = group_index, metrics["iou"]
                    if tag == "b" and metrics["iou"] > best_b_iou:
                        best_b, best_b_iou = group_index, metrics["iou"]
        panels = [rgb]
        truth_panel = np.zeros(truth.shape + (3,))
        truth_panel[truth] = 1.0
        panels.append(truth_panel.transpose(2, 0, 1))
        for source, group_index in ((mass, best_a), (mass_cf, best_b)):
            panel = np.zeros(truth.shape + (3,))
            if group_index is not None:
                panel[source[group_index] > MASK_THRESHOLD] = 1.0
            panels.append(panel.transpose(2, 0, 1))
        if best_a is not None:
            error = np.zeros(truth.shape + (3,))
            error[truth & (mass[best_a] <= MASK_THRESHOLD)] = [1.0, 0.0, 0.0]
            error[(~truth) & (mass[best_a] > MASK_THRESHOLD)] = [0.0, 0.6, 1.0]
            panels.append(error.transpose(2, 0, 1))
        else:
            panels.append(np.zeros(truth.shape + (3,)).transpose(2, 0, 1))
        tile = np.concatenate(panels, axis=2)
        image = Image.fromarray(
            (np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        image = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.text((4, 4), f"{item['scene']} key {key} v{view} | GT | original best (g{best_a} "
                          f"IoU {best_a_iou:.3f}) | counterfactual best (g{best_b} IoU "
                          f"{best_b_iou:.3f}) | GT errors", fill=(255, 255, 0))
        path = out_dir / f"{name}_{item['scene']}_key{key}.png"
        image.save(path)
        paths.append(str(path))
        del forward, cf, rendered_cf

    # stuff-region figure from a validation scene
    val_entries = build_val_entries(
        config_defaults["train_siu3r_group_locusgs_ab"].evolve(
            seed=42, group_arm="g0", group_bg_supervision=True, batch_size=1,
            num_workers=0, num_input_views=2, num_views=4,
            dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        ),
        json.loads(Path("workspace_recon_diag/cross_scene/split.json").read_text()),
        torch.device("cuda"), scenes=["scene0059_00"],
    )
    entry = val_entries[0]
    opt_plus, model_plus, _, _ = load_model(
        CHECKPOINTS["g0plus_step6000"], "train_siu3r_group_locusgs_ab", 42,
        torch.device("cuda"),
    )
    with torch.no_grad():
        forward = forward_group(model_plus, entry["batch"], opt_plus)
        cf = counterfactual_assignment(
            forward["group"]["slot_logits"], forward["group"]["slot_prob"]
        )
        rendered_cf = model_plus.render_group_masks(
            forward["output"]["gaussians"], cf["assignment"],
            ModelInputDecoder(cam_view=entry["batch"]["cam_view_all"],
                              intrinsics=entry["batch"]["intrinsics_all"]),
        )
    view = 2
    sem = entry["batch"]["semantic_label_all"][0, view].long().cpu().numpy()
    rgb = entry["batch"]["images_all"][0, view].float().cpu().numpy()
    stuff = (sem == 0) | (sem == 1)
    background = forward["masks"]["background_mass"][0, view, 0].float().cpu().numpy()
    background_cf = rendered_cf["background_mass"][0, view, 0].float().cpu().numpy()
    label = np.zeros(sem.shape + (3,))
    label[stuff] = [1.0, 1.0, 1.0]
    label[(sem >= 2) & (sem < 20)] = [0.2, 0.2, 0.2]
    panels = [rgb, label.transpose(2, 0, 1),
              np.repeat(background[None], 3, 0), np.repeat(background_cf[None], 3, 0)]
    tile = np.concatenate(panels, axis=2)
    image = Image.fromarray((np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
    image = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
    ImageDraw.Draw(image).text(
        (4, 4), f"{entry['scene']} v{view} stuff region | GT(sem) | original background "
                f"mass | counterfactual background mass (0 by construction)",
        fill=(255, 255, 0),
    )
    path = out_dir / f"stuff_region_{entry['scene']}.png"
    image.save(path)
    paths.append(str(path))
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
