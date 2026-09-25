#!/usr/bin/env python3
"""Unified (v2) GT-free scoring for the group experiment.

Class convention (SIU3R / processed ScanNet, checked against the code):
classes 0 and 1 are *stuff* (wall, floor), 2..19 are *thing* instances, and the
21st logit (index 20) is the **no-object** logit.  Two diagnostics are reported
and never mixed up:

* ``P(foreground) = 1 - softmax(21 logits)[20]``  (stuff + thing)
* ``P(thing)      = softmax(21 logits)[2:20].sum()``  (thing only)

The frozen GT-free instance rule of this round is
``P(thing) >= 0.5`` and the predicted class (argmax of the 20 group class
logits) must be a thing class, then ``mask > 0.5`` and ``area >= 50 px``.
A group predicted as wall/floor is never emitted as a thing instance.

The legacy score ``sigmoid(z20) = P(no-object)`` and the old
``group_view_predictions`` reader stay in ``scripts/group_locusgs_eval.py``
untouched, so the earlier reports remain reproducible.
"""
from __future__ import annotations

import numpy as np
import torch

from tokengs.models.ssst_contracts import NO_OBJECT_CLASS, SEMANTIC_CLASS_COUNT
from scripts.group_locusgs_eval import forward_group  # noqa: F401  (re-exported)
from scripts.object_locusgs_eval import (
    build_val_entries,
    gt_instances,
    instance_metrics,
    miou_from_confusion,
    purity_diagnostic,
    reconstruction_row,
    semantic_confusion,
)
from scripts.audit_group_scores import query_instance_iou_matrix

STUFF_CLASSES = (0, 1)
THING_CLASS_MIN = 2
THING_CLASS_MAX = SEMANTIC_CLASS_COUNT - 1  # 19

SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
MIN_PRED_PIXELS = 50


def group_score_table(forward: dict) -> dict:
    """Both diagnostics plus the class evidence, from one 21-way softmax."""
    raw = forward["group"]["objectness"][0].float()
    class_logits = forward["group"]["class_logits"][0].float()
    joint = torch.cat([class_logits, raw.unsqueeze(-1)], dim=-1)
    probability = torch.softmax(joint, dim=-1)
    argmax = class_logits.argmax(dim=-1)
    return {
        "raw_no_object_logit": raw,
        "p_no_object": probability[..., NO_OBJECT_CLASS],
        "p_foreground": 1.0 - probability[..., NO_OBJECT_CLASS],
        "p_thing": probability[..., THING_CLASS_MIN:SEMANTIC_CLASS_COUNT].sum(-1),
        "p_stuff": probability[..., :THING_CLASS_MIN].sum(-1),
        "class_argmax20": argmax,
        "is_thing_class": (argmax >= THING_CLASS_MIN) & (argmax <= THING_CLASS_MAX),
    }


def group_predictions_v2(
    forward: dict,
    view: int,
    *,
    score_key: str = "p_thing",
    require_thing_class: bool = True,
    score_threshold: float = SCORE_THRESHOLD,
    mask_threshold: float = MASK_THRESHOLD,
    min_area: int = MIN_PRED_PIXELS,
    table: dict | None = None,
):
    """Frozen v2 reader: score -> thing class -> mask > 0.5 -> area >= 50."""
    table = group_score_table(forward) if table is None else table
    scores = table[score_key]
    mass = forward["masks"]["group_mass"][0, view].float()
    counts = {"score": 0, "thing_class": 0, "mask": 0, "area": 0}
    predictions = []
    for group_index in range(mass.shape[0]):
        if float(scores[group_index]) < score_threshold:
            continue
        counts["score"] += 1
        if require_thing_class and not bool(table["is_thing_class"][group_index]):
            continue
        counts["thing_class"] += 1
        mask = (mass[group_index] > mask_threshold).cpu().numpy()
        if not mask.any():
            continue
        counts["mask"] += 1
        area = int(mask.sum())
        if area < min_area:
            continue
        counts["area"] += 1
        predictions.append({
            "group": group_index,
            "class": int(table["class_argmax20"][group_index]),
            "mask": mask,
            "area": area,
            "score": float(scores[group_index]),
            "p_thing": float(table["p_thing"][group_index]),
            "p_foreground": float(table["p_foreground"][group_index]),
            "raw_no_object_logit": float(table["raw_no_object_logit"][group_index]),
        })
    return predictions, counts


def background_stats(forward: dict, semantic_gt: np.ndarray, instance_gt: np.ndarray,
                     context_views) -> dict:
    """Mean background-slot mass / probability on GT stuff vs GT thing pixels."""
    mass_bg = forward["masks"]["background_mass"][0].float().cpu().numpy()  # [V,1,H,W]
    alpha = forward["masks"]["alpha"][0].float().cpu().numpy()
    group_mass = forward["masks"]["group_mass"][0].float().cpu().numpy()
    probability = mass_bg / np.clip(alpha, 0.5, None)
    rows = {"stuff": [], "thing": []}
    pixels = {"stuff": 0, "thing": 0}
    del context_views  # the diagnostic covers every rendered record
    for view in range(semantic_gt.shape[0]):
        sem = semantic_gt[view]
        ins = instance_gt[view]
        covered = alpha[view, 0] > 0.5
        stuff = ((sem == 0) | (sem == 1)) & covered
        thing = (sem >= THING_CLASS_MIN) & (sem <= THING_CLASS_MAX) & (ins > 0) & covered
        for name, mask in (("stuff", stuff), ("thing", thing)):
            if not mask.any():
                continue
            pixels[name] += int(mask.sum())
            rows[name].append({
                "mass": float(mass_bg[view, 0][mask].mean()),
                "probability": float(probability[view, 0][mask].mean()),
                "alpha": float(alpha[view, 0][mask].mean()),
            })
    out = {"pixels": pixels, "views": len(rows["stuff"]) + len(rows["thing"])}
    for name in ("stuff", "thing"):
        if rows[name]:
            out[f"{name}_mass_mean"] = float(np.mean([r["mass"] for r in rows[name]]))
            out[f"{name}_probability_mean"] = float(
                np.mean([r["probability"] for r in rows[name]])
            )
            out[f"{name}_alpha_mean"] = float(np.mean([r["alpha"] for r in rows[name]]))
        else:
            out[f"{name}_mass_mean"] = None
            out[f"{name}_probability_mean"] = None
            out[f"{name}_alpha_mean"] = None
    total_mass = max(float(group_mass.sum() + mass_bg.sum()), 1e-6)
    out["background_mass_fraction"] = float(mass_bg.sum() / total_mass)
    out["group_mass_fraction"] = float(group_mass.sum() / total_mass)
    return out


def best_over_groups(semantic_gt, instance_gt, forward, view) -> dict:
    """GT-assisted best IoU of the *current* masks over all 100 groups.

    No score gate is used; ``mask > 0.5`` and ``area >= 50`` stay fixed.  This
    describes what the current group masks can reach under the fixed
    thresholds - it is not a theoretical upper bound of the architecture or the
    token representation.
    """
    matrix = query_instance_iou_matrix(forward, view, semantic_gt, instance_gt)
    mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
    areas = (mass > MASK_THRESHOLD).reshape(mass.shape[0], -1).sum(axis=1)
    masked = matrix.copy()
    if masked.size:
        masked[areas < MIN_PRED_PIXELS] = 0.0
    per_gt = masked.max(axis=0) if masked.size else np.zeros(0)
    return {
        "per_gt_mean_best_iou": float(per_gt.mean()) if per_gt.size else 0.0,
        "per_gt_recall50": float((per_gt >= 0.5).mean()) if per_gt.size else 0.0,
        "n_gt_visible": int(per_gt.size),
        "note": "current masks at fixed mask>0.5 / area>=50, no score gate; "
                "not a theoretical upper bound",
    }


def evaluate_entry_v2(model, entry, opt, *, include_diagnostics: bool = False,
                      context_views=(0, 1)) -> dict:
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        row = reconstruction_row(entry, forward["output"]["render"], opt)
        semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
        confusion = semantic_confusion(
            forward["semantic_prob"], batch["semantic_label_all"].long(),
            forward["output"]["render"]["alphas_pred"],
        )
        row["sem_miou"], row["sem_iou_per_class"] = miou_from_confusion(confusion)
        table = group_score_table(forward)
        mass = forward["masks"]["group_mass"][0].float()
        alpha = forward["masks"]["alpha"][0, :, 0].float()
        row["mask_alpha_max_error"] = float(
            # mass is [V, Q, H, W]: sum over the 100 groups, then add the
            # background slot; the result must equal the rendered alpha.
            (mass.sum(dim=1) + forward["masks"]["background_mass"][0, :, 0] - alpha)
            .abs().max()
        )
        row["background"] = background_stats(forward, semantic_gt, instance_gt,
                                             context_views)
        row["scores"] = {
            "p_thing_mean": float(table["p_thing"].mean()),
            "p_thing_p90": float(table["p_thing"].quantile(0.9)),
            "p_foreground_mean": float(table["p_foreground"].mean()),
            "p_no_object_mean": float(table["p_no_object"].mean()),
            "raw_no_object_p50": float(table["raw_no_object_logit"].median()),
            "thing_class_fraction": float(table["is_thing_class"].float().mean()),
        }
        assignment = forward["group"]["slot_prob"][0].float().cpu().numpy()
        entropy = -(np.clip(assignment, 1e-8, 1) * np.log(np.clip(assignment, 1e-8, 1))).sum(-1)
        row["assignment"] = {
            "slot_entropy_mean": float(entropy.mean()),
            "slot_max_prob_mean": float(assignment.max(-1).mean()),
            "background_prob_mean": float(assignment[:, -1].mean()),
            "group_usage_active": int(
                (assignment[:, : model.num_groups].mean(0) > 1.0 / model.num_groups).sum()
            ),
        }
        views = {}
        for view in range(semantic_gt.shape[0]):
            predictions, counts = group_predictions_v2(forward, view, table=table)
            instances = gt_instances(semantic_gt, instance_gt, view)
            metrics = instance_metrics(predictions, instances)
            metrics["gate_counts"] = counts
            metrics["mean_score_passed"] = float(
                np.mean([p["score"] for p in predictions])
            ) if predictions else 0.0
            metrics["mean_p_foreground_passed"] = float(
                np.mean([p["p_foreground"] for p in predictions])
            ) if predictions else 0.0
            metrics["best_over_groups"] = best_over_groups(
                semantic_gt, instance_gt, forward, view
            )
            views[int(view)] = metrics
        row["views"] = views
        if include_diagnostics:
            row["purity"] = purity_diagnostic(model, batch, opt)
            means2d = forward["output"]["render"]["means2d_pred"][0].float()
            opacity = forward["output"]["gaussians"][0, :, 3].float()
            height, width = opt.img_size
            inside = (
                (means2d[..., 0] >= 0) & (means2d[..., 0] <= width)
                & (means2d[..., 1] >= 0) & (means2d[..., 1] <= height)
            ).any(dim=0)
            contributing = inside & (opacity > 0.05)
            row["purity_contributing"] = purity_diagnostic(
                model, batch, opt, gs_mask=contributing
            )
        row["_render"] = forward["output"]["render"]["images_pred"][0].float().cpu().numpy()
        row["_forward"] = forward
        row["_semantic_gt"] = semantic_gt
        row["_instance_gt"] = instance_gt
    return row


def summarise_v2(rows, *, novel_views=(2, 3)) -> dict:
    def mean(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values)) if values else float("nan")

    def view_stats(kind):
        ap, tp, fp, fn, preds, gt = [], 0, 0, 0, 0, 0
        gates = {"score": 0, "thing_class": 0, "mask": 0, "area": 0}
        best = []
        buckets = {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0},
                   "large": {"gt": 0, "tp": 0}}
        for row in rows:
            for view, metrics in row["views"].items():
                is_novel = int(view) in novel_views
                if (kind == "novel") != is_novel:
                    continue
                ap.append(metrics["ap50"])
                tp += metrics["tp"]
                fp += metrics["fp"]
                fn += metrics["fn"]
                preds += metrics["n_pred"]
                gt += metrics["n_gt"]
                for key in gates:
                    gates[key] += metrics["gate_counts"][key]
                best.append(metrics["best_over_groups"]["per_gt_mean_best_iou"])
                for stage in buckets:
                    buckets[stage]["gt"] += metrics["buckets"][stage]["gt"]
                    buckets[stage]["tp"] += metrics["buckets"][stage]["tp"]
        return {"ap50": float(np.mean(ap)) if ap else 0.0, "tp": tp, "fp": fp, "fn": fn,
                "n_pred": preds, "n_gt": gt, "gate_counts": gates,
                "best_over_groups_iou": float(np.mean(best)) if best else 0.0,
                "buckets": buckets}

    unseen = view_stats("novel")
    context = view_stats("context")
    summary = {
        "ctx_psnr": mean(["ctx_psnr"]),
        "novel_psnr": mean(["novel_psnr"]),
        "ctx_ssim": mean(["ctx_ssim"]),
        "novel_ssim": mean(["novel_ssim"]),
        "ctx_grey": mean(["ctx_grey"]),
        "novel_grey": mean(["novel_grey"]),
        "sem_miou": mean(["sem_miou"]),
        "mask_alpha_max_error": mean(["mask_alpha_max_error"]),
        "background_mass_fraction": mean(["background", "background_mass_fraction"]),
        "background_stuff_mass_mean": mean(["background", "stuff_mass_mean"]),
        "background_thing_mass_mean": mean(["background", "thing_mass_mean"]),
        "background_stuff_probability_mean": mean(
            ["background", "stuff_probability_mean"]),
        "background_thing_probability_mean": mean(
            ["background", "thing_probability_mean"]),
        "slot_entropy_mean": mean(["assignment", "slot_entropy_mean"]),
        "slot_max_prob_mean": mean(["assignment", "slot_max_prob_mean"]),
        "background_prob_mean": mean(["assignment", "background_prob_mean"]),
        "group_usage_active": mean(["assignment", "group_usage_active"]),
        "p_thing_mean": mean(["scores", "p_thing_mean"]),
        "p_thing_p90": mean(["scores", "p_thing_p90"]),
        "p_foreground_mean": mean(["scores", "p_foreground_mean"]),
        "thing_class_fraction": mean(["scores", "thing_class_fraction"]),
    }
    for name, stats in (("novel", unseen), ("context", context)):
        for key, value in stats.items():
            summary[f"{name}_{key}"] = value
    summary["novel_ap50"] = unseen["ap50"]
    summary["novel_tp"] = unseen["tp"]
    summary["novel_fp"] = unseen["fp"]
    summary["novel_fn"] = unseen["fn"]
    summary["ctx_ap50"] = context["ap50"]
    return summary


__all__ = [
    "MASK_THRESHOLD",
    "MIN_PRED_PIXELS",
    "SCORE_THRESHOLD",
    "STUFF_CLASSES",
    "THING_CLASS_MAX",
    "THING_CLASS_MIN",
    "background_stats",
    "best_over_groups",
    "build_val_entries",
    "evaluate_entry_v2",
    "group_predictions_v2",
    "group_score_table",
    "summarise_v2",
]
