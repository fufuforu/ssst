#!/usr/bin/env python3
"""Development-set evaluation for the from-scratch G0/G1 group experiment.

Reuses, unchanged: the fixed 2+2 windows (`build_val_entries`), the
reconstruction/semantic metrics and the purity/embedding diagnostics of
`scripts/object_locusgs_eval.py`, the GT instance extraction, the instance
metrics and the class-agnostic `ap50` of the audited instance-query tooling, and
the token contribution maps of `scripts/train_instance_query_overfit.py`.

The GT-free reader is the frozen rule of this round: group objectness >= 0.5,
mask > 0.5, predicted area >= 50 px; the class comes from the group semantic
head.  No threshold is tuned on the validation scenes and no GT picks a group.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.object_locusgs import render_semantic_probability
from tokengs.models.ssst_contracts import SEMANTIC_CLASS_COUNT
from scripts.object_locusgs_eval import (
    READOUT_ALPHA,
    build_val_entries,
    embedding_similarity_diagnostic,
    gt_instances,
    instance_metrics,
    miou_from_confusion,
    psnr,
    purity_diagnostic,
    reconstruction_row,
    ssim_value,
)
from scripts.object_locusgs_eval import semantic_confusion  # noqa: F401  (re-exported)

OBJECTNESS_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
MIN_PRED_PIXELS = 50


def forward_group(model, batch, opt, *, all_views=True):
    """One forward: RGB/depth render, layer-10 group output, 101 mask channels."""
    model_input, _ = split_data(batch, opt)
    decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    output = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
    )
    group = model.layer10_group
    if group is None:
        raise RuntimeError("the group hook did not run")
    if all_views:
        mask_input = decoder_input
    else:
        mask_input = decoder_input.select_batch(slice(None), slice(0, int(opt.num_input_views)))
    rendered = model.render_group_masks(output["gaussians"], group["slot_prob"], mask_input)
    semantic_logits = model.decode_semantics(output["states"][-1]["tokens"])
    semantic_prob, semantic_alpha = render_semantic_probability(
        model.gs, output["gaussians"], semantic_logits, decoder_input.cam_view,
        decoder_input.intrinsics,
    )
    return {
        "output": output,
        "group": group,
        "masks": rendered,
        "semantic_prob": semantic_prob,
        "semantic_alpha": semantic_alpha,
    }


def group_view_predictions(forward: dict, view: int, *, with_classes=True):
    """Frozen GT-free reader for one rendered record."""
    objectness = torch.sigmoid(forward["group"]["objectness"][0]).float().cpu().numpy()
    class_logits = forward["group"]["class_logits"][0].float().cpu().numpy()
    mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
    predictions = []
    for group_index in range(mass.shape[0]):
        if objectness[group_index] < OBJECTNESS_THRESHOLD:
            continue
        mask = mass[group_index] > MASK_THRESHOLD
        area = int(mask.sum())
        if area < MIN_PRED_PIXELS:
            continue
        class_index = int(class_logits[group_index].argmax()) if with_classes else -1
        predictions.append({
            "group": group_index,
            "class": class_index,
            "mask": mask,
            "area": area,
            "score": float(objectness[group_index]),
        })
    return predictions


def _view_metrics(predictions, instances, forward, view):
    metrics = instance_metrics(predictions, instances)
    class_logits = forward["group"]["class_logits"][0].float()
    probability = torch.softmax(class_logits, dim=-1)
    # per visible GT instance: best IoU among the GT-free kept predictions
    per_instance = []
    order = sorted(
        range(len(predictions)), key=lambda i: -predictions[i]["score"]
    )
    matched_gt = set()
    for position in order:
        prediction = predictions[position]
        best, best_gt = 0.0, None
        for index, mask in instances.items():
            union = int((prediction["mask"] | mask).sum())
            if not union:
                continue
            iou = int((prediction["mask"] & mask).sum()) / union
            if iou > best:
                best, best_gt = iou, index
        if best_gt is not None and best >= 0.5 and best_gt not in matched_gt:
            matched_gt.add(best_gt)
    for index, mask in instances.items():
        candidates = []
        for prediction in predictions:
            union = int((prediction["mask"] | mask).sum())
            if not union:
                continue
            candidates.append(int((prediction["mask"] & mask).sum()) / union)
        best = max(candidates) if candidates else 0.0
        per_instance.append({
            "area": int(mask.sum()),
            "best_pred_iou": float(best),
            "recalled_at_50": bool(index in matched_gt),
            "pred_area": int(
                max(
                    (p["area"] for p in predictions
                     if int((p["mask"] & mask).sum()) > 0),
                    default=0,
                )
            ),
        })
    metrics["per_instance"] = per_instance
    metrics["recall50"] = float(
        np.mean([entry["recalled_at_50"] for entry in per_instance])
    ) if per_instance else 0.0
    metrics["mean_best_iou"] = float(
        np.mean([entry["best_pred_iou"] for entry in per_instance])
    ) if per_instance else 0.0
    metrics["mean_pred_class_prob"] = float(
        np.mean([
            float(probability[p["group"]].max())
            for p in predictions
        ])
    ) if predictions else 0.0
    return metrics


def evaluate_group_entry(model, entry, opt, *, include_diagnostics=False) -> dict:
    """All metrics for one fixed scene window."""
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        row = reconstruction_row(entry, forward["output"]["render"], opt)
        semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
        semantic_prob = forward["semantic_prob"]
        confusion = semantic_confusion(
            semantic_prob, batch["semantic_label_all"].long(),
            forward["output"]["render"]["alphas_pred"],
        )
        row["sem_miou"], row["sem_iou_per_class"] = miou_from_confusion(confusion)
        alpha = forward["semantic_alpha"][0].float().cpu().numpy()
        mass = forward["masks"]["group_mass"][0].float()
        alpha_masks = forward["masks"]["alpha"][0, :, 0].float()
        row["mask_alpha_max_error"] = float(
            (mass.sum(dim=1) + forward["masks"]["background_mass"][0, :, 0] - alpha_masks)
            .abs().max()
        )
        row["group_mass_mean"] = float(mass.mean())
        row["background_mass_fraction"] = float(
            forward["masks"]["background_mass"].sum() / mass.sum().clamp_min(1e-6)
        )
        objectness = torch.sigmoid(forward["group"]["objectness"][0]).float().cpu().numpy()
        row["objectness_p50"] = float(np.percentile(objectness, 50))
        row["objectness_p90"] = float(np.percentile(objectness, 90))
        row["objectness_ge_threshold"] = float((objectness >= OBJECTNESS_THRESHOLD).mean())
        assignment = forward["group"]["slot_prob"][0].float().cpu().numpy()
        entropy = -(np.clip(assignment, 1e-8, 1) * np.log(np.clip(assignment, 1e-8, 1))).sum(-1)
        row["slot_entropy_mean"] = float(entropy.mean())
        row["slot_max_prob_mean"] = float(assignment.max(-1).mean())
        row["background_prob_mean"] = float(assignment[:, -1].mean())
        row["group_usage_active"] = int(
            (assignment[:, : model.num_groups].mean(0) > 1.0 / model.num_groups).sum()
        )
        views = {}
        for view in range(semantic_gt.shape[0]):
            predictions = group_view_predictions(forward, view)
            instances = gt_instances(semantic_gt, instance_gt, view)
            views[int(view)] = _view_metrics(predictions, instances, forward, view)
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
            row["contributing_gs_fraction"] = float(contributing.float().mean())
            row["token_group_purity"] = token_group_purity(
                model, batch, opt, forward, entry
            )
    return row


def token_group_purity(model, batch, opt, forward, entry) -> dict:
    """Diagnostic: which fraction of a token's pixel contribution follows its
    dominant group (analytic per-token contribution maps, ``no_grad``, never a
    training path)."""
    from types import SimpleNamespace

    from scripts.train_instance_query_overfit import token_maps

    device = batch["images_all"].device
    maps, _alphas, _out = token_maps(model, batch, opt, SimpleNamespace(cell=16), device)
    token_mass = torch.stack([m.float() for m in maps]).sum(dim=0)  # [T, H*W]
    del entry
    assignment = forward["group"]["slot_prob"][0].detach().float().cpu().numpy()
    dominance = assignment[:, : model.num_groups].max(axis=1)
    dominant = assignment[:, : model.num_groups].argmax(axis=1)
    per_token = np.abs(token_mass.cpu().numpy()).sum(axis=1)
    valid = per_token > 0
    weighted = float((dominance[valid] * per_token[valid]).sum() / per_token[valid].sum())
    group_mass = np.zeros(assignment.shape[0])
    for group_index in np.unique(dominant):
        group_mass[int(group_index)] = per_token[dominant == group_index].sum()
    return {
        "tokens_with_mass": int(valid.sum()),
        "mean_dominant_group_probability": weighted,
        "generated_groups": int((group_mass > 0).sum()),
        "largest_group_share": float(group_mass.max() / max(1e-9, group_mass.sum())),
        "note": "analytic token contribution mass x the token's dominant-group "
                "probability; diagnostic only",
    }


def summarise_group(rows) -> dict:
    def mean(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values)) if values else float("nan")

    def view_stats(kind):
        ap, tp, fp, fn, recall, iou, preds, gt = [], 0, 0, 0, [], [], 0, 0
        buckets = {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0},
                   "large": {"gt": 0, "tp": 0}}
        for row in rows:
            n_views = len(row["views"])
            for view, metrics in row["views"].items():
                if kind == "novel" and int(view) < int(n_views // 2):
                    continue
                if kind == "context" and int(view) >= int(n_views // 2):
                    continue
                ap.append(metrics["ap50"])
                tp += metrics["tp"]
                fp += metrics["fp"]
                fn += metrics["fn"]
                recall.append(metrics["recall50"])
                iou.append(metrics["mean_best_iou"])
                preds += metrics["n_pred"]
                gt += metrics["n_gt"]
                for stage in buckets:
                    buckets[stage]["gt"] += metrics["buckets"][stage]["gt"]
                    buckets[stage]["tp"] += metrics["buckets"][stage]["tp"]
        return {
            "ap50": float(np.mean(ap)) if ap else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "n_pred": preds, "n_gt": gt,
            "recall50": float(np.mean(recall)) if recall else 0.0,
            "mean_best_iou": float(np.mean(iou)) if iou else 0.0,
            "buckets": buckets,
        }

    novel = view_stats("novel")
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
        "group_mass_mean": mean(["group_mass_mean"]),
        "background_mass_fraction": mean(["background_mass_fraction"]),
        "objectness_p90": mean(["objectness_p90"]),
        "objectness_ge_threshold": mean(["objectness_ge_threshold"]),
        "slot_entropy_mean": mean(["slot_entropy_mean"]),
        "slot_max_prob_mean": mean(["slot_max_prob_mean"]),
        "group_usage_active": mean(["group_usage_active"]),
    }
    for name, stats in (("novel", novel), ("context", context)):
        for key, value in stats.items():
            summary[f"{name}_{key}"] = value
    summary["novel_ap50"] = novel["ap50"]
    summary["novel_tp"] = novel["tp"]
    summary["novel_fp"] = novel["fp"]
    summary["novel_fn"] = novel["fn"]
    summary["ctx_ap50"] = context["ap50"]
    return summary


__all__ = [
    "MASK_THRESHOLD",
    "MIN_PRED_PIXELS",
    "OBJECTNESS_THRESHOLD",
    "build_val_entries",
    "evaluate_group_entry",
    "forward_group",
    "group_view_predictions",
    "summarise_group",
    "token_group_purity",
]
