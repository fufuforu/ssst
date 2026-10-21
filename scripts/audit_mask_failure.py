#!/usr/bin/env python3
"""Read-only localisation of mid/large instance-mask failures (G0 vs G0+).

For a *fixed*, pre-registered sample of mid and large GT thing instances taken
from training windows that are provably part of the executed 6000-step plan,
this reports - per checkpoint (G0 and G0+ at step3000 and step6000) and per
instance - coverage, the Hungarian match on the training sampling, the best of
the existing 100 group masks, mask-shape/read-out diagnostics, a read-only
recomputation of the training signal, and a full-valid-pixel counterfactual
matching.

No training, no optimizer step, no checkpoint write, no threshold change.
GT is used only for diagnostic matching and scoring, never to pick a query for a
GT-free prediction.  ``best-over-groups`` is the best of the *current* 100 group
masks under the fixed thresholds, not a theoretical upper bound.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, split_data  # noqa: E402
from tokengs.models.ssst_loss import (  # noqa: E402
    _pairwise_bce,
    _pairwise_dice,
    _sample_points,
    build_context_segments,
    hungarian_match,
)
from tokengs.options import config_defaults  # noqa: E402
from scripts.eval_group_locusgs import build_train_entries  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    MASK_THRESHOLD,
    MIN_PRED_PIXELS,
    SCORE_THRESHOLD,
    forward_group,
    group_predictions_v2,
    group_score_table,
)
from scripts.object_locusgs_eval import gt_instances  # noqa: E402

CHECKPOINTS = (
    ("g0_step3000", "workspace_group_locusgs/arm_g0/ckpt_step3000"),
    ("g0_step6000", "workspace_group_locusgs/arm_g0/ckpt_step6000"),
    ("g0plus_step3000", "workspace_group_plus/arm_g0plus/ckpt_step3000"),
    ("g0plus_step6000", "workspace_group_plus/arm_g0plus/ckpt_step6000"),
)
TRAIN_LOGS = ("workspace_group_locusgs/arm_g0/train_log.jsonl",
              "workspace_group_plus/arm_g0plus/train_log.jsonl")
INSTANCE_KEYS = ("key", "scene", "context", "novel")


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
    return opt, model, arm, int(payload["step"]), payload["meta"]


def proven_training_tuples() -> set:
    proven = set()
    for path in TRAIN_LOGS:
        if not Path(path).is_file():
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            proven.add((row["scene"], tuple(row["context"]), tuple(row["novel"])))
    return proven


def thing_masks_and_keys(semantic_t, instance_t):
    """Thing segments (context-merged) plus their scene-stable keys."""
    classes, masks = build_context_segments(semantic_t, instance_t, (0, 1))
    labels = classes[0]
    masks = masks[0]
    keep = labels >= 2
    labels, masks = labels[keep], masks[keep]
    keys = []
    for index in range(masks.shape[0]):
        key = None
        for view in range(masks.shape[1]):
            sub = masks[index, view].bool()
            if sub.any():
                position = int(torch.nonzero(sub, as_tuple=False)[0, 0])
                row = int(torch.nonzero(sub, as_tuple=False)[0, 1])
                sem_value = int(semantic_t[0, view, position, row])
                ins_value = int(instance_t[0, view, position, row])
                key = (sem_value + 1) * 1000 + ins_value
                break
        keys.append(key)
    return labels, masks, keys


def sampled_cost_terms(class_logits21, mask_logits, labels, masks):
    probs = class_logits21.softmax(-1)
    return (
        -probs[:, labels.long()],
        _pairwise_bce(mask_logits, masks),
        _pairwise_dice(mask_logits, masks),
    )


def full_pixel_cost_terms(class_logits21, group_mass_ctx, labels, masks, valid):
    """Diagnostic matching cost over all valid annotated context pixels."""
    probs = class_logits21.softmax(-1)
    cost_class = -probs[:, labels.long()]
    queries = group_mass_ctx.shape[0]
    instances = masks.shape[0]
    bce = torch.zeros(queries, instances, device=group_mass_ctx.device)
    dice = torch.zeros_like(bce)
    valid_flat = valid.reshape(-1)
    probability = group_mass_ctx.reshape(queries, -1)[:, valid_flat].clamp(1e-5, 1 - 1e-5)
    for index in range(instances):
        truth = masks[index].reshape(-1)[valid_flat]
        bce[:, index] = F.binary_cross_entropy(
            probability, truth.unsqueeze(0).expand_as(probability)
        )
        intersection = (probability * truth.unsqueeze(0)).sum(-1)
        dice[:, index] = 1.0 - (2.0 * intersection + 1.0) / (
            probability.sum(-1) + truth.sum() + 1.0
        )
    return cost_class, bce, dice


def hard_metrics(mass, truth):
    prediction = mass > MASK_THRESHOLD
    intersection = int((prediction & truth).sum())
    union = int((prediction | truth).sum())
    return {
        "pred_area": int(prediction.sum()),
        "gt_area": int(truth.sum()),
        "precision": intersection / max(1, int(prediction.sum())),
        "recall": intersection / max(1, int(truth.sum())),
        "iou": intersection / max(1, union),
    }


def soft_dice(mass, truth, valid):
    p = mass[valid].clamp(1e-5, 1 - 1e-5)
    y = truth[valid].float()
    return float(1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0))


def context_state(forward, semantic_t, instance_t):
    mass = forward["masks"]["group_mass"]                       # [1,V,Q,H,W]
    class_logits21 = torch.cat(
        [forward["group"]["class_logits"][0], forward["group"]["objectness"][0].unsqueeze(-1)],
        dim=-1,
    )
    labels, masks, keys = thing_masks_and_keys(semantic_t, instance_t)
    # [Q, V_ctx, H, W] - the layout the audited matcher and loss expect
    mask_logits = torch.logit(mass[0, :2].permute(1, 0, 2, 3).clamp(1e-5, 1 - 1e-5))
    return mass, class_logits21, labels, masks, keys, mask_logits


def analyse_instance(model, opt, entry, key, device, want_grad=False):
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        mass, class_logits21, labels, masks, keys, mask_logits = context_state(
            forward, batch["semantic_label_all"].long(), batch["instance_label_all"].long()
        )
        semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
        table = group_score_table(forward)
        position = keys.index(key) if key in keys else None
        matched_group = None
        matched_costs = None
        diagnostic = None
        if position is not None:
            cost_class, cost_bce, cost_dice = sampled_cost_terms(
                class_logits21, mask_logits, labels, masks
            )
            total = 1.0 * cost_class + 5.0 * cost_bce + 5.0 * cost_dice
            rows, cols = hungarian_match(class_logits21, mask_logits, labels, masks)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if col == position:
                    matched_group = row
                    matched_costs = {
                        "class": float(cost_class[row, col]),
                        "bce": float(cost_bce[row, col]),
                        "dice": float(cost_dice[row, col]),
                        "total": float(total[row, col]),
                        "rank_of_matched_group": int(
                            (total[:, col] < total[row, col]).sum()
                        ) + 1,
                        "argmin_total_group": int(total[:, col].argmin()),
                        "sampled_points": int(_sample_points(masks).shape[-1]),
                    }
            valid_context = (batch["semantic_label_all"][0, :2] != 255)
            fc_class, fc_bce, fc_dice = full_pixel_cost_terms(
                class_logits21, mass[0, :2].permute(1, 0, 2, 3), labels, masks, valid_context
            )
            fc_total = 1.0 * fc_class + 5.0 * fc_bce + 5.0 * fc_dice
            diagnostic = {
                "valid_context_pixels": int(valid_context.sum()),
                "argmin_total_group": int(fc_total[:, position].argmin()),
                "matched_group_rank": (
                    int((fc_total[:, position] < fc_total[matched_group, position]).sum()) + 1
                    if matched_group is not None else None
                ),
                "matched_group_cost": (
                    {
                        "class": float(fc_class[matched_group, position]),
                        "bce": float(fc_bce[matched_group, position]),
                        "dice": float(fc_dice[matched_group, position]),
                        "total": float(fc_total[matched_group, position]),
                    } if matched_group is not None else None
                ),
            }
            del fc_class, fc_bce, fc_dice, fc_total
        views = []
        for view in range(int(opt.num_views)):
            sem = semantic_gt[view]
            ins = instance_gt[view]
            truth_np = (sem >= 2) & (sem < 20) & (ins > 0) & ((sem + 1) * 1000 + ins == key)
            truth = torch.from_numpy(truth_np).to(device)
            valid = torch.from_numpy(sem != 255).to(device)
            alpha_view = forward["masks"]["alpha"][0, view, 0]
            background = forward["masks"]["background_mass"][0, view, 0]
            record = {"view": view, "kind": "context" if view < 2 else "novel",
                      "gt_area": int(truth_np.sum())}
            if truth.any():
                record["alpha_coverage_in_gt"] = float(alpha_view[truth].mean())
                record["mean_group_mass_in_gt"] = float(
                    mass[0, view][:, truth].sum() / max(1, int(truth.sum()))
                )
                record["mean_background_mass_in_gt"] = float(background[truth].mean())
                record["conservation_error"] = float(
                    (mass[0, view].sum(0) + background - alpha_view).abs().max()
                )
                best_group, best_iou, best_metrics = None, 0.0, None
                for group_index in range(mass.shape[2]):
                    metrics = hard_metrics(mass[0, view, group_index], truth)
                    if metrics["pred_area"] >= MIN_PRED_PIXELS and metrics["iou"] > best_iou:
                        best_group, best_iou, best_metrics = group_index, metrics["iou"], metrics
                record["best_group"] = best_group
                record["best_iou"] = best_iou
                if best_group is not None:
                    best_metrics = dict(best_metrics)
                    best_metrics["soft_dice"] = soft_dice(
                        mass[0, view, best_group], truth, valid
                    )
                    record["best_metrics"] = best_metrics
                if matched_group is not None:
                    metrics = hard_metrics(mass[0, view, matched_group], truth)
                    metrics["soft_dice"] = soft_dice(
                        mass[0, view, matched_group], truth, valid
                    )
                    metrics["p_thing"] = float(table["p_thing"][matched_group])
                    metrics["p_foreground"] = float(table["p_foreground"][matched_group])
                    metrics["is_thing_class"] = bool(table["is_thing_class"][matched_group])
                    metrics["mean_group_mass_in_gt"] = float(
                        mass[0, view, matched_group][truth].mean()
                    )
                    metrics["mean_background_mass_in_gt"] = float(background[truth].mean())
                    record["matched_metrics"] = metrics
                predictions, gate_counts = group_predictions_v2(
                    {"group": forward["group"], "masks": forward["masks"]}, view
                )
                del gate_counts
                record["n_gt_free_predictions"] = len(predictions)
                record["gt_free_detected"] = any(
                    int((p["mask"] & truth_np).sum())
                    / max(1, int((p["mask"] | truth_np).sum())) >= 0.5
                    for p in predictions
                )
                if best_group is not None:
                    kept = [p for p in predictions if p["group"] == best_group]
                    record["best_group_kept_by_reader"] = bool(kept)
            views.append(record)
        result = {
            "key": key, "scene": entry["scene"], "context": list(entry["context"]),
            "novel": list(entry["novel"]), "matched_group": matched_group,
            "matched_costs": matched_costs,
            "full_pixel_counterfactual": diagnostic, "views": views,
        }
    if want_grad and matched_group is not None and position is not None:
        result["training_signal"] = training_signal(
            model, opt, batch, matched_group, position
        )
    return result


def training_signal(model, opt, batch, matched_group, position):
    """Read-only: sampled BCE/Dice for the matched pair and the mask-output gradient."""
    model.zero_grad(set_to_none=True)
    model_input, _ = split_data(batch, opt)
    from tokengs.models.input_types import ModelInputDecoder

    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                intrinsics=batch["intrinsics_all"])
    out = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder), render_decoder_input=decoder
    )
    mask_decoder = decoder.select_batch(slice(None), slice(0, 2))
    rendered = model.render_group_masks(out["gaussians"], model.layer10_group["slot_prob"],
                                        mask_decoder)
    masses = rendered["group_mass"].permute(0, 2, 1, 3, 4)[0]        # [Q,2,H,W]
    truth = thing_masks_and_keys(
        batch["semantic_label_all"].long(), batch["instance_label_all"].long()
    )[1][position].unsqueeze(0)
    prediction = torch.logit(masses[matched_group].clamp(1e-5, 1 - 1e-5)).unsqueeze(0)
    sampled_pred = _sample_points(prediction)
    sampled_truth = _sample_points(truth)
    bce = F.binary_cross_entropy_with_logits(sampled_pred, sampled_truth)
    dice = _pairwise_dice(prediction, truth)[0, 0]
    loss = 5.0 * bce + 5.0 * dice
    grad = torch.autograd.grad(loss, masses, allow_unused=True)[0]
    norm = float(grad[matched_group].norm()) if grad is not None else None
    model.zero_grad(set_to_none=True)
    return {
        "positive_sampled_points": int(sampled_truth.sum()),
        "sampled_bce": float(bce.detach()),
        "sampled_dice": float(dice.detach()),
        "mask_output_grad_norm": norm,
        "mask_output_grad_finite": norm is not None and math.isfinite(norm),
        "mask_output_grad_zero": (norm == 0.0) if norm is not None else None,
    }


def mean_valid_area(semantic_gt, instance_gt, key):
    areas = []
    for view in range(semantic_gt.shape[0]):
        sem, ins = semantic_gt[view], instance_gt[view]
        mask = (sem >= 2) & (sem < 20) & (ins > 0) & ((sem + 1) * 1000 + ins == key)
        areas.append(int(mask.sum()))
    nonzero = [a for a in areas if a > 0]
    return areas, (float(np.mean(nonzero)) if nonzero else 0.0)


def best_over_groups_iou(mass_view, truth):
    best = 0.0
    for group_index in range(mass_view.shape[0]):
        metrics = hard_metrics(mass_view[group_index], truth)
        if metrics["pred_area"] >= MIN_PRED_PIXELS and metrics["iou"] > best:
            best = metrics["iou"]
    return best


def select_instances(model, opt, entries, provenance, device, *, n_mid, n_large):
    candidates = []
    for entry in entries:
        batch = entry["batch"]
        with torch.no_grad():
            forward = forward_group(model, batch, opt)
        semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
        instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
        keys = set()
        for view in range(semantic_gt.shape[0]):
            keys |= set(gt_instances(semantic_gt, instance_gt, view).keys())
        for key in sorted(keys):
            areas, mean_area = mean_valid_area(semantic_gt, instance_gt, key)
            bucket = ("mid" if 1310 <= mean_area < 6553 else
                      "large" if mean_area >= 6553 else "small")
            novel_iou = 0.0
            for view in (2, 3):
                sem = semantic_gt[view]
                ins = instance_gt[view]
                truth = torch.from_numpy(
                    (sem >= 2) & (sem < 20) & (ins > 0) & ((sem + 1) * 1000 + ins == key)
                ).to(device)
                if not bool(truth.any()):
                    continue
                novel_iou = max(
                    novel_iou,
                    best_over_groups_iou(forward["masks"]["group_mass"][0, view], truth),
                )
            candidates.append({
                "key": int(key), "scene": entry["scene"],
                "context": list(entry["context"]), "novel": list(entry["novel"]),
                "per_view_area": areas, "mean_valid_area": mean_area, "bucket": bucket,
                "best_over_groups_iou": novel_iou,
                "provenance": provenance[(entry["scene"], tuple(entry["context"]),
                                          tuple(entry["novel"]))],
            })
    selected = []
    for bucket, count in (("mid", n_mid), ("large", n_large)):
        pool = [c for c in candidates if c["bucket"] == bucket and
                any(a > 0 for a in c["per_view_area"][2:])]
        pool.sort(key=lambda c: c["best_over_groups_iou"])
        selected.extend(pool[:count])
    reference = max(
        (c for c in candidates if any(a > 0 for a in c["per_view_area"][2:])),
        key=lambda c: c["best_over_groups_iou"],
    )
    if reference["key"] not in {s["key"] for s in selected}:
        reference = dict(reference)
        reference["role"] = "success_reference"
        selected.append(reference)
    for row in selected:
        row.setdefault("role", "failure")
    return selected, candidates


def draw_figure(path: Path, entry, models, view: int, key: int):
    batch = entry["batch"]
    rgb = batch["images_all"][0, view].float().cpu().numpy()
    semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
    instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
    sem, ins = semantic_gt[view], instance_gt[view]
    truth = (sem >= 2) & (sem < 20) & (ins > 0) & ((sem + 1) * 1000 + ins == key)
    panels = [rgb]
    truth_panel = np.zeros(truth.shape + (3,), dtype=np.float64)
    truth_panel[truth] = 1.0
    panels.append(truth_panel.transpose(2, 0, 1))
    text = []
    for name in ("g0_step6000", "g0plus_step6000"):
        info = models[name]
        with torch.no_grad():
            forward = forward_group(info["model"], batch, info["opt"])
        masses_t = forward["masks"]["group_mass"][0, view]
        masses = masses_t.float().cpu().numpy()
        best, best_iou = None, 0.0
        truth_t = torch.from_numpy(truth).to(masses_t.device)
        for group_index in range(masses_t.shape[0]):
            metrics = hard_metrics(masses_t[group_index], truth_t)
            if metrics["pred_area"] >= MIN_PRED_PIXELS and metrics["iou"] > best_iou:
                best, best_iou = group_index, metrics["iou"]
        # Hungarian group is recomputed on the identical context batch
        matched = None
        _, _, _, _, _, mask_logits = context_state(
            forward, batch["semantic_label_all"].long(), batch["instance_label_all"].long()
        )
        keys = thing_masks_and_keys(
            batch["semantic_label_all"].long(), batch["instance_label_all"].long()
        )[2]
        if key in keys:
            position = keys.index(key)
            rows, cols = hungarian_match(
                torch.cat([forward["group"]["class_logits"][0],
                           forward["group"]["objectness"][0].unsqueeze(-1)], dim=-1),
                mask_logits,
                thing_masks_and_keys(batch["semantic_label_all"].long(),
                                     batch["instance_label_all"].long())[0],
                thing_masks_and_keys(batch["semantic_label_all"].long(),
                                     batch["instance_label_all"].long())[1],
            )
            for row, col in zip(rows.tolist(), cols.tolist()):
                if col == position:
                    matched = row
                    break
        for tag, group_index in (("Hungarian", matched), ("best", best)):
            panel = np.zeros(truth.shape + (3,), dtype=np.float64)
            if group_index is not None:
                panel[masses[group_index] > MASK_THRESHOLD] = 1.0
                iou = hard_metrics(torch.from_numpy(masses[group_index]),
                                   torch.from_numpy(truth))["iou"]
                text.append(f"{name} {tag} g{group_index} IoU {iou:.3f} "
                            f"area {int((masses[group_index] > MASK_THRESHOLD).sum())}")
            else:
                text.append(f"{name} {tag} none")
            panels.append(panel.transpose(2, 0, 1))
        predictions, _ = group_predictions_v2(
            {"group": forward["group"], "masks": forward["masks"]}, view
        )
        panel = np.zeros(truth.shape + (3,), dtype=np.float64)
        for prediction in predictions:
            panel[prediction["mask"]] = 0.65
        panel[truth] = np.maximum(panel[truth], np.array([0.0, 0.4, 1.0]))
        panels.append(panel.transpose(2, 0, 1))
    tile = np.concatenate(panels, axis=2)
    image = Image.fromarray((np.clip(tile, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
    image = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    header = (f"{entry['scene']} ctx={entry['context']} novel={entry['novel']} view={view} "
              f"key={key} {' | '.join(text[:2])}")
    draw.text((4, 4), header, fill=(255, 255, 0))
    draw.text((4, 18), " | ".join(text[2:]), fill=(255, 255, 0))
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--out-dir", default="workspace_group_plus/mask_failure")
    parser.add_argument("--n-mid", type=int, default=4)
    parser.add_argument("--n-large", type=int, default=4)
    parser.add_argument("--train-windows", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-from", default="g0_step6000",
                        help="checkpoint used only to rank instances for the fixed sample")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    proven = proven_training_tuples()

    loaded = {}
    for name, directory in CHECKPOINTS:
        opt, model, arm, step, meta = load_model(directory, args.preset, args.seed, device)
        loaded[name] = {"opt": opt, "model": model, "arm": arm, "step": step,
                        "meta": meta, "dir": directory}
        del opt, model
    identity = {
        name: {
            "checkpoint": info["dir"], "arm": info["arm"], "step": info["step"],
            "plan_sha256": info["meta"].get("plan_sha256"),
            "model_only_sha256": hashlib.sha256(
                (Path(info["dir"]) / "model.pt").read_bytes()
            ).hexdigest(),
        }
        for name, info in loaded.items()
    }
    if len({v["plan_sha256"] for v in identity.values()}) != 1:
        raise SystemExit("checkpoints do not share the same plan")

    reference_opt = loaded[args.reference_from]["opt"]
    reference_model = loaded[args.reference_from]["model"]
    entries = build_train_entries(reference_opt, split, plan, device, args.train_windows)
    provenance = {}
    for entry in entries:
        tuple_key = (entry["scene"], tuple(entry["context"]), tuple(entry["novel"]))
        provenance[tuple_key] = (
            "proven_trained_window" if tuple_key in proven
            else "evaluation_window_of_training_scene"
        )
    selected, candidates = select_instances(
        reference_model, reference_opt, entries, provenance, device,
        n_mid=args.n_mid, n_large=args.n_large,
    )
    manifest = {
        "scope": "read-only mask-failure localisation; no training, no checkpoint writes",
        "rule": (
            "from the training windows of build_train_entries (plan stride 750, one per "
            "scene), rank thing instances by best-over-groups IoU at mask>0.5 / area>=50 "
            "(GT-assisted, no score gate) using the reference checkpoint; take the "
            f"{args.n_mid} lowest-IoU mid instances (1310<=mean per-view GT area<6553) and "
            f"the {args.n_large} lowest-IoU large instances (>=6553) that are visible in at "
            "least one novel view, plus the single highest-IoU instance as success "
            "reference.  Selected before any per-checkpoint analysis and not swapped."
        ),
        "reference_checkpoint": args.reference_from,
        "checkpoints": identity,
        "provenance_legend": {
            "proven_trained_window": "scene/context/novel tuple appears in the executed "
                                     "6000-step train log (so it was really trained on)",
            "evaluation_window_of_training_scene": "frames come from the plan for that "
                                                   "training scene but are not in the log",
        },
        "selected": selected,
        "candidate_pool_size": len(candidates),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[audit] manifest written with {len(selected)} instances "
          f"({sum(1 for s in selected if s['bucket']=='mid')} mid, "
          f"{sum(1 for s in selected if s['bucket']=='large')} large, "
          f"{sum(1 for s in selected if s['role']=='success_reference')} reference)",
          flush=True)
    for row in selected:
        print(f"[audit] {row['role']:17s} {row['bucket']:5s} {row['scene']} key {row['key']} "
              f"mean area {row['mean_valid_area']:.0f} best IoU {row['best_over_groups_iou']:.3f} "
              f"{row['provenance']}", flush=True)

    entry_by_tuple = {
        (e["scene"], tuple(e["context"]), tuple(e["novel"])): e for e in entries
    }
    per_instance = {}
    rows_for_csv = []
    for row in selected:
        tuple_key = (row["scene"], tuple(row["context"]), tuple(row["novel"]))
        entry = entry_by_tuple[tuple_key]
        per_instance[row["key"]] = {}
        for name in ("g0_step3000", "g0_step6000", "g0plus_step3000", "g0plus_step6000"):
            info = loaded[name]
            result = analyse_instance(
                info["model"], info["opt"], entry, row["key"], device,
                want_grad=True,
            )
            per_instance[row["key"]][name] = result
            for view_record in result["views"]:
                metrics = view_record.get("matched_metrics") or {}
                best = view_record.get("best_metrics") or {}
                rows_for_csv.append({
                    "instance_key": row["key"], "scene": row["scene"],
                    "bucket": row["bucket"], "role": row["role"], "checkpoint": name,
                    "view": view_record["view"], "kind": view_record["kind"],
                    "gt_area": view_record["gt_area"],
                    "matched_group": result["matched_group"],
                    "best_group": view_record.get("best_group"),
                    "matched_iou": metrics.get("iou"),
                    "best_iou": view_record.get("best_iou"),
                    "matched_precision": metrics.get("precision"),
                    "matched_recall": metrics.get("recall"),
                    "matched_pred_area": metrics.get("pred_area"),
                    "matched_soft_dice": metrics.get("soft_dice"),
                    "matched_p_thing": metrics.get("p_thing"),
                    "matched_is_thing_class": metrics.get("is_thing_class"),
                    "alpha_coverage_in_gt": view_record.get("alpha_coverage_in_gt"),
                    "mean_group_mass_in_gt": view_record.get("mean_group_mass_in_gt"),
                    "mean_background_mass_in_gt": view_record.get("mean_background_mass_in_gt"),
                    "conservation_error": view_record.get("conservation_error"),
                    "gt_free_detected": view_record.get("gt_free_detected"),
                    "best_group_kept": view_record.get("best_group_kept_by_reader"),
                    "sampled_bce": (result.get("training_signal") or {}).get("sampled_bce"),
                    "sampled_dice": (result.get("training_signal") or {}).get("sampled_dice"),
                    "positive_sampled_points": (
                        result.get("training_signal") or {}
                    ).get("positive_sampled_points"),
                    "mask_grad_norm": (
                        result.get("training_signal") or {}
                    ).get("mask_output_grad_norm"),
                    "full_pixel_argmin_group": (
                        result.get("full_pixel_counterfactual") or {}
                    ).get("argmin_total_group"),
                })
            for view_record in result["views"]:
                if view_record["kind"] != "novel" or view_record["gt_area"] == 0:
                    continue
                best_iou = view_record.get("best_iou")
                print(f"[audit] {name:16s} {row['scene']} key {row['key']} "
                      f"v{view_record['view']} gt {view_record['gt_area']} "
                      f"matched {result['matched_group']} "
                      f"best {view_record.get('best_group')} "
                      f"iou {best_iou if best_iou is None else round(best_iou, 3)} "
                      f"detected {view_record.get('gt_free_detected')} "
                      f"bg {view_record.get('mean_background_mass_in_gt')}", flush=True)
    # figures: two mid failures, two large failures, one success reference
    figure_rows = [r for r in selected if r["role"] == "failure"][:2]
    figure_rows += [r for r in selected if r["role"] == "failure" and r["bucket"] == "large"][:2]
    figure_rows += [r for r in selected if r["role"] == "success_reference"][:1]
    figure_paths = []
    for row in figure_rows:
        tuple_key = (row["scene"], tuple(row["context"]), tuple(row["novel"]))
        entry = entry_by_tuple[tuple_key]
        if row["key"] not in per_instance:
            continue
        view = 2 if any(a > 0 for a in row["per_view_area"][2:3]) else 3
        path = out_dir / f"{row['scene']}_key{row['key']}_{row['bucket']}.png"
        try:
            draw_figure(path, entry, loaded, view, row["key"])
            figure_paths.append(str(path))
        except Exception as error:  # noqa: BLE001
            print(f"[audit] figure failed for {row['key']}: {error}", flush=True)
    for info in loaded.values():
        info.pop("model", None)

    with (out_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows_for_csv for k in r}))
        writer.writeheader()
        writer.writerows(rows_for_csv)
    summary = summarise(per_instance, rows_for_csv, figures=figure_paths)
    (out_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "summary": summary,
                    "per_instance": per_instance}, indent=1, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1, default=str))
    print(f"[audit] wrote {out_dir}")
    return 0


def classify(record, view_record) -> str:
    best = view_record.get("best_iou", 0.0)
    if best < 0.5:
        return "no_good_mask"
    if view_record.get("gt_free_detected"):
        return "detected"
    if not view_record.get("best_group_kept_by_reader", False):
        return "good_mask_blocked_by_reader"
    return "good_mask_used_for_another_instance"


def summarise(per_instance, rows, *, figures) -> dict:
    counts = {}
    for key, checkpoints in per_instance.items():
        for name, record in checkpoints.items():
            for view_record in record["views"]:
                if view_record["kind"] != "novel" or view_record["gt_area"] == 0:
                    continue
                label = classify(record, view_record)
                counts.setdefault(name, {}).setdefault(label, 0)
                counts[name][label] += 1
    counterfactual = {"match_changed": 0, "match_same": 0,
                      "no_good_mask_even_with_full_pixels": 0}
    for checkpoints in per_instance.values():
        for record in checkpoints.values():
            diagnostic = record.get("full_pixel_counterfactual") or {}
            argmin = diagnostic.get("argmin_total_group")
            if argmin is None:
                continue
            if argmin == record.get("matched_group"):
                counterfactual["match_same"] += 1
            else:
                counterfactual["match_changed"] += 1
    progress = []
    for key, checkpoints in per_instance.items():
        early = checkpoints.get("g0_step6000")
        late = checkpoints.get("g0plus_step6000")
        if not early or not late:
            continue
        progress.append({
            "key": key,
            "matched_group_g0": early.get("matched_group"),
            "matched_group_g0plus": late.get("matched_group"),
            "group_identity_switch": early.get("matched_group") != late.get("matched_group"),
            "novel_best_iou_g0": early["views"][2].get("best_iou"),
            "novel_best_iou_g0plus": late["views"][2].get("best_iou"),
            "background_mass_in_gt_g0": early["views"][2].get("mean_background_mass_in_gt"),
            "background_mass_in_gt_g0plus": late["views"][2].get("mean_background_mass_in_gt"),
        })
    return {"failure_mode_counts_novel_views": counts,
            "counterfactual_matching": counterfactual,
            "g0_vs_g0plus_progress": progress,
            "figures": figures}


if __name__ == "__main__":
    raise SystemExit(main())
