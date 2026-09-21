# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joint reconstruction + understanding + spatial loss for SSST.

The criterion is inherited from the audited SIU3R-aligned implementation
(Hungarian matching over the unified query bank, class CE, mask BCE and Dice,
plus the context depth-smoothness term) and extended with two conservative
spatial regularizers that keep the token-to-Gaussian locality meaningful.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from tokengs.models.losses import compute_tokengs_loss
from tokengs.models.ssst_contracts import (
    INSTANCE_DEPTH_SMOOTHNESS_WEIGHT,
    LOSS_WEIGHT_CLASS_CE,
    LOSS_WEIGHT_DICE,
    LOSS_WEIGHT_MASK_BCE,
    MATCH_COST_CLASS,
    MATCH_COST_DICE,
    MATCH_COST_MASK_BCE,
    NO_OBJECT_CE_WEIGHT,
    NO_OBJECT_CLASS,
    OUTER_SEGMENTATION_WEIGHT,
    POINT_SAMPLE_COUNT,
    QUERY_COUNT,
    SEMANTIC_CLASS_COUNT,
)

try:  # scipy is used for the exact assignment, as in the audited criterion
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def understanding_weight(step: int, opt) -> float:
    """Linear understanding-loss warmup: one joint model, ramped supervision."""
    start = float(opt.understanding_start_weight)
    final = float(opt.understanding_final_weight)
    warmup = int(opt.understanding_warmup_steps)
    if warmup <= 0:
        return final
    progress = min(1.0, max(0.0, float(step) / float(warmup)))
    return start + progress * (final - start)


def build_context_segments(
    semantic_labels: torch.Tensor,
    instance_labels: torch.Tensor,
    positions: tuple[int, ...],
    *,
    stuff_class_count: int = 2,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build one supervision segment per stuff class and per thing instance.

    Stuff regions are merged across the context views (one segment per class);
    thing segments use the scene-global instance ID, so the same physical object
    shares one segment across views.  This mirrors the audited SIU3R target
    construction used by the matching criterion.
    """
    if semantic_labels.ndim != 4 or instance_labels.shape != semantic_labels.shape:
        raise ValueError(
            "semantic/instance labels must be [B,V,H,W] with identical shapes, got "
            f"{tuple(semantic_labels.shape)} / {tuple(instance_labels.shape)}"
        )
    if len(positions) != 2:
        raise ValueError(f"exactly two context positions are required, got {positions}")
    all_classes: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    for batch_index in range(semantic_labels.shape[0]):
        sem = semantic_labels[batch_index][list(positions)].long()
        ins = instance_labels[batch_index][list(positions)].long()
        keys: dict[tuple, int] = {}
        for view in range(len(positions)):
            for cls in torch.unique(sem[view]).tolist():
                cls = int(cls)
                if cls == 255:
                    continue
                pixels = sem[view] == cls
                if cls < stuff_class_count:
                    keys.setdefault(("stuff", cls), cls)
                else:
                    for instance_id in torch.unique(ins[view][pixels]).tolist():
                        if int(instance_id) > 0:
                            keys.setdefault(("thing", int(instance_id)), cls)
        classes, masks = [], []
        for (kind, key), cls in sorted(keys.items(), key=lambda item: str(item[0])):
            mask = torch.zeros(
                len(positions), sem.shape[-2], sem.shape[-1], dtype=torch.float32, device=sem.device
            )
            for view in range(len(positions)):
                if kind == "stuff":
                    mask[view] = (sem[view] == key).float()
                else:
                    mask[view] = ((sem[view] == cls) & (ins[view] == key)).float()
            if mask.any():
                classes.append(int(cls))
                masks.append(mask)
        if not masks:
            all_classes.append(torch.empty(0, dtype=torch.long, device=sem.device))
            all_masks.append(
                torch.empty(
                    0, len(positions), sem.shape[-2], sem.shape[-1],
                    dtype=torch.float32, device=sem.device,
                )
            )
            continue
        all_classes.append(torch.tensor(classes, dtype=torch.long, device=sem.device))
        all_masks.append(torch.stack(masks))
    return all_classes, all_masks


def _sample_points(values: torch.Tensor, *, point_count: int = POINT_SAMPLE_COUNT) -> torch.Tensor:
    """Deterministic uniform sampling over the [view, H, W] tail."""
    if values.ndim < 2:
        raise ValueError("point-sampled values must have a view/spatial tail")
    flat = values.reshape(values.shape[0], -1)
    count = min(int(point_count), flat.shape[1])
    if count <= 0:
        raise ValueError("point-sampled values are empty")
    if count == flat.shape[1]:
        return flat
    indices = torch.linspace(
        0, flat.shape[1] - 1, count, device=values.device, dtype=torch.float32
    ).round().long()
    return flat.index_select(1, indices)


def _pairwise_dice(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    p = _sample_points(logits.sigmoid())
    y = _sample_points(masks.float())
    return 1.0 - (2.0 * p @ y.t() + 1.0) / (
        p.sum(-1, keepdim=True) + y.sum(-1).view(1, -1) + 1.0
    )


def _pairwise_bce(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    q = _sample_points(logits)[:, None, :]
    y = _sample_points(masks.float())[None, :, :]
    return F.binary_cross_entropy_with_logits(
        q.expand(-1, masks.shape[0], -1),
        y.expand(logits.shape[0], -1, -1),
        reduction="none",
    ).mean(-1)


def hungarian_match(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the query bank to the ground-truth instances of one scene."""
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for the fixed Hungarian contract")
    if class_logits.ndim != 2 or class_logits.shape[-1] != SEMANTIC_CLASS_COUNT + 1:
        raise ValueError(f"class_logits must be [{QUERY_COUNT},21]")
    if labels.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=class_logits.device)
        return empty, empty
    probs = class_logits.softmax(-1)
    cost_class = -probs[:, labels.long()]
    cost_bce = _pairwise_bce(mask_logits, masks)
    cost_dice = _pairwise_dice(mask_logits, masks)
    cost = (
        MATCH_COST_CLASS * cost_class
        + MATCH_COST_MASK_BCE * cost_bce
        + MATCH_COST_DICE * cost_dice
    )
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=class_logits.device),
        torch.as_tensor(cols, dtype=torch.long, device=class_logits.device),
    )


def class_aware_context_loss(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    gt_classes: Sequence[torch.Tensor],
    gt_masks: Sequence[torch.Tensor],
    *,
    depth_pred: torch.Tensor | None = None,
    gt_instance_maps: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Set-prediction loss on the context views.

    ``class_logits`` is [B, M, 21] (last class = no-object), ``mask_logits`` is
    [B, M, V_ctx, H, W]; ``gt_classes``/``gt_masks`` are per-scene lists with
    masks of shape [n_i, V_ctx, H, W].
    """
    batch = class_logits.shape[0]
    device = class_logits.device
    no_object = class_logits.new_full((batch, QUERY_COUNT), NO_OBJECT_CLASS, dtype=torch.long)
    empty_weight = class_logits.new_ones(SEMANTIC_CLASS_COUNT + 1)
    empty_weight[NO_OBJECT_CLASS] = NO_OBJECT_CE_WEIGHT
    ce_terms, bce_terms, dice_terms = [], [], []
    for index in range(batch):
        labels = gt_classes[index].to(device).long()
        masks = gt_masks[index].to(device).float()
        if masks.ndim == 4 and masks.shape[0] != labels.numel():
            raise ValueError(
                f"GT mask count {masks.shape[0]} does not match GT class count {labels.numel()}"
            )
        rows, cols = hungarian_match(class_logits[index], mask_logits[index], labels, masks)
        target = no_object[index].clone()
        if rows.numel():
            target[rows] = labels[cols]
            pred_points = _sample_points(mask_logits[index, rows])
            truth_points = _sample_points(masks[cols])
            bce_terms.append(F.binary_cross_entropy_with_logits(pred_points, truth_points))
            dice_terms.append(_pairwise_dice(pred_points, truth_points).diagonal().mean())
        ce_terms.append(F.cross_entropy(class_logits[index], target, weight=empty_weight))

    zero = class_logits.sum() * 0.0
    ce = torch.stack(ce_terms).mean() if ce_terms else zero
    bce = torch.stack(bce_terms).mean() if bce_terms else zero
    dice = torch.stack(dice_terms).mean() if dice_terms else zero
    depth_smooth = zero
    if depth_pred is not None and gt_instance_maps is not None:
        depth = depth_pred[:, :, 0] if depth_pred.ndim == 5 else depth_pred
        inst = gt_instance_maps.to(depth.device)
        dx = depth[..., 1:] - depth[..., :-1]
        dy = depth[..., 1:, :] - depth[..., :-1, :]
        keep_x = inst[..., 1:] == inst[..., :-1]
        keep_y = inst[..., 1:, :] == inst[..., :-1, :]
        depth_smooth = (dx.abs() * keep_x).mean() + (dy.abs() * keep_y).mean()
    total = OUTER_SEGMENTATION_WEIGHT * (
        LOSS_WEIGHT_CLASS_CE * ce + LOSS_WEIGHT_MASK_BCE * bce + LOSS_WEIGHT_DICE * dice
    ) + INSTANCE_DEPTH_SMOOTHNESS_WEIGHT * depth_smooth
    return {
        "loss": total,
        "class_ce": ce,
        "mask_bce": bce,
        "mask_dice": dice,
        "instance_depth_smoothness": depth_smooth,
    }


def spatial_regularization(
    gaussians: torch.Tensor,
    anchors: torch.Tensor,
    radii: torch.Tensor,
    *,
    gaussians_per_token: int,
    compactness_weight: float,
    radius_weight: float,
    radius_soft_min: float,
    radius_soft_max: float,
) -> dict[str, torch.Tensor]:
    """Keep the per-token Gaussian cloud local and the support radius bounded."""
    batch, num_gaussians, _ = gaussians.shape
    if num_gaussians % gaussians_per_token != 0:
        raise ValueError(
            f"{num_gaussians} Gaussians are not a multiple of "
            f"{gaussians_per_token} per token"
        )
    num_tokens = num_gaussians // gaussians_per_token
    if anchors.shape != (batch, num_tokens, 3) or radii.shape != (batch, num_tokens):
        raise ValueError(
            f"anchor/radius shape mismatch: {tuple(anchors.shape)}, {tuple(radii.shape)}"
        )
    positions = gaussians[..., :3].reshape(batch, num_tokens, gaussians_per_token, 3)
    offsets = positions - anchors.unsqueeze(2)
    offset_norm = offsets.norm(dim=-1)
    ratios = offset_norm / (radii.unsqueeze(-1) + 1e-8)
    # Diagnostic only: the largest Gaussian axis relative to the token radius.
    # A large value means the token's footprint is carried by scale rather than
    # by the anchor support, which would weaken the locality claim.
    scales = gaussians[..., 4:7].reshape(batch, num_tokens, gaussians_per_token, 3)
    scale_over_radius = scales.max(dim=-1).values / (radii.unsqueeze(-1) + 1e-8)

    zero = gaussians.sum() * 0.0
    loss_compactness = (
        ratios.pow(2).mean() * compactness_weight if compactness_weight > 0 else zero
    )
    if radius_weight > 0:
        too_large = F.relu(radii - radius_soft_max) / max(radius_soft_max, 1e-8)
        too_small = F.relu(radius_soft_min - radii) / max(radius_soft_min, 1e-8)
        loss_radius = (too_large.pow(2).mean() + too_small.pow(2).mean()) * radius_weight
    else:
        loss_radius = zero
    stats = {
        "local_offset_norm_mean": offset_norm.detach().mean(),
        "local_offset_norm_p95": offset_norm.detach().flatten().quantile(0.95),
        "local_offset_norm_max": offset_norm.detach().max(),
        "local_offset_ratio_mean": ratios.detach().mean(),
        "gs_scale_over_radius_mean": scale_over_radius.detach().mean(),
        "gs_scale_over_radius_p95": scale_over_radius.detach().flatten().quantile(0.95),
        "gs_scale_over_radius_max": scale_over_radius.detach().max(),
    }
    return {
        "loss": loss_compactness + loss_radius,
        "loss_compactness": loss_compactness,
        "loss_radius": loss_radius,
        **stats,
    }


def compute_joint_loss(
    *,
    opt,
    step: int,
    img_size,
    render_results: dict,
    supervision,
    decoder_input,
    gaussians: torch.Tensor,
    lpips_loss,
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    gt_classes: Sequence[torch.Tensor],
    gt_masks: Sequence[torch.Tensor],
    depth_pred: torch.Tensor | None = None,
    gt_instance_maps: torch.Tensor | None = None,
    anchors: torch.Tensor | None = None,
    radii: torch.Tensor | None = None,
    gaussians_per_token: int = 1,
) -> dict[str, torch.Tensor]:
    """Compose the one-stage objective.

    ``L = L_recon + lambda_u(step) * L_understanding + lambda_spatial * L_spatial``
    """
    # LPIPS is evaluated outside the per-scene vmap: the frozen VGG trunk
    # contains dropout, which torch.vmap rejects in randomness-error mode.
    recon = compute_tokengs_loss(
        opt=opt,
        img_size=img_size,
        render_results=render_results,
        supervision=supervision,
        decoder_input=decoder_input,
        gaussians=gaussians,
        lpips_loss=None,
    )
    loss_lpips = recon["loss"] * 0.0
    if float(getattr(opt, "lambda_lpips", 0.0)) > 0 and lpips_loss is not None:
        loss_lpips = torch.stack(
            [
                lpips_loss(
                    supervision.images_output[:, view],
                    render_results["images_pred"][:, view],
                    normalize=True,
                ).mean()
                for view in range(render_results["images_pred"].shape[1])
            ]
        ).mean()
    understanding = class_aware_context_loss(
        class_logits,
        mask_logits,
        gt_classes,
        gt_masks,
        depth_pred=depth_pred,
        gt_instance_maps=gt_instance_maps,
    )
    lambda_understanding = understanding_weight(step, opt)
    total = (
        recon["loss"]
        + float(getattr(opt, "lambda_lpips", 0.0)) * loss_lpips
        + lambda_understanding * understanding["loss"]
    )

    spatial = None
    if anchors is not None and radii is not None:
        spatial = spatial_regularization(
            gaussians,
            anchors,
            radii,
            gaussians_per_token=gaussians_per_token,
            compactness_weight=float(opt.spatial_compactness_weight),
            radius_weight=float(opt.spatial_radius_weight),
            radius_soft_min=float(opt.anchor_radius_soft_min),
            radius_soft_max=float(opt.anchor_radius_soft_max),
        )
        total = total + spatial["loss"]

    results = {
        "loss": total,
        "loss_recon": recon["loss"] + float(getattr(opt, "lambda_lpips", 0.0)) * loss_lpips,
        "loss_lpips": loss_lpips,
        "loss_understanding": understanding["loss"],
        "lambda_understanding": class_logits.new_tensor(lambda_understanding),
        "loss_class_ce": understanding["class_ce"],
        "loss_mask_bce": understanding["mask_bce"],
        "loss_mask_dice": understanding["mask_dice"],
        "loss_instance_depth_smoothness": understanding["instance_depth_smoothness"],
    }
    if spatial is not None:
        results.update(
            {
                "loss_spatial": spatial["loss"],
                "loss_spatial_compactness": spatial["loss_compactness"],
                "loss_spatial_radius": spatial["loss_radius"],
                "local_offset_norm_mean": spatial["local_offset_norm_mean"],
                "local_offset_norm_p95": spatial["local_offset_norm_p95"],
                "local_offset_norm_max": spatial["local_offset_norm_max"],
                "local_offset_ratio_mean": spatial["local_offset_ratio_mean"],
                "gs_scale_over_radius_mean": spatial["gs_scale_over_radius_mean"],
                "gs_scale_over_radius_p95": spatial["gs_scale_over_radius_p95"],
                "gs_scale_over_radius_max": spatial["gs_scale_over_radius_max"],
            }
        )
    for key in ("loss_rgb", "loss_ssim", "loss_visibility", "loss_opacity", "psnr"):
        if key in recon:
            results[key] = recon[key]
    return results


__all__ = [
    "build_context_segments",
    "class_aware_context_loss",
    "compute_joint_loss",
    "hungarian_match",
    "spatial_regularization",
    "understanding_weight",
]
