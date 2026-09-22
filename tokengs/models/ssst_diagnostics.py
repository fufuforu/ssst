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

"""Diagnostics that test the SSST hypothesis without touching the training loss.

`token_purity_metrics` answers "does one spatial token explain one object?" by
measuring, per token, the rendered mass that falls on each context-view ground
truth instance.  `shared_gradient_diagnostic` separates the reconstruction and
understanding gradients on the shared token path and reports whether the
understanding objective actually reaches the spatial grounding parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SHARED_PARAMETER_PREFIXES = (
    "gs_tokens",
    "enc_dec_backbone.",
    "spatial_decoder.",
)
SPATIAL_GROUNDING_PARAMETERS = (
    "spatial_decoder.anchor_pre",
    "spatial_decoder.radius_pre",
    "spatial_decoder.refine_heads.0.weight",
    "spatial_decoder.anchor_pe_proj.weight",
    "spatial_decoder.ray_bias_raw",
)


def _pairwise_stats(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean/p95 of a square similarity matrix with the diagonal excluded."""
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected a square matrix, got {tuple(values.shape)}")
    count = values.shape[0]
    if count < 2:
        zero = values.new_zeros(()).float()
        return {"mean": zero, "p95": zero}
    mask = ~torch.eye(count, dtype=torch.bool, device=values.device)
    # bfloat16 autocast promotes matmul-like ops, but quantile() only accepts
    # float32/float64, so aggregate explicitly in float32.
    off_diagonal = values[mask].float()
    return {
        "mean": off_diagonal.mean(),
        "p95": off_diagonal.quantile(0.95),
    }


def query_pairwise_cosine(query_features: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pairwise cosine similarity between query features (diagonal excluded).

    Args:
        query_features: [B, M, C] final (LayerNorm-ed) query representations.

    Returns:
        ``{"mean": ..., "p95": ...}`` averaged over the batch.  Values close to
        1 mean the query bank has collapsed onto near-identical vectors.
    """
    if query_features.ndim != 3:
        raise ValueError(f"query_features must be [B,M,C], got {tuple(query_features.shape)}")
    normalized = F.normalize(query_features.detach().float(), dim=-1, eps=1e-6)
    similarity = torch.einsum("bmc,bnc->bmn", normalized, normalized)
    per_batch = [_pairwise_stats(similarity[index]) for index in range(similarity.shape[0])]
    return {
        "mean": torch.stack([item["mean"] for item in per_batch]).mean(),
        "p95": torch.stack([item["p95"] for item in per_batch]).mean(),
    }


def query_scene_update(
    query_features: torch.Tensor,
    query_seed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """How far the decoder moved the queries away from the learnable seed.

    ``query_features`` is the final LayerNorm-ed query output and
    ``query_seed`` is the same seed expanded to the batch, so the reported norm
    includes the effect of that final LayerNorm.  A value near 0 means the
    scene-conditioned blocks did not change the queries at all.
    """
    if query_features.ndim != 3 or query_seed.ndim != 3:
        raise ValueError("query features and seed must both be [B,M,C]")
    if query_features.shape != query_seed.shape:
        raise ValueError(
            f"query feature/seed shape mismatch: {tuple(query_features.shape)} vs "
            f"{tuple(query_seed.shape)}"
        )
    delta = (query_features.detach().float() - query_seed.detach().float()).norm(dim=-1).float()
    return {
        "mean": delta.mean(),
        "p95": delta.flatten().quantile(0.95),
    }


def query_mask_pairwise_similarity(
    mask_prob: torch.Tensor,
    *,
    sample_count: int = 4096,
) -> dict[str, dict[str, torch.Tensor]]:
    """Pairwise similarity between different query masks (diagonal excluded).

    ``mask_prob`` is [B, M, V, H, W].  For determinism and cost the [V, H, W]
    tail is flattened and a fixed evenly spaced pixel subset of
    ``sample_count`` positions is used for every query.

    Two variants are returned because raw soft Dice on probabilities is
    dominated by mask magnitude: when every query receives an equal share of the
    token mass the masks are all the same tiny field, yet
    ``dice = 2 sum(p_i p_j) / (sum(p_i) + sum(p_j))`` collapses towards
    ``mean(p)`` (~1/M) instead of revealing the identical shape.

    * ``cosine``: cosine similarity of the sampled mask vectors.  This is the
      scale-invariant "same field?" measure: exactly 1 for identical shapes and
      ~0 for disjoint masks, independent of magnitude.
    * ``dice``: the literally requested raw soft Dice on the probabilities.

    Values near 1 mean the query masks are the same field, which is what
    destroys instance structure at argmax time.
    """
    if mask_prob.ndim != 5:
        raise ValueError(f"mask_prob must be [B,M,V,H,W], got {tuple(mask_prob.shape)}")
    probability = mask_prob.detach().float()
    flat = probability.reshape(probability.shape[0], probability.shape[1], -1)
    positions = min(int(sample_count), flat.shape[-1])
    if positions <= 0:
        raise ValueError("mask_prob has no pixels to sample")
    if positions < flat.shape[-1]:
        index = torch.linspace(
            0, flat.shape[-1] - 1, positions, device=flat.device, dtype=torch.float32
        ).round().long()
        flat = flat.index_select(-1, index)

    def _summarize(matrix: torch.Tensor) -> dict[str, torch.Tensor]:
        per_batch = [_pairwise_stats(matrix[index]) for index in range(matrix.shape[0])]
        return {
            "mean": torch.stack([item["mean"] for item in per_batch]).mean(),
            "p95": torch.stack([item["p95"] for item in per_batch]).mean(),
        }

    numerator = torch.einsum("bmp,bnp->bmn", flat, flat)
    mass = flat.sum(-1)
    dice = 2.0 * numerator / (mass.unsqueeze(2) + mass.unsqueeze(1)).clamp_min(1e-8)
    normalized = F.normalize(flat, dim=-1, eps=1e-6)
    cosine = torch.einsum("bmp,bnp->bmn", normalized, normalized)
    return {"cosine": _summarize(cosine), "dice": _summarize(dice)}


def query_layer_scale_stats(blocks) -> dict[str, torch.Tensor]:
    """LayerScale magnitudes of the query decoder blocks.

    ``DecoderBlock`` applies ``gs_cross_attn_scale`` (scene-conditioned
    cross-attention), ``gs_self_attn_scale`` (query self-attention) and
    ``mlp_scale``.  Each is a per-channel LayerScale whose gamma starts at
    ``query_block_init_values``; a value that stayed at the initialization means
    the block never became active.
    """
    stats: dict[str, torch.Tensor] = {}
    for index, block in enumerate(blocks):
        for name, attribute in (
            ("cross", "gs_cross_attn_scale"),
            ("self", "gs_self_attn_scale"),
            ("mlp", "mlp_scale"),
        ):
            scale = getattr(block, attribute, None)
            gamma = getattr(scale, "gamma", None) if scale is not None else None
            if gamma is None:
                continue
            stats[f"query_layer{index}_{name}_scale_mean"] = gamma.detach().float().mean()
            stats[f"query_layer{index}_{name}_scale_max"] = gamma.detach().float().max()
    return stats


def shared_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    """Parameters shared by the reconstruction and understanding objectives."""
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(name == prefix or name.startswith(prefix) for prefix in SHARED_PARAMETER_PREFIXES):
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError("no shared parameters found; check the model layout")
    return selected


def _thing_index_map(
    semantic_labels: torch.Tensor,
    instance_labels: torch.Tensor,
    *,
    stuff_class_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map GT instance IDs to contiguous classes, with stuff/void as background.

    Returns ``(index_map, instance_ids)`` where ``index_map`` holds -1 for
    background (void, stuff class, or missing instance ID) and ``0..C-1`` for
    the thing instances listed in ``instance_ids``.
    """
    if semantic_labels.shape != instance_labels.shape:
        raise ValueError(
            f"semantic/instance label shape mismatch: {tuple(semantic_labels.shape)} vs "
            f"{tuple(instance_labels.shape)}"
        )
    is_thing = (semantic_labels != 255) & (semantic_labels >= stuff_class_count)
    thing_ids = torch.unique(instance_labels[(instance_labels > 0) & is_thing])
    thing_ids = thing_ids.sort().values
    index_map = torch.full_like(instance_labels, -1)
    for class_index, instance_id in enumerate(thing_ids.tolist()):
        index_map[(instance_labels == instance_id) & is_thing] = class_index
    return index_map, thing_ids


@torch.no_grad()
def token_instance_mass(
    model,
    gaussians: torch.Tensor,
    decoder_input,
    semantic_labels: torch.Tensor,
    instance_labels: torch.Tensor,
    *,
    stuff_class_count: int = 2,
    chunk_size: int = 64,
) -> dict:
    """Rendered mass of every token on every context-view GT instance.

    Every Gaussian of token ``i`` is given channel value 1 for token ``i`` and 0
    otherwise, so the alpha compositor returns that token's actual rendering
    contribution per pixel.  Mass is then accumulated over the GT instance
    regions of the provided views.
    """
    if decoder_input.cam_view is None or decoder_input.intrinsics is None:
        raise ValueError("token mass requires cameras and intrinsics")
    batch, num_gaussians, _ = gaussians.shape
    gaussians_per_token = int(model.gaussians_per_token)
    if num_gaussians % gaussians_per_token:
        raise ValueError(
            f"{num_gaussians} Gaussians are not a multiple of {gaussians_per_token}"
        )
    num_tokens = num_gaussians // gaussians_per_token
    index_map, thing_ids = _thing_index_map(
        semantic_labels, instance_labels, stuff_class_count=stuff_class_count
    )
    views = index_map.shape[1]
    pixels = index_map.shape[-2] * index_map.shape[-1]
    thing_class_count = int(thing_ids.numel())
    mass = gaussians.new_zeros(batch, num_tokens, thing_class_count)
    background = gaussians.new_zeros(batch, num_tokens)
    for start in range(0, num_tokens, chunk_size):
        stop = min(start + chunk_size, num_tokens)
        channels = stop - start
        features = gaussians.new_zeros(batch, num_gaussians, channels)
        for offset in range(channels):
            token = start + offset
            features[:, token * gaussians_per_token : (token + 1) * gaussians_per_token, offset] = 1.0
        rendered = model.gs.render_feature_channels(
            gaussians,
            features,
            decoder_input.cam_view,
            intrinsics=decoder_input.intrinsics,
        )["images_pred"]  # [B, V, C, H, W]
        flat = rendered.permute(0, 1, 3, 4, 2).reshape(batch, views * pixels, channels)
        flat = flat.transpose(1, 2)  # [B, C, P]
        labels = index_map.reshape(batch, views * pixels)
        valid = labels >= 0
        one_hot = F.one_hot(labels.clamp_min(0), num_classes=max(thing_class_count, 1)).to(flat.dtype)
        one_hot = one_hot * valid.unsqueeze(-1).to(flat.dtype)
        mass[:, start:stop] = torch.einsum("btp,bpc->btc", flat, one_hot)
        background[:, start:stop] = torch.einsum(
            "btp,bp->bt", flat, (~valid).to(flat.dtype)
        )
    return {
        "mass": mass,
        "background": background,
        "thing_ids": thing_ids,
        "views": views,
    }


@torch.no_grad()
def token_purity_metrics(
    model,
    gaussians: torch.Tensor,
    decoder_input,
    semantic_labels: torch.Tensor,
    instance_labels: torch.Tensor,
    *,
    stuff_class_count: int = 2,
    chunk_size: int = 64,
    min_mass: float = 1.0,
) -> dict:
    """Object coherence of the spatial tokens (validation / offline only).

    ``purity_i = max_c P_i(instance=c)`` over thing instances, with the
    background (void + stuff) mass reported separately.  Tokens whose total
    rendered mass is below ``min_mass`` pixels are excluded.
    """
    result = token_instance_mass(
        model,
        gaussians,
        decoder_input,
        semantic_labels,
        instance_labels,
        stuff_class_count=stuff_class_count,
        chunk_size=chunk_size,
    )
    mass = result["mass"]
    background = result["background"]
    total = mass.sum(dim=-1)
    with_background = total + background
    valid = with_background >= float(min_mass)
    device = mass.device
    if not bool(valid.any()):
        return {
            "token_gt_purity_mean": mass.new_tensor(float("nan")),
            "token_gt_purity_median": mass.new_tensor(float("nan")),
            "token_gt_purity_gt_08_ratio": mass.new_tensor(float("nan")),
            "token_gt_instance_entropy_mean": mass.new_tensor(float("nan")),
            "token_gt_background_mass_ratio": mass.new_tensor(float("nan")),
            "token_gt_valid_token_ratio": mass.new_tensor(0.0),
            "token_gt_thing_instance_count": torch.tensor(
                result["thing_ids"].numel(), device=device, dtype=mass.dtype
            ),
        }
    selected_mass = mass[valid]
    selected_total = total[valid].clamp_min(1e-8)
    probability = selected_mass / selected_total.unsqueeze(-1)
    purity = probability.max(dim=-1).values
    entropy = -(probability.clamp_min(1e-8).log() * probability).sum(dim=-1)
    background_ratio = (background[valid] / with_background[valid].clamp_min(1e-8)).mean()
    return {
        "token_gt_purity_mean": purity.mean(),
        "token_gt_purity_median": purity.median(),
        "token_gt_purity_gt_08_ratio": (purity > 0.8).float().mean(),
        "token_gt_instance_entropy_mean": entropy.mean(),
        "token_gt_background_mass_ratio": background_ratio,
        "token_gt_valid_token_ratio": valid.float().mean(),
        "token_gt_thing_instance_count": torch.tensor(
            result["thing_ids"].numel(), device=device, dtype=mass.dtype
        ),
    }


def _flatten_gradients(gradients) -> torch.Tensor:
    parts = []
    for gradient in gradients:
        if gradient is None:
            continue
        parts.append(gradient.detach().reshape(-1).float())
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def shared_gradient_diagnostic(
    model,
    batch: dict,
    *,
    step: int,
    phase: str = "train",
) -> dict:
    """Separate reconstruction/understanding gradients on the shared token path.

    Uses its own forward pass and `torch.autograd.grad`, so the optimizer's
    `.grad` buffers are never touched and no optimizer step is performed.

    Both the raw understanding gradient and the effective one are reported:
    the joint objective is ``L = L_recon + lambda_u(step) * L_understanding``,
    so only ``lambda_u * grad(L_understanding)`` competes with ``grad(L_recon)``
    on the shared parameters.
    """
    parameters = shared_parameters(model)
    inputs = [parameter for _, parameter in parameters]
    with torch.enable_grad():
        _, metrics = model.compute_joint_step(batch, step=step, phase=phase)
        loss_recon = metrics["loss_recon"]
        loss_understanding = metrics["loss_understanding"]
        if not loss_recon.requires_grad or not loss_understanding.requires_grad:
            raise RuntimeError("gradient diagnostic requires both losses to carry grad")
        grad_recon = list(
            torch.autograd.grad(loss_recon, inputs, retain_graph=True, allow_unused=True)
        )
        grad_understanding = list(
            torch.autograd.grad(loss_understanding, inputs, retain_graph=False, allow_unused=True)
        )
    vector_recon = _flatten_gradients(grad_recon)
    vector_understanding = _flatten_gradients(grad_understanding)
    norm_recon = float(vector_recon.norm())
    norm_understanding_raw = float(vector_understanding.norm())
    lambda_understanding = float(metrics["lambda_understanding"].detach())
    norm_understanding_effective = lambda_understanding * norm_understanding_raw
    if norm_recon > 0 and norm_understanding_raw > 0:
        cosine = float(
            torch.dot(vector_recon, vector_understanding).item()
            / (norm_recon * norm_understanding_raw)
        )
        cosine_defined = True
    else:
        cosine = float("nan")
        cosine_defined = False
    if norm_recon > 0:
        effective_ratio = norm_understanding_effective / norm_recon
    else:
        effective_ratio = float("nan")

    reachability = {}
    for name in SPATIAL_GROUNDING_PARAMETERS:
        index = next((i for i, (key, _) in enumerate(parameters) if key == name), None)
        if index is None:
            continue
        gradient = grad_understanding[index]
        reachability[name] = {
            "received_understanding_grad": gradient is not None,
            "understanding_grad_norm": float(gradient.norm()) if gradient is not None else 0.0,
        }
    shared_names = [name for name, _ in parameters]
    return {
        "grad_recon_norm": norm_recon,
        # Raw gradient of L_understanding and the gradient that actually enters
        # the joint objective through the curriculum weight.
        "grad_understanding_raw_norm": norm_understanding_raw,
        "grad_understanding_effective_norm": norm_understanding_effective,
        "grad_understanding_to_recon_ratio": effective_ratio,
        "lambda_understanding": lambda_understanding,
        # Backwards-compatible alias for the raw understanding gradient.
        "grad_understanding_norm": norm_understanding_raw,
        "grad_recon_understanding_cosine": cosine,
        "grad_cosine_defined": cosine_defined,
        "grad_shared_parameter_count": len(shared_names),
        "grad_shared_parameters_with_understanding_grad": sum(
            1 for gradient in grad_understanding if gradient is not None
        ),
        "grad_spatial_grounding": reachability,
        "grad_shared_names_sample": shared_names[:4],
    }


__all__ = [
    "SHARED_PARAMETER_PREFIXES",
    "SPATIAL_GROUNDING_PARAMETERS",
    "query_layer_scale_stats",
    "query_mask_pairwise_similarity",
    "query_pairwise_cosine",
    "query_scene_update",
    "shared_gradient_diagnostic",
    "shared_parameters",
    "token_instance_mass",
    "token_purity_metrics",
]
