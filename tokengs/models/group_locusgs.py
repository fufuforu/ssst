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

"""From-scratch object-aware LocusGS with a group -> token feedback switch.

Two arms share one architecture and one training recipe and differ only in
whether the instance-group information is written back into the reconstruction
tokens:

* **G0 (read-out)** -- 100 learnable instance group queries plus one background
  slot read the layer-10 LocusGS tokens/anchors; each token gets a 101-way slot
  distribution, each group predicts objectness and a 20-class label.  Nothing is
  written back.
* **G1 (feedback)** -- exactly the same group head, the same read position, the
  same 101-way token -> slot direction and the same losses, plus a residual
  write-back between decoder layers 10 and 11:

  ``token'_i = token_i + tanh(g) * Proj(LayerNorm(sum_q A[i,q] * q_feat_q))``

  with ``g`` initialised to exactly 0, so the first forward is bit-identical to
  G0 and the write-back is bounded by the LayerNorm + projection.

The masks are composited with the RGB renderer from the very same Gaussians that
produce RGB/depth (``SIU3RJointSSST.render_query_masks`` uses the same
primitive), so ``sum_groups mask + background mask == rendered alpha``; the path
is differentiable and no ``no_grad`` contribution map is used for training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from tokengs.models.canonical_recon_models import LocusGSRecon, _full_supervision
from tokengs.models.input_types import ModelInput, ModelInputDecoder
from tokengs.models.instance_query_head import InstanceQueryHead
from tokengs.models.object_locusgs import (
    GSAttributeHead,
    SEMANTIC_CLASS_COUNT,
    render_semantic_probability,
    semantic_pixel_nll,
)
from tokengs.models.spatial_grounded_tokens import build_anchor_encoding
from tokengs.models.ssst_contracts import (
    NO_OBJECT_CE_WEIGHT,
    NO_OBJECT_CLASS,
    SEMANTIC_CLASS_COUNT as SIU3R_SEMANTIC_CLASS_COUNT,
)
from tokengs.models.ssst_loss import (
    _pairwise_bce,
    _pairwise_dice,
    _sample_points,
    build_context_segments,
    hungarian_match,
)

# The audited SIU3R composition (the outer 0.05 is applied once by this round's
# lambda(step) * 0.05, so it must not be applied again inside the loss).
_INSTANCE_WEIGHT_CLASS_CE = 2.0
_INSTANCE_WEIGHT_MASK_BCE = 5.0
_INSTANCE_WEIGHT_MASK_DICE = 5.0

_GROUP_HEAD_SEED_OFFSET = 201
_FEEDBACK_SEED_OFFSET = 202
_NEW_PARAM_STD = 0.01


class GroupQueryHead(InstanceQueryHead):
    """Instance group queries (100 + 1 background slot) over LocusGS tokens.

    Reuses ``InstanceQueryHead`` unchanged (queries, token projection,
    cross-attention, MLP, objectness and the reserved 20-class semantic head) and
    adds two things: an additive anchor/radius positional projection built from
    the existing ``build_anchor_encoding``, and ``forward_full`` which exposes the
    101-way slot logits, the objectness logits, the group class logits and the
    group features.  The original ``forward`` is untouched.
    """

    def __init__(self, opt):
        super().__init__(
            dim=int(opt.enc_embed_dim),
            num_queries=int(opt.num_object_queries),
            num_semantic_classes=int(
                opt.semantic_class_count if opt.semantic_class_count else SEMANTIC_CLASS_COUNT
            ),
            num_heads=int(getattr(opt, "group_num_heads", 4)),
            mlp_ratio=float(getattr(opt, "group_mlp_ratio", 2.0)),
        )
        self.opt = opt
        self.semantic_classes = int(self.semantic.out_features)
        self.anchor_num_freqs = int(opt.anchor_num_freqs)
        self.anchor_extent = float(opt.anchor_extent)
        spatial_dim = 4 + 6 * self.anchor_num_freqs
        self.spatial_proj = nn.Linear(spatial_dim, self.dim)
        trunc_normal_(self.spatial_proj.weight, std=float(opt.query_spatial_pe_std))
        nn.init.zeros_(self.spatial_proj.bias)
        # Recipe v1 (default off): a four-layer group decoder replaces the single
        # cross-attention + MLP path.  The old single-layer parameters stay
        # registered so old checkpoints still load with strict=True; with the
        # switch off this module does not exist and the forward is unchanged.
        self.recipe = bool(getattr(opt, "group_recipe", False))
        if self.recipe:
            self.deep = DeepGroupDecoder(opt)
        else:
            self.deep = None

    def forward_full(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor | None = None,
        radii: torch.Tensor | None = None,
        recipe: bool | None = None,
    ):
        """Return ``(slot_logits[B,T,Q+1], objectness[B,Q], class_logits[B,Q,20], q_feat[B,Q,C])``."""
        batch, num_tokens, _ = tokens.shape
        token_features = self.token_proj(self.token_norm(tokens))
        if anchors is not None and radii is not None:
            token_features = token_features + self.spatial_proj(
                build_anchor_encoding(
                    anchors,
                    radii,
                    num_freqs=self.anchor_num_freqs,
                    extent=self.anchor_extent,
                )
            )
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.cross_attn(queries, token_features, token_features)
        query_features = self.query_norm(queries + attended)
        query_features = query_features + self.mlp(query_features)
        use_recipe = self.recipe if recipe is None else bool(recipe)
        if use_recipe:
            if self.deep is None:
                raise RuntimeError("recipe forward requested but the deep decoder "
                                   "was not constructed (group_recipe=False)")
            query_features = self.deep(query_features, token_features)
        similarity = (
            torch.einsum("btc,bqc->btq", token_features, query_features) * self.logit_scale
        )
        # 100 instance groups + 1 background slot, normalised per token.
        background = (
            torch.zeros(batch, num_tokens, 1, device=similarity.device, dtype=similarity.dtype)
            if use_recipe
            else self.background_bias.expand(batch, num_tokens, 1)
        )
        slot_logits = torch.cat([similarity, background], dim=-1)
        objectness = self.objectness(query_features).squeeze(-1)
        class_logits = self.semantic(query_features)
        return slot_logits, objectness, class_logits, query_features


class DeepGroupDecoder(nn.Module):
    """Four-layer group decoder (recipe v1).

    Each layer, in this exact order: LayerNorm(q) -> query self-attention ->
    residual; LayerNorm(q) -> cross-attention to the token features (which already
    carry the anchor/radius spatial encoding) -> residual; LayerNorm(q) ->
    Linear(1024,2048) -> GELU -> Linear(2048,1024) -> residual.  A final
    LayerNorm is applied after the four layers.  Width 1024, 8 heads, dropout 0.
    """

    def __init__(self, opt, *, layers: int = 4, heads: int = 8, hidden: int = 2048):
        super().__init__()
        dim = int(opt.enc_embed_dim)
        self.dim = dim
        self.layers = nn.ModuleList()
        for _ in range(int(layers)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_norm": nn.LayerNorm(dim),
                        "self_attn": nn.MultiheadAttention(
                            dim, int(heads), dropout=0.0, batch_first=True
                        ),
                        "cross_norm": nn.LayerNorm(dim),
                        "cross_attn": nn.MultiheadAttention(
                            dim, int(heads), dropout=0.0, batch_first=True
                        ),
                        "mlp_norm": nn.LayerNorm(dim),
                        "mlp": nn.Sequential(
                            nn.Linear(dim, int(hidden)),
                            nn.GELU(),
                            nn.Linear(int(hidden), dim),
                        ),
                    }
                )
            )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, token_features: torch.Tensor) -> torch.Tensor:
        q = queries
        for layer in self.layers:
            attended, _ = layer["self_attn"](layer["self_norm"](q), layer["self_norm"](q),
                                             layer["self_norm"](q))
            q = q + attended
            attended, _ = layer["cross_attn"](layer["cross_norm"](q), token_features,
                                              token_features)
            q = q + attended
            q = q + layer["mlp"](layer["mlp_norm"](q))
        return self.final_norm(q)

    def reset_parameters(self, seed: int = 1743) -> None:
        """Deterministic re-initialisation of every parameter of this module."""
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.dim() > 1:
                    fan = parameter[0].numel() if parameter.dim() > 2 else parameter.shape[1]
                    bound = 1.0 / math.sqrt(max(1, fan))
                    values = torch.empty(parameter.shape, dtype=parameter.dtype,
                                         device="cpu")
                    values.uniform_(-bound, bound, generator=generator)
                    parameter.copy_(values.to(parameter.device))
                else:
                    parameter.zero_()
        # LayerNorm weights start at one (they are 1-D, so zeroed above)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    module.weight.fill_(1.0)


class GroupToTokenFeedback(nn.Module):
    """Bounded, gated write-back of the group-aggregated information."""

    def __init__(self, opt):
        super().__init__()
        dim = int(opt.enc_embed_dim)
        self.dim = dim
        self.num_queries = int(opt.num_object_queries)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))  # tanh(0) = 0 exactly
        generator = torch.Generator(device="cpu").manual_seed(
            int(opt.seed) + _FEEDBACK_SEED_OFFSET
        )
        with torch.no_grad():
            self.proj.weight.normal_(mean=0.0, std=_NEW_PARAM_STD, generator=generator)
            self.proj.bias.zero_()

    def forward(self, tokens: torch.Tensor, slot_prob: torch.Tensor, query_features: torch.Tensor):
        assignment = slot_prob[..., : self.num_queries]
        aggregated = torch.einsum("btq,bqc->btc", assignment, query_features)
        return tokens + torch.tanh(self.gate) * self.proj(self.norm(aggregated))


def thing_segment_keys(masks, semantic, instance):
    """Packed instance key per thing segment (context-merged), same convention as
    the verified routing/audit tooling: ``(semantic + 1) * 1000 + instance``."""
    keys = []
    for index in range(masks.shape[0]):
        key = None
        for view in range(masks.shape[1]):
            sub = masks[index, view].bool()
            if not bool(sub.any()):
                continue
            position = torch.nonzero(sub, as_tuple=False)[0]
            sem_value = int(semantic[0, view, position[0], position[1]])
            ins_value = int(instance[0, view, position[0], position[1]])
            key = (sem_value + 1) * 1000 + ins_value
            break
        keys.append(key)
    return keys


def group_instance_loss(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    gt_classes,
    gt_masks,
) -> dict:
    """Audited SIU3R set-prediction loss, without its outer 0.05 constant.

    ``class_logits`` is ``[B,100,20+1]`` (last = no-object), ``mask_logits`` is
    ``[B,100,V_ctx,H,W]``; the matching, the BCE/Dice point sampling and the
    no-object CE weight all come from ``tokengs/models/ssst_loss.py``.
    """
    batch = class_logits.shape[0]
    device = class_logits.device
    no_object = class_logits.new_full(
        (batch, class_logits.shape[1]), NO_OBJECT_CLASS, dtype=torch.long
    )
    empty_weight = class_logits.new_ones(SIU3R_SEMANTIC_CLASS_COUNT + 1)
    empty_weight[NO_OBJECT_CLASS] = NO_OBJECT_CE_WEIGHT
    ce_terms, bce_terms, dice_terms, matched, targets = [], [], [], [], []
    for index in range(batch):
        labels = gt_classes[index].to(device).long()
        masks = gt_masks[index].to(device).float()
        rows, cols = hungarian_match(class_logits[index], mask_logits[index], labels, masks)
        target = no_object[index].clone()
        if rows.numel():
            target[rows] = labels[cols]
            pred_points = _sample_points(mask_logits[index, rows])
            truth_points = _sample_points(masks[cols])
            bce_terms.append(F.binary_cross_entropy_with_logits(pred_points, truth_points))
            dice_terms.append(_pairwise_dice(pred_points, truth_points).diagonal().mean())
        ce_terms.append(F.cross_entropy(class_logits[index], target, weight=empty_weight))
        matched.append(int(rows.numel()))
        targets.append(int(labels.numel()))
    zero = class_logits.sum() * 0.0
    ce = torch.stack(ce_terms).mean() if ce_terms else zero
    bce = torch.stack(bce_terms).mean() if bce_terms else zero
    dice = torch.stack(dice_terms).mean() if dice_terms else zero
    total = (
        _INSTANCE_WEIGHT_CLASS_CE * ce
        + _INSTANCE_WEIGHT_MASK_BCE * bce
        + _INSTANCE_WEIGHT_MASK_DICE * dice
    )
    return {
        "loss": total,
        "class_ce": ce,
        "mask_bce": bce,
        "mask_dice": dice,
        "matched": matched,
        "targets": targets,
    }


class LocusGSGroupRecon(LocusGSRecon):
    """LocusGS reconstruction + instance groups (+ optional token feed-back)."""

    architecture_name = "LOCUSGS_GROUP_FEEDBACK_SCANNET_V1"

    def __init__(self, opt):
        super().__init__(opt)
        self.attributes = GSAttributeHead(opt, use_instance_head=False)
        self.groups = GroupQueryHead(opt)
        self.feedback = GroupToTokenFeedback(opt)
        self.arm = str(getattr(opt, "group_arm", "g0")).lower()
        if self.arm not in ("g0", "g1"):
            raise ValueError(f"unknown group_arm {self.arm!r}")
        self.num_groups = int(self.groups.num_queries)
        self.min_alpha = float(opt.object_min_alpha)
        self.sem_weight = float(opt.group_sem_loss_weight)
        self.inst_weight = float(opt.group_inst_loss_weight)
        self.ramp_steps = int(opt.group_loss_ramp_steps)
        # The single training change of the G0+ round: direct pixel supervision of
        # the background slot on the two context views.  Disabled (weight 0 or
        # flag off) the model is bit-identical to G0.
        self.bg_supervision = bool(getattr(opt, "group_bg_supervision", False))
        self.bg_weight = (
            float(getattr(opt, "group_bg_loss_weight", 1.0)) if self.bg_supervision else 0.0
        )
        # ---- recipe v1 (default off) ------------------------------------- #
        # Four modifications applied together: deep group decoder, segmentation
        # outer weight 0.1 with its own 1..1500 ramp, fixed zero void logit instead
        # of the learnable shared background_bias (and no background pixel
        # supervision), plus a token-assignment auxiliary CE.
        self.recipe = bool(getattr(opt, "group_recipe", False))
        self.seg_weight = float(getattr(opt, "group_recipe_seg_weight", 0.1))
        self.seg_ramp_steps = int(getattr(opt, "group_recipe_seg_ramp_steps", 1500))
        self.assign_coef = float(getattr(opt, "group_recipe_assign_coef", 0.02))
        self.assign_every = int(getattr(opt, "group_recipe_assign_every", 1))
        self.last_assignment_metrics: dict | None = None
        # smoke-only: when set, both ramps use this constant value instead of
        # the step schedule (harness measurement switch; default None)
        self.ramp_fixed: float | None = None
        if self.recipe:
            # the recipe forward does not use the G0+ background pixel supervision
            self.bg_supervision = False
            self.bg_weight = 0.0
        self.layer10_group: dict | None = None
        self.step_metrics: dict | None = None
        self.set_feedback_enabled(self.arm == "g1")

    # -- arm switch -------------------------------------------------------- #
    def set_feedback_enabled(self, enabled: bool) -> None:
        """G1 installs the write-back hook; G0 keeps the hook read-only."""
        self.use_feedback = bool(enabled) and self.arm == "g1"
        # Both arms run the *same* read-out at the same layer; only the residual
        # is switched, so the group read position never differs between arms.
        self.anchor_decoder.set_group_feedback_hook(
            self._group_hook, layer=int(self.opt.group_feedback_layer)
        )

    def _group_hook(self, tokens, mu, radii):
        slot_logits, objectness, class_logits, query_features = self.groups.forward_full(
            tokens, mu, radii, recipe=self.recipe
        )
        slot_prob = torch.softmax(slot_logits, dim=-1)
        self.layer10_group = {
            "slot_logits": slot_logits,
            "slot_prob": slot_prob,
            "objectness": objectness,
            "class_logits": class_logits,
            "query_features": query_features,
        }
        if not self.use_feedback:
            return tokens
        return self.feedback(tokens, slot_prob, query_features)

    # -- rendering --------------------------------------------------------- #
    def render_group_masks(self, gaussians, slot_prob, decoder_input):
        """Composite the 101 slot channels with the RGB renderer's own weights.

        Every Gaussian of a token inherits that token's slot distribution, so
        ``sum over the 101 slots`` reproduces the rendered alpha exactly.
        """
        batch, num_gaussians, _ = gaussians.shape
        num_tokens = slot_prob.shape[1]
        patches = int(self.opt.dec_patch_size) ** 2
        if num_gaussians != num_tokens * patches:
            raise ValueError(
                f"{num_gaussians} Gaussians do not match {num_tokens} tokens x {patches}"
            )
        gaussian_slots = (
            # slot_prob is [B, T, Q+1]: every Gaussian of a token inherits that
            # token's full 101-way distribution (token-major, like the Gaussian
            # head's token -> 64 GS order).
            slot_prob.unsqueeze(2)
            .expand(-1, -1, patches, -1)
            .reshape(batch, num_gaussians, -1)
        )
        rendered = self.gs.render_feature_channels(
            gaussians,
            gaussian_slots,
            decoder_input.cam_view,
            intrinsics=decoder_input.intrinsics,
        )
        mass = rendered["images_pred"]  # [B,V,101,H,W]
        group_mass = mass[:, :, : self.num_groups]
        background_mass = mass[:, :, self.num_groups : self.num_groups + 1]
        return {
            "group_mass": group_mass,
            "background_mass": background_mass,
            "alpha": rendered["alphas_pred"],
        }

    def decode_semantics(self, tokens):
        semantic_logits, _ = self.attributes(tokens)
        return semantic_logits

    # -- supervision ------------------------------------------------------- #
    def group_loss_terms(self, batch, gaussians, decoder_input, context_views):
        """Instance-group losses on the context views (mask + class + objectness)."""
        group = self.layer10_group
        if group is None:
            raise RuntimeError("the group hook did not run; check group_feedback_layer")
        mask_decoder = decoder_input.select_batch(slice(None), slice(0, len(context_views)))
        rendered = self.render_group_masks(gaussians, group["slot_prob"], mask_decoder)
        group_mass = rendered["group_mass"].permute(0, 2, 1, 3, 4)  # [B,Q,V,H,W]
        mask_prob = group_mass.clamp(1e-5, 1.0 - 1e-5)
        mask_logits = torch.logit(mask_prob)
        # 20 class logits + the objectness logit as the no-object competitor,
        # exactly the audited 21-way contract of ssst_loss.hungarian_match.
        class_logits = torch.cat(
            [group["class_logits"], group["objectness"].unsqueeze(-1)], dim=-1
        )
        gt_classes, gt_masks = build_context_segments(
            batch["semantic_label_all"],
            batch["instance_label_all"],
            tuple(context_views),
        )
        things = [
            (labels[labels >= 2], masks[labels >= 2])
            for labels, masks in zip(gt_classes, gt_masks)
        ]
        # Matching is recomputed here (identical inputs to group_instance_loss) so
        # the recipe's assignment CE can reuse the exact same pairing.
        if class_logits.shape[0] != 1:
            raise ValueError("group recipe assumes batch_size=1")
        rows, cols = hungarian_match(
            class_logits[0], mask_logits[0], things[0][0], things[0][1]
        )
        matched_rows = [-1] * int(things[0][0].numel())
        for row, col in zip(rows.tolist(), cols.tolist()):
            matched_rows[col] = row
        loss = group_instance_loss(
            class_logits, mask_logits, [t[0] for t in things], [t[1] for t in things]
        )
        loss["matched_rows"] = matched_rows
        loss["segment_keys"] = thing_segment_keys(
            things[0][1], batch["semantic_label_all"], batch["instance_label_all"]
        )
        alpha = rendered["alpha"]
        conservation = (group_mass.sum(dim=1) + rendered["background_mass"][:, :, 0]).sub(
            alpha[:, :, 0]
        ).abs().max()
        objectness_prob = torch.sigmoid(group["objectness"])
        loss.update(
            {
                "mask_alpha_max_error": conservation.detach(),
                "objectness_mean": objectness_prob.detach().mean(),
                "objectness_max": objectness_prob.detach().max(),
                "group_mass_mean": group_mass.detach().mean(),
                "background_mass_mean": rendered["background_mass"].detach().mean(),
                "alpha_mean": alpha.detach().mean(),
                "thing_targets": int(sum(t[0].numel() for t in things)),
            }
        )
        # G0+ only: the background slot is supervised directly on the same
        # rendered context-view masks (single rasterization pass).
        background = self._background_loss_of(batch, rendered, context_views)
        if background is not None:
            background_loss, background_stats = background
            loss["loss_bg"] = background_loss
            loss.update(background_stats)
            loss["loss_inst_total"] = (
                loss["loss"] + self.bg_weight * background_loss
            )
            loss["background_weight"] = self.bg_weight
        else:
            loss["loss_inst_total"] = loss["loss"]
        if group["slot_prob"].numel():
            prob = group["slot_prob"].detach()
            entropy = -(prob.clamp_min(1e-8).log() * prob).sum(-1).mean()
            loss["slot_entropy"] = entropy
            loss["slot_max_prob_mean"] = prob.max(dim=-1).values.mean()
            loss["background_prob_mean"] = prob[..., -1].mean()
            loss["active_group_share"] = (
                prob[..., : self.num_groups].mean(dim=(0, 1)) > 1.0 / self.num_groups
            ).float().mean()
        return loss

    def semantic_loss_terms(self, batch, tokens, gaussians, decoder_input, alpha=None):
        """Per-GS semantic pixel loss (unchanged from the previous rounds)."""
        semantic_logits = self.decode_semantics(tokens)
        probability, semantic_alpha = render_semantic_probability(
            self.gs, gaussians, semantic_logits, decoder_input.cam_view,
            decoder_input.intrinsics,
        )
        loss, stats = semantic_pixel_nll(
            probability,
            semantic_alpha,
            batch["semantic_label_all"].long(),
            min_alpha=self.min_alpha,
        )
        stats["pixel_alpha_gap"] = (
            (semantic_alpha - alpha).abs().max().detach()
            if alpha is not None
            else torch.zeros((), device=semantic_alpha.device)
        )
        return loss, stats, semantic_logits

    def background_slot_loss(self, batch, rendered, context_views):
        """Binary supervision of the rendered background-slot mass (G0+ only).

        Targets on the two context views: valid *stuff* pixels (semantic 0/1) want
        background 1, valid *thing* pixels (semantic 2..19 with a valid instance
        id) want background 0.  Semantic 255, unannotated pixels and pixels where
        the rendered alpha is <= 0.5 are excluded.  The alpha in the denominator
        is detached so the term cannot be reduced by shrinking coverage.
        """
        views = len(context_views)
        alpha = rendered["alpha"][:, :views]
        mass_bg = rendered["background_mass"][:, :views]
        p_bg = (mass_bg / alpha.detach().clamp_min(0.5)).clamp(1e-5, 1.0 - 1e-5)
        p_bg = p_bg[:, :, 0]
        semantic = batch["semantic_label_all"][:, :views].long()
        instance = batch["instance_label_all"][:, :views].long()
        covered = alpha[:, :, 0] > 0.5
        stuff = ((semantic == 0) | (semantic == 1)) & covered
        thing = (
            (semantic >= 2) & (semantic < SEMANTIC_CLASS_COUNT) & (instance > 0) & covered
        )
        zero = p_bg.sum() * 0.0
        loss_stuff = (
            F.binary_cross_entropy(p_bg[stuff], torch.ones_like(p_bg[stuff]))
            if bool(stuff.any()) else zero
        )
        loss_thing = (
            F.binary_cross_entropy(p_bg[thing], torch.zeros_like(p_bg[thing]))
            if bool(thing.any()) else zero
        )
        if bool(stuff.any()) and bool(thing.any()):
            loss = 0.5 * loss_stuff + 0.5 * loss_thing
        elif bool(stuff.any()):
            loss = loss_stuff
        elif bool(thing.any()):
            loss = loss_thing
        else:
            loss = zero
        return loss, {
            "loss_bg": loss,
            "loss_bg_stuff": loss_stuff.detach(),
            "loss_bg_thing": loss_thing.detach(),
            "bg_pixels_stuff": int(stuff.sum()),
            "bg_pixels_thing": int(thing.sum()),
            "bg_pixels_ignored": int((~covered).sum()),
            "bg_prob_stuff_mean": (
                p_bg[stuff].mean().detach() if bool(stuff.any()) else zero.detach()
            ),
            "bg_prob_thing_mean": (
                p_bg[thing].mean().detach() if bool(thing.any()) else zero.detach()
            ),
        }

    def _background_loss_of(self, batch, rendered, context_views):
        if self.bg_weight <= 0:
            return None
        return self.background_slot_loss(batch, rendered, context_views)

    # -- recipe v1: token-assignment auxiliary CE --------------------------- #
    def token_instance_contributions(self, gaussians, batch, alpha, context_views):
        """Per-token contribution mass to each context thing instance and to rest.

        Same algorithm as the verified routing-v2 phase-0 statistics: render T
        one-hot token channels through the model's own compositing weights on the
        two context views and accumulate, on pixels with alpha > 0.5 and a valid
        annotation, the mass falling on each packed thing instance (stuff pixels
        accumulate into ``rest``; 255 / unannotated / thing pixels without a valid
        instance id are ignored).
        """
        semantic = batch["semantic_label_all"].long()
        instance = batch["instance_label_all"].long()
        num_gaussians = gaussians.shape[1]
        num_tokens = num_gaussians // 64
        token_of = torch.arange(num_gaussians, device=gaussians.device) // 64
        onehot = torch.zeros(1, num_gaussians, num_tokens, device=gaussians.device)
        onehot[0, torch.arange(num_gaussians, device=gaussians.device), token_of] = 1.0
        per_instance: dict[int, torch.Tensor] = {}
        rest = torch.zeros(num_tokens, device=gaussians.device)
        for view in context_views:
            rendered = self.gs.render_feature_channels(
                gaussians, onehot, batch["cam_view_all"][:, view:view + 1],
                intrinsics=batch["intrinsics_all"][:, view:view + 1],
            )["images_pred"][0, 0]
            sem, ins = semantic[0, view], instance[0, view]
            covered = alpha[0, view, 0] > 0.5
            valid_annotation = covered & (sem != 255)
            thing = valid_annotation & (sem >= 2) & (sem < SEMANTIC_CLASS_COUNT) & (ins > 0)
            stuff = valid_annotation & ((sem == 0) | (sem == 1))
            if bool(stuff.any()):
                rest = rest + rendered[:, stuff].sum(-1)
            if bool(thing.any()):
                packed = (sem + 1) * 1000 + ins
                for key in torch.unique(packed[thing]).tolist():
                    mask = thing & (packed == int(key))
                    mass = rendered[:, mask].sum(-1)
                    per_instance[int(key)] = per_instance.get(int(key), 0.0) + mass
            del rendered
        total = rest.clone()
        for mass in per_instance.values():
            total = total + mass
        return {"instance": per_instance, "rest": rest, "total": total}

    def assignment_target(self, contributions, keys, matched_rows):
        """[T,101] soft target: matched instance columns (rows) + rest at 100."""
        num_tokens = contributions["rest"].shape[0]
        target = torch.zeros(num_tokens, self.num_groups + 1,
                             device=contributions["rest"].device)
        total = contributions["total"]
        kept = total > 1e-6
        for index, key in enumerate(keys):
            row = matched_rows[index] if index < len(matched_rows) else -1
            if row is None or row < 0 or int(key) not in contributions["instance"]:
                continue
            target[:, row] = contributions["instance"][int(key)]
        target[:, self.num_groups] = contributions["rest"]
        target = target / total.clamp_min(1e-12).unsqueeze(-1)
        target = torch.where(kept.unsqueeze(-1), target, torch.zeros_like(target))
        return target.detach(), kept, total

    def assignment_ce(self, target, kept):
        """Class-balanced CE of the current token->101 softmax against the target."""
        group = self.layer10_group
        log_prob = torch.log_softmax(group["slot_logits"][0].float(), dim=-1)
        ce = -(target * log_prob).sum(-1)
        thing_tokens = kept & (target[:, : self.num_groups].sum(-1) > 0)
        rest_tokens = kept & (~thing_tokens)
        terms = []
        if bool(thing_tokens.any()):
            terms.append(0.5 * ce[thing_tokens].mean())
        if bool(rest_tokens.any()):
            terms.append(0.5 * ce[rest_tokens].mean())
        zero = group["slot_logits"].sum() * 0.0
        loss = torch.stack(terms).sum() if terms else zero
        stats = {
            "assign_ce": loss.detach(),
            "assign_thing_tokens": int(thing_tokens.sum()),
            "assign_rest_tokens": int(rest_tokens.sum()),
            "assign_dropped_tokens": int((~kept).sum()),
            "assign_target_row_sum_error": (
                float((target.sum(-1)[kept] - 1).abs().max()) if bool(kept.any()) else 0.0
            ),
            "assign_argmax_agreement": (
                float((log_prob.argmax(-1)[kept] == target.argmax(-1)[kept]).float().mean())
                if bool(kept.any()) else 0.0
            ),
        }
        return loss, stats

    def recipe_probe_losses(self, batch, step: int):
        """Weighted main instance loss and weighted assignment CE for the probe.

        ``main = seg_ramp * seg_weight * L_instance`` and
        ``aux = seg_ramp * assign_coef * CE_balanced`` - exactly the two terms the
        gradient-ratio pre-registration compares (no reconstruction, no semantics).
        """
        from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data

        model_input, _ = split_data(batch, self.opt)
        decoder_input = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        states, _ = self._decode(ModelInput(model_input.encoder, decoder_input), decoder_input)
        supervision = _full_supervision(batch)
        _, _, final_gaussians, final_render, _ = self._layer_objective(
            states, decoder_input, supervision
        )
        context_views = tuple(range(int(self.opt.num_input_views)))
        instance = self.group_loss_terms(batch, final_gaussians, decoder_input, context_views)
        seg_ramp = (min(1.0, max(0.0, float(step) / float(self.seg_ramp_steps)))
                    if self.seg_ramp_steps > 0 else 1.0)
        with torch.no_grad():
            contributions = self.token_instance_contributions(
                final_gaussians, batch, final_render["alphas_pred"], context_views
            )
            target, kept, _ = self.assignment_target(
                contributions, instance["segment_keys"], instance["matched_rows"]
            )
        ce_loss, ce_stats = self.assignment_ce(target, kept)
        main = seg_ramp * self.seg_weight * instance["loss"]
        aux = seg_ramp * self.assign_coef * ce_loss
        stats = {
            "seg_ramp": seg_ramp,
            "instance_loss": float(instance["loss"].detach()),
            "assign_ce": float(ce_loss.detach()),
            "assign_thing_tokens": ce_stats["assign_thing_tokens"],
            "assign_rest_tokens": ce_stats["assign_rest_tokens"],
            "assign_dropped_tokens": ce_stats["assign_dropped_tokens"],
            "assign_target_row_sum_error": ce_stats["assign_target_row_sum_error"],
            "assign_argmax_agreement": ce_stats["assign_argmax_agreement"],
            "main_weight": seg_ramp * self.seg_weight,
            "aux_weight": seg_ramp * self.assign_coef,
        }
        return main, aux, stats

    def step_loss(self, batch: dict, *, step: int, phase: str) -> tuple[dict, dict]:
        """One joint step.  ``step`` is the **1-based** optimizer step, so
        ``lambda(1) = 1/2000 > 0``: the understanding loss is non-zero from the
        very first update (it is never delayed)."""
        del phase
        from tokengs.models.input_types import split_data

        model_input, _ = split_data(batch, self.opt)
        decoder_input = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        states, ray_stats = self._decode(
            ModelInput(model_input.encoder, decoder_input), decoder_input
        )
        supervision = _full_supervision(batch)
        recon_loss, metrics, final_gaussians, final_render, final_state = self._layer_objective(
            states, decoder_input, supervision
        )
        metrics["loss"] = recon_loss
        metrics["psnr"] = metrics[f"psnr_layer{self.supervised_layers[-1]}"]
        metrics["recon_loss"] = recon_loss.detach()

        context_views = tuple(range(int(self.opt.num_input_views)))
        instance = self.group_loss_terms(batch, final_gaussians, decoder_input, context_views)
        semantic, semantic_stats, _ = self.semantic_loss_terms(
            batch, final_state["tokens"], final_gaussians, decoder_input,
            final_render["alphas_pred"],
        )
        ramp = (self.ramp_fixed if self.ramp_fixed is not None else
                (min(1.0, max(0.0, float(step) / float(self.ramp_steps)))
                 if self.ramp_steps > 0 else 1.0))
        instance_weight = self.inst_weight * ramp
        semantic_weight = self.sem_weight * ramp
        # L_inst_new = L_inst_original + 1.0 * L_bg (the weight lives in the model
        # option; 0.0 disables the term entirely and reproduces G0 exactly).
        instance_total = instance["loss_inst_total"]
        assign_loss = None
        assign_stats = {}
        if self.recipe:
            # recipe v1: segmentation outer weight 0.1 with its own 1..1500 ramp;
            # the semantic branch keeps the original weight and ramp.
            seg_ramp = (self.ramp_fixed if self.ramp_fixed is not None else
                        (min(1.0, max(0.0, float(step) / float(self.seg_ramp_steps)))
                         if self.seg_ramp_steps > 0 else 1.0))
            instance_weight = self.seg_weight * seg_ramp
            semantic_weight = self.sem_weight * ramp
            if self.assign_every > 0 and (int(step) % self.assign_every == 0):
                with torch.no_grad():
                    contributions = self.token_instance_contributions(
                        final_gaussians, batch, final_render["alphas_pred"], context_views
                    )
                    target, kept, _ = self.assignment_target(
                        contributions, instance["segment_keys"], instance["matched_rows"]
                    )
                assign_loss, assign_stats = self.assignment_ce(target, kept)
        else:
            seg_ramp = ramp
        total = recon_loss + instance_weight * instance_total + semantic_weight * semantic
        if assign_loss is not None:
            total = total + seg_ramp * self.assign_coef * assign_loss
        metrics.update(
            {
                "loss_inst": instance["loss"].detach(),
                "loss_inst_total": instance_total.detach(),
                "loss_sem": semantic.detach(),
                "ramp": torch.tensor(ramp, device=recon_loss.device),
                "seg_ramp": torch.tensor(seg_ramp, device=recon_loss.device),
                "instance_weight": torch.tensor(instance_weight, device=recon_loss.device),
                "semantic_weight": torch.tensor(semantic_weight, device=recon_loss.device),
                "assign_coef": torch.tensor(self.assign_coef, device=recon_loss.device),
                "assign_applied": torch.tensor(
                    1.0 if assign_loss is not None else 0.0, device=recon_loss.device
                ),
                "loss_inst_ce": instance["class_ce"].detach(),
                "loss_inst_bce": instance["mask_bce"].detach(),
                "loss_inst_dice": instance["mask_dice"].detach(),
                "mask_alpha_max_error": instance["mask_alpha_max_error"],
                "objectness_mean": instance["objectness_mean"],
                "objectness_max": instance["objectness_max"],
                "group_mass_mean": instance["group_mass_mean"],
                "background_mass_mean": instance["background_mass_mean"],
                "thing_targets": torch.tensor(
                    float(instance["thing_targets"]), device=recon_loss.device
                ),
                "matched_groups": torch.tensor(
                    float(sum(instance["matched"])), device=recon_loss.device
                ),
                "sem_coverage": torch.as_tensor(
                    semantic_stats["coverage"], device=recon_loss.device
                ),
                "sem_supervised_pixels": torch.tensor(
                    float(semantic_stats["supervised_pixels"]), device=recon_loss.device
                ),
                "semantic_pixel_alpha_gap": semantic_stats["pixel_alpha_gap"],
                "alpha_mean": final_render["alphas_pred"].detach().mean(),
                "radius_min": final_state["radii"].detach().min(),
                "radius_max": final_state["radii"].detach().max(),
                "anchor_max": final_state["mu"].detach().abs().max(),
            }
        )
        for key in ("loss_bg", "loss_bg_stuff", "loss_bg_thing", "bg_pixels_stuff",
                    "bg_pixels_thing", "bg_pixels_ignored", "bg_prob_stuff_mean",
                    "bg_prob_thing_mean"):
            if key in instance:
                value = instance[key]
                metrics[key] = (
                    value if torch.is_tensor(value)
                    else torch.tensor(float(value), device=recon_loss.device)
                )
        for key in ("slot_entropy", "slot_max_prob_mean", "background_prob_mean",
                    "active_group_share"):
            if key in instance:
                metrics[key] = instance[key]
        for key, value in assign_stats.items():
            metrics[key] = (value if torch.is_tensor(value)
                            else torch.tensor(float(value), device=recon_loss.device))
        if self.recipe:
            metrics["fixed_void_max_abs"] = (
                self.layer10_group["slot_logits"][..., -1].abs().max().detach()
            )
        metrics["feedback_gate_abs_tanh"] = torch.tanh(self.feedback.gate.detach()).abs().mean()
        if self.anchor_decoder.last_group_update_norm is not None:
            metrics["group_update_norm"] = self.anchor_decoder.last_group_update_norm
        if ray_stats:
            metrics["gamma_mean"] = torch.stack([r["gamma"] for r in ray_stats]).mean()
        metrics["loss"] = total
        output = {
            "states": states,
            "gaussians": final_gaussians,
            "render": final_render,
            "group": self.layer10_group,
        }
        return output, metrics


__all__ = [
    "GroupQueryHead",
    "GroupToTokenFeedback",
    "LocusGSGroupRecon",
    "group_instance_loss",
]
