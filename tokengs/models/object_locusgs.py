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

"""Object-aware LocusGS: per-Gaussian semantic/instance attributes plus a
predicted instance-relation token update (strictly paired A/B development run).

Both arms start from the *same* step-2000 LocusGS reconstruction checkpoint and
share the same pre-registered 6000-step batch plan; they differ in exactly one
structural variable:

* **A** -- GS semantic/instance attribute heads are trained jointly with the
  shared LocusGS reconstruction path (nothing frozen);
* **B** -- additionally, one predicted-instance-relation token update runs
  between decoder layer 10 and layer 11 (``ObjectRelationTokenUpdate``).

The 14-d Gaussian geometry/RGB format, the ``tanh(delta)`` bounded offsets, the
frozen decode radius, fp32, the 2-context input window, the layer-6/12
reconstruction objective and the 2+2 supervision window are inherited unchanged
from ``LocusGSRecon``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.canonical_recon_models import LocusGSRecon, _full_supervision
from tokengs.models.input_types import ModelInput, ModelInputDecoder

# ScanNet / SIU3R panoptic convention used by the processed-ScanNet provider:
# classes 0..19 with 255 = void; 0 (wall) and 1 (floor) are *stuff*, 2..19 are
# *things* (SIU3R `src/utils/scannet_constant.py`).
SEMANTIC_CLASS_COUNT = 20
STUFF_CLASS_COUNT = 2
THING_CLASS_MIN = STUFF_CLASS_COUNT
IGNORE_SEMANTIC = 255

# CHOICE: deterministic seed offsets for the two new heads and the relation
# module, so both arms instantiate bit-identical new parameters.
_ATTRIBUTE_HEAD_SEED_OFFSET = 101
_RELATION_SEED_OFFSET = 102
_NEW_PARAM_STD = 0.01


def instance_keys(semantic: torch.Tensor, instance: torch.Tensor) -> torch.Tensor:
    """Scene-stable cross-view key ``(semantic_id + 1) * 1000 + instance_id``.

    The key is only meaningful inside one scene (ScanNet instance ids are
    scene-global), so callers must never pool keys across scenes.
    """
    return (semantic.long() + 1) * 1000 + instance.long()


def thing_instance_mask(semantic: torch.Tensor, instance: torch.Tensor) -> torch.Tensor:
    """Valid *thing* instance pixels: semantic in 2..19 and instance id > 0."""
    sem = semantic.long()
    return (
        (sem != IGNORE_SEMANTIC)
        & (sem >= THING_CLASS_MIN)
        & (sem < SEMANTIC_CLASS_COUNT)
        & (instance.long() > 0)
    )


def semantic_supervision_mask(semantic: torch.Tensor) -> torch.Tensor:
    """Valid semantic pixels: class 0 is a real class, 255 is the only ignore."""
    sem = semantic.long()
    return (sem != IGNORE_SEMANTIC) & (sem >= 0) & (sem < SEMANTIC_CLASS_COUNT)


def ramp_weight(step: int, ramp_steps: int) -> float:
    """``ramp(t) = min(1, t / ramp_steps)`` for the experiment-local step."""
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(ramp_steps)))


class GSAttributeHead(nn.Module):
    """Per-Gaussian semantic logits and unit instance embeddings.

    Both are plain ``Linear(C, 64 * D)`` maps applied to the layer-12 scene
    tokens and reshaped with the *same* token -> 64 Gaussian order as
    ``LocusGSGaussianHead`` (token-major: Gaussian index = token * 64 + slot),
    so every Gaussian carries its own 20-d logits and 16-d embedding.
    """

    def __init__(self, opt, *, semantic_classes=None, instance_dim=None):
        super().__init__()
        dim = int(opt.enc_embed_dim)
        patches = int(opt.dec_patch_size) ** 2
        self.dim = dim
        self.patches = patches
        self.semantic_classes = int(
            opt.object_semantic_classes if semantic_classes is None else semantic_classes
        )
        self.instance_dim = int(opt.object_instance_dim if instance_dim is None else instance_dim)
        self.semantic = nn.Linear(dim, patches * self.semantic_classes)
        self.instance = nn.Linear(dim, patches * self.instance_dim)
        # Same seed and same distribution for both heads (CHOICE: std=0.01
        # normal, zero bias, matching this model's initialisation scale).
        generator = torch.Generator(device="cpu").manual_seed(
            int(opt.seed) + _ATTRIBUTE_HEAD_SEED_OFFSET
        )
        with torch.no_grad():
            for layer in (self.semantic, self.instance):
                layer.weight.normal_(mean=0.0, std=_NEW_PARAM_STD, generator=generator)
                layer.bias.zero_()

    def forward(self, tokens: torch.Tensor):
        batch, num_tokens, _ = tokens.shape
        semantic = self.semantic(tokens).reshape(
            batch, num_tokens, self.patches, self.semantic_classes
        ).reshape(batch, num_tokens * self.patches, self.semantic_classes)
        instance = self.instance(tokens).reshape(
            batch, num_tokens, self.patches, self.instance_dim
        ).reshape(batch, num_tokens * self.patches, self.instance_dim)
        return semantic, F.normalize(instance, dim=-1)


class ObjectRelationTokenUpdate(nn.Module):
    """Predicted-instance-relation token update (applied once, before layer 11).

    ``h_i  = LayerNorm(token_i)``

    ``z_i  = L2Normalize(Linear(C,32)(h_i))``

    ``a_ij = softmax_j(z_i . z_j / 0.2)`` over the 16 nearest anchors (by the
    layer-10 refined anchor coordinates, self included)

    ``m_i  = sum_j a_ij Linear(C,C)(h_j)``

    ``token'_i = token_i + tanh(g) * Linear(C,C)(m_i)``, with ``g`` initialised
    to 0.

    The neighbour indices are discrete and are detached (and computed under
    ``no_grad``), while every token/anchor that enters the update keeps its
    differentiable path.  ``g = 0`` makes the residual exactly zero, so arm B
    starts from exactly the same reconstruction forward as arm A.
    """

    def __init__(self, opt, *, hidden=32):
        super().__init__()
        dim = int(opt.enc_embed_dim)
        self.dim = dim
        self.neighbours = int(opt.object_relation_neighbours)
        self.temperature = float(opt.object_relation_temperature)
        if self.neighbours <= 0:
            raise ValueError("object_relation_neighbours must be positive")
        if self.temperature <= 0:
            raise ValueError("object_relation_temperature must be positive")
        self.norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, int(hidden), bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))
        generator = torch.Generator(device="cpu").manual_seed(int(opt.seed) + _RELATION_SEED_OFFSET)
        with torch.no_grad():
            for layer in (self.key, self.value, self.out):
                layer.weight.normal_(mean=0.0, std=_NEW_PARAM_STD, generator=generator)
                if layer.bias is not None:
                    layer.bias.zero_()

    def neighbour_indices(self, mu: torch.Tensor) -> torch.Tensor:
        """[B,T,K] nearest-anchor indices from the refined anchor coordinates.

        A stable argsort replaces ``topk`` so ties (and therefore the A/B
        pairing) are resolved deterministically on GPU.
        """
        with torch.no_grad():
            distances = torch.cdist(mu.detach().float(), mu.detach().float())
            order = torch.argsort(distances, dim=-1, stable=True)
            return order[..., : self.neighbours].detach()

    def _gather_tokens(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, channels = values.shape
        base = torch.arange(batch, device=values.device).view(batch, 1, 1) * num_tokens
        flat = (indices + base).reshape(-1)
        return values.reshape(batch * num_tokens, channels).index_select(0, flat).reshape(
            batch, num_tokens, indices.shape[-1], channels
        )

    def forward(self, tokens: torch.Tensor, mu: torch.Tensor, *, indices=None):
        if indices is None:
            indices = self.neighbour_indices(mu)
        h = self.norm(tokens)
        z = F.normalize(self.key(h), dim=-1)
        z_neighbour = self._gather_tokens(z, indices)
        attention = torch.einsum("bic,bikc->bik", z, z_neighbour) / self.temperature
        attention = torch.softmax(attention, dim=-1)
        values = self.value(self._gather_tokens(h, indices))
        message = torch.einsum("bik,bikc->bic", attention, values)
        return tokens + torch.tanh(self.gate) * self.out(message)


class LocusGSObjectRecon(LocusGSRecon):
    """LocusGS reconstruction + GS attributes (+ optional relation token update)."""

    architecture_name = "LOCUSGS_OBJECT_AWARE_SCANNET_V1"

    def __init__(self, opt):
        super().__init__(opt)
        self.attributes = GSAttributeHead(opt)
        self.relation = ObjectRelationTokenUpdate(opt)
        self.arm = str(getattr(opt, "object_arm", "a")).lower()
        if self.arm not in ("a", "b"):
            raise ValueError(f"unknown object_arm {self.arm!r}")
        # Arm A keeps the module instantiated (identical initial parameters) but
        # never applies it; arm B installs it as the decoder's single hook.
        self.use_relation_update = self.arm == "b"
        self.set_relation_enabled(self.use_relation_update)
        self.sem_loss_weight = float(opt.object_sem_loss_weight)
        self.inst_loss_weight = float(opt.object_inst_loss_weight)
        self.ramp_steps = int(opt.object_loss_ramp_steps)
        self.min_alpha = float(opt.object_min_alpha)
        self.instance_pixels = int(opt.object_instance_pixels)
        self.instance_budget = int(opt.object_instance_budget)
        self.last_instance_selection: dict | None = None

    # -- arm switch -------------------------------------------------------- #
    def _relation_hook(self, tokens, mu):
        if not self.use_relation_update:  # pragma: no cover - hook not installed
            return tokens
        return self.relation(tokens, mu)

    def set_relation_enabled(self, enabled: bool) -> None:
        """Arm A/B switch (arm A: module kept, parameter-identical, not applied)."""
        self.use_relation_update = bool(enabled) and self.arm == "b"
        if self.use_relation_update:
            self.anchor_decoder.set_token_update_hook(
                self._relation_hook, layer=int(self.opt.object_relation_layer)
            )
        else:
            self.anchor_decoder.set_token_update_hook(None, layer=0)

    # -- forward ----------------------------------------------------------- #
    def decode_attributes(self, tokens: torch.Tensor) -> dict:
        semantic_logits, instance_embedding = self.attributes(tokens)
        return {
            "semantic_logits": semantic_logits,
            "instance_embedding": instance_embedding,
        }

    def forward_reconstruction_only(self, model_input: ModelInput, *, render_decoder_input=None) -> dict:
        output = super().forward_reconstruction_only(
            model_input, render_decoder_input=render_decoder_input
        )
        output.update(self.decode_attributes(output["states"][-1]["tokens"]))
        return output

    # -- differentiable attribute compositing ------------------------------ #
    def render_attributes(self, gaussians, attributes, decoder_input):
        """Composite the attributes with the RGB geometry and blending weights.

        Both rasterizer calls consume the *same* Gaussian tensor, cameras,
        opacity, scale and rotation as the RGB render, so the compositing weights
        (and therefore alpha) describe identical geometry.  Nothing is detached.
        """
        probs = torch.softmax(attributes["semantic_logits"].float(), dim=-1)
        semantic_render = self.gs.render_feature_channels(
            gaussians, probs, decoder_input.cam_view, decoder_input.intrinsics
        )
        instance_render = self.gs.render_feature_channels(
            gaussians,
            attributes["instance_embedding"],
            decoder_input.cam_view,
            decoder_input.intrinsics,
        )
        alpha_sem = semantic_render["alphas_pred"]
        alpha_inst = instance_render["alphas_pred"]
        semantic_prob = semantic_render["images_pred"] / (alpha_sem + 1e-6)
        semantic_prob = semantic_prob / semantic_prob.sum(dim=2, keepdim=True).clamp_min(1e-6)
        instance_pred = instance_render["images_pred"] / (alpha_inst + 1e-6)
        instance_pred = F.normalize(instance_pred, dim=2)
        return {
            "semantic_prob": semantic_prob,
            "semantic_alpha": alpha_sem,
            "instance_embedding": instance_pred,
            "instance_alpha": alpha_inst,
            "alpha_gap": (alpha_sem - alpha_inst).abs().max().detach(),
        }

    # -- supervision -------------------------------------------------------- #
    def semantic_loss(self, semantic_prob, alpha, semantic_gt):
        """Per-view NLL over valid classes with ``alpha > min_alpha``, averaged."""
        views = semantic_prob.shape[1]
        target = semantic_gt.long().clamp(0, SEMANTIC_CLASS_COUNT - 1).unsqueeze(2)
        gathered = semantic_prob.gather(2, target).squeeze(2).clamp_min(1e-12)
        losses = []
        coverages = []
        supervised = 0
        for view in range(views):
            valid = semantic_supervision_mask(semantic_gt[:, view]) & (
                alpha[:, view, 0] > self.min_alpha
            )
            gt_valid = semantic_supervision_mask(semantic_gt[:, view])
            supervised += int(valid.sum())
            coverages.append(valid.float().sum() / gt_valid.float().sum().clamp_min(1.0))
            if valid.any():
                losses.append(-gathered[:, view].log()[valid].mean())
        coverage = torch.stack(coverages).mean().detach()
        stats = {"coverage": coverage, "supervised_pixels": supervised}
        if not losses:
            return semantic_prob.sum() * 0.0, stats
        return torch.stack(losses).mean(), stats

    def instance_loss(self, instance_pred, alpha, semantic_gt, instance_gt, *, step: int):
        """Pull/push contrastive loss over uniformly sampled thing-instance pixels."""
        device = instance_pred.device
        batch, views, dim, _, _ = instance_pred.shape
        if batch != 1:
            raise ValueError("instance loss is defined for batch_size=1 (one scene)")
        valid = thing_instance_mask(semantic_gt, instance_gt) & (alpha[:, :, 0] > self.min_alpha)
        keys_per_view = instance_keys(semantic_gt, instance_gt)
        seed_base = int(self.opt.seed)

        per_view_groups: list[dict[int, torch.Tensor]] = []
        for view in range(views):
            mask = valid[0, view]
            if not mask.any():
                per_view_groups.append({})
                continue
            flat_index = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
            flat_keys = keys_per_view[0, view].reshape(-1)[flat_index]
            order = torch.argsort(flat_keys, stable=True)
            flat_index, flat_keys = flat_index[order], flat_keys[order]
            unique, counts = torch.unique_consecutive(flat_keys, return_counts=True)
            generator = torch.Generator(device="cpu").manual_seed(
                (seed_base * 1_000_003 + step * 7919 + view * 104_729) % (2**31 - 1)
            )
            groups: dict[int, torch.Tensor] = {}
            offset = 0
            for key, count in zip(unique.tolist(), counts.tolist()):
                block = flat_index[offset : offset + count]
                offset += count
                take = min(int(count), self.instance_pixels)
                if take < count:
                    picked = torch.randperm(int(count), generator=generator)[:take]
                    block = block[picked.sort().values.to(block.device)]
                groups[int(key)] = block
            per_view_groups.append(groups)

        present = sorted({key for groups in per_view_groups for key in groups})
        if not present:
            return instance_pred.sum() * 0.0, {
                "instances_present": 0,
                "instances_used": 0,
                "pixels": 0,
                "push": 0.0,
            }
        if len(present) > self.instance_budget:
            generator = torch.Generator(device="cpu").manual_seed(
                (seed_base * 1_000_003 + step * 104_729 + 977) % (2**31 - 1)
            )
            picked = torch.randperm(len(present), generator=generator)[: self.instance_budget]
            selected = [present[i] for i in picked.sort().values.tolist()]
        else:
            selected = present

        centroids = []
        pulls = []
        pixel_count = 0
        for key in selected:
            vectors = []
            for view, groups in enumerate(per_view_groups):
                block = groups.get(key)
                if block is None:
                    continue
                vectors.append(
                    instance_pred[0, view].reshape(dim, -1)[:, block.to(device)].transpose(0, 1)
                )
            if not vectors:
                continue
            pixels = torch.cat(vectors, dim=0)
            pixel_count += int(pixels.shape[0])
            centroid = F.normalize(pixels.mean(dim=0), dim=-1)
            centroids.append(centroid)
            pulls.append((1.0 - (pixels * centroid).sum(-1)).mean())
        if not centroids:
            return instance_pred.sum() * 0.0, {
                "instances_present": len(present),
                "instances_used": 0,
                "pixels": 0,
                "push": 0.0,
            }
        pull = torch.stack(pulls).mean()
        push = torch.zeros((), device=device, dtype=pull.dtype)
        if len(centroids) >= 2:
            stacked = torch.stack(centroids)
            gram = stacked @ stacked.t()
            upper = torch.triu_indices(len(centroids), len(centroids), offset=1)
            push = (F.relu(gram[upper[0], upper[1]] - 0.2) ** 2).mean()
        self.last_instance_selection = {
            "instances_present": len(present),
            "instances_used": len(centroids),
            "pixels": pixel_count,
            "selected_keys": list(selected),
        }
        return pull + push, {
            "instances_present": len(present),
            "instances_used": len(centroids),
            "pixels": pixel_count,
            "push": float(push.detach()),
        }

    # -- training step ------------------------------------------------------ #
    def step_loss(self, batch: dict, *, step: int, phase: str) -> tuple[dict, dict]:
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

        attributes = self.decode_attributes(final_state["tokens"])
        rendered = self.render_attributes(final_gaussians, attributes, decoder_input)
        semantic_gt = batch["semantic_label_all"].long()
        instance_gt = batch["instance_label_all"].long()
        sem_loss, sem_stats = self.semantic_loss(
            rendered["semantic_prob"], rendered["semantic_alpha"], semantic_gt
        )
        inst_loss, inst_stats = self.instance_loss(
            rendered["instance_embedding"],
            rendered["instance_alpha"],
            semantic_gt,
            instance_gt,
            step=int(step),
        )
        ramp = ramp_weight(int(step), self.ramp_steps)
        metrics["loss_sem"] = sem_loss.detach()
        metrics["loss_inst"] = inst_loss.detach()
        metrics["ramp"] = torch.tensor(ramp, device=recon_loss.device)
        metrics["loss_understanding"] = (
            self.sem_loss_weight * sem_loss + self.inst_loss_weight * inst_loss
        ).detach()
        metrics["attribute_alpha_gap"] = rendered["alpha_gap"]
        metrics["sem_coverage"] = torch.as_tensor(
            sem_stats["coverage"], device=recon_loss.device
        )
        metrics["sem_supervised_pixels"] = torch.as_tensor(
            float(sem_stats["supervised_pixels"]), device=recon_loss.device
        )
        metrics["instances_present"] = torch.as_tensor(
            float(inst_stats["instances_present"]), device=recon_loss.device
        )
        metrics["instances_used"] = torch.as_tensor(
            float(inst_stats["instances_used"]), device=recon_loss.device
        )
        metrics["instance_pixels"] = torch.as_tensor(
            float(inst_stats["pixels"]), device=recon_loss.device
        )
        metrics["instance_push"] = torch.as_tensor(
            float(inst_stats["push"]), device=recon_loss.device
        )
        metrics["gate_abs_tanh"] = torch.tanh(self.relation.gate.detach()).abs().mean()
        if self.anchor_decoder.last_token_update_norm is not None:
            metrics["token_update_norm"] = self.anchor_decoder.last_token_update_norm

        radii_final = final_state["radii"].detach()
        metrics.update(
            {
                "radius_mean": radii_final.mean(),
                "radius_std": final_state["radii"].detach().std(),
                "radius_min": radii_final.min(),
                "radius_max": radii_final.max(),
                "anchor_max": final_state["mu"].detach().abs().max(),
                "alpha_mean": final_render["alphas_pred"].detach().mean(),
                "alpha_nonzero_fraction": (
                    final_render["alphas_pred"].detach() > 0
                ).float().mean(),
            }
        )
        if ray_stats:
            metrics["gamma_mean"] = torch.stack([r["gamma"] for r in ray_stats]).mean()
        total = recon_loss + ramp * (
            self.sem_loss_weight * sem_loss + self.inst_loss_weight * inst_loss
        )
        metrics["loss"] = total
        output = {
            "states": states,
            "gaussians": final_gaussians,
            "render": final_render,
            "attributes": rendered,
        }
        return output, metrics


__all__ = [
    "GSAttributeHead",
    "IGNORE_SEMANTIC",
    "LocusGSObjectRecon",
    "ObjectRelationTokenUpdate",
    "SEMANTIC_CLASS_COUNT",
    "STUFF_CLASS_COUNT",
    "THING_CLASS_MIN",
    "instance_keys",
    "ramp_weight",
    "semantic_supervision_mask",
    "thing_instance_mask",
]
