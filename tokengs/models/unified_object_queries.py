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

"""Unified object queries over the spatially grounded shared tokens.

The queries are object-level slots; the shared tokens stay the local 3D
representation.  Queries attend to the tokens and then produce, from one query
bank, both the class prediction and the token-to-query assignment.  The
assignment is the only bridge to the Gaussians: every Gaussian of token ``i``
inherits the assignment of that token, so no separate per-Gaussian instance
head exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.spatial_grounded_tokens import build_anchor_encoding
from tokengs.models.ssst_diagnostics import (
    query_layer_scale_stats,
    query_pairwise_cosine,
    query_scene_update,
)


def inverse_softplus(value: float) -> float:
    """Numerically stable inverse of ``softplus`` (double precision).

    ``softplus(inverse_softplus(t)) == t`` for ``t > 0``; the computation runs
    in float64 so large temperatures do not overflow before the log.
    """
    if not value > 0:
        raise ValueError(f"inverse_softplus expects a positive value, got {value}")
    return math.log(math.expm1(float(value)))


@dataclass
class UnifiedQueryOutput:
    """Unified object-query outputs."""

    query_features: torch.Tensor  # [B, M, C]
    class_logits: torch.Tensor  # [B, M, num_classes + 1] (last = no-object)
    assignment_logits: torch.Tensor  # [B, M, N] query-token logits
    assignment_prob: torch.Tensor  # [B, M, N] softmax over queries per token
    stats: dict[str, torch.Tensor]


class UnifiedObjectQueryHead(nn.Module):
    """Object-level queries that group spatially grounded tokens."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        dim = int(opt.enc_embed_dim)
        num_heads = int(opt.enc_num_heads)
        self.dim = dim
        self.num_queries = int(opt.num_object_queries)
        self.num_classes = int(opt.semantic_class_count)

        self.query_seed = nn.Parameter(
            float(opt.query_seed_std) * torch.randn(self.num_queries, dim)
        )
        self.token_norm = nn.LayerNorm(dim)
        self.token_kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.token_k_norm = nn.LayerNorm(dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim,
                    num_heads,
                    float(opt.mlp_ratio),
                    qkv_bias=True,
                    ffn_bias=True,
                    init_values=float(opt.query_block_init_values),
                    qk_norm=True,
                )
                for _ in range(int(opt.num_object_query_layers))
            ]
        )
        self.query_norm = nn.LayerNorm(dim)
        self.class_head = nn.Linear(dim, self.num_classes + 1)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.token_proj = nn.Linear(dim, dim, bias=False)
        # Explicit spatial state of every token: T_i = (f_i, mu_i, r_i).  The
        # query head is new (never warm-started), so it consumes the geometry
        # from the first forward instead of starting as a no-op.
        spatial_dim = 4 + 6 * int(opt.anchor_num_freqs)
        self.spatial_proj = nn.Linear(spatial_dim, dim)
        trunc_normal_(self.spatial_proj.weight, std=float(opt.query_spatial_pe_std))
        nn.init.zeros_(self.spatial_proj.bias)
        self.anchor_extent = float(opt.anchor_extent)
        self.anchor_num_freqs = int(opt.anchor_num_freqs)
        # The assignment logits are scaled by softplus(raw_temperature), so the
        # raw value must be the *inverse* softplus of the configured initial
        # temperature; storing log(init) would make the first forward use
        # softplus(log(init)) = log(1 + init) instead of init.
        self.raw_temperature = nn.Parameter(
            torch.tensor(inverse_softplus(float(opt.assignment_temperature_init)))
        )
        trunc_normal_(self.class_head.weight, std=0.02)
        nn.init.zeros_(self.class_head.bias)
        # Checkpoints written before the rename stored the same raw value under
        # `log_temperature`; the functional form (softplus) never changed, so
        # the value can be copied across unchanged.
        self.register_load_state_dict_pre_hook(self._remap_legacy_temperature)

    @staticmethod
    def _remap_legacy_temperature(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        legacy = prefix + "log_temperature"
        current = prefix + "raw_temperature"
        if legacy in state_dict and current not in state_dict:
            state_dict[current] = state_dict.pop(legacy)
        if legacy in unexpected_keys:
            unexpected_keys.remove(legacy)
        if current in missing_keys:
            missing_keys.remove(current)

    @property
    def no_object_index(self) -> int:
        return self.num_classes

    def _token_kv(self, tokens: torch.Tensor):
        keys, values = rearrange(
            self.token_kv_proj(self.token_norm(tokens)),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=int(self.opt.enc_num_heads),
        )
        return self.token_k_norm(keys), values

    def spatial_tokens(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        """Token feature augmented with its anchor/radius positional encoding."""
        if anchors.shape != tokens.shape[:2] + (3,) or radii.shape != tokens.shape[:2]:
            raise ValueError(
                f"spatial state must match tokens {tuple(tokens.shape)}: "
                f"anchors {tuple(anchors.shape)}, radii {tuple(radii.shape)}"
            )
        return tokens + self.spatial_proj(
            build_anchor_encoding(
                anchors,
                radii,
                num_freqs=self.anchor_num_freqs,
                extent=self.anchor_extent,
            )
        )

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        radii: torch.Tensor,
        batch_size: int | None = None,
    ) -> UnifiedQueryOutput:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,C], got {tuple(tokens.shape)}")
        batch = tokens.shape[0]
        if batch_size is not None and batch_size != batch:
            raise ValueError(f"batch_size {batch_size} does not match tokens {batch}")

        tokens = self.spatial_tokens(tokens, anchors, radii)
        keys, values = self._token_kv(tokens)
        query_seed = self.query_seed.unsqueeze(0).expand(batch, -1, -1).contiguous()
        queries = query_seed
        for block in self.blocks:
            queries = block(gs_tokens=queries, keys=keys, values=values)
        queries = self.query_norm(queries)

        class_logits = self.class_head(queries)
        temperature = F.softplus(self.raw_temperature).clamp(1.0, 100.0)
        query_embed = F.normalize(self.query_proj(queries).float(), dim=-1, eps=1e-6)
        token_embed = F.normalize(self.token_proj(self.token_norm(tokens)).float(), dim=-1, eps=1e-6)
        assignment_logits = temperature * torch.einsum("bmd,bnd->bmn", query_embed, token_embed)
        # Each spatial token is grouped by exactly one query (softmax over queries).
        assignment_prob = torch.softmax(assignment_logits, dim=1)

        with torch.no_grad():
            prob = assignment_prob.detach()
            entropy = -(prob.clamp_min(1e-8).log() * prob).sum(dim=1).mean()
            used = prob.max(dim=1).values
            no_object = (class_logits.detach().argmax(dim=-1) == self.no_object_index).float().mean()
            # Query-grouping diagnostics (detached, never part of the loss).
            scene_update = query_scene_update(queries, query_seed)
            pairwise_cosine = query_pairwise_cosine(queries)
        stats = {
            "assignment_entropy": entropy,
            "query_usage_mean": prob.mean(dim=(0, 2)).mean(),
            "query_usage_max_share": prob.mean(dim=(0, 2)).max(),
            "active_query_count": (prob.mean(dim=(0, 2)) > 1.0 / (2.0 * self.num_queries)).sum().float(),
            "no_object_ratio": no_object,
            "token_max_assignment_mean": used.mean(),
            "assignment_temperature": temperature.detach(),
            "query_scene_update_norm_mean": scene_update["mean"],
            "query_scene_update_norm_p95": scene_update["p95"],
            "query_pairwise_cosine_mean": pairwise_cosine["mean"],
            "query_pairwise_cosine_p95": pairwise_cosine["p95"],
        }
        stats.update(query_layer_scale_stats(self.blocks))
        return UnifiedQueryOutput(
            query_features=queries,
            class_logits=class_logits,
            assignment_logits=assignment_logits,
            assignment_prob=assignment_prob,
            stats=stats,
        )


__all__ = ["UnifiedObjectQueryHead", "UnifiedQueryOutput", "inverse_softplus"]
