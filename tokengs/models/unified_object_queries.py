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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

from tokengs.models.enc_dec import DecoderBlock


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
        self.log_temperature = nn.Parameter(
            torch.tensor(float(opt.assignment_temperature_init)).log()
        )
        trunc_normal_(self.class_head.weight, std=0.02)
        nn.init.zeros_(self.class_head.bias)

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

    def forward(self, tokens: torch.Tensor, batch_size: int = 1) -> UnifiedQueryOutput:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,C], got {tuple(tokens.shape)}")
        batch = tokens.shape[0]
        if batch_size != batch:
            raise ValueError(f"batch_size {batch_size} does not match tokens {batch}")

        keys, values = self._token_kv(tokens)
        queries = self.query_seed.unsqueeze(0).expand(batch, -1, -1).contiguous()
        for block in self.blocks:
            queries = block(gs_tokens=queries, keys=keys, values=values)
        queries = self.query_norm(queries)

        class_logits = self.class_head(queries)
        temperature = F.softplus(self.log_temperature).clamp(1.0, 100.0)
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
        stats = {
            "assignment_entropy": entropy,
            "query_usage_mean": prob.mean(dim=(0, 2)).mean(),
            "query_usage_max_share": prob.mean(dim=(0, 2)).max(),
            "active_query_count": (prob.mean(dim=(0, 2)) > 1.0 / (2.0 * self.num_queries)).sum().float(),
            "no_object_ratio": no_object,
            "token_max_assignment_mean": used.mean(),
            "assignment_temperature": temperature.detach(),
        }
        return UnifiedQueryOutput(
            query_features=queries,
            class_logits=class_logits,
            assignment_logits=assignment_logits,
            assignment_prob=assignment_prob,
            stats=stats,
        )


__all__ = ["UnifiedObjectQueryHead", "UnifiedQueryOutput"]
