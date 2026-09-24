# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class-agnostic instance query head for a frozen reconstruction backbone.

The head consumes the *scene tokens* of the frozen decoder (which depend only on
the context frames) and predicts, **for every token**, a distribution over
``num_queries`` instance slots plus one background slot.  With the verified
per-token pixel contribution maps ``M[t, p]`` this yields per-query masks
``mask[q, p] = sum_t A[t, q] * M[t, p]`` and a background mask, so that
``sum_q mask[q] + mask_bg = alpha`` by construction.

The token->slot normalisation is deliberate: summing over the 100 queries plus
background must reproduce the rendered alpha.  A 20-class semantic head is
reserved for the later phase and is *not* used by this module yet.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class InstanceQueryHead(nn.Module):
    def __init__(self, dim: int, num_queries: int = 100, num_semantic_classes: int = 20,
                 num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * (dim ** -0.5))
        self.token_norm = nn.LayerNorm(dim)
        self.token_proj = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.objectness = nn.Linear(dim, 1)
        # background slot: a learned scalar bias per token (shared), so the
        # token->slot distribution is over num_queries + 1 slots
        self.background_bias = nn.Parameter(torch.zeros(1))
        # reserved for phase 2 (ScanNet 20 classes, 1-based ids in the data)
        self.semantic = nn.Linear(dim, num_semantic_classes)
        self.logit_scale = float(dim) ** -0.5

    def forward(self, scene_tokens: torch.Tensor):
        """scene_tokens: [B, T, C] -> token->slot logits [B, T, Q+1], objectness [B, Q]."""
        b, t, _ = scene_tokens.shape
        tok = self.token_proj(self.token_norm(scene_tokens))
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        att, _ = self.cross_attn(q, tok, tok)
        qf = self.query_norm(q + att)
        qf = qf + self.mlp(qf)
        sim = torch.einsum("btc,bqc->btq", tok, qf) * self.logit_scale
        logits = torch.cat([sim, self.background_bias.expand(b, t, 1)], dim=-1)
        return logits, self.objectness(qf).squeeze(-1)


__all__ = ["InstanceQueryHead"]
