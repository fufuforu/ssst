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

"""Spatially grounded shared tokens.

Every shared token carries an explicit 3D anchor and a support radius in the
same normalized world frame that TokenGS uses for Gaussian positions (the first
context camera defines the origin, and the dataset scene scale is applied to the
camera translations by the provider).  The token feature, the anchor and the
radius are refined jointly inside the decoder: each layer injects a positional
encoding of the current anchor, attends to the multi-view image features, and
then predicts a bounded update of the anchor and the radius.

The corresponding Gaussian head keeps the TokenGS activation contract but
replaces the free global XYZ regression with a bounded local offset
``x_ik = mu_i + r_i * local_offset_ik``, so one token can only ever explain a
local 3D region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.activations import ClipActivationHead


def build_anchor_encoding(
    anchors: torch.Tensor,
    radii: torch.Tensor,
    *,
    num_freqs: int,
    extent: float,
) -> torch.Tensor:
    """Fourier encoding of anchor center and support radius.

    Args:
        anchors: [B, N, 3] anchor centers in the normalized world frame.
        radii: [B, N] support radii.

    Returns:
        [B, N, 4 + 6 * num_freqs] positional encoding.
    """
    if anchors.ndim != 3 or anchors.shape[-1] != 3:
        raise ValueError(f"anchors must be [B,N,3], got {tuple(anchors.shape)}")
    if radii.shape != anchors.shape[:2]:
        raise ValueError(f"radii must be [B,N], got {tuple(radii.shape)}")
    freq = 2.0 ** torch.arange(num_freqs, device=anchors.device, dtype=anchors.dtype)
    normed = (anchors / float(extent)).clamp(-1.0, 1.0)
    normed_radius = (radii / float(extent)).clamp(0.0, 1.0).unsqueeze(-1)
    angles = normed.unsqueeze(-2) * freq.view(1, 1, -1, 1) * math.pi
    return torch.cat(
        [
            normed,
            normed_radius,
            torch.sin(angles).flatten(-2),
            torch.cos(angles).flatten(-2),
        ],
        dim=-1,
    )


@dataclass
class SpatialTokenRefinement:
    """Spatially grounded token outputs."""

    tokens: torch.Tensor  # [B, N, C]
    anchors: torch.Tensor  # [B, N, 3]
    radii: torch.Tensor  # [B, N]
    stats: dict[str, torch.Tensor]


class SpatiallyGroundedTokenDecoder(nn.Module):
    """Iterative anchor refinement around the shared TokenGS decoder layers.

    The decoder blocks are *not* re-registered here (they live in the
    `EncDecBackbone`), so checkpoint keys stay exactly as in TokenGS.
    """

    def __init__(self, opt, decoder_blocks):
        super().__init__()
        self.opt = opt
        self.num_tokens = int(opt.num_gs_tokens)
        dim = int(opt.enc_embed_dim)
        self.dim = dim
        self.extent = float(opt.anchor_extent)
        self.center_z = float(opt.anchor_center_z)

        generator = torch.Generator(device="cpu").manual_seed(int(opt.seed) + 17)
        init_pre = torch.empty(self.num_tokens, 3).uniform_(-1.0, 1.0, generator=generator)
        self.anchor_pre = nn.Parameter(init_pre)
        init_radius = max(float(opt.anchor_init_radius), 1e-4)
        self.radius_pre = nn.Parameter(
            torch.full((self.num_tokens,), math.log(math.expm1(init_radius)))
        )

        pe_dim = 4 + 6 * int(opt.anchor_num_freqs)
        self.anchor_pe_proj = nn.Linear(pe_dim, dim)
        # Identity-at-initialization: a zero projection keeps the first forward
        # numerically identical to the plain TokenGS decoder.
        nn.init.zeros_(self.anchor_pe_proj.weight)
        nn.init.zeros_(self.anchor_pe_proj.bias)

        # Plain tuple: these modules are already registered (and saved) by the
        # backbone, so holding a ModuleList here would duplicate every key.
        self.decoder_blocks = tuple(decoder_blocks)
        self.refine_heads = nn.ModuleList(
            [nn.Linear(dim, 4) for _ in range(len(self.decoder_blocks))]
        )
        for head in self.refine_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _decode_pre(self, pre_anchors: torch.Tensor, pre_radius: torch.Tensor):
        """Map unconstrained parameters to bounded anchors and positive radii."""
        center = torch.tensor(
            [0.0, 0.0, self.center_z], dtype=pre_anchors.dtype, device=pre_anchors.device
        )
        anchors = center + self.extent * torch.tanh(pre_anchors)
        radii = F.softplus(pre_radius).clamp(
            float(self.opt.anchor_radius_min), float(self.opt.anchor_radius_max)
        )
        return anchors, radii

    def forward(self, tokens: torch.Tensor, encoder_latent) -> SpatialTokenRefinement:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,C], got {tuple(tokens.shape)}")
        batch = tokens.shape[0]
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"expected {self.num_tokens} shared tokens, got {tokens.shape[1]}"
            )
        if any(block is None for block in self.decoder_blocks):
            raise ValueError("decoder blocks are required for anchor refinement")

        pre_anchors = self.anchor_pre.unsqueeze(0).expand(batch, -1, -1).contiguous()
        pre_radius = self.radius_pre.unsqueeze(0).expand(batch, -1).contiguous()
        anchors, radii = self._decode_pre(pre_anchors, pre_radius)

        update_norms: list[torch.Tensor] = []
        radius_update_norms: list[torch.Tensor] = []
        for block, head in zip(self.decoder_blocks, self.refine_heads):
            pe = build_anchor_encoding(
                anchors, radii, num_freqs=int(self.opt.anchor_num_freqs), extent=self.extent
            )
            tokens = block(
                gs_tokens=tokens + self.anchor_pe_proj(pe),
                keys=encoder_latent.keys,
                values=encoder_latent.values,
            )
            delta = head(tokens)
            pre_anchors = pre_anchors + float(self.opt.anchor_refine_step) * torch.tanh(
                delta[..., :3]
            )
            pre_radius = pre_radius + float(self.opt.anchor_refine_step) * torch.tanh(
                delta[..., 3]
            )
            new_anchors, new_radii = self._decode_pre(pre_anchors, pre_radius)
            update_norms.append((new_anchors - anchors).norm(dim=-1).mean().detach())
            radius_update_norms.append((new_radii - radii).abs().mean().detach())
            anchors, radii = new_anchors, new_radii

        stats = {
            "anchor_update_norm_mean": torch.stack(update_norms).mean() if update_norms else tokens.new_zeros(()),
            "anchor_update_norm_max": torch.stack(update_norms).max() if update_norms else tokens.new_zeros(()),
            "anchor_update_norm_per_layer": (
                torch.stack(update_norms) if update_norms else tokens.new_zeros(0)
            ),
            "radius_update_abs_mean": torch.stack(radius_update_norms).mean() if radius_update_norms else tokens.new_zeros(()),
            "anchor_abs_mean": anchors.detach().abs().mean(),
            "anchor_std": anchors.detach().std(dim=(0, 1)).mean(),
            "anchor_min": anchors.detach().min(),
            "anchor_max": anchors.detach().max(),
            "radius_mean": radii.detach().mean(),
            "radius_std": radii.detach().std(),
            "radius_min": radii.detach().min(),
            "radius_max": radii.detach().max(),
        }
        return SpatialTokenRefinement(tokens=tokens, anchors=anchors, radii=radii, stats=stats)


class LocalGaussianHead(ClipActivationHead):
    """Token-to-Gaussian head with bounded offsets around the token anchor.

    Channel layout is kept identical to `ClipActivationHead`
    (`pos, rgb, scale, rot, opacity`) so the pre-trained reconstruction head
    remains shape-compatible; only the position channels change meaning: they
    now predict a bounded local offset instead of a free global XYZ.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.local_offset_bound = float(opt.anchor_local_offset_bound)

    def forward(self, x, anchors=None, radii=None):
        if anchors is None or radii is None:
            raise ValueError("LocalGaussianHead requires token anchors and radii")
        batch, num_tokens, _ = x.shape
        if anchors.shape != (batch, num_tokens, 3) or radii.shape != (batch, num_tokens):
            raise ValueError(
                f"anchor/radius shape mismatch with tokens {tuple(x.shape)}: "
                f"{tuple(anchors.shape)} / {tuple(radii.shape)}"
            )
        patches = self.num_gaussians_per_token
        raw = self.deconv(x).reshape(batch, num_tokens, patches, self.output_dims)

        offsets = torch.tanh(raw[..., 0:3]) * self.local_offset_bound
        positions = anchors.unsqueeze(2) + radii.unsqueeze(2).unsqueeze(-1) * offsets
        rgbs = self.rgb_act(raw[..., 3:6])
        scales = self.scale_act(raw[..., 6:9])
        rotations = self.rot_act(raw[..., 9:13])
        opacity = self.opacity_act(raw[..., 13:14])

        positions = positions + self.opt.gaussian_z_offset * torch.tensor(
            [0.0, 0.0, 1.0], dtype=positions.dtype, device=positions.device
        )
        gaussians = torch.cat(
            [
                positions.reshape(batch, num_tokens * patches, 3),
                opacity.reshape(batch, num_tokens * patches, 1),
                scales.reshape(batch, num_tokens * patches, 3),
                rotations.reshape(batch, num_tokens * patches, 4),
                rgbs.reshape(batch, num_tokens * patches, 3),
            ],
            dim=-1,
        )
        return gaussians


__all__ = [
    "LocalGaussianHead",
    "SpatialTokenRefinement",
    "SpatiallyGroundedTokenDecoder",
    "build_anchor_encoding",
]
