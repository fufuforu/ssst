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

"""LocusGS-faithful feed-forward reconstruction, adapted to ScanNet 2-view pairs.

Every formula below is taken from arXiv:2608.12825 (LocusGS: Spatially Grounded
Tokens for Feed-Forward 3D Gaussian Splatting); equation numbers in the comments
refer to that paper.  Items the paper does not specify (numerical epsilon, the
"predefined" initial support radius, PE frequency count, MLP depth, whether the
intermediate Gaussian head is shared) are marked "unspecified" and use the
documented ScanNet-adapted choice instead of guessing silently.

Nothing here touches the historical `siu3r_joint_ssst` model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tokengs.models.activations import ClipActivationHead
from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.tokengs import TokenGS


# --------------------------------------------------------------------------- #
# paper primitives
# --------------------------------------------------------------------------- #
def inverse_softplus(value: float) -> float:
    """``softplus(inverse_softplus(t)) == t`` (float64, numerically stable)."""
    if not value > 0:
        raise ValueError(f"inverse_softplus expects a positive value, got {value}")
    return math.log(math.expm1(float(value)))


def plucker_point_distance(mu: torch.Tensor, moment: torch.Tensor, direction: torch.Tensor):
    """Point-to-ray distance in Plücker coordinates (Eq. 22-24).

    ``d``/``m`` are normalized together first (Eq. 23) so the distance is
    ``|| mu x d_hat - m_hat ||_2`` with ``m = o x d`` (Eq. 20).

    Args:
        mu: [B, N, 3] anchor centers.
        moment: [B, P, 3] patch ray moments ``m = o x d``.
        direction: [B, P, 3] patch ray directions.

    Returns:
        [B, N, P] non-negative distances.
    """
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    d_hat = direction / norm
    m_hat = moment / norm
    cross = torch.cross(mu.unsqueeze(2).expand(-1, -1, d_hat.shape[1], -1), d_hat.unsqueeze(1).expand(-1, mu.shape[1], -1, -1), dim=-1)
    return (cross - m_hat.unsqueeze(1)).norm(dim=-1)


def anchor_ray_geometric_bias(
    mu: torch.Tensor,
    radii: torch.Tensor,
    moment: torch.Tensor,
    direction: torch.Tensor,
    *,
    sigma0: float,
    bandwidth_floor: float,
    clamp_min: float,
) -> torch.Tensor:
    """Radius-adaptive anchor-to-ray bias (Eq. 3 / Eq. 25).

    ``b_ij = -0.5 * (D(mu_i, l_j) / (sigma0 * r_i))^2`` with a lower bound on
    the squared bandwidth ("for stability, the squared bandwidth is
    lower-bounded", App. A.5) and clamping to ``[clamp_min, 0]``
    ("the geometric bias is clamped to the interval [-20, 0]").

    Returns ``[B, 1, N, P]`` so it broadcasts over attention heads.

    The lower bound is applied to the *squared bandwidth* exactly as App. A.5
    states ("for stability, the squared bandwidth is lower-bounded"); with
    ``bandwidth_floor`` interpreted as the floor on the bandwidth itself the
    result is numerically identical to the previous formulation
    ``(sigma0*r).clamp_min(floor)``.
    """
    distance = plucker_point_distance(mu, moment, direction)
    bandwidth_sq = (float(sigma0) * radii).square()
    bandwidth_sq = bandwidth_sq.clamp_min(float(bandwidth_floor) ** 2)
    bias = -0.5 * distance.square() / bandwidth_sq.unsqueeze(-1)
    return bias.clamp(min=float(clamp_min), max=0.0).unsqueeze(1)


def sinusoidal_positional_encoding(points: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Sinusoidal PE of 3D points (App. A.1: PE of the anchor center only)."""
    freq = 2.0 ** torch.arange(num_freqs, device=points.device, dtype=points.dtype)
    angles = points.unsqueeze(-2) * freq.view(1, 1, -1, 1) * math.pi
    return torch.cat([points, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)


class LocusGSAnchorDecoder(nn.Module):
    """Anchor-token decoder with dynamic anchor refinement (Sec. 3.2.3-3.2.5).

    The TokenGS decoder blocks are referenced as a plain tuple, so no parameter is
    registered twice and the backbone keys stay identical to TokenGS.  Each layer
    applies the TokenGS sub-block order (cross-attention -> self-attention -> FFN)
    with the anchor positional embedding injected between cross- and
    self-attention (Eq. 2) and the anchor-to-ray bias on the cross-attention
    logits (Eq. 4).
    """

    def __init__(self, opt, decoder_blocks, *, token_update_hook=None, token_update_layer=0):
        super().__init__()
        self.opt = opt
        self.num_tokens = int(opt.num_gs_tokens)
        dim = int(opt.enc_embed_dim)
        self.dim = dim
        # anchor centers are learnable and randomly initialized in the normalized
        # scene space (Sec. 3.2.3); the paper does not give the distribution, so we
        # sample uniformly inside the measured ScanNet normalized scene box.
        generator = torch.Generator(device="cpu").manual_seed(int(opt.seed) + 23)
        box = float(opt.locusgs_anchor_init_extent)
        center_z = float(opt.locusgs_anchor_init_center_z)
        init_mu = torch.empty(self.num_tokens, 3).uniform_(-box, box, generator=generator)
        init_mu[:, 2] = init_mu[:, 2] + center_z
        self.mu = nn.Parameter(init_mu)
        # raw radius: inverse softplus of the desired initial support radius
        # (App. A.3); the paper does not state the value, we use
        # `locusgs_radius_init`.
        self.rho = nn.Parameter(
            torch.full((self.num_tokens,), inverse_softplus(float(opt.locusgs_radius_init)))
        )
        self.epsilon = float(opt.locusgs_radius_epsilon)
        self.sigma0 = float(opt.locusgs_sigma0)
        self.bandwidth_floor = float(opt.locusgs_bandwidth_floor)
        self.bias_clamp_min = float(opt.locusgs_bias_clamp)

        pe_dim = 3 + 6 * int(opt.locusgs_pe_num_freqs)
        hidden = int(opt.locusgs_pe_hidden_dim) if int(opt.locusgs_pe_hidden_dim) > 0 else dim
        # Two implementations are available:
        #   * "legacy_v1"    -- one shared PE MLP + single-Linear refinement heads
        #                       (the first-guess implementation).
        #   * "inferred_v2"  -- per-layer PE MLP + per-layer bottleneck refinement
        #                       MLPs, chosen so that the LocusGS-specific parameter
        #                       count matches the paper's reported TokenGS->LocusGS
        #                       delta (222.0M -> 241.5M, i.e. +19.5M).
        # The paper writes PE(cdot) -> MLP (Eq. 2) and f_mu/f_rho (Eq. 6) but never
        # states whether these are shared across decoder layers or how wide they
        # are, so both readings are inferences from the parameter budget.
        self.impl = str(getattr(opt, "locusgs_impl", "legacy_v1"))
        if self.impl not in ("legacy_v1", "inferred_v2"):
            raise ValueError(f"unknown locusgs_impl {self.impl!r}")
        self.refine_hidden = int(getattr(opt, "locusgs_refine_hidden", 256))
        num_layers = len(decoder_blocks)
        if self.impl == "legacy_v1":
            self.pe_mlp = nn.Sequential(
                nn.Linear(pe_dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
            )
            self.pe_mlps = None
            # per-layer residual heads (Eq. 6): raw additive residuals, no tanh/step
            self.refine_mu = nn.ModuleList([nn.Linear(dim, 3) for _ in range(num_layers)])
            self.refine_rho = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_layers)])
            # The paper states the residual form (Eq. 6-7) but not the head
            # initialization.  Default-initialized heads move the anchors by O(1) in
            # a single step, so we zero-init them: the refinement then starts at
            # mu^{l+1} = mu^l (and rho^{l+1} = rho^l) and learns residuals, which is
            # the standard DETR-style refinement initialization.
            for head in list(self.refine_mu) + list(self.refine_rho):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            self.pe_mlp = None
            self.pe_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(pe_dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
                    )
                    for _ in range(num_layers)
                ]
            )
            self.refine_mu = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim, self.refine_hidden),
                        nn.GELU(),
                        nn.Linear(self.refine_hidden, 3),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.refine_rho = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim, self.refine_hidden),
                        nn.GELU(),
                        nn.Linear(self.refine_hidden, 1),
                    )
                    for _ in range(num_layers)
                ]
            )
            # UNSPECIFIED BY PAPER: only the final residual projection is
            # zero-initialized, so step 0 has delta_mu = delta_rho = 0 while the
            # hidden layers still carry a non-trivial default-initialized
            # representation.
            for head in list(self.refine_mu) + list(self.refine_rho):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        # learnable non-negative geometric bias scale (Eq. 4).  The paper gives no
        # initialization; `locusgs_gamma_raw_init` selects it (0 -> gamma 0.693,
        # -6 -> gamma ~2.5e-3).
        self.gamma_raw = nn.Parameter(
            torch.full((len(decoder_blocks),), float(opt.locusgs_gamma_raw_init))
        )
        # Eq. 2 is written as one residual step; the paper does not say whether the
        # anchor embedding stays in the residual stream ("persistent") or is only
        # the self-attention input ("injected").
        self.pe_mode = str(opt.locusgs_pe_mode)
        if self.pe_mode not in ("persistent", "injected"):
            raise ValueError(f"unknown locusgs_pe_mode {self.pe_mode!r}")
        # referenced, never re-registered: these live in `enc_dec_backbone`
        self.decoder_blocks = tuple(decoder_blocks)
        # Optional, additive hook: when set, it is applied exactly once, right
        # after the state of decoder layer `token_update_layer` has been emitted
        # and before the next layer runs.  Default `None` keeps the original
        # LocusGS anchor decoder bit-identical (the hook is the only switch the
        # object-aware variant adds, so every old checkpoint still loads).
        self.token_update_hook = token_update_hook
        self.token_update_layer = int(token_update_layer)
        self.last_token_update_norm = None

    def set_token_update_hook(self, hook, layer: int) -> None:
        """Install (or clear with ``hook=None``) the single pre-layer update."""
        self.token_update_hook = hook
        self.token_update_layer = int(layer)
        self.last_token_update_norm = None

    def activated_radius(self, rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + self.epsilon

    def forward(self, tokens: torch.Tensor, encoder_latent, patch_rays):
        """Run all decoder layers, returning per-layer states for supervision."""
        if patch_rays is None:
            raise ValueError("LocusGS anchor decoder requires patch-level rays")
        moment, direction = patch_rays
        batch = tokens.shape[0]
        expected = int(encoder_latent.keys.shape[-2])
        if moment.shape[1] != expected:
            raise ValueError(
                f"patch-level rays ({moment.shape[1]}) must match encoder keys ({expected})"
            )
        mu = self.mu.unsqueeze(0).expand(batch, -1, -1).contiguous()
        rho = self.rho.unsqueeze(0).expand(batch, -1).contiguous()
        states = []
        ray_bias_stats = []
        for index, (block, head_mu, head_rho) in enumerate(
            zip(self.decoder_blocks, self.refine_mu, self.refine_rho)
        ):
            radii = self.activated_radius(rho)
            bias = self.gamma_raw.new_zeros(())
            attn_bias = None
            if float(self.opt.locusgs_ray_bias_scale) > 0:
                geometric = anchor_ray_geometric_bias(
                    mu,
                    radii,
                    moment,
                    direction,
                    sigma0=self.sigma0,
                    bandwidth_floor=self.bandwidth_floor,
                    clamp_min=self.bias_clamp_min,
                )
                attn_bias = F.softplus(self.gamma_raw[index]) * geometric
                bias = attn_bias.detach()
                ray_bias_stats.append(
                    dict(
                        ray_bias_mean=bias.mean(),
                        ray_bias_clamped_fraction=(
                            geometric.detach() <= self.bias_clamp_min + 1e-6
                        ).float().mean(),
                        gamma=F.softplus(self.gamma_raw[index]).detach(),
                    )
                )
            pe_module = self.pe_mlp if self.pe_mlp is not None else self.pe_mlps[index]
            anchor_pe = pe_module(
                sinusoidal_positional_encoding(mu, int(self.opt.locusgs_pe_num_freqs))
            )
            # anchor-guided cross-attention (Eq. 4) -> anchor PE (Eq. 2) ->
            # anchor-aware self-attention -> FFN, mirroring DecoderBlock.
            tokens = tokens + block.gs_cross_attn_scale(
                block.gs_cross_attn(
                    tokens, encoder_latent.keys, encoder_latent.values, attn_bias=attn_bias
                )
            )
            if self.pe_mode == "persistent":
                # anchor embedding stays in the residual stream
                tokens = tokens + anchor_pe
                tokens = tokens + block.gs_self_attn_scale(block.gs_self_attn(tokens))
            else:
                # anchor embedding only conditions the self-attention input
                tokens = tokens + block.gs_self_attn_scale(
                    block.gs_self_attn(tokens + anchor_pe)
                )
            tokens = tokens + block.mlp_scale(block.mlp(tokens))
            # Eq. 6-7: raw additive residual refinement
            previous_mu = mu
            previous_radii = radii
            mu = mu + head_mu(tokens)
            rho = rho + head_rho(tokens).squeeze(-1)
            radii = self.activated_radius(rho)
            states.append(
                dict(
                    layer=index + 1,
                    tokens=tokens,
                    mu=mu,
                    rho=rho,
                    radii=radii,
                    anchor_update=(mu - previous_mu).norm(dim=-1).mean().detach(),
                    radius_update=(radii - previous_radii).abs().mean().detach(),
                )
            )
            if self.token_update_hook is not None and (index + 1) == self.token_update_layer:
                updated = self.token_update_hook(tokens, mu)
                if updated.shape != tokens.shape:
                    raise ValueError(
                        f"token update hook must preserve the token shape, got "
                        f"{tuple(updated.shape)} for {tuple(tokens.shape)}"
                    )
                self.last_token_update_norm = (
                    (updated - tokens).detach().norm(dim=-1).mean()
                )
                tokens = updated
        return states, ray_bias_stats


class LocusGSGaussianHead(ClipActivationHead):
    """Anchor-centered Gaussian decoding (Sec. 3.2.6, Eq. 8-9).

    Position channels predict the local offset ``delta`` directly (the paper does
    not bound them) and the center is ``mu + r * delta``; scale/rotation/color/
    opacity reuse the TokenGS activation head ("the standard token-based
    prediction head as in TokenGS").  Subclassing `ClipActivationHead` keeps the
    parameter names/shapes identical to TokenGS, so the variant has no unused
    parameters.
    """

    def __init__(self, opt):
        super().__init__(opt)

    def forward(self, tokens: torch.Tensor, mu: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, _ = tokens.shape
        patches = self.num_gaussians_per_token
        raw = self.deconv(tokens).reshape(batch, num_tokens, patches, self.output_dims)
        offsets = raw[..., 0:3]  # Eq. 8: f_delta
        if bool(getattr(self.opt, "locusgs_bound_delta", False)):
            # Single-variable locality experiment: bounding delta puts every
            # Gaussian within the token's support radius (||r*delta|| <= r).
            offsets = torch.tanh(offsets)
        if bool(getattr(self.opt, "locusgs_freeze_decode_radius", False)):
            # Decoding radius held at the initial support radius; the learned
            # radii still feed the anchor-to-ray bias only.
            radii = torch.full_like(radii, float(self.opt.locusgs_radius_init))
        self.last_decode_radius = radii.detach()
        centers = mu.unsqueeze(2) + radii.unsqueeze(2).unsqueeze(-1) * offsets  # Eq. 9
        rgbs = self.rgb_act(raw[..., 3:6])
        scales = self.scale_act(raw[..., 6:9])
        rotations = self.rot_act(raw[..., 9:13])
        opacity = self.opacity_act(raw[..., 13:14])
        centers = centers + self.opt.gaussian_z_offset * torch.tensor(
            [0.0, 0.0, 1.0], dtype=centers.dtype, device=centers.device
        )
        return torch.cat(
            [
                centers.reshape(batch, num_tokens * patches, 3),
                opacity.reshape(batch, num_tokens * patches, 1),
                scales.reshape(batch, num_tokens * patches, 3),
                rotations.reshape(batch, num_tokens * patches, 4),
                rgbs.reshape(batch, num_tokens * patches, 3),
            ],
            dim=-1,
        )


__all__ = [
    "AnchorGuidedDecoderBlock",
    "LocusGSAnchorDecoder",
    "LocusGSGaussianHead",
    "anchor_ray_geometric_bias",
    "inverse_softplus",
    "plucker_point_distance",
    "sinusoidal_positional_encoding",
]
