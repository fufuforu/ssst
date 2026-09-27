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

"""Instance-state-driven LocusGS (``LOCUSGS_INSTANCE_STATE_V1``).

The reconstruction path of ``LocusGSRecon`` is reproduced bit-exactly when the
coupling coefficient ``beta`` is zero; every new module lives under
``instance_state.*`` and is initialised inside a ``torch.random.fork_rng`` island
seeded with 31415 so both experiment arms start from the same state.

All formulae follow ``docs/instance_state_v1_codex_prompt.md``; the constants are
mirrored in ``group_plus/instance_state_v1/spec.json``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.canonical_recon import canonical_layer_loss
from tokengs.models.canonical_recon_models import LocusGSRecon, _full_supervision
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.locusgs_recon import (
    LocusGSAnchorDecoder,
    anchor_ray_geometric_bias,
    sinusoidal_positional_encoding,
)

NUM_THING = 100
NUM_STUFF = 2
VOID_INDEX = 102
NUM_STATES = NUM_THING + NUM_STUFF + 1
STATE_DIM = 256
ID_DIM = 16
FEAT_DIM = 2 * STATE_DIM + 6
BETA_RAMP_STEPS = 200
SEG_RAMP_STEPS = 200
GS_DECODE_RADIUS = 0.15
ASSIGN_TEMPERATURE = 0.1
ASSIGN_TIGHTNESS = 0.1
ASSIGN_CLAMP = 25.0


def xavier_(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def zero_(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def inverse_softplus_stable(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


class InstanceStateController(nn.Module):
    """Shared update network for the 100 thing + 2 stuff instance states."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.C = int(opt.enc_embed_dim)
        self.D = STATE_DIM
        self.patches = int(opt.dec_patch_size) ** 2
        self.ln_h = nn.LayerNorm(self.C, eps=1e-5)
        self.proj_h = nn.Linear(self.C, self.D)
        self.proj_m = nn.Linear(4, self.D)
        self.ln_x = nn.LayerNorm(self.D, eps=1e-5)
        self.ln_e = nn.LayerNorm(self.D, eps=1e-5)
        self.proj_e = nn.Linear(self.D, ID_DIM)
        self.ln_u = nn.LayerNorm(self.D, eps=1e-5)
        self.proj_u = nn.Linear(self.D, ID_DIM)
        # 100 thing + 2 stuff learnable states; the void channel is not a query
        self.query_init = nn.Parameter(torch.empty(NUM_THING + NUM_STUFF, self.D))
        self.gru = nn.GRUCell(self.D, self.D)
        self.ln_gru = nn.LayerNorm(self.D, eps=1e-5)
        self.ffn_fc1 = nn.Linear(self.D, 2 * self.D)
        self.ffn_fc2 = nn.Linear(2 * self.D, self.D)
        self.ln_ffn = nn.LayerNorm(self.D, eps=1e-5)
        self.token_void = nn.Linear(self.D, 1)
        self.thing_classifier = nn.Linear(self.D, 19)
        self.ln_vq = nn.LayerNorm(self.D, eps=1e-5)
        self.proj_vq = nn.Linear(self.D, self.D)
        self.ln_fx = nn.LayerNorm(self.D, eps=1e-5)
        self.ln_fm = nn.LayerNorm(self.D, eps=1e-5)
        self.proj_wh = nn.Linear(FEAT_DIM, self.C)
        self.proj_wmu = nn.Linear(FEAT_DIM, 3)
        self.proj_wr = nn.Linear(FEAT_DIM, 1)
        self.proj_wgs = nn.Linear(FEAT_DIM, self.patches * 3)
        self.ln_de = nn.LayerNorm(self.C, eps=1e-5)
        self.proj_de = nn.Linear(self.C, self.patches * ID_DIM)
        self.proj_off = nn.Linear(3, ID_DIM)
        self.ln_void = nn.LayerNorm(self.C, eps=1e-5)
        self.proj_gvoid = nn.Linear(self.C, self.patches)
        self._init_weights()
        self.reset_diagnostics()

    def _init_weights(self) -> None:
        for module in (self.proj_h, self.proj_m, self.proj_e, self.proj_u,
                       self.ffn_fc1, self.ffn_fc2, self.proj_vq, self.proj_wh,
                       self.proj_wmu, self.proj_wr, self.proj_off,
                       self.thing_classifier):
            xavier_(module)
        for module in (self.token_void, self.proj_wgs, self.proj_de, self.proj_gvoid):
            zero_(module)
        hidden = self.gru.hidden_size
        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if name.startswith("weight_ih") or name.startswith("weight_hh"):
                    for gate in range(3):
                        nn.init.xavier_uniform_(param[gate * hidden:(gate + 1) * hidden])
                else:
                    nn.init.zeros_(param)
            nn.init.normal_(self.query_init, std=0.02)
        norms = (self.ln_h, self.ln_x, self.ln_e, self.ln_u, self.ln_gru,
                 self.ln_ffn, self.ln_vq, self.ln_fx, self.ln_fm, self.ln_de,
                 self.ln_void)
        for ln in norms:
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

    def reset_diagnostics(self) -> None:
        self.last_low_mass_states = 0
        self.last_radius_clamp_hits = 0
        self.last_assign_entropy = None

    # ------------------------------------------------------------------ #
    def encode_token(self, h, mu, r, ell):
        """X(h, mu, r) -> [B, T, D]; ``r`` is [B, T], ``ell`` is [B]."""
        feat = torch.cat([mu / ell.reshape(-1, 1, 1),
                          torch.log(torch.clamp(r / ell.reshape(-1, 1), 1e-4, 1e4)
                                    ).unsqueeze(-1)], dim=-1)
        return self.ln_x(self.proj_h(self.ln_h(h)) + self.proj_m(feat))

    def embed(self, x):
        return F.normalize(self.proj_e(self.ln_e(x)), dim=-1, eps=1e-6)

    def embed_query(self, q):
        return F.normalize(self.proj_u(self.ln_u(q)), dim=-1, eps=1e-6)

    def assign(self, e, pos, q, c, s, void_logit):
        """103-channel softmax over 100 thing + 2 stuff + 1 void."""
        u_thing = self.embed_query(q[:, :NUM_THING])
        u_stuff = self.embed_query(q[:, NUM_THING:NUM_THING + NUM_STUFF])
        dot_thing = torch.einsum("bmd,bqd->bmq", e, u_thing) * 10.0
        delta = (pos.unsqueeze(2) - c[:, :NUM_THING].unsqueeze(1)) / s[:, :NUM_THING].unsqueeze(1)
        tight = torch.clamp((delta ** 2).sum(-1), max=ASSIGN_CLAMP) * ASSIGN_TIGHTNESS
        stuff = torch.einsum("bmd,bqd->bmq", e, u_stuff) * 10.0
        logits = torch.cat([dot_thing - tight, stuff, void_logit], dim=-1)
        return torch.softmax(logits / ASSIGN_TEMPERATURE, dim=-1)

    def update_states(self, x, mu, r, q, c, s, void_logit):
        """One registration update (spec step 5.2)."""
        ell = torch.clamp(
            (mu - mu.mean(dim=1, keepdim=True)).pow(2).sum(-1).mean(-1).sqrt(),
            min=0.05).detach()
        e = self.embed(x)
        A_pre = self.assign(e, mu, q, c, s, void_logit)
        mass = A_pre.sum(dim=1)
        w = A_pre / (mass.unsqueeze(1) + 1e-6)
        q_old = q
        z = torch.einsum("btq,btd->bqd", w[:, :, :NUM_THING + NUM_STUFF], x)
        flat = z.reshape(-1, self.D)
        v = self.ln_gru(self.gru(flat, q_old.reshape(-1, self.D))).reshape_as(q_old)
        q_new = self.ln_ffn(v + self.ffn_fc2(F.gelu(self.ffn_fc1(v))))
        w_thing = w[:, :, :NUM_THING]
        chat = torch.einsum("btq,btd->bqd", w_thing, mu)
        diff = mu.unsqueeze(2) - chat.unsqueeze(1)
        var = torch.einsum("btq,btqd->bqd", w_thing, diff ** 2)
        shat = torch.sqrt(var + (0.05 * ell).reshape(-1, 1, 1) ** 2)
        c_new = 0.5 * c[:, :NUM_THING] + 0.5 * chat
        lo = (0.05 * ell).reshape(-1, 1, 1)
        hi = (2.0 * ell).reshape(-1, 1, 1)
        s_new = torch.clamp(0.5 * s[:, :NUM_THING] + 0.5 * shat, lo, hi)
        low_mass = (mass[:, :NUM_THING] < 1e-4).unsqueeze(-1)
        self.last_low_mass_states = int(low_mass.sum())
        c_new = torch.where(low_mass, c[:, :NUM_THING], c_new)
        s_new = torch.where(low_mass, s[:, :NUM_THING], s_new)
        c_full = torch.cat([c_new, c[:, NUM_THING:]], dim=1)
        s_full = torch.cat([s_new, s[:, NUM_THING:]], dim=1)
        A_post = self.assign(self.embed(x), mu, q_new, c_full, s_full, void_logit)
        with torch.no_grad():
            p = A_post.clamp_min(1e-9)
            self.last_assign_entropy = float(-(p * p.log()).sum(-1).mean())
        return q_new, c_full, s_full, A_post, ell

    def token_message(self, x, mu, A_post, q, c, s, ell):
        """F_i (518-d) plus the thing mass t_i."""
        vq = self.proj_vq(self.ln_vq(q))
        a_query = A_post[:, :, :NUM_THING + NUM_STUFF]
        mass_all = torch.clamp(a_query.sum(dim=-1, keepdim=True), min=1e-6)
        m = torch.einsum("btq,bqd->btd", a_query, vq) / mass_all
        a_thing = A_post[:, :, :NUM_THING]
        t = a_thing.sum(dim=-1, keepdim=True)
        t_safe = torch.clamp(t, min=1e-6)
        delta = c[:, :NUM_THING].unsqueeze(2) - mu.unsqueeze(1)
        d = torch.einsum("btq,bqtd->btd", a_thing, delta)
        d = d / (ell.reshape(-1, 1, 1) * t_safe)
        v = torch.einsum("btq,bqd->btd", a_thing,
                         torch.log(s[:, :NUM_THING] / ell.reshape(-1, 1, 1))) / t_safe
        f_vec = torch.cat([self.ln_fx(x), self.ln_fm(m), d, v], dim=-1)
        return f_vec, t

    def custom_fwd(self, tokens, mu, r, f_vec, beta):
        """h/mu/r write-back (spec step 5.4); identity when beta == 0."""
        if beta == 0.0:
            return tokens, mu, r
        h_new = tokens + beta * 0.1 * torch.tanh(self.proj_wh(f_vec))
        mu_new = mu + beta * 0.1 * r.unsqueeze(-1) * torch.tanh(self.proj_wmu(f_vec))
        r_new = r * torch.exp(beta * 0.05 * torch.tanh(self.proj_wr(f_vec).squeeze(-1)))
        return h_new, mu_new, r_new

    def rho_from_radius(self, r, epsilon):
        clamped = torch.clamp(r - epsilon, min=1e-6)
        with torch.no_grad():
            self.last_radius_clamp_hits = int((r - epsilon < 1e-6).sum())
        return inverse_softplus_stable(clamped)

    def compactness_bias(self, A_post, beta):
        """Additive [B,1,T,T] self-attention bias, or None when beta == 0."""
        if beta == 0.0:
            return None
        a = A_post[:, :, :NUM_THING]
        u = F.normalize(a, dim=-1, eps=1e-6)
        t = a.sum(dim=-1)
        cos = torch.clamp(torch.einsum("bid,bjd->bij", u, u), 1e-4, 1.0)
        bias = 0.5 * beta * t.unsqueeze(-1) * t.unsqueeze(-2) * torch.log(cos)
        eye = torch.eye(bias.shape[-1], device=bias.device, dtype=bias.dtype)
        return (bias * (1.0 - eye)).unsqueeze(1)


@torch.no_grad()
def deterministic_fps(mu: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic farthest-point selection over token centres (ties -> lowest index)."""
    if mu.shape[0] != 1:
        raise ValueError(f"deterministic FPS expects B=1, got {mu.shape[0]}")
    centroid = mu.mean(dim=1, keepdim=True)
    dist = (mu - centroid).pow(2).sum(-1)[0]
    selected = [int(torch.argmin(dist).item())]
    min_d = (mu[0] - mu[0, selected[0]]).pow(2).sum(-1).clone()
    while len(selected) < count:
        min_d[selected] = -1.0
        nxt = int(torch.argmax(min_d).item())
        selected.append(nxt)
        min_d = torch.minimum(min_d, (mu[0] - mu[0, nxt]).pow(2).sum(-1))
    if len(set(selected)) != count:
        raise RuntimeError("deterministic FPS produced duplicate indices")
    return torch.tensor(selected, device=mu.device, dtype=torch.long).unsqueeze(0)


def gather_tokens(mu: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(mu, 1, index.unsqueeze(-1).expand(-1, -1, mu.shape[-1]))


class InstanceStateDecoder(LocusGSAnchorDecoder):
    """Anchor decoder that interleaves the instance-state registration update."""

    def __init__(self, opt, decoder_blocks):
        super().__init__(opt, decoder_blocks)
        self.state_layers = tuple(int(x) for x in opt.instance_state_layers)
        self.last_fps_index = None

    @staticmethod
    def _self_attn_with_bias(block, tokens: torch.Tensor, bias):
        """`DecoderBlock.SelfAttnBlock.forward` with an optional additive bias."""
        attn = block.gs_self_attn
        if bias is None:
            return block(tokens)
        if getattr(attn, "rope", None) is not None \
                or getattr(attn, "flex_attn_score_mod", None) is not None \
                or getattr(attn, "flex_attn_block_mask", None) is not None \
                or not attn.fused_attn:
            raise RuntimeError(
                "instance_state requires the verified default SDPA path "
                "(no rope, no flex score_mod/block_mask, fused_attn=True)")
        x = block.norm(tokens)
        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=attn.attn_drop.p if attn.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return attn.proj_drop(attn.proj(out))

    def forward_stateful(self, tokens, encoder_latent, patch_rays, controller,
                         coupled: bool, step: int):
        """Original layer order plus the state update at the registered layers."""
        if patch_rays is None:
            raise ValueError("LocusGS anchor decoder requires patch-level rays")
        moment, direction = patch_rays
        batch = tokens.shape[0]
        if moment.shape[1] != int(encoder_latent.keys.shape[-2]):
            raise ValueError("patch-level rays must match encoder keys")
        mu = self.mu.unsqueeze(0).expand(batch, -1, -1).contiguous()
        rho = self.rho.unsqueeze(0).expand(batch, -1).contiguous()
        beta = (1.0 if coupled else 0.0) * min(max(float(step), 0.0) / BETA_RAMP_STEPS, 1.0)
        states, ray_bias_stats = [], []
        q = c = s = A_post = f_vec = ell = None
        pending_bias = None
        for index, (block, head_mu, head_rho) in enumerate(
                zip(self.decoder_blocks, self.refine_mu, self.refine_rho)):
            radii = self.activated_radius(rho)
            attn_bias = None
            if float(self.opt.locusgs_ray_bias_scale) > 0:
                geometric = anchor_ray_geometric_bias(
                    mu, radii, moment, direction, sigma0=self.sigma0,
                    bandwidth_floor=self.bandwidth_floor,
                    clamp_min=self.bias_clamp_min)
                attn_bias = F.softplus(self.gamma_raw[index]) * geometric
                bias_stat = attn_bias.detach()
                ray_bias_stats.append(dict(
                    ray_bias_mean=bias_stat.mean(),
                    ray_bias_clamped_fraction=(
                        geometric.detach() <= self.bias_clamp_min + 1e-6).float().mean(),
                    gamma=F.softplus(self.gamma_raw[index]).detach()))
            pe_module = self.pe_mlp if self.pe_mlp is not None else self.pe_mlps[index]
            anchor_pe = pe_module(
                sinusoidal_positional_encoding(mu, int(self.opt.locusgs_pe_num_freqs)))
            tokens = tokens + block.gs_cross_attn_scale(
                block.gs_cross_attn(tokens, encoder_latent.keys,
                                    encoder_latent.values, attn_bias=attn_bias))
            if self.pe_mode == "persistent":
                tokens = tokens + anchor_pe
                tokens = tokens + block.gs_self_attn_scale(
                    self._self_attn_with_bias(block.gs_self_attn, tokens, pending_bias))
            else:
                tokens = tokens + block.gs_self_attn_scale(
                    self._self_attn_with_bias(block.gs_self_attn, tokens + anchor_pe,
                                              pending_bias))
            tokens = tokens + block.mlp_scale(block.mlp(tokens))
            previous_mu, previous_radii = mu, radii
            mu = mu + head_mu(tokens)
            rho = rho + head_rho(tokens).squeeze(-1)
            radii = self.activated_radius(rho)

            layer = index + 1
            if layer == self.state_layers[0]:
                ell = torch.clamp(
                    (mu - mu.mean(dim=1, keepdim=True)).pow(2).sum(-1).mean(-1).sqrt(),
                    min=0.05).detach()
                sel = deterministic_fps(mu.detach(), NUM_THING)
                self.last_fps_index = sel
                x6 = controller.encode_token(tokens, mu, radii, ell)
                q = torch.cat([
                    controller.query_init[:NUM_THING].unsqueeze(0) + gather_tokens(x6, sel),
                    controller.query_init[NUM_THING:NUM_THING + NUM_STUFF].unsqueeze(0)
                    + x6.mean(dim=1, keepdim=True)], dim=1)
                c = gather_tokens(mu, sel)
                s = ell.reshape(-1, 1, 1).expand(-1, NUM_THING, 3).clone()
            if layer in self.state_layers:
                x = controller.encode_token(tokens, mu, radii, ell)
                void_logit = controller.token_void(x)
                q, c, s, A_post, ell = controller.update_states(
                    x, mu, radii, q, c, s, void_logit)
                f_vec, _ = controller.token_message(x, mu, A_post, q, c, s, ell)
                if beta != 0.0:
                    tokens, mu, r_new = controller.custom_fwd(
                        tokens, mu, radii, f_vec, beta)
                    rho = controller.rho_from_radius(r_new, self.epsilon)
                    radii = self.activated_radius(rho)
                pending_bias = controller.compactness_bias(A_post, beta)
            states.append(dict(
                layer=layer, tokens=tokens, mu=mu, rho=rho, radii=radii,
                anchor_update=(mu - previous_mu).norm(dim=-1).mean().detach(),
                radius_update=(radii - previous_radii).abs().mean().detach(),
                q=q, c=c, s=s, A_post=A_post, F=f_vec, ell=ell, beta=beta,
                fps_index=self.last_fps_index))
        return states, ray_bias_stats


class LocusGSInstanceStateRecon(LocusGSRecon):
    """Full-data reconstruction + instance state + 20-class / panoptic read-out."""

    architecture_name = "LOCUSGS_INSTANCE_STATE_V1"

    def __init__(self, opt):
        super().__init__(opt)
        self.anchor_decoder = InstanceStateDecoder(opt, self.enc_dec_backbone.decoder_blocks)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(getattr(opt, "instance_state_init_seed", 31415)))
            self.instance_state = InstanceStateController(opt)
        self.instance_state_layers = tuple(int(x) for x in opt.instance_state_layers)
        self.understanding_step = 0

    # ------------------------------------------------------------------ #
    def decode_stateful(self, model_input, decoder_input, coupled, step=None):
        from tokengs.models.canonical_recon_models import patch_plucker_rays
        encoder_latent = self.forward_encoder(model_input.encoder)
        patch_rays = patch_plucker_rays(
            model_input.encoder.rays_os, model_input.encoder.rays_ds,
            patch_size=int(self.opt.patch_size))
        use_step = self.understanding_step if step is None else step
        return self.anchor_decoder.forward_stateful(
            self.get_gs_tokens(batch_size=model_input.batch_size), encoder_latent,
            patch_rays, self.instance_state, bool(coupled), use_step)

    def _gaussians_from_state(self, state, beta: float) -> torch.Tensor:
        """Single geometry helper shared by reconstruction and the identity head."""
        base = self.activation_head(state["tokens"], state["mu"], state["radii"])
        B, T = state["tokens"].shape[0], state["tokens"].shape[1]
        patches = base.shape[1] // T
        delta = torch.tanh(self.instance_state.proj_wgs(state["F"]))
        delta = delta.reshape(B, T, patches, 3)
        base_xyz = base[..., 0:3].reshape(B, T, patches, 3)
        xyz = base_xyz + beta * 0.1 * GS_DECODE_RADIUS * delta
        return torch.cat([xyz.reshape(B, -1, 3), base[..., 3:]], dim=-1)

    def forward_instance_state(self, model_input, *, render_decoder_input=None,
                               coupled=None, step=None) -> dict:
        decoder_input = render_decoder_input or model_input.decoder
        if coupled is None:
            coupled = bool(getattr(self.opt, "instance_state_coupled", False))
        states, ray_stats = self.decode_stateful(model_input, decoder_input, coupled, step)
        final = states[-1]
        beta = float(final["beta"])
        gaussians = self._gaussians_from_state(final, beta)
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        render = self.render_reconstruction(reconstruction, decoder_input)

        ctrl = self.instance_state
        B, T = final["tokens"].shape[0], final["tokens"].shape[1]
        patches = gaussians.shape[1] // T
        e_token = ctrl.embed(ctrl.encode_token(
            final["tokens"], final["mu"], final["radii"], final["ell"]))
        de = ctrl.proj_de(ctrl.ln_de(final["tokens"])).reshape(B, T, patches, ID_DIM)
        flat_xyz = gaussians[..., 0:3].reshape(B, T, patches, 3)
        centre = final["mu"].unsqueeze(2).expand(-1, -1, patches, 3)
        offset = torch.clamp((flat_xyz - centre) / GS_DECODE_RADIUS, -2.0, 2.0)
        e_gs = F.normalize(e_token.unsqueeze(2) + 0.25 * torch.tanh(de)
                           + 0.1 * torch.tanh(ctrl.proj_off(offset)), dim=-1, eps=1e-6)
        e_gs = e_gs.reshape(B, -1, ID_DIM)
        void_gs = ctrl.proj_gvoid(ctrl.ln_void(final["tokens"])).reshape(B, -1, 1)
        A_g = ctrl.assign(e_gs, flat_xyz.reshape(B, -1, 3), final["q"], final["c"],
                          final["s"], void_gs)
        channel = self.gs.render_feature_channels(
            gaussians, torch.cat([A_g, e_gs], dim=-1),
            decoder_input.cam_view, decoder_input.intrinsics)
        M = channel["images_pred"][..., :NUM_STATES, :, :]
        E_render = channel["images_pred"][..., NUM_STATES:, :, :]
        alpha = render["alphas_pred"]
        p_class = torch.softmax(ctrl.thing_classifier(final["q"][:, :NUM_THING]), dim=-1)
        S = torch.zeros(B, M.shape[1], 20, *M.shape[-2:], device=M.device, dtype=M.dtype)
        thing_mass = M[:, :, :NUM_THING]
        S[:, :, 0] = M[:, :, NUM_THING]
        S[:, :, 1] = M[:, :, NUM_THING + 1]
        for cls in range(2, 20):
            weight = p_class[:, :, cls - 2].reshape(B, 1, NUM_THING, 1, 1)
            S[:, :, cls] = (thing_mass * weight).sum(dim=2)
        no_object = p_class[:, :, 18].reshape(B, 1, NUM_THING, 1, 1)
        Svoid = (1.0 - alpha) + M[:, :, VOID_INDEX:VOID_INDEX + 1] \
            + (thing_mass * no_object).sum(dim=2).unsqueeze(2)
        logits21 = ctrl.thing_classifier(final["q"][:, :NUM_THING])
        pad = torch.full_like(logits21[:, :, :1], -1e4)
        thing_logits = torch.cat([pad, pad, logits21], dim=-1)
        return {
            "gaussians": gaussians, "render": render, "states": states,
            "thing_class_logits": thing_logits, "assignment": A_g,
            "region_mass": M, "semantic_scores": S, "pixel_void_mass": Svoid,
            "identity_render": E_render, "alpha": alpha, "ray_stats": ray_stats,
            "p_class": p_class, "beta": beta,
        }

    # -- training ------------------------------------------------------- #
    def step_loss(self, batch: dict, *, step: int, phase: str = "train",
                  coupled=None, rseg_override=None):
        from tokengs.models.instance_state_loss import instance_state_losses
        del phase
        model_input, _ = split_data(batch, self.opt)
        decoder_input = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
        if coupled is None:
            coupled = bool(getattr(self.opt, "instance_state_coupled", False))
        prediction = self.forward_instance_state(
            ModelInput(model_input.encoder, decoder_input),
            render_decoder_input=decoder_input, coupled=coupled, step=step)
        supervision = _full_supervision(batch)
        metrics: dict = {}
        total = None
        for layer, weight in zip(self.supervised_layers, self.layer_weights):
            state = prediction["states"][layer - 1]
            gaussians = self._gaussians_from_state(state, float(state["beta"]))
            render = self.render_reconstruction(
                self._reconstruction_from_gaussians(gaussians), decoder_input)
            layer_loss = canonical_layer_loss(
                opt=self.opt, img_size=self.img_size, render_results=render,
                supervision=supervision, decoder_input=decoder_input,
                gaussians=gaussians, anchor_centers=state["mu"],
                anchor_weight=float(self.opt.canonical_anchor_visibility_weight))
            total = layer_loss["loss"] * weight if total is None else \
                total + layer_loss["loss"] * weight
            metrics[f"loss_layer{layer}"] = layer_loss["loss"]
            for key in ("loss_rgb", "loss_ssim", "loss_gaussian_visibility",
                        "loss_anchor_visibility", "psnr"):
                if key in layer_loss:
                    metrics[f"{key}_layer{layer}"] = layer_loss[key]
        metrics["loss_recon"] = total
        metrics["psnr"] = metrics[f"psnr_layer{self.supervised_layers[-1]}"]
        rseg = 0.2 + 0.8 * min(max(float(step), 0.0) / SEG_RAMP_STEPS, 1.0)
        if rseg_override is not None:
            rseg = float(rseg_override)
        seg, seg_metrics = instance_state_losses(
            prediction=prediction, batch=batch, opt=self.opt, context_views=2)
        metrics.update(seg_metrics)
        metrics["rseg"] = torch.tensor(rseg, device=total.device, dtype=torch.float32)
        metrics["loss_understanding"] = seg
        metrics["loss"] = total + rseg * seg
        return {"prediction": prediction}, metrics

    def forward(self, data, skip_loss: bool = False):
        del skip_loss
        if isinstance(data, ModelInput):
            return self.forward_instance_state(data)
        if isinstance(data, dict):
            output, _ = self.step_loss(data, step=self.understanding_step)
            return output
        raise TypeError(f"unsupported forward input type: {type(data)!r}")


__all__ = [
    "LocusGSInstanceStateRecon",
    "InstanceStateController",
    "InstanceStateDecoder",
    "deterministic_fps",
    "gather_tokens",
    "NUM_THING",
    "NUM_STUFF",
    "NUM_STATES",
]
