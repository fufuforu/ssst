"""Fixed Object-Locus Joint V1 interleaved token-only generation coupling."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.locusgs_recon import (LocusGSAnchorDecoder,
    anchor_ray_geometric_bias, sinusoidal_positional_encoding)
from tokengs.models.canonical_recon_models import LocusGSRecon, patch_plucker_rays, _full_supervision
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.object_locus_v3_set import LocusGSObjectLocusV3SetRecon
from tokengs.models.object_locus_v3_set_controller import ObjectLocusV3SetController, scene_normalization


def joint_beta(step):
    return min(max(float(step), 0.0) / 200.0, 1.0)


def joint_route(a, q, mu, c, s):
    a_hat = F.normalize(F.layer_norm(a, (256,), eps=1e-5), dim=-1, eps=1e-6)
    values = F.layer_norm(q, (256,), eps=1e-5)
    q_hat = F.normalize(values, dim=-1, eps=1e-6)
    d2 = (((mu[:, :, None] - c[:, None]) / (s[:, None] + 1e-6)) ** 2).sum(-1)
    if not torch.isfinite(d2).all():
        raise FloatingPointError('nonfinite joint route d2')
    geo = torch.cat((-torch.log1p(d2), d2.new_zeros((*d2.shape[:2], 2))), -1)
    route = torch.softmax(torch.matmul(a_hat, q_hat.transpose(-1, -2)) / 0.1 + geo, dim=-1)
    return route, torch.matmul(route, values), values


def joint_residual(tokens, raw, beta):
    sigma = tokens.square().mean(-1, keepdim=True).sqrt().detach()
    return float(beta) * 0.1 * sigma * torch.tanh(raw)


class ObjectLocusJointAnchorDecoder(LocusGSAnchorDecoder):
    def forward_stateful(self, tokens, encoder_latent, patch_rays, controller, injection, *, enabled, step):
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
        q = c = s = origin = ell = seed_indices = None
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
            layer = index + 1
            result = {}
            beta = joint_beta(step) if enabled else 0.0
            if layer in (6, 8, 10, 12):
                if layer == 6:
                    origin, ell = scene_normalization(mu)
                a = controller.encode_token(tokens, mu, radii, ell)
                if layer == 6:
                    q, c, s, seed_indices, neighbors, weights = controller.initialize_states(
                        a, tokens, mu, origin, ell)
                    result.update(seed_neighbors=neighbors, seed_pool_weights=weights)
                result.update(controller.forward_registered_layer(
                    tokens, mu, radii, ell, q, c, s, anchor_embedding=a))
                q, c, s = result['q'], result['c'], result['s']
                result.update(seed_indices=seed_indices, ell=ell, scene_origin=origin)
                route, message, values = joint_route(a, q, mu, c, s)
                if getattr(self, 'object_mean_message', False):
                    message = values.mean(1, keepdim=True).expand_as(message)
                active = enabled and beta > 0 and layer not in getattr(self, 'disabled_layers', ())
                raw = injection[f'L{layer}'](message)
                delta = joint_residual(tokens, raw, beta if active else 0.0)
                result.update(joint_T=route, joint_u=message, joint_raw=raw,
                              joint_delta=delta, joint_h_norm=tokens.detach().norm(dim=-1),
                              joint_direct_mu=0.0, joint_direct_rho=0.0,
                              joint_direct_radius=0.0)
                if active:
                    tokens = tokens + delta
            states.append(
                dict(
                    **result,
                    beta=beta,
                    layer=index + 1,
                    tokens=tokens,
                    mu=mu,
                    rho=rho,
                    radii=radii,
                    anchor_update=(mu - previous_mu).norm(dim=-1).mean().detach(),
                    radius_update=(radii - previous_radii).abs().mean().detach(),
                )
            )
        return states, ray_bias_stats


class LocusGSObjectLocusJointV1Recon(LocusGSObjectLocusV3SetRecon):
    architecture_name = 'LOCUSGS_OBJECT_LOCUS_JOINT_V1'

    def __init__(self, opt):
        LocusGSRecon.__init__(self, opt)
        self.anchor_decoder = ObjectLocusJointAnchorDecoder(opt, self.enc_dec_backbone.decoder_blocks)
        self.reconstruction_only = False
        if tuple(map(int, opt.instance_state_layers)) != self.state_layers:
            raise ValueError('joint registered layers must be (6,8,10,12)')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31415)
            self.object_locus_v3_set = ObjectLocusV3SetController(1024)
            self.object_locus_joint_injection = nn.ModuleDict({
                f'L{layer}': nn.Linear(256, 1024, bias=False) for layer in self.state_layers})
            for projection in self.object_locus_joint_injection.values():
                nn.init.zeros_(projection.weight)
        self.inject_enabled = False
        self.understanding_step = 0

    def decode_object_locus(self, model_input, *, coupled=None, step=None):
        enabled = self.inject_enabled if coupled is None else bool(coupled)
        current_step = self.understanding_step if step is None else int(step)
        latent = self.forward_encoder(model_input.encoder)
        rays = patch_plucker_rays(model_input.encoder.rays_os, model_input.encoder.rays_ds,
                                 patch_size=int(self.opt.patch_size))
        return self.anchor_decoder.forward_stateful(
            self.get_gs_tokens(batch_size=model_input.batch_size), latent, rays,
            self.object_locus_v3_set, self.object_locus_joint_injection,
            enabled=enabled, step=current_step)

    def _decode(self, model_input, decoder_input):
        return self.decode_object_locus(ModelInput(model_input.encoder, decoder_input))

    def forward_object_locus(self, model_input, *, render_decoder_input=None,
                             context_decoder=None, coupled=None, step=None):
        decoder = render_decoder_input or model_input.decoder
        context = context_decoder or decoder
        input_with_decoder = ModelInput(model_input.encoder, decoder)
        states, ray_stats = self.decode_object_locus(input_with_decoder, coupled=coupled, step=step)
        final = states[-1]
        gaussians = self.activation_head(final["tokens"], final["mu"], final["radii"])
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        render = self.render_reconstruction(reconstruction, decoder)
        output = {
            "reconstruction": reconstruction,
            "gaussians": gaussians,
            "render": render,
            "states": states,
            "beta": final["beta"],
            "ray_stats": ray_stats,
        }
        output.update(self._readout(final, gaussians, context))
        return output

    def forward_instance_state(self, model_input, *, render_decoder_input=None,
                               context_decoder=None, coupled=None, step=None):
        return self.forward_object_locus(
            model_input, render_decoder_input=render_decoder_input,
            context_decoder=context_decoder, coupled=coupled, step=step
        )

    def forward_reconstruction_only(self, model_input, *, render_decoder_input=None):
        return self.forward_object_locus(model_input, render_decoder_input=render_decoder_input)

    def step_loss(self, batch, *, step, phase="train", coupled=None,
                  understanding_weight=1.0):
        del phase
        self.understanding_step = int(step)
        model_input, _ = split_data(batch, self.opt)
        decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                    intrinsics=batch["intrinsics_all"])
        context = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :2],
                                    intrinsics=batch["intrinsics_all"][:, :2])
        prediction = self.forward_object_locus(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
            context_decoder=context, coupled=coupled, step=step
        )
        recon, recon_metrics, *_ = self._layer_objective(
            prediction["states"], decoder, _full_supervision(batch)
        )
        from tokengs.models.object_locus_v3_set_loss import object_locus_v3_set_losses
        understanding, loss_metrics = object_locus_v3_set_losses(
            prediction, batch, self.opt
        )
        weight = float(understanding_weight)
        metrics = dict(recon_metrics)
        metrics.update(loss_metrics)
        metrics["loss_recon"] = recon
        metrics["loss_understanding"] = understanding
        metrics["understanding_weight"] = weight
        metrics["loss"] = recon + weight * understanding
        metrics["loss_total"] = metrics["loss"]
        metrics["psnr"] = recon_metrics[f"psnr_layer{self.supervised_layers[-1]}"]
        return {"prediction": prediction}, metrics

