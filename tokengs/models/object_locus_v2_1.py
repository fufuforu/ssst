"""LOCUSGS_OBJECT_LOCUS_V2_1: independent masks with Gaussian child residuals."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.canonical_recon_models import LocusGSRecon, _full_supervision
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.object_locus_v2_1_controller import (
    ID_DIM, NUM_QUERIES, NUM_THING, ObjectLocusV2_1Controller, scene_normalization,
)


def alpha_normalize_membership(membership_mass, alpha):
    value = torch.where(alpha > 0, membership_mass / alpha.clamp_min(1e-6),
                        torch.zeros_like(membership_mass))
    return value.clamp(0.0, 1.0)


class LocusGSObjectLocusV2_1Recon(LocusGSRecon):
    architecture_name = "LOCUSGS_OBJECT_LOCUS_V2_1"
    reconstruction_only = False
    state_layers = (6, 8, 10, 12)

    def __init__(self, opt):
        super().__init__(opt)
        self.reconstruction_only = False
        if tuple(int(x) for x in opt.instance_state_layers) != self.state_layers:
            raise ValueError("Object-Locus V2.1 state layers are fixed to (6, 8, 10, 12)")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(getattr(opt, "object_locus_init_seed", 31415)))
            self.object_locus_v2_1 = ObjectLocusV2_1Controller(int(opt.enc_embed_dim))
        self.understanding_step = 0

    def decode_object_locus(self, model_input):
        states, ray_stats = super()._decode(model_input, model_input.decoder)
        # One scene coordinate system, fixed from the detached layer-6 geometry.
        mu6 = states[5]["mu"]
        origin, ell = scene_normalization(mu6)
        q = c = s = seed_indices = None
        for state in states:
            layer = int(state["layer"])
            state["beta"] = 0.0
            if layer not in self.state_layers:
                continue
            tokens, mu, radii = state["tokens"], state["mu"], state["radii"]
            anchor_embedding = self.object_locus_v2_1.encode_token(tokens, mu, radii, ell)
            if layer == 6:
                q, c, s, seed_indices, neighborhoods, init_weights = self.object_locus_v2_1.initialize_states(
                    anchor_embedding, tokens, mu, origin, ell
                )
                state["seed_neighbors"] = neighborhoods
                state["seed_pool_weights"] = init_weights
            result = self.object_locus_v2_1.forward_registered_layer(
                tokens, mu, radii, ell, q, c, s, anchor_embedding=anchor_embedding
            )
            result["seed_indices"] = seed_indices
            result["ell"] = ell
            result["scene_origin"] = origin
            state.update(result)
            q, c, s = result["q"], result["c"], result["s"]
        return states, ray_stats

    def _readout(self, final, gaussians, decoder):
        batch, anchors = final["tokens"].shape[:2]
        children = gaussians.shape[1] // anchors
        if children != 64:
            raise RuntimeError(f"expected 64 child Gaussians per anchor, got {children}")
        anchor_membership = final["anchor_membership"]
        f_gaussian, child_residual = self.object_locus_v2_1.gaussian_child_features(
            final["anchor_embedding"], final["f_anchor"], gaussians,
            final["mu"], final["radii"]
        )
        gaussian_mask_logits, gaussian_membership = self.object_locus_v2_1.gaussian_membership(
            f_gaussian, final["m_query"], final["mask_bias"]
        )
        u_gaussian, gaussian_pool_mass = self.object_locus_v2_1.pool_gaussian_features(
            f_gaussian, gaussian_membership, gaussians, final["u_anchor"]
        )
        classification = self.object_locus_v2_1.classify(final["q"], u_gaussian)
        # L12's exported/matching/loss classifier is the Gaussian-mask pooled readout.
        final["anchor_pooled_feature"] = final["u_anchor"]
        final.update(classification)
        final["u_gaussian"] = u_gaussian
        final["gaussian_pool_mass"] = gaussian_pool_mass
        identity = self.object_locus_v2_1.identity_features(final, gaussians)
        rendered = self.gs.render_feature_channels(
            gaussians, torch.cat((gaussian_membership, identity), dim=-1),
            decoder.cam_view, decoder.intrinsics
        )
        membership_mass = rendered["images_pred"][:, :, :102]
        identity_render = rendered["images_pred"][:, :, 102:]
        alpha = rendered["alphas_pred"]
        membership_pixel = alpha_normalize_membership(membership_mass, alpha)
        class_logits19 = final["thing_logits19"]
        p_class = final["p_class"]
        semantic_raw = membership_pixel.new_zeros(
            (batch, membership_pixel.shape[1], 20, *membership_pixel.shape[-2:]))
        semantic_raw[:, :, 0] = membership_pixel[:, :, 100]
        semantic_raw[:, :, 1] = membership_pixel[:, :, 101]
        thing_mass = membership_pixel[:, :, :NUM_THING]
        for class_index in range(18):
            semantic_raw[:, :, class_index + 2] = (
                thing_mass * p_class[:, :, class_index].reshape(batch, 1, NUM_THING, 1, 1)
            ).sum(2)
        semantic = semantic_raw / (semantic_raw.sum(2, keepdim=True) + 1e-6)
        pixel_void = 1.0 - alpha
        logits21 = final["thing_class_logits"]
        return {
            "assignment": gaussian_membership,
            "gaussian_membership": gaussian_membership,
            "gaussian_mask_logits": gaussian_mask_logits,
            "gaussian_feature": f_gaussian,
            "child_feature_residual": child_residual,
            "anchor_assignment": anchor_membership,
            "anchor_membership": anchor_membership,
            "anchor_mask_logits": final["anchor_mask_logits"],
            "membership_mass": membership_mass,
            "region_mass": membership_pixel,
            "pixel_membership": membership_pixel,
            "semantic_scores": semantic,
            "pixel_void_mass": pixel_void,
            "identity_render": identity_render,
            "alpha": alpha,
            "p_class": p_class,
            "conditional_class_prob": final["conditional_class_prob"],
            "objectness_prob": final["objectness_prob"],
            "pooled_feature": final["pooled_feature"],
            "anchor_pool_mass": final["anchor_pool_mass"],
            "gaussian_pool_mass": gaussian_pool_mass,
            "thing_class_logits": logits21,
        }

    def forward_object_locus(self, model_input, *, render_decoder_input=None,
                             context_decoder=None, coupled=False, step=None):
        if coupled:
            raise RuntimeError("Object-Locus V2.1 has no legacy beta coupling; beta is fixed at zero.")
        decoder = render_decoder_input or model_input.decoder
        context = context_decoder or decoder
        input_with_decoder = ModelInput(model_input.encoder, decoder)
        states, ray_stats = self.decode_object_locus(input_with_decoder)
        final = states[-1]
        gaussians = self.activation_head(final["tokens"], final["mu"], final["radii"])
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        render = self.render_reconstruction(reconstruction, decoder)
        output = {
            "reconstruction": reconstruction,
            "gaussians": gaussians,
            "render": render,
            "states": states,
            "beta": 0.0,
            "ray_stats": ray_stats,
        }
        output.update(self._readout(final, gaussians, context))
        return output

    def forward_instance_state(self, model_input, *, render_decoder_input=None,
                               context_decoder=None, coupled=False, step=None):
        return self.forward_object_locus(
            model_input, render_decoder_input=render_decoder_input,
            context_decoder=context_decoder, coupled=coupled, step=step
        )

    def forward_reconstruction_only(self, model_input, *, render_decoder_input=None):
        return self.forward_object_locus(model_input, render_decoder_input=render_decoder_input)

    def step_loss(self, batch, *, step, phase="train", coupled=False,
                  understanding_weight=1.0):
        del phase
        if coupled:
            raise RuntimeError("Object-Locus V2.1 has no legacy beta coupling; beta is fixed at zero.")
        model_input, _ = split_data(batch, self.opt)
        decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                    intrinsics=batch["intrinsics_all"])
        context = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :2],
                                    intrinsics=batch["intrinsics_all"][:, :2])
        prediction = self.forward_object_locus(
            ModelInput(model_input.encoder, decoder), render_decoder_input=decoder,
            context_decoder=context, coupled=False, step=step
        )
        recon, recon_metrics, *_ = self._layer_objective(
            prediction["states"], decoder, _full_supervision(batch)
        )
        from tokengs.models.object_locus_v2_1_loss import object_locus_v2_1_losses
        understanding, loss_metrics = object_locus_v2_1_losses(
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

    def forward(self, data, skip_loss=False):
        del skip_loss
        if isinstance(data, ModelInput):
            return self.forward_object_locus(data)
        if isinstance(data, dict):
            return self.step_loss(data, step=self.understanding_step)[0]
        raise TypeError(f"unsupported input type: {type(data)!r}")
