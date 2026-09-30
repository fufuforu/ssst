"""LOCUSGS_OBJECT_LOCUS_V1: scene-conditioned object states over canonical anchors."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.canonical_recon_models import LocusGSRecon, _full_supervision
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.object_locus_v1_controller import (
    ID_DIM, NUM_QUERIES, NUM_REGION_CHANNELS, NUM_THING,
    ObjectLocusV1Controller, scene_normalization,
)


def inherit_gaussian_ownership(anchor_assignment, children_per_anchor=64):
    if anchor_assignment.ndim != 3 or anchor_assignment.shape[1:] != (1024, NUM_REGION_CHANNELS):
        raise ValueError("anchor_assignment must be [B,1024,103]")
    batch = anchor_assignment.shape[0]
    return (anchor_assignment[:, :, None, :]
            .expand(batch, 1024, int(children_per_anchor), NUM_REGION_CHANNELS)
            .reshape(batch, 1024 * int(children_per_anchor), NUM_REGION_CHANNELS))


class LocusGSObjectLocusV1Recon(LocusGSRecon):
    architecture_name = "LOCUSGS_OBJECT_LOCUS_V1_1"
    reconstruction_only = False
    state_layers = (6, 8, 10, 12)

    def __init__(self, opt):
        super().__init__(opt)
        self.reconstruction_only = False
        if tuple(int(x) for x in opt.instance_state_layers) != self.state_layers:
            raise ValueError("Object-Locus V1 state layers are fixed to (6, 8, 10, 12)")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(getattr(opt, "object_locus_init_seed", 31415)))
            self.object_locus = ObjectLocusV1Controller(int(opt.enc_embed_dim))
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
            anchor_embedding = self.object_locus.encode_token(tokens, mu, radii, ell)
            if layer == 6:
                q, c, s, seed_indices, neighborhoods, init_weights = self.object_locus.initialize_states(
                    anchor_embedding, tokens, mu, origin, ell
                )
                state["seed_neighbors"] = neighborhoods
                state["seed_pool_weights"] = init_weights
            result = self.object_locus.forward_registered_layer(
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
        A_anchor = final["anchor_assignment"]
        A_g = inherit_gaussian_ownership(A_anchor, children)
        identity = self.object_locus.identity_features(final, gaussians)
        rendered = self.gs.render_feature_channels(
            gaussians, torch.cat((A_g, identity), dim=-1), decoder.cam_view, decoder.intrinsics
        )
        mass = rendered["images_pred"][:, :, :NUM_REGION_CHANNELS]
        identity_render = rendered["images_pred"][:, :, NUM_REGION_CHANNELS:]
        alpha = rendered["alphas_pred"]
        class_logits19 = final["thing_logits19"]
        p_class = torch.softmax(class_logits19, dim=-1)
        semantic = mass.new_zeros((batch, mass.shape[1], 20, *mass.shape[-2:]))
        semantic[:, :, 0] = mass[:, :, 100]
        semantic[:, :, 1] = mass[:, :, 101]
        thing_mass = mass[:, :, :NUM_THING]
        for class_index in range(18):
            semantic[:, :, class_index + 2] = (
                thing_mass * p_class[:, :, class_index].reshape(batch, 1, NUM_THING, 1, 1)
            ).sum(2)
        pixel_void = (1.0 - alpha) + mass[:, :, 102:103] + (
            thing_mass * p_class[:, :, 18].reshape(batch, 1, NUM_THING, 1, 1)
        ).sum(2, keepdim=True)
        logits21 = final["thing_class_logits"]
        return {
            "assignment": A_g,
            "anchor_assignment": A_anchor,
            "region_mass": mass,
            "semantic_scores": semantic,
            "pixel_void_mass": pixel_void,
            "identity_render": identity_render,
            "alpha": alpha,
            "p_class": p_class,
            "thing_class_logits": logits21,
        }

    def forward_object_locus(self, model_input, *, render_decoder_input=None,
                             context_decoder=None, coupled=False, step=None):
        if coupled:
            raise RuntimeError("Object-Locus V1 has no legacy beta coupling; beta is fixed at zero.")
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

    def step_loss(self, batch, *, step, phase="train", coupled=False):
        del phase
        if coupled:
            raise RuntimeError("Object-Locus V1 has no legacy beta coupling; beta is fixed at zero.")
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
        from tokengs.models.object_locus_v1_loss import object_locus_v1_losses
        understanding, loss_metrics = object_locus_v1_losses(
            prediction, batch, self.opt
        )
        from scripts.object_locus_v1_runtime import understanding_weight
        weight = understanding_weight(int(step))
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
