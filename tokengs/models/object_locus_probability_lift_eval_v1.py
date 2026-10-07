"""Inference-only probability-domain readout for the fixed paired evaluation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.object_locus_v3_set import alpha_normalize_membership


class LocusGSObjectLocusProbabilityLiftEvalV1(LocusGSObjectLocusPanopticV1Recon):
    """Adds no state; ``readout_mode`` is an ordinary instance attribute."""

    def _readout(self, final, gaussians, fm, read_context_decoder, render_decoder_input):
        mode = getattr(self, "readout_mode", "C")
        if mode == "C":
            return super()._readout(final, gaussians, fm, read_context_decoder, render_decoder_input)
        if mode != "P":
            raise ValueError(f"unknown readout_mode={mode!r}")

        b = gaussians.shape[0]
        feature_grid = F.interpolate(
            fm.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False
        ).reshape(b, 2, 256, 256, 256)
        mq = self.understanding.mask_embedder(final["q"])
        pixel_logits = torch.einsum("bvdhw,bqd->bvqhw", feature_grid, mq)
        pixel_prob = torch.sigmoid(pixel_logits)
        evidence, gate, mass = lift_features(pixel_prob, gaussians, read_context_decoder, self.gs)

        child, residual = self.panoptic.gaussian_child_features(
            final["anchor_embedding"], final["f_anchor"], gaussians, final["mu"], final["radii"]
        )
        z_child = child @ mq.transpose(1, 2)
        child_prob = torch.sigmoid(z_child)
        mixed = gate * evidence + (1.0 - gate) * child_prob
        delta_feature = 0.1 * torch.tanh(self.panoptic.W_res(child))
        residual_logits = delta_feature @ mq.transpose(1, 2)
        logits = torch.logit(mixed.clamp(1e-6, 1.0 - 1e-6)) + residual_logits
        fallback = (child + delta_feature) @ mq.transpose(1, 2)
        logits = torch.where(mass == 0, fallback, logits)
        membership = torch.sigmoid(logits)

        rendered = self.gs.render_feature_channels(
            gaussians, membership, render_decoder_input.cam_view, render_decoder_input.intrinsics
        )
        alpha = rendered["alphas_pred"]
        region = alpha_normalize_membership(rendered["images_pred"], alpha)
        cls = self.panoptic.classify(final["q"])
        semantic = region.new_zeros((b, region.shape[1], 20, *region.shape[-2:]))
        semantic[:, :, 0:2] = region[:, :, 100:102]
        semantic[:, :, 2:20] = torch.einsum(
            "bvqhw,bqc->bvchw", region[:, :, :100], cls["p_class"][..., :18]
        )
        semantic = semantic / (semantic.sum(2, keepdim=True) + 1e-6)
        final.update(cls)
        return dict(
            **cls, assignment=membership, gaussian_membership=membership,
            gaussian_mask_logits=logits, gaussian_feature=None,
            child_feature_residual=residual, anchor_membership=final["anchor_membership"],
            anchor_assignment=final["anchor_membership"], anchor_mask_logits=final["anchor_mask_logits"],
            membership_mass=rendered["images_pred"], region_mass=region, pixel_membership=region,
            semantic_scores=semantic, pixel_void_mass=1-alpha, alpha=alpha,
            lifting_mass=mass, lifting_gate=gate, readout_domain="probability",
        )
