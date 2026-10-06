"""Region-conditioned classification for the isolated paired experiment."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.object_locus_v3_set import alpha_normalize_membership


def pool_region_features(weights: torch.Tensor, feature_grid: torch.Tensor):
    """Jointly pool two context feature maps using rendered alpha-composited mass."""
    if weights.ndim != 5 or feature_grid.ndim != 5:
        raise ValueError("weights and feature_grid must be rank-5 tensors")
    if weights.shape[:2] != feature_grid.shape[:2] or weights.shape[-2:] != feature_grid.shape[-2:]:
        raise ValueError("context views and image dimensions must match")
    if weights.shape[2] != 100 or feature_grid.shape[2] != 256:
        raise ValueError("expected 100 thing channels and 256 feature channels")
    w = weights.float()
    f = feature_grid.float()
    denominator = w.sum(dim=(1, 3, 4))
    numerator = torch.einsum("bvjhw,bvdhw->bjd", w, f)
    pooled = numerator / denominator.clamp_min(1e-6).unsqueeze(-1)
    pooled = torch.where((denominator > 1e-6).unsqueeze(-1), pooled, torch.zeros_like(pooled))
    return pooled, denominator


def classify_with_region(panoptic, q: torch.Tensor, pooled: torch.Tensor, *, enabled: bool = True):
    """Use region features only as an additive input to the final thing classifier."""
    if not enabled:
        return panoptic.classify(q), q
    if pooled.shape != (q.shape[0], 100, 256):
        raise ValueError(f"pooled region feature shape mismatch: {tuple(pooled.shape)}")
    z_norm = F.layer_norm(pooled, (256,), eps=1e-5)
    delta = panoptic.region_class_proj(z_norm)
    q_class = torch.cat((q[:, :100] + delta, q[:, 100:102]), dim=1)
    return panoptic.classify(q_class), q_class


def add_region_class_projection(panoptic: nn.Module, seed: int = 31417):
    """Register the sole new parameter without advancing process RNG state."""
    if hasattr(panoptic, "region_class_proj"):
        raise RuntimeError("region_class_proj is already installed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        projection = nn.Linear(256, 256, bias=False)
        nn.init.zeros_(projection.weight)
    device = next(panoptic.parameters()).device
    projection = projection.to(device=device, dtype=torch.float32)
    panoptic.add_module("region_class_proj", projection)


class RegionClassObjectLocusPanopticV1Recon(LocusGSObjectLocusPanopticV1Recon):
    """Baseline forward path with region-conditioned final classification only."""

    def _readout(self, final, gaussians, fm, read_context_decoder, render_decoder_input):
        b = gaussians.shape[0]
        feature_grid = F.interpolate(
            fm.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False
        ).reshape(b, 2, 256, 256, 256)
        evidence, gate, mass = lift_features(feature_grid, gaussians, read_context_decoder, self.gs)
        child, residual = self.panoptic.gaussian_child_features(
            final["anchor_embedding"], final["f_anchor"], gaussians, final["mu"], final["radii"]
        )
        features = gate * evidence + (1 - gate) * child + 0.1 * torch.tanh(self.panoptic.W_res(child))
        mq = self.understanding.mask_embedder(final["q"])
        logits = features @ mq.transpose(1, 2)
        membership = logits.sigmoid()

        rendered = self.gs.render_feature_channels(
            gaussians, membership, render_decoder_input.cam_view, render_decoder_input.intrinsics
        )
        alpha = rendered["alphas_pred"]
        region = alpha_normalize_membership(rendered["images_pred"], alpha)

        same_context_views = (
            render_decoder_input.cam_view.shape == read_context_decoder.cam_view.shape
            and render_decoder_input.intrinsics.shape == read_context_decoder.intrinsics.shape
            and torch.equal(render_decoder_input.cam_view, read_context_decoder.cam_view)
            and torch.equal(render_decoder_input.intrinsics, read_context_decoder.intrinsics)
        )
        context_rendered = rendered if same_context_views else self.gs.render_feature_channels(
            gaussians, membership, read_context_decoder.cam_view, read_context_decoder.intrinsics
        )
        weights = context_rendered["images_pred"][:, :, :100]
        pooled, denominator = pool_region_features(weights, feature_grid)
        enabled = bool(getattr(self, "region_class_enabled", True))
        cls, q_class = classify_with_region(self.panoptic, final["q"], pooled, enabled=enabled)
        semantic = region.new_zeros((b, region.shape[1], 20, *region.shape[-2:]))
        semantic[:, :, 0:2] = region[:, :, 100:102]
        semantic[:, :, 2:20] = torch.einsum("bvqhw,bqc->bvchw", region[:, :, :100], cls["p_class"][..., :18])
        semantic = semantic / (semantic.sum(2, keepdim=True) + 1e-6)
        final.update(cls)

        # CPU scalars only: formal logging may read these values without retaining
        # feature maps or a training graph on the model between steps.
        self._last_region_stats = {
            "region_mass_mean": float(denominator.detach().mean().cpu()),
            "region_projection_norm": float(self.panoptic.region_class_proj.weight.detach().norm().cpu()),
            "classification_delta_norm": float((q_class[:, :100] - final["q"][:, :100]).detach().norm(dim=-1).mean().cpu()),
        }
        if bool(getattr(self, "capture_region_grads", False)) and torch.is_grad_enabled():
            self._last_region_grad_norms = {}
            for name, tensor in (("feature_grid", feature_grid), ("weights", weights), ("pooled", pooled)):
                tensor.register_hook(lambda grad, key=name: self._last_region_grad_norms.__setitem__(
                    key, float(grad.detach().norm().cpu())))

        return dict(**cls, assignment=membership, gaussian_membership=membership, gaussian_mask_logits=logits,
            gaussian_feature=features, child_feature_residual=residual, anchor_membership=final["anchor_membership"],
            anchor_assignment=final["anchor_membership"], anchor_mask_logits=final["anchor_mask_logits"],
            membership_mass=rendered["images_pred"], region_mass=region, pixel_membership=region,
            semantic_scores=semantic, pixel_void_mass=1-alpha, alpha=alpha,
            lifting_mass=mass, lifting_gate=gate)
