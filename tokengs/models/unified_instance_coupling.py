"""Controlled geometry->instance evidence and instance->RGB boundary coupling.

The lifting adjoint deliberately freezes rasterization weights: this path trains
image features, not geometry through a second derivative of the CUDA renderer.
Original rendered mask/instance losses still differentiate Gaussian geometry.
Boundary weights come from rendered embeddings (an adaptation of InstanceSplat,
whose boundary features are pixel-aligned before rasterization).
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.object_locus_posefree_geometry import camera_vectors
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.unified_instance_grounding import InstanceGroundedUnifiedCore, LocusGSInstanceGroundedUnified


@dataclass(frozen=True)
class CouplingRecipe:
    lift: bool = False
    boundary: bool = False
    lift_residual_scale: float = .1
    lift_geometry_gradients: bool = False
    boundary_weight: float = .02
    boundary_threshold: float = .15
    boundary_temperature: float = .05
    boundary_window: int = 5
    charbonnier_epsilon: float = .001
    boundary_source_detached: bool = True
    boundary_source: str = 'rendered_context_instance_embeddings'
    boundary_ramp_reference_understanding_weight: float = .05


def instance_boundary_weights(features, recipe=CouplingRecipe()):
    """Detached max 4-neighbor cosine distance, dilation and soft threshold."""
    if features.ndim != 5:
        raise ValueError('Expected [B,V,C,H,W] features')
    with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
        z = F.normalize(features.detach().float(), dim=2, eps=1e-6)
        if not torch.isfinite(z).all():
            raise FloatingPointError('Nonfinite boundary source')
        horizontal = (1 - (z[..., 1:] * z[..., :-1]).sum(2)).clamp_min(0)
        vertical = (1 - (z[..., 1:, :] * z[..., :-1, :]).sum(2)).clamp_min(0)
        response = torch.maximum(F.pad(horizontal, (0, 1)), F.pad(horizontal, (1, 0)))
        response = torch.maximum(response, F.pad(vertical, (0, 0, 0, 1)))
        response = torch.maximum(response, F.pad(vertical, (0, 0, 1, 0)))
        response = F.max_pool2d(response.flatten(0, 1)[:, None], recipe.boundary_window,
                              stride=1, padding=recipe.boundary_window // 2)
        return ((response - recipe.boundary_threshold) / recipe.boundary_temperature).sigmoid().reshape(
            *features.shape[:2], 1, *features.shape[-2:])


def boundary_rgb_loss(rgb, target, features, recipe=CouplingRecipe()):
    if rgb.shape != target.shape or rgb.shape[:2] != features.shape[:2] or rgb.shape[-2:] != features.shape[-2:]:
        raise ValueError('Boundary RGB/features must share cameras and pixel grid')
    weights = instance_boundary_weights(features, recipe)
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        error = ((rgb.float() - target.float()).square() + recipe.charbonnier_epsilon ** 2).sqrt().mean(2, keepdim=True)
        loss = ((weights * error).sum((-1, -2, -3)) / weights.sum((-1, -2, -3)).clamp_min(1e-6)).mean()
    return loss, weights


class LiftedInstanceCore(InstanceGroundedUnifiedCore):
    def __init__(self, opt, config):
        super().__init__(opt, config)
        self.image_size = tuple(opt.img_size)
        with torch.random.fork_rng(devices=[]):
            self.lift_residual = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim))

    def forward(self, cloud, memory, superpoint_ids=None):
        result = super().forward(cloud, memory['features'], superpoint_ids)
        grid = memory['grid']
        pixels = F.interpolate(grid.flatten(0, 1), size=self.image_size, mode='bilinear', align_corners=False)
        pixels = pixels.reshape(*grid.shape[:2], grid.shape[2], *self.image_size)
        evidence, gate, mass = lift_features(pixels, result['gaussians'], memory['decoder'], memory['renderer'])
        residual = .1 * gate * self.lift_residual(evidence)
        # Unseen Gaussians retain exactly their original child features.
        children = result['gaussian_features'] + residual
        semantic = self.semantic_adapter(result['unified_query'])
        descriptors = torch.cat((self.query_mask(semantic), self.stuff_feature(semantic.mean(1)).reshape(
            len(semantic), 2, self.config.dim)), 1)
        logits = children @ descriptors.transpose(1, 2) / self.config.dim ** .5
        result.update(gaussian_mask_logits=logits, gaussian_membership=logits.sigmoid(),
                      gaussian_features=children, gaussian_instance_features=self.instance_head(children),
                      lift_mass=mass.detach(), lift_gate=gate.detach(), lift_residual=residual,
                      gaussian_features_before_lift=result['gaussian_features'])
        return result

    def parameter_groups(self):
        groups = super().parameter_groups()
        parameters = list(self.lift_residual.parameters()); ids = {id(p) for p in parameters}
        groups['shared'] = [p for p in groups['shared'] if id(p) not in ids]
        groups['understanding'] += parameters
        return groups


class LocusGSInstanceBoundary(LocusGSInstanceGroundedUnified):
    architecture_name = 'OMEGA_UNIFIED_INSTANCE_BOUNDARY_V1'
    coupling_recipe = CouplingRecipe(boundary=True)

    @property
    def extra_checkpoint_metadata(self):
        return dict(super().extra_checkpoint_metadata, instance_coupling_recipe=asdict(self.coupling_recipe))

    @torch.no_grad()
    def probe_diagnostics(self, prediction, batch):
        """Fixed-probe evidence; GT is used only after generation for evaluation."""
        p = prediction
        features = F.normalize(p['rendered_instance_features'].float(), dim=2, eps=1e-6)
        masks = p['final_targets']['gt_pixel_masks'][0]
        valid = p['final_targets']['valid_pixels'][0]
        prototypes = []
        compactness = []
        shuffled_compactness = []
        pixel_features = features[0].permute(0, 2, 3, 1)
        # Fixed cyclic permutation of feature-to-GT correspondence; preserves
        # feature distribution and mask sizes without consuming training RNG.
        shuffled_features = pixel_features.clone()
        if valid.any():
            selected = pixel_features[valid]
            shuffled_features[valid] = selected.roll(max(1, len(selected) // 3), dims=0)
        for mask in masks:
            mask = mask & valid
            if not mask.any():
                continue
            values = pixel_features[mask]
            proto = F.normalize(values.mean(0), dim=0, eps=1e-6)
            compactness.append((values @ proto).mean())
            shuffled = shuffled_features[mask]
            shuffled_proto = F.normalize(shuffled.mean(0), dim=0, eps=1e-6)
            shuffled_compactness.append((shuffled @ shuffled_proto).mean())
            prototypes.append(proto)
        result = dict(instance_feature_compactness=float(torch.stack(compactness).mean()) if compactness else 0.,
                      instance_feature_compactness_shuffled=float(torch.stack(shuffled_compactness).mean()) if shuffled_compactness else 0.,
                      instance_feature_evaluated_gt=len(prototypes))
        if len(prototypes) > 1:
            proto = torch.stack(prototypes); similarities = proto @ proto.T
            result['instance_feature_between_gt_cosine'] = float(similarities[torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)].mean())
        # GT boundary and its complement are independent of learned weights.
        labels = batch['instance_label_all'][:, :2]
        semantic = batch['semantic_label_all'][:, :2]
        thing = (semantic >= 2) & (semantic < 20) & (labels > 0)
        dx = (labels[..., 1:] != labels[..., :-1]) & (thing[..., 1:] | thing[..., :-1])
        dy = (labels[..., 1:, :] != labels[..., :-1, :]) & (thing[..., 1:, :] | thing[..., :-1, :])
        edge = F.pad(dx, (0, 1)) | F.pad(dx, (1, 0)) | F.pad(dy, (0, 0, 0, 1)) | F.pad(dy, (0, 0, 1, 0))
        edge = F.max_pool2d(edge.flatten(0, 1).float()[:, None], 5, stride=1, padding=2).reshape_as(labels).bool()
        error = (p['render']['images_pred'] - batch['images_input']).float().square().mean(2)
        known = (semantic >= 0) & (semantic < 20)
        weights = instance_boundary_weights(p['rendered_instance_features'])[:, :, 0]
        for name, region in (('boundary', edge & known), ('interior', ~edge & thing & known), ('background', ~edge & ~thing & known)):
            result[f'context_gt_{name}_pixels'] = int(region.sum())
            result[f'context_gt_{name}_rgb_mse'] = float(error[region].mean()) if region.any() else 0.
            result[f'predicted_weight_at_gt_{name}'] = float(weights[region].mean()) if region.any() else 0.
        return result

    def step_loss(self, batch, *, step=0, phase='train', understanding_weight=1.):
        output, metrics = super().step_loss(batch, step=step, phase=phase, understanding_weight=understanding_weight)
        p = output['prediction']; recipe = self.coupling_recipe
        if recipe.lift:
            metrics.update(lift_visible_fraction=(p['lift_mass'] > 0).float().mean().detach(),
                           lift_gate_mean=p['lift_gate'].mean(),
                           lift_residual_relative_norm=(p['lift_residual'].float().norm() /
                               p['gaussian_features_before_lift'].float().norm().clamp_min(1e-6)).detach())
        if recipe.boundary:
            loss, weights = boundary_rgb_loss(p['render']['images_pred'], batch['images_input'],
                                             p['rendered_instance_features'], recipe)
            # Same 20-update ramp as the short-test understanding objective,
            # but .02 is an independent reconstruction coefficient, not .05*.02.
            coefficient = recipe.boundary_weight * min(1., understanding_weight /
                recipe.boundary_ramp_reference_understanding_weight)
            metrics.update(loss_recon_base=metrics['loss_recon'].detach(), loss_boundary_rgb=loss.detach(),
                           boundary_rgb_effective_weight=coefficient, boundary_weight_mean=weights.mean(),
                           loss_boundary_rgb_weighted=(coefficient * loss).detach())
            metrics['loss_recon'] = metrics['loss_recon'] + coefficient * loss
            metrics['loss'] = metrics['loss_recon'] + understanding_weight * metrics['loss_understanding']
        return output, metrics


class LocusGSInstanceLift(LocusGSInstanceBoundary):
    architecture_name = 'OMEGA_UNIFIED_INSTANCE_LIFT_V1'
    core_type = LiftedInstanceCore
    coupling_recipe = CouplingRecipe(lift=True)

    def encode_context(self, rgb):
        omega, cloud, features = super().encode_context(rgb)
        # The same projected GFM grid used by the shared decoder, no extra teacher.
        side = 32
        grid = features.reshape(1, 2, side, side, self.config.dim).permute(0, 1, 4, 2, 3)
        decoder = ModelInputDecoder(cam_view=torch.linalg.inv(omega['c2w']).transpose(-1, -2),
                                    intrinsics=camera_vectors(omega['k']))
        return omega, cloud, dict(features=features, grid=grid, decoder=decoder, renderer=self.gs)


class LocusGSInstanceBidirectional(LocusGSInstanceLift):
    architecture_name = 'OMEGA_UNIFIED_INSTANCE_BIDIRECTIONAL_V1'
    coupling_recipe = CouplingRecipe(lift=True, boundary=True)
