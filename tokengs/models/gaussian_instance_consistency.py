"""Direct, geometry-detached Gaussian identity supervision on original Unified.

No new parameters or inference changes. The extra training readout recomputes
the existing child/8D heads with detached local geometry, so only this loss's
explicit geometry path is cut. Original RGB/mask/rendered-instance losses keep
their original gradients, including geometry gradients.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.object_locus_panoptic_v1_lift import RendererTranspose
from tokengs.models.unified_instance_grounding import InstanceGroundedUnifiedCore, LocusGSInstanceGroundedUnified


@dataclass(frozen=True)
class GaussianConsistencyRecipe:
    loss_weight: float = .01  # Inside the original understanding weight u.
    purity_threshold: float = .7
    minimum_contribution_mass: float = .001  # Sum of valid-pixel compositing weights.
    pull_cosine_margin: float = .1
    push_cosine_margin: float = .5
    negative_weight: float = 1.
    negative_neighbors: int = 8
    negative_radius_scene_fraction: float = .1
    neighbor_chunk: int = 256
    normalization_epsilon: float = 1e-6
    geometry_gradients: bool = False
    supervised_views: int = 2


RECIPE = GaussianConsistencyRecipe()


@torch.no_grad()
def contribution_instance_targets(gaussians, decoder, renderer, targets, recipe=RECIPE):
    """Detached thing labels; denominator also includes valid stuff contributions."""
    if gaussians.ndim != 3 or gaussians.shape[0] != 1:
        raise ValueError('Direct consistency supports one scene per rank')
    masks = targets['gt_pixel_masks'][0]
    valid = targets['valid_pixels']
    if valid.ndim != 4 or valid.shape[:2] != (1, recipe.supervised_views):
        raise ValueError('Exactly two context GT views required')
    if masks.shape != (len(targets['gt_instance_ids'][0]), *valid.shape[1:]):
        raise ValueError('GT instance masks and valid-pixel grids differ')
    if decoder.cam_view.shape[:2] != valid.shape[:2] or decoder.intrinsics.shape[:2] != valid.shape[:2]:
        raise ValueError('Context cameras and GT views differ')
    if masks.dtype != torch.bool or valid.dtype != torch.bool:
        raise ValueError('Boolean masks and valid-pixel grids required')
    g = gaussians.detach()
    mass = g.new_zeros(g.shape[:2])
    purity = mass.clone()
    labels = torch.full(g.shape[:2], -1, dtype=torch.long, device=g.device)
    if masks.shape[0]:
        mass = RendererTranspose.apply(valid[:, :, None].float(), g, decoder.cam_view.detach(),
                                       decoder.intrinsics.detach(), renderer)[..., 0]
        evidence = RendererTranspose.apply((masks & valid[0][None]).permute(1, 0, 2, 3)[None].float(),
                                           g, decoder.cam_view.detach(), decoder.intrinsics.detach(), renderer)
        if not torch.isfinite(mass).all() or not torch.isfinite(evidence).all():
            raise FloatingPointError('Nonfinite Gaussian ownership contributions')
        if (mass < -1e-6).any() or (evidence < -1e-6).any():
            raise ValueError('Negative compositing contributions')
        top, labels = evidence.max(-1)
        purity = (top / mass.clamp_min(recipe.normalization_epsilon)).clamp(0, 1)
        labels = labels.masked_fill((mass < recipe.minimum_contribution_mass) |
                                    (purity < recipe.purity_threshold), -1)
    return dict(labels=labels.detach(), purity=purity.detach(), mass=mass.detach())


@torch.no_grad()
def local_different_instance_neighbors(xyz, labels, scene_scale, recipe=RECIPE):
    """Nearest different-GT neighbors within a scene-relative radius, no fallback.

    xyz/labels already contain only reliable thing Gaussians. Search other-GT
    candidates directly, so dense same-GT Children cannot fill all k slots.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or labels.shape != xyz.shape[:1]:
        raise ValueError('Expected reliable [N,3] positions and [N] labels')
    if not torch.isfinite(xyz).all() or not torch.isfinite(scene_scale).all() or scene_scale.numel() != 1 or float(scene_scale) <= 0:
        raise ValueError('Finite coordinates and positive scene scale required')
    k = min(recipe.negative_neighbors, len(xyz))
    if not len(xyz):
        return labels.new_empty((0, 0)), torch.empty((0, 0), dtype=torch.bool, device=xyz.device)
    radius = recipe.negative_radius_scene_fraction
    points = xyz.detach().float() / scene_scale.detach().float()
    indices = []; eligible = []
    for start in range(0, len(points), recipe.neighbor_chunk):
        stop = min(start + recipe.neighbor_chunk, len(points))
        distance = torch.cdist(points[start:stop], points)
        distance.masked_fill_(labels[start:stop, None] == labels[None], float('inf'))
        value, index = distance.topk(k, largest=False)
        indices.append(index)
        eligible.append(torch.isfinite(value) & (value <= radius))
    return torch.cat(indices), torch.cat(eligible)


def _sum_groups(values, index, count):
    return values.new_zeros((count, *values.shape[1:])).index_add(0, index, values)


def _balanced_mean(values, weights, group, group_instance, instance_count):
    """Gaussian weighted mean -> equal parent means -> equal GT means."""
    group_count = len(group_instance)
    mass = _sum_groups(weights, group, group_count)
    mean = _sum_groups(values * weights, group, group_count) / mass.clamp_min(1e-12)
    active = mass > 0
    count = _sum_groups(active.to(values.dtype), group_instance, instance_count)
    means = _sum_groups(mean * active, group_instance, instance_count) / count.clamp_min(1)
    return means[count > 0].mean() if (count > 0).any() else values.sum() * 0.


def direct_gaussian_consistency_loss(features, xyz, parent, assignment, scene_scale, recipe=RECIPE):
    """[1,N,8] features, GT-balanced cross-parent pull and local other-GT push."""
    if features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != 8:
        raise ValueError('Expected one-scene [1,N,8] instance features')
    n = features.shape[1]
    if xyz.shape != (1, n, 3) or parent.shape != (n,) or parent.dtype != torch.long or (parent < 0).any():
        raise ValueError('Gaussian geometry/parent identity differs')
    if any(assignment[key].shape != (1, n) for key in ('labels', 'purity', 'mass')):
        raise ValueError('Assignment shape differs')
    if assignment['labels'].dtype != torch.long:
        raise ValueError('Integer instance labels required')
    with torch.autocast(device_type=features.device.type, enabled=False):
        f = features.float()
        if not torch.isfinite(f).all():
            raise FloatingPointError('Nonfinite direct instance features')
        zero = f.sum() * 0.
        labels = assignment['labels'][0].detach()
        purity = assignment['purity'][0].detach()
        mass = assignment['mass'][0].detach()
        if not torch.isfinite(purity).all() or not torch.isfinite(mass).all() or (purity < 0).any() or (purity > 1).any() or (mass < 0).any():
            raise ValueError('Invalid ownership confidence')
        reliable = (labels >= 0) & (purity >= recipe.purity_threshold) & (mass >= recipe.minimum_contribution_mass)
        metrics = dict(direct_gaussian_total=n, direct_gaussian_reliable_count=int(reliable.sum()),
                       direct_gaussian_reliable_fraction=float(reliable.float().mean()) if n else 0.,
                       direct_reliable_gt_count=0, direct_positive_gt_count=0, direct_positive_parent_count=0,
                       direct_positive_degenerate_parent_count=0,
                       direct_positive_gaussians=0, direct_negative_gt_count=0, direct_negative_anchors=0,
                       direct_negative_edges=0, direct_positive_distance=zero.detach(), direct_negative_distance=zero.detach(),
                       direct_negative_violation_fraction=zero.detach(), direct_reliable_purity_mean=zero.detach(),
                       direct_reliable_mass_mean=zero.detach())
        pull = push = zero
        if reliable.any():
            vectors = F.normalize(f[0, reliable], dim=-1, eps=recipe.normalization_epsilon)
            confidence = purity[reliable]
            gt = labels[reliable]
            owners = parent[reliable]
            keys, group = torch.unique(torch.stack((gt, owners), -1), dim=0, sorted=True, return_inverse=True)
            _, group_instance = keys[:, 0].unique(sorted=True, return_inverse=True)
            instances = int(group_instance.max()) + 1
            groups = len(keys)
            parent_mass = _sum_groups(confidence, group, groups)
            parent_proto = F.normalize(_sum_groups(vectors * confidence[:, None], group, groups) /
                                      parent_mass[:, None].clamp_min(1e-12), dim=-1, eps=recipe.normalization_epsilon)
            parent_count = torch.bincount(group_instance, minlength=instances)
            total_proto = _sum_groups(parent_proto, group_instance, instances)
            # Equal weight per OTHER parent, never own parent's Children.
            other_mean = (total_proto[group_instance] - parent_proto) / (parent_count[group_instance] - 1).clamp_min(1)[:, None]
            other_proto = F.normalize(other_mean, dim=-1, eps=recipe.normalization_epsilon)
            multi_parent = parent_count[group_instance] >= 2
            usable_parent = multi_parent & (other_mean.detach().norm(dim=-1) > recipe.normalization_epsilon)
            positive = usable_parent[group]
            positive_distance = (1 - (vectors * other_proto[group]).sum(-1)).clamp(0, 2)
            positive_weights = confidence * positive
            pull = _balanced_mean((positive_distance - recipe.pull_cosine_margin).relu(), positive_weights,
                                  group, group_instance, instances)
            neighbors, within = local_different_instance_neighbors(xyz[0, reliable].detach(), gt, scene_scale.detach(), recipe)
            distances = (1 - (vectors[:, None] * vectors[neighbors]).sum(-1)).clamp(0, 2)
            neighbor_weights = confidence[neighbors] * within
            neighbor_mass = neighbor_weights.sum(-1)
            has_negative = neighbor_mass > 0
            raw_push = ((recipe.push_cosine_margin - distances).relu() * neighbor_weights).sum(-1) / neighbor_mass.clamp_min(1e-12)
            negative_weights = confidence * has_negative
            push = _balanced_mean(raw_push, negative_weights, group, group_instance, instances)
            metrics.update(direct_reliable_gt_count=instances,
                direct_positive_gt_count=int(group_instance[usable_parent].unique().numel()),
                direct_positive_parent_count=int(usable_parent.sum()),
                direct_positive_degenerate_parent_count=int((multi_parent & ~usable_parent).sum()),
                direct_positive_gaussians=int(positive.sum()),
                direct_negative_gt_count=int(gt[has_negative].unique().numel()),
                direct_negative_anchors=int(has_negative.sum()), direct_negative_edges=int(within.sum()),
                direct_positive_distance=_balanced_mean(positive_distance, positive_weights, group, group_instance, instances).detach(),
                direct_negative_distance=_balanced_mean((distances * neighbor_weights).sum(-1) / neighbor_mass.clamp_min(1e-12),
                    negative_weights, group, group_instance, instances).detach(),
                direct_negative_violation_fraction=_balanced_mean(((distances < recipe.push_cosine_margin) * neighbor_weights).sum(-1) /
                    neighbor_mass.clamp_min(1e-12), negative_weights, group, group_instance, instances).detach(),
                direct_reliable_purity_mean=confidence.mean().detach(), direct_reliable_mass_mean=mass[reliable].mean().detach())
        metrics.update(loss_gaussian_direct_pull=pull.detach(), loss_gaussian_direct_push=push.detach())
        # Retain a zero connection even when no trustworthy supervision exists.
        return pull + recipe.negative_weight * push + zero, metrics


class GaussianConsistencyCore(InstanceGroundedUnifiedCore):
    def direct_instance_features(self, prediction):
        """Same 8D values as native readout, with explicit geometry detached."""
        state = prediction['states'][-1]
        semantic = self.semantic_adapter(prediction['unified_query'])
        children = self.child_index.num_embeddings
        parent = semantic.repeat_interleave(children, 1)
        index = self.child_index.weight.repeat(semantic.shape[1], 1)[None].expand(len(semantic), -1, -1)
        centers = state['mu'].detach().repeat_interleave(children, 1)
        radii = state['radii'].detach().repeat_interleave(children, 1)[..., None]
        local = (prediction['gaussians'][..., :3].detach() - centers) / radii
        return self.instance_head(self.child_feature(torch.cat((parent, index, local), -1)))


class LocusGSGaussianInstanceConsistency(LocusGSInstanceGroundedUnified):
    architecture_name = 'OMEGA_UNIFIED_GAUSSIAN_INSTANCE_CONSISTENCY_V1'
    core_type = GaussianConsistencyCore

    @property
    def extra_checkpoint_metadata(self):
        return dict(super().extra_checkpoint_metadata, gaussian_consistency_recipe=asdict(RECIPE))

    def direct_consistency_objective(self, prediction, targets):
        decoder = ModelInputDecoder(cam_view=torch.linalg.inv(prediction['predicted_context_c2w']).transpose(-1, -2),
                                    intrinsics=prediction['predicted_context_intrinsics'])
        assignment = contribution_instance_targets(prediction['gaussians'], decoder, self.gs, targets)
        features = self.core.direct_instance_features(prediction)
        loss, metrics = direct_gaussian_consistency_loss(features, prediction['gaussians'][..., :3],
            prediction['gaussian_parent_query'], assignment, prediction['cloud']['scale'])
        return loss, metrics

    def step_loss(self, batch, *, step=0, phase='train', understanding_weight=1.):
        output, metrics = super().step_loss(batch, step=step, phase=phase, understanding_weight=understanding_weight)
        direct, terms = self.direct_consistency_objective(output['prediction'], output['prediction']['final_targets'])
        previous = metrics['loss_understanding']
        metrics.update(terms, loss_understanding_before_gaussian_direct=previous.detach(),
            loss_gaussian_direct=direct.detach(), gaussian_direct_effective_weight=understanding_weight * RECIPE.loss_weight,
            loss_gaussian_direct_weighted=(understanding_weight * RECIPE.loss_weight * direct).detach())
        metrics['loss_understanding'] = previous + RECIPE.loss_weight * direct
        metrics['loss'] = metrics['loss_recon'] + understanding_weight * metrics['loss_understanding']
        return output, metrics
