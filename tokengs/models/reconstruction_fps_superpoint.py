"""Fixed-budget point positions plus persistent slots and superpoint content."""
import torch
from torch import nn


@torch.no_grad()
def deterministic_fps(points, count):
    """Global XYZ FPS; deterministic centroid-farthest start, no label quotas.

    Select distinct input indices until exhausted, then repeat their FPS order.
    Coordinates are never jittered or replaced with superpoint centers.
    """
    if points.ndim != 2 or points.shape[-1] != 3 or not len(points) or not torch.isfinite(points).all():
        raise ValueError('FPS requires finite nonempty [M,3] points')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Positive integer FPS count required')
    xyz = points.detach().float()
    distance = torch.full((len(xyz),), float('inf'), device=xyz.device)
    selected = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    current = (xyz - xyz.mean(0)).square().sum(-1).argmax()
    indices = torch.empty(min(count, len(xyz)), dtype=torch.long, device=xyz.device)
    for i in range(len(indices)):
        indices[i] = current
        selected[current] = True
        distance = torch.minimum(distance, (xyz - xyz[current]).square().sum(-1))
        current = distance.masked_fill(selected, -1).argmax()
    return indices.repeat((count + len(indices) - 1) // len(indices))[:count]


class FPSWithSuperpointFeatures(nn.Module):
    def __init__(self, count=1024, dim=1024, std=.02):
        super().__init__()
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('Positive integer anchor count required')
        self.count = count
        self.learned_embedding = nn.Parameter(torch.empty(count, dim))
        nn.init.normal_(self.learned_embedding, std=std)

    def forward(self, cloud, seed, projected_features):
        points = cloud['points']
        ids = seed['point_superpoint_ids']  # compact IDs matching seed query order
        if ids.shape != (len(points),) or ids.dtype != torch.long:
            raise ValueError('Point labels must match the sampled cloud')
        if projected_features.ndim != 3 or projected_features.shape[0] != 1 or projected_features.shape[-1] != self.learned_embedding.shape[-1]:
            raise ValueError('Expected single-scene projected superpoint features')
        if (ids < 0).any() or (ids >= projected_features.shape[1]).any():
            raise ValueError('Invalid point-to-superpoint mapping')
        index = deterministic_fps(points, self.count)
        parent = ids[index]
        covered = parent.unique().numel()
        return dict(tokens=self.learned_embedding[None] + projected_features[:, parent],
            centers=points[index].detach()[None],
            # Retain existing superpoint scale; only center changes to sampled XYZ.
            radii=seed['radii'][:, parent],
            anchor_to_superpoint=parent,
            anchor_slot_ids=torch.arange(self.count, device=points.device),
            anchor_point_indices=index,
            num_sampled_superpoints=covered,
            sampled_superpoint_coverage=covered / projected_features.shape[1])
