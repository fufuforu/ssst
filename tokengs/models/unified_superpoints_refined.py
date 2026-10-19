"""Region-preserving query-capacity variant of the original Unified model."""
import torch

from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import labelled_voxel_partition

# Initial scan candidate. The scan selects and records the execution constant
# before timing/training; no learned weights or shared cloud lifting change.
QUERY_VOXEL = 0.08


def refine_cloud(cloud, query_voxel, max_superpoints):
    points, regions = cloud['points'], cloud['point_superpoint_ids']
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
    assert regions.shape == (len(points),) and regions.dtype == torch.long
    assert torch.isfinite(points).all() and (regions >= 0).all()
    assert torch.as_tensor(cloud['scale']).shape == ()
    assert query_voxel > 0
    qids, qvox = labelled_voxel_partition(points, regions, cloud['scale'], query_voxel, max_superpoints)
    assert qids.shape == regions.shape and qids.dtype == torch.long
    q = int(qids.max()) + 1
    assert 1 <= q <= max_superpoints
    # Every refined cell has exactly one original region, without averaging IDs.
    pairs = torch.unique(torch.stack((qids, regions), dim=1), dim=0)
    assert pairs.shape == (q, 2), 'Refined query crosses original SAM partition'
    return dict(cloud, region_superpoint_ids=regions, point_superpoint_ids=qids,
                query_voxel_size=qvox, refined_query_count=q)


class LocusGSRefinedSuperpoints(LocusGSUnifiedQueryV1):
    architecture_name = 'OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_REFINED_V1'

    def encode_context(self, rgb):
        omega, cloud, memory = super().encode_context(rgb)
        cloud = refine_cloud(cloud, QUERY_VOXEL, self.config.max_superpoints)
        assert cloud['point_superpoint_ids'].shape == cloud['region_superpoint_ids'].shape
        return omega, cloud, memory
