"""Independent reconstruction/understanding seeds on the same 3D superpoints.

Preparatory interface only: no legacy decoder is silently replaced or resized.
The caller supplies features already attached to the common fused 3D cloud.
"""
import torch
from torch import nn
from tokengs.models.unified_superpoints import SuperpointQueryInitializer


class DualSuperpointInitialization(nn.Module):
    def __init__(self,reconstruction_config,understanding_config):
        super().__init__()
        if reconstruction_config.max_superpoints!=understanding_config.max_superpoints:raise ValueError('Both streams must preserve the same superpoint budget')
        self.reconstruction=SuperpointQueryInitializer(reconstruction_config)
        self.understanding=SuperpointQueryInitializer(understanding_config)

    def forward(self,cloud,reconstruction_features,understanding_features):
        if 'point_superpoint_ids' not in cloud:raise ValueError('Explicit common 3D superpoint IDs required')
        count=len(cloud['points'])
        for features,module in ((reconstruction_features,self.reconstruction),(understanding_features,self.understanding)):
            if features.shape!=(count,module.config.dim):raise ValueError('Features must match the fused 3D point order and branch width')
        reconstruction=self.reconstruction(dict(cloud,features=reconstruction_features))
        understanding=self.understanding(dict(cloud,features=understanding_features))
        if not torch.equal(reconstruction['point_superpoint_ids'],understanding['point_superpoint_ids']):raise ValueError('Branch superpoint identities differ')
        return dict(reconstruction=reconstruction,understanding=understanding)
