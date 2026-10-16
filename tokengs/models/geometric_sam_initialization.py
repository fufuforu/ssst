"""ESAM-inspired shape-aware pooling of semantic GFM inside SAM proposals.

No ESAM dependency or pretrained weights. Proposal selection/association and
query position/scale encoding are inherited unchanged from the V3 initializer.
"""
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.gfm_dual_stream_components import SemanticSAMQueryInitializer


class GeometricSAMQueryInitializer(SemanticSAMQueryInitializer):
    pooling_description='SAM geometric-aware semantic pooling then query projection'

    def __init__(self,max_region_points=256):
        super().__init__()
        if max_region_points<1:
            raise ValueError('max_region_points must be positive')
        self.max_region_points=max_region_points
        self.local_shape=nn.Sequential(nn.Linear(3,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,64))
        self.point_weight=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        self.shape_projection=nn.Linear(64,256)

    def pool_region(self,feature_map,mask,point_map):
        # Selection and predicted geometry are fixed inputs, while semantic
        # samples and all pooling MLPs retain their gradients.
        valid=mask & torch.isfinite(point_map).all(-1)
        ids=valid.flatten().nonzero().flatten()
        if not ids.numel():
            raise ValueError('Geometric pooling requires a finite point in the SAM region')
        if ids.numel()>self.max_region_points:
            pick=torch.linspace(0,ids.numel()-1,self.max_region_points,device=ids.device).long()
            ids=ids[pick]
        xyz=point_map.reshape(-1,3)[ids].detach().float()
        centered=xyz-xyz.mean(0)
        diameter=(xyz.amax(0)-xyz.amin(0)).amax().clamp_min(1e-6)
        local=self.local_shape(centered/diameter)
        shape=local.amax(0)
        weights=2*self.point_weight(torch.cat((local,shape.expand_as(local)),-1))
        # Sample semantic features at the exact source pixels. Bilinear lifting
        # avoids dropping sub-feature-pixel proposals during mask downsampling.
        h,w=mask.shape
        xy=torch.stack(((ids%w+.5)*2/w-1,(ids//w+.5)*2/h-1),-1)
        samples=F.grid_sample(feature_map[None],xy[None,:,None].to(feature_map.dtype),
                              mode='bilinear',padding_mode='border',align_corners=False)
        samples=samples[0,:,:,0].T
        # ESAM uses mean(weight * feature) plus a global shape feature, rather
        # than a softmax that forces point weights to sum to one.
        return (samples*weights).mean(0)+self.shape_projection(shape)
