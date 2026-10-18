"""Independent geometry controls; legacy A and Unified retain their behavior."""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tokengs.models.dual_stream_decoder import PointSeededAnchorDecoder
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.locusgs_recon import anchor_ray_geometric_bias
from tokengs.models.object_locus_posefree_geometry import pixel_rays


def bounded_anchor_update(mu, rho, raw_center, raw_radius, epsilon, step=.1):
    """Bound center displacement by step*r and positive-radius ratio by exp(±step).

    Scale the softplus part of r=epsilon+softplus(rho), preserving the original
    decoder's radius floor. No new parameters; zero heads retain seed geometry.
    """
    positive = F.softplus(rho)
    radius = positive + epsilon
    center = mu + (step / math.sqrt(3)) * radius[..., None] * raw_center.tanh()
    updated = positive * torch.exp(step * raw_radius.tanh())
    updated = updated.clamp_min(torch.finfo(updated.dtype).tiny)
    new_rho = updated + torch.log(-torch.expm1(-updated))
    return center, new_rho


class BoundedPointSeededAnchorDecoder(PointSeededAnchorDecoder):
    def update_geometry(self, h, mu, rho, head_mu, head_rho):
        return bounded_anchor_update(mu,rho,head_mu(h),head_rho(h).squeeze(-1),self.epsilon)


class BoundedSuperpointDualCore(SuperpointDualCore):
    anchor_decoder_type = BoundedPointSeededAnchorDecoder


def feature_grid_rays(c2w, intrinsics, image_hw=(256,256), grid_hw=(32,32)):
    """Rays at feature-cell centers, in the same view/row/column order as K/V.

    Intrinsics and poses come exclusively from the context Omega prediction.
    Pixel edge coordinates and align_corners=False are preserved by K scaling.
    """
    height,width = grid_hw
    scale = intrinsics.new_tensor([width/image_hw[1],height/image_hw[0],1.])
    grid_k = scale[...,None] * intrinsics
    origins,directions = pixel_rays(c2w,grid_k,height,width)
    origins = origins.movedim(-3,-1).reshape(c2w.shape[0],-1,3)
    directions = directions.movedim(-3,-1).reshape(c2w.shape[0],-1,3)
    return torch.cross(origins,directions,dim=-1),directions


class GeometryBiasedUnifiedCore(UnifiedQueryCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        # softplus(-4.6) ~= .01, hence initial attention penalty in [-.2,0].
        self.ray_gate_raw = nn.Parameter(torch.full((config.depth,),-4.6))

    def attention_bias(self,centers,radii,moment,direction,index):
        # Compute distances in FP32 under autocast. A finite penalty retains
        # global context; it is not a visibility test or an attention mask.
        with torch.autocast(device_type=centers.device.type,enabled=False):
            geometric = anchor_ray_geometric_bias(centers.float(),radii.float(),moment.float(),direction.float(),
                sigma0=2.,bandwidth_floor=1e-8,clamp_min=-20.)
            return F.softplus(self.ray_gate_raw[index].float()) * geometric

    def decode_query(self,block,q,memory,centers,radii,index):
        keys = memory['features'];moment,direction = memory['rays']
        if moment.shape != direction.shape or moment.shape[:2] != keys.shape[:2]:
            raise ValueError('Unified ray/key layout mismatch')
        # Bind block/index at creation, and pass geometry as checkpoint inputs:
        # backward must replay this layer's geometry, not the final geometry.
        def decode(query,features,mu,radius,m,d,layer=block,idx=index):
            bias = self.attention_bias(mu,radius,m,d,idx).to(query.dtype)
            bias = bias.expand(-1,self.config.heads,-1,-1).reshape(-1,query.shape[1],features.shape[1])
            return layer(query,features,memory_mask=bias)
        args = (q,keys,centers,radii,moment,direction)
        if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(decode,*args,use_reentrant=False)
        return decode(*args)
