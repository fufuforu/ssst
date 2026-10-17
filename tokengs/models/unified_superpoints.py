"""Context-derived 3D points -> neighborhood features -> superpoint queries.

Geometric voxel oversegmentation is the explicit first implementation, not a
claim of reproducing ESAM's dataset-specific superpoint generator. No SAM or
learned fallback query table is used. All discrete geometry is detached.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class UnifiedQueryConfig:
    dim:int=256
    heads:int=8
    depth:int=12
    point_grid:int=48
    max_points:int=4096
    max_superpoints:int=512
    point_voxel:float=.01
    superpoint_voxel:float=.08
    neighbors:int=12
    point_layers:int=2
    checkpoint_blocks:bool=True
    support_matching:bool=True
    support_purity:float=.5

    def __post_init__(self):
        if self.dim<8 or self.heads<1 or self.dim%self.heads or min(self.depth,self.point_grid,self.max_points,self.max_superpoints,self.neighbors,self.point_layers)<1:raise ValueError('Invalid unified dimensions/budgets')
        if self.max_superpoints>999:raise ValueError('Official packed instance IDs must remain below1000')
        if min(self.point_voxel,self.superpoint_voxel)<=0 or not 0<self.support_purity<=1:raise ValueError('Invalid geometry/matching settings')

    def to_dict(self):return asdict(self)


def scatter_mean(values,ids,count):
    output=values.new_zeros((count,)+values.shape[1:]).index_add(0,ids,values)
    denominator=torch.bincount(ids,minlength=count).clamp_min(1).to(values.dtype)
    return output/denominator.reshape((count,)+(1,)*(values.ndim-1))


@torch.no_grad()
def voxel_partition(points,scale,relative_size,budget):
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not torch.isfinite(points).all():raise ValueError('Finite nonempty XYZ required')
    origin=points.amin(0);size=float(scale)*relative_size
    if size<=0:raise ValueError('Positive voxel size required')
    for _ in range(128):
        keys=torch.floor((points-origin)/size).long()
        _,ids=torch.unique(keys,dim=0,sorted=True,return_inverse=True)
        if int(ids.max())+1<=budget:return ids,size
        size*=1.25
    raise RuntimeError('Could not construct bounded spatial partition')


@torch.no_grad()
def scene_frame(points):
    center=points.mean(0);scale=(points-center).square().sum(-1).mean().sqrt().clamp_min(.05)
    return center,scale


def lift_context_cloud(omega,feature_grid,rgb,config):
    """One scene, source pixel -> predicted XYZ + bilinear feature sample.

    Per-image grid cells retain their best finite-confidence pixel. Subsequent
    3D voxel fusion averages observations from either view; no source observation
    is discarded to enforce the 3D point budget.
    """
    points=omega['points'][0].detach();v,h,w,_=points.shape
    confidence=omega['confidence'][0].reshape(v,h,w).detach()
    source=[]
    with torch.no_grad():
        for view in range(v):
            valid=torch.isfinite(points[view]).all(-1)&torch.isfinite(confidence[view])
            score=confidence[view].masked_fill(~valid,-torch.inf)[None,None]
            _,ids=F.adaptive_max_pool2d(score,(min(h,config.point_grid),min(w,config.point_grid)),return_indices=True)
            ids=ids.flatten();ids=ids[valid.flatten()[ids]].unique(sorted=True)
            source.append(ids+view*h*w)
        source=torch.cat(source)
        if not len(source):raise ValueError('No finite context geometry')
        xyz=points.reshape(-1,3)[source];origin,scale=scene_frame(xyz)
        point_ids,voxel_size=voxel_partition(xyz,scale,config.point_voxel,config.max_points)
    view=source//(h*w);pixel=source%(h*w)
    xy=torch.stack(((pixel%w+.5)*2/w-1,(pixel//w+.5)*2/h-1),-1)
    # Sample each view, then gather its actual source observation.
    samples=F.grid_sample(feature_grid[0],xy[None,:,None].expand(v,-1,-1,-1),align_corners=False,padding_mode='border')[:,:, :,0]
    samples=samples.permute(2,0,1)[torch.arange(len(source),device=source.device),view]
    count=int(point_ids.max())+1
    return dict(points=scatter_mean(xyz,point_ids,count).detach(),features=scatter_mean(samples,point_ids,count),
        source_indices=source,source_to_point=point_ids,source_points=xyz,origin=origin,scale=scale,point_voxel_size=voxel_size)


@torch.no_grad()
def knn(points,k,chunk=256):
    return torch.cat([torch.cdist(part,points).topk(min(k,len(points)),largest=False).indices for part in points.split(chunk)])


class PointNeighborhoodEncoder(nn.Module):
    def __init__(self,dim,layers=2,neighbors=12):
        super().__init__();self.neighbors=neighbors
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(2*dim+3,dim),nn.LayerNorm(dim),nn.GELU(),nn.Linear(dim,dim)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])

    def forward(self,features,points,scale):
        ids=knn(points.detach(),self.neighbors);delta=(points[ids]-points[:,None])/scale
        for layer,norm in zip(self.layers,self.norms):
            local=torch.cat((features[:,None].expand(-1,ids.shape[1],-1),features[ids]-features[:,None],delta),-1)
            features=norm(features+layer(local).amax(1))
        return features


class SuperpointQueryInitializer(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d=config.dim
        self.point_encoder=PointNeighborhoodEncoder(d,config.point_layers,config.neighbors)
        self.local_shape=nn.Sequential(nn.Linear(3,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d))
        self.point_weight=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1),nn.Sigmoid())
        self.projection=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d))
        self.position=nn.Sequential(nn.Linear(4,d),nn.GELU(),nn.Linear(d,d));self.norm=nn.LayerNorm(d)

    def forward(self,cloud,superpoint_ids=None):
        points=cloud['points'].detach();scale=cloud['scale'].detach();origin=cloud['origin'].detach()
        features=self.point_encoder(cloud['features'],points,scale)
        if superpoint_ids is None:
            ids,voxel_size=voxel_partition(points,scale,self.config.superpoint_voxel,self.config.max_superpoints)
        else:
            supplied=torch.as_tensor(superpoint_ids,device=points.device)
            if supplied.shape!=(len(points),) or supplied.dtype not in (torch.int32,torch.int64) or (supplied<0).any():raise ValueError('External superpoint IDs must be nonnegative integer IDs for fused points')
            _,ids=torch.unique(supplied,sorted=True,return_inverse=True);voxel_size=None
            if int(ids.max())+1>self.config.max_superpoints:raise ValueError('External superpoints exceed configured budget; no silent truncation')
        count=int(ids.max())+1;center=scatter_mean(points,ids,count)
        delta=points-center[ids]
        radius=scatter_mean(delta.square().sum(-1,keepdim=True),ids,count).sqrt().squeeze(-1).clamp_min(.02*scale)
        local=self.local_shape(delta/radius[ids,None])
        shape=local.new_full((count,self.config.dim),-torch.inf).scatter_reduce(0,ids[:,None].expand_as(local),local,reduce='amax',include_self=True)
        weights=2*self.point_weight(torch.cat((local,shape[ids]),-1))
        pooled=scatter_mean(features*weights,ids,count)+shape
        position=torch.cat(((center-origin)/scale,(radius/scale)[:,None]),-1)
        query=self.norm(self.projection(pooled)+self.position(position))
        return dict(query=query[None],centers=center[None],radii=radius[None],point_features=features,
            point_superpoint_ids=ids,source_superpoint_ids=ids[cloud['source_to_point']],superpoint_counts=torch.bincount(ids,minlength=count),superpoint_voxel_size=voxel_size)
