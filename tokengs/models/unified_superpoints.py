"""VGGT geometry + ESAM-style SAM partitions -> encoded 3D superpoint queries."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class UnifiedQueryConfig:
    superpoint_source:str="esam_sam"
    uncovered_clusters:int=8
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
    matching_mode:str="source_many_to_one"
    mask_nms_iou:float=.7
    support_matching:bool=True
    support_purity:float=.5

    def __post_init__(self):
        if self.superpoint_source not in ("esam_sam","geometric_voxel") or self.uncovered_clusters<1:raise ValueError("Invalid superpoint source/settings")
        if self.matching_mode not in ("hungarian","source_many_to_one") or not 0<self.mask_nms_iou<=1:raise ValueError("Invalid assignment/NMS settings")
        if self.matching_mode=="source_many_to_one" and not self.support_matching:raise ValueError("Many-to-one requires source support")
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


@torch.no_grad()
def esam_image_partition(masks,height,width,device):
    """ESAM: descending area, smaller regions overwrite larger regions."""
    labels=torch.full((height,width),-1,dtype=torch.long,device=device)
    converted=[torch.as_tensor(m.get('mask',m.get('segmentation')),device=device,dtype=torch.bool) for m in masks]
    if any(m.shape!=(height,width) for m in converted):raise ValueError('SAM/geometry pixel grids differ')
    for index,mask in enumerate(sorted(converted,key=lambda m:-int(m.sum()))):labels[mask]=index
    return labels


@torch.no_grad()
def labelled_voxel_partition(points,labels,scale,relative_size,budget):
    """Fuse only observations of the SAME superpoint; never erase its boundary."""
    if len(labels.unique())>budget:raise ValueError('Point budget cannot retain every superpoint')
    origin=points.amin(0);size=float(scale)*relative_size
    for _ in range(128):
        keys=torch.cat((labels[:,None],torch.floor((points-origin)/size).long()),-1)
        _,ids=torch.unique(keys,dim=0,sorted=True,return_inverse=True)
        if int(ids.max())+1<=budget:return ids,size
        size*=1.25
    raise RuntimeError('Could not construct region-preserving bounded cloud')


def lift_context_cloud(omega,feature_grid,rgb,config,proposals=None):
    """Lift context pixels; retain every finite SAM region before point fusion.

    ESAM MV uncovered labels use deterministic XYZRGB KMeans (RGB 0..255).
    Region IDs are view-specific, not speculative cross-view instance matches.
    """
    points=omega['points'][0].detach();v,h,w,_=points.shape
    confidence=omega['confidence'][0].reshape(v,h,w).detach()
    source=[];source_labels=[];offset=0
    esam=config.superpoint_source=="esam_sam"
    if esam and (proposals is None or len(proposals)!=v):raise ValueError("ESAM mode requires per-view SAM masks")
    depth=omega["depth"][0].reshape(v,h,w).detach()
    colors=F.interpolate(rgb[0],size=(h,w),mode="bilinear",align_corners=False).permute(0,2,3,1).detach()
    with torch.no_grad():
        for view in range(v):
            valid=torch.isfinite(points[view]).all(-1)&torch.isfinite(confidence[view])&torch.isfinite(depth[view])&(depth[view]>0)
            score=confidence[view].masked_fill(~valid,-torch.inf)[None,None]
            _,ids=F.adaptive_max_pool2d(score,(min(h,config.point_grid),min(w,config.point_grid)),return_indices=True)
            ids=ids.flatten();ids=ids[valid.flatten()[ids]].unique(sorted=True)
            if esam:
                labels=esam_image_partition(proposals[view],h,w,points.device).flatten()
                # Add one valid source per region, including tiny regions missed by grid sampling.
                retained=[]
                for label in labels[valid.flatten()].unique():
                    if label>=0:retained.append(torch.where(valid.flatten()&(labels==label))[0][0])
                if retained:ids=torch.cat((ids,torch.stack(retained))).unique(sorted=True)
                groups=labels[ids].clone();uncovered=groups<0;n=int(uncovered.sum())
                next_id=int(groups.max())+1 if len(groups) else 0
                if n:
                    if n<config.uncovered_clusters:extra=torch.zeros(n,device=points.device,dtype=torch.long)
                    else:
                        from sklearn.cluster import KMeans
                        values=torch.cat((points[view].reshape(-1,3)[ids[uncovered]],(colors[view].reshape(-1,3)[ids[uncovered]]*255).round()),-1)
                        extra=torch.as_tensor(KMeans(n_clusters=config.uncovered_clusters,n_init=10,random_state=42).fit_predict(values.cpu().numpy()),device=points.device,dtype=torch.long)
                    groups[uncovered]=extra+next_id
                if len(groups):
                    _,groups=torch.unique(groups,sorted=True,return_inverse=True)
                    source_labels.append(groups+offset);offset+=int(groups.max())+1
            source.append(ids+view*h*w)
        source=torch.cat(source)
        if not len(source):raise ValueError('No finite context geometry')
        xyz=points.reshape(-1,3)[source];origin,scale=scene_frame(xyz)
        if esam:
            labels=torch.cat(source_labels)
            if offset>config.max_superpoints:raise ValueError(f'ESAM generated {offset} superpoints, budget {config.max_superpoints}; no truncation')
            point_ids,voxel_size=labelled_voxel_partition(xyz,labels,scale,config.point_voxel,config.max_points)
        else:point_ids,voxel_size=voxel_partition(xyz,scale,config.point_voxel,config.max_points)
    view=source//(h*w);pixel=source%(h*w)
    xy=torch.stack(((pixel%w+.5)*2/w-1,(pixel//w+.5)*2/h-1),-1)
    # Sample each view, then gather its actual source observation.
    samples=F.grid_sample(feature_grid[0],xy[None,:,None].expand(v,-1,-1,-1),align_corners=False,padding_mode='border')[:,:, :,0]
    samples=samples.permute(2,0,1)[torch.arange(len(source),device=source.device),view]
    count=int(point_ids.max())+1
    result=dict(points=scatter_mean(xyz,point_ids,count).detach(),features=scatter_mean(samples,point_ids,count),
        source_indices=source,source_to_point=point_ids,source_points=xyz,origin=origin,scale=scale,point_voxel_size=voxel_size)
    if esam:
        # All members of a fused point have identical labels, so the mean is exact.
        result['point_superpoint_ids']=scatter_mean(labels.double()[:,None],point_ids,count).squeeze(-1).long()
        result['superpoint_source']='esam_sam'
    return result


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
        if superpoint_ids is None and 'point_superpoint_ids' in cloud:superpoint_ids=cloud['point_superpoint_ids']
        if superpoint_ids is None:
            if self.config.superpoint_source!='geometric_voxel':raise ValueError('ESAM cloud is missing superpoint IDs')
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
