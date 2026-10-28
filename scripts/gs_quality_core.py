"""Frozen, GT-free GS features; paired small-head and consistency diagnostics."""
from pathlib import Path
import math
import torch
from torch import nn
import torch.nn.functional as F
from scripts.frozen_iou_head_core import TARGET,digest,write_json,atomic_save

SOURCE_SHA='bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5'

PARENT=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline4_iou_mlp_20261023_v1')
POLICY='frozen_baseline8_token_and_gs_quality_v1'
FEATURE_NAMES=['support_fraction','opacity_mass_fraction','effective_support_fraction','weighted_membership','support_entropy',
    'cov_eig0','cov_eig1','cov_eig2','rms_radius','extent_x','extent_y','extent_z','mean_gs_scale','std_gs_scale','max_gs_scale',
    'opacity_mean','opacity_std','linearity','planarity','top64_largest_component_mass','top64_component_fraction','top64_isolated_mass',
    'visible0_fraction','visible1_fraction','visible_both_fraction','visibility_imbalance','view_support_cosine','view_support_absdiff',
    'projected_membership0','projected_membership1','single_view_support_fraction','pixel_area_imbalance']

def graph_stats(xyz,weights,radii,scene_scale,parent):
    """Top64 parent-anchor graph, edges <= max(.01 scene,3*(ri+rj))."""
    import numpy as np
    from scipy.sparse.csgraph import connected_components
    count=int(parent.max())+1;mass=weights.new_zeros((count,weights.shape[1]));mass.index_add_(0,parent,weights)
    parent_xyz=weights.new_zeros((count,3));parent_xyz.index_add_(0,parent,xyz)
    counts=torch.bincount(parent,minlength=count).clamp_min(1)
    parent_xyz/=counts[:,None]
    parent_r=radii.new_zeros(count);parent_r.index_add_(0,parent,radii);parent_r/=counts
    k=min(64,count);values,indices=mass.T.topk(k,dim=1)
    pos=parent_xyz[indices];rad=parent_r[indices]
    adj=(torch.cdist(pos,pos)<=torch.maximum(3*(rad[:,:,None]+rad[:,None,:]),scene_scale*.01)).cpu().numpy()
    values=values.cpu().numpy();result=[]
    for w,a in zip(values,adj):
        keep=(w>=w.max()*.01)&(w>0);w=w[keep];a=a[keep][:,keep]
        if not len(w):result.append([0.,0.,0.]);continue
        groups,labels=connected_components(a,directed=False);sums=np.bincount(labels,weights=w)
        isolated=(a.sum(1)<=1)
        result.append([float(sums.max()/w.sum()),groups/len(w),float(w[isolated].sum()/w.sum())])
    return torch.tensor(result,device=xyz.device,dtype=xyz.dtype)

@torch.no_grad()
def geometry_features(pred):
    """No labels accepted. Gaussian shape, opacity, Membership, predicted cameras."""
    g=pred['gaussians'][0].float();xyz=g[:,:3];opacity=g[:,3].clamp(0,1);scales=g[:,4:7].mean(-1)
    n=pred['num_thing_queries'];m=pred['gaussian_membership'][0,:,:n].float().clamp(0,1)
    weights=(m>=.5)*m*opacity[:,None];total=weights.sum(0);p=weights/total.clamp_min(1e-8)
    active=weights>0;nonempty=total>1e-8;N=len(g)
    center=p.T@xyz;delta=xyz[:,None,:]-center[None]
    scale=torch.linalg.vector_norm(xyz-xyz.median(0).values,dim=-1).quantile(.9).clamp_min(1e-6)
    covariance=torch.einsum('nq,nqi,nqj->qij',p,delta,delta)/(scale**2)
    eig=torch.linalg.eigvalsh(covariance).clamp_min(0)
    high=torch.where(active[:,:,None],xyz[:,None,:],-float('inf')).amax(0)
    low=torch.where(active[:,:,None],xyz[:,None,:],float('inf')).amin(0)
    extent=torch.where(nonempty[:,None],(high-low)/scale,torch.zeros_like(high))
    mean_scale=p.T@scales;scale_std=(p*(scales[:,None]-mean_scale)**2).sum(0).clamp_min(0).sqrt()
    max_scale=torch.where(active,scales[:,None],0).amax(0)
    mean_op=p.T@opacity;op_std=(p*(opacity[:,None]-mean_op)**2).sum(0).clamp_min(0).sqrt()
    entropy=-(p*p.clamp_min(1e-8).log()).sum(0)/math.log(max(2,N))
    columns=[active.float().mean(0),total/N,1/(p.square().sum(0).clamp_min(1e-8)*N),
        (p*m).sum(0),entropy,*eig.unbind(-1),eig.sum(-1).sqrt(),*extent.unbind(-1),
        mean_scale/scale,scale_std/scale,max_scale/scale,mean_op,op_std,
        (eig[:,2]-eig[:,1])/eig[:,2].clamp_min(1e-8),(eig[:,1]-eig[:,0])/eig[:,2].clamp_min(1e-8)]
    parent=pred['gaussian_parent_anchor'].long()
    columns.extend(graph_stats(xyz,weights,scales,scale,parent).unbind(-1))
    # Visibility is a proxy from predicted RGB+expected-depth rendering, not GT.
    c2w=pred['predicted_context_c2w'][0];K=pred['predicted_context_intrinsics_matrix'][0]
    w2c=torch.linalg.inv(c2w);cam=torch.einsum('vij,nj->vni',w2c[:,:3,:3],xyz)+w2c[:,None,:3,3]
    pix=torch.einsum('vij,vnj->vni',K,cam);uv=pix[...,:2]/pix[...,2:].clamp_min(1e-6)
    h,w=pred['region_mass'].shape[-2:];grid=uv.clone();grid[...,0]=2*uv[...,0]/w-1;grid[...,1]=2*uv[...,1]/h-1
    grid=grid[:,None]
    depth=F.grid_sample(pred['render']['depths_pred'][0].float(),grid,align_corners=False).squeeze(1).squeeze(1)
    alpha=F.grid_sample(pred['alpha'][0].float(),grid,align_corners=False).squeeze(1).squeeze(1)
    visible=(cam[...,2]>1e-6)&(uv[...,0]>=0)&(uv[...,0]<w)&(uv[...,1]>=0)&(uv[...,1]<h)&(alpha>.05)
    visible&=(cam[...,2]-depth).abs()<=torch.maximum(.05*depth.abs(),scale*.03)
    sampled=F.grid_sample(pred['region_mass'][0,:,:n].float(),grid,align_corners=False).squeeze(2).permute(0,2,1)
    vis=visible.float();v0=(p*vis[0,:,None]).sum(0);v1=(p*vis[1,:,None]).sum(0)
    both=vis[0]*vis[1];both_mass=(p*both[:,None]).sum(0)
    s0,s1=sampled[0],sampled[1];shared=p*both[:,None]
    denom=shared.sum(0).clamp_min(1e-8)
    cosine=(shared*s0*s1).sum(0)/((shared*s0.square()).sum(0)*(shared*s1.square()).sum(0)).clamp_min(1e-12).sqrt()
    diff=(shared*(s0-s1).abs()).sum(0)/denom
    area=(pred['region_mass'][0,:,:n]>=.5)&(pred['alpha'][0]>.05)
    areas=area.flatten(2).sum(-1).float()
    columns.extend([v0,v1,both_mass,(v0-v1).abs()/(v0+v1).clamp_min(1e-8),cosine,diff,
        (p*s0*vis[0,:,None]).sum(0)/v0.clamp_min(1e-8),(p*s1*vis[1,:,None]).sum(0)/v1.clamp_min(1e-8),
        (p*(vis[0]-vis[1]).abs()[:,None]).sum(0),(areas[0]-areas[1]).abs()/(areas.sum(0)+1)])
    result=torch.stack(columns,-1);result=torch.where(nonempty[:,None],result,torch.zeros_like(result))
    if result.shape!=(n,32) or not torch.isfinite(result).all():raise ValueError('Invalid GS descriptor')
    return result

def cross_scene_shuffle(scenes,seed=31415):
    """A true permutation preserving marginals, with no same-scene donor."""
    unique=sorted(set(scenes));gen=torch.Generator().manual_seed(seed)
    unique=[unique[i] for i in torch.randperm(len(unique),generator=gen).tolist()]
    buckets={s:[] for s in unique}
    for i,s in enumerate(scenes):buckets[s].append(i)
    groups=[buckets[s] for s in unique]
    largest=max(map(len,groups));N=len(scenes)
    if largest*2>N:raise ValueError('Cannot make a cross-scene permutation')
    order=torch.tensor([i for group in groups for i in group]);donor=torch.empty_like(order)
    donor[order]=order.roll(largest)
    if sorted(donor.tolist())!=list(range(N)) or any(scenes[i]==scenes[j] for i,j in enumerate(donor.tolist())):raise ValueError('Invalid shuffle')
    return donor

class GSQualityHead(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.token_norm=nn.LayerNorm(256) if arm in ('A','C','D') else nn.Identity()
        self.net=nn.Sequential(nn.Linear(288 if arm in ('C','D') else 256 if arm=='A' else 32,256),nn.GELU(),nn.Linear(256,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,token,geo):
        x=torch.cat((self.token_norm(token.detach().float()),geo.detach().float()),-1) if self.arm in ('C','D') else self.token_norm(token.detach().float()) if self.arm=='A' else geo.detach().float()
        return self.net(x).squeeze(-1).sigmoid()

def view_gt_diagnostics(masks,semantic,instance,n):
    """GT-only diagnostic outputs; never input to a head or quality ranking."""
    out=torch.full((n,4),-1.,device=masks.device)
    for view in (0,1):
        valid=(semantic[view]>=0)&(semantic[view]<20)
        ids=instance[view][valid&(semantic[view]>=2)&(instance[view]>0)].unique()
        q=masks[view].long();selected=valid&(q>0)&(q<=n)
        hist=torch.bincount((q[selected]-1)*20+semantic[view][selected].long(),minlength=n*20).reshape(n,20)
        areas=hist.sum(-1);out[:,view]=0.;out[:,view+2]=torch.where(areas>0,hist.argmax(-1),-1)
        if len(ids):
            gt=valid&(semantic[view]>=2)&(instance[view]>0)
            gi=torch.searchsorted(ids,instance[view][gt]).long()
            gtareas=torch.bincount(gi,minlength=len(ids))
            overlap_pixels=gt&selected
            paired=torch.searchsorted(ids,instance[view][overlap_pixels]).long()
            overlap=torch.bincount((q[overlap_pixels]-1)*len(ids)+paired,minlength=n*len(ids)).reshape(n,len(ids))
            out[:,view]=(overlap/(areas[:,None]+gtareas[None]-overlap).clamp_min(1)).max(-1).values
    return out
