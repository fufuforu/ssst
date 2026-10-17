"""Independent GFM reconstruction memory and mask-pool-before-projection queries."""
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.input_types import EncoderLatent
from tokengs.models.dual_stream_initialization import SAMQueryInitializer


def gfm_grid(layers):
    """Frozen, parameter-free mean of normalized Omega layers, native 32x32 grid."""
    if len(layers)!=4 or any(x.shape[-2:]!=(1024,2048) for x in layers):
        raise ValueError('Omega GFM expects four [B,V,1024,2048] patch layers')
    value=sum(F.layer_norm(x.float(),(2048,)) for x in layers)/4
    b,v=value.shape[:2]
    return value.reshape(b,v,32,32,2048).permute(0,1,4,2,3).contiguous()


class GFMReconstructionAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norms=nn.ModuleList([nn.LayerNorm(2048) for _ in range(4)])
        self.layer_logits=nn.Parameter(torch.zeros(4))
        self.projection=nn.Sequential(nn.Linear(2048,1024),nn.GELU(),nn.LayerNorm(1024))
        self.kv=nn.Linear(1024,2048)
        self.key_norm=nn.LayerNorm(64)
    def forward(self,layers):
        if len(layers)!=4:raise ValueError('Four Omega layers required')
        mixed=sum(w*norm(x.float()) for w,norm,x in zip(self.layer_logits.softmax(0),self.layer_norms,layers))
        b,v,p,_=mixed.shape
        if (v,p)!=(2,1024):raise ValueError('Reconstruction K/V must have two 32x32 context grids')
        memory=self.projection(mixed.reshape(b,v*p,2048))
        keys,values=self.kv(memory).chunk(2,-1)
        keys=keys.reshape(b,v*p,16,64).transpose(1,2).contiguous()
        values=values.reshape(b,v*p,16,64).transpose(1,2).contiguous()
        return EncoderLatent(keys=self.key_norm(keys),values=values)


def sample_anchor_gfm(grid,source_ids,image_hw=(256,256)):
    """Bilinear lift from the exact source view/pixel selected by point-cloud FPS."""
    b,v,c,_,_=grid.shape;h,w=image_hw;n=source_ids.shape[1]
    view=source_ids//(h*w);pixel=source_ids%(h*w)
    x=(pixel%w+.5)*2/w-1;y=(pixel//w+.5)*2/h-1
    xy=torch.stack((x,y),-1)[:,None].expand(-1,v,-1,-1).reshape(b*v,n,1,2)
    values=F.grid_sample(grid.flatten(0,1),xy,mode='bilinear',padding_mode='border',align_corners=False)
    values=values.reshape(b,v,c,n).permute(0,3,1,2)
    return values.gather(2,view[:,:,None,None].expand(-1,-1,1,c)).squeeze(2)


class GFMSAMQueryInitializer(SAMQueryInitializer):
    """SAM area-weights pool native GFM BEFORE the learned 2048->256 projection.

    Geometry/dedup use full-resolution mask support so sub-patch objects do not
    disappear under nearest-neighbor mask resizing. No semantic-adapter input.
    """
    def __init__(self,input_dim=2048):
        super().__init__()
        self.input_dim=input_dim
        self.content_projection=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,256),nn.GELU(),nn.Linear(256,256))

    def pool_region(self,feature_map,mask,point_map):
        # Keep the V2/V3 computation unchanged; V4 replaces only this operation.
        weights=F.interpolate(mask.float()[None,None],size=feature_map.shape[-2:],mode='area')[0,0]
        return (feature_map*weights[None]).sum((1,2))/weights.sum().clamp_min(1e-8)

    def compose_seed(self,base,seed):
        return seed

    def make_seed(self,group,center,scale,origin,ell,seed_context,batch_index):
        return self.content_projection(group['feat'])+self.geometry(torch.cat(((center-origin)/ell,scale/ell)))

    def forward(self,gfm,proposals,points,mu,*,seed_context=None):
        b,v,d,h,w=gfm.shape
        if d!=self.input_dim:raise ValueError(f'Query initializer expects {self.input_dim} feature channels, got {d}')
        ph,pw=points.shape[2:4]
        allq=[];allc=[];alls=[];logs=[]
        for bi in range(b):
            origin=mu[bi].mean(0)
            ell=(mu[bi]-origin).square().sum(-1).mean().sqrt().clamp_min(.05).detach()
            groups=[];accepted=[[] for _ in range(v)]
            candidates=sorted([(float(p['score']),vi,p['mask']) for vi,ps in enumerate(proposals[bi]) for p in ps],key=lambda x:-x[0])
            for score,vi,raw_mask in candidates:
                mask=torch.as_tensor(raw_mask,device=gfm.device).float()[None,None]
                mask=F.interpolate(mask,size=(ph,pw),mode='nearest')[0,0].bool()
                if not mask.any():continue
                if any(float((mask&m).sum()/(mask|m).sum().clamp_min(1))>.8 for m in accepted[vi]):continue
                xyz=points[bi,vi][mask];finite=torch.isfinite(xyz).all(-1)
                if not finite.any():continue
                xyz=xyz[finite].detach()
                feat=self.pool_region(gfm[bi,vi],mask,points[bi,vi])
                center=xyz.median(0).values;spread=(xyz-center).square().mean(0).sqrt().clamp_min(.02*ell)
                support=xyz[torch.linspace(0,len(xyz)-1,min(128,len(xyz)),device=xyz.device).long()]
                match=None
                for gi,group in enumerate(groups):
                    if vi in group['views']:continue
                    distances=torch.cdist(support,group['support'])
                    overlap=min(float((distances.amin(0)<.05*ell).float().mean()),float((distances.amin(1)<.05*ell).float().mean()))
                    cosine=float(F.cosine_similarity(feat.detach(),group['feat'].detach(),dim=0))
                    if overlap>.5 and cosine>.7:match=gi;break
                accepted[vi].append(mask)
                if match is None:groups.append(dict(feat=feat,center=center,spread=spread,support=support,seed_supports=[support],views=[vi],count=1))
                else:
                    group=groups[match];count=group['count']
                    group['feat']=(group['feat']*count+feat)/(count+1)
                    group['center']=(group['center']*count+center)/(count+1)
                    group['spread']=torch.maximum(group['spread'],spread);group['views'].append(vi);group['count']+=1
                    group['seed_supports'].append(support)
            groups=groups[:self.slots-self.reserve]
            q=self.fallback.clone();c=origin.expand(self.slots,3).clone();s=ell.expand(self.slots,3).clone()
            for i,group in enumerate(groups):
                c[i]=group['center'];s[i]=group['spread']
                seed=self.make_seed(group,c[i],s[i],origin,ell,seed_context,bi)
                q[i]=self.compose_seed(self.fallback[i],seed)
            allq.append(self.norm(q));allc.append(c);alls.append(s)
            logs.append(dict(proposal_queries=len(groups),learned_queries=self.slots-len(groups),cross_view_groups=sum(len(g['views'])>1 for g in groups),pooling=getattr(self,'pooling_description','SAM area-weighted semantic adapter output then query projection' if self.input_dim==256 else 'SAM area-weighted raw GFM then projection')))
        return torch.stack(allq),torch.stack(allc),torch.stack(alls),logs


class SemanticSAMQueryInitializer(GFMSAMQueryInitializer):
    """Pool trainable semantic memory, with intact adapter -> query gradients."""
    def __init__(self):super().__init__(input_dim=256)
