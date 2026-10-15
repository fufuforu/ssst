"""Context-only, detached geometry/proposal selection; trainable content pooling."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def unproject(depth, k, c2w):
    b,v,_,h,w = depth.shape
    y,x = torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
    pixels = torch.stack((x+.5,y+.5,torch.ones_like(x)),-1).float()
    camera = torch.einsum('bvij,hwj->bvhwi',torch.linalg.inv(k),pixels)*depth.permute(0,1,3,4,2)
    return torch.einsum('bvij,bvhwj->bvhwi',c2w[...,:3,:3],camera)+c2w[...,:3,3][:,:,None,None]


@torch.no_grad()
def seed_anchors(points, confidence, rgb, count=1024, candidate_side=48):
    """Per-view image-grid coverage, confidence filtering, deterministic 3D FPS.

    Keep top confidence point per grid cell before FPS (not global top-k).
    Returns scene coordinates, kNN support radius and RGB content, no GT input.
    """
    if count<2: raise ValueError("At least two reconstruction anchors required")
    b,v,h,w,_ = points.shape
    if confidence.ndim==5: confidence=confidence.squeeze(2)
    colors=F.interpolate(rgb.flatten(0,1),size=(h,w),mode='bilinear',align_corners=False).reshape(b,v,3,h,w).permute(0,1,3,4,2)
    outputs=[]
    for bi in range(b):
        valid=torch.isfinite(points[bi]).all(-1)&torch.isfinite(confidence[bi])
        ids=[]
        for vi in range(v):
            # Moderate threshold, separately per view: retain 80% rather than erase small objects.
            vals=confidence[bi,vi][valid[vi]]
            if not vals.numel(): continue
            keep=valid[vi]&(confidence[bi,vi]>=vals.quantile(.2))
            score=confidence[bi,vi].masked_fill(~keep,-torch.inf)[None,None]
            _,index=F.adaptive_max_pool2d(score,(min(h,candidate_side),min(w,candidate_side)),return_indices=True)
            local=index.flatten(); local=local[keep.flatten()[local]]
            ids.append(local+vi*h*w)
        if not ids: raise ValueError('No valid VGGT points; refusing fabricated anchors')
        ids=torch.cat(ids).unique(sorted=True)
        p=points[bi].reshape(-1,3)[ids]; col=colors[bi].reshape(-1,3)[ids]
        chosen=[]; dist=torch.full((len(p),),torch.inf,device=p.device)
        far=(p-p.mean(0)).square().sum(-1).argmax()
        for _ in range(min(count,len(p))):
            chosen.append(far); dist=torch.minimum(dist,(p-p[far]).square().sum(-1)); dist[torch.stack(chosen)]=-1
            far=dist.argmax()
        ix=torch.stack(chosen)
        if len(ix)<count: ix=ix.repeat((count+len(ix)-1)//len(ix))[:count]
        mu=p[ix]
        d=torch.cdist(mu,mu); d.fill_diagonal_(torch.inf)
        # Sixth-neighbor radius; duplicated/degenerate points use scene-relative floor.
        ell=(p-p.mean(0)).square().sum(-1).mean().sqrt().clamp_min(.05)
        radius=d.topk(min(6,count-1),largest=False).values[:,-1].clamp(min=.01*ell,max=.3*ell)
        outputs.append((mu,radius,col[ix]))
    return tuple(torch.stack([out[i] for out in outputs]) for i in range(3))


class SAMQueryInitializer(nn.Module):
    """Pool current semantic memory; SAM supplies masks, never SAM latent tokens.

    80 proposal slots + 20 permanently learned fallback slots. Conservative
    cross-view merge requires both 3D support overlap and appearance similarity;
    masks from the same view are deduplicated by IoU but never identity-merged.
    """
    def __init__(self, dim=256, slots=100, reserve=20):
        super().__init__(); self.slots=slots; self.reserve=reserve
        self.fallback=nn.Parameter(torch.randn(slots,dim)*.02)
        self.geometry=nn.Sequential(nn.Linear(6,dim),nn.GELU(),nn.Linear(dim,dim))
        self.norm=nn.LayerNorm(dim)

    def forward(self, fm, proposals, points, mu):
        b,v,d,h,w=fm.shape
        point_grid=F.interpolate(points.permute(0,1,4,2,3).flatten(0,1),size=(h,w),mode='nearest').reshape(b,v,3,h,w).permute(0,1,3,4,2)
        allq=[]; allc=[]; alls=[]; logs=[]
        for bi in range(b):
            ell=(mu[bi]-mu[bi].mean(0)).square().sum(-1).mean().sqrt().clamp_min(.05).detach()
            groups=[]; accepted=[[] for _ in range(v)]
            candidates=sorted([(float(p['score']),vi,p['mask']) for vi,ps in enumerate(proposals[bi]) for p in ps],key=lambda t:-t[0])
            for score,vi,mask in candidates:
                mask=torch.as_tensor(mask,device=fm.device).float()[None,None]
                mask=F.interpolate(mask,size=(h,w),mode='nearest')[0,0].bool()
                if not mask.any(): continue
                if any(float((mask&m).sum()/((mask|m).sum().clamp_min(1)))>.8 for m in accepted[vi]): continue
                xyz=point_grid[bi,vi][mask]; finite=torch.isfinite(xyz).all(-1)
                if not finite.any(): continue
                xyz=xyz[finite].detach(); feat=fm[bi,vi].permute(1,2,0)[mask][finite].mean(0)
                center=xyz.median(0).values; spread=(xyz-center).square().mean(0).sqrt().clamp_min(.02*ell)
                # Compact point support for symmetric nearest-distance overlap.
                support=xyz[torch.linspace(0,len(xyz)-1,min(128,len(xyz)),device=xyz.device).long()]
                match=None
                for gi,g in enumerate(groups):
                    if vi in g['views']: continue
                    distances=torch.cdist(support,g['support'])
                    overlap=min(float((distances.amin(0)<.05*ell).float().mean()),float((distances.amin(1)<.05*ell).float().mean()))
                    cosine=float(F.cosine_similarity(feat.detach(),g['feat'].detach(),dim=0))
                    if overlap>.5 and cosine>.7: match=gi; break
                accepted[vi].append(mask)
                if match is None:
                    groups.append(dict(feat=feat,center=center,spread=spread,support=support,views=[vi],score=score,count=1))
                else:
                    g=groups[match]; n=g['count']; g['feat']=(g['feat']*n+feat)/(n+1)
                    g['center']=(g['center']*n+center)/(n+1); g['spread']=torch.maximum(g['spread'],spread)
                    g['views'].append(vi); g['count']+=1
            groups=groups[:self.slots-self.reserve]
            q=self.fallback.clone(); c=mu[bi].mean(0).expand(self.slots,3).clone(); s=ell.expand(self.slots,3).clone()
            for i,g in enumerate(groups):
                c[i]=g['center']; s[i]=g['spread']
                q[i]=g['feat']+self.geometry(torch.cat(((c[i]-mu[bi].mean(0))/ell,s[i]/ell)))
            allq.append(self.norm(q)); allc.append(c); alls.append(s)
            logs.append(dict(proposal_queries=len(groups),learned_queries=self.slots-len(groups),cross_view_groups=sum(len(g['views'])>1 for g in groups)))
        return torch.stack(allq),torch.stack(allc),torch.stack(alls),logs
