"""Anchor-domain GT construction and unified pixel/anchor Hungarian losses."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tokengs.models.instance_state_loss import (
    _check_labels, _flat_regions, _linspace_indices, _matching_cost,
    stuff_loss, semantic_loss, identity_loss, NUM_THING, NO_OBJECT_INDEX,
    CLASS_CE_WEIGHT, MASK_BCE_WEIGHT, MASK_DICE_WEIGHT, UNMATCHED_CLASS_WEIGHT,
)

IGNORE, THING, WALL, FLOOR = -1, 0, 1, 2

def resolve_anchor_observations(observations):
    """Resolve valid (kind, semantic class, instance id) observations by consensus."""
    if not observations: return IGNORE,0,0
    first=observations[0]
    if first[0]==THING and all(x[0]==THING and x[1:]==first[1:] for x in observations):
        return THING,int(first[2]),int(first[1])
    if all(x[0]==WALL for x in observations): return WALL,0,0
    if all(x[0]==FLOOR for x in observations): return FLOOR,0,0
    return IGNORE,0,0

def project_points(xyz, cam_view, intrinsics):
    """Verified scripts/instance_state_s1_local3d.py convention (same compositing helper)."""
    c2w = torch.inverse(cam_view.transpose(0,1).float())
    xc=(xyz-c2w[:3,3]) @ c2w[:3,:3]
    z=xc[:,2]; fx,fy,cx,cy=[float(v) for v in intrinsics[:4]]
    u=fx*xc[:,0]/z.clamp_min(1e-6)+cx; v=fy*xc[:,1]/z.clamp_min(1e-6)+cy
    return u,v,z

def build_anchor_targets(mu, semantic, instance, cam_view, intrinsics, *, num_thing=100):
    """Return deterministic per-scene instances and two-view consensus anchor labels."""
    B,T,_=mu.shape; device=mu.device
    classes=[]; ids_all=[]; masks_all=[]; kinds=[]; inst_labels=[]; cls_labels=[]
    for b in range(B):
        sem=semantic[b,:2].long(); ins=instance[b,:2].long(); _check_labels(sem,ins,where="anchor GT")
        thing=(sem>=2)&(sem<=19)&(ins>0)
        ids=torch.unique(ins[thing],sorted=True)
        rows=[]; cls=[]
        for iid in ids.tolist():
            cc=torch.unique(sem[thing & (ins==int(iid))])
            if cc.numel()!=1: raise RuntimeError(f"instance {iid} maps to multiple semantic classes: {cc.tolist()}")
            cls.append(int(cc.item())); rows.append(torch.stack([thing[v]&(ins[v]==int(iid)) for v in range(2)]))
        if len(rows)>num_thing: raise RuntimeError(f"{len(rows)} GT instances exceed {num_thing} thing queries")
        classes.append(torch.tensor(cls,device=device,dtype=torch.long)); ids_all.append(ids.to(device))
        masks_all.append(torch.stack(rows) if rows else torch.zeros((0,2,*sem.shape[-2:]),device=device,dtype=torch.bool))
        kind=torch.full((T,),IGNORE,device=device,dtype=torch.long); ai=torch.zeros(T,device=device,dtype=torch.long); ac=torch.zeros(T,device=device,dtype=torch.long)
        observations=[[] for _ in range(T)]
        H,W=sem.shape[-2:]
        for v in range(2):
            u,vv,z=project_points(mu[b].detach(),cam_view[b,v],intrinsics[b,v])
            ok=(z>0)&(u>=0)&(u<W)&(vv>=0)&(vv<H)
            ix=torch.nonzero(ok,as_tuple=False).flatten()
            if ix.numel():
                uu=u[ix].long(); vy=vv[ix].long(); sv=sem[v,vy,uu]; iv=ins[v,vy,uu]
                valid=(sv>=0)&(sv<=19)
                for j in torch.nonzero(valid,as_tuple=False).flatten().tolist():
                    t=int(ix[j]); sc=int(sv[j]); iid=int(iv[j])
                    obs=THING if 2<=sc<=19 and iid>0 else (WALL if sc==0 else FLOOR if sc==1 else IGNORE)
                    if obs!=IGNORE: observations[t].append((obs,sc,iid))
        for t,obs in enumerate(observations):
            kk,ii,cc=resolve_anchor_observations(obs); kind[t]=kk; ai[t]=ii; ac[t]=cc
        kinds.append(kind); inst_labels.append(ai); cls_labels.append(ac)
    K=max((x.numel() for x in classes),default=0)
    Y=torch.zeros((B,K,T),device=device,dtype=torch.float32)
    for b in range(B):
        for k,iid in enumerate(ids_all[b]): Y[b,k]=((kinds[b]==THING)&(inst_labels[b]==iid)).float()
    valid=torch.stack([(k==THING)|(k==WALL)|(k==FLOOR) for k in kinds])
    return dict(gt_classes=classes,gt_instance_ids=ids_all,gt_pixel_masks=masks_all,Y_anchor=Y,anchor_valid=valid,anchor_kind=torch.stack(kinds),anchor_instance_id=torch.stack(inst_labels),anchor_semantic_class=torch.stack(cls_labels))

def unified_hungarian(prediction,batch,targets=None,match_points=4096):
    """One scene Hungarian solve combines exact legacy pixel terms and anchor terms."""
    device=prediction["gaussians"].device; sem=batch["semantic_label_all"][:,:2]; ins=batch["instance_label_all"][:,:2]
    t=targets or build_anchor_targets(prediction["states"][-1]["mu"],sem,ins,batch["cam_view_all"],batch["intrinsics_all"])
    out=[]; region=_flat_regions(prediction["region_mass"][:,:,:100],100); logits=prediction["thing_class_logits"][:,:,2:]
    P=prediction["anchor_assignment"][:,:,:100].transpose(1,2)
    for b in range(sem.shape[0]):
        cls=t["gt_classes"][b]; masks=t["gt_pixel_masks"][b]; K=len(cls); valid=(sem[b]>=0)&(sem[b]<=19)&((sem[b]<2)|(ins[b]>0))
        if K==0 or not valid.any():
            out.append((torch.empty(0,device=device,dtype=torch.long),torch.empty(0,device=device,dtype=torch.long))); continue
        flat=torch.nonzero(valid.reshape(-1),as_tuple=False).flatten(); sel=_linspace_indices(len(flat),match_points).to(device); flat=flat[sel]
        z=torch.logit(region[b].clamp(1e-6,1-1e-6))[:,flat]; y=masks.flatten(1)[:,flat].float()
        pixel=_matching_cost(logits[b],z,y,cls)
        av=t["anchor_valid"][b]; pa=P[b,:,av].float(); ya=t["Y_anchor"][b,:,av].float()
        if av.any() and K:
            bce=F.softplus(torch.logit(pa.clamp(1e-6,1-1e-6))).mean(-1,keepdim=True)-(pa.logit()@ya.T)/int(av.sum())
            dice=1-(2*(pa@ya.T)+1)/(pa.sum(-1,keepdim=True)+ya.sum(-1)[None,:]+1)
            support=t["Y_anchor"][b].sum(-1)>0
            bce=bce*support[None,:]; dice=dice*support[None,:]
        else: bce=dice=torch.zeros_like(pixel)
        cost=pixel+2*bce+2*dice
        qi,ki=linear_sum_assignment(cost.detach().float().cpu().numpy())
        out.append((torch.as_tensor(qi,device=device),torch.as_tensor(ki,device=device)))
    return t,out

def anchor_group_losses(prediction,batch,opt=None):
    del opt
    device=prediction["gaussians"].device; sem=batch["semantic_label_all"][:,:2].long(); ins=batch["instance_label_all"][:,:2].long()
    _check_labels(sem,ins,where="anchor_group_losses")
    t,pairs=unified_hungarian(prediction,batch)
    region=_flat_regions(prediction["region_mass"][:,:,:100],100); logits=prediction["thing_class_logits"][:,:,2:]
    zero=prediction["gaussians"].sum()*0; thing_total=zero; ce_terms=[]; bce_terms=[]; dice_terms=[]
    ownership=prediction["anchor_assignment"]; anchor_ce=[]; anchor_dice=[]
    class_weight=torch.ones(19,device=device); class_weight[18]=UNMATCHED_CLASS_WEIGHT
    for b,(qi,ki) in enumerate(pairs):
        cls=t["gt_classes"][b]; masks=t["gt_pixel_masks"][b]; target=torch.full((100,),NO_OBJECT_INDEX,device=device,dtype=torch.long)
        target[qi]=cls[ki]-2
        ce=F.cross_entropy(logits[b].float(),target,weight=class_weight,reduction="mean"); ce_terms.append(ce)
        if qi.numel():
            flatvalid=((sem[b]>=0)&(sem[b]<=19)&((sem[b]<2)|(ins[b]>0))).flatten(); z=torch.logit(region[b,:,flatvalid].clamp(1e-6,1-1e-6)); y=masks.flatten(1)[:,flatvalid].float()
            bce=F.binary_cross_entropy_with_logits(z[qi],y[ki],reduction="none").mean(); p=z[qi].sigmoid(); inter=(p*y[ki]).sum(1); den=p.sum(1)+y[ki].sum(1); dice=(1-(2*inter+1)/(den+1)).mean()
        else: bce=dice=zero
        bce_terms.append(bce); dice_terms.append(dice)
        # Per valid confident anchor ownership target, ignore unmatched GT only if impossible (fail closed).
        id_to_gt={int(iid):k for k,iid in enumerate(t["gt_instance_ids"][b].tolist())}; kind=t["anchor_kind"][b]; aid=t["anchor_instance_id"][b]
        tgt=torch.full((kind.numel(),),-1,device=device,dtype=torch.long)
        tgt[kind==WALL]=100; tgt[kind==FLOOR]=101
        q_for_gt={int(k):int(q) for q,k in zip(qi.tolist(),ki.tolist())}
        for iid,k in id_to_gt.items():
            mask=(kind==THING)&(aid==iid)
            if mask.any() and k not in q_for_gt: raise RuntimeError("confident thing anchor belongs to unmatched GT")
            if k in q_for_gt: tgt[mask]=q_for_gt[k]
        av=t["anchor_valid"][b] & (tgt>=0)
        if av.any(): anchor_ce.append(-torch.log(ownership[b,av,tgt[av]].clamp_min(1e-6)).mean())
        for q,k in zip(qi.tolist(),ki.tolist()):
            y=t["Y_anchor"][b,k]; valid=t["anchor_valid"][b]
            if y.sum()>0:
                p=ownership[b,:,q][valid]; yy=y[valid]; anchor_dice.append(1-(2*(p*yy).sum()+1)/(p.sum()+yy.sum()+1))
    L2d=2*torch.stack(ce_terms).mean()+5*torch.stack(bce_terms).mean()+5*torch.stack(dice_terms).mean()
    Lce=torch.stack(anchor_ce).mean() if anchor_ce else zero; Ldice=torch.stack(anchor_dice).mean() if anchor_dice else zero
    Lgroup=Lce+Ldice
    stuff,sm=stuff_loss(prediction,batch); semantic,smm=semantic_loss(prediction,batch); ident,idm=identity_loss(prediction,batch)
    total=.1*L2d+.1*stuff+.1*semantic+.01*ident+.1*Lgroup
    thing_count=int((t["anchor_kind"]==THING).sum()); wall_count=int((t["anchor_kind"]==WALL).sum()); floor_count=int((t["anchor_kind"]==FLOOR).sum()); valid_count=thing_count+wall_count+floor_count
    support=t["Y_anchor"].sum(-1)>0
    metrics={"loss_anchor_group":float(Lgroup.detach()),"anchor_ce":float(Lce.detach()),"anchor_dice":float(Ldice.detach()),"anchor_valid_count":valid_count,"anchor_thing_count":thing_count,"anchor_wall_count":wall_count,"anchor_floor_count":floor_count,"anchor_ignore_count":int(t["anchor_kind"].eq(IGNORE).sum()),"gt_with_anchor_support":int(support.sum()),"gt_without_anchor_support":int((~support).sum()),"loss_thing_2d":float(L2d.detach()),"loss_stuff_2d":float(stuff.detach()),"loss_semantic":float(semantic.detach()),"loss_identity":float(ident.detach()),"loss_understanding":float(total.detach())}
    metrics.update(sm); metrics.update(smm); metrics.update(idm)
    return total,metrics

__all__=["build_anchor_targets","unified_hungarian","anchor_group_losses","project_points","resolve_anchor_observations"]
