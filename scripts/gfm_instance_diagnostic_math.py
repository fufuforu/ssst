"""Observation-only masks and initializer trace; never consumed by model predictions."""
import torch
import torch.nn.functional as F


def iou_matrix(pred,gt,valid):
    p=pred.bool().flatten(1)[:,valid.flatten()].float();y=gt.bool().flatten(1)[:,valid.flatten()].float()
    inter=p@y.T
    return inter/(p.sum(1)[:,None]+y.sum(1)[None]-inter).clamp_min(1)


def stage_masks(render):
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    member=render['region_mass'][0];alpha=render['alpha'][0,:,0];pc=render['p_class'][0]
    prob,cls=pc[:,:18].max(-1);eligible=(pc.argmax(-1)!=18)&(prob>=.05)
    raw=(member[:,:100]>=.5)&(alpha[:,None]>.05)
    candidates=torch.cat((raw&eligible[None,:,None,None],(member[:,100:102]>=.5)&(alpha[:,None]>.05)),1)
    scores=torch.cat((member[:,:100]*prob[None,:,None,None],member[:,100:102]),1)
    maximum,winner=torch.where(candidates,scores,torch.full_like(scores,-1.)).max(1)
    ids=torch.arange(100,device=member.device)[None,:,None,None]
    won=(winner[:,None]==ids)&(maximum[:,None]>=0)
    _,ins,_=assemble_panoptic(render)
    return dict(raw=raw,eligible=raw&eligible[None,:,None,None],competition=won,final=ins[:,None]==ids+1),cls+2,eligible


class InitializerTrace:
    """Wrap pool_region without modifying values; replay identical grouping for mask supports."""
    def __init__(self,initializer):
        self.initializer=initializer;self.original=initializer.pool_region;self.records=[];self.inputs=None
        self.handle=initializer.register_forward_pre_hook(self.pre)
        initializer.pool_region=self.pool

    def pre(self,module,args):
        self.inputs=args;self.records=[]

    def pool(self,feature,mask,points):
        result=self.original(feature,mask,points)
        memory,_,_,_=self.inputs
        if len(memory)!=1:raise ValueError('Diagnostic trace requires batch1')
        matches=[v for v in range(2) if memory[0,v].data_ptr()==feature.data_ptr()]
        if len(matches)!=1:raise ValueError('Cannot identify proposal view')
        self.records.append((matches[0],mask.detach().clone(),result.detach().clone()))
        return result

    def selected_masks(self,generated):
        memory,proposals,points,mu=self.inputs;origin=mu[0].mean(0)
        ell=(mu[0]-origin).square().sum(-1).mean().sqrt().clamp_min(.05).detach()
        groups=[]
        for vi,mask,feat in self.records:
            xyz=points[0,vi][mask];xyz=xyz[torch.isfinite(xyz).all(-1)].detach()
            center=xyz.median(0).values;spread=(xyz-center).square().mean(0).sqrt().clamp_min(.02*ell)
            support=xyz[torch.linspace(0,len(xyz)-1,min(128,len(xyz)),device=xyz.device).long()]
            match=None
            for gi,group in enumerate(groups):
                if vi in group['views']:continue
                distances=torch.cdist(support,group['support'])
                overlap=min(float((distances.amin(0)<.05*ell).float().mean()),float((distances.amin(1)<.05*ell).float().mean()))
                cosine=float(F.cosine_similarity(feat,group['feat'],dim=0))
                if overlap>.5 and cosine>.7:match=gi;break
            if match is None:
                masks=torch.zeros((2,*mask.shape),dtype=torch.bool,device=mask.device);masks[vi]=mask
                groups.append(dict(feat=feat,center=center,spread=spread,support=support,views=[vi],count=1,masks=masks))
            else:
                group=groups[match];count=group['count']
                group['feat']=(group['feat']*count+feat)/(count+1);group['center']=(group['center']*count+center)/(count+1)
                group['spread']=torch.maximum(group['spread'],spread);group['views'].append(vi);group['count']+=1;group['masks'][vi]=mask
        groups=groups[:self.initializer.slots-self.initializer.reserve]
        if len(groups)!=generated['proposal_initialization'][0]['proposal_queries']:raise ValueError('Initializer grouping replay differs')
        q=self.initializer.fallback.clone();c=origin.expand(self.initializer.slots,3).clone();s=ell.expand(self.initializer.slots,3).clone()
        for i,group in enumerate(groups):
            c[i]=group['center'];s[i]=group['spread']
            q[i]=self.initializer.content_projection(group['feat'])+self.initializer.geometry(torch.cat(((c[i]-origin)/ell,s[i]/ell)))
        torch.testing.assert_close(self.initializer.norm(q),generated['q_pre'][0],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(c,generated['initial_query_centers'][0],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(s,generated['initial_query_scales'][0],rtol=1e-5,atol=1e-6)
        masks=torch.stack([g['masks'] for g in groups]) if groups else torch.zeros((0,2,256,256),device=memory.device,dtype=torch.bool)
        return masks

    def raw_proposal_best_per_view(self,gt,valid):
        proposals=self.inputs[1][0];result=gt.new_zeros((len(gt),2),dtype=torch.float32)
        for vi in range(2):
            if not proposals[vi]:continue
            masks=torch.stack([torch.as_tensor(p['mask'],device=gt.device).bool() for p in proposals[vi]])
            masks=F.interpolate(masks[:,None].float(),size=gt.shape[-2:],mode='nearest')[:,0].bool()
            result[:,vi]=iou_matrix(masks,gt[:,vi],valid[vi]).max(0).values
        # Per-view oracle; distinct from cross-view linked slot coverage.
        return result

    def close(self):
        self.handle.remove();self.initializer.pool_region=self.original;self.records=[];self.inputs=None


def repeat_calibrated_witness(native_a,native_b,observed):
    """Bound observer differences by native repeat jitter, with fixed hard ceilings.

    Rendering/lifting uses GPU reductions; no bitwise determinism is assumed.
    These bounds certify the observer only and never change prediction thresholds.
    """
    limits={'gaussians':(1e-7,1e-5),'gaussian_membership':(1e-7,2e-5),
            'p_class':(1e-8,1e-6),'region_mass':(1e-7,1e-4)}
    receipt={'policy':'native_repeat_5x_plus_floor_with_fixed_ceiling_v1','tensors':{},'passed':True}
    for key,(floor,ceiling) in limits.items():
        a,b,c=native_a[key],native_b[key],observed[key]
        if a.shape!=b.shape or a.shape!=c.shape:raise ValueError('Witness shape changed: '+key)
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(c).all())
        repeat=float((a-b).abs().max());error=max(float((c-a).abs().max()),float((c-b).abs().max()))
        tolerance=min(ceiling,5*repeat+floor)
        passed=finite and repeat<=ceiling and error<=tolerance
        receipt['tensors'][key]=dict(native_repeat_max_abs=repeat,observer_max_abs=error,tolerance=tolerance,hard_ceiling=ceiling,passed=passed)
        receipt['passed'] &= passed
    a,b,c=native_a['packed'],native_b['packed'],observed['packed']
    repeat=float((a!=b).float().mean());error=max(float((a!=c).float().mean()),float((b!=c).float().mean()))
    ceiling=1e-3;tolerance=min(ceiling,5*repeat+1/a.numel())
    passed=repeat<=ceiling and error<=tolerance
    receipt['packed']=dict(native_repeat_mismatch_fraction=repeat,observer_mismatch_fraction=error,tolerance=tolerance,hard_ceiling=ceiling,passed=passed)
    receipt['passed'] &= passed
    return receipt
