"""Read-only gradient/parameter observation, CPU storage and bounded GPU chunks."""
import math
import torch


def module_group(name):
    for prefix,label in (
        ('vggt_memory_adapter.','memory_adapter'),
        ('understanding.encoder.','mast3r_encoder'),
        ('understanding.adapter.','visual_adapter'),
        ('understanding.mask2former.pixel_decoder.','pixel_decoder'),
        ('understanding.mask2former.transformer_module.','query_decoder'),
        ('understanding.mask2former.mask_embedder.','mask_embedder'),
        ('panoptic.','panoptic_controller'),
        ('enc_dec_backbone.dec','shared_gs_decoder'),
        ('anchor_decoder.','gs_anchor_head'),
        ('activation_head.','gs_activation_head'),
        ('understanding.','understanding_other')):
        if name.startswith(prefix):return label
    return 'reconstruction_other'


class GradientObserver:
    def __init__(self,world_size=4,adapter_scale=.01,chunk_elements=262144):
        self.world_size=world_size;self.adapter_scale=adapter_scale;self.chunk_elements=chunk_elements
        self.rec={};self.under={};self.before={};self.result={}

    def microbatch(self,names,rec,under):
        for name,gr,gu in zip(names,rec,under):
            for dest,g in ((self.rec,gr),(self.under,gu)):
                if g is not None:
                    value=g.detach().cpu()
                    if name in dest:dest[name].add_(value)
                    else:dest[name]=value.clone()

    def before_clip(self,named):
        from scripts.object_locus_frozen_vggt_posefree_runtime import parameter_family
        groups={}
        for name,p in named:
            label=module_group(name)
            stats=groups.setdefault(label,dict(rec_squared=0.,weighted_U_squared=0.,dot=0.,
                combined_squared=0.,combined_error_squared=0.,parameter_squared=0.,parameters=0))
            stats['parameters']+=p.numel()
            before=p.detach().cpu().clone();self.before[name]=before
            stats['parameter_squared']+=float(before.double().square().sum())
            r=self.rec.get(name);u=self.under.get(name)
            scale=self.adapter_scale if name.startswith('vggt_memory_adapter.') else (.01 if parameter_family(name)=='reconstruction' else 1.)
            installed=None if p.grad is None else p.grad.detach().reshape(-1)
            for start in range(0,p.numel(),self.chunk_elements):
                stop=min(p.numel(),start+self.chunk_elements)
                rr=torch.zeros(stop-start,dtype=p.dtype) if r is None else r.reshape(-1)[start:stop]
                uu=torch.zeros_like(rr) if u is None else u.reshape(-1)[start:stop]
                packed=torch.stack((rr,uu)).to(p.device)
                if self.world_size>1:
                    torch.distributed.all_reduce(packed,op=torch.distributed.ReduceOp.SUM)
                    packed.div_(self.world_size)
                gr,gu=packed.cpu().double();del packed
                actual=torch.zeros_like(gr) if installed is None else installed[start:stop].cpu().double()
                stats['rec_squared']+=float(gr.square().sum())
                stats['weighted_U_squared']+=float(gu.square().sum())
                stats['dot']+=float((gr*gu).sum())
                stats['combined_squared']+=float(actual.square().sum())
                stats['combined_error_squared']+=float((actual-gr-scale*gu).square().sum())
        self.rec.clear();self.under.clear()
        for s in groups.values():
            s.update(reconstruction_norm=math.sqrt(s['rec_squared']),
                weighted_understanding_norm=math.sqrt(s['weighted_U_squared']),
                combined_preclip_norm=math.sqrt(s['combined_squared']),
                recombination_error_norm=math.sqrt(s['combined_error_squared']),
                global_R_U_cosine=s['dot']/math.sqrt(s['rec_squared']*s['weighted_U_squared'])
                    if s['rec_squared']>0 and s['weighted_U_squared']>0 else None)
        error=math.sqrt(sum(s['combined_error_squared'] for s in groups.values()))
        norm=math.sqrt(sum(s['combined_squared'] for s in groups.values()))
        if not math.isfinite(error) or error>1e-4*max(norm,1.):
            raise RuntimeError('global R/U recombination differs from actual installed GC gradient')
        self.result=dict(groups=groups,global_combined_preclip_norm_fp64=norm,
            global_recombination_error_norm=error,global_R_U_vectors_averaged=True,
            understanding_gradient_includes_native_warmup_and_accumulation=True,
            gradients_mutated=False)

    def after_clip(self,named,norm):
        groups=self.result['groups']
        for s in groups.values():s['postclip_squared']=0.
        for name,p in named:
            if p.grad is not None:
                groups[module_group(name)]['postclip_squared']+=float(p.grad.detach().cpu().double().square().sum())
        for s in groups.values():s['combined_postclip_norm']=math.sqrt(s['postclip_squared'])
        self.result.update(native_preclip_norm=norm,global_clip_multiplier=min(1.,1./(norm+1e-6)))

    def after_optimizer(self,named,optimizer):
        groups=self.result['groups']
        for s in groups.values():s['actual_update_squared']=0.
        for name,p in named:
            delta=p.detach().cpu()-self.before.pop(name)
            groups[module_group(name)]['actual_update_squared']+=float(delta.double().square().sum())
        for s in groups.values():
            s['actual_parameter_update_norm']=math.sqrt(s['actual_update_squared'])
            s['relative_parameter_update_norm']=math.sqrt(s['actual_update_squared']/s['parameter_squared']) if s['parameter_squared']>0 else None
        self.result['optimizer_group_lrs']={g['name']:g['lr'] for g in optimizer.param_groups}

