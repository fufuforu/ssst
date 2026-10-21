"""Observation and explicit post-initializer interventions for fixed-weight audits."""
import hashlib
import torch
import torch.nn.functional as F

VARIANTS=('native','embedding_fallback','geometry_global','both_fallback')


def intervene(initializer,inputs,output,variant):
    if variant not in VARIANTS:raise ValueError('Unknown initializer intervention')
    if variant=='native':return output
    q,c,s,logs=output;mu=inputs[3]
    q=q.clone();c=c.clone();s=s.clone()
    for b,log in enumerate(logs):
        n=log['proposal_queries']
        if variant in ('embedding_fallback','both_fallback'):q[b,:n]=initializer.norm(initializer.fallback)[:n]
        if variant in ('geometry_global','both_fallback'):
            origin=mu[b].mean(0);ell=(mu[b]-origin).square().sum(-1).mean().sqrt().clamp_min(.05).detach()
            c[b,:n]=origin;s[b,:n]=ell
    return q,c,s,logs


def candidate_hashes(masks):
    return [hashlib.sha256(m.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for m in masks]


def group_norms(grad,n):
    norms=grad.detach().float().norm(dim=-1)[0]
    def summarize(x):return dict(count=len(x),sum=float(x.sum()),mean=float(x.mean()) if len(x) else None)
    return dict(per_query=norms.cpu().tolist(),SAM=summarize(norms[:n]),fallback=summarize(norms[n:]))


class InitializationObserver:
    """Read-only hooks, capturing live initializer output only for gradient probes."""
    def __init__(self,model):
        self.model=model;self.handles=[];self.active=False;self.components={};self.layers={};self.seed=None
        def pre(module,inputs):self.active=True;self.components={};self.layers={};self.seed=None
        def component(name):
            def hook(module,inputs,output):
                if self.active:self.components.setdefault(name,[]).append(output.detach().clone())
            return hook
        def post(module,inputs,output):
            self.active=False;self.seed=output
        init=model.query_initializer
        self.handles.append(init.register_forward_pre_hook(pre))
        self.handles.append(init.content_projection.register_forward_hook(component('content_projection')))
        self.handles.append(init.geometry.register_forward_hook(component('geometry_encoding')))
        for name in ('point_weight','shape_projection'):
            if hasattr(init,name):self.handles.append(getattr(init,name).register_forward_hook(component(name)))
        self.handles.append(init.register_forward_hook(post))
        for name,layer in model.panoptic.layers.items():
            def layer_hook(module,inputs,output,name=name):
                mu,q,c,s,ell=inputs[2],inputs[3],inputs[4],inputs[5],inputs[6]
                distance=((mu[:,None]-c[:,:,None])/s[:,:,None]).square().sum(-1)[0]
                read=output['evidence_attention'][0].mean(0)[:100]
                entropy=-(read*read.clamp_min(1e-12).log()).sum(-1)
                self.layers[name]=dict(input_centers=c[0].detach().cpu().tolist(),input_scales=s[0].detach().cpu().tolist(),
                    scales_over_scene=(s[0]/ell[0]).detach().cpu().tolist(),
                    anchors_within_3sigma=(distance<=9).sum(-1).detach().cpu().tolist(),
                    attention_mass_within_3sigma=(read*(distance<=9)).sum(-1).detach().cpu().tolist(),
                    attention_effective_anchors=entropy.exp().detach().cpu().tolist(),
                    attention_entropy=(entropy/torch.log(read.new_tensor(read.shape[-1]))).detach().cpu().tolist())
            self.handles.append(layer.register_forward_hook(layer_hook))

    def initial_stats(self,region_records=None):
        q,c,s,logs=self.seed;n=logs[0]['proposal_queries'];init=self.model.query_initializer
        if len(self.components.get('content_projection',[]))!=n or len(self.components.get('geometry_encoding',[]))!=n:raise ValueError('Initializer component hook count differs')
        content=torch.stack(self.components['content_projection']) if n else q.new_empty((0,256))
        geometry=torch.stack(self.components['geometry_encoding']) if n else q.new_empty((0,256))
        fallback=init.fallback[:n].detach();pc=self.model.panoptic.classify(q)['p_class'][0]
        region_stats=[]
        if region_records is not None and hasattr(init,'point_weight'):
            weights=self.components.get('point_weight',[]);shapes=self.components.get('shape_projection',[])
            if len(weights)!=len(region_records) or len(shapes)!=len(region_records):raise ValueError('Geometric pooling observation count differs')
            for (view,mask,feature),weight,shape in zip(region_records,weights,shapes):
                semantic=feature-shape
                region_stats.append(dict(view=view,mask_sha256=hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),mask_pixels=int(mask.sum()),sampled_points=weight.numel(),
                    sigmoid_weight_mean=float(weight.mean()),sigmoid_weight_min=float(weight.min()),sigmoid_weight_max=float(weight.max()),
                    sigmoid_weight_saturated_fraction=float(((weight<.01)|(weight>.99)).float().mean()),
                    shape_projection_norm=float(shape.norm()),weighted_semantic_norm=float(semantic.norm()),
                    shape_semantic_cosine=float(F.cosine_similarity(shape,semantic,dim=0))))
        return dict(geometric_pooling_regions=region_stats,content_norm=content.norm(dim=-1).cpu().tolist(),geometry_encoding_norm=geometry.norm(dim=-1).cpu().tolist(),
            learned_base_norm=init.fallback.detach().norm(dim=-1).cpu().tolist(),
            content_geometry_cosine=F.cosine_similarity(content,geometry,dim=-1).cpu().tolist(),
            seed_learned_cosine=F.cosine_similarity(q[0,:n].detach(),init.norm(init.fallback)[:n].detach(),dim=-1).cpu().tolist(),
            initial_class_probabilities=pc.detach().cpu().tolist())

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.seed=None;self.components={};self.layers={}
