"""Point-seeded reconstruction with strictly isolated residual understanding reads."""
import os
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.superpoint_dual_core import SuperpointDualCore,superpoint_scene_normalization
from tokengs.models.object_locus_panoptic_v1_lift import lift_features


class DetachedResidualAdapter(nn.Module):
    """Identity at initialization; LU trains this adapter, never its input."""
    def __init__(self, dim=1024, hidden=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,dim))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,tokens):
        x=tokens.detach()
        return x+self.net(x)


class Point1024IsolatedAdapterCore(SuperpointDualCore):
    read_layers=(6,8,10,12)

    def __init__(self,opt,config):
        for name in ('RG_IOUHEAD','RG_EXTRA','RG_ADAPT','RG_ISO'):
            if os.environ.get(name) not in (None,'','0'):
                raise ValueError('Explicit isolated architecture does not mix legacy overrides: '+name)
        super().__init__(opt,config)
        del self.initializer.reconstruction
        self.understanding_adapters=nn.ModuleDict({f'L{i}':DetachedResidualAdapter() for i in self.read_layers})

    def forward(self,cloud,memory,superpoint_ids=None):
        if superpoint_ids is not None:cloud=dict(cloud,point_superpoint_ids=superpoint_ids)
        # Reuse geometry/SAM/fusion identities; initialize from semantic memory.
        # Keep this gradient: LU must train FM through the query seed path.
        under_features=memory['understanding_point_features']
        if under_features.shape!=(len(cloud['points']),256):raise ValueError('FM features must match existing fused point order')
        under=self.initializer.understanding(dict(cloud,features=under_features))
        rec=memory['point_seed'];tokens=self.reconstruction_seed_projection(rec['features'])
        fm=memory['fm']
        # Write=0 allows the reconstruction and object computations to be scheduled
        # separately. Only original reconstruction tensors are stored as state tokens.
        states,_=self.anchor_decoder.forward_stateful(tokens,memory['latent'],memory['rays'],
            self.panoptic,fm,under['query'],self.semantic_memory.mask_embedder,0,
            seed_mu=rec['centers'],seed_radius=rec['radii'],seed_c=under['centers'],
            seed_s=under['radii'][...,None].expand(-1,-1,3),write_strength=0.,
            reconstruction_only=True,normalization_fn=superpoint_scene_normalization)
        image=F.interpolate(fm.flatten(0,1),size=(32,32),mode='bilinear',align_corners=False)
        image=image.reshape(1,2,256,1024).permute(0,1,3,2).reshape(1,2048,256)
        q=c=s=ell=origin=None
        for layer in self.read_layers:
            state=states[layer-1]
            mu=state['mu'].detach();radii=state['radii'].detach()
            h=self.understanding_adapters[f'L{layer}'](state['tokens'])
            if layer==6:
                origin,ell=superpoint_scene_normalization(mu)
                stuff=self.panoptic.stuff_ln(fm.mean((1,3,4))[:,None]+self.panoptic.stuff_seed[None])
                q=torch.cat((under['query'],stuff),1)
                c=under['centers'];s=under['radii'][...,None].expand(-1,-1,3)
            a=self.panoptic.encode_token(h,mu,radii,ell)
            f_anchor=self.panoptic.ln_mask_a(self.panoptic.W_mask_a(a))
            result=self.panoptic.layers[f'L{layer}'](h,a,mu,q,c,s,ell,image,
                self.semantic_memory.mask_embedder,f_anchor,0)
            q,c,s=result['q'],result['c'],result['s']
            # The write path is disabled. Keep original reconstruction state tokens.
            result['joint_delta']=result['joint_delta']*0.
            state.update(result,scene_origin=origin)
        final=states[-1];n=under['query'].shape[1];anchors=tokens.shape[1]
        gaussians=self.activation_head(final['tokens'],final['mu'],final['radii'])
        semantic_gaussians=gaussians.detach()
        grid=F.interpolate(fm.flatten(0,1),size=self.activation_head.opt.img_size,
            mode='bilinear',align_corners=False).reshape(1,2,256,*self.activation_head.opt.img_size)
        evidence,gate,_=lift_features(grid,semantic_gaussians,memory['decoder'],memory['renderer'])
        child,_=self.panoptic.gaussian_child_features(final['anchor_embedding'],final['f_anchor'],
            semantic_gaussians,final['mu'].detach(),final['radii'].detach())
        features=gate*evidence+(1-gate)*child+.1*self.panoptic.W_res(child).tanh()
        logits=features@self.semantic_memory.mask_embedder(q).transpose(1,2)
        classification=self.panoptic.classify(q);final.update(classification)
        return dict(states=states,gaussians=gaussians,gaussian_membership=logits.sigmoid(),gaussian_mask_logits=logits,
            thing_logits19=classification['thing_logits19'],p_class=classification['p_class'],q_pre=under['query'],
            unified_query=q[:,:n],superpoints=under,cloud=cloud,pred_iou=None,
            understanding_point_features=under_features,
            gaussian_parent_anchor=torch.arange(anchors,device=tokens.device).repeat_interleave(64),
            num_thing_queries=n,num_superpoint_queries=n,num_object_queries=n,num_stuff_queries=2,
            num_reconstruction_anchors=anchors,num_gaussians=gaussians.shape[1],
            initial_anchor_mu=rec['centers'],initial_anchor_radius=rec['radii'],initial_anchor_source_ids=rec['source_indices'])
