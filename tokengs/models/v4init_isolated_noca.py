"""V4 initialization, geometric No-Anchor-CA, and detached reconstruction reads.

No adapters. Write=0 makes the two-stage ordering forward-equivalent to the
interleaved decoder. Original reconstruction tensors remain attached for LR;
all understanding reads and mask rasterization use detached reconstruction.
"""
import os
import torch
import torch.nn.functional as F
from tokengs.models.olda4_initialization_ablation import (
    OldA4V4InitCore, OldA4GeometricObjectLayer, LocusGSOldA4V4Init,
)
from tokengs.models.superpoint_dual_core import superpoint_scene_normalization
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.canonical_recon import supervised_layer_weights
from tokengs.models.object_locus_panoptic_v1_lift import lift_features

ARCH='OMEGA_V4INIT_NO_ANCHOR_CA_ISOLATED_V4_RECIPE_V1'
KEY='dual_v4init_isolated_noca'


class V4InitIsolatedNoCACore(OldA4V4InitCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        for layer in self.panoptic.layers.values():
            layer.__class__=OldA4GeometricObjectLayer
            del layer.anchor_attention

    def reconstruction_states(self,args,kwargs,memory):
        states,_=self.anchor_decoder.forward_stateful(*args,**kwargs,reconstruction_only=True)
        return states

    def semantic_gaussians(self,state,gaussians):
        return gaussians.detach()

    def decode_state(self,state,memory):
        return self.activation_head(state['tokens'],state['mu'],state['radii'])

    def forward(self,cloud,memory,superpoint_ids=None):
        if superpoint_ids is not None:
            cloud=dict(cloud,point_superpoint_ids=superpoint_ids)
        under=self.initializer.understanding(cloud)
        rec=self.initialize_reconstruction_anchors(None,cloud,memory)
        tokens=rec['tokens'];fm=memory['fm']
        kwargs=dict(seed_mu=rec['centers'],seed_radius=rec['radii'],
            seed_c=under['centers'],seed_s=under['radii'][...,None].expand(-1,-1,3),
            write_strength=0.,normalization_fn=superpoint_scene_normalization)
        args=(tokens,memory['latent'],memory['rays'],self.panoptic,fm,
              under['query'],self.semantic_memory.mask_embedder,0)
        states=self.reconstruction_states(args,kwargs,memory)
        # The existing second-stage API detaches tokens/mu/rho at every layer.
        semantic_states,_=self.anchor_decoder.forward_stateful(*args,**kwargs,reconstruction_states=states)
        protected=('tokens','mu','rho','radii','layer','anchor_update','radius_update','beta')
        for state,semantic in zip(states,semantic_states):
            state.update({k:v for k,v in semantic.items() if k not in protected})
        final=states[-1];semantic=semantic_states[-1]
        gaussians=self.decode_state(final,memory)
        final['gaussians']=gaussians
        semantic_gaussians=self.semantic_gaussians(final,gaussians)
        grid=F.interpolate(fm.flatten(0,1),size=getattr(self,'reconstruction_image_size',None) or self.activation_head.opt.img_size,
            mode='bilinear',align_corners=False).reshape(1,2,256,*(getattr(self,'reconstruction_image_size',None) or self.activation_head.opt.img_size))
        evidence,gate,_=lift_features(grid,semantic_gaussians,memory['decoder'],memory['renderer'])
        child,_=self.panoptic.gaussian_child_features(semantic['anchor_embedding'],semantic['f_anchor'],
            semantic_gaussians,semantic['mu'].detach(),semantic['radii'].detach())
        features=gate*evidence+(1-gate)*child+.1*self.panoptic.W_res(child).tanh()
        q=semantic['q'];n=under['query'].shape[1]
        logits=features@self.semantic_memory.mask_embedder(q).transpose(1,2)
        classification=self.panoptic.classify(q);final.update(classification)
        parent=torch.arange(tokens.shape[1],device=tokens.device).repeat_interleave(64)
        return dict(states=states,gaussians=gaussians,gaussian_membership=logits.sigmoid(),
            gaussian_mask_logits=logits,thing_logits19=classification['thing_logits19'],
            p_class=classification['p_class'],q_pre=under['query'],unified_query=q[:,:n],
            superpoints=under,cloud=cloud,pred_iou=None,gaussian_parent_anchor=parent,
            gaussian_parent_query=parent,gaussian_parent_query_semantics='legacy_alias_of_anchor_index_not_object_query',
            num_thing_queries=n,num_superpoint_queries=n,num_object_queries=n,num_stuff_queries=2,
            num_reconstruction_anchors=tokens.shape[1],num_gaussians=gaussians.shape[1],
            initial_anchor_mu=rec['centers'],initial_anchor_radius=rec['radii'],
            anchor_point_indices=rec['anchor_point_indices'])


class LocusGSV4InitIsolatedNoCA(LocusGSUnifiedQueryV1):
    architecture_name=ARCH
    core_type=V4InitIsolatedNoCACore
    # Exactly the same frozen inputs, source pixels and 9 initialization tensors.
    encode_context=LocusGSOldA4V4Init.encode_context

    def __init__(self,opt,**kwargs):
        if os.environ.get('RG_SUPERVISED') not in (None,'','6,12'):
            raise ValueError('V4 recipe requires L6/L12 supervision')
        for name in ('RG_IOUHEAD','RG_EXTRA','RG_ADAPT','RG_ISO','RG_DEPTH','RG_TSEM'):
            if os.environ.get(name) not in (None,'','0'):
                raise ValueError('Unexpected experiment override: '+name)
        super().__init__(opt,**kwargs)
        self.supervised_layers=(6,12)
        self.layer_weights=supervised_layer_weights(self.supervised_layers)

    def _render_understanding(self,generated,decoder):
        return super()._render_understanding(dict(generated,gaussians=generated['gaussians'].detach()),decoder)

    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        pred=output['prediction']
        for key in ('num_superpoint_queries','num_reconstruction_anchors','num_object_queries','num_gaussians'):
            metrics[key]=pred[key]
        metrics['gradient_isolation']=1
        with torch.no_grad():
            for layer in self.supervised_layers:
                state=pred['states'][layer-1]
                metrics[f'anchor_update_layer{layer}']=state['anchor_update']
                metrics[f'radius_update_layer{layer}']=state['radius_update']
            final=pred['states'][-1]
            metrics['anchor_total_displacement_mean']=(final['mu']-pred['initial_anchor_mu']).norm(dim=-1).mean()
            metrics['anchor_radius_mean']=final['radii'].mean()
            metrics['anchor_radius_ratio_mean']=(final['radii']/pred['initial_anchor_radius'].clamp_min(1e-6)).mean()
        return output,metrics
