"""Two independent OLD A4 controls; joint gradients, original four read/loss layers."""
import inspect
import os
import torch
from torch import nn
from tokengs.options import config_defaults
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.dynamic_panoptic_controller import DynamicObjectLayer
from tokengs.models.dual_stream_initialization import seed_anchors
from tokengs.models.gfm_dual_stream_components import gfm_grid,sample_anchor_gfm

V4_ARCH='OMEGA_OLD_A4_V4_RECON_INITIALIZATION_JOINT_V1'
NOCA_ARCH='OMEGA_OLD_A4_NO_ANCHOR_CA_GEOMETRIC_JOINT_V1'


class V4Initialization(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.gs_tokens=nn.Parameter(opt.gs_token_std*torch.randn(1024,1024))
        self.anchor_content=nn.Sequential(nn.Linear(7,1024),nn.GELU(),nn.Linear(1024,1024))
        self.anchor_visual_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,1024))

    def forward(self,seed):
        mu,radius,color,visual=(seed[k] for k in ('centers','radii','color','visual'))
        if mu.shape!=(1,1024,3) or radius.shape!=(1,1024) or color.shape!=(1,1024,3) or visual.shape!=(1,1024,2048):
            raise ValueError('V4 reconstruction seed dimensions differ')
        origin=mu.mean(1,keepdim=True)
        ell=(mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
        content=torch.cat(((mu-origin)/ell[:,None,None],(radius/ell[:,None])[...,None],color),-1)
        return self.gs_tokens[None].expand(len(mu),-1,-1).clone()+self.anchor_content(content)+self.anchor_visual_projection(visual)


class OldA4V4InitCore(SuperpointDualCore):
    def __init__(self,opt,config):
        # Construct all original modules before adding new parameters to retain RNG order.
        super().__init__(opt,config)
        del self.initializer.reconstruction
        del self.reconstruction_seed_projection
        self.v4_initialization=V4Initialization(config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4'])

    def initialize_branch_queries(self,cloud,memory):
        return dict(reconstruction=None,understanding=self.initializer.understanding(cloud))

    def initialize_reconstruction_anchors(self,reconstruction_seed,cloud=None,memory=None):
        seed=memory['v4_seed'];tokens=self.v4_initialization(seed)
        # Legacy query spelling denotes Gaussian parent ANCHOR, not a superpoint assignment.
        # Loss support matching uses the unchanged understanding source_superpoint_ids.
        return dict(tokens=tokens,centers=seed['centers'],radii=seed['radii'],
            anchor_to_superpoint=torch.arange(1024,device=tokens.device),
            anchor_slot_ids=torch.arange(1024,device=tokens.device),
            anchor_point_indices=seed['source_indices'][0])

    def forward(self,*args,**kwargs):
        result=super().forward(*args,**kwargs)
        result.pop('anchor_to_superpoint')
        result['gaussian_parent_query_semantics']='legacy_alias_of_anchor_index_not_object_query'
        return result

    def parameter_groups(self):
        groups=super().parameter_groups()
        new=list(self.v4_initialization.parameters());ids={id(p) for p in new}
        groups['understanding']=[p for p in groups['understanding'] if id(p) not in ids]
        groups['reconstruction']+=new
        return groups


class OldA4GeometricObjectLayer(DynamicObjectLayer):
    def _read_anchor(self,q,a,bias):
        if not torch.isfinite(bias).all():raise ValueError('Nonfinite geometry bias')
        return torch.zeros_like(q),bias.softmax(-1)[:,None]


class OldA4NoCACore(SuperpointDualCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        for layer in self.panoptic.layers.values():
            layer.__class__=OldA4GeometricObjectLayer
            del layer.anchor_attention


class OldA4DualBase(LocusGSUnifiedQueryV1):
    def encode_context(self,rgb):
        omega,cloud,_=super().encode_context(rgb)
        return omega,cloud,self.core.prepare_memory(omega,rgb,self.gs)


class OldA4Recipe:
    def __init__(self,opt,**kwargs):
        if os.environ.get('RG_SUPERVISED') not in (None,'','6,8,10,12'):
            raise ValueError('Old A4 controls require original L6/8/10/12 supervision')
        if os.environ.get('RG_IOUHEAD') not in (None,'','0'):raise ValueError('IoU head is outside old A4 controls')
        super().__init__(opt,**kwargs)
        if self.supervised_layers!=(6,8,10,12) or self.layer_weights!=[.1,.2,.3,.4]:
            raise ValueError('Old A4 supervision changed')

    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        p=output['prediction']
        for k in ('num_superpoint_queries','num_reconstruction_anchors','num_object_queries','num_gaussians'):
            metrics[k]=p[k]
        metrics['gradient_isolation']=0
        return output,metrics


class LocusGSOldA4V4Init(OldA4Recipe,OldA4DualBase):
    architecture_name=V4_ARCH
    core_type=OldA4V4InitCore

    def encode_context(self,rgb):
        if 'return_source_indices' not in inspect.signature(seed_anchors).parameters:
            raise ValueError('seed_anchors must return exact source indices')
        omega,cloud,_=LocusGSUnifiedQueryV1.encode_context(self,rgb)
        memory=self.core.prepare_memory(omega,rgb,self.gs)
        mu,radius,color,source=seed_anchors(omega['points'],omega['confidence'],rgb,1024,return_source_indices=True)
        memory['v4_seed']=dict(centers=mu,radii=radius,color=color,source_indices=source,
            visual=sample_anchor_gfm(gfm_grid(omega['patch_layers']),source,image_hw=tuple(omega['points'].shape[2:4])))
        return omega,cloud,memory


class LocusGSOldA4NoCA(OldA4Recipe,OldA4DualBase):
    architecture_name=NOCA_ARCH
    core_type=OldA4NoCACore
