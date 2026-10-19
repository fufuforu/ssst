"""Legacy 1024/256 dual streams with genuine 3D superpoint seeds in both."""
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.options import config_defaults
from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.dual_stream_decoder import PointSeededAnchorDecoder
from tokengs.models.dual_superpoint_initialization import DualSuperpointInitialization
from tokengs.models.dynamic_panoptic_controller import DynamicPanopticController
from tokengs.models.gfm_dual_stream_components import GFMReconstructionAdapter
from tokengs.models.object_locus_dual_stream_v1 import SemanticMemory
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.canonical_recon_models import patch_plucker_rays
from tokengs.models.locusgs_recon import LocusGSGaussianHead
from tokengs.models.object_locus_posefree_geometry import pixel_rays,camera_vectors
from tokengs.models.input_types import ModelInputDecoder


def superpoint_scene_normalization(mu):
    if mu.ndim!=3 or mu.shape[-1]!=3 or not mu.shape[1]:raise ValueError('Nonempty superpoint centers required')
    origin=mu.mean(1,keepdim=True).detach()
    ell=(mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05).detach()
    return origin,ell


class SuperpointDualCore(nn.Module):
    anchor_decoder_type = PointSeededAnchorDecoder

    def __init__(self,opt,config):
        super().__init__()
        if config.dim!=256 or config.depth!=12 or opt.dec_patch_size!=8:raise ValueError('Legacy dual streams preserve widths1024/256, depth12, children64')
        self.config=config
        legacy=config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4'].evolve(img_size=opt.img_size)
        self.initializer=DualSuperpointInitialization(config,config)
        self.reconstruction_seed_projection=nn.Linear(256,1024)
        self.reconstruction_memory=GFMReconstructionAdapter();self.semantic_memory=SemanticMemory()
        self.blocks=nn.ModuleList([DecoderBlock(1024,legacy.enc_num_heads,legacy.mlp_ratio,qkv_bias=True,ffn_bias=True,qk_norm=True,
            init_values=legacy.dec_init_values if legacy.dec_init_values is not None else 5e-3*legacy.gs_token_std) for _ in range(12)])
        self.anchor_decoder=self.anchor_decoder_type(legacy,self.blocks)
        self.anchor_decoder.register_parameter('mu',None);self.anchor_decoder.register_parameter('rho',None)
        self.panoptic=DynamicPanopticController();self.activation_head=LocusGSGaussianHead(legacy)

    def prepare_memory(self,omega,rgb,renderer):
        rays_o,rays_d=pixel_rays(omega['c2w'],omega['k'],256,256)
        return dict(latent=self.reconstruction_memory(omega['patch_layers']),fm=self.semantic_memory(omega['patch_layers'],rgb),
            rays=patch_plucker_rays(rays_o,rays_d,patch_size=8),renderer=renderer,
            decoder=ModelInputDecoder(cam_view=torch.linalg.inv(omega['c2w']).transpose(-1,-2),intrinsics=camera_vectors(omega['k'])))

    def initialize_streams(self,cloud,memory):
        seed=self.initializer(cloud,cloud['features'],cloud['features']);rec,under=seed['reconstruction'],seed['understanding']
        return self.reconstruction_seed_projection(rec['query']),rec,under

    def parent_metadata(self,tokens,n):
        return dict(gaussian_parent_query=torch.arange(n,device=tokens.device).repeat_interleave(64))

    def decode_streams(self,tokens,rec,under,memory):
        radius=under['radii'][...,None].expand(-1,-1,3)
        states,_=self.anchor_decoder.forward_stateful(tokens,memory['latent'],memory['rays'],self.panoptic,memory['fm'],under['query'],self.semantic_memory.mask_embedder,0,
            seed_mu=rec['centers'],seed_radius=rec['radii'],seed_c=under['centers'],seed_s=radius,write_strength=0.,normalization_fn=superpoint_scene_normalization)
        return states

    def forward(self,cloud,memory,superpoint_ids=None):
        if superpoint_ids is not None:cloud=dict(cloud,point_superpoint_ids=superpoint_ids)
        tokens,rec,under=self.initialize_streams(cloud,memory)
        fm=memory['fm'];states=self.decode_streams(tokens,rec,under,memory)
        final=states[-1];q=final['q'];n=q.shape[1]-2
        gaussians=self.activation_head(final['tokens'],final['mu'],final['radii'])
        grid=F.interpolate(fm.flatten(0,1),size=self.activation_head.opt.img_size if hasattr(self.activation_head,'opt') else (256,256),mode='bilinear',align_corners=False)
        grid=grid.reshape(1,2,256,*grid.shape[-2:])
        evidence,gate,_=lift_features(grid,gaussians,memory['decoder'],memory['renderer'])
        child,_=self.panoptic.gaussian_child_features(final['anchor_embedding'],final['f_anchor'],gaussians,final['mu'],final['radii'])
        features=gate*evidence+(1-gate)*child+.1*self.panoptic.W_res(child).tanh()
        logits=features@self.semantic_memory.mask_embedder(q).transpose(1,2);classification=self.panoptic.classify(q)
        final.update(classification)
        return dict(states=states,gaussians=gaussians,gaussian_membership=logits.sigmoid(),gaussian_mask_logits=logits,
            thing_logits19=classification['thing_logits19'],p_class=classification['p_class'],q_pre=under['query'],unified_query=q[:,:n],
            superpoints=under,cloud=cloud,**self.parent_metadata(tokens,n),num_thing_queries=n)

    def parameter_groups(self):
        groups=dict(shared=[],reconstruction=[],understanding=[])
        prefixes=('initializer.reconstruction.','reconstruction_seed_projection.','reconstruction_memory.','blocks.','anchor_decoder.','activation_head.')
        for name,p in self.named_parameters():groups['reconstruction' if name.startswith(prefixes) else 'understanding'].append(p)
        return groups
