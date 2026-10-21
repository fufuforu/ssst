"""Legacy 1024/256 dual streams with genuine 3D superpoint seeds in both."""
import os as _os2
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
        self.anchor_decoder=PointSeededAnchorDecoder(legacy,self.blocks)
        self.anchor_decoder.register_parameter('mu',None);self.anchor_decoder.register_parameter('rho',None)
        self.panoptic=DynamicPanopticController();self.activation_head=LocusGSGaussianHead(legacy)
        import os as _os_iou
        self.iou_head=None
        if _os_iou.environ.get('RG_IOUHEAD') not in (None,'','0'):
            self.iou_head=nn.Sequential(nn.LayerNorm(config.dim),nn.Linear(config.dim,config.dim),nn.GELU(),nn.Linear(config.dim,1))
            nn.init.zeros_(self.iou_head[-1].weight);nn.init.zeros_(self.iou_head[-1].bias)

    def prepare_memory(self,omega,rgb,renderer):
        rays_o,rays_d=pixel_rays(omega['c2w'],omega['k'],256,256)
        return dict(latent=self.reconstruction_memory(omega['patch_layers']),fm=self.semantic_memory(omega['patch_layers'],rgb),
            rays=patch_plucker_rays(rays_o,rays_d,patch_size=8),renderer=renderer,
            decoder=ModelInputDecoder(cam_view=torch.linalg.inv(omega['c2w']).transpose(-1,-2),intrinsics=camera_vectors(omega['k'])))

    def initialize_reconstruction_anchors(self,reconstruction_seed,cloud=None):
        tokens=self.reconstruction_seed_projection(reconstruction_seed['query'])
        n=tokens.shape[1]
        return dict(tokens=tokens,centers=reconstruction_seed['centers'],radii=reconstruction_seed['radii'],
            anchor_to_superpoint=torch.arange(n,device=tokens.device),
            anchor_slot_ids=torch.zeros(n,dtype=torch.long,device=tokens.device))

    def forward(self,cloud,memory,superpoint_ids=None):
        if superpoint_ids is not None:cloud=dict(cloud,point_superpoint_ids=superpoint_ids)
        seed=self.initializer(cloud,cloud['features'],cloud['features']);rec,under=seed['reconstruction'],seed['understanding']
        rec_anchors=self.initialize_reconstruction_anchors(rec,cloud)
        tokens=rec_anchors['tokens'];fm=memory['fm'];radius=under['radii'][...,None].expand(-1,-1,3)
        states,_=self.anchor_decoder.forward_stateful(tokens,memory['latent'],memory['rays'],self.panoptic,fm,under['query'],self.semantic_memory.mask_embedder,0,
            seed_mu=rec_anchors['centers'],seed_radius=rec_anchors['radii'],seed_c=under['centers'],seed_s=radius,write_strength=0.,normalization_fn=superpoint_scene_normalization)
        final=states[-1];q=final['q'];n=q.shape[1]-2
        num_anchors=final['tokens'].shape[1]
        if under['query'].shape[-1]!=256 or n!=under['query'].shape[1]:raise ValueError('Understanding queries must retain native count and 256D width')
        gaussians=self.activation_head(final['tokens'],final['mu'],final['radii'])
        grid=F.interpolate(fm.flatten(0,1),size=self.activation_head.opt.img_size if hasattr(self.activation_head,'opt') else (256,256),mode='bilinear',align_corners=False)
        grid=grid.reshape(1,2,256,*grid.shape[-2:])
        evidence,gate,_=lift_features(grid,gaussians,memory['decoder'],memory['renderer'])
        child,_=self.panoptic.gaussian_child_features(final['anchor_embedding'],final['f_anchor'],gaussians,final['mu'],final['radii'])
        features=gate*evidence+(1-gate)*child+.1*self.panoptic.W_res(child).tanh()
        logits=features@self.semantic_memory.mask_embedder(q).transpose(1,2);classification=self.panoptic.classify(q)
        final.update(classification)
        pred_iou=(torch.sigmoid(self.iou_head(q[:, :n])).squeeze(-1) if self.iou_head is not None else None)
        result=dict(states=states,gaussians=gaussians,gaussian_membership=logits.sigmoid(),gaussian_mask_logits=logits,
            thing_logits19=classification['thing_logits19'],p_class=classification['p_class'],q_pre=under['query'],unified_query=q[:,:n],
            superpoints=under,cloud=cloud,
            anchor_to_superpoint=rec_anchors['anchor_to_superpoint'],anchor_slot_ids=rec_anchors['anchor_slot_ids'],
            gaussian_parent_anchor=torch.arange(num_anchors,device=q.device).repeat_interleave(64),
            gaussian_parent_query=rec_anchors['anchor_to_superpoint'].repeat_interleave(64),
            num_superpoint_queries=under['query'].shape[1],num_reconstruction_anchors=num_anchors,
            num_object_queries=n,num_stuff_queries=2,num_gaussians=gaussians.shape[1],
            recon_qp_slots=getattr(getattr(self,'reconstruction_qp',None),'num_slots',1),
            num_thing_queries=n,pred_iou=pred_iou)
        if 'anchor_point_indices' in rec_anchors:
            result.update(initial_anchor_mu=rec_anchors['centers'],initial_anchor_radius=rec_anchors['radii'])
        for key in ('anchor_point_indices','num_sampled_superpoints','sampled_superpoint_coverage'):
            if key in rec_anchors:result[key]=rec_anchors[key]
        return result

    def parameter_groups(self):
        groups=dict(shared=[],reconstruction=[],understanding=[])
        prefixes=('reconstruction_fps.','reconstruction_qp.','initializer.reconstruction.','reconstruction_seed_projection.','reconstruction_memory.','blocks.','anchor_decoder.','activation_head.')
        for name,p in self.named_parameters():groups['reconstruction' if name.startswith(prefixes) else 'understanding'].append(p)
        return groups
