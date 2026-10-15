"""Shared Omega GFM -> two adapters; raw-GFM SAM pooling initializes queries."""
import torch
from torch import nn
from tokengs.models.object_locus_dual_stream_v1 import LocusGSDualStreamV1
from tokengs.models.canonical_recon_models import patch_plucker_rays
from tokengs.models.dual_stream_initialization import seed_anchors
from tokengs.models.gfm_dual_stream_components import GFMReconstructionAdapter,GFMSAMQueryInitializer,gfm_grid,sample_anchor_gfm

REMOVED_ENCODER_PREFIXES=('patch_embed.','patch_plucker_embed.','enc_dec_backbone.encoder.','enc_dec_backbone.encoder_norm.','enc_dec_backbone.kv_proj.','enc_dec_backbone.k_proj_norm.','enc_dec_backbone.multiscale_norms.','enc_dec_backbone.latents','enc_dec_backbone.latent_feature_','enc_dec_backbone.latent_blocks.','enc_dec_backbone.latent_kv_proj.')
NEW_GFM_PREFIXES=('reconstruction_adapter.','anchor_visual_projection.','query_initializer.content_projection.')

class LocusGSGFMDualStreamV2(LocusGSDualStreamV1):
    architecture_name='LOCUSGS_OMEGA_GFM_TWO_ADAPTER_SAM_POOL_V2'
    def __init__(self,opt,**kwargs):
        super().__init__(opt,**kwargs)
        self.patch_embed=nn.Identity();self.patch_plucker_embed=nn.Identity()
        backbone=self.enc_dec_backbone;backbone.encoder=nn.Identity()
        for name in ('encoder_norm','kv_proj','k_proj_norm','multiscale_norms','latent_feature_kv_proj','latent_feature_k_norm','latent_blocks','latent_kv_proj'):
            if hasattr(backbone,name):setattr(backbone,name,nn.Identity())
        if hasattr(backbone,'latents'):backbone.register_parameter('latents',None)
        self.reconstruction_adapter=GFMReconstructionAdapter()
        self.query_initializer=GFMSAMQueryInitializer()
        self.anchor_visual_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,1024))

    def query_seed_memory(self,raw_gfm,fm):
        return raw_gfm

    def generate(self,context_rgb,runtime_config=None):
        if runtime_config: raise ValueError('configure architecture at construction; no implicit runtime overrides')
        if context_rgb.ndim!=5 or context_rgb.shape[1:]!=(2,3,256,256):
            raise ValueError('Expected two context RGBs [B,2,3,256,256] in [0,1]')
        if not torch.isfinite(context_rgb).all() or context_rgb.min()<0 or context_rgb.max()>1:
            raise ValueError('context RGB must be finite in [0,1]')
        omega=self.frozen_omega(context_rgb)
        model_input,_=self._predicted_input(context_rgb,None,omega['c2w'],omega['k'])
        latent=self.reconstruction_adapter(omega['patch_layers'])
        # 512 / patch16 = 32 keys per side -> 256 / stride8 context rays.
        rays=patch_plucker_rays(model_input.encoder.rays_os,model_input.encoder.rays_ds,patch_size=8)
        raw_gfm=gfm_grid(omega['patch_layers'])
        mu,radius,color,source_ids=seed_anchors(omega['points'],omega['confidence'],context_rgb,1024,return_source_indices=True)
        origin=mu.mean(1,keepdim=True); ell=(mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
        content=torch.cat(((mu-origin)/ell[:,None,None],(radius/ell[:,None])[...,None],color),-1)
        anchor_visual=sample_anchor_gfm(raw_gfm,source_ids)
        tokens=self.get_gs_tokens(batch_size=len(context_rgb))+self.anchor_content(content)+self.anchor_visual_projection(anchor_visual)
        fm=self.understanding(omega['patch_layers'],context_rgb)
        proposals=self.sam_proposals(context_rgb)
        q,c,s,proposal_log=self.query_initializer(self.query_seed_memory(raw_gfm,fm),proposals,omega['points'],mu)
        states,stats=self.anchor_decoder.forward_stateful(tokens,latent,rays,self.panoptic,fm,q,self.understanding.mask_embedder,self.understanding_step,
            seed_mu=mu,seed_radius=radius,seed_c=c,seed_s=s,write_strength=self.write_strength)
        final=states[-1]; gaussians=self.activation_head(final['tokens'],final['mu'],final['radii'])
        reconstruction=self._reconstruction_from_gaussians(gaussians)
        output=dict(states=states,gaussians=gaussians,reconstruction=reconstruction,render=self.render_reconstruction(reconstruction,model_input.decoder),
            F_m=fm,q_pre=q,ray_stats=stats,beta=final['beta'],predicted_context_c2w=omega['c2w'],
            predicted_context_intrinsics=model_input.decoder.intrinsics,predicted_context_intrinsics_matrix=omega['k'],
            depth=omega['depth'],confidence=omega['confidence'],predicted_points=omega['points'],generation_context_rgb=context_rgb.detach(),
            initial_anchor_mu=mu,initial_anchor_radius=radius,initial_anchor_source_ids=source_ids,initial_query_centers=c,initial_query_scales=s,
            proposal_initialization=proposal_log,omega_source_identity=self.frozen_omega.source_identity)
        output.update(self._readout(final,gaussians,fm,model_input.decoder,model_input.decoder))
        return output

