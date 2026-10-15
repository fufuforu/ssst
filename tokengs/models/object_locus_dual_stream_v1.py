"""VGGT point-seeded reconstruction + SAM region-seeded understanding.

New model: no implicit restoration of old pretrained understanding checkpoints.
Reuse proven TokenGS encoder, Gaussian head, child lifting and losses explicitly.
"""
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.canonical_recon_models import LocusGSRecon,patch_plucker_rays
from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
from tokengs.models.object_locus_frozen_vggt_posefree import LocusGSObjectLocusFrozenVGGT
from tokengs.models.object_locus_panoptic_v1_controller import ObjectLocusPanopticV1Controller
from tokengs.models.dual_stream_decoder import PointSeededAnchorDecoder
from tokengs.models.dual_stream_initialization import SAMQueryInitializer,seed_anchors
from tokengs.models.dual_stream_frozen_backbones import FrozenOmega,FrozenSAM

class SemanticMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256)) for _ in range(4)])
        self.layer_logits=nn.Parameter(torch.zeros(4))
        # High-resolution local RGB residual; randomly initialized, no MASt3R.
        self.local=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.GELU(),nn.Conv2d(64,256,3,padding=1))
        self.refine=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.GELU(),nn.Conv2d(256,256,1))
        self.mask_embedder=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256))
    def forward(self,layers,rgb):
        b,v=rgb.shape[:2]
        patches=sum(w*proj(x.float()) for w,proj,x in zip(self.layer_logits.softmax(0),self.projections,layers))
        grid=patches.reshape(b*v,32,32,256).permute(0,3,1,2)
        grid=F.interpolate(grid,size=(128,128),mode='bilinear',align_corners=False)
        local=self.local(F.interpolate(rgb.flatten(0,1),size=(128,128),mode='bilinear',align_corners=False))
        return self.refine(grid+local).reshape(b,v,256,128,128)

class LocusGSDualStreamV1(LocusGSRecon):
    architecture_name='LOCUSGS_OMEGA_POINT_SAM_DUAL_STREAM_V1'
    reconstruction_only=False
    def __init__(self,opt,*,omega_checkpoint='/datasets2/mengxl/vggt_omega/vggt_omega_1b_512.pt',sam_checkpoint='/space/mawb/ssst_dual_stream_assets/sam_vit_b_01ec64.pth',write_strength=0.):
        if opt.enc_embed_dim!=1024 or opt.num_gs_tokens!=1024 or opt.dec_depth!=12:
            raise ValueError('v1 reuses 1024 anchors, 1024 reconstruction width, 12 layers')
        if opt.num_dynamic_gs_tokens or opt.locusgs_freeze_decode_radius:
            raise ValueError('Point-seeded v1 needs static token count and adaptive decode radii')
        if not 0<=write_strength<=1: raise ValueError('write_strength must be in [0,1]')
        super().__init__(opt)
        self.reconstruction_only=False
        self.anchor_decoder.__class__=PointSeededAnchorDecoder
        # These learned scene-independent centers/radii are replaced, not optimized unused.
        self.anchor_decoder.register_parameter('mu',None); self.anchor_decoder.register_parameter('rho',None)
        self.frozen_omega=FrozenOmega(omega_checkpoint)
        self.sam_proposals=FrozenSAM(sam_checkpoint)
        self.understanding=SemanticMemory()
        self.query_initializer=SAMQueryInitializer()
        self.panoptic=ObjectLocusPanopticV1Controller()
        self.anchor_content=nn.Sequential(nn.Linear(7,1024),nn.GELU(),nn.Linear(1024,1024))
        self.write_strength=write_strength
        self.state_layers=(6,8,10,12)

    # Exact existing heads/lifting/rendering; no duplicate implementation or hidden pretraining.
    _readout=LocusGSObjectLocusPanopticV1Recon._readout
    _predicted_input=LocusGSObjectLocusFrozenVGGT._predicted_input
    render_generated_at=LocusGSObjectLocusFrozenVGGT.render_generated_at

    def generate(self,context_rgb,runtime_config=None):
        if runtime_config: raise ValueError('configure architecture at construction; no implicit runtime overrides')
        if context_rgb.ndim!=5 or context_rgb.shape[1:]!=(2,3,256,256):
            raise ValueError('Expected two context RGBs [B,2,3,256,256] in [0,1]')
        if not torch.isfinite(context_rgb).all() or context_rgb.min()<0 or context_rgb.max()>1:
            raise ValueError('context RGB must be finite in [0,1]')
        omega=self.frozen_omega(context_rgb)
        model_input,_=self._predicted_input(context_rgb,None,omega['c2w'],omega['k'])
        # Deliberately restore RGB + predicted Pluecker TokenGS memory, unlike old VGGT-only path.
        latent=self.forward_encoder(model_input.encoder)
        rays=patch_plucker_rays(model_input.encoder.rays_os,model_input.encoder.rays_ds,patch_size=int(self.opt.patch_size))
        mu,radius,color=seed_anchors(omega['points'],omega['confidence'],context_rgb,1024)
        origin=mu.mean(1,keepdim=True); ell=(mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
        content=torch.cat(((mu-origin)/ell[:,None,None],(radius/ell[:,None])[...,None],color),-1)
        tokens=self.get_gs_tokens(batch_size=len(context_rgb))+self.anchor_content(content)
        fm=self.understanding(omega['patch_layers'],context_rgb)
        proposals=self.sam_proposals(context_rgb)
        q,c,s,proposal_log=self.query_initializer(fm,proposals,omega['points'],mu)
        states,stats=self.anchor_decoder.forward_stateful(tokens,latent,rays,self.panoptic,fm,q,self.understanding.mask_embedder,self.understanding_step,
            seed_mu=mu,seed_radius=radius,seed_c=c,seed_s=s,write_strength=self.write_strength)
        final=states[-1]; gaussians=self.activation_head(final['tokens'],final['mu'],final['radii'])
        reconstruction=self._reconstruction_from_gaussians(gaussians)
        output=dict(states=states,gaussians=gaussians,reconstruction=reconstruction,render=self.render_reconstruction(reconstruction,model_input.decoder),
            F_m=fm,q_pre=q,ray_stats=stats,beta=final['beta'],predicted_context_c2w=omega['c2w'],
            predicted_context_intrinsics=model_input.decoder.intrinsics,predicted_context_intrinsics_matrix=omega['k'],
            depth=omega['depth'],confidence=omega['confidence'],predicted_points=omega['points'],generation_context_rgb=context_rgb.detach(),
            initial_anchor_mu=mu,initial_anchor_radius=radius,initial_query_centers=c,initial_query_scales=s,
            proposal_initialization=proposal_log,omega_source_identity=self.frozen_omega.source_identity)
        output.update(self._readout(final,gaussians,fm,model_input.decoder,model_input.decoder))
        return output

    def forward(self,context_rgb,skip_loss=False):
        del skip_loss
        if isinstance(context_rgb,dict): return self.step_loss(context_rgb,step=self.understanding_step)[0]
        if not torch.is_tensor(context_rgb): raise TypeError('Use generate(context_rgb); no GT-camera ModelInput forward')
        return self.generate(context_rgb)

    @torch.no_grad()
    def calibrate_targets(self,context_and_target_rgb,generated):
        from tokengs.models.dual_stream_calibration import align_omega_targets
        if not torch.equal(context_and_target_rgb[:,:2],generated['generation_context_rgb']):
            raise ValueError('Calibration must share identical contexts')
        return align_omega_targets(self.frozen_omega(context_and_target_rgb),generated)

    step_loss=LocusGSObjectLocusFrozenVGGT.step_loss
