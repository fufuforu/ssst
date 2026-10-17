"""Pose-free unified superpoint queries for reconstruction and instance understanding.

Independent architecture: no old Anchor/Object decoder, no random query table,
no MASt3R weights, no SAM dependency, and no implicit checkpoint migration.
"""
import torch
from torch import nn
from tokengs.models.dual_stream_frozen_backbones import FrozenOmega
from tokengs.models.gfm_dual_stream_components import gfm_grid
from tokengs.models.unified_superpoints import UnifiedQueryConfig,lift_context_cloud
from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.object_locus_posefree_geometry import camera_vectors,pixel_rays
from tokengs.models.object_locus_v3_set import alpha_normalize_membership
from tokengs.models.canonical_recon_models import LocusGSRecon,_full_supervision
from tokengs.models.canonical_recon import supervised_layer_weights
from tokengs.models.tokengs import TokenGS


class LocusGSUnifiedQueryV1(nn.Module):
    architecture_name='OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_V1'
    reconstruction_only=False

    def __init__(self,opt,*,config=None,omega=None,renderer=None,omega_checkpoint='/datasets2/mengxl/vggt_omega/vggt_omega_1b_512.pt'):
        super().__init__();self.config=config or UnifiedQueryConfig();self.opt=opt
        self.img_size=tuple(opt.img_size)
        self.frozen_omega=omega if omega is not None else FrozenOmega(omega_checkpoint)
        self.frozen_omega.requires_grad_(False).eval()
        self.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,self.config.dim))
        self.core=UnifiedQueryCore(opt,self.config)
        if renderer is None:
            from tokengs.rendering.gs import GaussianRenderer
            renderer=GaussianRenderer(opt)
        self.gs=renderer
        self.supervised_layers=tuple(x for x in (6,8,10,12) if x<=self.config.depth)
        if not self.supervised_layers or self.supervised_layers[-1]!=self.config.depth:self.supervised_layers=tuple(sorted(set(self.supervised_layers+(self.config.depth,))))
        self.layer_weights=supervised_layer_weights(self.supervised_layers);self.understanding_step=0

    @property
    def activation_head(self):return self.core.activation_head
    _background_color=TokenGS._background_color
    _reconstruction_from_gaussians=TokenGS._reconstruction_from_gaussians
    render_reconstruction=TokenGS.render_reconstruction
    _layer_objective=LocusGSRecon._layer_objective

    def train(self,mode=True):
        super().train(mode);self.frozen_omega.eval();return self

    def _render_understanding(self,generated,decoder):
        render=self.render_reconstruction(generated['reconstruction'],decoder)
        rendered=self.gs.render_feature_channels(generated['gaussians'],generated['gaussian_membership'],decoder.cam_view,decoder.intrinsics)
        alpha=rendered['alphas_pred'];regions=alpha_normalize_membership(rendered['images_pred'],alpha)
        n=generated['num_thing_queries'];p=generated['p_class']
        semantic=regions.new_zeros((1,regions.shape[1],20,*regions.shape[-2:]))
        semantic[:,:,:2]=regions[:,:,n:n+2]
        semantic[:,:,2:]=torch.einsum('bvqhw,bqc->bvchw',regions[:,:,:n],p[...,:18])
        semantic=semantic/(semantic.sum(2,keepdim=True)+1e-6)
        return dict(**render,region_mass=regions,pixel_membership=regions,alpha=alpha,semantic_scores=semantic,p_class=p,
            states=generated['states'],gaussians=generated['gaussians'],gaussian_membership=generated['gaussian_membership'],num_thing_queries=n)

    def encode_context(self,context_rgb):
        if context_rgb.shape!=(1,2,3,256,256):raise ValueError('Expected one two-view 256 scene')
        if not torch.isfinite(context_rgb).all() or context_rgb.min()<0 or context_rgb.max()>1:raise ValueError('Finite RGB in[0,1] required')
        omega=self.frozen_omega(context_rgb)
        raw=gfm_grid(omega['patch_layers']);grid=self.gfm_projection(raw.permute(0,1,3,4,2)).permute(0,1,4,2,3).contiguous()
        cloud=lift_context_cloud(omega,grid,context_rgb,self.config)
        memory=grid.permute(0,1,3,4,2).reshape(1,-1,self.config.dim)
        return omega,cloud,memory

    def generate(self,context_rgb,*,superpoint_ids=None):
        if context_rgb.shape!=(1,2,3,256,256):raise ValueError('Unified v1 uses one scene per rank: [1,2,3,256,256]')
        if not torch.isfinite(context_rgb).all() or context_rgb.min()<0 or context_rgb.max()>1:raise ValueError('Finite RGB in[0,1] required')
        omega,cloud,memory=self.encode_context(context_rgb)
        output=self.core(cloud,memory,superpoint_ids)
        decoder=ModelInputDecoder(cam_view=torch.linalg.inv(omega['c2w']).transpose(-1,-2),intrinsics=camera_vectors(omega['k']))
        output.update(reconstruction=self._reconstruction_from_gaussians(output['gaussians']),predicted_context_c2w=omega['c2w'],
            predicted_context_intrinsics=decoder.intrinsics,predicted_context_intrinsics_matrix=omega['k'],predicted_points=omega['points'],
            depth=omega['depth'],confidence=omega['confidence'],generation_context_rgb=context_rgb.detach(),omega_source_identity=self.frozen_omega.source_identity)
        rendered=self._render_understanding(output,decoder)
        output.update(render={k:rendered[k] for k in ('images_pred','depths_pred','alphas_pred')},
            **{k:rendered[k] for k in ('region_mass','pixel_membership','alpha','semantic_scores')})
        return output

    @torch.no_grad()
    def calibrate_targets(self,context_and_target_rgb,generated):
        if not torch.equal(context_and_target_rgb[:,:2],generated['generation_context_rgb']):raise ValueError('Calibration context mismatch')
        from tokengs.models.dual_stream_calibration import align_omega_targets
        return align_omega_targets(self.frozen_omega(context_and_target_rgb),generated)

    def render_generated_at(self,generated,cam_view,intrinsics):
        return self._render_understanding(generated,ModelInputDecoder(cam_view=cam_view,intrinsics=intrinsics))

    def step_loss(self,batch,*,step=0,phase='train',understanding_weight=1.):
        from tokengs.models.unified_query_loss import unified_instance_loss
        if understanding_weight<0:raise ValueError('Negative understanding loss weight')
        self.understanding_step=int(step);prediction=self.generate(batch['images_input'])
        calibration=self.calibrate_targets(batch['images_all'],prediction)
        decoder=ModelInputDecoder(cam_view=torch.linalg.inv(calibration['c2w']).transpose(-1,-2),intrinsics=calibration['intrinsics'])
        rays_o,rays_d=pixel_rays(calibration['c2w'],calibration['intrinsics_matrix'],256,256)
        loss_batch=dict(batch,rays_os=rays_o,rays_ds=rays_d)
        reconstruction,metrics,*_=self._layer_objective(prediction['states'],decoder,_full_supervision(loss_batch))
        understanding,under_metrics=unified_instance_loss(prediction,batch,self.config)
        metrics.update(under_metrics,loss_recon=reconstruction,loss_understanding=understanding,loss=reconstruction+understanding_weight*understanding,
            psnr=metrics[f'psnr_layer{self.supervised_layers[-1]}'],num_thing_queries=prediction['num_thing_queries'])
        prediction['target_camera_calibration']=calibration
        return dict(prediction=prediction),metrics

    def forward(self,context_rgb,skip_loss=False):
        if isinstance(context_rgb,dict):return self.step_loss(context_rgb,step=self.understanding_step)[0]
        return self.generate(context_rgb)

    def parameter_groups(self):
        result=self.core.parameter_groups();result['shared']=list(self.gfm_projection.parameters())+result['shared']
        return result
