"""QPanchor1 fixed anchors/XYZ offsets; optional Context-only global scale, GT render-only poses."""
import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from tokengs.models.qp_anchor_joint4 import AnchorQPJoint4,scale_gaussians
from tokengs.rendering.sh1 import SH1Renderer
from tokengs.models.object_locus_v3_set import alpha_normalize_membership
from tokengs.models.unified_query_nms import context_mask_nms
from native_scannet.core import schedule,ssim_loss,visibility,chamfer
from scripts.unposed_gt_protocol import validate_K,context_geometry,gt_decoder,camera_audit,NEAR_PUBLIC,FAR_PUBLIC

class AnchorQPUnposedGT(AnchorQPJoint4):
    def __init__(self,*args,scene_scale_head=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.scene_scale_head_enabled=bool(scene_scale_head)
        self.architecture_name='QP_ANCHOR1_UNPOSED_GT_SCALE_'+('ON' if self.scene_scale_head_enabled else 'OFF')+'_V1'
        # Preserve every original parameter initialization AND subsequent RNG clock in both arms.
        if self.scene_scale_head_enabled:
            with torch.random.fork_rng(devices=[]):
                head=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,1))
                nn.init.zeros_(head[1].weight);nn.init.zeros_(head[1].bias)
            self.core.reconstruction.scene_scale_head=head
        self.metric_gs=SH1Renderer(self.opt.evolve(znear=NEAR_PUBLIC,zfar=FAR_PUBLIC))
        self._context_intrinsics=None
        original=self.frozen_omega.forward
        def context_only(rgb):
            if self._context_intrinsics is None or rgb.shape[1]!=2:raise RuntimeError('Bind Context RGB+K; no Target images/GT poses')
            if not torch.equal(rgb,self._context_rgb):raise ValueError('Context RGB identity differs')
            omega=context_geometry(original(rgb),self._context_intrinsics)
            self._scale_features=omega['patch_layers'][-1].float().mean((1,2)).detach()
            return omega
        self.frozen_omega.forward=context_only

    @contextmanager
    def context_inputs(self,rgb,intrinsics):
        validate_K(intrinsics)
        if self._context_intrinsics is not None:raise RuntimeError('Nested Context binding')
        self._context_intrinsics=intrinsics.detach();self._context_rgb=rgb.detach();self._scale_features=None
        try:yield
        finally:self._context_intrinsics=None;self._context_rgb=None;self._scale_features=None

    def generate(self,*args,**kwargs):
        raise RuntimeError('Use generate_unposed(Context_RGB,Context_K); no pose argument')

    def generate_unposed(self,context_rgb,context_intrinsics):
        with self.context_inputs(context_rgb,context_intrinsics):
            pred=super().generate(context_rgb)
            if self.scene_scale_head_enabled:
                raw=self.core.reconstruction.scene_scale_head(self._scale_features)
                scale=.001+F.softplus(raw+math.log(math.expm1(.999)))
            else:scale=pred['gaussians'].new_ones((1,1))
            pred['scene_scale']=scale
            return pred

    def calibrate_targets(self,*args,**kwargs):
        raise RuntimeError('Target RGB camera calibration forbidden')

    def physical_gaussians(self,pred):
        return scale_gaussians(pred['gaussians'],pred['scene_scale'].reshape(-1,1,1))

    def render_generated_gt(self,pred,batch):
        decoder=gt_decoder(batch);g=self.physical_gaussians(pred)
        rgb=self.metric_gs.render(g,decoder.cam_view,g.new_zeros(3),decoder.intrinsics)
        rgb['images_pred']=rgb['images_pred'].clamp(0,1)
        rgb.update(self.render_gt_features(pred,decoder))
        rgb.update(p_class=pred['p_class'],num_thing_queries=pred['num_thing_queries'])
        return rgb

    def render_gt_features(self,pred,decoder):
        feature=self.metric_gs.render_feature_channels(self.physical_gaussians(pred).detach(),pred['gaussian_membership'],decoder.cam_view,decoder.intrinsics)
        alpha=feature['alphas_pred'];regions=alpha_normalize_membership(feature['images_pred'],alpha)
        n=pred['num_thing_queries'];p=pred['p_class']
        semantic=regions.new_zeros((1,regions.shape[1],20,*regions.shape[-2:]))
        semantic[:,:,:2]=regions[:,:,n:n+2]
        semantic[:,:,2:]=torch.einsum('bvqhw,bqc->bvchw',regions[:,:,:n],p[...,:18])
        semantic=semantic/(semantic.sum(2,keepdim=True)+1e-6)
        eligible,ranking=context_mask_nms(p,regions[:,:2],alpha[:,:2],self.config.mask_nms_iou)
        return dict(region_mass=regions,pixel_membership=regions,alpha=alpha,semantic_scores=semantic,
            instance_eligible=eligible,instance_ranking_scores=ranking)

    def step_loss(self,batch,*,step=0,phase='train',understanding_weight=1.):
        from tokengs.models.unified_query_loss import unified_instance_loss
        self.understanding_step=int(step)
        pred=self.generate_unposed(batch['images_input'],batch['intrinsics_all'][:,:2])
        decoder=gt_decoder(batch);g_phys=self.physical_gaussians(pred)
        rendered=self.metric_gs.render(g_phys,decoder.cam_view,g_phys.new_zeros(3),decoder.intrinsics)
        images=rendered['images_pred'].clamp(0,1);truth=batch['images_all'];flat=images.flatten(0,1);gt=truth.flatten(0,1)
        g=pred['native_gaussians'];weights=schedule(step,4172)
        row=dict(cam_view=decoder.cam_view,intrinsics=decoder.intrinsics)
        terms=dict(l1=F.l1_loss(images,truth),ssim=ssim_loss(flat,gt),visibility=visibility(g_phys[0,:,:3],row,near=NEAR_PUBLIC))
        # This prior remains in INTERNAL normalized units for both arms; it cannot shrink just by changing the scene scale.
        points=pred['predicted_points'].reshape(-1,3)*4;points=points[torch.isfinite(points).all(-1)]
        cloud=points[torch.linspace(0,len(points)-1,4096,device=points.device).long()]
        terms['chamfer']=chamfer(g[0,:,:3],cloud) if weights['chamfer'] else images.new_zeros(())
        terms['opacity']=F.relu(math.log(.1)-g[...,3].clamp_min(1e-8).log()).mean()
        with torch.autocast(images.device.type,enabled=False):
            if weights['lpips'] and self.perceptual is None:raise ValueError('Frozen LPIPS missing')
            terms['lpips']=self.perceptual(flat.float()*2-1,gt.float()*2-1).mean() if weights['lpips'] else images.new_zeros(())
        lr=terms['l1']+.2*terms['ssim']+terms['visibility']+weights['chamfer']*terms['chamfer']+weights['opacity']*terms['opacity']+weights['lpips']*terms['lpips']
        # Detach the physical GS including scene scale for LU; only membership/classification remain trainable here.
        pred.update(self.render_gt_features(pred,gt_decoder(batch,2)))
        lu,um=unified_instance_loss(pred,batch,self.config)
        mse=(images-truth).square().flatten(2).mean(-1);psnr=-10*mse.clamp_min(1e-12).log10()
        metrics=dict(um,**terms,loss_recon=lr,loss_understanding=lu,loss=lr+understanding_weight*lu,
            psnr=psnr.mean(),psnr_context=psnr[:,:2].mean(),psnr_novel=psnr[:,2:].mean(),scene_scale=pred['scene_scale'].mean(),
            gradient_isolation=1,num_gaussians=g.shape[1],num_reconstruction_anchors=self.core.reconstruction.opt.num_gs_tokens,
            num_thing_queries=pred['num_thing_queries'],num_object_queries=pred['num_thing_queries'])
        pred['target_camera_calibration']=camera_audit(batch,pred['scene_scale'],self.scene_scale_head_enabled)
        pred['render']={k:rendered[k][:,:2] for k in ['images_pred','depths_pred','alphas_pred'] if k in rendered}
        return dict(prediction=pred),metrics
