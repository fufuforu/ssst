"""Original QPanchor1 parameters and initialization; GT cameras for every supervised render."""
import math
import torch
from torch.nn import functional as F
from tokengs.models.qp_anchor_joint4 import AnchorQPJoint4,scale_cam_view
from native_scannet.core import schedule,ssim_loss,visibility,chamfer
from tokengs.models.object_locus_v3_set import alpha_normalize_membership
from tokengs.models.unified_query_nms import context_mask_nms
from contextlib import contextmanager
from scripts.gt_camera_protocol import gt_decoder,camera_audit,context_camera_batch,reframe_omega
ARCH='QP_ANCHOR1_GT_CONTEXT_GEOMETRY_GAUGE_V2'

class AnchorQPJoint4GTCamera(AnchorQPJoint4):
    architecture_name=ARCH

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original=self.frozen_omega.forward
        self._GT_context=None
        def context_only(rgb):
            if rgb.shape[1]!=2 or self._GT_context is None:raise RuntimeError('GT Context camera binding required; Target RGB forbidden')
            if not torch.equal(rgb,self._GT_context_rgb):raise ValueError('Context RGB identity differs')
            value,gauge=reframe_omega(original(rgb),self._GT_context)
            self._GT_gauge=gauge
            return value
        self.frozen_omega.forward=context_only

    @contextmanager
    def gt_context(self,batch):
        if self._GT_context is not None:raise RuntimeError('Nested GT Context binding')
        self._GT_context=context_camera_batch(batch);self._GT_context_rgb=batch['images_input'].detach();self._GT_gauge=None
        try:yield
        finally:self._GT_context=None;self._GT_context_rgb=None;self._GT_gauge=None

    def generate(self,*args,**kwargs):
        raise RuntimeError('Use generate_gt with explicit Context GT cameras')

    def generate_gt(self,batch):
        with self.gt_context(batch):
            pred=super().generate(batch['images_input'])
            pred['GT_camera_gauge']=dict(self._GT_gauge)
            return pred

    def calibrate_targets(self,*args,**kwargs):
        raise RuntimeError('Predicted target camera calibration is forbidden in GT-camera training/evaluation')

    def render_gt_features(self,pred,decoder):
        feature=self.gs.render_feature_channels(pred['gaussians'].detach(),pred['gaussian_membership'],decoder.cam_view,decoder.intrinsics)
        alpha=feature['alphas_pred'];regions=alpha_normalize_membership(feature['images_pred'],alpha)
        n=pred['num_thing_queries'];p=pred['p_class']
        semantic=regions.new_zeros((1,regions.shape[1],20,*regions.shape[-2:]))
        semantic[:,:,:2]=regions[:,:,n:n+2]
        semantic[:,:,2:]=torch.einsum('bvqhw,bqc->bvchw',regions[:,:,:n],p[...,:18])
        semantic=semantic/(semantic.sum(2,keepdim=True)+1e-6)
        eligible,ranking=context_mask_nms(p,regions[:,:2],alpha[:,:2],self.config.mask_nms_iou)
        return dict(region_mass=regions,pixel_membership=regions,alpha=alpha,semantic_scores=semantic,
            instance_eligible=eligible,instance_ranking_scores=ranking)

    def render_generated_gt(self,pred,batch):
        decoder=gt_decoder(batch,gauge=pred['GT_camera_gauge'])
        result=self.render_reconstruction(pred['reconstruction'],decoder)
        result.update(self.render_gt_features(pred,decoder))
        result.update(p_class=pred['p_class'],num_thing_queries=pred['num_thing_queries'])
        return result

    def step_loss(self,batch,*,step=0,phase='train',understanding_weight=1.):
        from tokengs.models.unified_query_loss import unified_instance_loss
        self.understanding_step=int(step);pred=self.generate_gt(batch)
        gauge=pred['GT_camera_gauge']
        decoder=gt_decoder(batch,gauge=gauge)
        calibration=camera_audit(batch,gauge)
        rendered=self.render_reconstruction(pred['reconstruction'],decoder)
        images=rendered['images_pred'];truth=batch['images_all'];flat=images.flatten(0,1);gt=truth.flatten(0,1)
        g=pred['native_gaussians']
        row=dict(cam_view=scale_cam_view(decoder.cam_view,4.),intrinsics=decoder.intrinsics)
        terms=dict(l1=F.l1_loss(images,truth),ssim=ssim_loss(flat,gt),visibility=visibility(g[0,:,:3],row))
        weights=schedule(step,4172)
        points=pred['predicted_points'].reshape(-1,3)*4
        points=points[torch.isfinite(points).all(-1)]
        cloud=points[torch.linspace(0,len(points)-1,4096,device=points.device).long()]
        terms['chamfer']=chamfer(g[0,:,:3],cloud) if weights['chamfer'] else images.new_zeros(())
        terms['opacity']=F.relu(math.log(.1)-g[...,3].clamp_min(1e-8).log()).mean()
        with torch.autocast(images.device.type,enabled=False):
            if weights['lpips'] and self.perceptual is None:raise ValueError('Frozen LPIPS missing')
            terms['lpips']=self.perceptual(flat.float()*2-1,gt.float()*2-1).mean() if weights['lpips'] else images.new_zeros(())
        reconstruction=terms['l1']+.2*terms['ssim']+terms['visibility']+weights['chamfer']*terms['chamfer']+weights['opacity']*terms['opacity']+weights['lpips']*terms['lpips']
        # All supervised Context mask projections also use GT cameras.
        context_render=self.render_gt_features(pred,gt_decoder(batch,2,gauge=gauge))
        pred.update(context_render)
        understanding,um=unified_instance_loss(pred,batch,self.config)
        mse=(images-truth).square().flatten(2).mean(-1)
        metrics=dict(um,**terms,loss_recon=reconstruction,loss_understanding=understanding,
            loss=reconstruction+understanding_weight*understanding,psnr=(-10*mse.clamp_min(1e-12).log10()).mean(),
            gradient_isolation=1,num_gaussians=g.shape[1],num_reconstruction_anchors=self.core.reconstruction.opt.num_gs_tokens,
            num_thing_queries=pred['num_thing_queries'],num_object_queries=pred['num_thing_queries'])
        pred['target_camera_calibration']=calibration
        pred['render']={k:rendered[k][:,:2] for k in ['images_pred','depths_pred','alphas_pred'] if k in rendered}
        metrics['GT_camera_supervision']=1.
        metrics['target_RGB_camera_calls']=0.
        metrics['GT_scene_scale']=gauge['GT_public_to_generated_scale']
        metrics['GT_scale_guard_active']=float(gauge['scale_rule']!='unclipped_Context_baseline_scale')
        return dict(prediction=pred),metrics
