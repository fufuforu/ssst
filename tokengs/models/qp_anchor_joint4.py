"""Original QuerySplat reconstruction + unchanged Baseline4 understanding modules.

Native reconstruction computes in median-depth1 units. Public/understanding
geometry uses Baseline4 median-depth.25 units; positions, scales, camera
translations and clipping planes are converted together, not learned.
Understanding reads detached native L6/8/10/12 states through a new projection.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from native_scannet.anchor_geometry import install as install_anchor,make_seed
from native_scannet.core import build_model,model_input,options,chamfer,ssim_loss,visibility,schedule
from scripts.models.input_types import ModelInputDecoder as NativeDecoder
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import UnifiedQueryConfig,SuperpointQueryInitializer
from tokengs.models.dual_stream_frozen_backbones import FrozenOmega,FrozenESAMSAM
from tokengs.models.object_locus_dual_stream_v1 import SemanticMemory
from tokengs.models.dynamic_panoptic_controller import DynamicPanopticController
from tokengs.models.olda4_initialization_ablation import OldA4GeometricObjectLayer
from tokengs.models.superpoint_dual_core import superpoint_scene_normalization
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.object_locus_posefree_geometry import camera_vectors
from tokengs.models.input_types import ModelInputDecoder
from tokengs.rendering.sh1 import SH1Renderer

KEY='qp_anchor1'
ARCH='QP_ANCHOR1_NATIVE_JOINT4_20261026_V1'
READ_LAYERS=(6,8,10,12)


def scale_gaussians(g,factor):
    return torch.cat((g[...,:3]*factor,g[...,3:4],g[...,4:7]*factor,g[...,7:]),-1)


def scale_cam_view(cam,factor):
    # Public convention is transposed w2c: translation is the last ROW.
    out=cam.clone();out[...,3,:3]=out[...,3,:3]*factor
    return out


def native_row(omega,rgb):
    cams=torch.linalg.inv(omega['c2w']).transpose(-1,-2)
    return dict(rgb=rgb,layers=omega['native_layers'],patch_start=omega['native_patch_start'],
        context_cam_view=scale_cam_view(cams,4.),context_intrinsics=camera_vectors(omega['k']),
        cam_view=scale_cam_view(cams,4.),intrinsics=camera_vectors(omega['k']))


class AnchorBaselineCore(nn.Module):
    anchor_dynamic=False
    def __init__(self,opt,config,native_options=None):
        super().__init__();self.config=config
        # First construct the exact original scratch network with seed42. The
        # baseline modules are copied from the original zero checkpoint later.
        self.reconstruction=build_model(native_options)
        self.initializer=nn.Module();self.initializer.understanding=SuperpointQueryInitializer(config)
        self.semantic_memory=SemanticMemory();self.panoptic=DynamicPanopticController()
        for layer in self.panoptic.layers.values():
            layer.__class__=OldA4GeometricObjectLayer;del layer.anchor_attention
        width=self.reconstruction.opt.enc_embed_dim
        self.native_to_baseline=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1024))
        self.read_states={};self.capture=False
        for i,block in enumerate(self.reconstruction.enc_dec_backbone.decoder_blocks,1):
            if i in READ_LAYERS:block.register_forward_hook(self._hook(i))
        install_anchor(self.reconstruction,opt,self.anchor_dynamic)

    def _hook(self,index):
        def record(module,args,output):
            if self.capture:
                # Eye token is present in the native sequence, never a GS parent.
                self.read_states[index]=output[:,1:].detach()
        return record

    def prepare_memory(self,omega,rgb,renderer):
        return dict(fm=self.semantic_memory(omega['patch_layers'],rgb),renderer=renderer,
            decoder=ModelInputDecoder(cam_view=torch.linalg.inv(omega['c2w']).transpose(-1,-2),intrinsics=camera_vectors(omega['k'])),
            native=native_row(omega,rgb),anchor_seed=make_seed(omega,rgb))

    def forward(self,cloud,memory,superpoint_ids=None):
        if superpoint_ids is not None:cloud=dict(cloud,point_superpoint_ids=superpoint_ids)
        under=self.initializer.understanding(cloud)
        fm=memory['fm'];row=memory['native'];self.read_states={};self.capture=True
        self.reconstruction.vggt_encoder.row=row
        self.reconstruction.anchor_seed=memory['anchor_seed']
        try:
            with torch.autocast(row['rgb'].device.type,dtype=torch.bfloat16,enabled=self.training and row['rgb'].is_cuda):
                g_native=self.reconstruction.forward_gaussians(model_input(row)).float()
        finally:self.capture=False;self.reconstruction.vggt_encoder.row=None
        reads=self.read_states;self.read_states={}
        if set(reads)!=set(READ_LAYERS):raise ValueError('Native semantic read states missing')
        g=scale_gaussians(g_native,.25)
        image=F.interpolate(fm.flatten(0,1),size=(32,32),mode='bilinear',align_corners=False)
        image=image.reshape(1,2,256,1024).permute(0,1,3,2).reshape(1,2048,256)
        q=c=s=ell=None;states=[]
        for layer in READ_LAYERS:
            raw=reads[layer]
            anchor_mu,anchor_radius=self.reconstruction.anchor_reads[layer]
            with torch.no_grad(),torch.autocast(raw.device.type,dtype=torch.bfloat16,enabled=self.training and raw.is_cuda):
                partial=self.reconstruction._decode_partial_gaussians_at(raw,anchor_mu,anchor_radius).float()
                mu=partial[...,:3].reshape(1,-1,64,3).mean(2)*.25
                radii=partial[...,3:6].reshape(1,-1,64,3).mean((2,3)).clamp_min(1e-4)*.25
            h=self.native_to_baseline(raw)
            if layer==6:
                origin,ell=superpoint_scene_normalization(mu)
                stuff=self.panoptic.stuff_ln(fm.mean((1,3,4))[:,None]+self.panoptic.stuff_seed[None])
                q=torch.cat((under['query'],stuff),1)
                c=under['centers'];s=under['radii'][...,None].expand(-1,-1,3)
            a=self.panoptic.encode_token(h,mu,radii,ell)
            f=self.panoptic.ln_mask_a(self.panoptic.W_mask_a(a))
            semantic=self.panoptic.layers[f'L{layer}'](h,a,mu,q,c,s,ell,image,self.semantic_memory.mask_embedder,f,0)
            q,c,s=semantic['q'],semantic['c'],semantic['s']
            state=dict(semantic,layer=layer,tokens=h,mu=mu,radii=radii,scene_origin=origin)
            states.append(state)
        final=states[-1];n=under['query'].shape[1]
        # All LU geometry paths, including feature lift and mask rasterization,
        # use detached GS. No semantic module feeds native generation.
        semantic_g=g.detach()
        grid=F.interpolate(fm.flatten(0,1),size=(256,256),mode='bilinear',align_corners=False).reshape(1,2,256,256,256)
        evidence,gate,_=lift_features(grid,semantic_g,memory['decoder'],memory['renderer'])
        child,_=self.panoptic.gaussian_child_features(final['anchor_embedding'],final['f_anchor'],semantic_g,final['mu'],final['radii'])
        features=gate*evidence+(1-gate)*child+.1*self.panoptic.W_res(child).tanh()
        logits=features@self.semantic_memory.mask_embedder(q).transpose(1,2)
        classification=self.panoptic.classify(q);final.update(classification)
        count=self.reconstruction.opt.num_gs_tokens
        parent=torch.arange(count,device=g.device).repeat_interleave(64)
        return dict(states=states,gaussians=g,native_gaussians=g_native,gaussian_membership=logits.sigmoid(),gaussian_mask_logits=logits,
            thing_logits19=classification['thing_logits19'],p_class=classification['p_class'],q_pre=under['query'],
            unified_query=q[:,:n],superpoints=under,cloud=cloud,pred_iou=None,gaussian_parent_anchor=parent,
            gaussian_parent_query=parent,gaussian_parent_query_semantics='legacy_alias_of_anchor_index_not_object_query',
            num_thing_queries=n,num_superpoint_queries=n,num_object_queries=n,num_stuff_queries=2,
            num_reconstruction_anchors=count,num_gaussians=g.shape[1],
            initial_anchor_mu=memory['anchor_seed']['mu']*.25,initial_anchor_radius=memory['anchor_seed']['radius']*.25,anchor_point_indices=memory['anchor_seed']['indices'][0])

    def parameter_groups(self):
        r=list(self.reconstruction.parameters());ids={id(p) for p in r}
        return dict(shared=[],reconstruction=r,understanding=[p for p in self.parameters() if id(p) not in ids])


class AnchorQPJoint4(LocusGSUnifiedQueryV1):
    architecture_name=ARCH
    core_type=AnchorBaselineCore

    def __init__(self,opt,*,config=None,omega=None,sam=None,renderer=None,native_options=None,**kwargs):
        nn.Module.__init__(self);self.config=config or UnifiedQueryConfig();self.opt=opt;self.img_size=(256,256)
        # Construct native before frozen asset loading changes random clocks.
        self.core=self.core_type(opt,self.config,native_options)
        self.frozen_omega=omega if omega is not None else FrozenOmega()
        self.frozen_omega.retain_native_tokens=True;self.frozen_omega.requires_grad_(False).eval()
        self.sam_proposals=sam if sam is not None else FrozenESAMSAM()
        self.sam_proposals.requires_grad_(False).eval()
        self.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256))
        # Same coordinate conversion as native GS/cameras, including clipping.
        self.gs=renderer or SH1Renderer(opt.evolve(znear=.025*.25,zfar=125.*.25))
        self.supervised_layers=(12,);self.layer_weights=(1.,);self.understanding_step=0
        self.perceptual=None

    def encode_context(self,rgb):
        omega,cloud,_=LocusGSUnifiedQueryV1.encode_context(self,rgb)
        return omega,cloud,self.core.prepare_memory(omega,rgb,self.gs)

    def render_reconstruction(self,reconstruction,decoder_input):
        g=scale_gaussians(reconstruction.gaussians,4.)
        decoder=NativeDecoder(cam_view=scale_cam_view(decoder_input.cam_view,4.),intrinsics=decoder_input.intrinsics)
        result=self.core.reconstruction.render_gaussians(g,decoder)
        result['depths_pred']=result['depths_pred']*.25
        return result

    def _render_understanding(self,generated,decoder):
        # Feature rendering is detached. RGB still uses the original attached
        # gaussians via reconstruction, exactly as the baseline isolation path.
        return super()._render_understanding(dict(generated,gaussians=generated['gaussians'].detach()),decoder)

    def step_loss(self,batch,*,step=0,phase='train',understanding_weight=1.):
        from tokengs.models.unified_query_loss import unified_instance_loss
        self.understanding_step=int(step);pred=self.generate(batch['images_input'])
        calibration=self.calibrate_targets(batch['images_all'],pred)
        decoder=ModelInputDecoder(cam_view=torch.linalg.inv(calibration['c2w']).transpose(-1,-2),intrinsics=calibration['intrinsics'])
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
        understanding,um=unified_instance_loss(pred,batch,self.config)
        mse=(images-truth).square().flatten(2).mean(-1)
        metrics=dict(um,**terms,loss_recon=reconstruction,loss_understanding=understanding,
            loss=reconstruction+understanding_weight*understanding,psnr=(-10*mse.clamp_min(1e-12).log10()).mean(),
            gradient_isolation=1,num_gaussians=g.shape[1],num_reconstruction_anchors=self.core.reconstruction.opt.num_gs_tokens,
            num_thing_queries=pred['num_thing_queries'],num_object_queries=pred['num_thing_queries'])
        pred['target_camera_calibration']=calibration
        return dict(prediction=pred),metrics

    def parameter_groups(self):
        groups=self.core.parameter_groups();groups['understanding']+=list(self.gfm_projection.parameters())
        return groups

class AnchorDynamicCore(AnchorBaselineCore):
    anchor_dynamic=True
class AnchorQP2Joint4(AnchorQPJoint4):
    architecture_name='QP_ANCHOR2_NATIVE_JOINT4_20261026_V1'
    core_type=AnchorDynamicCore
