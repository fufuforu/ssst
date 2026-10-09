"""VGGT-assisted pose-free migration path; historical Object-Locus is unchanged."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.frozen_vggt_posefree import FrozenVGGT
from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
from tokengs.models.object_locus_posefree_geometry import (
    align_cameras_by_shared_context, camera_vectors, intrinsics518_to_256,
    pixel_rays, posefree_scene, rays_to_patch_plucker,
)
from tokengs.models.input_types import EncoderLatent, ModelInput, ModelInputDecoder, ModelInputEncoder
from tokengs.models.canonical_recon_models import _full_supervision


class VGGTMemoryAdapter(nn.Module):
    """Four-scale patch adapter and 1024->16x64 K/V projection."""
    def __init__(self):
        super().__init__()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(2048, eps=1e-5) for _ in range(4)])
        self.layer_logits = nn.Parameter(torch.zeros(4))
        self.projection = nn.Linear(2048, 1024)
        self.output_norm = nn.LayerNorm(1024)
        self.kv = nn.Linear(1024, 2048)
        self.key_norm = nn.LayerNorm(64)

    def forward(self, patch_layers):
        if len(patch_layers) != 4:
            raise ValueError("exactly four official VGGT layers are required")
        reference = patch_layers[0]
        if reference.ndim != 4 or reference.shape[1:] != (2,1369,2048):
            raise ValueError(f"expected each patch layer [B,2,1369,2048], got {tuple(reference.shape)}")
        normalized=[]
        for norm, layer in zip(self.layer_norms, patch_layers):
            if layer.shape != reference.shape:
                raise ValueError("VGGT patch layer shape mismatch")
            normalized.append(norm(layer.float()))
        weights=self.layer_logits.softmax(0)
        mixed=sum(weight*layer for weight,layer in zip(weights,normalized))
        # reshape preserves view-major then row-major spatial order
        memory=self.output_norm(self.projection(mixed.reshape(reference.shape[0],2738,2048)))
        keys,values=self.kv(memory).chunk(2,dim=-1)
        b=memory.shape[0]
        keys=keys.reshape(b,2738,16,64).transpose(1,2).contiguous()
        values=values.reshape(b,2738,16,64).transpose(1,2).contiguous()
        return memory, self.key_norm(keys), values


def initialize_memory_adapter(adapter: nn.Module, seed: int = 31415):
    """Initialize only newly introduced adapter parameters in an isolated RNG fork."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
        nn.init.zeros_(adapter.layer_logits)


class LocusGSObjectLocusFrozenVGGT(LocusGSObjectLocusPanopticV1Recon):
    architecture_name = "LOCUSGS_OBJECT_LOCUS_FROZEN_VGGT_POSEFREE_V1"

    def __init__(self, opt, *, vggt: FrozenVGGT | None = None):
        super().__init__(opt)
        self.frozen_vggt = vggt if vggt is not None else FrozenVGGT()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31415)
            self.vggt_memory_adapter = VGGTMemoryAdapter()
            initialize_memory_adapter(self.vggt_memory_adapter)
        # Retain only the shared LocusGS decoder/reference graph from EncDecBackbone.
        # The old pixel embedding, reconstruction encoder and old K/V are dead in
        # this architecture and must not be registered or optimized.
        self.patch_embed = nn.Identity()
        self.patch_plucker_embed = nn.Identity()
        backbone=self.enc_dec_backbone
        backbone.encoder=nn.Identity()
        for name in ("encoder_norm","kv_proj","k_proj_norm","multiscale_norms",
                     "latent_feature_kv_proj","latent_feature_k_norm","latent_blocks",
                     "latent_kv_proj"):
            if hasattr(backbone,name): setattr(backbone,name,nn.Identity())
        if hasattr(backbone,"latents"):
            backbone.register_parameter("latents",None)
        if any(p.requires_grad for p in self.frozen_vggt.parameters()):
            raise RuntimeError("VGGT parameters must be frozen")
        self.frozen_vggt.eval()

    def train(self, mode=True):
        super().train(mode)
        self.frozen_vggt.eval()
        return self

    def _latent_from_vggt(self, result):
        _,keys,values=self.vggt_memory_adapter(result.patch_layers)
        return EncoderLatent(keys=keys,values=values)

    def _predicted_input(self, context_rgb, result, c2w_scene, k256):
        b=context_rgb.shape[0]
        ro256,rd256=pixel_rays(c2w_scene,k256,256,256)
        plucker256=torch.cat((torch.cross(ro256,rd256,dim=2),rd256),dim=2)
        encoder=ModelInputEncoder(images_rgb=context_rgb.mul(2).sub(1),plucker=plucker256,
            rays_os=ro256,rays_ds=rd256,intrinsics_input=camera_vectors(k256),
            cam_to_world_input=c2w_scene,images_rgb_unnormalized=context_rgb)
        decoder=ModelInputDecoder(cam_view=torch.linalg.inv(c2w_scene).transpose(-1,-2),
                                  intrinsics=camera_vectors(k256))
        return ModelInput(encoder,decoder), (ro256,rd256)

    def generate(self, context_rgb: torch.Tensor, runtime_config: dict | None = None):
        """Generate from two RGB contexts and runtime config only; no target data accepted."""
        del runtime_config
        if context_rgb.ndim!=5 or context_rgb.shape[1:]!=(2,3,256,256):
            raise ValueError(f"generate expects [B,2,3,256,256], got {tuple(context_rgb.shape)}")
        vggt_result=self.frozen_vggt(context_rgb)
        c2w_scene,k256,depth_scaled,points,coordinate_record=posefree_scene(
            vggt_result.c2w_cv,vggt_result.intrinsics518,vggt_result.depth518)
        model_input,_=self._predicted_input(context_rgb,vggt_result,c2w_scene,k256)
        latent=self._latent_from_vggt(vggt_result)
        patch_rays=rays_to_patch_plucker(c2w_scene,k256,patch_size=14)
        fm,qpre=self.understanding(context_rgb)
        states,ray_stats=self.anchor_decoder.forward_stateful(
            self.get_gs_tokens(batch_size=context_rgb.shape[0]),latent,patch_rays,
            self.panoptic,fm,qpre,self.understanding.mask_embedder,self.understanding_step)
        final=states[-1]
        gaussians=self.activation_head(final['tokens'],final['mu'],final['radii'])
        reconstruction=self._reconstruction_from_gaussians(gaussians)
        render_decoder=model_input.decoder
        output_decoder=model_input.decoder
        render=self.render_reconstruction(reconstruction,render_decoder)
        output=dict(states=states,gaussians=gaussians,reconstruction=reconstruction,render=render,
                    ray_stats=ray_stats,beta=final['beta'],F_m=fm,q_pre=qpre,
                    predicted_context_c2w=c2w_scene,predicted_context_intrinsics=camera_vectors(k256),
                    predicted_context_intrinsics_matrix=k256,depth=depth_scaled,
                    predicted_points=points,coordinates=coordinate_record,
                    vggt_source_identity=vggt_result.source_identity)
        output.update(self._readout(final,gaussians,fm,render_decoder,output_decoder))
        return output

    def calibrate_targets(self, context_and_target_rgb: torch.Tensor, generated: dict):
        """Independent frozen image calibration pass; returns only aligned camera data."""
        result=self.frozen_vggt.camera_only(context_and_target_rgb)
        raw=result['c2w_cv']
        context_raw=generated['coordinates']['raw_c2w_cv']
        aligned,sim3=align_cameras_by_shared_context(raw,context_raw,
            generated['predicted_context_c2w'],generated['coordinates']['a_scale'],
            generated['coordinates']['first_camera_inverse'])
        k256,A=intrinsics518_to_256(result['intrinsics518'])
        k256=k256.clone(); k256[:,:2]=generated['predicted_context_intrinsics_matrix']
        aligned=aligned.clone(); aligned[:,:2]=generated['predicted_context_c2w']
        return {"c2w":aligned,"intrinsics_matrix":k256,"intrinsics":camera_vectors(k256),
                "sim3":sim3,"A_518_to_256":A.detach(),"source_identity":result['source_identity']}

    def render_generated_at(self, generated: dict, cam_view: torch.Tensor, intrinsics: torch.Tensor):
        """Render existing generated Gaussians for an externally supplied target camera."""
        decoder=ModelInputDecoder(cam_view=cam_view,intrinsics=intrinsics)
        return self.render_reconstruction(generated['reconstruction'],decoder)

    def step_loss(self,batch,*,step,phase="train",understanding_weight=1.0):
        del phase
        self.understanding_step=int(step)
        context=batch['images_input']
        prediction=self.generate(context)
        calibration=self.calibrate_targets(batch['images_all'],prediction)
        c2w=calibration['c2w']
        cam_view=torch.linalg.inv(c2w).transpose(-1,-2)
        all_rays_o,all_rays_d=pixel_rays(c2w,calibration['intrinsics_matrix'],256,256)
        # Clone only the loss view of the provider batch. RGB and semantic masks stay
        # exactly as supplied; provider storage and source dictionary are untouched.
        loss_batch=dict(batch)
        loss_batch['cam_view_all']=cam_view
        loss_batch['intrinsics_all']=calibration['intrinsics']
        loss_batch['rays_os']=all_rays_o
        loss_batch['rays_ds']=all_rays_d
        loss_batch['cam_to_world_input']=c2w[:,:2]
        loss_batch['intrinsics_input']=calibration['intrinsics'][:,:2]
        decoder=ModelInputDecoder(cam_view=cam_view,intrinsics=calibration['intrinsics'])
        recon,metrics,*_=self._layer_objective(prediction['states'],decoder,_full_supervision(loss_batch))
        from tokengs.models.object_locus_v3_set_loss import object_locus_v3_set_losses
        under,under_metrics=object_locus_v3_set_losses(prediction,loss_batch,self.opt)
        metrics.update(under_metrics)
        metrics.update(loss_recon=recon,loss_understanding=under,
                       understanding_weight=float(understanding_weight),
                       loss=recon+float(understanding_weight)*under,
                       loss_total=recon+float(understanding_weight)*under)
        metrics['psnr']=metrics[f"psnr_layer{self.supervised_layers[-1]}"]
        prediction['target_camera_calibration']=calibration
        return {'prediction':prediction},metrics

    def forward(self,data,skip_loss=False):
        del skip_loss
        if isinstance(data,torch.Tensor):
            return self.generate(data)
        if isinstance(data,dict):
            return self.step_loss(data,step=self.understanding_step)[0]
        raise TypeError("pose-free forward accepts context RGB or training batch")
