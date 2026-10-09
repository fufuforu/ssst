"""Joint pretrained panoptic understanding inside the original LocusGS loop."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokengs.models.canonical_recon_models import LocusGSRecon, patch_plucker_rays, _full_supervision
from tokengs.models.locusgs_recon import LocusGSAnchorDecoder, anchor_ray_geometric_bias, sinusoidal_positional_encoding
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.object_locus_v3_set_controller import scene_normalization
from tokengs.models.object_locus_v3_set import alpha_normalize_membership
from tokengs.models.object_locus_panoptic_v1_lift import lift_features


def object_image_memory(fm, size=32):
    """View-major image memory; configuration changes no model state."""
    if size not in (32, 128):
        raise ValueError('object_image_memory_size must be 32 or 128')
    if fm.ndim != 5 or tuple(fm.shape[1:]) != (2, 256, 128, 128):
        raise ValueError('expected F_m [B,2,256,128,128]')
    batch = fm.shape[0]
    image = F.interpolate(fm.flatten(0,1),size=(32,32),mode='bilinear',align_corners=False) if size == 32 else fm
    return image.reshape(batch,2,256,size*size).permute(0,1,3,2).reshape(batch,2*size*size,256)


class PanopticAnchorDecoder(LocusGSAnchorDecoder):
    def forward_stateful(self, tokens, latent, rays, controller, fm, qpre, mask_embedder, exposure):
        moment, direction = rays
        batch = tokens.shape[0]
        if moment.shape[1] != latent.keys.shape[-2]: raise ValueError('ray/key count mismatch')
        mu = self.mu[None].expand(batch,-1,-1).contiguous()
        rho = self.rho[None].expand(batch,-1).contiguous()
        image = object_image_memory(fm, getattr(self.opt, 'object_image_memory_size', 32))
        states, ray_stats = [], []
        q = c = s = ell = origin = None
        for index, (block, head_mu, head_rho) in enumerate(zip(self.decoder_blocks,self.refine_mu,self.refine_rho)):
            def reconstruction_block(h, m, r, keys, values, moments, directions, idx=index, blk=block, hm=head_mu, hr=head_rho):
                rad = self.activated_radius(r)
                attn_bias = None
                if float(self.opt.locusgs_ray_bias_scale)>0:
                    geo = anchor_ray_geometric_bias(m,rad,moments,directions,sigma0=self.sigma0,
                        bandwidth_floor=self.bandwidth_floor,clamp_min=self.bias_clamp_min)
                    attn_bias = F.softplus(self.gamma_raw[idx])*geo
                pe_module = self.pe_mlp if self.pe_mlp is not None else self.pe_mlps[idx]
                pe = pe_module(sinusoidal_positional_encoding(m,int(self.opt.locusgs_pe_num_freqs)))
                h = h + blk.gs_cross_attn_scale(blk.gs_cross_attn(h,keys,values,attn_bias=attn_bias))
                if self.pe_mode=='persistent':
                    h = h + pe
                    h = h + blk.gs_self_attn_scale(blk.gs_self_attn(h))
                else:
                    h = h + blk.gs_self_attn_scale(blk.gs_self_attn(h+pe))
                h = h + blk.mlp_scale(blk.mlp(h))
                return h,m+hm(h),r+hr(h).squeeze(-1)
            previous_mu, previous_radii = mu,self.activated_radius(rho)
            args = (tokens,mu,rho,latent.keys,latent.values,moment,direction)
            tokens,mu,rho = checkpoint(reconstruction_block,*args,use_reentrant=False,preserve_rng_state=True) if self.training and torch.is_grad_enabled() else reconstruction_block(*args)
            radii = self.activated_radius(rho)
            layer, result = index+1, {}
            if layer in (6,8,10,12):
                if layer==6: origin,ell = scene_normalization(mu)
                a = controller.encode_token(tokens,mu,radii,ell)
                if layer==6: q,c,s = controller.initialize_states(qpre,fm,a,mu,ell)
                f_anchor = controller.ln_mask_a(controller.W_mask_a(a))
                result = controller.layers[f'L{layer}'](tokens,a,mu,q,c,s,ell,image,mask_embedder,f_anchor,exposure)
                q,c,s = result['q'],result['c'],result['s']
                tokens = tokens + result['joint_delta']
                result['scene_origin'] = origin
            states.append(dict(**result,layer=layer,tokens=tokens,mu=mu,rho=rho,radii=radii,
                beta=0.1*min(max(float(exposure),0)/1000,1),
                anchor_update=(mu-previous_mu).norm(dim=-1).mean().detach(),
                radius_update=(radii-previous_radii).abs().mean().detach()))
        return states,ray_stats


class LocusGSObjectLocusPanopticV1Recon(LocusGSRecon):
    architecture_name = 'LOCUSGS_OBJECT_LOCUS_PANOPTIC_V1'
    reconstruction_only = False
    state_layers = (6,8,10,12)

    def __init__(self,opt):
        super().__init__(opt)
        self.reconstruction_only = False
        if tuple(opt.instance_state_layers)!=self.state_layers: raise ValueError('registered layers fixed to 6/8/10/12')
        # Same decoder instance/state tensors, new loop implementation.
        self.anchor_decoder.__class__ = PanopticAnchorDecoder
        self.understanding_step = 0

    def initialize_understanding(self,mapping):
        from tokengs.models.object_locus_panoptic_v1_pretrained import PretrainedUnderstanding
        from tokengs.models.object_locus_panoptic_v1_controller import ObjectLocusPanopticV1Controller
        self.understanding = PretrainedUnderstanding(mapping)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31415)
            self.panoptic = ObjectLocusPanopticV1Controller()

    def _readout(self, final, gaussians, fm, read_context_decoder, render_decoder_input):
        b = gaussians.shape[0]
        feature_grid = F.interpolate(fm.flatten(0,1),size=(256,256),mode='bilinear',align_corners=False).reshape(b,2,256,256,256)
        evidence,gate,mass = lift_features(feature_grid,gaussians,read_context_decoder,self.gs)
        child,residual = self.panoptic.gaussian_child_features(final['anchor_embedding'],final['f_anchor'],gaussians,final['mu'],final['radii'])
        features = gate*evidence + (1-gate)*child + 0.1*torch.tanh(self.panoptic.W_res(child))
        mq = self.understanding.mask_embedder(final['q'])
        logits = features @ mq.transpose(1,2)
        membership = logits.sigmoid()
        rendered = self.gs.render_feature_channels(gaussians,membership,render_decoder_input.cam_view,render_decoder_input.intrinsics)
        alpha = rendered['alphas_pred']
        region = alpha_normalize_membership(rendered['images_pred'],alpha)
        cls = self.panoptic.classify(final['q'])
        semantic = region.new_zeros((b,region.shape[1],20,*region.shape[-2:]))
        semantic[:,:,0:2] = region[:,:,100:102]
        semantic[:,:,2:20] = torch.einsum('bvqhw,bqc->bvchw',region[:,:,:100],cls['p_class'][...,:18])
        semantic = semantic/(semantic.sum(2,keepdim=True)+1e-6)
        final.update(cls)
        return dict(**cls,assignment=membership,gaussian_membership=membership,gaussian_mask_logits=logits,
            gaussian_feature=features,child_feature_residual=residual,anchor_membership=final['anchor_membership'],
            anchor_assignment=final['anchor_membership'],anchor_mask_logits=final['anchor_mask_logits'],
            membership_mass=rendered['images_pred'],region_mass=region,pixel_membership=region,
            semantic_scores=semantic,pixel_void_mass=1-alpha,alpha=alpha,
            lifting_mass=mass,lifting_gate=gate)

    def forward_object_locus(self,model_input,*,render_decoder_input=None,read_context_decoder=None,context_decoder=None,coupled=None,step=None):
        del coupled
        render_decoder = render_decoder_input or model_input.decoder
        # Evaluator's context_decoder can contain requested novel cameras. Reading
        # always uses the encoder's actual context camera identities.
        read_decoder = read_context_decoder or ModelInputDecoder(
            cam_view=torch.linalg.inv(model_input.encoder.cam_to_world_input).transpose(-1,-2),
            intrinsics=model_input.encoder.intrinsics_input)
        output_decoder = context_decoder or render_decoder
        exposure = self.understanding_step if step is None else int(step)
        images = model_input.encoder.images_rgb_unnormalized
        if images is None: raise ValueError('missing original context crop')
        fm,qpre = self.understanding(images)
        latent = self.forward_encoder(model_input.encoder)
        rays = patch_plucker_rays(model_input.encoder.rays_os,model_input.encoder.rays_ds,patch_size=int(self.opt.patch_size))
        states,ray_stats = self.anchor_decoder.forward_stateful(self.get_gs_tokens(batch_size=model_input.batch_size),latent,rays,self.panoptic,fm,qpre,self.understanding.mask_embedder,exposure)
        final = states[-1]
        gaussians = self.activation_head(final['tokens'],final['mu'],final['radii'])
        reconstruction = self._reconstruction_from_gaussians(gaussians)
        output = dict(states=states,gaussians=gaussians,reconstruction=reconstruction,
                      render=self.render_reconstruction(reconstruction,render_decoder),ray_stats=ray_stats,beta=final['beta'],F_m=fm,q_pre=qpre)
        output.update(self._readout(final,gaussians,fm,read_decoder,output_decoder))
        return output

    forward_instance_state = forward_object_locus
    forward_reconstruction_only = forward_object_locus

    def step_loss(self,batch,*,step,phase='train',coupled=None,understanding_weight=1.0):
        del phase,coupled
        self.understanding_step = int(step)
        mi,_ = split_data(batch,self.opt)
        decoder = ModelInputDecoder(cam_view=batch['cam_view_all'],intrinsics=batch['intrinsics_all'])
        context = ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2])
        prediction = self.forward_object_locus(mi,render_decoder_input=decoder,read_context_decoder=context,context_decoder=context,step=step)
        rec,metrics,*_ = self._layer_objective(prediction['states'],decoder,_full_supervision(batch))
        from tokengs.models.object_locus_v3_set_loss import object_locus_v3_set_losses
        under,um = object_locus_v3_set_losses(prediction,batch,self.opt)
        metrics.update(um)
        metrics.update(loss_recon=rec,loss_understanding=under,understanding_weight=float(understanding_weight),
                       loss=rec+float(understanding_weight)*under,loss_total=rec+float(understanding_weight)*under)
        metrics['psnr'] = metrics[f'psnr_layer{self.supervised_layers[-1]}']
        return {'prediction':prediction},metrics

    def forward(self,data,skip_loss=False):
        del skip_loss
        if isinstance(data,ModelInput): return self.forward_object_locus(data)
        return self.step_loss(data,step=self.understanding_step)[0]
