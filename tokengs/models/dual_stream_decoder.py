"""Point/SAM-seeded variant of the existing LocusGS interaction loop.

The old decoder is untouched. Write defaults off; enabled write reuses the
old soft route + null slot and bounded residual, not a claimed new ownership mechanism.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokengs.models.locusgs_recon import LocusGSAnchorDecoder, anchor_ray_geometric_bias, sinusoidal_positional_encoding
from tokengs.models.object_locus_v3_set_controller import scene_normalization

class PointSeededAnchorDecoder(LocusGSAnchorDecoder):
    def forward_stateful(self, tokens, latent, rays, controller, fm, qpre, mask_embedder, exposure, *, seed_mu, seed_radius, seed_c, seed_s, write_strength=0.0):
        moment, direction = rays
        batch = tokens.shape[0]
        if moment.shape[1] != latent.keys.shape[-2]: raise ValueError('ray/key count mismatch')
        mu = seed_mu
        positive = (seed_radius-self.epsilon).clamp_min(1e-6)
        rho = positive + torch.log(-torch.expm1(-positive))
        image = F.interpolate(fm.flatten(0,1),size=(32,32),mode='bilinear',align_corners=False)
        image = image.reshape(batch,2,256,1024).permute(0,1,3,2).reshape(batch,2048,256)
        states, ray_stats = [], []
        q = c = s = ell = origin = None
        reading = getattr(controller, 'reading_control', None)
        previous_support = None
        for index, (block, head_mu, head_rho) in enumerate(zip(self.decoder_blocks,self.refine_mu,self.refine_rho)):
            def reconstruction_block(h, m, r, keys, values, moments, directions, idx=index, blk=block, hm=head_mu, hr=head_rho):
                rad = self.activated_radius(r)
                attn_bias = None
                if float(self.opt.locusgs_ray_bias_scale)>0:
                    geo = anchor_ray_geometric_bias(m,rad,moments,directions,sigma0=self.sigma0,
                        bandwidth_floor=self.bandwidth_floor,clamp_min=self.bias_clamp_min)
                    attn_bias = F.softplus(self.gamma_raw[idx])*geo
                if reading is not None:
                    attn_bias = reading.geometry(attn_bias, m, rad, directions, self.bandwidth_floor)
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
                if reading is not None:
                    a = reading.anchor(a, mu)
                if layer==6:
                    stuff = controller.stuff_ln(fm.mean((1,3,4))[:,None] + controller.stuff_seed[None])
                    q = torch.cat((qpre,stuff),1)
                    c,s = seed_c,seed_s
                f_anchor = controller.ln_mask_a(controller.W_mask_a(a))
                kwargs = {} if reading is None else dict(reading_control=reading, previous_support=previous_support)
                result = controller.layers[f'L{layer}'](tokens,a,mu,q,c,s,ell,image,mask_embedder,f_anchor,exposure, **kwargs)
                previous_support = result['anchor_membership'].detach()
                q,c,s = result['q'],result['c'],result['s']
                result['joint_delta'] = result['joint_delta'] * write_strength
                tokens = tokens + result['joint_delta']
                result['scene_origin'] = origin
            states.append(dict(**result,layer=layer,tokens=tokens,mu=mu,rho=rho,radii=radii,
                beta=write_strength*0.1*min(max(float(exposure),0)/1000,1),
                anchor_update=(mu-previous_mu).norm(dim=-1).mean().detach(),
                radius_update=(radii-previous_radii).abs().mean().detach()))
        return states,ray_stats

