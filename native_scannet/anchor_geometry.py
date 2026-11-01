"""Scene-conditioned QuerySplat tokens and optional iterative geometric routing."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from native_scannet.core import native
from tokengs.models.dual_stream_initialization import seed_anchors
from tokengs.models.gfm_dual_stream_components import gfm_grid,sample_anchor_gfm
from tokengs.models.locusgs_recon import anchor_ray_geometric_bias


def make_seed(omega,rgb):
    mu,r,col,indices=seed_anchors(omega['points'],omega['confidence'],rgb,1024,return_source_indices=True)
    if indices[0].unique().numel()!=1024:raise ValueError('Nonunique seed pixels')
    visual=sample_anchor_gfm(gfm_grid(omega['patch_layers']),indices,image_hw=tuple(rgb.shape[-2:]))
    return dict(mu=mu*4,radius=r*4,color=col,visual=visual,indices=indices,omega=omega)


def memory_rays(seed,key_count):
    omega=seed['omega'];special=int(omega['native_patch_start']);perview=key_count//2
    side=math.isqrt(perview-special)
    if key_count%2 or side*side!=perview-special:raise ValueError('Native memory grid layout differs')
    k=omega['k'].float();c2w=omega['c2w'].float()
    y,x=torch.meshgrid((torch.arange(side,device=k.device)+.5)*256/side,(torch.arange(side,device=k.device)+.5)*256/side,indexing='ij')
    pixels=torch.stack((x,y,torch.ones_like(x)),-1).reshape(-1,3)
    direction=torch.einsum('bvij,pj->bvpi',torch.linalg.inv(k),pixels)
    direction=torch.einsum('bvij,bvpj->bvpi',c2w[...,:3,:3],direction)
    direction=F.normalize(direction,dim=-1);origin=c2w[...,:3,3]*4
    moment=torch.cross(origin[:,:,None,:].expand_as(direction),direction,dim=-1)
    return moment,direction,special


def geometry_bias(mu,radius,moment,direction,special,key_count,gamma,*,sigma0,bandwidth_floor,clamp_min):
    # Preserve native per-view [camera,registers,patches] key order.
    bias=anchor_ray_geometric_bias(mu.float(),radius.float(),moment.flatten(1,2),direction.flatten(1,2),
        sigma0=sigma0,bandwidth_floor=bandwidth_floor,clamp_min=clamp_min)
    b,_,n,_=bias.shape;patches=moment.shape[2]
    bias=bias.reshape(b,1,n,2,patches)
    neutral=bias.new_zeros((b,1,n,2,special))
    bias=torch.cat((neutral,bias),-1).flatten(-2)
    if bias.shape[-1]!=key_count:raise ValueError('Bias/key alignment differs')
    # The native eye query has no spatial anchor. All its attention remains unbiased.
    return torch.cat((bias.new_zeros((b,1,1,key_count)),F.softplus(gamma)*bias),dim=2)


class AnchoredQuerySplat(native.QuerySplat):
    def _forward_decoder_tokens(self,latent):
        seed=self.anchor_seed
        if seed is None:raise ValueError('Missing Context-only anchor seed')
        mu=seed['mu'].float();radius=seed['radius'].float();origin=mu.mean(1,keepdim=True)
        ell=(mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.2)
        content=torch.cat(((mu-origin)/ell[:,None,None],(radius/ell[:,None])[...,None],seed['color']),-1)
        tokens=self.get_gs_tokens(len(mu))+self.anchor_content(content)+self.anchor_visual(seed['visual'])
        if latent.eye_token is None:raise ValueError('Native eye token required')
        tokens=torch.cat((latent.eye_token[:,None].to(tokens.dtype),tokens),1)
        positive=(radius-self.anchor_epsilon).clamp_min(1e-6)
        rho=positive+torch.log(-torch.expm1(-positive))
        if self.anchor_dynamic:moment,direction,special=memory_rays(seed,latent.keys.shape[-2])
        self.anchor_reads={};self.anchor_update_norms=[];self.anchor_bias_shape=None
        for i,block in enumerate(self.enc_dec_backbone.decoder_blocks):
            bias=None
            if self.anchor_dynamic:
                key_count=latent.keys.shape[-2]
                def bias_fn(m,r,mo,di,g):
                    return geometry_bias(m,r,mo,di,special,key_count,g,
                        sigma0=self.anchor_sigma0,bandwidth_floor=self.anchor_bandwidth_floor,clamp_min=self.anchor_bias_clamp)
                # Checkpoint the ray-distance graph as well as attention to fit the native1.1B model on24GB cards.
                bias=(checkpoint(bias_fn,mu,radius,moment,direction,self.anchor_gamma[i],use_reentrant=False)
                      if self.training and torch.is_grad_enabled() else bias_fn(mu,radius,moment,direction,self.anchor_gamma[i])).to(tokens.dtype)
            if bias is not None:self.anchor_bias_shape=list(bias.shape)
            tokens=block(tokens,latent.keys,latent.values,attn_bias=bias)
            if self.anchor_dynamic:
                delta=self.anchor_refine_mu[i](tokens[:,1:]).float()
                mu=mu+delta;rho=rho+self.anchor_refine_rho[i](tokens[:,1:]).squeeze(-1).float()
                radius=F.softplus(rho)+self.anchor_epsilon
                self.anchor_update_norms.append(delta.detach().norm(dim=-1).mean())
            if i+1 in (6,8,10,12):self.anchor_reads[i+1]=(mu.detach(),radius.detach())
        self.anchor_final=(mu,radius)
        return tokens[:,1:]

    def _decode_partial_gaussians(self,tokens):
        return self._decode_partial_gaussians_at(tokens,*self.anchor_final)

    def _decode_partial_gaussians_at(self,tokens,mu,radius):
        # Native scale/rotation activations are unchanged; raw XYZ becomes a local offset.
        raw=self.non_rgb_activation_head.deconv(tokens).reshape(tokens.shape[0],tokens.shape[1],64,10)
        native_value=self.non_rgb_activation_head(tokens)
        pos=mu[:,:,None,:]+radius[:,:,None,None]*raw[...,:3].float()
        return torch.cat((pos.flatten(1,2),native_value[...,3:].float()),-1)


def install(model,opt,dynamic):
    model.__class__=AnchoredQuerySplat
    width=model.opt.enc_embed_dim
    model.anchor_content=nn.Sequential(nn.Linear(7,width),nn.GELU(),nn.Linear(width,width))
    model.anchor_visual=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,width))
    model.anchor_dynamic=bool(dynamic);model.anchor_seed=None;model.anchor_reads={};model.anchor_final=None
    model.anchor_epsilon=opt.locusgs_radius_epsilon*4
    model.anchor_sigma0=opt.locusgs_sigma0
    model.anchor_bandwidth_floor=opt.locusgs_bandwidth_floor*4
    model.anchor_bias_clamp=opt.locusgs_bias_clamp
    if dynamic:
        model.anchor_refine_mu=nn.ModuleList([nn.Linear(width,3) for _ in range(12)])
        model.anchor_refine_rho=nn.ModuleList([nn.Linear(width,1) for _ in range(12)])
        for head in list(model.anchor_refine_mu)+list(model.anchor_refine_rho):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        model.anchor_gamma=nn.Parameter(torch.full((12,),float(opt.locusgs_gamma_raw_init)))
    return model
