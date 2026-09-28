"""Anchor-Group V1: all-anchor soft grouping for LocusGS."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import math

from tokengs.models.canonical_recon import canonical_layer_loss
from tokengs.models.canonical_recon_models import LocusGSRecon, _full_supervision, patch_plucker_rays
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.locusgs_recon import LocusGSAnchorDecoder, sinusoidal_positional_encoding, anchor_ray_geometric_bias

NUM_THING, NUM_STUFF = 100, 2
NUM_QUERIES, VOID_INDEX, NUM_REGION_CHANNELS = 102, 102, 103
STATE_DIM, ID_DIM, GROUP_TEMPERATURE = 256, 16, 0.1
GS_DECODE_RADIUS = 0.15

def anchor_group_understanding_weight(step: int) -> float:
    if step <= 200:
        return 0.0
    if step < 1000:
        return (step - 200) / 800.0
    return 1.0

def anchor_group_lr_multiplier(step: int, total_steps: int = 5000) -> float:
    if step <= 0:
        return 0.0
    if step <= 200:
        return step / 200.0
    t = (step - 200) / (total_steps - 200)
    return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * t))


class AnchorGroupController(nn.Module):
    """Independent grouping controller. Void is a token logit, never a query."""
    def __init__(self, opt):
        super().__init__()
        C, D, P = int(opt.enc_embed_dim), STATE_DIM, int(opt.dec_patch_size) ** 2
        self.C, self.D, self.patches = C, D, P
        self.ln_h, self.proj_h, self.proj_m = nn.LayerNorm(C), nn.Linear(C,D), nn.Linear(4,D)
        self.ln_x = nn.LayerNorm(D)
        self.ln_e, self.proj_e = nn.LayerNorm(D), nn.Linear(D,ID_DIM)
        self.ln_u, self.proj_u = nn.LayerNorm(D), nn.Linear(D,ID_DIM)
        self.query_init = nn.Parameter(torch.empty(NUM_QUERIES,D))
        self.gru = nn.GRUCell(D,D); self.ln_gru = nn.LayerNorm(D)
        self.ffn_fc1, self.ffn_fc2 = nn.Linear(D,2*D), nn.Linear(2*D,D)
        self.ln_ffn = nn.LayerNorm(D)
        self.token_void = nn.Linear(D,1)
        self.thing_classifier = nn.Linear(D,19)
        self.ln_vq, self.proj_vq = nn.LayerNorm(D), nn.Linear(D,D)
        self.ln_fx, self.ln_fm = nn.LayerNorm(D), nn.LayerNorm(D)
        self.proj_wh, self.proj_wmu, self.proj_wr = nn.Linear(2*D+6,C), nn.Linear(2*D+6,3), nn.Linear(2*D+6,1)
        self.proj_wgs = nn.Linear(2*D+6,P*3)
        self.ln_de, self.proj_de = nn.LayerNorm(C), nn.Linear(C,P*ID_DIM)
        self.proj_off = nn.Linear(3,ID_DIM); self.ln_void = nn.LayerNorm(C); self.proj_gvoid = nn.Linear(C,P)
        self._init_weights()

    @staticmethod
    def _xavier(module: nn.Linear) -> None:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    @staticmethod
    def _zero(module: nn.Linear) -> None:
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def _init_weights(self) -> None:
        # Keep every shared controller primitive exactly aligned with
        # InstanceStateController._init_weights; this is initialization parity,
        # not parameter sharing between the architectures.
        for module in (self.proj_h,self.proj_m,self.proj_e,self.proj_u,
                       self.ffn_fc1,self.ffn_fc2,self.proj_vq,self.proj_off,
                       self.thing_classifier):
            self._xavier(module)
        for module in (self.token_void,self.proj_wgs,self.proj_de,self.proj_gvoid,
                       self.proj_wh,self.proj_wmu,self.proj_wr):
            self._zero(module)
        hidden=self.gru.hidden_size
        with torch.no_grad():
            for name,param in self.gru.named_parameters():
                if name.startswith("weight_ih") or name.startswith("weight_hh"):
                    for gate in range(3):
                        nn.init.xavier_uniform_(param[gate*hidden:(gate+1)*hidden])
                else:
                    nn.init.zeros_(param)
            nn.init.normal_(self.query_init,std=0.02)
        for ln in (self.ln_h,self.ln_x,self.ln_e,self.ln_u,self.ln_gru,
                   self.ln_ffn,self.ln_vq,self.ln_fx,self.ln_fm,self.ln_de,
                   self.ln_void):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

    def encode_token(self, tokens, mu, radii, ell):
        feat = torch.cat([mu / ell.reshape(-1,1,1), torch.log((radii / ell[:,None]).clamp(1e-4,1e4)).unsqueeze(-1)],-1)
        return self.ln_x(self.proj_h(self.ln_h(tokens)) + self.proj_m(feat))

    def assign_group(self, a, q, void_logit):
        e = F.normalize(self.proj_e(self.ln_e(a)), dim=-1, eps=1e-6)
        u = F.normalize(self.proj_u(self.ln_u(q)), dim=-1, eps=1e-6)
        group_logits = torch.einsum("btd,bqd->btq",e,u) / GROUP_TEMPERATURE
        return torch.softmax(torch.cat([group_logits,void_logit],-1),dim=-1)

    def update_group_states(self, a, mu, q, void_logit, ell):
        A_pre = self.assign_group(a,q,void_logit)
        mass = A_pre[:,:,:NUM_QUERIES].sum(1)
        w = A_pre[:,:,:NUM_QUERIES] / (mass.unsqueeze(1)+1e-6)
        z = torch.einsum("btq,btd->bqd",w,a)
        v = self.ln_gru(self.gru(z.reshape(-1,self.D),q.reshape(-1,self.D))).reshape_as(q)
        q_new = self.ln_ffn(v+self.ffn_fc2(F.gelu(self.ffn_fc1(v))))
        q_new = torch.where((mass < 1e-4).unsqueeze(-1),q,q_new)
        A_post = self.assign_group(a,q_new,void_logit)
        wt = A_post[:,:,:NUM_THING]
        mt = wt.sum(1)
        wn = wt/(mt.unsqueeze(1)+1e-6)
        c = torch.einsum("btq,btd->bqd",wn,mu)
        diff = mu.unsqueeze(2)-c.unsqueeze(1)
        var = torch.einsum("btq,btqd->bqd",wn,diff.pow(2))
        s = torch.sqrt(var+(0.05*ell).reshape(-1,1,1).pow(2))
        lo,hi=(0.05*ell).reshape(-1,1,1),(2.0*ell).reshape(-1,1,1)
        s=torch.clamp(s,lo,hi)
        return q_new,A_pre,A_post,c,s

    def token_message(self,x,mu,A,q,c,s,ell):
        vq=self.proj_vq(self.ln_vq(q)); ag=A[:,:,:NUM_QUERIES]
        m=torch.einsum("btq,bqd->btd",ag,vq)/(ag.sum(-1,keepdim=True).clamp_min(1e-6))
        at=A[:,:,:NUM_THING]; t=at.sum(-1,keepdim=True)
        delta=c[:,:NUM_THING].unsqueeze(2)-mu.unsqueeze(1)
        d=torch.einsum("btq,bqtd->btd",at,delta)/(ell[:,None,None]*t.clamp_min(1e-6))
        v=torch.einsum("btq,bqd->btd",at,torch.log(s/ell[:,None,None]))/t.clamp_min(1e-6)
        return torch.cat([self.ln_fx(x),self.ln_fm(m),d,v],-1),t


class AnchorGroupDecoder(LocusGSAnchorDecoder):
    """LocusGS decoder with 1024 anchors retained at every registered layer."""
    def __init__(self,opt,blocks):
        super().__init__(opt,blocks); self.state_layers=tuple(opt.instance_state_layers)

    def forward_group(self,tokens,latent,patch_rays,ctrl):
        # Run the canonical decoder unchanged. Grouping is a read/update branch over
        # its emitted states; with beta=0 it cannot alter reconstruction tensors.
        states,ray_stats=super().forward(tokens,latent,patch_rays)
        B=tokens.shape[0]; q=ell=None
        for st in states:
            layer=st["layer"]; mu,radii=st["mu"],st["radii"]
            if layer==self.state_layers[0]:
                ell=torch.clamp((mu-mu.mean(1,keepdim=True)).pow(2).sum(-1).mean(-1).sqrt(),min=.05).detach()
                q=ctrl.query_init.unsqueeze(0).expand(B,-1,-1)
            if layer in self.state_layers:
                a=ctrl.encode_token(st["tokens"],mu,radii,ell); void=ctrl.token_void(a)
                q,Apre,Apost,c,s=ctrl.update_group_states(a,mu,q,void,ell)
                Fmsg,_=ctrl.token_message(a,mu,Apost,q,c,s,ell)
                st.update(q=q,A_pre=Apre,A_post=Apost,anchor_embedding=a,c=c,s=s,F=Fmsg,ell=ell,beta=0.0)
            else: st.update(q=q,A_pre=None,A_post=None,anchor_embedding=None,c=None,s=None,F=None,ell=ell,beta=0.0)
            st["fps_index"]=None
        return states,ray_stats


class LocusGSAnchorGroupRecon(LocusGSRecon):
    architecture_name="LOCUSGS_ANCHOR_GROUP_V1"
    def __init__(self,opt):
        super().__init__(opt)
        original_decoder=self.anchor_decoder
        self.anchor_decoder=AnchorGroupDecoder(opt,self.enc_dec_backbone.decoder_blocks)
        self.anchor_decoder.load_state_dict(original_decoder.state_dict(),strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(getattr(opt,"anchor_group_init_seed",31415)))
            self.anchor_group=AnchorGroupController(opt)
        self.instance_state_layers=tuple(opt.instance_state_layers); self.understanding_step=0

    def decode_group(self,model_input):
        latent=self.forward_encoder(model_input.encoder)
        rays=patch_plucker_rays(model_input.encoder.rays_os,model_input.encoder.rays_ds,patch_size=int(self.opt.patch_size))
        return self.anchor_decoder.forward_group(self.get_gs_tokens(batch_size=model_input.batch_size),latent,rays,self.anchor_group)

    def _gaussians_from_group(self,state):
        # beta is structurally fixed at zero; this exactly preserves the canonical head.
        return self.activation_head(state["tokens"],state["mu"],state["radii"])

    def _group_readout(self,final,gaussians,decoder):
        ctrl=self.anchor_group; B,T=final["tokens"].shape[:2]; P=gaussians.shape[1]//T
        A=final["A_post"]; A_g=A[:,:,None,:].expand(B,T,P,NUM_REGION_CHANNELS).reshape(B,T*P,NUM_REGION_CHANNELS)
        et=F.normalize(ctrl.proj_e(ctrl.ln_e(final["anchor_embedding"])),dim=-1,eps=1e-6)
        de=ctrl.proj_de(ctrl.ln_de(final["tokens"])).reshape(B,T,P,ID_DIM)
        xyz=gaussians[...,:3].reshape(B,T,P,3); off=((xyz-final["mu"][:,:,None,:])/GS_DECODE_RADIUS).clamp(-2,2)
        eg=F.normalize(et[:,:,None,:]+.25*torch.tanh(de)+.1*torch.tanh(ctrl.proj_off(off)),dim=-1,eps=1e-6).reshape(B,T*P,ID_DIM)
        ch=self.gs.render_feature_channels(gaussians,torch.cat([A_g,eg],-1),decoder.cam_view,decoder.intrinsics)
        M=ch["images_pred"][...,:NUM_REGION_CHANNELS,:,:]; E=ch["images_pred"][...,NUM_REGION_CHANNELS:,:,:]; alpha=ch["alphas_pred"]
        pc=torch.softmax(ctrl.thing_classifier(final["q"][:,:NUM_THING]),-1)
        S=torch.zeros(B,M.shape[1],20,*M.shape[-2:],device=M.device,dtype=M.dtype); things=M[:,:,:NUM_THING]
        S[:,:,0]=M[:,:,100]; S[:,:,1]=M[:,:,101]
        for k in range(2,20): S[:,:,k]=(things*pc[:,:,k-2].reshape(B,1,100,1,1)).sum(2)
        Svoid=(1-alpha)+M[:,:,102:103]+(things*pc[:,:,18].reshape(B,1,100,1,1)).sum(2).unsqueeze(2)
        logits=ctrl.thing_classifier(final["q"][:,:100]); pad=torch.full_like(logits[:,:,:1],-1e4)
        return dict(gaussians=gaussians,assignment=A_g,anchor_assignment=A,region_mass=M,semantic_scores=S,pixel_void_mass=Svoid,identity_render=E,alpha=alpha,p_class=pc,thing_class_logits=torch.cat([pad,pad,logits],-1))

    def forward_anchor_group(self,model_input,*,render_decoder_input=None,context_decoder=None,coupled=False,step=None):
        if coupled: raise RuntimeError("Anchor-Group V1 uses joint optimization through shared anchor/reconstruction features and does not use the legacy beta coupling path.")
        decoder=render_decoder_input or model_input.decoder; ctx=context_decoder or decoder
        states,_=self.decode_group(ModelInput(model_input.encoder,decoder)); final=states[-1]
        gs=self._gaussians_from_group(final); rec=self._reconstruction_from_gaussians(gs); render=self.render_reconstruction(rec,decoder)
        out=dict(reconstruction=rec,gaussians=gs,render=render,states=states,beta=0.0); out.update(self._group_readout(final,gs,ctx)); return out

    def forward_reconstruction_only(self,model_input,*,render_decoder_input=None): return self.forward_anchor_group(model_input,render_decoder_input=render_decoder_input)
    def _decode(self,model_input,decoder_input): return self.decode_group(ModelInput(model_input.encoder,decoder_input))

    def forward_instance_state(self,model_input,*,render_decoder_input=None,context_decoder=None,coupled=False,step=None):
        return self.forward_anchor_group(model_input,render_decoder_input=render_decoder_input,context_decoder=context_decoder,coupled=coupled,step=step)

    def _layer_objective(self,states,decoder,supervision):
        total=None; metrics={}
        for layer,w in zip(self.supervised_layers,self.layer_weights):
            st=states[layer-1]; gs=self._gaussians_from_group(st); rr=self.render_reconstruction(self._reconstruction_from_gaussians(gs),decoder)
            ll=canonical_layer_loss(opt=self.opt,img_size=self.img_size,render_results=rr,supervision=supervision,decoder_input=decoder,gaussians=gs,anchor_centers=st["mu"],anchor_weight=float(self.opt.canonical_anchor_visibility_weight))
            total=ll["loss"]*w if total is None else total+ll["loss"]*w; metrics.update({f"{k}_layer{layer}":v for k,v in ll.items() if k!="loss"})
        return total,metrics

    def step_loss(self,batch,*,step,phase="train",coupled=False,rseg_override=None,understanding_weight_override=None):
        from tokengs.models.anchor_group_loss import anchor_group_losses
        del phase
        if coupled: raise RuntimeError("Anchor-Group V1 uses joint optimization through shared anchor/reconstruction features and does not use the legacy beta coupling path.")
        mi,_=split_data(batch,self.opt); dec=ModelInputDecoder(cam_view=batch["cam_view_all"],intrinsics=batch["intrinsics_all"]); ctx=ModelInputDecoder(cam_view=batch["cam_view_all"][:,:2],intrinsics=batch["intrinsics_all"][:,:2])
        pred=self.forward_anchor_group(ModelInput(mi.encoder,dec),render_decoder_input=dec,context_decoder=ctx,coupled=False,step=step)
        recon,metrics=self._layer_objective(pred["states"],dec,_full_supervision(batch)); seg,sm=anchor_group_losses(pred,batch,self.opt)
        uweight=anchor_group_understanding_weight(int(step)) if understanding_weight_override is None else float(understanding_weight_override)
        metrics.update(sm); metrics["loss_recon"]=recon; metrics["loss_understanding"]=seg
        metrics["understanding_weight"]=uweight
        metrics["loss"]=recon+uweight*seg
        metrics["loss_total"]=metrics["loss"]
        return {"prediction":pred},metrics

    def forward(self,data,skip_loss=False):
        del skip_loss
        if isinstance(data,ModelInput): return self.forward_anchor_group(data)
        if isinstance(data,dict): return self.step_loss(data,step=self.understanding_step)[0]
        raise TypeError(f"unsupported input {type(data)!r}")
