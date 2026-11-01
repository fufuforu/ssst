# Portions derived from QuerySplat, Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0
"""Sealed ABC Baseline8 continuation and scratch QuerySplat-style D.

D ports QuerySplat 3465a1d decoder order, partial activation and appearance
activation. Explicit adaptation: width1024, context256, 1024 queries, native
Omega four-layer memory adapter, supervised geometry L6/L12 with shared
six-layer appearance decoder. No pretrained task weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tokengs.models.v4init_isolated_noca import LocusGSV4InitIsolatedNoCA,V4InitIsolatedNoCACore
from tokengs.models.olda4_initialization_ablation import V4Initialization
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.dual_stream_initialization import seed_anchors
from tokengs.models.gfm_dual_stream_components import gfm_grid,sample_anchor_gfm
from tokengs.models.object_locus_posefree_geometry import pixel_rays

ARCHS={a:'RECONSTRUCTION_ABCD_'+a+'_20261025_V1' for a in 'ABCD'}


class Appearance(nn.Module):
    def __init__(self,count=1024,residual=False):
        super().__init__();self.residual=residual
        # Official RGB + Pluecker patch sum, separate normalization and K/V.
        self.rgb=nn.Conv2d(3,1024,16,stride=16)
        self.plucker=nn.Conv2d(6,1024,16,stride=16)
        self.patch_norm=nn.LayerNorm(1024)
        self.norm=nn.LayerNorm(1024);self.kv=nn.Linear(1024,2048);self.key_norm=nn.LayerNorm(64)
        self.tokens=nn.Parameter(torch.zeros(count,1024))
        self.blocks=nn.ModuleList([DecoderBlock(1024,16,4.,qkv_bias=True,ffn_bias=True,qk_norm=True,init_values=5e-5) for _ in range(6)])
        self.output=nn.Linear(1024,64*(3 if residual else 13))
        if residual:nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def memory(self,rgb,omega):
        o,d=pixel_rays(omega['c2w'],omega['k'],256,256)
        # pixel_rays returns channels first; convert to world vectors.
        o=o.movedim(-3,-1);d=d.movedim(-3,-1)
        plucker=torch.cat((torch.cross(o,d,dim=-1),d),-1).permute(0,1,4,2,3)
        b,v=rgb.shape[:2]
        value=self.rgb(rgb.flatten(0,1))+self.plucker(plucker.flatten(0,1))
        value=self.patch_norm(value.flatten(2).transpose(1,2)).reshape(b,v*256,1024)
        keys,values=self.kv(self.norm(value)).chunk(2,-1)
        keys=keys.reshape(b,-1,16,64).transpose(1,2)
        values=values.reshape(b,-1,16,64).transpose(1,2)
        return self.key_norm(keys),values

    def forward(self,geometry,memory):
        h=(geometry.detach() if self.residual else geometry)+self.tokens[None]
        for block in self.blocks:
            h=checkpoint(block,h,*memory,use_reentrant=False) if self.training and torch.is_grad_enabled() else block(h,*memory)
        return self.output(h).reshape(h.shape[0],-1,3 if self.residual else 13)


def activate_geometry(raw):
    position,scale,rotation=raw.split((3,3,4),-1)
    position=position.sign()*position.abs().expm1()
    position=position+position.new_tensor([0.,0.,1.])
    scale=torch.exp((scale-(1-math.log(.075))).clamp(max=math.log(.075)))
    return position,scale,F.normalize(rotation,dim=-1)


def activate_appearance(raw):
    sh=raw[...,:12].reshape(*raw.shape[:-1],4,3)
    dc=(.5*sh[...,0,:].tanh())/.28209479177387814
    color=torch.cat((dc.unsqueeze(-2),.1*sh[...,1:,:].tanh()),-2).flatten(-2)
    return color,torch.sigmoid(raw[...,12:13]-2.)


class ABCDCore(V4InitIsolatedNoCACore):
    arm='A'
    def __init__(self,opt,config):
        super().__init__(opt,config)
        self.count=4096 if self.arm=='B' else 1024
        if self.arm=='B':
            self.v4_initialization.gs_tokens=nn.Parameter(torch.empty(4096,1024))
        if self.arm in ('C','D'):self.appearance=Appearance(residual=self.arm=='C')
        if self.arm=='D':
            # Reuse the same DecoderBlock primitives as official QuerySplat,
            # with no anchor PE, ray bias, local refinement or geometric query seed.
            del self.v4_initialization
            self.gs_tokens=nn.Parameter(.01*torch.randn(1024,1024))
            self.geometry_head=nn.Linear(1024,64*10)
            nn.init.trunc_normal_(self.geometry_head.weight,std=.002);nn.init.zeros_(self.geometry_head.bias)
            del self.activation_head
            self.reconstruction_image_size=tuple(opt.img_size)
            # Anchor decoder remains solely an adapter for detached semantic reads.
            # Its geometric refinement/PE parameters are never trained or loaded.
            for name,p in self.anchor_decoder.named_parameters():
                if not name.startswith('decoder_blocks.'):p.requires_grad_(False)

    def prepare_memory(self,omega,rgb,renderer):
        memory=super().prepare_memory(omega,rgb,renderer)
        if self.arm in ('C','D'):memory['appearance']=self.appearance.memory(rgb,omega)
        return memory

    def initialize_reconstruction_anchors(self,reconstruction_seed,cloud=None,memory=None):
        if self.arm=='D':
            tokens=self.gs_tokens[None];p,s,_=activate_geometry(self.geometry_head(tokens).reshape(1,-1,10))
            mu=p.reshape(1,self.count,64,3).mean(2);radius=s.reshape(1,self.count,64,3).mean((2,3)).clamp_min(1e-4)
            return dict(tokens=tokens,centers=mu,radii=radius,anchor_point_indices=torch.full((1024,),-1,device=tokens.device))
        seed=memory['v4_seed']
        # V4Initialization validates its count dynamically in this variant only.
        mu,rad,col,vis=(seed[k] for k in ('centers','radii','color','visual'))
        origin=mu.mean(1,keepdim=True);ell=(mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
        content=torch.cat(((mu-origin)/ell[:,None,None],(rad/ell[:,None])[...,None],col),-1)
        init=self.v4_initialization
        tokens=init.gs_tokens[None].expand(len(mu),-1,-1).clone()+init.anchor_content(content)+init.anchor_visual_projection(vis)
        return dict(tokens=tokens,centers=mu,radii=rad,anchor_point_indices=seed['source_indices'][0])

    def reconstruction_states(self,args,kwargs,memory):
        if self.arm!='D':
            states=super().reconstruction_states(args,kwargs,memory)
            if self.arm=='C':
                for i in (5,11):states[i]['gaussians']=self.decode_state(states[i],memory)
            return states
        h=args[0];latent=memory['latent'];states=[]
        for index,block in enumerate(self.blocks):
            h=checkpoint(block,h,latent.keys,latent.values,use_reentrant=False) if self.training and torch.is_grad_enabled() else block(h,latent.keys,latent.values)
            p,s,rot=activate_geometry(self.geometry_head(h).reshape(1,-1,10))
            mu=p.reshape(1,self.count,64,3).mean(2);rad=s.reshape(1,self.count,64,3).mean((2,3)).clamp_min(1e-4)
            positive=(rad-self.anchor_decoder.epsilon).clamp_min(1e-6)
            state=dict(tokens=h,mu=mu,radii=rad,rho=positive+torch.log(-torch.expm1(-positive)),layer=index+1,anchor_update=h.new_zeros(()),radius_update=h.new_zeros(()),beta=0.)
            if index in (5,11):
                color,alpha=activate_appearance(self.appearance(h,memory['appearance']))
                state['gaussians']=torch.cat((p,alpha,s,rot,color),-1)
            states.append(state)
        return states

    def semantic_gaussians(self,state,gaussians):
        if self.arm=='C':
            return self.activation_head(state['tokens'],state['mu'],state['radii']).detach()
        return gaussians.detach()

    def decode_state(self,state,memory):
        if 'gaussians' in state:return state['gaussians']
        base=super().decode_state(state,memory)
        if self.arm=='C':
            delta=self.appearance(state['tokens'],memory['appearance'])
            # Bounded RGB residual, exact identity for zero output, no clamp
            # derivative ambiguity at initialization.
            base=torch.cat((base[...,:11],base[...,11:14]+.25*delta.tanh()),-1)
        return base

    def parameter_groups(self):
        if self.arm=='D':
            result=dict(shared=[],reconstruction=[],understanding=[])
            for n,p in self.named_parameters():
                group='reconstruction' if n.startswith(('reconstruction_memory.','blocks.','anchor_decoder.','geometry_head.','gs_tokens','appearance.')) else 'understanding'
                result[group].append(p)
        else:result=super().parameter_groups()
        if self.arm=='C':
            extra=list(self.appearance.parameters())
            ids={id(p) for p in extra}
            result['understanding']=[p for p in result['understanding'] if id(p) not in ids]
            result['reconstruction']+=extra
        return {k:[p for p in ps if p.requires_grad] for k,ps in result.items()}


class ReconstructionABCD(LocusGSV4InitIsolatedNoCA):
    arm='A'
    core_type=ABCDCore
    def encode_context(self,rgb):
        omega,cloud,_=LocusGSUnifiedQueryV1.encode_context(self,rgb)
        memory=self.core.prepare_memory(omega,rgb,self.gs)
        if self.arm!='D':
            mu,rad,col,source=seed_anchors(omega['points'],omega['confidence'],rgb,self.core.count,return_source_indices=True,minimum_candidates=4096)
            if source[0].unique().numel()!=self.core.count:raise ValueError('Duplicated reconstruction source indices')
            memory['v4_seed']=dict(centers=mu,radii=rad,color=col,source_indices=source,visual=sample_anchor_gfm(gfm_grid(omega['patch_layers']),source,image_hw=tuple(omega['points'].shape[2:4])))
        return omega,cloud,memory

    def step_loss(self,*args,**kwargs):
        # D has no iterative anchor refinement. Same objective and monitoring
        # except displacement is omitted, rather than fabricated from old heads.
        if self.arm!='D':return super().step_loss(*args,**kwargs)
        out,metrics=LocusGSUnifiedQueryV1.step_loss(self,*args,**kwargs)
        for k in ('num_superpoint_queries','num_reconstruction_anchors','num_object_queries','num_gaussians'):metrics[k]=out['prediction'][k]
        metrics['gradient_isolation']=1
        return out,metrics

    def parameter_groups(self):
        return {k:[p for p in ps if p.requires_grad] for k,ps in super().parameter_groups().items()}

    def __init__(self,opt,**kwargs):
        if self.arm=='D' and kwargs.get('renderer') is None:
            from tokengs.rendering.sh1 import SH1Renderer
            kwargs['renderer']=SH1Renderer(opt)
        super().__init__(opt,**kwargs)


def variants():
    result={}
    for arm in 'ABCD':
        core=type('Reconstruction'+arm+'Core',(ABCDCore,),dict(arm=arm))
        model=type('Reconstruction'+arm,(ReconstructionABCD,),dict(arm=arm,core_type=core,architecture_name=ARCHS[arm]))
        result['reconstruction_'+arm.lower()]=model
    return result
