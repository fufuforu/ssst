"""Opt-in classification readback; original query/mask/feedback stay unchanged."""
import torch
from torch import nn


class MaskSupportReadback(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ('object_support', 'scene_support'):
            raise ValueError('unknown mask support mode')
        self.mode = mode
        self.projection = nn.Sequential(nn.LayerNorm(256, eps=1e-5),
            nn.Linear(256,256), nn.GELU(), nn.Linear(256,256))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)
        self.last_observation = None

    def pool(self, features, masks, visibility):
        # Frozen context-generated Gaussian features and masks only. No target
        # cameras, RGB, class probabilities, GT, or gradient into the core.
        features, masks, visibility = features.detach(), masks.detach(), visibility.detach()
        if features.ndim!=3 or features.shape[-1]!=256 or masks.shape[:2]!=features.shape[:2]:
            raise ValueError('Gaussian feature/mask shapes differ')
        if masks.shape[-1]!=102 or visibility.shape!=features.shape[:2]+(1,):
            raise ValueError('expected100 query masks,2 stuff and per-Gaussian visibility')
        if not all(torch.isfinite(x).all() for x in (features,masks,visibility)):
            raise FloatingPointError('nonfinite mask support input')
        weights=visibility.clamp_min(0)*masks[...,:100] if self.mode=='object_support' else visibility.clamp_min(0).expand(-1,-1,100)
        pooled=weights.transpose(1,2)@features
        mass=weights.sum(1).unsqueeze(-1)
        return torch.where(mass>0,pooled/mass.clamp_min(1e-6),torch.zeros_like(pooled))

    def forward(self,q,features,masks,visibility):
        if q.ndim!=3 or q.shape[1:]!=(102,256):
            raise ValueError('expected100 object and2 stuff queries')
        if not torch.isfinite(q).all():raise FloatingPointError('nonfinite classification query')
        pooled=self.pool(features,masks,visibility)
        delta=self.projection(pooled)
        if not torch.isfinite(delta).all():raise FloatingPointError('nonfinite mask support residual')
        self.last_observation=dict(mode=self.mode,
            residual_query_norm_ratio_mean=float((delta.detach().norm(dim=-1)/q[:,:100].detach().norm(dim=-1).clamp_min(1e-6)).mean()))
        return torch.cat((q[:,:100].detach()+delta,q[:,100:].detach()),1)


def install(model,arm):
    if arm not in ('head_only','object_support','scene_support'):
        raise ValueError('unknown classification control')
    if hasattr(model.panoptic,'mask_support_readback'):
        raise ValueError('readback already installed')
    if arm!='head_only':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(271828)
            module=MaskSupportReadback(arm)
        parameter=next(model.panoptic.parameters())
        model.panoptic.mask_support_readback=module.to(device=parameter.device,dtype=parameter.dtype)
    model.requires_grad_(False).eval()
    for name,p in model.named_parameters():
        if name.startswith(('panoptic.class_ln.','panoptic.class_head.','panoptic.mask_support_readback.')):
            p.requires_grad_(True)
    model.classification_readback_arm=arm
