"""Degree-one SH evaluated per camera, same pinned gsplat RGB rasterizer."""
import torch
from torch.nn import functional as F
from tokengs.rendering.gs import GaussianRenderer


def sh1_rgb(sh,direction):
    x,y,z=direction.unbind(-1);c=.4886025119029199
    basis=torch.stack((torch.ones_like(x)*.28209479177387814,-c*y,c*z,-c*x),-1)
    return ((sh*basis[...,None]).sum(-2)+.5).clamp_min(0.)


class SH1Renderer(GaussianRenderer):
    def render(self,gaussians,cam_view,bg_color=None,intrinsics=None):
        if gaussians.shape[-1]!=23:raise ValueError('D requires full SH1 GS23')
        if self.opt.deferred_bp:raise ValueError('Use standard differentiable renderer')
        views=[]
        for i in range(cam_view.shape[1]):
            center=torch.linalg.inv(cam_view[:,i].transpose(-1,-2))[...,:3,3]
            direction=F.normalize(gaussians[...,:3]-center[:,None],dim=-1)
            rgb=sh1_rgb(gaussians[...,11:].reshape(*gaussians.shape[:2],4,3),direction)
            gs=torch.cat((gaussians[...,:11],rgb),-1)
            views.append(super().render(gs,cam_view[:,i:i+1],bg_color,intrinsics[:,i:i+1]))
        return {k:torch.cat([v[k] for v in views],1) for k in views[0]}
