"""Diagnostic-only explicit renderer matrix blocks; never calls reverse AD."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import torch


def forbidden(*args,**kwargs):
    raise RuntimeError('Diagnostic forbids backward/autograd.grad')


@contextmanager
def no_reverse_ad():
    with ExitStack() as stack:
        for target in ('torch.autograd.grad','torch.autograd.backward','torch.Tensor.backward'):
            stack.enter_context(patch(target,forbidden))
        with torch.no_grad():yield


class ForwardLift:
    def __init__(self,semantic,instance,chunk=64):
        valid=(semantic>=2)&(semantic<20)&(instance>0)
        # Collision-free composite key, shared across context views.
        pairs=torch.stack([semantic[valid].long(),instance[valid].long()],-1)
        self.gt_pairs=torch.unique(pairs,dim=0)
        self.gt=torch.stack([(semantic==s)&(instance==i)&valid for s,i in self.gt_pairs],-1).float() if len(self.gt_pairs) else semantic.new_zeros((*semantic.shape,0),dtype=torch.float32)
        self.chunk=chunk;self.cache=None;self.checks={}

    def apply(self,features,gaussians,cam,intr,renderer):
        if self.cache is not None:
            if gaussians is not self.g or features.shape[2]!=1 or not bool((features==1).all()):
                raise ValueError('Unexpected extra renderer transpose call')
            return self.mass[None,:,None]
        self.g=gaussians
        assert gaussians.shape[0]==1 and features.shape[:2]==(1,2)
        _,v,d,h,w=features.shape;g=gaussians.shape[1]
        if self.gt.shape[:3]!=(v,h,w):raise ValueError('GT/render resolution mismatch')
        fm=features[0].permute(0,2,3,1).reshape(-1,d).float()
        gt=self.gt.reshape(v*h*w,-1)
        lifted=torch.empty((g,d),device=gaussians.device);mass=torch.empty(g,device=gaussians.device)
        contribution=torch.empty((g,gt.shape[1]),device=gaussians.device)
        summed=torch.zeros((v,h,w),device=gaussians.device)
        # Positive deterministic per-GS probe colors; four channels avoid channel-specific padding.
        idx=torch.arange(g,device=gaussians.device,dtype=torch.float32)
        probe=torch.stack([.5+.5*torch.sin(idx*(j+1)*.137) for j in range(4)],-1)
        recomposed=torch.zeros((v,h,w,4),device=gaussians.device)
        adjoint_right=torch.zeros((),device=gaussians.device,dtype=torch.float64)
        # Independent pixel probe for a dot-product identity (not the FM being lifted).
        pixel_index=torch.arange(v*h*w,device=gaussians.device,dtype=torch.float32)
        u=torch.stack([.5+.5*torch.cos(pixel_index*(j+1)*.019) for j in range(4)],-1)
        for start in range(0,g,self.chunk):
            end=min(g,start+self.chunk);c=end-start
            basis=torch.zeros((1,g,c),device=gaussians.device)
            basis[0,torch.arange(start,end,device=basis.device),torch.arange(c,device=basis.device)]=1.
            render=renderer.render_feature_channels(gaussians,basis,cam,intr)
            image=render['images_pred'][0]
            weights=image.permute(1,0,2,3).reshape(c,-1)
            if not bool(torch.isfinite(weights).all()) or float(weights.min())< -1e-6:raise ValueError('Invalid contribution weights')
            lifted[start:end]=weights@fm
            mass[start:end]=weights.sum(-1)
            contribution[start:end]=weights@gt
            summed+=image.sum(1)
            recomposed+=(weights.T@probe[start:end]).reshape(v,h,w,4)
            adjoint_right+=(probe[start:end].double()*(weights@u).double()).sum()
        direct=renderer.render_feature_channels(gaussians,probe[None],cam,intr)
        direct_image=direct['images_pred'][0].permute(0,2,3,1)
        alpha=direct['alphas_pred'][0,:,0]
        alpha_error=float((summed-alpha).abs().max())
        linear_error=float((recomposed-direct_image).abs().max())
        left=(direct_image.reshape(-1,4).double()*u.double()).sum()
        adj_error=float((left-adjoint_right).abs()/left.abs().clamp_min(1.))
        self.checks=dict(alpha_conservation_max_abs=alpha_error,feature_render_linearity_max_abs=linear_error,
                         adjoint_inner_product_relative_error=adj_error,gs=g,channels=d,chunk=self.chunk)
        if alpha_error>2e-4 or linear_error>2e-4 or adj_error>2e-5:
            raise ValueError('Forward lift equivalence failed: '+str(self.checks))
        self.cache=lifted[None];self.mass=mass;self.contribution=contribution
        self.evidence=torch.where(mass[:,None]>0,lifted/mass[:,None].clamp_min(1e-6),torch.zeros_like(lifted))
        self.gate=mass[:,None]/(mass[:,None]+1.)
        return self.cache
