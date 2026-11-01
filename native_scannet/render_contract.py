"""Actual GPU comparison of native SH1 and explicit SH evaluation, same rasterizer."""
import torch
from torch.nn import functional as F
from gsplat.rendering import rasterization
from native_scannet.core import options,camera_matrix
from scripts.rendering.gs import GaussianRenderer


def verify(device):
    generator=torch.Generator(device=device).manual_seed(963)
    xyz=torch.randn(1,64,3,generator=generator,device=device)*.12;xyz[...,2]+=1.
    alpha=torch.full((1,64,1),.5,device=device)
    scale=torch.full((1,64,3),.03,device=device)
    quat=torch.zeros(1,64,4,device=device);quat[...,0]=1
    sh=torch.randn(1,64,12,generator=generator,device=device)*.05
    original=torch.cat((xyz,alpha,scale,quat,sh),-1)
    native=original.clone().requires_grad_();manual=original.clone().requires_grad_()
    cams=torch.eye(4,device=device).expand(1,2,4,4).clone();cams[:,1,3,0]=-.1
    intr=torch.tensor([256.,256.,128.,128.],device=device).expand(1,2,4)
    bg=torch.full((3,),.5,device=device)
    actual=GaussianRenderer(options()).render(native,cams,bg,intr)['images_pred']
    expected=[]
    for view in range(2):
        camera=torch.linalg.inv(cams[:,view].transpose(-1,-2))[...,:3,3]
        direction=F.normalize(manual[...,:3]-camera[:,None],dim=-1)
        x,y,z=direction.unbind(-1);c=.4886025119029199
        basis=torch.stack((torch.ones_like(x)*.28209479177387814,-c*y,c*z,-c*x),-1)
        color=((manual[...,11:].reshape(1,64,4,3)*basis[...,None]).sum(-2)+.5).clamp_min(0)
        image,_,_=rasterization(means=manual[0,:,:3],quats=manual[0,:,7:11],scales=manual[0,:,4:7],
            opacities=manual[0,:,3],colors=color[0],viewmats=cams[0,view:view+1].transpose(-1,-2),
            Ks=camera_matrix(intr)[0,view:view+1],width=256,height=256,near_plane=.025,far_plane=125.,
            backgrounds=bg[None],render_mode='RGB+ED',packed=False)
        expected.append(image[...,:3].permute(0,3,1,2))
    expected=torch.stack(expected,1)
    actual.square().mean().backward();expected.square().mean().backward()
    forward_error=float((actual-expected).abs().max())
    gradient_error=float((native.grad-manual.grad).abs().max())
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=3e-5)
    torch.testing.assert_close(native.grad,manual.grad,rtol=.05,atol=1e-4)
    return dict(status='PASSED',forward_max_abs=forward_error,gradient_max_abs=gradient_error,
                all23_gaussian_channels_compared=True,native_SH_degree=1)
