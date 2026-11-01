"""Real-GPU roundtrip between fixed output units and meters; no learned scene/GT fitting."""
import torch
from tokengs.models.qp_anchor_joint4 import scale_gaussians,scale_cam_view
from tokengs.rendering.sh1 import SH1Renderer
from scripts.unposed_gt_protocol import UNITS_PER_METER

@torch.no_grad()
def verify(model,device):
    g=torch.zeros(1,40,23,device=device);x=torch.linspace(-.25,.25,40,device=device)
    g[0,:,0]=x;g[0,:,1]=x.flip(0)*.2;g[0,:,2]=2.;g[...,3]=.5;g[...,4:7]=.025;g[...,7]=1;g[...,11:14]=.3
    # Include near/far-excluded points to exercise the physical frustum too.
    g[0,0,2]=.05;g[0,1,2]=110.
    cam=torch.eye(4,device=device)[None,None].repeat(1,2,1,1);cam[0,1,3,0]=-.15
    K=torch.tensor([240.,241.,127.5,129.],device=device)[None,None].expand(1,2,4);bg=torch.zeros(3,device=device)
    meter=SH1Renderer(model.opt.evolve(znear=.1,zfar=100.))
    a=meter.render(g,cam,bg,K);b=model.metric_gs.render(scale_gaussians(g,UNITS_PER_METER),scale_cam_view(cam,UNITS_PER_METER),bg,K)
    errors={}
    for key in ['images_pred','alphas_pred','depths_pred']:
        target=a[key]*(UNITS_PER_METER if key=='depths_pred' else 1.)
        torch.testing.assert_close(b[key],target,rtol=3e-4,atol=5e-5);errors[key]=float((b[key]-target).abs().max())
    return dict(status='PASSED',output_units_per_meter=.15,physical_near_m=.1,physical_far_m=100.,background='black',meter_vs_output_max_abs=errors,GT_baseline_fit=False)
