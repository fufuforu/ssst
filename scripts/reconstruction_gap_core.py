"""Explicitly bounded diagnostic helpers; no model or novel-GT optimization."""
import torch,math

def align_gt_cameras(gt,predicted_context):
    # Two centers alone do not identify rotation. Use both camera orientations
    # to identify R, baseline norm for scale, and mean center for translation.
    gt=gt.double();p=predicted_context.double()
    u,_,vh=torch.linalg.svd((p[:,:3,:3]@gt[:2,:3,:3].transpose(-1,-2)).sum(0))
    fix=torch.eye(3,device=gt.device,dtype=gt.dtype);fix[2,2]=torch.det(u@vh);r=u@fix@vh
    bg=(gt[1,:3,3]-gt[0,:3,3]).norm();bp=(p[1,:3,3]-p[0,:3,3]).norm()
    if min(float(bg),float(bp))<1e-5:return None,dict(reliable=False,reason='degenerate context baseline',gt_baseline=float(bg),predicted_baseline=float(bp))
    scale=bp/bg;t=p[:,:3,3].mean(0)-scale*(r@gt[:2,:3,3].mean(0))
    aligned=gt.clone();aligned[:,:3,:3]=r@gt[:,:3,:3];aligned[:,:3,3]=scale*(gt[:,:3,3]@r.T)+t
    rotation=aligned[:2,:3,:3].transpose(-1,-2)@p[:,:3,:3]
    angle=torch.rad2deg(torch.acos(((rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
    center=(aligned[:2,:3,3]-p[:,:3,3]).norm(dim=-1)/bp
    reliable=bool(torch.isfinite(aligned).all() and angle.max()<15 and center.max()<.15)
    return aligned.float() if reliable else None,dict(reliable=reliable,scale=float(scale),rotation=r.cpu().tolist(),translation=t.cpu().tolist(),
        gt_baseline=float(bg),predicted_baseline=float(bp),context_rotation_deg=angle.cpu().tolist(),context_center_residual_over_baseline=center.cpu().tolist(),
        rotation_identification='both Context orientations; centers alone insufficient',thresholds=dict(rotation_deg=15,center_over_baseline=.15),
        limitation='GT-assisted pose+intrinsics diagnostic conditional on orientation/baseline Sim3; not a uniquely identifiable causal attribution')

def appearance_gaussians(fixed,colours):
    if fixed.shape[-1]!=14 or colours.shape!=fixed[...,11:].shape:raise ValueError('RGB-only Gaussian schema differs')
    return torch.cat((fixed[...,:11].detach(),colours),-1)
