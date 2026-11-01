"""One Context-only scene gauge for GT-conditioned geometry, losses and export."""
import torch
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.dual_stream_initialization import unproject
PUBLIC_UNITS_PER_METER=.15
NATIVE_UNITS_PER_PUBLIC=4.
PROTOCOL='GT_context_reprojected_Omega_depth_bounded_context_gauge_v2'
SCALE_MIN=.25
SCALE_MAX=4.
BASELINE_EPS=1e-6


def gt_decoder(batch,views=None,gauge=None):
    cam=batch['cam_view_all'];intr=batch['intrinsics_all']
    if cam.ndim!=4 or cam.shape[0]!=1 or cam.shape[-2:]!=(4,4) or intr.shape!=cam.shape[:2]+(4,):raise ValueError('GT camera batch contract differs')
    if not torch.isfinite(cam).all() or not torch.isfinite(intr).all() or not (intr[...,:2]>0).all():raise ValueError('Invalid GT camera values')
    if not torch.allclose(cam[:,0],torch.eye(4,device=cam.device,dtype=cam.dtype)[None],rtol=0,atol=2e-5):raise ValueError('GT cameras must be first-context canonical')
    if views is not None:cam=cam[:,:views];intr=intr[:,:views]
    if gauge is not None:
        a=float(gauge['GT_public_to_generated_scale'])
        if not SCALE_MIN<=a<=SCALE_MAX:raise ValueError('Invalid effective scene scale')
        cam=cam.clone();cam[...,3,:3]*=a
    return ModelInputDecoder(cam_view=cam,intrinsics=intr)


def context_camera_batch(batch):
    # Deliberately retain only first2 camera values; never bind Target cameras, RGB or GT depth to generation.
    out={k:batch[k][:,:2].detach() for k in ['cam_view_all','intrinsics_all','frame_ids']}
    if out['frame_ids'].shape!=(1,2):raise ValueError('Exactly two Context cameras required')
    gt_decoder(out);return out


@torch.no_grad()
def context_gauge(context,predicted_c2w):
    if predicted_c2w.shape!=(1,2,4,4) or not torch.isfinite(predicted_c2w).all():raise ValueError('Invalid RGB-only Context camera prediction')
    gt=gt_decoder(context);c2w=torch.linalg.inv(gt.cam_view.transpose(-1,-2))
    dp=float((predicted_c2w[0,1,:3,3]-predicted_c2w[0,0,:3,3]).norm());dg=float((c2w[0,1,:3,3]-c2w[0,0,:3,3]).norm())
    raw=dp/dg if dg>BASELINE_EPS else None
    if dp<=BASELINE_EPS or dg<=BASELINE_EPS:a=1.;reason='degenerate_Context_baseline_unit_fallback'
    elif raw<SCALE_MIN:a=SCALE_MIN;reason='lower_scale_bound'
    elif raw>SCALE_MAX:a=SCALE_MAX;reason='upper_scale_bound'
    else:a=raw;reason='unclipped_Context_baseline_scale'
    return dict(protocol=PROTOCOL,GT_public_to_generated_scale=a,raw_GT_public_to_generated_scale=raw,
        generated_units_per_meter=PUBLIC_UNITS_PER_METER*a,meters_per_generated_unit=1/(PUBLIC_UNITS_PER_METER*a),
        scale_bounds=[SCALE_MIN,SCALE_MAX],baseline_epsilon=BASELINE_EPS,scale_rule=reason,
        predicted_Context_baseline=dp,GT_Context_baseline_public=dg,Context_frame_ids=context['frame_ids'][0].cpu().tolist(),
        Context_only=True,GT_Context_K=True,GT_Context_extrinsics=True,Omega_depth_reprojected=True,
        target_RGB_used_for_camera=False,Target_cameras_used_for_generation=False,GT_depth_used_for_alignment=False,no_window_filtering=True)


@torch.no_grad()
def reframe_omega(omega,context):
    gauge=context_gauge(context,omega['c2w']);decoder=gt_decoder(context,gauge=gauge)
    c2w=torch.linalg.inv(decoder.cam_view.transpose(-1,-2));intr=decoder.intrinsics
    k=intr.new_zeros((*intr.shape[:-1],3,3));k[...,0,0]=intr[...,0];k[...,1,1]=intr[...,1];k[...,0,2]=intr[...,2];k[...,1,2]=intr[...,3];k[...,2,2]=1
    depth=omega['depth'];valid=torch.isfinite(depth)&(depth>0)
    points=unproject(depth,k,c2w);points=points.masked_fill(~valid.permute(0,1,3,4,2),float('nan'))
    # Check real point/camera projection at a sparse Context grid on every forward (no GT depth involved).
    sample=points[:,:,::16,::16];w2c=torch.linalg.inv(c2w)
    xyz=torch.einsum('bvij,bvhwj->bvhwi',w2c[...,:3,:3],sample)+w2c[...,:3,3][:,:,None,None]
    uv=torch.einsum('bvij,bvhwj->bvhwi',k,xyz);positive=torch.isfinite(uv).all(-1)&(uv[...,2]>0)
    if not positive.any():raise ValueError('No finite positive Context reprojection')
    y,x=torch.meshgrid(torch.arange(0,depth.shape[-2],16,device=depth.device)+.5,torch.arange(0,depth.shape[-1],16,device=depth.device)+.5,indexing='ij')
    expected=torch.stack([x,y],-1);error=(uv[...,:2]/uv[...,2:]-expected).norm(dim=-1)[positive]
    gauge['Context_reprojection_max_px']=float(error.max())
    if gauge['Context_reprojection_max_px']>.02:raise ValueError('Context generation/GT camera projection disagrees')
    # Frozen features/confidence/predicted depths are unchanged. Replace geometry BEFORE cloud, seeds and native input construction.
    return dict(omega,c2w=c2w,k=k,points=points),gauge


def camera_audit(batch,gauge):
    gt_decoder(batch,gauge=gauge)
    if batch['frame_ids'][0,:2].cpu().tolist()!=gauge['Context_frame_ids']:raise ValueError('Gauge belongs to a different Context pair')
    return dict(gauge,GT_intrinsics=True,GT_extrinsics=True,camera_source='processed GT K/c2w with one Context-only scene scale',
        public_units_per_meter=gauge['generated_units_per_meter'],native_units_per_meter=gauge['generated_units_per_meter']*4,
        frame_ids=batch['frame_ids'][0].cpu().tolist())
