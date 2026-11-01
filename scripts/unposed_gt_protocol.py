"""GT poses are render-only; known Context K and RGB-predicted poses guide generation."""
import torch
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.dual_stream_initialization import unproject
PROTOCOL='Unposed_RGB_K_fixed_metric_GT_render_scale_ablation_v1'
UNITS_PER_METER=.15
NEAR_PUBLIC=.015
FAR_PUBLIC=15.


def validate_K(intrinsics):
    if intrinsics.shape!=(1,2,4) or not torch.isfinite(intrinsics).all() or not (intrinsics[...,:2]>0).all():raise ValueError('Known Context pixel K must be [1,2,4],finite,positive focal')


@torch.no_grad()
def context_geometry(omega,intrinsics):
    validate_K(intrinsics)
    if omega['c2w'].shape!=(1,2,4,4) or not torch.isfinite(omega['c2w']).all():raise ValueError('Invalid RGB-predicted Context pose')
    if not torch.allclose(omega['c2w'][:,0],torch.eye(4,device=omega['c2w'].device,dtype=omega['c2w'].dtype)[None],atol=2e-5,rtol=0):raise ValueError('Predicted geometry must use the first Context frame')
    k=intrinsics.new_zeros((1,2,3,3));k[...,0,0]=intrinsics[...,0];k[...,1,1]=intrinsics[...,1];k[...,0,2]=intrinsics[...,2];k[...,1,2]=intrinsics[...,3];k[...,2,2]=1
    depth=omega['depth'];points=unproject(depth,k,omega['c2w']);valid=torch.isfinite(depth)&(depth>0)
    points=points.masked_fill(~valid.permute(0,1,3,4,2),float('nan'))
    return dict(omega,k=k,points=points)


def gt_decoder(batch,views=None):
    cam=batch['cam_view_all'];intr=batch['intrinsics_all']
    if cam.ndim!=4 or cam.shape[0]!=1 or cam.shape[-2:]!=(4,4) or intr.shape!=cam.shape[:2]+(4,):raise ValueError('GT render camera shape differs')
    if not torch.isfinite(cam).all() or not torch.isfinite(intr).all() or not (intr[...,:2]>0).all():raise ValueError('Invalid GT render camera')
    if not torch.allclose(cam[:,0],torch.eye(4,device=cam.device,dtype=cam.dtype)[None],rtol=0,atol=2e-5):raise ValueError('GT render camera must be first-Context canonical')
    if views is not None:cam=cam[:,:views];intr=intr[:,:views]
    return ModelInputDecoder(cam_view=cam,intrinsics=intr)


def camera_audit(batch,scene_scale,enabled):
    gt_decoder(batch)
    s=float(scene_scale.detach().reshape(-1)[0])
    if not s>0 or not torch.isfinite(scene_scale).all():raise ValueError('Invalid learned scene scale')
    return dict(protocol=PROTOCOL,GT_intrinsics=True,GT_extrinsics=True,GT_Context_pose_used_for_generation=False,
        Target_pose_used_for_generation=False,Target_RGB_used_for_generation=False,GT_depth_used_for_generation=False,
        Context_K_known=True,Context_pose_source='RGB-predicted frozen Omega',GT_pose_source='render-only provider first-Context frame',
        public_units_per_meter=UNITS_PER_METER,scene_scale_head=bool(enabled),scene_scale=s,
        scene_scale_source='learned_Context_RGB_features' if enabled else 'fixed_identity',GT_baseline_alignment=False,
        background='black',physical_near_m=.1,physical_far_m=100.,frame_ids=batch['frame_ids'][0].cpu().tolist())
