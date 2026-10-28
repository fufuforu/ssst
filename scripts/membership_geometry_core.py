"""Geometry-only, fixed teacher Membership pilot. No new model parameters."""
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F

SOURCE_SHA='694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc'
POLICY='fixed_baseline4_membership_geometry_pilot_v1'
PREFIXES=('core.anchor_decoder.refine_mu.','core.anchor_decoder.refine_rho.')
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
G0=BASE/'v4init_isolated_noca_v4recipe_20261022_v1/eval/evaluation'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def tensor_sha(x):return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def freeze(model):
    names=[]
    for n,p in model.named_parameters():
        p.requires_grad_(n.startswith(PREFIXES));p.grad=None
        if p.requires_grad:names.append(n)
    if len(names)!=48:raise ValueError(f'Expected 48 geometry tensors, got {len(names)}')
    model.eval()
    # Enable its existing non-reentrant activation checkpointing. Decoder blocks
    # have no stochastic dropout; other frozen modules retain evaluation mode.
    model.core.anchor_decoder.train()
    return names

def validate_source(saved):
    from scripts.v4init_isolated_recipe import ARCH,recipe_identity
    m=saved['meta']
    if saved['progress']!=dict(epoch=4,update_in_epoch=0,full_updates=4172):raise ValueError('Not completed Baseline4')
    if m['architecture']!=ARCH or any(m.get(k)!=v for k,v in recipe_identity().items()):raise ValueError('Source recipe differs')
    if len(saved['rank_rng'])!=8 or len(saved['optimizer']['state'])!=604:raise ValueError('Source Adam/RNG incomplete')
    if {float(s['step']) for s in saved['optimizer']['state'].values()}!={4172.}:raise ValueError('Source Adam clock differs')

@torch.no_grad()
def teacher_record(pred,batch,config):
    from tokengs.models.unified_query_loss import unified_instance_loss
    unified_instance_loss(pred,batch,config)
    qi,ki=pred['final_pairs'][0];targets=pred['final_targets'];n=pred['num_thing_queries']
    m=pred['gaussian_membership'].detach().float()
    if m.ndim!=3 or m.shape[-1]!=n+2:raise ValueError('Membership slot schema differs')
    represented=ki.unique(sorted=True)
    features=[m[...,n],m[...,n+1]]
    for k in represented:features.append(m[...,qi[ki==k]].amax(-1))
    unmatched=torch.ones(n,dtype=torch.bool,device=m.device);unmatched[qi]=False
    features.append(m[...,:n][...,unmatched].amax(-1) if unmatched.any() else torch.zeros_like(m[...,0]))
    features=torch.stack(features,-1).clamp_min(1e-8);features=features/features.sum(-1,keepdim=True)
    sem=batch['semantic_label_all'][0,:2];valid=targets['valid_pixels'][0]
    y=torch.full_like(sem,-1,dtype=torch.long)
    for c in (0,1):y[valid&(sem==c)]=c
    for j,k in enumerate(represented):y[targets['gt_pixel_masks'][0][k]]=2+j
    if not (y>=0).any():raise ValueError('No supported Context pixels')
    return dict(features=features.half().cpu(),target=y.cpu(),
        cam_view=torch.linalg.inv(pred['predicted_context_c2w']).transpose(-1,-2).cpu(),
        intrinsics=pred['predicted_context_intrinsics'].cpu(),anchor_point_indices=pred['anchor_point_indices'].cpu(),
        source_xyz_scale=pred['gaussians'][...,[0,1,2,4,5,6]].float().cpu(),
        rgb_sha256=tensor_sha(batch['images_input']),matched_queries=len(qi),matched_instances=len(represented),
        gt_instances=len(targets['gt_instance_ids'][0]))

def probability(rendered,alpha):
    # Fixed categorical features are premultiplied by coverage. Empty space is
    # an explicit unknown class, so removing geometry cannot erase the loss.
    residual=torch.zeros_like(rendered);residual[:,:,-1:]=1-alpha
    p=(rendered+residual).clamp_min(1e-8)
    return p/p.sum(2,keepdim=True).clamp_min(1e-8)

def mask_objective(p,y):
    # Equal CE/Dice weights; average Dice only over represented positive classes.
    flat=p.permute(0,1,3,4,2).reshape(-1,p.shape[2]);target=y.reshape(-1);valid=target>=0
    if not valid.any():raise ValueError('Empty geometry supervision')
    flat=flat[valid];target=target[valid]
    ce=F.nll_loss(flat.log(),target)
    dice=[]
    for c in target.unique():
        gt=(target==c).float();pr=flat[:,c]
        dice.append(1-(2*(gt*pr).sum()+1)/(gt.sum()+pr.sum()+1))
    d=torch.stack(dice).mean()
    return .5*(ce+d),ce,d

def same_slots(left,right):
    # DDP may migrate every tensor in the nested teacher record to CUDA.
    # Refuse nonintegral indices before normalizing device/dtype.
    integral=(torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8)
    if left.dtype not in integral or right.dtype not in integral:
        raise ValueError('Noninteger slot indices')
    return left.shape==right.shape and torch.equal(left.detach().to(device='cpu',dtype=torch.int64),right.detach().to(device='cpu',dtype=torch.int64))

def geometry_mask_loss(model,pred,record):
    g=pred['gaussians'];device=g.device
    if not same_slots(pred['anchor_point_indices'],record['anchor_point_indices']):raise ValueError('Teacher/student slot alignment differs')
    # New branch gradients enter xyz/scale only, not opacity/rotation/colour.
    geometry=torch.cat((g[...,:3],g[...,3:4].detach(),g[...,4:7],g[...,7:].detach()),-1)
    features=record['features'].to(device=device,dtype=torch.float32).detach()
    features=features/features.sum(-1,keepdim=True)
    rendered=model.gs.render_feature_channels(geometry,features,record['cam_view'].to(device),record['intrinsics'].to(device))
    p=probability(rendered['images_pred'],rendered['alphas_pred'])
    return mask_objective(p,record['target'].to(device))

def reconstruction(model,pred,batch):
    from tokengs.models.unified_query_v1 import ModelInputDecoder,pixel_rays,_full_supervision
    calibration=model.calibrate_targets(batch['images_all'],pred)
    decoder=ModelInputDecoder(cam_view=torch.linalg.inv(calibration['c2w']).transpose(-1,-2),intrinsics=calibration['intrinsics'])
    rays_o,rays_d=pixel_rays(calibration['c2w'],calibration['intrinsics_matrix'],256,256)
    loss,metrics,*_=model._layer_objective(pred['states'],decoder,_full_supervision(dict(batch,rays_os=rays_o,rays_ds=rays_d)))
    return loss

def coefficient(arm,step):
    if arm not in ('G1','G2'):raise ValueError(arm)
    return .01*min(1.,(step+1)/100) if arm=='G2' else 0.


def verify_teacher_renderer(configuration):
    import sys
    module=sys.modules.get('gsplat_cuda')
    path=None if module is None else module.__file__
    expected=configuration['teacher_renderer']
    if path is None or Path(path).resolve()!=Path(expected['path']).resolve() or sha(path)!=expected['sha256']:
        raise ValueError('Loaded renderer differs from reused teacher renderer')
    return dict(path=path,sha256=sha(path))
