"""Locked assets and explicit synchronous FP32 GC for global batch eight."""
from __future__ import annotations
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v3_set_runtime import (
    build_batch, seed_everything, capture_rng, restore_rng, write_json, jsonable,
    PRETRAINED, PRETRAINED_SHA, load_checkpoint_state, transfer_object_locus_reconstruction_weights)
from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
REPORTS = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu')
RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_v1_8gpu')
FRESH = Path('/space/mawb/ssst/group_plus/object_locus_v3_set_fresh128')
REGISTERED = (0,2,4,8,16,32,64)
OFFICIAL = (0,8,16,32,64)
TOTAL_UPDATES, TOTAL_EXPOSURES = 8064,64512


def rank_world():
    return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)


def init_distributed():
    world = int(os.environ.get('WORLD_SIZE','1'))
    if world not in (1,8): raise RuntimeError('only single-card smoke or global8 is allowed')
    local = int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local)
    if world==8: dist.init_process_group('nccl',timeout=datetime.timedelta(hours=4))
    if not os.uname().nodename.startswith('3dimage-13'): raise RuntimeError('fixed node 3dimage-13 required')
    if torch.cuda.get_device_name(local)!='NVIDIA GeForce RTX 3090': raise RuntimeError('RTX3090 required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda',local)


def build_options():
    from tokengs.options import config_defaults
    return config_defaults['train_siu3r_object_locus_panoptic_v1']


def build_model(device='cpu', *, report=True):
    seed_everything(42)
    opt=build_options()
    model=LocusGSObjectLocusPanopticV1Recon(opt)
    source=load_checkpoint_state()
    # Reuse the exact existing 450-tensor strict transfer before adding modules.
    transfer=transfer_object_locus_reconstruction_weights(model,opt,source)
    if transfer['matched_reconstruction_tensor_count']!=450: raise RuntimeError('reconstruction count !=450')
    mapping=[dict(source=k,target=k,source_shape=list(v.shape),target_shape=list(model.state_dict()[k].shape),status='LOADED',family='reconstruction') for k,v in source.items()]
    del source
    model.initialize_understanding(mapping)
    model=model.to(device).float()
    if any(not p.requires_grad for p in model.parameters()): raise RuntimeError('unexpected frozen parameters')
    rank,world=rank_world()
    if world==8:
        for value in model.state_dict().values(): dist.broadcast(value,src=0)
    seed_everything(42+100003*rank)
    if report and rank==0:
        write_json(REPORTS/'weights_mapping.json',dict(counts=dict(reconstruction=450,encoder=292,mast3r_excluded=725,adapter=187,mask_decoder=326),mapping=mapping,
            transfer=transfer,new_parameters=[dict(name=n,shape=list(p.shape),numel=p.numel()) for n,p in model.named_parameters() if n.startswith('panoptic.')]))
        import transformers,gsplat
        write_json(REPORTS/'dependencies.json',dict(torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,gsplat=getattr(gsplat,'__version__','unknown'),numpy=np.__version__,python=sys.version,siu3r_commit='8ea80166be76854f938e90521f1a5b688b755c87',precision='FP32',tf32=False))
    return model,opt


def family(name):
    return 'pretrained' if name.startswith('understanding.') else 'new' if name.startswith('panoptic.') else 'reconstruction'


def build_optimizer(model):
    no_decay=set()
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.LayerNorm,torch.nn.modules.batchnorm._BatchNorm,torch.nn.Embedding)):
            no_decay.update(name+'.'+n for n,_ in module.named_parameters(recurse=False))
    no_decay.add('panoptic.stuff_seed')
    # SIU3R adapter/pixel-decoder level embeddings are bare Parameters.
    no_decay.update(n for n,p in model.named_parameters() if n.endswith('.level_embed') or getattr(p,'_no_weight_decay',False))
    peaks=dict(reconstruction=1e-6,pretrained=1e-5,new=1e-4)
    groups=[]
    for fam in ('reconstruction','pretrained','new'):
        for decay in (False,True):
            selected=[(n,p) for n,p in sorted(model.named_parameters()) if family(n)==fam and
                (fam!='reconstruction' and p.ndim>1 and not n.endswith('.bias') and n not in no_decay)==decay]
            if selected: groups.append(dict(name=f'{fam}_{"decay" if decay else "nodecay"}',params=[p for _,p in selected],param_names=[n for n,_ in selected],lr=peaks[fam],peak_lr=peaks[fam],weight_decay=0.05 if decay else 0.0))
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters()}: raise RuntimeError('optimizer coverage mismatch')
    if rank_world()[0]==0:
        write_json(REPORTS/'optimizer_groups.json',dict(groups=[{k:v for k,v in g.items() if k!='params'} | dict(tensor_count=len(g['params']),numel=sum(p.numel() for p in g['params'])) for g in groups],all_trainable_once=True,no_decay_embeddings=sorted(no_decay)))
    return torch.optim.AdamW(groups,betas=(0.9,0.95),eps=1e-8)


def lr_multiplier(update):
    t=8*(int(update)+1)
    if not 1<=t<=TOTAL_EXPOSURES: raise ValueError('update out of range')
    return t/200 if t<=200 else 0.1+0.9*(1+math.cos(math.pi*(t-200)/(TOTAL_EXPOSURES-200)))/2


def sample_index(epoch,k,rank):
    return int(np.random.default_rng(42+epoch).permutation(1008)[8*k+rank])


def assets():
    # SHA and job association were verified in preflight; reuse that result.
    verified=json.loads((REPORTS/'preflight_audit.json').read_text())
    if verified['weights']['mast3r']['excluded_expected']!=725: raise RuntimeError('erratum absent')
    manifest=json.loads((FRESH/'data_manifest.json').read_text())
    source_plan=json.loads((FRESH/'training_plan.json').read_text())
    windows=manifest['expanded_train_windows']
    if len(windows)!=1008 or len({w['scene'] for w in windows})!=128: raise RuntimeError('Fresh128 identity mismatch')
    original=json.loads(Path(manifest['source_v3_manifest']).read_text())
    manifest['train_all56']=manifest['original_train_all56']
    manifest['same_scene_holdout8']=original['same_scene_holdout8']
    names=['expanded_train_probe32','original_train_all56','same_scene_holdout16','dev8','val32','train_all56','same_scene_holdout8']
    # Keep every registered Fresh128 split, and expose required synonymous names.
    train_scenes={w['scene'] for w in windows}
    intersections={n:sorted(train_scenes & {w['scene'] for w in manifest[n]}) for n in names}
    leaks=[]
    for name in ('same_scene_holdout8','same_scene_holdout16'):
        for h in manifest[name]:
            hf=set(h['context']+h['novel'])
            for w in windows:
                if h['scene']==w['scene'] and hf & set(w['context']+w['novel']): leaks.append((name,h,w))
    if leaks: raise RuntimeError(f'holdout frame leak {leaks[:1]}')
    return manifest,source_plan,names,dict(scene_intersections=intersections,holdout_frame_leak=False,posed_setting=True,
        windows=1008,scenes=128,epochs=64,updates=8064,exposures=64512,exposures_per_window=64)


def synchronized_check(error, device, stage, update, batch=None, metrics=None):
    rank,world=rank_world()
    flag=torch.tensor(int(error is not None),device=device,dtype=torch.int32)
    if world>1: dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    if flag.item():
        write_json(REPORTS/f'failure_update{update}_rank{rank}_{stage}.json',dict(stage=stage,update=update,exposure=8*update,
            error=str(error) if error else 'peer rank failure',traceback=traceback.format_exc() if error else None,
            frame_ids=batch.get('frame_ids') if batch else None,
            metrics={k:v for k,v in (metrics or {}).items() if not torch.is_tensor(v) or v.ndim==0}))
        raise RuntimeError(f'synchronized {stage} failure at {update}: {error}')


def combine_gradients(params,names,rec,under):
    return [None if gr is None and gu is None else
            ((gr if gr is not None else torch.zeros_like(p)) + (0.01 if family(n)=='reconstruction' else 1.0)*(gu if gu is not None else torch.zeros_like(p)))
            for p,n,gr,gu in zip(params,names,rec,under)]


def average_gradients(params,grads,bucket_bytes=25*1024*1024):
    _,world=rank_world()
    present=torch.tensor([g is not None for g in grads],device=params[0].device,dtype=torch.uint8)
    if world>1: dist.all_reduce(present,op=dist.ReduceOp.MAX)
    bits=present.tolist()
    indices=[];size=0
    def flush(indices):
        if not indices: return
        flat=torch.cat([(grads[i] if grads[i] is not None else torch.zeros_like(params[i])).reshape(-1) for i in indices])
        if world>1: dist.all_reduce(flat,op=dist.ReduceOp.SUM); flat.div_(world)
        offset=0
        for i in indices:
            count=params[i].numel();params[i].grad=flat[offset:offset+count].view_as(params[i]).clone();offset+=count
    for i,(p,exists) in enumerate(zip(params,bits)):
        if not exists: p.grad=None;continue
        amount=p.numel()*p.element_size()
        if indices and size+amount>bucket_bytes: flush(indices);indices=[];size=0
        indices.append(i);size+=amount
    flush(indices)
    return [i for i,v in enumerate(bits) if not v]


def train_one_step(model,opt,optimizer,batch,update,*,check_rec_under=False):
    device=next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups: group['lr']=group['peak_lr']*lr_multiplier(update)
    exposure=8*update;weight=min(exposure/200,1)
    error=None;output=metrics=None
    start=time.perf_counter()
    try:
        output,metrics=model.step_loss(batch,step=exposure,understanding_weight=weight)
        if not all(torch.isfinite(metrics[k]).all() for k in ('loss_recon','loss_understanding')): raise FloatingPointError('nonfinite loss')
        if not torch.isfinite(output['prediction']['gaussians']).all(): raise FloatingPointError('nonfinite Gaussian')
    except Exception as exc: error=exc
    synchronized_check(error,device,'forward',update,batch,metrics)
    named=sorted(model.named_parameters());names=[n for n,_ in named];params=[p for _,p in named]
    error=None;grads=None;rec_under={}
    try:
        rec=torch.autograd.grad(metrics['loss_recon'],params,allow_unused=True,retain_graph=weight>0)
        if check_rec_under:
            rec_under={n:float(g.norm()) for n,g in zip(names,rec) if family(n)=='pretrained' and g is not None and g.norm()>0}
        under=torch.autograd.grad(weight*metrics['loss_understanding'],params,allow_unused=True) if weight>0 else (None,)*len(params)
        norms={fam:dict(rec=float(sum((g.detach().square().sum() for n,g in zip(names,rec) if family(n)==fam and g is not None),torch.zeros((),device=device)).sqrt()),under=float(sum((g.detach().square().sum() for n,g in zip(names,under) if family(n)==fam and g is not None),torch.zeros((),device=device)).sqrt())) for fam in ('reconstruction','pretrained','new')}
        selected_under={label:float(sum((g.detach().square().sum() for n,g in zip(names,under) if needle in n and g is not None),torch.zeros((),device=device)).sqrt()) for label,needle in [('mask_decoder','understanding.mask2former.transformer_module'),('child','panoptic.child_mlp.2'),('class','panoptic.class_head')]}
        grads=combine_gradients(params,names,rec,under)
        if any(g is not None and not torch.isfinite(g).all() for g in grads): raise FloatingPointError('nonfinite local gradient')
        del rec,under
    except Exception as exc: error=exc
    synchronized_check(error,device,'autograd',update,batch,metrics)
    sync_start=time.perf_counter();unused=average_gradients(params,grads);del grads
    torch.cuda.synchronize();sync_time=time.perf_counter()-sync_start
    error=None
    try:
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params): raise FloatingPointError('nonfinite averaged gradient')
    except Exception as exc: error=exc
    synchronized_check(error,device,'averaged_gradient',update,batch,metrics)
    combined_norms={fam:float(sum((p.grad.detach().square().sum() for n,p in named if family(n)==fam and p.grad is not None),torch.zeros((),device=device)).sqrt()) for fam in ('reconstruction','pretrained','new')}
    total_norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
    optimizer.step()
    prediction=output['prediction']
    activity={}
    for state in prediction['states']:
        if 'route' not in state: continue
        w=model.panoptic.layers[f'L{state["layer"]}'].W_inject.weight
        activity[f'L{state["layer"]}']=dict(weight_norm=float(w.detach().norm()),gradient_norm=float(w.grad.norm()) if w.grad is not None else None,
            delta_norm=float(state['joint_delta'].detach().norm()),relative_delta=float((state['joint_delta'].detach().norm(dim=-1)/(state['joint_h_norm'].detach()+1e-6)).mean()),
            route_thing=float(state['route'][...,:100].detach().sum(-1).mean()),route_stuff=float(state['route'][...,100:102].detach().sum(-1).mean()),route_void=float(state['route'][...,102].detach().mean()))
    row={k:float(v.detach()) if torch.is_tensor(v) and v.ndim==0 else v for k,v in metrics.items() if not torch.is_tensor(v) or v.ndim==0}
    row.update(update=update+1,exposure=8*(update+1),epoch=(update+1)/126,beta=prediction['beta'],activity=activity,
        combined_gradient_norms=combined_norms,selected_under_gradients=selected_under,gradient_norms=norms,preclip_norm=float(total_norm),lrs={g['name']:g['lr'] for g in optimizer.param_groups},
        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        synchronization_seconds=sync_time,seconds=time.perf_counter()-start,unused=[names[i] for i in unused],rec_to_pretrained=rec_under,
        lifting_support=float((prediction['lifting_mass']>0).float().mean()),fallback_fraction=float((1-prediction['lifting_gate']).mean()),frame_ids=batch['frame_ids'])
    return output,jsonable(row)
