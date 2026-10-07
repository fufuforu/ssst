"""Runtime for the preregistered Object-Locus GC alpha sweep."""
from __future__ import annotations
import dataclasses, datetime, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
from tokengs.models.object_locus_competition_v1 import LocusGSObjectLocusCompetitionV1Recon
from scripts.object_locus_v3_set_runtime import build_batch, capture_rng, restore_rng, write_json, jsonable
from scripts.object_locus_panoptic_v1_runtime import build_model as _registered_build_model

SOURCE_CHECKPOINT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')
SOURCE_SHA256 = '68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a'
SOURCE_UPDATES, SOURCE_EXPOSURES = 6258, 50064
SOURCE_MANIFEST = Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/source_manifest.json')
REPORT_ROOT = Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1')
RUN_ROOT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_competition_gc001_v1')
ARMS = {'comp_gc001': 0.01}
COMPETITION_LAMBDA = 2.0

def build_comp_model(device='cpu'):
    # The local wrapper performs the strict Full1201 load and returns its
    # provenance; the low-level registered constructor returns only
    # (model, options).
    model, opt, source = build_model(device, report=False)
    if not isinstance(model, LocusGSObjectLocusPanopticV1Recon):
        raise TypeError('wrong registered Panoptic V1 class')
    before = tuple(model.state_dict().keys())
    model.__class__ = LocusGSObjectLocusCompetitionV1Recon
    model.competition_lambda = COMPETITION_LAMBDA
    if tuple(model.state_dict().keys()) != before:
        raise RuntimeError('competition class changed state_dict')
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError('competition model unexpectedly freezes a parameter')
    return model, opt, source
EPOCHS, WINDOWS, WORLD, UPDATES_PER_EPOCH = 8, 1008, 8, 126
TOTAL_UPDATES, TOTAL_EXPOSURES = 1008, 8064
SPLITS = ('expanded_train_probe32','train_all56','same_scene_holdout8','dev8','val32')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def rank_world(): return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def init_distributed():
    world=int(os.environ.get('WORLD_SIZE','1')); local=int(os.environ.get('LOCAL_RANK','0'))
    if world not in (1,8): raise RuntimeError(f'world size must be 1 or 8, got {world}')
    if not os.uname().nodename.startswith('3dimage-11'): raise RuntimeError('fixed node 3dimage-11 required')
    torch.cuda.set_device(local)
    if torch.cuda.get_device_name(local)!='NVIDIA GeForce RTX 3090': raise RuntimeError('RTX3090 required')
    if world==8: dist.init_process_group('nccl',timeout=datetime.timedelta(hours=4))
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda',local)

def load_source_blob():
    if not SOURCE_CHECKPOINT.is_file(): raise FileNotFoundError(SOURCE_CHECKPOINT)
    actual=sha256(SOURCE_CHECKPOINT)
    if actual!=SOURCE_SHA256: raise RuntimeError(f'source checkpoint SHA mismatch: {actual}')
    blob=torch.load(SOURCE_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    expected={'epoch':6,'completed_updates':6258,'completed_exposures':50064}
    got={k:blob.get(k) for k in expected}
    if got!=expected: raise RuntimeError(f'source metadata mismatch: {got}')
    if not isinstance(blob.get('model'),dict): raise RuntimeError('source checkpoint missing complete model state')
    return blob

def build_model(device='cpu', report=False):
    seed_everything(42)
    blob=load_source_blob()
    config=blob.get('config')
    if not isinstance(config,dict) or config.get('model_type')!='siu3r_object_locus_panoptic_v1':
        raise RuntimeError('source model configuration mismatch')
    # The registered V1 construction entry creates the complete reconstruction,
    # understanding and object modules. Its temporary transferred values are
    # replaced immediately by the strict complete checkpoint load below.
    from scripts.object_locus_panoptic_v1_runtime import build_model as build_panoptic_model
    model,opt=build_panoptic_model(device,report=False)
    if not isinstance(model,LocusGSObjectLocusPanopticV1Recon): raise TypeError('wrong registered model class')
    if dataclasses.asdict(opt)!=config: raise RuntimeError('source checkpoint config differs from original V1 model construction config')
    result=model.load_state_dict(blob['model'],strict=True)
    if result.missing_keys or result.unexpected_keys: raise RuntimeError('strict source model load failed')
    model=model.to(device=device,dtype=torch.float32)
    if not all(p.requires_grad for p in model.parameters()): raise RuntimeError('source contains frozen parameters')
    rank,world=rank_world()
    if world==8:
        for value in model.state_dict().values(): dist.broadcast(value,src=0)
    seed_everything(42+100003*rank)
    return model,opt,{'path':str(SOURCE_CHECKPOINT),'sha256':SOURCE_SHA256,**{k:blob[k] for k in ('epoch','completed_updates','completed_exposures')},'strict_load':True,'state_tensors':len(blob['model'])}

def family(name): return 'pretrained' if name.startswith('understanding.') else 'new' if name.startswith('panoptic.') else 'reconstruction'

def build_optimizer(model):
    no_decay=set()
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.LayerNorm,torch.nn.modules.batchnorm._BatchNorm,torch.nn.Embedding)):
            no_decay.update(name+'.'+n for n,_ in module.named_parameters(recurse=False))
    no_decay.add('panoptic.stuff_seed')
    no_decay.update(n for n,p in model.named_parameters() if n.endswith('.level_embed') or getattr(p,'_no_weight_decay',False))
    peaks={'reconstruction':1e-6,'pretrained':1e-5,'new':1e-4}; groups=[]
    named=sorted(model.named_parameters())
    for fam in ('reconstruction','pretrained','new'):
        for decay in (False,True):
            selected=[(n,p) for n,p in named if family(n)==fam and ((fam!='reconstruction' and p.ndim>1 and not n.endswith('.bias') and n not in no_decay)==decay)]
            if selected: groups.append(dict(name=f'{fam}_{"decay" if decay else "nodecay"}',params=[p for _,p in selected],param_names=[n for n,_ in selected],peak_lr=peaks[fam],lr=peaks[fam],weight_decay=0.05 if decay else 0.0))
    flat=[id(p) for g in groups for p in g['params']]
    trainable={id(p) for p in model.parameters() if p.requires_grad}
    if len(flat)!=len(set(flat)) or set(flat)!=trainable: raise RuntimeError('optimizer duplicate/omission')
    opt=torch.optim.AdamW(groups,betas=(0.9,0.95),eps=1e-8)
    return opt

def lr_multiplier(update):
    t=int(update)+1
    if not 1<=t<=TOTAL_UPDATES: raise ValueError('update out of range')
    if t<=25: return t/25
    return 0.1+0.9*(1+math.cos(math.pi*(t-25)/983))/2

def combine_gradients(params,names,rec,under,alpha):
    out=[]
    for p,n,gr,gu in zip(params,names,rec,under):
        if gr is None and gu is None: out.append(None); continue
        grec=gr if gr is not None else torch.zeros_like(p)
        gunder=gu if gu is not None else torch.zeros_like(p)
        out.append(grec+(alpha if family(n)=='reconstruction' else 1.0)*gunder)
    return out

def average_gradients_into_params(params,grads):
    _,world=rank_world()
    flags=torch.tensor([g is not None for g in grads],device=params[0].device,dtype=torch.uint8)
    if world>1: dist.all_reduce(flags,op=dist.ReduceOp.MAX)
    bits=flags.tolist(); ids=[]; amount=0; limit=25*1024*1024
    def flush(indices):
        if not indices:return
        flat=torch.cat([(grads[i] if grads[i] is not None else torch.zeros_like(params[i])).reshape(-1) for i in indices])
        if world>1: dist.all_reduce(flat,op=dist.ReduceOp.SUM);flat.div_(world)
        offset=0
        for i in indices:
            n=params[i].numel();params[i].grad=flat[offset:offset+n].view_as(params[i]).clone();offset+=n
            grads[i]=None
    for i,(p,present) in enumerate(zip(params,bits)):
        if not present:params[i].grad=None;continue
        size=p.numel()*p.element_size()
        if ids and amount+size>limit:flush(ids);ids=[];amount=0
        ids.append(i);amount+=size
    flush(ids)
    return None

def average_gradient_norm(params,grads,names,*,selected_family='reconstruction'):
    """Streaming norm of the rank-mean component gradient, without copies."""
    _,world=rank_world();indices=[i for i,n in enumerate(names) if family(n)==selected_family]
    flags=torch.tensor([grads[i] is not None for i in indices],device=params[0].device,dtype=torch.uint8)
    if world>1:dist.all_reduce(flags,op=dist.ReduceOp.MAX)
    total=torch.zeros((),device=params[0].device,dtype=torch.float64);bucket=[];amount=0;limit=25*1024*1024
    def flush(items):
        if not items:return
        flat=torch.cat([(grads[i] if grads[i] is not None else torch.zeros_like(params[i])).reshape(-1) for i in items])
        if world>1:dist.all_reduce(flat,op=dist.ReduceOp.SUM);flat.div_(world)
        total.add_(flat.double().square().sum());del flat
    for i,present in zip(indices,flags.tolist()):
        if not present:continue
        size=params[i].numel()*params[i].element_size()
        if bucket and amount+size>limit:flush(bucket);bucket=[];amount=0
        bucket.append(i);amount+=size
    flush(bucket)
    return float(total.sqrt())

def synchronized_error(error,device,stage,update,metrics=None):
    rank,world=rank_world();flag=torch.tensor(int(error is not None),device=device,dtype=torch.int32)
    if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    if flag.item():
        if rank==0:
            REPORT_ROOT.mkdir(parents=True,exist_ok=True)
            write_json(REPORT_ROOT/f'failure_{os.environ.get("TASK_ARM","smoke")}_update{update}_{stage}.json',{'stage':stage,'update':update,'error':str(error) if error else 'peer rank failure','traceback':traceback.format_exc() if error else None,'metrics':{k:float(v.detach()) for k,v in (metrics or {}).items() if torch.is_tensor(v) and v.ndim==0}})
        raise RuntimeError(f'synchronized {stage} error at update {update}: {error}')

def train_step(model,opt,optimizer,batch,update,alpha,*,source_exposure=SOURCE_EXPOSURES,diagnose=False):
    device=next(model.parameters()).device;rank,world=rank_world();optimizer.zero_grad(set_to_none=True)
    mult=lr_multiplier(update)
    for g in optimizer.param_groups:g['lr']=g['peak_lr']*mult
    local_exposure=8*update;model_exposure=source_exposure+local_exposure;uw=min(local_exposure/200,1.0)
    output=metrics=None;error=None
    try:
        output,metrics=model.step_loss(batch,step=model_exposure,understanding_weight=uw)
        for k in ('loss_recon','loss_understanding'):
            if k not in metrics or not torch.isfinite(metrics[k]).all():raise FloatingPointError('nonfinite '+k)
        for key in ('gaussians','RGB','gaussian_membership','membership_mass','region_mass','semantic_scores','alpha','p_class'):
            value=output['prediction'].get(key)
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise FloatingPointError('nonfinite '+key)
        for state in output['prediction'].get('states',[]):
            for key in ('q','c','s','route'):
                value=state.get(key)
                if torch.is_tensor(value) and not torch.isfinite(value).all():raise FloatingPointError('nonfinite state '+key)
    except Exception as e:error=e
    synchronized_error(error,device,'forward',update,metrics)
    named=sorted(model.named_parameters());names=[n for n,_ in named];params=[p for _,p in named]
    error=None
    try:
        rec=torch.autograd.grad(metrics['loss_recon'],params,allow_unused=True,retain_graph=uw>0)
        under=torch.autograd.grad(uw*metrics['loss_understanding'],params,allow_unused=True) if uw>0 else (None,)*len(params)
        if any(g is not None and not torch.isfinite(g).all() for g in (*rec,*under)):raise FloatingPointError('nonfinite component gradient')
        if diagnose:
            nrec=average_gradient_norm(params,rec,names);nunder=average_gradient_norm(params,under,names)
            diag={'g_rec':nrec,'g_under':nunder,'alpha_g_under':alpha*nunder}
        else:diag=None
        grads=combine_gradients(params,names,rec,under,alpha)
        if any(g is not None and not torch.isfinite(g).all() for g in grads):raise FloatingPointError('nonfinite combined gradient')
        del rec,under
    except Exception as e:error=e;rec=under=grads=None
    synchronized_error(error,device,'autograd',update,metrics)
    average_gradients_into_params(params,grads)
    del grads
    error=None
    try:
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('nonfinite averaged gradient')
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('nonfinite parameter after optimizer step')
    except Exception as e:error=e;norm=float('nan')
    synchronized_error(error,device,'optimizer_step',update,metrics)
    row={'local_update':update+1,'local_epoch':(update+1)/UPDATES_PER_EPOCH,'new_exposures':8*(update+1),'model_exposure':model_exposure,'endpoint_model_exposure':source_exposure+8*(update+1),'understanding_weight':uw,'loss_recon':float(metrics['loss_recon'].detach()),'loss_understanding':float(metrics['loss_understanding'].detach()),'loss_under_old':float(metrics.get('loss_under_old',metrics['loss_understanding']).detach()),'loss_competition':float(metrics.get('loss_competition',metrics['loss_understanding']*0.0).detach()),'weighted_loss_competition':float(metrics.get('weighted_loss_competition',metrics['loss_understanding']*0.0).detach()),'loss_under_new':float(metrics['loss_understanding'].detach()),'competition_lambda':COMPETITION_LAMBDA,'monitor_total_loss':float(metrics['loss_recon'].detach()+uw*metrics['loss_understanding'].detach()),'monitor_total_excludes_parameter_group_specific_gc':True,'alpha':alpha,'group_lr':{g['name']:g['lr'] for g in optimizer.param_groups},'preclip_global_grad_norm':norm,'clipped':norm>1.0,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'diagnostic':diag}
    return output,jsonable(row)

def state_equal(a,b):
    sa,sb=a.state_dict(),b.state_dict()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k].detach().cpu(),sb[k].detach().cpu()) for k in sa)
