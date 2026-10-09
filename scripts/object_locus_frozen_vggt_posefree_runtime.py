"""Review-locked configuration, migration, optimizer, and CPU plan helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path
import random
import traceback

import numpy as np
import torch
from torch import nn
from tokengs.models.frozen_vggt_posefree import FrozenVGGT

REPO=Path(__file__).resolve().parents[1]
MANIFEST=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')
CHECKPOINT=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')
EXPECTED_CHECKPOINT_SHA='68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a'
EXPECTED_VGGT_COMMIT='a288dd0f14786c93483e45524328726ab7b1b4ce'
EXPECTED_SCENES=1191
EXPECTED_WINDOWS=8337
WORLD_SIZE=4
MICRO_BATCH=1
ACCUMULATION=2
EPOCHS=8
GLOBAL_BATCH=8
WINDOWS_PER_EPOCH=8344
UPDATES_PER_EPOCH=1043
TOTAL_UPDATES=8344
TOTAL_EXPOSURES=66752
VGGT_ARTIFACT_MANIFEST=REPO/'vggt_artifact_manifest.json'

REMOVED_SOURCE_PREFIXES=(
    'patch_embed.', 'patch_plucker_embed.', 'enc_dec_backbone.encoder.',
    'enc_dec_backbone.encoder_norm.', 'enc_dec_backbone.multiscale_norms.',
    'enc_dec_backbone.kv_proj.', 'enc_dec_backbone.k_proj_norm.',
    'enc_dec_backbone.latents', 'enc_dec_backbone.latent_feature_',
    'enc_dec_backbone.latent_blocks.', 'enc_dec_backbone.latent_kv_proj.',
)
RETAINED_PREFIXES=('gs_tokens','gs_tokens_dynamic','anchor_decoder.',
    'enc_dec_backbone.decoder_blocks.','activation_head.','understanding.','panoptic.')


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_epoch6_cpu(path: Path=CHECKPOINT):
    if not path.is_file(): raise FileNotFoundError(f"required epoch-06 migration source missing: {path}")
    digest=sha256(path)
    if digest!=EXPECTED_CHECKPOINT_SHA: raise RuntimeError(f"epoch-06 checkpoint SHA mismatch: {digest}")
    blob=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    meta={k:blob.get(k) for k in ('epoch','completed_updates','completed_exposures')}
    # Historical full1201 checkpoint metadata uses completed-count convention.
    if meta['completed_updates']!=6258 or meta['completed_exposures']!=50064:
        raise RuntimeError(f"epoch-06 progress metadata mismatch: {meta}")
    state=blob.get('model')
    if not isinstance(state,dict): raise TypeError("epoch-06 checkpoint has no model state dict")
    return blob,state,{'path':str(path),'sha256':digest,'epoch':6,
                       'source_completed_updates':6258,'source_completed_exposures':50064,
                       'new_training_completed_updates':0,'new_training_completed_exposures':0}


def migrate_model_state(model: nn.Module, source: dict[str,torch.Tensor]):
    """Copy an explicit retained whitelist by key and shape; never mask with strict=False."""
    target=model.state_dict()
    required={k for k in target if k.startswith(RETAINED_PREFIXES)}
    absent_targets=sorted(required-set(source))
    extra_retained=sorted(k for k in source if k.startswith(RETAINED_PREFIXES) and k not in required)
    mismatched=sorted(k for k in required&set(source) if target[k].shape!=source[k].shape)
    if absent_targets or extra_retained or mismatched:
        raise RuntimeError(f"migration whitelist mismatch: missing={absent_targets[:30]}, extra={extra_retained[:30]}, shape={mismatched[:30]}")
    # Every transferred tensor is copied explicitly. No generic partial-load API.
    with torch.no_grad():
        for key in sorted(required):
            target[key].copy_(source[key])
    source_loaded=sorted(required)
    new_keys=sorted(k for k in target if k.startswith(('vggt_memory_adapter.',)))
    frozen_keys=sorted(k for k in target if k.startswith('frozen_vggt.'))
    excluded=[]; unknown=[]
    for key in sorted(set(source)-required):
        if key.startswith(REMOVED_SOURCE_PREFIXES): excluded.append(key)
        elif key in ('optimizer','scheduler') or key.startswith(('optimizer.','scheduler.')): excluded.append(key)
        else: unknown.append(key)
    if unknown:
        raise RuntimeError(f"source checkpoint has unclassified keys outside explicit migration map: {unknown[:30]}")
    # Verify each retained tensor after copy without loading unrelated source state.
    bad_after=sorted(k for k in required if not torch.equal(target[k].cpu(),source[k].cpu()))
    if bad_after: raise RuntimeError(f"migration copy verification failed: {bad_after[:20]}")
    return {'loaded':source_loaded,'explicitly_excluded':excluded,
            'newly_initialized':new_keys,'external_frozen_vggt_keys':frozen_keys,
            'key_map':[{'source':k,'target':k,'source_shape':list(source[k].shape),
                        'target_shape':list(target[k].shape),'status':'LOADED'} for k in source_loaded],
            'excluded_key_map':[{'source':k,'target':None,'source_shape':list(source[k].shape),
                                 'target_shape':None,'status':'EXPLICITLY_EXCLUDED',
                                 'reason':'legacy reconstruction encoder/embedding/KV removed'} for k in excluded],
            'new_parameter_shapes':[{'target':k,'shape':list(target[k].shape),'status':'NEWLY_INITIALIZED'} for k in new_keys],
            'missing':[],'shape_mismatch':[],'unknown_source_keys':[],
            'counts':{'loaded':len(source_loaded),'explicitly_excluded':len(excluded),
                      'newly_initialized':len(new_keys),'external_frozen_vggt_keys':len(frozen_keys)}}


def load_manifest(path: Path=MANIFEST):
    manifest=json.loads(Path(path).read_text())
    scenes=manifest.get('train_scenes') or manifest.get('scenes')
    windows=manifest.get('expanded_train_windows') or manifest.get('train_windows')
    if scenes is None or windows is None:
        # Existing full1201 manifest stores named train split fields; preserve its
        # entries and derive only their counts without resampling.
        windows=manifest.get('train',manifest.get('windows'))
        scenes=sorted({w.get('scene') for w in (windows or []) if isinstance(w,dict)})
    elif isinstance(scenes,int):
        scenes=manifest.get('actual_train_scenes',[])
    if len(scenes or [])!=EXPECTED_SCENES or len(windows or [])!=EXPECTED_WINDOWS:
        raise ValueError(f"locked full1201 manifest count mismatch: scenes={len(scenes or [])}, windows={len(windows or [])}")
    return manifest,scenes,windows


def epoch_order(epoch: int, n: int=EXPECTED_WINDOWS) -> np.ndarray:
    order=np.random.default_rng(42+int(epoch)).permutation(n)
    padded=math.ceil(n/GLOBAL_BATCH)*GLOBAL_BATCH
    if padded>n: order=np.concatenate((order,order[:padded-n]))
    if len(order)!=WINDOWS_PER_EPOCH: raise AssertionError("epoch padding contract")
    return order


def rank_microbatch_indices(epoch: int, update: int, rank: int) -> tuple[int,int]:
    if not 0<=rank<WORLD_SIZE or not 0<=update<UPDATES_PER_EPOCH: raise ValueError("rank/update out of range")
    order=epoch_order(epoch)
    start=update*GLOBAL_BATCH+rank*ACCUMULATION
    return int(order[start]),int(order[start+1])


def exposure_schedule(exposure: int):
    exposure=max(int(exposure),0)
    return min(exposure/200,1.0),0.1*min(exposure/1000,1.0)


def lr_multiplier(update: int):
    step=int(update)+1
    if not 1<=step<=TOTAL_UPDATES: raise ValueError("optimizer update outside planned range")
    return step/200 if step<=200 else .1+.9*(1+math.cos(math.pi*(step-200)/(TOTAL_UPDATES-200)))/2


def parameter_family(name: str):
    if name.startswith('frozen_vggt.'): return 'frozen_vggt'
    if name.startswith('understanding.'): return 'understanding'
    if name.startswith(('panoptic.','vggt_memory_adapter.')): return 'object_memory'
    return 'reconstruction'


def build_optimizer(model: nn.Module):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(parameter_family(n)=='frozen_vggt' for n,_ in named): raise RuntimeError("frozen VGGT must not enter optimizer")
    no_decay=set()
    for module_name,module in model.named_modules():
        if isinstance(module,(nn.LayerNorm,nn.modules.batchnorm._BatchNorm,nn.Embedding)):
            no_decay.update(f'{module_name}.{name}' if module_name else name for name,_ in module.named_parameters(recurse=False))
    no_decay.update(n for n,p in named if n.endswith('.level_embed') or getattr(p,'_no_weight_decay',False))
    no_decay.add('panoptic.stuff_seed')
    peak={'reconstruction':1e-5,'understanding':1e-5,'object_memory':1e-4}
    groups=[]
    for family in ('reconstruction','understanding','object_memory'):
        for decay in (False,True):
            selected=[(n,p) for n,p in sorted(named) if parameter_family(n)==family and
                      (family!='reconstruction' and p.ndim>1 and not n.endswith('.bias') and n not in no_decay)==decay]
            if selected: groups.append({'name':f'{family}_{"decay" if decay else "nodecay"}',
                'params':[p for _,p in selected],'param_names':[n for n,_ in selected],
                'peak_lr':peak[family],'lr':peak[family],
                'weight_decay':.05 if decay else 0.})
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for _,p in named}:
        raise RuntimeError("optimizer groups do not cover each trainable tensor exactly once")
    return torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8)


def gc_combine(parameter_names, reconstruction_gradients, understanding_gradients):
    combined=[]
    for name,gr,gu in zip(parameter_names,reconstruction_gradients,understanding_gradients):
        if gr is None and gu is None: combined.append(None); continue
        base=torch.zeros_like(gu if gr is None else gr)
        if gr is not None: base=base+gr
        if gu is not None: base=base+(.01 if parameter_family(name)=='reconstruction' else 1.)*gu
        combined.append(base)
    return combined


def install_gradients(parameters, gradients, *, world_size=None):
    """Average per-rank gradients when distributed is initialized (CPU testable)."""
    world_size=world_size or 1
    distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        for p,g in zip(parameters,gradients): p.grad=None if g is None else g.clone()
        return
    present=torch.tensor([g is not None for g in gradients],device=parameters[0].device,dtype=torch.uint8)
    torch.distributed.all_reduce(present,op=torch.distributed.ReduceOp.MAX)
    active=present.tolist()
    buckets=[];indices=[];numel=0;bucket_limit=25*1024*1024
    def flush(which):
        if not which: return
        values=[gradients[i] if gradients[i] is not None else torch.zeros_like(parameters[i]) for i in which]
        flat=torch.cat([value.reshape(-1) for value in values])
        torch.distributed.all_reduce(flat,op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        offset=0
        for i in which:
            count=parameters[i].numel()
            parameters[i].grad=flat[offset:offset+count].view_as(parameters[i]).clone()
            offset+=count
    for i,(parameter,is_present) in enumerate(zip(parameters,active)):
        if not is_present:
            parameter.grad=None
            continue
        amount=parameter.numel()*parameter.element_size()
        if indices and numel+amount>bucket_limit:
            flush(indices);indices=[];numel=0
        indices.append(i);numel+=amount
    flush(indices)


def synchronized_error(error,device,stage,update):
    """Make local failures visible to every rank before the next collective."""
    distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
    rank=torch.distributed.get_rank() if distributed else 0
    flag=torch.tensor(int(error is not None),device=device,dtype=torch.int32)
    if distributed: torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
    if flag.item():
        detail=f"{type(error).__name__}: {error}" if error is not None else "peer rank failed"
        if error is not None: detail += "\n"+traceback.format_exc()
        raise RuntimeError(f"synchronized {stage} error at update {update} on rank {rank}: {detail}") from error


def train_microbatch_window(model,optimizer,batches,update,*,base_exposure,world_size=None):
    """Two-microbatch GC step with one accumulated gradient family per parameter."""
    if len(batches)!=ACCUMULATION: raise ValueError("each rank must receive exactly two accumulation microbatches")
    named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
    names=[n for n,_ in named];params=[p for _,p in named]
    optimizer.zero_grad(set_to_none=True)
    combined_acc=[None]*len(params)
    weight,beta=exposure_schedule(base_exposure)
    device=next(model.parameters()).device
    local_rows=[]
    for micro_index,batch in enumerate(batches):
        output=metrics=None;error=None
        try:
            output,metrics=model.step_loss(batch,step=base_exposure,understanding_weight=weight)
            if not all(torch.isfinite(metrics[key]).all() for key in ('loss_recon','loss_understanding')):
                raise FloatingPointError('nonfinite reconstruction/understanding loss')
        except Exception as exc:error=exc
        synchronized_error(error,device,'forward',int(update))
        error=None
        try:
            rec=torch.autograd.grad(metrics['loss_recon']/ACCUMULATION,params,
                retain_graph=weight>0,allow_unused=True)
            under=(torch.autograd.grad(metrics['loss_understanding']*weight/ACCUMULATION,params,
                allow_unused=True) if weight>0 else (None,)*len(params))
            mixed=gc_combine(names,rec,under)
            if any(g is not None and not torch.isfinite(g).all() for g in mixed):
                raise FloatingPointError('nonfinite per-microbatch GC-combined gradient')
            combined_acc=[None if g is None and old is None else
                (torch.zeros_like(p) if old is None else old)+(torch.zeros_like(p) if g is None else g)
                for p,old,g in zip(params,combined_acc,mixed)]
            local_rows.append({'loss_recon':float(metrics['loss_recon'].detach()),
                'loss_understanding':float(metrics['loss_understanding'].detach())})
            del rec,under,mixed
        except Exception as exc:
            error=exc
        try: synchronized_error(error,device,'autograd',int(update))
        finally:
            if output is not None: del output
            if metrics is not None: del metrics
            del batch
    grads=combined_acc
    error=None
    try:
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('nonfinite accumulated GC gradient')
    except Exception as exc:error=exc
    synchronized_error(error,device,'local_finite',int(update))
    install_gradients(params,grads,world_size=world_size or WORLD_SIZE)
    del grads,combined_acc
    error=None
    try:
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        mult=lr_multiplier(update)
        for group in optimizer.param_groups: group['lr']=group['peak_lr']*mult
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params): raise FloatingPointError('nonfinite parameter after optimizer.step')
    except Exception as exc:error=exc;norm=torch.tensor(float('nan'))
    synchronized_error(error,device,'optimizer',int(update))
    row={'completed_updates':int(update)+1,'completed_exposures':(int(update)+1)*(GLOBAL_BATCH if (world_size or WORLD_SIZE)==4 else ACCUMULATION),
         'understanding_weight':weight,'beta':beta,'lr_multiplier':mult,'preclip_norm':float(norm),
         'loss_recon':sum(x['loss_recon'] for x in local_rows)/len(local_rows),
         'loss_understanding':sum(x['loss_understanding'] for x in local_rows)/len(local_rows),
         'group_lr':{g['name']:g['lr'] for g in optimizer.param_groups}}
    if device.type=='cuda':
        row.update(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
                   peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device))
    return row


def plan_record():
    return {'manifest':str(MANIFEST),'manifest_sha256':sha256(MANIFEST) if MANIFEST.exists() else None,
            'expected_scenes':EXPECTED_SCENES,'expected_windows':EXPECTED_WINDOWS,'context_views_per_window':2,
            'world_size':4,'gpu_model':'RTX4090','microbatch_per_rank':1,'accumulation':2,
            'global_batch':8,'epochs':8,'windows_per_epoch_padded':WINDOWS_PER_EPOCH,
            'updates_per_epoch':UPDATES_PER_EPOCH,'total_updates':TOTAL_UPDATES,
            'total_exposures':TOTAL_EXPOSURES,'sampler':'default_rng(42+epoch); permutation padded by prefix',
            'new_training_clock_origin':0}


def seed_everything(seed: int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(*, checkpoint: Path=CHECKPOINT, opt=None,
                artifact_manifest: Path=VGGT_ARTIFACT_MANIFEST):
    """Construct, initialize the retained panoptic modules, then strict-copy epoch 06."""
    from tokengs.models.object_locus_frozen_vggt_posefree import LocusGSObjectLocusFrozenVGGT
    revision=os.environ.get('VGGT_HF_REVISION')
    if not revision:
        raise RuntimeError('set VGGT_HF_REVISION to the reviewed Hugging Face commit before model construction')
    vggt=FrozenVGGT.from_pretrained(local_files_only=True,revision=revision,
                                    artifact_manifest=artifact_manifest)
    if opt is None:
        from tokengs.options import config_defaults
        opt=config_defaults['train_siu3r_object_locus_frozen_vggt_posefree_v1']
    model=LocusGSObjectLocusFrozenVGGT(opt,vggt=vggt)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source_blob,source_state,provenance=load_epoch6_cpu(checkpoint)
    mapping=[]
    model.initialize_understanding(mapping)
    transfer=migrate_model_state(model,source_state)
    provenance.update({'source_epoch':source_blob['epoch'],'source_git_sha':source_blob.get('git_sha'),
                       'source_science_sha':source_blob.get('science_sha'),
                       'source_optimizer_restored':False,'source_scheduler_restored':False,
                       'source_rng_restored':False,'migration_counts':transfer['counts']})
    model.initialization_provenance=provenance
    model.float()  # FrozenVGGT._apply restores BF16 only for the aggregator.
    model.frozen_vggt.eval()
    for name,parameter in model.named_parameters():
        if name.startswith('frozen_vggt.') and parameter.requires_grad:
            raise RuntimeError(f'VGGT parameter trainable: {name}')
    return model,opt,transfer,provenance


def capture_rank_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
        'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rank_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state.get('torch_cuda'):torch.cuda.set_rng_state_all(state['torch_cuda'])


def checkpoint_model_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('frozen_vggt.')}


def restore_model_state_strict(model, saved):
    current=model.state_dict();expected={k:v for k,v in current.items() if not k.startswith('frozen_vggt.')}
    if set(saved)!=set(expected):
        raise RuntimeError(f"checkpoint non-VGGT key set mismatch: missing={sorted(set(expected)-set(saved))[:20]}, extra={sorted(set(saved)-set(expected))[:20]}")
    mismatched=[k for k in expected if saved[k].shape!=expected[k].shape]
    if mismatched:raise RuntimeError(f"checkpoint non-VGGT shape mismatch: {mismatched[:20]}")
    merged=dict(current);merged.update(saved)
    model.load_state_dict(merged,strict=True)


def _atomic_save(payload,path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    torch.save(payload,temp);os.replace(temp,path)


def run_training(*, manifest_path: Path=MANIFEST, checkpoint: Path=CHECKPOINT,
                 run_dir: Path=Path('/space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1'),
                 hf_revision: str, resume: bool=False):
    """Explicit next-phase 4xRTX4090 launcher. Never called by prepare-only/help."""
    import torch.distributed as dist
    if socket.gethostname().split('.')[0] != '3dimage-14':
        raise RuntimeError('formal training is pinned to 3dimage-14')
    world=int(os.environ.get('WORLD_SIZE','1')); local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=WORLD_SIZE: raise RuntimeError(f'formal world size must be {WORLD_SIZE}, got {world}')
    torch.cuda.set_device(local)
    if torch.cuda.get_device_name(local) not in ('NVIDIA GeForce RTX 4090','NVIDIA RTX 4090'):
        raise RuntimeError(f'RTX4090 required, got {torch.cuda.get_device_name(local)}')
    dist.init_process_group('nccl')
    rank=dist.get_rank();device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    latest=run_dir/'checkpoint_latest.pt'
    if resume:
        if not latest.is_file():raise FileNotFoundError(f'--resume requires {latest}')
    elif run_dir.exists() and any(run_dir.iterdir()): raise RuntimeError(f'run directory is not empty: {run_dir}; pass --resume to restore')
    manifest,scenes,windows=load_manifest(manifest_path)
    manifest_digest=sha256(manifest_path)
    os.environ['VGGT_HF_REVISION']=hf_revision
    seed_everything(42)
    model,opt,transfer,provenance=build_model(checkpoint=checkpoint)
    seed_everything(42+rank)
    model=model.to(device)
    model.train()
    model.frozen_vggt.eval()
    optimizer=build_optimizer(model)
    from scripts import object_locus_v3_set_runtime as provider_runtime
    if rank==0:
        run_dir.mkdir(parents=True,exist_ok=True)
        if not resume:(run_dir/'run_manifest.json').write_text(json.dumps({
            'status':'TRAINING_STARTED_AFTER_EXPLICIT_FLAG','branch':'object-locus-frozen-vggt-posefree-v1',
            'source_science_sha':'9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0',
            'manifest_sha256':manifest_digest,'checkpoint_provenance':provenance,
            'transfer_counts':transfer['counts'],'plan':plan_record(),
            'vggt_source':model.frozen_vggt.source_identity,'optimizer':'AdamW',
            'seeds':{'common_model_initialization':42,'per_rank_training_rng':[42+r for r in range(WORLD_SIZE)],'memory_adapter_isolated':31415},
            'optimizer_state_restored':False,'scheduler_state_restored':False,'rng_state_restored':False,
            'evaluation_launched':False},indent=2)+'\n')
    dist.barrier()
    total_counts=np.zeros(EXPECTED_WINDOWS,dtype=np.int64)
    completed=0;start_epoch=0;last_row=None
    if resume:
        payload=torch.load(latest,map_location='cpu',weights_only=False)
        execution_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
        artifact_keys=('repository','model_id','revision','files','loaded_subtrees',
                       'loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')
        current_artifact={k:model.frozen_vggt.source_identity.get(k) for k in artifact_keys}
        required={'manifest_sha256':manifest_digest,'source_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,
                  'vggt_revision':hf_revision,'world_size':WORLD_SIZE,'total_updates':TOTAL_UPDATES,
                  'git_sha':execution_sha,'vggt_artifact_identity':current_artifact}
        for key,value in required.items():
            if payload.get(key)!=value:raise RuntimeError(f'resume provenance mismatch for {key}: {payload.get(key)!r} != {value!r}')
        restore_model_state_strict(model,payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        completed=int(payload['completed_updates']);start_epoch=int(payload['epoch'])
        if completed!=start_epoch*UPDATES_PER_EPOCH or payload.get('next_position')!=0:
            raise RuntimeError('resume checkpoint is not at the recorded epoch sampler boundary')
        total_counts=np.asarray(payload['rank_window_exposure_counts'][rank],dtype=np.int64)
        rngs=payload['rank_rng']
        if len(rngs)!=WORLD_SIZE:raise RuntimeError('resume rank RNG count mismatch')
        restore_rank_rng(rngs[rank])
        del payload
    elif rank==0:
        run_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch,EPOCHS):
        order=epoch_order(epoch)
        for epoch_update in range(UPDATES_PER_EPOCH):
            update=epoch*UPDATES_PER_EPOCH+epoch_update
            indices=rank_microbatch_indices(epoch,epoch_update,rank)
            batches=[];error=None
            try:batches=[provider_runtime.build_batch(opt,windows[index],device) for index in indices]
            except Exception as exc:error=exc
            synchronized_error(error,device,'batch',update)
            result=train_microbatch_window(model,optimizer,batches,update,base_exposure=update*GLOBAL_BATCH,world_size=WORLD_SIZE)
            for index in indices: total_counts[index]+=1
            completed=update+1;last_row=result
            if rank==0 and (completed%20==0 or completed==TOTAL_UPDATES):
                with (run_dir/'training_rank0.jsonl').open('a') as stream:
                    stream.write(json.dumps({'epoch':epoch+1,'update':completed,'exposure':completed*GLOBAL_BATCH,
                                             **result},allow_nan=False)+'\n')
        dist.barrier()
        counts=torch.as_tensor(total_counts,device=device,dtype=torch.int64)
        dist.all_reduce(counts,op=dist.ReduceOp.SUM)
        expected=np.zeros(EXPECTED_WINDOWS,dtype=np.int64)
        for completed_epoch in range(epoch+1): np.add.at(expected,epoch_order(completed_epoch),1)
        if not np.array_equal(counts.cpu().numpy(),expected):
            raise RuntimeError(f'window exposure accounting mismatch at epoch {epoch+1}')
        rank_rng=[None for _ in range(WORLD_SIZE)];dist.all_gather_object(rank_rng,capture_rank_rng())
        rank_counts=[None for _ in range(WORLD_SIZE)];dist.all_gather_object(rank_counts,total_counts.tolist())
        if rank==0:
            payload={'model':checkpoint_model_state(model),'optimizer':optimizer.state_dict(),
                     'completed_updates':completed,'completed_exposures':completed*GLOBAL_BATCH,
                     'epoch':epoch+1,'next_position':0,'manifest_sha256':manifest_digest,
                     'window_exposure_counts':counts.cpu().tolist(),'rank_window_exposure_counts':rank_counts,'rank_rng':rank_rng,
                     'world_size':WORLD_SIZE,'total_updates':TOTAL_UPDATES,
                     'vggt_revision':hf_revision,
                     'git_sha':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                     'source_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,
                     'initialization_provenance':provenance,
                     'vggt_artifact_identity':{k:model.frozen_vggt.source_identity.get(k) for k in ('repository','model_id','revision','files','loaded_subtrees','loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')},
                     'config':{'global_batch':GLOBAL_BATCH,'microbatch':MICRO_BATCH,'accumulation':ACCUMULATION,
                        'epochs':EPOCHS,'gc_alpha':.01,'understanding_weight':'min(new_exposure/200,1)',
                        'beta':'0.1*min(new_exposure/1000,1)','lr_peak':{'reconstruction':1e-5,
                        'understanding':1e-5,'object_memory':1e-4},'warmup_updates':200,
                        'cosine_final_fraction':.1,'clip_global_norm':1.0,'weight_decay':.05,
                        'source_exposure':50064,'new_exposure_origin':0},
                     'sampler':{'rule':'default_rng(42+epoch) permutation padded by prefix to multiple of 8',
                        'completed_epoch':epoch+1,'next_position':0},
                     'initial_seeds':{'common_model_initialization':42,'rank_training_rng':[42+r for r in range(WORLD_SIZE)],
                        'adapter_isolated':31415},
                     'scheduler':{'warmup_updates':200,'total_updates':TOTAL_UPDATES,'completed_updates':completed},
                     'loss_config':{'gc_alpha':.01}}
            _atomic_save(payload,latest)
            if epoch+1 in (4,8):_atomic_save(payload,run_dir/f'checkpoint_epoch_{epoch+1:02d}.pt')
            (run_dir/'progress.json').write_text(json.dumps({'completed_updates':completed,
                'completed_exposures':completed*GLOBAL_BATCH,'epoch':epoch+1,
                'window_counts':counts.cpu().tolist()},indent=2)+'\n')
        dist.barrier()
    dist.destroy_process_group()
    return {'completed_updates':completed,'completed_exposures':completed*GLOBAL_BATCH,'last_row':last_row}


def run_real_smoke(*, mode: str, manifest_path: Path=MANIFEST, checkpoint: Path=CHECKPOINT,
                   output_dir: Path, hf_revision: str, artifact_manifest: Path=VGGT_ARTIFACT_MANIFEST):
    """Future real-weight GPU smoke. This function is never called by CPU contracts."""
    import time
    import torch.distributed as dist
    from scripts.object_locus_v3_set_runtime import build_batch
    if mode not in ('single','four'):raise ValueError('smoke mode must be single or four')
    distributed=mode=='four'
    if socket.gethostname().split('.')[0]!='3dimage-14':raise RuntimeError('real VGGT smoke is pinned to 3dimage-14')
    if not torch.cuda.is_available():raise RuntimeError('real VGGT smoke requires an allocated CUDA device')
    local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    expected=4 if distributed else 1
    if world!=expected:raise RuntimeError(f'{mode} smoke requires torchrun world size {expected}, got {world}')
    torch.cuda.set_device(local)
    if '4090' not in torch.cuda.get_device_name(local):raise RuntimeError('RTX4090 smoke device required')
    if distributed:dist.init_process_group('nccl')
    rank=dist.get_rank() if distributed else 0
    seed_everything(42);os.environ['VGGT_HF_REVISION']=hf_revision
    model,opt,transfer,provenance=build_model(checkpoint=checkpoint,artifact_manifest=artifact_manifest)
    seed_everything(42+rank)
    model.to(device=f'cuda:{local}',dtype=torch.float32);model.train();model.frozen_vggt.eval()
    optimizer=build_optimizer(model)
    manifest,_,windows=load_manifest(manifest_path)
    order=epoch_order(0)
    all_indices=[rank_microbatch_indices(0,u,r) for u in range(2) for r in range(4)]
    if distributed:
        selected=[all_indices[u*4+rank] for u in range(2)]
    else:
        selected=[(int(order[0]),int(order[0])),(int(order[0]),int(order[0]))]
    if distributed:
        torch.cuda.synchronize();dist.barrier()
    if rank==0:output_dir.mkdir(parents=True,exist_ok=False)
    if distributed:dist.barrier()
    device=torch.device(f'cuda:{local}')
    version_before={id(p):p._version for p in model.frozen_vggt.parameters()}
    first_window=windows[int(order[0])]
    preview_batch=build_batch(opt,first_window,device)
    with torch.no_grad():
        raw_vggt=model.frozen_vggt(preview_batch['images_input'])
        if len(raw_vggt.patch_layers)!=4 or any(x.shape!=(1,2,1369,2048) for x in raw_vggt.patch_layers):
            raise RuntimeError('smoke VGGT selected patch layer shape mismatch')
        if raw_vggt.c2w_cv.shape!=(1,2,4,4) or raw_vggt.depth518.shape!=(1,2,1,518,518):
            raise RuntimeError('smoke VGGT camera/depth output shape mismatch')
        if not all(torch.isfinite(x).all() for x in (*raw_vggt.patch_layers,raw_vggt.c2w_cv,raw_vggt.intrinsics518,raw_vggt.depth518)):
            raise FloatingPointError('smoke VGGT outputs contain nonfinite values')
        del raw_vggt
        preview_generated=model.generate(preview_batch['images_input'])
        preview_calibration=model.calibrate_targets(preview_batch['images_all'],preview_generated)
        preview_view=model.render_generated_at(preview_generated,
            torch.linalg.inv(preview_calibration['c2w']).transpose(-1,-2),preview_calibration['intrinsics'])
    for key in ('gaussians','gaussian_membership','p_class','region_mass','semantic_scores','alpha'):
        value=preview_view.get(key)
        if value is None or not torch.isfinite(value).all():raise FloatingPointError(f'smoke invalid {key}')
    if rank==0:
        torch.save({k:preview_view[k].detach().cpu() for k in ('gaussians','gaussian_membership','p_class','region_mass','semantic_scores','alpha')},output_dir/'generated_preview.pt')
    del preview_view,preview_generated,preview_calibration,preview_batch
    torch.cuda.reset_peak_memory_stats(device)
    rows=[]
    for update,index_pair in enumerate(selected):
        batches=[build_batch(opt,windows[index],device) for index in index_pair]
        base_exposure=update*(GLOBAL_BATCH if distributed else ACCUMULATION)
        started=time.perf_counter()
        row=train_microbatch_window(model,optimizer,batches,update,base_exposure=base_exposure,
                                    world_size=expected)
        torch.cuda.synchronize(device);row['seconds']=time.perf_counter()-started
        row['smoke_mode']=mode;row['rank']=rank;row['window_indices']=list(index_pair)
        row['gradient_family_norms']={family:float(torch.sqrt(sum((p.grad.detach().float().square().sum() for n,p in model.named_parameters() if p.grad is not None and parameter_family(n)==family),torch.zeros((),device=device))))
            for family in ('reconstruction','understanding','object_memory')}
        if not any(p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters() if n.startswith('vggt_memory_adapter.')):
            raise RuntimeError('smoke memory adapter has no reconstruction gradient')
        if any(p.grad is not None for p in model.frozen_vggt.parameters()) or model.frozen_vggt.training:
            raise RuntimeError('frozen VGGT grad/eval contract failed')
        if any(p._version!=version_before[id(p)] for p in model.frozen_vggt.parameters()):
            raise RuntimeError('frozen VGGT parameter version changed')
        rows.append(row)
    # Separate path-only understanding backward: no optimizer step and no clock update.
    formal_completed_exposure=4 if not distributed else 16
    model.understanding_step=formal_completed_exposure
    diagnostic_batch=build_batch(opt,windows[int(order[0])],device)
    diagnostic_output,diagnostic_metrics=model.step_loss(diagnostic_batch,step=formal_completed_exposure,understanding_weight=1.0)
    adapter_params=[p for n,p in model.named_parameters() if n.startswith('vggt_memory_adapter.') and p.requires_grad]
    diagnostic_grads=torch.autograd.grad(diagnostic_metrics['loss_understanding'],adapter_params,allow_unused=True)
    if not any(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in diagnostic_grads):
        raise RuntimeError('weight=1 understanding diagnostic has no memory-adapter gradient')
    del diagnostic_output,diagnostic_metrics,diagnostic_batch,diagnostic_grads
    if rank==0:
        payload={'mode':mode,'world_size':expected,'rows':rows,'updates':2,
            'exposures':formal_completed_exposure,'formal_schedule_start_exposure':0,
            'understanding_diagnostic':'separate weight=1 backward; no optimizer step or clock increment',
            'source_checkpoint':provenance,'migration_counts':transfer['counts'],
            'initial_seeds':{'common_model_initialization':42,'rank_training_rng':[42+r for r in range(expected)],'memory_adapter_isolated':31415},
            'vggt_artifact':model.frozen_vggt.source_identity,
            'manifest_sha256':sha256(manifest_path),'window_indices':selected,
            'lr_schedule':'formal schedule indices 0 and 1; single-card world_size=1 averaging',
            'status':'GPU_SMOKE_COMPLETED'}
        _atomic_save({'model':checkpoint_model_state(model),'optimizer':optimizer.state_dict(),
            'completed_updates':2,'completed_exposures':payload['exposures'],'mode':mode},output_dir/'smoke_state.pt')
        (output_dir/'smoke_report.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    if distributed:dist.barrier();dist.destroy_process_group()
    return {'mode':mode,'updates':2,'exposures':formal_completed_exposure}
