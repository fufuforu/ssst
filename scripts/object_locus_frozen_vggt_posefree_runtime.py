"""Review-locked configuration, migration, optimizer, and CPU plan helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import sys
import subprocess
import time
from pathlib import Path
import random
import traceback

import numpy as np
import torch
from torch import nn

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))
# Must run before importing tokengs.models: its package initializer imports the
# legacy renderer, which expects this cluster's documented fused-SSIM shim.
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)

from tokengs.models.frozen_vggt_posefree import FrozenVGGT
from tokengs.models.object_locus_posefree_geometry import ContextDepthSim3Error, old_shared_context_alignment_diagnostics

MANIFEST=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')
CHECKPOINT=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')
EXPECTED_CHECKPOINT_SHA='68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a'
EXPECTED_VGGT_COMMIT='a288dd0f14786c93483e45524328726ab7b1b4ce'
EXPECTED_SCENES=1191
EXPECTED_WINDOWS=8337
WORLD_SIZE=8
MICRO_BATCH=1
ACCUMULATION=1
EPOCHS=8
GLOBAL_BATCH=8
WINDOWS_PER_EPOCH=8344
UPDATES_PER_EPOCH=1043
TOTAL_UPDATES=8344
TOTAL_EXPOSURES=66752
VGGT_ARTIFACT_MANIFEST=REPO/'vggt_artifact_manifest.json'
CALIBRATION_PROTOCOL='shared_context_depth_sim3_v2'
GEOMETRY_QUALITY_POLICY='monitor_v1'
if WORLD_SIZE*MICRO_BATCH*ACCUMULATION!=GLOBAL_BATCH:
    raise RuntimeError('resource adaptation must preserve global batch 8')

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


def rank_microbatch_indices(epoch: int, update: int, rank: int, *, world_size: int=WORLD_SIZE) -> tuple[int,...]:
    if world_size not in (4,8): raise ValueError('supported training world sizes are four or eight')
    if not 0<=rank<world_size or not 0<=update<UPDATES_PER_EPOCH: raise ValueError("rank/update out of range")
    order=epoch_order(epoch)
    per_rank=GLOBAL_BATCH//world_size
    start=update*GLOBAL_BATCH+rank*per_rank
    return tuple(int(order[start+i]) for i in range(per_rank))


def exposure_schedule(exposure: int):
    exposure=max(int(exposure),0)
    return min(exposure/200,1.0),0.1*min(exposure/1000,1.0)


def lr_multiplier(update: int, total_updates: int | None=None):
    step=int(update)+1
    total=TOTAL_UPDATES if total_updates is None else int(total_updates)
    if not 1<=step<=total: raise ValueError("optimizer update outside planned range")
    return step/200 if step<=200 else .1+.9*(1+math.cos(math.pi*(step-200)/(total-200)))/2


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


def gc_combine(parameter_names, reconstruction_gradients, understanding_gradients, *, vggt_reconstruction=False,
               adapter_understanding_scale=1.):
    if adapter_understanding_scale not in (0., .01, 1.):
        raise ValueError('unauthorized adapter understanding-gradient scale')
    combined=[]
    for name,gr,gu in zip(parameter_names,reconstruction_gradients,understanding_gradients):
        if gr is None and gu is None: combined.append(None); continue
        base=torch.zeros_like(gu if gr is None else gr)
        if gr is not None: base=base+gr
        reconstruction = parameter_family(name)=='reconstruction' or (vggt_reconstruction and name.startswith('frozen_vggt.'))
        scale = adapter_understanding_scale if name.startswith('vggt_memory_adapter.') else (.01 if reconstruction else 1.)
        if gu is not None: base=base+scale*gu
        combined.append(base)
    return combined


def install_gradients(parameters, gradients, *, world_size=None):
    """Average per-rank gradients when distributed is initialized (CPU testable)."""
    world_size=world_size or 1
    distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        for p,g in zip(parameters,gradients): p.grad=None if g is None else g.to(p.device).clone()
        return
    present=torch.tensor([g is not None for g in gradients],device=parameters[0].device,dtype=torch.uint8)
    torch.distributed.all_reduce(present,op=torch.distributed.ReduceOp.MAX)
    active=present.tolist()
    buckets=[];indices=[];numel=0;bucket_limit=25*1024*1024
    def flush(which):
        if not which: return
        values=[gradients[i] if gradients[i] is not None else torch.zeros_like(parameters[i]) for i in which]
        flat=torch.cat([value.to(parameters[0].device).reshape(-1) for value in values])
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


def save_calibration_failure(error,update,rank,stage='formal_training'):
    if not isinstance(error,ContextDepthSim3Error): return None
    root=os.environ.get('POSEFREE_V2_EVIDENCE_DIR')
    if not root: return None
    directory=Path(root)/'geometry_failures'/stage/f'update_{int(update):06d}_rank_{int(rank)}'
    directory.mkdir(parents=True,exist_ok=True)
    _write_json_atomic(directory/'calibration_failure.json',{'protocol':CALIBRATION_PROTOCOL,
        'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,'status':'HARD_VALIDITY_FAILED','update':int(update),'rank':int(rank),
        'error':str(error),'diagnostics':error.diagnostics})
    if error.points:
        np.savez_compressed(directory/'calibration_failure_points.npz',**error.points)
    return str(directory)


def train_microbatch_window(model,optimizer,batches,update,*,base_exposure,world_size=None,
                            monitor=None,window_metadata=None,stage="formal_training",total_updates=None,
                            reconstruction_adaptation=False,accumulation=ACCUMULATION,
                            cpu_saved_tensors=False,cpu_gradient_accumulation=False,vggt_reconstruction=False,lr_update=None,
                            adapter_understanding_scale=1.,understanding_only=False,gradient_diagnostic=False):
    """Two-microbatch GC step with one accumulated gradient family per parameter."""
    if accumulation not in (1,2) or len(batches)!=accumulation:
        raise ValueError(f"each rank must receive exactly {accumulation} accumulation microbatches")
    named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
    names=[n for n,_ in named];params=[p for _,p in named]
    optimizer.zero_grad(set_to_none=True)
    combined_acc=[None]*len(params)
    weight,beta=exposure_schedule(base_exposure)
    if reconstruction_adaptation: weight,beta=0.,0.
    device=next(model.parameters()).device
    local_rows=[]
    for micro_index,batch in enumerate(batches):
        output=metrics=None;error=None
        try:
            from contextlib import nullcontext
            context = torch.autograd.graph.save_on_cpu(pin_memory=True) if cpu_saved_tensors else nullcontext()
            with context:
                output,metrics=model.step_loss(batch,step=base_exposure,understanding_weight=weight)
            if not all(torch.isfinite(metrics[key]).all() for key in ('loss_recon','loss_understanding')):
                raise FloatingPointError('nonfinite reconstruction/understanding loss')
        except Exception as exc:
            error=exc
            if isinstance(error,ContextDepthSim3Error):
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                try: save_calibration_failure(error,update,rank,stage)
                except Exception as save_exc:
                    error=RuntimeError(f"{exc}; saving geometry evidence failed: {save_exc}")
        synchronized_error(error,device,'forward',int(update))
        error=None
        try:
            if understanding_only:
                if metrics['loss_recon'].requires_grad:
                    raise RuntimeError('understanding-only probe reconstruction depends on trainable parameters')
                rec=(None,)*len(params)
            else:
                rec=torch.autograd.grad(metrics['loss_recon']/accumulation,params,
                    retain_graph=weight>0,allow_unused=True)
            if cpu_gradient_accumulation:
                rec=tuple(None if g is None else g.cpu() for g in rec)
            under=(torch.autograd.grad(metrics['loss_understanding']*weight/accumulation,params,
                allow_unused=True) if weight>0 else (None,)*len(params))
            if cpu_gradient_accumulation:
                under=tuple(None if g is None else g.cpu() for g in under)
            mixed=gc_combine(names,rec,under,vggt_reconstruction=vggt_reconstruction,
                             adapter_understanding_scale=adapter_understanding_scale)
            grad_stats=None
            if gradient_diagnostic:
                grad_stats={}
                for prefix in ('anchor_decoder.','vggt_memory_adapter.','understanding.','panoptic.'):
                    triples=[]
                    for name,gr,gu in zip(names,rec,under):
                        if not name.startswith(prefix) or (gr is None and gu is None):continue
                        rr=gr.square().sum() if gr is not None else gu.new_zeros(())
                        uu=gu.square().sum() if gu is not None else gr.new_zeros(())
                        dot=(gr*gu).sum() if gr is not None and gu is not None else rr.new_zeros(())
                        triples.append(torch.stack((rr,uu,dot)))
                    if triples:
                        rr,uu,dot=torch.stack(triples).sum(0).detach().cpu().tolist()
                        grad_stats[prefix]=dict(reconstruction_norm=math.sqrt(rr),weighted_understanding_norm=math.sqrt(uu),
                            cosine=dot/math.sqrt(rr*uu) if rr>0 and uu>0 else None)
            if any(g is not None and not torch.isfinite(g).all() for g in mixed):
                raise FloatingPointError('nonfinite per-microbatch GC-combined gradient')
            combined_acc=[None if g is None and old is None else
                (torch.zeros_like(g) if old is None else old)+(torch.zeros_like(old) if g is None else g)
                for p,old,g in zip(params,combined_acc,mixed)]
            calibration_row=None
            calibration=output.get('prediction',{}).get('target_camera_calibration') if isinstance(output,dict) else None
            if calibration is not None:
                ds=calibration['diagnostics'][0]
                calibration_row={'protocol':CALIBRATION_PROTOCOL,'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,
                    'fit_status':ds['fit_status'],'quality_status':ds['quality_status'],
                    'quality_warning_reasons':[v['quality_warning_reasons'] for v in ds['views']],
                    'diagnostic_unavailable_reasons':[v['diagnostic_unavailable_reasons'] for v in ds['views']],
                    'scale':ds['s'],
                    'valid_points':[v['valid_count'] for v in ds['views']],
                    'positive_z_ratio':[v['positive_z_ratio'] for v in ds['views']],
                    'reprojection_median_px':[v['reprojection_median_px'] for v in ds['views']],
                    'reprojection_p90_px':[v['reprojection_p90_px'] for v in ds['views']],
                    'residual_3d_median_over_scene_median_depth':[
                        v['residual_3d_median_over_scene_median_depth'] for v in ds['views']]}
            if monitor is not None and calibration is not None:
                monitor.record(calibration,update,(window_metadata or [None])[micro_index])
            local_rows.append({'loss_recon':float(metrics['loss_recon'].detach()),
                'loss_understanding':float(metrics['loss_understanding'].detach()),
                'calibration':calibration_row,'gradient_diagnostic':grad_stats})
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
        mult=lr_multiplier(update if lr_update is None else lr_update,total_updates)
        for group in optimizer.param_groups: group['lr']=group['peak_lr']*mult
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params): raise FloatingPointError('nonfinite parameter after optimizer.step')
    except Exception as exc:error=exc;norm=torch.tensor(float('nan'))
    synchronized_error(error,device,'optimizer',int(update))
    effective_world=world_size or (torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized() else 1)
    row={'completed_updates':int(update)+1,'completed_exposures':(int(update)+1)*effective_world*MICRO_BATCH*accumulation,
         'understanding_weight':weight,'beta':beta,'lr_multiplier':mult,'preclip_norm':float(norm),
         'loss_recon':sum(x['loss_recon'] for x in local_rows)/len(local_rows),
         'loss_understanding':sum(x['loss_understanding'] for x in local_rows)/len(local_rows),
         'calibration':local_rows[0]['calibration'],
         'group_lr':{g['name']:g['lr'] for g in optimizer.param_groups}}
    if gradient_diagnostic:row['microbatch_gradient_diagnostics']=[x['gradient_diagnostic'] for x in local_rows]
    if accumulation>1:
        row['microbatch_calibrations']=[x['calibration'] for x in local_rows]
    if device.type=='cuda':
        row.update(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
                   peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device))
    return row


def plan_record():
    return {'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,'calibration_protocol':CALIBRATION_PROTOCOL,
            'manifest':str(MANIFEST),'manifest_sha256':sha256(MANIFEST) if MANIFEST.exists() else None,
            'expected_scenes':EXPECTED_SCENES,'expected_windows':EXPECTED_WINDOWS,'context_views_per_window':2,
            'world_size':WORLD_SIZE,'gpu_model':'RTX3090','node':'3dimage-13','microbatch_per_rank':MICRO_BATCH,'accumulation':ACCUMULATION,
            'global_batch':8,'epochs':8,'windows_per_epoch_padded':WINDOWS_PER_EPOCH,
            'updates_per_epoch':UPDATES_PER_EPOCH,'total_updates':TOTAL_UPDATES,
            'total_exposures':TOTAL_EXPOSURES,'sampler':'default_rng(42+epoch); permutation padded by prefix',
            'new_training_clock_origin':0}


def training_configuration():
    return {'calibration_protocol':CALIBRATION_PROTOCOL,
        'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,
        'calibration_acceptance':{
            'hard_validity':{'min_valid_points_per_view':32,'finite_inputs_cameras_fit_loss_gradients_parameters':True,
                'positive_scale':True,'invertible_camera_and_K':True,'SO3_atol':1e-8,
                'min_covariance_second_eigen_ratio':1e-6,'positive_rms':True},
            'quality_reference':{'min_positive_z_ratio':.95,'max_reprojection_median_px':4.,
                'max_reprojection_p90_px':12.,'handling':'diagnostic warning only; never changes cameras or loss'},
            'huber_irls_iterations':5},
        'node':'3dimage-13','gpu_model':'RTX3090','world_size':WORLD_SIZE,
        'global_batch':GLOBAL_BATCH,'microbatch':MICRO_BATCH,
        'accumulation':ACCUMULATION,'epochs':EPOCHS,'updates_per_epoch':UPDATES_PER_EPOCH,
        'total_updates':TOTAL_UPDATES,'gc_alpha':.01,
        'understanding_weight':'min(new_exposure/200,1)',
        'beta':'0.1*min(new_exposure/1000,1)',
        'lr_peak':{'reconstruction':1e-5,'understanding':1e-5,'object_memory':1e-4},
        'warmup_updates':200,'cosine_final_fraction':.1,'clip_global_norm':1.0,
        'weight_decay':.05,'source_exposure':50064,'new_exposure_origin':0}


def smoke_exposure_count(mode: str, updates: int=2) -> int:
    if mode not in ('single','eight'):raise ValueError('smoke mode must be single or eight')
    world=1 if mode=='single' else WORLD_SIZE
    return int(updates)*world*MICRO_BATCH*ACCUMULATION


def seed_everything(seed: int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(*, checkpoint: Path=CHECKPOINT, opt=None,
                artifact_manifest: Path=VGGT_ARTIFACT_MANIFEST,
                preserve_fp32_aggregator: bool=False, fresh_reconstruction_state=None):
    """Construct, initialize the retained panoptic modules, then strict-copy epoch 06."""
    from tokengs.models.object_locus_frozen_vggt_posefree import LocusGSObjectLocusFrozenVGGT
    revision=os.environ.get('VGGT_HF_REVISION')
    if not revision:
        raise RuntimeError('set VGGT_HF_REVISION to the reviewed Hugging Face commit before model construction')
    vggt=FrozenVGGT.from_pretrained(local_files_only=True,revision=revision,
                                    artifact_manifest=artifact_manifest,
                                    preserve_fp32_aggregator=preserve_fp32_aggregator)
    if opt is None:
        from tokengs.options import config_defaults
        opt=config_defaults['train_siu3r_object_locus_frozen_vggt_posefree_v1']
    model=LocusGSObjectLocusFrozenVGGT(opt,vggt=vggt)
    # Optional diagnostic capture, before any checkpoint migration. Constructors
    # retain their original initialization, including zero residual heads,
    # anchor geometry and shared decoder aliases; default callers are unchanged.
    if fresh_reconstruction_state is not None:
        if fresh_reconstruction_state:
            raise ValueError('fresh reconstruction capture must be empty')
        fresh_reconstruction_state.update({k:v.detach().cpu().clone()
            for k,v in model.state_dict().items() if k.startswith(RETAINED_PREFIXES[:5])})
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


def _json_finite(value):
    if torch.is_tensor(value): return _json_finite(value.detach().cpu().tolist())
    if isinstance(value,np.ndarray): return _json_finite(value.tolist())
    if isinstance(value,dict): return {str(k):_json_finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_finite(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)): return None
    if isinstance(value,(np.integer,np.floating)): return value.item()
    return value


def _write_json_atomic(path: Path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(_json_finite(value),indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)



class GeometryMonitor:
    """Scalar per-window evidence; two shared warning slots, no tensor retention."""
    def __init__(self,directory,rank,stage,updates_per_epoch=UPDATES_PER_EPOCH):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.rank=int(rank);self.stage=stage
        self.updates_per_epoch=int(updates_per_epoch)
        self.counts={'windows':0,'OK':0,'WARNING':0,'warning_reasons':{}}
        self.latest=None

    def record(self,calibration,update,metadata):
        ds=calibration['diagnostics'][0]
        views=ds['views'];metadata=metadata or {}
        row={'stage':self.stage,'epoch':int(update)//self.updates_per_epoch+1,
            'update':int(update)+1,'rank':self.rank,**metadata,
            'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,'fit_status':ds['fit_status'],
            'quality_status':ds['quality_status'],'quality_warning_reasons':[v['quality_warning_reasons'] for v in views],
            'diagnostic_unavailable_reasons':[v['diagnostic_unavailable_reasons'] for v in views],
            'scale':ds['s'],'valid_points':[v['valid_count'] for v in views],
            **{key:[v[key] for v in views] for key in ('positive_z_ratio','reprojection_median_px',
                'reprojection_p90_px','residual_3d_median_over_scene_median_depth',
                'residual_3d_p90_over_scene_median_depth','residual_3d_rmse_over_scene_median_depth')}}
        row=_json_finite(row)
        with (self.directory/f'geometry_monitor_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        self.latest=row;self.counts['windows']+=1;self.counts[ds['quality_status']]+=1
        for reason in {reason for view in views for reason in view['quality_warning_reasons']}:
            self.counts['warning_reasons'][reason]=self.counts['warning_reasons'].get(reason,0)+1
        if ds['quality_status']=='WARNING':
            evidence=os.environ.get('POSEFREE_V2_EVIDENCE_DIR')
            if evidence:
                root=Path(evidence)/'geometry_warnings';root.mkdir(parents=True,exist_ok=True)
                # Exclusive window claims prevent duplicate evidence across ranks/stages.
                claim=root/f"window_{metadata.get('manifest_index','unknown')}"
                try:claim.mkdir()
                except FileExistsError:return
                for slot in range(2):
                    directory=root/f'example_{slot}'
                    try:directory.mkdir()
                    except FileExistsError:continue
                    _write_json_atomic(directory/'calibration_warning.json',{'monitor':row,'diagnostics':ds})
                    cameras={key:calibration[key].detach().cpu().numpy() for key in ('c2w','intrinsics_matrix')}
                    np.savez_compressed(directory/'calibration_warning_points.npz',**calibration['point_records'][0],**cameras)
                    break
                # Only the first two distinct warnings need a persistent claim.
                if not any((root/f'example_{i}'/'calibration_warning.json').is_file() and
                    json.loads((root/f'example_{i}'/'calibration_warning.json').read_text())['monitor'].get('manifest_index')==metadata.get('manifest_index') for i in range(2)):
                    claim.rmdir()

    def summary(self):
        return {'counts':self.counts,'latest':self.latest,'stage':self.stage,'rank':self.rank}

    def write_summary(self,completed):
        with (self.directory/f'geometry_monitor_summary_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps({'completed_updates':completed,**self.summary()},allow_nan=False)+'\n')


def window_identity(index,window):
    return {'manifest_index':int(index),'scene':window['scene'],
            'context_ids':window['context'],'novel_ids':window['novel']}


def validated_smoke(path,mode,world,execution_sha,manifest_digest,artifact):
    path=Path(path)
    if path.is_dir():path=path/'smoke_report.json'
    report=json.loads(path.read_text())
    if (report.get('status')!='GPU_SMOKE_COMPLETED' or report.get('mode')!=mode or
        report.get('world_size')!=world or report.get('updates')!=2 or
        report.get('exposures')!=smoke_exposure_count(mode) or
        report.get('manifest_sha256')!=manifest_digest or
        report.get('calibration_protocol')!=CALIBRATION_PROTOCOL or
        report.get('source_checkpoint',{}).get('sha256')!=EXPECTED_CHECKPOINT_SHA):
        raise RuntimeError(f'smoke evidence contract/identity mismatch: {path}')
    keys=('repository','model_id','revision','files','loaded_subtrees','loaded_key_sha256')
    if any(report['vggt_artifact'].get(k)!=artifact.get(k) for k in keys):
        raise RuntimeError(f'smoke asset identity mismatch: {path}')
    if mode=='eight' and report.get('window_indices')!=epoch_order(0)[:16].astype(int).tolist():
        raise RuntimeError('eight smoke did not use the fixed first sixteen windows including 6923')
    reused=mode=='single' and report.get('geometry_quality_policy')!=GEOMETRY_QUALITY_POLICY
    sha=report.get('execution_git_sha')
    if reused:
        attempt=json.loads((path.parents[2]/'attempt_manifest.json').read_text())
        sha=attempt['execution_git_sha']
        if sha!='d4107c881b0c5ce4e0bb187f620d0607e9e41454' or attempt['slurm_job_id']!='59658':
            raise RuntimeError('unrecognized historical single smoke provenance')
    elif sha!=execution_sha or report.get('geometry_quality_policy')!=GEOMETRY_QUALITY_POLICY:
        raise RuntimeError('current smoke must execute the fixed monitor snapshot')
    return {'path':str(path),'status':report['status'],'world_size':world,'updates':2,
        'exposures':report['exposures'],'execution_git_sha':sha,
        'geometry_quality_policy':report.get('geometry_quality_policy','strict_v2_historical'),
        'single_smoke_reused':reused,'reuse_reason':
            'Only quality error handling and evidence changed; fitting, precision, gradients and science unchanged.' if reused else None,
        'geometry_monitor_by_rank':report.get('geometry_monitor_by_rank')}

def run_window4253_calibration(*, output_dir: Path, manifest_path: Path=MANIFEST,
                               checkpoint: Path=CHECKPOINT, hf_revision: str,
                               artifact_manifest: Path=VGGT_ARTIFACT_MANIFEST):
    """Capture old signed-baseline evidence and validate v2 on fixed window 4253."""
    if socket.gethostname().split('.')[0]!='3dimage-13':
        raise RuntimeError('window 4253 real calibration check is pinned to 3dimage-13')
    if not torch.cuda.is_available(): raise RuntimeError('window 4253 calibration requires its allocated CUDA device')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('window 4253 calibration evidence must run as a single process')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    device=torch.device(f'cuda:{local}')
    if '3090' not in torch.cuda.get_device_name(local): raise RuntimeError('RTX3090 required for window 4253 calibration')
    seed_everything(42);os.environ['VGGT_HF_REVISION']=hf_revision
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    from scripts.object_locus_v3_set_runtime import build_batch
    model,opt,transfer,provenance=build_model(checkpoint=checkpoint,artifact_manifest=artifact_manifest)
    model.to(device);model.eval();model.frozen_vggt.eval()
    manifest,_,windows=load_manifest(manifest_path)
    if len(windows)<=4253: raise RuntimeError('locked manifest has no window index 4253')
    window=windows[4253]
    batch=build_batch(opt,window,device)
    context=batch['images_input'];all_rgb=batch['images_all']
    shared_equal=torch.equal(context,all_rgb[:,:2])
    if not shared_equal: raise RuntimeError('window 4253 context RGB differs between generation and calibration inputs')
    with torch.no_grad():
        generated=model.generate(context)
        calibration_pass=model.frozen_vggt.calibration_with_context_depth(all_rgb)
    old=old_shared_context_alignment_diagnostics(
        calibration_pass['c2w_cv'][:,:3],generated['coordinates']['raw_c2w_cv'])
    old_payload={'window_index':4253,'window':window,'status':'RECORDED',
        'old_protocol':'orientation constrained SO(3) plus signed least squares center baseline scale',
        'context_rgb_exactly_shared':shared_equal,
        'c2w_all_raw':calibration_pass['c2w_cv'].detach().cpu().tolist(),
        'c2w_context_only_raw':generated['coordinates']['raw_c2w_cv'].detach().cpu().tolist(),
        'old_alignment':old,'official_vggt':generated['vggt_source_identity'],
        'source_checkpoint':provenance,'manifest_sha256':sha256(manifest_path)}
    _write_json_atomic(output_dir/'window4253_old_alignment.json',old_payload)
    observed={key:generated[key].detach().clone() for key in
        ('gaussians','gaussian_membership','p_class','predicted_points') if torch.is_tensor(generated.get(key))}
    if generated.get('states'):
        observed['last_state_tokens']=generated['states'][-1]['tokens'].detach().clone()
    try:
        calibrated=model._calibrate_result(generated,calibration_pass)
    except Exception as exc:
        status='GEOMETRY_BLOCKED' if isinstance(exc,ContextDepthSim3Error) else 'ENGINEERING_ERROR'
        new_payload={'window_index':4253,'window':window,'status':status,
            'error_type':type(exc).__name__,'error':str(exc),
            'diagnostics':getattr(exc,'diagnostics',None),
            'c2w_all_raw':calibration_pass['c2w_cv'].detach().cpu().tolist(),
            'k518_all':calibration_pass['intrinsics518'].detach().cpu().tolist(),
            'c2w_context_only_raw':generated['coordinates']['raw_c2w_cv'].detach().cpu().tolist(),
            'k518_context_only':generated['predicted_context_intrinsics518'].detach().cpu().tolist(),
            'depth_context_from_full_window_aggregator':True,
            'old_alignment_file':'window4253_old_alignment.json'}
        # Record actual context K and the generated scene projection matrix.
        new_payload['k256_context_only']=generated['predicted_context_intrinsics_matrix'].detach().cpu().tolist()
        new_payload['official_vggt']=generated['vggt_source_identity']
        _write_json_atomic(output_dir/'window4253_context_sim3_v2.json',new_payload)
        point_rows=getattr(exc,'points',{})
        np.savez_compressed(output_dir/'window4253_context_sim3_v2_points.npz',**point_rows,
            c2w_all_raw=calibration_pass['c2w_cv'].detach().cpu().numpy(),
            k518_all=calibration_pass['intrinsics518'].detach().cpu().numpy(),
            c2w_context_only_raw=generated['coordinates']['raw_c2w_cv'].detach().cpu().numpy(),
            k256_context_only=generated['predicted_context_intrinsics_matrix'].detach().cpu().numpy())
        raise
    changed=[key for key,value in observed.items() if not torch.equal(value,
        generated['states'][-1]['tokens'] if key=='last_state_tokens' else generated[key])]
    if changed: raise RuntimeError(f'calibration mutated generated model outputs: {changed}')
    point_row=calibrated['point_records'][0]
    np.savez_compressed(output_dir/'window4253_context_sim3_v2_points.npz',**point_row,
        c2w_all_raw=calibration_pass['c2w_cv'].detach().cpu().numpy(),
        k518_all=calibration_pass['intrinsics518'].detach().cpu().numpy(),
        c2w_aligned=calibrated['c2w'].detach().cpu().numpy(),
        c2w_context_only=generated['predicted_context_c2w'].detach().cpu().numpy(),
        k256_context_only=generated['predicted_context_intrinsics_matrix'].detach().cpu().numpy())
    payload={'window_index':4253,'window':window,'status':'PASS',
        'calibration_protocol':CALIBRATION_PROTOCOL,'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,'context_rgb_exactly_shared':shared_equal,
        'depth_context_from_full_window_aggregator':True,
        'diagnostics':calibrated['diagnostics'],
        'c2w_all_raw':calibration_pass['c2w_cv'].detach().cpu().tolist(),
        'c2w_aligned':calibrated['c2w'].detach().cpu().tolist(),
        'c2w_context_only':generated['predicted_context_c2w'].detach().cpu().tolist(),
        'k518_all':calibration_pass['intrinsics518'].detach().cpu().tolist(),
        'k256_aligned':calibrated['intrinsics_matrix'].detach().cpu().tolist(),
        'old_alignment_file':'window4253_old_alignment.json',
        'points_file':'window4253_context_sim3_v2_points.npz',
        'generated_outputs_unchanged_after_calibration':True,
        'official_vggt':generated['vggt_source_identity'],'source_checkpoint':provenance,
        'migration_counts':transfer['counts'],'manifest_sha256':sha256(manifest_path)}
    _write_json_atomic(output_dir/'window4253_context_sim3_v2.json',payload)
    print(json.dumps({'window_index':4253,'status':'PASS','s':calibrated['diagnostics'][0]['s'],
        'reprojection_median_px':[v['reprojection_median_px'] for v in calibrated['diagnostics'][0]['views']],
        'reprojection_p90_px':[v['reprojection_p90_px'] for v in calibrated['diagnostics'][0]['views']]},allow_nan=False),flush=True)
    del model,generated,calibration_pass,batch
    torch.cuda.empty_cache()
    return payload


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


STAGED_RECIPE = 'vggt_reconstruction_adapt2_frozen_joint4_v1'
STAGED_ADAPTER_U001_RECIPE = 'vggt_reconstruction_adapt2_frozen_joint4_adapter_u001_v1'
STAGED_U001_BOUNDARY = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_reconstruction_adaptation_epoch_02.pt')
STAGED_U001_BOUNDARY_SHA = 'a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c'
STAGED_NODE_GPU = {'3dimage-11':'3090','3dimage-13':'3090',
                   '3dimage-14':'4090','3dimage-17':'4090','3dimage-18':'4090'}


def staged_training_configuration(world_size=8, adapter_understanding_scale=1.):
    if world_size not in (4,8): raise ValueError('staged training requires four or eight ranks')
    if adapter_understanding_scale not in (1., .01): raise ValueError('unauthorized staged adapter gradient scale')
    config = training_configuration()
    config.update(recipe=STAGED_RECIPE,world_size=world_size,accumulation=GLOBAL_BATCH//world_size,
        node='3dimage-13' if world_size==4 else 'one of 3dimage-[11,13,14,17,18]',
        gpu_model='RTX3090' if world_size==4 else 'RTX3090 or RTX4090', epochs=6, total_updates=6*UPDATES_PER_EPOCH,
        total_exposures=6*WINDOWS_PER_EPOCH, stage_epochs={'reconstruction_adaptation':2,'frozen_joint':4},
        stage_updates={'reconstruction_adaptation':2*UPDATES_PER_EPOCH,'frozen_joint':4*UPDATES_PER_EPOCH},
        adaptation_trainable_vggt=['aggregator.frame_blocks','aggregator.global_blocks'],
        adaptation_frozen_vggt=['aggregator.patch_embed','aggregator.camera_token','aggregator.register_token','camera_head','depth_head'],
        adaptation_lr_peak={'vggt':1e-6,'reconstruction':1e-5,'memory_adapter':1e-4},
        adaptation_precision='FP32 AA parameters/master/moments; BF16 autocast; original FP32 decoder/head/loss',
        adaptation_optimizer='native FP32 AdamW CPU offload; betas=(0.9,0.95), eps=1e-8',
        adaptation_understanding='frozen and unused; object feedback disabled; original reconstruction loss only',
        stage_transition='retain adapted VGGT/decoder/adapter and original understanding; fresh joint optimizer and exposure warmup',
        frozen_joint_precision='adapted VGGT aggregator BF16, camera/depth FP32; unchanged downstream precision',
        camera_inputs='predicted/detached in both phases; original shared_context_depth_sim3_v2; no GT camera input')
    if adapter_understanding_scale == .01:
        config.update(recipe=STAGED_ADAPTER_U001_RECIPE,adapter_understanding_scale=.01,
            adapter_gc_rule='g_rec + 0.01*g_under; original understanding weight included',
            understanding_panoptic_gc_rule='unchanged g_rec + g_under')
    return config


def validate_staged_boundary(payload, config, revision, manifest_digest):
    """Only the exact completed, unchanged reconstruction stage may be reused."""
    if config != staged_training_configuration(4, .01):
        raise RuntimeError('boundary reuse requires the authorized four-rank adapter-u001 recipe')
    expected = dict(recipe=STAGED_RECIPE,config=staged_training_configuration(4),world_size=4,
        phase='reconstruction_adaptation',phase_completed_updates=2*UPDATES_PER_EPOCH,
        completed_updates=2*UPDATES_PER_EPOCH,completed_exposures=2*WINDOWS_PER_EPOCH,
        epoch=2,next_position=0,manifest_sha256=manifest_digest,
        source_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,vggt_revision=revision)
    if any(payload.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('reconstruction boundary identity/clock mismatch')
    if len(payload['rank_rng']) != 4 or not payload.get('vggt_model'):
        raise RuntimeError('reconstruction boundary weights/RNG absent')
    counts=np.asarray(payload['rank_window_exposure_counts'])
    expected_counts=np.zeros((4,EXPECTED_WINDOWS),dtype=np.int64)
    for epoch in range(2):
        for update in range(UPDATES_PER_EPOCH):
            for rank in range(4):
                np.add.at(expected_counts[rank],list(rank_microbatch_indices(epoch,update,rank,world_size=4)),1)
    if not np.array_equal(counts,expected_counts) or not np.array_equal(counts.sum(0),payload['window_exposure_counts']):
        raise RuntimeError('reconstruction boundary exposure ownership mismatch')


def configure_staged_phase(model, phase, device):
    if phase not in ('reconstruction_adaptation','frozen_joint'): raise ValueError(phase)
    if not hasattr(model,'_staged_original_trainable'):
        model._staged_original_trainable={n:p.requires_grad for n,p in model.named_parameters() if not n.startswith('frozen_vggt.')}
    for name,parameter in model.named_parameters():
        if name in model._staged_original_trainable:
            parameter.requires_grad_(model._staged_original_trainable[name])
    adapting = phase == 'reconstruction_adaptation'
    model.reconstruction_adaptation = adapting
    model.frozen_vggt.set_reconstruction_adaptation(adapting)
    for module in (model.understanding,model.panoptic):
        if adapting: module.requires_grad_(False)
        module.to('cpu' if adapting else device)
        module.train(not adapting)
    model.train()
    if not adapting: return build_optimizer(model)
    from scripts.posefree_cpu_adamw import CPUOffloadAdamW
    groups=[]
    for family,peak in (('vggt',1e-6),('reconstruction',1e-5),('memory_adapter',1e-4)):
        def selected(name):
            if family=='vggt': return name.startswith('frozen_vggt.')
            if family=='memory_adapter': return name.startswith('vggt_memory_adapter.')
            return not name.startswith(('frozen_vggt.','vggt_memory_adapter.'))
        for decay in (False,True):
            named=[(n,p) for n,p in sorted(model.named_parameters()) if p.requires_grad and selected(n)
                   and (family!='reconstruction' and p.ndim>1 and not n.endswith('.bias') and not getattr(p,'_no_weight_decay',False))==decay]
            if named: groups.append(dict(name=f'{family}_{"decay" if decay else "nodecay"}',
                params=[p for _,p in named],param_names=[n for n,_ in named],peak_lr=peak,lr=peak,weight_decay=.05 if decay else 0.))
    identifiers=[id(p) for group in groups for p in group['params']]
    assert len(identifiers)==len(set(identifiers)) and set(identifiers)=={id(p) for p in model.parameters() if p.requires_grad}
    return CPUOffloadAdamW(groups)


def restore_adapted_vggt(model, path, expected_sha, expected_updates=2*UPDATES_PER_EPOCH):
    if sha256(path) != expected_sha: raise RuntimeError('adapted VGGT artifact SHA mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('recipe')!=STAGED_RECIPE or payload.get('adaptation_updates')!=expected_updates:
        raise RuntimeError('adapted VGGT artifact recipe/clock mismatch')
    if payload.get('base_hf_revision')!=model.frozen_vggt.source_identity['revision']:
        raise RuntimeError('adapted VGGT base revision mismatch')
    model.frozen_vggt.model.load_state_dict(payload['model'],strict=True)
    model.frozen_vggt.source_identity['derived_asset']={'path':str(path),'sha256':expected_sha,'recipe':STAGED_RECIPE}


def restore_staged_phase_state(model, optimizer, payload, phase, rank):
    if phase=='reconstruction_adaptation':
        model.frozen_vggt.model.load_state_dict(payload['vggt_model'],strict=True)
    if payload['phase']==phase:
        optimizer.load_state_dict(payload['optimizer'])
    # Across the boundary the optimizer is fresh, but RNG must still continue.
    restore_rank_rng(payload['rank_rng'][rank])


def run_staged_training(*, run_dir, hf_revision, mode='train', resume=False,
                        artifact_manifest=VGGT_ARTIFACT_MANIFEST,training_world_size=8,
                        adapter_understanding_scale=1., boundary_checkpoint=None):
    """Execute the authorized two-stage recipe using the existing data/step runtime."""
    import torch.distributed as dist
    from scripts import object_locus_v3_set_runtime as provider_runtime
    world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if mode not in ('train','single_smoke','eight_smoke','four_smoke'): raise ValueError(mode)
    smoke=mode!='train'
    config=staged_training_configuration(training_world_size,adapter_understanding_scale)
    recipe=config["recipe"]
    if boundary_checkpoint is not None and (resume or smoke or config!=staged_training_configuration(4,.01)):
        raise RuntimeError("boundary reuse is fresh formal four-rank joint training only")
    expected_world={'single_smoke':1,'eight_smoke':8,'four_smoke':4,'train':training_world_size}[mode]
    if world!=expected_world or (mode in ('four_smoke','eight_smoke') and world!=training_world_size):
        raise RuntimeError('staged recipe world-size mismatch')
    accumulation=1 if mode=='single_smoke' else GLOBAL_BATCH//world
    node=socket.gethostname().split('.')[0]
    if node not in STAGED_NODE_GPU: raise RuntimeError('staged recipe node outside authorized pool')
    if training_world_size==4 and node!='3dimage-13': raise RuntimeError('four-card staged training requires node13')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if STAGED_NODE_GPU[node] not in torch.cuda.get_device_name(local):
        raise RuntimeError('staged recipe GPU does not match authorized node type')
    hardware={'node':node,'gpu_models':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    if world>1: dist.init_process_group('nccl')
    rank=dist.get_rank() if world>1 else 0
    execution_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    manifest,_,windows=load_manifest();manifest_digest=sha256(MANIFEST)
    latest=run_dir/'checkpoint_latest.pt'
    if not resume and run_dir.exists() and any(run_dir.iterdir()): raise RuntimeError('staged output is not empty')
    if rank==0: run_dir.mkdir(parents=True,exist_ok=True)
    if world>1: dist.barrier()
    seed_everything(42);os.environ['VGGT_HF_REVISION']=hf_revision
    model,opt,transfer,provenance=build_model(artifact_manifest=artifact_manifest,preserve_fp32_aggregator=True)
    model.to(device);seed_everything(42+rank)
    reused_updates=0;boundary_identity=None
    boundary_payload=None
    if boundary_checkpoint is not None:
        boundary_checkpoint=Path(boundary_checkpoint)
        if sha256(boundary_checkpoint)!=STAGED_U001_BOUNDARY_SHA:
            raise RuntimeError('reconstruction boundary checkpoint SHA mismatch')
        boundary_payload=torch.load(boundary_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        validate_staged_boundary(boundary_payload,config,hf_revision,manifest_digest)
        restore_model_state_strict(model,boundary_payload['model'])
        reused_updates=2*UPDATES_PER_EPOCH
        boundary_identity=dict(path=str(boundary_checkpoint.resolve()),sha256=STAGED_U001_BOUNDARY_SHA,
            execution_git_sha=boundary_payload['git_sha'],reused_updates=reused_updates,
            reused_exposures=reused_updates*GLOBAL_BATCH,phase_optimizer_restored=False,
            joint_phase_clock_origin=0,rank_rng_restore_required=True)

    if rank==0:
        manifest_name=f'resume_manifest_{os.environ.get("SLURM_JOB_ID","manual")}.json' if resume else 'run_manifest.json'
        _write_json_atomic(run_dir/manifest_name,dict(status='SMOKE_STARTED' if smoke else 'TRAINING_STARTED',
            execution_git_sha=execution_sha,recipe=recipe,training_configuration=config,source_checkpoint=provenance,
            vggt_source=model.frozen_vggt.source_identity,manifest_sha256=manifest_digest,mode=mode,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),hardware=hardware,
            optimizer_clock_origin=0,smoke_clock_in_formal_training=False,
            reused_reconstruction_boundary=boundary_identity,adapter_understanding_scale=adapter_understanding_scale))
    completed=reused_updates;total_counts=np.zeros(EXPECTED_WINDOWS,dtype=np.int64);adapted_identity=None;payload=boundary_payload
    boundary_payload=None
    if payload is not None:
        total_counts=np.asarray(payload['rank_window_exposure_counts'][rank],dtype=np.int64).copy()
        adapted_identity=dict(path=str(STAGED_U001_BOUNDARY.parent/'adapted_vggt.pt'),
            sha256=CONTINUATION_VGGT_SHA,recipe=STAGED_RECIPE)
    if resume:
        payload=torch.load(latest,map_location='cpu',weights_only=False,mmap=True)
        for key,value in {'recipe':recipe,'config':config,'manifest_sha256':manifest_digest,
                          'source_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,'vggt_revision':hf_revision,'world_size':world}.items():
            if payload.get(key)!=value: raise RuntimeError(f'staged resume identity mismatch: {key}')
        boundary_identity=payload.get('reused_reconstruction_boundary');reused_updates=(boundary_identity or {}).get('reused_updates',0)
        completed=payload['completed_updates'];total_counts=np.asarray(payload['rank_window_exposure_counts'][rank])
        restore_model_state_strict(model,payload['model']);adapted_identity=payload.get('adapted_vggt_identity')
        if completed>=2*UPDATES_PER_EPOCH and adapted_identity is None:
            # An epoch-2 adaptation checkpoint precedes writing the derived asset.
            # Recover from its exact FP32 tensors, preserving an existing artifact.
            identity=[None]
            if rank==0:
                path=run_dir/f'adapted_vggt_recovered_{os.environ.get("SLURM_JOB_ID","manual")}.pt'
                _atomic_save(dict(recipe=STAGED_RECIPE,adaptation_updates=2*UPDATES_PER_EPOCH,
                    base_hf_revision=hf_revision,base_artifact=model.frozen_vggt.source_identity,
                    model=payload['vggt_model']),path)
                identity[0]={'path':str(path.resolve()),'sha256':sha256(path),'recipe':STAGED_RECIPE}
            dist.broadcast_object_list(identity,src=0);adapted_identity=identity[0]
    phase_rows=[]
    for phase_idx,(phase,phase_epochs) in enumerate((('reconstruction_adaptation',2),('frozen_joint',4))):
        phase_total=phase_epochs*UPDATES_PER_EPOCH
        offset=0 if phase_idx==0 else 2*UPDATES_PER_EPOCH
        limit=2 if smoke else phase_total
        if completed>=offset+limit: continue
        optimizer=configure_staged_phase(model,phase,device)
        initial_rng=None
        if boundary_identity is not None and phase_idx==1 and completed==offset:
            initial_rng=payload['rank_rng'][rank] if payload is not None else None
            if initial_rng is None or optimizer.state:
                raise RuntimeError('reused boundary requires original RNG and a fresh joint optimizer')
        if payload is not None:
            restore_staged_phase_state(model,optimizer,payload,phase,rank)
            payload=None
        if phase_idx==1:
            restore_adapted_vggt(model,Path(adapted_identity['path']),adapted_identity['sha256'],
                                 expected_updates=2 if smoke else 2*UPDATES_PER_EPOCH)
        if initial_rng is not None:
            actual_rng=capture_rank_rng()
            if not torch.equal(actual_rng['torch_cpu'],initial_rng['torch_cpu']) or len(actual_rng['torch_cuda'])!=len(initial_rng['torch_cuda']) or any(
                    not torch.equal(a,b) for a,b in zip(actual_rng['torch_cuda'],initial_rng['torch_cuda'])):
                raise RuntimeError('reused boundary RNG restoration mismatch')
            _write_json_atomic(run_dir/f'joint_initialization_rank{rank}.json',dict(status='EXACT_BOUNDARY_RESTORED',
                rank=rank,source_sha256=STAGED_U001_BOUNDARY_SHA,rank_torch_rng_restored=True,
                optimizer_state_empty=True,joint_phase_completed_updates=0,new_exposures=0,
                adapter_understanding_scale=adapter_understanding_scale))
            del initial_rng,actual_rng
        monitor=GeometryMonitor(run_dir,rank,mode+'_'+phase)
        probe=next(p for n,p in model.frozen_vggt.named_parameters() if 'frame_blocks.0.attn.qkv.weight' in n)
        probe_before=probe.detach().clone();frozen_versions={id(p):p._version for p in model.frozen_vggt.parameters()}
        phase_start=max(0,completed-offset)
        phase_started=time.monotonic()
        for phase_update in range(phase_start,limit):
            absolute_update=offset+phase_update
            epoch,epoch_update=divmod(absolute_update,UPDATES_PER_EPOCH)
            indices=(int(epoch_order(0)[0]),) if mode=='single_smoke' else rank_microbatch_indices(epoch,epoch_update,rank,world_size=world)
            batches=[];error=None
            try: batches=[provider_runtime.build_batch(opt,windows[index],device) for index in indices]
            except Exception as exc: error=exc
            synchronized_error(error,device,'staged_batch',absolute_update)
            row=train_microbatch_window(model,optimizer,batches,phase_update,
                base_exposure=phase_update*world*accumulation,world_size=world,accumulation=accumulation,
                monitor=monitor,window_metadata=[window_identity(i,windows[i]) for i in indices],
                stage=mode+'_'+phase,total_updates=phase_total,reconstruction_adaptation=phase_idx==0,
                adapter_understanding_scale=adapter_understanding_scale)
            row.update(phase=phase,phase_update=phase_update+1,window_indices=list(indices),
                       adapter_understanding_scale=adapter_understanding_scale,
                       reused_reconstruction_updates=reused_updates,
                       new_completed_updates=absolute_update+1-reused_updates if not smoke else 0,
                       new_completed_exposures=(absolute_update+1-reused_updates)*GLOBAL_BATCH if not smoke else 0,
                       absolute_update=absolute_update+1 if not smoke else None,
                       absolute_new_exposures=(absolute_update+1)*GLOBAL_BATCH if not smoke else None)
            if phase=='reconstruction_adaptation':
                aa=[p for p in model.frozen_vggt.parameters() if p.requires_grad]
                if not aa or not any(p.grad is not None and p.grad.norm()>0 for p in aa): raise RuntimeError('VGGT adaptation gradient missing')
            elif any(p.grad is not None for p in model.frozen_vggt.parameters()): raise RuntimeError('frozen joint VGGT gradient present')
            if smoke and phase_idx==1 and phase_update==1:
                for prefix in ('understanding.','vggt_memory_adapter.'):
                    if not any(p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters() if n.startswith(prefix)):
                        raise RuntimeError(f'joint smoke gradient path missing: {prefix}')
            if phase=='frozen_joint' and any(p._version!=frozen_versions[id(p)] for p in model.frozen_vggt.parameters()):
                raise RuntimeError('frozen joint mutated adapted VGGT parameters')
            for index in indices: total_counts[index]+=1
            completed=absolute_update+1
            if rank==0 and ((phase_update+1)%20==0 or smoke or phase_update+1==limit):
                with (run_dir/'training_rank0.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
                if not smoke:
                    elapsed=time.monotonic()-phase_started
                    eta=elapsed/(phase_update-phase_start+1)*(phase_total-phase_update-1)
                    _write_json_atomic(run_dir/'progress.json',dict(phase=phase,phase_update=phase_update+1,
                        phase_total_updates=phase_total,completed_updates=completed,completed_exposures=completed*GLOBAL_BATCH,
                        reused_updates=reused_updates,reused_exposures=reused_updates*GLOBAL_BATCH,
                        new_completed_updates=completed-reused_updates,new_completed_exposures=(completed-reused_updates)*GLOBAL_BATCH,
                        total_updates=6*UPDATES_PER_EPOCH,estimated_phase_remaining_seconds=eta,
                        updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            if (phase_update+1)%20==0: monitor.write_summary(phase_update+1)
            startup_due=phase_update+1==20 or (resume and phase_update==phase_start)
            if smoke or startup_due:
                local_row={'rank':rank,**row,'geometry_monitor':monitor.summary(),
                           'vggt_probe_change_norm':float((probe.detach()-probe_before).norm())}
                gathered=[None]*world
                if world>1: dist.all_gather_object(gathered,local_row)
                else: gathered=[local_row]
                if rank==0:
                    if mode in ('eight_smoke','four_smoke') and phase_idx==0 and phase_update==1:
                        if not any(6923 in r['window_indices'] for r in gathered):
                            raise RuntimeError('distributed smoke did not visit mandatory window 6923')
                    if not smoke:
                        if any(r['phase_update']!=phase_update+1 for r in gathered): raise RuntimeError('staged startup clocks disagree')
                        if phase_idx==0 and any(r['vggt_probe_change_norm']<=0 for r in gathered): raise RuntimeError('VGGT weights did not update')
                        _write_json_atomic(run_dir/f'startup_confirmation_{phase}.json',dict(status='TRAINING_CONFIRMED',
                            recipe=recipe,phase=phase,confirmed_phase_updates=phase_update+1,confirmed_phase_exposures=(phase_update+1)*8,
                            execution_git_sha=execution_sha,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                            geometry_quality_policy='monitor_v1',rank_rows=gathered,training_configuration=config,hardware=hardware))
                    else: phase_rows.append(gathered)
            save_due=not smoke and (phase_update+1==20 or completed%UPDATES_PER_EPOCH==0)
            if save_due:
                rank_rng=[None]*world;rank_counts=[None]*world
                dist.all_gather_object(rank_rng,capture_rank_rng());dist.all_gather_object(rank_counts,total_counts.tolist())
                counts=np.sum(np.asarray(rank_counts),axis=0);expected=np.zeros(EXPECTED_WINDOWS,dtype=np.int64)
                full_epochs,position=divmod(completed,UPDATES_PER_EPOCH)
                for e in range(full_epochs): np.add.at(expected,epoch_order(e),1)
                np.add.at(expected,epoch_order(full_epochs)[:position*GLOBAL_BATCH],1)
                if not np.array_equal(counts,expected): raise RuntimeError('staged full exposure accounting mismatch')
                if rank==0:
                    saved=dict(recipe=recipe,reused_reconstruction_boundary=boundary_identity,config=config,model=checkpoint_model_state(model),optimizer=optimizer.state_dict(),
                        phase=phase,phase_completed_updates=phase_update+1,completed_updates=completed,completed_exposures=completed*GLOBAL_BATCH,
                        epoch=completed//UPDATES_PER_EPOCH,next_position=completed%UPDATES_PER_EPOCH,
                        rank_rng=rank_rng,rank_window_exposure_counts=rank_counts,window_exposure_counts=counts.tolist(),
                        manifest_sha256=manifest_digest,source_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,vggt_revision=hf_revision,
                        world_size=world,git_sha=execution_sha,geometry_quality_policy='monitor_v1',adapted_vggt_identity=adapted_identity,
                        total_updates=6*UPDATES_PER_EPOCH,
                        vggt_artifact_identity={k:model.frozen_vggt.source_identity.get(k) for k in ('repository','model_id','revision','files','loaded_subtrees','loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')})
                    if phase_idx==0: saved['vggt_model']={k:v.detach().cpu() for k,v in model.frozen_vggt.model.state_dict().items()}
                    _atomic_save(saved,latest)
                    if completed%UPDATES_PER_EPOCH==0: _atomic_save(saved,run_dir/f'checkpoint_{phase}_epoch_{(phase_update+1)//UPDATES_PER_EPOCH:02d}.pt')
                    _write_json_atomic(run_dir/'progress.json',dict(phase=phase,completed_updates=completed,completed_exposures=completed*GLOBAL_BATCH,
                        phase_update=phase_update+1,phase_total_updates=phase_total,total_updates=6*UPDATES_PER_EPOCH,
                        reused_updates=reused_updates,reused_exposures=reused_updates*GLOBAL_BATCH,
                        new_completed_updates=completed-reused_updates,new_completed_exposures=(completed-reused_updates)*GLOBAL_BATCH,
                        estimated_phase_remaining_seconds=(time.monotonic()-phase_started)/(phase_update-phase_start+1)*(phase_total-phase_update-1),
                        updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                dist.barrier()
        change=float((probe.detach()-probe_before).norm())
        if phase_idx==0 and change<=0: raise RuntimeError('adaptation produced no VGGT change')
        optimizer.zero_grad(set_to_none=True);del optimizer,probe_before,probe
        if phase_idx==0:
            identity=[None]
            if rank==0:
                path=run_dir/('adapted_vggt_smoke.pt' if smoke else 'adapted_vggt.pt')
                _atomic_save(dict(recipe=STAGED_RECIPE,adaptation_updates=2 if smoke else phase_total,
                    base_hf_revision=hf_revision,base_artifact=model.frozen_vggt.source_identity,
                    model={k:v.detach().cpu() for k,v in model.frozen_vggt.model.state_dict().items()}),path)
                identity[0]={'path':str(path.resolve()),'sha256':sha256(path),'recipe':STAGED_RECIPE}
            if world>1: dist.broadcast_object_list(identity,src=0)
            adapted_identity=identity[0];model.frozen_vggt.source_identity['derived_asset']=adapted_identity
            # Smoke has two isolated updates per phase, not 2086 adaptation updates.
            completed=2*UPDATES_PER_EPOCH
        torch.cuda.empty_cache()
    if rank==0:
        if smoke:
            _write_json_atomic(run_dir/'smoke_report.json',dict(status='GPU_SMOKE_COMPLETED',mode=mode,recipe=recipe,
                world_size=world,accumulation=accumulation,updates_per_phase=2,
                exposures_per_phase=2*world*accumulation,rank_rows=phase_rows,
                execution_git_sha=execution_sha,manifest_sha256=manifest_digest,source_checkpoint=provenance,
                geometry_quality_policy='monitor_v1',formal_updates=0,training_configuration=config,
                understanding_and_vggt_gradient_paths_checked=True,hardware=hardware))
        else:
            _write_json_atomic(run_dir/'COMPLETE.json',dict(status='TRAINING_COMPLETED',recipe=recipe,
                reused_reconstruction_boundary=boundary_identity,
                new_completed_updates=completed-reused_updates,new_completed_exposures=(completed-reused_updates)*GLOBAL_BATCH,
                completed_updates=completed,completed_exposures=completed*GLOBAL_BATCH,stage_epochs=config['stage_epochs'],
                execution_git_sha=execution_sha,adapted_vggt_identity=adapted_identity))
    if world>1: dist.barrier();dist.destroy_process_group()
    return {'status':'GPU_SMOKE_COMPLETED' if smoke else 'TRAINING_COMPLETED',
            'completed_updates':4 if smoke else completed,'formal_updates':0 if smoke else completed-reused_updates}


CONTINUATION_RECIPE = 'vggt_joint_continuation_ab2_v1'
CONTINUATION_BATCH16_RECIPE = 'vggt_joint_continuation_ab2_batch16_v1'
CONTINUATION_SOURCE = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_frozen_joint_epoch_04.pt')
CONTINUATION_SOURCE_SHA = '01d49b092c8dfc2e02f15b5216e5d6ed1cfd88fae2cf693ee69334e45a9410ba'
CONTINUATION_VGGT_SHA = 'c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace'


def continuation_configuration(branch, world_size=4, global_batch=8, gpu_model='RTX3090'):
    if branch not in ('frozen', 'joint_aa'): raise ValueError(branch)
    if world_size not in (4,8): raise ValueError(world_size)
    if global_batch not in (8,16) or (global_batch==16 and world_size!=8): raise ValueError('batch16 requires eight ranks')
    if gpu_model not in ('RTX3090','RTX4090'): raise ValueError(gpu_model)
    if gpu_model=='RTX4090' and (branch,world_size,global_batch)!=('frozen',8,16):
        raise ValueError('4090 migration is authorized only for the eight-rank batch16 frozen arm')
    config = training_configuration()
    config.update(recipe=CONTINUATION_RECIPE, branch=branch, world_size=world_size, accumulation=8//world_size,
        epochs=2, total_updates=2*UPDATES_PER_EPOCH, total_exposures=2*WINDOWS_PER_EPOCH,
        source_checkpoint_sha256=CONTINUATION_SOURCE_SHA, source_adapted_vggt_sha256=CONTINUATION_VGGT_SHA,
        source_completed_updates=6*UPDATES_PER_EPOCH, source_completed_exposures=6*WINDOWS_PER_EPOCH,
        sampler_epoch_offset=6, optimizer='fresh native FP32 CPU-offload AdamW for both branches',
        precision='FP32 AA parameters in BOTH branches, BF16 autocast; DINO BF16; heads/decoder/loss FP32',
        activation_storage='saved tensors and accumulated gradients offloaded to CPU in BOTH branches',
        understanding_weight='continue source exposure; saturated at1', beta='continue source exposure; saturated at0.1',
        lr_peak={'reconstruction':1e-5,'understanding':1e-5,'object_memory':1e-4,'vggt_aa':1e-6},
        aa_gc='g_rec + 0.01*g_under', aa_trainable=branch=='joint_aa',
        aa_frozen=['aggregator.patch_embed','aggregator.camera_token','aggregator.register_token','camera_head','depth_head'])
    if world_size==8: config['node']='3dimage-11,3dimage-13'
    if global_batch==16:
        config.update(recipe=CONTINUATION_BATCH16_RECIPE,global_batch=16,accumulation=2,
            updates_per_epoch=math.ceil(WINDOWS_PER_EPOCH/16),total_updates=2*math.ceil(WINDOWS_PER_EPOCH/16),
            warmup_updates=100,tail_global_batch=8,
            lr_clock='unchanged 200-step warmup/cosine on eight-sample-equivalent exposure clock; peaks unchanged',
            physical_batch='eight ranks x microbatch1 x accumulation2; accumulation1 for the final eight windows each epoch')
    if gpu_model=='RTX4090': config.update(node='3dimage-18',gpu_model=gpu_model)
    return config


def continuation_clock(completed, global_batch=8):
    """Return epoch, next update position and exact consumed continuation windows."""
    if global_batch not in (8,16) or completed<0: raise ValueError('invalid continuation clock')
    epoch,position=divmod(completed,math.ceil(WINDOWS_PER_EPOCH/global_batch))
    return epoch,position,epoch*WINDOWS_PER_EPOCH+position*global_batch


def continuation_rank_indices(epoch, update, rank, *, world_size=8, global_batch=8):
    if global_batch==8: return rank_microbatch_indices(epoch,update,rank,world_size=world_size)
    if global_batch!=16 or world_size!=8 or not 0<=rank<8 or not 0<=update<522: raise ValueError('invalid batch16 sampler')
    order=epoch_order(epoch);size=min(16,WINDOWS_PER_EPOCH-update*16)//8
    start=update*16+rank*size
    return tuple(int(i) for i in order[start:start+size])


def reshard_continuation_counts(saved, start_epoch, completed, world_size):
    """Reconstruct historical sampler ownership; retain exactly the same totals."""
    saved=np.asarray(saved,dtype=np.int64)
    old_world=len(saved)
    if old_world==world_size: return saved.copy()
    if (old_world,world_size)!=(4,8): raise RuntimeError('unsupported continuation reshard')
    result=np.zeros((8,EXPECTED_WINDOWS),dtype=np.int64)
    full,partial=divmod(completed,UPDATES_PER_EPOCH)
    for offset in range(full+1):
        rows=UPDATES_PER_EPOCH if offset<full else partial
        order=epoch_order(start_epoch+offset)[:rows*8].reshape(-1,8)
        for rank in range(8): np.add.at(result[rank],order[:,rank],1)
    if not np.array_equal(result.reshape(4,2,EXPECTED_WINDOWS).sum(axis=1),saved):
        raise RuntimeError('historical sampler ownership does not match saved counts')
    return result


def continuation_rank_rng(states, rank, world_size):
    """4→8 preserves each parent's stream on its even child, seeds the odd child.

    Physical rank changes do not promise bitwise stochastic equivalence. CUDA
    states refer to old local GPU ordinals, so map the actual parent's device.
    """
    if len(states)==world_size: return states[rank]
    if (len(states),world_size)!=(4,8): raise RuntimeError('unsupported RNG reshard')
    parent=rank//2;state=dict(states[parent])
    if rank%2:
        seed=int.from_bytes(hashlib.sha256(state['torch_cpu'].numpy().tobytes()+
            f'continuation-4to8-child-{rank}'.encode()).digest()[:8],'little')%(2**63)
        state.update(python=random.Random(seed).getstate(),numpy=np.random.RandomState(seed%(2**32)).get_state(),
            torch_cpu=torch.Generator().manual_seed(seed).get_state())
        cuda=torch.Generator(device='cuda').manual_seed(seed).get_state() if torch.cuda.is_available() else None
    else:
        cuda=state['torch_cuda'][parent] if state.get('torch_cuda') else None
    state['torch_cuda']=[cuda.clone() for _ in range(world_size)] if cuda is not None else []
    return state


def configure_joint_continuation(model, branch, device):
    from scripts.posefree_cpu_adamw import CPUOffloadAdamW
    # Restore the original joint trainable flags and exactly its optimizer groups.
    native = configure_staged_phase(model, 'frozen_joint', device)
    groups = native.param_groups
    del native
    model.reconstruction_adaptation = False  # Understanding/feedback stays enabled in BOTH arms.
    model.frozen_vggt.set_reconstruction_adaptation(branch=='joint_aa', keep_fp32=True)
    if branch=='joint_aa':
        for decay in (False,True):
            named=[(n,p) for n,p in sorted(model.named_parameters()) if n.startswith('frozen_vggt.') and p.requires_grad
                   and (p.ndim>1 and not n.endswith('.bias') and not getattr(p,'_no_weight_decay',False))==decay]
            if named: groups.append(dict(name=f'vggt_aa_{"decay" if decay else "nodecay"}',
                params=[p for _,p in named],param_names=[n for n,_ in named],peak_lr=1e-6,lr=1e-6,
                weight_decay=.05 if decay else 0.))
    ids=[id(p) for group in groups for p in group['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    model.train()
    return CPUOffloadAdamW(groups)


def run_joint_continuation(*, run_dir, hf_revision, branch, mode='train', resume=False,
                          artifact_manifest=VGGT_ARTIFACT_MANIFEST, training_world_size=4, global_batch=8, gpu_model='RTX3090'):
    """Two authorized continuation arms, reusing the original sampler/data/GC/step."""
    import torch.distributed as dist
    from scripts import object_locus_v3_set_runtime as provider_runtime
    config=continuation_configuration(branch,training_world_size,global_batch,gpu_model);recipe=config['recipe']
    if mode not in ('train','four_smoke','eight_smoke'): raise ValueError(mode)
    smoke=mode!='train';world=int(os.environ.get('WORLD_SIZE','1'))
    allowed_nodes=('3dimage-13',) if training_world_size==4 else ('3dimage-11','3dimage-13')
    if gpu_model=='RTX4090': allowed_nodes=('3dimage-18',)
    if world!=training_world_size or socket.gethostname().split('.')[0] not in allowed_nodes:
        raise RuntimeError('continuation rank count or authorized node mismatch')
    if smoke and mode!=('four_smoke' if world==4 else 'eight_smoke'): raise RuntimeError('smoke world mismatch')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    if gpu_model.removeprefix('RTX') not in torch.cuda.get_device_name(local): raise RuntimeError('continuation GPU mismatch')
    dist.init_process_group('nccl');rank=dist.get_rank()
    if not resume and run_dir.exists() and any(run_dir.iterdir()): raise RuntimeError('continuation output is not empty')
    if rank==0: run_dir.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    execution_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha256(CONTINUATION_SOURCE)!=CONTINUATION_SOURCE_SHA: raise RuntimeError('continuation source hash mismatch')
    source=torch.load(CONTINUATION_SOURCE,map_location='cpu',weights_only=False,mmap=True)
    assert source['recipe']==STAGED_RECIPE and source['config']==staged_training_configuration(4)
    assert (source['epoch'],source['completed_updates'],source['completed_exposures'])==(6,6258,50064)
    assert source['phase']=='frozen_joint' and source['phase_completed_updates']==4*UPDATES_PER_EPOCH
    assert source['adapted_vggt_identity']['sha256']==CONTINUATION_VGGT_SHA and len(source['rank_rng'])==4
    manifest,_,windows=load_manifest();manifest_digest=sha256(MANIFEST)
    assert source['manifest_sha256']==manifest_digest and source['vggt_revision']==hf_revision
    seed_everything(42);os.environ['VGGT_HF_REVISION']=hf_revision
    model,opt,_,_=build_model(artifact_manifest=artifact_manifest,preserve_fp32_aggregator=True)
    model.to(device);restore_model_state_strict(model,source['model'])
    optimizer=configure_joint_continuation(model,branch,device)
    # Restore AFTER phase configuration: converting a frozen BF16 model back to
    # FP32 must not quantize either branch's common FP32 adaptation source.
    source_asset=CONTINUATION_SOURCE.parent/'adapted_vggt.pt'
    restore_adapted_vggt(model,source_asset,CONTINUATION_VGGT_SHA)
    # CPU masters were allocated before the strict source reload.
    for parameter,master in optimizer.pairs:
        with torch.no_grad(): master.copy_(parameter.detach().cpu())
    adapted_identity=dict(path=str(source_asset.resolve()),sha256=CONTINUATION_VGGT_SHA,recipe=STAGED_RECIPE)
    restore_rank_rng(continuation_rank_rng(source['rank_rng'],rank,world))
    source_counts=reshard_continuation_counts(source['rank_window_exposure_counts'],0,6258,world)
    base_counts=source_counts.sum(axis=0)
    identity={k:model.frozen_vggt.source_identity.get(k) for k in ('repository','model_id','revision','files','loaded_subtrees','loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')}
    del source
    completed=0;counts=np.zeros(EXPECTED_WINDOWS,dtype=np.int64)
    if resume:
        payload=torch.load(run_dir/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
        previous_world=payload['world_size']
        if previous_world!=world and (previous_world,world)!=(4,8): raise RuntimeError('unsupported rank migration')
        for key,value in {'recipe':recipe,'config':continuation_configuration(branch,previous_world,global_batch,gpu_model),'branch':branch,
                          'manifest_sha256':manifest_digest,'continuation_source_sha256':CONTINUATION_SOURCE_SHA,
                          'vggt_revision':hf_revision,'vggt_artifact_identity':identity}.items():
            if payload.get(key)!=value: raise RuntimeError(f'continuation resume mismatch: {key}')
        completed=payload['continuation_completed_updates']
        restore_model_state_strict(model,payload['model'])
        if branch=='joint_aa': model.frozen_vggt.model.load_state_dict(payload['vggt_model'],strict=True)
        optimizer.load_state_dict(payload['optimizer']);restore_rank_rng(continuation_rank_rng(payload['rank_rng'],rank,world))
        counts=reshard_continuation_counts(payload['continuation_rank_window_exposure_counts'],6,completed,world)[rank]
        adapted_identity=payload['adapted_vggt_identity'];del payload
    hardware=dict(node=socket.gethostname().split('.')[0],gpu_models=[torch.cuda.get_device_name(i) for i in range(world)])
    if rank==0:
        _write_json_atomic(run_dir/(f'resume_manifest_{os.environ.get("SLURM_JOB_ID")}.json' if resume else 'run_manifest.json'),
            dict(status='SMOKE_STARTED' if smoke else 'TRAINING_STARTED',recipe=recipe,branch=branch,
                 execution_git_sha=execution_sha,training_configuration=config,continuation_source_sha256=CONTINUATION_SOURCE_SHA,
                 manifest_sha256=manifest_digest,hardware=hardware,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                 restored_common_source_rank_rng=world==4,smoke_clock_in_formal_training=False,
                 resumed_continuation_update=completed,optimizer_state_restored=resume,
                 rng_policy='native restore' if world==4 else '4-to-8 parent-even stream plus deterministic independent odd child; no bitwise equivalence claim'))
        if resume and not smoke:
            resumed_exposures=continuation_clock(completed,global_batch)[2]
            _write_json_atomic(run_dir/'progress.json',dict(phase='joint_continuation',branch=branch,
                completed_updates=6258+completed,completed_exposures=50064+resumed_exposures,
                continuation_completed_updates=completed,continuation_completed_exposures=resumed_exposures,
                phase_update=completed,phase_total_updates=config['total_updates'],total_updates=6258+config['total_updates'],
                resumed_from_checkpoint=True,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    probe=next(p for n,p in model.frozen_vggt.named_parameters() if 'frame_blocks.0.attn.qkv.weight' in n)
    before=probe.detach().clone();fixed={n:p._version for n,p in model.frozen_vggt.named_parameters() if not p.requires_grad}
    attempt_stage=mode+'_'+branch+'_job'+os.environ.get('SLURM_JOB_ID','local')
    monitor=GeometryMonitor(run_dir,rank,attempt_stage,updates_per_epoch=config['updates_per_epoch'])
    total=config['total_updates'];limit=2 if smoke else total;start=completed;started=time.monotonic();smoke_rows=[]
    start_exposures=continuation_clock(start,global_batch)[2]
    exposures_after=start_exposures
    for update in range(start,limit):
        epoch,position,exposures_before=continuation_clock(update,global_batch)
        indices=continuation_rank_indices(0 if smoke else 6+epoch,position,rank,world_size=world,global_batch=global_batch)
        exposures_after=continuation_clock(update+1,global_batch)[2]
        error=None;batches=[]
        try: batches=[provider_runtime.build_batch(opt,windows[index],device) for index in indices]
        except Exception as exc: error=exc
        synchronized_error(error,device,'continuation_batch',update)
        row=train_microbatch_window(model,optimizer,batches,update,base_exposure=50064+exposures_before,
            world_size=world,accumulation=len(indices),monitor=monitor,window_metadata=[window_identity(i,windows[i]) for i in indices],
            stage=attempt_stage,total_updates=2*UPDATES_PER_EPOCH,lr_update=exposures_after//8-1,
            cpu_saved_tensors=True,cpu_gradient_accumulation=True,vggt_reconstruction=True)
        completed=update+1
        aa=[p for p in model.frozen_vggt.parameters() if p.requires_grad]
        if branch=='joint_aa' and (not aa or not any(p.grad is not None and p.grad.norm()>0 for p in aa)):
            raise RuntimeError('joint AA gradient missing')
        if any(p._version!=fixed[n] or p.grad is not None for n,p in model.frozen_vggt.named_parameters() if n in fixed):
            raise RuntimeError('continuation mutated a fixed VGGT parameter')
        for index in indices: counts[index]+=1
        row.update(branch=branch,phase='joint_continuation',phase_update=completed,window_indices=list(indices),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),execution_git_sha=execution_sha,
            completed_exposures=exposures_after,continuation_completed_updates=completed,continuation_completed_exposures=exposures_after,
            absolute_update=6258+completed,absolute_new_exposures=50064+exposures_after,
            effective_global_batch=exposures_after-exposures_before)
        if rank==0 and (completed%20==0 or smoke or completed==limit):
            with (run_dir/'training_rank0.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
            _write_json_atomic(run_dir/'progress.json',dict(phase='joint_continuation',branch=branch,
                completed_updates=6258+completed,completed_exposures=50064+exposures_after,
                continuation_completed_updates=completed,continuation_completed_exposures=exposures_after,
                total_updates=6258+total,phase_update=completed,phase_total_updates=total,
                estimated_phase_remaining_seconds=(time.monotonic()-started)/(completed-start)*(total-completed),
                updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        if completed%20==0: monitor.write_summary(completed)
        startup_due=completed==start+20
        if smoke or startup_due:
            for prefix in ('understanding.','vggt_memory_adapter.'):
                if not any(p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters() if n.startswith(prefix)):
                    raise RuntimeError(f'continuation joint gradient missing: {prefix}')
            gathered=[None]*world
            dist.all_gather_object(gathered,dict(rank=rank,**row,geometry_monitor=monitor.summary(),
                vggt_probe_change_norm=float((probe.detach()-before).norm()),fixed_vggt_parameters_unchanged=True))
            if rank==0:
                assert all(r['phase_update']==completed for r in gathered)
                assert all((r['vggt_probe_change_norm']>0 if branch=='joint_aa' else r['vggt_probe_change_norm']==0) for r in gathered)
                if smoke:
                    smoke_rows.append(gathered)
                    if completed==2: assert any(6923 in r['window_indices'] for rows in smoke_rows for r in rows)
                else:
                    _write_json_atomic(run_dir/'startup_confirmation.json',dict(status='TRAINING_CONFIRMED',recipe=recipe,
                        branch=branch,confirmed_phase_updates=completed,confirmed_phase_exposures=exposures_after,
                        confirmed_attempt_updates=completed-start,confirmed_attempt_exposures=exposures_after-start_exposures,
                        rank_rows=gathered,training_configuration=config,execution_git_sha=execution_sha,
                        slurm_job_id=os.environ.get('SLURM_JOB_ID'),geometry_quality_policy='monitor_v1',hardware=hardware))
        save_due=not smoke and (startup_due or completed%(800//global_batch)==0 or completed%config['updates_per_epoch']==0)
        if save_due:
            rngs=[None]*world;rank_counts=[None]*world
            dist.all_gather_object(rngs,capture_rank_rng());dist.all_gather_object(rank_counts,counts.tolist())
            summed=np.sum(np.asarray(rank_counts),axis=0);expected=np.zeros(EXPECTED_WINDOWS,dtype=np.int64)
            full,partial,_=continuation_clock(completed,global_batch)
            for e in range(full): np.add.at(expected,epoch_order(6+e),1)
            np.add.at(expected,epoch_order(6+full)[:partial*global_batch],1)
            assert np.array_equal(summed,expected)
            if rank==0:
                if branch=='joint_aa' and completed==total:
                    path=run_dir/'adapted_vggt_joint_final.pt'
                    _atomic_save(dict(recipe=recipe,branch=branch,continuation_completed_updates=completed,
                        source_adapted_vggt_sha256=CONTINUATION_VGGT_SHA,base_hf_revision=hf_revision,
                        model={k:v.detach().cpu() for k,v in model.frozen_vggt.model.state_dict().items()}),path)
                    adapted_identity=dict(path=str(path.resolve()),sha256=sha256(path),recipe=recipe)
                saved=dict(recipe=recipe,branch=branch,config=config,model=checkpoint_model_state(model),
                    optimizer=optimizer.state_dict(),phase='joint_continuation',phase_completed_updates=completed,
                    continuation_completed_updates=completed,continuation_completed_exposures=exposures_after,
                    completed_updates=6258+completed,completed_exposures=50064+exposures_after,
                    epoch=6+full,next_position=partial,
                    rank_rng=rngs,continuation_rank_window_exposure_counts=rank_counts,
                    rank_window_exposure_counts=(source_counts+np.asarray(rank_counts)).tolist(),
                    window_exposure_counts=(base_counts+summed).tolist(),manifest_sha256=manifest_digest,
                    source_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,continuation_source_sha256=CONTINUATION_SOURCE_SHA,
                    world_size=world,vggt_revision=hf_revision,vggt_artifact_identity=identity,git_sha=execution_sha,
                    geometry_quality_policy='monitor_v1',adapted_vggt_identity=adapted_identity,total_updates=6258+total)
                if branch=='joint_aa': saved['vggt_model']={k:v.detach().cpu() for k,v in model.frozen_vggt.model.state_dict().items()}
                _atomic_save(saved,run_dir/'checkpoint_latest.pt')
                if completed%config['updates_per_epoch']==0: _atomic_save(saved,run_dir/f'checkpoint_continuation_epoch_{full:02d}.pt')
                del saved
            dist.barrier()
    if rank==0:
        if smoke:
            _write_json_atomic(run_dir/'smoke_report.json',dict(status='GPU_SMOKE_COMPLETED',recipe=recipe,
                branch=branch,world_size=world,updates=2,exposures=2*global_batch,formal_updates=0,rank_rows=smoke_rows,
                execution_git_sha=execution_sha,training_configuration=config,geometry_quality_policy='monitor_v1'))
        else:
            _write_json_atomic(run_dir/'COMPLETE.json',dict(status='TRAINING_COMPLETED',recipe=recipe,
                branch=branch,continuation_completed_updates=completed,continuation_completed_exposures=exposures_after,
                completed_updates=6258+completed,completed_exposures=50064+exposures_after,execution_git_sha=execution_sha,
                continuation_source_sha256=CONTINUATION_SOURCE_SHA,adapted_vggt_identity=adapted_identity))
    dist.barrier();dist.destroy_process_group()
    return dict(status='GPU_SMOKE_COMPLETED' if smoke else 'TRAINING_COMPLETED',formal_updates=0 if smoke else completed)


def run_training(*, manifest_path: Path=MANIFEST, checkpoint: Path=CHECKPOINT,
                 run_dir: Path=Path('/space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1_calibration_v2_monitor'),
                 hf_revision: str, resume: bool=False,
                 calibration_report: Path | None=None,
                 single_smoke_report: Path | None=None,
                 eight_smoke_report: Path | None=None):
    """Explicit 8xRTX3090 launcher for the reviewed resource adaptation."""
    import torch.distributed as dist
    if socket.gethostname().split('.')[0] != '3dimage-13':
        raise RuntimeError('formal training is pinned to 3dimage-13')
    world=int(os.environ.get('WORLD_SIZE','1')); local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=WORLD_SIZE: raise RuntimeError(f'formal world size must be {WORLD_SIZE}, got {world}')
    torch.cuda.set_device(local)
    if '3090' not in torch.cuda.get_device_name(local):
        raise RuntimeError(f'RTX3090 required, got {torch.cuda.get_device_name(local)}')
    dist.init_process_group('nccl')
    rank=dist.get_rank();device=torch.device('cuda',local)
    execution_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    latest=run_dir/'checkpoint_latest.pt'
    if resume:
        if not latest.is_file():raise FileNotFoundError(f'--resume requires {latest}')
    elif run_dir.exists() and any(run_dir.iterdir()): raise RuntimeError(f'run directory is not empty: {run_dir}; pass --resume to restore')
    if not resume:
        required_reports=(('window4253',calibration_report,'window4253_context_sim3_v2.json','PASS'),
                          ('single smoke',single_smoke_report,'smoke_report.json','GPU_SMOKE_COMPLETED'),
                          ('eight-card smoke',eight_smoke_report,'smoke_report.json','GPU_SMOKE_COMPLETED'))
        for label,path,filename,status in required_reports:
            if path is None:raise RuntimeError(f'fresh training requires the passed {label} report')
            path=Path(path)
            if path.is_dir():path=path/filename
            if not path.is_file():raise FileNotFoundError(f'required {label} report missing: {path}')
            report=json.loads(path.read_text())
            if report.get('status')!=status:
                raise RuntimeError(f'{label} has not passed: {path} status={report.get("status")!r}')
            if label=='window4253' and report.get('calibration_protocol')!=CALIBRATION_PROTOCOL:
                raise RuntimeError('window 4253 report uses the wrong calibration protocol')
            if label=='single smoke' and (report.get('mode')!='single' or report.get('world_size')!=1):
                raise RuntimeError('single-card smoke report identity mismatch')
            if label=='eight-card smoke' and (report.get('mode')!='eight' or report.get('world_size')!=WORLD_SIZE):
                raise RuntimeError('eight-card smoke report identity mismatch')
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
    single_evidence=validated_smoke(single_smoke_report,'single',1,execution_sha,manifest_digest,model.frozen_vggt.source_identity)
    eight_evidence=validated_smoke(eight_smoke_report,'eight',WORLD_SIZE,execution_sha,manifest_digest,model.frozen_vggt.source_identity)
    from scripts import object_locus_v3_set_runtime as provider_runtime
    if rank==0:
        run_dir.mkdir(parents=True,exist_ok=True)
        if not resume:(run_dir/'run_manifest.json').write_text(json.dumps({
            'status':'TRAINING_STARTED_AFTER_EXPLICIT_FLAG','branch':'object-locus-frozen-vggt-posefree-v1',
            'training_code_sha':execution_sha,'postprocessing_code_sha':execution_sha,
            'source_science_sha':'9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0',
            'manifest_sha256':manifest_digest,'checkpoint_provenance':provenance,
            'transfer_counts':transfer['counts'],'plan':plan_record(),
            'vggt_source':model.frozen_vggt.source_identity,'optimizer':'AdamW',
            'seeds':{'common_model_initialization':42,'per_rank_training_rng':[42+r for r in range(WORLD_SIZE)],'memory_adapter_isolated':31415},
            'optimizer_state_restored':False,'scheduler_state_restored':False,'rng_state_restored':False,
            'calibration_protocol':CALIBRATION_PROTOCOL,'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,
            'training_configuration':training_configuration(),'single_card_smoke':single_evidence,'eight_card_smoke':eight_evidence,
            'evaluation_launched':False},indent=2)+'\n')
        if not resume:
            _write_json_atomic(run_dir/'training_plan.json',{
                'calibration_protocol':CALIBRATION_PROTOCOL,'plan':plan_record(),
                'training_configuration':training_configuration(),
                'manifest_sha256':manifest_digest,'execution_git_sha':execution_sha})
    dist.barrier()
    monitor=GeometryMonitor(run_dir,rank,'formal_training')
    total_counts=np.zeros(EXPECTED_WINDOWS,dtype=np.int64)
    completed=0;start_epoch=0;last_row=None
    if resume:
        payload=torch.load(latest,map_location='cpu',weights_only=False)
        artifact_keys=('repository','model_id','revision','files','loaded_subtrees',
                       'loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')
        current_artifact={k:model.frozen_vggt.source_identity.get(k) for k in artifact_keys}
        required={'manifest_sha256':manifest_digest,'source_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,
                  'vggt_revision':hf_revision,'world_size':WORLD_SIZE,'total_updates':TOTAL_UPDATES,
                  'vggt_artifact_identity':current_artifact}
        for key,value in required.items():
            if payload.get(key)!=value:raise RuntimeError(f'resume provenance mismatch for {key}: {payload.get(key)!r} != {value!r}')
        if payload.get('config')!=training_configuration():
            raise RuntimeError('resume training configuration differs; only code/logging/interface fixes can resume')
        resume_code_sha=payload.get('git_sha')
        restore_model_state_strict(model,payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        completed=int(payload['completed_updates']);start_epoch=int(payload['epoch'])
        if completed!=start_epoch*UPDATES_PER_EPOCH or payload.get('next_position')!=0:
            raise RuntimeError('resume checkpoint is not at the recorded epoch sampler boundary')
        total_counts=np.asarray(payload['rank_window_exposure_counts'][rank],dtype=np.int64)
        rngs=payload['rank_rng']
        if len(rngs)!=WORLD_SIZE:raise RuntimeError('resume rank RNG count mismatch')
        restore_rank_rng(rngs[rank])
        if rank==0:
            with (run_dir/'resume_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'resumed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'checkpoint_code_sha':resume_code_sha,'execution_code_sha':execution_sha,
                    'world_size':WORLD_SIZE,'config':training_configuration()},allow_nan=False)+'\n')
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
            result=train_microbatch_window(model,optimizer,batches,update,base_exposure=update*GLOBAL_BATCH,world_size=WORLD_SIZE,monitor=monitor,
                window_metadata=[window_identity(index,windows[index]) for index in indices])
            for index in indices: total_counts[index]+=1
            completed=update+1;last_row=result
            if completed%20==0:monitor.write_summary(completed)
            if rank==0 and (completed%20==0 or completed==TOTAL_UPDATES):
                with (run_dir/'training_rank0.jsonl').open('a') as stream:
                    stream.write(json.dumps({'epoch':epoch+1,'update':completed,'exposure':completed*GLOBAL_BATCH,
                                             **result},allow_nan=False)+'\n')
            startup_due=(completed==20 or (resume and update==start_epoch*UPDATES_PER_EPOCH))
            if startup_due:
                dist.barrier()
                local_update={'rank':rank,'completed_updates':completed,
                    'completed_exposures':completed*GLOBAL_BATCH,
                    **{key:result[key] for key in ('loss_recon','loss_understanding','understanding_weight',
                        'beta','group_lr','preclip_norm')},
                    'geometry_monitor':monitor.summary(),
                    **{key:result[key] for key in ('allocated','reserved','peak_allocated','peak_reserved') if key in result}}
                rank_updates=[None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(rank_updates,local_update)
                startup_error=None
                if rank==0:
                    try:
                        single=single_evidence
                        eight=eight_evidence
                        if len(rank_updates)!=WORLD_SIZE or not all(row['completed_updates']==completed and row['completed_exposures']==completed*GLOBAL_BATCH for row in rank_updates):
                            raise RuntimeError('startup confirmation rank update accounting mismatch')
                        if sorted(row['rank'] for row in rank_updates)!=list(range(WORLD_SIZE)):
                            raise RuntimeError('startup rank set mismatch')
                        expected_weight,expected_beta=exposure_schedule((completed-1)*GLOBAL_BATCH)
                        for row in rank_updates:
                            if row['understanding_weight']!=expected_weight or row['beta']!=expected_beta:
                                raise RuntimeError('startup exposure weight/beta mismatch')
                            if not all(math.isfinite(row[key]) for key in ('loss_recon','loss_understanding','preclip_norm')):
                                raise FloatingPointError('startup row contains nonfinite training values')
                            for group in optimizer.param_groups:
                                if row['group_lr'][group['name']]!=group['peak_lr']*lr_multiplier(completed-1):
                                    raise RuntimeError('startup learning rate mismatch')
                        confirmation={'status':'TRAINING_CONFIRMED','slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                            'node':socket.gethostname().split('.')[0],
                            'gpu_models':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                            'world_size':WORLD_SIZE,'global_batch':GLOBAL_BATCH,'microbatch_per_rank':MICRO_BATCH,
                            'accumulation':ACCUMULATION,'code_sha':execution_sha,
                            'training_code_sha':execution_sha,'postprocessing_code_sha':execution_sha,
                            'remote_verified_sha':os.environ.get('TASK_CODE_SHA'),
                            'calibration_protocol':CALIBRATION_PROTOCOL,'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,
                            'single_smoke_reused':single['single_smoke_reused'],'single_smoke_reuse_reason':single['reuse_reason'],
                            'geometry_quality_status':'WARNING' if any(row['geometry_monitor']['counts']['WARNING'] for row in rank_updates) else 'OK',
                            'training_configuration':training_configuration(),
                            'loss_gradient_updated_parameters_finite':True,
                            'geometry_warning_evidence_root':str(Path(os.environ.get('POSEFREE_V2_EVIDENCE_DIR',''))/'geometry_warnings'),
                            'vggt_artifact':model.frozen_vggt.source_identity,
                            'single_card_smoke':single,'eight_card_smoke':eight,
                            'confirmed_updates':completed,'confirmed_new_exposures':completed*GLOBAL_BATCH,
                            'all_ranks_completed_same_update':True,'rank_update_rows':rank_updates,
                            'latest_log':str(run_dir/'training_rank0.jsonl'),
                            'logged_row':{'loss_recon':result['loss_recon'],'loss_understanding':result['loss_understanding'],
                                'understanding_weight':result['understanding_weight'],'beta':result['beta'],
                                'group_lr':result['group_lr'],'preclip_norm':result['preclip_norm']},
                            'run_manifest':str(run_dir/'run_manifest.json'),
                            'training_continues':True,'evaluation_started':False}
                        confirmation_text=json.dumps(confirmation,indent=2,allow_nan=False)+'\n'
                        job_id=os.environ.get('SLURM_JOB_ID','manual')
                        for confirmation_path in (run_dir/'startup_confirmation.json',
                                run_dir/f'startup_confirmation_{job_id}.json'):
                            temp=confirmation_path.with_suffix('.json.tmp')
                            temp.write_text(confirmation_text)
                            os.replace(temp,confirmation_path)
                    except Exception as exc:
                        startup_error=exc
                synchronized_error(startup_error,device,'startup_confirmation',update)
                dist.barrier()
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
                     'config':training_configuration(),'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,
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
    if mode not in ('single','eight'):raise ValueError('smoke mode must be single or eight')
    distributed=mode=='eight'
    if socket.gethostname().split('.')[0]!='3dimage-13':raise RuntimeError('real VGGT smoke is pinned to 3dimage-13')
    if not torch.cuda.is_available():raise RuntimeError('real VGGT smoke requires an allocated CUDA device')
    local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    expected=WORLD_SIZE if distributed else 1
    if world!=expected:raise RuntimeError(f'{mode} smoke requires torchrun world size {expected}, got {world}')
    torch.cuda.set_device(local)
    if '3090' not in torch.cuda.get_device_name(local):raise RuntimeError('RTX3090 smoke device required')
    if distributed:dist.init_process_group('nccl')
    rank=dist.get_rank() if distributed else 0
    seed_everything(42);os.environ['VGGT_HF_REVISION']=hf_revision
    model,opt,transfer,provenance=build_model(checkpoint=checkpoint,artifact_manifest=artifact_manifest)
    seed_everything(42+rank)
    model.to(device=f'cuda:{local}',dtype=torch.float32);model.train();model.frozen_vggt.eval()
    optimizer=build_optimizer(model)
    execution_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    manifest,_,windows=load_manifest(manifest_path)
    order=epoch_order(0)
    planned_windows=(order[:16].astype(int).tolist() if distributed else [int(order[0]),int(order[0])])
    all_indices=[rank_microbatch_indices(0,u,r) for u in range(2) for r in range(WORLD_SIZE)]
    if distributed:
        selected=[all_indices[u*WORLD_SIZE+rank] for u in range(2)]
    else:
        selected=[(int(order[0]),) for _ in range(2)]
    if distributed:
        torch.cuda.synchronize();dist.barrier()
    if rank==0:output_dir.mkdir(parents=True,exist_ok=False)
    if distributed:dist.barrier()
    device=torch.device(f'cuda:{local}')
    monitor=GeometryMonitor(output_dir,rank,'eight_smoke' if distributed else 'single_smoke')
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
        if raw_vggt.confidence518.shape!=(1,2,1,518,518) or not torch.isfinite(raw_vggt.confidence518).all():
            raise FloatingPointError('smoke VGGT confidence output is malformed or nonfinite')
        del raw_vggt
        preview_generated=model.generate(preview_batch['images_input'])
        preview_calibration=model.calibrate_targets(preview_batch['images_all'],preview_generated)
        if (preview_calibration['c2w'].shape[:1]!=(1,) or
            preview_calibration['intrinsics'].shape!=(1,preview_batch['images_all'].shape[1],4) or
            not torch.isfinite(preview_calibration['c2w']).all() or
            not torch.isfinite(preview_calibration['intrinsics']).all()):
            raise RuntimeError('smoke calibrated camera outputs are malformed or nonfinite')
        preview_calibration_diagnostics=preview_calibration['diagnostics']
        preview_view=model.render_generated_at(preview_generated,
            torch.linalg.inv(preview_calibration['c2w']).transpose(-1,-2),preview_calibration['intrinsics'])
    for key in ('images_pred','gaussians','gaussian_membership','p_class','region_mass','semantic_scores','alpha'):
        value=preview_view.get(key)
        if value is None or not torch.isfinite(value).all():raise FloatingPointError(f'smoke invalid {key}')
    if rank==0:
        preview_keys=('images_pred','depths_pred','gaussians','gaussian_membership','p_class','region_mass','semantic_scores','alpha')
        torch.save({k:preview_view[k].detach().cpu() for k in preview_keys if k in preview_view},output_dir/'generated_preview.pt')
    del preview_view,preview_generated,preview_calibration,preview_batch
    torch.cuda.reset_peak_memory_stats(device)
    rows=[]
    for update,index_pair in enumerate(selected):
        batches=[build_batch(opt,windows[index],device) for index in index_pair]
        base_exposure=update*(GLOBAL_BATCH if distributed else MICRO_BATCH*ACCUMULATION)
        started=time.perf_counter()
        row=train_microbatch_window(model,optimizer,batches,update,base_exposure=base_exposure,
                                    world_size=expected,monitor=monitor,stage=monitor.stage,
                                    window_metadata=[window_identity(index,windows[index]) for index in index_pair])
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
    formal_completed_exposure=2 if not distributed else 16
    model.understanding_step=formal_completed_exposure
    diagnostic_batch=build_batch(opt,windows[int(order[0])],device)
    diagnostic_output,diagnostic_metrics=model.step_loss(diagnostic_batch,step=formal_completed_exposure,understanding_weight=1.0)
    adapter_params=[p for n,p in model.named_parameters() if n.startswith('vggt_memory_adapter.') and p.requires_grad]
    diagnostic_grads=torch.autograd.grad(diagnostic_metrics['loss_understanding'],adapter_params,allow_unused=True)
    if not any(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in diagnostic_grads):
        raise RuntimeError('weight=1 understanding diagnostic has no memory-adapter gradient')
    del diagnostic_output,diagnostic_metrics,diagnostic_batch,diagnostic_grads
    if distributed:
        dist.barrier()
        rank_rows=[None for _ in range(expected)]
        dist.all_gather_object(rank_rows,{'rows':rows,'geometry_monitor':monitor.summary()})
    else:
        rank_rows=[{'rows':rows,'geometry_monitor':monitor.summary()}]
    if rank==0:
        payload={'mode':mode,'world_size':expected,'rows_by_rank':[item['rows'] for item in rank_rows],
            'geometry_monitor_by_rank':[item['geometry_monitor'] for item in rank_rows],
            'geometry_quality_policy':GEOMETRY_QUALITY_POLICY,'execution_git_sha':execution_sha,'updates':2,
            'exposures':formal_completed_exposure,'formal_schedule_start_exposure':0,
            'understanding_diagnostic':'separate weight=1 backward; no optimizer step or clock increment',
            'source_checkpoint':provenance,'migration_counts':transfer['counts'],
            'calibration_protocol':CALIBRATION_PROTOCOL,'training_configuration':training_configuration(),
            'target_calibration_diagnostics':preview_calibration_diagnostics,
            'initial_seeds':{'common_model_initialization':42,'rank_training_rng':[42+r for r in range(expected)],'memory_adapter_isolated':31415},
            'vggt_artifact':model.frozen_vggt.source_identity,
            'manifest_sha256':sha256(manifest_path),'window_indices':planned_windows,
            'lr_schedule':'formal schedule indices 0 and 1; gradients averaged over the actual world size',
            'status':'GPU_SMOKE_COMPLETED'}
        _atomic_save({'model':checkpoint_model_state(model),'optimizer':optimizer.state_dict(),
            'completed_updates':2,'completed_exposures':payload['exposures'],'mode':mode},output_dir/'smoke_state.pt')
        (output_dir/'smoke_report.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    if distributed:dist.barrier();dist.destroy_process_group()
    return {'mode':mode,'updates':2,'exposures':formal_completed_exposure}
