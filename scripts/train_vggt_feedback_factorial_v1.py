"""Paired visual-freeze/feedback controls using the original joint kernels."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.train_vggt_joint_short_controls_v1 import digest, parameter_hash
ARMS = ('joint_full', 'visual_fixed', 'feedback_small', 'visual_fixed_feedback_small')
RECIPE = 'vggt_feedback_factorial_v1'
SMALL_PEAK = .025
SOURCE = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_reconstruction_adaptation_epoch_02.pt')
SOURCE_SHA = 'a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c'
REVISION = '860abec7937da0a4c03c41d3c269c366e82abdf9'
COHORT = REPO/'docs/vggt_understanding_probe_cohort_v1.json'
TOTAL_UPDATES = 500
SAVE_STEPS = (0, 20, 100, 300, 500)
EVAL_STEPS = (0, 100, 300, 500)


def configuration(arm):
    from scripts.train_vggt_understanding_probes_v1 import configuration as base_configuration
    if arm not in ARMS or SMALL_PEAK not in (.025,.05):raise ValueError('invalid factorial')
    result=base_configuration('joint_full')
    result.update(recipe=RECIPE,arm=arm,small_feedback_peak=SMALL_PEAK,
        frozen_visual_features=arm.startswith('visual_fixed'),
        feedback_peak=SMALL_PEAK if 'feedback_small' in arm else .1,
        trainable_modules='native joint except MASt3R/ViT adapter/pixel decoder' if arm.startswith('visual_fixed') else 'native joint',
        feedback_beta='feedback_peak*min(new_exposure/1000,1)',
        feedback_nominal_beta='original runtime records0.1 warmup; actual_feedback_beta records applied coefficient')
    return result


def slots(update,rank,world=4):
    if world not in (1,4) or not 0<=rank<world or not 0<=update<500:raise ValueError('invalid sampler')
    stream=json.loads(COHORT.read_text())['train_source_indices']
    return stream[update*8+rank*2:update*8+rank*2+2]


def expected_counts(step):
    result=np.zeros((4,8337),dtype=np.int64)
    stream=json.loads(COHORT.read_text())['train_source_indices']
    for update in range(step):
        for rank in range(4):np.add.at(result[rank],stream[update*8+rank*2:update*8+rank*2+2],1)
    return result


def prepare_output(output,rank,resume,synchronize):
    """Only rank0 tests/creates the path, then every rank accepts its result."""
    error=None
    if rank==0:
        try:
            if output.exists() and not resume:raise ValueError('preserve existing attempt')
            output.mkdir(parents=True,exist_ok=resume)
        except Exception as exc:error=exc
    synchronize(error)


def check_clock(b,arm,step,sha):
    expected = dict(recipe=RECIPE, arm=arm, config=configuration(arm), execution_git_sha=sha,
        source_checkpoint_sha256=SOURCE_SHA, completed_updates=step, completed_exposures=step*8,
        reused_updates=2086, reused_exposures=16688, world_size=4, resumable=step!=0)
    if any(b.get(k)!=v for k,v in expected.items()): raise ValueError('checkpoint identity/clock differs')
    if len(b['rank_rng'])!=4 or not np.array_equal(b['rank_window_counts'],expected_counts(step)):
        raise ValueError('rank RNG/sampling ownership differs')
    if step:
        state=b['optimizer']['state']
        if not state or any(int(s['step']) not in (step-1,step) for s in state.values()):
            raise ValueError('optimizer clock differs')
        if b['optimizer']['param_groups'][0]['betas']!=(.9,.95): raise ValueError('AdamW betas differ')


def build(arm,device):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.train_vggt_understanding_probes_v1 import build as base_build
    from tokengs.models.object_locus_feedback_factorial_v1 import install
    model,opt,optimizer,rng=base_build('joint_full',device)
    del optimizer
    install(model,arm,SMALL_PEAK)
    optimizer=native.build_optimizer(model)
    if optimizer.state or any(p.requires_grad for p in model.frozen_vggt.parameters()):
        raise ValueError('joint optimizer/freeze differs')
    return model,opt,optimizer,rng


def run(a):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts import object_locus_v3_set_runtime as provider
    from tokengs.models.object_locus_interaction_controls_v1 import observations
    import torch.distributed as dist
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    smoke=a.mode!='train'
    if world not in (1,4) or (not smoke and world!=4): raise ValueError('wrong world size')
    gpu_class=os.environ.get('TASK_GPU_CLASS','4090')
    if gpu_class not in ('4090','3090') or (gpu_class=='3090' and a.arm!='visual_fixed_feedback_small'):
        raise ValueError('3090 migration is restricted to visual_fixed_feedback_small')
    allowed_nodes=('3dimage-11','3dimage-13') if gpu_class=='3090' else ('3dimage-14','3dimage-17','3dimage-18')
    if socket.gethostname().split('.')[0] not in allowed_nodes:
        raise ValueError('wrong training node')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if gpu_class not in torch.cuda.get_device_name(local): raise ValueError('wrong GPU type')
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha!=os.environ['TASK_CODE_SHA']: raise ValueError('immutable code differs')
    identities=[None]
    if rank==0: identities[0]=digest(SOURCE)
    dist.broadcast_object_list(identities,src=0)
    if identities[0]!=SOURCE_SHA: raise ValueError('source SHA differs')
    cohort=json.loads(COHORT.read_text());heldout=cohort['heldout']
    _,_,original=native.load_manifest()
    windows=[dict(w,source_index=i) for i,w in enumerate(original)]
    if len(windows)!=8337 or len(heldout)!=64 or {w['scene'] for w in windows}&{w['scene'] for w in heldout}:
        raise ValueError('cohort differs')
    if digest(native.MANIFEST)!=cohort['train_manifest_sha256']:raise ValueError('train manifest changed')
    prepare_output(a.output,rank,a.resume,
        lambda error:native.synchronized_error(error,device,'understanding_probe_output_directory',0))
    dist.barrier()
    model,opt,optimizer,source_rng=build(a.arm,device)
    named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
    common=parameter_hash((n,p) for n,p in model.named_parameters() if not n.startswith('frozen_vggt.') and '.geometry_fusion.' not in n)
    initrows=[None]*world
    dist.all_gather_object(initrows,dict(rank=rank,common_parameters_sha256=common,
        trainable_sha256=parameter_hash(named),additional_parameters=sum(p.numel() for n,p in named if '.geometry_fusion.' in n)))
    if len({r['trainable_sha256'] for r in initrows})!=1: raise ValueError('initial rank weights differ')
    config=configuration(a.arm);counts=np.zeros(8337,dtype=np.int64);start=0
    if a.resume:
        receipt=json.loads((a.output/'latest_checkpoint.json').read_text())
        path=a.output/receipt['name']
        if digest(path)!=receipt['sha256']:raise ValueError('resume checkpoint SHA differs')
        b=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        start=b['completed_updates'];check_clock(b,a.arm,start,b['execution_git_sha'])
        native.restore_model_state_strict(model,b['model']);optimizer.load_state_dict(b['optimizer'])
        native.restore_rank_rng(b['rank_rng'][rank]);counts=np.asarray(b['rank_window_counts'][rank],dtype=np.int64)
        del b
    elif world==4:
        native.restore_rank_rng(source_rng[rank])
        actual=native.capture_rank_rng()
        if not torch.equal(actual['torch_cpu'],source_rng[rank]['torch_cpu']) or len(actual['torch_cuda'])!=4 or any(
            not torch.equal(x,y) for x,y in zip(actual['torch_cuda'],source_rng[rank]['torch_cuda'])):
            raise ValueError('boundary rank RNG restoration differs')
    else: native.seed_everything(42+rank)
    del source_rng
    if rank==0:
        native._write_json_atomic(a.output/('resume_manifest_'+os.environ['SLURM_JOB_ID']+'.json' if a.resume else 'run_manifest.json'),
            dict(config=config,execution_git_sha=sha,rank_initialization=initrows,mode=a.mode,
                 source_checkpoint_sha256=SOURCE_SHA,slurm_job_id=os.environ['SLURM_JOB_ID'],
                 node=socket.gethostname(),gpu_models=[torch.cuda.get_device_name(i) for i in range(world)],
                 original_boundary_rng_restored=world==4 and not a.resume,
                 fresh_optimizer=not a.resume,formal_updates=0 if smoke else start))
    fixed={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    buffers={n:p._version for n,p in model.named_buffers()}
    prefixes=('anchor_decoder.','activation_head.','vggt_memory_adapter.','understanding.','panoptic.')
    probes={prefix:next((n,p.detach().clone()) for n,p in named if n.startswith(prefix) and p.ndim>1) for prefix in prefixes}
    monitor=native.GeometryMonitor(a.output,rank,'feedback_factorial_'+a.mode,updates_per_epoch=1043)
    limit=2 if smoke else TOTAL_UPDATES
    started=time.monotonic()

    def save(step):
        rng=[None]*world;cs=[None]*world
        dist.all_gather_object(rng,native.capture_rank_rng());dist.all_gather_object(cs,counts.tolist())
        error=None
        if rank==0:
            try:
                b=dict(recipe=RECIPE,arm=a.arm,config=config,execution_git_sha=sha,
                    source_checkpoint_sha256=SOURCE_SHA,completed_updates=step,completed_exposures=step*8,
                    reused_updates=2086,reused_exposures=16688,world_size=4,model=native.checkpoint_model_state(model),
                    rank_rng=rng,rank_window_counts=cs,resumable=step!=0,
                    adapted_vggt_sha256=native.CONTINUATION_VGGT_SHA,initialization=initrows,
                    frozen_parameters_unchanged=True)
                if step:b['optimizer']=optimizer.state_dict()
                check_clock(b,a.arm,step,sha)
                path=a.output/f'checkpoint_update_{step:04d}.pt';native._atomic_save(b,path)
                loaded=torch.load(path,map_location='cpu',mmap=True,weights_only=False);check_clock(loaded,a.arm,step,sha)
                if any(not torch.equal(v,loaded['model'][k]) for k,v in b['model'].items()):raise ValueError('serialized model differs')
                receipt=dict(status='CHECKPOINT_VERIFIED',name=path.name,sha256=digest(path),step=step,resumable=step!=0)
                native._write_json_atomic(path.with_suffix('.json'),receipt)
                if step:native._write_json_atomic(a.output/'latest_checkpoint.json',receipt)
                del loaded,b
            except Exception as exc:error=exc
        native.synchronized_error(error,device,'understanding_probe_checkpoint',step)
    if not smoke and not a.resume:save(0)
    for update in range(start,limit):
        indices=slots(update,rank,world)
        selected=[windows[i] for i in indices]
        if smoke:
            # Same mandatory real windows as the native four-rank preflight,
            # including6923, regardless of the score-independent train subset.
            source_indices=native.rank_microbatch_indices(0,update,rank,world_size=4)
            selected=[dict(original[i],source_index=i) for i in source_indices]
        error=None;batches=[]
        try:batches=[provider.build_batch(opt,w,device) for w in selected]
        except Exception as exc:error=exc
        native.synchronized_error(error,device,'understanding_probe_batch',update)
        row=native.train_microbatch_window(model,optimizer,batches,update,base_exposure=update*8,
            world_size=world,accumulation=2,monitor=monitor,
            window_metadata=[native.window_identity(w['source_index'],w) for w in selected],
            stage='feedback_factorial_'+a.mode,total_updates=4*native.UPDATES_PER_EPOCH,adapter_understanding_scale=.01,
            understanding_only=False,gradient_diagnostic=smoke or update<20 or (update+1) in (100,300,500))
        del batches
        step=update+1
        np.add.at(counts,indices,1)
        error=None
        try:
            if any(p._version!=fixed[n] or p.grad is not None for n,p in model.named_parameters() if n in fixed):
                raise ValueError('frozen parameters changed')
            if any(p._version!=buffers[n] for n,p in model.named_buffers() if n in buffers):raise ValueError('fixed buffers changed')
            if step>=2 and any(not any(p.grad is not None and bool((p.grad!=0).any()) for n,p in named if n.startswith(prefix)) for prefix in prefixes):
                raise ValueError('intended module gradient absent')
            stats=observations(model.panoptic)
            if set(stats)!={'L6','L8','L10','L12'} or any(v['delta_ratio_max']>row['beta']*(model.feedback_peak/.1)*.25+1e-6 for v in stats.values()):
                raise ValueError('feedback layer/norm contract differs')
        except Exception as exc:error=exc
        native.synchronized_error(error,device,'understanding_probe_boundaries',step)
        row.update(arm=a.arm,mode=a.mode,window_slots=indices,source_indices=[w['source_index'] for w in selected],
            adapter_understanding_scale=.01,actual_feedback_beta=row['beta']*(model.feedback_peak/.1),
            interactions=observations(model.panoptic),frozen_parameters_unchanged=True)
        with (a.output/f'training_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if rank==0:
            native._write_json_atomic(a.output/'progress.json',dict(arm=a.arm,completed_updates=step,
                completed_exposures=step*world*2,total_updates=limit,
                seconds_per_update=(time.monotonic()-started)/(step-start),
                estimated_remaining_seconds=(time.monotonic()-started)/(step-start)*(limit-step)))
            if smoke or step%20==0:print(json.dumps({k:row[k] for k in ('arm','completed_updates','loss_recon','loss_understanding','preclip_norm','peak_allocated')}),flush=True)
        if (smoke and step==2) or step==20:
            changes={prefix:float((dict(named)[n].detach()-before).norm()) for prefix,(n,before) in probes.items()}
            proof=[None]*world
            dist.all_gather_object(proof,dict(rank=rank,row=row,trainable_sha256=parameter_hash(named),
                trainable_probe_changes=changes,fixed_parameters_and_buffers_unchanged=True))
            error=None
            if len({r['trainable_sha256'] for r in proof})!=1 or any(not all(v>0 for v in r['trainable_probe_changes'].values()) for r in proof):
                error=ValueError('rank synchronization/real updates differ')
            native.synchronized_error(error,device,'understanding_probe_update_proof',step)
            if rank==0:native._write_json_atomic(a.output/('smoke_update_proof.json' if smoke else 'startup_confirmation.json'),
                dict(status='GPU_STEP_VERIFIED' if smoke else 'TRAINING_CONFIRMED',arm=a.arm,
                    execution_git_sha=sha,completed_updates=step,completed_exposures=step*world*2,
                    formal_updates=0 if smoke else step,rank_rows=proof,config=config))
        if not smoke and step in SAVE_STEPS:save(step)
        if step%20==0:monitor.write_summary(step)
    if smoke:
        error=None
        if rank==0:
            try:
                from scripts.eval_vggt_joint_short_controls_v1 import export_window,write
                model.eval().requires_grad_(False)
                versions={n:p._version for n,p in model.named_parameters()}
                with torch.no_grad():record=export_window(model,opt,heldout[0],a.output/'evaluation_preflight',device)
                if any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('evaluation mutated parameters')
                write(a.output/'evaluation_preflight/export_complete.json',dict(status='PREFLIGHT_EXPORTED',
                    records=[record],optimizer_updates=0,training_updates=2,formal_updates=0,execution_git_sha=sha))
            except Exception as exc:error=exc
        native.synchronized_error(error,device,'understanding_probe_preflight_export',2)
    if rank==0:native._write_json_atomic(a.output/('smoke_report.json' if smoke else 'COMPLETE.json'),dict(
        status='GPU_SMOKE_COMPLETED' if smoke else 'INTERACTION_TRAINING_COMPLETED',arm=a.arm,
        execution_git_sha=sha,completed_updates=limit,completed_exposures=limit*world*2,
        formal_updates=0 if smoke else limit,source_checkpoint_sha256=SOURCE_SHA,
        slurm_job_id=os.environ['SLURM_JOB_ID'],frozen_parameters_unchanged=True,config=config))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--mode',choices=('train','single_smoke','four_smoke'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--small-peak',type=float,choices=(.025,.05),required=True)
    a=p.parse_args();SMALL_PEAK=a.small_peak;run(a)
