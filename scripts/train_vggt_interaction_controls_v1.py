"""Paired short decoder controls using native frozen-joint numerical kernels."""
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
ARMS = ('bidirectional', 'forward_only', 'gated_feedback', 'independent_feedback')
RECIPE = 'vggt_interaction_controls_v1'
SOURCE = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_reconstruction_adaptation_epoch_02.pt')
SOURCE_SHA = 'a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c'
REVISION = '860abec7937da0a4c03c41d3c269c366e82abdf9'
COHORT = REPO/'docs/vggt_interaction_cohort_v1.json'
TOTAL_UPDATES = 500
SAVE_STEPS = (0, 20, 100, 300, 500)
EVAL_STEPS = (0, 100, 300, 500)


def configuration(arm):
    if arm not in ARMS: raise ValueError('unknown arm')
    return dict(recipe=RECIPE, arm=arm, diagnostic_only=True,
        source_checkpoint_sha256=SOURCE_SHA, cohort_sha256=digest(COHORT), vggt_revision=REVISION,
        world_size=4, microbatch=1, accumulation=2, global_batch=8,
        total_updates=500, total_exposures=4000, train_scenes=128, train_windows=512,
        heldout_scenes=64, heldout_windows=64, state_layers=[6,8,10,12],
        vggt_trainable=False, understanding_trainable=True, decoder_initialization='exact adaptation2 boundary',
        adapter_understanding_scale=.01, gs_understanding_scale=.01,
        understanding_panoptic_scale=1., understanding_weight='min(new_exposure/200,1)',
        feedback_beta='0 for forward_only; otherwise 0.1*min(new_exposure/1000,1)',
        feedback_norm_bound='beta * 0.25 * token_norm (gate also in [0,1])',
        gate_initial_value=.9 if arm=='gated_feedback' else None,
        reverse_attention='independent Q/K/V/O, 8 heads; same geometric prior and zero null key/value'
                          if arm=='independent_feedback' else 'original normalized mask-space dot product',
        lr_peak=dict(reconstruction=1e-5,understanding=1e-5,object_memory=1e-4),
        lr_clock='original 200-update warmup and 4172-update frozen-joint cosine; truncate at500',
        optimizer='fresh native GPU AdamW; beta=(0.9,0.95),eps=1e-8,original decay groups',
        clip=1., geometry_quality_policy='monitor_v1', calibration_protocol='shared_context_depth_sim3_v2',
        precision='adapted VGGT aggregator BF16; camera/depth FP32; downstream FP32',
        sampling='seed42+epoch permutation of512; consecutive global8, rank2; original four-rank boundary RNG')


def slots(update, rank, world=4):
    if world not in (1,4) or not 0<=rank<world or not 0<=update<500: raise ValueError('invalid sampler')
    epoch, position = divmod(update, 64)
    order = np.random.default_rng(42+epoch).permutation(512)
    return [int(i) for i in order[position*8+rank*2:position*8+rank*2+2]]


def expected_counts(step):
    result = np.zeros((4,512), dtype=np.int64)
    for update in range(step):
        for rank in range(4): np.add.at(result[rank], slots(update,rank), 1)
    return result


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
    from tokengs.models.object_locus_interaction_controls_v1 import install_interaction
    native.seed_everything(42); os.environ['VGGT_HF_REVISION']=REVISION
    model,opt,_,_=native.build_model(preserve_fp32_aggregator=True)
    boundary=torch.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    native.validate_staged_boundary(boundary,native.staged_training_configuration(4,.01),REVISION,digest(native.MANIFEST))
    native.restore_model_state_strict(model,boundary['model'])
    install_interaction(model.panoptic,arm)
    model.to(device)
    optimizer=native.configure_staged_phase(model,'frozen_joint',device)
    native.restore_adapted_vggt(model,SOURCE.parent/'adapted_vggt.pt',native.CONTINUATION_VGGT_SHA)
    model.understanding_step=0
    if optimizer.state or any(p.requires_grad for p in model.frozen_vggt.parameters()):
        raise ValueError('joint optimizer/freeze contract differs')
    if not any(p.requires_grad for p in model.understanding.parameters()): raise ValueError('understanding frozen')
    return model,opt,optimizer,boundary['rank_rng']


def run(a):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts import object_locus_v3_set_runtime as provider
    from tokengs.models.object_locus_interaction_controls_v1 import observations
    import torch.distributed as dist
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    smoke=a.mode!='train'
    if world not in (1,4) or (not smoke and world!=4): raise ValueError('wrong world size')
    if socket.gethostname().split('.')[0] not in ('3dimage-14','3dimage-17','3dimage-18'):
        raise ValueError('wrong training node')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '4090' not in torch.cuda.get_device_name(local): raise ValueError('wrong GPU type')
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha!=os.environ['TASK_CODE_SHA']: raise ValueError('immutable code differs')
    identities=[None]
    if rank==0: identities[0]=digest(SOURCE)
    dist.broadcast_object_list(identities,src=0)
    if identities[0]!=SOURCE_SHA: raise ValueError('source SHA differs')
    cohort=json.loads(COHORT.read_text());windows=cohort['train'];heldout=cohort['heldout']
    if len(windows)!=512 or len(heldout)!=64 or {w['scene'] for w in windows}&{w['scene'] for w in heldout}:
        raise ValueError('cohort differs')
    _,_,original=native.load_manifest()
    if any(any(w[k]!=original[w['source_index']][k] for k in ('scene','context','novel')) for w in windows):
        raise ValueError('train source changed')
    if a.output.exists() and not a.resume: raise ValueError('preserve existing attempt')
    if rank==0:a.output.mkdir(parents=True,exist_ok=a.resume)
    dist.barrier()
    model,opt,optimizer,source_rng=build(a.arm,device)
    named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
    common=parameter_hash((n,p) for n,p in model.named_parameters() if not n.startswith('frozen_vggt.') and '.interaction_feedback.' not in n)
    initrows=[None]*world
    dist.all_gather_object(initrows,dict(rank=rank,common_parameters_sha256=common,
        trainable_sha256=parameter_hash(named),additional_parameters=sum(p.numel() for n,p in named if '.interaction_feedback.' in n)))
    if len({r['trainable_sha256'] for r in initrows})!=1: raise ValueError('initial rank weights differ')
    config=configuration(a.arm);counts=np.zeros(512,dtype=np.int64);start=0
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
    newprobes={n:p.detach().clone() for n,p in named if '.interaction_feedback.' in n}
    monitor=native.GeometryMonitor(a.output,rank,'interaction_'+a.mode,updates_per_epoch=64)
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
        native.synchronized_error(error,device,'interaction_checkpoint',step)
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
        native.synchronized_error(error,device,'interaction_batch',update)
        row=native.train_microbatch_window(model,optimizer,batches,update,base_exposure=update*8,
            world_size=world,accumulation=2,monitor=monitor,
            window_metadata=[native.window_identity(w['source_index'],w) for w in selected],
            stage='interaction_'+a.mode,total_updates=4*native.UPDATES_PER_EPOCH,adapter_understanding_scale=.01)
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
            if set(stats)!={'L6','L8','L10','L12'} or any(v['delta_ratio_max']>row['beta']*.25+1e-6 for v in stats.values()):
                raise ValueError('feedback layer/norm contract differs')
            if a.arm=='forward_only' and any(v['delta_ratio_max']!=0 for v in stats.values()):
                raise ValueError('forward-only control injected feedback')
        except Exception as exc:error=exc
        native.synchronized_error(error,device,'interaction_boundaries',step)
        row.update(arm=a.arm,mode=a.mode,window_slots=indices,source_indices=[w['source_index'] for w in selected],
            adapter_understanding_scale=.01,actual_feedback_beta=0. if a.arm=='forward_only' else row['beta'],
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
            added={n:float((dict(named)[n].detach()-before).norm()) for n,before in newprobes.items()}
            proof=[None]*world
            dist.all_gather_object(proof,dict(rank=rank,row=row,trainable_sha256=parameter_hash(named),
                trainable_probe_changes=changes,new_parameter_changes=added,fixed_parameters_and_buffers_unchanged=True))
            error=None
            if len({r['trainable_sha256'] for r in proof})!=1 or any(not all(v>0 for v in r['trainable_probe_changes'].values()) for r in proof):
                error=ValueError('rank synchronization/real updates differ')
            if added and not any(v>0 for v in added.values()):error=ValueError('new interaction parameters never update')
            native.synchronized_error(error,device,'interaction_update_proof',step)
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
        native.synchronized_error(error,device,'interaction_preflight_export',2)
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
    run(p.parse_args())
