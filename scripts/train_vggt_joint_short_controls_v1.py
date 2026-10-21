"""Small, paired joint-task controls; reuse the native model and optimizer step."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import timedelta
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
RECIPE='vggt_joint_short_controls_v1'
ARMS=('frozen_warm','aa_warm','frozen_scratch')
SOURCE=Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_frozen_joint_epoch_04.pt')
SOURCE_SHA='01d49b092c8dfc2e02f15b5216e5d6ed1cfd88fae2cf693ee69334e45a9410ba'
REVISION='860abec7937da0a4c03c41d3c269c366e82abdf9'
COHORT=REPO/'docs/vggt_joint_short_controls_cohort_v1.json'
GS_PREFIXES=('gs_tokens','gs_tokens_dynamic','anchor_decoder.','enc_dec_backbone.decoder_blocks.','activation_head.')
AA_PREFIXES=('model.aggregator.frame_blocks.','model.aggregator.global_blocks.')
TOTAL_UPDATES=100


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def configuration(arm):
    if arm not in ARMS:raise ValueError('unknown control arm')
    return dict(recipe=RECIPE,arm=arm,diagnostic_only=True,source_checkpoint_sha256=SOURCE_SHA,
        vggt_revision=REVISION,cohort_sha256=digest(COHORT),world_size=8,global_batch=16,
        microbatch=1,accumulation=2,total_updates=100,total_exposures=1600,
        train_scenes=16,heldout_scenes=16,understanding_trainable=True,understanding_weight=1.,
        feedback_enabled=True,beta=.1,adapter_understanding_scale=.01,gs_understanding_scale=.01,
        aa_trainable=arm=='aa_warm',aa_understanding_scale=.01,
        decoder_initialization='native constructor seed42' if arm=='frozen_scratch' else 'common VGGT epoch6',
        reset_prefixes=list(GS_PREFIXES) if arm=='frozen_scratch' else [],
        adapter_initialization='common trained epoch6; unchanged in all arms',
        lr_peak=dict(reconstruction=1e-5,understanding=1e-5,object_memory=1e-4,vggt_aa=1e-6),
        lr_clock='20 update diagnostic warmup then hold original peaks; no production cosine claim',
        optimizer='fresh native_fp32_cpu_adamw_v1',clip=1.,geometry_quality_policy='monitor_v1',
        precision='FP32 AA in all arms; BF16 DINO/autocast; downstream FP32',
        train_sampling='seed42+update permutation of all16; each window once/update')


def slots(update,rank,world):
    if not 0<=update<100 or world not in (1,8) or not 0<=rank<world:raise ValueError('invalid sampler clock')
    order=np.random.default_rng(42+update).permutation(16)
    return [int(i) for i in order[rank*2:rank*2+2]]


def restore_fresh_decoder(model,fresh):
    target=model.state_dict()
    expected={k for k in target if k.startswith(GS_PREFIXES)}
    if set(fresh)!=expected or any(fresh[k].shape!=target[k].shape for k in expected):
        raise ValueError('fresh GS state keys/shapes mismatch')
    blocks=model.enc_dec_backbone.decoder_blocks
    references=model.anchor_decoder.decoder_blocks
    if len(blocks)!=len(references) or any(a is not b for a,b in zip(blocks,references)):
        raise ValueError('anchor decoder lost its shared backbone references')
    # Check shared aliases before copying so one cannot overwrite another.
    for key in expected:
        if key.startswith('enc_dec_backbone.decoder_blocks.'):
            alias='anchor_decoder.decoder_blocks.'+key.removeprefix('enc_dec_backbone.decoder_blocks.')
            # Production uses an unregistered tuple of references. Some test
            # graphs register both names; if present those aliases must agree.
            if alias in fresh and not torch.equal(fresh[key],fresh[alias]):raise ValueError('shared decoder alias mismatch')
    with torch.no_grad():
        for k in sorted(expected):target[k].copy_(fresh[k])
    if any(not torch.equal(target[k].cpu(),fresh[k]) for k in expected):raise ValueError('fresh GS copy failed')
    return sorted(expected)


def parameter_hash(named):
    h=hashlib.sha256()
    for n,p in sorted(named):
        h.update(n.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_clock(b,arm,step,sha):
    expected=dict(recipe=RECIPE,arm=arm,diagnostic_only=True,execution_git_sha=sha,
        completed_updates=step,completed_exposures=step*16,source_checkpoint_sha256=SOURCE_SHA,
        config=configuration(arm))
    if any(b.get(k)!=v for k,v in expected.items()):raise ValueError('short-control checkpoint contract mismatch')
    if len(b['rank_rng'])!=8 or len(b['rank_window_counts'])!=8:raise ValueError('checkpoint rank state mismatch')
    counts=np.asarray(b['rank_window_counts'])
    if counts.shape!=(8,16) or not np.array_equal(counts.sum(0),np.full(16,step)):raise ValueError('window exposure count mismatch')
    if bool(b['vggt_aa'])!=(arm=='aa_warm'):raise ValueError('AA weights absent/unexpected')
    if b['resumable']!=(step in (20,100)):raise ValueError('resume-state contract mismatch')
    if b['resumable']:
        o=b['optimizer']
        if o['kind']!='native_fp32_cpu_adamw_v1' or not o['optimizer']['state'] or any(int(s['step'])!=step for s in o['optimizer']['state'].values()):raise ValueError('optimizer clock mismatch')


def build(arm,device):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _restore_checkpoint
    native.seed_everything(42)
    os.environ['VGGT_HF_REVISION']=REVISION
    fresh={} if arm=='frozen_scratch' else None
    model,opt,_,_=native.build_model(preserve_fp32_aggregator=True,fresh_reconstruction_state=fresh)
    model.to(device=device,dtype=torch.float32)
    _restore_checkpoint(model,SOURCE,REVISION,digest(native.MANIFEST))
    reset=restore_fresh_decoder(model,fresh) if fresh is not None else []
    del fresh
    optimizer=native.configure_joint_continuation(model,'joint_aa' if arm=='aa_warm' else 'frozen',device)
    native.restore_adapted_vggt(model,SOURCE.parent/'adapted_vggt.pt',native.CONTINUATION_VGGT_SHA)
    # Phase setup may convert the evaluation-restored BF16 AA to FP32. Reload
    # the original FP32 asset, then synchronize native CPU masters as in A/B.
    with torch.no_grad():
        for parameter,master in optimizer.pairs:master.copy_(parameter.detach().cpu())
    if any(not torch.equal(parameter.detach().cpu(),master) for parameter,master in optimizer.pairs):
        raise ValueError('CPU optimizer masters differ from common initial parameters')
    model.understanding_step=50064
    if not any(p.requires_grad for p in model.understanding.parameters()):raise ValueError('understanding unexpectedly frozen')
    actual={n for n,p in model.frozen_vggt.named_parameters() if p.requires_grad}
    expected={n for n,p in model.frozen_vggt.named_parameters() if n.startswith(AA_PREFIXES)} if arm=='aa_warm' else set()
    if actual!=expected:raise ValueError('AA/DINO freeze contract mismatch')
    return model,opt,optimizer,reset


def run(a):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts import object_locus_v3_set_runtime as provider
    import torch.distributed as dist
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if world not in (1,8) or (a.mode=='train' and world!=8) or socket.gethostname().split('.')[0]!='3dimage-17':raise ValueError('wrong resource allocation')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '4090' not in torch.cuda.get_device_name(local):raise ValueError('wrong GPU model')
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    if a.output.exists():raise ValueError('preserve existing run; use fresh attempt')
    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha!=os.environ['TASK_CODE_SHA'] or digest(SOURCE)!=SOURCE_SHA:raise ValueError('execution/source identity mismatch')
    cohort=json.loads(COHORT.read_text());windows=cohort['train'];heldout=cohort['heldout']
    if len(windows)!=16 or len(heldout)!=16 or {w['scene'] for w in windows}&{w['scene'] for w in heldout}:raise ValueError('cohort separation mismatch')
    _,_,original=native.load_manifest()
    if any(any(original[w['source_index']][k]!=w[k] for k in ('scene','context','novel')) for w in windows):raise ValueError('train provider differs')
    if rank==0:a.output.mkdir(parents=True)
    dist.barrier()
    model,opt,optimizer,reset=build(a.arm,device)
    named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
    initial=parameter_hash(named)
    unchanged=parameter_hash((n,p) for n,p in model.named_parameters() if n.startswith(('understanding.','panoptic.','vggt_memory_adapter.')))
    initrows=[None]*world
    dist.all_gather_object(initrows,dict(rank=rank,initial_trainable_sha256=initial,
        common_understanding_panoptic_adapter_sha256=unchanged,reset_keys=reset))
    if len({r['initial_trainable_sha256'] for r in initrows})!=1:raise ValueError('initial rank state differs')
    config=configuration(a.arm)
    if rank==0:native._write_json_atomic(a.output/'run_manifest.json',dict(config=config,execution_git_sha=sha,
        source_checkpoint_sha256=SOURCE_SHA,slurm_job_id=os.environ['SLURM_JOB_ID'],mode=a.mode,
        formal_updates=0,rank_initialization=initrows,reset_keys=reset,smoke_clock_in_formal_training=False))
    native.seed_everything(42+rank)
    fixed={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    fixedbuf={n:b._version for n,b in model.named_buffers()}
    prefixes=['anchor_decoder.','activation_head.','vggt_memory_adapter.','understanding.','panoptic.']
    if a.arm=='aa_warm':prefixes+=['frozen_vggt.model.aggregator.frame_blocks.','frozen_vggt.model.aggregator.global_blocks.']
    probes={prefix:next((n,p.detach().clone()) for n,p in named if n.startswith(prefix) and p.ndim>1) for prefix in prefixes}
    counts=np.zeros(16,dtype=np.int64)
    monitor=native.GeometryMonitor(a.output,rank,'joint_short_'+a.mode,updates_per_epoch=100)
    smoke=a.mode!='train';limit=2 if smoke else 100;started=time.monotonic()

    def save(step):
        rng=[None]*world;cs=[None]*world
        dist.all_gather_object(rng,native.capture_rank_rng());dist.all_gather_object(cs,counts.tolist())
        error=None
        if rank==0:
            try:
                aa={k:v.detach().cpu() for k,v in model.frozen_vggt.state_dict().items() if k.startswith(AA_PREFIXES)} if a.arm=='aa_warm' else {}
                b=dict(recipe=RECIPE,arm=a.arm,diagnostic_only=True,config=config,execution_git_sha=sha,
                    completed_updates=step,completed_exposures=step*16,source_checkpoint_sha256=SOURCE_SHA,
                    model=native.checkpoint_model_state(model),vggt_aa=aa,rank_rng=rng,rank_window_counts=cs,
                    adapted_vggt_sha256=native.CONTINUATION_VGGT_SHA,resumable=step in (20,100),
                    initialization=initrows,slurm_job_id=os.environ['SLURM_JOB_ID'],frozen_parameters_unchanged=True)
                if b['resumable']:b['optimizer']=optimizer.state_dict()
                check_clock(b,a.arm,step,sha)
                path=a.output/f'checkpoint_update_{step:04d}.pt'
                native._atomic_save(b,path)
                restored=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
                check_clock(restored,a.arm,step,sha)
                if set(restored['model'])!=set(b['model']) or any(not torch.equal(v,restored['model'][k]) for k,v in b['model'].items()):raise ValueError('serialized model differs')
                if set(restored['vggt_aa'])!=set(aa) or any(not torch.equal(v,restored['vggt_aa'][k]) for k,v in aa.items()):raise ValueError('serialized AA differs')
                native._write_json_atomic(a.output/f'checkpoint_update_{step:04d}.json',dict(status='CHECKPOINT_VERIFIED',
                    step=step,sha256=digest(path),path=str(path),resumable=b['resumable']))
                del b,restored,aa
            except Exception as exc:error=exc
        native.synchronized_error(error,device,'short_checkpoint',step)
    if not smoke:save(0)
    for update in range(limit):
        indices=slots(update,rank,world);error=None;batches=[]
        try:batches=[provider.build_batch(opt,windows[i],device) for i in indices]
        except Exception as exc:error=exc
        native.synchronized_error(error,device,'short_batch',update)
        row=native.train_microbatch_window(model,optimizer,batches,update,base_exposure=50064+update*16,
            world_size=world,accumulation=2,monitor=monitor,window_metadata=[native.window_identity(windows[i]['source_index'],windows[i]) for i in indices],
            stage='joint_short_'+a.mode,total_updates=native.TOTAL_UPDATES,lr_update=min((update+1)*10,200)-1,
            cpu_saved_tensors=True,cpu_gradient_accumulation=True,vggt_reconstruction=True,adapter_understanding_scale=.01)
        del batches
        step=update+1;counts[indices]+=1;error=None
        try:
            if any(p._version!=fixed[n] or p.grad is not None for n,p in model.named_parameters() if n in fixed):raise ValueError('frozen parameter changed')
            if any(b._version!=fixedbuf[n] for n,b in model.named_buffers() if n in fixedbuf):raise ValueError('fixed buffer changed')
            if any(not any(p.grad is not None and bool((p.grad!=0).any()) for n,p in named if n.startswith(prefix)) for prefix in prefixes):raise ValueError('intended gradient missing')
        except Exception as exc:error=exc
        native.synchronized_error(error,device,'short_boundaries',step)
        row.update(arm=a.arm,mode=a.mode,window_slots=indices,adapter_understanding_scale=.01,
            aa_trainable=a.arm=='aa_warm',understanding_trainable=True,frozen_parameters_unchanged=True)
        with (a.output/f'training_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if rank==0:
            native._write_json_atomic(a.output/'progress.json',dict(arm=a.arm,completed_updates=step,
                completed_exposures=step*world*2,total_updates=limit,
                estimated_remaining_seconds=(time.monotonic()-started)/step*(limit-step)))
            if smoke or step%10==0:print(json.dumps({k:row[k] for k in ('arm','completed_updates','loss_recon','loss_understanding','preclip_norm','peak_allocated')},allow_nan=False),flush=True)
        if smoke or step==20:
            changes={prefix:float((dict(named)[n].detach()-before).norm()) for prefix,(n,before) in probes.items()}
            proof=[None]*world
            dist.all_gather_object(proof,dict(rank=rank,row=row,trainable_sha256=parameter_hash(named),
                trainable_probe_changes=changes,fixed_parameters_and_buffers_unchanged=True))
            if len({r['trainable_sha256'] for r in proof})!=1 or any(not all(v>0 for v in r['trainable_probe_changes'].values()) for r in proof):raise ValueError('rank synchronization or real update failed')
            if rank==0:native._write_json_atomic(a.output/('smoke_update_proof.json' if smoke else 'startup_confirmation.json'),
                dict(status='GPU_STEP_VERIFIED' if smoke else 'TRAINING_CONFIRMED',arm=a.arm,
                execution_git_sha=sha,completed_updates=step,completed_exposures=step*world*2,
                formal_updates=0 if smoke else step,rank_rows=proof,config=config))
        if not smoke and step in (20,50,100):save(step)
        if step%20==0:monitor.write_summary(step)
    if smoke:
        error=None
        if rank==0:
            try:
                from scripts.eval_vggt_joint_short_controls_v1 import export_window,write
                model.frozen_vggt.set_reconstruction_adaptation(False,keep_fp32=True)
                model.eval().requires_grad_(False)
                versions={n:p._version for n,p in model.named_parameters()}
                with torch.no_grad():row=export_window(model,opt,heldout[0],a.output/'evaluation_preflight',device)
                if any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('preflight export mutated parameters')
                write(a.output/'evaluation_preflight/export_complete.json',dict(status='PREFLIGHT_EXPORTED',
                    records=[row],optimizer_updates=0,training_updates=2,formal_updates=0,execution_git_sha=sha))
            except Exception as exc:error=exc
        native.synchronized_error(error,device,'preflight_export',2)
    if rank==0:native._write_json_atomic(a.output/('smoke_report.json' if smoke else 'COMPLETE.json'),dict(
        status='GPU_SMOKE_COMPLETED' if smoke else 'JOINT_SHORT_TRAINING_COMPLETED',arm=a.arm,execution_git_sha=sha,
        completed_updates=limit,completed_exposures=limit*world*2,formal_updates=0 if smoke else limit,
        source_checkpoint_sha256=SOURCE_SHA,frozen_parameters_unchanged=True,understanding_trainable=True,
        adapter_understanding_scale=.01,slurm_job_id=os.environ['SLURM_JOB_ID']))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--mode',choices=('train','single_smoke','eight_smoke'),required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
