"""Small classification controls using the original loss/GC/AdamW kernels."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.train_vggt_joint_short_controls_v1 import digest,parameter_hash
from scripts.eval_vggt_joint_short_controls_v1 import write,export_window
from scripts.probe_vggt_raw_query_readout_v1 import restored_full

ARMS=('head_only','object_support','scene_support')
RECIPE='vggt_classification_readback_v1'
SOURCE=Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/checkpoint_frozen_joint_epoch_04.pt')
SOURCE_SHA='ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0'
OFFICIAL=Path('/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/evaluation/final_adapt2_joint4_official')
TRAIN_COHORT=REPO/'docs/vggt_understanding_probe_cohort_v1.json'
HELDOUT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json')
PREFIXES=('panoptic.class_ln.','panoptic.class_head.','panoptic.mask_support_readback.')
TOTAL_UPDATES=300


def config(arm):
    if arm not in ARMS:raise ValueError('unknown arm')
    return dict(recipe=RECIPE,arm=arm,source_checkpoint_sha256=SOURCE_SHA,diagnostic_only=True,
        total_updates=300,total_exposures=2400,world_size=4,global_batch=8,microbatch=1,accumulation=2,
        train_cohort_sha256=digest(TRAIN_COHORT),heldout_sha256=digest(HELDOUT),
        train_stream='same first2400 unique windows of existing full8337 pool stream',
        heldout='same fixed random128 scenes, one window each; GT only for loss/evaluation',
        trainable=list(PREFIXES[:2] if arm=='head_only' else PREFIXES),
        all_other_parameters_frozen=True,frozen_core_eval_mode=True,
        adapter_understanding_scale=.01,feedback_peak=.05,source_feedback_exposure=33376,
        loss='unchanged native full set loss/matching/weights; reconstruction computed and verified has no trainable dependency',
        optimizer='fresh native AdamW, original parameter groups/betas/decay/clip1',
        learning_rate='native object_memory peak1e-4;200 update warmup,4172 cosine truncated at300',
        geometry_quality_policy='monitor_v1',support='none' if arm=='head_only' else arm,
        support_inputs='context-generated Gaussian features, masks and visibility; all detached; classifier only',
        readback_initialization='zero output residual; C/D same seed271828 and parameter count')


def slots(update,rank,world=4):
    stream=json.loads(TRAIN_COHORT.read_text())['train_source_indices'][:2400]
    if len(stream)!=2400 or len(set(stream))!=2400:raise ValueError('train stream differs')
    if world not in (1,4) or not 0<=update<300:raise ValueError('sampler clock differs')
    return stream[update*8+rank*2:update*8+rank*2+2]


def expected_names(model,arm):
    return {n for n,p in model.named_parameters() if n.startswith(PREFIXES[:2] if arm=='head_only' else PREFIXES)}


def patch_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(PREFIXES)}


def apply_patch_state(model,blob,arm):
    if blob['recipe']!=RECIPE or blob['config']!=config(arm) or blob['source_checkpoint_sha256']!=SOURCE_SHA:
        raise ValueError('patch identity/config differs')
    target=dict(model.named_parameters());expected=expected_names(model,arm)
    if set(blob['parameter_patch'])!=expected:raise ValueError('classification patch ownership differs')
    with torch.no_grad():
        for n,v in blob['parameter_patch'].items():
            if v.shape!=target[n].shape:raise ValueError('classification patch shape differs')
            target[n].copy_(v)


def build_base(device):
    model,opt,identity=restored_full(SOURCE,OFFICIAL,device)
    original=model.step_loss
    def observed(*args,**kwargs):
        output,metrics=original(*args,**kwargs)
        model.classification_loss_terms.append({k:float(metrics[k]) for k in
            ('loss_classification','loss_thing_bce','loss_thing_dice','loss_stuff_bce','loss_stuff_dice','matched_classification_accuracy','matched_gt_count')})
        return output,metrics
    model.step_loss=observed;model.classification_loss_terms=[]
    source=torch.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    rng=source['rank_rng'];del source
    initial={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith(PREFIXES[:2])}
    return model,opt,identity,rng,initial


def configure(model,arm,initial):
    from tokengs.models.object_locus_mask_support_readback_v1 import install
    if hasattr(model.panoptic,'mask_support_readback'):del model.panoptic.mask_support_readback
    with torch.no_grad():
        for n,v in initial.items():dict(model.named_parameters())[n].copy_(v)
    install(model,arm)
    if {n for n,p in model.named_parameters() if p.requires_grad}!=expected_names(model,arm):
        raise ValueError('trainable ownership differs')


def official_reproduction(model,opt,w,root,device,check_segmentation):
    row=export_window(model,opt,w,root,device)
    golden=None
    for rank in range(4):
        records=json.loads((OFFICIAL/f'rank{rank:02d}/export_complete.json').read_text())['records']
        if any(r['name']==row['name'] for r in records):golden=OFFICIAL/f'rank{rank:02d}/all'/row['name'];break
    if golden is None:raise ValueError('heldout source window absent')
    pair=root/'all'/row['name']
    for sub in ('rgb','depth','rgb_gt','depth_gt')+ (('context_seg_pred','target_seg_pred') if check_segmentation else ()):
        paths=[p for p in (golden/sub).iterdir() if p.is_file()]
        if not paths:raise ValueError('native source files missing')
        if any(digest(p)!=digest(pair/sub/p.name) for p in paths):raise ValueError('frozen/initial official reproduction differs: '+sub)
    write(root/'native_reproduction.json',dict(status='NATIVE_REPRODUCTION_VERIFIED',name=row['name'],
        RGB_depth_and_GT_reproduced=True,segmentation_reproduced=check_segmentation,optimizer_updates=0))
    return row


def run(a):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts import object_locus_v3_set_runtime as provider
    from scripts.eval_vggt_feedback_expanded_v1 import windows as validation_windows
    import torch.distributed as dist
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world not in (1,4) or (a.mode=='train' and world!=4):raise ValueError('wrong worker count')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '3090' not in torch.cuda.get_device_name(local):raise ValueError('matched3090 controls required')
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha!=os.environ['TASK_CODE_SHA']:raise ValueError('immutable code differs')
    hashed=[digest(SOURCE) if rank==0 else None];dist.broadcast_object_list(hashed,src=0)
    if hashed[0]!=SOURCE_SHA:raise ValueError('source checkpoint bytes differ')
    _,_,train=native.load_manifest();validation,_=validation_windows()
    heldout=json.loads(HELDOUT.read_text())['records'];stream=json.loads(TRAIN_COHORT.read_text())['train_source_indices'][:2400]
    if len(stream)!=2400 or len(set(stream))!=2400 or {w['scene'] for w in train}&{w['scene'] for w in validation}:
        raise ValueError('train/validation ownership differs')
    if digest(native.MANIFEST)!=json.loads(TRAIN_COHORT.read_text())['train_manifest_sha256']:raise ValueError('train manifest differs')
    model,opt,identity,source_rng,initial=build_base(device)
    for arm in ([a.arm] if a.arm else ARMS):
        output=a.output/arm if a.arm is None else a.output
        error=None
        if rank==0:
            try:output.mkdir(parents=True,exist_ok=False)
            except Exception as exc:error=exc
        native.synchronized_error(error,device,'readback_output',0);dist.barrier()
        configure(model,arm,initial);optimizer=native.build_optimizer(model)
        if optimizer.state:raise ValueError('fresh optimizer not empty')
        if world==4:native.restore_rank_rng(source_rng[rank])
        else:native.seed_everything(42)
        named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
        frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
        buffers={n:p._version for n,p in model.named_buffers()}
        initialized=[None]*world;dist.all_gather_object(initialized,dict(rank=rank,trainable_sha256=parameter_hash(named),
            trainable_parameters=sum(p.numel() for n,p in named),source_classifier_sha256=parameter_hash((n,p) for n,p in named if n.startswith(PREFIXES[:2]))))
        if len({r['trainable_sha256'] for r in initialized})!=1:raise ValueError('rank initial parameters differ')
        if rank==0:write(output/'run_manifest.json',dict(status='INITIALIZED',config=config(arm),execution_sha=sha,
            source=identity,rank_initialization=initialized,mode=a.mode,formal_updates=0,slurm_job_id=os.environ['SLURM_JOB_ID']))
        if a.mode!='train':
            witness=validation[heldout[rank]['source_index']]
            model.understanding_step=33376
            with torch.no_grad():official_reproduction(model,opt,witness,output/f'initial_rank{rank:02d}',device,True)
        counts=np.zeros(8337,dtype=np.int64);started=time.monotonic();limit=300 if a.mode=='train' else 2
        monitor=native.GeometryMonitor(output,rank,'classification_readback_'+a.mode)
        before={n:p.detach().clone() for n,p in named}

        def save(step):
            rng=[None]*world;cs=[None]*world
            dist.all_gather_object(rng,native.capture_rank_rng());dist.all_gather_object(cs,counts.tolist())
            if rank==0:
                blob=dict(recipe=RECIPE,config=config(arm),source_checkpoint_sha256=SOURCE_SHA,execution_sha=sha,
                    completed_updates=step,completed_exposures=step*8,source_joint_exposure=33376,
                    parameter_patch=patch_state(model),optimizer=optimizer.state_dict(),rank_rng=rng,rank_window_counts=cs,
                    frozen_core_unchanged=True,diagnostic_only=True,resumable=True)
                path=output/f'classification_patch_update{step:04d}.pt';native._atomic_save(blob,path)
                write(path.with_suffix('.json'),dict(status='PATCH_SAVED',step=step,sha256=digest(path),source_checkpoint_sha256=SOURCE_SHA))
            dist.barrier()

        for update in range(limit):
            indices=slots(update,rank,world)
            if a.mode!='train':indices=[4253,6923] if update==0 else list(native.rank_microbatch_indices(0,update,rank,world_size=4))
            batches=[provider.build_batch(opt,train[i],device) for i in indices]
            model.classification_loss_terms=[]
            row=native.train_microbatch_window(model,optimizer,batches,update,base_exposure=33376+update*8,
                world_size=world,accumulation=2,total_updates=4172,adapter_understanding_scale=.01,
                understanding_only=True,monitor=monitor,window_metadata=[native.window_identity(i,train[i]) for i in indices],
                stage='classification_readback_'+a.mode)
            del batches
            if any(p._version!=frozen[n] or p.grad is not None for n,p in model.named_parameters() if n in frozen):raise ValueError('frozen core changed')
            if any(p._version!=buffers[n] for n,p in model.named_buffers() if n in buffers):raise ValueError('frozen buffers changed')
            np.add.at(counts,indices,1);step=update+1
            row.update(arm=arm,source_indices=indices,actual_feedback_beta=.05,class_terms=model.classification_loss_terms,
                frozen_core_unchanged=True,readback_observation=getattr(getattr(model.panoptic,'mask_support_readback',None),'last_observation',None))
            with (output/f'training_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if rank==0:
                write(output/'progress.json',dict(completed_updates=step,total_updates=limit,formal_updates=step if a.mode=='train' else 0,
                    completed_exposures=step*world*2,seconds_per_update=(time.monotonic()-started)/step,
                    remaining_seconds=(time.monotonic()-started)/step*(limit-step)))
                if step%20==0 or a.mode!='train':print('CLASSIFICATION_READBACK',arm,step,row['loss_understanding'],flush=True)
            if step in (2,20):
                proof=[None]*world;dist.all_gather_object(proof,dict(rank=rank,trainable_sha256=parameter_hash(named),
                    changed_norm=float(sum((p.detach()-before[n]).square().sum() for n,p in named).sqrt()),
                    frozen_core_unchanged=True,peak_allocated=row['peak_allocated']))
                if len({p['trainable_sha256'] for p in proof})!=1 or any(p['changed_norm']<=0 for p in proof):raise ValueError('real updates/rank synchronization failed')
                if rank==0:write(output/f'update_proof_{step:04d}.json',dict(status='REAL_UPDATES_VERIFIED',rows=proof,formal_updates=step if a.mode=='train' else 0))
            if a.mode=='train' and step in (20,100,300):save(step)
        if a.mode!='train':
            model.understanding_step=33376
            with torch.no_grad():official_reproduction(model,opt,witness,output/f'final_rank{rank:02d}',device,False)
            dist.barrier()
        if rank==0:write(output/'COMPLETE.json',dict(status='CLASSIFICATION_TRAINED' if a.mode=='train' else 'CLASSIFICATION_SMOKE_PASSED',
            arm=arm,mode=a.mode,completed_updates=limit,formal_updates=limit if a.mode=='train' else 0,
            new_formal_exposures=limit*8 if a.mode=='train' else 0,source_checkpoint_sha256=SOURCE_SHA,
            frozen_core_unchanged=True,execution_sha=sha,world_size=world))
        del optimizer
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--mode',choices=('single_smoke','four_smoke','train'),required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='train' and args.arm is None:p.error('training requires one arm')
    run(args)
