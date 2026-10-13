"""Replay original A/B/C updates301..441 without altering GC/clip/AdamW."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.vggt_joint_gradient_observer_v1 import GradientObserver
from scripts.eval_vggt_joint_short_controls_v1 import write

RUNS=dict(
    A=Path('/space/mawb/ssst/workspace_group_plus/vggt_understanding_probes_v1/attempt01/joint_full'),
    B=Path('/space/mawb/ssst/workspace_group_plus/vggt_feedback_factorial_v1/attempt01/visual_fixed'),
    C=Path('/space/mawb/ssst/workspace_group_plus/vggt_feedback_factorial_v1/attempt01/feedback_small'))
ARMS=dict(A='joint_full',B='visual_fixed',C='feedback_small')


def restored(letter,step,device,collective_hash=True,restore_optimizer=True):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts import train_vggt_feedback_factorial_v1 as factorial
    from scripts import train_vggt_understanding_probes_v1 as probes
    factorial.SMALL_PEAK=.05
    trainer=probes if letter=='A' else factorial
    root=RUNS[letter];path=root/f'checkpoint_update_{step:04d}.pt'
    receipt=json.loads(path.with_suffix('.json').read_text())
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    valid=[None]
    if rank==0 or not collective_hash:valid[0]=trainer.digest(path)==receipt['sha256']
    if world>1 and collective_hash:torch.distributed.broadcast_object_list(valid,src=0)
    if not valid[0]:raise ValueError('saved checkpoint hash mismatch')
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    trainer.check_clock(payload,ARMS[letter],step,payload['execution_git_sha'])
    model,opt,optimizer,_=trainer.build(ARMS[letter],device)
    native.restore_model_state_strict(model,payload['model'])
    provenance=dict(checkpoint=str(path),checkpoint_sha256=receipt['sha256'],
        training_execution_sha=payload['execution_git_sha'],config=payload['config'],restored_step=step)
    if step and restore_optimizer:
        optimizer.load_state_dict(payload['optimizer'])
        native.restore_rank_rng(payload['rank_rng'][rank])
    del payload
    return model,opt,optimizer,trainer,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=RUNS,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('matched four-rank replay required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '4090' not in torch.cuda.get_device_name(local):raise ValueError('original matched4090 replay required')
    torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=10))
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.object_locus_v3_set_runtime import build_batch
    from tokengs.models.object_locus_feedback_factorial_v1 import fixed_eval
    if rank==0:a.output.mkdir(parents=True,exist_ok=False)
    torch.distributed.barrier()
    model,opt,optimizer,trainer,identity=restored(a.arm,300,device)
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    original=[json.loads(x) for x in (RUNS[a.arm]/f'training_rank{rank}.jsonl').read_text().splitlines()]
    _,_,windows=native.load_manifest()
    monitor=native.GeometryMonitor(a.output,rank,'diagnostic_replay',updates_per_epoch=1043)
    probes={301} if a.smoke else {301,400,440,441}
    limit=301 if a.smoke else 441
    if rank==0:write(a.output/'run_manifest.json',dict(status='DIAGNOSTIC_REPLAY_STARTED',arm=a.arm,
        replay_updates=limit-300,new_research_training=False,source=identity,
        observer_steps=sorted(probes),gc_and_clip_unchanged=True,source_checkpoints_never_overwritten=True))
    for update in range(300,limit):
        indices=trainer.slots(update,rank)
        batches=[build_batch(opt,windows[i],device) for i in indices]
        fixed_eval(model)
        observer=GradientObserver(4,.01) if update+1 in probes else None
        row=native.train_microbatch_window(model,optimizer,batches,update,base_exposure=update*8,
            world_size=4,accumulation=2,total_updates=4172,adapter_understanding_scale=.01,
            monitor=monitor,window_metadata=[native.window_identity(i,windows[i]) for i in indices],
            stage='diagnostic_replay',gradient_observer=observer)
        if any(p._version!=frozen[n] for n,p in model.named_parameters() if n in frozen):raise ValueError('frozen parameters changed')
        reference=original[update]
        if reference['source_indices']!=indices:raise ValueError('original replay sampler differs')
        row.update(source_indices=indices,original_logged_values={k:reference[k] for k in
                   ('loss_recon','loss_understanding','preclip_norm','lr_multiplier')},
                   diagnostic_replay=True,new_research_training=False)
        with (a.output/f'replay_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if observer is not None:
            proof=[None]*4;torch.distributed.all_gather_object(proof,observer.result)
            if any(abs(x['native_preclip_norm']-row['preclip_norm'])>1e-5*max(row['preclip_norm'],1.) for x in proof):
                raise ValueError('replay global gradients differ by rank')
            if rank==0:write(a.output/f'global_gradients_update{update+1:04d}.json',dict(status='GLOBAL_PATHS_MEASURED',
                step=update+1,rank_rows=proof,source=identity,original_norm=reference['preclip_norm'],
                replay_norm=row['preclip_norm'],not_an_optimizer_intervention=True))
            del observer,proof
    if rank==0:write(a.output/('smoke_report.json' if a.smoke else 'COMPLETE.json'),dict(
        status='GRADIENT_REPLAY_SMOKE_PASSED' if a.smoke else 'GRADIENT_REPLAY_COMPLETED',arm=a.arm,
        replayed_updates=limit-300,replayed_global_exposures=(limit-300)*8,source=identity,
        model_or_optimizer_interventions=0,not_formal_training=True,frozen_parameters_unchanged=True))
    torch.distributed.barrier();torch.distributed.destroy_process_group()


if __name__=='__main__':main()
