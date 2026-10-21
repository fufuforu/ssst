"""Stop after capturing a real norm>=20 event, never a prespecified step."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.diagnose_vggt_joint_gradient_spikes_v1 import restored,RUNS
from scripts.eval_vggt_joint_short_controls_v1 import write
from scripts.vggt_joint_gradient_observer_v1 import TriggeredGradientObserver


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=RUNS,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('matched original four-rank replay required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '4090' not in torch.cuda.get_device_name(local):raise ValueError('matched4090 replay required')
    torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=10))
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.object_locus_v3_set_runtime import build_batch
    from tokengs.models.object_locus_feedback_factorial_v1 import fixed_eval
    if rank==0:a.output.mkdir(parents=True,exist_ok=False)
    torch.distributed.barrier()
    model,opt,optimizer,trainer,identity=restored(a.arm,300,device)
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    original=[json.loads(x) for x in (RUNS[a.arm]/f'training_rank{rank}.jsonl').read_text().splitlines()]
    _,_,windows=native.load_manifest();monitor=native.GeometryMonitor(a.output,rank,'actual_spike_replay',updates_per_epoch=1043)
    limit=301 if a.smoke else 441;captured=[];threshold=20.
    if rank==0:write(a.output/'run_manifest.json',dict(status='ACTUAL_SPIKE_REPLAY_STARTED',arm=a.arm,source=identity,
        max_replay_calls=limit-300,threshold=threshold,stop_after_first_actual_spike=True,clip_unchanged=1.,
        no_new_research_training=True,no_large_event_checkpoint=True,main_training_untouched=True,
        historical799_attribution_not_assumed=True))
    for update in range(300,limit):
        before_rng=native.capture_rank_rng();indices=trainer.slots(update,rank)
        batches=[build_batch(opt,windows[i],device) for i in indices];fixed_eval(model)
        observer=TriggeredGradientObserver(4,.01,threshold,force=update==300)
        row=native.train_microbatch_window(model,optimizer,batches,update,base_exposure=update*8,
            world_size=4,accumulation=2,total_updates=4172,adapter_understanding_scale=.01,
            monitor=monitor,window_metadata=[native.window_identity(i,windows[i]) for i in indices],
            stage='actual_spike_replay',gradient_observer=observer)
        if any(p._version!=frozen[n] for n,p in model.named_parameters() if n in frozen):raise ValueError('frozen parameters changed')
        reference=original[update]
        if reference['source_indices']!=indices or reference['lr_multiplier']!=row['lr_multiplier']:raise ValueError('sampler/LR differs')
        row.update(source_indices=indices,original_logged_values={k:reference[k] for k in
            ('loss_recon','loss_understanding','preclip_norm','lr_multiplier')},diagnostic_replay=True,new_research_training=False)
        with (a.output/f'replay_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if observer.captured:
            proof=[None]*4;torch.distributed.all_gather_object(proof,observer.result)
            if any(abs(x['native_preclip_norm']-row['preclip_norm'])>1e-5*max(row['preclip_norm'],1.) for x in proof):
                raise ValueError('global gradient norms differ by rank')
            if rank==0:write(a.output/f'global_gradients_update{update+1:04d}.json',dict(status='ACTUAL_GLOBAL_PATHS_MEASURED',
                step=update+1,actual_spike=observer.triggered,rank_rows=proof,source=identity,
                original_norm=reference['preclip_norm'],replay_norm=row['preclip_norm'],main_training_untouched=True))
            if observer.triggered:
                torch.save(dict(rank=rank,step=update+1,source_indices=indices,before_step_rng=before_rng,
                    source=identity,not_resumable_without_event_model_and_optimizer=True),a.output/f'spike_batch_rng_rank{rank}.pt')
                captured.append(update+1)
        if rank==0 and (update==300 or observer.triggered or (update+1)%20==0):
            print('ACTUAL_SPIKE_PROGRESS',a.arm,update+1,row['preclip_norm'],observer.triggered,flush=True)
        del observer,before_rng
        if captured:break
    if rank==0:write(a.output/('smoke_report.json' if a.smoke else 'COMPLETE.json'),dict(
        status='ACTUAL_SPIKE_SMOKE_PASSED' if a.smoke else 'ACTUAL_SPIKE_CAPTURED' if captured else 'NO_SPIKE_WITHIN_BOUNDED_REPLAY',
        arm=a.arm,actual_spike_steps=captured,replayed_calls=update+1-300,replayed_exposures=(update+1-300)*8,
        source=identity,frozen_parameters_unchanged=True,clip_unchanged=True,new_research_training=False,
        historical_step441_not_required=True,main_training_untouched=True))
    torch.distributed.barrier();torch.distributed.destroy_process_group()


if __name__=='__main__':main()
