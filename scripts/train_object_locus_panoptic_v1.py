"""One fixed fresh 64-epoch global8 run; smoke state cannot enter this run."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import torch
import torch.distributed as dist
from scripts.object_locus_panoptic_v1_runtime import *
from scripts.eval_object_locus_panoptic_v1 import evaluate_epoch,parameter_drift,endpoint_report


def atomic_save(path,blob):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(blob,tmp)
    # Verify the complete serialization can be opened before exposing the file.
    loaded=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
    if loaded['completed_updates']!=blob['completed_updates']: raise RuntimeError('checkpoint validation failed')
    del loaded
    os.replace(tmp,path)


def save_checkpoint(model,optimizer,update,epoch,sha,*,model_only=False,exposure_counts=None):
    rank,world=rank_world()
    rng=capture_rng();rng['window_exposure_counts']=exposure_counts.tolist() if exposure_counts is not None else None
    all_rng=[None]*world if rank==0 else None
    dist.gather_object(rng,all_rng,dst=0)
    if rank==0:
        metadata=dict(completed_updates=update,completed_exposures=8*update,epoch=epoch,git_sha=sha,
                      recipe='OBJECT_LOCUS_PANOPTIC_V1_8GPU',world_size=8)
        if model_only:
            atomic_save(RUN/f'checkpoint_epoch_{epoch:02}.pt',dict(**metadata,model=model.state_dict()))
        else:
            latest=RUN/'resume_latest.pt';previous=RUN/'resume_previous.pt'
            # Write/validate new recovery first, then rotate only this run's files.
            new=RUN/'resume_new.pt'
            atomic_save(new,dict(**metadata,model=model.state_dict(),optimizer=optimizer.state_dict(),rank_rng=all_rng))
            if latest.exists(): os.replace(latest,previous)
            os.replace(new,latest)
    dist.barrier()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    device=init_distributed();rank,world=rank_world()
    if world!=8: raise RuntimeError('formal training requires eight ranks')
    status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
    if status.strip(): raise RuntimeError('formal training requires a clean pushed checkout')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    receipt=json.loads((REPORTS/'git_provenance.json').read_text())
    if receipt['training_sha']!=sha or not receipt['clean']:
        raise RuntimeError('training SHA does not match submit-host remote verification')
    lines=receipt['remote_verification'].strip().splitlines()
    if not any(line.split()==[sha,'refs/heads/object-locus-panoptic-v1-8gpu'] for line in lines):
        raise RuntimeError('remote branch verification absent')
    if not (REPORTS/'single_smoke.json').is_file() or not (REPORTS/'eight_smoke.json').is_file(): raise RuntimeError('required smoke reports absent')
    for f in ('single_smoke.json','eight_smoke.json'):
        if json.loads((REPORTS/f).read_text())['status']!='PASS': raise RuntimeError(f'failed gate {f}')
    manifest,source_plan,splits,audit=assets()
    model,opt=build_model(device);optimizer=build_optimizer(model)
    update=0;exposure_counts=np.zeros(1008,dtype=np.int64)
    if args.resume:
        blob=torch.load(RUN/'resume_latest.pt',map_location=device,weights_only=False)
        if blob['git_sha']!=sha or blob['world_size']!=8: raise RuntimeError('resume provenance mismatch')
        model.load_state_dict(blob['model'],strict=True);optimizer.load_state_dict(blob['optimizer'])
        update=blob['completed_updates'];restore_rng(blob['rank_rng'][rank])
        exposure_counts=np.asarray(blob['rank_rng'][rank]['window_exposure_counts'],dtype=np.int64);del blob
    elif (RUN/'resume_latest.pt').exists(): raise RuntimeError('existing run needs explicit same-recipe resume')
    if rank==0:
        RUN.mkdir(parents=True,exist_ok=True)
        size=sum(x.numel()*x.element_size() for x in model.state_dict().values())
        required=size*13+2*1024**3 # 7 model + 2*(model + 2 optimizer moments)
        if shutil.disk_usage(RUN).free<required: raise RuntimeError(f'insufficient checkpoint budget: {required} bytes required')
        shutil.copyfile(FRESH/'training_plan.json',REPORTS/'source_training_plan.json')
        write_json(REPORTS/'data_manifest.json',manifest)
        entries=[dict(update=u,epoch=u//126,rank_windows=[sample_index(u//126,u%126,r) for r in range(8)]) for u in range(8064)]
        write_json(REPORTS/'training_plan.json',dict(seed=42,epochs=64,global_updates=8064,exposures=64512,entries=entries))
        write_json(REPORTS/'run_manifest.json',dict(git_sha=sha,main_sha=receipt['main_sha'],
            job_id=os.environ.get('SLURM_JOB_ID'),node=os.uname().nodename,global_batch=8,per_rank_batch=1,
            audit=audit,checkpoint_epochs=REGISTERED,official_epochs=OFFICIAL,optimizer_updates=8064,
            exposure_count=64512,checkpoint_budget_bytes=required,scientific_changes=[],erratum='723→725 excluded MASt3R states',
            comparison='Recipe comparison: batch, LR groups, pretraining and coupling differ; exposure equality does not imply update equality.'))
    dist.barrier()
    if update==0:
        save_checkpoint(model,optimizer,0,0,sha,model_only=True,exposure_counts=exposure_counts)
        error=None
        if rank==0:
            try:
                evaluate_epoch(model,opt,manifest,splits,0,device);parameter_drift(model,0)
            except Exception as exc:error=exc
        synchronized_check(error,device,'eval0',0)
        dist.barrier()
    model.train();start=time.perf_counter()
    while update<8064:
        epoch,k=divmod(update,126);wi=sample_index(epoch,k,rank)
        error=None;batch=None
        try: batch=build_batch(opt,manifest['expanded_train_windows'][wi],device)
        except Exception as exc: error=exc
        synchronized_check(error,device,'batch',update,batch)
        out,row=train_one_step(model,opt,optimizer,batch,update)
        row.update(rank=rank,window_index=wi,identity=manifest['expanded_train_windows'][wi])
        update+=1;exposure_counts[wi]+=1
        if update%10==0 or update%126==0:
            with (REPORTS/f'training_rank{rank}.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        del out,batch
        if update%126==0:
            done=update//126
            save_checkpoint(model,optimizer,update,done,sha,exposure_counts=exposure_counts)
            if done in REGISTERED:
                save_checkpoint(model,optimizer,update,done,sha,model_only=True,exposure_counts=exposure_counts)
                error=None
                if rank==0:
                    try:
                        evaluate_epoch(model,opt,manifest,splits,done,device);parameter_drift(model,done)
                    except Exception as exc: error=exc
                synchronized_check(error,device,'eval',update)
                dist.barrier()
        if rank==0 and (update%10==0 or update%126==0):
            write_json(REPORTS/'progress.json',dict(completed_updates=update,completed_exposures=8*update,epoch=update/126,
                windows_per_second=8*update/(time.perf_counter()-start),updates_per_second=update/(time.perf_counter()-start)))
    global_counts=torch.tensor(exposure_counts,device=device,dtype=torch.int64)
    dist.all_reduce(global_counts,op=dist.ReduceOp.SUM)
    if not torch.all(global_counts==64):raise RuntimeError('actual per-window exposure differs from64')
    if rank==0:
        write_json(REPORTS/'actual_window_exposures.json',dict(counts=global_counts.cpu().tolist(),total=int(global_counts.sum()),per_window=64))
        write_json(REPORTS/'endpoint_status.json',dict(status='TRAINING_AND_REGISTERED_EVAL_COMPLETE',completed_updates=8064,completed_exposures=64512,exposures_per_window=64,comparison='PENDING_REPORT'))
    dist.barrier();dist.destroy_process_group()
    if rank==0:
        endpoint_report()
        write_json(REPORTS/'endpoint_status.json',dict(status='COMPLETE',completed_updates=8064,completed_exposures=64512,exposures_per_window=64))

if __name__=='__main__': main()
