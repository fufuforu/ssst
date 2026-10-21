"""Formal three-arm Object-Locus GC sweep. Each Slurm task is a fresh arm."""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from scripts.object_locus_gc_sweep_runtime import *

def checkpoint(model,optimizer,u,alpha,source,plan_sha,code_sha,counts):
    rank,world=rank_world(); state=capture_rng();state['window_exposure_counts']=counts.tolist()
    all_rng=[None]*world if rank==0 else None
    dist.gather_object(state,all_rng,dst=0)
    if rank==0:
        arm=os.environ['TASK_ARM']; path=RUN_ROOT/arm/f'checkpoint_epoch{u//UPDATES_PER_EPOCH}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rank_rng':all_rng,'completed_updates':u,'new_exposures':u*8,'source_exposure':source['completed_exposures'],'model_exposure':source['completed_exposures']+u*8,'epoch':u//UPDATES_PER_EPOCH,'alpha':alpha,'config':dataclasses.asdict(model.opt),'plan_sha256':plan_sha,'source_checkpoint':source,'code_sha':code_sha,'world_size':world}
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp)
        check=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
        if check['completed_updates']!=u or check['alpha']!=alpha or check['plan_sha256']!=plan_sha:raise RuntimeError('checkpoint verification failed')
        del check;os.replace(tmp,path)
    dist.barrier()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=tuple(ARMS),required=True);args=ap.parse_args()
    os.environ['TASK_ARM']=args.arm
    device=init_distributed();rank,world=rank_world()
    if world!=8:raise RuntimeError('formal job requires eight ranks')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise RuntimeError('formal worktree must be clean')
    receipt=json.loads((REPORT_ROOT/'git_provenance.json').read_text())
    if receipt.get('training_sha')!=code or not receipt.get('clean'):raise RuntimeError('local pushed provenance mismatch')
    plan_path=REPORT_ROOT/'training_plan.json';plan_sha=sha256(plan_path)
    if receipt.get('plan_sha256')!=plan_sha:raise RuntimeError('plan SHA mismatch')
    for gate in ('single_smoke.json','eight_smoke.json'):
        if json.loads((REPORT_ROOT/gate).read_text()).get('status')!='PASS':raise RuntimeError('smoke gate missing or failed: '+gate)
    manifest=json.loads(SOURCE_MANIFEST.read_text());plan=json.loads(plan_path.read_text());alpha=ARMS[args.arm]
    arm_report=REPORT_ROOT/args.arm;arm_run=RUN_ROOT/args.arm
    if rank==0:
        arm_report.mkdir(parents=True,exist_ok=True);arm_run.mkdir(parents=True,exist_ok=True)
        if any(arm_run.glob('checkpoint_epoch*.pt')):raise RuntimeError('formal arm output exists; refusing overwrite')
    dist.barrier()
    model,opt,source=build_model(device);optimizer=build_optimizer(model);counts=np.zeros(WINDOWS,dtype=np.int64);u=0
    if rank==0:
        write_json(arm_report/'run_manifest.json',{'arm':args.arm,'alpha':alpha,'unique_scientific_variable':'alpha','git_sha':code,'source_checkpoint':source,'source_manifest_path':str(SOURCE_MANIFEST),'source_manifest_sha256':plan['source_manifest_sha256'],'plan_sha256':plan_sha,'scenes':128,'windows':1008,'epochs':8,'updates':1008,'new_exposures':8064,'model_exposure_endpoint':58128,'checkpoint_epochs':[0,2,4,8],'scene_independence':plan['dev_val_scene_intersections'],'optimizer':'AdamW fresh moments, betas=(0.9,0.95), eps=1e-8, FP32, TF32 disabled, clip=1.0'})
        with (arm_report/'training_rank0.jsonl').open('w') as f: pass
        write_json(arm_report/'progress.json',{'arm':args.arm,'status':'INITIALIZING','completed_updates':0,'new_exposures':0,'model_exposure':SOURCE_EXPOSURES})
    dist.barrier();checkpoint(model,optimizer,0,alpha,source,plan_sha,code,counts)
    model.train();torch.cuda.reset_peak_memory_stats()
    while u<TOTAL_UPDATES:
        entry=plan['entries'][u];wi=entry['rank_windows'][rank];batch=None;error=None
        try:batch=build_batch(opt,manifest['expanded_train_windows'][wi],device)
        except Exception as e:error=e
        synchronized_error(error,device,'batch',u)
        row_out,row=train_step(model,opt,optimizer,batch,u,alpha,diagnose=((u+1)%100==0))
        counts[wi]+=1;u+=1
        checkpoint_node=u in (252,504,1008)
        if checkpoint_node:checkpoint(model,optimizer,u,alpha,source,plan_sha,code,counts)
        log_node=u==1 or u%10==0 or checkpoint_node
        if log_node:
            # Logged scalar losses are the arithmetic mean over the eight rank batches.
            vals=torch.tensor([row['loss_recon'],row['loss_understanding'],row['monitor_total_loss']],device=device,dtype=torch.float64)
            dist.all_reduce(vals,op=dist.ReduceOp.SUM);vals/=world
            if rank==0:
                row.update(loss_recon=float(vals[0]),loss_understanding=float(vals[1]),monitor_total_loss=float(vals[2]),local_epoch=u/126,local_update=u,new_exposures=8*u,model_exposure=SOURCE_EXPOSURES+8*(u-1),endpoint_model_exposure=SOURCE_EXPOSURES+8*u,alpha=alpha,window_index=wi,window_identity=manifest['expanded_train_windows'][wi],recent_checkpoint=str(RUN_ROOT/args.arm/f'checkpoint_epoch{max(x for x in (0,2,4,8) if x*126<=u)}.pt'))
                with (arm_report/'training_rank0.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                write_json(arm_report/'progress.json',{'arm':args.arm,'status':'TRAINING','completed_updates':u,'local_epoch':u/126,'new_exposures':8*u,'model_exposure':row['model_exposure'],'endpoint_model_exposure':SOURCE_EXPOSURES+8*u,'loss_recon':row['loss_recon'],'loss_understanding':row['loss_understanding'],'understanding_weight':row['understanding_weight'],'monitor_total_loss':row['monitor_total_loss'],'alpha':alpha,'job_id':os.environ.get('SLURM_JOB_ID'),'recent_checkpoint':row['recent_checkpoint']})
            if u==1 and rank==0:
                write_json(arm_report/'startup_confirmation.json',{'status':'FIRST_FORMAL_UPDATE_FINITE','arm':args.arm,'alpha':alpha,'job_id':os.environ.get('SLURM_JOB_ID'),'epoch0_checkpoint':str(RUN_ROOT/args.arm/'checkpoint_epoch0.pt'),'completed_updates':1,'new_exposures':8,'model_exposure':row['model_exposure'],'endpoint_model_exposure':SOURCE_EXPOSURES+8,'finite':True,'loss_recon':float(vals[0]),'loss_understanding':float(vals[1]),'code_sha':code,'plan_sha256':plan_sha})
            dist.barrier()
        del row_out,batch
    totals=torch.tensor(counts,device=device,dtype=torch.int64);dist.all_reduce(totals)
    if totals.cpu().tolist()!=plan['window_exposure_counts']:raise RuntimeError('formal exposure plan mismatch')
    if rank==0:
        write_json(arm_report/'progress.json',{'arm':args.arm,'status':'COMPLETE','completed_updates':u,'new_exposures':u*8,'model_exposure':SOURCE_EXPOSURES+u*8,'job_id':os.environ.get('SLURM_JOB_ID'),'recent_checkpoint':str(RUN_ROOT/args.arm/'checkpoint_epoch8.pt')})
        write_json(arm_report/'training_complete.json',{'arm':args.arm,'completed_updates':u,'new_exposures':u*8,'model_exposure':SOURCE_EXPOSURES+u*8,'window_exposure_counts':totals.cpu().tolist(),'evaluation':'WAIT_USER_NOTIFICATION'})
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
