"""No evaluation: necessary smoke or independently scheduled formal arm."""
import argparse, json, os, subprocess, traceback
import torch
import torch.distributed as dist
from scripts.object_locus_image_memory_runtime import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--mode',choices=('single','four','train'),required=True);parser.add_argument('--resume')
    args=parser.parse_args();device=init_device();rank,world=base.rank_world()
    manifest,plan,plan_sha=prepare_plan();report=ROOT/args.arm
    base.REPORT_ROOT=report;os.environ['TASK_ARM']=args.arm
    code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip()
    if args.mode=='train':
        receipt=json.loads((ROOT/'git_provenance.json').read_text())
        if receipt['training_sha']!=code_sha or receipt['plan_sha256']!=plan_sha or not receipt['remote_verified']:raise RuntimeError('launch provenance mismatch')
        if subprocess.check_output(['git','status','--porcelain'],cwd=base.REPO,text=True).strip():raise RuntimeError('dirty formal checkout')
        for gate in ('cpu_contracts.json','single_smoke.json','four_smoke_c32.json','four_smoke_u128.json','checkpoint_loading_contract.json','checkpoint_recovery_contract.json'):
            if json.loads((ROOT/gate).read_text())['status']!='PASS':raise RuntimeError('missing required gate '+gate)
    model,opt,optimizer,identity=construct(args.arm,device);model.train();torch.cuda.reset_peak_memory_stats()
    counts=[0]*plan['N'];start=0
    if args.resume:start,counts=restore(args.resume,model,optimizer,args.arm,plan_sha)
    if rank==0:write_json(report/(args.mode+'_initialization.json'),dict(source=identity,arm=args.arm,object_image_memory_size=ARMS[args.arm],fresh_optimizer=not bool(args.resume),code_sha=code_sha))
    single=args.mode=='single';limit=1 if single else 2 if args.mode=='four' else plan['total_updates']
    rows=[]
    for step in range(start,limit):
        epoch=step//plan['U'];offset=step%plan['U'];ids=plan['orders'][epoch][8*offset:8*offset+8]
        # Single smoke uses first real window but nonzero warm-up to exercise understanding backward.
        train_step=1 if single else step
        row=update(model,opt,optimizer,manifest['windows'],ids,train_step,plan['total_updates'],single)
        for wi in row['rank_windows']:counts[wi]+=1
        rows.append(row)
        if rank==0 and (args.mode!='train' or (step+1)%10==0):
            row.update(epoch=epoch,new_update=step+1,global_window_ids=ids,code_sha=code_sha)
            print(json.dumps(row),flush=True)
            with (report/(args.mode+'_updates.jsonl')).open('a') as stream:stream.write(json.dumps(row)+'\n')
        if args.mode=='train':
            if step==9:
                actual=torch.tensor(counts,device=device,dtype=torch.int64)
                dist.all_reduce(actual)
                expected=torch.tensor(np.bincount(plan['orders'][0][:80],minlength=plan['N']),device=device)
                if not torch.equal(actual,expected):raise RuntimeError('startup exposure counts differ from fixed plan')
            if step==9 and rank==0:
                write_json(report/'startup_confirmation.json',dict(status='PASS',job_id=os.environ.get('SLURM_JOB_ID'),arm=args.arm,code_sha=code_sha,
                    plan_sha256=plan_sha,fresh_initialization=not bool(args.resume),completed_new_updates=10,completed_new_exposures=80,
                    source_checkpoint_sha256=base.SOURCE_SHA256,finite=True,window_ids=plan['orders'][0][:80],last_update=row))
            if (step+1)%plan['U']==0:save_checkpoint(model,opt,optimizer,args.arm,epoch+1,step+1,plan_sha,code_sha,counts)
    if args.mode!='train':
        from scripts.smoke_object_locus_gc_sweep import digest_state,digest_optimizer
        digest=[digest_state(model),digest_optimizer(model,optimizer)];digests=[None]*world
        if world>1:dist.all_gather_object(digests,digest)
        else:digests=[digest]
        if any(value!=digests[0] for value in digests):raise RuntimeError('rank model or optimizer desynchronized')
        all_rows=[None]*world
        if world>1:dist.all_gather_object(all_rows,rows)
        else:all_rows=[rows]
        if rank==0:write_json(ROOT/('single_smoke.json' if single else f'four_smoke_{args.arm}.json'),
            dict(status='PASS',discarded=True,arm=args.arm,code_sha=code_sha,updates=limit,ranks=world,global_batch=1 if single else 8,
                 synchronized=True,source=identity,per_rank=all_rows))
    elif rank==0:write_json(report/'training_complete.json',dict(status='PASS',completed_new_updates=limit,completed_new_exposures=8*limit,code_sha=code_sha))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except Exception:
        print(traceback.format_exc(),flush=True);raise
