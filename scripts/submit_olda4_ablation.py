"""Submit only authorized old A4 V4init/NoCA: smoke -> full4 -> official1860, one bounded check."""
from pathlib import Path
import hashlib,json,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/oldA4_v4init_noca_ablation_20261022_v1')

def main():
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty execution snapshot'
 assert json.loads((RUN/'CPU_VALIDATION.json').read_text())['status']=='PASSED'
 assert not (RUN/'SUBMITTED.json').exists() and not (RUN/'SUBMISSION_REQUESTED.json').exists()
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 for branch in ('V4init','NoCA'):
  for stage in ('smoke','train','eval'):
   r=RUN/branch/stage;cfg=json.loads((r/'configuration.json').read_text());cfg['code_sha']=sha
   assert not (r/'STARTED.json').exists() and not (r/'SMOKE_STARTED.json').exists()
   assert hashlib.sha256(Path(cfg['manifest']).read_bytes()).hexdigest()==cfg['manifest_sha256']
   cfg_path=r/'configuration.json';cfg_path.write_text(json.dumps(cfg,indent=2)+'\n');(r/'code.sha').write_text(sha+'\n')
 common=['--partition=3090,4090','--exclude=3dimage-11','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60']
 # All preparation, scheduler argument checks and immutable code sealing precede first submission.
 for gpus,cpus,mem,limit in [(1,8,'128G','01:30:00'),(8,32,'192G','1-00:00:00'),(8,32,'192G','08:00:00')]:
  check=subprocess.run(['sbatch','--test-only']+common+[f'--gres=gpu:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={limit}','scripts/run_olda4_ablation.sh',str(RUN/'V4init/smoke'),'smoke'],cwd=ROOT,capture_output=True,text=True,timeout=30)
  assert check.returncode==0,check.stderr
 receipt=dict(code_root=str(ROOT),code_sha=sha,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),eligible_nodes=[12,13,14,17,18],branches={},stop_polling_after_single_check=True)
 (RUN/'SUBMISSION_REQUESTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
 def save():(RUN/'SUBMITTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
 save()
 for branch in ('V4init','NoCA'):
  jobs={};receipt['branches'][branch]=jobs
  for stage,gpus,cpus,mem,limit,dependency in [('smoke',1,8,'128G','01:30:00',None),('train',8,32,'192G','1-00:00:00','smoke'),('eval',8,32,'192G','08:00:00','train')]:
   r=RUN/branch/stage;argv=['sbatch','--parsable']+common+[f'--job-name=A4{branch}_{stage}',f'--gres=gpu:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={limit}',f'--output={r}/slurm-%j.out']
   if dependency:argv+=[f'--dependency=afterok:{jobs[dependency]}','--kill-on-invalid-dep=yes']
   argv+=['scripts/run_olda4_ablation.sh',str(r),stage]
   jobs[stage]=int(subprocess.check_output(argv,cwd=ROOT,text=True).strip().split(';')[0]);save()
   (r/'SUBMITTED.json').write_text(json.dumps(dict(job=jobs[stage],stage=stage,dependency=jobs.get(dependency),submission_argv=argv,code_sha=sha),indent=2)+'\n')
 print(json.dumps(receipt,indent=2),flush=True)
 # Exactly one bounded scheduler acceptance check for all six jobs.
 ids=','.join(str(job) for jobs in receipt['branches'].values() for job in jobs.values())
 try:
  q=subprocess.run(['squeue','-j',ids,'-o','%.18i %.18j %.10T %.15P %.20N %R'],capture_output=True,text=True,timeout=20);status=q.stdout+q.stderr
 except subprocess.TimeoutExpired:status='Single bounded query timed out; sbatch IDs above accepted.\n'
 (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);print(status,flush=True)

if __name__=='__main__':main()
