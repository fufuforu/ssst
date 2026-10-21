"""Submit one No-CA full4 run on any single 8-GPU node in the user pool."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_isoadapter_noca_full4_20261021_v1_8gpu')
ALLOWED={f'3dimage-{i}' for i in (12,13,14,17,18)}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Immutable snapshot dirty')
    pre=json.loads((RUN/'PREFLIGHT.json').read_text())
    if pre['status']!='PASSED_NOCA_PREFLIGHT':raise ValueError('No-CA preparation incomplete')
    if (RUN/'STARTED.json').exists() or (RUN/'checkpoint_latest.pt').exists():raise ValueError('Refusing duplicate training')
    for key in ('training_window_manifest','initial_checkpoint','model_config'):
        if digest(pre[key])!=pre[key+'_sha256']:raise ValueError('Asset changed: '+key)
    inventory=set(subprocess.check_output(['sinfo','-h','-N','-p','3090,4090','-o','%N'],text=True).split())
    if not ALLOWED<=inventory:raise ValueError('Requested nodes absent from eligible partitions')
    excluded=sorted(inventory-ALLOWED)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    argv=['sbatch','--parsable','--job-name=P1024isoNoCA','--partition=3090,4090','--nodes=1','--ntasks=1','--gres=gpu:8','--cpus-per-task=32','--mem=192G','--time=1-00:00:00','--signal=B:TERM@60','--output='+str(RUN/'slurm-%j.out')]
    if excluded:argv+=['--exclude='+','.join(excluded)]
    argv+=['scripts/run_point1024_fmquery_full4.sh',str(RUN)]
    # --nodelist with five hosts would require them all; exclusion gives an OR pool.
    test=subprocess.run([argv[0],'--test-only']+argv[2:],cwd=ROOT,capture_output=True,text=True,timeout=30)
    if test.returncode:raise ValueError('Scheduler preflight failed: '+test.stderr)
    (RUN/'SCHEDULER_TEST_ONLY.json').write_text(json.dumps(dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr,allowed=sorted(ALLOWED),excluded=excluded),indent=2)+'\n')
    with (RUN/'SUBMISSION_REQUESTED.json').open('x') as f:json.dump(dict(code_sha=sha,training_only=True),f)
    (RUN/'code.sha').write_text(sha+'\n')
    job=int(subprocess.check_output(argv,cwd=ROOT,text=True).strip().split(';')[0])
    record=dict(training_job=job,evaluation_job=None,summary_job=None,architecture=pre['architecture'],ablation=pre['ablation'],code_root=str(ROOT),code_sha=sha,run_root=str(RUN),eligible_nodes=sorted(ALLOWED),excluded_nodes=excluded,gpus=8,nodes=1,global_batch=8,local_batch=1,epochs=4,full_updates=4172,context_views=2,novel_views=4,initial_updates=0,gpu_first_step_verified=False,submission_argv=argv,stop_polling_after_submission=True)
    (RUN/'SUBMITTED.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        p=subprocess.run(['squeue','-j',str(job),'-o','%.18i %.18j %.10T %.6D %R'],capture_output=True,text=True,timeout=15);status=p.stdout+p.stderr
    except subprocess.TimeoutExpired:status='Bounded query timed out; sbatch accepted '+str(job)+'\n'
    (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);record['bounded_status_check']=status;print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
