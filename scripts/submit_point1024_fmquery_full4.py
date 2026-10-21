"""Submit only the authorized Point1024 2+4 full4 training, then query once."""
import datetime,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_ROOT=Path('/space/mawb/ssst_point1024_fmquery_2ctx4novel_full4_code_20261021_v1_node13')
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_2ctx4novel_full4_20261021_v1_node13_8gpu')
SMOKE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_2ctx4novel_gpu_smoke_20261021_v1_node13')

def main():
    if ROOT!=EXPECTED_ROOT:raise ValueError('Use dedicated immutable full4 snapshot')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    from scripts.verify_point1024_fmquery_smoke import verify
    smoke=verify(SMOKE)
    if smoke!=json.loads((SMOKE/'SMOKE_PASSED.json').read_text()):raise ValueError('Smoke evidence differs')
    pre=json.loads((RUN/'PREFLIGHT.json').read_text())
    if pre['status']!='PASSED_CPU_AND_GPU_SMOKE' or pre['full_updates']!=4172 or pre['initial_updates']!=0:raise ValueError('Preflight incomplete')
    if (RUN/'STARTED.json').exists() or (RUN/'checkpoint_latest.pt').exists():raise ValueError('Refusing duplicate training')
    for key in ('training_window_manifest','initial_checkpoint','model_config'):
        if hashlib.sha256(Path(pre[key]).read_bytes()).hexdigest()!=pre[key+'_sha256']:raise ValueError('Changed preflight asset: '+key)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    with (RUN/'SUBMISSION_REQUESTED.json').open('x') as f:json.dump(dict(code_sha=sha,training_only=True),f)
    (RUN/'code.sha').write_text(sha+'\n')
    argv=['sbatch','--parsable','--job-name=P1024iso2p4A4','--partition=3090','--nodelist=3dimage-13',
        '--nodes=1','--ntasks=1','--gres=gpu:3090:8','--cpus-per-task=32','--mem=192G',
        '--time=1-00:00:00','--signal=B:TERM@60','--output='+str(RUN/'slurm-%j.out'),
        'scripts/run_point1024_fmquery_full4.sh',str(RUN)]
    job=int(subprocess.check_output(argv,cwd=ROOT,text=True).strip().split(';')[0])
    record=dict(training_job=job,evaluation_job=None,summary_job=None,node='3dimage-13',gpus=8,world_size=8,
        global_batch=8,local_batch=1,epochs=4,full_updates=4172,context_views=2,novel_views=4,
        code_root=str(ROOT),code_sha=sha,run_root=str(RUN),smoke_job=69992,initial_updates=0,
        submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),submission_argv=argv,
        stop_polling_after_submission=True,gpu_first_step_verified=False,multirank_synchronization_verified=False)
    (RUN/'SUBMITTED.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        result=subprocess.run(['squeue','-j',str(job),'-o','%.18i %.18j %.10T %.6D %R'],capture_output=True,text=True,timeout=15)
        status=result.stdout+result.stderr
    except subprocess.TimeoutExpired:status='Bounded query timed out; sbatch accepted '+str(job)+'\n'
    (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);record['bounded_status_check']=status
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
