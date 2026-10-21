"""Submit the prepared smoke once, then make exactly one bounded status check."""
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
SMOKE=BASE/'fps1024_superpoint_smoke_20261021_v1'
FULL=BASE/'fps1024_superpoint_single16_20261021_v1'

def main():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code','HEAD','--','tokengs','scripts','tests'],cwd=ROOT,check=True)
    for root in (SMOKE,FULL):
        if json.loads((root/'configuration.json').read_text())['code_sha']!=sha or (root/'code.sha').read_text().strip()!=sha:raise ValueError('Code preparation differs')
    preflight=json.loads((SMOKE/'PREFLIGHT.json').read_text())
    if preflight['status']!='PASSED_CPU_PREFLIGHT':raise ValueError('CPU preparation incomplete')
    with (SMOKE/'SUBMISSION_REQUESTED.json').open('x') as stream:
        json.dump(dict(code_sha=sha,full_training_authorized_after_gpu_smoke=True),stream,indent=2)
    argv=['sbatch','--parsable','--job-name=FPSsp_smoke','--partition=3090,4090',
        '--exclude=3dimage-11,3dimage-12,3dimage-15,3dimage-16,3dimage-20,3dimage-21',
        '--nodes=1','--gres=gpu:8','--cpus-per-task=32','--mem=192G','--time=01:30:00',
        '--output='+str(SMOKE/'slurm-%j.out'),'scripts/run_fps_superpoint.sh',str(SMOKE),'smoke',str(FULL)]
    job=int(subprocess.check_output(argv,cwd=ROOT,text=True).strip().split(';')[0])
    record=dict(smoke_job=job,code_sha=sha,code_root=str(ROOT),smoke_root=str(SMOKE),full_root=str(FULL),gpus=8,
        eligible_nodes=[13,14,17,18],submission_argv=argv,full_job_submitted=False,
        full_training_auto_submitted_only_after_gpu_smoke_pass=True,stop_polling_after_submission=True)
    (SMOKE/'SUBMITTED.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        status=subprocess.run(['squeue','-j',str(job),'-o','%.18i %.12j %.10T %.6D %R'],capture_output=True,text=True,timeout=15)
        (SMOKE/'STATUS_AFTER_SUBMIT.txt').write_text(status.stdout+status.stderr)
        record['bounded_status_check']=status.stdout+status.stderr
    except subprocess.TimeoutExpired:
        (SMOKE/'STATUS_AFTER_SUBMIT.txt').write_text('Acceptance query timed out; sbatch accepted '+str(job)+'\n')
        record['bounded_status_check']='query timed out; sbatch accepted'
    print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
