"""Submit the authorized 8-GPU full run only after the GPU verifier passes.

No polling: one bounded acceptance check after submission. Full training also
requires afterok of the smoke job and rechecks this gate on startup.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def check_gate(full):
    config=json.loads((full/'configuration.json').read_text())
    smoke=Path(config['smoke_root'])
    receipt=json.loads((smoke/'SMOKE_PASSED.json').read_text())
    if not (receipt['status']=='PASSED' and receipt['gpu_forward_backward_verified'] and receipt['ranks']==8 and receipt['anchors']==1024 and not receipt['color_prior']):
        raise ValueError('GPU smoke has not passed')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if sha!=config['code_sha'] or sha!=(smoke/'code.sha').read_text().strip():raise ValueError('Smoke/full code differs')
    return config,smoke

def submit(full):
    config,smoke=check_gate(full)
    smoke_job=int(os.environ['SLURM_JOB_ID'])
    if json.loads((smoke/'SUBMITTED.json').read_text())['smoke_job']!=smoke_job:raise ValueError('Wrong smoke allocation')
    # Exclusive marker prevents repeated submissions even if sbatch receipt is lost.
    with (full/'SUBMISSION_REQUESTED.json').open('x') as stream:
        json.dump(dict(smoke_job=smoke_job,gate=str(smoke/'SMOKE_PASSED.json')),stream,indent=2)
    argv=['sbatch','--parsable','--job-name=FPSsp16','--partition=3090,4090',
        '--exclude=3dimage-11,3dimage-12,3dimage-15,3dimage-16,3dimage-20,3dimage-21',
        '--nodes=1','--gres=gpu:8','--cpus-per-task=32','--mem=192G','--time=48:00:00',
        '--kill-on-invalid-dep=yes','--dependency=afterok:'+str(smoke_job),
        '--output='+str(full/'slurm-%j.out'),'scripts/run_fps_superpoint.sh',str(full),'full',str(full)]
    job=int(subprocess.check_output(argv,cwd=ROOT,text=True).strip().split(';')[0])
    record=dict(train_job=job,smoke_job=smoke_job,dependency='afterok:'+str(smoke_job),
        gpus=8,eligible_nodes=[13,14,17,18],code_sha=config['code_sha'],submission_argv=argv,
        stop_polling_after_submission=True)
    (full/'SUBMITTED.json').write_text(json.dumps(record,indent=2)+'\n')
    # Exactly one check; no retry if the scheduler query is unavailable.
    try:
        status=subprocess.run(['squeue','-j',str(job),'-o','%.18i %.12j %.10T %.6D %R'],capture_output=True,text=True,timeout=15)
        (full/'STATUS_AFTER_SUBMIT.txt').write_text(status.stdout+status.stderr)
    except subprocess.TimeoutExpired:
        (full/'STATUS_AFTER_SUBMIT.txt').write_text('Bounded acceptance query timed out; sbatch accepted job '+str(job)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-gate',action='store_true');parser.add_argument('full',type=Path);args=parser.parse_args()
    if args.check_gate:check_gate(args.full)
    else:submit(args.full)
