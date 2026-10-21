"""Dedicated smoke-only submission; exactly one bounded acceptance check."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_isoadapter_gpu_smoke_20261021_v1')
SOURCE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')
SHA='3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744'

def main():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code','HEAD','--','tokengs','scripts','tests'],cwd=ROOT,check=True)
    if json.loads((RUN/'PREFLIGHT.json').read_text())['status']!='PASSED_CPU_PREFLIGHT':raise ValueError('CPU preparation incomplete')
    with (RUN/'SUBMISSION_REQUESTED.json').open('x') as stream:json.dump(dict(code_sha=sha,smoke_only=True),stream)
    if not (RUN/'checkpoint_untrained_initial.pt').exists():os.link(SOURCE,RUN/'checkpoint_untrained_initial.pt')
    (RUN/'untrained_initial.sha256').write_text(SHA+'\n');(RUN/'code.sha').write_text(sha+'\n')
    argv=['sbatch','--parsable','--job-name=P1024isoSmoke','--partition=3090,4090',
        '--exclude=3dimage-11,3dimage-12,3dimage-15,3dimage-16,3dimage-20,3dimage-21',
        '--nodes=1','--gres=gpu:1','--cpus-per-task=8','--mem=128G','--time=01:30:00','--signal=B:TERM@60',
        '--output='+str(RUN/'slurm-%j.out'),'scripts/run_point1024_fmquery_smoke.sh',str(RUN)]
    job=int(subprocess.check_output(argv,cwd=ROOT,text=True).strip().split(';')[0])
    record=dict(smoke_job=job,code_root=str(ROOT),code_sha=sha,run_root=str(RUN),gpus=1,world_size=1,
        global_batch=8,accumulation=8,optimizer_updates=2,unique_windows=1,eligible_nodes=[13,14,17,18],
        full_training_submitted=False,evaluation_submitted=False,submission_argv=argv,stop_polling_after_submission=True)
    (RUN/'SUBMITTED.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        result=subprocess.run(['squeue','-j',str(job),'-o','%.18i %.16j %.10T %.6D %R'],capture_output=True,text=True,timeout=15)
        status=result.stdout+result.stderr
    except subprocess.TimeoutExpired:status='Bounded query timed out; sbatch accepted '+str(job)+'\n'
    (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);record['bounded_status_check']=status
    print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
