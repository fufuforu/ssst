"""One frozen-token cache -> head-only fit -> paired export -> score chain."""
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline4_iou_mlp_20261023_v1')


def main():
    assert json.loads((RUN/'PREFLIGHT.json').read_text())['status']=='PASSED'
    assert json.loads((RUN/'CPU_TESTS.json').read_text())['status']=='PASSED'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    common=['--partition=3090,4090','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60','--kill-on-invalid-dep=yes']
    specs=[('cache',6,24,'192G','12:00:00'),('fit',0,8,'32G','02:00:00'),('export',6,24,'192G','12:00:00'),('score',1,8,'96G','02:00:00')]
    commands=[];tests=[]
    for mode,gpu,cpu,mem,limit in specs:
        command=common+[f'--job-name=FrozenIoU_{mode}',f'--cpus-per-task={cpu}',f'--mem={mem}',f'--time={limit}',f'--output={RUN}/{mode}-%j.out']
        if gpu:command.append(f'--gres=gpu:{gpu}')
        if mode in ('cache','export'):command.append('--nodelist=3dimage-18')
        commands.append(command)
        tested=subprocess.run(['sbatch','--test-only']+command+['scripts/run_frozen_iou_mlp.sh',str(RUN),mode],cwd=ROOT,
                              check=True,capture_output=True,text=True,timeout=30)
        tests.append(dict(mode=mode,stdout=tested.stdout,stderr=tested.stderr))
    (RUN/'SLURM_TEST_ONLY.json').write_text(json.dumps(tests,indent=2)+'\n')
    with (RUN/'SUBMISSION_ATTEMPT.json').open('x') as f:json.dump(dict(code_sha=sha,time=datetime.now(timezone(timedelta(hours=8))).isoformat()),f)
    (RUN/'code.sha').write_text(sha+'\n')
    records=[];previous=None
    for command,spec in zip(commands,specs):
        mode,gpu,cpu,mem,limit=spec;dependency=[f'--dependency=afterok:{previous}'] if previous else []
        result=subprocess.run(['sbatch','--parsable']+command+dependency+['scripts/run_frozen_iou_mlp.sh',str(RUN),mode],
                              cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
        job=int(result.stdout.strip().split(';')[0]);records.append(dict(stage=mode,job_id=job,gpus=gpu,cpus=cpu,memory=mem,time_limit=limit,dependency=previous))
        previous=job
        (RUN/'SUBMITTED.json').write_text(json.dumps(dict(code_sha=sha,jobs=records),indent=2)+'\n')
    try:
        status=subprocess.check_output(['squeue','-j',','.join(str(r['job_id']) for r in records),'-o','%.12i %.12T %.20N %.8C %.25b %.35R'],text=True,timeout=30)
    except subprocess.TimeoutExpired:status='Single bounded acceptance check timed out; sbatch IDs saved.\n'
    (RUN/'ACCEPTANCE_STATUS.txt').write_text(status)
    print(json.dumps(records,indent=2));print(status);print('已停止轮询，等你通知完成后再检查。')


if __name__=='__main__':main()
