"""Submit the two authorized four-card pilot arms; one acceptance check only."""
import subprocess,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/membership_guided_geometry_20261023_v2_sdpafix')

def main():
    assert json.loads((RUN/'PREFLIGHT.json').read_text())['status']=='PASSED'
    assert json.loads((RUN/'CPU_TESTS.json').read_text())['status']=='PASSED'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert json.loads((RUN/'configuration.json').read_text())['code_sha']==code
    common=['--partition=3090,4090','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60','--kill-on-invalid-dep=yes']
    specs=[('cache','',4,24,'160G','12:00:00',[]),('train','G1',4,24,'160G','16:00:00',['cache']),
        ('train','G2',4,24,'160G','16:00:00',['cache']),('export','G1',4,24,'160G','06:00:00',['trainG1']),
        ('export','G2',4,24,'160G','06:00:00',['trainG2']),('score','',1,8,'96G','06:00:00',['exportG1','exportG2'])]
    commands=[];tests=[]
    for mode,arm,gpu,cpu,mem,limit,deps in specs:
        stage=mode+arm
        command=common+[f'--job-name=MGeom_{stage}',f'--cpus-per-task={cpu}',f'--mem={mem}',f'--time={limit}',f'--gres=gpu:{gpu}',f'--output={RUN}/{stage}-%j.out']
        commands.append(command)
        result=subprocess.run(['sbatch','--test-only']+command+['scripts/run_membership_geometry.sh',str(RUN),mode,arm],cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
        tests.append(dict(stage=stage,stdout=result.stdout,stderr=result.stderr))
    (RUN/'SLURM_TEST_ONLY.json').write_text(json.dumps(tests,indent=2)+'\n')
    with (RUN/'SUBMISSION_ATTEMPT.json').open('x') as f:json.dump(dict(code_sha=code,time=datetime.now(timezone.utc).isoformat()),f)
    (RUN/'code.sha').write_text(code+'\n');records=[];ids={}
    for command,spec in zip(commands,specs):
        mode,arm,gpu,cpu,mem,limit,deps=spec;stage=mode+arm;dependency=[ids[d] for d in deps]
        dep=['--dependency=afterok:'+':'.join(map(str,dependency))] if dependency else []
        result=subprocess.run(['sbatch','--parsable']+command+dep+['scripts/run_membership_geometry.sh',str(RUN),mode,arm],cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
        job=int(result.stdout.strip().split(';')[0]);ids[stage]=job;records.append(dict(stage=stage,job_id=job,gpus=gpu,cpus=cpu,memory=mem,time_limit=limit,dependencies=dependency,node='flexible_3090_4090_single_node'))
        (RUN/'SUBMITTED.json').write_text(json.dumps(dict(code_sha=code,jobs=records),indent=2)+'\n')
    try:status=subprocess.check_output(['squeue','-j',','.join(str(r['job_id']) for r in records),'-o','%.12i %.12T %.20N %.8C %.25b %.35R'],text=True,timeout=30)
    except subprocess.TimeoutExpired:status='Single bounded acceptance check timed out; sbatch IDs saved.\n'
    (RUN/'ACCEPTANCE_STATUS.txt').write_text(status)
    print(json.dumps(records,indent=2));print(status);print('已停止轮询，等你通知完成后再检查。')

if __name__=='__main__':main()
