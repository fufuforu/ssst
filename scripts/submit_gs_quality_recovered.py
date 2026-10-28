"""Submit only CPU fits -> paired export -> score, never redo cached forwards."""
from pathlib import Path
import json,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline4_gs_quality_20261024_v4_cachefinalized')
SPECS=[('fit','gs_fit',0,12,'64G','06:00:00',[]),('export','gs_export',4,24,'160G','06:00:00',['fit']),('score','gs_score',1,8,'96G','10:00:00',['export'])]
def save(name,value):(RUN/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty snapshot')
    cfg=json.loads((RUN/'configuration.json').read_text());proof=json.loads((RUN/'CACHE_RECOVERY.json').read_text())
    if cfg['code_sha']!=code or proof['status']!='PASSED' or proof['official_forward_hashes_verified']!=1860 or json.loads((RUN/'PREFLIGHT.json').read_text())['status']!='PASSED':raise ValueError('Preflight incomplete')
    if (RUN/'SUBMISSION_ATTEMPT.json').exists():raise FileExistsError('Existing attempt, inspect rather than duplicate')
    plans=[];dry=[]
    for stage,mode,gpu,cpu,mem,limit,deps in SPECS:
        cmd=['--partition=3090','--nodelist=3dimage-11','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60','--kill-on-invalid-dep=yes',f'--job-name=Qrecover_{stage}',f'--cpus-per-task={cpu}',f'--mem={mem}',f'--time={limit}',f'--output={RUN}/{stage}-%j.out']
        if gpu:cmd.append(f'--gres=gpu:{gpu}')
        script=['scripts/run_reciprocity_reconstruction.sh',str(RUN),mode,'']
        result=subprocess.run(['sbatch','--test-only']+cmd+script,cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
        plans.append((cmd,script));dry.append(dict(stage=stage,stdout=result.stdout,stderr=result.stderr))
    save('SLURM_TEST_ONLY.json',dry)
    with (RUN/'SUBMISSION_ATTEMPT.json').open('x') as f:json.dump(dict(code_sha=code,time=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
    ids={};jobs=[]
    for spec,(cmd,script) in zip(SPECS,plans):
        stage,mode,gpu,cpu,mem,limit,deps=spec;dependencies=[ids[k] for k in deps];dep=['--dependency=afterok:'+':'.join(map(str,dependencies))] if dependencies else []
        result=subprocess.run(['sbatch','--parsable']+cmd+dep+script,cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
        job=int(result.stdout.strip().split(';')[0]);ids[stage]=job;jobs.append(dict(stage=stage,job_id=job,node='3dimage-11',gpus=gpu,cpus=cpu,memory=mem,dependencies=dependencies,time_limit=limit));save('SUBMITTED.json',dict(code_sha=code,jobs=jobs))
    try:status=subprocess.check_output(['squeue','-j',','.join(str(j['job_id']) for j in jobs),'-o','%.12i %.12T %.20N %.25b %.45R'],text=True,timeout=30)
    except subprocess.TimeoutExpired:status='Single bounded acceptance query timed out; IDs saved.\n'
    (RUN/'ACCEPTANCE_STATUS.txt').write_text(status);print(json.dumps(jobs,indent=2));print(status);print('已停止轮询，等你通知完成后再检查。')
if __name__=='__main__':main()
