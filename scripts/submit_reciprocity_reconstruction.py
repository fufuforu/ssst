"""Submit all prepared authorized chains, then exactly one bounded acceptance query."""
from pathlib import Path
import sys,json,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
M=BASE/'membership_guided_geometry_20261024_v3_ddpgate';Q=BASE/'frozen_baseline4_gs_quality_20261024_v3_freshpair';R=BASE/'baseline16_reconstruction_gap_20261024_v1'

def specs():
    # stage, run, mode, arm, GPUs, CPU, memory, time, prior keys, node
    return [
      ('Mgate',M,'membership_gate','',4,24,'160G','01:00:00',[],'3dimage-14'),
      ('MG1',M,'membership_train','G1',4,24,'160G','16:00:00',['Mgate'],'3dimage-14'),
      ('MG2',M,'membership_train','G2',4,24,'160G','16:00:00',['Mgate'],'3dimage-14'),
      ('ME1',M,'membership_export','G1',4,24,'160G','06:00:00',['MG1'],'3dimage-14'),
      ('ME2',M,'membership_export','G2',4,24,'160G','06:00:00',['MG2'],'3dimage-14'),
      ('Mscore',M,'membership_score','',1,8,'96G','08:00:00',['ME1','ME2'],None),
      ('Qcache',Q,'gs_cache','',4,24,'160G','12:00:00',[],None),
      ('Qfit',Q,'gs_fit','',0,12,'64G','06:00:00',['Qcache'],None),
      ('Qexport',Q,'gs_export','',4,24,'160G','06:00:00',['Qfit'],None),
      ('Qscore',Q,'gs_score','',1,8,'96G','10:00:00',['Qexport'],None),
      ('Rexport',R,'recon_export','',4,24,'160G','04:00:00',[],None),
      ('Rscore',R,'recon_score','',1,8,'64G','04:00:00',['Rexport'],None)]

def main():
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    for run in (M,Q,R):
        if json.loads((run/'PREFLIGHT.json').read_text())['status']!='PASSED' or json.loads((run/'configuration.json').read_text())['code_sha']!=code:raise ValueError('Preflight identity differs')
        if (run/'SUBMISSION_ATTEMPT.json').exists():raise FileExistsError('Existing submission; never duplicate')
    plans=[];dry=[]
    for key,run,mode,arm,gpu,cpu,mem,limit,deps,node in specs():
        cmd=['--partition=3090,4090','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60','--kill-on-invalid-dep=yes',f'--job-name=Recip_{key}',f'--cpus-per-task={cpu}',f'--mem={mem}',f'--time={limit}',f'--output={run}/{key}-%j.out']
        if gpu:cmd+=[f'--gres=gpu:{gpu}']
        if node:cmd+=[f'--nodelist={node}']
        script=['scripts/run_reciprocity_reconstruction.sh',str(run),mode,arm]
        result=subprocess.run(['sbatch','--test-only']+cmd+script,cwd=ROOT,text=True,capture_output=True,check=True,timeout=30)
        dry.append(dict(stage=key,stdout=result.stdout,stderr=result.stderr));plans.append((cmd,script))
    for run in (M,Q,R):
        (run/'SLURM_TEST_ONLY.json').write_text(json.dumps(dry,indent=2)+'\n')
        with (run/'SUBMISSION_ATTEMPT.json').open('x') as f:json.dump(dict(code_sha=code,time=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
    records=[];ids={}
    for spec,(cmd,script) in zip(specs(),plans):
        key,run,mode,arm,gpu,cpu,mem,limit,deps,node=spec;dependency=[ids[k] for k in deps]
        dep=['--dependency=afterok:'+':'.join(map(str,dependency))] if dependency else []
        result=subprocess.run(['sbatch','--parsable']+cmd+dep+script,cwd=ROOT,text=True,capture_output=True,check=True,timeout=30)
        job=int(result.stdout.strip().split(';')[0]);ids[key]=job;records.append(dict(stage=key,run=str(run),job_id=job,gpus=gpu,cpus=cpu,memory=mem,time_limit=limit,dependencies=dependency,node=node or 'flexible_3090_4090'))
        for root in (M,Q,R):(root/'SUBMITTED_ALL.json').write_text(json.dumps(dict(code_sha=code,jobs=records),indent=2)+'\n')
    try:status=subprocess.check_output(['squeue','-j',','.join(str(r['job_id']) for r in records),'-o','%.12i %.12T %.20N %.8C %.25b %.40R'],text=True,timeout=30)
    except subprocess.TimeoutExpired:status='Single bounded acceptance check timed out; IDs persisted.\n'
    for run in (M,Q,R):(run/'ACCEPTANCE_STATUS.txt').write_text(status)
    print(json.dumps(records,indent=2));print(status);print('已停止轮询，等你通知完成后再检查。')
if __name__=='__main__':main()
