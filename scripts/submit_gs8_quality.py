from pathlib import Path
import json,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline8_gs_quality_20261025_v3_runtimefix')
SPECS=[('cache',8,32,'192G','06:00:00'),('fit',0,12,'64G','02:00:00'),('export',8,32,'160G','03:00:00'),('score',5,20,'256G','03:00:00')]


def main():
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip() or json.loads((RUN/'PREFLIGHT.json').read_text())['code_sha']!=code:raise ValueError('Preflight/code differs')
    if (RUN/'SUBMISSION_ATTEMPT.json').exists():raise FileExistsError('Existing submission; inspect instead')
    cfg=json.loads((RUN/'configuration.json').read_text())
    allowed=set(cfg['allowed_nodes'])
    nodes=set(subprocess.check_output(['sinfo','-N','-h','-o','%N'],text=True,timeout=30).split())
    if not allowed<=nodes:raise ValueError('Requested nodes missing from cluster')
    excluded=sorted(nodes-allowed)
    # --nodelist with six names requests all six, not one of them. Use the two
    # partitions and exclude their complement while keeping --nodes=1.
    (RUN/'NODE_ELIGIBILITY.json').write_text(json.dumps(dict(allowed_nodes=sorted(allowed),excluded_nodes=excluded,partitions=cfg['partitions'],single_node=True),indent=2)+'\n')
    plans=[];dry=[]
    for phase,gpus,cpus,memory,limit in SPECS:
        args=['--partition=3090,4090','--exclude='+','.join(excluded),'--nodes=1','--ntasks=1','--no-requeue','--kill-on-invalid-dep=yes','--signal=B:TERM@60',
              f'--cpus-per-task={cpus}',f'--mem={memory}',f'--time={limit}',f'--job-name=GS8_{phase}',f'--output={RUN}/{phase}-%j.out']
        if gpus:args.append(f'--gres=gpu:{gpus}')
        script=['scripts/run_gs8_quality.sh',str(RUN),phase]
        value=subprocess.run(['sbatch','--test-only']+args+script,cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
        plans.append((args,script));dry.append(dict(phase=phase,stdout=value.stdout,stderr=value.stderr))
    (RUN/'SLURM_TEST_ONLY.json').write_text(json.dumps(dry,indent=2)+'\n')
    with (RUN/'SUBMISSION_ATTEMPT.json').open('x') as f:json.dump(dict(code_sha=code,time=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
    jobs=[]
    for (phase,gpus,cpus,memory,limit),(args,script) in zip(SPECS,plans):
        dependencies=[jobs[-1]['job_id']] if jobs else [];dep=['--dependency=afterok:'+':'.join(map(str,dependencies))] if dependencies else []
        value=subprocess.run(['sbatch','--parsable']+args+dep+script,cwd=ROOT,check=True,capture_output=True,text=True,timeout=30)
        jobs.append(dict(phase=phase,job_id=int(value.stdout.strip().split(';')[0]),gpus=gpus,allowed_nodes=sorted(allowed),single_node=True,partitions=cfg['partitions'],dependencies=dependencies))
        (RUN/'SUBMITTED.json').write_text(json.dumps(dict(code_sha=code,jobs=jobs),indent=2)+'\n')
    try:status=subprocess.check_output(['squeue','-j',','.join(str(j['job_id']) for j in jobs),'-o','%.12i %.12T %.20N %.25b %.45R'],text=True,timeout=30)
    except subprocess.TimeoutExpired:status='One bounded acceptance query timed out; job IDs saved.'
    (RUN/'ACCEPTANCE_STATUS.txt').write_text(status);print(json.dumps(jobs,indent=2));print(status);print('已停止轮询，等你通知完成后再检查。')


if __name__=='__main__':main()
