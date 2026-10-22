"""Submit ONE authorized chain, then query scheduler acceptance once and stop."""
from pathlib import Path
import datetime,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1')

def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty execution snapshot'
    assert json.loads((RUN/'CPU_VALIDATION.json').read_text())['status']=='PASSED'
    assert json.loads((RUN/'CPU_TESTS.json').read_text())['status']=='PASSED'
    assert not (RUN/'SUBMISSION_REQUESTED.json').exists() and not (RUN/'SUBMITTED.json').exists(),'Refusing duplicate submission'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for stage in ('smoke','train','eval'):
        p=RUN/stage;cfg=json.loads((p/'configuration.json').read_text());cfg['code_sha']=sha
        assert not (p/'STARTED.json').exists() and not (p/'SMOKE_STARTED.json').exists()
        assert hashlib.sha256(Path(cfg['manifest']).read_bytes()).hexdigest()==cfg['manifest_sha256']
        (p/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n');(p/'code.sha').write_text(sha+'\n')
    # These two partitions contain exactly nodes11/12/13/14/17/18. Excluding11
    # permits any ONE node of the approved pool, without a multi-node nodelist.
    common=['--partition=3090,4090','--exclude=3dimage-11','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60']
    specs=[('smoke',1,8,'128G','01:30:00'),('train',8,32,'192G','1-00:00:00'),('eval',8,32,'192G','08:00:00')]
    for stage,gpus,cpus,mem,limit in specs:
        cmd=['sbatch','--test-only']+common+[f'--gres=gpu:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={limit}','scripts/run_v4init_isolated.sh',str(RUN/stage),stage]
        check=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert check.returncode==0,check.stderr
    receipt=dict(code_root=str(ROOT),code_sha=sha,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        eligible_nodes=[12,13,14,17,18],jobs={},dependencies={},stop_polling_after_single_check=True)
    with (RUN/'SUBMISSION_REQUESTED.json').open('x') as f:json.dump(receipt,f,indent=2)
    def save():(RUN/'SUBMITTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save();previous=None
    for stage,gpus,cpus,mem,limit in specs:
        p=RUN/stage
        cmd=['sbatch','--parsable']+common+[f'--job-name=V4isoNoCA_{stage}',f'--gres=gpu:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={limit}',f'--output={p}/slurm-%j.out']
        if previous is not None:cmd += [f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes']
        cmd+=['scripts/run_v4init_isolated.sh',str(p),stage]
        job=int(subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0])
        receipt['jobs'][stage]=job;receipt['dependencies'][stage]=previous;save()
        (p/'SUBMITTED.json').write_text(json.dumps(dict(job=job,stage=stage,dependency=previous,submission_argv=cmd,code_sha=sha),indent=2)+'\n')
        previous=job
    print(json.dumps(receipt,indent=2),flush=True)
    ids=','.join(map(str,receipt['jobs'].values()))
    try:
        q=subprocess.run(['squeue','-j',ids,'-o','%.18i %.20j %.10T %.15P %.20N %R'],capture_output=True,text=True,timeout=20);status=q.stdout+q.stderr
    except subprocess.TimeoutExpired:status='Single acceptance query timed out; sbatch returned IDs above.\n'
    (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);print(status,flush=True)

if __name__=='__main__':main()
