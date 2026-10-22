"""Submit TWO authorized one-factor chains plus dependent CPU summary, then query scheduler acceptance once and stop."""
from pathlib import Path
import datetime,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_weight_views_20261022_v1')

def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty execution snapshot'
    assert json.loads((RUN/'CPU_VALIDATION.json').read_text())['status']=='PASSED'
    assert json.loads((RUN/'CPU_TESTS.json').read_text())['status']=='PASSED'
    assert not (RUN/'SUBMISSION_REQUESTED.json').exists() and not (RUN/'SUBMITTED.json').exists(),'Refusing duplicate submission'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for branch in ('U005','Views2ctx4'):
        for stage in ('smoke','train','eval'):
            p=RUN/branch/stage;cfg=json.loads((p/'configuration.json').read_text());cfg['code_sha']=sha
            assert not (p/'STARTED.json').exists() and not (p/'SMOKE_STARTED.json').exists()
            assert hashlib.sha256(Path(cfg['manifest']).read_bytes()).hexdigest()==cfg['manifest_sha256']
            if branch=='Views2ctx4':assert hashlib.sha256(Path(cfg['training_window_manifest']).read_bytes()).hexdigest()==cfg['training_window_manifest_sha256']
            (p/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n');(p/'code.sha').write_text(sha+'\n')
    (RUN/'code.sha').write_text(sha+'\n')
    # These two partitions contain exactly nodes11/12/13/14/17/18. Excluding11
    # permits any ONE node of the approved pool, without a multi-node nodelist.
    common=['--partition=3090,4090','--exclude=3dimage-11','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60']
    specs=[('smoke',1,8,'128G','01:30:00'),('train',8,32,'192G','1-00:00:00'),('eval',8,32,'192G','08:00:00')]
    for stage,gpus,cpus,mem,limit in specs:
        cmd=['sbatch','--test-only']+common+[f'--gres=gpu:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={limit}','scripts/run_v4init_weight_views.sh',str(RUN/'U005'/stage),stage]
        check=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert check.returncode==0,check.stderr
    check=subprocess.run(['sbatch','--test-only']+common+['--cpus-per-task=4','--mem=16G','--time=01:00:00','scripts/run_v4init_weight_views_summary.sh',str(RUN)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert check.returncode==0,check.stderr
    receipt=dict(code_root=str(ROOT),code_sha=sha,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        eligible_nodes=[12,13,14,17,18],branches={},baseline_train_job=71001,baseline_eval_job=71002,stop_polling_after_single_check=True)
    with (RUN/'SUBMISSION_REQUESTED.json').open('x') as f:json.dump(receipt,f,indent=2)
    def save():(RUN/'SUBMITTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for branch in ('U005','Views2ctx4'):
        previous=None;receipt['branches'][branch]={}
        for stage,gpus,cpus,mem,limit in specs:
            p=RUN/branch/stage
            cmd=['sbatch','--parsable']+common+[f'--job-name=V4{branch}_{stage}',f'--gres=gpu:{gpus}',f'--cpus-per-task={cpus}',f'--mem={mem}',f'--time={limit}',f'--output={p}/slurm-%j.out']
            if previous is not None:cmd += [f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes']
            cmd+=['scripts/run_v4init_weight_views.sh',str(p),stage]
            job=int(subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0])
            receipt['branches'][branch][stage]=job;save()
            (p/'SUBMITTED.json').write_text(json.dumps(dict(job=job,stage=stage,dependency=previous,submission_argv=cmd,code_sha=sha),indent=2)+'\n')
            previous=job
    dependency=[71002]+[j['eval'] for j in receipt['branches'].values()]
    cmd=['sbatch','--parsable']+common+['--job-name=V4weightviews_summary','--cpus-per-task=4','--mem=16G','--time=01:00:00',f'--dependency=afterok:{":".join(map(str,dependency))}','--kill-on-invalid-dep=yes',f'--output={RUN}/summary-%j.out','scripts/run_v4init_weight_views_summary.sh',str(RUN)]
    summary=int(subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0]);receipt['summary']=dict(job=summary,dependency=dependency,submission_argv=cmd);save()
    print(json.dumps(receipt,indent=2),flush=True)
    ids=','.join(map(str,[j for jobs in receipt['branches'].values() for j in jobs.values()]+[summary]))
    try:
        q=subprocess.run(['squeue','-j',ids,'-o','%.18i %.20j %.10T %.15P %.20N %R'],capture_output=True,text=True,timeout=20);status=q.stdout+q.stderr
    except subprocess.TimeoutExpired:status='Single acceptance query timed out; sbatch returned IDs above.\n'
    (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);print(status,flush=True)

if __name__=='__main__':main()
