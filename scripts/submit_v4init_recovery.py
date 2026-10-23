"""Submit exactly two JSON-fixed recovery trains and their afterok epoch8 evaluations."""
from pathlib import Path
import os,json,subprocess,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_recovery_20261023_v2_jsonfix')


def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty snapshot'
    assert not (RUN/'SUBMISSION_REQUESTED.json').exists(),'Refuse duplicate submission'
    audit=json.loads((RUN/'CPU_SOURCE_VALIDATION.json').read_text())
    assert audit['status']=='PASSED' and set(audit['arms'])=={'Baseline','U005'}
    assert not (RUN/'Views2ctx4').exists(),'Do not recreate the active 2+4 arm'
    assert json.loads((RUN/'CPU_TESTS.json').read_text())['status']=='PASSED'
    assert os.statvfs(RUN).f_bavail*os.statvfs(RUN).f_frsize>28*2**30,'Capacity budget no longer fits'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for label in ['Baseline','U005']:
        for stage in ['train','eval']:
            r=RUN/label/stage;cfg=json.loads((r/'configuration.json').read_text())
            assert not (r/'STARTED.json').exists() and not (r/'checkpoint_latest.pt').exists()
            cfg['code_sha']=sha;assert digest(cfg['manifest'])==cfg['manifest_sha256']
            if 'training_window_manifest' in cfg:assert digest(cfg['training_window_manifest'])==cfg['training_window_manifest_sha256']
            write(r/'configuration.json',cfg);(r/'code.sha').write_text(sha+'\n')
    common=['--partition=3090,4090','--exclude=3dimage-11','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60','--gres=gpu:8','--cpus-per-task=32','--mem=192G','--kill-on-invalid-dep=yes']
    for stage,limit in [('train','1-00:00:00'),('eval','08:00:00')]:
        q=subprocess.run(['sbatch','--test-only']+common+[f'--time={limit}','scripts/run_v4init_recovery.sh',str(RUN/'Baseline'/stage),stage],cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert q.returncode==0,q.stderr
    receipt=dict(code_root=str(ROOT),code_sha=sha,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branches={},eligible_nodes=[12,13,14,17,18],gpus_per_job=8,unchanged_existing_jobs=dict(Views2ctx4_train=71188,Views2ctx4_eval=71189))
    with (RUN/'SUBMISSION_REQUESTED.json').open('x') as f:json.dump(receipt,f,indent=2)
    write(RUN/'SUBMITTED.json',receipt)
    for label in ['Baseline','U005']:
        jobs={};receipt['branches'][label]=dict(jobs=jobs,recovery_progress=json.loads((RUN/label/'train/PREFLIGHT.json').read_text())['progress'])
        dependency=None
        for stage,limit in [('train','1-00:00:00'),('eval','08:00:00')]:
            r=RUN/label/stage
            cmd=['sbatch','--parsable']+common+[f'--job-name=V4jsonfix_{label}_{stage}',f'--time={limit}',f'--output={r}/slurm-%j.out']
            if dependency is not None:cmd += [f'--dependency=afterok:{dependency}']
            cmd += ['scripts/run_v4init_recovery.sh',str(r),stage]
            job=int(subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0]);jobs[stage]=job
            write(RUN/'SUBMITTED.json',receipt);write(r/'SUBMITTED.json',dict(job=job,stage=stage,dependency=dependency,submission_argv=cmd,code_sha=sha));dependency=job
    print(json.dumps(receipt,indent=2),flush=True)
    ids=','.join(str(j) for b in receipt['branches'].values() for j in b['jobs'].values())
    try:
        q=subprocess.run(['squeue','-j',ids,'-o','%.18i %.32j %.10T %.15P %.20N %R'],capture_output=True,text=True,timeout=20);status=q.stdout+q.stderr
    except subprocess.TimeoutExpired:status='Only bounded status check timed out; sbatch returned IDs above.\n'
    (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);print(status,flush=True)


if __name__=='__main__':main()
