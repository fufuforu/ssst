"""Exactly three afterok continuations and their three epoch8 evaluations."""
from pathlib import Path
import json,subprocess,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_20261022_v1')

def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Dirty execution snapshot'
    assert json.loads((RUN/'CPU_SOURCE_VALIDATION.json').read_text())['status']=='PASSED'
    assert json.loads((RUN/'CPU_TESTS.json').read_text())['status']=='PASSED'
    assert not (RUN/'SUBMISSION_REQUESTED.json').exists() and not (RUN/'SUBMITTED.json').exists(),'Refuse duplicate submission'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for label in ('Baseline','U005','Views2ctx4'):
        for stage in ('train','eval'):
            r=RUN/label/stage;cfg=json.loads((r/'configuration.json').read_text())
            assert not (r/'STARTED.json').exists() and not (r/'checkpoint_latest.pt').exists()
            cfg['code_sha']=sha
            assert hashlib.sha256(Path(cfg['manifest']).read_bytes()).hexdigest()==cfg['manifest_sha256']
            if 'training_window_manifest' in cfg:assert hashlib.sha256(Path(cfg['training_window_manifest']).read_bytes()).hexdigest()==cfg['training_window_manifest_sha256']
            (r/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n');(r/'code.sha').write_text(sha+'\n')
    common=['--partition=3090,4090','--exclude=3dimage-11','--nodes=1','--ntasks=1','--no-requeue','--signal=B:TERM@60','--gres=gpu:8','--cpus-per-task=32','--mem=192G','--kill-on-invalid-dep=yes']
    for stage,limit in [('train','1-00:00:00'),('eval','08:00:00')]:
        cmd=['sbatch','--test-only']+common+[f'--time={limit}','scripts/run_v4init_extend8.sh',str(RUN/'Baseline'/stage),stage]
        q=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=30);assert q.returncode==0,q.stderr
    receipt=dict(code_root=str(ROOT),code_sha=sha,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branches={},eligible_nodes=[12,13,14,17,18],stop_polling_after_single_check=True)
    with (RUN/'SUBMISSION_REQUESTED.json').open('x') as f:json.dump(receipt,f,indent=2)
    def save():(RUN/'SUBMITTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    for label in ('Baseline','U005','Views2ctx4'):
        cfg=json.loads((RUN/label/'train/configuration.json').read_text());dependency=cfg['source_train_job'];jobs={};receipt['branches'][label]=dict(source_train=dependency,source_eval=cfg['source_eval_job'],jobs=jobs)
        for stage,limit in [('train','1-00:00:00'),('eval','08:00:00')]:
            r=RUN/label/stage
            cmd=['sbatch','--parsable']+common+[f'--job-name=V4{label}_ep8_{stage}',f'--time={limit}',f'--dependency=afterok:{dependency}',f'--output={r}/slurm-%j.out','scripts/run_v4init_extend8.sh',str(r),stage]
            job=int(subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0]);jobs[stage]=job;save()
            (r/'SUBMITTED.json').write_text(json.dumps(dict(job=job,stage=stage,dependency=dependency,submission_argv=cmd,code_sha=sha),indent=2)+'\n');dependency=job
    print(json.dumps(receipt,indent=2),flush=True)
    ids=','.join(str(j) for b in receipt['branches'].values() for j in b['jobs'].values())
    try:
        q=subprocess.run(['squeue','-j',ids,'-o','%.18i %.24j %.10T %.15P %.20N %R'],capture_output=True,text=True,timeout=20);status=q.stdout+q.stderr
    except subprocess.TimeoutExpired:status='One bounded acceptance check timed out; sbatch returned IDs above.\n'
    (RUN/'STATUS_AFTER_SUBMIT.txt').write_text(status);print(status,flush=True)
if __name__=='__main__':main()
