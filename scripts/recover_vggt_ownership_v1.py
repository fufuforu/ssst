"""Resume only missing inference records; preserve the strict-PNG failed job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1')
ATTEMPT=ROOT/'attempts/attempt01'


def cmd(args,env=None):return subprocess.check_output(args,text=True,env=env).strip()


def main():
    assert not (ATTEMPT/'resume_submission.json').exists()
    assert not cmd(['git','-C',str(REPO),'status','--porcelain'])
    assert 'FAILED' in cmd(['sacct','-n','-X','-j','64911','-o','State','-P'])
    sha=cmd(['git','-C',str(REPO),'rev-parse','HEAD'])
    assert cmd(['git','-c','http.https://github.com/fufuforu/ssst.git.proxy=http://127.0.0.1:17899','ls-remote','origin','refs/heads/vggt-overfit-diagnostics-v1']).split()[0]==sha
    proof=json.loads((ATTEMPT/'comparability.json').read_text())
    old=ROOT/'snapshots'/proof['execution_sha']
    mathfile='scripts/vggt_ownership_math_v1.py'
    assert hashlib.sha256((REPO/mathfile).read_bytes()).digest()==hashlib.sha256((old/mathfile).read_bytes()).digest()
    from vggt_reading_numerical_contract_v1 import contract
    assert contract(REPO)==contract(old)
    expected=json.loads(Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json').read_text())['records']
    expected={r['name']:r for r in expected}
    counts={}
    for arm in ('C','baseline300'):
        files=list((ATTEMPT/arm).glob('rank*/scene*.json'))
        for f in files:
            r=json.loads(f.read_text());assert all(r[k]==expected[r['name']][k] for k in ('source_index','frame_ids','input_rgb_sha256'))
            assert f.with_suffix('.npz').exists()
        counts[arm]=len(files)
    assert counts['C']==128 and counts['baseline300']==92
    snap=ROOT/'snapshots'/sha
    env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
    cmd(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(snap)],env)
    cmd(['git','-C',str(snap),'sparse-checkout','init','--no-cone'],env)
    cmd(['git','-C',str(snap),'sparse-checkout','set','--no-cone','/*','!/group_plus/'],env)
    cmd(['git','-C',str(snap),'checkout','--quiet','--detach',sha],env)
    args=['sbatch','--parsable','--partition=3090','--exclude=3dimage-12','--nodes=1','--ntasks=1','--gres=gpu:3','--cpus-per-task=12','--mem=120G','--time=02:00:00',
        '--job-name=vggt-ownership-resume','--output='+str(ATTEMPT/'slurm/%x-%j.out'),'--error='+str(ATTEMPT/'slurm/%x-%j.err'),
        '--export='+f'ALL,TASK_REPO={snap},TASK_OUTPUT={ATTEMPT},TASK_CODE_SHA={sha},TASK_WORKERS=3,TASK_RESUME=1',str(snap/'scripts/run_vggt_ownership_v1.sh')]
    jid=cmd(args).split(';')[0]
    receipt=dict(status='RESUME_MISSING_FROZEN_INFERENCE_ONLY',job_id=jid,execution_sha=sha,prior_job='64911',reused_windows=counts,
        missing_baseline300_windows=36,new_optimizer_updates=0,model_and_observation_math_identical=True,
        reproduction_policy='Exact existing PNG remains required. Up to5 identical-seed repeats for numerical reproduction; preserve each mismatch and native-repeat witnesses; never select by GT score.',command=args)
    (ATTEMPT/'resume_submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
