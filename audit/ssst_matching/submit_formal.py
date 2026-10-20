"""Freeze pushed audit code; submit formal32 then dependent formal128, no polling."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/space/mawb/ssst_matching_audit_runs_20261020')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,data):Path(p).write_text(json.dumps(data,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--push-receipt',type=Path,required=True)
    args=p.parse_args();formal=RUN/'formal_v1';formal.mkdir(exist_ok=True)
    push=json.loads(args.push_receipt.read_text());smoke=json.loads((RUN/'smoke8/SMOKE_PASSED.json').read_text())
    review=json.loads((ROOT/'audit/ssst_matching/reports/SMOKE_REVIEWED.json').read_text())
    pre=json.loads((RUN/'cpu_preflight128/CPU_PREFLIGHT_PASSED.json').read_text())
    assert push['remote_verified'] and smoke['status']==review['status']==pre['status']=='PASSED'
    assert smoke['windows']==8 and pre['windows']==128 and smoke['parameter_updates']==0
    assert review['saved_matrix_consistency']
    for file,sha in smoke['implementation_sha256'].items():
        assert digest(ROOT/file)==sha,file
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert sha==push['git_sha']
    target=formal/'code_snapshot'
    if args.prepare_only:
        assert not target.exists(),'Refuse overwriting prepared formal snapshot'
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit first'
        remote=subprocess.check_output(['git','ls-remote',push['remote'],push['ref']],cwd=ROOT,text=True).split()[0]
        assert remote==sha
        target.mkdir()
        for name in ('tokengs','scripts','third_party','acc_configs','audit'):
            shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'pyproject.toml',target/'pyproject.toml')
        (target/'.audit_git_sha').write_text(sha+'\n')
        for file in target.rglob('*'):
            if file.is_file():file.chmod(file.stat().st_mode & ~0o222)
        for file,expected in smoke['implementation_sha256'].items():assert digest(target/file)==expected
        shutil.copy2(args.push_receipt,formal/'push_verified.json')
        write(formal/'snapshot_identity.json',dict(git_sha=sha,code_snapshot=str(target),checkpoint_sha256=smoke['checkpoint_sha256'],implementation_sha256=smoke['implementation_sha256'],smoke_receipt=str(RUN/'smoke8/SMOKE_PASSED.json'),push_ref=push['ref'],cohorts=['heldout32','heldout128'],overlap_note='heldout32 is first32 of heldout128; never sum as160 independent scenes'))
        script='''#!/bin/bash
set -euo pipefail
cd "${1:?snapshot required}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GIT_TERMINAL_PROMPT=0
unset RG_EXTRA RG_DEPTH RG_TSEM RG_SUPERVISED
/space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.run --stage formal --cohort "${2:?cohort required}" --smoke-receipt /space/mawb/ssst_matching_audit_runs_20261020/smoke8/SMOKE_PASSED.json --push-receipt /space/mawb/ssst_matching_audit_runs_20261020/formal_v1/push_verified.json --output "${3:?new output required}"
'''
        (formal/'run_formal.sbatch').write_text(script)
        print(json.dumps(dict(status='PREPARED',git_sha=sha,code_snapshot=str(target))));return
    assert target.is_dir()
    for file,expected in smoke['implementation_sha256'].items():assert digest(target/file)==expected
    request=formal/'submission_request.json'
    assert not request.exists() and not (formal/'submission.json').exists(),'Do not duplicate/guess submission'
    commands=[];jobs=[]
    # A successful first32 run gates full128. Both jobs include their own summaries.
    for cohort in ('heldout32','heldout128'):
        command=['sbatch','--parsable',f'--job-name=ssst_match_{cohort}','--partition=4090','--nodelist=3dimage-14','--gres=gpu:4090:1','--cpus-per-task=8','--mem=64G','--time=00:45:00',f'--output={formal}/{cohort}-slurm-%j.out']
        if jobs:command += [f'--dependency=afterok:{jobs[0]}','--kill-on-invalid-dep=yes']
        command += [str(formal/'run_formal.sbatch'),str(target),cohort,str(formal/cohort)]
        commands.append(command)
        write(request,dict(stage='SUBMISSION_IN_PROGRESS',commands=commands,accepted_jobs=jobs))
        result=subprocess.run(command,cwd=target,check=True,text=True,capture_output=True)
        jobs.append(int(result.stdout.strip().split(';')[0]))
        write(formal/'submission.json',dict(jobs=dict(zip(('heldout32','heldout128'),jobs)),commands=commands,git_sha=sha,node='3dimage-14',gpu_per_job='1xRTX4090',summary='inside each inference job',training_jobs=[],stop_polling=True))
    print(json.dumps(json.loads((formal/'submission.json').read_text()),indent=2))


if __name__=='__main__':main()
