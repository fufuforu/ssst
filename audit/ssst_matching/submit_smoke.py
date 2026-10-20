"""Prepare an independent immutable code snapshot; submit smoke only, no polling."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=Path('/space/mawb/ssst_matching_audit_runs_20261020')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=DEFAULT_RUN)
    p.add_argument('--cpu-receipt',type=Path,required=True)
    p.add_argument('--contract-receipt',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();root=args.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    cpu=json.loads(args.cpu_receipt.read_text());contract=json.loads(args.contract_receipt.read_text())
    assert cpu['status']=='PASSED' and cpu['windows']>=8
    assert contract['status']=='PASSED' and contract['tests']>=13
    # Validate tested code before snapshotting or submission.
    for path,sha in cpu['implementation_sha256'].items():
        assert digest(ROOT/path)==sha, f'CPU preflight implementation changed: {path}'
    for path,sha in contract['implementation_sha256'].items():
        assert digest(ROOT/path)==sha, f'Contract-tested implementation changed: {path}'
    target=root/'code_snapshot';request=root/'smoke_submission_request.json'
    if args.prepare_only:
        if target.exists():raise ValueError('Snapshot already exists; do not overwrite')
        target.mkdir()
        for name in ('tokengs','scripts','third_party','acc_configs','audit'):
            shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'pyproject.toml',target/'pyproject.toml')
        base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (target/'.audit_git_sha').write_text(base+'\n')
        # Freeze the snapshot bytes; group/world read and traversal are retained.
        for f in target.rglob('*'):
            if f.is_file():f.chmod(f.stat().st_mode & ~0o222)
        identity=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),worktree=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                      base_git_sha=base,code_snapshot=str(target),implementation_sha256=cpu['implementation_sha256'],
                      cpu_receipt=str(args.cpu_receipt.resolve()),contract_receipt=str(args.contract_receipt.resolve()),
                      stage='SMOKE_ONLY_NO_FORMAL_AUTO_START',checkpoint_sha256=cpu['checkpoint_sha256'])
        (root/'snapshot_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        job='''#!/bin/bash
set -euo pipefail
cd "${1:?snapshot required}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RG_EXTRA RG_DEPTH RG_TSEM RG_SUPERVISED
/space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.run --stage smoke --cohort smoke8 --output "${2:?new output required}"
'''
        (root/'run_smoke.sbatch').write_text(job)
        print(json.dumps(identity,indent=2));return
    if (root/'smoke_submission.json').exists() or request.exists():
        raise ValueError('Refusing duplicate submission or ambiguous retry; inspect existing submission first')
    identity=json.loads((root/'snapshot_identity.json').read_text())
    for path,sha in identity['implementation_sha256'].items():
        assert digest(target/path)==sha, f'Snapshot changed: {path}'
    argv=['sbatch','--parsable','--job-name=ssst_match_smoke8','--partition=3090','--gres=gpu:3090:1',
          '--cpus-per-task=8','--mem=64G','--time=00:45:00',f'--output={root}/smoke-slurm-%j.out',
          str(root/'run_smoke.sbatch'),str(target),str(root/'smoke8')]
    request.write_text(json.dumps(dict(command=argv,stop_polling_after_submission=True,stage='smoke8',formal_auto_start=False),indent=2)+'\n')
    completed=subprocess.run(argv,check=True,text=True,capture_output=True,cwd=target)
    job_id=int(completed.stdout.strip().split(';')[0])
    record=dict(job_id=job_id,stage='smoke8',node='scheduler-selected',gpus='1 x RTX3090',command=argv,
                code_snapshot=str(target),output=str(root/'smoke8'),formal_auto_start=False,stop_polling_after_submission=True)
    (root/'smoke_submission.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
