#!/usr/bin/env python3
"""Run the exact module entrypoints used by the fixed C/D Slurm scripts."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.train_parallel_probe_heads import worker_command

ROOT = Path(__file__).resolve().parents[1]
TOKENGS = Path('/space/mawb/anaconda3/envs/tokengs/bin/python')
OFFICIAL = Path('/space/mawb/SIU3R/.venv_gpu_v4/bin/python')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['CUDA_VISIBLE_DEVICES'] = ''
    root_arg = '/__entrypoint_help_only__'
    checks = [
        ('C_parent', [str(TOKENGS), '-m', 'scripts.train_parallel_probe_heads', '--help'], TOKENGS),
        *[(f'C_{head}_worker', worker_command(head, root_arg, python=TOKENGS) + ['--help'], TOKENGS)
          for head in ('H1', 'H2', 'H3')],
        ('D_eval', [str(TOKENGS), '-m', 'scripts.eval_object_locus_frozen_probe', '--help'], TOKENGS),
        ('D_report', [str(OFFICIAL), '-m', 'scripts.report_object_locus_frozen_probe', '--help'], OFFICIAL),
    ]
    records = []
    for name, argv, python in checks:
        proc = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True)
        records.append({
            'name': name, 'python': str(python), 'cwd': str(ROOT), 'argv': argv,
            'env_overrides': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': 'removed'},
            'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr,
            'help_usage_present': 'usage:' in proc.stdout.lower(),
            'module_not_found': 'ModuleNotFoundError' in proc.stderr,
        })
    files = ['scripts/preflight_frozen_probe_entrypoints.py', 'scripts/train_parallel_probe_heads.py',
             'scripts/train_object_locus_frozen_probe_head_worker.py',
             'scripts/eval_object_locus_frozen_probe.py', 'scripts/report_object_locus_frozen_probe.py',
             'slurm/train_probe_heads_parallel.sbatch', 'slurm/unified_eval_report.sbatch']
    result = {'status': 'PASS' if all(x['exit_code'] == 0 and x['help_usage_present'] and not x['module_not_found'] for x in records) else 'FAIL',
              'checks': records,
              'source_files': {p: {'sha256': sha(ROOT / p), 'size': (ROOT / p).stat().st_size} for p in files}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': {x['name']: x['exit_code'] for x in records}}), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
