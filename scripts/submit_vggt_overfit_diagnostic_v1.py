"""Submit one eight-GPU validation or the three verified, sequential diagnostics."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
REPORT = Path('/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1')
RUN = Path('/space/mawb/ssst/workspace_group_plus/vggt_overfit_diagnostic_v1')
BRANCH = 'vggt-overfit-diagnostics-v1'
ARMS = ('pred_joint', 'pred_recon', 'gt_recon')


def command(*args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def submit(snapshot, sha, name, output, mode, arm=None, dependency=None):
    env = dict(os.environ, TASK_REPO=str(snapshot), TASK_CODE_SHA=sha,
               TASK_OUTPUT=str(output), TASK_MODE=mode)
    if arm:
        env['TASK_ARM'] = arm
    args = ['sbatch', '--parsable', '--partition=4090', '--nodelist=3dimage-17',
            '--nodes=1', '--ntasks=1', '--gres=gpu:8', '--cpus-per-task=32',
            '--mem=160G', '--time=12:00:00', '--job-name=' + name,
            '--output=' + str(REPORT / 'slurm/%x-%j.out'),
            '--error=' + str(REPORT / 'slurm/%x-%j.err'), '--export=ALL']
    if dependency:
        args.append('--dependency=afterok:' + str(dependency))
    args.append(str(snapshot / 'scripts/run_vggt_overfit_diagnostic_v1.sh'))
    job = command(*args, env=env).split(';')[0]
    if not job.isdigit():
        raise RuntimeError('unrecognized sbatch receipt: ' + job)
    return dict(job_id=job, arm=arm, name=name, output=str(output), dependency=dependency,
                execution_git_sha=sha, snapshot=str(snapshot), node='3dimage-17',
                gpu_count=8, gpu_model='RTX4090', cpus=32, memory='160G', time_limit='12h')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('preflight', 'formal'), required=True)
    p.add_argument('--preflight-output', type=Path)
    p.add_argument('--preflight-job')
    a = p.parse_args()
    if command('git', '-C', str(REPO), 'branch', '--show-current') != BRANCH or command('git', '-C', str(REPO), 'status', '--porcelain'):
        raise RuntimeError('submission requires clean diagnostic branch')
    sha = command('git', '-C', str(REPO), 'rev-parse', 'HEAD')
    if command('git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/' + BRANCH).split()[0] != sha:
        raise RuntimeError('execution commit is not pushed')
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j').splitlines()
    if any(n.startswith('vggt-ofit-') for n in active):
        raise RuntimeError('active diagnostic job exists; no duplicate submission')
    (REPORT / 'slurm').mkdir(parents=True, exist_ok=True)
    (REPORT / 'checks').mkdir(parents=True, exist_ok=True)
    snapshot = REPORT / 'snapshots' / sha
    if not snapshot.exists():
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        command('git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(snapshot), env=env)
        command('git', '-C', str(snapshot), 'checkout', '--quiet', '--detach', sha, env=env)
    if command('git', '-C', str(snapshot), 'rev-parse', 'HEAD') != sha or command('git', '-C', str(snapshot), 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('immutable diagnostic snapshot mismatch')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if a.stage == 'preflight':
        output = REPORT / 'checks' / ('preflight_' + stamp + '_' + sha[:8])
        receipt = submit(snapshot, sha, 'vggt-ofit-preflight', output, 'preflight')
        write(REPORT / 'checks' / ('preflight_submission_' + stamp + '.json'), receipt)
        print(json.dumps(receipt, indent=2))
        return
    if not a.preflight_output or not a.preflight_job:
        raise RuntimeError('formal launch requires completed exact-code three-arm preflight')
    states = command('sacct', '-n', '-X', '-j', a.preflight_job, '-o', 'JobIDRaw,State,ExitCode', '-P').splitlines()
    if not any(line.split('|')[:3] == [a.preflight_job, 'COMPLETED', '0:0'] for line in states):
        raise RuntimeError('preflight Slurm job did not complete successfully')
    manifests = []
    for arm in ARMS:
        report = json.loads((a.preflight_output / arm / 'smoke_report.json').read_text())
        manifest = json.loads((a.preflight_output / arm / 'run_manifest.json').read_text())
        if report['status'] != 'GPU_SMOKE_COMPLETED' or report['execution_git_sha'] != sha or report['formal_updates'] != 0 or report['completed_updates'] != 2 or report['completed_exposures'] != 32:
            raise RuntimeError('unverified preflight for ' + arm)
        if not (a.preflight_output / arm / 'checkpoint_update_0002.pt').is_file():
            raise RuntimeError('preflight checkpoint missing')
        manifests.append(manifest)
    initials = {p['initial_trainable_sha256'] for m in manifests for p in m['rank_preparation']}
    if len(initials) != 1:
        raise RuntimeError('three arms do not share exact source parameters')
    pixels = {}
    for m in manifests:
        for rank in m['rank_preparation']:
            for proof in rank['proofs']:
                key = proof['window']['source_index']
                pixels.setdefault(key, set()).add(proof['input_rgb_sha256'])
    if len(pixels) != 4 or any(len(v) != 1 for v in pixels.values()):
        raise RuntimeError('three arms have different input pixels')
    if RUN.exists() and any(RUN.iterdir()):
        raise RuntimeError('formal diagnostic output already exists; preserve it')
    RUN.mkdir(parents=True, exist_ok=True)
    receipt_path = REPORT / ('launch_plan_' + stamp + '.json')
    receipt = dict(status='SUBMITTING', preflight_job=a.preflight_job,
                   preflight_output=str(a.preflight_output), execution_git_sha=sha,
                   evaluations_submitted=False, runs=[])
    write(receipt_path, receipt)
    dependency = a.preflight_job
    for arm in ARMS:
        job = submit(snapshot, sha, 'vggt-ofit-' + arm, RUN / arm, 'train', arm, dependency)
        receipt['runs'].append(job)
        write(receipt_path, receipt)
        dependency = job['job_id']
    receipt['status'] = 'THREE_DIAGNOSTICS_SUBMITTED'
    write(receipt_path, receipt)
    write(REPORT / 'launch_plan.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
