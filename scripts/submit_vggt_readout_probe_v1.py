"""Submit immutable readout preflight, then selected4/8 streaming cache/train/eval."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/space/mawb/ssst/group_plus/vggt_readout_probe_v1')


def command(args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'formal'), required=True)
    parser.add_argument('--attempt', default='attempt01')
    args = parser.parse_args()
    assert Path(args.attempt).name == args.attempt and args.attempt.startswith('attempt')
    assert not command(['git', '-C', str(REPO), 'status', '--porcelain'])
    sha = command(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
    assert command(['git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/vggt-overfit-diagnostics-v1']).split()[0] == sha
    assert not any(name.startswith('vggt-readout-') for name in command(['squeue', '-h', '-u', 'mawb', '-o', '%j']).splitlines())
    assert shutil.disk_usage(ROOT.parent).free > 5 * 2**30
    attempt = ROOT / 'attempts' / args.attempt
    snapshot = ROOT / 'snapshots' / sha
    if args.phase == 'preflight':
        attempt.mkdir(parents=True, exist_ok=False)
        (attempt / 'slurm').mkdir()
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
            command(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(snapshot)], env)
            command(['git', '-C', str(snapshot), 'checkout', '--quiet', '--detach', sha], env)
        jobs, workers = [], 8
    else:
        receipt = json.loads((attempt / 'submission.json').read_text())
        assert receipt['execution_sha'] == sha
        assert json.loads((attempt / 'preflight/PREFLIGHT_VERIFIED.json').read_text())['status'] == 'READOUT_FULL_PREFLIGHT_PASSED'
        numerical = json.loads((attempt / 'preflight/numerical_preflight.json').read_text())
        workers = numerical['selected_cache_workers']
        assert workers in (4, 8)
        assert not (attempt / 'training').exists()
        jobs = receipt['jobs']
        assert len(jobs) == 1 and jobs[0]['mode'] == 'preflight'
    receipt_path = attempt / 'submission.json'

    def submit(mode, gpu_count, dependency=None):
        exports = f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={attempt},TASK_WORKERS={workers}'
        cmd = ['sbatch', '--parsable', '--partition=3090', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=' + ('32' if gpu_count else '4'), '--mem=' + ('384G' if gpu_count == 8 else '192G' if gpu_count else '32G'),
            '--time=' + ('03:00:00' if mode == 'train_eval' else '01:00:00'),
            '--job-name=vggt-readout-' + mode, '--output=' + str(attempt / 'slurm/%x-%j.out'),
            '--error=' + str(attempt / 'slurm/%x-%j.err'), '--export=' + exports]
        if gpu_count:
            cmd += ['--gres=gpu:' + str(gpu_count), '--nodelist=3dimage-13']
        if dependency:
            cmd += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
        cmd += [str(snapshot / 'scripts/run_vggt_readout_probe_v1.sh')]
        job = command(cmd).split(';')[0]
        jobs.append(dict(job_id=job, mode=mode, gpus=gpu_count, dependency=dependency, command=cmd))
        receipt_path.write_text(json.dumps(dict(status='SUBMITTING', execution_sha=sha, jobs=jobs), indent=2) + '\n')
        return job

    if args.phase == 'preflight':
        submit('preflight', 8)
        status = 'READOUT_REAL8GPU_PREFLIGHT_SUBMITTED'
    else:
        formal = submit('train_eval', workers)
        submit('summary', 0, formal)
        status = 'READOUT_SHARED_CACHE_TWO_HEADS_AND_EVALUATION_SUBMITTED'
    receipt_path.write_text(json.dumps(dict(status=status, execution_sha=sha, attempt=str(attempt), jobs=jobs,
        cache_workers=workers, head_optimizer_workers=1, head_global_batch=8,
        updates_per_arm=300, exposures_per_arm=2400, shared_frozen_C_windows=2400,
        evaluation_windows=128, evaluation_scenes=128, C_core_frozen=True,
        original_loss_definitions_unchanged=True, new_diagnostic_losses=['quality SmoothL1 beta1', 'objectness weighted BCE negative0.1'],
        automatic_official128_scoring=True, official_columns=13, text_columns_blank=True), indent=2) + '\n')
    print(json.dumps(dict(receipt=str(receipt_path), status=status, execution_sha=sha, cache_workers=workers,
                         jobs=[{k: j[k] for k in ('job_id', 'mode', 'gpus')} for j in jobs]), indent=2))


if __name__ == '__main__':
    main()
