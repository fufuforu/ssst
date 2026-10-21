"""Eight-GPU fixed-C feature diagnosis on node11, independent of Q full312."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/space/mawb/ssst/group_plus/vggt_feature_reading_v1')


def command(args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', default='attempt01')
    args = p.parse_args()
    assert Path(args.attempt).name == args.attempt and args.attempt.startswith('attempt')
    assert not command(['git', '-C', str(REPO), 'status', '--porcelain'])
    sha = command(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
    assert command(['git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/vggt-overfit-diagnostics-v1']).split()[0] == sha
    assert not any(n.startswith('vggt-feature-reading-') for n in command(['squeue', '-h', '-u', 'mawb', '-o', '%j']).splitlines())
    assert shutil.disk_usage(ROOT.parent).free > 6 * 2**30
    attempt = ROOT / 'attempts' / args.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / 'slurm').mkdir()
    snapshot = ROOT / 'snapshots' / sha
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    if not snapshot.exists():
        command(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(snapshot)], env)
        command(['git', '-C', str(snapshot), 'checkout', '--quiet', '--detach', sha], env)
    jobs = []
    def submit(mode, gpu, dependency=None):
        cmd = ['sbatch', '--parsable', '--partition=3090', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=' + ('32' if gpu else '4'), '--mem=' + ('384G' if gpu else '32G'),
            '--time=' + ('02:00:00' if gpu else '01:00:00'), '--job-name=vggt-feature-reading-' + mode,
            '--output=' + str(attempt / 'slurm/%x-%j.out'), '--error=' + str(attempt / 'slurm/%x-%j.err'),
            '--export=' + f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={attempt}']
        if gpu:
            cmd += ['--gres=gpu:8', '--nodelist=3dimage-11']
        if dependency:
            cmd += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
        cmd += [str(snapshot / 'scripts/run_vggt_feature_reading_v1.sh')]
        job = command(cmd).split(';')[0]
        jobs.append(dict(job_id=job, mode=mode, gpus=8 if gpu else 0, dependency=dependency, command=cmd))
        (attempt / 'submission.json').write_text(json.dumps(dict(status='SUBMITTING', execution_sha=sha, jobs=jobs), indent=2) + '\n')
        return job
    cache = submit('cache', True)
    submit('analyze', False, cache)
    (attempt / 'submission.json').write_text(json.dumps(dict(status='FEATURE_READING_DIAGNOSIS_SUBMITTED',
        execution_sha=sha, jobs=jobs, train_windows=1024, validation_scenes=128, feature_views=19,
        analytical_ridge_budgets=[256, 512, 1024], additional_core_training_updates=0,
        fixed_C_geometry_masks=True, GT_only_targets_and_offline_diagnostics=True,
        node='3dimage-11', Q_full312_jobs_unchanged=[64392, 64393]), indent=2) + '\n')
    print(json.dumps(dict(receipt=str(attempt / 'submission.json'), execution_sha=sha,
        jobs=[{k: j[k] for k in ('job_id', 'mode', 'gpus')} for j in jobs]), indent=2))


if __name__ == '__main__':
    main()
