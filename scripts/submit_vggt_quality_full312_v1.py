"""Submit fixed Q all312 evaluation on eight3090; no new training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/space/mawb/ssst/group_plus/vggt_quality_full312_v1')


def command(args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='attempt01')
    args = parser.parse_args()
    assert Path(args.attempt).name == args.attempt and args.attempt.startswith('attempt')
    assert not command(['git', '-C', str(REPO), 'status', '--porcelain'])
    sha = command(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
    assert command(['git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/vggt-overfit-diagnostics-v1']).split()[0] == sha
    assert not any(n.startswith('vggt-Q-full312-') for n in command(['squeue', '-h', '-u', 'mawb', '-o', '%j']).splitlines())
    assert shutil.disk_usage(ROOT.parent).free > 5 * 2**30
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

    def submit(mode, gpu_count, dependency=None):
        cmd = ['sbatch', '--parsable', '--partition=3090', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=' + ('32' if gpu_count else '4'), '--mem=' + ('384G' if gpu_count else '64G'),
            '--time=' + ('03:00:00' if gpu_count else '01:00:00'), '--job-name=vggt-Q-full312-' + mode,
            '--output=' + str(attempt / 'slurm/%x-%j.out'), '--error=' + str(attempt / 'slurm/%x-%j.err'),
            '--export=' + f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={attempt}']
        if gpu_count:
            cmd += ['--gres=gpu:8', '--nodelist=3dimage-13']
        if dependency:
            cmd += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
        cmd += [str(snapshot / 'scripts/run_vggt_quality_full312_v1.sh')]
        job = command(cmd).split(';')[0]
        jobs.append(dict(job_id=job, mode=mode, gpus=gpu_count, dependency=dependency, command=cmd))
        (attempt / 'submission.json').write_text(json.dumps(dict(status='SUBMITTING', execution_sha=sha, jobs=jobs), indent=2) + '\n')
        return job

    evaluation = submit('evaluate', 8)
    submit('reduce', 0, evaluation)
    receipt = dict(status='FIXED_Q_FULL312_SUBMITTED', execution_sha=sha, attempt=str(attempt), jobs=jobs,
        scenes=312, windows=1860, original_Q128_outputs_and_states_reused=True, new_Q_windows=1732,
        all_C_RGB_depth_masks_GT_reused=True, Q_frozen_update300=True, additional_training_updates=0,
        official_columns=13, text_columns_blank=True, confirmation184_AP=True,
        source_Q_sha256='61dd9a8fbd8915c1c8779e61c6e0a3997b2b3d15a2a9639e297b1c0946ffcee5')
    (attempt / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(attempt / 'submission.json'), execution_sha=sha,
        jobs=[{k: j[k] for k in ('job_id', 'mode', 'gpus')} for j in jobs]), indent=2))


if __name__ == '__main__':
    main()
