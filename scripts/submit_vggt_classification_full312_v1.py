"""Submit evaluation only: reused A, four-GPU C preflight/full export, CPU summary."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/space/mawb/ssst/group_plus/vggt_classification_readback_full312_v1')


def command(args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='attempt01')
    parser.add_argument('--reference-from', help='Reuse an already successful reference in a preserved prior attempt.')
    args = parser.parse_args()
    assert Path(args.attempt).name == args.attempt and args.attempt.startswith('attempt')
    assert not command(['git', '-C', str(REPO), 'status', '--porcelain'])
    sha = command(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
    assert command(['git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/vggt-overfit-diagnostics-v1']).split()[0] == sha
    assert not any(name.startswith('vggt-cls-full312-') for name in command(['squeue', '-h', '-u', 'mawb', '-o', '%j']).splitlines())
    assert shutil.disk_usage(ROOT.parent).free > 8 * 2**30
    attempt = ROOT / 'attempts' / args.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    logs = attempt / 'slurm'
    logs.mkdir()
    snapshot = ROOT / 'snapshots' / sha
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    command(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(snapshot)], env)
    command(['git', '-C', str(snapshot), 'checkout', '--quiet', '--detach', sha], env)
    jobs = []
    receipt = attempt / 'submission.json'

    def submit(mode, output, gpu=False, dependency=None):
        exports = f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={output},TASK_PREFLIGHT={attempt}/preflight'
        cmd = ['sbatch', '--parsable', '--partition=3090', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=' + ('16' if gpu else '4'), '--mem=' + ('192G' if gpu else '32G'),
            '--time=' + ('03:00:00' if mode == 'evaluate' else '01:00:00'),
            '--job-name=vggt-cls-full312-' + mode, '--output=' + str(logs / '%x-%j.out'),
            '--error=' + str(logs / '%x-%j.err'), '--export=' + exports]
        if gpu:
            cmd += ['--gres=gpu:4', '--nodelist=3dimage-13']
        if dependency:
            cmd += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
        cmd += [str(snapshot / 'scripts/run_vggt_classification_full312_v1.sh')]
        job = command(cmd).split(';')[0]
        jobs.append(dict(job_id=job, mode=mode, output=str(output), gpus=4 if gpu else 0, dependency=dependency, command=cmd))
        receipt.write_text(json.dumps(dict(status='SUBMITTING', execution_sha=sha, jobs=jobs), indent=2) + '\n')
        return job

    if args.reference_from:
        assert Path(args.reference_from).name == args.reference_from
        previous = ROOT / 'attempts' / args.reference_from / 'reference'
        done = json.loads((previous / 'COMPLETE.json').read_text())
        assert done['status'] == 'FULL312_REFERENCE_REUSED' and done['native_reference_metrics_reproduced']
        assert done['native_prior128_C_merge_reproduced']
        shutil.copytree(previous, attempt / 'reference')
        (attempt / 'reference/REFERENCE_REUSE.json').write_text(json.dumps(dict(
            status='SUCCESSFUL_PRIOR_REFERENCE_REUSED', source=str(previous),
            new_evaluation_execution_sha=sha, model_forward_calls=0, optimizer_updates=0), indent=2) + '\n')
        reference = None
    else:
        reference = submit('reference', attempt / 'reference')
    preflight = submit('preflight', attempt / 'preflight', True)
    formal = submit('evaluate', attempt / 'object_support', True, ':'.join([job for job in (reference, preflight) if job]))
    submit('comparison', attempt, dependency=formal)
    receipt.write_text(json.dumps(dict(status='FULL312_A_C_EVALUATION_SUBMITTED', execution_sha=sha,
        attempt=str(attempt), jobs=jobs, reused_reference_from=args.reference_from, scenes=312, windows=1860, images=11160,
        A_all1860_outputs_and_native_states_reused=True, C_existing128_outputs_and_native_states_reused=True,
        C_new_windows=1732, additional_training_updates=0, official_columns=13, text_columns_blank=True), indent=2) + '\n')
    print(json.dumps(dict(receipt=str(receipt), execution_sha=sha, jobs=[{k: j[k] for k in ('job_id', 'mode', 'gpus', 'dependency')} for j in jobs]), indent=2))


if __name__ == '__main__':
    main()
