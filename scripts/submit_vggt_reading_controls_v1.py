"""Immutable matched8GPU reading controls; preflight, two resource waves, evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1')


def cmd(args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', default='attempt01')
    a = p.parse_args()
    assert Path(a.attempt).name == a.attempt and a.attempt.startswith('attempt')
    assert not cmd(['git', '-C', str(REPO), 'status', '--porcelain'])
    sha = cmd(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
    assert cmd(['git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/vggt-overfit-diagnostics-v1']).split()[0] == sha
    assert not any(n.startswith('vggt-reading-') for n in cmd(['squeue', '-h', '-u', 'mawb', '-o', '%j']).splitlines())
    assert shutil.disk_usage(ROOT.parent).free > 12 * 2**30
    attempt = ROOT / 'attempts' / a.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / 'slurm').mkdir()
    snapshot = ROOT / 'snapshots' / sha
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    if not snapshot.exists():
        cmd(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(REPO), str(snapshot)], env)
        cmd(['git', '-C', str(snapshot), 'checkout', '--quiet', '--detach', sha], env)
    jobs = []
    def submit(mode, node, output, arm=None, dependency=None):
        gpu = mode not in ('summary', 'startup_gate')
        arguments = ['sbatch', '--parsable', '--partition=3090', '--nodelist=' + node,
            '--nodes=1', '--ntasks=1', '--cpus-per-task=' + ('32' if gpu else '4'),
            '--mem=' + ('384G' if gpu else '24G'), '--time=' + ('04:00:00' if gpu else '01:00:00'),
            '--job-name=vggt-reading-' + (arm or mode), '--output=' + str(attempt / 'slurm/%x-%j.out'),
            '--error=' + str(attempt / 'slurm/%x-%j.err'),
            '--export=' + f'ALL,TASK_REPO={snapshot},TASK_CODE_SHA={sha},TASK_MODE={mode},TASK_OUTPUT={output},TASK_PREFLIGHT={attempt}/preflight' + (',TASK_ARM=' + arm if arm else '')]
        if gpu:
            arguments += ['--gres=gpu:8']
        if dependency:
            arguments += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
        arguments += [str(snapshot / 'scripts/run_vggt_reading_controls_v1.sh')]
        jid = cmd(arguments).split(';')[0]
        jobs.append(dict(job_id=jid, mode=mode, arm=arm, node=node, gpus=8 if gpu else 0, dependency=dependency, output=str(output), command=arguments))
        (attempt / 'submission.json').write_text(json.dumps(dict(status='SUBMITTING', execution_sha=sha, jobs=jobs), indent=2) + '\n')
        return jid
    pre = submit('preflight', '3dimage-11', attempt / 'preflight')
    bstart = submit('startup', '3dimage-11', attempt / 'baseline', 'baseline', pre)
    rstart = submit('startup', '3dimage-13', attempt / 'object_local_read', 'object_local_read', pre)
    vstart = submit('startup', '3dimage-11', attempt / 'anchor_visual', 'anchor_visual', bstart)
    dstart = submit('startup', '3dimage-13', attempt / 'reliable_depth', 'reliable_depth', rstart)
    gate = submit('startup_gate', '3dimage-11', attempt, dependency=':'.join([bstart, rstart, vstart, dstart]))
    baseline = submit('train_eval', '3dimage-11', attempt / 'baseline', 'baseline', gate)
    read = submit('train_eval', '3dimage-13', attempt / 'object_local_read', 'object_local_read', gate)
    visual = submit('train_eval', '3dimage-11', attempt / 'anchor_visual', 'anchor_visual', baseline)
    depth = submit('train_eval', '3dimage-13', attempt / 'reliable_depth', 'reliable_depth', read)
    submit('summary', '3dimage-11', attempt, dependency=':'.join([baseline, read, visual, depth]))
    receipt = dict(status='FOUR_MATCHED_READING_CONTROLS_SUBMITTED', execution_sha=sha, jobs=jobs,
        updates_per_arm=300, new_exposures_per_arm=2400, heldout_scenes=128, global_batch=8,
        geometry_quality_policy='monitor_v1', Q_applied=False, checkpoint_policy='one atomic rolling resumable downstream patch per arm',
        polling_endpoint='all four arms actual20 updates/160 exposures and startup synchronization verified')
    (attempt / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(attempt / 'submission.json'), execution_sha=sha, jobs=jobs), indent=2))


if __name__ == '__main__':
    main()
