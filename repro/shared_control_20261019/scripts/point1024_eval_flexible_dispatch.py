"""Two dependent official evaluations and ONE replacement of unstarted67976.

No model edits, no in-flight training cancellation, no polling after one check.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path('/space/mawb/ssst/group_plus/unified_query_training_v1')
ALLOWED = {'3dimage-13', '3dimage-14', '3dimage-17', '3dimage-18'}
EVALS = [
    dict(train_job=67952, run=BASE/'attempt21_A_dual_point1024_full_node13_8gpu',
         snapshot=Path('/space/mawb/ssst_A_point1024_full_code_20261019'), node='3dimage-13', gpu='3090'),
    dict(train_job=67973, run=BASE/'attempt22_concat_shared_full_node17_8gpu',
         snapshot=Path('/space/mawb/ssst_concat_shared_full_code_20261019'), node='3dimage-17', gpu='4090')]
FLEX_RUN = BASE/'attempt23_recon_then_object_full_node18_8gpu'
FLEX_CODE = Path('/space/mawb/ssst_recon_then_object_full_code_20261019')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_control():
    for name, sha in read(ROOT/'SOURCE_MANIFEST.json')['sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Control source changed: '+name)


def verify_snapshot(snapshot):
    manifest = read(snapshot/'SOURCE_MANIFEST.json')
    for name, sha in manifest['overlay_sha256'].items():
        if digest(snapshot/name) != sha:
            raise ValueError('Historical training snapshot changed: '+name)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=snapshot, text=True).strip()
    if sha != manifest['base_git_sha']:
        raise ValueError('Historical training HEAD changed')


def is_unstarted(state, run):
    return state == 'PENDING' and not any((run/n).exists() for n in
        ('STARTED.json', 'checkpoint_latest.pt', 'checkpoint_initial.pt', 'TRAINING_COMPLETE.json'))


def evaluation_is_new(run):
    return not any((run/n).exists() for n in ('EVALUATION_SUBMISSION.json', 'EVALUATION_SUBMISSION_REQUESTED.json',
        'EVALUATION_STARTED.json', 'COMPLETE.json', 'evaluation/export_complete.json'))


def evaluation_command(spec):
    return ['sbatch', '--parsable', '--job-name=point1024_eval_'+str(spec['train_job']),
        '--partition='+spec['gpu'], '--nodelist='+spec['node'], '--nodes=1', '--gres=gpu:'+spec['gpu']+':1',
        '--cpus-per-task=4', '--mem=64G', '--time=12:00:00', '--kill-on-invalid-dep=yes',
        '--dependency=afterok:'+str(spec['train_job']), '--chdir='+str(spec['snapshot']),
        '--output='+str(spec['run']/'evaluation-slurm-%j.out'), str(ROOT/'scripts/run_point1024_official_eval.sh'),
        str(spec['run']), str(spec['snapshot']), str(ROOT)]


def flexible_command(excluded):
    return ['sbatch', '--parsable', '--job-name=recon_then_object_flexible8', '--partition=3090,4090',
        '--nodes=1', '--gres=gpu:8', '--cpus-per-task=32', '--mem=192G', '--time=1-00:00:00',
        '--exclude='+','.join(sorted(excluded)), '--chdir='+str(FLEX_CODE),
        '--output='+str(FLEX_RUN/'slurm-%j.out'), str(ROOT/'scripts/run_recon_then_object_flexible.sh'),
        str(FLEX_RUN), str(FLEX_CODE), str(ROOT)]


def state_from_oneline(output):
    fields = dict(x.split('=', 1) for x in output.split() if '=' in x)
    return fields.get('JobState')


def runtime_check(args):
    verify_control(); verify_snapshot(args.snapshot)
    pre = read(args.run/'preflight.json')
    if Path(pre['snapshot']).resolve() != args.snapshot.resolve():
        raise ValueError('Run/snapshot mismatch')
    node = os.environ.get('SLURMD_NODENAME')
    if args.runtime_eval_check:
        spec = next((s for s in EVALS if s['run'].resolve() == args.run.resolve()), None)
        if spec is None or node != spec['node']:
            raise ValueError('Evaluation must stay on its corresponding node')
        complete = read(args.run/'TRAINING_COMPLETE.json')
        started = read(args.run/'STARTED.json')
        if complete.get('status') != 'TRAINING_COMPLETE' or complete['progress'] != dict(epoch=4, update_in_epoch=0, full_updates=4172):
            raise ValueError('Not completed full4 training')
        if started['architecture'] != pre['architecture'] or started['config'] != pre['config'] or started['short_validation']:
            raise ValueError('Training architecture/config differs')
        write(args.run/'EVALUATION_STARTED.json', dict(job_id=os.environ['SLURM_JOB_ID'], node=node,
            training_job=spec['train_job'], original_snapshot=str(args.snapshot), started_utc=now(),
            protocol='unchanged official heldout128, 11 metrics, no parameter updates'), exclusive=True)
    else:
        if node not in ALLOWED or not is_unstarted('PENDING', args.run):
            raise ValueError('Unexpected node or already-started training; never replay updates')
        if digest(args.run/'checkpoint_untrained_initial.pt') != pre['initial_sha256']:
            raise ValueError('Initial checkpoint changed')
        write(args.run/'FLEX_ALLOCATION.json', dict(job_id=os.environ['SLURM_JOB_ID'], predecessor=67976,
            node=node, gpu_type='3090' if node == '3dimage-13' else '4090', world_size=8,
            slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            actual_start_utc=now(), source_snapshot=str(args.snapshot)), exclusive=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-eval-check', action='store_true')
    p.add_argument('--runtime-training-check', action='store_true')
    p.add_argument('--run', type=Path); p.add_argument('--snapshot', type=Path)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if args.runtime_eval_check or args.runtime_training_check:
        runtime_check(args); return
    verify_control()
    for spec in EVALS:
        verify_snapshot(spec['snapshot'])
        if not evaluation_is_new(spec['run']):
            raise ValueError('Existing evaluation; inspect rather than resubmit: '+str(spec['run']))
        configuration = read(spec['run']/'configuration.json')
        if not (configuration.get('recipe') or configuration.get('architecture')):
            raise ValueError('Missing evaluation model label')
    verify_snapshot(FLEX_CODE)
    pre = read(FLEX_RUN/'preflight.json')
    if not is_unstarted('PENDING', FLEX_RUN) or digest(FLEX_RUN/'checkpoint_untrained_initial.pt') != pre['initial_sha256']:
        raise ValueError('Only a genuinely unstarted zero-step run can change placement')
    from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO, SIU3R_COMMIT
    if subprocess.check_output(['git', '-C', SIU3R_REPO, 'rev-parse', 'HEAD'], text=True).strip() != SIU3R_COMMIT:
        raise ValueError('Official evaluator commit differs')
    if subprocess.check_output(['git', '-C', SIU3R_REPO, 'status', '--porcelain', '--', 'src'], text=True).strip():
        raise ValueError('Official evaluator src changed')
    inventory = subprocess.check_output(['sinfo', '-h', '-N', '-p', '3090,4090', '-o', '%N'], text=True).split()
    if not ALLOWED <= set(inventory):
        raise ValueError('Allowed node inventory differs')
    excluded = set(inventory)-ALLOWED
    if not excluded:
        raise ValueError('Unexpected unrestricted node set')
    flex_cmd = flexible_command(excluded)
    test = subprocess.run([flex_cmd[0], '--test-only']+flex_cmd[2:], capture_output=True, text=True, timeout=30)
    if test.returncode:
        raise ValueError('Flexible scheduling validation failed: '+test.stderr)
    plan = dict(prepared_utc=now(), evaluations=[evaluation_command(s) for s in EVALS],
        flexible_training=flex_cmd, allowed_nodes=sorted(ALLOWED), excluded_nodes=sorted(excluded),
        source_snapshot_unchanged=True, initial_sha256=pre['initial_sha256'],
        official_commit=SIU3R_COMMIT, flexible_sbatch_test_returncode=test.returncode)
    write(ROOT/'PREPARED_PLAN.json', plan)
    if args.preflight_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    write(ROOT/'ACTION_REQUESTED.json', dict(requested_utc=now(), plan=plan), exclusive=True)
    receipt = dict(started_utc=now(), evaluations=[], replacement=None)
    def save():
        write(ROOT/'DISPATCH_RECEIPT.json', receipt)
    save()
    for spec in EVALS:
        command = evaluation_command(spec)
        payload = dict(training_job=spec['train_job'], node=spec['node'], gpu=spec['gpu']+':1',
            dependency='afterok:'+str(spec['train_job']), original_snapshot=str(spec['snapshot']),
            command=command, control_code=str(ROOT), submitted_utc=now())
        write(spec['run']/'EVALUATION_SUBMISSION_REQUESTED.json', payload, exclusive=True)
        output = subprocess.check_output(command, cwd=spec['snapshot'], text=True).strip()
        payload['job_id'] = int(output.split(';')[0])
        write(spec['run']/'EVALUATION_SUBMISSION.json', payload)
        receipt['evaluations'].append(payload); save()
    # Prevent a queued-to-running race before cancelling the old node18-only job.
    old = subprocess.check_output(['scontrol', 'show', 'job', '-o', '67976'], text=True)
    if not is_unstarted(state_from_oneline(old), FLEX_RUN):
        receipt['replacement'] = dict(status='NOT_REPLACED_ALREADY_STARTED', old_job=67976, scheduler=old)
    else:
        held = subprocess.run(['scontrol', 'hold', '67976'], capture_output=True, text=True)
        after_hold = subprocess.check_output(['scontrol', 'show', 'job', '-o', '67976'], text=True)
        if held.returncode or not is_unstarted(state_from_oneline(after_hold), FLEX_RUN):
            release = None
            if held.returncode == 0:
                release = subprocess.run(['scontrol', 'release', '67976'], capture_output=True, text=True)
            receipt['replacement'] = dict(status='NOT_CANCELLED_HOLD_RACE', old_job=67976,
                hold_returncode=held.returncode, hold_stderr=held.stderr, scheduler=after_hold,
                release_returncode=None if release is None else release.returncode)
        else:
            request = dict(old_job=67976, old_state='PENDING', old_held_scheduler=after_hold,
                replacement_command=flex_cmd, source_snapshot=str(FLEX_CODE), initial_sha256=pre['initial_sha256'],
                run_directory_reused_without_updates=True, requested_utc=now())
            write(FLEX_RUN/'FLEX_SUBMISSION_REQUESTED.json', request, exclusive=True)
            subprocess.run(['scancel', '67976'], check=True)
            request['old_cancel_request_accepted'] = True
            write(FLEX_RUN/'FLEX_SUBMISSION_REQUESTED.json', request)
            output = subprocess.check_output(flex_cmd, cwd=FLEX_CODE, text=True).strip()
            request.update(job_id=int(output.split(';')[0]), status='REPLACED_UNSTARTED_JOB',
                allowed_nodes=sorted(ALLOWED), excluded_nodes=sorted(excluded), gpu_request='generic_gpu:8', submitted_utc=now())
            write(FLEX_RUN/'FLEX_SUBMISSION.json', request)
            write(FLEX_RUN/'ACTIVE_TRAINING_JOB.json', dict(job_id=request['job_id'], predecessor_cancelled=67976,
                placement='one node, eight GPUs, any of13/14/17/18', source_snapshot=str(FLEX_CODE)))
            receipt['replacement'] = request
    save()
    jobs = [str(p['job_id']) for p in receipt['evaluations']]
    jobs.append(str((receipt['replacement'] or {}).get('job_id', 67976)))
    # Exactly one bounded post-action status check; no log reads or retries.
    try:
        state = subprocess.run(['squeue', '-h', '-j', ','.join(jobs), '-o', '%i|%T|%N|%R|%P|%b'],
            capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        state = subprocess.CompletedProcess([], 124, '', 'Single bounded check timed out; not retried')
    receipt['once_check'] = dict(checked_utc=now(), jobs=jobs, stdout=state.stdout.strip(),
        stderr=state.stderr.strip(), returncode=state.returncode, polling_stopped=True)
    if receipt['replacement'].get('job_id'):
        try:
            cancelled = subprocess.run(['scontrol', 'show', 'job', '-o', '67976'],
                capture_output=True, text=True, timeout=15)
            receipt['once_check']['old_job_state'] = state_from_oneline(cancelled.stdout)
            receipt['once_check']['old_job_check_returncode'] = cancelled.returncode
        except subprocess.TimeoutExpired:
            receipt['once_check']['old_job_check_returncode'] = 124
    write(ROOT/'SUBMISSION_ONCE_CHECK.json', receipt['once_check']); save()
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
