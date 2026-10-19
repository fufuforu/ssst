"""Submit one prepared eight4090 concat/shared full4 job on node17; inspect once."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(run):
    pre = read(run/'preflight.json')
    assert pre['status'] == 'PREFLIGHT_PASSED' and pre['architecture_key'] == 'concat_shared'
    assert Path(pre['snapshot']).resolve() == ROOT.resolve()
    assert Path(pre['run']).resolve() == run.resolve()
    assert pre['shared_value_mismatches'] == [] and pre['shared_exact_tensor_count'] > 300
    assert pre['optimizer_empty'] and pre['initial_progress'] == dict(epoch=0, update_in_epoch=0, full_updates=0)
    assert pre['budget']['world_size'] == 8 and pre['budget']['updates'] == 4172
    assert pre['gaussians'] == 65536 and pre['rank_rng_exactly_matches_source']
    assert pre['concat_shared']['gfm_adapters'] == 1 and pre['concat_shared']['decoder_depth'] == 12
    assert read(run/'VALIDATION_EVIDENCE.json')['targeted_cpu_tests']['passed'] == 13
    assert read(run/'VALIDATION_EVIDENCE.json')['status'] == 'PASSED'
    assert read(run/'model_config.json') == pre['config']
    assert digest(run/'checkpoint_untrained_initial.pt') == pre['initial_sha256']
    manifest = read(ROOT/'SOURCE_MANIFEST.json')
    for name, sha in manifest['overlay_sha256'].items():
        assert digest(ROOT/name) == sha, name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == manifest['base_git_sha']
    # Check the entire versioned code tree, excluding only the explicitly
    # recorded overlay. This also protects unchanged trainer dependencies.
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    assert set(changed) <= set(manifest['overlay_sha256']), changed
    return pre, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run = args.run
    pre, manifest = verify(run)
    if args.verify_only:
        print('SOURCE_AND_INITIAL_PREFLIGHT_PASSED', flush=True)
        return
    assert not any((run/n).exists() for n in ('submission.json', 'SUBMISSION_REQUESTED.json', 'STARTED.json', 'checkpoint_latest.pt'))
    command = ['sbatch', '--parsable', '--job-name=concat_shared_full4', '--partition=4090',
               '--nodelist=3dimage-17', '--nodes=1', '--gres=gpu:4090:8', '--cpus-per-task=32',
               '--mem=192G', '--time=1-00:00:00', '--chdir='+str(ROOT),
               '--output='+str(run/'slurm-%j.out'), str(ROOT/'scripts/run_concat_shared_full.sh'), str(run)]
    receipt = dict(requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=command,
                   node='3dimage-17', gpus='4090:8', training_jobs=1, evaluation_jobs=0,
                   snapshot=str(ROOT), run=str(run), initial_sha256=pre['initial_sha256'],
                   source_manifest=manifest, monitoring_policy='One bounded status check, then stop; no background monitoring')
    # Exclusive creation prevents accidental duplicate submissions.
    with (run/'SUBMISSION_REQUESTED.json').open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    output = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    receipt['job_id'] = int(output.split(';')[0])
    write(run/'submission.json', receipt)
    try:
        state = subprocess.run(['squeue', '-h', '-j', str(receipt['job_id']), '-o', '%i|%T|%N|%R'],
                               capture_output=True, text=True, timeout=15)
        check = dict(stdout=state.stdout.strip(), stderr=state.stderr.strip(), returncode=state.returncode)
    except subprocess.TimeoutExpired:
        check = dict(stdout='', stderr='Single bounded check timed out; not retried', returncode=124)
    check.update(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 job_failed=read(run/'JOB_FAILED.json') if (run/'JOB_FAILED.json').exists() else None,
                 gpu_first_update_verified=False, eight_rank_sync_verified=False, polling_stopped=True)
    write(run/'SUBMISSION_ONCE_CHECK.json', check)
    print(json.dumps(dict(job_id=receipt['job_id'], once_check=check, run=str(run)), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
