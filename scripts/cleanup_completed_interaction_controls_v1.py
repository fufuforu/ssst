"""User-authorized pruning of completed controls; exact checkpoint allowlist only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import subprocess

WORK = Path('/space/mawb/ssst/workspace_group_plus/vggt_interaction_controls_v1/attempt02')
REPORT = Path('/space/mawb/ssst/group_plus/vggt_interaction_controls_v1/attempts/attempt02')
OUT = REPORT / 'checkpoint_cleanup_20261013'
ARMS = ('bidirectional', 'forward_only', 'gated_feedback', 'independent_feedback')
JOBS = tuple(str(i) for i in range(63630, 63639)) + ('63647',)


def read(p):
    return json.loads(p.read_text())


def stat(p, small=False):
    assert p.is_file() and not p.is_symlink(), str(p)
    s = p.stat()
    d = dict(path=str(p), bytes=s.st_size, inode=s.st_ino, device=s.st_dev,
             mtime_ns=s.st_mtime_ns, nlink=s.st_nlink)
    if small and s.st_size < 8*1024**2:
        d['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d


def verify_jobs():
    active = subprocess.check_output(['squeue', '-h', '-u', 'mawb', '-o', '%i|%j|%T|%o'], text=True)
    assert 'vggt-interact-' not in active and str(WORK) not in active and str(REPORT) not in active
    completed = subprocess.check_output(['sacct', '-n', '-X', '-j', ','.join(JOBS),
                                        '-o', 'JobIDRaw,State,ExitCode', '-P'], text=True)
    seen = {x.split('|')[0]: x.split('|')[1:3] for x in completed.splitlines() if x.strip()}
    assert all(seen[j] == ['COMPLETED', '0:0'] for j in JOBS)
    return dict(active=active, completed=completed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    assert not (OUT/'receipt.json').exists(), 'preserve completed cleanup'
    jobs = verify_jobs()
    assert read(REPORT/'evaluation/COMPLETE.json')['conditions'] == 16
    assert read(REPORT/'analysis_review_20261013/audit.json')['status'] == 'SAVED_ARTIFACTS_VERIFIED'
    candidates = []
    protected = [WORK/'bidirectional/checkpoint_update_0500.pt']
    source = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1')
    protected += [source/'checkpoint_reconstruction_adaptation_epoch_02.pt', source/'adapted_vggt.pt']
    for arm in ARMS:
        complete = read(WORK/arm/'COMPLETE.json')
        assert (complete['completed_updates'], complete['completed_exposures']) == (500, 4000)
        protected += [WORK/arm/'COMPLETE.json', WORK/arm/'run_manifest.json']
        for step in (0, 20, 100, 300, 500):
            path = WORK/arm/f'checkpoint_update_{step:04d}.pt'
            receipt = read(path.with_suffix('.json'))
            assert receipt['status'] == 'CHECKPOINT_VERIFIED' and receipt['step'] == step
            protected.append(path.with_suffix('.json'))
            if step != 20:
                ev = REPORT/'evaluation'/arm/f'update{step:04d}'
                assert read(ev/'metrics.json')['checkpoint_sha256'] == receipt['sha256']
                protected += [ev/'metrics.json', ev/'export_complete.json', ev/'score_complete.json']
            if arm == 'bidirectional' and step == 500:
                continue
            fingerprint = stat(path)
            assert fingerprint['nlink'] == 1 and path.resolve().parent == (WORK/arm).resolve()
            candidates.append(dict(fingerprint, saved_receipt=receipt, completed_training=complete))
    assert len(candidates) == 19
    if not a.apply:
        OUT.mkdir(exist_ok=False)
        data = dict(status='AUDITED_NOT_DELETED', authorization='User requested deletion of obsolete four-control files',
                    jobs=jobs, entries=candidates, protected=[stat(x, True) for x in protected],
                    free_bytes_before=os.statvfs(WORK).f_bavail*os.statvfs(WORK).f_frsize)
        (OUT/'manifest.json').write_text(json.dumps(data, indent=2)+'\n')
        print('AUDITED', len(candidates), sum(x['bytes'] for x in candidates)/2**30, 'GiB')
        return
    manifest = read(OUT/'manifest.json')
    assert [x['path'] for x in candidates] == [x['path'] for x in manifest['entries']]
    for row in manifest['entries'] + manifest['protected']:
        actual = stat(Path(row['path']), 'sha256' in row)
        assert all(actual[k] == v for k, v in row.items() if k in actual), row['path']
    with (OUT/'deletions.jsonl').open('x') as stream:
        for row in manifest['entries']:
            Path(row['path']).unlink()
            stream.write(json.dumps(dict(path=row['path'], bytes=row['bytes'],
                deleted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
            stream.flush(); os.fsync(stream.fileno())
    for row in manifest['protected']:
        assert stat(Path(row['path']), 'sha256' in row) == row
    result = dict(status='CLEANUP_COMPLETED', deleted_files=19,
                  deleted_bytes=sum(x['bytes'] for x in manifest['entries']),
                  free_bytes_after=os.statvfs(WORK).f_bavail*os.statvfs(WORK).f_frsize,
                  retained_checkpoint=str(protected[0]), protected_fingerprints_unchanged=True,
                  all_scores_exports_logs_failure_records_snapshots_and_active_runs_preserved=True,
                  jobs_after=verify_jobs())
    (OUT/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
