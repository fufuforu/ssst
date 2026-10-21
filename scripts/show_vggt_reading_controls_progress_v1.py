"""Read-only standard-library progress view; no model import or GPU request."""
import argparse
import json
from pathlib import Path
import subprocess

BASE = Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    a = p.parse_args()
    candidates = [r for r in sorted(BASE.glob('attempt*')) if (r / 'submission.json').exists() and not (r / 'SUPERSEDED_BEFORE_FORMAL_TRAINING.json').exists()]
    root = a.root or candidates[-1]
    submission = json.loads((root / 'submission.json').read_text())
    try:
        queue = subprocess.check_output(['squeue', '-h', '-u', 'mawb', '-o', '%i|%T|%M|%R'], text=True)
        jobs = {line.split('|')[0]: line.split('|')[1:] for line in queue.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        jobs = {}
    print('Evidence:', root)
    print('Snapshot:', submission['execution_sha'])
    for arm in ('baseline', 'object_local_read', 'anchor_visual', 'reliable_depth'):
        run = root / arm
        progress = json.loads((run / 'training/progress.json').read_text()) if (run / 'training/progress.json').exists() else {}
        step = progress.get('formal_updates', 0)
        speed = progress.get('seconds_per_update')
        remaining = None if speed is None else round((300 - step) * speed / 60, 1)
        evaluated = (run / 'evaluation/COMPLETE.json').exists()
        details = [(j['job_id'], j['mode'], jobs.get(j['job_id'], ['not in queue'])) for j in submission['jobs'] if j.get('arm') == arm]
        print(json.dumps(dict(arm=arm, updates=f'{step}/300', exposures=step * 8,
            approximate_remaining_training_minutes=remaining, estimate_excludes_queue_and_evaluation=True,
            evaluated=evaluated, jobs=details), ensure_ascii=False))
    print('All four startup proofs:', (root / 'ALL_STARTUPS_VERIFIED.json').exists())
    print('Final comparison:', root / 'analysis/report.md')


if __name__ == '__main__':
    main()
