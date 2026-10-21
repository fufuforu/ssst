"""Read-only Slurm/export/scoring progress; no model, GPU or training imports."""
import argparse
import json
import subprocess
from pathlib import Path

BASE = Path('/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2_monitor/evaluation')


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def show(root):
    job = read(root / 'job.json').get('job_id')
    if job:
        status = subprocess.run(['squeue', '-j', str(job), '-o', '%i %T %M %N'], capture_output=True, text=True)
        print(status.stdout.strip())
    if (root / 'metrics.json').exists():
        report = read(root / 'metrics.json')
        print(report['status'], report['windows'], 'windows /', report['unique_scenes'], 'scenes')
        table = root / 'siu3r_table1.md'
        print(table.read_text() if table.exists() else f"Metrics: {root / 'metrics.json'}")
        return
    total = complete = exported = scored = 0
    export_eta, score_eta = [], []
    ranks = sorted(root.glob('rank*'))
    for rank in ranks:
        identity = read(rank / 'export_identity.json')
        progress = read(rank / 'progress.json')
        n = identity.get('windows', 0)
        export_done = (rank / 'export_complete.json').exists()
        score_done = (rank / 'score_complete.json').exists()
        done = n if export_done else progress.get('completed', 0)
        total += n
        complete += done
        exported += export_done
        scored += score_done
        if progress and not export_done:
            export_eta.append(progress['estimated_remaining_seconds'])
        score_progress = read(rank / 'score_progress.json')
        label = 'COMPLETE' if score_done else 'pending'
        if score_progress and not score_done:
            label = f"{score_progress['arm']} {score_progress['completed']}/{score_progress['total']}"
            passed = score_progress['completed'] + (n if score_progress['arm'] == 'novel' else 0)
            if passed:
                score_eta.append(score_progress['elapsed_seconds'] / passed * (2*n - passed))
        elif export_done and not score_done:
            records = read(rank / 'export_complete.json').get('records', [])
            native = sum((rank / 'all' / row['name'] / 'depth_scores.json').exists() for row in records)
            label = f'RGB/depth {native}/{n}'
        print(rank.name, 'export', f'{done}/{n}', 'score', label)
    print(f'Export {complete}/{total or 1860}; exported shards {exported}/{len(ranks)}; scored shards {scored}/{len(ranks)}')
    if export_eta:
        print(f'Estimated remaining export: {max(export_eta)/60:.1f} min; scoring follows.')
    if score_eta:
        print(f'Estimated remaining segmentation: {max(score_eta)/60:.1f} min; global AP reduction follows.')
    if not ranks or (not complete and not export_eta):
        print('Workers are initializing; not enough completed windows for an ETA.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    show(BASE / ('epoch04_official' if args.epoch == 4 else 'final_epoch08_official'))
