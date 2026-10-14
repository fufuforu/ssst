"""Show saved evaluation progress without querying Slurm or triggering work."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_full312_v1/attempts/attempt01'))
    args = parser.parse_args()
    root = args.root
    if (root / 'COMPLETE.json').exists():
        print((root / 'COMPLETE.json').read_text())
        print('Table:', root / 'analysis/siu3r_comparison.md')
        return
    print('Reference reused:', (root / 'reference/COMPLETE.json').exists())
    print('Real GPU preflight passed:', (root / 'preflight/PREFLIGHT_VERIFIED.json').exists())
    for stage in ('preflight', 'object_support'):
        for rank in range(4):
            part = root / stage / f'rank{rank:02d}'
            if (part / 'score_complete.json').exists():
                status = 'scored'
            elif (part / 'score_progress.json').exists():
                data = json.loads((part / 'score_progress.json').read_text())
                status = f"scoring {data['completed']}/{data['total']}"
                status += f"; remaining estimate {data['elapsed_seconds'] / data['completed'] * (data['total'] - data['completed']) / 60:.1f} min"
            elif (part / 'export_progress.json').exists():
                data = json.loads((part / 'export_progress.json').read_text())
                status = f"export {data['completed']}/{data['total']} (reused {data['reused']})"
                fresh = data['newly_inferred']
                remaining = data['total'] - data['completed']
                status += f"; export remaining estimate {data['elapsed_seconds'] / fresh * remaining / 60:.1f} min"
            else:
                status = 'awaiting start/model load'
            print(stage, f'rank{rank}', status)
    print('Remaining estimates exclude subsequent phases and are not guarantees.')


if __name__ == '__main__':
    main()
