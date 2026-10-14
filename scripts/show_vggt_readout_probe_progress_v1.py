"""Read saved streaming cache/head/evaluation progress without Slurm polling."""
import json
from pathlib import Path

root = Path('/space/mawb/ssst/group_plus/vggt_readout_probe_v1/attempts')
attempts = sorted(p.parent for p in root.glob('attempt*/submission.json'))
if not attempts:
    print('No readout attempt submitted.')
else:
    root = attempts[-1]
    print('Attempt:', root)
    print('Real GPU preflight passed:', (root / 'preflight/PREFLIGHT_VERIFIED.json').exists())
    if (root / 'training/progress.json').exists():
        p = json.loads((root / 'training/progress.json').read_text())
        print('Q and O:', p['completed_updates'], '/', p['total_updates'], 'updates each;', p['exposures_per_head'], 'exposures each')
        print('Shared-cache/head-training remaining estimate:', round(p['remaining_seconds'] / 60, 1), 'min; excludes evaluation')
    for file in sorted((root / 'evaluation').glob('export_progress_rank*.json')):
        p = json.loads(file.read_text())
        print(file.stem, p['completed'], '/', p['total'])
    print('Automatic evaluation done:', (root / 'COMPLETE.json').exists())
    print('Comparison ready:', (root / 'analysis/results.json').exists())
