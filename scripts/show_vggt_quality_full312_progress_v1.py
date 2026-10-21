"""Read saved fixed-Q export/scoring progress without Slurm polling."""
import json
from pathlib import Path

root = Path('/space/mawb/ssst/group_plus/vggt_quality_full312_v1/attempts')
attempts = sorted(p.parent for p in root.glob('attempt*/submission.json'))
if not attempts:
    print('No fixed-Q full312 evaluation submitted.')
else:
    root = attempts[-1]
    print('Attempt:', root)
    print('Real8GPU startup passed:', (root / 'STARTUP_VERIFIED.json').exists())
    for file in sorted(root.glob('rank*/export_progress.json')):
        p = json.loads(file.read_text())
        print(file.parent.name, 'new Q export', p['completed_new'], '/', p['total_new'],
            '; remaining estimate', round(p['remaining_seconds'] / 60, 1), 'min, excludes scoring; reused128', p['reused'])
    for file in sorted(root.glob('rank*/score_progress.json')):
        p = json.loads(file.read_text())
        print(file.parent.name, 'official scoring', p['completed'], '/', p['total'])
    print('Complete:', (root / 'COMPLETE.json').exists())
    print('13-column comparison ready:', (root / 'analysis/results.json').exists())
