"""Pinned official per-window scores for associations, never averaged into global AP."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from scripts.score_object_locus_posefree_official import create_worker
from scripts.posefree_official_state_helpers import update, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(4)
    assert subprocess.check_output(['git', '-C', '/space/mawb/SIU3R', 'rev-parse', 'HEAD'], text=True).strip() == '8ea80166be76854f938e90521f1a5b688b755c87'
    assert not subprocess.check_output(['git', '-C', '/space/mawb/SIU3R', 'diff', '--name-only', 'HEAD', '--', 'src'], text=True).strip()
    paths = dict(C=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation'),
                 baseline300=Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04/baseline/evaluation'))
    out = a.root / 'official_per_window_association_scores.json'
    if a.merge:
        data = dict(status='OFFICIAL_WINDOW_ASSOCIATION_SCORES_COMPLETE', conditions={k: {} for k in paths},
            scope='Per-window official scores for associations only; full global metrics are not window averages.', optimizer_updates=0)
        shards = list(a.root.glob('official_window_shard*.json'))
        assert len(shards) == a.shards
        for f in shards:
            d = json.loads(f.read_text()); assert d['status'] == 'OFFICIAL_WINDOW_ASSOCIATION_SHARD_COMPLETE'
            for arm in paths:
                for n, value in d['conditions'][arm].items():
                    assert n not in data['conditions'][arm] or data['conditions'][arm][n] == value
                    data['conditions'][arm][n] = value
        assert all(len(x) == 128 for x in data['conditions'].values())
        out.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        return
    assert 0 <= a.shard < a.shards
    seed = out
    if a.shards > 1:
        out = a.root / f'official_window_shard{a.shard:02d}.json'
    data = json.loads(out.read_text()) if out.exists() else json.loads(seed.read_text()) if seed.exists() else dict(status='IN_PROGRESS', conditions={k: {} for k in paths},
        scope='Pinned official metrics on each individual window for associations only; do not average window AP/PQ into a model score.', optimizer_updates=0)
    index = -1
    for arm, path in paths.items():
        for receipt in sorted(path.glob('rank*/export_complete.json')):
            for r in json.loads(receipt.read_text())['records']:
                index += 1
                if index % a.shards != a.shard:
                    continue
                if r['name'] in data['conditions'][arm]:
                    continue
                pair = receipt.parent / 'all' / r['name']
                worker = create_worker(receipt.parent / 'all', 'cpu')
                for view in ('context', 'target'):
                    update(worker, worker.process_segmentation(pair / f'{view}_seg_pred', pair / f'{view}_seg_gt'), view)
                v = result(worker)
                selected = {f'{view}_{metric}': (v[f'{view}_map']['map'] if metric == 'map' else v[f'{view}_{metric}'])
                            for view in ('context', 'target') for metric in ('miou', 'pq', 'map')}
                data['conditions'][arm][r['name']] = selected
                tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(out)
                print('OFFICIAL_WINDOW_SCORED', arm, len(data['conditions'][arm]), r['name'], flush=True)
    if a.shards == 1:
        assert all(len(x) == 128 for x in data['conditions'].values())
    data['status'] = 'OFFICIAL_WINDOW_ASSOCIATION_SCORES_COMPLETE' if a.shards == 1 else 'OFFICIAL_WINDOW_ASSOCIATION_SHARD_COMPLETE'
    out.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
