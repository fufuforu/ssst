"""Audit completed full312 exports/states and inspect gains without model execution."""
import argparse
import ast
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.train_vggt_classification_readback_v1 import OFFICIAL, SOURCE_SHA, HELDOUT
from scripts.eval_vggt_classification_full312_v1 import OLD_C, TRAIN
from scripts.format_object_locus_posefree_siu3r_table import table_values
from scripts.posefree_official_state_helpers import create, itemize


def read(path):
    return json.loads(path.read_text())


def equal_tree(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_tree(x, y) for x, y in zip(left, right))
    return left == right


def ap_subset(parts, chosen, view, path):
    names = [n for part in parts for n in part['names']]
    indices = sorted((i for i, name in enumerate(names) if name in chosen), key=lambda i: names[i])
    assert len(indices) == len(chosen)
    key = view + '_mAP'
    states = {k: [value for part in parts for value in part['states'][key][k]] for k in parts[0]['states'][key]}
    assert all(len(value) in (0, 1860) for value in states.values())
    states = {k: [value[i] for i in indices] if len(value) else [] for k, value in states.items()}
    metric = create(path).target_mAP
    for key, value in states.items():
        setattr(metric, key, value)
    metric._update_count = len(indices)
    answer = itemize(metric.compute())
    return {k: answer[k] for k in ('map', 'map_50', 'map_75', 'classes', 'map_per_class')}, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    completed = read(root / 'COMPLETE.json')
    assert completed['status'] == 'FULL312_A_C_COMPARISON_COMPLETE' and completed['optimizer_updates'] == 0
    summary = read(root / 'analysis/results.json')
    assert summary['status'] == 'FULL312_CLASSIFICATION_COMPARISON_COMPLETE'
    assert summary['identity']['source_checkpoint_sha256'] == SOURCE_SHA
    patch = TRAIN / 'classification_patch_update0300.pt'
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == summary['identity']['classification_patch_sha256']
    a = read(root / 'reference/metrics.json')
    c = read(root / 'object_support/metrics.json')
    av, cv = table_values(a), table_values(c)
    assert av[:5] == cv[:5]
    assert len(av) == len(cv) == 13 and all(v[8] is None and v[12] is None for v in (av, cv))
    assert all(math.isfinite(x) for v in (av, cv) for x in v if x is not None)
    assert a['windows'] == c['windows'] == 1860 and a['unique_scenes'] == c['unique_scenes'] == 312
    assert a['official_commit'] == c['official_commit']
    manifest = read(Path('/space/mawb/SIU3R/data/scannet/val_pair.json'))
    expected = {r['scan'] + '_context' + '_'.join(map(str, r['context_ids'])) for r in manifest}
    golden = {r['name']: r for rank in range(4) for r in read(OFFICIAL / f'rank{rank:02d}/export_complete.json')['records']}
    prior = {r['name']: r for rank in range(4) for r in read(OLD_C / f'rank{rank:02d}/export_complete.json')['records']}
    records, parts = [], []
    for rank in range(4):
        path = root / 'object_support' / f'rank{rank:02d}'
        exported = read(path / 'export_complete.json')
        assert exported['status'] == 'EXPORTED' and exported['identity'] == summary['identity']
        assert exported['frozen_RGB_depth_GT_reproduced'] and not exported['smoke']
        assert exported['reused_windows'] == 32 and exported['newly_inferred_windows'] == 433
        assert len(exported['records']) == 465
        scored = read(path / 'score_complete.json')
        assert scored['status'] == 'SCORED' and scored['windows'] == 465 and scored['optimizer_updates'] == 0
        assert scored['reused_native_scored_windows'] == 32 and scored['newly_scored_windows'] == 433
        assert scored['official_commit'] == c['official_commit']
        assert read(path / 'official_merge_contract.json')['status'] == 'PASS'
        shard_parts = torch.load(path / 'official_states.pt', map_location='cpu', mmap=True, weights_only=False)['parts']
        assert [len(p['names']) for p in shard_parts] == [433, 32]
        old_part = torch.load(OLD_C / f'rank{rank:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all']
        assert equal_tree(shard_parts[1], old_part)
        parts += shard_parts
        records += exported['records']
        for row in exported['records']:
            assert row['frame_ids'] == golden[row['name']]['frame_ids']
            if row['reused_existing128']:
                assert row['input_rgb_sha256'] == prior[row['name']]['input_rgb_sha256']
                assert row['diagnostics'] == prior[row['name']]['diagnostics']
    assert len(records) == len({r['name'] for r in records}) == 1860
    assert {r['name'] for r in records} == expected
    assert sorted(r['source_index'] for r in records) == list(range(1860))
    assert len({r['name'].split('_context')[0] for r in records}) == 312
    assert sum(r['reused_existing128'] for r in records) == 128
    assert {n for part in parts for n in part['names']} == expected
    totals = {k: sum(r['diagnostics'][k] for r in records) for k in
        ('gt', 'good_matched_mask', 'good_matched_but_no_object', 'matched_no_object', 'matched_class_correct')}
    assert totals == read(root / 'object_support/matched_query_diagnostics.json')['totals']
    mapping = ast.literal_eval(ast.parse(Path('/space/mawb/SIU3R/src/utils/scannet_constant.py').read_text()).body[0].value)
    names = {label: mapping[label + 1] for label in range(20)}
    counts, scenes = Counter(), defaultdict(set)
    for r in records:
        for label in r['diagnostics']['gt_internal_semantic_classes']:
            counts[label] += 1
            scenes[label].add(r['name'].split('_context')[0])
    concentration, per_class = {}, {}
    for view in ('context', 'target'):
        d = summary['per_class_deltas'][view]
        ap = dict(zip(d['AP_classes'], d['AP_delta_pp']))
        per_class[view] = [dict(label=label, name=names[label], IoU_delta_pp=d['IoU_delta_pp'][label],
            PQ_delta_pp=d['PQ_delta_pp'][label], AP_delta_pp=ap.get(label),
            input_GT_window_instances=counts[label], input_GT_scenes=len(scenes[label])) for label in range(20)]
        concentration[view] = dict(
            mean_IoU_delta_pp=np.mean(d['IoU_delta_pp']),
            positive_IoU_classes=sum(x > 0 for x in d['IoU_delta_pp']),
            positive_PQ_classes=sum(x > 0 for x in d['PQ_delta_pp']),
            positive_AP_classes=sum(x > 0 for x in d['AP_delta_pp']),
            curtain_refrigerator_shower_contribution_pp=sum(d['IoU_delta_pp'][i] for i in (13, 14, 15)) / 20,
            mean_IoU_delta_pp_excluding_curtain_refrigerator_shower=np.mean([x for i, x in enumerate(d['IoU_delta_pp']) if i not in (13, 14, 15)]),
            mean_AP_delta_pp=np.mean(d['AP_delta_pp']),
            shower_AP_contribution_pp=ap[15] / 18,
            mean_AP_delta_pp_excluding_shower=np.mean([x for label, x in ap.items() if label != 15]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_scenes = {r['name'].split('_context')[0] for r in read(HELDOUT)['records']}
    assert len(old_scenes) == 128
    chosen = {n for n in expected if n.split('_context')[0] not in old_scenes}
    assert len({n.split('_context')[0] for n in chosen}) == 184
    a_parts = [torch.load(OFFICIAL / f'rank{rank:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all'] for rank in range(4)]
    confirmation = dict(unique_scenes=184, windows=len(chosen), selection='All official windows in scenes outside original128 screening scenes', results={})
    for view in ('context', 'target'):
        aa, ga = ap_subset(a_parts, chosen, view, args.output.parent)
        cc, gc = ap_subset(parts, chosen, view, args.output.parent)
        assert all(equal_tree(ga[k], gc[k]) for k in ga if k.startswith('groundtruth_'))
        confirmation['results'][view] = dict(A=aa, C=cc, delta_map_pp=100 * (cc['map'] - aa['map']))
    result = dict(status='FULL312_CLASSIFICATION_RESULTS_REVIEWED', checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        identity=summary['identity'], all_export_score_reuse_and_manifest_checks_passed=True,
        windows=1860, unique_scenes=312, images=11160, zero_training_updates=True, model_forward_calls=0,
        native13_A=av, native13_C=cv, per_class=per_class, concentration=concentration,
        confirmation184_AP_only=confirmation, C_full_native_matching=totals,
        caveat='AP confirmation is read-only native recomputation from saved states. It is not a full13-column subset score or a second training seed. Full A native-matcher diagnostics do not exist. GT oracle scores are not model performance.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], concentration=concentration,
        confirmation184=dict(windows=len(chosen), scenes=184, delta_pp={v: x['delta_map_pp'] for v, x in confirmation['results'].items()})), indent=2))


if __name__ == '__main__':
    main()
