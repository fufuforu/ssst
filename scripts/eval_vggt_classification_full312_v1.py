"""Expand fixed A/C classification controls to all official 312 scenes/1860 windows."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.train_vggt_classification_readback_v1 import (
    OFFICIAL, SOURCE_SHA, build_base, configure, apply_patch_state, digest)
from scripts.eval_vggt_classification_readback_v1 import check_official, diagnose, diagnostic_text_scopes
from scripts.eval_vggt_joint_short_controls_v1 import export_window, write
from scripts.eval_vggt_feedback_expanded_v1 import windows

OLD = Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01')
TRAIN = OLD / 'object_support/training'
OLD_C = OLD / 'object_support/evaluation'
PAIR = Path('/space/mawb/SIU3R/data/scannet/val_pair.json')


def read(path):
    return json.loads(path.read_text())


def identity():
    done = read(TRAIN / 'COMPLETE.json')
    assert done['status'] == 'CLASSIFICATION_TRAINED' and done['completed_updates'] == 300
    assert done['new_formal_exposures'] == 2400 and done['frozen_core_unchanged']
    assert done['source_checkpoint_sha256'] == SOURCE_SHA
    patch = TRAIN / 'classification_patch_update0300.pt'
    proof = read(patch.with_suffix('.json'))
    assert proof['sha256'] == digest(patch) and proof['source_checkpoint_sha256'] == SOURCE_SHA
    reuse = read(OLD / 'numerical_preflight_reuse.json')
    assert done['execution_sha'] == reuse['execution_sha']
    for relative, expected in reuse['numerical_files_sha256'].items():
        assert digest(REPO / relative) == expected, relative
    # Existing trained-model export/matching math must also stay unchanged.
    old_snapshot = OLD.parents[1] / 'snapshots' / done['execution_sha']
    relative = 'scripts/eval_vggt_classification_readback_v1.py'
    assert digest(REPO / relative) == digest(old_snapshot / relative)
    assert read(OLD_C / 'COMPLETE.json')['status'] == 'CLASSIFICATION_HELDOUT128_EVALUATED'
    return dict(source_checkpoint_sha256=SOURCE_SHA, classification_patch_sha256=proof['sha256'],
        training_execution_sha=done['execution_sha'], evaluation_execution_sha=os.environ['TASK_CODE_SHA'],
        numerical_training_and_model_files_unchanged=True, optimizer_updates=0, additional_training_updates=0)


def catalogs():
    choices, _ = windows()
    full = {}
    old = {}
    for rank in range(4):
        for row in read(OFFICIAL / f'rank{rank:02d}/export_complete.json')['records']:
            full[row['name']] = (row, OFFICIAL / f'rank{rank:02d}/all' / row['name'])
        for row in read(OLD_C / f'rank{rank:02d}/export_complete.json')['records']:
            assert row['name'] not in old
            old[row['name']] = (row, OLD_C / f'rank{rank:02d}/all' / row['name'], rank)
    assert len(choices) == len(full) == 1860 and len(old) == 128
    indexed = {w['scene'] + '_context' + '_'.join(map(str, w['context'])): w for w in choices}
    assert set(indexed) == set(full) and set(old) <= set(full)
    for name, (row, _, _) in old.items():
        assert row['source_index'] == indexed[name]['source_index']
        assert row['frame_ids'] == full[name][0]['frame_ids']
    return choices, full, old, indexed


def worker():
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    assert '3090' in torch.cuda.get_device_name(local)
    return rank, torch.device('cuda', local)


def verify_pair(pair, reference, subs):
    for sub in subs:
        files = sorted(p.name for p in (reference / sub).iterdir() if p.is_file())
        assert files and files == sorted(p.name for p in (pair / sub).iterdir() if p.is_file())
        assert all(digest(reference / sub / name) == digest(pair / sub / name) for name in files), (pair.name, sub)


def export(args):
    rank, device = worker()
    proof = identity()
    choices, full, old, indexed = catalogs()
    root = args.root / f'rank{rank:02d}'
    root.mkdir(parents=True, exist_ok=False)
    model, opt, _, _, initial = build_base(device)
    configure(model, 'object_support', initial)
    blob = torch.load(TRAIN / 'classification_patch_update0300.pt', map_location='cpu', mmap=True, weights_only=False)
    apply_patch_state(model, blob, 'object_support')
    del blob, initial
    model.eval().requires_grad_(False)
    model.understanding_step = 33376 + 2400
    versions = {name: p._version for name, p in model.named_parameters()}
    unseen = [w for w in choices if w['scene'] + '_context' + '_'.join(map(str, w['context'])) not in old]
    assert len(unseen) == 1732
    known = [row for row, _, r in old.values() if r == rank]
    assert len(known) == 32
    selected = [indexed[known[0]['name']], unseen[rank]] if args.smoke else unseen[rank::4]
    records = []
    started = time.monotonic()
    if not args.smoke:
        for previous in known:
            dest = root / 'all' / previous['name']
            dest.parent.mkdir(exist_ok=True)
            shutil.copytree(old[previous['name']][1], dest)
            records.append(dict(previous, reused_existing128=True))
    for w in selected:
        with torch.no_grad():
            row = export_window(model, opt, w, root, device, observer=diagnose)
        reference_row, reference = full[row['name']]
        assert row['frame_ids'] == reference_row['frame_ids']
        pair = root / 'all' / row['name']
        verify_pair(pair, reference, ('rgb', 'depth', 'rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt'))
        if row['name'] in old:
            previous, previous_path, _ = old[row['name']]
            assert row['input_rgb_sha256'] == previous['input_rgb_sha256']
            assert row['diagnostics'] == previous['diagnostics']
            verify_pair(pair, previous_path, ('context_seg_pred', 'target_seg_pred'))
        for filename in ('render_scores.json', 'depth_scores.json'):
            shutil.copy2(reference / filename, pair / filename)
        records.append(dict(row, reused_existing128=False))
        write(root / 'export_progress.json', dict(completed=len(records), total=2 if args.smoke else 465,
            newly_inferred=sum(not r['reused_existing128'] for r in records), reused=0 if args.smoke else 32,
            elapsed_seconds=time.monotonic() - started, latest=row['name']))
    assert all(p._version == versions[name] for name, p in model.named_parameters())
    assert len(records) == (2 if args.smoke else 465)
    write(root / 'export_complete.json', dict(status='EXPORTED', records=records, identity=proof,
        smoke=args.smoke, frozen_RGB_depth_GT_reproduced=True, source_pair_sha256=digest(PAIR),
        reused_windows=0 if args.smoke else 32, newly_inferred_windows=2 if args.smoke else 433))


def score(args):
    from scripts.posefree_official_state_helpers import update, states, merge
    from scripts.score_object_locus_posefree_official import create_worker, cpu_tree
    from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_COMMIT
    rank, device = worker()
    check_official()
    root = args.root / f'rank{rank:02d}'
    exported = read(root / 'export_complete.json')
    assert exported['smoke'] == args.smoke and exported['identity'] == identity()
    fresh = sorted(r['name'] for r in exported['records'] if not r['reused_existing128'])
    reused = sorted(r['name'] for r in exported['records'] if r['reused_existing128'])
    chosen = sorted(r['name'] for r in exported['records'])[:2]
    with tempfile.TemporaryDirectory(dir=root, prefix='native-contract-') as tmp:
        tmp = Path(tmp)
        for name in chosen:
            (tmp / name).symlink_to((root / 'all' / name).resolve(), target_is_directory=True)
        expected = evaluate(tmp, device='cpu', image_quality=False, depth_quality=False)
        parts = []
        for name in chosen:
            metric = create_worker(root / 'all', str(device))
            for view in ('context', 'target'):
                update(metric, metric.process_segmentation(root / 'all' / name / f'{view}_seg_pred', root / 'all' / name / f'{view}_seg_gt'), view)
            parts.append(dict(names=[name], states=cpu_tree(states(metric))))
        actual = merge(parts[::-1], root)
        assert_segmentation_equal(actual, expected, 1e-7)
    write(root / 'official_merge_contract.json', dict(status='PASS', official_commit=SIU3R_COMMIT))
    metric = create_worker(root / 'all', str(device))
    started = time.monotonic()
    for i, name in enumerate(fresh):
        for view in ('context', 'target'):
            update(metric, metric.process_segmentation(root / 'all' / name / f'{view}_seg_pred', root / 'all' / name / f'{view}_seg_gt'), view)
        write(root / 'score_progress.json', dict(completed=i + 1, total=len(fresh), elapsed_seconds=time.monotonic() - started))
    parts = [dict(names=fresh, states=cpu_tree(states(metric)))]
    if reused:
        prior = torch.load(OLD_C / f'rank{rank:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all']
        assert sorted(prior['names']) == reused and len(reused) == 32
        assert read(OLD_C / f'rank{rank:02d}/score_complete.json')['official_commit'] == SIU3R_COMMIT
        parts.append(prior)
    torch.save(dict(parts=parts), root / 'official_states.pt')
    write(root / 'score_complete.json', dict(status='SCORED', windows=len(fresh) + len(reused),
        reused_native_scored_windows=len(reused), newly_scored_windows=len(fresh), optimizer_updates=0,
        official_commit=SIU3R_COMMIT))


def assert_segmentation_equal(actual, expected, tolerance):
    for view in ('context', 'target'):
        for field in ('miou', 'pq'):
            assert abs(actual[view + '_' + field] - expected[view + '_' + field]) < tolerance
        for field in ('map', 'map_50', 'map_75'):
            assert abs(actual[view + '_map'][field] - expected[view + '_map'][field]) < tolerance


def write_table(report, root):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS, table_values
    from scripts.analyze_vggt_feedback_factorial_v1 import markdown
    values = table_values(diagnostic_text_scopes(report))
    assert len(values) == 13 and values[8] is None and values[12] is None
    assert all(x is None or np.isfinite(x) for x in values)
    with (root / 'siu3r_table1.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        writer.writerow(values)
    (root / 'siu3r_table1.md').write_text(markdown([(report['arm'], values)]) + '\n')
    write(root / 'siu3r_table1.json', dict(columns=FIELDS, values=values, windows=report['windows'], unique_scenes=report['unique_scenes']))
    return values


def reference(args):
    from scripts.posefree_official_state_helpers import merge
    check_official()
    proof = identity()
    _, full, _, _ = catalogs()
    args.root.mkdir(parents=True, exist_ok=False)
    report = read(OFFICIAL / 'metrics.json')
    assert report['windows'] == 1860 and report['unique_scenes'] == 312
    assert report['checkpoint_sha256'] == SOURCE_SHA
    parts = [torch.load(OFFICIAL / f'rank{i:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all'] for i in range(4)]
    names = [name for part in parts for name in part['names']]
    assert len(names) == len(set(names)) == 1860 and set(names) == set(full)
    result = merge(parts, args.root)
    assert_segmentation_equal(result, report['official_segmentation']['all'], 1e-12)
    # Also validate the exact native old128 state reuse before expansion.
    prior = [torch.load(OLD_C / f'rank{i:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all'] for i in range(4)]
    assert_segmentation_equal(merge(prior, args.root), read(OLD_C / 'metrics.json')['official_segmentation']['all'], 1e-12)
    report.update(arm='A frozen reference', identity=proof, source_outputs_reused=True)
    write(args.root / 'metrics.json', report)
    write_table(report, args.root)
    write(args.root / 'COMPLETE.json', dict(status='FULL312_REFERENCE_REUSED', windows=1860, scenes=312,
        native_reference_metrics_reproduced=True, native_prior128_C_merge_reproduced=True, model_forward_calls=0, optimizer_updates=0))


def reduce(args):
    from scripts.posefree_official_state_helpers import merge
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    check_official()
    parts, records, diagnostics, observations = [], [], [], []
    for rank in range(4):
        root = args.root / f'rank{rank:02d}'
        assert read(root / 'score_complete.json')['status'] == 'SCORED'
        assert read(root / 'official_merge_contract.json')['status'] == 'PASS'
        exported = read(root / 'export_complete.json')
        assert exported['identity'] == identity() and exported['smoke'] == args.smoke
        records += exported['records']
        parts += torch.load(root / 'official_states.pt', map_location='cpu', mmap=True, weights_only=False)['parts']
        for row in exported['records']:
            diagnostics.append(row['diagnostics'])
            rgb = read(root / 'all' / row['name'] / 'render_scores.json')
            depth = read(root / 'all' / row['name'] / 'depth_scores.json')
            assert len(rgb) == len(depth) == 6
            by_item = {r['item']: r for r in depth}
            observations.extend([dict(absrel=by_item[r['item']]['absrel'], rmse=by_item[r['item']]['rmse'],
                psnr=r['psnr'], ssim=r['ssim'], lpips=r['lpips']) for r in rgb])
    names = [n for part in parts for n in part['names']]
    assert len(names) == len(set(names)) == len(records) == (8 if args.smoke else 1860)
    if not args.smoke:
        assert set(names) == set(catalogs()[1])
        assert len({n.split('_context')[0] for n in names}) == 312
        assert sorted(r['source_index'] for r in records) == list(range(1860))
        assert sum(r['reused_existing128'] for r in records) == 128
    report = dict(status='EVALUATED', arm='C object_support', windows=len(names),
        unique_scenes=len({n.split('_context')[0] for n in names}), images=len(observations), official_commit=SIU3R_COMMIT,
        source_checkpoint_sha256=SOURCE_SHA, identity=identity(), optimizer_updates=0,
        scopes={'target-all': {k: float(np.mean([row[k] for row in observations])) for k in observations[0]}},
        official_segmentation={'all': merge(parts, args.root)}, original_core_and_masks_frozen=True,
        frozen_RGB_depth_GT_reproduced=True, reused128_windows=0 if args.smoke else 128,
        newly_inferred_windows=8 if args.smoke else 1732)
    write(args.root / 'metrics.json', diagnostic_text_scopes(report))
    write_table(report, args.root)
    totals = {k: sum(row[k] for row in diagnostics) for k in
        ('gt', 'good_matched_mask', 'good_matched_but_no_object', 'matched_no_object', 'matched_class_correct')}
    counts, scenes = Counter(), defaultdict(set)
    for row in records:
        for label in row['diagnostics']['gt_internal_semantic_classes']:
            counts[label] += 1
            scenes[label].add(row['name'].split('_context')[0])
    write(args.root / 'matched_query_diagnostics.json', dict(totals=totals,
        good_mask_no_object_rate=totals['good_matched_but_no_object'] / totals['good_matched_mask'],
        context_GT_window_instances_by_internal_class=dict(counts), context_GT_scenes_by_internal_class={k: len(v) for k, v in scenes.items()},
        caveat='GT counts repeat instances across windows. Full A native-matcher diagnostics do not exist; compare A/C matched queries only on original128.'))
    if args.smoke:
        write(args.root / 'PREFLIGHT_VERIFIED.json', dict(status='FULL312_EVALUATION_PREFLIGHT_PASSED',
            identity=identity(), windows=8, four_workers=4, original_C_outputs_reproduced=True,
            new_windows_frozen_RGB_depth_GT_reproduced=True, native_scoring_contracts_passed=True, optimizer_updates=0))
    else:
        a_values = read(args.root.parent / 'reference/siu3r_table1.json')['values']
        c_values = read(args.root / 'siu3r_table1.json')['values']
        assert all(abs(x - y) < 1e-12 for x, y in zip(a_values[:5], c_values[:5]))
        write(args.root / 'COMPLETE.json', dict(status='CLASSIFICATION_FULL312_EVALUATED', windows=1860, scenes=312,
            original_core_and_masks_frozen=True, native_states_reused128=True, identity=identity(), optimizer_updates=0))


def comparison(args):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS, table_values
    from scripts.analyze_vggt_feedback_factorial_v1 import markdown
    from scripts.probe_vggt_saved_mask_ranking_v1 import ranking_result
    assert read(args.root / 'reference/COMPLETE.json')['status'] == 'FULL312_REFERENCE_REUSED'
    assert read(args.root / 'object_support/COMPLETE.json')['status'] == 'CLASSIFICATION_FULL312_EVALUATED'
    out = args.root / 'analysis'
    out.mkdir(exist_ok=False)
    reports = {arm: read(args.root / sub / 'metrics.json') for arm, sub in [('A', 'reference'), ('C', 'object_support')]}
    rows = [(report['arm'], table_values(report)) for report in reports.values()]
    with (out / 'siu3r_comparison.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('experiment',) + FIELDS)
        for label, values in rows:
            writer.writerow((label,) + tuple(values))
    (out / 'siu3r_comparison.md').write_text(markdown(rows) + '\n')
    a_parts = [torch.load(OFFICIAL / f'rank{i:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all'] for i in range(4)]
    c_parts = [part for i in range(4) for part in torch.load(args.root / 'object_support' / f'rank{i:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['parts']]
    ranking, per_class = {}, {}
    for arm, parts in [('A', a_parts), ('C', c_parts)]:
        names = [n for part in parts for n in part['names']]
        order = sorted(range(len(names)), key=lambda i: names[i])
        assert len(names) == len(set(names)) == 1860
        ranking[arm] = {}
        for view in ('context', 'target'):
            key = view + '_mAP'
            keys = parts[0]['states'][key]
            merged = {k: [v for part in parts for v in part['states'][key][k]] for k in keys}
            assert all(len(v) in (0, 1860) for v in merged.values())
            merged = {k: [v[i] for i in order] if len(v) == 1860 else v for k, v in merged.items()}
            ranking[arm][view] = ranking_result(merged, out)
            assert abs(ranking[arm][view]['baseline']['map'] - reports[arm]['official_segmentation']['all'][view + '_map']['map']) < 1e-12
    for view in ('context', 'target'):
        a = reports['A']['official_segmentation']['all']
        c = reports['C']['official_segmentation']['all']
        assert a[view + '_map']['classes'] == c[view + '_map']['classes']
        per_class[view] = dict(
            IoU_delta_pp=[100 * (y - x) for x, y in zip(a[view + '_ious_per_class'], c[view + '_ious_per_class'])],
            PQ_delta_pp=[100 * (y - x) for x, y in zip(a[view + '_pqs_per_class'][0], c[view + '_pqs_per_class'][0])],
            AP_classes=a[view + '_map']['classes'],
            AP_delta_pp=[100 * (y - x) for x, y in zip(a[view + '_map']['map_per_class'], c[view + '_map']['map_per_class'])])
    delta = [None if x is None else y - x for x, y in zip(rows[0][1], rows[1][1])]
    write(out / 'results.json', dict(status='FULL312_CLASSIFICATION_COMPARISON_COMPLETE', identity=identity(),
        windows=1860, unique_scenes=312, images=11160, delta_C_minus_A=delta,
        per_class_deltas=per_class, ranking_diagnosis_only=ranking, reconstruction_unchanged=True,
        prior128_matched_query_comparison=read(OLD / 'analysis/results.json'),
        caveat='Single trained C patch; expanded validation, not a new training seed. GT ranking is diagnosis only. No full312 A native-matching diagnosis.'))
    write(args.root / 'COMPLETE.json', dict(status='FULL312_A_C_COMPARISON_COMPLETE', optimizer_updates=0,
        windows=1860, unique_scenes=312, official_columns=13, text_columns_blank=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('export', 'score', 'reduce', 'reference', 'comparison'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.mode](args)
