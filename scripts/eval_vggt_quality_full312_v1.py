"""Fixed trained Q: canonical C masks, all1860 official windows, no optimization."""
import argparse
import copy
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.vggt_readout_probe_v1 import (
    ScalarReadout, frozen_C, workers, core_versions, assert_core, verify_probe_segmentation)
from scripts.eval_vggt_classification_full312_v1 import write_table, verify_pair, read
from scripts.eval_vggt_classification_readback_v1 import check_official
from scripts.eval_vggt_joint_short_controls_v1 import write
from scripts.eval_vggt_feedback_expanded_v1 import windows, PAIR
from scripts.train_vggt_classification_readback_v1 import SOURCE_SHA
from scripts.train_vggt_joint_short_controls_v1 import digest

PRIOR = Path('/space/mawb/ssst/group_plus/vggt_readout_probe_v1/attempts/attempt03')
FULL = Path('/space/mawb/ssst/group_plus/vggt_classification_readback_full312_v1/attempts/attempt02/object_support')
CHECKPOINT = PRIOR / 'training/head_checkpoint_update0300.pt'
Q_SHA = '61dd9a8fbd8915c1c8779e61c6e0a3997b2b3d15a2a9639e297b1c0946ffcee5'
PATCH_SHA = 'ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9'


@lru_cache(None)
def identity():
    done = read(PRIOR / 'analysis/completion_independent_audit.json')
    assert done['status'] == 'INDEPENDENT_FORMAL300_COMPLETION_AUDIT_PASSED'
    assert done['updates_per_arm'] == 300 and done['exposures_per_arm'] == 2400
    assert done['checkpoint_sha256'] == Q_SHA == read(CHECKPOINT.with_suffix('.json'))['sha256']
    assert digest(CHECKPOINT) == Q_SHA
    assert done['identity']['source_checkpoint_sha256'] == SOURCE_SHA
    assert done['identity']['C_classification_patch_sha256'] == PATCH_SHA
    prior_snapshot = PRIOR.parents[1] / 'snapshots' / done['identity']['execution_sha']
    numerical_files = ('scripts/vggt_readout_probe_v1.py', 'scripts/export_object_locus_v3_set_official.py',
        'scripts/eval_object_locus_frozen_vggt_posefree_v1.py', 'tokengs/models/object_locus_readout_heads_v1.py',
        'tokengs/models/object_locus_frozen_vggt_posefree.py', 'tokengs/models/frozen_vggt_posefree.py',
        'tokengs/models/object_locus_panoptic_v1.py', 'tokengs/models/object_locus_panoptic_v1_controller.py',
        'tokengs/models/object_locus_mask_support_readback_v1.py')
    assert all(digest(REPO / f) == digest(prior_snapshot / f) for f in numerical_files)
    full = read(FULL / 'COMPLETE.json')
    assert full['status'] == 'CLASSIFICATION_FULL312_EVALUATED' and full['windows'] == 1860
    assert full['identity']['source_checkpoint_sha256'] == SOURCE_SHA
    assert full['identity']['classification_patch_sha256'] == PATCH_SHA
    return dict(source_checkpoint_sha256=SOURCE_SHA, C_patch_sha256=PATCH_SHA,
        Q_checkpoint_sha256=Q_SHA, Q_training_execution_sha=done['identity']['execution_sha'],
        evaluation_execution_sha=os.environ['TASK_CODE_SHA'], numerical_files_unchanged=True,
        adapter=.01, feedback=.05, additional_training_updates=0)


def catalogs():
    choices, _ = windows()
    full = {}
    prior = {}
    for rank in range(4):
        for row in read(FULL / f'rank{rank:02d}/export_complete.json')['records']:
            assert row['name'] not in full
            full[row['name']] = (row, FULL / f'rank{rank:02d}/all' / row['name'])
    for rank in range(8):
        for row in read(PRIOR / 'evaluation' / f'export_complete_rank{rank:02d}.json')['records']:
            assert row['name'] not in prior
            prior[row['name']] = (row, PRIOR / f'evaluation/quality/rank{rank:02d}/all' / row['name'], rank)
    indexed = {w['scene'] + '_context' + '_'.join(map(str, w['context'])): w for w in choices}
    assert len(full) == len(indexed) == 1860 and len(prior) == 128
    assert len({w['scene'] for w in choices}) == 312 and set(full) == set(indexed)
    assert set(prior) <= set(full)
    for name, (row, _, _) in prior.items():
        assert row['source_index'] == full[name][0]['source_index'] == indexed[name]['source_index']
        assert row['frame_ids'] == full[name][0]['frame_ids']
    return choices, full, prior, indexed


def quality_rows(rows, multipliers):
    """Only replace confidence, using original C's canonical score and query id."""
    multipliers = np.asarray(multipliers, dtype=np.float32)
    assert multipliers.shape == (100,) and np.isfinite(multipliers).all()
    assert ((multipliers >= 0) & (multipliers <= 1)).all()
    result = []
    for row in rows:
        assert 1 <= row['id'] <= 100 and np.isfinite(row['score'])
        result.append(dict(row, score=row['score'] * float(multipliers[row['id'] - 1])))
    return result


def link_pair(destination, canonical, multipliers=None, prior_quality=None):
    """Reuse immutable images/GT; keep every JSON owned by this evaluation."""
    destination.mkdir(parents=True, exist_ok=False)
    for sub in ('rgb', 'depth', 'rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt'):
        assert (canonical / sub).is_dir()
        (destination / sub).symlink_to((canonical / sub).resolve(), target_is_directory=True)
    for view in ('context', 'target'):
        sub = view + '_seg_pred'
        (destination / sub).mkdir()
        for file in (canonical / sub).glob('*.png'):
            (destination / sub / file.name).symlink_to(file.resolve())
        original = read(canonical / sub / 'pred.json')
        if prior_quality is None:
            rows = quality_rows(original, multipliers)
        else:
            rows = read(prior_quality / sub / 'pred.json')
        assert [{k: v for k, v in x.items() if k != 'score'} for x in original] == [
            {k: v for k, v in x.items() if k != 'score'} for x in rows]
        assert all(np.isfinite(x['score']) for x in rows)
        write(destination / sub / 'pred.json', rows)
    for filename in ('render_scores.json', 'depth_scores.json'):
        shutil.copy2(canonical / filename, destination / filename)


def predict(model, opt, head, window, device, *, witness=None):
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.object_locus_frozen_vggt_posefree_runtime import seed_everything
    seed_everything(42 + window['source_index'])
    batch = build_batch(opt, window, device)
    generated = model.generate(batch['images_input'])
    feature = model._readout_feature
    assert feature.shape == (1, 100, 256) and torch.isfinite(feature).all()
    assert torch.allclose(model.panoptic.class_head(feature), generated['states'][-1]['thing_logits19'], atol=1e-6, rtol=1e-6)
    multiplier = head(feature).sigmoid()[0].cpu().numpy()
    metadata = dict(source_index=window['source_index'], scene=window['scene'],
        frame_ids=batch['frame_ids'][0].cpu().tolist(),
        input_rgb_sha256=hashlib.sha256(batch['images_all'].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest())
    if witness is not None:
        # Only witnesses repeat calibration/rendering. Q needs context-generated
        # features, and rendering must not regenerate or change that feature.
        from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _official_export_view
        from scripts.export_object_locus_v3_set_official import write_official_pair
        from scripts.evaluate_ssst_validation import save_rgb, save_depth
        canonical, prior_quality, scratch = witness
        calibration = model.calibrate_targets(batch['images_all'], generated)
        rendered = model.render_generated_at(generated, torch.linalg.inv(calibration['c2w']).transpose(-1, -2), calibration['intrinsics'])
        assert torch.equal(model._readout_feature, feature)
        view = _official_export_view(rendered)
        write_official_pair(view, batch, window, scratch)
        name = window['scene'] + '_context' + '_'.join(map(str, window['context']))
        pair = scratch / name
        for sub in ('rgb', 'depth', 'rgb_gt', 'depth_gt'):
            (pair / sub).mkdir()
        for i, frame in enumerate(metadata['frame_ids']):
            filename = f'{window["scene"]}_{frame}.png'
            save_rgb(pair / 'rgb' / filename, rendered['images_pred'][0, i])
            save_rgb(pair / 'rgb_gt' / filename, batch['images_all'][0, i])
            save_depth(pair / 'depth' / filename, rendered['depths_pred'][0, i] / .15)
            save_depth(pair / 'depth_gt' / filename, batch['depth_gt_m_all'][0, i])
        verify_pair(pair, canonical, ('rgb', 'depth', 'rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt'))
        verify_probe_segmentation(pair, canonical)
        if prior_quality is not None:
            for scope in ('context', 'target'):
                actual = quality_rows(read(canonical / f'{scope}_seg_pred/pred.json'), multiplier)
                expected = read(prior_quality / f'{scope}_seg_pred/pred.json')
                assert len(actual) == len(expected)
                assert all({k: v for k, v in x.items() if k != 'score'} == {k: v for k, v in y.items() if k != 'score'}
                    and abs(x['score'] - y['score']) <= 1e-7 for x, y in zip(actual, expected))
        metadata['witness_RGB_depth_masks_and_Q_scores_passed'] = True
    return multiplier, metadata


def export(args):
    rank, world, device = workers()
    assert world == 8
    proof = identity()
    choices, full, prior, indexed = catalogs()
    root = args.root / f'rank{rank:02d}'
    root.mkdir(parents=True, exist_ok=False)
    model, opt = frozen_C(device)
    blob = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    assert blob['completed_updates'] == 300 and blob['core_frozen']
    head = ScalarReadout().to(device)
    head.load_state_dict(blob['states']['quality'], strict=True)
    head.eval().requires_grad_(False)
    del blob
    versions = core_versions(model)
    head_versions = {n: p._version for n, p in head.named_parameters()}
    known = [(row, path) for row, path, previous_rank in prior.values() if previous_rank == rank]
    unseen = [w for w in choices if w['scene'] + '_context' + '_'.join(map(str, w['context'])) not in prior]
    assert len(known) == 16 and len(unseen) == 1732
    witnesses = []
    with torch.no_grad(), tempfile.TemporaryDirectory(dir=root, prefix='witness-') as tmp:
        known_row, known_path = known[0]
        for i, (w, previous) in enumerate(((indexed[known_row['name']], known_path), (unseen[rank], None))):
            name = w['scene'] + '_context' + '_'.join(map(str, w['context']))
            _, metadata = predict(model, opt, head, w, device, witness=(full[name][1], previous, Path(tmp) / str(i)))
            assert metadata['frame_ids'] == full[name][0]['frame_ids']
            assert metadata['input_rgb_sha256'] == full[name][0]['input_rgb_sha256']
            assert_core(model, versions)
            witnesses.append(dict(name=name, **metadata))
    assert all(not p.requires_grad and p.grad is None and p._version == head_versions[n] for n, p in head.named_parameters())
    write(args.root / f'startup_witness_rank{rank:02d}.json', dict(status='FIXED_Q_FULL312_STARTUP_WITNESSES_PASSED',
        identity=proof, witnesses=witnesses, core_and_Q_frozen=True, optimizer_updates=0))
    dist.barrier()
    if rank == 0:
        assert all(read(args.root / f'startup_witness_rank{i:02d}.json')['status'] == 'FIXED_Q_FULL312_STARTUP_WITNESSES_PASSED' for i in range(world))
        write(args.root / 'STARTUP_VERIFIED.json', dict(status='FIXED_Q_FULL312_REAL8GPU_STARTUP_VERIFIED', identity=proof,
            workers=8, witnesses=16, prior_Q_scores_reproduced=True, C_RGB_depth_masks_reproduced=True,
            context_only_feature_equals_prior_full_render_path=True, optimizer_updates=0))
    records = []
    for previous, path in known:
        link_pair(root / 'all' / previous['name'], full[previous['name']][1], prior_quality=path)
        records.append(dict(previous, reused_existing_Q128=True))
    started = time.monotonic()
    selected = unseen[rank::world]
    for i, w in enumerate(selected):
        name = w['scene'] + '_context' + '_'.join(map(str, w['context']))
        with torch.no_grad():
            multiplier, metadata = predict(model, opt, head, w, device)
        assert metadata['frame_ids'] == full[name][0]['frame_ids']
        assert metadata['input_rgb_sha256'] == full[name][0]['input_rgb_sha256']
        link_pair(root / 'all' / name, full[name][1], multiplier)
        assert_core(model, versions)
        assert all(not p.requires_grad and p.grad is None and p._version == head_versions[n] for n, p in head.named_parameters())
        records.append(dict(name=name, **metadata, reused_existing_Q128=False))
        elapsed = time.monotonic() - started
        write(root / 'export_progress.json', dict(completed_new=i + 1, total_new=len(selected), reused=16,
            elapsed_seconds=elapsed, remaining_seconds=elapsed / (i + 1) * (len(selected) - i - 1), latest=name))
    write(root / 'export_complete.json', dict(status='EXPORTED', records=records, identity=proof,
        reused_Q128_windows=16, new_windows=len(selected), C_masks_RGB_depth_GT_reused=True,
        source_pair_sha256=digest(PAIR), core_and_Q_frozen=True, optimizer_updates=0))
    dist.destroy_process_group()


def score(args):
    from scripts.score_object_locus_posefree_official import create_worker, cpu_tree
    from scripts.posefree_official_state_helpers import update, states, merge
    from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_COMMIT
    rank, world, device = workers()
    check_official()
    root = args.root / f'rank{rank:02d}'
    exported = read(root / 'export_complete.json')
    assert exported['identity'] == identity() and exported['optimizer_updates'] == 0
    fresh = sorted(row['name'] for row in exported['records'] if not row['reused_existing_Q128'])
    reused = sorted(row['name'] for row in exported['records'] if row['reused_existing_Q128'])
    with tempfile.TemporaryDirectory(dir=root, prefix='native-contract-') as tmp:
        tmp = Path(tmp)
        (tmp / fresh[0]).symlink_to((root / 'all' / fresh[0]).resolve(), target_is_directory=True)
        expected = evaluate(tmp, device='cpu', image_quality=False, depth_quality=False)
        worker = create_worker(root / 'all', str(device))
        for view in ('context', 'target'):
            update(worker, worker.process_segmentation(root / 'all' / fresh[0] / f'{view}_seg_pred', root / 'all' / fresh[0] / f'{view}_seg_gt'), view)
        actual = merge([dict(names=fresh[:1], states=cpu_tree(states(worker)))], root)
        from scripts.eval_vggt_classification_full312_v1 import assert_segmentation_equal
        assert_segmentation_equal(actual, expected, 1e-7)
    write(root / 'official_merge_contract.json', dict(status='PASS', official_commit=SIU3R_COMMIT))
    worker = create_worker(root / 'all', str(device))
    started = time.monotonic()
    for i, name in enumerate(fresh):
        for view in ('context', 'target'):
            update(worker, worker.process_segmentation(root / 'all' / name / f'{view}_seg_pred', root / 'all' / name / f'{view}_seg_gt'), view)
        write(root / 'score_progress.json', dict(completed=i + 1, total=len(fresh), elapsed_seconds=time.monotonic() - started))
    prior = torch.load(PRIOR / f'evaluation/quality/rank{rank:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all']
    assert sorted(prior['names']) == reused and len(reused) == 16
    assert read(PRIOR / f'evaluation/quality/rank{rank:02d}/score_complete.json')['official_commit'] == SIU3R_COMMIT
    torch.save(dict(parts=[dict(names=fresh, states=cpu_tree(states(worker))), prior]), root / 'official_states.pt')
    write(root / 'score_complete.json', dict(status='SCORED', native_contract='PASS', windows=len(fresh) + 16,
        reused_Q128_windows=16, new_windows=len(fresh), optimizer_updates=0, official_commit=SIU3R_COMMIT))
    dist.destroy_process_group()


def reduce(args):
    from scripts.posefree_official_state_helpers import merge
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS, table_values
    from scripts.analyze_vggt_feedback_factorial_v1 import markdown
    from scripts.review_vggt_classification_full312_v1 import ap_subset, equal_tree
    check_official()
    proof = identity()
    _, full, prior, _ = catalogs()
    parts, records = [], []
    for rank in range(8):
        root = args.root / f'rank{rank:02d}'
        assert read(root / 'score_complete.json')['native_contract'] == 'PASS'
        assert read(root / 'official_merge_contract.json')['status'] == 'PASS'
        exported = read(root / 'export_complete.json')
        assert exported['identity'] == proof and exported['core_and_Q_frozen']
        records += exported['records']
        parts += torch.load(root / 'official_states.pt', map_location='cpu', mmap=True, weights_only=False)['parts']
    names = [n for p in parts for n in p['names']]
    assert len(names) == len(set(names)) == len(records) == 1860 and set(names) == set(full)
    assert len({r['scene'] for r in records}) == 312
    assert sorted(r['source_index'] for r in records) == list(range(1860))
    assert sum(r['reused_existing_Q128'] for r in records) == 128
    output = args.root / 'analysis'
    output.mkdir(exist_ok=False)
    reference = read(FULL / 'metrics.json')
    segmentation = merge(parts, output)
    write(output / 'fresh_native_segmentation.json', segmentation)
    canonical = reference['official_segmentation']['all']
    for view in ('context', 'target'):
        for field in ('miou', 'pq'):
            assert abs(segmentation[view + '_' + field] - canonical[view + '_' + field]) < 1e-7
    # Same canonical PNGs by construction. Preserve fresh native evidence before
    # canonicalizing unchanged semantic values; Q AP is always recomputed.
    for key, value in canonical.items():
        if '_map' not in key:
            segmentation[key] = value
    report = copy.deepcopy(reference)
    report.update(arm='Q fixed quality', identity=proof, optimizer_updates=0,
        official_commit=SIU3R_COMMIT, official_segmentation={'all': segmentation},
        unchanged_images_and_semantic_metrics_reused=True, prior_Q128_states_reused=True)
    write(output / 'metrics.json', report)
    q_values = write_table(report, output)
    c_values = table_values(reference)
    assert q_values[:5] == c_values[:5]
    assert all(q_values[i] == c_values[i] for i in (5, 7, 9, 11))
    rows = [('R frozen C', c_values), ('Q fixed quality', q_values)]
    (output / 'siu3r_comparison.md').write_text(markdown(rows) + '\n')
    with (output / 'siu3r_comparison.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('experiment',) + FIELDS)
        writer.writerows((label, *values) for label, values in rows)
    c_parts = [p for rank in range(4) for p in torch.load(FULL / f'rank{rank:02d}/official_states.pt', map_location='cpu', mmap=True, weights_only=False)['parts']]
    old_scenes = {row['scene'] for row, _, _ in prior.values()}
    chosen = {n for n in names if n.split('_context')[0] not in old_scenes}
    assert len(old_scenes) == 128 and len({n.split('_context')[0] for n in chosen}) == 184
    confirmation = dict(scenes=184, windows=len(chosen), AP_only_diagnostic={})
    per_class = {}
    for view in ('context', 'target'):
        c_ap, c_gt = ap_subset(c_parts, chosen, view, output)
        q_ap, q_gt = ap_subset(parts, chosen, view, output)
        assert all(equal_tree(c_gt[k], q_gt[k]) for k in c_gt if k.startswith('groundtruth_'))
        confirmation['AP_only_diagnostic'][view] = dict(C=c_ap, Q=q_ap, delta_map_pp=100 * (q_ap['map'] - c_ap['map']))
        prior_ap, _ = ap_subset(parts, set(prior), view, output)
        expected = read(PRIOR / 'evaluation/quality/metrics.json')['official_segmentation']['all'][view + '_map']['map']
        assert abs(prior_ap['map'] - expected) < 1e-12
        assert canonical[view + '_map']['classes'] == segmentation[view + '_map']['classes']
        per_class[view] = dict(classes=canonical[view + '_map']['classes'],
            AP_delta_pp=[100 * (q - c) for c, q in zip(canonical[view + '_map']['map_per_class'], segmentation[view + '_map']['map_per_class'])])
    write(output / 'results.json', dict(status='FIXED_Q_FULL312_COMPARISON_COMPLETE', identity=proof,
        windows=1860, scenes=312, images=11160, optimizer_updates=0, Q_trained_updates=300,
        native13_C=c_values, native13_Q=q_values, confirmation184=confirmation, per_class=per_class,
        prior128_native_AP_reproduced=True, C_masks_RGB_depth_GT_unchanged=True,
        caveat='Fixed Q checkpoint, no further training or score tuning; single seed. Confirmation184 is AP-only diagnosis, not another13-column table.'))
    write(args.root / 'COMPLETE.json', dict(status='FIXED_Q_FULL312_EVALUATED', identity=proof,
        windows=1860, scenes=312, optimizer_updates=0, official_columns=13, text_columns_blank=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('export', 'score', 'reduce'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.mode](args)
