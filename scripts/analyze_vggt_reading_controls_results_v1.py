"""CPU-only audit and paired analysis of completed reading controls; no inference."""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_vggt_instance_bottlenecks_v1 import rles, mask_matrix, best_masks, score_matches

ARMS = ('baseline', 'object_local_read', 'anchor_visual', 'reliable_depth')
RECON = ('AbsRel', 'RMSE', 'PSNR', 'SSIM', 'LPIPS')
COMMIT = '8ea80166be76854f938e90521f1a5b688b755c87'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite_tree(x):
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    if isinstance(x, float):
        return np.isfinite(x)
    return True


def native_signature(values, labels):
    return [(tuple(size), counts) for size, counts in values], labels.tolist()


def checkpoint_audit(path, stream, manifest):
    receipt = read(path.with_suffix('.json'))
    assert digest(path) == receipt['sha256']
    b = torch.load(path, map_location='cpu', weights_only=False)
    assert b['resumable'] and b['world_size'] == 8
    assert b['completed_updates'] == 300 and b['completed_exposures'] == 2400
    assert b['config'] == manifest['config']
    assert set(b['parameter_patch']) == set(manifest['rank_initialization'][0]['names'])
    assert sum(v.numel() for v in b['parameter_patch'].values()) == 233188331
    assert len(b['rank_rng']) == len(b['rank_window_counts']) == 8
    for rank, counts in enumerate(b['rank_window_counts']):
        expected = np.zeros(8337, dtype=int)
        expected[stream[rank::8]] = 1
        assert np.array_equal(counts, expected)
    tensors = list(b['parameter_patch'].values()) + list(b['buffers'].values())
    steps = Counter()
    for state in b['optimizer']['state'].values():
        tensors += [v for v in state.values() if torch.is_tensor(v)]
        steps[int(state['step'])] += 1
    assert all(torch.isfinite(v).all().item() for v in tensors)
    result = dict(sha256=receipt['sha256'], parameters_optimizer_buffers_finite=True,
                  rank_sampling_counts_exact=True, rank_rng_keys=[sorted(r) for r in b['rank_rng']],
                  optimizer_step_histogram=dict(steps), parameter_count=233188331)
    del b, tensors
    gc.collect()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--checkpoints', action='store_true')
    a = ap.parse_args()
    assert not a.output.exists(), 'Preserve earlier analyses; choose a new output directory.'
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[1]
    cohort_path = repo / 'docs/vggt_understanding_probe_cohort_v1.json'
    stream = read(cohort_path)['train_source_indices'][:2400]
    assert len(stream) == len(set(stream)) == 2400
    heldout = read(Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json'))
    choices = {r['name']: r for r in heldout['records']}
    assert len(choices) == 128
    pool = read(Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json'))['windows']
    # Scene identity is encoded in the fixed cohort names, not a nonexistent scene field.
    train_scenes = {pool[i]['scene'] for i in stream}
    assert not train_scenes & {n.split('_context')[0] for n in choices}
    assert subprocess.check_output(['git', '-C', '/space/mawb/SIU3R', 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', '/space/mawb/SIU3R', 'diff', '--name-only', 'HEAD', '--', 'src'], text=True).strip()
    summary = read(a.root / 'analysis/results.json')
    assert read(a.root / 'COMPLETE.json')['status'] == 'FOUR_READING_CONTROLS_EVALUATED'
    assert summary['identical_GT_and_training_stream'] and not summary['Q_applied']
    initial = None
    audit, observations, matched, retained, recon, raw, candidates, semantic = {}, {}, {}, {}, {}, {}, {}, {}
    reference_gt, reference_records, reference_scores = {}, {}, {}
    per_window = []
    gate = read(a.root / 'ALL_STARTUPS_VERIFIED.json')
    for arm in ARMS:
        root = a.root / arm
        manifest = read(root / 'training/run_manifest.json')
        resume = read(root / 'training/resume_manifest.json')
        cfg = manifest['config']
        assert cfg['stream_sha256'] == digest(cohort_path)
        assert cfg['geometry_quality_policy'] == 'monitor_v1' and cfg['Q'].startswith('absent')
        assert cfg == resume['config']
        ids = manifest['rank_initialization']
        assert len(ids) == 8 and len({x['parameter_sha256'] for x in ids}) == 1
        identity = {k: v for k, v in ids[0].items() if k != 'rank'}
        if initial is not None:
            assert initial == identity
        initial = identity
        assert resume['start_update'] == 20
        assert all(x['parameter_sha256'] == gate['rows'][arm]['rank_rows'][i]['parameter_sha256']
                   for i, x in enumerate(resume['rank_initialization']))
        done = read(root / 'training/COMPLETE.json')
        assert done['completed_updates'] == done['formal_updates'] == 300 and done['completed_exposures'] == 2400
        assert done['config'] == cfg
        warnings, rows_by_rank, records, values = Counter(), [], {}, {}
        peak = 0
        for rank in range(8):
            logs = [json.loads(x) for x in (root / f'training/training_rank{rank:02d}.jsonl').read_text().splitlines()]
            assert len(logs) == 300 and finite_tree(logs)
            for i, row in enumerate(logs):
                assert row['completed_updates'] == row['formal_updates'] == i + 1
                assert row['completed_exposures'] == (i + 1) * 8
                assert row['source_indices'] == [stream[i * 8 + rank]]
                assert row['frozen_encoders_unchanged'] and row['understanding_weight'] == 1
                assert row['actual_feedback_peak'] == .05
                assert row['calibration']['geometry_quality_policy'] == 'monitor_v1'
                warnings[row['calibration']['quality_status']] += 1
            peak = max(peak, max(r['peak_allocated'] for r in logs))
            rows_by_rank.append(logs)
            e = root / f'evaluation/rank{rank:02d}'
            exported = read(e / 'export_complete.json')
            assert exported['status'] == 'EXPORTED' and exported['optimizer_updates'] == 0
            assert exported['config'] == cfg and len(exported['records']) == 16
            assert exported['checkpoint_sha256'] == read(root / 'training/checkpoint_latest.json')['sha256']
            score = read(e / 'score_complete.json')
            contract = read(e / 'official_merge_contract.json')
            assert score['status'] == 'SCORED' and score['windows'] == 16 and score['official_commit'] == COMMIT
            assert contract == dict(status='PASS', official_commit=COMMIT)
            for r in exported['records']:
                n = r['name']
                assert n not in records
                records[n] = r
                assert all(r[k] == choices[n][k] for k in ('frame_ids', 'source_index', 'input_rgb_sha256'))
                assert r['official_pair']['target_frames'] == r['frame_ids'] and len(r['frame_ids']) == 6
                assert len(r['official_pair']['context_frames']) == 2 and r['official_pair']['target_set'] == 'all'
                rid = [r['frame_ids'], r['input_rgb_sha256'], r['diagnostics']['gt_instance_ids'], r['diagnostics']['gt_internal_semantic_classes']]
                if arm == 'baseline':
                    reference_records[n] = rid
                else:
                    assert reference_records[n] == rid
                pair = e / 'all' / n
                rgb, depth = read(pair / 'render_scores.json'), read(pair / 'depth_scores.json')
                assert len(rgb) == len(depth) == 6 and {x['item'] for x in rgb} == {x['item'] for x in depth}
                if arm == 'baseline':
                    reference_scores[n] = sorted(x['item'] for x in rgb)
                else:
                    assert reference_scores[n] == sorted(x['item'] for x in rgb)
                values[n] = [np.mean([x[k] for x in depth]) for k in ('absrel', 'rmse')] + [np.mean([x[k] for x in rgb]) for k in ('psnr', 'ssim', 'lpips')]
            blob = torch.load(e / 'official_states.pt', map_location='cpu', weights_only=False)
            assert sorted(blob['names']) == sorted(r['name'] for r in exported['records'])
            for view in ('context', 'target'):
                state = blob['states'][view + '_mAP']
                for j, n in enumerate(blob['names']):
                    gtlabels = state['groundtruth_labels'][j].numpy()
                    sig = native_signature(state['groundtruth_mask'][j], state['groundtruth_labels'][j])
                    if arm == 'baseline':
                        reference_gt[n, view] = sig
                    else:
                        assert reference_gt[n, view] == sig
                    gt = rles(state['groundtruth_mask'][j]); pred = rles(state['detection_mask'][j])
                    labels, scores = state['detection_labels'][j].numpy(), state['detection_scores'][j].numpy()
                    assert np.isfinite(scores).all()
                    iou, _, _, _ = mask_matrix(pred, gt)
                    best, same = best_masks(iou, labels, gtlabels)
                    assigned = score_matches(iou, labels, gtlabels, scores)
                    row = dict(arm=arm, view=view, name=n, gt=len(gt), pred=len(pred),
                               agnostic50=int((best >= .5).sum()), same50=int((same >= .5).sum()),
                               same75=int((same >= .75).sum()), tp50=int((assigned >= 0).sum()),
                               best_iou_sum=float(best.sum()))
                    per_window.append(row)
            del blob
        # Rank synchronization of the globally reduced pre-clipping norm at every update.
        assert all(len({logs[i]['preclip_norm'] for logs in rows_by_rank}) == 1 for i in range(300))
        assert set(records) == set(choices) and len(values) == 128
        names = sorted(records)
        recon[arm] = np.array([values[n] for n in names])
        assert np.isfinite(recon[arm]).all()
        assert np.allclose(recon[arm].mean(0), summary['conditions'][arm][:5], rtol=0, atol=1e-12)
        assert len(summary['conditions'][arm]) == 13 and all(summary['conditions'][arm][i] is None for i in (8, 12))
        assert all(np.isfinite(x) for x in summary['conditions'][arm] if x is not None)
        raw[arm] = {n: records[n]['diagnostics']['raw_best_iou_per_GT'] for n in names}
        matched[arm] = read(root / 'evaluation/matched_query_diagnostics.json')['totals']
        assert matched[arm] == {k: sum(records[n]['diagnostics'][k] for n in names) for k in matched[arm]}
        matched[arm]['good_mask_no_object_rate'] = matched[arm]['good_matched_but_no_object'] / matched[arm]['good_matched_mask']
        observations[arm] = {}
        for layer in ('L6', 'L8', 'L10', 'L12'):
            observations[arm][layer] = dict(
                V_head_norm_mean=np.mean([records[n]['reading_observation'][layer + '_V_head_norm'] for n in names], axis=0).tolist(),
                output_norm_mean=float(np.mean([records[n]['reading_observation'][layer]['output_norm_mean'] for n in names])),
                normalized_entropy_mean=float(np.mean([records[n]['reading_observation'][layer]['per_head_entropy_mean'] for n in names])))
        for key in ('local_support_bias_abs_mean', 'visual_valid_fraction', 'depth_bias_abs_mean', 'reliability_mean'):
            v = [r['reading_observation'][key] for r in records.values() if key in r['reading_observation']]
            if v:
                observations[arm][key] = dict(mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)))
        retained[arm] = {}
        for view in ('context', 'target'):
            chosen = [r for r in per_window if r['arm'] == arm and r['view'] == view]
            total = {k: sum(r[k] for r in chosen) for k in ('gt', 'pred', 'agnostic50', 'same50', 'same75', 'tp50', 'best_iou_sum')}
            total['precision50'] = total['tp50'] / total['pred']
            total['recall50'] = total['tp50'] / total['gt']
            retained[arm][view] = total
        metrics = read(root / 'evaluation/metrics.json')
        assert not metrics['Q_applied'] and metrics['official_commit'] == COMMIT
        semantic[arm] = {v: metrics['official_segmentation']['all'][v + '_ious_per_class'] for v in ('context', 'target')}
        audit[arm] = dict(updates=300, exposures=2400, ranks=8, continuous_unique_sampling=True,
                          all_logged_numbers_finite=True, every_update_frozen_parameters_and_buffers_checked=True,
                          every_update_global_gradient_norm_synchronized=True, startup_hash_matches_resume=True,
                          geometry_quality_counts=dict(warnings), peak_allocated_GiB=peak / 2**30,
                          exported_windows=128, official_merge_contracts=8, execution_sha=done['execution_sha'])
        if a.checkpoints:
            audit[arm]['checkpoint'] = checkpoint_audit(root / 'training/checkpoint_latest.pt', stream, manifest)
        print('AUDITED', arm, '300 updates; 128 windows; 8 official merge contracts', flush=True)
    # Scene-paired bootstrap, same 128 scene samples for each metric/arm. No AP averaging.
    rng = np.random.default_rng(20261015)
    samples = rng.integers(0, 128, size=(5000, 128))
    paired = {}
    for arm in ARMS[1:]:
        delta = recon[arm] - recon['baseline']
        ci = np.quantile(delta[samples].mean(1), [.025, .975], axis=0)
        paired[arm] = {k: dict(delta=float(delta[:, i].mean()), ci95=ci[:, i].tolist(),
                               improved_scenes=int(((delta[:, i] > 0) if k in ('PSNR', 'SSIM') else (delta[:, i] < 0)).sum()))
                       for i, k in enumerate(RECON)}
    raw_summary = {}
    for arm in ARMS:
        v = np.array([x for n in sorted(raw[arm]) for x in raw[arm][n]])
        raw_summary[arm] = dict(gt=len(v), mean_best_iou=float(v.mean()), best50=int((v >= .5).sum()), best75=int((v >= .75).sum()))
    report = dict(status='COMPLETED_READING_CONTROLS_AUDITED', root=str(a.root), audit=audit,
                  train_unique_windows=2400, train_scenes=len(train_scenes), heldout_unique_scenes=128,
                  official_summary_and_GT_file_hash_verification_receipt_sha256=digest(a.root / 'analysis/results.json'),
                  official_13_columns=summary, paired_reconstruction=paired,
                  bootstrap=dict(seed=20261015, samples=5000, unit='one window per heldout scene',
                                 limitation='Conditional on these four trained endpoints; no multi-seed uncertainty or multiplicity correction.'),
                  matched_context_diagnostics=matched, raw_class_agnostic_context=raw_summary,
                  retained_official_candidates=retained, real_CA_observations=observations, per_class_semantic_iou=semantic,
                  limitations=['GT image file hashes reuse the successful automatic summary; native AP GT RLEs independently compared here.',
                               'Final per-rank parameter hashes were not recorded; startup hashes, resume hashes, update counts and all-update global norm synchrony are verified.',
                               'Retained-mask recall/precision are diagnostic and do not replace official global mAP or PQ.',
                               'Matched query no-object conflicts are not unique missed GT instances; duplicate candidates may survive.',
                               'No new optimizer updates, inference calls, GPU jobs or changes to official evaluator.'])
    a.output.mkdir(parents=True)
    (a.output / 'audit_and_analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (a.output / 'retained_window_diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_window[0])); writer.writeheader(); writer.writerows(per_window)
    print(json.dumps(dict(paired=paired, matched=matched, raw=raw_summary, retained=retained, observations=observations), indent=2))


if __name__ == '__main__':
    main()
