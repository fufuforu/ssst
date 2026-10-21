"""CPU-only independent coverage audit and clustered sensitivity summaries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import csv
from collections import defaultdict

from scripts.analyze_vggt_ownership_v1 import partial_rank, corr


def read(path):
    return json.loads(path.read_text())


def read_csv(path):
    rows = []
    for row in csv.DictReader(path.open()):
        for key, value in row.items():
            if value in ('True', 'False'):
                row[key] = value == 'True'
            elif value == '':
                row[key] = None
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        rows.append(row)
    return rows


def clustered_mean(frame, column, names, samples):
    selected = [r for r in frame if r[column] is not None and np.isfinite(r[column])]
    if not selected:
        return dict(n=0, mean=None, ci95=None)
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['name']].append(row[column])
    sums = np.array([sum(grouped[n]) for n in names])
    counts = np.array([len(grouped[n]) for n in names])
    den = counts[samples].sum(1)
    bs = sums[samples].sum(1) / np.maximum(den, 1)
    return dict(n=len(selected), mean=float(sums.sum()/counts.sum()), ci95=np.quantile(bs[den>0], [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    out = args.analysis / 'completion_audit_and_sensitivity.json'
    assert not out.exists()
    result = read(args.analysis / 'results.json')
    assert result['status'] == 'OWNERSHIP_DIAGNOSIS_ANALYZED'
    cohort = read(Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json'))
    cohort = {r['name']: r for r in cohort['records']}
    records, coverage, region_rows = {}, {}, []
    for arm in ('C', 'baseline300'):
        paths = sorted((args.root / arm).glob('rank*/scene*.json'))
        rs = {p.stem: read(p) for p in paths}
        assert set(rs) == set(cohort) and len(paths) == 128
        receipts = [read(p) for p in (args.root / arm).glob('rank*/COMPLETE.json')]
        receipt_names = [n for r in receipts for n in r['names']]
        assert len(receipts) == 3 and len(receipt_names) == len(set(receipt_names)) == 128
        assert set(receipt_names) == set(rs)
        assert all(r['new_optimizer_updates'] == 0 and r['model_parameters_buffers_unchanged'] for r in receipts)
        for p in paths:
            r = rs[p.stem]
            assert p.with_suffix('.npz').stat().st_size > 0
            assert all(r[k] == cohort[p.stem][k] for k in ('source_index', 'frame_ids', 'input_rgb_sha256'))
            assert r['GT_used_only_after_native_prediction']
            assert r['reproduction']['segmentation_and_GT_PNGs_identical']
            assert r['reproduction']['rgb_max_uint8_delta'] <= 1
            for layer in ('6', '8', '10', '12'):
                assert r['ownership'][layer]['feedback_reconstruction_max_error'] < 1e-6
            for frame in range(6):
                assert sum(r['regional_rgb'][key][frame]['pixels'] for key in r['regional_rgb']) == 256 * 256
            for key, frames in r['regional_rgb'].items():
                region_rows.append(dict(arm=arm, name=p.stem, region=key,
                    sse=sum(f['sse'] for f in frames), pixels=sum(f['pixels'] for f in frames),
                    frame_PSNR=np.mean([f['psnr'] for f in frames if f['psnr'] is not None])))
        coverage[arm] = dict(windows=len(rs), GT_instances=sum(len(r['gt_ids']) for r in rs.values()),
            receipts=len(receipts), new_optimizer_updates=0,
            reproduction_attempt_indices=dict(Counter(r['reproduction'].get('attempt_index', 0) for r in rs.values())))
        records[arm] = rs
    for name in records['C']:
        assert all(records['C'][name][key] == records['baseline300'][name][key]
                   for key in ('gt_ids', 'gt_classes', 'gt_pixels', 'input_rgb_sha256', 'frame_ids'))
    names = sorted(cohort)
    rng = np.random.default_rng(19412026)
    samples = rng.integers(0, 128, size=(3000, 128))
    queries = read_csv(args.analysis / 'query_ownership.csv')
    gt = read_csv(args.analysis / 'GT_aligned_instances.csv')
    controls, groups, associations = {}, {}, {}
    for arm in ('C', 'baseline300'):
        controls[arm], groups[arm], associations[arm] = {}, {}, {}
        for layer in (6, 8, 10, 12):
            rows = [r.copy() for r in queries if r['arm']==arm and r['layer']==layer and r['native_matched']]
            controls[arm][str(layer)] = {}
            for key in ('read_child_js', 'write_child_js', 'contribution_child_js', 'anchor_child_dice'):
                for row in rows:
                    row['gain'] = row[key + '_query_shuffle'] - row[key] if key.endswith('_js') else row[key] - row[key + '_query_shuffle']
                controls[arm][str(layer)][key + '_gain_over_query_shuffle'] = clustered_mean(rows, 'gain', names, samples)
            valid = [r for r in rows if r['write_GT_supported'] and not r['ownership_GT_differs_from_matcher'] and r['GT_count']>=2]
            controls[arm][str(layer)]['own_GT_preference_over_label_shuffle'] = clustered_mean(valid, 'write_own_GT_preference_over_null', names, samples)
            if layer == 12:
                for group in sorted({r['group'] for r in rows}):
                    frame = [r for r in rows if r['group']==group]
                    v = [r for r in frame if r['write_GT_supported'] and not r['ownership_GT_differs_from_matcher'] and r['GT_count']>=2]
                    groups[arm][group] = dict(n=len(frame), raw_IoU=float(np.mean([r['raw_IoU'] for r in frame])), final_IoU=float(np.mean([r['final_IoU'] for r in frame])),
                        GT_selectivity=clustered_mean(v, 'write_own_GT_preference_over_null', names, samples))
                valid = [r for r in rows if not r['ownership_GT_differs_from_matcher']]
                for x in ('write_child_js', 'anchor_child_dice', 'write_own_GT_preference_over_null'):
                    v = [r for r in valid if r[x] is not None]
                    if x == 'write_own_GT_preference_over_null':
                        v = [r for r in v if r['write_GT_supported'] and r['GT_count']>=2]
                    associations[arm][x] = {}
                    by_name = {n: [r for r in v if r['name']==n] for n in names}
                    for y in ('raw_IoU', 'final_IoU'):
                        boot = []
                        for indices in samples[:800]:
                            rs = [r for idx in indices for r in by_name.get(names[idx], [])]
                            estimate = partial_rank(rs, x, y)
                            if estimate is not None:
                                boot.append(estimate)
                        associations[arm][x][y] = dict(n=len(v), partial_rank=partial_rank(v, x, y),
                            ci95=np.quantile(boot, [.025, .975]).tolist(), bootstrap_scene_resamples=len(boot))
    stages = {}
    for arm in ('C', 'baseline300'):
        rows = [r for r in gt if r['arm']==arm]
        stages[arm] = dict(duplicate_GT_instances=sum(r['raw_good_query_count']>=2 for r in rows),
            total_raw_good_queries=int(sum(r['raw_good_query_count'] for r in rows)), groups={})
        for label, subset in [('all', rows), ('small', [r for r in rows if r['GT_pixels']<.01*2*256*256]),
                              ('tiny', [r for r in rows if r['GT_pixels']<.001*2*256*256])]:
            stages[arm]['groups'][label] = dict(n=len(subset), **{stage: dict(
                matched50=sum(r[stage+'_matched_IoU']>=.5 for r in subset),
                best50=sum(r[stage+'_best_IoU']>=.5 for r in subset),
                mean_matched_IoU=float(np.mean([r[stage+'_matched_IoU'] for r in subset])))
                for stage in ('inherited_anchor', 'geometry_child', 'raw_all', 'eligible_raw', 'winner_before_area_filter', 'final')})
    regional = {}
    for key in sorted({r['region'] for r in region_rows}):
        c = {r['name']:r for r in region_rows if r['arm']=='C' and r['region']==key}
        b = {r['name']:r for r in region_rows if r['arm']=='baseline300' and r['region']==key}
        assert all(c[n]['pixels']==b[n]['pixels'] for n in names)
        paired = [dict(name=n, delta_sse=b[n]['sse']-c[n]['sse'], delta_frame_PSNR=b[n]['frame_PSNR']-c[n]['frame_PSNR']) for n in names]
        regional[key] = dict(mean_window_SSE_delta=clustered_mean(paired, 'delta_sse', names, samples),
            mean_window_regional_PSNR_delta=clustered_mean(paired, 'delta_frame_PSNR', names, samples),
            pixel_fraction=float(sum(c[n]['pixels'] for n in names)/ (128*6*256*256)),
            relative_MSE_increase=float(sum(b[n]['sse'] for n in names)/sum(c[n]['sse'] for n in names)-1))
    drift = read_csv(args.analysis / 'GT_aligned_drift.csv')
    drift_stats = dict(classification_correct_to_wrong=sum(r['correct_C'] and not r['correct_baseline300'] for r in drift),
        classification_wrong_to_correct=sum(not r['correct_C'] and r['correct_baseline300'] for r in drift),
        no_object_recovered=sum(r['no_object_C'] and not r['no_object_baseline300'] for r in drift),
        new_no_object=sum(not r['no_object_C'] and r['no_object_baseline300'] for r in drift))
    for key in ('delta_raw_all_best_IoU', 'delta_final_best_IoU'):
        drift_stats[key] = clustered_mean(drift, key, names, samples)
    for x in ('delta_write_child_js', 'delta_anchor_child_dice', 'delta_write_other_GT_fraction', 'delta_instance_rgb_sse'):
        drift_stats[x] = {y: corr([r[x] for r in drift], [r[y] for r in drift]) for y in ('delta_raw_all_matched_IoU', 'delta_final_matched_IoU')}
    audit = dict(status='COMPLETE_256_WINDOWS_AUDITED', coverage=coverage, paired_GT_identical=True,
        official_13_columns_unchanged=True, controls=controls, group_sensitivity=groups,
        partial_association_scene_bootstrap=associations, stage_groups=stages,
        regional_sensitivity=regional, drift_sensitivity=drift_stats,
        source_results_sha256=hashlib.sha256((args.analysis/'results.json').read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).with_name('analyze_vggt_ownership_v1.py').read_bytes()).hexdigest(),
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Exploratory scene-cluster intervals are not multiple-comparison corrected.',
            'No model intervention, optimizer, or new GPU inference was used by this audit.'])
    out.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(out)


if __name__ == '__main__':
    main()
