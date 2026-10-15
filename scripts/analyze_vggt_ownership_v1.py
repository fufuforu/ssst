"""Analyze frozen ownership observations; use GT pairing, negative controls and saved outputs."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr, rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.format_object_locus_posefree_siu3r_table import table_values

LAYERS = ('6', '8', '10', '12')
METRICS = ('read_child_js', 'write_child_js', 'contribution_child_js', 'anchor_child_dice',
           'read_child_support', 'write_child_support', 'contribution_child_support',
           'write_other_GT_fraction', 'write_other_GT_fraction_pure', 'write_signed_other_GT', 'read_entropy', 'contribution_norm',
           'write_own_GT_preference_over_null', 'write_other_GT_label_shuffle',
           'write_object_fraction_of_visible_write', 'write_other_GT_fraction_of_visible_write')


def read(p):
    return json.loads(p.read_text())


def mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    return float(spearmanr(x[ok], y[ok]).statistic)


def partial_rank(rows, x, y):
    if len(rows) < 10:
        return None
    xx, yy = rankdata([r[x] for r in rows]), rankdata([r[y] for r in rows])
    cov = np.column_stack((np.ones(len(rows)), rankdata([np.log1p(r['GT_pixels']) for r in rows]),
                           rankdata([np.log1p(r['contribution_norm']) for r in rows]),rankdata([r['GT_count'] for r in rows])))
    rx = xx - cov @ np.linalg.lstsq(cov, xx, rcond=None)[0]
    ry = yy - cov @ np.linalg.lstsq(cov, yy, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 1e-10 and np.std(ry) > 1e-10 else None


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def gt_write_null(record, source):
    """Reassign post-hoc GT ownership while holding every actual write fixed.

    Archived norms are FP16 solely for storage. Require reconstruction of the
    original FP32 scalar within 5e-4 before using this sensitivity control.
    Single-object scenes cannot identify cross-instance selectivity.
    """
    k = len(record['gt_ids'])
    result = {}
    z = np.load(source.with_suffix('.npz'))
    gt = z['parent_GT_mass'].astype(np.float64)
    owner = np.array(record['diagnostic_query_owner'])
    rng = np.random.default_rng(941 + record['source_index'])
    perms = []
    if k >= 2:
        for _ in range(16):
            p = rng.permutation(k)
            while (p == np.arange(k)).any():
                p = rng.permutation(k)
            perms.append(p)
    for layer in LAYERS:
        norm = z[f'L{layer}_contribution'].astype(np.float64)
        written = norm.T @ gt
        denominator = written.sum(1)
        visible = norm.T @ z['parent_mass'].astype(np.float64)
        object_fraction = denominator/np.maximum(visible,1e-12)
        if k:
            own = written[np.arange(100), np.maximum(owner, 0)] / np.maximum(denominator, 1e-12)
            actual = 1-own
            expected = np.array(record['ownership'][layer]['write_other_GT_fraction'])
            valid = (owner >= 0) & (denominator > 1e-10)
            assert np.max(np.abs(actual[valid]-expected[valid]), initial=0) < 5e-4
        if perms:
            null = np.mean([1-written[np.arange(100), p[np.maximum(owner, 0)]] / np.maximum(denominator, 1e-12) for p in perms], 0)
            result[layer] = dict(write_other_GT_label_shuffle=null.tolist(), write_own_GT_preference_over_null=(null-actual).tolist())
        else:
            result[layer] = dict(write_other_GT_label_shuffle=[None]*100, write_own_GT_preference_over_null=[None]*100)
        result[layer]['write_object_fraction_of_visible_write']=object_fraction.tolist()
        result[layer]['write_other_GT_fraction_of_visible_write']=(object_fraction*actual).tolist() if k else [0.]*100
    return result


def case_gallery(root, out, cases, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    paths = dict(C=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation'),
                 baseline300=Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04/baseline/evaluation'))
    pairs = {arm: {r['name']: f.parent / 'all' / r['name'] for f in path.glob('rank*/export_complete.json') for r in read(f)['records']} for arm, path in paths.items()}
    gallery = out / 'cases'; gallery.mkdir()
    for i, case in enumerate(cases):
        n, iid = case['name'], case['GT_id']
        fig, axes = plt.subplots(2, 8, figsize=(24, 8.3))
        for row, arm in enumerate(('C', 'baseline300')):
            r, src = records[arm][n]
            g = r['gt_ids'].index(iid)
            q = r['matched_query'][r['matched_gt'].index(g)]
            z = np.load(src.with_suffix('.npz'))
            pair = pairs[arm][n]
            context = r['frame_ids'][:2]
            gt_paths = [pair / 'context_seg_gt' / f"{n.split('_context')[0]}_gt{frame}.png" for frame in context]
            gts = []
            for path in gt_paths:
                arr = np.asarray(Image.open(path), dtype=np.int64)
                packed = arr[..., 0] + 256 * arr[..., 1] + 65536 * arr[..., 2]
                gts.append(packed % 1000 == iid)
            view = int(np.argmax([x.sum() for x in gts])); gt = gts[view]
            rgb = np.asarray(Image.open(pair / 'rgb_gt' / f"{n.split('_context')[0]}_{context[view]}.png")) / 255
            raw = np.unpackbits(z['raw_context'][view, q], axis=-1).astype(bool)
            winner = np.unpackbits(z['winner_context'][view, q], axis=-1).astype(bool)
            final = z['final_context'][view] == q + 1
            columns = [('GT', None), ('Read (context allowed)', z['L12_read'][:, q]), ('Actual Write contribution', z['L12_contribution'][:, q]),
                       ('Anchor membership', z['L12_anchor_membership'][:, q]), ('Child membership', z['child_membership'][:, q]),
                       ('Raw mask', raw), ('Pixel winner', winner), ('Final mask', final)]
            for col, (title, val) in enumerate(columns):
                ax = axes[row, col]; ax.imshow(rgb); ax.axis('off'); ax.set_title(title, fontsize=10)
                if col == 0:
                    ax.contour(gt, levels=[.5], colors=['yellow'], linewidths=.8)
                    pc = np.array(r['query_class_probabilities'][q]); pred = int(pc.argmax())
                    ax.set_title(f'{arm}: Q{q}\npred={pred}, GT={r["gt_classes"][g]-2}', fontsize=9)
                elif col < 5:
                    keep = z['anchor_xy_valid'][view] & (z['parent_mass'] > 1e-7)
                    idx = np.flatnonzero(keep); idx = idx[np.argsort(val[idx])[-200:]]
                    xy = z['anchor_xy'][view, idx]
                    ax.scatter(xy[:, 0], xy[:, 1], c=val[idx], s=7, cmap='magma', alpha=.85, vmin=0, vmax=max(float(np.max(val[idx])) if len(idx) else 0, 1e-8))
                    ax.contour(gt, levels=[.5], colors=['cyan'], linewidths=.6)
                    ax.set_xlim(0, 256); ax.set_ylim(256, 0)
                else:
                    overlay = np.zeros((256, 256, 4))
                    overlay[val & gt] = [0, 1, 0, .65]
                    overlay[val & ~gt] = [1, 0, 0, .65]
                    overlay[~val & gt] = [0, .5, 1, .65]
                    ax.imshow(overlay)
                    inter = np.count_nonzero(val & gt); union = np.count_nonzero(val | gt)
                    ax.text(.02, .02, f'frame IoU {inter/max(union,1):.3f}', color='white', transform=ax.transAxes, fontsize=10,
                            bbox=dict(facecolor='black', alpha=.6, pad=2))
        fig.suptitle(f'{case["stratum"]}: {n}, GT instance {iid}\nGT-selected diagnostic example; green TP / red FP / blue FN; anchor colors independently scaled', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .9), h_pad=4)
        filename = f'{i+1:02d}_{case["stratum"]}.png'
        fig.savefig(gallery / filename, dpi=130); plt.close(fig)
        case['figure'] = str(gallery / filename)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    records, gt_rows, query_rows, ceilings, regional, windows = {}, [], [], {}, {}, []
    source_paths = dict(C=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation'),
                        baseline300=Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04/baseline/evaluation'))
    window_scores = read(a.root / 'official_per_window_association_scores.json')
    assert window_scores['status'] == 'OFFICIAL_WINDOW_ASSOCIATION_SCORES_COMPLETE'
    official = {}
    for arm in ('C', 'baseline300'):
        done = list((a.root / arm).glob('rank*/COMPLETE.json'))
        assert sum(len(read(f)['names']) for f in done) == 128
        assert all(read(f)['model_parameters_buffers_unchanged'] and read(f)['new_optimizer_updates'] == 0 for f in done)
        rs = {f.stem: (read(f), f) for f in (a.root / arm).glob('rank*/scene*.json')}
        assert len(rs) == 128
        records[arm] = rs
        official[arm] = table_values(read(source_paths[arm] / 'metrics.json'))
        ceiling = defaultdict(Counter)
        regions = {k: dict(pixels=0, sse=0., frame_psnr=[]) for k in ('object_interior', 'boundary', 'background_stuff', 'other')}
        for n, (r, source) in rs.items():
            write_null = gt_write_null(r, source)
            for key, frames in r['regional_rgb'].items():
                for frame in frames:
                    regions[key]['pixels'] += frame['pixels']; regions[key]['sse'] += frame['sse']
                    if frame['psnr'] is not None:
                        regions[key]['frame_psnr'].append(frame['psnr'])
            k = len(r['gt_ids']); mats = {key: np.array(v).reshape(100, k) for key, v in r['mask_IoU'].items()}
            pc = np.array(r['query_class_probabilities']); pred = pc.argmax(1)
            class18 = pc[:, :18].argmax(1) + 2
            raw = mats['raw_all']
            maxraw = raw.max(1) if k else np.zeros(100)
            bestgt = raw.argmax(1) if k else np.zeros(100, int)
            matched = dict(zip(r['matched_query'], r['matched_gt']))
            query_to_gt = {q: g for q, g in matched.items()}
            query_to_gt.update({q: int(bestgt[q]) for q in range(100) if q not in matched and maxraw[q] >= .5})
            for stage, mat in mats.items():
                best = mat.max(0) if k else np.zeros(0)
                same = np.where(class18[:, None] == np.array(r['gt_classes'])[None], mat, 0).max(0) if k else np.zeros(0)
                ceiling[stage].update(dict(gt=k, best50=int((best >= .5).sum()), best75=int((best >= .75).sum()),
                                           same50=int((same >= .5).sum()), same75=int((same >= .75).sum())))
            wownership = []
            for q, g in query_to_gt.items():
                quality = float(raw[q, g]); gt_class = r['gt_classes'][g] - 2
                if q not in matched:
                    group = 'duplicate_raw_good'
                elif quality < .5:
                    group = 'poor_mask'
                elif pred[q] == 18:
                    group = 'good_mask_no_object'
                elif pred[q] != gt_class:
                    group = 'good_mask_wrong_class'
                else:
                    group = 'good_mask_correct_class'
                for layer in LAYERS:
                    o = r['ownership'][layer]
                    row = dict(arm=arm, name=n, layer=int(layer), query=q, GT_id=r['gt_ids'][g], group=group,
                               native_matched=q in matched, GT_count=k, GT_pixels=r['gt_pixels'][g], small=r['gt_pixels'][g] < .01*2*256*256,
                               tiny=r['gt_pixels'][g] < .001*2*256*256, raw_IoU=quality, final_IoU=float(mats['final'][q, g]),
                               predicted_class=int(pred[q]), GT_class=gt_class,
                               write_GT_mass=o['write_GT_mass'][q], write_GT_supported=o['write_GT_mass'][q] > 1e-8,
                               ownership_GT_differs_from_matcher=int(r['diagnostic_query_owner'][q]) != g)
                    for metric in METRICS:
                        row[metric] = o[metric][q] if metric in o else write_null[layer][metric][q]
                        if metric + '_query_shuffle' in o:
                            row[metric + '_query_shuffle'] = o[metric + '_query_shuffle'][q]
                    row['write_child_js_anchor_shuffle'] = o['write_child_js_anchor_shuffle'][q]
                    row['anchor_child_dice_anchor_shuffle'] = o['anchor_child_dice_anchor_shuffle'][q]
                    query_rows.append(row)
                    if layer == '12' and q in matched:
                        wownership.append(row)
            for g, iid in enumerate(r['gt_ids']):
                q = r['matched_query'][r['matched_gt'].index(g)]
                row = dict(arm=arm, name=n, GT_id=iid, GT_class=r['gt_classes'][g], query=q,
                           GT_pixels=r['gt_pixels'][g], predicted_class=int(pred[q]), class_correct=bool(pred[q] == r['gt_classes'][g]-2),
                           no_object=bool(pred[q] == 18), raw_good_query_count=int((raw[:, g] >= .5).sum()),
                           write_GT_mass=r['ownership']['12']['write_GT_mass'][q])
                for stage, mat in mats.items():
                    row[stage + '_matched_IoU'] = float(mat[q, g]); row[stage + '_best_IoU'] = float(mat[:, g].max())
                for metric in METRICS:
                    row[metric] = r['ownership']['12'][metric][q] if metric in r['ownership']['12'] else write_null['12'][metric][q]
                irgb = r['instance_rgb_unquantized'][g]
                row['instance_rgb_pixels'] = sum(irgb['pixels']); row['instance_rgb_sse'] = sum(irgb['sse'])
                gt_rows.append(row)
            score = window_scores['conditions'][arm][n]
            # Only original matched foreground queries contribute to window means.
            windows.append(dict(arm=arm, name=n, **score, **{m: mean([r[m] for r in wownership]) for m in METRICS}))
        ceilings[arm] = dict(ceiling)
        for key, value in regions.items():
            value['mean_frame_PSNR'] = mean(value.pop('frame_psnr'))
            value['pooled_MSE'] = value['sse'] / value['pixels']
        regional[arm] = regions
    assert set(records['C']) == set(records['baseline300'])
    for n in records['C']:
        c, b = records['C'][n][0], records['baseline300'][n][0]
        assert all(c[k] == b[k] for k in ('gt_ids', 'gt_classes', 'gt_pixels', 'input_rgb_sha256', 'frame_ids'))
    summaries, correlations = {}, {}
    for arm in ('C', 'baseline300'):
        summaries[arm] = {}
        for layer in (6, 8, 10, 12):
            selected = [r for r in query_rows if r['arm'] == arm and r['layer'] == layer]
            groups = sorted({r['group'] for r in selected}) + ['all_native_matched', 'small_native_matched', 'tiny_native_matched']
            summaries[arm][str(layer)] = {}
            for group in groups:
                if group == 'all_native_matched':
                    rows = [r for r in selected if r['native_matched']]
                elif group in ('small_native_matched', 'tiny_native_matched'):
                    rows = [r for r in selected if r['native_matched'] and r['small' if group.startswith('small') else 'tiny']]
                else:
                    rows = [r for r in selected if r['group'] == group]
                keys = [k for k in selected[0] if k in METRICS or k.endswith('_query_shuffle') or k.endswith('_anchor_shuffle')]
                summaries[arm][str(layer)][group] = dict(n=len(rows), GT_supported_writes=sum(r['write_GT_supported'] for r in rows),
                    **{k: mean([r[k] for r in rows if not (k.startswith('write_other_GT') or k=='write_own_GT_preference_over_null') or r['write_GT_supported']]) for k in keys})
        selected = [r for r in query_rows if r['arm'] == arm and r['layer'] == 12 and r['native_matched'] and not r['ownership_GT_differs_from_matcher']]
        correlations[arm] = {}
        for x in ('write_other_GT_fraction', 'write_child_js', 'contribution_child_js', 'anchor_child_dice', 'read_child_js', 'write_own_GT_preference_over_null'):
            use = [r for r in selected if r[x] is not None and (not x.startswith('write_other_GT') and x!='write_own_GT_preference_over_null' or r['write_GT_supported'])]
            correlations[arm][x] = {y: dict(spearman=corr([r[x] for r in use], [r[y] for r in use]),
                partial_rank_controlling_GT_area_count_and_contribution=partial_rank(use, x, y)) for y in ('raw_IoU', 'final_IoU')}
        ws = [r for r in windows if r['arm'] == arm and r['write_child_js'] is not None]
        correlations[arm]['window_performance'] = {x: {y: corr([r[x] for r in ws], [r[y] for r in ws]) for y in ('context_map', 'context_pq', 'target_map', 'target_pq')}
            for x in ('write_other_GT_fraction', 'write_child_js', 'anchor_child_dice', 'read_child_js')}
    indexed = {(r['arm'], r['name'], r['GT_id']): r for r in gt_rows}
    drift = []
    for row in gt_rows:
        if row['arm'] != 'C':
            continue
        b = indexed['baseline300', row['name'], row['GT_id']]
        d = dict(name=row['name'], GT_id=row['GT_id'], GT_class=row['GT_class'], GT_pixels=row['GT_pixels'],
                 query_C=row['query'], query_baseline300=b['query'], correct_C=row['class_correct'], correct_baseline300=b['class_correct'],
                 no_object_C=row['no_object'], no_object_baseline300=b['no_object'])
        for key in ['raw_all_matched_IoU', 'eligible_raw_matched_IoU', 'winner_before_area_filter_matched_IoU', 'final_matched_IoU',
                    'inherited_anchor_matched_IoU', 'geometry_child_matched_IoU', 'raw_all_best_IoU', 'final_best_IoU',
                    'write_other_GT_fraction', 'write_child_js', 'anchor_child_dice', 'instance_rgb_sse']:
            d[key + '_C'] = row[key]; d[key + '_baseline300'] = b[key]; d['delta_' + key] = b[key] - row[key]
        drift.append(d)
    drift_summary = {k: mean([r[k] for r in drift]) for k in drift[0] if k.startswith('delta_')}
    drift_summary.update(instances=len(drift), classification_correct_C=sum(r['correct_C'] for r in drift),
                         classification_correct_baseline300=sum(r['correct_baseline300'] for r in drift),
                         no_object_C=sum(r['no_object_C'] for r in drift), no_object_baseline300=sum(r['no_object_baseline300'] for r in drift))
    total_sse_delta = sum(regional['baseline300'][k]['sse'] - regional['C'][k]['sse'] for k in regional['C'])
    region_drift = {k: dict(delta_sse=regional['baseline300'][k]['sse']-regional['C'][k]['sse'],
                           share_of_net_SSE_increase=(regional['baseline300'][k]['sse']-regional['C'][k]['sse'])/total_sse_delta,
                           delta_pooled_MSE=regional['baseline300'][k]['pooled_MSE']-regional['C'][k]['pooled_MSE'],
                           delta_mean_frame_PSNR=regional['baseline300'][k]['mean_frame_PSNR']-regional['C'][k]['mean_frame_PSNR']) for k in regional['C']}
    transitions = {}
    for arm in ('C', 'baseline300'):
        rows = [r for r in gt_rows if r['arm'] == arm]
        transitions[arm] = {}
        stages = ['raw_all', 'eligible_raw', 'winner_before_area_filter', 'final']
        for x, y in zip(stages, stages[1:]):
            transitions[arm][x+'->'+y] = dict(lost_matched_good50=sum(r[x+'_matched_IoU']>=.5 and r[y+'_matched_IoU']<.5 for r in rows),
                gained_matched_good50=sum(r[x+'_matched_IoU']<.5 and r[y+'_matched_IoU']>=.5 for r in rows),
                lost_any_query_good50=sum(r[x+'_best_IoU']>=.5 and r[y+'_best_IoU']<.5 for r in rows),
                gained_any_query_good50=sum(r[x+'_best_IoU']<.5 and r[y+'_best_IoU']>=.5 for r in rows))
    cases, used = [], set()
    def choose(stratum, candidates, score):
        for r in sorted(candidates, key=score, reverse=True):
            if r['name'] not in used:
                cases.append(dict(stratum=stratum, name=r['name'], GT_id=r['GT_id'])); used.add(r['name']); return
    crows = [r for r in gt_rows if r['arm'] == 'C']
    choose('classification_rejection', [r for r in crows if r['raw_all_matched_IoU']>=.5 and r['no_object']], lambda r:r['raw_all_matched_IoU']-r['final_matched_IoU'])
    choose('pixel_assembly_loss', [r for r in crows if r['eligible_raw_matched_IoU']>=.5], lambda r:r['eligible_raw_matched_IoU']-r['final_matched_IoU'])
    choose('visual_rescue_weak_anchor', [r for r in crows if r['raw_all_matched_IoU']>=.5], lambda r:r['raw_all_matched_IoU']-r['inherited_anchor_matched_IoU'])
    choose('cross_instance_write_association', [r for r in crows if r['raw_all_matched_IoU']<.5 and r['GT_pixels']>131 and r['write_GT_mass']>1e-8], lambda r:r['write_other_GT_fraction'])
    choose('joint_training_drift', drift, lambda r:-r['delta_raw_all_matched_IoU'])
    choose('healthy_global_read_control', [r for r in crows if r['raw_all_matched_IoU']>=.75 and r['final_matched_IoU']>=.75], lambda r:r['read_child_js'])
    case_gallery(a.root, a.output, cases, records)
    feedback = {arm:{layer:dict(mean_actual_residual_fraction=mean([r['ownership'][layer]['actual_feedback_to_token_norm'] for r,_ in records[arm].values()]),
                    max_decomposition_error=max(r['ownership'][layer]['feedback_reconstruction_max_error'] for r,_ in records[arm].values())) for layer in LAYERS} for arm in records}
    # Scene-cluster bootstrap: unequal GT counts are preserved within sampled scenes.
    rng = np.random.default_rng(20261015)
    cluster_names = sorted(records['C'])
    samples = rng.integers(0, len(cluster_names), size=(3000, len(cluster_names)))
    cluster_ci = {}
    for metric in ('delta_raw_all_matched_IoU','delta_final_matched_IoU','delta_anchor_child_dice','delta_write_child_js'):
        sums = np.array([sum(r[metric] for r in drift if r['name']==n) for n in cluster_names])
        counts = np.array([sum(r['name']==n for r in drift) for n in cluster_names])
        bootstrap = sums[samples].sum(1)/counts[samples].sum(1)
        cluster_ci[metric] = dict(mean=float(sums.sum()/counts.sum()), ci95=np.quantile(bootstrap,[.025,.975]).tolist())
    report = dict(status='OWNERSHIP_DIAGNOSIS_ANALYZED', models=['C', 'baseline300'], scenes=128, new_optimizer_updates=0,
                  official_13_columns=official, ownership_by_group_layer=summaries, associations=correlations,
                  stage_ceilings=ceilings, stage_transitions=transitions, regional_rgb=regional, regional_drift=region_drift,
                  GT_aligned_drift_summary=drift_summary, drift_scene_cluster_bootstrap=cluster_ci, feedback_magnitude=feedback, cases=cases,
                  limitations=['Read is allowed to gather global context; equality of Read/Write/Child is not a correctness requirement.',
                    'Feedback attribution is additive under the observed gate, not a causal query-removal intervention.',
                    'Ownership/IoU/AP/PQ correlations and a single training trajectory do not locate a unique causal bottleneck.',
                    'Per-window official AP/PQ are used only for associations; full model scores use original globally merged states.',
                    'GT-aligned native matcher queries can change; best-query IoU is supplied as a sensitivity check.',
                    'Case selection uses diagnostic GT strata and is illustrative, not a representative evaluation sample.'])
    (a.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_csv(a.output/'query_ownership.csv', query_rows); write_csv(a.output/'GT_aligned_instances.csv', gt_rows)
    write_csv(a.output/'GT_aligned_drift.csv', drift); write_csv(a.output/'window_associations.csv', windows)
    (a.output/'case_gallery.md').write_text('# GT-selected diagnostic cases\n\n'+'\n\n'.join(f'## {c["stratum"]}\n\n{c["name"]}, GT instance {c["GT_id"]}\n\n![case]({c["figure"]})' for c in cases)+'\n')
    print(json.dumps(dict(official=official, transitions=transitions, region_drift=region_drift, drift=drift_summary), indent=2))


if __name__ == '__main__':
    main()
