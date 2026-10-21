"""Audit persisted factorial runs and official scores; no inference or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_vggt_understanding_probes_v1 import gradient_summary
from analyze_vggt_interaction_controls_v1 import CLASSES, metric_class_names, read, write
from format_object_locus_posefree_siu3r_table import FIELDS, HEADERS, PRECISION, table_values

ARMS = ('joint_full', 'visual_fixed', 'feedback_small', 'visual_fixed_feedback_small')
STEPS = (0, 100, 300, 500)
LABELS = ('A: visual train, beta .1', 'B: visual fixed, beta .1',
          'C: visual train, beta .05', 'D: visual fixed, beta .05 (3090 training)')


def markdown(rows):
    lines = ['| Experiment | ' + ' | '.join(HEADERS) + ' |',
             '| --- | ' + ' | '.join(['---:']*13) + ' |']
    for label, values in rows:
        lines.append('| ' + label + ' | ' + ' | '.join(
            '' if v is None else f'{v:.{p}f}' for v, p in zip(values, PRECISION)) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    reduced = read(a.report_root/'evaluation/analysis/results.json')
    startup = read(a.report_root/'startup_validation.json')
    assert reduced['status'] == 'FEEDBACK_FACTORIAL_EVALUATED'
    assert reduced['heldout_gt_identical'] and reduced['initial_rgb_identical']
    assert reduced['optimizer_updates'] == 0
    assert startup['common_initial_parameters_identical'] and not startup['pending']
    cohort_sha = hashlib.sha256(a.cohort.read_bytes()).hexdigest()
    expected = read(a.cohort)['train_source_indices']
    assert len(expected) == len(set(expected)) == 4000
    assert np.array_equal(expected, np.random.default_rng(42).permutation(8337)[:4000])
    ranks_by_arm, audit, native, scenes = {}, {}, {}, {}
    sampling_reference = records_reference = None
    for arm in ARMS:
        root = (Path(startup['original_baseline_reused']['run_root'])
                if arm == 'joint_full' else a.run_root/arm)
        done = read(root/'COMPLETE.json'); cfg = done['config']
        assert done['status'] == 'INTERACTION_TRAINING_COMPLETED'
        assert (done['completed_updates'], done['completed_exposures']) == (500, 4000)
        assert done['frozen_parameters_unchanged']
        assert done['source_checkpoint_sha256'] == reduced['source_checkpoint_sha256']
        assert cfg['cohort_sha256'] == cohort_sha and cfg['global_batch'] == 8
        assert cfg['adapter_understanding_scale'] == cfg['gs_understanding_scale'] == .01
        assert cfg['vggt_trainable'] is False and cfg['state_layers'] == [6,8,10,12]
        assert cfg['reconstruction_update'] and cfg['feedback_enabled']
        assert cfg['geometry_quality_policy'] == 'monitor_v1'
        peak = .05 if arm in ('feedback_small', 'visual_fixed_feedback_small') else .1
        frozen = arm in ('visual_fixed', 'visual_fixed_feedback_small')
        assert cfg.get('feedback_peak', .1) == peak
        assert cfg.get('frozen_visual_features', False) == frozen
        ranks = [[json.loads(x) for x in (root/f'training_rank{i}.jsonl').read_text().splitlines()]
                 for i in range(4)]
        assert all(len(r) == 500 for r in ranks)
        for rank in ranks:
            assert [x['completed_updates'] for x in rank] == list(range(1,501))
            assert [x['completed_exposures'] for x in rank] == list(range(8,4001,8))
            assert all(x['frozen_parameters_unchanged'] for x in rank)
            assert all(np.isfinite(x[k]) for x in rank for k in
                       ('loss_recon','loss_understanding','preclip_norm','lr_multiplier'))
            assert all(abs(x['actual_feedback_beta']-x['beta']*peak/.1) < 1e-14 for x in rank)
        sampling = [[(x['window_slots'], x['source_indices']) for x in rank] for rank in ranks]
        if sampling_reference is None: sampling_reference = sampling
        assert sampling == sampling_reference
        assert [i for step in range(500) for rank in ranks for i in rank[step]['source_indices']] == expected
        ranks_by_arm[arm] = ranks
        audit[arm] = dict(
            complete=done,
            preclip_over1_fraction=float(np.mean([x['preclip_norm'] > 1 for r in ranks for x in r])),
            max_preclip_norm=max(x['preclip_norm'] for x in ranks[0]),
            last100_max_preclip_norm=max(x['preclip_norm'] for x in ranks[0][400:]),
            last100_mean_clip_multiplier=float(np.mean([
                min(1.,1./(x['preclip_norm']+1e-6)) for x in ranks[0][400:]])),
            losses={k: dict(first100_mean=float(np.mean([x[k] for r in ranks for x in r[:100]])),
                            last100_mean=float(np.mean([x[k] for r in ranks for x in r[400:]])))
                    for k in ('loss_recon','loss_understanding','preclip_norm')},
            gradients_first20=gradient_summary(ranks, set(range(1,21))),
            gradients_late=gradient_summary(ranks, {100,300,500}),
            late_feedback={f'L{i}': {k: float(np.mean([x['interactions'][f'L{i}'][k]
                for r in ranks for x in r[400:]])) for k in
                ('delta_ratio_mean','delta_ratio_max','feedback_route_entropy')} for i in (6,8,10,12)})
        for step in STEPS:
            condition = f'{arm}/{step}'
            provenance = reduced['provenance'][condition]
            ev = Path(provenance['artifact_root'])
            saved = read(root/f'checkpoint_update_{step:04d}.json')
            metrics, exported, scored = [read(ev/name) for name in
                                        ('metrics.json','export_complete.json','score_complete.json')]
            assert saved['status'] == 'CHECKPOINT_VERIFIED' and saved['step'] == step
            assert saved['sha256'] == metrics['checkpoint_sha256'] == exported['checkpoint_sha256'] == provenance['checkpoint_sha256']
            assert done['execution_git_sha'] == provenance['execution_git_sha'] == exported['execution_git_sha']
            assert metrics['optimizer_updates'] == exported['optimizer_updates'] == scored['optimizer_updates'] == 0
            assert metrics['windows'] == metrics['unique_scenes'] == 64
            assert scored['status'] == 'SCORED' and exported['status'] == 'EXPORTED'
            assert metrics['official_commit'] == '8ea80166be76854f938e90521f1a5b688b755c87'
            assert table_values(metrics) == reduced['conditions'][condition]
            records = sorted(exported['records'], key=lambda x:x['name'])
            identifiers = [(x['name'],x['source_index'],x['frame_ids'],x['input_rgb_sha256']) for x in records]
            if records_reference is None: records_reference = identifiers
            assert identifiers == records_reference and len(set(x[0] for x in identifiers)) == 64
            means = []
            for record in records:
                pair = ev/'all'/record['name']
                render, depth = read(pair/'render_scores.json'), read(pair/'depth_scores.json')
                assert len(render) == len(depth) == 6
                assert sorted(x['item'] for x in render) == sorted(x['item'] for x in depth)
                means.append([np.mean([x[k] for x in depth]) for k in ('absrel','rmse')]+
                             [np.mean([x[k] for x in render]) for k in ('psnr','ssim','lpips')])
            scenes[condition] = np.asarray(means)
            assert np.isfinite(scenes[condition]).all()
            assert np.allclose(scenes[condition].mean(0), reduced['conditions'][condition][:5], rtol=0, atol=1e-12)
            native[condition] = read(ev/'all/results.json')
            assert native[condition] == metrics['official_native_result']
    tables = reduced['conditions']
    assert all(tables[f'{arm}/0'] == tables['joint_full/0'] for arm in ARMS)
    spike_index = max(range(500),key=lambda i:ranks_by_arm['joint_full'][0][i]['preclip_norm'])
    spike = {arm:{k:ranks_by_arm[arm][0][spike_index][k] for k in
                   ('completed_updates','preclip_norm','loss_recon','loss_understanding','source_indices')}
             for arm in ARMS}
    comparisons = [(f'{arm}/500','joint_full/500') for arm in ARMS[1:]]
    comparisons += [(f'{arm}/500',f'{arm}/0') for arm in ARMS]
    comparisons += [('visual_fixed_feedback_small/500','feedback_small/500'),
                    ('visual_fixed_feedback_small/500','visual_fixed/500')]
    indices = np.random.default_rng(42).integers(0,64,size=(10000,64))
    deltas, bootstrap, classes = {}, {}, {}
    for condition, reference in comparisons:
        key = f'{condition} minus {reference}'
        deltas[key] = {k:None if x is None else y-x for k,x,y in zip(FIELDS,tables[reference],tables[condition])}
        delta = scenes[condition]-scenes[reference]
        bounds = np.quantile(delta[indices].mean(1),[.025,.975],axis=0)
        bootstrap[key] = {k:dict(mean_delta=float(delta[:,i].mean()),
            scene_bootstrap_95_interval=bounds[:,i].tolist(),positive_scene_count=int((delta[:,i]>0).sum()))
            for i,k in enumerate(FIELDS[:5])}
        classes[key] = {}
        for view in ('context','target'):
            for metric in ('ious','pqs'):
                change = (np.asarray(native[condition][f'{view}_{metric}_per_class']).reshape(-1)-
                          np.asarray(native[reference][f'{view}_{metric}_per_class']).reshape(-1))
                assert len(change) == len(CLASSES) == 20
                classes[key][f'{view}_{metric}'] = dict(zip(metric_class_names(metric),change.tolist()))
    interaction = {k:None if av is None else dv-cv-bv+av for k,av,bv,cv,dv in
        zip(FIELDS,*(tables[f'{arm}/500'] for arm in ARMS))}
    write(a.output/'audit.json', dict(status='SAVED_FACTORIAL_ARTIFACTS_VERIFIED',training=audit,
        baseline_largest_spike_matched_batch=spike,
        all_four_sampling_identical=True,unique_windows=4000,sampled_scenes=1177,
        all16_evaluation_receipt_chains_verified=True,official_score_aggregates_reproduced=True,
        optimizer_updates=0,
        gradient_scope='Local rank/microbatch norms, not global cosines. Weighted U includes warmup/accumulation; shared branches scaled by .01 here. Anchor prefix excludes decoder blocks registered under enc_dec_backbone.',
        limitation='Single seed,64 validation scenes. D trained on3090; A/B/C on4090, all evaluation4090. Interaction and D comparisons confounded by hardware. First/last loss windows differ.'))
    write(a.output/'comparisons.json', dict(deltas=deltas,interaction_D_minus_C_minus_B_plus_A=interaction,
        bootstrap_reconstruction=bootstrap,bootstrap_samples=10000,seed=42,
        bootstrap_scope='Paired scene dataset uncertainty, not training seed uncertainty. No AP/PQ/IoU scene averaging.',
        per_class_changes=classes))
    write(a.output/'scene_scores.json',dict(scene_names=[x[0] for x in records_reference],columns=FIELDS[:5],
        conditions={k:v.tolist() for k,v in scenes.items()}))
    with (a.output/'siu3r_table1.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(('experiment',)+FIELDS)
        writer.writerow(('common_step0',)+tuple(tables['joint_full/0']))
        for arm,label in zip(ARMS,LABELS):writer.writerow((label,)+tuple(tables[f'{arm}/500']))
    (a.output/'siu3r_table1.md').write_text(markdown(
        [('Common step0',tables['joint_full/0'])]+[(label,tables[f'{arm}/500']) for arm,label in zip(ARMS,LABELS)])+'\n')
    fig,axes=plt.subplots(3,4,figsize=(15,10),constrained_layout=True)
    for ax,col in zip(axes.flat,[i for i in range(13) if i not in (8,12)]):
        for arm,label in zip(ARMS,LABELS):
            ax.plot(STEPS,[tables[f'{arm}/{s}'][col] for s in STEPS],'o-',label=label)
        ax.set_title(FIELDS[col]);ax.set_xlabel('New optimizer steps');ax.grid(alpha=.3)
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',fontsize=8)
    fig.suptitle('Full-pool joint factorial /64 held-out scenes /single seed')
    fig.savefig(a.output/'metric_curves.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='VERIFIED',deltas=deltas,bootstrap=bootstrap),indent=2))


if __name__ == '__main__':main()
