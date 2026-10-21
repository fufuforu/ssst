"""Review completed controls without model execution or new optimizer updates."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    summary = read(root / 'analysis/results.json')
    assert summary['status'] == 'CLASSIFICATION_CONTROLS_AUDITED'
    assert read(root / 'COMPLETE.json')['training_updates'] == 900
    cohort = read(Path(__file__).resolve().parents[1] / 'docs/vggt_understanding_probe_cohort_v1.json')
    stream = cohort['train_source_indices'][:2400]
    assert len(stream) == len(set(stream)) == 2400
    reference = read(root / 'analysis/reference_canonical_metrics.json')
    official_sha = reference['official_commit']
    baseline = reference['official_segmentation']['all']
    tree = ast.parse(Path('/space/mawb/SIU3R/src/utils/scannet_constant.py').read_text())
    names = ast.literal_eval(tree.body[0].value)
    # Official output labels are zero based; the name dictionary is one based.
    label_names = {i: names[i + 1] for i in range(20)}
    arms = {}
    records_by_arm = {}
    initial_hashes = set()
    for arm in ('head_only', 'object_support', 'scene_support'):
        train = root / arm / 'training'
        evaluation = root / arm / 'evaluation'
        done = read(train / 'COMPLETE.json')
        assert done['status'] == 'CLASSIFICATION_TRAINED'
        assert done['completed_updates'] == done['formal_updates'] == 300
        assert done['new_formal_exposures'] == 2400 and done['frozen_core_unchanged']
        assert done['source_checkpoint_sha256'] == summary['source_checkpoint_sha256']
        manifest = read(train / 'run_manifest.json')
        initial_hashes.update(x['source_classifier_sha256'] for x in manifest['rank_initialization'])
        patch = train / 'classification_patch_update0300.pt'
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == read(patch.with_suffix('.json'))['sha256']
        logs = []
        records = {}
        for rank in range(4):
            rows = [json.loads(line) for line in (train / f'training_rank{rank}.jsonl').read_text().splitlines()]
            assert len(rows) == 300
            for update, row in enumerate(rows):
                assert row['completed_updates'] == update + 1
                assert row['completed_exposures'] == 8 * (update + 1)
                assert row['source_indices'] == stream[update * 8 + rank * 2:update * 8 + rank * 2 + 2]
                assert row['frozen_core_unchanged'] and row['actual_feedback_beta'] == .05
                assert row['understanding_weight'] == 1
                assert row['calibration']['geometry_quality_policy'] == 'monitor_v1'
                assert all(math.isfinite(row[k]) for k in ('loss_recon', 'loss_understanding', 'preclip_norm'))
            logs.extend(rows)
            part = evaluation / f'rank{rank:02d}'
            exported = read(part / 'export_complete.json')
            assert exported['status'] == 'EXPORTED' and exported['optimizer_updates'] == 0
            assert exported['frozen_rgb_depth_and_GT_reproduced']
            assert exported['classification_patch_sha256'] == read(patch.with_suffix('.json'))['sha256']
            assert len(exported['records']) == 32
            scored = read(part / 'score_complete.json')
            assert scored['status'] == 'SCORED' and scored['windows'] == 32
            assert scored['optimizer_updates'] == 0 and scored['official_commit'] == official_sha
            assert read(part / 'official_merge_contract.json')['status'] == 'PASS'
            for row in exported['records']:
                assert row['name'] not in records
                records[row['name']] = row
        assert len(records) == 128
        records_by_arm[arm] = records
        metrics = read(evaluation / 'metrics.json')
        assert metrics['windows'] == metrics['unique_scenes'] == 128
        assert metrics['optimizer_updates'] == 0
        assert metrics['official_commit'] == official_sha
        assert metrics['frozen_RGB_depth_and_GT_exactly_reproduced']
        values = summary['arms'][arm]['native_table']
        assert len(values) == 13 and values[8] is None and values[12] is None
        assert all(math.isfinite(x) for x in values if x is not None)
        gt_counts = Counter()
        gt_scenes = defaultdict(set)
        for name, row in records.items():
            for label in row['diagnostics']['gt_internal_semantic_classes']:
                gt_counts[label] += 1
                gt_scenes[label].add(name)
        assert sum(gt_counts.values()) == 492
        per_class = {}
        current = metrics['official_segmentation']['all']
        for view in ('context', 'target'):
            ious0, ious1 = baseline[view + '_ious_per_class'], current[view + '_ious_per_class']
            pq0, pq1 = baseline[view + '_pqs_per_class'][0], current[view + '_pqs_per_class'][0]
            ap0, ap1 = baseline[view + '_map'], current[view + '_map']
            assert ap0['classes'] == ap1['classes']
            ap_deltas = dict(zip(ap0['classes'], [100 * (y - x) for x, y in zip(ap0['map_per_class'], ap1['map_per_class'])]))
            per_class[view] = [dict(label=label, name=label_names[label],
                context_GT_instances=gt_counts[label], context_GT_scenes=len(gt_scenes[label]),
                IoU_delta_pp=100 * (ious1[label] - ious0[label]),
                PQ_delta_pp=100 * (pq1[label] - pq0[label]), AP_delta_pp=ap_deltas.get(label))
                for label in range(20)]
        norms = [row['preclip_norm'] for row in logs]
        arms[arm] = dict(completed_updates=300, new_exposures=2400, training_rank_rows=len(logs),
            gradient_norm_max=max(norms), gradient_norm_median=statistics.median(norms),
            updates_with_preclip_norm_above1=sum(row['preclip_norm'] > 1 for row in logs[:300]),
            per_class_deltas_vs_A=per_class)
    assert len(initial_hashes) == 1
    first = records_by_arm['head_only']
    for arm, records in records_by_arm.items():
        assert set(records) == set(first)
        for name, row in records.items():
            assert row['input_rgb_sha256'] == first[name]['input_rgb_sha256']
            for key in ('gt_instance_ids', 'gt_internal_semantic_classes', 'raw_best_iou_per_GT'):
                assert row['diagnostics'][key] == first[name]['diagnostics'][key]
    concentration = {}
    for view in ('context', 'target'):
        rows = arms['object_support']['per_class_deltas_vs_A'][view]
        excluded = [r for r in rows if r['name'] not in ('curtain', 'shower curtain')]
        ap_rest = [r['AP_delta_pp'] for r in rows if r['name'] != 'shower curtain' and r['AP_delta_pp'] is not None]
        concentration[view] = dict(
            mean_IoU_delta_pp_excluding_curtain_and_shower_curtain=statistics.mean(r['IoU_delta_pp'] for r in excluded),
            mean_AP_delta_pp_excluding_shower_curtain=statistics.mean(ap_rest))
    ranking = {}
    for arm in ('reference', 'head_only', 'object_support', 'scene_support'):
        path = root / 'reference/recovery02/ranking_diagnostics.json' if arm == 'reference' else root / arm / 'evaluation/ranking_diagnostics.json'
        results = read(path)['results']
        ranking[arm] = {view: dict(baseline=results[view]['baseline']['map'],
            GT_ranking_diagnostic=results[view]['GT_iou_ranking_diagnostic']['map'],
            diagnostic_gap_pp=100 * (results[view]['GT_iou_ranking_diagnostic']['map'] - results[view]['baseline']['map']))
            for view in ('context', 'target')}
    report = dict(status='COMPLETED_CLASSIFICATION_CONTROLS_REVIEWED',
        source_checkpoint_sha256=summary['source_checkpoint_sha256'], official_commit=official_sha,
        heldout_scenes=128, new_optimizer_updates=0, model_forward_calls=0,
        completed_budget_and_export_score_contracts_passed=True,
        same_GT_input_and_class_agnostic_raw_mask_IoU_verified=True,
        initial_classifiers_identical=True, arms=arms,
        C_gain_concentration=concentration, ranking_diagnosis_only=ranking,
        caveat='Single-seed 300-update screen. Class-excluded averages diagnose concentration and are not official scores. GT scores are not deployable results.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'C_gain_concentration', 'ranking_diagnosis_only')}, indent=2))


if __name__ == '__main__':
    main()
