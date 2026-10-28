"""Unmodified SIU3R segmentation scoring and explicit diagnostic summaries."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fixed_query_oracles import ARMS
from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_REPO, SIU3R_COMMIT
from scripts.score_dual_stream_v1 import itemize, json_safe
from scripts.paired_e0_checks import verify_export_control

REFERENCE = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1/eval/evaluation')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def flatten(raw, recon):
    row = {k: recon[k] for k in ('psnr', 'ssim', 'lpips', 'absrel', 'rmse')}
    for scope in ('context', 'target'):
        for k in ('miou', 'pq'):
            row[scope + '_' + k] = 100 * raw[scope + '_' + k]
        for k in ('map', 'map_50', 'map_75', 'mar_1', 'mar_10', 'mar_100'):
            row[scope + '_' + k] = 100 * raw[scope + '_map'][k]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    exported = json.loads((root / 'export_complete.json').read_text())
    verify_export_control(exported, json.loads((root / 'PAIRED_BASELINE_VERIFIED.json').read_text()))
    if subprocess.check_output(['git', '-C', SIU3R_REPO, 'rev-parse', 'HEAD'], text=True).strip() != SIU3R_COMMIT:
        raise ValueError('Official evaluator commit changed')
    if subprocess.check_output(['git', '-C', SIU3R_REPO, 'status', '--porcelain', '--', 'src'], text=True).strip():
        raise ValueError('Official evaluator src changed')
    reference = json.loads((REFERENCE / 'metrics.json').read_text())['official_native_result']
    summary = []
    historical_metric_audit = None
    for arm in ARMS:
        # RGB/depth are shared immutable Baseline outputs: no need to recompute
        # their five metrics six times. All segmentation metrics are rescored.
        raw = itemize(evaluate(root / arm / 'evaluation' / 'all', device='cuda:0', image_quality=False, depth_quality=False))
        row = flatten(raw, reference)
        if any(not math.isfinite(v) or v < 0 for v in row.values()):
            raise FloatingPointError('Invalid required metric: ' + arm)
        write_json(root / arm / 'evaluation' / 'metrics.json', dict(status='SCORED', diagnostic_oracle=arm != 'E0',
                   oracle_gt_scope='context_only', target_scope='all6_including_context', windows=1860, scenes=312,
                   parameter_updates=0, official_commit=SIU3R_COMMIT, metrics=row, official_native_result=json_safe(raw),
                   reconstruction_metrics_source=str(REFERENCE / 'metrics.json')))
        if arm == 'E0':
            refrow = flatten(reference, reference)
            delta = {k: row[k] - refrow[k] for k in row}
            historical_metric_audit = dict(metric_delta=delta, reference_metrics=refrow, current_E0_metrics=row,
                historical_differing_pngs=exported['baseline_differing_pngs'],
                historical_differing_pixels=sum(r['historical_differing_pixels'] for r in exported['records']),
                decoded_pixels_equal_to_history=exported['baseline_differing_pngs'] == 0,
                comparison_control='Current E0 from same forward, native assembly verified',
                reconstruction_metric_deltas_zero_by_reuse=['psnr', 'ssim', 'lpips', 'absrel', 'rmse'],
                tolerance_pass_claimed=False)
            write_json(root / 'HISTORICAL_BASELINE_METRIC_AUDIT.json', historical_metric_audit)
        summary.append(dict(arm=arm, **row))
    baseline = summary[0]
    fields = list(baseline)
    with (root / 'metrics_and_deltas.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
            if row['arm'] != 'E0':
                writer.writerow(dict(arm=row['arm'] + '_minus_E0', **{k: row[k] - baseline[k] for k in fields if k != 'arm'}))
    audits = [json.loads(p.read_text()) for p in sorted((root / 'query_audit').glob('*.json'))]
    if len(audits) != 1860 or {a['name'] for a in audits} != {r['name'] for r in exported['records']}:
        raise ValueError('Incomplete per-window diagnostic audit')
    histories = [a['historical_baseline_audit'] for a in audits]
    history_audit = {k:sum(h[k] for h in histories) for k in ('checked_pngs', 'compared_pixels',
        'historical_byte_differing_pngs', 'historical_differing_pngs', 'historical_differing_pixels',
        'historical_semantic_differing_pixels', 'historical_instance_differing_pixels')}
    history_audit['max_abs_score_difference'] = max(h['historical_max_score_difference'] for h in histories)
    history_audit['windows_with_id_changes'] = sum(bool(h['historical_id_changes']) for h in histories)
    history_audit['windows_with_label_changes'] = sum(bool(h['historical_label_changes']) for h in histories)
    history_audit['metric_audit'] = historical_metric_audit
    write_json(root / 'HISTORICAL_BASELINE_AUDIT.json', history_audit)
    instances = [i for a in audits for i in a['instances']]
    coverage = {}
    for scope in ('raw', 'baseline_kept'):
        items = [i[scope] for i in instances]
        coverage[scope] = dict(gt_instances=len(items),
            multiquery_instances=sum(i['count'] >= 2 for i in items),
            fragmented_recoverable_at_iou50=sum(i['best_iou'] < .5 <= i['union_iou'] for i in items),
            mean_best_iou=sum(i['best_iou'] for i in items)/max(1, len(items)),
            mean_union_iou=sum(i['union_iou'] for i in items)/max(1, len(items)),
            mean_best_coverage=sum(i['best_coverage'] for i in items)/max(1, len(items)),
            mean_union_coverage=sum(i['union_coverage'] for i in items)/max(1, len(items)))
    write_json(root / 'diagnostic_summary.json', dict(status='COMPLETE', metrics=summary, coverage=coverage,
               baseline_control='same_forward_native_E0', historical_baseline_audit=history_audit,
               totals={k:sum(a[k] for a in audits) for k in ('queries', 'baseline_gate_count', 'baseline_keep_count', 'supported_queries',
                       'supported_rejected_no_object', 'supported_rejected_confidence', 'supported_rejected_nms', 'wrong_class_supported')},
               recall_definition='official COCO AR@1/10/100, averaged over IoU .50:.05:.95, percentage; distinct from candidate union coverage',
               units='segmentation/AP/AR in percent; differences in percentage points',
               caution='Context-GT-assisted oracle diagnostics, single seed, not deployable improvement or proof of geometry guidance. E2 joint has no query merge.',
               oracle_gt_scope='context_only', target_scope='all6_including_context', parameter_updates=0))
    write_json(root / 'COMPLETE.json', dict(status='FIXED_WEIGHT_DIAGNOSTICS_COMPLETE', parameter_updates=0, windows=1860, scenes=312,
               arms=ARMS, baseline_control='same_forward_native_E0', historical_drift_audit='HISTORICAL_BASELINE_AUDIT.json'))


if __name__ == '__main__':
    main()
