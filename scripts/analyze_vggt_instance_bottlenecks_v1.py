"""Read saved official RLE states; diagnose retained instances without inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
import torch


def rles(values):
    return [dict(size=list(x[0]), counts=x[1]) for x in values]


def mask_matrix(pred, gt):
    """COCO IoU and areas, preserving native multi-view instance identities."""
    pa = np.asarray(coco_mask.area(pred), dtype=float) if pred else np.zeros(0)
    ga = np.asarray(coco_mask.area(gt), dtype=float) if gt else np.zeros(0)
    iou = np.asarray(coco_mask.iou(pred, gt, [0] * len(gt)), dtype=float) if pred and gt else np.zeros((len(pred), len(gt)))
    assert iou.shape == (len(pred), len(gt)) and np.isfinite(iou).all()
    intersection = iou * (pa[:, None] + ga[None, :]) / (1 + iou)
    return iou, pa, ga, intersection


def best_masks(iou, labels, gtlabels):
    best = iou.max(0) if len(iou) else np.zeros(iou.shape[1])
    same = np.where(labels[:, None] == gtlabels[None, :], iou, 0)
    samebest = same.max(0) if len(same) else np.zeros(iou.shape[1])
    return best, samebest


def score_matches(iou, labels, gtlabels, scores, threshold=.5):
    """Diagnostic score-ordered one-to-one matches, not an official AP score."""
    used = set(); matched = np.full(len(labels), -1, dtype=int)
    for p in np.argsort(-scores, kind='stable'):
        candidates = [g for g in range(len(gtlabels)) if g not in used and labels[p] == gtlabels[g] and iou[p, g] >= threshold]
        if candidates:
            g = max(candidates, key=lambda k: iou[p, k]); matched[p] = g; used.add(g)
    return matched


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expanded-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    objects = []; windows = []; detections = []; identities = {}; results = {}
    for arm in 'ABC':
        counters = {view: dict(gt=0, pred=0, best50=0, best75=0, same50=0, same75=0, tp50=0,
                    no_overlap=0, partial=0, class_gap=0, split_pattern=0, merge_pattern=0,
                    best_iou_sum=0., confidence_tp_sum=0., confidence_fp_sum=0.) for view in ('context', 'target')}
        for rank in range(4):
            blob = torch.load(a.expanded_root / arm / f'rank{rank:02d}' / 'official_states.pt', map_location='cpu', mmap=True, weights_only=False)['confirmation248']
            for view in ('context', 'target'):
                state = blob['states'][view + '_mAP']
                assert len(state['groundtruth_mask']) == len(blob['names'])
                for index, name in enumerate(blob['names']):
                    gtvalues = state['groundtruth_mask'][index]; gtlabels = state['groundtruth_labels'][index].numpy()
                    identity = (gtvalues, tuple(gtlabels))
                    key = (name, view)
                    if arm == 'A': identities[key] = identity
                    else: assert identity == identities[key], 'native AP ground truth differs between arms'
                    pred = rles(state['detection_mask'][index]); gt = rles(gtvalues)
                    labels = state['detection_labels'][index].numpy(); scores = state['detection_scores'][index].numpy()
                    assert len(pred) == len(labels) == len(scores) and np.isfinite(scores).all()
                    iou, pa, ga, overlap = mask_matrix(pred, gt); best, samebest = best_masks(iou, labels, gtlabels)
                    matched = score_matches(iou, labels, gtlabels, scores)
                    shape = gt[0]['size'] if gt else pred[0]['size'] if pred else (256 * (2 if view == 'context' else 6), 256)
                    pixels = np.prod(shape); count = counters[view]
                    split = ((overlap / np.maximum(ga[None, :], 1) >= .2) & (overlap / np.maximum(pa[:, None], 1) >= .5)).sum(0) >= 2
                    merge = ((overlap / np.maximum(ga[None, :], 1) >= .2) & (overlap / np.maximum(pa[:, None], 1) >= .2)).sum(1) >= 2
                    increments = dict(gt=len(gt), pred=len(pred), best50=int((best >= .5).sum()), best75=int((best >= .75).sum()),
                        same50=int((samebest >= .5).sum()), same75=int((samebest >= .75).sum()), tp50=int((matched >= 0).sum()),
                        no_overlap=int((best < .1).sum()), partial=int(((best >= .1) & (best < .5)).sum()),
                        class_gap=int(((best >= .5) & (samebest < .5)).sum()), split_pattern=int(split.sum()), merge_pattern=int(merge.sum()),
                        best_iou_sum=float(best.sum()), confidence_tp_sum=float(scores[matched >= 0].sum()), confidence_fp_sum=float(scores[matched < 0].sum()))
                    for k, v in increments.items(): count[k] += v
                    windows.append(dict(arm=arm, view=view, name=name, **increments))
                    for g in range(len(gt)):
                        fraction = ga[g] / pixels
                        objects.append(dict(arm=arm, view=view, name=name, gt_index=g, class_id=int(gtlabels[g]) + 1,
                            pixels=int(ga[g]), area_fraction=float(fraction), size_group='tiny<0.1%' if fraction < .001 else 'small0.1-1%' if fraction < .01 else 'large>=1%',
                            best_iou=float(best[g]), same_class_best_iou=float(samebest[g]), split_pattern=bool(split[g])))
                    for d in range(len(pred)):
                        ag = float(iou[d].max()) if len(gt) else 0.
                        sc = float(np.where(gtlabels == labels[d], iou[d], 0).max()) if len(gt) else 0.
                        reason = 'matched' if matched[d] >= 0 else 'duplicate_or_assignment' if sc >= .5 else 'class_error' if ag >= .5 else 'partial_mask' if ag >= .1 else 'no_geometric_match'
                        detections.append(dict(arm=arm, view=view, name=name, pred_index=d, class_id=int(labels[d]) + 1,
                            score=float(scores[d]), best_iou=ag, same_class_best_iou=sc, outcome=reason, merge_pattern=bool(merge[d])))
            print('STATES_ANALYZED', arm, rank, len(blob['names']), flush=True)
        results[arm] = {}
        for view, c in counters.items():
            results[arm][view] = dict(**c, agnostic_best_mask_recall50=c['best50'] / c['gt'], same_class_best_mask_recall50=c['same50'] / c['gt'],
                agnostic_best_mask_recall75=c['best75'] / c['gt'], same_class_best_mask_recall75=c['same75'] / c['gt'],
                score_ordered_precision50=c['tp50'] / c['pred'], score_ordered_recall50=c['tp50'] / c['gt'],
                mean_gt_best_iou=c['best_iou_sum'] / c['gt'], unmatched_gt_rate=(c['gt'] - c['tp50']) / c['gt'])
    for fname, rows in (('objects.csv', objects), ('windows.csv', windows), ('detections.csv', detections)):
        with (a.output / fname).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    slices = {}
    for arm in 'ABC':
        slices[arm] = {}
        for view in ('context', 'target'):
            selected = [r for r in objects if r['arm'] == arm and r['view'] == view]
            slices[arm][view] = {}
            for group, key in [('size', 'size_group'), ('class', 'class_id')]:
                for value in sorted({r[key] for r in selected}, key=str):
                    rows = [r for r in selected if r[key] == value]
                    slices[arm][view][f'{group}/{value}'] = dict(gt=len(rows), best50=float(np.mean([r['best_iou'] >= .5 for r in rows])),
                        same50=float(np.mean([r['same_class_best_iou'] >= .5 for r in rows])), mean_best_iou=float(np.mean([r['best_iou'] for r in rows])))
    # Failure-stratified examples; diagnostic selection, never a validation score.
    candidates = [r for r in windows if r['arm'] == 'C' and r['view'] == 'target']
    chosen = {}; used_scenes = set()
    for stratum, key in [('missing', 'no_overlap'), ('partial', 'partial'), ('class', 'class_gap'), ('split', 'split_pattern')]:
        for r in sorted(candidates, key=lambda x: (-x[key] / max(x['gt'], 1), x['name'])):
            scene = r['name'].split('_context')[0]
            if scene not in used_scenes:
                chosen[r['name']] = stratum; used_scenes.add(scene)
                if sum(v == stratum for v in chosen.values()) == 4: break
    # Add four reasonably fitted windows, without interpreting this selected set as representative.
    for r in sorted(candidates, key=lambda x: (-x['tp50'] / max(x['gt'], 1), x['name'])):
        scene = r['name'].split('_context')[0]
        if scene not in used_scenes:
            chosen[r['name']] = 'fitted'; used_scenes.add(scene)
            if sum(v == 'fitted' for v in chosen.values()) == 4: break
    records = {}
    for rank in range(4):
        receipt = json.loads((a.expanded_root / 'A' / f'rank{rank:02d}' / 'export_complete.json').read_text())
        for record in receipt['records']:
            if record['name'] in chosen: records[record['name']] = dict(record, selection_stratum=chosen[record['name']], original_rank=rank)
    assert len(records) == len(chosen) == 20 and len(identities) == 1478 * 2
    report = dict(status='RETAINED_INSTANCE_BOTTLENECKS_ANALYZED',cohort='confirmation248',scenes=248,windows=1478,
        metric_scope='Diagnostic recall/precision, not replacement SIU3R AP/PQ. Native multi-view RLE masks; no GT void filtering of AP objects.',
        definitions={'best_mask_recall':'Each GT independently finds its best retained mask; no one-to-one constraint.',
            'score_matches':'Score-ordered same-class IoU>=0.5 one-to-one diagnostic.',
            'split_merge':'Overlap patterns, not definitive causal split/merge classification.',
            'size':'GT mask area fraction across concatenated window; not COCO size or 3D object size.'},
        identical_native_AP_ground_truth=True,summary=results,strata=slices,
        limitations=['Only retained assembled predictions are saved. No conclusions about rejected/raw queries without a new read-only forward.',
            'Repeated appearances of the same physical instance in different windows are not unique scene-instance counts.',
            'Error categories describe saved output; they do not identify a causal network module.'],
        new_optimizer_updates=0,new_inference_calls=0)
    (a.output / 'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (a.output / 'raw_readout_cohort.json').write_text(json.dumps(dict(status='FAILURE_STRATIFIED_DIAGNOSTIC_ONLY',records=[records[n] for n in sorted(records)],
        excluded_selection64=True,not_a_representative_validation_cohort=True),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__': main()
