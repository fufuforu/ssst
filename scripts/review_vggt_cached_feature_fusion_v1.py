"""Cached-only equal512-width fusion controls and paired-scene feature review."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.vggt_feature_probe_math_v1 import design, targets, RIDGE
from scripts.diagnose_vggt_feature_reading_v1 import metrics, canonical_identity
from scripts.eval_vggt_joint_short_controls_v1 import write

PAIRS = (('q12', 'q_pre'), ('q_class', 'q12'), ('anchor_local', 'anchor_read12'),
         ('dino_local', 'q_class'), ('gfm4_local', 'gfm23_local'), ('gfm_memory_local', 'gfm4_local'))
BRANCHES = ('duplicate', 'noise', 'dino_local', 'gfm_memory_local', 'object_local', 'visual_local')


def read(path):
    return json.loads(path.read_text())


def bootstrap_difference(a, b, denominator, weights):
    # Each row is a scene/window, preserving its matched-query count.
    values = (weights @ (a - b)) / np.maximum(weights @ denominator, 1)
    return dict(delta_pp=float(100 * (a.sum() - b.sum()) / denominator.sum()),
        paired_scene_bootstrap95_pp=(100 * np.quantile(values, [.025, .975])).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = read(args.root / 'analysis/results.json')
    train, val, tm, vm = [], [], [], []
    for rank in range(8):
        blob = torch.load(args.root / f'rank{rank:02d}/features.pt', map_location='cpu', mmap=True, weights_only=False)
        assert canonical_identity(blob['identity']) == report['identity']
        train += blob['train']; val += blob['validation']
        tm += blob['train_metadata']; vm += blob['validation_metadata']
    assert len(train) == 1024 and len(val) == 128
    assert len({m['scene'] for m in vm}) == 128
    target = {key: torch.stack([r[key] for r in train]).to(device) for key in ('class_target', 'quality_target', 'quality_valid')}
    wc, yc, wq, yq = targets(target)
    q = {split: F.layer_norm(torch.stack([r['features']['q_class'] for r in rows]).to(device), (256,))
         for split, rows in [('train', train), ('validation', val)]}
    coefficients, fusion_results, predictions = {}, {}, {}
    def fusion_x(rows, split, branch):
        if branch == 'duplicate':
            auxiliary = q[split]
        elif branch == 'noise':
            generator = torch.Generator().manual_seed(66452 if split == 'train' else 66453)
            auxiliary = F.layer_norm(torch.randn(len(rows), 100, 256, generator=generator).to(device), (256,))
        else:
            auxiliary = F.layer_norm(torch.stack([r['features'][branch] for r in rows]).to(device), (256,))
        feature = torch.cat((q[split], auxiliary), -1)
        return torch.cat((feature, torch.ones_like(feature[..., :1])), -1).flatten(0, 1)
    for branch in BRANCHES:
        x = fusion_x(train, 'train', branch)
        beta = {}
        for name, weight, y in [('class', wc, yc), ('quality', wq, yq)]:
            w = weight.flatten()[:, None]
            xw = x * w.sqrt()
            gram = (xw.T @ xw).double()
            rhs = (x.T @ (w * y.flatten(0, 1))).double()
            regularizer = torch.eye(513, device=device, dtype=torch.float64) * (RIDGE * len(train))
            regularizer[-1, -1] = 0
            beta[name] = torch.linalg.solve(gram + regularizer, rhs).float()
            assert torch.isfinite(beta[name]).all()
        coefficients[branch] = {k: v.cpu() for k, v in beta.items()}
        fusion_results[branch] = {}
        for split, rows in [('train', train), ('validation', val)]:
            xx = x if split == 'train' else fusion_x(val, 'validation', branch)
            cs = (xx @ beta['class']).cpu().numpy()
            qs = (xx @ beta['quality']).flatten().cpu().numpy()
            fusion_results[branch][split] = metrics(rows, cs, qs)
            if split == 'validation':
                predictions['fusion_' + branch] = cs.reshape(128, 100, 19)
    original = torch.load(args.root / 'probe_coefficients_windows1024.pt', map_location='cpu', weights_only=False)
    for name in {x for pair in PAIRS for x in pair}:
        x = design(torch.stack([r['features'][name] for r in val])).to(device)
        predictions[name] = (x @ original['coefficients'][name]['class'].to(device)).cpu().numpy()
    labels = torch.stack([r['class_target'] for r in val]).numpy()
    matched = labels != 18
    good = matched & (torch.stack([r['matched_iou'] for r in val]).numpy() >= .5)
    generator = np.random.default_rng(91374)
    weights = generator.multinomial(128, np.full(128, 1 / 128), size=5000)
    comparisons = {}
    comparisons_list = list(PAIRS) + [('fusion_' + b, 'fusion_duplicate') for b in BRANCHES if b != 'duplicate']
    for a, b in comparisons_list:
        scores = {}
        for condition, keep in [('matched', matched), ('good_mask', good)]:
            den = keep.sum(-1)
            ac = ((predictions[a][..., :18].argmax(-1) == labels) & keep).sum(-1)
            bc = ((predictions[b][..., :18].argmax(-1) == labels) & keep).sum(-1)
            scores[condition + '_conditional18'] = bootstrap_difference(ac, bc, den, weights)
        comparisons[a + '_minus_' + b] = scores
    disagreement = {}
    for name in ('dino_local',):
        a, b = predictions[name][..., :18].argmax(-1) == labels, predictions['q_class'][..., :18].argmax(-1) == labels
        disagreement[name] = dict(local_correct_query_wrong=int((matched & a & ~b).sum()),
            query_correct_local_wrong=int((matched & b & ~a).sum()),
            caveat='GT case selection is diagnosis, not an achievable oracle score.')
    quality = torch.stack([r['quality_target'] for r in val]).numpy()
    quality_valid = torch.stack([r['quality_valid'] for r in val]).numpy()
    conflicts = dict(unmatched_raw_good_mask_queries=int(((labels == 18) & quality_valid & (quality >= .5)).sum()),
        matched_good_queries=int(good.sum()),
        caveat='An unmatched good raw mask may duplicate a matched object; it does not establish a bad Hungarian label.')
    confidence_deltas = np.asarray([np.mean([c['bins_low_to_high'][-1]['absrel'] - c['bins_low_to_high'][0]['absrel'] for c in m['confidence']]) for m in vm])
    confidence_boot = weights @ confidence_deltas / 128
    result = dict(status='CACHED_FEATURE_FUSION_AND_PAIRED_REVIEW_COMPLETE', cache_identity=report['identity'],
        review_execution_sha=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        fusion_width=512, coefficients_per_task=dict(classification=513 * 19, quality=513),
        same_capacity_fusion_controls=True, fusion_results=fusion_results, paired_scene_review=comparisons,
        DINO_query_disagreements=disagreement, query_assignment_context=conflicts,
        confidence_high_minus_low_absrel=float(confidence_deltas.mean()),
        confidence_paired_scene_bootstrap95_absrel=np.quantile(confidence_boot, [.025, .975]).tolist(),
        encoder_model_forward_calls=0, optimizer_updates=0,
        caveat='Fusion probes have512 inputs versus256 for individual probes; compare them primarily to equal512 duplicate/noise controls. Average-head R@a is a proxy, not the full learned multihead V/O output. Resampled-scene intervals are descriptive under the fixed selected checkpoint and single probe cohort; not independent architecture validation.')
    torch.save(dict(cache_identity=report['identity'], coefficients=coefficients), args.root / 'analysis/fusion_probe_coefficients.pt')
    write(args.root / 'analysis/fusion_paired_review.json', result)
    print(json.dumps(dict(status=result['status'], fusion_validation={k:v['validation'] for k,v in fusion_results.items()},
        paired_review=comparisons, confidence95=result['confidence_paired_scene_bootstrap95_absrel']), indent=2))


if __name__ == '__main__':
    main()
