"""Fixed C: compare query stages, predicted local features, and readout information."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.vggt_readout_probe_v1 import frozen_C, workers, core_versions, assert_core, gather
from scripts.train_vggt_classification_readback_v1 import SOURCE_SHA, TRAIN_COHORT, HELDOUT
from scripts.eval_vggt_joint_short_controls_v1 import write
from scripts.eval_vggt_feedback_expanded_v1 import windows
from scripts.train_vggt_joint_short_controls_v1 import digest
from scripts.vggt_feature_probe_math_v1 import design, new_stats, accumulate, solve, pool_image, RIDGE

TRAIN_WINDOWS = 1024
VIEWS = ('q_pre', 'q6', 'q8', 'q10', 'q12', 'q_class', 'anchor_read12', 'anchor_local',
    'object_local', 'visual_local', 'gfm_memory_local', 'gfm4_local', 'gfm11_local', 'gfm17_local',
    'gfm23_local', 'dino_local', 'scene_local', 'shuffled_qclass', 'null')
PRIOR = Path('/space/mawb/ssst/group_plus/vggt_readout_probe_v1/attempts/attempt03')
PATCH_SHA = 'ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9'


def read(path):
    return json.loads(path.read_text())


def identity():
    audit = read(PRIOR / 'analysis/completion_independent_audit.json')
    assert audit['identity']['source_checkpoint_sha256'] == SOURCE_SHA
    assert audit['identity']['C_classification_patch_sha256'] == PATCH_SHA
    snapshot = PRIOR.parents[1] / 'snapshots' / audit['identity']['execution_sha']
    for name in ('scripts/vggt_readout_probe_v1.py', 'tokengs/models/object_locus_panoptic_v1.py',
                 'tokengs/models/object_locus_panoptic_v1_controller.py', 'tokengs/models/object_locus_frozen_vggt_posefree.py',
                 'tokengs/models/object_locus_mask_support_readback_v1.py'):
        assert digest(REPO / name) == digest(snapshot / name)
    return dict(source_checkpoint_sha256=SOURCE_SHA, C_patch_sha256=PATCH_SHA,
        execution_sha=os.environ['TASK_CODE_SHA'], adapter=.01, feedback=.05,
        train_windows=TRAIN_WINDOWS, val_windows=128, feature_views=VIEWS,
        feature_width=256, projection_seed=92831, ridge=RIDGE, optimizer_updates=0,
        primary_scope='Frozen-feature understanding diagnosis; not official model scores or reconstruction training')


def hooks(model, device):
    model.__dict__['_feature_diag'] = {}
    def vggt_hook(module, inputs, output):
        model._feature_diag['vggt'] = output
    def memory_hook(module, inputs, output):
        model._feature_diag['memory'] = output[0].detach()
    def dino_hook(module, inputs, output):
        model._feature_diag['dino'] = (output['x_norm_patchtokens'] if isinstance(output, dict) else output).detach()
    model.frozen_vggt.register_forward_hook(vggt_hook)
    model.vggt_memory_adapter.register_forward_hook(memory_hook)
    model.frozen_vggt.model.aggregator.patch_embed.register_forward_hook(dino_hook)
    projections = {}
    for width in (1024, 2048):
        generator = torch.Generator().manual_seed(92831 + width)
        projections[width] = (torch.randn(width, 256, generator=generator) / math.sqrt(width)).to(device)
    return projections


def depth_confidence(generated, batch):
    # One fit per entire context frame, shared by all bins; no per-bin fitting.
    prediction = F.interpolate(generated['depth'].flatten(0, 1), (256, 256), mode='bilinear', align_corners=False)[:, 0].double()
    confidence = F.interpolate(generated['confidence518'].flatten(0, 1), (256, 256), mode='bilinear', align_corners=False)[:, 0]
    gt = batch['depth_gt_m_all'][0, :2, 0].double()
    valid = batch['depth_gt_valid_all'][0, :2, 0].bool() & (gt > 0)
    assert prediction.shape == gt.shape == confidence.shape
    result = []
    for i in range(2):
        keep = valid[i] & torch.isfinite(prediction[i]) & torch.isfinite(confidence[i])
        x, y, c = prediction[i][keep], gt[i][keep], confidence[i][keep]
        if len(x) < 10:
            continue
        xc, yc = x - x.mean(), y - y.mean()
        slope = (xc * yc).sum() / xc.square().sum().clamp_min(1e-12)
        error = ((slope * xc + y.mean()) - y).abs() / y
        order = c.argsort(stable=True)
        groups = []
        for idx in order.tensor_split(5):
            groups.append(dict(pixels=len(idx), mean_confidence=float(c[idx].mean()), absrel=float(error[idx].mean())))
        result.append(dict(context_frame=i, bins_low_to_high=groups, confidence_range=float(c.max() - c.min()),
            all_pixels_absrel=float(error.mean()), fit_scale=float(slope)))
    return result


def collect(model, opt, projections, window, device, confidence=False):
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.object_locus_frozen_vggt_posefree_runtime import seed_everything
    from tokengs.models.object_locus_v3_set_loss import final_hungarian
    seed_everything(42 + window['source_index'])
    model._feature_diag.clear()
    batch = build_batch(opt, window, device)
    with torch.no_grad():
        generated = model.generate(batch['images_input'])
        targets, pairs = final_hungarian(generated, batch)
        qi, ki = pairs[0]
        regions = generated['region_mass'][:, :2, :100]
        alpha = generated['alpha'][:, :2]
        mask = ((regions >= .5) & (alpha > .05)).permute(0, 2, 1, 3, 4).flatten(2)
        gt = targets['gt_pixel_masks'][0].flatten(1)
        valid = targets['valid_pixels'][0].flatten()
        raw = mask[0, :, valid]
        target = gt[:, valid]
        iou = raw.float() @ target.float().T if len(gt) else raw.new_zeros((100, 0), dtype=torch.float32)
        if len(gt):
            union = raw.sum(-1)[:, None] + target.sum(-1)[None] - iou
            iou = iou / union.clamp_min(1)
        labels = torch.full((100,), 18, dtype=torch.long, device=device)
        labels[qi] = targets['gt_classes'][0][ki] - 2
        quality = iou.max(-1).values if len(gt) else regions.new_zeros(100)
        matched_iou = regions.new_zeros(100)
        matched_iou[qi] = iou[qi, ki]
        final = generated['states'][-1]
        visibility = generated['lifting_mass'].clamp_min(0)
        gm = generated['gaussian_membership'][..., :100]
        weight = visibility * gm
        pooled = weight.transpose(1, 2) @ generated['gaussian_feature'] / weight.sum(1).unsqueeze(-1).clamp_min(1e-6)
        b, anchors = final['anchor_embedding'].shape[:2]
        assert gm.shape[1] == anchors * 64
        aw = weight.reshape(b, anchors, 64, 100).sum(2)
        anchor = aw.transpose(1, 2) @ final['anchor_embedding'] / aw.sum(1).unsqueeze(-1).clamp_min(1e-6)
        visual_masks = regions * alpha
        features = dict(q_pre=generated['q_pre'][:, :100], q_class=model._readout_feature,
            object_local=pooled, anchor_local=anchor,
            anchor_read12=final['evidence_attention'].mean(1)[:, :100] @ final['anchor_embedding'],
            visual_local=pool_image(generated['F_m'], visual_masks))
        scene = (visibility.transpose(1, 2) @ generated['gaussian_feature']) / visibility.sum(1).unsqueeze(-1).clamp_min(1e-6)
        features['scene_local'] = scene.expand(-1, 100, -1)
        def compress(value):
            return F.layer_norm(value.float(), (value.shape[-1],)) @ projections[value.shape[-1]]
        memory = model._feature_diag['memory'].reshape(b, 2, 37, 37, 1024).permute(0, 1, 4, 2, 3)
        features['gfm_memory_local'] = compress(pool_image(memory, visual_masks))
        for layer, value in zip((4, 11, 17, 23), model._feature_diag['vggt'].patch_layers):
            grid = value.reshape(b, 2, 37, 37, 2048).permute(0, 1, 4, 2, 3)
            features[f'gfm{layer}_local'] = compress(pool_image(grid, visual_masks))
        dino = model._feature_diag['dino']
        assert dino.shape == (b * 2, 1369, 1024)
        features['dino_local'] = compress(pool_image(dino.reshape(b, 2, 37, 37, 1024).permute(0, 1, 4, 2, 3), visual_masks))
        attention = {}
        support = (visibility * (gm >= .5)).reshape(b, anchors, 64, 100).sum(2) / visibility.reshape(b, anchors, 64, 1).sum(2).clamp_min(1e-6)
        uniform = support.mean(1)[0]
        for state in generated['states']:
            if state['layer'] not in (6, 8, 10, 12):
                continue
            layer = state['layer']
            features[f'q{layer}'] = state['q'][:, :100]
            r = state['evidence_attention'].mean(1)[:, :100]
            attention[f'entropy{layer}'] = (-(r * r.clamp_min(1e-12).log()).sum(-1) / math.log(anchors))[0].cpu()
            attention[f'top10_mass{layer}'] = r.topk(10, -1).values.sum(-1)[0].cpu()
            attention[f'own_support{layer}'] = (r * support.transpose(1, 2)).sum(-1)[0].cpu()
        attention['uniform_support'] = uniform.cpu()
        permutation = torch.randperm(100, generator=torch.Generator().manual_seed(773 + window['source_index'])).to(device)
        features['shuffled_qclass'] = features['q_class'][:, permutation]
        features['null'] = torch.zeros_like(features['q_class'])
        assert set(features) == set(VIEWS)
        assert all(v.shape == (1, 100, 256) and torch.isfinite(v).all() for v in features.values())
        native = generated['states'][-1]['thing_logits19'].detach().float()[0]
        assert torch.allclose(model.panoptic.class_head(features['q_class'])[0], native, atol=1e-6, rtol=1e-6)
        item = dict(features={k: v[0].detach().cpu() for k, v in features.items()},
            class_target=labels.cpu(), quality_target=quality.cpu(), quality_valid=raw.any(-1).cpu(),
            matched_iou=matched_iou.cpu(), native_logits=native.cpu(), attention=attention)
        metadata = dict(source_index=window['source_index'], scene=window['scene'],
            frame_ids=batch['frame_ids'][0].cpu().tolist(), gt=len(gt), matched=len(qi),
            input_rgb_sha256=hashlib.sha256(batch['images_all'].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
            confidence=depth_confidence(generated, batch) if confidence else [])
    return item, metadata


def batch_targets(items, device):
    return {key: torch.stack([row[key] for row in items]).to(device) for key in ('class_target', 'quality_target', 'quality_valid')}


def cache(args):
    rank, world, device = workers()
    assert world == 8
    proof = identity()
    from scripts.object_locus_frozen_vggt_posefree_runtime import load_manifest
    _, _, train = load_manifest()
    train = [dict(w, source_index=i) for i, w in enumerate(train)]
    validation, _ = windows()
    stream = read(TRAIN_COHORT)['train_source_indices'][:TRAIN_WINDOWS]
    heldout = read(HELDOUT)['records']
    assert len(stream) == len(set(stream)) == TRAIN_WINDOWS and len(heldout) == 128
    assert not ({w['scene'] for w in train} & {w['scene'] for w in validation})
    root = args.root / f'rank{rank:02d}'
    root.mkdir(parents=True, exist_ok=False)
    model, opt = frozen_C(device)
    projections = hooks(model, device)
    versions = core_versions(model)
    old = torch.load(PRIOR / f'training/cache_rank{rank:02d}.pt', map_location='cpu', mmap=True, weights_only=False)
    assert len(old['items']) == 300
    stats = {name: new_stats(device) for name in VIEWS} if rank == 0 else None
    local_train, train_meta = [], []
    started = time.monotonic()
    checkpoints = []
    for step in range(TRAIN_WINDOWS // 8):
        position = step * 8 + rank
        item, meta = collect(model, opt, projections, train[stream[position]], device)
        assert meta['input_rgb_sha256'] == old['metadata'][step]['input_rgb_sha256']
        assert meta['frame_ids'] == old['metadata'][step]['frame_ids'] and meta['source_index'] == old['metadata'][step]['source_index']
        assert torch.allclose(item['features']['q_class'], old['items'][step]['features'][0], atol=1e-6, rtol=1e-6)
        assert torch.allclose(item['native_logits'], old['items'][step]['class_logits'][0], atol=1e-6, rtol=1e-6)
        assert torch.equal(item['class_target'] < 18, old['items'][step]['object_target'][0].bool())
        assert_core(model, versions)
        local_train.append(item)
        train_meta.append(dict(meta, stream_position=position))
        entries = gather(item, world)
        if rank == 0:
            target = batch_targets(entries, device)
            for name in VIEWS:
                accumulate(stats[name], torch.stack([x['features'][name] for x in entries]).to(device), target)
            completed = (step + 1) * 8
            if completed in (256, 512, 1024):
                path = args.root / f'probe_coefficients_windows{completed:04d}.pt'
                torch.save(dict(identity=proof, train_windows=completed, optimizer_updates=0,
                    coefficients={name: {k: v.cpu() for k, v in solve(stats[name]).items()} for name in VIEWS}), path)
                checkpoints.append(path.name)
            elapsed = time.monotonic() - started
            write(args.root / 'progress.json', dict(stage='frozen_train_features', completed_windows=completed,
                total_windows=TRAIN_WINDOWS, cache_batches=step + 1, optimizer_updates=0,
                remaining_cache_seconds=elapsed / (step + 1) * (TRAIN_WINDOWS // 8 - step - 1),
                core_frozen=True, original_cached_query_and_matching_reproduced=True))
            if step + 1 == 20:
                assert all(s['windows'] == 160 and torch.isfinite(s['class_gram']).all() for s in stats.values())
                write(args.root / 'STARTUP_VERIFIED.json', dict(status='FEATURE_READING_REAL8GPU_CACHE160_VERIFIED',
                    identity=proof, train_windows=160, cache_batches=20, original_C_query_and_match_reproduced=True,
                    core_parameters_buffers_frozen=True, GT_only_in_targets_and_offline_diagnostics=True,
                    optimizer_updates=0, equal_probe_parameter_counts=True))
        dist.barrier()
    local_val, val_meta = [], []
    for i, record in enumerate(heldout[rank::world]):
        item, meta = collect(model, opt, projections, validation[record['source_index']], device, confidence=True)
        assert meta['frame_ids'] == record['frame_ids'] and meta['input_rgb_sha256'] == record['input_rgb_sha256']
        assert_core(model, versions)
        local_val.append(item)
        val_meta.append(dict(meta, name=record['name']))
        write(root / 'validation_progress.json', dict(completed=i + 1, total=16, optimizer_updates=0))
    torch.save(dict(identity=proof, train=local_train, train_metadata=train_meta,
        validation=local_val, validation_metadata=val_meta), root / 'features.pt')
    write(root / 'COMPLETE.json', dict(status='FROZEN_FEATURES_CACHED', identity=proof, train_windows=len(local_train),
        val_windows=len(local_val), core_parameters_buffers_frozen=True, optimizer_updates=0))
    dist.barrier()
    if rank == 0:
        write(args.root / 'CACHE_COMPLETE.json', dict(status='FEATURE_READING_CACHE_COMPLETE', identity=proof,
            train_windows=TRAIN_WINDOWS, val_windows=128, coefficients=checkpoints, optimizer_updates=0))
    dist.destroy_process_group()


def score_vector(items, key):
    return np.concatenate([row[key].numpy() for row in items])


def metrics(items, class_scores, quality_scores):
    from scipy.stats import spearmanr
    label = score_vector(items, 'class_target')
    quality = score_vector(items, 'quality_target')
    valid = score_vector(items, 'quality_valid').astype(bool)
    matched = label != 18
    good = matched & (score_vector(items, 'matched_iou') >= .5)
    predicted = class_scores.argmax(-1)
    conditional = class_scores[:, :18].argmax(-1)
    macro = [float((conditional[label == k] == k).mean()) for k in range(18) if (label == k).any()]
    quality_scores = np.clip(quality_scores, 0, 1)
    def mean(mask, values):
        return float(values[mask].mean()) if mask.any() else None
    correlations = []
    for i in range(len(items)):
        v = valid[i * 100:(i + 1) * 100]
        x, y = quality_scores[i * 100:(i + 1) * 100][v], quality[i * 100:(i + 1) * 100][v]
        if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            correlations.append(float(spearmanr(x, y)[0]))
    return dict(matched_accuracy19=mean(matched, predicted == label),
        matched_conditional_accuracy18=mean(matched, conditional == label),
        macro_matched_conditional_accuracy18=float(np.mean(macro)),
        good_mask_matched_accuracy19=mean(good, predicted == label),
        good_mask_predicted_no_object=int((good & (predicted == 18)).sum()),
        quality_MAE=mean(valid, abs(quality_scores - quality)),
        quality_MSE=mean(valid, (quality_scores - quality) ** 2),
        quality_within_window_spearman=float(np.mean(correlations)) if correlations else None,
        matched_queries=int(matched.sum()), good_matched_queries=int(good.sum()), quality_valid_queries=int(valid.sum()),
        per_class_conditional_accuracy={str(k): float((conditional[label == k] == k).mean()) for k in range(18) if (label == k).any()})


def analyze(args):
    done = read(args.root / 'CACHE_COMPLETE.json')
    assert done['status'] == 'FEATURE_READING_CACHE_COMPLETE' and done['identity'] == identity()
    train, val, train_meta, val_meta = [], [], [], []
    for rank in range(8):
        root = args.root / f'rank{rank:02d}'
        assert read(root / 'COMPLETE.json')['core_parameters_buffers_frozen']
        blob = torch.load(root / 'features.pt', map_location='cpu', mmap=True, weights_only=False)
        assert blob['identity'] == done['identity']
        train += blob['train']; val += blob['validation']
        train_meta += blob['train_metadata']; val_meta += blob['validation_metadata']
    assert len(train) == TRAIN_WINDOWS and len(val) == 128
    assert len({row['stream_position'] for row in train_meta}) == TRAIN_WINDOWS
    assert len({row['name'] for row in val_meta}) == len({row['scene'] for row in val_meta}) == 128
    assert not ({row['scene'] for row in train_meta} & {row['scene'] for row in val_meta})
    # Keep cache entries paired with their labels/metadata, including native query
    # identities. Diagnostics evaluate heads; no official model outputs change.
    out = args.root / 'analysis'
    out.mkdir(exist_ok=False)
    results = {}
    for filename in done['coefficients']:
        checkpoint = torch.load(args.root / filename, map_location='cpu', weights_only=False)
        assert checkpoint['identity'] == done['identity']
        budget = checkpoint['train_windows']
        subset = [row for row, meta in zip(train, train_meta) if meta['stream_position'] < budget]
        assert len(subset) == budget
        result = {}
        for name in VIEWS:
            scores = {}
            for split, items in [('train', subset), ('validation', val)]:
                x = design(torch.stack([row['features'][name] for row in items]))
                beta = checkpoint['coefficients'][name]
                class_scores = (x @ beta['class']).flatten(0, 1).numpy()
                quality_scores = (x @ beta['quality']).flatten().numpy()
                scores[split] = metrics(items, class_scores, quality_scores)
            result[name] = scores
        results[str(budget)] = result
    native = {}
    for split, items in [('train', train), ('validation', val)]:
        logits = torch.stack([row['native_logits'] for row in items])
        native[split] = metrics(items, logits.flatten(0, 1).numpy(), logits.softmax(-1)[..., :18].max(-1).values.flatten().numpy())
    attention = {}
    labels = score_vector(val, 'class_target')
    good = (labels != 18) & (score_vector(val, 'matched_iou') >= .5)
    bad = (labels != 18) & ~good
    for layer in (6, 8, 10, 12):
        entry = {}
        for group, keep in [('matched_good', good), ('matched_bad', bad), ('unmatched', labels == 18)]:
            entry[group] = {key: float(np.concatenate([x['attention'][key].numpy() for x in val])[keep].mean())
                for key in (f'entropy{layer}', f'top10_mass{layer}', f'own_support{layer}', 'uniform_support')}
        attention[str(layer)] = entry
    conf = [r for meta in val_meta for r in meta['confidence'] if r['confidence_range'] > 0]
    confidence = dict(frames=len(conf), bins_low_to_high=[dict(bin=i,
        mean_frame_absrel=float(np.mean([r['bins_low_to_high'][i]['absrel'] for r in conf])),
        mean_frame_confidence=float(np.mean([r['bins_low_to_high'][i]['mean_confidence'] for r in conf]))) for i in range(5)],
        high_minus_low_absrel=float(np.mean([r['bins_low_to_high'][-1]['absrel'] - r['bins_low_to_high'][0]['absrel'] for r in conf])),
        caveat='Per-frame equal-count bins; one scale/shift fit for all pixels. Correlation, not a causal attention intervention; this is VGGT depth, not rendered GS depth.')
    coverage = dict(train={str(k): int((score_vector(train, 'class_target') == k).sum()) for k in range(19)},
        validation={str(k): int((labels == k).sum()) for k in range(19)})
    report = dict(status='FEATURE_READING_DIAGNOSIS_COMPLETE', identity=done['identity'],
        train_windows=TRAIN_WINDOWS, train_scenes=len({r['scene'] for r in train_meta}), val_windows=128,
        feature_probe_results=results, native_C=native, attention=attention, confidence=confidence, class_coverage=coverage,
        same_GT_targets_and_prediction_support=True, optimizer_updates=0,
        caveat='Single-seed equal-capacity linear decodability probes, not official13 scores. Fixed random projection for1024/2048 widths may discard information. Predicted final support has timing/localization advantages; probes and attention dependence do not prove a fusion will improve the model. Cannot explain remaining PSNR gap by classification probes alone.')
    write(out / 'results.json', report)
    lines = ['# 固定C读取瓶颈诊断', '', '固定1024训练窗口、128留出场景；以下为探针诊断，不是官方模型成绩。', '',
        '| 特征位置 | 匹配query19类准确率 | 匹配query条件18类准确率 | 好mask误判no-object数 | mask质量MAE↓ | 窗口内质量Spearman↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for name, values in results[str(TRAIN_WINDOWS)].items():
        m = values['validation']
        fmt = lambda x: '' if x is None else f'{x:.4f}'
        lines.append(f"| {name} | {fmt(m['matched_accuracy19'])} | {fmt(m['matched_conditional_accuracy18'])} | {m['good_mask_predicted_no_object']} | {fmt(m['quality_MAE'])} | {fmt(m['quality_within_window_spearman'])} |")
    lines += ['', '256/512/1024窗口预算曲线、训练/留出比较、逐类与注意力/confidence统计见results.json。', '', report['caveat']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    write(args.root / 'COMPLETE.json', dict(status=report['status'], identity=done['identity'], optimizer_updates=0))
    print(json.dumps(dict(status=report['status'], native_val=native['validation'], confidence=confidence), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cache', 'analyze'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.mode](args)
