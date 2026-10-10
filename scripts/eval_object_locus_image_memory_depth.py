"""Depth-only supplement to the completed U128 epoch8 official evaluation.

No segmentation metrics, training, optimizer, or backward. The pinned SIU3R
Evaluator performs depth alignment and metric computation without modification.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from PIL import Image
from scripts import eval_object_locus_image_memory_official as parent

ROOT = parent.ROOT/'depth_supplement'
SHARDS = 8


def source():
    data = parent.source()
    data.update(depth_protocol_sha256=parent.sha(ROOT/'protocol.json'),
                supplementation='depth only; prior segmentation and RGB metrics reused')
    return data


def register():
    done = parent.read(parent.ROOT/'evaluation_complete.json')
    assert done['status'] == 'COMPLETE'
    for path, digest in done['outputs'].items():
        assert parent.sha(parent.ROOT/path) == digest
    assert parent.sha(parent.ROOT/'protocol.json') == done['source']['protocol_sha256']
    rows = parent.windows()
    spec = dict(parent_protocol_sha256=parent.sha(parent.ROOT/'protocol.json'),
        checkpoint_sha256=done['source']['checkpoint_sha256'], epoch=8,
        object_image_memory_size=128, understanding_step=66752,
        validation_sha256=parent.sha(parent.VAL), windows=len(rows), scenes=312,
        frame_exposures=sum(len(w['context'])+len(w['novel']) for w in rows),
        scopes=list(parent.SCOPES), official_depth_aggregate='target-all; all requested frames including context',
        scope_reduction='Filter official per-frame depth_scores.json by original context/novel frame IDs',
        depth_metric='Pinned SIU3R Evaluator: per-frame scale+shift fit on GT>0, mean frame AbsRel/RMSE',
        official_commit=parent.OFFICIAL_SHA,
        official_evaluator_sha256=parent.sha('/space/mawb/SIU3R/src/evaluator.py'),
        save_depth_sha256=parent.sha(parent.REPO/'scripts/evaluate_ssst_validation.py'),
        provider_sha256=parent.sha(parent.REPO/'scripts/object_locus_v3_set_runtime.py'),
        depth_export='Existing save_depth: clipped [0,65.535] metres, rounded uint16 millimetres PNG',
        prediction_unit_scale=1/0.15,
        ground_truth='Existing provider nearest-resized metric GT on unchanged RGB crop; invalid GT remains zero',
        resolution=256, dtype='FP32', tf32=False, camera_condition='GT camera poses',
        allowed_gpu_partitions=['3090','4090'], excluded_gpu='A6000', shards=SHARDS,
        optimizer_updates=0, backward=0, segmentation_metrics_recomputed=False)
    path = ROOT/'protocol.json'
    if path.exists():
        assert parent.read(path) == spec
    else:
        parent.write(path, spec)
    parent.write(ROOT/'cpu_contracts.json', dict(status='PASS', windows=1860, scenes=312,
        frame_exposures=spec['frame_exposures'], parent_outputs_unchanged=True,
        official_depth_alignment_unchanged=True, optimizer_updates=0, backward=0))
    print('DEPTH_CONTRACTS_PASS', flush=True)


def selected_windows(shard, smoke=False):
    rows = parent.windows()
    if smoke:
        return rows[:1]
    scenes = sorted({w['scene'] for w in rows})[shard::SHARDS]
    return [w for w in rows if w['scene'] in scenes]


def folder(shard, smoke=False):
    return ROOT/('smoke' if smoke else f'shard{shard:02}')


def predict(shard, smoke=False):
    from scripts import object_locus_panoptic_v1_runtime as base
    from scripts.eval_object_locus_v3_set import _run
    from scripts.evaluate_ssst_validation import save_depth
    provenance = source()
    root = folder(shard, smoke)
    rows = selected_windows(shard, smoke)
    root.mkdir(parents=True, exist_ok=True)
    done = root/'prediction_complete.json'
    if done.exists():
        saved = parent.read(done)
        assert saved['source'] == provenance and saved['windows'] == len(rows)
        assert parent.sha(root/'windows.json') == saved['windows_sha256']
        print('DEPTH_PREDICTIONS_REUSED', root, flush=True)
        return
    assert not torch.distributed.is_initialized()
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    assert 'A6000' not in gpu and ('RTX 3090' in gpu or 'RTX 4090' in gpu), gpu
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', 0)
    model, opt, restoration = parent.load_model(device)
    records = []
    start = time.monotonic()
    for index, window in enumerate(rows):
        name = parent.pair_name(window)
        receipt = root/'pairs'/f'{name}.json'
        if receipt.exists():
            saved = parent.read(receipt)
            assert saved['source'] == provenance and saved['window'] == window
            for path, digest in saved['files'].items():
                assert parent.sha(root/path) == digest
            records.append(saved)
            print('DEPTH_PAIR_REUSED', name, flush=True)
            continue
        with torch.no_grad():
            assert not model.training and not torch.is_grad_enabled()
            batch, out = _run(model, opt, window, base.build_batch, device)
            pred = out['render']['depths_pred'][0]
            truth = batch['depth_gt_m_all'][0]
            frames = batch['frame_ids'][0].cpu().tolist()
            assert frames == window['context']+window['novel']
            assert pred.shape == truth.shape == (len(frames), 1, 256, 256)
            assert torch.isfinite(pred).all() and torch.isfinite(truth).all()
            assert abs(float(out['beta'])-.1) < 1e-6
            pair = root/'official_depth'/name
            (pair/'depth').mkdir(parents=True, exist_ok=True)
            (pair/'depth_gt').mkdir(exist_ok=True)
            for v, frame in enumerate(frames):
                filename = f'{window["scene"]}_{frame}.png'
                save_depth(pair/'depth'/filename, pred[v]/0.15)
                save_depth(pair/'depth_gt'/filename, truth[v])
            files = {str(p.relative_to(root)): parent.sha(p) for p in sorted(pair.rglob('*.png'))}
            record = dict(window=window, frame_ids=frames,
                valid_gt_pixels=[int((truth[v]>0).sum()) for v in range(len(frames))],
                source=provenance, files=files)
            parent.write(receipt, record)
            records.append(record)
            del out, batch, pred, truth
        print('DEPTH_PREDICT', shard, index+1, len(rows), name, flush=True)
    parent.write(root/'windows.json', records)
    parent.write(done, dict(status='COMPLETE', source=provenance, windows=len(records),
        windows_sha256=parent.sha(root/'windows.json'), restoration=restoration,
        seconds=time.monotonic()-start, gpu=gpu, node=os.uname().nodename,
        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        job_id=os.environ.get('SLURM_JOB_ID'), optimizer_updates=0, backward=0))


def score(shard, smoke=False):
    from scripts.invoke_siu3r_official_evaluator import evaluate
    provenance = source()
    root = folder(shard, smoke)
    done = parent.read(root/'prediction_complete.json')
    assert done['status'] == 'COMPLETE' and done['source'] == provenance
    assert parent.sha(root/'windows.json') == done['windows_sha256']
    records = parent.read(root/'windows.json')
    per_frame = []
    for row in records:
        pair = root/'official_depth'/parent.pair_name(row['window'])
        for path, digest in row['files'].items():
            assert parent.sha(root/path) == digest
        expected = {f'{row["window"]["scene"]}_{frame}.png' for frame in row['frame_ids']}
        assert {p.name for p in (pair/'depth').glob('*.png')} == expected
        assert {p.name for p in (pair/'depth_gt').glob('*.png')} == expected
        for name in expected:
            with Image.open(pair/'depth'/name) as image:
                assert image.size == (256, 256)
                assert np.asarray(image).dtype == np.uint16
    result = evaluate(root/'official_depth', device='cpu', segmentation=False,
                      image_quality=False, depth_quality=True)
    assert set(result) == {'absrel', 'rmse'}
    for row in records:
        w = row['window']
        values = parent.read(root/'official_depth'/parent.pair_name(w)/'depth_scores.json')
        expected = {f'{w["scene"]}_{f}.png': f for f in row['frame_ids']}
        assert len(values) == len(expected) and {v['item'] for v in values} == set(expected)
        for v in values:
            frame = expected[v['item']]
            per_frame.append(dict(scene=w['scene'], context=w['context'], official_index=w['official_index'],
                frame_id=frame, scope='context' if frame in w['context'] else 'true novel',
                absrel=v['absrel'], rmse=v['rmse'], matching_source='Pinned official depth_scores.json'))
    for key in ('absrel', 'rmse'):
        mean = float(np.mean([v[key] for v in per_frame]))
        assert (math.isnan(mean) and math.isnan(result[key])) or abs(mean-result[key]) < 1e-12
    parent.write(root/'official_depth_metrics.json', parent.clean(dict(result=result,
        frames=len(per_frame), windows=len(records), source=provenance,
        official_evaluator_used=True, optimizer_updates=0, backward=0)))
    parent.write(root/'per_frame_depth.json', parent.clean(per_frame))
    parent.write(root/'complete.json', dict(status='COMPLETE', source=provenance,
        windows=len(records), frames=len(per_frame),
        files={name: parent.sha(root/name) for name in ('official_depth_metrics.json','per_frame_depth.json','windows.json')},
        optimizer_updates=0, backward=0, job_id=os.environ.get('SLURM_JOB_ID')))
    if smoke:
        parent.write(ROOT/'smoke_check.json', dict(status='PASS', windows=1,
            official_depth_metrics=parent.clean(result), restored_memory_size=128,
            restored_exposures=66752, uint16_mm_export=True, unchanged_official_evaluator=True,
            optimizer_updates=0, backward=0))
    print('DEPTH_OFFICIAL_COMPLETE', shard, len(records), len(per_frame), result, flush=True)


def mean_metric(rows, key):
    values = [r[key] for r in rows]
    if not values or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        return 'UNDEFINED'
    return float(np.mean(values))


def reduce():
    assert parent.read(ROOT/'smoke_check.json')['status'] == 'PASS'
    provenance = source()
    frames = []
    windows = []
    for shard in range(SHARDS):
        root = folder(shard)
        done = parent.read(root/'complete.json')
        assert done['status'] == 'COMPLETE' and done['source'] == provenance
        for name, digest in done['files'].items():
            assert parent.sha(root/name) == digest
        frames.extend(parent.read(root/'per_frame_depth.json'))
        windows.extend(parent.read(root/'windows.json'))
    expected = parent.windows()
    assert len(windows) == len(expected) == 1860
    assert {parent.pair_name(w['window']) for w in windows} == {parent.pair_name(w) for w in expected}
    assert len(frames) == sum(len(w['context'])+len(w['novel']) for w in expected)
    assert len({(r['official_index'], r['frame_id']) for r in frames}) == len(frames)
    frames.sort(key=lambda r: (r['official_index'], r['frame_id']))
    dev = {w['scene'] for w in parent.read(parent.REPORT/'manifest.json')['monitor_splits']['dev8']}
    metrics = []
    for cohort in ('all', 'excluding_dev8_scenes'):
        selected = [r for r in frames if cohort == 'all' or r['scene'] not in dev]
        for scope in parent.SCOPES:
            rows = [r for r in selected if scope == 'target-all' or r['scope'] == scope]
            metrics.append(dict(model='U128', epoch=8, cohort=cohort, scope=scope,
                frames=len(rows), windows=len({r['official_index'] for r in rows}),
                scenes=len({r['scene'] for r in rows}), absrel=mean_metric(rows,'absrel'),
                rmse=mean_metric(rows,'rmse'), units='AbsRel ratio; RMSE metres',
                alignment='Official per-frame scale+shift on GT>0', status='COMPLETE'))
    parent.write(ROOT/'depth_metrics.json', metrics)
    parent.csvout(ROOT/'depth_metrics.csv', metrics)
    parent.write(ROOT/'per_frame_depth.json', frames)
    parent.csvout(ROOT/'per_frame_depth.csv', frames)
    original = parent.read(parent.ROOT/'full_validation_metrics.json')
    combined = []
    for row in original:
        matched = [r for r in metrics if r['cohort'] == row['cohort'] and r['scope'] == row['scope']]
        assert len(matched) == 1
        d = matched[0]
        combined.append(dict(row, absrel=d['absrel'], rmse=d['rmse'], depth_frames=d['frames'],
                             depth_alignment=d['alignment'], rmse_unit='metres'))
    parent.write(ROOT/'full_validation_metrics_with_depth.json', combined)
    parent.csvout(ROOT/'full_validation_metrics_with_depth.csv', combined)
    overall = next(r for r in metrics if r['cohort'] == 'all' and r['scope'] == 'target-all')
    context = next(r for r in original if r['cohort'] == 'all' and r['scope'] == 'context')
    novel = next(r for r in original if r['cohort'] == 'all' and r['scope'] == 'true novel')
    def fmt(v, decimals=4):
        return f'{v:.{decimals}f}' if isinstance(v, (float, int)) and math.isfinite(v) else '--'
    latex = r'$\star$ \textbf{Ours (U128, epoch8)}'+'\n& '+' & '.join([
        fmt(overall['absrel']),fmt(overall['rmse']),fmt(novel['psnr'],2),fmt(novel['ssim']),fmt(novel['lpips']),
        fmt(context['official_miou']),fmt(context['official_map']),fmt(context['official_pq']),'--',
        fmt(novel['official_miou']),fmt(novel['official_map']),fmt(novel['official_pq']),'--'])+r' \\'+'\n'
    (ROOT/'paper_ours_row.tex').write_text(latex)
    lines = [(parent.ROOT/'report.md').read_text().rstrip(), '', '## 官方深度补测', '',
        '深度指标来自原SIU3R官方Evaluator：GT>0有效像素上逐帧scale+shift对齐；帧平均AbsRel和RMSE。',
        '深度采用原crop/nearest GT和scene_scale=0.15固定单位转换；导出为uint16毫米PNG。',
        '论文Depth Estimation两列使用官方全target帧汇总（含context）；NVS仍使用true novel。',
        '以下scope拆分复用官方逐帧depth_scores，不引入新的指标实现。', '',
        '| Cohort | Scope | Frames | AbsRel | RMSE (m) |', '|---|---|---:|---:|---:|']
    for row in metrics:
        lines.append(f'|{row["cohort"]}|{row["scope"]}|{row["frames"]}|{fmt(row["absrel"],6)}|{fmt(row["rmse"],6)}|')
    lines += ['', '原分割和RGB结果直接复用，未重复评测；optimizer updates=0、backward=0。',
        '未测mIoU_t继续标为--。不沿用旧checkpoint的深度数值。']
    (ROOT/'report_with_depth.md').write_text('\n'.join(lines)+'\n')
    files = ['depth_metrics.json','depth_metrics.csv','per_frame_depth.json','per_frame_depth.csv',
             'full_validation_metrics_with_depth.json','full_validation_metrics_with_depth.csv',
             'paper_ours_row.tex','report_with_depth.md']
    # Verify original results still match their completion receipt, and new tables decode.
    old_done = parent.read(parent.ROOT/'evaluation_complete.json')
    for name, digest in old_done['outputs'].items():
        assert parent.sha(parent.ROOT/name) == digest
    for name in files:
        if name.endswith('.json'):
            parent.read(ROOT/name)
    parent.write(ROOT/'depth_complete.json', dict(status='COMPLETE', source=provenance,
        windows=1860, scenes=312, frames=len(frames),
        outputs={name: parent.sha(ROOT/name) for name in files},
        parent_results_unchanged=True, optimizer_updates=0, backward=0,
        job_id=os.environ.get('SLURM_JOB_ID')))
    print('DEPTH_FULL_COMPLETE', overall, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['contracts','smoke','predict','score','reduce'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    functions = {'contracts': register, 'smoke': lambda: predict(0, True),
        'predict': lambda: predict(a.shard), 'score': lambda: score(a.shard,a.smoke), 'reduce': reduce}
    try:
        functions[a.mode]()
    except Exception as exc:
        parent.write(ROOT/f'failure_{a.mode}_{a.shard:02}.json', dict(status='FAILED',
            exception=repr(exc), job_id=os.environ.get('SLURM_JOB_ID'), optimizer_updates=0, backward=0))
        raise


if __name__ == '__main__':
    main()
