"""U128 endpoint evaluation with the registered Full1201 SIU3R protocol.

Prediction jobs are independent, eval/no-grad, and never construct an optimizer.
Packed export and official sufficient-state reduction reuse the existing code.
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
REPORT = Path('/space/mawb/ssst/group_plus/object_locus_image_memory_u128_full8gpu_v1')
SOURCE = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
ROOT = REPORT/'evaluation_epoch08'
CHECKPOINT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_image_memory_u128_full8gpu_v1/u128/checkpoint_epoch_08.pt')
VAL = Path('/space/mawb/SIU3R/data/scannet/val_pair.json')
TRAIN_SHA = '574889048498d33df2542a81f4768f3758b72e11'
OFFICIAL_SHA = '8ea80166be76854f938e90521f1a5b688b755c87'
SCOPES = {'context': ('all', 'context'), 'target-all': ('all', 'target'), 'true novel': ('novel', 'target')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'UNDEFINED'
    return value


def windows():
    registration = read(SOURCE/'deferred_evaluation_plan.json')
    assert sha(VAL) == sha(SOURCE/'full_validation_manifest.json') == registration['full_manifest_sha256']
    result = []
    for i, row in enumerate(read(VAL)):
        context = list(map(int, row['context_ids']))
        target = list(map(int, row['target_ids']))
        assert len(context) == 2 and set(context) <= set(target)
        result.append(dict(scene=row['scan'], context=context, target=target,
                           novel=[f for f in target if f not in context], pair_iou=row.get('iou'),
                           official_index=i, camera_source='GT camera poses'))
    assert len(result) == 1860 and len({w['scene'] for w in result}) == 312
    assert len({pair_name(w) for w in result}) == len(result)
    return result


def pair_name(window):
    return window['scene']+'_context'+'_'.join(map(str, window['context']))


def contracts():
    assert read(REPORT/'u128/training_verification.json')['status'] == 'PASS'
    blob = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    assert (blob['epoch'], blob['completed_updates'], blob['completed_exposures']) == (8, 8344, 66752)
    assert blob['git_sha'] == TRAIN_SHA and blob['object_image_memory_size'] == 128
    assert blob['plan_sha256'] == sha(REPORT/'training_plan.json')
    rows = windows()
    manifest = read(REPORT/'manifest.json')
    assert not {w['scene'] for w in rows} & set(manifest['actual_train_scenes'])
    assert subprocess.check_output(['git', '-C', '/space/mawb/SIU3R', 'rev-parse', 'HEAD'], text=True).strip() == OFFICIAL_SHA
    # Metric reduction helpers must be byte-identical to the completed Full1201.
    old = Path('/space/mawb/ssst_object_locus_panoptic_full1201/scripts/aggregate_object_locus_panoptic_full1201_official.py').read_text()
    core = old[old.index('def itemize('):old.index('\ndef shard(')]
    ours = (REPO/'scripts/object_locus_image_memory_official_metrics.py').read_text()
    assert core in ours
    for name in ['export_object_locus_v3_set_official.py', 'invoke_siu3r_official_evaluator.py']:
        assert sha(REPO/'scripts'/name) == sha(Path('/space/mawb/ssst_object_locus_panoptic_full1201/scripts')/name)
    provenance = dict(checkpoint_path=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
        epoch=8, completed_updates=8344, completed_exposures=66752, understanding_step=66752,
        object_image_memory_size=128, training_sha=TRAIN_SHA,
        validation_path=str(VAL), validation_sha256=sha(VAL), windows=1860, scenes=312,
        scopes=SCOPES, official_commit=OFFICIAL_SHA,
        official_evaluator_sha256=sha('/space/mawb/SIU3R/src/evaluator.py'),
        packed_export_sha256=sha(REPO/'scripts/export_object_locus_v3_set_official.py'),
        provider_sha256=sha(REPO/'scripts/object_locus_v3_set_runtime.py'),
        full1201_metric_helpers_sha256=sha(REPO/'scripts/object_locus_image_memory_official_metrics.py'),
        checkpoint_config=blob['config'], eval_mode=True, no_grad=True, backward=0, optimizer_updates=0,
        filters='No training eligibility or thing-count/area filtering',
        posed_setting=True, model_input='2 context RGB, GT camera poses',
        quality_aggregation='Same Full1201: scope-MSE PSNR per window; per-frame SSIM/LPIPS then per-window mean',
        selection='U128 epoch8 fixed by user; no validation-based checkpoint selection',
        comparison='Full1201 epoch6 historical selected baseline; different epoch, not an equal-duration ablation')
    provenance = json.loads(json.dumps(provenance))
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path) == provenance, 'registered evaluation assets changed'
    else:
        write(path, provenance)
    write(ROOT/'cpu_contracts.json', dict(status='PASS', windows=len(rows), scenes=312,
                                        official_metric_helpers_unchanged=True, optimizer_updates=0))
    print('CONTRACTS_PASS', flush=True)


def source():
    p = read(ROOT/'protocol.json')
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    expected = os.environ.get('EVAL_CODE_SHA')
    if expected:
        assert current == expected, 'Slurm checkout differs from submitted SHA'
    assert subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == ''
    assert sha(VAL) == p['validation_sha256']
    assert sha('/space/mawb/SIU3R/src/evaluator.py') == p['official_evaluator_sha256']
    assert subprocess.check_output(['git', '-C', '/space/mawb/SIU3R', 'rev-parse', 'HEAD'], text=True).strip() == OFFICIAL_SHA
    return dict(protocol_sha256=sha(ROOT/'protocol.json'), checkpoint_sha256=p['checkpoint_sha256'],
                evaluation_sha=current, training_sha=TRAIN_SHA, object_image_memory_size=128,
                understanding_step=66752, optimizer_updates=0, backward=0)


def load_model(device):
    from scripts import object_locus_panoptic_v1_runtime as base
    from scripts.object_locus_image_memory_checkpoint import load_model_checkpoint
    assert sha(CHECKPOINT) == read(ROOT/'protocol.json')['checkpoint_sha256']
    model, opt = base.build_model('cpu', report=False)
    metadata = load_model_checkpoint(model, CHECKPOINT)
    expected = read(ROOT/'protocol.json')['checkpoint_config']
    assert json.loads(json.dumps(dataclasses.asdict(opt))) == {k: v for k, v in expected.items() if k != 'object_image_memory_size'}
    assert metadata['object_image_memory_size'] == 128 and metadata['understanding_step'] == 66752
    model = model.to(device).float().eval()
    return model, opt, metadata


def predict(shard, smoke=False):
    from scripts import object_locus_panoptic_v1_runtime as base
    from scripts.eval_object_locus_v3_set import _run
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    provenance = source()
    torch.cuda.set_device(0)
    assert not torch.distributed.is_initialized()
    assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 3090'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = windows()
    if smoke:
        selected = selected[:1]
        root = ROOT/'smoke'
    else:
        scenes = sorted({w['scene'] for w in selected})[shard::8]
        selected = [w for w in selected if w['scene'] in scenes]
        root = ROOT/f'shard{shard:02}'
    root.mkdir(parents=True, exist_ok=True)
    done = root/'complete.json'
    if done.exists():
        saved = read(done)
        assert saved['source'] == provenance and saved['windows'] == len(selected)
        for name, digest in saved['files'].items():
            assert sha(root/name) == digest
        print('REUSED', root, flush=True)
        return
    device = torch.device('cuda', 0)
    model, opt, metadata = load_model(device)
    ssim = StructuralSimilarityIndexMeasure(sync_on_compute=False).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity('vgg', normalize=True, sync_on_compute=False).to(device).eval()
    records = []
    begin = time.monotonic()
    for index, window in enumerate(selected):
        name = pair_name(window)
        receipt = root/'pairs'/f'{name}.json'
        if receipt.exists():
            row = read(receipt)
            assert row['source'] == provenance and row['window'] == window
            for path, digest in row['files'].items():
                assert sha(root/path) == digest
            records.append(row)
            print('REUSED_PAIR', name, flush=True)
            continue
        with torch.no_grad():
            assert not model.training and not torch.is_grad_enabled()
            batch, out = _run(model, opt, window, base.build_batch, device)
            assert out['F_m'].shape[:3] == (1, 2, 256)
            for key in ('p_class', 'region_mass', 'semantic_scores', 'gaussians'):
                assert torch.isfinite(out[key]).all(), (name, key)
            assert torch.isfinite(out['render']['images_pred']).all(), name
            assert abs(float(out['beta'])-.1) < 1e-6
            frames = batch['frame_ids'][0].cpu().tolist()
            indices = {'context': [0, 1], 'target-all': list(range(len(frames))),
                       'true novel': [i for i, f in enumerate(frames) if f in window['novel']]}
            quality = {}
            for scope, ids in indices.items():
                a = out['render']['images_pred'][0, ids]
                b = batch['images_all'][0, ids]
                psnr = float(-10*torch.log10((a-b).square().mean().clamp_min(1e-12)))
                vals = []
                for av, bv in zip(a.clamp(0, 1), b.clamp(0, 1)):
                    ssim.reset()
                    lpips.reset()
                    vals.append((float(ssim(av[None], bv[None])), float(lpips(av[None], bv[None]))))
                quality[scope] = dict(psnr=psnr, ssim=float(np.mean([v[0] for v in vals])),
                                      lpips=float(np.mean([v[1] for v in vals])))
            for arm in ('all', 'novel'):
                write_official_pair(out, batch, window, root/'official'/arm, target_frames=arm)
            files = {str(p.relative_to(root)): sha(p) for arm in ('all', 'novel')
                     for p in sorted((root/'official'/arm/name).rglob('*')) if p.is_file()}
            row = dict(window=window, quality=quality, source=provenance, files=files)
            write(receipt, row)
            records.append(row)
            del out, batch
        print('PREDICT', shard, index+1, len(selected), name, flush=True)
    write(root/'windows.json', records)
    write(done, dict(status='COMPLETE', source=provenance, windows=len(records),
                     files={'windows.json': sha(root/'windows.json')}, restoration=metadata,
                     seconds=time.monotonic()-begin, peak_allocated=torch.cuda.max_memory_allocated(),
                     peak_reserved=torch.cuda.max_memory_reserved(), job_id=os.environ.get('SLURM_JOB_ID')))


def smoke_check():
    from scripts.eval_object_locus_v1 import _official_run
    from scripts.object_locus_image_memory_official_metrics import create, update, states, merge
    root = ROOT/'smoke'
    assert read(root/'complete.json')['status'] == 'COMPLETE'
    for arm in ('all', 'novel'):
        path = root/'official'/arm
        expected = _official_run(path, root/f'official_{arm}.json')['result']
        e = create(path)
        names = sorted(p.name for p in path.iterdir() if p.is_dir())
        for name in names:
            for view in ('context', 'target'):
                update(e, e.process_segmentation(path/name/f'{view}_seg_pred', path/name/f'{view}_seg_gt'), view)
        actual = merge([dict(names=names, states=states(e))], path)
        for view in ('context', 'target'):
            for key in ('miou', 'pq'):
                assert abs(actual[view+'_'+key]-expected[view+'_'+key]) < 1e-7
            for key in ('map', 'map_50'):
                assert abs(actual[view+'_map'][key]-expected[view+'_map'][key]) < 1e-7
    write(ROOT/'smoke_check.json', dict(status='PASS', official_evaluator=True,
        official_state_merge_matches=True, restored_memory_size=128, restored_exposures=66752,
        optimizer_updates=0, backward=0))
    print('OFFICIAL_SMOKE_PASS', flush=True)


def prepare():
    assert read(ROOT/'smoke_check.json')['status'] == 'PASS'
    rows = []
    for shard in range(8):
        root = ROOT/f'shard{shard:02}'
        done = read(root/'complete.json')
        assert done['status'] == 'COMPLETE' and done['source'] == source()
        assert sha(root/'windows.json') == done['files']['windows.json']
        for row in read(root/'windows.json'):
            for path, digest in row['files'].items():
                assert sha(root/path) == digest
            rows.append(row)
            for arm in ('all', 'novel'):
                parent = ROOT/'aggregated_exports'/arm
                parent.mkdir(parents=True, exist_ok=True)
                link = parent/pair_name(row['window'])
                target = (root/'official'/arm/link.name).resolve()
                if link.exists():
                    assert link.resolve() == target
                else:
                    link.symlink_to(target, target_is_directory=True)
    expected = windows()
    assert len(rows) == 1860 and {pair_name(r['window']) for r in rows} == {pair_name(w) for w in expected}
    write(ROOT/'all_windows.json', sorted(rows, key=lambda r: r['window']['official_index']))
    write(ROOT/'prepare_complete.json', dict(status='PASS', windows=1860, optimizer_updates=0))


def aggregate(shard):
    from scripts.object_locus_image_memory_official_metrics import create, update, states
    assert read(ROOT/'prepare_complete.json')['status'] == 'PASS'
    dev = {w['scene'] for w in read(REPORT/'manifest.json')['monitor_splits']['dev8']}
    scenes = sorted({w['scene'] for w in windows()})[shard::8]
    names = sorted(pair_name(w) for w in windows() if w['scene'] in scenes)
    saved = {}
    for arm in ('all', 'novel'):
        path = ROOT/'aggregated_exports'/arm
        totals = {c: create(path) for c in ('all', 'excluding_dev8_scenes')}
        cohort_names = {c: [] for c in totals}
        for index, name in enumerate(names):
            for view in ('context', 'target'):
                data = totals['all'].process_segmentation(path/name/f'{view}_seg_pred', path/name/f'{view}_seg_gt')
                update(totals['all'], data, view)
                if name.split('_context')[0] not in dev:
                    update(totals['excluding_dev8_scenes'], data, view)
            cohort_names['all'].append(name)
            if name.split('_context')[0] not in dev:
                cohort_names['excluding_dev8_scenes'].append(name)
            print('OFFICIAL_RAW', shard, arm, index+1, len(names), name, flush=True)
        saved[arm] = {c: dict(names=cohort_names[c], states=states(e)) for c, e in totals.items()}
    path = ROOT/f'official_states_shard{shard:02}.pt'
    tmp = path.with_suffix('.tmp')
    torch.save(saved, tmp)
    os.replace(tmp, path)
    write(ROOT/f'official_shard{shard:02}_complete.json', dict(status='COMPLETE', source=source(),
        windows=len(names), state_sha256=sha(path), optimizer_updates=0))


def finish():
    from scripts.object_locus_image_memory_official_metrics import merge
    blobs = []
    for shard in range(8):
        record = read(ROOT/f'official_shard{shard:02}_complete.json')
        path = ROOT/f'official_states_shard{shard:02}.pt'
        assert record['status'] == 'COMPLETE' and record['source'] == source()
        assert sha(path) == record['state_sha256']
        blobs.append(torch.load(path, map_location='cpu', weights_only=False))
    cohorts = {c: {a: merge([b[a][c] for b in blobs], ROOT) for a in ('all', 'novel')}
               for c in ('all', 'excluding_dev8_scenes')}
    write(ROOT/'official_aggregated.json', clean(dict(cohorts=cohorts, source=source(),
        aggregation='Original Full1201: sufficient-state sum and canonical pair order for global AP, never averaged scene AP')))
    records = read(ROOT/'all_windows.json')
    dev = {w['scene'] for w in read(REPORT/'manifest.json')['monitor_splits']['dev8']}
    metrics = []
    for cohort, official in cohorts.items():
        selected = [r for r in records if cohort == 'all' or r['window']['scene'] not in dev]
        for scope, (arm, view) in SCOPES.items():
            data = official[arm]
            ap = data[view+'_map']
            row = dict(model='U128', epoch=8, cohort=cohort, scope=scope, windows=len(selected),
                scenes=len({r['window']['scene'] for r in selected}), official_miou=data[view+'_miou'],
                official_pq=data[view+'_pq'], official_map=ap['map'], official_ap50=ap['map_50'],
                psnr=float(np.mean([r['quality'][scope]['psnr'] for r in selected])),
                ssim=float(np.mean([r['quality'][scope]['ssim'] for r in selected])),
                lpips=float(np.mean([r['quality'][scope]['lpips'] for r in selected])), status='COMPLETE')
            for k in ('official_miou', 'official_pq', 'official_map', 'official_ap50'):
                if not math.isfinite(row[k]) or row[k] < 0:
                    row[k] = 'UNDEFINED'
            metrics.append(clean(row))
    write(ROOT/'full_validation_metrics.json', metrics)
    csvout(ROOT/'full_validation_metrics.csv', metrics)
    baseline = list(csv.DictReader((SOURCE/'full_validation_metrics.csv').open()))
    comparison = []
    for row in metrics:
        oldscope = {'target-all': 'target_all', 'true novel': 'novel'}.get(row['scope'], row['scope'])
        matches = [b for b in baseline if b['epoch'] == '6' and b['scope'] == oldscope and b['cohort'] == row['cohort']]
        assert len(matches) == 1
        old = matches[0]
        for key in ('official_miou', 'official_pq', 'official_map', 'official_ap50', 'psnr', 'ssim', 'lpips'):
            try:
                oldval = float(old[key])
            except ValueError:
                oldval = old[key]
            newval = row[key]
            comparison.append(dict(cohort=row['cohort'], scope=row['scope'], metric=key,
                u128_epoch8=newval, full1201_epoch6=oldval,
                difference=newval-oldval if isinstance(newval, (int, float)) and isinstance(oldval, (int, float)) else 'UNDEFINED'))
    write(ROOT/'comparison_full1201_epoch6.json', comparison)
    csvout(ROOT/'comparison_full1201_epoch6.csv', comparison)
    report = ['# U128 epoch8：SIU3R 官方协议评测', '',
        '固定完整验证清单：1860窗口、312个scene。输入为两张context RGB，使用GT camera poses。',
        'U128 epoch8由用户指定，不进行选模。恢复 image memory=128、understanding_step=66752。',
        'packed-panoptic与官方指标实现沿用Full1201；AP从所有原始预测的充分状态整体重算，不平均scene AP。',
        '比例指标为0–1；PSNR为dB。UNDEFINED表示指标不成立，不填0。', '',
        '| Cohort | Scope | mIoU | PQ | mAP | AP50 | PSNR | SSIM | LPIPS |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics:
        def fmt(k):
            return f'{row[k]:.6f}' if isinstance(row[k], (float, int)) else row[k]
        report.append('|'+row['cohort']+'|'+row['scope']+'|'+'|'.join(fmt(k) for k in ('official_miou','official_pq','official_map','official_ap50','psnr','ssim','lpips'))+'|')
    report += ['', '与旧Full1201 epoch6的同清单、同scope结果比较见 comparison_full1201_epoch6.csv/json。',
        '训练节点不同：U128 epoch8（8344 updates）与历史选出的C32 epoch6（6258 updates）；不能归因于分辨率这一单一因素。',
        '旧Full1201用dev8选模；dev8属于val32。额外汇总排除dev8全部scene的验证子集。',
        '当前为GT-pose条件，不宣称等同unposed SIU3R。未新增local实例诊断或定性图片。',
        '本轮 optimizer updates=0、backward=0；未修改模型、loss、阈值或官方导出协议。']
    (ROOT/'report.md').write_text('\n'.join(report)+'\n')
    outputs = ['full_validation_metrics.json','full_validation_metrics.csv','comparison_full1201_epoch6.json',
               'comparison_full1201_epoch6.csv','report.md','official_aggregated.json']
    write(ROOT/'evaluation_complete.json', dict(status='COMPLETE', source=source(), windows=1860, scenes=312,
        outputs={n: sha(ROOT/n) for n in outputs}, optimizer_updates=0, backward=0,
        job_id=os.environ.get('SLURM_JOB_ID')))


def csvout(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['contracts', 'smoke', 'predict', 'smoke-check', 'prepare', 'aggregate', 'finish'])
    p.add_argument('--shard', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4)
    functions = {'contracts': contracts, 'smoke': lambda: predict(0, True), 'predict': lambda: predict(args.shard),
                 'smoke-check': smoke_check, 'prepare': prepare, 'aggregate': lambda: aggregate(args.shard), 'finish': finish}
    try:
        functions[args.mode]()
    except Exception as exc:
        write(ROOT/f'failure_{args.mode}_{args.shard:02}.json', dict(status='FAILED', exception=repr(exc),
            job_id=os.environ.get('SLURM_JOB_ID'), optimizer_updates=0, backward=0))
        raise


if __name__ == '__main__':
    main()
