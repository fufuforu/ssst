"""Read-only registered-checkpoint reevaluation for Object-Locus V3-Set.

This module never constructs an optimizer, calls backward, or performs updates.
Official metrics are reused verbatim from the original registered result JSONs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.object_locus_v3_set_runtime import build_model, build_options, build_batch, write_json
from scripts.eval_object_locus_v3_set import evaluate_windows
from scripts.train_object_locus_v3_set import _scope_official_metrics

SOURCE_REPORTS = Path('/space/mawb/ssst/group_plus/object_locus_v3_set')
SOURCE_RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_v3_set')
OUTPUT_DEFAULT = Path('/space/mawb/ssst/group_plus/object_locus_v3_set_evalfix')
REGISTERED = {
    0: (0, ('train_probe8', 'same_scene_holdout8', 'dev8', 'train_all56', 'val32')),
    8: (448, ('train_probe8', 'same_scene_holdout8', 'dev8')),
    16: (896, ('train_probe8', 'same_scene_holdout8', 'dev8')),
    32: (1792, ('train_probe8', 'same_scene_holdout8', 'dev8', 'train_all56', 'val32')),
    64: (3584, ('train_probe8', 'same_scene_holdout8', 'dev8', 'train_all56', 'val32')),
}
OFFICIAL_EPOCHS = {0, 32, 64}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    path = SOURCE_REPORTS / 'data_manifest.json'
    data = json.loads(path.read_text())
    needed = {'train_probe8', 'same_scene_holdout8', 'dev8', 'train_all56', 'val32'}
    missing = sorted(needed - set(data))
    if missing:
        raise RuntimeError(f'Original V3-Set manifest lacks fixed splits: {missing}')
    return data, {'path': str(path), 'sha256': sha256(path)}


def _registered_official(epoch, step, split):
    if epoch not in OFFICIAL_EPOCHS:
        return {}, {'status': 'NOT_REGISTERED'}
    root = SOURCE_REPORTS / 'official' / f'step_{step:04d}' / split
    all_path = root / 'official_all.json'
    novel_path = root / 'official_novel.json'
    if not all_path.is_file() or not novel_path.is_file():
        missing = [str(p) for p in (all_path, novel_path) if not p.is_file()]
        return {}, {'status': 'MISSING_REGISTERED_OFFICIAL', 'missing': missing}
    all_blob = json.loads(all_path.read_text())
    novel_blob = json.loads(novel_path.read_text())
    official = {
        'all': all_blob.get('result', {}),
        'novel': novel_blob.get('result', {}),
        '_provenance': {
            'all': {'path': str(all_path), 'sha256': sha256(all_path)},
            'novel': {'path': str(novel_path), 'sha256': sha256(novel_path)},
        },
    }
    return official, {
        'status': 'REUSED_REGISTERED_JSON',
        'all': {'path': str(all_path), 'sha256': sha256(all_path)},
        'novel': {'path': str(novel_path), 'sha256': sha256(novel_path)},
    }


def _copy_registered_official(official, epoch, split, out):
    if epoch not in OFFICIAL_EPOCHS:
        return
    root = out / 'official_source' / f'epoch_{epoch:02d}' / split
    root.mkdir(parents=True, exist_ok=True)
    for arm in ('all', 'novel'):
        src = Path(official['_provenance'][arm]['path'])
        shutil.copy2(src, root / f'official_{arm}.json')


def _write_rows(out, nodes, pergt, queryrows):
    rows = []
    confusion = {}
    for node in nodes:
        for split, result in node['splits'].items():
            official = result.get('official', {})
            confusion_key = f"{node['step']}:{split}"
            for scope, metrics in result['local'].items():
                row = {'step': node['step'], 'epoch': node['epoch'], 'split': split, 'scope': scope}
                for key, value in metrics.items():
                    if isinstance(value, (str, int, float)) or value is None:
                        row[key] = value
                    elif key in ('candidate_ca', 'candidate_cw', 'panoptic_ca', 'panoptic_cw', 'candidate_ap'):
                        row.update({f'{key}_{k}': v for k, v in value.items()})
                row.update(_scope_official_metrics(official, scope))
                for arm in ('all', 'novel'):
                    for metric_key, value in official.get(arm, {}).items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    row[f'official_{arm}_{metric_key}_{subkey}'] = subvalue
                rows.append(row)
                confusion[f'{confusion_key}:{scope}'] = metrics['classification_confusion']

    keys = sorted({key for row in rows for key in row})
    with (out / 'task_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / 'task_metrics.json', rows)
    write_json(out / 'classification_confusion.json', confusion)
    for filename, values in (('per_gt_candidate_and_panoptic.csv', pergt),
                             ('candidate_panoptic_queries.csv', queryrows)):
        fields = sorted({key for value in values for key in value})
        with (out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)


def _write_acceptance(out, nodes):
    by_step = {node['step']: node for node in nodes}
    endpoint = by_step.get(3584, {}).get('splits', {}).get('train_all56')
    start = by_step.get(0, {}).get('splits', {}).get('train_all56')
    if not endpoint or not start:
        payload = {'status': 'MISSING_CHECKPOINT', 'passed': None,
                   'missing_steps': [s for s in (0, 3584) if s not in by_step]}
        write_json(out / 'train_set_acceptance.json', payload)
        return payload
    ec = endpoint['local']['context']
    en0, en64 = start['local']['novel'], endpoint['local']['novel']
    official_ap50 = endpoint.get('official', {}).get('all', {}).get('context_map', {}).get('map_50')
    checks = {
        'raw_best_mask_iou50_fraction_ge_0_80': ec['raw_best_iou_ge_0_5_fraction'] >= .80,
        'matched_19_class_accuracy_ge_0_85': ec['matched_19_class_accuracy'] >= .85,
        'candidate_ca_recall_ge_0_70': ec['candidate_ca']['recall'] >= .70,
        'candidate_cw_recall_ge_0_60': ec['candidate_cw']['recall'] >= .60,
        'candidate_cw_precision_ge_0_60': ec['candidate_cw']['precision'] >= .60,
        'candidate_ap50_ge_0_60': (ec.get('candidate_ap', {}).get('map_50') or 0.) >= .60,
        'panoptic_cw_recall_ge_0_50': ec['panoptic_cw']['recall'] >= .50,
        'official_ap50_ge_0_40': (official_ap50 or 0.) >= .40,
        'context_psnr_drop_le_0_5db': start['local']['context']['psnr'] - ec['psnr'] <= .5,
        'true_novel_psnr_drop_le_0_5db': en0['psnr'] - en64['psnr'] <= .5,
    }
    payload = {'passed': all(checks.values()), 'checks': checks,
               'measured': {'train_all56_context': ec,
                            'train_all56_true_novel_psnr_step0': en0['psnr'],
                            'train_all56_true_novel_psnr_step3584': en64['psnr'],
                            'official_context_ap50': official_ap50},
               'corrected_candidate_calculation': True}
    write_json(out / 'train_set_acceptance.json', payload)
    return payload


def _write_report(out, nodes, missing, acceptance):
    lines = [
        '# Object-Locus V3-Set evaluator fix：checkpoint 重评', '',
        '只执行了 eval/no-grad；未 backward、未创建 optimizer、未更新参数。模型预测与官方 evaluator 未变；candidate 统计按修正后的 prediction-only confidence 与 valid-domain IoU 重算。',
        '', f"训练集验收（train_all56 context，原注册阈值）：**{'PASS' if acceptance.get('passed') else 'FAIL'}**。", '',
        '| step | split | scope | candidate数 | CA TP/FP/FN | CW TP/FP/FN | candidate P/R | candidate mAP/AP50 | official packed mIoU/PQ/mAP/AP50 | 官方JSON来源 |',
        '|---:|---|---|---:|---:|---:|---:|---:|---|---|'
    ]
    for node in nodes:
        for split, result in node['splits'].items():
            for scope, m in result['local'].items():
                ca, cw, ap = m['candidate_ca'], m['candidate_cw'], m['candidate_ap']
                off = _scope_official_metrics(result.get('official', {}), scope)
                offvals = '/'.join(str(off[k]) for k in ('scope_official_miou','scope_official_pq','scope_official_map','scope_official_ap50'))
                apv = f"{ap.get('map','MISSING')}/{ap.get('map_50','MISSING')}"
                lines.append(f"| {node['step']} | {split} | {scope} | {m['candidate_count']} | {ca['tp']}/{ca['fp']}/{ca['fn']} | {cw['tp']}/{cw['fp']}/{cw['fn']} | {cw['precision']:.4f}/{cw['recall']:.4f} | {apv} | {offvals} | {off['scope_official_source']} |")
    lines += ['', '## 缺失节点', '', *(missing or ['无']), '',
              '官方指标严格按 scope 映射：context 使用 all/context；target_all 使用 all/target；novel 使用 novel/target。epoch8/16 未注册官方评估，故显示 MISSING，不以其他scope填充。',
              '官方结果直接复用原注册 JSON；原始文件副本位于 `official_source/`，来源路径与 SHA256 写入 `official_provenance.json`。']
    (out / 'analysis_report.md').write_text('\n'.join(lines) + '\n')


def _attach_old_comparison(out, nodes):
    old_path = SOURCE_REPORTS / 'curves_step_3584.json'
    if not old_path.is_file():
        write_json(out / 'epoch64_candidate_before_after.json', {'status': 'MISSING_OLD_CURVE'})
        return
    old = json.loads(old_path.read_text())
    new64 = next((n for n in nodes if n['step'] == 3584), None)
    comparisons = []
    if new64:
        for split, result in new64['splits'].items():
            if split not in old.get('splits', {}):
                continue
            for scope, newm in result['local'].items():
                oldm = old['splits'][split]['local'][scope]
                comparisons.append({'split': split, 'scope': scope,
                    'before': {k: oldm.get(k) for k in ('candidate_count','candidate_ca','candidate_cw','candidate_ap')},
                    'after': {k: newm.get(k) for k in ('candidate_count','candidate_ca','candidate_cw','candidate_ap')}})
    write_json(out / 'epoch64_candidate_before_after.json', {'source_curve': str(old_path), 'comparisons': comparisons,
        'unchanged_by_evaluator_fix': ['model predictions','panoptic output','official metrics','PSNR']})


def run_smoke(device='cuda'):
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires CUDA')
    reports = SOURCE_REPORTS
    manifest, _ = load_manifest()
    checkpoint = SOURCE_RUN / 'checkpoint_epoch_64.pt'
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model, opt, _ = build_model('cuda')
    model.load_state_dict(blob['model'], strict=True)
    model.eval()
    split = manifest['train_probe8'][:1]
    smoke_dir = OUTPUT_DEFAULT / 'smoke'
    result, pergt, queries = evaluate_windows(model, opt, split, 3584, 'train_probe8_smoke', smoke_dir,
                                               'cuda', build_batch, official=False, panels=False)
    metrics = result['local']
    if set(metrics) != {'context','target_all','novel'}:
        raise RuntimeError('smoke omitted an evaluation scope')
    for scope, row in metrics.items():
        for key in ('semantic_miou','psnr','panoptic_pq'):
            if not torch.isfinite(torch.tensor(float(row[key]))):
                raise RuntimeError(f'nonfinite {scope}.{key}')
    if len(queries) != 300:
        raise RuntimeError(f'expected 300 query diagnostic rows for three scopes, got {len(queries)}')
    if any(not {'prediction_raw_area','evaluation_raw_area','prediction_candidate_area',
                'evaluation_candidate_area','joint_eligible'}.issubset(row) for row in queries):
        raise RuntimeError('smoke query rows do not match the corrected prediction/evaluation schema')
    torch.cuda.synchronize()
    return {'status':'PASS','optimizer_updates':0,'backward_calls':0,'checkpoint':str(checkpoint),
            'checkpoint_sha256':sha256(checkpoint),'node':__import__('os').uname().nodename,
            'gpu':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'scene':split[0]['scene'],
            'window':split[0],'scope_metrics':{k:{'candidate_count':v['candidate_count'],
            'candidate_ca':v['candidate_ca'],'candidate_cw':v['candidate_cw'],'candidate_ap':v['candidate_ap']}
            for k,v in metrics.items()},'per_gt_rows':len(pergt),'query_rows':len(queries)}


def run_all(device='cuda', output=OUTPUT_DEFAULT):
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('checkpoint reevaluation requires CUDA')
    output.mkdir(parents=True, exist_ok=True)
    manifest, manifest_meta = load_manifest()
    windows_by_name = {name: manifest[name] for name in ('train_probe8','same_scene_holdout8','dev8','train_all56','val32')}
    opt = build_options()
    model, _, _ = build_model('cuda', opt=opt)
    nodes, all_pergt, all_queries, missing, official_provenance = [], [], [], [], {}
    checkpoint_manifest = {}
    for epoch, (step, split_names) in REGISTERED.items():
        checkpoint = SOURCE_RUN / f'checkpoint_epoch_{epoch:02d}.pt'
        if not checkpoint.is_file():
            missing.append(f'epoch{epoch}/step{step}: MISSING_CHECKPOINT {checkpoint}')
            checkpoint_manifest[str(epoch)] = {'status':'MISSING_CHECKPOINT','path':str(checkpoint)}
            continue
        blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if blob.get('architecture_name') != 'LOCUSGS_OBJECT_LOCUS_V3_SET' or int(blob.get('stage_step',-1)) != step:
            raise RuntimeError(f'checkpoint provenance mismatch at epoch {epoch}: architecture/step')
        model.load_state_dict(blob['model'], strict=True)
        model.eval()
        node = {'step':step,'epoch':epoch,'global_optimizer_step':step,'splits':{}}
        checkpoint_manifest[str(epoch)] = {'status':'LOADED_STRICT','path':str(checkpoint),
                                           'sha256':sha256(checkpoint),'git_sha':blob.get('git_sha'),
                                           'architecture_name':blob.get('architecture_name'),'step':blob.get('stage_step')}
        for split in split_names:
            result, pergt, queries = evaluate_windows(model, opt, windows_by_name[split], step, split,
                output, 'cuda', build_batch, official=False, panels=True)
            official, provenance = _registered_official(epoch, step, split)
            if official:
                result['official'] = official
                _copy_registered_official(official, epoch, split, output)
            elif epoch in OFFICIAL_EPOCHS:
                missing.append(f'epoch{epoch}/step{step}/{split}: {provenance}')
            official_provenance[f'{step}:{split}'] = provenance
            node['splits'][split] = result
            all_pergt.extend(pergt); all_queries.extend(queries)
        nodes.append(node)
        del blob
        torch.cuda.empty_cache()

    write_json(output/'source_manifest.json', {'original_manifest':manifest_meta,'source_manifest_sha256':manifest.get('source_sha256')})
    write_json(output/'checkpoint_provenance.json', checkpoint_manifest)
    write_json(output/'official_provenance.json', official_provenance)
    _write_rows(output, nodes, all_pergt, all_queries)
    acceptance = _write_acceptance(output, nodes)
    _attach_old_comparison(output, nodes)
    _write_report(output, nodes, missing, acceptance)
    summary = {'optimizer_updates':0,'backward_calls':0,'registered_epochs':list(REGISTERED),
               'completed_epochs':[n['epoch'] for n in nodes],'missing':missing,
               'source_manifest':manifest_meta,'source_run':str(SOURCE_RUN)}
    write_json(output/'reevaluation_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke','reevaluate'), required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    if args.phase == 'smoke':
        result = run_smoke(args.device)
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'evaluator_smoke.json', result)
    else:
        result = run_all(args.device, args.output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
