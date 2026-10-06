#!/usr/bin/env python3
"""Execute only the fixed recent-experiment cleanup list with an audit trail."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import argparse
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)

MG_REPORT = Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1')
MG_RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_mask_guided_v1')
FZ_REPORT = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
FZ_RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
TEXT_REPORT = Path('/space/mawb/ssst/group_plus/object_locus_text_refer_v1')
TEXT_RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_text_refer_v1_train')
NEW_REPORT = Path('/space/mawb/ssst/group_plus/object_locus_mh_feedback_v1')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_bitwise_equal(left, right):
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(a, b)


def disk_free():
    return shutil.disk_usage('/space').free


def original_meta(checkpoint):
    x = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    keys = ('epoch', 'completed_updates', 'exposures', 'global_seed', 'object_seed',
            'code_sha', 'plan_sha256', 'data_plan', 'data_manifest', 'config',
            'source_hashes', 'weights_provenance', 'rank_rng')
    return x, {k: x[k] for k in keys if k in x}


def model_only(source, target, model, provenance):
    source_sha = sha256(source)
    old, metadata = original_meta(source)
    source_state = old['model']
    expected = {k: (tuple(v.shape), v.dtype) for k, v in source_state.items()}
    # Snapshot each tensor into independent storage. Some inherited checkpoints
    # contain tied/view storage; independent contiguous copies avoid storage alias
    # rewrites while retaining each state_dict tensor's exact bytes.
    state = {k: v.detach().cpu().contiguous().clone() for k, v in source_state.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    source_s = str(source)
    if 'object_locus_mask_guided_v1' in source_s:
        experiment_report = MG_REPORT / ('control' if '/control/' in source_s else 'mask_guided')
    else:
        experiment_report = FZ_REPORT
    config = {}
    for name in ('run_manifest.json','optimizer_groups.json','weights_mapping.json','data_manifest.json','training_plan.json'):
        config_path = experiment_report/name
        if config_path.is_file():
            try: config[name] = json.loads(config_path.read_text())
            except (ValueError, OSError): config[name] = {'source_path':str(config_path),'sha256':sha256(config_path)}
    envelope = dict(model=state, metadata=dict(metadata,
        source_checkpoint=str(source), source_checkpoint_sha256=source_sha,
        pretrained_sources=provenance, experiment_config=config,
        optimizer_recipe={'type':'AdamW','betas':[0.9,0.95],'eps':1e-8,'precision':'FP32','global_grad_clip':1.0,
            'peak_lr':{'reconstruction':1e-6,'understanding':1e-5,'new':1e-4},'warmup_exposures':200,
            'warmup_updates':25,'weight_decay_matrix':0.05,'weight_decay_bias_norm_embedding':0.0},
        model_only=True))
    torch.save(envelope, temp)
    saved = torch.load(temp, map_location='cpu', mmap=True, weights_only=False)
    got = saved['model']
    if set(got) != set(state):
        raise RuntimeError(f'model-only keys differ for {source}')
    for key, value in state.items():
        other = got[key]
        if tuple(other.shape) != expected[key][0] or other.dtype != expected[key][1] or not tensor_bitwise_equal(value, other):
            raise RuntimeError(f'model-only tensor mismatch: {source}: {key}')
    model.load_state_dict(got, strict=True)
    if saved['metadata']['source_checkpoint_sha256'] != source_sha:
        raise RuntimeError(f'model-only provenance SHA mismatch: {source}')
    os.replace(temp, target)
    target_sha = sha256(target)
    del old, saved, source_state, state
    return dict(path=str(target), sha256=target_sha, source=str(source), source_sha256=source_sha,
                tensor_count=len(expected), strict_load=True, values_exact=True)


def iter_files(root, extensions, *, exclude_qualitative=True):
    if not root.exists():
        return
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not (exclude_qualitative and d.lower() == 'qualitative')]
        for name in files:
            path = Path(current) / name
            if path.suffix.lower() in extensions:
                yield path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--finalize-model-only',action='store_true')
    if parser.parse_args().finalize_model_only:
        finalize_model_only()
        return
    NEW_REPORT.mkdir(parents=True, exist_ok=True)
    initial_free = disk_free()
    removed = []
    retained = []
    models = {}

    # Existing C epoch0 is checked against the original fresh builder; mask-guided
    # endpoints are loaded by their corresponding arm class before deletion.
    from scripts.object_locus_mask_guided_runtime import build_model as build_mg
    c_model, _ = build_mg('control', 'cpu', report=False)
    c_provenance = json.loads((MG_REPORT / 'control' / 'initialization_contract.json').read_text())['weights']
    c_specs = [('control_epoch_00', MG_RUN/'control/checkpoint_epoch_00.pt', c_model),
               ('control_epoch_64', MG_RUN/'control/checkpoint_epoch_64.pt', c_model)]
    c_saved = {}
    for label, source, model in c_specs:
        target = MG_RUN/'model_only'/f'{label}.pt'
        models[label] = model_only(source, target, model, c_provenance)
        c_saved[label] = target
    del c_model

    m_model, _ = build_mg('mask_guided', 'cpu', report=False)
    m_provenance = json.loads((MG_REPORT / 'mask_guided' / 'initialization_contract.json').read_text())['weights']
    m_source = MG_RUN/'mask_guided/checkpoint_epoch_64.pt'
    m_target = MG_RUN/'model_only/mask_guided_epoch_64.pt'
    models['mask_guided_epoch_64'] = model_only(m_source, m_target, m_model, m_provenance)
    del m_model

    selection = json.loads((FZ_REPORT/'evaluation/dev8_selection.json').read_text())
    if selection.get('selected_epoch') != 8:
        raise RuntimeError('frozen experiment selection no longer identifies epoch 8')
    f_source = FZ_RUN/'checkpoint_epoch_08.pt'
    if sha256(f_source) != selection['selected_checkpoint_sha256']:
        raise RuntimeError('selected frozen epoch-8 source SHA differs from selection record')
    f_report = json.loads((FZ_REPORT/'run_manifest.json').read_text())
    provenance = f_report.get('weights_provenance', f_report.get('pretrained_sources', {}))
    # Same architecture/state keys; all encoder tensors are still present in the model-only file.
    f_model, _ = __import__('scripts.object_locus_panoptic_v1_runtime', fromlist=['build_model']).build_model('cpu', report=False)
    f_target = FZ_RUN/'model_only/selected_epoch_08.pt'
    models['frozen_selected_epoch_08'] = model_only(f_source, f_target, f_model, provenance)
    del f_model

    # Validate every aggregate and per-object detail before removing any image predictions.
    for arm in ('control','mask_guided'):
        for split in ('train_all56','same_scene_holdout8','dev8','val32'):
            p = MG_REPORT/arm/f'per_gt_epoch64_{split}.json'
            rows = json.loads(p.read_text())
            if not isinstance(rows, list) or not rows:
                raise RuntimeError(f'paired per-GT detail is unreadable: {p}')
    json.loads((MG_REPORT/'paired_endpoint_comparison.json').read_text())
    json.loads((MG_REPORT/'paired_scene_bootstrap.json').read_text())
    json.loads((FZ_REPORT/'evaluation/dev8_selection.json').read_text())
    json.loads((FZ_REPORT/'evaluation_accelerated/frozen_excluding_dev8_official_segmentation.json').read_text())
    for p in (FZ_REPORT/'evaluation_accelerated/frozen_vs_unfrozen_full_and_excluded.json',
              FZ_REPORT/'evaluation_accelerated/evaluation_coverage.json'):
        json.loads(p.read_text())

    # Replaced checkpoint files are deleted only after save, exact tensor checks and strict load.
    mg_keep = {c_saved['control_epoch_00'], c_saved['control_epoch_64'], m_target}
    mg_delete = [p for p in MG_RUN.glob('control/checkpoint_epoch_*.pt') if p.name != 'checkpoint_epoch_00.pt' and p.name != 'checkpoint_epoch_64.pt']
    mg_delete += [MG_RUN/'control/checkpoint_epoch_00.pt', MG_RUN/'control/checkpoint_epoch_64.pt']
    mg_delete += [p for p in MG_RUN.glob('mask_guided/checkpoint_epoch_*.pt') if p.name != 'checkpoint_epoch_64.pt']
    mg_delete += [MG_RUN/'mask_guided/checkpoint_epoch_64.pt']
    mg_alternative = str(MG_REPORT/'paired_endpoint_comparison.json')
    for path in mg_delete:
        if path.exists():
            replacement = next((v for v in models.values() if v['source'] == str(path)), None)
            alt_path = replacement['path'] if replacement else mg_alternative
            alt_sha = replacement['sha256'] if replacement else sha256(alt_path)
            removed.append(delete_file(path, 'full training checkpoint (optimizer removed)', alt_path, alt_sha))
    for arm in ('control','mask_guided'):
        pass
    for folder in (MG_RUN/'control', MG_RUN/'mask_guided'):
        for path in folder.glob('checkpoint_epoch_*.pt'):
            if path not in mg_keep and path.exists():
                removed.append(delete_file(path,'non-endpoint training checkpoint','registered endpoint model-only checkpoints',sha256(MG_RUN/'model_only'/'control_epoch_64.pt')))

    # Frozen encoder experiment: retain only selected epoch 8 as a model-only checkpoint.
    for path in FZ_RUN.glob('checkpoint_epoch_*.pt'):
        if path.name != 'checkpoint_epoch_08.pt':
            removed.append(delete_file(path,'non-selected training checkpoint','selected epoch-08 model-only checkpoint',models['frozen_selected_epoch_08']['sha256']))
    removed.append(delete_file(f_source,'selected full checkpoint including optimizer state',f_target,models['frozen_selected_epoch_08']['sha256']))

    # Completed text smoke left one temporary head file; retain the formal head checkpoints and reports.
    text_temp = TEXT_REPORT/'temporary_two_update_head.pt'
    if text_temp.exists():
        removed.append(delete_file(text_temp,'completed text smoke temporary model','preserved formal text head checkpoints',sha256(TEXT_RUN/'head_update_12000.pt')))

    image_exts = {'.png','.jpg','.jpeg','.npy','.npz'}
    mg_roots = [MG_REPORT/'control/official', MG_REPORT/'mask_guided/official']
    frozen_roots = [FZ_REPORT/'evaluation/aggregate',FZ_REPORT/'evaluation/best',FZ_REPORT/'evaluation/selection',
        FZ_REPORT/'evaluation/smoke',FZ_REPORT/'evaluation/smoke_final',FZ_REPORT/'evaluation_accelerated/aggregate',
        FZ_REPORT/'evaluation_accelerated/recovery/val32_exports',FZ_REPORT/'evaluation_accelerated/recovery/incomplete_snapshot_full_exports',
        FZ_REPORT/'evaluation_accelerated/smoke',FZ_REPORT/'smoke_eval',FZ_REPORT/'smoke_runtime']
    metric_alt = FZ_REPORT/'evaluation_accelerated/frozen_excluding_dev8_official_segmentation.json'
    metric_sha = sha256(metric_alt)
    for root in mg_roots+frozen_roots:
        for path in iter_files(root,image_exts):
            if path.exists():
                removed.append(delete_file(path,'bulk official prediction or completed smoke image/array export',str(metric_alt),metric_sha))

    final_free = disk_free()
    manifest = dict(text_job_status='COMPLETED; no cancellation',text_job_id='58689',
        text_formal_training_preserved=True, scope=[str(MG_REPORT),str(MG_RUN),str(FZ_REPORT),str(FZ_RUN),str(TEXT_REPORT),str(TEXT_RUN)],
        protected={'unfrozen_Full1201_epoch6': 'untouched', 'reconstruction_step47500':'untouched',
            'shared_weights':'untouched','datasets':'untouched','other_experiments':'untouched','new_experiment':'untouched'},
        disk={'free_before_bytes':initial_free,'free_after_bytes':final_free,'freed_bytes':final_free-initial_free},
        model_only=models, removed=removed, retained=retained)
    (NEW_REPORT/'cleanup_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (NEW_REPORT/'cleanup_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=('deleted_path','type','bytes','retained_alternative','alternative_sha256'))
        writer.writeheader()
        for row in removed:
            writer.writerow(row)
    print(json.dumps({'disk':manifest['disk'],'model_only':models,'deleted_count':len(removed)},indent=2))


def finalize_model_only():
    manifest_path=NEW_REPORT/'cleanup_manifest.json'
    if not manifest_path.is_file():raise FileNotFoundError(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    # The frozen task also held unfrozen-epoch depth render exports. They are
    # batch prediction products; keep the per-image depth CSVs and aggregates.
    depth_metric=FZ_REPORT/'evaluation_accelerated/frozen_vs_unfrozen_full_and_excluded.json'
    if depth_metric.is_file():
        for path in iter_files(FZ_REPORT/'evaluation/unfrozen_depth',{'.png','.jpg','.jpeg','.npy','.npz'}):
            if path.exists():
                manifest['removed'].append(delete_file(path,'bulk depth prediction export',str(depth_metric),sha256(depth_metric)))
    reports={
        'control_epoch_00':MG_REPORT/'control',
        'control_epoch_64':MG_REPORT/'control',
        'mask_guided_epoch_64':MG_REPORT/'mask_guided',
        'frozen_selected_epoch_08':FZ_REPORT,
    }
    recipe={'type':'AdamW','betas':[0.9,0.95],'eps':1e-8,'precision':'FP32','global_grad_clip':1.0,
        'peak_lr':{'reconstruction':1e-6,'understanding':1e-5,'new':1e-4},
        'warmup_exposures':200,'warmup_updates':25,'weight_decay_matrix':0.05,
        'weight_decay_bias_norm_embedding':0.0}
    for label,row in manifest['model_only'].items():
        path=Path(row['path'])
        blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        source_state=blob['model']
        model={k:v.detach().cpu().contiguous().clone() for k,v in source_state.items()}
        report=reports[label]
        config={}
        for name in ('run_manifest.json','optimizer_groups.json','weights_mapping.json','data_manifest.json','training_plan.json'):
            p=report/name
            if p.is_file():
                try:config[name]=json.loads(p.read_text())
                except (OSError,ValueError):config[name]={'path':str(p),'sha256':sha256(p)}
        metadata=dict(blob['metadata'],experiment_config=config,optimizer_recipe=recipe,
            finalization='config/provenance metadata attached after exact model-only conversion')
        tmp=path.with_suffix('.finalize.tmp')
        torch.save(dict(model=model,metadata=metadata),tmp)
        check=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False)
        if set(check['model'])!=set(model) or any(not tensor_bitwise_equal(model[k],check['model'][k]) for k in model):
            raise RuntimeError(f'final model-only checkpoint tensor mismatch: {label}')
        os.replace(tmp,path)
        row['sha256']=sha256(path)
        row['metadata_finalized']=True
        del blob,source_state,model,check
    manifest['disk']['free_after_bytes']=disk_free()
    manifest['disk']['freed_bytes']=manifest['disk']['free_after_bytes']-manifest['disk']['free_before_bytes']
    manifest['checks']={
        'text_job_status_checked_once':'58689 COMPLETED; no cancellation submitted',
        'text_formal_head_checkpoints':'preserved under workspace_group_plus/object_locus_text_refer_v1_train',
        'text_smoke_temporary_head':'temporary_two_update_head.pt deleted',
        'text_optimizer_or_batch_cache':'no additional smoke optimizer/cache batch files found in specified text task directories',
        'mask_guided_endpoint_metrics_and_per_gt':'all four split per_gt files in both arms and paired JSON summaries parsed before export deletion',
        'frozen_selection_and_metrics':'dev8 epoch-8 selection plus aggregate segmentation and coverage JSON parsed before export deletion',
        'mask_guided_analysis_zips':'preserved',
        'qualitative_directories':'preserved; deletion walker pruned directories named qualitative',
        'cleanup_scope':'only the named Object-Locus Mask-Guided V1, frozen-encoder Full1201, and text smoke products',
        'protected_assets':'unfrozen Full1201 epoch6, reconstruction step47500, shared pretrained weights, datasets, source and all other experiment directories untouched'}
    mg_alt=MG_REPORT/'paired_endpoint_comparison.json'
    for item in manifest['removed']:
        if item['deleted_path'].startswith(str(MG_REPORT/'control/official')) or item['deleted_path'].startswith(str(MG_REPORT/'mask_guided/official')):
            item['retained_alternative']=str(mg_alt)
            item['alternative_sha256']=sha256(mg_alt)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    csv_path=NEW_REPORT/'cleanup_manifest.csv'
    if csv_path.is_file():
        rows=list(csv.DictReader(csv_path.open()))
        existing={row['deleted_path'] for row in rows}
        for item in manifest['removed']:
            if item['deleted_path'] not in existing:
                rows.append(dict(deleted_path=item['deleted_path'],type=item['type'],bytes=item['bytes'],
                    retained_alternative=item['retained_alternative'],alternative_sha256=item['alternative_sha256']))
        current={str(Path(x['path'])):x['sha256'] for x in manifest['model_only'].values()}
        for row in rows:
            if row['retained_alternative'] in current:
                row['alternative_sha256']=current[row['retained_alternative']]
        with csv_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=('deleted_path','type','bytes','retained_alternative','alternative_sha256'))
            writer.writeheader();writer.writerows(rows)
    print(json.dumps({'model_only_finalized':manifest['model_only'],'disk':manifest['disk']},indent=2))


def delete_file(path, kind, alternative, alternative_sha):
    path = Path(path)
    size = path.stat().st_size
    path.unlink()
    return dict(deleted_path=str(path), type=kind, bytes=size,
        retained_alternative=str(alternative), alternative_sha256=str(alternative_sha))


if __name__ == '__main__':
    main()
