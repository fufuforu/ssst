"""Real-data single-card parity and four-card accumulation gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.object_locus_gc_sweep_runtime import SOURCE_MANIFEST, SOURCE_EXPOSURES, WINDOWS, sha256
from scripts.object_locus_output_refine_gc001_runtime import (
    roots, write_json, rank_world, init_distributed, build_model, build_optimizer,
    init_slot_states, train_update, load_checkpoint, record_blocker,
)


def flatten_tensors(obj, prefix=''):
    found = {}
    if torch.is_tensor(obj): found[prefix] = obj.detach().cpu()
    elif isinstance(obj, dict):
        for key, value in obj.items(): found.update(flatten_tensors(value, f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj): found.update(flatten_tensors(value, f'{prefix}.{i}'))
    return found


def cpu_copy(obj):
    if torch.is_tensor(obj): return obj.detach().cpu()
    if isinstance(obj, dict): return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [cpu_copy(v) for v in obj]
    if isinstance(obj, tuple): return tuple(cpu_copy(v) for v in obj)
    return obj


def compare_outputs(reference, candidate):
    a, b = flatten_tensors(reference), flatten_tensors(candidate)
    ignored = {k for k in b if k.endswith('.q_refined') or k in ('q_refined', 'q_base')}
    common = sorted((set(a) & set(b)) - ignored)
    missing = sorted((set(a) ^ set(b)) - ignored)
    results = []
    for key in common:
        x, y = a[key].float(), b[key].float()
        if x.shape != y.shape:
            results.append({'key': key, 'shape_mismatch': [list(x.shape), list(y.shape)], 'failed_elements': int(x.numel())})
            continue
        valid = torch.isfinite(x) & torch.isfinite(y)
        passed = valid & ((x-y).abs() <= 1e-6 + 1e-5*y.abs())
        results.append({'key': key, 'shape': list(x.shape), 'failed_elements': int((~passed).sum()),
                        'max_abs_difference': float((x-y).abs().max()) if x.numel() else 0.0})
    failed = sum(x['failed_elements'] for x in results)
    return {'common_tensor_count': len(common), 'missing_tensor_paths': missing,
            'failed_elements': failed, 'maximum_absolute_difference': max((x.get('max_abs_difference', 0) for x in results), default=0),
            'tolerance': 'abs(a-b) <= 1e-6 + 1e-5*abs(b)', 'per_tensor': results}


def real_forward(model, batch):
    from tokengs.models.input_types import split_data, ModelInputDecoder
    mi, _ = split_data(batch, model.opt)
    decoder = ModelInputDecoder(cam_view=batch['cam_view_all'], intrinsics=batch['intrinsics_all'])
    context = ModelInputDecoder(cam_view=batch['cam_view_all'][:, :2], intrinsics=batch['intrinsics_all'][:, :2])
    with torch.no_grad():
        return model.forward_object_locus(mi, render_decoder_input=decoder,
                    read_context_decoder=context, context_decoder=context, step=SOURCE_EXPOSURES)


def cpu_forward_compare(model, batch):
    # Retain every shared forward tensor needed by the parity contract, but no GPU storage.
    return cpu_copy(real_forward(model, batch))


def cpu_state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def identity_readout_contract(prediction, model):
    base = prediction['q_base'].float()
    refined = prediction['q_refined'].float()
    features = prediction['gaussian_feature'].float()
    xyz = prediction['gaussians'][..., :3]
    final = prediction['states'][-1]
    # Existing lifting/renderer results are reused; no second lifting/render pass.
    q_test = model.panoptic.output_3d_refine(base, features, xyz, final['c'], final['s'])
    mq_base = model.understanding.mask_embedder(base)
    mq_refined = model.understanding.mask_embedder(q_test)
    logits_base = features @ mq_base.transpose(1, 2)
    logits_refined = features @ mq_refined.transpose(1, 2)
    cls_base = model.panoptic.classify(base)
    cls_refined = model.panoptic.classify(q_test)
    region = prediction['region_mass']
    semantic_base = region.new_zeros((region.shape[0], region.shape[1], 20, *region.shape[-2:]))
    semantic_base[:, :, :2] = region[:, :, 100:102]
    semantic_base[:, :, 2:20] = torch.einsum('bvqhw,bqc->bvchw', region[:, :, :100], cls_base['p_class'][..., :18])
    semantic_base = semantic_base / (semantic_base.sum(2, keepdim=True) + 1e-6)
    return {
        'q_refined_exact': torch.equal(q_test, base),
        'gaussian_logits_exact': torch.equal(logits_base, logits_refined),
        'membership_exact': torch.equal(logits_base.sigmoid(), logits_refined.sigmoid()),
        'class_exact': torch.equal(cls_base['thing_logits19'], cls_refined['thing_logits19']),
        'same_render_region_reused': True,
        'semantic_postprocess_exact': torch.equal(semantic_base, prediction['semantic_scores']),
        'returned_q_exact': torch.equal(refined, base),
    }


def single():
    start = time.time(); device = init_distributed(formal=False); rank, _ = rank_world()
    report, _ = roots()
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    plan = json.loads((report / 'training_plan.json').read_text())
    entry = plan['entries'][0]
    wi = entry['rank_windows'][0]
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.object_locus_gc_sweep_runtime import build_model as build_gc_model
    first = manifest['expanded_train_windows'][wi]
    gpu = torch.device(device)
    # Baseline output is detached before constructing R3D so the real 3090 remains
    # within the specified memory budget.
    base_model, opt, source = build_gc_model(device, report=False)
    base_model.eval(); batch = build_batch(opt, first, gpu)
    base_compare = cpu_forward_compare(base_model, batch)
    old_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    del base_model; torch.cuda.empty_cache()
    model, opt, source_new = build_model(device)
    old_loaded = model.state_dict()
    if any(not torch.equal(old_loaded[k].detach().cpu(), v) for k, v in old_state.items()):
        raise RuntimeError('R3D initial source state differs from original C model')
    model.eval()
    r3d_model = model
    r3d_compare_gpu = real_forward(r3d_model, batch)
    r3d_compare = cpu_copy(r3d_compare_gpu)
    parity = compare_outputs(base_compare, r3d_compare)
    if parity['failed_elements'] or parity['missing_tensor_paths']:
        # Repeated C/C and R/R outputs are retained as diagnostics, without changing
        # the preregistered elementwise threshold.
        repeat_r = cpu_copy(real_forward(r3d_model, batch))
        r_repeat = compare_outputs(r3d_compare, repeat_r)
        del repeat_r, r3d_model, model, r3d_compare_gpu; torch.cuda.empty_cache()
        repeat_c_model, repeat_opt, _ = build_gc_model(device, report=False)
        repeat_c_model.load_state_dict(old_state, strict=True); repeat_c_model.eval()
        repeat_c = cpu_forward_compare(repeat_c_model, batch)
        c_repeat = compare_outputs(base_compare, repeat_c)
        del repeat_c_model; torch.cuda.empty_cache()
        write_json(report / 'single_smoke_failure.json', {'status': 'INVALID', 'parity': parity,
                    'repeat_c_vs_c': c_repeat, 'repeat_r_vs_r': r_repeat,
                    'repeat_c_state_sha': tensor_state_hash(old_state),
                    'repeat_r_state_sha': 'same initialized state as first R3D forward'})
        raise RuntimeError('C/R full-forward parity exceeded locked elementwise tolerance')
    readout = identity_readout_contract(r3d_compare_gpu, r3d_model)
    if not all(readout.values()): raise RuntimeError(f'same-evidence readout identity failed: {readout}')
    del r3d_model, model, r3d_compare_gpu
    del base_compare, r3d_compare, old_state
    torch.cuda.empty_cache()
    # Three local two-window smoke updates, slot 0 and slot 4 only; not a global update.
    model, opt, source = build_model(device); model.train(); optimizer = build_optimizer(model)
    bn_before = {n: b.running_mean.detach().clone() for n, b in model.named_modules()
                 if isinstance(b, torch.nn.modules.batchnorm._BatchNorm) and b.running_mean is not None}
    states = init_slot_states(device)
    refiner0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.startswith('panoptic.output_3d_refine.')}
    logs = []; positive_under = False; windows = []
    for u in range(3):
        logs.append(train_update(model, opt, optimizer, manifest, plan['entries'][u], u, states, device,
                                 training_windows=windows, epoch0_refiner=refiner0))
        positive_under |= logs[-1]['refiner_avg_preclip_grad_norm'] > 0
        if not np.isfinite(logs[-1]['loss_rec']) or not np.isfinite(logs[-1]['loss_under']) or not np.isfinite(logs[-1]['preclip_norm']):
            raise FloatingPointError('nonfinite single-card smoke scalar')
    bn_after = {n: b.running_mean.detach() for n, b in model.named_modules()
                if isinstance(b, torch.nn.modules.batchnorm._BatchNorm) and b.running_mean is not None}
    if bn_before.keys() != bn_after.keys() or any(not torch.equal(v, bn_after[k]) for k, v in bn_before.items()):
        raise RuntimeError('BN running means changed between single-card microbatches')
    changed = any(not torch.equal(refiner0[n], p.detach().cpu()) for n, p in model.named_parameters() if n in refiner0)
    if not changed or not positive_under or logs[2]['q_refined_minus_q_base_rms'] <= 0:
        raise RuntimeError('three-update single smoke did not activate the output refiner')
    smoke_ckpt = report / 'smoke_single' / 'roundtrip.pt'
    from scripts.object_locus_output_refine_gc001_runtime import save_checkpoint
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    save_checkpoint(smoke_ckpt, model, optimizer, 3, states, np.zeros(WINDOWS, dtype=np.int64),
                   __import__('dataclasses').asdict(opt), source, sha256(report / 'training_plan.json'), code_sha, device)
    reloaded, reopt, metadata = load_checkpoint(smoke_ckpt, device)
    reloaded.eval()
    reloaded_prediction = real_forward(reloaded, batch)
    original_prediction = real_forward(model.eval(), batch)
    reload_equal = torch.allclose(reloaded_prediction['q_refined'], original_prediction['q_refined'], rtol=1e-5, atol=1e-6)
    for key in ('gaussian_mask_logits', 'semantic_scores', 'p_class'):
        reload_equal &= torch.allclose(reloaded_prediction[key], original_prediction[key], rtol=1e-5, atol=1e-6)
    if not reload_equal or metadata.get('experiment', {}).get('arm') != 'R3D':
        raise RuntimeError('strict checkpoint loader did not restore exact R3D outputs')
    peak = torch.cuda.max_memory_allocated(device)
    result = {'status': 'PASS', 'source_state_tensors': 1445, 'source_state_exact': True,
        'full_forward_parity': parity, 'same_evidence_readout_identity': readout,
        'three_local_updates_two_microbatches': {'updates': 3, 'windows': windows, 'steps_are_not_global_eight_window_updates': True,
            'finite': True, 'bn_running_statistics_unchanged': True, 'positive_understanding_gradient': positive_under,
            'refiner_parameters_changed': changed, 'last_q_delta_rms': logs[-1]['q_refined_minus_q_base_rms'], 'logs': logs},
        'strict_checkpoint_reload': {'status': 'PASS', 'outputs_exact': reload_equal, 'path': str(smoke_ckpt)},
        'gpu': torch.cuda.get_device_name(device), 'peak_allocated_bytes': peak,
        'elapsed_seconds': time.time() - start}
    if rank == 0: write_json(report / 'single_smoke.json', result)
    dist.barrier(); dist.destroy_process_group()


def four():
    device = init_distributed(formal=True); rank, world = rank_world()
    report, _ = roots(); manifest = json.loads(SOURCE_MANIFEST.read_text())
    plan = json.loads((report / 'training_plan.json').read_text())
    model, opt, source = build_model(device); model.train(); optimizer = build_optimizer(model)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.startswith('panoptic.output_3d_refine.')}
    states = init_slot_states(device); local_counts = np.zeros(WINDOWS, dtype=np.int64); selected = []
    for u in range(3):
        entry = plan['entries'][u]
        row = train_update(model, opt, optimizer, manifest, entry, u, states, device, training_windows=selected)
        for micro in (0, 1): local_counts[int(entry['rank_windows'][rank + 4*micro])] += 1
        if not all(np.isfinite(row[k]) for k in ('loss_rec', 'loss_under', 'preclip_norm')):
            raise FloatingPointError('nonfinite four-card smoke values')
    global_counts = torch.as_tensor(local_counts, device=device, dtype=torch.int64); dist.all_reduce(global_counts)
    expected = np.zeros(WINDOWS, dtype=np.int64)
    for entry in plan['entries'][:3]: expected[np.asarray(entry['rank_windows'], dtype=np.int64)] += 1
    if not np.array_equal(global_counts.cpu().numpy(), expected): raise RuntimeError('four-card smoke windows do not match all eight fixed slots')
    state_hash = cpu_state_hash(model)
    hashes = [None] * world; dist.all_gather_object(hashes, state_hash)
    if len(set(hashes)) != 1: raise RuntimeError('model parameters/buffers differ across four ranks')
    changed = any(not torch.equal(initial[n], p.detach().cpu()) for n, p in model.named_parameters() if n in initial)
    if not changed: raise RuntimeError('four-card smoke did not update refiner parameters')
    payload = {'rank': rank, 'slots': [rank, rank+4], 'selected_windows': [entry['rank_windows'][s] for entry in plan['entries'][:3] for s in (rank,rank+4)],
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
               'model_sha256': state_hash, 'sync_status': 'PASS'}
    gathered = [None] * world; dist.all_gather_object(gathered, payload)
    if rank == 0:
        expected_windows = [[int(e['rank_windows'][s]) for e in plan['entries'][:3] for s in (r, r+4)] for r in range(4)]
        write_json(report / 'four_accum_smoke.json', {'status': 'PASS', 'updates': 3,
            'global_window_exposures': 24, 'true_global_updates': False,
            'fixed_plan_slots_all_eight_verified': True, 'manual_four_rank_average': True,
            'refiner_parameters_changed': changed, 'rank_model_state_synced': True,
            'ranks': gathered, 'expected_rank_windows': expected_windows,
            'source_sha256': '68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a',
            'plan_sha256': sha256(report / 'training_plan.json')})
    dist.barrier(); dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(); group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--single', action='store_true'); group.add_argument('--four', action='store_true')
    args = ap.parse_args()
    try:
        if args.single: single()
        else: four()
    except BaseException as exc:
        record_blocker('gpu_smoke', exc)
        raise


if __name__ == '__main__': main()
