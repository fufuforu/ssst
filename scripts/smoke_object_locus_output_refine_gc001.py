"""Real-data single-card parity and four-card accumulation gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
import gc
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.object_locus_gc_sweep_runtime import SOURCE_MANIFEST, SOURCE_EXPOSURES, WINDOWS, sha256
from scripts.object_locus_output_refine_gc001_runtime import (
    roots, write_json, rank_world, init_distributed, build_model, build_optimizer,
    init_slot_states, train_update, load_checkpoint, record_blocker,
)
from scripts.object_locus_output_refine_smoke_contracts import (
    compare_independent_forward, flattened_tensors, clone_tree, assert_tensor_tree_equal,
    assert_value_tree_equal,
    checked_lift_replacement, patch_shared_readout_inputs, cleanup_distributed_if_initialized,
)


def cpu_copy(obj):
    if torch.is_tensor(obj): return obj.detach().cpu()
    if isinstance(obj, dict): return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [cpu_copy(v) for v in obj]
    if isinstance(obj, tuple): return tuple(cpu_copy(v) for v in obj)
    return obj


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
    return tensor_state_hash(model.state_dict())


def tensor_state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def memory_snapshot(stage, device=None):
    payload = {'stage': stage, 'device': None, 'max_memory_allocated_bytes': None,
               'max_memory_reserved_bytes': None, 'cuda_available': bool(torch.cuda.is_available()),
               'unavailable_reason': None}
    if not torch.cuda.is_available():
        payload['unavailable_reason'] = 'torch.cuda.is_available() is false'
        return payload
    try:
        index = torch.cuda.current_device() if device is None else torch.device(device).index
        if index is None: index = torch.cuda.current_device()
        payload.update(device=torch.cuda.get_device_name(index), device_index=index,
                       max_memory_allocated_bytes=torch.cuda.max_memory_allocated(index),
                       max_memory_reserved_bytes=torch.cuda.max_memory_reserved(index))
    except Exception as exc:
        payload['unavailable_reason'] = repr(exc)
    return payload


def reset_peak_memory(device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def record_smoke_exception(stage, exc):
    try:
        report, _ = roots()
        rank = int(os.environ.get('LOCAL_RANK', '0'))
        path = report / ('single_smoke_failure.json' if stage == 'single_gpu_smoke'
                         else f'four_accum_smoke_failure_rank{rank}.json')
        prior = json.loads(path.read_text()) if path.exists() else {}
        prior.update({'status': 'INVALID', 'stage': stage, 'failure': repr(exc),
                      'traceback': traceback.format_exc(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                      'gpu_memory': memory_snapshot(stage)})
        write_json(path, prior)
    except Exception:
        pass


def _finite_tree(value, label):
    for key, tensor in flattened_tensors(value).items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f'nonfinite {label}.{key}')


def _readout_identity_contract(model, prediction, batch, report):
    """Run both actual readout methods against one shared true lift/render result."""
    import hashlib as _hashlib
    import torch.nn.functional as F
    from scripts.export_object_locus_v3_set_official import assemble_panoptic, _save_packed
    from tokengs.models.input_types import ModelInputDecoder
    import tokengs.models.object_locus_panoptic_v1 as old_module
    import tokengs.models.object_locus_output_refine_gc001 as new_module

    old_final = clone_tree(prediction['states'][-1])
    new_final = clone_tree(prediction['states'][-1])
    old_final.pop('q_refined', None); new_final.pop('q_refined', None)
    old_original = clone_tree(old_final); new_original = clone_tree(new_final)
    gaussians = prediction['gaussians']
    fm = prediction['F_m']
    b = gaussians.shape[0]
    render_decoder = ModelInputDecoder(cam_view=batch['cam_view_all'][:, :2], intrinsics=batch['intrinsics_all'][:, :2])
    read_decoder = ModelInputDecoder(cam_view=batch['cam_view_all'][:, :2], intrinsics=batch['intrinsics_all'][:, :2])

    feature_grid = F.interpolate(fm.flatten(0, 1), size=(256, 256), mode='bilinear',
                                  align_corners=False).reshape(b, 2, 256, 256, 256)
    from tokengs.models.object_locus_panoptic_v1_lift import lift_features
    with torch.no_grad():
        cached_lift = lift_features(feature_grid, gaussians, read_decoder, model.gs)
    _finite_tree(cached_lift, 'cached_lift')
    counts = {'true_lifting_calls': 1, 'shared_lift_consumers': 0,
              'real_renderer_calls': 0, 'shared_renderer_consumers': 0,
              'output_refiner_calls': 0}
    lift_replacement = checked_lift_replacement(
        feature_grid.detach().clone(), gaussians.detach().clone(),
        read_decoder.cam_view.detach().clone(), read_decoder.intrinsics.detach().clone(),
        model.gs, cached_lift, counts)
    original_renderer = model.gs.render_feature_channels
    original_old_lift = old_module.lift_features
    original_new_lift = new_module.lift_features
    hook = model.panoptic.output_3d_refine.register_forward_hook(
        lambda _module, _inputs, _output: counts.__setitem__('output_refiner_calls', counts['output_refiner_calls'] + 1))
    try:
        with torch.no_grad(), patch_shared_readout_inputs(
                old_module, new_module, model.gs, lift_replacement, original_renderer, counts) as renderer_state:
            old_output = old_module.LocusGSObjectLocusPanopticV1Recon._readout(
                model, old_final, gaussians, fm, read_decoder, render_decoder)
            new_output = new_module.LocusGSObjectLocusOutputRefineV1Recon._readout(
                model, new_final, gaussians, fm, read_decoder, render_decoder)
            if counts['shared_lift_consumers'] != 2 or counts['real_renderer_calls'] != 1 or counts['shared_renderer_consumers'] != 1:
                raise AssertionError(f'shared evidence/render call counts invalid: {counts}')
            if counts['output_refiner_calls'] != 1:
                raise AssertionError(f'actual output_3d_refine call count must be 1, got {counts["output_refiner_calls"]}')
            if renderer_state['output'] is None:
                raise AssertionError('old readout did not execute the real feature renderer')
            # Each method mutates only its own final mapping with the same class outputs.
            old_final_common = {k: v for k, v in old_final.items() if k in old_original}
            new_final_common = {k: v for k, v in new_final.items() if k in new_original}
            assert_value_tree_equal(old_original, {k: old_final_common[k] for k in old_original}, 'old final input values')
            assert_value_tree_equal(new_original, {k: new_final_common[k] for k in new_original}, 'new final input values')
            assert_value_tree_equal(old_final_common, new_final_common, 'mutated public final state')
            q = new_original['q']
            for name, value in (('q_base', new_output['q_base']), ('q_refined', new_output['q_refined']),
                                ('final.q_refined', new_final['q_refined'])):
                if not torch.equal(value, q): raise AssertionError(f'{name} is not exactly the original q')
            old_tensors, new_tensors = flattened_tensors(old_output), flattened_tensors(new_output)
            expected_extras = {'q_base', 'q_refined'}
            if set(new_tensors) - set(old_tensors) != expected_extras or set(old_tensors) - set(new_tensors):
                raise AssertionError('old/new readout public tensor keys differ beyond q_base/q_refined')
            old_public = {k: old_tensors[k] for k in sorted(old_tensors)}
            new_public = {k: new_tensors[k] for k in sorted(old_tensors)}
            assert_tensor_tree_equal(old_public, new_public, 'shared evidence readout public tensors')
            if not torch.equal(new_output['q_base'], q) or not torch.equal(new_output['q_refined'], q):
                raise AssertionError('new readout query identity failed')
            old_panoptic = assemble_panoptic(old_output)
            new_panoptic = assemble_panoptic(new_output)
            assert_tensor_tree_equal(dict(zip(('semantic', 'instance', 'raw_sem'), old_panoptic)),
                                     dict(zip(('semantic', 'instance', 'raw_sem'), new_panoptic)),
                                     'official assemble_panoptic outputs')
            packed_root = report / 'smoke_single' / 'shared_evidence_readout_identity'
            packed_root.mkdir(parents=True, exist_ok=True)
            packed = []
            for view in (0, 1):
                row = {}
                for label, panoptic in (('old', old_panoptic), ('new', new_panoptic)):
                    path = packed_root / f'{label}_context{view}.png'
                    _save_packed(path, panoptic[0][view], panoptic[1][view])
                    payload = path.read_bytes()
                    row[label] = {'path': str(path), 'sha256': _hashlib.sha256(payload).hexdigest(),
                                  'bytes': len(payload)}
                if row['old']['sha256'] != row['new']['sha256'] or (packed_root / f'old_context{view}.png').read_bytes() != (packed_root / f'new_context{view}.png').read_bytes():
                    raise AssertionError(f'official packed PNG bytes differ at context {view}')
                packed.append(row)
    finally:
        hook.remove()

    restored_renderer = model.gs.render_feature_channels
    patches_restored = (old_module.lift_features is original_old_lift and
                        new_module.lift_features is original_new_lift and
                        restored_renderer.__func__ is original_renderer.__func__ and
                        restored_renderer.__self__ is original_renderer.__self__)
    if not patches_restored: raise AssertionError('shared readout patches did not restore original functions')
    record = {
        'status': 'PASS',
        'protocol': 'shared_evidence_identity_v2',
        'actual_methods': {
            'old': 'LocusGSObjectLocusPanopticV1Recon._readout',
            'new': 'LocusGSObjectLocusOutputRefineV1Recon._readout',
        },
        'shared_inputs': {
            'real_r3d_final_decoder_state': True, 'same_gaussians_object': True,
            'same_F_m': True, 'read_context_cam_view_sha256': _hashlib.sha256(read_decoder.cam_view.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            'read_context_intrinsics_sha256': _hashlib.sha256(read_decoder.intrinsics.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            'render_cam_view_sha256': _hashlib.sha256(render_decoder.cam_view.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            'render_intrinsics_sha256': _hashlib.sha256(render_decoder.intrinsics.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        },
        'calls': counts,
        'all_public_readout_tensors_torch_equal': True,
        'public_readout_tensor_count': len(old_tensors),
        'new_q_base_q_refined_and_final_q_refined_equal_original_q': True,
        'official_assemble_panoptic_semantic_instance_raw_sem_equal': True,
        'official_packed_context_pngs': packed,
        'patches_restored': patches_restored,
    }
    write_json(report / 'same_evidence_readout_identity.json', record)
    return record


def single():
    start = time.time(); device = init_distributed(formal=False); rank, world = rank_world()
    if world != 1 or dist.is_initialized(): raise RuntimeError('single smoke must run without a process group')
    reset_peak_memory(device)
    report, _ = roots()
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    plan = json.loads((report / 'training_plan.json').read_text())
    entry = plan['entries'][0]
    wi = entry['rank_windows'][0]
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.object_locus_gc_sweep_runtime import build_model as build_gc_model
    first = manifest['expanded_train_windows'][wi]
    gpu = torch.device(device)
    # Exactly one complete model resides on the GPU at a time.
    base_model, opt, source = build_gc_model(device, report=False)
    base_model.eval(); batch = build_batch(opt, first, gpu)
    base_compare = cpu_forward_compare(base_model, batch)
    old_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    base_state_sha = tensor_state_hash(old_state)
    del base_model; gc.collect(); torch.cuda.empty_cache()
    model, opt, source_new = build_model(device)
    old_loaded = model.state_dict()
    if set(old_loaded) - {f'panoptic.output_3d_refine.{k}' for k in model.panoptic.output_3d_refine.state_dict()} != set(old_state):
        raise RuntimeError('R3D source-state keys differ from C source keys')
    if len(old_state) != 1445 or any(
            old_loaded[k].shape != v.shape or old_loaded[k].dtype != v.dtype or
            not torch.equal(old_loaded[k].detach().cpu(), v) for k, v in old_state.items()):
        raise RuntimeError('R3D initial source state differs from original C model')
    r3d_source_state_sha = tensor_state_hash({k: old_loaded[k] for k in old_state})
    if r3d_source_state_sha != base_state_sha:
        raise RuntimeError('C/R full source-state SHA mismatch despite source loading')
    refiner = model.panoptic.output_3d_refine
    if sum(p.numel() for p in refiner.parameters()) != 527616:
        raise RuntimeError('new output refiner parameter count differs from 527616')
    for name, value in refiner.state_dict().items():
        if name.startswith(('W_O.', 'W_2.')) and not torch.equal(value, torch.zeros_like(value)):
            raise RuntimeError(f'zero output projection init failed for {name}')
    initial_state_sha = cpu_state_hash(model)
    refiner_state_sha = tensor_state_hash(refiner.state_dict())
    model.eval()
    r3d_model = model
    r3d_compare_gpu = real_forward(r3d_model, batch)
    r3d_compare = cpu_copy(r3d_compare_gpu)
    independent = compare_independent_forward(base_compare, r3d_compare)
    write_json(report / 'independent_full_forward_diagnostic.json', {
        'status': 'DIAGNOSTIC_ONLY', 'classification': independent['classification'],
        'independent_full_forward_diagnostic': independent['independent_full_forward_diagnostic']})
    readout = _readout_identity_contract(r3d_model, r3d_compare_gpu, batch, report)
    del r3d_model, model, r3d_compare_gpu, base_compare, r3d_compare, old_state, old_loaded, refiner
    del batch, opt, source, source_new
    gc.collect(); torch.cuda.empty_cache()
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
    batch = build_batch(opt, first, gpu)
    smoke_ckpt = report / 'smoke_single' / 'roundtrip.pt'
    from scripts.object_locus_output_refine_gc001_runtime import save_checkpoint
    import dataclasses
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    saved_config = dataclasses.asdict(opt)
    save_checkpoint(smoke_ckpt, model, optimizer, 3, states, np.zeros(WINDOWS, dtype=np.int64),
                   saved_config, source, sha256(report / 'training_plan.json'), code_sha, device)
    roundtrip_state_cpu = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    roundtrip_state_sha = tensor_state_hash(roundtrip_state_cpu)
    roundtrip_refiner_sha = tensor_state_hash({k: v for k, v in roundtrip_state_cpu.items()
        if k.startswith('panoptic.output_3d_refine.')})
    roundtrip_prediction = real_forward(model.eval(), batch)
    _finite_tree(roundtrip_prediction, 'checkpoint_preload_forward')
    final = roundtrip_prediction['states'][-1]
    fixed_inputs_gpu = {
        'q_base': roundtrip_prediction['q_base'].detach().contiguous(),
        'gaussian_feature': roundtrip_prediction['gaussian_feature'].detach().contiguous(),
        'xyz': roundtrip_prediction['gaussians'][..., :3].detach().contiguous(),
        'c': final['c'].detach().contiguous(), 's': final['s'].detach().contiguous(),
    }
    with torch.no_grad():
        fixed_q_refined = model.panoptic.output_3d_refine(
            fixed_inputs_gpu['q_base'], fixed_inputs_gpu['gaussian_feature'], fixed_inputs_gpu['xyz'],
            fixed_inputs_gpu['c'], fixed_inputs_gpu['s'])
        fixed_mask_embed = model.understanding.mask_embedder(fixed_q_refined)
        fixed_logits = fixed_inputs_gpu['gaussian_feature'] @ fixed_mask_embed.transpose(1, 2)
        fixed_membership = fixed_logits.sigmoid()
        fixed_class = model.panoptic.classify(fixed_q_refined)
    fixed_reference_gpu = {'q_refined': fixed_q_refined, 'gaussian_mask_logits': fixed_logits,
                           'gaussian_membership': fixed_membership, 'classification': fixed_class}
    _finite_tree(fixed_reference_gpu, 'checkpoint_fixed_input_reference')
    fixed_inputs_cpu = {k: v.detach().cpu().contiguous().clone() for k, v in fixed_inputs_gpu.items()}
    fixed_reference_cpu = cpu_copy(fixed_reference_gpu)
    del fixed_inputs_gpu, fixed_reference_gpu, fixed_q_refined, fixed_mask_embed, fixed_logits, fixed_membership, fixed_class
    del roundtrip_prediction, final, batch, model, opt, optimizer, states, refiner0, bn_before, bn_after
    del source
    gc.collect(); torch.cuda.empty_cache()
    # Only one complete GPU model is created by the public strict loader.
    reloaded, reopt, metadata = load_checkpoint(smoke_ckpt, device)
    reloaded.eval()
    loaded_state = reloaded.state_dict()
    if set(loaded_state) != set(roundtrip_state_cpu): raise RuntimeError('checkpoint state keys differ on load')
    for name, expected in roundtrip_state_cpu.items():
        value = loaded_state[name].detach().cpu().contiguous()
        if value.shape != expected.shape or value.dtype != expected.dtype or not torch.equal(value, expected):
            raise RuntimeError(f'strict checkpoint state mismatch at {name}')
    loaded_state_sha = tensor_state_hash({k: v.detach().cpu().contiguous() for k, v in loaded_state.items()})
    loaded_refiner_sha = tensor_state_hash({k: v.detach().cpu().contiguous() for k, v in loaded_state.items()
        if k.startswith('panoptic.output_3d_refine.')})
    if loaded_state_sha != roundtrip_state_sha or loaded_refiner_sha != roundtrip_refiner_sha:
        raise RuntimeError('checkpoint state SHA mismatch')
    if metadata.get('experiment', {}).get('arm') != 'R3D' or metadata.get('source_sha256') != '68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a':
        raise RuntimeError('strict checkpoint experiment/source metadata mismatch')
    if metadata.get('plan_sha256') != sha256(report / 'training_plan.json') or metadata.get('completed_updates') != 3:
        raise RuntimeError('strict checkpoint plan/update metadata mismatch')
    if metadata.get('config') != saved_config or dataclasses.asdict(reopt) != saved_config:
        raise RuntimeError('strict checkpoint configuration metadata mismatch')
    if metadata.get('source_manifest_sha256') != 'a03e6842e9657d512a0de2b9dd20ed73aa9bcc61ec8f4fd8d33ed07e8cc11249' or metadata.get('code_sha') != code_sha:
        raise RuntimeError('strict checkpoint manifest/code metadata mismatch')
    loaded_inputs = {k: v.to(device=device) for k, v in fixed_inputs_cpu.items()}
    with torch.no_grad():
        loaded_q_refined = reloaded.panoptic.output_3d_refine(
            loaded_inputs['q_base'], loaded_inputs['gaussian_feature'], loaded_inputs['xyz'],
            loaded_inputs['c'], loaded_inputs['s'])
        loaded_mask_embed = reloaded.understanding.mask_embedder(loaded_q_refined)
        loaded_logits = loaded_inputs['gaussian_feature'] @ loaded_mask_embed.transpose(1, 2)
        loaded_outputs = {'q_refined': loaded_q_refined, 'gaussian_mask_logits': loaded_logits,
                          'gaussian_membership': loaded_logits.sigmoid(),
                          'classification': reloaded.panoptic.classify(loaded_q_refined)}
    loaded_outputs_cpu = cpu_copy(loaded_outputs)
    assert_tensor_tree_equal(fixed_reference_cpu, loaded_outputs_cpu, 'same_input_heads_exact')
    checkpoint_record = {
        'status': 'PASS', 'path': str(smoke_ckpt), 'completed_updates': 3,
        'full_state_sha256': roundtrip_state_sha, 'loaded_full_state_sha256': loaded_state_sha,
        'new_block_state_sha256': roundtrip_refiner_sha, 'loaded_new_block_state_sha256': loaded_refiner_sha,
        'full_state_tensor_count': len(roundtrip_state_cpu), 'all_state_keys_shapes_dtypes_values_exact': True,
        'same_input_heads_exact': True,
        'fixed_head_outputs': ['q_refined', 'gaussian_mask_logits', 'gaussian_membership',
                               'thing_logits19', 'class_logits19', 'p_class', 'conditional_class_prob',
                               'objectness_prob', 'thing_class_logits'],
        'independent_full_forward_exactness_claimed': False,
        'optimizer_moments_restored_for_continued_training': False,
        'metadata': {'arm': metadata['experiment']['arm'], 'source_sha256': metadata['source_sha256'],
                     'source_manifest_sha256': metadata['source_manifest_sha256'],
                     'plan_sha256': metadata['plan_sha256'], 'code_sha': metadata['code_sha'],
                     'configuration_exact': True, 'completed_updates': metadata['completed_updates']},
    }
    write_json(report / 'checkpoint_roundtrip_exact.json', checkpoint_record)
    del reloaded, reopt, loaded_inputs, loaded_outputs, loaded_outputs_cpu, loaded_q_refined, loaded_mask_embed, loaded_logits
    del loaded_state, roundtrip_state_cpu, fixed_inputs_cpu, fixed_reference_cpu
    gc.collect(); torch.cuda.empty_cache()
    peak = memory_snapshot('single_gpu_smoke', device)
    result = {'status': 'PASS', 'protocol': 'shared_evidence_identity_v2',
        'source_state_tensors': 1445, 'source_state_exact': True,
        'source_state_sha256': base_state_sha, 'initial_r3d_source_state_sha256': r3d_source_state_sha,
        'initial_r3d_state_sha256': initial_state_sha,
        'initial_r3d_new_block_state_sha256': refiner_state_sha,
        'new_block_parameters': 527616, 'new_block_zero_output_projection': True,
        'independent_full_forward_diagnostic': independent['independent_full_forward_diagnostic'],
        'independent_hard_gate_classification': independent['classification'],
        'same_evidence_readout_identity': readout,
        'three_local_updates_two_microbatches': {'updates': 3, 'windows': windows, 'steps_are_not_global_eight_window_updates': True,
            'finite': True, 'bn_running_statistics_unchanged': True, 'positive_understanding_gradient': positive_under,
            'refiner_parameters_changed': changed, 'last_q_delta_rms': logs[-1]['q_refined_minus_q_base_rms'], 'logs': logs},
        'checkpoint_roundtrip_exact': checkpoint_record,
        'gpu': torch.cuda.get_device_name(device), 'gpu_memory': peak,
        'elapsed_seconds': time.time() - start}
    if rank == 0: write_json(report / 'single_smoke.json', result)
    cleanup_distributed_if_initialized(dist)


def four():
    device = init_distributed(formal=True); rank, world = rank_world()
    reset_peak_memory(device)
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
               'stage': 'four_gpu_accumulation_smoke', 'device': torch.cuda.get_device_name(device),
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
    cleanup_distributed_if_initialized(dist)


def main():
    ap = argparse.ArgumentParser(); group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--single', action='store_true'); group.add_argument('--four', action='store_true')
    args = ap.parse_args()
    try:
        if args.single: single()
        else: four()
    except BaseException as exc:
        record_smoke_exception('single_gpu_smoke' if args.single else 'four_gpu_accumulation_smoke', exc)
        record_blocker('gpu_smoke', exc)
        if dist.is_initialized():
            try: dist.destroy_process_group()
            except Exception: pass
        raise


if __name__ == '__main__': main()
