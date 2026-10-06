"""Fixed CPU, one-card and eight-card temporary contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.object_locus_region_class_runtime import (
    REPORT_ROOT, arm_dirs, build_model, build_optimizer, build_batch,
    capture_rng, manifest_and_plan, rank_micro_windows, rank_world, state_sha,
    train_accumulated_step, train_one_step, write_json,
)


def optimizer_sha(optimizer):
    h = hashlib.sha256()
    for group in optimizer.param_groups:
        h.update(group['name'].encode())
        for p in group['params']:
            state = optimizer.state.get(p, {})
            for key, value in sorted(state.items()):
                h.update(str(key).encode())
                if torch.is_tensor(value):
                    t = value.detach().cpu().contiguous()
                    h.update(str(t.dtype).encode())
                    h.update(json.dumps(list(t.shape)).encode())
                    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
                else:
                    h.update(repr(value).encode())
    return h.hexdigest()


def build_prediction(model, opt, batch, exposure):
    from tokengs.models.input_types import ModelInputDecoder, split_data
    mi, _ = split_data(batch, opt)
    decoder = ModelInputDecoder(cam_view=batch['cam_view_all'], intrinsics=batch['intrinsics_all'])
    context = ModelInputDecoder(cam_view=batch['cam_view_all'][:, :2], intrinsics=batch['intrinsics_all'][:, :2])
    return model.forward_object_locus(mi, render_decoder_input=decoder,
        read_context_decoder=context, context_decoder=context, step=exposure)


def initialize_device(mode):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    expected = 1 if mode == 'single' else 2
    if world != expected:
        raise RuntimeError(f'{mode} smoke requires {expected} process(es)')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires CUDA')
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    if mode == 'two':
        import datetime
        dist.init_process_group('nccl', timeout=datetime.timedelta(hours=4))
    if os.uname().nodename != '3dimage-13' or torch.cuda.get_device_name(local) != 'NVIDIA GeForce RTX 3090':
        raise RuntimeError('smoke node/device must be 3dimage-13 RTX3090')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda', local)


def single_smoke(device, manifest, plan):
    rank, _ = rank_world()
    wi = plan['entries'][0]['rank_windows'][0]
    refs = None
    c_hash = None
    results = {}
    for arm in ('control', 'region_class'):
        torch.cuda.reset_peak_memory_stats(device)
        model, opt = build_model(arm, device, report=False)
        optimizer = build_optimizer(model)
        init_hash = state_sha(model.state_dict(), exclude_region=(arm == 'region_class'))
        if c_hash is None:
            c_hash = init_hash
        elif init_hash != c_hash:
            raise RuntimeError('C/R common parameter or buffer initialization differs')
        batch = build_batch(opt, manifest['train_all56'][wi], device)
        if arm == 'region_class':
            model.capture_region_grads = True

        if arm == 'control':
            model.eval()
            with torch.no_grad():
                first = build_prediction(model, opt, batch, 8)
                first_values = {k: first[k].detach().cpu().clone() for k in
                    ('gaussians', 'gaussian_membership', 'thing_logits19')}
                first_values['rgb'] = first['render']['images_pred'].detach().cpu().clone()
                repeats = [first_values]
                for _ in range(5):
                    repeat = build_prediction(model, opt, batch, 8)
                    values = {k: repeat[k].detach().cpu().clone() for k in
                        ('gaussians', 'gaussian_membership')}
                    values['rgb'] = repeat['render']['images_pred'].detach().cpu().clone()
                    repeats.append(values)
                noise = {}
                for metric, key in (('gaussian', 'gaussians'), ('membership', 'gaussian_membership'), ('rgb', 'rgb')):
                    noise[metric] = max(float((repeats[i][key] - repeats[j][key]).abs().max())
                        for i in range(len(repeats)) for j in range(i + 1, len(repeats)))
            refs = (first_values, noise)
            model.train()
        else:
            model.eval()
            with torch.no_grad():
                pred = build_prediction(model, opt, batch, 8)
                now = {k: pred[k].detach().cpu() for k in ('gaussians', 'gaussian_membership', 'thing_logits19')}
                now['rgb'] = pred['render']['images_pred'].detach().cpu()
            ref, noise = refs
            differences = {
                'gaussian': float((now['gaussians'] - ref['gaussians']).abs().max()),
                'membership': float((now['gaussian_membership'] - ref['gaussian_membership']).abs().max()),
                'rgb': float((now['rgb'] - ref['rgb']).abs().max()),
            }
            if any(differences[k] > noise[k] for k in differences):
                raise RuntimeError(f'zero-projection output differs beyond measured C repeat envelope: {differences} vs {noise}')
            if not torch.equal(now['thing_logits19'], ref['thing_logits19']):
                raise RuntimeError('zero-projection public classification logits are not exactly equal')
            model.train()
            results['initialization_comparison'] = dict(common_state_equal=True,
                gaussian_rgb_membership_difference=differences, c_repeat_noise=noise,
                zero_projection_classifier_logits_exact=True)

        started = time.monotonic()
        first_region_projection_grad = None
        for update in (1, 2):
            output, row = train_one_step(model, optimizer, batch, update)
            required = ('loss', 'loss_recon', 'loss_understanding', 'preclip_norm')
            if not all(k in row and math.isfinite(float(row[k])) for k in required):
                raise FloatingPointError(f'{arm} nonfinite or missing smoke scalar')
            pred = output['prediction']
            for key, value in (('gaussian', pred['gaussians']), ('RGB', pred['render']['images_pred']),
                               ('mask', pred['gaussian_membership']), ('classification', pred['thing_logits19'])):
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f'nonfinite {arm} {key}')
            if arm == 'region_class':
                projection = model.panoptic.region_class_proj.weight
                if update == 1:
                    first_region_projection_grad = float(projection.grad.norm()) if projection.grad is not None else 0.0
                    if not math.isfinite(first_region_projection_grad) or first_region_projection_grad <= 0:
                        raise RuntimeError('region projection first active gradient must be finite and nonzero')
                if update == 2:
                    hook = getattr(model, '_last_region_grad_norms', {})
                    if not all(k in hook and math.isfinite(hook[k]) and hook[k] > 0 for k in ('feature_grid', 'weights', 'pooled')):
                        raise RuntimeError(f'region feature/mask-weight gradients absent: {hook}')
            del output
        grads = [p.grad for n, p in model.named_parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f'{arm} gradient path is absent/nonfinite')
        reconstruction_parameters = [p for group in optimizer.param_groups
                                     if group['name'].startswith('reconstruction_')
                                     for p in group['params']]
        if not any(p.grad is not None and float(p.grad.norm()) > 0 for p in reconstruction_parameters):
            raise RuntimeError('reconstruction gradient path is missing')
        understanding_parameters = [p for name, p in model.named_parameters()
                                    if name.startswith('understanding.')]
        if not any(p.grad is not None and float(p.grad.norm()) > 0 for p in understanding_parameters):
            raise RuntimeError('understanding gradient path is missing')
        if arm == 'region_class' and float(model.panoptic.region_class_proj.weight.detach().norm()) <= 0:
            raise RuntimeError('region projection did not update')
        results[arm] = dict(status='PASS', updates=2, forward_exposures=[8, 16],
            last_loss=row, projection_first_active_gradient=first_region_projection_grad,
            region_gradient_norms=getattr(model, '_last_region_grad_norms', None),
            parameter_numel=sum(p.numel() for p in model.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            elapsed_seconds=time.monotonic() - started, temporary_state='DISCARDED')
        del model, opt, optimizer, batch
        torch.cuda.empty_cache()

    # Exercise the unchanged local evaluator and official packed export with one
    # real window after both arm update smokes. No artifacts are retained.
    model, opt = build_model('region_class', device, report=False)
    optimizer = build_optimizer(model)
    win = manifest['train_all56'][wi]
    from scripts.eval_object_locus_panoptic_v1 import evaluate_windows
    temp_report = Path('/tmp/object_locus_region_class_single_smoke')
    if temp_report.exists():
        import shutil
        shutil.rmtree(temp_report)
    result, per_gt, queries = evaluate_windows(model, opt, [win], 0, 'smoke_pair', temp_report,
        device, build_batch, official=True, panels=False)
    if result['local']['context'].get('candidate_ap', {}).get('error'):
        raise RuntimeError('local evaluator candidate AP failed')
    if not per_gt or not queries or not (temp_report / 'official/step_0000/smoke_pair/official_all.json').is_file():
        raise RuntimeError('single-pair official export/evaluator artifacts missing')
    results['evaluator_contract'] = dict(status='PASS', local_windows=result['local']['context']['windows'],
        official_all=(temp_report / 'official/step_0000/smoke_pair/official_all.json').is_file(),
        artifacts_temporary=True)
    if rank == 0:
        for arm in ('control', 'region_class'):
            reports, _ = arm_dirs(arm)
            write_json(reports / 'single_smoke.json', results[arm] | dict(arm=arm,
                initialization_comparison=results.get('initialization_comparison') if arm == 'region_class' else None,
                evaluator_contract=results['evaluator_contract']))
    del model, opt, optimizer
    torch.cuda.empty_cache()


def two_gpu_smoke(device, manifest, plan):
    rank, world = rank_world()
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    all_plan = [None] * world
    dist.all_gather_object(all_plan, plan_sha)
    if len(set(all_plan)) != 1:
        raise RuntimeError('two-rank plan SHA mismatch')
    for arm in ('control', 'region_class'):
        torch.cuda.reset_peak_memory_stats(device)
        model, opt = build_model(arm, device, report=False)
        optimizer = build_optimizer(model)
        if arm == 'region_class':
            model.capture_region_grads = True
        started = time.monotonic()
        local_window_counts = torch.zeros(56, dtype=torch.int64, device=device)
        local_micro_batches = 0
        optimizer_updates = 0
        for update in range(40):
            local_windows = rank_micro_windows(plan['entries'][update], rank)
            local_micro_batches += len(local_windows)
            for wi in local_windows:
                local_window_counts[wi] += 1
            batches = (build_batch(opt, manifest['train_all56'][wi], device) for wi in local_windows)
            row = train_accumulated_step(model, optimizer, batches, update)
            optimizer_updates += 1
            if not all(math.isfinite(float(row[k])) for k in ('loss', 'loss_recon', 'loss_understanding', 'preclip_norm')):
                raise FloatingPointError(f'{arm} nonfinite smoke update {update + 1}')
            if row.get('accumulated_micro_batches') != 4 or row.get('global_batch') != 8:
                raise RuntimeError('two-card smoke did not accumulate four micro-batches into global batch eight')
            if arm == 'region_class':
                row.update(getattr(model, '_last_region_stats', {}))
            del row
        dist.all_reduce(local_window_counts, op=dist.ReduceOp.SUM)
        expected_counts = torch.zeros(56, dtype=torch.int64, device=device)
        for entry in plan['entries'][:40]:
            for idx in entry['rank_windows']:
                expected_counts[idx] += 1
        if not torch.equal(local_window_counts, expected_counts):
            raise RuntimeError(f'{arm} two-rank smoke exposure plan mismatch')
        micro_counts = [None] * world
        dist.all_gather_object(micro_counts, local_micro_batches)
        if micro_counts != [160, 160]:
            raise RuntimeError(f'{arm} accumulated micro-batch count mismatch: {micro_counts}')
        model_hash = state_sha(model.state_dict())
        hashes = [None] * world
        dist.all_gather_object(hashes, model_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f'{arm} model state differs across ranks')
        opt_hash = optimizer_sha(optimizer)
        opt_hashes = [None] * world
        dist.all_gather_object(opt_hashes, opt_hash)
        if len(set(opt_hashes)) != 1:
            raise RuntimeError(f'{arm} optimizer state differs across ranks')
        state_steps = [int(value['step']) for value in optimizer.state.values() if 'step' in value]
        if optimizer_updates != 40 or not state_steps or min(state_steps) < 39 or max(state_steps) > 40:
            raise RuntimeError(f'{arm} optimizer global/moment step counts mismatch: updates={optimizer_updates}, steps={set(state_steps)}')
        reconstruction_probe = next(p for group in optimizer.param_groups if group['name'].startswith('reconstruction_') for p in group['params'])
        reconstruction_step = int(optimizer.state[reconstruction_probe]['step'])
        if reconstruction_step != 40:
            raise RuntimeError(f'{arm} reconstruction probe should receive all 40 optimizer steps, got {reconstruction_step}')
        projection_norm = None
        projection_grad = None
        if arm == 'region_class':
            projection = model.panoptic.region_class_proj.weight
            projection_step = int(optimizer.state[projection]['step'])
            if projection_step != 39:
                raise RuntimeError(f'zero warm-up update means region projection should have 39 optimizer states, got {projection_step}')
            projection_norm = float(projection.detach().norm())
            projection_grad = float(projection.grad.norm()) if projection.grad is not None else 0.0
            if not math.isfinite(projection_norm) or projection_norm <= 0 or not math.isfinite(projection_grad) or projection_grad <= 0:
                raise RuntimeError('region projection did not activate during two-rank smoke')
            hook = getattr(model, '_last_region_grad_norms', {})
            if not all(k in hook and math.isfinite(hook[k]) for k in ('feature_grid', 'weights', 'pooled')):
                raise RuntimeError(f'region mask/feature gradient hooks missing: {hook}')
        result = dict(status='PASS', arm=arm, updates=40, world_size=world, accumulation_steps=4,
            per_rank_micro_batches=160, completed_exposures=320, optimizer_update_calls=optimizer_updates,
            reconstruction_probe_optimizer_steps=reconstruction_step,
            region_projection_optimizer_steps=projection_step if arm == 'region_class' else None,
            data_plan_sha256=plan_sha,
            model_state_sha256=model_hash, optimizer_state_sha256=opt_hash,
            rank_model_and_optimizer_synchronized=True,
            region_projection_norm=projection_norm, region_projection_gradient_norm=projection_grad,
            last_region_gradient_norms=getattr(model, '_last_region_grad_norms', None),
            all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
            parameter_numel=sum(p.numel() for p in model.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            elapsed_seconds=time.monotonic() - started, temporary_state='DISCARDED')
        if rank == 0:
            reports, _ = arm_dirs(arm)
            write_json(reports / 'two_card_smoke.json', result)
        del model, opt, optimizer
        torch.cuda.empty_cache()
        dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('single', 'two'), required=True)
    args = parser.parse_args()
    device = initialize_device(args.mode)
    manifest, plan = manifest_and_plan()
    if args.mode == 'single':
        single_smoke(device, manifest, plan)
    else:
        two_gpu_smoke(device, manifest, plan)


if __name__ == '__main__':
    main()
