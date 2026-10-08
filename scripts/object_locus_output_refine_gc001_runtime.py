"""R3D training helpers with isolated report/run roots and logical-slot RNG."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.runtime_bootstrap import prepare_runtime
REPO = Path(__file__).resolve().parents[1]
prepare_runtime(REPO)
from scripts.object_locus_v3_set_runtime import build_batch, jsonable
from scripts.object_locus_gc_sweep_runtime import (
    SOURCE_CHECKPOINT, SOURCE_SHA256, SOURCE_EXPOSURES, SOURCE_MANIFEST,
    SOURCE_PLAN, WINDOWS, UPDATES_PER_EPOCH, TOTAL_UPDATES, sha256,
    family, lr_multiplier,
)
from tokengs.models.object_locus_output_refine_gc001 import attach_output_refiner

EXPERIMENT = {
    'name': 'object_locus_output_refine_gc001_v1', 'arm': 'R3D',
    'model_variant': 'output_3d_refine_v1',
    'base_scientific_sha': '9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0',
    'refiner_seed': 31416, 'refiner_heads': 8, 'refiner_query_chunk': 8,
    'geometry_bias_detached': True, 'alpha': 0.01,
    'physical_world_size': 4, 'logical_global_slots': 8,
    'gradient_accumulation_steps': 2,
}


def roots():
    report = Path(os.environ['TASK_REPORT_ATTEMPT'])
    run = Path(os.environ['TASK_RUN_ATTEMPT'])
    return report, run


def write_json(path, payload):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + '\n')


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def init_distributed(*, formal=False):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world not in (1, 4): raise RuntimeError(f'world size must be 1 or 4, got {world}')
    if formal and world != 4: raise RuntimeError('formal training requires four physical ranks')
    if local < 0 or local > 3: raise RuntimeError(f'local rank must be 0..3, got {local}')
    torch.cuda.set_device(local)
    if world == 4: dist.init_process_group('nccl', timeout=datetime.timedelta(hours=4))
    if not os.uname().nodename.startswith('3dimage-11'):
        raise RuntimeError(f'fixed node 3dimage-11 required, got {os.uname().nodename}')
    if torch.cuda.get_device_name(local) != 'NVIDIA GeForce RTX 3090':
        raise RuntimeError(f'RTX3090 required, got {torch.cuda.get_device_name(local)}')
    props = torch.cuda.get_device_properties(local)
    if props.total_memory < 23 * 1024**3: raise RuntimeError('expected approximately 24GB RTX3090 memory')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda', local)


def build_model(device):
    from scripts.object_locus_gc_sweep_runtime import build_model as build_gc_model, load_source_blob
    blob = load_source_blob()
    old_state = blob['model']
    model, opt, source = build_gc_model('cpu', report=False)
    if source['state_tensors'] != 1445 or len(old_state) != 1445:
        raise RuntimeError('expected complete 1445-tensor Full1201 source')
    loaded = model.state_dict()
    unequal = [k for k, v in old_state.items() if k not in loaded or not torch.equal(loaded[k], v)]
    if unequal: raise RuntimeError(f'GC source builder did not strictly preserve source tensors: {unequal[:8]}')
    model = attach_output_refiner(model)
    model = model.to(device=device, dtype=torch.float32)
    if not all(p.requires_grad for p in model.parameters()): raise RuntimeError('all original/new parameters must train')
    rank, world = rank_world()
    if world == 4:
        # The original builder only broadcasts in world=8; broadcast the complete
        # 4-rank model including every buffer and new refiner tensor here.
        for tensor in model.state_dict().values(): dist.broadcast(tensor, src=0)
        signatures = [None] * world
        signature = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            signature.update(name.encode()); signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        dist.all_gather_object(signatures, signature.hexdigest())
        if len(set(signatures)) != 1: raise RuntimeError('full model broadcast mismatch across ranks')
    # Preserve each process RNG after the refiner's isolated CPU seed scope.
    return model, opt, source


def build_optimizer(model):
    no_decay = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm, torch.nn.Embedding)):
            no_decay.update(name + '.' + n for n, _ in module.named_parameters(recurse=False))
    no_decay.add('panoptic.stuff_seed')
    no_decay.update(n for n, p in model.named_parameters() if n.endswith('.level_embed') or getattr(p, '_no_weight_decay', False))
    peaks = {'reconstruction': 1e-6, 'pretrained': 1e-5, 'new': 1e-4}
    groups = []
    for fam in ('reconstruction', 'pretrained', 'new'):
        for decay in (False, True):
            selected = [(n, p) for n, p in sorted(model.named_parameters()) if family(n) == fam and
                        (fam != 'reconstruction' and p.ndim > 1 and not n.endswith('.bias') and n not in no_decay) == decay]
            if selected:
                groups.append({'name': f'{fam}_{"decay" if decay else "nodecay"}',
                               'params': [p for _, p in selected], 'param_names': [n for n, _ in selected],
                               'peak_lr': peaks[fam], 'lr': peaks[fam], 'weight_decay': .05 if decay else 0.})
    flat = [id(p) for g in groups for p in g['params']]
    if len(flat) != len(set(flat)) or set(flat) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('optimizer groups do not cover each trainable parameter exactly once')
    refiner = {id(p) for n, p in model.named_parameters() if n.startswith('panoptic.output_3d_refine.')}
    if not refiner or any(sum(id(p) == i for g in groups for p in g['params']) != 1 for i in refiner):
        raise RuntimeError('new refiner parameter family/optimizer coverage mismatch')
    return torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8)


def load_checkpoint(path, device='cpu'):
    """Strict future-evaluation loader; the public return is always three values."""
    model, opt, source = build_model(device)
    blob = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if blob.get('experiment') != EXPERIMENT or blob.get('source_sha256') != SOURCE_SHA256:
        raise RuntimeError('checkpoint experiment/source metadata mismatch')
    result = model.load_state_dict(blob['model'], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {result}")
    metadata = {k: v for k, v in blob.items() if k not in ('model', 'optimizer')}
    metadata['optimizer_state_in_checkpoint'] = 'optimizer' in blob
    metadata['source_build'] = source
    return model, opt, metadata


def init_slot_states(device):
    rank, world = rank_world()
    if world == 4:
        slots = (rank, rank + 4)
    elif world == 1:
        slots = (0, 4)
    else:
        raise RuntimeError('slot stream setup supports world=1 smoke or world=4')
    states = {}
    for slot in slots:
        seed = 42 + 100003 * slot
        random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        states[slot] = capture_slot_rng(device)
    return states


def capture_slot_rng(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(device) if device is not None else None}


def restore_slot_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device is not None and state.get('torch_cuda') is not None: torch.cuda.set_rng_state(state['torch_cuda'], device)


def capture_all_rng_states(slot_states, device):
    return {'slots': {int(k): v for k, v in slot_states.items()},
            'rank_python': random.getstate(), 'rank_numpy': np.random.get_state(),
            'rank_torch_cpu': torch.get_rng_state(), 'rank_torch_cuda': torch.cuda.get_rng_state(device)}


def synchronized_error(error, device, stage, update, micro=None):
    rank, world = rank_world()
    flag = torch.tensor(int(error is not None), device=device, dtype=torch.int32)
    if world > 1: dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        report, _ = roots()
        if rank == 0:
            write_json(report / 'blocker.json', {'status': 'INVALID', 'stage': stage, 'update': update,
                'micro': micro, 'error': str(error) if error else 'peer rank failure',
                'traceback': traceback.format_exc() if error else None, 'job_id': os.environ.get('SLURM_JOB_ID')})
        raise RuntimeError(f'synchronized {stage} failure at update={update} micro={micro}: {error}')


def record_blocker(stage, error):
    """Write an INVALID receipt if a gate or formal process exits unexpectedly."""
    if int(os.environ.get('LOCAL_RANK', '0')) != 0: return
    try:
        report, _ = roots()
        blocker = {'status': 'INVALID', 'stage': stage, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'job_id': os.environ.get('SLURM_JOB_ID')}
        write_json(report / 'blocker.json', blocker)
        progress_path = report / 'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            progress.update(status='INVALID', blocker=blocker)
            write_json(progress_path, progress)
    except Exception:
        pass


def _accumulate(dst, src, scale):
    for parameter, gradient in zip(dst, src):
        if gradient is None: continue
        if parameter.grad is None: parameter.grad = gradient.detach().mul(scale)
        else: parameter.grad.add_(gradient.detach(), alpha=scale)


def average_accumulated_gradients(params, device, bucket_bytes=25 * 1024 * 1024):
    """In-place 4-rank mean using only bounded FP32 buckets."""
    _, world = rank_world()
    present = torch.tensor([p.grad is not None for p in params], device=device, dtype=torch.uint8)
    if world > 1: dist.all_reduce(present, op=dist.ReduceOp.MAX)
    flags = present.tolist(); bucket = []; amount = 0
    def flush(ids):
        if not ids: return
        flat = torch.cat([(params[i].grad if params[i].grad is not None else torch.zeros_like(params[i])).reshape(-1) for i in ids])
        if world > 1: dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for i in ids:
            count = params[i].numel()
            params[i].grad = flat[offset:offset+count].view_as(params[i]).clone()
            offset += count
    for i, (p, exists) in enumerate(zip(params, flags)):
        if not exists:
            p.grad = None
            continue
        size = p.numel() * p.element_size()
        if bucket and amount + size > bucket_bytes: flush(bucket); bucket = []; amount = 0
        bucket.append(i); amount += size
    flush(bucket)


def _slot_for_micro(rank, micro):
    if micro not in (0, 1): raise ValueError('exactly two microbatches per update')
    return rank + 4 * micro


def train_update(model, opt, optimizer, manifest, plan_entry, update, slot_states, device,
                 *, training_windows=None, epoch0_refiner=None):
    rank, world = rank_world()
    if world not in (1, 4): raise RuntimeError('updates require one smoke rank or four physical ranks')
    optimizer.zero_grad(set_to_none=True)
    multiplier = lr_multiplier(update)
    for group in optimizer.param_groups: group['lr'] = group['peak_lr'] * multiplier
    forward_exposure = SOURCE_EXPOSURES + 8 * update
    uw = min(8 * update / 200.0, 1.0)
    slots = (_slot_for_micro(rank, 0), _slot_for_micro(rank, 1)) if world == 4 else (0, 4)
    rows, q_stats = [], []
    names_params = sorted(model.named_parameters())
    names = [n for n, _ in names_params]; params = [p for _, p in names_params]
    refiner_params = [p for n, p in names_params if n.startswith('panoptic.output_3d_refine.')]
    for micro, slot in enumerate(slots):
        wi = int(plan_entry['rank_windows'][slot])
        error = None; batch = None
        try:
            restore_slot_rng(slot_states[slot], device)
            identity = manifest['expanded_train_windows'][wi]
            if training_windows is not None: training_windows.append((slot, wi, identity))
            batch = build_batch(opt, identity, device)
        except Exception as exc: error = exc
        synchronized_error(error, device, 'build_batch', update, micro)
        output = metrics = None; error = None
        try:
            output, metrics = model.step_loss(batch, step=forward_exposure, understanding_weight=uw)
            if not torch.isfinite(metrics['loss_recon']).all() or not torch.isfinite(metrics['loss_understanding']).all():
                raise FloatingPointError('nonfinite reconstruction/understanding loss')
            prediction = output['prediction']
            for key in ('gaussians', 'gaussian_membership', 'region_mass', 'semantic_scores', 'alpha', 'q_refined'):
                value = prediction.get(key)
                if torch.is_tensor(value) and not torch.isfinite(value).all(): raise FloatingPointError('nonfinite ' + key)
            q_delta = prediction['q_refined'].detach().float() - prediction['q_base'].detach().float()
            q_base = prediction['q_base'].detach().float()
            q_stats.append((float(q_delta.square().mean().sqrt()), float(q_base.square().mean().sqrt())))
        except Exception as exc: error = exc
        synchronized_error(error, device, 'forward', update, micro)
        error = None
        try:
            g_rec = torch.autograd.grad(metrics['loss_recon'], params, allow_unused=True, retain_graph=uw > 0)
            if any(g is not None and not torch.isfinite(g).all() for g in g_rec): raise FloatingPointError('nonfinite L_rec gradient')
            _accumulate(params, g_rec, .5)
            del g_rec
            if uw > 0:
                g_under = torch.autograd.grad(uw * metrics['loss_understanding'], params, allow_unused=True)
                if any(g is not None and not torch.isfinite(g).all() for g in g_under): raise FloatingPointError('nonfinite L_under gradient')
                for p, name, gradient in zip(params, names, g_under):
                    if gradient is None: continue
                    scale = .5 * (.01 if family(name) == 'reconstruction' else 1.0)
                    if p.grad is None: p.grad = gradient.detach().mul(scale)
                    else: p.grad.add_(gradient.detach(), alpha=scale)
                del g_under
            slot_states[slot] = capture_slot_rng(device)
            scalar = torch.stack((metrics['loss_recon'].detach().float(), metrics['loss_understanding'].detach().float()))
            rows.append(scalar)
        except Exception as exc: error = exc
        synchronized_error(error, device, 'autograd_accumulate', update, micro)
        del output, metrics, batch
    average_accumulated_gradients(params, device)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        synchronized_error(FloatingPointError('nonfinite averaged gradient'), device, 'gradient_average', update)
    refiner_grad_norm = torch.stack([p.grad.detach().float().square().sum() for p in refiner_params if p.grad is not None]).sum().sqrt() if any(p.grad is not None for p in refiner_params) else torch.zeros((), device=device)
    error = None; preclip = float('nan')
    try:
        preclip = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):
            raise FloatingPointError('nonfinite parameter after optimizer step')
    except Exception as exc: error = exc
    synchronized_error(error, device, 'optimizer_step', update)
    local_losses = torch.stack(rows).mean(0)
    if world > 1: dist.all_reduce(local_losses, op=dist.ReduceOp.SUM); local_losses.div_(world)
    q_delta = sum(x[0] for x in q_stats) / 2
    q_base = sum(x[1] for x in q_stats) / 2
    query_stats = torch.tensor([q_delta, q_base], device=device, dtype=torch.float64)
    if world > 1: dist.all_reduce(query_stats, op=dist.ReduceOp.SUM); query_stats.div_(world)
    q_delta, q_base = (float(query_stats[0]), float(query_stats[1]))
    row = {
        'update': update + 1, 'zero_based_update': update,
        'epoch': (update + 1) / UPDATES_PER_EPOCH,
        'new_exposures': 8 * (update + 1), 'total_exposure': SOURCE_EXPOSURES + 8 * (update + 1),
        'forward_exposure': forward_exposure, 'understanding_weight': uw, 'alpha': .01,
        'beta': .1, 'lr_multiplier': multiplier,
        'group_lr': {g['name']: g['lr'] for g in optimizer.param_groups},
        'loss_rec': float(local_losses[0]), 'loss_under': float(local_losses[1]),
        'monitor_total': float(local_losses[0] + uw * local_losses[1]),
        'monitor_total_is_not_gc_equivalent_scalar': True,
        'preclip_norm': preclip, 'clipped': preclip > 1,
        'refiner_avg_preclip_grad_norm': float(refiner_grad_norm),
        'q_refined_minus_q_base_rms': q_delta,
        'q_base_rms': q_base,
        'q_refined_relative_rms': q_delta / max(q_base, 1e-12),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'window_slots': [{'slot': s, 'window_index': int(plan_entry['rank_windows'][s]),
                          'identity': manifest['expanded_train_windows'][int(plan_entry['rank_windows'][s])]}
                         for s in slots],
        'global_update_window_slots': [{'slot': s, 'window_index': int(plan_entry['rank_windows'][s]),
                                        'identity': manifest['expanded_train_windows'][int(plan_entry['rank_windows'][s])]}
                                       for s in range(8)],
    }
    if epoch0_refiner is not None:
        sq = torch.zeros((), device=device, dtype=torch.float64)
        with torch.no_grad():
            for (name, p) in model.named_parameters():
                if name in epoch0_refiner: sq += (p.detach().double() - epoch0_refiner[name].to(device).double()).square().sum()
        row['refiner_parameter_change_from_epoch0_norm'] = float(sq.sqrt())
    return row


def save_checkpoint(path, model, optimizer, update, slot_states, window_counts, config, source, plan_sha, code_sha, device):
    rank, world = rank_world()
    state = capture_all_rng_states(slot_states, device)
    state['rank'] = rank
    state['window_counts'] = np.asarray(window_counts, dtype=np.int64).tolist()
    all_states = [None] * world if rank == 0 else None
    if world > 1: dist.gather_object(state, all_states, dst=0)
    else: all_states = [state]
    error = None
    if rank == 0:
        try:
            payload = {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'logical_slot_rng_states': sorted(all_states, key=lambda x: x['rank']),
                'physical_rank_window_counts': [row['window_counts'] for row in sorted(all_states, key=lambda x: x['rank'])],
                'config': config, 'experiment': EXPERIMENT, 'source_checkpoint': source,
                'source_sha256': SOURCE_SHA256, 'source_manifest_sha256': sha256(SOURCE_MANIFEST),
                'plan_sha256': plan_sha, 'code_sha': code_sha, 'world_size': world,
                'physical_world_size': world, 'logical_global_slots': 8,
                'gradient_accumulation_steps': 2, 'epoch': update // UPDATES_PER_EPOCH,
                'completed_updates': update, 'new_exposures': update * 8,
                'source_exposure': SOURCE_EXPOSURES, 'model_exposure': SOURCE_EXPOSURES + update * 8,
                'alpha': .01, 'actual_training_settings': {'precision': 'FP32', 'tf32': False,
                    'physical_ranks': world, 'per_rank_microbatch': 1, 'accumulation_steps': 2,
                    'global_batch': 8, 'clip_grad_norm': 1.0},
            }
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            torch.save(payload, temporary)
            check = torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
            if check['completed_updates'] != update or check['experiment'] != EXPERIMENT or check['plan_sha256'] != plan_sha:
                raise RuntimeError('temporary checkpoint readback validation failed')
            if update == 0:
                result = model.load_state_dict(check['model'], strict=True)
                if result.missing_keys or result.unexpected_keys:
                    raise RuntimeError(f'epoch0 strict checkpoint reload mismatch: {result}')
                saved, live = check['model'], model.state_dict()
                if saved.keys() != live.keys() or any(not torch.equal(saved[k], live[k].detach().cpu()) for k in saved):
                    raise RuntimeError('epoch0 saved checkpoint does not preserve the complete initialized model')
            del check
            os.replace(temporary, path)
        except Exception as exc:
            error = exc
    synchronized_error(error, device, 'checkpoint_save', update)
