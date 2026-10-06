"""Task-local builder, locked-plan checks, and optimizer helpers."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts import object_locus_mask_guided_runtime as locked
from scripts.object_locus_v3_set_runtime import write_json, jsonable, sha256_file
from tokengs.models.object_locus_region_class import (
    RegionClassObjectLocusPanopticV1Recon, add_region_class_projection,
)

REPORT_ROOT = Path('/space/mawb/ssst/group_plus/object_locus_region_class_v1')
RUN_ROOT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_region_class_v1')
CM_CONTROL = Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1/control')
SOURCE_PROVENANCE = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/weights_provenance.json')
ARMS = ('control', 'region_class')
LAYERS = ('L6', 'L8', 'L10', 'L12')
TOTAL_UPDATES = 448
TOTAL_EXPOSURES = 3584
BASE_COMMIT = 'd606e194d358727fefd7daa6848268e14fea3347'


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def arm_dirs(arm):
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    reports, run = REPORT_ROOT / arm, RUN_ROOT / arm
    reports.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    return reports, run


def state_sha(state, exclude_region=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if exclude_region and name == 'panoptic.region_class_proj.weight':
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(json.dumps(list(tensor.shape)).encode('ascii'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_model(arm, device='cpu', *, report=True):
    """Use the original fresh control builder, then install R's sole module."""
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    model, opt = locked.build_model('control', device, report=False)
    baseline_state = model.state_dict()
    pre_params = sum(p.numel() for p in model.parameters())
    if pre_params != 572_432_103:
        raise RuntimeError(f'baseline parameter count mismatch: {pre_params}')
    if arm == 'region_class':
        # Keep the baseline architecture object and all existing parameter values;
        # only dispatch _readout through the task-local subclass.
        model.__class__ = RegionClassObjectLocusPanopticV1Recon
        add_region_class_projection(model.panoptic, seed=31417)
        if sum(p.numel() for p in model.parameters()) != 572_497_639:
            raise RuntimeError('region-class parameter count must be 572,497,639')
        if model.panoptic.region_class_proj.weight.count_nonzero().item() != 0:
            raise RuntimeError('region class projection must be zero initialized')
        if model.panoptic.region_class_proj.weight.shape != (256, 256):
            raise RuntimeError('region class projection shape mismatch')
        after = model.state_dict()
        if set(after) != set(baseline_state) | {'panoptic.region_class_proj.weight'}:
            raise RuntimeError('R state dict must add exactly one projection tensor')
        for key, value in baseline_state.items():
            if not torch.equal(value, after[key]):
                raise RuntimeError(f'R initialization changed common state: {key}')
    elif arm != 'control':
        raise ValueError(f'unknown arm: {arm}')
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError('all baseline and new parameters must remain trainable')

    rank, world = rank_world()
    if world > 1:
        # Explicitly broadcast the complete task model from rank 0. The legacy
        # eight-rank builder does not cover this two-rank task configuration.
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)

    if report and rank == 0:
        reports, _ = arm_dirs(arm)
        common = state_sha(model.state_dict(), exclude_region=(arm == 'region_class'))
        contract = dict(arm=arm, base_commit=BASE_COMMIT, global_seed=42, object_seed=31415,
            region_projection_seed=31417 if arm == 'region_class' else None,
            common_state_sha256=common, common_parameter_numel=572_432_103,
            added_parameter_numel=65_536 if arm == 'region_class' else 0,
            total_parameter_numel=sum(p.numel() for p in model.parameters()),
            projection_zero=(arm == 'region_class'),
            pretrained_transfer_counts={'reconstruction': 450, 'mast3r_encoder': 292,
                'adapter': 187, 'mask_decoder': 326, 'mast3r_excluded': 725},
            all_parameters_trainable=True, state_dict_keys=len(model.state_dict()))
        if arm == 'region_class':
            control_init = CM_CONTROL / 'initialization_contract.json'
            if control_init.is_file():
                old = json.loads(control_init.read_text())
                old_sha = old.get('initial_state_sha256')
                if old_sha and old_sha != common:
                    raise RuntimeError('shared fresh initialization differs from saved C initialization')
                contract['mask_guided_control_initialization_sha256'] = old_sha
        write_json(reports / 'initialization_contract.json', contract)
    return model, opt


def build_optimizer(model, reports=None):
    optimizer = locked.build_optimizer(model)
    names = {id(p): n for n, p in model.named_parameters()}
    found = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if names[id(param)] == 'panoptic.region_class_proj.weight':
                found.append(group)
    if model.__class__ is RegionClassObjectLocusPanopticV1Recon:
        if len(found) != 1 or found[0]['name'] != 'new_decay' or found[0]['peak_lr'] != 1e-4 or found[0]['weight_decay'] != 0.05:
            raise RuntimeError('region_class_proj must be in new_decay at peak LR 1e-4 / WD 0.05')
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError('optimizer parameter coverage mismatch')
    if reports is not None:
        names_by_id = {id(p): n for n, p in model.named_parameters()}
        rows = []
        for g in optimizer.param_groups:
            rows.append(dict(name=g['name'], peak_lr=g['peak_lr'], weight_decay=g['weight_decay'],
                tensor_count=len(g['params']), numel=sum(p.numel() for p in g['params']),
                param_names=[names_by_id[id(p)] for p in g['params']]))
        write_json(Path(reports) / 'optimizer_groups.json', dict(groups=rows, all_trainable_once=True))
    return optimizer


def manifest_and_plan():
    manifest, plan = locked.manifest_and_plan()
    fixed_manifest = json.loads((CM_CONTROL / 'data_manifest.json').read_text())
    fixed_plan = json.loads((CM_CONTROL / 'training_plan.json').read_text())
    if manifest != fixed_manifest or plan != fixed_plan:
        raise RuntimeError('manifest/plan differ from the locked C artifacts')
    if len(plan['entries']) != 448 or len(manifest['train_all56']) != 56:
        raise RuntimeError('fixed training size mismatch')
    counts = np.zeros(56, dtype=np.int64)
    for epoch in range(64):
        expected = np.random.default_rng(42 + epoch).permutation(56).tolist()
        entries = plan['entries'][epoch * 7:(epoch + 1) * 7]
        order = [i for row in entries for i in row['rank_windows']]
        if order != expected:
            raise RuntimeError(f'fixed sampling order mismatch at epoch {epoch}')
        counts[order] += 1
    if not np.all(counts == 64):
        raise RuntimeError('each fixed window must have 64 exposures')
    return manifest, plan


def train_one_step(model, optimizer, batch, update):
    return locked.train_one_step(model, optimizer, batch, update)


def train_accumulated_step(model, optimizer, batches, update):
    """One global update from four local micro-batches, then one rank mean/clip/step."""
    device = next(model.parameters()).device
    rank, world = rank_world()
    if world != 2:
        raise RuntimeError(f'accumulated paired update requires world_size=2, got {world}')
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr'] * locked.lr_multiplier(update)
    exposure = 8 * int(update)
    understanding_weight = min(exposure / 200, 1)
    named = sorted(model.named_parameters())
    names = [name for name, _ in named]
    params = [param for _, param in named]
    accumulated = [None] * len(params)
    metric_sums = {key: 0.0 for key in ('loss', 'loss_recon', 'loss_understanding')}
    latest_stats = {}
    started = __import__('time').perf_counter()

    def family(name):
        return 'pretrained' if name.startswith('understanding.') else 'new' if name.startswith('panoptic.') else 'reconstruction'

    micro_count = 0
    for micro, batch in enumerate(batches):
        micro_count += 1
        output = metrics = None
        error = None
        try:
            output, metrics = model.step_loss(batch, step=exposure, understanding_weight=understanding_weight)
            if not all(torch.isfinite(metrics[key]).all() for key in ('loss_recon', 'loss_understanding')):
                raise FloatingPointError('nonfinite micro-batch loss')
            pred = output['prediction']
            for key in ('gaussians', 'gaussian_membership', 'thing_logits19'):
                if key in pred and not torch.isfinite(pred[key]).all():
                    raise FloatingPointError(f'nonfinite micro-batch {key}')
            if not torch.isfinite(pred['render']['images_pred']).all():
                raise FloatingPointError('nonfinite micro-batch RGB')
        except Exception as exc:
            error = exc
        locked.synchronize_failure(error, device, f'accumulation forward {micro}', update)
        for key in metric_sums:
            metric_sums[key] += float(metrics[key].detach()) / 4.0
        error = None
        try:
            scale = 0.25
            rec = torch.autograd.grad(metrics['loss_recon'] * scale, params,
                allow_unused=True, retain_graph=understanding_weight > 0)
            for index, gr in enumerate(rec):
                if gr is None:
                    continue
                accumulated[index] = (gr if accumulated[index] is None else accumulated[index].add_(gr))
            del rec
            if understanding_weight > 0:
                under = torch.autograd.grad(metrics['loss_understanding'] * (understanding_weight * scale), params,
                    allow_unused=True)
                for index, (param, name, gu) in enumerate(zip(params, names, under)):
                    if gu is None:
                        continue
                    if family(name) == 'reconstruction':
                        gu = gu.mul_(0.01)
                    accumulated[index] = gu if accumulated[index] is None else accumulated[index].add_(gu)
                del under
        except Exception as exc:
            error = exc
        locked.synchronize_failure(error, device, f'accumulation backward {micro}', update)
        if getattr(model, '_last_region_stats', None):
            latest_stats = dict(model._last_region_stats)
        del metrics, output, batch

    if micro_count != 4:
        raise RuntimeError(f'two-rank global batch requires four local micro-batches, got {micro_count}')

    unused = locked.average_gradients(params, accumulated)
    error = None
    try:
        if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in params):
            raise FloatingPointError('nonfinite accumulated/global-mean gradient')
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
    except Exception as exc:
        error = exc
        norm = torch.tensor(float('nan'), device=device)
    locked.synchronize_failure(error, device, 'accumulated clip/optimizer_step', update)
    reduced = torch.tensor([metric_sums[k] for k in ('loss', 'loss_recon', 'loss_understanding')],
        dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world
    row = dict(zip(('loss', 'loss_recon', 'loss_understanding'), map(float, reduced.cpu().tolist())))
    row.update(update=int(update) + 1, exposure=8 * (int(update) + 1), forward_exposure=exposure,
        understanding_weight=understanding_weight, preclip_norm=float(norm),
        lr_multiplier=locked.lr_multiplier(update), beta=0.1 * min(exposure / 1000, 1),
        accumulated_micro_batches=4, global_batch=8, rank=rank, elapsed_seconds=__import__('time').perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device), unused_count=len(unused))
    row.update(latest_stats)
    return row


def build_batch(opt, window, device):
    return locked.build_batch(opt, window, device)


def rank_micro_windows(entry, rank):
    windows = entry.get('rank_windows')
    if rank not in (0, 1) or not isinstance(windows, list) or len(windows) != 8:
        raise ValueError('locked global batch must contain eight windows for a two-rank plan')
    return [windows[2 * offset + rank] for offset in range(4)]


def capture_rng():
    from scripts.object_locus_v3_set_runtime import capture_rng as capture
    return capture()


def restore_rng(state):
    from scripts.object_locus_v3_set_runtime import restore_rng as restore
    return restore(state)


def code_sha():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def source_hashes():
    paths = ('tokengs/models/object_locus_region_class.py', 'scripts/object_locus_region_class_runtime.py',
        'scripts/train_object_locus_region_class.py', 'scripts/eval_object_locus_region_class.py')
    return {p: sha256_file(REPO / p) for p in paths if (REPO / p).is_file()}


def provenance_hashes():
    data = json.loads(SOURCE_PROVENANCE.read_text())
    return {name: {'path': row['path'], 'sha256': row['sha256']} for name, row in data['weights'].items()}
