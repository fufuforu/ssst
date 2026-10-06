"""Fixed runtime and task-specific builders for the paired mask-route study."""
from __future__ import annotations

import hashlib
import json
import math
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
from scripts import object_locus_panoptic_v1_runtime as base
from scripts.object_locus_v3_set_runtime import build_batch, seed_everything, capture_rng, restore_rng, write_json, jsonable, sha256_file
from scripts.train_object_locus_v3_set import build_manifest
from tokengs.models.object_locus_mask_guided import MaskGuidedRegisteredObjectLayer
from tokengs.models.object_locus_panoptic_v1_controller import RegisteredObjectLayer
from tokengs.models.object_locus_panoptic_v1_pretrained import MAST, PANOPTIC

REPORT_ROOT = Path('/space/mawb/ssst/group_plus/object_locus_mask_guided_v1')
RUN_ROOT = Path('/space/mawb/ssst/workspace_group_plus/object_locus_mask_guided_v1')
PROVENANCE = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/weights_provenance.json')
SOURCE_MANIFEST = Path('/space/mawb/ssst/group_plus/object_locus_v2_1/data_manifest.json')
SOURCE_MANIFEST_SHA = 'ebed1a133d64ed38ef7afce17aaaf27bbe65c6b0edea950c229b5d1e4ff77bf0'
SCENES = ('scene0000_00', 'scene0003_02', 'scene0009_00', 'scene0013_01',
          'scene0018_00', 'scene0024_02', 'scene0031_00', 'scene0035_00')
EPOCHS, WINDOWS, UPDATES_PER_EPOCH = 64, 56, 7
TOTAL_UPDATES, TOTAL_EXPOSURES = 448, 3584
CHECKPOINT_EPOCHS = (0, 8, 16, 32, 64)
BASE_COMMIT = '7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3'


def arm_dirs(arm):
    if arm not in ('control', 'mask_guided'):
        raise ValueError(f'unknown arm: {arm}')
    reports, run = REPORT_ROOT / arm, RUN_ROOT / arm
    reports.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    return reports, run


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def build_model(arm, device='cpu', *, report=True):
    """Use the original strict pretrained path, then class-swap M layers in place."""
    rank, world = rank_world()
    provenance = None
    asset_error = None
    if rank == 0:
        try:
            if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA:
                raise RuntimeError('locked source manifest SHA mismatch')
            provenance = json.loads(PROVENANCE.read_text())
            weights = provenance['weights']
            for family, path, expected in (
                ('reconstruction', base.PRETRAINED, base.PRETRAINED_SHA),
                ('mast3r', MAST, 'e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2'),
                ('panoptic', PANOPTIC, '3f7d5d1a065913bfc0686942d979ba8b28f0230fec9a8eea1d4214d2e603eb20'),
            ):
                if weights[family]['sha256'] != expected or Path(weights[family]['path']) != path or sha256_file(path) != expected:
                    raise RuntimeError(f'{family} provenance disagrees with the original panoptic runtime')
        except Exception as exc:
            asset_error = f'{type(exc).__name__}: {exc}'
    status = [asset_error]
    if world == 8:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])
    provenance = json.loads(PROVENANCE.read_text())
    weights = provenance['weights']
    model, opt = base.build_model(device, report=False)
    if arm == 'mask_guided':
        for layer_id in ('L6', 'L8', 'L10', 'L12'):
            layer = model.panoptic.layers[layer_id]
            if type(layer) is not RegisteredObjectLayer:
                raise RuntimeError(f'{layer_id} is not the baseline registered layer')
            layer.__class__ = MaskGuidedRegisteredObjectLayer
    elif arm != 'control':
        raise ValueError(f'unknown arm: {arm}')
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError('all visual and object parameters must remain trainable')
    rank, world = rank_world()
    if world == 8:
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)
    if report and rank == 0:
        reports, _ = arm_dirs(arm)
        keys = {k: (list(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            tensor=value.detach().cpu().contiguous()
            digest.update(name.encode('utf-8'));digest.update(str(tensor.dtype).encode('ascii'))
            digest.update(json.dumps(list(tensor.shape)).encode('ascii'))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        state_digest=digest.hexdigest()
        if arm == 'mask_guided':
            control_path=REPORT_ROOT/'control'/'initialization_contract.json'
            control=json.loads(control_path.read_text())
            if control.get('initial_state_sha256') != state_digest:
                raise RuntimeError('fresh control and mask-guided model initialization differs')
        write_json(reports / 'initialization_contract.json', dict(
            arm=arm, global_seed=42, object_seed=31415, parameter_count=sum(p.numel() for p in model.parameters()),
            state_keys=len(keys), state_signature={k: [s, d] for k, (s, d) in keys.items()},
            initial_state_sha256=state_digest,
            reconstruction_step=47500,
            weights=weights, strict_counts={'reconstruction': 450, 'mast3r_encoder': 292,
                'adapter': 187, 'mask_decoder': 326, 'mast3r_excluded': 725},
            all_trainable=True, injected_layer_types={n: type(model.panoptic.layers[n]).__name__ for n in ('L6','L8','L10','L12')}))
    return model, opt


def build_optimizer(model, reports=None):
    # Preserve the exact panoptic grouping and no-decay rules.
    groups = []
    no_decay = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm, torch.nn.Embedding)):
            no_decay.update(name + '.' + n for n, _ in module.named_parameters(recurse=False))
    no_decay.add('panoptic.stuff_seed')
    no_decay.update(n for n, p in model.named_parameters() if n.endswith('.level_embed') or getattr(p, '_no_weight_decay', False))
    def family(name):
        return 'pretrained' if name.startswith('understanding.') else 'new' if name.startswith('panoptic.') else 'reconstruction'
    peaks = {'reconstruction': 1e-6, 'pretrained': 1e-5, 'new': 1e-4}
    for fam in ('reconstruction', 'pretrained', 'new'):
        for decay in (False, True):
            chosen = [(n, p) for n, p in sorted(model.named_parameters()) if family(n) == fam and
                      ((fam != 'reconstruction' and p.ndim > 1 and not n.endswith('.bias') and n not in no_decay) == decay)]
            if chosen:
                groups.append(dict(name=f'{fam}_'+('decay' if decay else 'nodecay'), params=[p for _, p in chosen],
                    param_names=[n for n, _ in chosen], lr=peaks[fam], peak_lr=peaks[fam],
                    weight_decay=0.05 if decay else 0.0))
    ids = [id(p) for g in groups for p in g['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError('optimizer has duplicate or missing parameters')
    if reports is not None:
        write_json(reports / 'optimizer_groups.json', dict(groups=[{k: v for k, v in g.items() if k != 'params'} |
            dict(tensor_count=len(g['params']), numel=sum(p.numel() for p in g['params'])) for g in groups],
            all_trainable_once=True, no_decay_embeddings=sorted(no_decay)))
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def lr_multiplier(update):
    t = int(update) + 1
    if not 1 <= t <= TOTAL_UPDATES:
        raise ValueError('update index outside the fixed 448-update plan')
    if t <= 25:
        return t / 25
    return 0.1 + 0.9 * (1 + math.cos(math.pi * (t - 25) / (TOTAL_UPDATES - 25))) / 2


def synchronize_failure(error, device, stage, update):
    failed=torch.tensor(int(error is not None),device=device,dtype=torch.int32)
    rank,world=rank_world()
    if world>1:
        dist.all_reduce(failed,op=dist.ReduceOp.MAX)
    if int(failed):
        detail=f'{type(error).__name__}: {error}' if error is not None else 'another rank failed'
        raise RuntimeError(f'{stage} failure at global update {update}: {detail}')


def manifest_and_plan():
    manifest = build_manifest()
    if manifest['source_sha256'] != SOURCE_MANIFEST_SHA or len(manifest['train_all56']) != 56:
        raise RuntimeError('fixed V3-set window manifest contract failed')
    entries = []
    for epoch in range(EPOCHS):
        perm = np.random.default_rng(42 + epoch).permutation(56)
        for update_in_epoch in range(7):
            wi = [int(x) for x in perm[update_in_epoch * 8:(update_in_epoch + 1) * 8]]
            entries.append(dict(update=len(entries), epoch=epoch, windows=wi,
                rank_windows=wi, identities=[manifest['train_all56'][i] for i in wi]))
    return manifest, dict(seed=42, epochs=64, global_updates=448, global_batch=8,
        windows=56, exposures=3584, exposures_per_window=64, entries=entries)


def train_one_step(model, optimizer, batch, update, *, check_rec_under=False):
    # Same two independent loss gradients, GC weighting and explicit averaging as V1.
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr'] * lr_multiplier(update)
    exposure = 8 * int(update)
    understanding_weight = min(exposure / 200, 1)
    output=metrics=None;error=None
    try:
        output, metrics = model.step_loss(batch, step=exposure, understanding_weight=understanding_weight)
        for key in ('loss_recon', 'loss_understanding'):
            if not torch.isfinite(metrics[key]).all():
                raise FloatingPointError(f'nonfinite {key}')
        if not torch.isfinite(output['prediction']['gaussians']).all() or not torch.isfinite(output['prediction']['render']['images_pred']).all():
            raise FloatingPointError('nonfinite Gaussian or rendered RGB output')
    except Exception as exc:
        error=exc
    synchronize_failure(error,device,'forward',update)
    named = sorted(model.named_parameters())
    names = [n for n, _ in named]
    params = [p for _, p in named]
    error=None;grads=None
    try:
        rec = torch.autograd.grad(metrics['loss_recon'], params, allow_unused=True, retain_graph=understanding_weight > 0)
        under = (torch.autograd.grad(understanding_weight * metrics['loss_understanding'], params, allow_unused=True)
                 if understanding_weight > 0 else (None,) * len(params))
        def family(name):
            return 'pretrained' if name.startswith('understanding.') else 'new' if name.startswith('panoptic.') else 'reconstruction'
        grads = [None if gr is None and gu is None else
            ((gr if gr is not None else torch.zeros_like(p)) +
             (0.01 if family(n) == 'reconstruction' else 1.0) * (gu if gu is not None else torch.zeros_like(p)))
            for p, n, gr, gu in zip(params, names, rec, under)]
        del rec, under
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('nonfinite local gradient')
    except Exception as exc:
        error=exc
    synchronize_failure(error,device,'backward',update)
    unused = base.average_gradients(params, grads)
    error=None;norm=None
    try:
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('nonfinite averaged gradient')
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    except Exception as exc:
        error=exc
    synchronize_failure(error,device,'clip',update)
    error=None
    try:
        optimizer.step()
    except Exception as exc:
        error=exc
    synchronize_failure(error,device,'optimizer_step',update)
    row = {k: float(v.detach()) for k, v in metrics.items() if torch.is_tensor(v) and v.ndim == 0}
    row.update(update=int(update) + 1, exposure=8 * (int(update) + 1),
        preclip_norm=float(norm), lr_multiplier=lr_multiplier(update),
        allocated=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        peak_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        unused_count=len(unused), beta=0.1 * min(exposure / 1000, 1))
    return output, jsonable(row)


def average_gradients(*args, **kwargs):
    return base.average_gradients(*args, **kwargs)


def code_sha():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def source_hashes():
    files = ('tokengs/models/object_locus_mask_guided.py', 'scripts/object_locus_mask_guided_runtime.py',
             'scripts/train_object_locus_mask_guided.py', 'scripts/eval_object_locus_mask_guided.py')
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in files if (REPO / f).is_file()}
