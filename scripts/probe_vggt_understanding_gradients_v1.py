"""Zero-update component attribution on identity-checked fitting checkpoints."""
from __future__ import annotations
import argparse
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import train_vggt_overfit_diagnostic_v1 as diagnostic

PARENT = Path('/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01')
CONDITIONS = [('pred_joint', s) for s in (0, 20, 100, 200, 300)] + [('pred_recon', s) for s in (100, 200, 300)]
COMPONENTS = ('classification', 'thing_bce', 'thing_dice', 'stuff_bce', 'stuff_dice')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def parameter_group(name):
    if name.startswith('vggt_memory_adapter.'): return 'adapter'
    if name in ('anchor_decoder.mu', 'anchor_decoder.rho'): return 'anchor_geometry'
    if name.startswith('activation_head.'): return 'gaussian_head'
    if name.startswith(('gs_tokens', 'gs_tokens_dynamic')): return 'gs_tokens'
    return 'decoder'


def paired_stats(names, rec, under):
    """Statistics on complete flattened parameter families, including unused tensors."""
    squares = {}
    for name, gr, gu in zip(names, rec, under):
        for group in ('all', parameter_group(name)):
            values = squares.setdefault(group, [0., 0., 0.])
            if gr is not None: values[0] += float(gr.flatten().double().square().sum())
            if gu is not None: values[1] += float(gu.flatten().double().square().sum())
            if gr is not None and gu is not None: values[2] += float((gr.flatten().double() * gu.flatten().double()).sum())
    result = {}
    for group, (r2, u2, dot) in squares.items():
        r, u = math.sqrt(r2), math.sqrt(u2)
        result[group] = dict(reconstruction_norm=r, understanding_norm=u, dot=dot,
            cosine=dot / (r * u) if r > 0 and u > 0 else None,
            norm_ratio=u / r if r > 0 else None)
    return result


def gaussian_stats(gradient):
    if gradient is None: return None
    return {key:dict(norm=float(gradient[..., lo:hi].double().norm()),
        max_abs=float(gradient[..., lo:hi].abs().max()))
        for key, lo, hi in [('xyz', 0, 3), ('opacity', 3, 4), ('scale', 4, 7), ('rotation', 7, 11), ('rgb', 11, 14)]}


def pixel_stats(probability, gradient):
    p = probability.detach()
    result = dict(min=float(p.min()), max=float(p.max()),
        exactly_zero_fraction=float((p == 0).float().mean()),
        exactly_one_fraction=float((p == 1).float().mean()),
        near_boundary_fraction=float(((p <= 1e-6) | (p >= 1-1e-6)).float().mean()))
    if gradient is not None:
        g = gradient.detach().abs()
        index = int(g.flatten().argmax())
        result.update(gradient_norm=float(g.double().norm()), max_abs_gradient=float(g.max()),
            probability_at_max_abs_gradient=float(p.flatten()[index]),
            max_gradient_flat_index=index, nonzero_gradient_fraction=float((g > 0).float().mean()))
    return result


def routing_predictions(names, rec, under, native):
    results = {}
    for scale in (1., .01, 0.):
        mixed = native.gc_combine(names, rec, under, adapter_understanding_scale=scale)
        norm2 = sum(float(g.double().square().sum()) for g in mixed if g is not None)
        norm = math.sqrt(norm2)
        coefficient = min(1., 1. / (norm + 1e-6))
        results[str(scale)] = dict(preclip_norm=norm, clip_coefficient=coefficient,
            per_family=paired_stats(names, rec, mixed),
            note='Static mixed gradient, before Adam history; not an optimizer-update prediction')
        del mixed
    return results


def records():
    previous = json.loads((PARENT / 'checkpoint_validation.json').read_text())
    known = {(r['arm'], r['step']):r for r in previous['records']}
    selected = []
    for arm, step in CONDITIONS:
        if (arm, step) in known:
            selected.append(known[arm, step])
        else:
            path = diagnostic.SOURCE.parents[1] / 'vggt_overfit_diagnostic_v1' / arm / f'checkpoint_update_{step:04d}.pt'
            selected.append(dict(arm=arm, step=step, path=str(path), sha256=diagnostic.digest(path)))
    return selected


def run(root):
    import torch.distributed as dist
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _restore_checkpoint
    from scripts.eval_vggt_overfit_diagnostic_v1 import validate_clock
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('four windows require four GPU workers')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    shard = root / f'rank{rank:02d}'
    if shard.exists(): raise ValueError('probe shard already exists; preserve evidence')
    shard.mkdir(parents=True)
    if diagnostic.digest(diagnostic.SOURCE) != diagnostic.SOURCE_SHA:
        raise ValueError('source checkpoint SHA mismatch')
    if diagnostic.digest(diagnostic.SOURCE.parent / 'adapted_vggt.pt') != native.CONTINUATION_VGGT_SHA:
        raise ValueError('adapted VGGT SHA mismatch')
    if rank == 0:
        selected = records()
        # Hash every actual selected file, including checkpoints already audited.
        for record in selected:
            if diagnostic.digest(record['path']) != record['sha256']:
                raise ValueError('selected checkpoint bytes changed')
        source_state = torch.load(diagnostic.SOURCE,map_location='cpu',mmap=True,weights_only=False)['model']
        for record in selected:
            if record['step'] == 20:
                blob = torch.load(record['path'],map_location='cpu',mmap=True,weights_only=False)
                validate_clock(blob,record['arm'],record['step'])
                if set(blob['model']) != set(source_state) or any(not torch.equal(blob['model'][k],v)
                    for k,v in source_state.items() if not k.startswith(diagnostic.TRAIN_PREFIXES)):
                    raise ValueError('unaudited step20 frozen state differs from source')
                del blob
        del source_state
    else: selected = None
    shared = [selected]
    dist.broadcast_object_list(shared, src=0)
    selected = shared[0]
    native.seed_everything(42)
    os.environ['VGGT_HF_REVISION'] = diagnostic.REVISION
    model, opt, _, _ = native.build_model(preserve_fp32_aggregator=True)
    model.to(device=device, dtype=torch.float32)
    _restore_checkpoint(model, diagnostic.SOURCE, diagnostic.REVISION, diagnostic.digest(native.MANIFEST))
    model.frozen_vggt.set_reconstruction_adaptation(False, keep_fp32=True)
    native.restore_adapted_vggt(model, diagnostic.SOURCE.parent / 'adapted_vggt.pt', native.CONTINUATION_VGGT_SHA)
    diagnostic.configure_trainable(model)
    windows = json.loads((REPO / 'docs/vggt_overfit_cohort_v1.json').read_text())['windows']
    batch_cpu, cache_cpu, proof = diagnostic.prepare_window(model, opt, windows[rank], device)
    model.frozen_vggt.cpu()
    model.understanding.encoder.cpu()
    model.understanding.adapter.cpu()
    model.understanding.mask2former.pixel_decoder.cpu()
    model.understanding.mask2former.transformer_module.cpu()
    model.understanding.mask_embedder.to(device)
    named = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
    names, params = [n for n, _ in named], [p for _, p in named]
    batch, cached = diagnostic.tree_to(batch_cpu, device), diagnostic.tree_to(cache_cpu, device)
    del batch_cpu, cache_cpu
    gc.collect(); torch.cuda.empty_cache()
    rows, global_rows = [], []
    for record in selected:
        blob = torch.load(record['path'], map_location='cpu', mmap=True, weights_only=False)
        validate_clock(blob, record['arm'], record['step'])
        matching = [p for r in blob['config']['rank_preparation'] for p in r['proofs']
                    if p['window']['source_index'] == windows[rank]['source_index']]
        if len(matching) != 4 or any(p['input_rgb_sha256'] != proof['input_rgb_sha256'] for p in matching):
            raise ValueError('probe provider pixels differ from training')
        native.restore_model_state_strict(model, blob['model'])
        del blob
        native.seed_everything(42 + rank)
        model.understanding_step = 50064 + 16 * record['step']
        versions = {n:p._version for n,p in model.named_parameters()}
        components, tensors = {}, {}
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            output, metrics = diagnostic.diagnostic_forward(model, batch, cached, 'pred_joint',
                understanding_components=components, probe_tensors=tensors)
        under = metrics['loss_understanding']
        total = sum(components.values())
        if not torch.allclose(total, under, rtol=1e-6, atol=1e-7):
            raise ValueError('weighted understanding components differ from native total')
        generated = output['prediction']
        rec_gaussians = tensors['reconstruction_final_gaussians']
        under_gaussians = tensors['understanding_gaussians']
        if not torch.equal(rec_gaussians.detach(), under_gaussians.detach()):
            raise ValueError('readout and final reconstruction Gaussians differ')
        regions = generated['region_mass']
        targets = params + [rec_gaussians, under_gaussians, regions]
        local_rec = global_rec = global_under = None
        component_sum = [None] * len(params)
        condition = dict(arm=record['arm'], step=record['step'], scene=windows[rank]['scene'],
            source_index=windows[rank]['source_index'], checkpoint_sha256=record['sha256'],
            native_understanding_loss=float(under.detach()), reconstruction_loss=float(metrics['loss_recon'].detach()),
            weighted_components={k:float(v.detach()) for k,v in components.items()},
            matching_pairs=[dict(query=q.tolist(), gt_target=k.tolist()) for q,k in generated['final_pairs']],
            alpha_mean=float(generated['alpha'].detach().mean()),
            low_alpha_fraction=float((generated['alpha'].detach() < 1e-6).float().mean()), component_gradients={})
        global_condition = dict(arm=record['arm'], step=record['step'], component_gradients={})
        losses = [('reconstruction', metrics['loss_recon']), ('understanding', under)] + list(components.items())
        for key, loss in losses:
            error = None
            try:
                gradient = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
                if any(g is not None and not torch.isfinite(g).all() for g in gradient):
                    raise FloatingPointError('nonfinite gradient in ' + key)
                local_gradient = [None if g is None else g.detach().cpu() for g in gradient[:len(params)]]
                condition['component_gradients'][key] = dict(
                    per_family=paired_stats(names, local_gradient if local_rec is None else local_rec, local_gradient),
                    final_gaussian_gradient=gaussian_stats(gradient[-3] if key == 'reconstruction' else gradient[-2]),
                    region_probability=pixel_stats(regions, gradient[-1]))
                del gradient
            except Exception as exc: error = exc
            native.synchronized_error(error, device, 'gradient_component_' + key, record['step'])
            native.install_gradients(params, local_gradient, world_size=4)
            averaged = [None if p.grad is None else p.grad.detach().cpu() for p in params]
            for p in params: p.grad = None
            if key == 'reconstruction':
                local_rec, global_rec = local_gradient, averaged
            else:
                global_condition['component_gradients'][key] = paired_stats(names, global_rec, averaged)
                if key == 'understanding':
                    global_under = averaged
                    global_condition['routing_predictions'] = routing_predictions(names, global_rec, global_under, native)
                else:
                    for i, g in enumerate(averaged):
                        if g is not None:
                            if component_sum[i] is None: component_sum[i] = g.clone()
                            else: component_sum[i].add_(g)
            if key != 'reconstruction': del local_gradient
            del averaged
        error2 = sum(float((a.double() - b.double()).square().sum())
            for a,b in zip(component_sum, global_under) if a is not None and b is not None)
        under2 = sum(float(g.double().square().sum()) for g in global_under if g is not None)
        relative_error = math.sqrt(error2) / max(math.sqrt(under2), 1e-12)
        if any((a is None) != (b is None) for a,b in zip(component_sum, global_under)) or relative_error > 1e-3:
            raise ValueError('component gradients do not recover native total gradient')
        global_condition['component_sum_relative_gradient_error'] = relative_error
        if any(p._version != versions[n] or p.grad is not None for n,p in model.named_parameters()):
            raise ValueError('zero-update probe modified model or retained parameter gradients')
        rows.append(condition)
        if rank == 0:
            global_rows.append(global_condition)
            write(root / 'progress.json', dict(completed_conditions=len(rows), total_conditions=len(selected),
                optimizer_updates=0, latest=dict(arm=record['arm'], step=record['step'])))
            print(json.dumps(dict(arm=record['arm'], step=record['step'],
                globally_averaged_families=global_condition['component_gradients']['understanding'])), flush=True)
        del output, metrics, generated, under, total, components, tensors, targets, losses, loss
        del rec_gaussians, under_gaussians, regions, local_rec, global_rec, global_under, component_sum
        gc.collect(); torch.cuda.empty_cache()
    write(shard / 'COMPLETE.json', dict(status='ZERO_UPDATE_GRADIENT_PROBE_COMPLETED', optimizer_updates=0,
        rank=rank, execution_git_sha=os.environ['TASK_CODE_SHA'], native_replay=proof, rows=rows))
    dist.barrier()
    if rank == 0:
        write(root / 'global_gradient_statistics.json', dict(status='ZERO_UPDATE_GRADIENT_PROBE_COMPLETED',
            optimizer_updates=0, execution_git_sha=os.environ['TASK_CODE_SHA'], rows=global_rows,
            averaging='Exact arithmetic mean of parameter gradients across four fixed windows; before clip/Adam',
            gaussian_gradient_scope='Local final-layer output gradients; reconstruction includes its native layer weight',
            frozen_modules='same as parent four-window fitting diagnostic', checkpoint_records=selected))
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)


if __name__ == '__main__': main()
