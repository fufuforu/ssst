"""Shared frozen-C streaming cache; independent quality/objectness heads and official128 eval."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.train_vggt_classification_readback_v1 import SOURCE, SOURCE_SHA, TRAIN_COHORT, HELDOUT, build_base, configure, apply_patch_state, digest
from scripts.eval_vggt_classification_full312_v1 import TRAIN, OLD_C, verify_pair, verify_segmentation
from scripts.eval_vggt_joint_short_controls_v1 import write
from scripts.eval_vggt_classification_readback_v1 import check_official
# Official scoring environment intentionally excludes the reconstruction package
# dependencies. Load this pure-Torch head file without tokengs.models.__init__.
import importlib.util
_spec = importlib.util.spec_from_file_location('readout_probe_scalar_v1', REPO / 'tokengs/models/object_locus_readout_heads_v1.py')
_readouts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_readouts)
ScalarReadout = _readouts.ScalarReadout
diagnostic_loss = _readouts.diagnostic_loss
candidate_eligibility = _readouts.candidate_eligibility
candidate_probabilities = _readouts.candidate_probabilities

ARMS = ('quality', 'objectness')
FIELDS = ('features', 'class_logits', 'quality_target', 'quality_valid', 'object_target')


def read(path):
    return json.loads(path.read_text())


def identity():
    proof = read((TRAIN / 'classification_patch_update0300.pt').with_suffix('.json'))
    assert digest(TRAIN / 'classification_patch_update0300.pt') == proof['sha256']
    assert proof['source_checkpoint_sha256'] == SOURCE_SHA
    assert read(TRAIN / 'COMPLETE.json')['completed_updates'] == 300
    return dict(source_checkpoint_sha256=SOURCE_SHA, C_classification_patch_sha256=proof['sha256'],
                execution_sha=os.environ['TASK_CODE_SHA'], adapter=.01, feedback=.05)


def workers():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    assert world in (4, 8)
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    assert '3090' in torch.cuda.get_device_name(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    return rank, world, torch.device('cuda', local)


def gather(value, world):
    answer = [None] * world
    dist.all_gather_object(answer, value)
    return answer


def frozen_C(device):
    model, opt, _, _, initial = build_base(device)
    configure(model, 'object_support', initial)
    patch = torch.load(TRAIN / 'classification_patch_update0300.pt', map_location='cpu', mmap=True, weights_only=False)
    apply_patch_state(model, patch, 'object_support')
    del patch, initial
    model.eval().requires_grad_(False)
    model.understanding_step = 35776
    model.panoptic.class_ln.register_forward_hook(lambda module, inputs, output: setattr(model, '_readout_feature', output.detach().float()))
    return model, opt


def core_versions(model):
    return ({n: p._version for n, p in model.named_parameters()}, {n: p._version for n, p in model.named_buffers()})


def assert_core(model, versions):
    assert all(not p.requires_grad and p.grad is None and p._version == versions[0][n] for n, p in model.named_parameters())
    assert all(p._version == versions[1][n] for n, p in model.named_buffers())


def extract(model, opt, window, device):
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render, _official_export_view
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    from tokengs.models.object_locus_v3_set_loss import final_hungarian
    # Pinned windows have identical input ownership at4/8 workers; per-window
    # seed also makes any provider/calibration sampling replayable for cache use.
    from scripts.object_locus_frozen_vggt_posefree_runtime import seed_everything
    seed_everything(42 + window['source_index'])
    batch = build_batch(opt, window, device)
    with torch.no_grad():
        output = generate_and_render(model, batch['images_input'], batch['images_all'])
        features = model._readout_feature
        logits = output['generated']['states'][-1]['thing_logits19'].detach().float()
        view = _official_export_view(output['render'])
        assert torch.allclose(model.panoptic.class_head(features), logits, atol=1e-6, rtol=1e-6)
        targets, pairs = final_hungarian(output['generated'], batch)
        object_target = logits.new_zeros((1, 100))
        object_target[0, pairs[0][0]] = 1
        sem, ins, _ = assemble_panoptic(view)
        masks = (ins[:2, None] == torch.arange(1, 101, device=device)[None, :, None, None]).permute(1, 0, 2, 3).reshape(100, -1)
        gt = targets['gt_pixel_masks'][0].flatten(1)
        classes = targets['gt_classes'][0]
        predicted = view['p_class'][0, :, :18].argmax(-1) + 2
        quality_target = logits.new_zeros((1, 100))
        if len(gt):
            intersection = masks.float() @ gt.float().T
            union = masks.sum(-1)[:, None] + gt.sum(-1)[None] - intersection
            iou = intersection / union.clamp_min(1)
            iou = torch.where(predicted[:, None] == classes[None], iou, torch.zeros_like(iou))
            quality_target[0] = iou.max(-1).values
        quality_valid = masks.any(-1)[None]
        good = logits.new_zeros((1, 100), dtype=torch.bool)
        qi, ki = pairs[0]
        if len(qi):
            raw = ((view['region_mass'][0, :2, :100] >= .5) & (view['alpha'][0, :2] > .05)).permute(1, 0, 2, 3).reshape(100, -1)
            intersect = (raw[qi] & gt[ki]).sum(-1)
            union = (raw[qi] | gt[ki]).sum(-1)
            good[0, qi] = intersect.float() / union.clamp_min(1) >= .5
        item = dict(features=features.cpu(), class_logits=logits.cpu(), quality_target=quality_target.cpu(),
                    quality_valid=quality_valid.cpu(), object_target=object_target.cpu())
        assert all(torch.isfinite(v).all() for v in item.values())
        metadata = dict(source_index=window['source_index'], scene=window['scene'],
            frame_ids=batch['frame_ids'][0].cpu().tolist(),
            input_rgb_sha256=hashlib.sha256(batch['images_all'].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
            quality_valid=int(quality_valid.sum()), matched_GT=int(object_target.sum()))
    return item, metadata, (batch, output, good)


def stack(items, device):
    return {k: torch.cat([item[k] for item in items], 0).to(device) for k in FIELDS}


def new_head(device):
    from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123456)
        head = ScalarReadout().to(device)
    registry = torch.nn.Module()
    registry.panoptic = torch.nn.Module()
    registry.panoptic.readout_probe = head
    optimizer = build_optimizer(registry)
    return head, optimizer


def step(head, optimizer, batch, arm, update, group=None, group_world=1):
    from scripts.object_locus_frozen_vggt_posefree_runtime import lr_multiplier
    optimizer.zero_grad(set_to_none=True)
    for entry in optimizer.param_groups:
        entry['lr'] = entry['peak_lr'] * lr_multiplier(update, 4172)
    loss = diagnostic_loss(head, batch, arm)
    loss.backward()
    if group_world > 1:
        for parameter in head.parameters():
            dist.all_reduce(parameter.grad, group=group)
            parameter.grad.div_(group_world)
    norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.))
    assert torch.isfinite(loss) and np.isfinite(norm)
    optimizer.step()
    return float(loss.detach()), norm


def preflight(args):
    rank, world, device = workers()
    assert world == 8
    proof = identity()
    hashes = gather(digest(SOURCE) if rank == 0 else None, world)
    assert hashes[0] == SOURCE_SHA
    from scripts.object_locus_frozen_vggt_posefree_runtime import load_manifest
    from scripts.eval_vggt_feedback_expanded_v1 import windows
    _, _, train = load_manifest()
    validation, _ = windows()
    stream = read(TRAIN_COHORT)['train_source_indices'][:16]
    assert len(stream) == len(set(stream)) == 16
    model, opt = frozen_C(device)
    versions = core_versions(model)
    for difficult in (4253, 6923)[rank::world]:
        extract(model, opt, train[difficult], device)
    chosen = read(HELDOUT)['records'][rank]
    witness, meta, payload = extract(model, opt, validation[chosen['source_index']], device)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    write_eval(model, witness, meta, payload, validation[chosen['source_index']], root / 'evaluation', rank,
               {}, device, conditions=('reference', 'quality_initial', 'objectness_initial'))
    torch.cuda.synchronize()
    # Counterbalance warm4x2 versus8x1 sampling; compare actual cached targets.
    timings = {4: [], 8: []}
    reference_items = {}
    for active in (4, 8, 8, 4):
        dist.barrier()
        started = time.monotonic()
        local = []
        if rank < active:
            for position in range(rank * (8 // active), (rank + 1) * (8 // active)):
                item, metadata, _ = extract(model, opt, train[stream[position]], device)
                local.append((position, item, metadata))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        rows = gather((elapsed, local), world)
        timings[active].append(max(row[0] for row in rows))
        for _, entries in rows:
            for position, item, metadata in entries:
                if position in reference_items:
                    prior, prior_meta = reference_items[position]
                    assert metadata == prior_meta
                    for key in FIELDS:
                        assert torch.allclose(item[key].float(), prior[key].float(), atol=1e-6, rtol=1e-6), (position, key)
                else:
                    reference_items[position] = item, metadata
    local = []
    for position in (rank, rank + 8):
        if position < 8:
            local.append((position, reference_items[position][0]))
        else:
            item, _, _ = extract(model, opt, train[stream[position]], device)
            local.append((position, item))
    all_items = {p: item for entries in gather(local, world) for p, item in entries}
    data = stack([all_items[i] for i in range(16)], device)
    groups = {size: dist.new_group(list(range(size))) for size in (1, 8)}
    snapshots, head_times = {}, {}
    for arm in ARMS:
        for size in (1, 8):
            head, optimizer = new_head(device)
            dist.barrier()
            if rank < size:
                for update in range(2):
                    first = update * 8 + rank * (8 // size)
                    batch = {k: value[first:first + 8 // size] for k, value in data.items()}
                    step(head, optimizer, batch, arm, update, groups[size], size)
            dist.barrier()
            if rank == 0:
                snapshots[(arm, size)] = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                if size == 8:
                    assert all(torch.allclose(snapshots[(arm, 1)][k], v, atol=2e-6, rtol=2e-6) for k, v in snapshots[(arm, 8)].items())
                assert float(head.net[-1].weight.norm()) > 0
            # Tiny-head benchmark excludes communicator initialization/update2.
            dist.barrier()
            torch.cuda.synchronize()
            started = time.monotonic()
            if rank < size:
                for repeat in range(20):
                    first = rank * (8 // size)
                    batch = {k: value[first:first + 8 // size] for k, value in data.items()}
                    step(head, optimizer, batch, arm, repeat + 2, groups[size], size)
            torch.cuda.synchronize()
            head_times[(arm, size)] = max(gather(time.monotonic() - started, world)) / 20
    # Both diagnostic heads train on GPU0 using gathered globalbatch8. The
    # expensive frozen teacher still runs across8 (or4) workers in parallel.
    assert_core(model, versions)
    if rank == 0:
        cache_speedup = statistics.median(timings[4]) / statistics.median(timings[8])
        assert all(torch.isfinite(v).all() for state in snapshots.values() for v in state.values())
        torch.save(dict(states={arm: snapshots[(arm, 1)] for arm in ARMS}), root / 'head_smoke.pt')
        write(root / 'numerical_preflight.json', dict(status='READOUT_NUMERICAL_PREFLIGHT_PASSED', identity=proof,
            cache_four_seconds=timings[4], cache_eight_seconds=timings[8], cache_speedup=cache_speedup,
            selected_cache_workers=8 if cache_speedup >= 1.05 else 4,
            head_seconds_per_update={arm: {str(size): head_times[(arm, size)] for size in (1, 8)} for arm in ARMS},
            head_optimizer_workers=1, global_batch=8, actual_two_updates_single_eight_equivalent=True,
            same_train_windows_targets_four_eight=True, frozen_C_parameters_buffers_unchanged=True,
            smoke_updates_not_formal=True, head_architecture='256-128-1 GELU, last linear zero', total_updates=300))
    dist.barrier()
    smoke = torch.load(root / 'head_smoke.pt', map_location='cpu', weights_only=False)['states']
    heads = {}
    for arm in ARMS:
        head, _ = new_head(device)
        head.load_state_dict(smoke[arm])
        heads[arm] = head.eval().requires_grad_(False)
    write_eval(model, witness, meta, payload, validation[chosen['source_index']], root / 'evaluation', rank,
               heads, device, conditions=('quality_smoke', 'objectness_smoke'))
    assert_core(model, versions)
    dist.destroy_process_group()


def save_checkpoint(root, update, heads, optimizers, local_cache, local_metadata, world):
    caches = gather((local_cache, local_metadata), world)
    if dist.get_rank() == 0:
        blob = dict(recipe='vggt_readout_probe_v1', identity=identity(), completed_updates=update,
            exposures_per_head=update * 8, head_optimizer_workers=1, cache_workers=world,
            states={arm: head.state_dict() for arm, head in heads.items()},
            optimizers={arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
            cache=caches, resumable=True, core_frozen=True)
        path = root / f'head_checkpoint_update{update:04d}.pt'
        torch.save(blob, path)
        write(path.with_suffix('.json'), dict(status='HEAD_CHECKPOINT_SAVED', updates=update, exposures=update * 8,
            sha256=digest(path), identity=identity()))
    dist.barrier()


def train_export(args):
    rank, world, device = workers()
    assert read(args.root / 'preflight/PREFLIGHT_VERIFIED.json')['status'] == 'READOUT_FULL_PREFLIGHT_PASSED'
    numerical = read(args.root / 'preflight/numerical_preflight.json')
    assert numerical['selected_cache_workers'] == world and numerical['identity'] == identity()
    from scripts.object_locus_frozen_vggt_posefree_runtime import load_manifest
    from scripts.eval_vggt_feedback_expanded_v1 import windows
    _, _, train = load_manifest()
    validation, _ = windows()
    stream = read(TRAIN_COHORT)['train_source_indices'][:2400]
    assert len(stream) == len(set(stream)) == 2400
    assert not ({w['scene'] for w in train} & {w['scene'] for w in validation})
    model, opt = frozen_C(device)
    versions = core_versions(model)
    root = args.root / 'training'
    if rank == 0:
        root.mkdir(exist_ok=False)
    dist.barrier()
    heads, optimizers = {}, {}
    for arm in ARMS:
        heads[arm], optimizers[arm] = new_head(device)
    local_cache, local_metadata = [], []
    started = time.monotonic()
    for update in range(300):
        positions = list(range(update * 8 + rank * (8 // world), update * 8 + (rank + 1) * (8 // world)))
        current, metadata = [], []
        for position in positions:
            item, info, payload = extract(model, opt, train[stream[position]], device)
            del payload
            current.append(item)
            metadata.append(dict(info, stream_position=position))
        local_cache += current
        local_metadata += metadata
        assembled = gather((current, metadata), world)
        row = {}
        if rank == 0:
            batch = stack([item for entries, _ in assembled for item in entries], device)
            assert len(batch['features']) == 8
            indices = [item['source_index'] for _, info in assembled for item in info]
            assert indices == stream[update * 8:(update + 1) * 8]
            for arm in ARMS:
                loss, norm = step(heads[arm], optimizers[arm], batch, arm, update)
                row[arm] = dict(loss=loss, preclip_norm=norm)
            row.update(completed_updates=update + 1, exposures_per_head=(update + 1) * 8,
                source_indices=indices, frozen_C_unchanged=True, head_optimizer_workers=1, cache_workers=world)
            with (root / 'training.jsonl').open('a') as file:
                file.write(json.dumps(row, allow_nan=False) + '\n')
            elapsed = time.monotonic() - started
            write(root / 'progress.json', dict(completed_updates=update + 1, total_updates=300,
                exposures_per_head=(update + 1) * 8, seconds_per_update=elapsed / (update + 1),
                remaining_seconds=elapsed / (update + 1) * (299 - update), row=row))
            if (update + 1) % 20 == 0:
                print('READOUT_HEADS', update + 1, row, flush=True)
        assert_core(model, versions)
        if update + 1 in (20, 100, 300):
            save_checkpoint(root, update + 1, heads, optimizers, local_cache, local_metadata, world)
            if rank == 0 and update + 1 == 20:
                assert all(optimizer.state and all(int(s['step']) == 20 for s in optimizer.state.values()) for optimizer in optimizers.values())
                assert all(float(head.net[-1].weight.norm()) > 0 for head in heads.values())
                write(args.root / 'startup_validation.json', dict(status='READOUT_FORMAL20_UPDATES_VERIFIED',
                    updates_per_arm=20, exposures_per_arm=160, shared_C_windows=160,
                    identity=identity(), native_globalbatch8=True, sampler_verified=True, core_frozen=True,
                    actual_nonzero_heads_verified=True, cache_workers=world, head_optimizer_workers=1))
        dist.barrier()
    torch.save(dict(items=local_cache, metadata=local_metadata, identity=identity()), root / f'cache_rank{rank:02d}.pt')
    assert_core(model, versions)
    if rank == 0:
        write(root / 'COMPLETE.json', dict(status='READOUT_HEADS_TRAINED', updates_per_arm=300, exposures_per_arm=2400,
            shared_C_windows=2400, identity=identity(), cache_workers=world, head_optimizer_workers=1, core_frozen=True))
    for arm in ARMS:
        for parameter in heads[arm].parameters():
            dist.broadcast(parameter.data, 0)
        heads[arm].eval().requires_grad_(False)
    choices = read(HELDOUT)['records']
    records = []
    for i, choice in enumerate(choices[rank::world]):
        window = validation[choice['source_index']]
        item, metadata, payload = extract(model, opt, window, device)
        assert metadata['input_rgb_sha256'] == choice['input_rgb_sha256']
        records.append(write_eval(model, item, metadata, payload, window, args.root / 'evaluation', rank,
            heads, device, conditions=('reference', 'quality', 'objectness')))
        write(args.root / 'evaluation' / f'export_progress_rank{rank:02d}.json', dict(completed=i + 1, total=128 // world))
    assert_core(model, versions)
    write(args.root / 'evaluation' / f'export_complete_rank{rank:02d}.json', dict(status='EXPORTED', records=records,
        identity=identity(), world_size=world, frozen_C_unchanged=True, optimizer_updates=0))
    dist.destroy_process_group()


def write_eval(model, item, metadata, payload, window, root, rank, heads, device, conditions):
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _official_export_view
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from scripts.evaluate_ssst_validation import save_rgb, save_depth
    batch, output, good = payload
    view = _official_export_view(output['render'])
    name = window['scene'] + '_context' + '_'.join(map(str, window['context']))
    golden = next(OLD_C / f'rank{i:02d}/all' / name for i in range(4) if (OLD_C / f'rank{i:02d}/all' / name).exists())
    features = item['features'].to(device)
    logits = item['class_logits'].to(device)
    original_eligible = candidate_eligibility(logits, torch.zeros((1, 100), device=device))[0]
    result = dict(name=name, **metadata, diagnostics={})
    for arm in conditions:
        path = root / arm / f'rank{rank:02d}'
        override, multiplier = None, None
        if arm.startswith('quality'):
            raw = heads['quality'](features) if 'quality' in heads else torch.zeros((1, 100), device=device)
            multiplier = raw.sigmoid()[0]
        elif arm.startswith('objectness'):
            raw = heads['objectness'](features) if 'objectness' in heads else torch.zeros((1, 100), device=device)
            override = candidate_eligibility(logits, raw)[0]
        with torch.no_grad():
            exported = write_official_pair(view, batch, window, path / 'all', eligible_override=override, score_multiplier=multiplier)
        pair = path / 'all' / name
        for sub in ('rgb', 'rgb_gt', 'depth', 'depth_gt'):
            (pair / sub).mkdir(exist_ok=True)
        for i, frame in enumerate(metadata['frame_ids']):
            filename = f'{window["scene"]}_{frame}.png'
            save_rgb(pair / 'rgb' / filename, output['render']['images_pred'][0, i])
            save_rgb(pair / 'rgb_gt' / filename, batch['images_all'][0, i])
            save_depth(pair / 'depth' / filename, output['render']['depths_pred'][0, i] / .15)
            save_depth(pair / 'depth_gt' / filename, batch['depth_gt_m_all'][0, i])
        verify_pair(pair, golden, ('rgb', 'depth', 'rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt'))
        if arm == 'reference' or arm.endswith('initial') or arm.startswith('quality'):
            for sub in ('context_seg_pred', 'target_seg_pred'):
                files = [p for p in (golden / sub).glob('*.png')]
                assert all(digest(p) == digest(pair / sub / p.name) for p in files), (arm, name, sub)
            if arm == 'reference' or arm == 'objectness_initial':
                verify_segmentation(pair, golden)
            # Canonical original C confidence avoids1ULP differences/tie order.
            for view_name in ('context', 'target'):
                original = read(golden / f'{view_name}_seg_pred/pred.json')
                actual = read(pair / f'{view_name}_seg_pred/pred.json')
                assert [(r['id'], r['label_id']) for r in original] == [(r['id'], r['label_id']) for r in actual]
                patched = [dict(r, score=r['score'] * (float(multiplier[r['id'] - 1]) if multiplier is not None else 1.)) for r in original]
                write(pair / f'{view_name}_seg_pred/pred.json', patched)
        for filename in ('render_scores.json', 'depth_scores.json'):
            shutil.copy2(golden / filename, pair / filename)
        eligible = override if override is not None else original_eligible
        result['diagnostics'][arm] = dict(good_mask_queries=int(good.sum()),
            good_mask_rejected_by_original_gate=int((good[0] & ~original_eligible).sum()),
            good_mask_rejected_by_effective_gate=int((good[0] & ~eligible).sum()),
            candidates=int(eligible.sum()), recovered_candidates=int((eligible & ~original_eligible).sum()),
            suppressed_candidates=int((original_eligible & ~eligible).sum()))
    return result


def score(args):
    from scripts.score_object_locus_posefree_official import create_worker, cpu_tree
    from scripts.posefree_official_state_helpers import update, states, merge
    from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_COMMIT
    rank, world, device = workers()
    check_official()
    root = args.root / 'evaluation'
    conditions = ('reference', 'quality_initial', 'objectness_initial', 'quality_smoke', 'objectness_smoke') if args.smoke else ('reference', 'quality', 'objectness')
    for arm in conditions:
        part = root / arm / f'rank{rank:02d}'
        names = sorted(p.name for p in (part / 'all').iterdir() if p.is_dir())
        assert len(names) == (1 if args.smoke else 128 // world)
        worker = create_worker(part / 'all', str(device))
        for name in names:
            for view in ('context', 'target'):
                update(worker, worker.process_segmentation(part / 'all' / name / f'{view}_seg_pred', part / 'all' / name / f'{view}_seg_gt'), view)
        state = dict(names=names, states=cpu_tree(states(worker)))
        torch.save(dict(all=state), part / 'official_states.pt')
        with tempfile.TemporaryDirectory(dir=part, prefix='official-contract-') as tmp:
            tmp = Path(tmp)
            for name in names[:1]:
                (tmp / name).symlink_to((part / 'all' / name).resolve(), target_is_directory=True)
            expected = evaluate(tmp, device='cpu', image_quality=False, depth_quality=False)
            chosen = create_worker(part / 'all', str(device))
            for view in ('context', 'target'):
                update(chosen, chosen.process_segmentation(part / 'all' / names[0] / f'{view}_seg_pred', part / 'all' / names[0] / f'{view}_seg_gt'), view)
            actual = merge([dict(names=names[:1], states=cpu_tree(states(chosen)))], part)
            for view in ('context', 'target'):
                for metric in ('miou', 'pq'):
                    assert abs(actual[view + '_' + metric] - expected[view + '_' + metric]) < 1e-7
                for metric in ('map', 'map_50', 'map_75'):
                    assert abs(actual[view + '_map'][metric] - expected[view + '_map'][metric]) < 1e-7
        write(part / 'score_complete.json', dict(status='SCORED', native_contract='PASS', windows=len(names), official_commit=SIU3R_COMMIT, optimizer_updates=0))
    dist.destroy_process_group()


def reduce(args):
    from scripts.posefree_official_state_helpers import merge
    from scripts.eval_vggt_classification_full312_v1 import write_table
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    check_official()
    root = args.root / 'evaluation'
    numerical = read(args.root / 'numerical_preflight.json') if args.smoke else read(args.root / 'preflight/numerical_preflight.json')
    world = 8 if args.smoke else numerical['selected_cache_workers']
    conditions = ('reference', 'quality_initial', 'objectness_initial', 'quality_smoke', 'objectness_smoke') if args.smoke else ('reference', 'quality', 'objectness')
    results = {}
    for arm in conditions:
        parts, observations = [], []
        for rank in range(world):
            part = root / arm / f'rank{rank:02d}'
            assert read(part / 'score_complete.json')['native_contract'] == 'PASS'
            blob = torch.load(part / 'official_states.pt', map_location='cpu', mmap=True, weights_only=False)['all']
            parts.append(blob)
            for name in blob['names']:
                pair = part / 'all' / name
                rgb, depth = read(pair / 'render_scores.json'), read(pair / 'depth_scores.json')
                by_item = {row['item']: row for row in depth}
                assert len(rgb) == len(depth) == 6
                observations += [dict(absrel=by_item[row['item']]['absrel'], rmse=by_item[row['item']]['rmse'],
                                     psnr=row['psnr'], ssim=row['ssim'], lpips=row['lpips']) for row in rgb]
        names = [n for p in parts for n in p['names']]
        assert len(names) == len(set(names)) == (8 if args.smoke else 128)
        if not args.smoke:
            assert set(names) == {row['name'] for row in read(HELDOUT)['records']}
        segmentation = merge(parts, root / arm)
        # Float32 MeanIoU summation may differ at4 versus8 shards. Exact PNG
        # identity permits reuse of the unchanged original C semantic metrics;
        # always preserve fresh native results and verify them first.
        write(root / arm / 'fresh_native_segmentation.json', segmentation)
        if not args.smoke and arm in ('reference', 'quality'):
            canonical = read(OLD_C / 'metrics.json')['official_segmentation']['all']
            for view in ('context', 'target'):
                for metric in ('miou', 'pq'):
                    assert abs(segmentation[view + '_' + metric] - canonical[view + '_' + metric]) < 1e-7
                if arm == 'reference':
                    for metric in ('map', 'map_50', 'map_75'):
                        assert abs(segmentation[view + '_map'][metric] - canonical[view + '_map'][metric]) < 1e-12
            for key, value in canonical.items():
                if arm == 'reference' or '_map' not in key:
                    segmentation[key] = value
        report = dict(status='EVALUATED', arm=arm, windows=len(names), unique_scenes=len(names), images=len(observations),
            official_commit=SIU3R_COMMIT, identity=identity(), optimizer_updates=0,
            scopes={'target-all': {k: float(np.mean([row[k] for row in observations])) for k in observations[0]}},
            official_segmentation={'all': segmentation},
            unchanged_semantic_metrics_reused=not args.smoke and arm in ('reference', 'quality'))
        write(root / arm / 'metrics.json', report)
        values = write_table(report, root / arm)
        results[arm] = values
    reference = results['reference']
    for arm, values in results.items():
        assert all(abs(x - y) < 1e-12 for x, y in zip(reference[:5], values[:5]))
        if arm.startswith('quality') or arm == 'objectness_initial':
            for i in (5, 7, 9, 11):
                assert abs(values[i] - reference[i]) < 1e-12
        if arm.endswith('initial'):
            assert all(x is None or abs(x - y) < 1e-12 for x, y in zip(reference, values))
    if args.smoke:
        write(args.root / 'PREFLIGHT_VERIFIED.json', dict(status='READOUT_FULL_PREFLIGHT_PASSED', identity=identity(),
            numerical=numerical, native_all_conditions_score_contracts_passed=True, original_C_RGB_depth_GT_reproduced=True,
            quality_changes_scores_only=True, objectness_zero_reproduces_C=True, formal_updates=0))
    else:
        from scripts.format_object_locus_posefree_siu3r_table import table_values
        old = table_values(read(OLD_C / 'metrics.json'))
        assert all(x is None or abs(x - y) < 1e-12 for x, y in zip(old, reference))
        write(args.root / 'COMPLETE.json', dict(status='READOUT128_TRAINED_AND_EVALUATED', updates_per_arm=300,
            exposures_per_arm=2400, shared_C_windows=2400, official_columns=13, text_columns_blank=True, results=results))


def summary(args):
    from scripts.analyze_vggt_feedback_factorial_v1 import markdown
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS
    import csv
    done = read(args.root / 'COMPLETE.json')
    assert done['status'] == 'READOUT128_TRAINED_AND_EVALUATED'
    logs = [json.loads(line) for line in (args.root / 'training/training.jsonl').read_text().splitlines()]
    assert len(logs) == 300 and [row['completed_updates'] for row in logs] == list(range(1, 301))
    stream = read(TRAIN_COHORT)['train_source_indices'][:2400]
    assert all(row['source_indices'] == stream[i * 8:(i + 1) * 8] and row['frozen_C_unchanged'] for i, row in enumerate(logs))
    records = [r for file in sorted((args.root / 'evaluation').glob('export_complete_rank*.json')) for r in read(file)['records']]
    assert len(records) == len({r['name'] for r in records}) == 128
    totals = {arm: {key: sum(r['diagnostics'][arm][key] for r in records) for key in records[0]['diagnostics'][arm]} for arm in ('reference',) + ARMS}
    out = args.root / 'analysis'
    out.mkdir(exist_ok=False)
    rows = [(label, done['results'][arm]) for label, arm in [('R frozen C', 'reference'), ('Q quality only', 'quality'), ('O objectness gate only', 'objectness')]]
    (out / 'siu3r_comparison.md').write_text(markdown(rows) + '\n')
    with (out / 'siu3r_comparison.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('experiment',) + FIELDS)
        for label, values in rows:
            writer.writerow((label,) + tuple(values))
    write(out / 'results.json', dict(status='READOUT128_CONTROLS_AUDITED', identity=identity(),
        training_updates_per_arm=300, shared_C_windows=2400, gate_diagnostics=totals,
        original_C_masks_and_features_frozen=True, native13=done['results'],
        caveat='O gate alters candidates and final pixel assignment; frozen original19-class no-object argmax is not changed. GT targets are training-only; no GT scoring at inference. Single-seed128 screen, not full312 validation.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'train_export', 'score', 'reduce', 'summary'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    globals()[arguments.mode](arguments)
