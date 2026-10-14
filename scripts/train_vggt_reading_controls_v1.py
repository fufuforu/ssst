"""Four eight-GPU, matched full-pool C decoder controls; native joint losses."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tokengs.models.object_locus_reading_controls_v1 import ARMS, install
from scripts.train_vggt_classification_readback_v1 import build_base, configure, apply_patch_state, TRAIN_COHORT, HELDOUT, SOURCE, SOURCE_SHA
from scripts.train_vggt_joint_short_controls_v1 import digest, parameter_hash
from scripts.eval_vggt_joint_short_controls_v1 import write, export_window
PATCH = Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/training/classification_patch_update0300.pt')
PATCH_SHA = 'ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9'
RECIPE = 'vggt_reading_controls_v1'
FROZEN = ('frozen_vggt.', 'understanding.encoder.')


def config(arm):
    return dict(recipe=RECIPE, arm=arm, source_sha256=SOURCE_SHA, C_patch_sha256=PATCH_SHA,
        world_size=8, microbatch=1, accumulation=1, global_batch=8, updates=300, exposures=2400,
        stream_sha256=digest(TRAIN_COHORT), heldout_sha256=digest(HELDOUT),
        sampling='same first2400 unique full8337 training-pool stream, consecutive global8',
        frozen_prefixes=list(FROZEN), other_trainability='native original flags plus C class readback',
        model_modes='train downstream; both encoders and all BatchNorm eval',
        adapter_understanding_scale=.01, reconstruction_understanding_scale=.01, feedback_peak=.05,
        source_joint_exposure=35776, understanding_weight=1.,
        loss='unchanged native reconstruction plus full V3 set loss, matching and auxiliary losses',
        optimizer='fresh native AdamW beta.9/.95 eps1e-8 clip1 original decay groups',
        lr='native peaks rec1e-5/understanding1e-5/object_memory1e-4; warmup200 and cosine4172 truncated300',
        intervention_ramp='min(completed_new_updates/20,1)', new_trainable_parameters=0,
        local_read_peak=.5, visual_peak=.1, reliable_depth_peak=.1,
        local_read='L8/10/12 previous predicted anchor soft masks detached; log(.1+.9*mask/max), stuff/empty/flat global fallback',
        visual='same CA equations, normalized existing Fm bilinear at anchor context projections; residual norm<=.025*anchor norm at peak',
        geometry='retain ray bias; add .1*reliability*clamp(-.5*alongray_distance^2/radius^2,-20,0)',
        confidence='same14x14 patch pooling as rays; per context-view q20/q80 normalization, fixed, not calibrated probability',
        geometry_quality_policy='monitor_v1', Q='absent; native scores first; equal-budget Q refit is subsequent separate work')


def build(arm, device, training=True):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    model, opt, identity, _, initial = build_base(device)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    configure(model, 'object_support', initial)
    blob = torch.load(PATCH, map_location='cpu', mmap=True, weights_only=False)
    apply_patch_state(model, blob, 'object_support')
    del blob, initial
    install(model, arm)
    for n, p in model.named_parameters():
        p.requires_grad_(training and not n.startswith(FROZEN) and (flags.get(n, False) or n.startswith('panoptic.mask_support_readback.')))
    model.train(training)
    model.understanding.encoder.eval()
    model.frozen_vggt.eval()
    model.understanding_step = 35776
    if any(p.requires_grad for n, p in model.named_parameters() if n.startswith(FROZEN)):
        raise ValueError('encoder trainability differs')
    return model, opt, identity


def slots(update, rank):
    stream = json.loads(TRAIN_COHORT.read_text())['train_source_indices'][:2400]
    if len(set(stream)) != 2400 or not 0 <= update < 300 or not 0 <= rank < 8:
        raise ValueError('stream/clock differs')
    return [stream[update * 8 + rank]]


def worker():
    import torch.distributed as dist
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    if world not in (1, 8) or '3090' not in torch.cuda.get_device_name(local):
        raise ValueError('matched 1/8-rank3090 required')
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    sha = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != os.environ['TASK_CODE_SHA']:
        raise ValueError('execution snapshot differs')
    return rank, world, device, sha


def zero_reproduction(model, opt, window, device):
    from scripts.object_locus_v3_set_runtime import build_batch
    pan = model.panoptic
    control = pan.reading_control
    del pan.reading_control
    batch = build_batch(opt, window, device)
    model.eval()
    keys = ('gaussians', 'class_logits19', 'gaussian_membership', 'F_m', 'q_pre')
    with torch.no_grad():
        original = model.generate(batch['images_input'])
        expected = {k: original[k].detach().clone() for k in keys}
        del original
        pan.reading_control = control
        output = model.generate(batch['images_input'])
        diffs = {k: float((output[k] - expected[k]).abs().max()) for k in keys}
        if any(v != 0 for v in diffs.values()):
            raise ValueError('zero-start control changes native C: ' + str(diffs))
    del output, batch, expected
    model.train()
    model.understanding.encoder.eval()
    model.frozen_vggt.eval()
    return diffs


def run(a):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts import object_locus_v3_set_runtime as provider
    from scripts.eval_vggt_feedback_expanded_v1 import windows
    import torch.distributed as dist
    rank, world, device, sha = worker()
    if a.mode == 'train' and world != 8:
        raise ValueError('formal training requires eight GPUs')
    identities = [dict(source=digest(SOURCE), patch=digest(PATCH), config=config(a.arm)) if rank == 0 else None]
    dist.broadcast_object_list(identities, src=0)
    if identities[0]['source'] != SOURCE_SHA or identities[0]['patch'] != PATCH_SHA:
        raise ValueError('source bytes differ')
    _, _, train = native.load_manifest()
    validation, _ = windows()
    choices = json.loads(HELDOUT.read_text())['records']
    if {w['scene'] for w in train} & {w['scene'] for w in validation}:
        raise ValueError('train/validation scenes overlap')
    if digest(native.MANIFEST) != json.loads(TRAIN_COHORT.read_text())['train_manifest_sha256']:
        raise ValueError('training manifest differs')
    error = None
    if rank == 0:
        try:
            a.output.mkdir(parents=True, exist_ok=a.resume)
            if not a.resume and any(a.output.iterdir()):
                raise ValueError('preserve prior outputs')
        except Exception as exc:
            error = exc
    native.synchronized_error(error, device, 'output', 0)
    dist.barrier()
    model, opt, identity = build(a.arm, device)
    named = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
    optimizer = native.build_optimizer(model)
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    buffers = {n: p._version for n, p in model.named_buffers()}
    initial_hash = parameter_hash(named)
    native.seed_everything(42 + rank)
    counts = np.zeros(8337, dtype=np.int64)
    start = 0
    if a.resume:
        b = torch.load(a.output / 'checkpoint_latest.pt', map_location='cpu', mmap=True, weights_only=False)
        if b['config'] != config(a.arm) or b['world_size'] != 8:
            raise ValueError('resume science identity differs')
        apply_state(model, b)
        optimizer.load_state_dict(b['optimizer'])
        native.restore_rank_rng(b['rank_rng'][rank])
        counts = np.asarray(b['rank_window_counts'][rank], dtype=np.int64)
        start = b['completed_updates']
        del b
        frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
        buffers = {n: p._version for n, p in model.named_buffers()}
    else:
        if a.mode != 'train':
            proof = zero_reproduction(model, opt, train[4253], device)
            rows = [None] * world
            dist.all_gather_object(rows, dict(rank=rank, max_abs_differences=proof))
            if rank == 0:
                write(a.output / 'zero_reproduction.json', dict(status='EXACT_NATIVE_C_REPRODUCED', rows=rows, formal_updates=0))
    # Every branch shares exactly the same existing trainable tensors.
    initialization = [None] * world
    dist.all_gather_object(initialization, dict(rank=rank, parameter_sha256=parameter_hash(named),
        parameter_count=sum(p.numel() for _, p in named), names=[n for n, _ in named]))
    if len({r['parameter_sha256'] for r in initialization}) != 1:
        raise ValueError('rank parameter initialization differs')
    if rank == 0:
        write(a.output / ('resume_manifest.json' if a.resume else 'run_manifest.json'), dict(config=config(a.arm), source=identity, C_patch_sha256=PATCH_SHA,
            execution_sha=sha, mode=a.mode, slurm_job_id=os.environ['SLURM_JOB_ID'],
            rank_initialization=initialization, start_update=start, initial_trainable_sha256=initial_hash))
    prefixes = ('anchor_decoder.', 'vggt_memory_adapter.', 'understanding.adapter.', 'panoptic.')
    probes = {}
    for prefix in prefixes:
        selected = next(((n, p) for n, p in named if n.startswith(prefix) and p.numel() > 100), None)
        if selected is None:
            raise ValueError('intended downstream family absent: ' + prefix)
        n, p = selected
        probes[prefix] = (n, p.detach().clone())
    monitor = native.GeometryMonitor(a.output, rank, 'reading_controls_' + a.mode)
    started = time.monotonic()
    limit = a.stop_after if a.mode == 'train' else 2

    def save(step):
        rng, cs = [None] * world, [None] * world
        dist.all_gather_object(rng, native.capture_rank_rng())
        dist.all_gather_object(cs, counts.tolist())
        error = None
        if rank == 0:
            try:
                state = {n: p.detach().cpu().clone() for n, p in named}
                state_buffers = {n: p.detach().cpu().clone() for n, p in model.named_buffers() if not n.startswith(FROZEN)}
                payload = dict(recipe=RECIPE, config=config(a.arm), parameter_patch=state, buffers=state_buffers,
                    optimizer=optimizer.state_dict(), rank_rng=rng, rank_window_counts=cs, world_size=world,
                    completed_updates=step, completed_exposures=step * 8, execution_sha=sha, resumable=True)
                path = a.output / 'checkpoint_latest.pt'
                # Serialize atomic replacements across the two nodes so only
                # one additional2.6GiB temporary checkpoint exists at a time.
                import fcntl
                with (a.output.parent.parent / '.checkpoint-write.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    native._atomic_save(payload, path)
                    fcntl.flock(lock, fcntl.LOCK_UN)
                write(path.with_suffix('.json'), dict(status='SAVED', sha256=digest(path), completed_updates=step, execution_sha=sha))
            except Exception as exc:
                error = exc
        native.synchronized_error(error, device, 'save_checkpoint', step)

    for update in range(start, limit):
        model.panoptic.reading_control.completed_updates = update if a.mode == 'train' else (0 if update == 0 else 20)
        indices = slots(update, rank) if a.mode == 'train' else [4253 if update == 0 else 6923]
        batch = provider.build_batch(opt, train[indices[0]], device)
        model.classification_loss_terms = []
        row = native.train_microbatch_window(model, optimizer, [batch], update,
            base_exposure=35776 + update * 8, world_size=world, accumulation=1, total_updates=4172,
            adapter_understanding_scale=.01, monitor=monitor,
            window_metadata=[native.window_identity(indices[0], train[indices[0]])], stage='reading_controls_' + a.mode,
            gradient_diagnostic=update in (0, 1, 19))
        del batch
        error = None
        try:
            if any(p._version != frozen[n] or p.grad is not None for n, p in model.named_parameters() if n in frozen):
                raise ValueError('frozen parameter changed')
            if any(p._version != buffers[n] for n, p in model.named_buffers()):
                raise ValueError('fixed BatchNorm/encoder buffer changed')
            if row['understanding_weight'] != 1 or model.feedback_peak != .05:
                raise ValueError('joint loss/feedback differs')
        except Exception as exc:
            error = exc
        native.synchronized_error(error, device, 'frozen_contract', update)
        np.add.at(counts, indices, 1)
        step = update + 1
        row.update(arm=a.arm, source_indices=indices, formal_updates=step if a.mode == 'train' else 0,
            control=model.panoptic.reading_control.describe(), actual_feedback_peak=.05,
            frozen_encoders_unchanged=True, class_terms=model.classification_loss_terms)
        with (a.output / f'training_rank{rank:02d}.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if rank == 0:
            seconds = (time.monotonic() - started) / (step - start)
            write(a.output / 'progress.json', dict(completed_updates=step, formal_updates=step if a.mode == 'train' else 0,
                completed_exposures=step * world, total_updates=limit, seconds_per_update=seconds,
                estimated_remaining_seconds=seconds * (limit - step)))
            if step % 20 == 0 or a.mode != 'train':
                print('READING_CONTROL', a.arm, step, row['loss_recon'], row['loss_understanding'], row['control'], flush=True)
        if step == 20 or (a.mode != 'train' and step == 2):
            rows = [None] * world
            changes = {pre: float((dict(named)[n].detach() - p).norm()) for pre, (n, p) in probes.items()}
            dist.all_gather_object(rows, dict(rank=rank, parameter_sha256=parameter_hash(named), changes=changes,
                peak_allocated=row['peak_allocated'], peak_reserved=row['peak_reserved'], control=row['control'],
                geometry=monitor.summary(), window_counts=int(counts.sum()), finite=True))
            if len({r['parameter_sha256'] for r in rows}) != 1 or any(any(v <= 0 for v in r['changes'].values()) for r in rows):
                raise ValueError('actual updates/rank synchronization differ')
            if rank == 0:
                write(a.output / ('startup_confirmation.json' if a.mode == 'train' else 'smoke_update_proof.json'),
                    dict(status='TRAINING_CONFIRMED' if a.mode == 'train' else 'GPU_STEP_VERIFIED',
                        formal_updates=step if a.mode == 'train' else 0, formal_exposures=step * 8 if a.mode == 'train' else 0,
                        execution_sha=sha, config=config(a.arm), rank_rows=rows))
        if a.mode == 'train' and step in (20, 100, 300):
            save(step)
        if step % 20 == 0:
            monitor.write_summary(step)
    if a.mode != 'train':
        model.eval().requires_grad_(False)
        # Test full-strength new computation and exact official output formats
        # after true joint backward/update at ramp onset.
        model.panoptic.reading_control.completed_updates = 20
        witness = validation[choices[rank]['source_index']]
        with torch.no_grad():
            record = export_window(model, opt, witness, a.output / f'evaluation_rank{rank:02d}', device)
        write(a.output / f'evaluation_rank{rank:02d}/export_complete.json', dict(status='EXPORTED', records=[record], optimizer_updates=0))
    if rank == 0:
        write(a.output / ('STARTUP_CHECKPOINTED.json' if a.mode == 'train' and limit == 20 else 'COMPLETE.json'), dict(status=('READING_CONTROL_TRAINED' if limit == 300 else 'READING_CONTROL_STARTUP_CHECKPOINTED') if a.mode == 'train' else 'READING_CONTROL_SMOKE_PASSED',
            arm=a.arm, completed_updates=limit, formal_updates=limit if a.mode == 'train' else 0,
            completed_exposures=limit * 8 if a.mode == 'train' else 0, execution_sha=sha, config=config(a.arm)))
    dist.barrier()
    dist.destroy_process_group()


def apply_state(model, blob):
    target = dict(model.named_parameters())
    expected = {n for n, p in target.items() if p.requires_grad}
    # Evaluation builds freeze the same registry; names are verified against
    # the manifest and source when restoring below.
    if expected and set(blob['parameter_patch']) != expected:
        raise ValueError('patch ownership differs')
    with torch.no_grad():
        for n, v in blob['parameter_patch'].items():
            if n not in target or n.startswith(FROZEN) or target[n].shape != v.shape:
                raise ValueError('patch key/shape invalid')
            target[n].copy_(v)
        buffers = dict(model.named_buffers())
        for n, v in blob['buffers'].items():
            if n not in buffers or n.startswith(FROZEN) or buffers[n].shape != v.shape:
                raise ValueError('buffer patch invalid')
            buffers[n].copy_(v)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--mode', choices=('single_smoke', 'eight_smoke', 'train'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, choices=(20, 300), default=300)
    run(p.parse_args())
