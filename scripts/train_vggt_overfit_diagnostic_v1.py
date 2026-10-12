"""Task-owned four-window diagnostic; never an official train/eval entry point."""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import types

import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_frozen_joint_epoch_04.pt')
SOURCE_SHA = '01d49b092c8dfc2e02f15b5216e5d6ed1cfd88fae2cf693ee69334e45a9410ba'
REVISION = '860abec7937da0a4c03c41d3c269c366e82abdf9'
RECIPE = 'vggt_four_window_overfit_diagnostic_v1'
ARMS = ('pred_joint', 'pred_recon', 'gt_recon')
TRAIN_PREFIXES = ('gs_tokens', 'gs_tokens_dynamic', 'anchor_decoder.',
                  'enc_dec_backbone.decoder_blocks.', 'activation_head.', 'vggt_memory_adapter.')
TOTAL_UPDATES = 300
GLOBAL_BATCH = 16


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    return value


def rank_window_slots(rank):
    if not 0 <= rank < 8:
        raise ValueError('diagnostic requires eight ranks')
    # Every update contains four repetitions of each of four fixed windows.
    return (rank % 4, (rank + 1) % 4)


def pilot_lr_update(update):
    if not 0 <= update < TOTAL_UPDATES:
        raise ValueError('invalid pilot update')
    # Reuse the native LR helper: map its 200-step warmup to 20 pilot steps,
    # then hold each original peak. This is explicitly a diagnostic schedule.
    return min((update + 1) * 10, 200) - 1


def gt_baseline_gauge(provider_c2w, predicted_context):
    """GT first-camera frame, positive scale matching predicted context baseline."""
    relative = torch.linalg.inv(provider_c2w[:, :1]) @ provider_c2w
    gt = (relative[:, 1, :3, 3] - relative[:, 0, :3, 3]).norm(dim=-1)
    pred = (predicted_context[:, 1, :3, 3] - predicted_context[:, 0, :3, 3]).norm(dim=-1)
    if not torch.isfinite(relative).all() or (gt <= 1e-8).any() or (pred <= 1e-8).any():
        raise ValueError('invalid/degenerate diagnostic camera gauge')
    scale = pred / gt
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('nonpositive/nonfinite diagnostic gauge')
    result = relative.clone()
    result[..., :3, 3] *= scale[:, None, None]
    rotation = result[..., :3, :3]
    identity = torch.eye(3, device=result.device, dtype=result.dtype)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity.expand_as(rotation), atol=2e-4, rtol=0):
        raise ValueError('GT camera rotation outside SO(3)')
    if not torch.allclose(torch.linalg.det(rotation), torch.ones_like(gt[:, None]).expand(rotation.shape[:2]), atol=2e-4, rtol=0):
        raise ValueError('improper GT camera rotation')
    return result, scale


def diagnostic_forward(model, batch, cached, arm, *, compute_understanding=True):
    """Identical native decoder/feedback/loss with explicit diagnostic interventions."""
    from tokengs.models.object_locus_posefree_geometry import pixel_rays, camera_vectors, rays_to_patch_plucker
    from tokengs.models.input_types import EncoderLatent, ModelInputDecoder
    from tokengs.models.canonical_recon_models import _full_supervision
    _, keys, values = model.vggt_memory_adapter(cached['patch_layers'])
    latent = EncoderLatent(keys=keys, values=values)
    gt = arm == 'gt_recon'
    context_cams = cached['gt_c2w'][:, :2] if gt else cached['context_c2w']
    context_k = cached['gt_intrinsics'][:, :2] if gt else cached['context_intrinsics']
    k518 = cached['gt_context_k518'] if gt else cached['context_k518']
    all_cams = cached['gt_c2w'] if gt else cached['calibration']['c2w']
    all_k = cached['gt_k'] if gt else cached['calibration']['intrinsics_matrix']
    intrinsics = camera_vectors(all_k)
    decoder = ModelInputDecoder(cam_view=torch.linalg.inv(all_cams).transpose(-1, -2), intrinsics=intrinsics)
    context_decoder = ModelInputDecoder(cam_view=torch.linalg.inv(context_cams).transpose(-1, -2), intrinsics=context_k)
    states, stats = model.anchor_decoder.forward_stateful(
        model.get_gs_tokens(batch_size=1), latent, rays_to_patch_plucker(context_cams, k518, patch_size=14),
        model.panoptic, cached['fm'], cached['qpre'], model.understanding.mask_embedder, model.understanding_step)
    final = states[-1]
    gaussians = model.activation_head(final['tokens'], final['mu'], final['radii'])
    prediction = dict(states=states, gaussians=gaussians,
        reconstruction=model._reconstruction_from_gaussians(gaussians),
        ray_stats=stats, beta=final['beta'], F_m=cached['fm'], q_pre=cached['qpre'])
    rays_o, rays_d = pixel_rays(all_cams, all_k, 256, 256)
    loss_batch = dict(batch, cam_view_all=decoder.cam_view, intrinsics_all=intrinsics,
        rays_os=rays_o, rays_ds=rays_d, cam_to_world_input=context_cams, intrinsics_input=context_k)
    recon, metrics, _, _, _ = model._layer_objective(states, decoder, _full_supervision(loss_batch))
    if arm == 'pred_joint' and compute_understanding:
        from tokengs.models.object_locus_v3_set_loss import object_locus_v3_set_losses
        # Native readout uses context cameras both for lifting and membership render.
        prediction.update(model._readout(final, gaussians, cached['fm'], context_decoder, context_decoder))
        under, under_metrics = object_locus_v3_set_losses(prediction, loss_batch, model.opt)
        metrics.update(under_metrics)
    else:
        # An independent zero leaf makes the absent objective have no gradients,
        # while permitting the unmodified native GC step's allow_unused call.
        under = recon.new_zeros((), requires_grad=torch.is_grad_enabled())
    if not gt:
        prediction['target_camera_calibration'] = cached['calibration']
    metrics.update(loss_recon=recon, loss_understanding=under, loss=recon + under,
                   loss_total=recon + under, psnr=metrics[f'psnr_layer{model.supervised_layers[-1]}'])
    return {'prediction': prediction}, metrics


def prepare_window(model, opt, window, device):
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.diagnose_vggt_reconstruction_v1 import k_matrix
    from tokengs.models.object_locus_posefree_geometry import camera_vectors
    batch = build_batch(opt, window, device)
    captured = []
    hook = model.frozen_vggt.register_forward_hook(lambda _m, _i, out: captured.append(out))
    with torch.no_grad():
        try:
            output, reference = model.step_loss(batch, step=50064, understanding_weight=1.)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError('must capture exactly one context-only VGGT forward')
        prediction = output['prediction']
        provider_c2w = torch.linalg.inv(batch['cam_view_all'].transpose(-1, -2))
        gt_c2w, scale = gt_baseline_gauge(provider_c2w, prediction['predicted_context_c2w'])
        gt_k = k_matrix(batch['intrinsics_all'])
        gt_k518 = gt_k[:, :2].clone()
        gt_k518[..., :2, :] *= 518 / 256
        cached = dict(patch_layers=captured[0].patch_layers, fm=prediction['F_m'], qpre=prediction['q_pre'],
            context_c2w=prediction['predicted_context_c2w'], context_intrinsics=camera_vectors(prediction['predicted_context_intrinsics_matrix']),
            context_k518=prediction['predicted_context_intrinsics518'], calibration=prediction['target_camera_calibration'],
            gt_c2w=gt_c2w, gt_k=gt_k, gt_context_k518=gt_k518, gt_intrinsics=camera_vectors(gt_k))
        replay, replay_metrics = diagnostic_forward(model, batch, cached, 'pred_joint')
        gs_error = float((replay['prediction']['gaussians'] - prediction['gaussians']).abs().max())
        errors = {k: float((replay_metrics[k] - reference[k]).abs()) for k in ('loss_recon', 'loss_understanding')}
        if gs_error > 1e-6 or any(v > 1e-6 for v in errors.values()):
            raise RuntimeError(f'native/cached replay mismatch: GS={gs_error}, losses={errors}')
        # Isolating understanding loss must preserve the exact reconstruction.
        _, rec_metrics = diagnostic_forward(model, batch, cached, 'pred_recon')
        if float((rec_metrics['loss_recon'] - reference['loss_recon']).abs()) > 1e-6:
            raise RuntimeError('recon-only arm changed native reconstruction objective')
        rgb_sha = hashlib.sha256(batch['images_all'].float().cpu().numpy().tobytes()).hexdigest()
        proof = dict(window=window, input_rgb_sha256=rgb_sha, native_gaussian_max_error=gs_error,
            native_loss_absolute_errors=errors, gt_positive_baseline_scale=scale.tolist(),
            gt_gauge='first GT context camera frame; norm ratio of predicted/GT context baseline',
            source_recon_loss=float(reference['loss_recon']), source_understanding_loss=float(reference['loss_understanding']))
        return tree_to(batch, 'cpu'), tree_to(cached, 'cpu'), proof


def configure_trainable(model):
    model.requires_grad_(False)
    named = list(model.named_parameters())
    for name, parameter in named:
        parameter.requires_grad_(name.startswith(TRAIN_PREFIXES))
    if any(p.requires_grad for n, p in named if n.startswith(('frozen_vggt.', 'understanding.', 'panoptic.'))):
        raise RuntimeError('diagnostic trainability boundary violated')
    if not all(any(n.startswith(prefix) and p.requires_grad for n, p in named)
               for prefix in ('anchor_decoder.', 'activation_head.', 'vggt_memory_adapter.')):
        raise RuntimeError('missing intended trainable module')
    model.train()
    model.understanding.eval()
    model.panoptic.eval()
    model.frozen_vggt.eval()
    model.reconstruction_adaptation = False
    model.understanding_step = 50064


def parameter_digest(named):
    h = hashlib.sha256()
    for name, parameter in sorted(named):
        h.update(name.encode())
        h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    import torch.distributed as dist
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _restore_checkpoint
    from scripts.posefree_cpu_adamw import CPUOffloadAdamW
    if int(os.environ.get('WORLD_SIZE', '0')) != 8 or socket.gethostname().split('.')[0] != '3dimage-17':
        raise RuntimeError('diagnostic authorized only on node17/eight ranks')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if '4090' not in torch.cuda.get_device_name(local):
        raise RuntimeError('diagnostic expects RTX4090')
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    rank = dist.get_rank()
    error = None
    try:
        if args.output.exists():
            raise RuntimeError('diagnostic output must be new; no overwrite/resume implied')
        if digest(SOURCE) != SOURCE_SHA:
            raise RuntimeError('common source checkpoint identity mismatch')
    except Exception as exc:
        error = exc
    native.synchronized_error(error, device, 'diagnostic_identity', 0)
    if rank == 0:
        args.output.mkdir(parents=True)
    dist.barrier()
    cohort_path = REPO / 'docs/vggt_overfit_cohort_v1.json'
    cohort = json.loads(cohort_path.read_text())
    windows = cohort['windows']
    if len(windows) != 4 or len({w['scene'] for w in windows}) != 4 or any(w['split'] != 'train' for w in windows):
        raise RuntimeError('diagnostic requires four distinct training scenes')
    _, _, full_windows = native.load_manifest()
    for w in windows:
        provider = full_windows[w['source_index']]
        if any(provider[k] != w[k] for k in ('scene', 'context', 'novel')):
            raise RuntimeError('fixed cohort differs from committed provider manifest')
    execution_sha = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if execution_sha != os.environ['TASK_CODE_SHA']:
        raise RuntimeError('diagnostic immutable code SHA mismatch')
    native.seed_everything(42)
    os.environ['VGGT_HF_REVISION'] = REVISION
    model, opt, _, _ = native.build_model(preserve_fp32_aggregator=True)
    model.to(device=device, dtype=torch.float32)
    _restore_checkpoint(model, SOURCE, REVISION, digest(native.MANIFEST))
    # Evaluation restoration normally freezes BF16 AA. Match the current A/B
    # continuation's common FP32 AA/BF16 DINO/autocast, then reload the original
    # FP32 adapted asset to avoid a BF16 round-trip of the source weights.
    model.frozen_vggt.set_reconstruction_adaptation(False, keep_fp32=True)
    native.restore_adapted_vggt(model, SOURCE.parent / 'adapted_vggt.pt', native.CONTINUATION_VGGT_SHA)
    configure_trainable(model)
    slots = rank_window_slots(rank)
    data, cached, proofs = {}, {}, []
    error = None
    try:
        for slot in slots:
            data[slot], cached[slot], proof = prepare_window(model, opt, windows[slot], device)
            proofs.append(proof)
    except Exception as exc:
        error = exc
    native.synchronized_error(error, device, 'native_replay_and_cache', 0)
    # The frozen VGGT and MASt3R encoder no longer execute after exact replay.
    # Keep only the mask embedder on GPU; no full feature cache is persisted.
    model.frozen_vggt.cpu()
    model.understanding.encoder.cpu()
    model.understanding.adapter.cpu()
    model.understanding.mask2former.pixel_decoder.cpu()
    model.understanding.mask2former.transformer_module.cpu()
    model.understanding.mask_embedder.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    frozen_buffers = {n: b._version for n, b in model.named_buffers()}
    named = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
    initial_digest = parameter_digest(named)
    probes = {prefix: next(p for n, p in named if n.startswith(prefix)).detach().clone()
              for prefix in ('anchor_decoder.', 'activation_head.', 'vggt_memory_adapter.')}
    optimizer_native = native.build_optimizer(model)
    optimizer = CPUOffloadAdamW(optimizer_native.param_groups)
    del optimizer_native
    native.seed_everything(42 + rank)
    def cached_step(self, batch, *, step, phase='train', understanding_weight=1.):
        del phase, understanding_weight
        self.understanding_step = int(step)
        slot = batch['_diagnostic_slot']
        return diagnostic_forward(self, batch, tree_to(cached[slot], device), args.arm)
    model.step_loss = types.MethodType(cached_step, model)
    gathered = [None] * 8
    dist.all_gather_object(gathered, dict(rank=rank, slots=slots, proofs=proofs, initial_trainable_sha256=initial_digest))
    if len({x['initial_trainable_sha256'] for x in gathered}) != 1:
        raise RuntimeError('ranks have different source initialization')
    config = dict(recipe=RECIPE, diagnostic_only=True, arm=args.arm, world_size=8, microbatch=1,
        accumulation=2, global_batch=16, total_updates=300, total_exposures=4800,
        unique_training_windows=4, windows=windows, cohort_sha256=digest(cohort_path),
        source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA, source_updates=6258,
        source_exposures=50064, source_optimizer_restored=False, source_scheduler_restored=False,
        vggt_revision=REVISION, geometry_quality_policy='monitor_v1',
        frozen_modules=['VGGT including AA/DINO/camera/depth', 'MASt3R and all understanding', 'panoptic/feedback controller', 'LPIPS'],
        trainable_prefixes=TRAIN_PREFIXES, feedback_enabled=True, beta=.1,
        understanding_weight=1. if args.arm == 'pred_joint' else 0.,
        gc='native reconstruction g_rec + .01*g_under; memory g_rec + g_under; absent under objective gives no gradients',
        reconstruction_loss='unchanged native layer losses/weights and visibility regularizers; no GT depth loss',
        precision='current A/B source: FP32 AA + BF16 DINO/autocast; camera/depth/decoder/loss FP32',
        optimizer='fresh native FP32 CPU-offload AdamW; original groups/betas/eps/decay/clip1',
        lr='original reconstruction1e-5/memory1e-4 peaks; common pilot20-update warmup then constant peak',
        cached_outputs='only frozen context-only features/understanding and detached geometry for two fixed windows/rank, RAM only',
        geometry='GT first-camera/baseline norm gauge for gt_recon; native monitor_v1 geometry otherwise',
        formal_official_results_eligible=False, auto_evaluation_submitted=False,
        mode=args.mode, execution_git_sha=execution_sha, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        hardware=dict(node=socket.gethostname().split('.')[0], gpu=torch.cuda.get_device_name(local)),
        rank_preparation=gathered)
    if rank == 0:
        native._write_json_atomic(args.output / 'run_manifest.json', config)
    monitor = native.GeometryMonitor(args.output, rank, 'diagnostic_' + args.arm, updates_per_epoch=300)
    limit = 2 if args.mode == 'smoke' else TOTAL_UPDATES
    counts = [0] * 4
    started = time.monotonic()

    def save_checkpoint(completed, filename):
        rngs, rank_counts = [None] * 8, [None] * 8
        dist.all_gather_object(rngs, native.capture_rank_rng())
        dist.all_gather_object(rank_counts, counts)
        totals = [sum(c[i] for c in rank_counts) for i in range(4)]
        if totals != [completed * 4] * 4:
            raise RuntimeError('diagnostic exposure accounting mismatch')
        if rank == 0:
            payload = dict(recipe=RECIPE, diagnostic_only=True, config=config, arm=args.arm,
                model=native.checkpoint_model_state(model), optimizer=optimizer.state_dict(),
                completed_updates=completed, completed_exposures=completed * GLOBAL_BATCH,
                rank_rng=rngs, rank_window_exposure_counts=rank_counts, window_exposure_counts=totals,
                source_checkpoint_sha256=SOURCE_SHA, adapted_vggt_identity=dict(path=str(SOURCE.parent / 'adapted_vggt.pt'),
                    sha256=native.CONTINUATION_VGGT_SHA), execution_git_sha=execution_sha,
                frozen_parameters_unchanged=True, initial_trainable_sha256=initial_digest)
            native._atomic_save(payload, args.output / filename)
            # Exercise serialization immediately in every smoke, including CPU
            # optimizer and strict model keys/shapes. Keep smoke clocks separate.
            check = torch.load(args.output / filename, map_location='cpu', mmap=True, weights_only=False)
            if check['completed_exposures'] != completed * GLOBAL_BATCH or len(check['rank_rng']) != 8:
                raise RuntimeError('saved diagnostic checkpoint clock/RNG mismatch')
            adam = check['optimizer']
            if adam['kind'] != 'native_fp32_cpu_adamw_v1':
                raise RuntimeError('wrong serialized optimizer')
            if completed and (not adam['optimizer']['state'] or any(
                int(s['step']) != completed or s['exp_avg'].dtype != torch.float32 or
                s['exp_avg_sq'].dtype != torch.float32 or s['exp_avg'].device.type != 'cpu'
                for s in adam['optimizer']['state'].values())):
                raise RuntimeError('serialized optimizer state clock/precision mismatch')
            state = check['model']
            native.restore_model_state_strict(model, state)
            if parameter_digest(named) != parameter_digest([(n, state[n]) for n, _ in named]):
                raise RuntimeError('saved diagnostic checkpoint parameter mismatch')
            del payload, check
        dist.barrier()

    if args.mode == 'train':
        save_checkpoint(0, 'checkpoint_update_0000.pt')
        # Strict load copies frozen tensors too; establish versions afterwards.
        frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
        frozen_buffers = {n: b._version for n, b in model.named_buffers()}
    for update in range(limit):
        batches = [dict(tree_to(data[slot], device), _diagnostic_slot=slot) for slot in slots]
        row = native.train_microbatch_window(model, optimizer, batches, update,
            base_exposure=50064 + update * GLOBAL_BATCH, world_size=8, accumulation=2,
            monitor=monitor, window_metadata=[native.window_identity(windows[slot]['source_index'], windows[slot]) for slot in slots],
            stage='diagnostic_' + args.arm, total_updates=TOTAL_UPDATES, lr_update=pilot_lr_update(update),
            cpu_saved_tensors=True, cpu_gradient_accumulation=True)
        del batches
        completed = update + 1
        for slot in slots:
            counts[slot] += 1
        error = None
        try:
            if any(p._version != frozen_versions[n] or p.grad is not None for n, p in model.named_parameters() if n in frozen_versions):
                raise RuntimeError('diagnostic modified a frozen parameter')
            if any(b._version != frozen_buffers[n] for n, b in model.named_buffers() if n in frozen_buffers):
                raise RuntimeError('diagnostic modified a frozen buffer')
            for prefix in probes:
                if not any(p.grad is not None and p.grad.norm() > 0 for n, p in named if n.startswith(prefix)):
                    raise RuntimeError('missing gradient in ' + prefix)
        except Exception as exc:
            error = exc
        native.synchronized_error(error, device, 'diagnostic_boundaries', update)
        row.update(arm=args.arm, diagnostic_only=True, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            window_slots=slots, objective_understanding_enabled=args.arm == 'pred_joint',
            feedback_enabled=True, frozen_parameters_unchanged=True)
        with (args.output / f'training_rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if rank == 0:
            native._write_json_atomic(args.output / 'progress.json', dict(arm=args.arm, mode=args.mode,
                completed_updates=completed, completed_exposures=completed * GLOBAL_BATCH, total_updates=limit,
                estimated_remaining_seconds=(time.monotonic() - started) / completed * (limit - completed),
                updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
            if completed % 20 == 0 or args.mode == 'smoke':
                print(json.dumps(row, allow_nan=False), flush=True)
        if args.mode == 'smoke' or completed == 20:
            proof_rows = [None] * 8
            changes = {prefix: float((next(p for n, p in named if n.startswith(prefix)).detach() - before).norm())
                       for prefix, before in probes.items()}
            dist.all_gather_object(proof_rows, dict(rank=rank, row=row, trainable_probe_changes=changes,
                trainable_sha256=parameter_digest(named), fixed_parameters_and_buffers_unchanged=True))
            if len({p['trainable_sha256'] for p in proof_rows}) != 1:
                raise RuntimeError('diagnostic rank parameters diverged')
            norms = [p['row']['preclip_norm'] for p in proof_rows]
            if max(norms) - min(norms) > 1e-6:
                raise RuntimeError('diagnostic averaged gradient norms disagree')
            if any(not all(v > 0 for v in p['trainable_probe_changes'].values()) for p in proof_rows):
                raise RuntimeError('intended parameters did not update')
            if rank == 0:
                native._write_json_atomic(args.output / ('smoke_update_proof.json' if args.mode == 'smoke' else 'startup_confirmation.json'),
                    dict(status='GPU_STEP_VERIFIED' if args.mode == 'smoke' else 'TRAINING_CONFIRMED',
                         completed_updates=completed, completed_exposures=completed * GLOBAL_BATCH,
                         diagnostic_only=True, formal_updates=0 if args.mode == 'smoke' else completed,
                         rank_rows=proof_rows, execution_git_sha=execution_sha))
        if (args.mode == 'smoke' and completed == limit) or (args.mode == 'train' and completed in (20, 100, 200, 300)):
            filename = f'checkpoint_update_{completed:04d}.pt'
            save_checkpoint(completed, filename)
            # A rank0 strict-load verification intentionally increments versions.
            frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
            frozen_buffers = {n: b._version for n, b in model.named_buffers()}
            if rank == 0:
                native._write_json_atomic(args.output / 'latest_checkpoint.json', dict(path=str(args.output / filename),
                    completed_updates=completed, completed_exposures=completed * GLOBAL_BATCH))
    if rank == 0:
        native._write_json_atomic(args.output / ('smoke_report.json' if args.mode == 'smoke' else 'COMPLETE.json'),
            dict(status='GPU_SMOKE_COMPLETED' if args.mode == 'smoke' else 'DIAGNOSTIC_TRAINING_COMPLETED',
                recipe=RECIPE, arm=args.arm, diagnostic_only=True, completed_updates=limit,
                completed_exposures=limit * GLOBAL_BATCH, formal_updates=0 if args.mode == 'smoke' else limit,
                source_checkpoint_sha256=SOURCE_SHA, frozen_parameters_unchanged=True,
                execution_git_sha=execution_sha, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                evaluation_pending_user_notification=True))
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
