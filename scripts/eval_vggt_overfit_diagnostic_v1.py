"""Read-only reconstruction evaluation of the completed four-window fitting study."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.train_vggt_overfit_diagnostic_v1 import (
    SOURCE, SOURCE_SHA, REVISION, RECIPE, ARMS, TRAIN_PREFIXES,
    digest, tree_to, configure_trainable, prepare_window, diagnostic_forward,
)
RUN = Path('/space/mawb/ssst/workspace_group_plus/vggt_overfit_diagnostic_v1')
TRAIN_SHA = 'bff606fc721f83a4040e4880bc082487de23b8f3'
STEPS = (0, 100, 200, 300)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_clock(blob, arm, step):
    expected = dict(recipe=RECIPE, diagnostic_only=True, arm=arm, completed_updates=step,
        completed_exposures=16 * step, source_checkpoint_sha256=SOURCE_SHA,
        execution_git_sha=TRAIN_SHA, frozen_parameters_unchanged=True)
    for key, value in expected.items():
        if blob.get(key) != value:
            raise ValueError('diagnostic checkpoint contract mismatch: ' + key)
    if len(blob['rank_rng']) != 8 or len(blob['rank_window_exposure_counts']) != 8:
        raise ValueError('eight-rank checkpoint contract mismatch')
    if blob['window_exposure_counts'] != [step * 4] * 4:
        raise ValueError('per-window exposure contract mismatch')
    actual = [sum(row[i] for row in blob['rank_window_exposure_counts']) for i in range(4)]
    if actual != blob['window_exposure_counts']:
        raise ValueError('rank exposure accounting mismatch')
    config = blob['config']
    for key, value in dict(arm=arm, world_size=8, global_batch=16, accumulation=2,
        unique_training_windows=4, total_updates=300, total_exposures=4800,
        feedback_enabled=True, beta=.1, geometry_quality_policy='monitor_v1',
        source_checkpoint_sha256=SOURCE_SHA, vggt_revision=REVISION,
        understanding_weight=1. if arm == 'pred_joint' else 0.).items():
        if config.get(key) != value:
            raise ValueError('diagnostic configuration mismatch: ' + key)


def validate_checkpoints(root):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    torch.set_num_threads(4)
    if (root / 'checkpoint_validation.json').exists():
        raise ValueError('validation receipt exists; preserve previous evidence')
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError('common source SHA mismatch')
    if digest(SOURCE.parent / 'adapted_vggt.pt') != native.CONTINUATION_VGGT_SHA:
        raise ValueError('adapted frozen VGGT asset SHA mismatch')
    cohort_file = REPO / 'docs/vggt_overfit_cohort_v1.json'
    windows = json.loads(cohort_file.read_text())['windows']
    source = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=False)
    state = source['model']
    records, initials = [], []
    jobs = []
    for arm in ARMS:
        complete = json.loads((RUN / arm / 'COMPLETE.json').read_text())
        if complete['status'] != 'DIAGNOSTIC_TRAINING_COMPLETED' or complete['completed_updates'] != 300 or complete['completed_exposures'] != 4800 or complete['execution_git_sha'] != TRAIN_SHA:
            raise ValueError('training completion receipt mismatch')
        jobs.append(complete['slurm_job_id'])
        for step in STEPS:
            path = RUN / arm / f'checkpoint_update_{step:04d}.pt'
            blob = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            validate_clock(blob, arm, step)
            if blob['config']['windows'] != windows or blob['config']['cohort_sha256'] != digest(cohort_file):
                raise ValueError('checkpoint cohort differs')
            if blob['adapted_vggt_identity']['sha256'] != native.CONTINUATION_VGGT_SHA:
                raise ValueError('frozen VGGT asset identity differs')
            if set(blob['model']) != set(state) or any(blob['model'][k].shape != state[k].shape for k in state):
                raise ValueError('strict non-VGGT model keys/shapes differ')
            frozen = [k for k in state if not k.startswith(TRAIN_PREFIXES)]
            if any(not torch.equal(blob['model'][k], state[k]) for k in frozen):
                raise ValueError('frozen model tensors changed')
            if step == 0 and any(not torch.equal(blob['model'][k], state[k]) for k in state):
                raise ValueError('initial checkpoint differs from common source')
            adam = blob['optimizer']
            if adam['kind'] != 'native_fp32_cpu_adamw_v1':
                raise ValueError('optimizer kind differs')
            if step and (not adam['optimizer']['state'] or any(int(s['step']) != step for s in adam['optimizer']['state'].values())):
                raise ValueError('optimizer step differs')
            if step == 0 and adam['optimizer']['state']:
                raise ValueError('initial optimizer is not fresh')
            initials.append(blob['initial_trainable_sha256'])
            records.append(dict(arm=arm, step=step, path=str(path), sha256=digest(path),
                frozen_non_vggt_tensors_unchanged=True, strict_model_keys_and_shapes=True,
                completed_exposures=step * 16, windows=windows))
            del blob
    if len(set(initials)) != 1:
        raise ValueError('arms do not share common initial trainable tensors')
    sacct = subprocess.check_output(['sacct', '-n', '-X', '-j', ','.join(jobs), '-o', 'JobIDRaw,State,ExitCode', '-P'], text=True)
    expected = {job + '|COMPLETED|0:0' for job in jobs}
    if not expected.issubset({row.rstrip('|') for row in sacct.splitlines()}):
        raise ValueError('Slurm completion/exit status mismatch')
    receipt = dict(status='ALL_CHECKPOINTS_VERIFIED', diagnostic_only=True, training_git_sha=TRAIN_SHA,
        records=records, slurm_completed_jobs=jobs, cohort_sha256=digest(cohort_file),
        common_source_checkpoint_sha256=SOURCE_SHA, adapted_vggt_sha256=native.CONTINUATION_VGGT_SHA,
        training_helper_sha256=digest(REPO / 'scripts/train_vggt_overfit_diagnostic_v1.py'))
    write(root / 'checkpoint_validation.json', receipt)
    print(json.dumps({'status': receipt['status'], 'checkpoints': len(records)}), flush=True)


def export(root):
    from scripts import object_locus_frozen_vggt_posefree_runtime as native
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _restore_checkpoint
    from scripts.diagnose_vggt_reconstruction_v1 import image_psnr, render
    from scripts.evaluate_ssst_validation import save_rgb
    from tokengs.models.object_locus_posefree_geometry import camera_vectors
    from tokengs.models.input_types import ModelInputDecoder
    from tokengs.models.object_locus_v3_set_loss import object_locus_v3_set_losses
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world != 4 or not 0 <= rank < 4:
        raise ValueError('evaluation requires four shards, one per fixed training window')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    receipt = json.loads((root / 'checkpoint_validation.json').read_text())
    if receipt['training_helper_sha256'] != digest(REPO / 'scripts/train_vggt_overfit_diagnostic_v1.py'):
        raise ValueError('evaluation helper differs from verified training')
    shard = root / f'rank{rank:02d}'
    if shard.exists():
        raise ValueError('evaluation shard already exists; preserve evidence')
    shard.mkdir()
    native.seed_everything(42)
    os.environ['VGGT_HF_REVISION'] = REVISION
    model, opt, _, _ = native.build_model(preserve_fp32_aggregator=True)
    model.to(device=device, dtype=torch.float32)
    _restore_checkpoint(model, SOURCE, REVISION, digest(native.MANIFEST))
    model.frozen_vggt.set_reconstruction_adaptation(False, keep_fp32=True)
    native.restore_adapted_vggt(model, SOURCE.parent / 'adapted_vggt.pt', native.CONTINUATION_VGGT_SHA)
    configure_trainable(model)
    windows = json.loads((REPO / 'docs/vggt_overfit_cohort_v1.json').read_text())['windows']
    window = windows[rank]
    data, cached_cpu, prep = prepare_window(model, opt, window, device)
    for record in receipt['records']:
        blob = torch.load(record['path'], map_location='cpu', mmap=True, weights_only=False)
        proofs = [p for r in blob['config']['rank_preparation'] for p in r['proofs'] if p['window']['source_index'] == window['source_index']]
        if len(proofs) != 4 or any(p['input_rgb_sha256'] != prep['input_rgb_sha256'] for p in proofs):
            raise ValueError('evaluation/training provider pixels differ')
        del blob
    model.frozen_vggt.cpu()
    model.understanding.encoder.cpu()
    model.understanding.adapter.cpu()
    model.understanding.mask2former.pixel_decoder.cpu()
    model.understanding.mask2former.transformer_module.cpu()
    model.understanding.mask_embedder.to(device)
    model.eval().requires_grad_(False)
    batch, cached = tree_to(data, device), tree_to(cached_cpu, device)
    del data, cached_cpu
    gc.collect()
    torch.cuda.empty_cache()
    rows = []
    name = window['scene'] + '_context' + '_'.join(map(str, window['context']))
    with torch.no_grad():
        for record in receipt['records']:
            arm, step = record['arm'], record['step']
            blob = torch.load(record['path'], map_location='cpu', mmap=True, weights_only=False)
            validate_clock(blob, arm, step)
            native.restore_model_state_strict(model, blob['model'])
            del blob
            model.understanding_step = 50064 + step * 16
            versions = {n: p._version for n, p in model.named_parameters()}
            output, metrics = diagnostic_forward(model, batch, cached, arm, compute_understanding=False)
            generated = output['prediction']
            gt = arm == 'gt_recon'
            cams = cached['gt_c2w'] if gt else cached['calibration']['c2w']
            intrinsics = cached['gt_intrinsics'] if gt else cached['calibration']['intrinsics']
            final_render = render(model, generated['reconstruction'], cams, intrinsics)
            pred, truth = final_render['images_pred'][0].float(), batch['images_all'][0].float()
            if not torch.isfinite(pred).all():
                raise FloatingPointError('nonfinite evaluated reconstruction')
            mse, psnr = image_psnr(pred, truth)
            # Probe the same understanding objective even for recon-only arms.
            context_cams = cams[:, :2]
            context_intr = intrinsics[:, :2]
            ctx = ModelInputDecoder(cam_view=torch.linalg.inv(context_cams).transpose(-1, -2), intrinsics=context_intr)
            generated.update(model._readout(generated['states'][-1], generated['gaussians'], cached['fm'], ctx, ctx))
            under, under_metrics = object_locus_v3_set_losses(generated, batch, model.opt)
            if any(p._version != versions[n] for n, p in model.named_parameters()):
                raise ValueError('read-only evaluation modified model parameters')
            directory = root / 'png' / arm / f'update{step:04d}' / name
            (directory / 'rgb').mkdir(parents=True)
            (directory / 'rgb_gt').mkdir()
            frames = batch['frame_ids'][0].tolist()
            for i, frame in enumerate(frames):
                save_rgb(directory / 'rgb' / f'{frame:06d}.png', pred[i])
                save_rgb(directory / 'rgb_gt' / f'{frame:06d}.png', truth[i])
            raw_dir = root / 'floating' / arm / f'update{step:04d}'
            raw_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_dir / (name + '.npz'), prediction=pred.cpu().numpy(), truth=truth.cpu().numpy())
            alpha = final_render['alphas_pred'][0]
            row = dict(arm=arm, step=step, scene=window['scene'], context=window['context'],
                source_index=window['source_index'], frame_ids=frames, raw_float_psnr=psnr.tolist(),
                raw_float_mse=mse.tolist(), reconstruction_loss=float(metrics['loss_recon']),
                understanding_loss_probe=float(under),
                matched_classification_accuracy=under_metrics['matched_classification_accuracy'],
                alpha_mean=float(alpha.mean()), rgb_outside_0_1=float(((pred < 0) | (pred > 1)).float().mean()),
                input_rgb_sha256=prep['input_rgb_sha256'], checkpoint_sha256=record['sha256'],
                geometry='GT first-camera/baseline norm gauge' if gt else 'frozen native monitor_v1 predicted geometry')
            rows.append(row)
            write(shard / 'export_progress.json', dict(completed=len(rows), total=12, latest=dict(arm=arm, step=step)))
            print(json.dumps({'rank': rank, 'arm': arm, 'step': step, 'float_psnr': float(psnr.mean())}), flush=True)
            del output, metrics, generated, final_render, pred, truth, mse, psnr, alpha
    write(shard / 'export_complete.json', dict(status='EXPORT_COMPLETED', diagnostic_only=True,
        rank=rank, rows=rows, initial_native_replay=prep, optimizer_updates=0,
        evaluation_git_sha=os.environ['TASK_CODE_SHA']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--validate-checkpoints', action='store_true')
    args = p.parse_args()
    if args.validate_checkpoints:
        validate_checkpoints(args.root)
    else:
        export(args.root)


if __name__ == '__main__':
    main()
