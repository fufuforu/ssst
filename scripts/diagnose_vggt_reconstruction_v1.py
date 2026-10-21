"""Paired inference interventions. DIAGNOSTIC ONLY; never a training/eval entry point."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_frozen_joint_epoch_04.pt')
SOURCE_SHA = '01d49b092c8dfc2e02f15b5216e5d6ed1cfd88fae2cf693ee69334e45a9410ba'
REVISION = '860abec7937da0a4c03c41d3c269c366e82abdf9'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def gt_cameras(provider_c2w, predicted_context, context_scene_depth):
    """Align GT to the first camera; preserve two explicitly different scale gauges."""
    relative = torch.linalg.inv(provider_c2w[:, :1]) @ provider_c2w
    gt_baseline = (relative[:, 1, :3, 3] - relative[:, 0, :3, 3]).norm(dim=-1)
    pred_baseline = (predicted_context[:, 1, :3, 3] - predicted_context[:, 0, :3, 3]).norm(dim=-1)
    medians = []
    for depths in context_scene_depth:
        valid = depths[torch.isfinite(depths) & (depths > 0)]
        if not valid.numel():
            raise ValueError('missing positive GT context depths')
        medians.append(valid.median())
    median = torch.stack(medians)
    if (gt_baseline <= 1e-8).any() or (pred_baseline <= 1e-8).any():
        raise ValueError('degenerate baseline: cannot identify this diagnostic gauge')
    baseline_factor = pred_baseline / gt_baseline
    depth_factor = .25 / median
    def scaled(factor):
        cams = relative.clone()
        cams[..., :3, 3] *= factor[:, None, None]
        return cams
    return scaled(baseline_factor), scaled(depth_factor), relative, {
        'baseline_factor': baseline_factor.tolist(), 'depth_factor': depth_factor.tolist(),
        'gt_context_median_scene_depth': median.tolist(),
        'gt_provider_baseline': gt_baseline.tolist(), 'predicted_baseline': pred_baseline.tolist(),
    }


def k_matrix(vector):
    k = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(*vector.shape[:-1], 3, 3).clone()
    k[..., 0, 0], k[..., 1, 1] = vector[..., 0], vector[..., 1]
    k[..., 0, 2], k[..., 1, 2] = vector[..., 2], vector[..., 3]
    return k


def image_psnr(pred, truth):
    mse = (pred - truth).square().flatten(1).mean(1)
    # Pinned torchmetrics initializes min_target/max_target at zero. Preserve
    # that detail rather than silently substituting a conventional range.
    ranges = truth.flatten(1).amax(1).clamp_min(0) - truth.flatten(1).amin(1).clamp_max(0)
    if (mse <= 0).any() or (ranges <= 0).any():
        raise ValueError('undefined image PSNR')
    return mse, 10 * torch.log10(ranges.square() / mse)


def cohort(n):
    from scripts.object_locus_frozen_vggt_posefree_runtime import MANIFEST, load_manifest
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import EXPECTED_FULL_VALIDATION_SHA256
    manifest, _, train = load_manifest(MANIFEST)
    path = Path(manifest['full_validation_source'])
    if sha(path) != EXPECTED_FULL_VALIDATION_SHA256:
        raise ValueError('validation cohort identity mismatch')
    valid = []
    for row in json.loads(path.read_text()):
        ctx = list(map(int, row['context_ids']))
        valid.append(dict(scene=row['scan'], context=ctx,
                          novel=[int(x) for x in row['target_ids'] if int(x) not in ctx], pair_iou=row.get('iou')))
    if len(train) != 8337 or len(valid) != 1860 or set(w['scene'] for w in train) & set(w['scene'] for w in valid):
        raise ValueError('cohort counts or train/validation separation mismatch')
    selected = []
    for name, windows in [('train', train), ('heldout', valid)]:
        rng = np.random.default_rng(42)
        scenes = sorted({w['scene'] for w in windows})
        if not 1 <= n <= len(scenes):
            raise ValueError('invalid per-split sample count')
        for scene in rng.choice(scenes, n, replace=False):
            candidates = [(i, w) for i, w in enumerate(windows) if w['scene'] == scene]
            i, w = candidates[int(rng.integers(len(candidates)))]
            selected.append(dict(w, split=name, source_index=i))
    return selected


def render(model, reconstruction, cams, intrinsics):
    from tokengs.models.input_types import ModelInputDecoder
    return model.render_reconstruction(reconstruction, ModelInputDecoder(
        cam_view=torch.linalg.inv(cams).transpose(-1, -2), intrinsics=intrinsics))


def decode(model, latent, fm, qpre, context_cams, context_k, feedback=True, *, original_k518=None):
    from tokengs.models.object_locus_posefree_geometry import rays_to_patch_plucker
    if original_k518 is None:
        k518 = context_k.clone()
        k518[..., :2, :] *= 518 / 256
    else:
        # Native predicted rays use the original K518. A K256 round trip can
        # change one ULP, which propagates through decoder/rasterizer boundaries.
        k518 = original_k518
    rays = rays_to_patch_plucker(context_cams, k518, patch_size=14)
    tokens = model.get_gs_tokens(batch_size=context_cams.shape[0])
    if feedback:
        states, _ = model.anchor_decoder.forward_stateful(tokens, latent, rays, model.panoptic,
            fm, qpre, model.understanding.mask_embedder, model.understanding_step)
    else:
        states, _ = model.anchor_decoder(tokens, latent, rays)
    last = states[-1]
    gs = model.activation_head(last['tokens'], last['mu'], last['radii'])
    return model._reconstruction_from_gaussians(gs)


def record(root, rows, window, variant, output, batch, **extras):
    from scripts.evaluate_ssst_validation import save_rgb
    pred, truth = output['images_pred'][0].float(), batch['images_all'][0].float()
    if not torch.isfinite(pred).all():
        raise FloatingPointError(f'nonfinite RGB in {variant}')
    mse, psnr = image_psnr(pred, truth)
    name = f"{window['split']}_{window['scene']}_" + '_'.join(map(str, window['context']))
    directory = root / 'images' / name / variant
    directory.mkdir(parents=True)
    for i, frame in enumerate(batch['frame_ids'][0].tolist()):
        save_rgb(directory / f'{frame:06d}.png', pred[i])
        gt_path = directory.parent / 'gt' / f'{frame:06d}.png'
        if not gt_path.exists():
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            save_rgb(gt_path, truth[i])
    alpha = output['alphas_pred'][0]
    row = dict(split=window['split'], scene=window['scene'], context=window['context'],
        novel=window['novel'], source_index=window['source_index'], variant=variant,
        frame_ids=batch['frame_ids'][0].tolist(), psnr=psnr.tolist(), mse=mse.tolist(),
        psnr_data_range_1=(-10 * torch.log10(mse)).tolist(),
        alpha_mean=float(alpha.mean()), alpha_below_0_1=float((alpha < .1).float().mean()),
        rgb_outside_0_1=float(((pred < 0) | (pred > 1)).float().mean()), **extras)
    row['input_rgb_sha256'] = hashlib.sha256(truth.detach().cpu().numpy().tobytes()).hexdigest()
    rows.append(row)
    with (root / 'rows.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps({k: row[k] for k in ('split', 'scene', 'variant')} |
                     {'psnr': float(psnr.mean()), 'alpha': row['alpha_mean']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-split', type=int, default=16)
    p.add_argument('--reuse-old-results', type=Path,
                   help='completed legacy inference from a preserved failed diagnostic attempt')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('diagnostic output must be new; no evidence overwrite')
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(REPO))
    from scripts.object_locus_frozen_vggt_posefree_runtime import build_model, MANIFEST, CHECKPOINT, EXPECTED_CHECKPOINT_SHA
    from scripts.object_locus_v3_set_runtime import build_batch, seed_everything
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _restore_checkpoint
    from tokengs.models.object_locus_posefree_geometry import camera_vectors
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    seed_everything(42)
    windows = cohort(args.per_split)
    if sha(SOURCE) != SOURCE_SHA or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError('source checkpoint identity mismatch')
    receipt = dict(diagnostic_only=True, gt_oracles_excluded_from_official_results=True,
        optimizer_updates=0, source=str(SOURCE), source_sha256=SOURCE_SHA,
        old_source=str(CHECKPOINT), old_source_sha256=EXPECTED_CHECKPOINT_SHA,
        manifest_sha256=sha(MANIFEST), vggt_revision=REVISION,
        code_sha=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), windows=windows,
        metric='Per-image floating RGB PSNR, default dynamic GT data range as pinned SIU3R; no scope subdivision. PNGs for visual inspection only.')
    (args.output / 'identity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    rows = []
    if args.reuse_old_results is not None:
        previous = json.loads((args.reuse_old_results / 'identity.json').read_text())
        if previous['windows'] != windows or previous['old_source_sha256'] != EXPECTED_CHECKPOINT_SHA:
            raise ValueError('reused legacy results have a different source/cohort')
        previous_rows = [json.loads(x) for x in (args.reuse_old_results / 'rows.jsonl').read_text().splitlines()]
        rows = [r for r in previous_rows if r['variant'] == 'old_locus_gt']
        expected = {(w['split'], w['scene'], tuple(w['context']), tuple(w['novel'])) for w in windows}
        actual = {(r['split'], r['scene'], tuple(r['context']), tuple(r['novel'])) for r in rows}
        if len(rows) != len(windows) or actual != expected:
            raise ValueError('reused legacy results are incomplete or duplicated')
        with (args.output / 'rows.jsonl').open('w') as f:
            for row in rows:
                row['reused_from'] = str(args.reuse_old_results)
                f.write(json.dumps(row, allow_nan=False) + '\n')
        receipt['old_inference_reused_from'] = str(args.reuse_old_results)
        receipt['old_inference_code_sha'] = previous['code_sha']
        (args.output / 'identity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # Sequential models prevent peak GPU memory competing with A/B.
    from scripts.object_locus_panoptic_v1_runtime import build_model as build_old
    from tokengs.models.input_types import split_data, ModelInputDecoder
    if args.reuse_old_results is None:
      old, opt = build_old('cpu', report=False)
      blob = torch.load(CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
      old.load_state_dict(blob['model'], strict=True)
      old.understanding_step = 50064
      del blob
      old = old.to('cuda').eval()
      old_versions = [(id(p), p._version) for p in old.parameters()]
      with torch.no_grad():
        for window in windows:
            batch = build_batch(opt, window, 'cuda')
            mi, _ = split_data(batch, opt)
            decoder = ModelInputDecoder(cam_view=batch['cam_view_all'], intrinsics=batch['intrinsics_all'])
            output = old.forward_object_locus(mi, render_decoder_input=decoder, step=50064)
            record(args.output, rows, window, 'old_locus_gt', output['render'], batch)
            if old_versions != [(id(p), p._version) for p in old.parameters()]:
                raise ValueError('old-model inference changed parameters')
            del output, mi, decoder, batch
      del old
    gc.collect()
    torch.cuda.empty_cache()
    os.environ['VGGT_HF_REVISION'] = REVISION
    model, opt, _, _ = build_model()
    model = model.to(device='cuda', dtype=torch.float32)
    _restore_checkpoint(model, SOURCE, REVISION, sha(MANIFEST))
    model.eval()
    versions = [(id(p), p._version) for p in model.parameters()]
    old_rgb = {(r['split'], r['scene']): r['input_rgb_sha256'] for r in rows}
    with torch.no_grad():
        for window in windows:
            batch = build_batch(opt, window, 'cuda')
            rgb_sha = hashlib.sha256(batch['images_all'][0].float().cpu().numpy().tobytes()).hexdigest()
            if rgb_sha != old_rgb[(window['split'], window['scene'])]:
                raise ValueError('old/new provider pixels differ on the paired window')
            captured = []
            hook = model.frozen_vggt.register_forward_hook(lambda _m, _i, out: captured.append(out))
            try:
                generated = model.generate(batch['images_input'])
            finally:
                hook.remove()
            if len(captured) != 1:
                raise ValueError('context features were not captured exactly once')
            latent = model._latent_from_vggt(captured[0])
            captured.clear()
            calibrated = model.calibrate_targets(batch['images_all'], generated)
            pred_cams, pred_k = calibrated['c2w'], calibrated['intrinsics']
            ctx_cams = generated['predicted_context_c2w']
            ctx_k = generated['predicted_context_intrinsics_matrix']
            cams = torch.linalg.inv(batch['cam_view_all'].transpose(-1, -2))
            gt_k = k_matrix(batch['intrinsics_all'])
            gt_base, gt_depth, gt_legacy, gauges = gt_cameras(cams, ctx_cams, batch['depth_gt_scene_all'][:, :2])
            rec = generated['reconstruction']
            original = generated['gaussians'].clone()
            baseline = render(model, rec, pred_cams, pred_k)
            official = model.render_generated_at(generated, torch.linalg.inv(pred_cams).transpose(-1, -2), pred_k)
            identity_error = float((baseline['images_pred'] - official['images_pred']).abs().max())
            if identity_error > 1e-6:
                raise ValueError('diagnostic renderer differs from native production RGB')
            del official
            # Cached-feature replay must reproduce native GS before any intervention.
            replay = decode(model, latent, generated['F_m'], generated['q_pre'], ctx_cams, ctx_k,
                            original_k518=generated['predicted_context_intrinsics518'])
            replay_rgb = render(model, replay, pred_cams, pred_k)['images_pred']
            replay_error = float((baseline['images_pred'] - replay_rgb).abs().max())
            if replay_error > 1e-5:
                raise ValueError(f"cached-feature decoder replay changed native RGB: max={replay_error}, scene={window['scene']}")
            del replay, replay_rgb
            record(args.output, rows, window, 'current', baseline, batch,
                   native_renderer_error=identity_error, cached_replay_error=replay_error, gauges=gauges,
                   cameras=dict(predicted_c2w=pred_cams.tolist(), gt_provider_c2w=cams.tolist(),
                                predicted_intrinsics=pred_k.tolist(), gt_intrinsics=batch['intrinsics_all'].tolist()))
            for variant, camera, intr in [
                ('same_gs_gt_k', pred_cams, camera_vectors(gt_k)),
                ('same_gs_gt_pose_baseline', gt_base, pred_k),
                ('same_gs_gt_camera_baseline', gt_base, camera_vectors(gt_k)),
                ('same_gs_gt_camera_depth', gt_depth, camera_vectors(gt_k)),
            ]:
                record(args.output, rows, window, variant, render(model, rec, camera, intr), batch)
                if not torch.equal(generated['gaussians'], original):
                    raise ValueError('camera intervention mutated Gaussians')
            for variant, ctx, k, target, intr, feedback in [
                ('same_features_gt_rays_baseline', gt_base[:, :2], gt_k[:, :2], gt_base, camera_vectors(gt_k), True),
                ('same_features_gt_rays_legacy_units', gt_legacy[:, :2], gt_k[:, :2], gt_legacy, camera_vectors(gt_k), True),
                ('feedback_off', ctx_cams, ctx_k, pred_cams, pred_k, False),
            ]:
                altered = decode(model, latent, generated['F_m'], generated['q_pre'], ctx, k, feedback,
                    original_k518=generated['predicted_context_intrinsics518'] if variant == 'feedback_off' else None)
                record(args.output, rows, window, variant, render(model, altered, target, intr), batch)
                del altered
            for scale in (.5, 2.):
                ctx, target = ctx_cams.clone(), pred_cams.clone()
                ctx[..., :3, 3] *= scale
                target[..., :3, 3] *= scale
                altered = decode(model, latent, generated['F_m'], generated['q_pre'], ctx, ctx_k,
                                 original_k518=generated['predicted_context_intrinsics518'])
                record(args.output, rows, window, f'predicted_ray_scale_{scale}', render(model, altered, target, pred_k), batch)
                del altered
            if versions != [(id(p), p._version) for p in model.parameters()]:
                raise ValueError('inference changed parameter tensors')
            del baseline, rec, generated, original, latent, batch, calibrated
    summary = {}
    for split in ('train', 'heldout'):
        subset = [r for r in rows if r['split'] == split]
        base = {r['scene']: float(np.mean(r['psnr'])) for r in subset if r['variant'] == 'current'}
        for variant in sorted({r['variant'] for r in subset}):
            rs = [r for r in subset if r['variant'] == variant]
            delta = [float(np.mean(r['psnr'])) - base[r['scene']] for r in rs]
            summary[f'{split}/{variant}'] = dict(windows=len(rs), images=sum(len(r['psnr']) for r in rs),
                psnr=float(np.mean([p for r in rs for p in r['psnr']])),
                paired_delta_mean=float(np.mean(delta)), paired_delta_median=float(np.median(delta)),
                improved_windows=sum(d > 0 for d in delta),
                paired_delta_standard_error=float(np.std(delta, ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else None)
    (args.output / 'summary.json').write_text(json.dumps(dict(diagnostic_only=True, results=summary), indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
