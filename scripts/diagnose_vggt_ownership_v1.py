"""Fixed C/Baseline-300 observations, stages, and regional RGB error; no updates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# Native compatibility bootstrap must precede every model-dependent import.
from scripts import object_locus_frozen_vggt_posefree_runtime as native
from scripts.vggt_ownership_math_v1 import feedback_attribution, ownership_metrics, parent_pool, spatial
from scripts.eval_vggt_joint_short_controls_v1 import write

ROOT = Path('/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1')
READING = Path('/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04')
C_EVAL = Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation')
BASE_EVAL = READING / 'baseline/evaluation'
HELDOUT = Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json')
LAYERS = (6, 8, 10, 12)


class OutputReproductionMismatch(RuntimeError):
    pass


def read(p):
    return json.loads(p.read_text())


def catalogs():
    result = {}
    for arm, path in [('C', C_EVAL), ('baseline300', BASE_EVAL)]:
        result[arm] = {}
        for f in path.glob('rank*/export_complete.json'):
            for row in read(f)['records']:
                result[arm][row['name']] = dict(record=row, pair=str(f.parent / 'all' / row['name']))
        assert len(result[arm]) == 128
    return result


class Observer:
    def __init__(self, model):
        self.values = {}
        self.enabled = True
        for key, module in model.panoptic.layers.items():
            def hook(layer, inputs, output, key=key):
                if not self.enabled:
                    return
                norm, signed, rebuilt, error = feedback_attribution(
                    inputs[0], output['q'], output['route'], layer.W_inject,
                    inputs[10], model.feedback_peak, output['joint_delta'])
                self.values[key] = dict(norm=norm[0].detach(), signed=signed[0].detach(), error=error,
                                        residual_fraction=float(output['joint_delta'].norm() / inputs[0].norm().clamp_min(1e-12)))
            module.register_forward_hook(hook)


def iou_matrix(masks, gt):
    p, t = masks.float().flatten(1), gt.float().flatten(1)
    inter = p @ t.T
    return inter / (p.sum(1)[:, None] + t.sum(1)[None] - inter).clamp_min(1)


def projected_xy(generated, mu):
    w2c = torch.linalg.inv(generated['predicted_context_c2w'][0])
    xyz = torch.einsum('vij,nj->vni', w2c[:, :3, :3], mu) + w2c[:, None, :3, 3]
    xyh = torch.einsum('vij,vnj->vni', generated['predicted_context_intrinsics_matrix'][0], xyz)
    xy = xyh[..., :2] / xyh[..., 2:].clamp_min(1e-6)
    valid = (xyz[..., 2] > 1e-6) & (xy[..., 0] >= 0) & (xy[..., 0] < 256) & (xy[..., 1] >= 0) & (xy[..., 1] < 256)
    return xy, valid


def regional_rgb(render, batch):
    """Disjoint GT object interior / boundary / stuff interior / unlabeled.

    Boundary is a 2-pixel Chebyshev band around valid semantic/instance changes.
    Report summed SSE and pixel counts as well as mean per-frame masked PSNR.
    This is a diagnostic partition, never a replacement official RGB metric.
    """
    rgb = render['images_pred'][0].clamp(0, 1)
    truth = batch['images_all'][0]
    # Use official PNG quantization for additive regional SSE. Regional PSNR
    # uses data_range=1; official torchmetrics infers the target image range.
    rgb = (rgb * 255 + .5).to(torch.uint8).float() / 255
    truth = (truth.clamp(0, 1) * 255 + .5).to(torch.uint8).float() / 255
    sem, ins = batch['semantic_label_all'][0], batch['instance_label_all'][0]
    valid = (sem >= 0) & (sem <= 19)
    obj = valid & (sem >= 2) & (ins > 0)
    stuff = valid & (sem < 2)
    label = torch.where(obj, ins + 100, torch.where(stuff, sem + 1, 0)).float()
    maximum = F.max_pool2d(label[:, None], 5, 1, 2)[:, 0]
    minimum = -F.max_pool2d(-label[:, None], 5, 1, 2)[:, 0]
    boundary = (maximum != minimum) & valid
    regions = dict(object_interior=obj & ~boundary, boundary=boundary,
                   background_stuff=stuff & ~boundary, other=~valid | (valid & ~obj & ~stuff & ~boundary))
    assert torch.stack(list(regions.values())).sum(0).eq(1).all()
    error = (rgb - truth).square().mean(1)
    result = {}
    for key, mask in regions.items():
        per_frame = []
        for e, m in zip(error, mask):
            count = int(m.sum())
            per_frame.append(dict(pixels=count, sse=float(e[m].sum()),
                                  psnr=float(-10 * e[m].mean().clamp_min(1e-12).log10()) if count else None))
        result[key] = per_frame
    return result


def reproduce(output, batch, window, golden, scratch):
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _official_export_view
    from scripts.evaluate_ssst_validation import save_rgb
    from scripts.train_vggt_joint_short_controls_v1 import digest
    row = write_official_pair(_official_export_view(output['render']), batch, window, scratch, target_frames='all')
    pair, reference = scratch / golden['record']['name'], Path(golden['pair'])
    for sub in ('context_seg_pred', 'target_seg_pred', 'context_seg_gt', 'target_seg_gt'):
        expected = sorted(f.name for f in (reference / sub).glob('*.png'))
        assert expected == sorted(f.name for f in (pair / sub).glob('*.png'))
        same = all(digest(reference / sub / n) == digest(pair / sub / n) for n in expected)
        if not same and sub.endswith('_pred'):
            raise OutputReproductionMismatch((window, sub))
        assert same, (window, sub)
    for view in ('context', 'target'):
        old = read(reference / f'{view}_seg_pred/pred.json')
        actual = row['pred_segments'][view]
        assert len(old) == len(actual)
        for x, y in zip(actual, old):
            assert {k: v for k, v in x.items() if k != 'score'} == {k: v for k, v in y.items() if k != 'score'}
            assert abs(x['score'] - y['score']) <= 1e-6
    max_rgb = 0
    for i, frame in enumerate(batch['frame_ids'][0].tolist()):
        rgb = output['render']['images_pred'][0, i].clamp(0, 1)
        arr = (rgb.cpu().numpy().transpose(1, 2, 0) * 255 + .5).astype(np.uint8)
        previous = np.asarray(Image.open(reference / 'rgb' / f"{window['scene']}_{frame}.png"))
        max_rgb = max(max_rgb, int(np.abs(arr.astype(int) - previous.astype(int)).max()))
    assert max_rgb <= 1, ('RGB reproduction exceeds one quantization bin', max_rgb)
    # Temporary PNGs certify existing official outputs; retain only small receipts.
    import shutil
    shutil.rmtree(pair)
    return dict(segmentation_and_GT_PNGs_identical=True, scores_max_tolerance=1e-6, rgb_max_uint8_delta=max_rgb)


def observe_window(model, opt, obs, window, choice, golden, destination, device, witness=False, reproduction_attempt=0):
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render
    from scripts.probe_vggt_raw_query_readout_v1 import raw_stages
    from tokengs.models.object_locus_v3_set_loss import final_hungarian
    from tokengs.models.object_locus_panoptic_v1_lift import RendererTranspose
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    from tokengs.models.object_locus_v3_set import alpha_normalize_membership
    native.seed_everything(42 + window['source_index'])
    batch = build_batch(opt, window, device)
    input_hash = hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest()
    assert input_hash == choice['input_rgb_sha256']
    assert batch['frame_ids'][0].tolist() == choice['frame_ids']
    witness_result = None
    if witness:
        obs.enabled = False
        references = []
        native_segments = []
        for _ in range(2):
            with torch.no_grad():
                old = generate_and_render(model, batch['images_input'], batch['images_all'])
            references.append({k: old['generated'][k].detach().clone() for k in ('gaussians', 'gaussian_membership', 'p_class')})
            references[-1]['query12'] = old['generated']['states'][-1]['q'].detach().clone()
            references[-1]['anchor12'] = old['generated']['states'][-1]['anchor_embedding'].detach().clone()
            sem, ins, _ = assemble_panoptic(old['render'])
            native_segments.append((sem * 1000 + ins).detach().clone())
            del old
        obs.enabled = True
    obs.values.clear()
    with torch.no_grad():
        output = generate_and_render(model, batch['images_input'], batch['images_all'])
        g, render = output['generated'], output['render']
        if witness:
            witness_result = {}
            actual_fields = dict(gaussians=g['gaussians'], gaussian_membership=g['gaussian_membership'], p_class=g['p_class'],
                                 query12=g['states'][-1]['q'], anchor12=g['states'][-1]['anchor_embedding'])
            for key in references[0]:
                native_jitter = float((references[0][key] - references[1][key]).abs().max())
                observed_jitter = float((references[1][key] - actual_fields[key]).abs().max())
                assert native_jitter <= 1e-5 and observed_jitter <= 1e-5, (key, native_jitter, observed_jitter)
                if key in ('gaussians','query12','anchor12'):
                    assert native_jitter == observed_jitter == 0
                witness_result[key] = dict(native_repeat=native_jitter, observed_difference=observed_jitter)
            sem, ins, _ = assemble_panoptic(render)
            packed = sem * 1000 + ins
            witness_result['native_repeat_panoptic_pixels'] = int((native_segments[0] != native_segments[1]).sum())
            witness_result['observed_vs_native_panoptic_pixels'] = [int((packed!=v).sum()) for v in native_segments]
            write(destination / f'observer_witness_{choice["name"]}_attempt{reproduction_attempt:02d}.json', witness_result)
            del references
            del native_segments, actual_fields
        reproduction = reproduce(output, batch, window, golden, destination / 'temporary_png_witness' / f'attempt{reproduction_attempt:02d}')
        reproduction['attempt_index'] = reproduction_attempt
        targets, pairs = final_hungarian(g, batch)
        gt = targets['gt_pixel_masks'][0]
        ids, classes = targets['gt_instance_ids'][0], targets['gt_classes'][0]
        qi, ki = pairs[0]
        stages, labels = raw_stages(render)
        # Match on native context; retain full query x GT matrices at every stage.
        matrices = {name: iou_matrix(mask[:2].permute(1, 0, 2, 3), gt) for name, mask in stages.items()}
        if len(ids):
            best_iou, best_gt = matrices['raw_all'].max(1)
        else:
            best_iou = torch.zeros(100, device=device)
            best_gt = torch.zeros(100, device=device, dtype=torch.long)
        matched_gt = torch.full((100,), -1, device=device, dtype=torch.long)
        matched_gt[qi] = ki
        diagnostic_owner = torch.where(best_iou >= .5, best_gt, matched_gt)
        anchors = g['states'][-1]['anchor_membership'].shape[1]
        mass = g['lifting_mass'][0, :, 0].clamp_min(0)
        child, parent_mass = parent_pool(g['gaussian_membership'][0], mass, anchors)
        child_uniform, _ = parent_pool(g['gaussian_membership'][0], torch.ones_like(mass), anchors)
        # GT appears only after the complete native prediction. Renderer adjoint
        # gives final-child visibility of context GT, not a new model input.
        cam = torch.linalg.inv(g['predicted_context_c2w']).transpose(-1, -2)
        gt_features = gt.permute(1, 0, 2, 3)[None].float()
        gt_lift = RendererTranspose.apply(gt_features, g['gaussians'], cam, g['predicted_context_intrinsics'], model.gs)[0] if len(ids) else torch.zeros((anchors * 64, 0), device=device)
        parent_gt_mass = gt_lift.reshape(anchors, 64, len(ids)).sum(1).clamp_min(0)
        object_mass = parent_gt_mass.sum(1)
        purity = parent_gt_mass.max(1).values / object_mass.clamp_min(1e-12) if len(ids) else torch.zeros(anchors, device=device)
        # Compare inherited anchor mask -> geometry child feature -> fused child
        # mask on identical final geometry and identical official cameras.
        final = g['states'][-1]
        child_features, _ = model.panoptic.gaussian_child_features(final['anchor_embedding'], final['f_anchor'], g['gaussians'], final['mu'], final['radii'])
        geom_membership = (child_features @ final['m_query'].transpose(1, 2)).sigmoid()
        render_cam = torch.linalg.inv(output['camera_calibration']['c2w']).transpose(-1, -2)
        intr = output['camera_calibration']['intrinsics']
        for name, membership in [('inherited_anchor', final['anchor_membership'].repeat_interleave(64, 1)), ('geometry_child', geom_membership)]:
            rr = model.gs.render_feature_channels(g['gaussians'], membership, render_cam, intr)
            regions = alpha_normalize_membership(rr['images_pred'], rr['alphas_pred'])
            mask = (regions[0, :2, :100] >= .5) & (rr['alphas_pred'][0, :2] > .05)
            matrices[name] = iou_matrix(mask.permute(1, 0, 2, 3), gt)
            del rr, regions, mask
        permutations = []
        gen = torch.Generator().manual_seed(7183 + window['source_index'])
        for _ in range(16):
            # Derangement avoids retaining any true query correspondence.
            p = torch.randperm(100, generator=gen)
            while (p == torch.arange(100)).any():
                p = torch.randperm(100, generator=gen)
            permutations.append(p.to(device))
        layer_metrics, arrays = {}, {}
        pc = g['p_class'][0]
        for state in g['states']:
            if state['layer'] not in LAYERS:
                continue
            layer = state['layer']; observed = obs.values[f'L{layer}']
            read_weights = state['evidence_attention'][0].mean(0).T
            route = state['route'][0]
            am = state['anchor_membership'][0]
            metrics = ownership_metrics(read_weights, route, observed['norm'], am, child, parent_mass, permutations)
            head_r = state['evidence_attention'][0, :, :100]
            metrics['read_entropy'] = (-(head_r * head_r.clamp_min(1e-12).log()).sum(-1) / np.log(anchors)).mean(0)
            # Soft GT-conditioned final-parent write contamination. Read may
            # intentionally use context; only Write is called cross-instance here.
            owners = diagnostic_owner.clamp_min(0)
            own = parent_gt_mass[:, owners] if len(ids) else torch.zeros((anchors, 100), device=device)
            other = (object_mass[:, None] - own).clamp_min(0)
            norm = observed['norm'][:, :100]
            metrics['write_other_GT_fraction'] = (norm * other).sum(0) / (norm * object_mass[:, None]).sum(0).clamp_min(1e-12)
            pure = (purity >= .8).float()[:, None]
            metrics['write_other_GT_fraction_pure'] = (norm * other * pure).sum(0) / (norm * object_mass[:, None] * pure).sum(0).clamp_min(1e-12)
            metrics['write_GT_mass'] = (norm * object_mass[:, None]).sum(0)
            metrics['write_signed_other_GT'] = (observed['signed'][:, :100] * other).sum(0) / object_mass.sum().clamp_min(1e-12)
            metrics['read_own_GT_fraction'] = (spatial(read_weights[:, :100]) * own / parent_mass[:, None].clamp_min(1e-12)).sum(0)
            metrics['contribution_norm'] = norm.sum(0)
            metrics['write_null_mass_mean'] = route[:, -1].mean().expand(100)
            metrics['write_child_uniform_js'] = __import__('scripts.vggt_ownership_math_v1', fromlist=['js_divergence']).js_divergence(route[:, :100], child_uniform[:, :100])
            layer_metrics[str(layer)] = {key: value.cpu().tolist() for key, value in metrics.items()}
            layer_metrics[str(layer)]['feedback_reconstruction_max_error'] = observed['error']
            layer_metrics[str(layer)]['actual_feedback_to_token_norm'] = observed['residual_fraction']
            for key, tensor in [('read', read_weights[:, :100]), ('route', route[:, :100]), ('contribution', norm), ('anchor_membership', am[:, :100])]:
                arrays[f'L{layer}_{key}'] = tensor.cpu().numpy().astype(np.float16)
        sem, ins, _ = assemble_panoptic(render)
        xy, valid_xy = projected_xy(g, final['mu'][0])
        arrays.update(child_membership=child[:, :100].cpu().numpy().astype(np.float16),
                      parent_mass=parent_mass.cpu().numpy(), parent_GT_mass=parent_gt_mass.cpu().numpy(),
                      anchor_xy=xy.cpu().numpy(), anchor_xy_valid=valid_xy.cpu().numpy(),
                      raw_context=np.packbits(stages['raw_all'][:2].cpu().numpy(), axis=-1),
                      winner_context=np.packbits(stages['winner_before_area_filter'][:2].cpu().numpy(), axis=-1),
                      final_context=ins[:2].cpu().numpy().astype(np.int16))
        np.savez_compressed(destination / (choice['name'] + '.npz'), **arrays)
        areas = stages['raw_all'][:2].sum((0, 2, 3)).cpu().tolist()
        rgb_error = (render['images_pred'][0].clamp(0, 1) - batch['images_all'][0]).square().mean(1)
        instance_rgb = []
        for iid in ids:
            mask = (batch['instance_label_all'][0] == iid) & (batch['semantic_label_all'][0] >= 2) & (batch['semantic_label_all'][0] <= 19)
            instance_rgb.append(dict(pixels=mask.sum((1, 2)).cpu().tolist(), sse=(rgb_error * mask).sum((1, 2)).cpu().tolist()))
        record = dict(name=choice['name'], source_index=choice['source_index'], frame_ids=choice['frame_ids'], input_rgb_sha256=input_hash,
                      gt_ids=ids.cpu().tolist(), gt_classes=classes.cpu().tolist(), gt_pixels=gt.sum((1, 2, 3)).cpu().tolist(),
                      matched_query=qi.cpu().tolist(), matched_gt=ki.cpu().tolist(), diagnostic_query_owner=diagnostic_owner.cpu().tolist(),
                      query_class_probabilities=pc.cpu().tolist(), query_raw_pixels=areas,
                      mask_IoU={name: matrix.cpu().tolist() for name, matrix in matrices.items()},
                      ownership=layer_metrics, regional_rgb=regional_rgb(render, batch),
                      instance_rgb_unquantized=instance_rgb,
                      reproduction=reproduction, observer_noop_witness=witness_result,
                      parent_child_order='1024 parent-major anchors x64 children; visibility-weighted and uniform control',
                      GT_used_only_after_native_prediction=True)
        assert all(torch.isfinite(t).all() for t in [g['gaussians'], g['gaussian_membership'], pc])
    del output, g, render, batch, child_features, geom_membership, stages
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    torch.cuda.set_device(local); torch.set_num_threads(4)
    device = torch.device('cuda', local)
    assert '3090' in torch.cuda.get_device_name(local)
    proof = read(a.root / 'comparability.json')
    assert proof['status'] == 'C_AND_BASELINE300_COMPARABLE'
    from scripts.vggt_readout_probe_v1 import frozen_C, core_versions, assert_core
    from scripts.eval_vggt_feedback_expanded_v1 import windows
    from scripts.train_vggt_reading_controls_v1 import apply_state
    from scripts.vggt_lossless_checkpoint_v1 import load
    model, opt = frozen_C(device)
    obs = Observer(model)
    all_windows, _ = windows()
    choices = read(HELDOUT)['records'][rank::world]
    golden = catalogs()
    for arm in ('C', 'baseline300'):
        if arm == 'baseline300':
            b = load(READING / 'baseline/training/checkpoint_latest.pt')
            assert b['completed_updates'] == 300 and b['completed_exposures'] == 2400
            assert set(b['parameter_patch']) == set(read(READING / 'baseline/training/run_manifest.json')['rank_initialization'][0]['names'])
            apply_state(model, b); del b; gc.collect()
            model.understanding_step = 38176
        destination = a.root / arm / f'rank{rank:02d}'
        destination.mkdir(parents=True, exist_ok=a.resume)
        if a.resume and (destination/'COMPLETE.json').exists():
            done = read(destination/'COMPLETE.json')
            assert done['status']=='FIXED_WEIGHT_OWNERSHIP_EXTRACTED' and done['names']==[r['name'] for r in choices]
            print('OWNERSHIP_COMPLETED_SHARD_REUSED', rank, arm, len(choices), flush=True)
            continue
        versions = core_versions(model)
        records = []
        started = time.monotonic()
        for i, choice in enumerate(choices):
            result_path=destination/(choice['name']+'.json')
            if a.resume and result_path.exists():
                result=read(result_path)
                assert all(result[k]==choice[k] for k in ('name','source_index','frame_ids','input_rgb_sha256'))
                assert result_path.with_suffix('.npz').exists()
                records.append(result)
                print('OWNERSHIP_WINDOW_REUSED', rank, arm, i+1, choice['name'], flush=True)
                continue
            orphan=result_path.with_suffix('.npz')
            if a.resume and orphan.exists():
                orphan.rename(orphan.with_suffix('.orphan_64911.npz'))
            for retry in range(5):
                try:
                    result = observe_window(model, opt, obs, all_windows[choice['source_index']], choice,
                        golden[arm][choice['name']], destination, device,
                        witness=(i==0 and rank==0) or retry>0 or choice['name']=='scene0608_02_context114_192',
                        reproduction_attempt=retry)
                    break
                except OutputReproductionMismatch as exc:
                    write(destination/f'reproduction_failure_{choice["name"]}_attempt{retry:02d}.json',
                        dict(status='STRICT_PNG_REPRODUCTION_RETRY',error=str(exc),attempt=retry,
                             canonical_output_not_changed=True,criterion='Exact original PNG, not better GT metrics',new_optimizer_updates=0))
                    if retry==4:
                        raise
            result['execution_sha']=os.environ['TASK_CODE_SHA']
            assert_core(model, versions)
            records.append(result)
            write(destination / (choice['name'] + '.json'), result)
            write(destination / 'progress.json', dict(arm=arm, completed=i + 1, total=len(choices), new_optimizer_updates=0,
                elapsed_seconds=time.monotonic() - started))
            if i == 0:
                write(destination / 'FIRST_WINDOW_VERIFIED.json', dict(status='NATIVE_OUTPUT_AND_OBSERVER_VERIFIED',
                    name=choice['name'], reproduction=result['reproduction'], model_parameters_buffers_unchanged=True,
                    actual_feedback_decomposition_verified=True, new_optimizer_updates=0))
            print('OWNERSHIP_OBSERVED', rank, arm, i + 1, choice['name'], flush=True)
        write(destination / 'COMPLETE.json', dict(status='FIXED_WEIGHT_OWNERSHIP_EXTRACTED', names=[r['name'] for r in records],
            rank=rank, world=world, new_optimizer_updates=0, model_parameters_buffers_unchanged=True,
            execution_sha=os.environ['TASK_CODE_SHA']))
    print('OWNERSHIP_RANK_COMPLETE', rank, flush=True)


if __name__ == '__main__':
    main()
