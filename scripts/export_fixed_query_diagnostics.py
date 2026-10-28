"""Load the validated checkpoint with the existing strict loader, infer once.

Use the ordinary export_unified_query_checkpoint CLI. --output is a fresh run
root, with E0/E1/E2_* subdirectories. No training/optimizer is instantiated.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
import torch.distributed as dist
from scripts.fixed_query_oracles import ARMS, oracle_variants, context_ranking_scores
from scripts.paired_e0_checks import capture_native, verify_unmodified, audit_pair, verify_gt_files

REFERENCE = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1/eval/evaluation')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def export(model, opt, windows, root, rank, world, device, *, expected_windows):
    from scripts.train_dual_stream_v1 import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render, _official_export_view
    from scripts.export_object_locus_v3_set_official import write_official_pair
    root = Path(root)
    if expected_windows != 1860 or len(windows) != 1860 or len({w['scene'] for w in windows}) != 312:
        raise ValueError('This experiment requires the fixed official1860 / 312-scene cohort')
    reference = json.loads((REFERENCE / 'export_complete.json').read_text())
    refs = {r['name']: r for r in reference['records']}
    if len(refs) != 1860:
        raise ValueError('Baseline reference incomplete')
    model.eval()
    versions = {k: v._version for k, v in model.named_parameters()}
    records = []
    # RendererTranspose locally enables autograd for a temporary dummy tensor.
    # Match the validated official exporter: no_grad permits that local adjoint,
    # while inference_mode disables it even inside torch.enable_grad().
    with torch.no_grad():
        for index in range(rank, len(windows), world):
            window = windows[index]
            batch = build_batch(opt, window, device)
            result = generate_and_render(model, batch['images_input'], batch['images_all'])
            out = _official_export_view(result['render'])
            if any(not torch.isfinite(out[k]).all() for k in ('region_mass', 'p_class', 'alpha', 'images_pred', 'depths_pred')):
                raise FloatingPointError('Nonfinite fixed-weight prediction')
            context_sem = batch['semantic_label_all'][0, :2]
            context_ins = batch['instance_label_all'][0, :2]
            native = capture_native(out)
            variants, audit = oracle_variants(out, context_sem, context_ins, nms_threshold=model.config.mask_nms_iou)
            verify_unmodified(out, native, *variants['E0'])
            name = window['scene'] + '_context' + '_'.join(map(str, window['context']))
            rgbsha = hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest()
            if rgbsha != refs[name]['input_rgb_sha256']:
                raise ValueError('Input differs from verified Baseline: ' + name)
            history = None
            for arm, (prediction, keep) in variants.items():
                armroot = root / arm / 'evaluation'
                row = write_official_pair(prediction, batch, window, armroot / 'all', target_frames='all', eligible_override=keep)
                pair = armroot / 'all' / name
                # Reconstruction was not changed by any intervention. Reuse its
                # already-verified RGB/depth artifacts without recompressing.
                for sub in ('rgb', 'rgb_gt', 'depth', 'depth_gt'):
                    (pair / sub).symlink_to((REFERENCE / 'all' / name / sub).resolve(), target_is_directory=True)
                verify_gt_files(pair, REFERENCE / 'all' / name)
                if arm in ('E2_rank', 'E2_joint'):
                    ranking = context_ranking_scores(prediction, keep, context_sem, context_ins)
                    for split in ('context', 'target'):
                        path = pair / (split + '_seg_pred') / 'pred.json'
                        rows = json.loads(path.read_text())
                        for r in rows:
                            r['score'] = ranking[r['id'] - 1]
                        write_json(path, rows)
                if arm == 'E0':
                    history = audit_pair(pair, REFERENCE / 'all' / name, native['packed'],
                                         batch['frame_ids'][0].cpu().tolist(), window['scene'])
                write_json(pair / 'diagnostic_export.json', dict(arm=arm, oracle=arm != 'E0', official_pair=row,
                           parameter_updates=0, input_rgb_sha256=rgbsha))
            audit.update(name=name, source_index=window['source_index'], baseline_differing_pngs=history['historical_differing_pngs'],
                         native_control_verified=True, historical_baseline_audit=history)
            write_json(root / 'query_audit' / (name + '.json'), audit)
            records.append(dict(name=name, source_index=window['source_index'], input_rgb_sha256=rgbsha,
                                baseline_differing_pngs=history['historical_differing_pngs'],
                                native_control_verified=True,
                                historical_differing_pixels=history['historical_differing_pixels']))
            write_json(root / f'progress_rank{rank:02d}.json', dict(completed=len(records), total=len(windows[rank::world]), parameter_updates=0))
            del variants, prediction, result, out, batch, context_sem, context_ins, native
    if any(v._version != versions[k] for k, v in model.named_parameters()):
        raise ValueError('Diagnostic export modified model parameters')
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    if rank == 0:
        rows = sorted([r for group in gathered for r in group], key=lambda x: x['source_index'])
        if len(rows) != 1860 or len({r['name'] for r in rows}) != 1860:
            raise ValueError('Incomplete export coverage')
        report = dict(status='EXPORTED', windows=1860, unique_scenes=312, records=rows,
                      baseline_control='same_forward_native_E0', historical_comparison='observational_audit',
                      parameter_updates=0, arms=ARMS, oracle_gt_scope='context_only',
                      target_scope='all6_including_context', reconstruction='unchanged; linked verified baseline artifacts',
                      depth_units=reference['depth_units'], prediction_protocol=reference['prediction_protocol'],
                      baseline_differing_pngs=sum(r['baseline_differing_pngs'] for r in rows))
        write_json(root / 'export_complete.json', report)
        write_json(root / 'PAIRED_BASELINE_VERIFIED.json', dict(status='NATIVE_SAME_FORWARD_E0_VERIFIED', windows=1860,
                   parameter_updates=0, oracle_input_tensors_unmodified=True,
                   historical_bitwise_reproduction_claimed=False))
        for arm in ARMS:
            write_json(root / arm / 'evaluation' / 'export_complete.json', dict(report, arm=arm, diagnostic_oracle=arm != 'E0'))
    dist.barrier()


if __name__ == '__main__':
    from scripts import export_unified_query_checkpoint as loader
    loader.export = export
    loader.main()
