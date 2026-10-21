"""Read completed paired diagnostics; never changes training or inference artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('analysis output must be fresh')
    summary = json.loads((args.attempt / 'summary.json').read_text())
    identity = json.loads((args.attempt / 'identity.json').read_text())
    rows = [json.loads(x) for x in (args.attempt / 'rows.jsonl').read_text().splitlines()]
    variants = {r['variant'] for r in rows}
    if len(rows) != len(identity['windows']) * len(variants):
        raise ValueError('incomplete intervention matrix')
    natives = [r for r in rows if r['variant'] == 'current']
    if len(natives) != len(identity['windows']):
        raise ValueError('incomplete native identity gates')
    payload = dict(diagnostic_only=True, source_identity=identity,
        native_renderer_max_error=max(r['native_renderer_error'] for r in natives),
        cached_replay_max_error=max(r['cached_replay_error'] for r in natives), results={})
    for key, aggregate in summary['results'].items():
        split, variant = key.split('/')
        base = {r['scene']: np.mean(r['psnr']) for r in natives if r['split'] == split}
        selected = [r for r in rows if r['split'] == split and r['variant'] == variant]
        delta = np.array([np.mean(r['psnr']) - base[r['scene']] for r in selected])
        rng = np.random.default_rng(42)
        boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        payload['results'][key] = dict(aggregate,
            paired_delta_bootstrap_95_ci=np.quantile(boot, [.025, .975]).tolist(),
            alpha_mean=float(np.mean([r['alpha_mean'] for r in selected])),
            alpha_below_0_1=float(np.mean([r['alpha_below_0_1'] for r in selected])))
    camera = {}
    for split in ('train', 'heldout'):
        focal, principal, rotation, direction, scale_consistency = [], [], [], [], []
        for r in natives:
            if r['split'] != split:
                continue
            c = r['cameras']
            pred, gt = np.array(c['predicted_c2w'])[0], np.array(c['gt_provider_c2w'])[0]
            pk, gk = np.array(c['predicted_intrinsics'])[0], np.array(c['gt_intrinsics'])[0]
            focal.extend((np.abs(pk[:, :2] / gk[:, :2] - 1) * 100).flatten().tolist())
            principal.extend(np.linalg.norm(pk[:, 2:] - gk[:, 2:], axis=-1).tolist())
            for a, b in zip(pred[1:], gt[1:]):
                cosine = (np.trace(a[:3, :3] @ b[:3, :3].T) - 1) / 2
                rotation.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
                u, v = a[:3, 3], b[:3, 3]
                if min(np.linalg.norm(u), np.linalg.norm(v)) > 1e-8:
                    cosine = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
                    direction.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
            scale_consistency.append(r['gauges']['baseline_factor'][0] / r['gauges']['depth_factor'][0])
        def stats(values):
            return dict(count=len(values), median=float(np.median(values)), p90=float(np.quantile(values, .9)))
        camera[split] = dict(focal_absolute_relative_error_percent=stats(focal),
            principal_point_error_pixels=stats(principal), nonreference_rotation_error_degrees=stats(rotation),
            nonreference_translation_direction_error_degrees=stats(direction),
            context_baseline_over_gt_depth_gauge=stats(scale_consistency))
    payload['external_gt_camera_diagnostics'] = camera
    payload['interpretation_limits'] = [
        '16 independent scenes per split; exploratory multiple interventions, not formal benchmark scores.',
        'Negative GT oracle scores do not exclude geometry errors: learned GS may compensate for predicted cameras.',
        'Feedback removal at inference does not isolate the effect of understanding gradients during training.',
        'Legacy-unit and ray-scale interventions alter conditioning while leaving learned anchor priors fixed.',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    for key, r in payload['results'].items():
        print(key, f"PSNR={r['psnr']:.4f}", f"delta={r['paired_delta_mean']:+.4f}",
              'CI=', [round(x, 4) for x in r['paired_delta_bootstrap_95_ci']])
    print(json.dumps(camera, indent=2))


if __name__ == '__main__':
    main()
