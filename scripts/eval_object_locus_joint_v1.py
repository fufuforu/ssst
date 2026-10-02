"""Registered eval orchestration; all prediction/evaluator rules remain V3 Evalfix."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import torch
from scripts.object_locus_joint_v1_runtime import (
    capture_rng, restore_rng, write_json, sha256_file, build_batch, SPLITS)
from scripts.eval_object_locus_v3_set import evaluate_windows


def evaluate_registered(model, opt, manifest, reports, step):
    rng, was = capture_rng(), model.training
    model.understanding_step = int(step)
    model.eval()
    result = {'step': step, 'epoch': step // 56, 'splits': {}, 'per_scene': {}}
    gt, queries = [], []
    try:
        with torch.no_grad():
            for split in SPLITS:
                windows = manifest[split]
                x, g, q = evaluate_windows(model, opt, windows, step, split, reports,
                    'cuda', build_batch, official=True, panels=True)
                add_provenance(x, reports, step, split)
                result['splits'][split] = x; gt.extend(g); queries.extend(q)
                if step in (0, 3584):
                    result['per_scene'][split] = {}
                    for scene in sorted({w['scene'] for w in windows}):
                        name = f'{split}__{scene}'
                        subset = [w for w in windows if w['scene'] == scene]
                        sx, _, _ = evaluate_windows(model, opt, subset, step, name,
                            reports, 'cuda', build_batch, official=True, panels=False)
                        add_provenance(sx, reports, step, name)
                        result['per_scene'][split][scene] = sx
            write_json(Path(reports) / f'curves_step_{step:04d}.json', result)
            write_json(Path(reports) / f'per_gt_step_{step:04d}.json', gt)
            write_json(Path(reports) / f'queries_step_{step:04d}.json', queries)
    finally:
        restore_rng(rng); model.train(was)
    return result


def add_provenance(result, reports, step, name):
    official = result.setdefault('official', {})
    official['_provenance'] = {}
    for scope in ('all', 'novel'):
        path = Path(reports) / f'official/step_{step:04d}/{name}/official_{scope}.json'
        official['_provenance'][scope] = {'path': str(path),
            'sha256': sha256_file(path) if path.exists() else 'MISSING'}
    write_json(Path(reports) / f'eval_{name}_step{step:04d}.json', result)


def main():
    import argparse
    from scripts.object_locus_joint_v1_runtime import build_model, build_manifest
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True)
    p.add_argument('--arm', choices=('control', 'joint'), required=True)
    p.add_argument('--reports', type=Path, required=True)
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['arm'] != args.arm: raise RuntimeError('checkpoint arm mismatch')
    model, opt, _ = build_model('cuda', arm=args.arm)
    model.load_state_dict(state['model'], strict=True)
    evaluate_registered(model, opt, build_manifest(), args.reports, state['completed_updates'])


if __name__ == '__main__': main()
