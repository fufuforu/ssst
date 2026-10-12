"""Pinned official SIU3R PNG metrics, one checkpoint stage per GPU worker."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_REPO, SIU3R_COMMIT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('four checkpoint-stage workers required')
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    actual = subprocess.check_output(['git', '-C', SIU3R_REPO, 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SIU3R_COMMIT:
        raise ValueError('official SIU3R source mismatch')
    exports = [json.loads((args.root / f'rank{i:02d}' / 'export_complete.json').read_text()) for i in range(4)]
    if any(e['status'] != 'EXPORT_COMPLETED' or len(e['rows']) != 12 for e in exports):
        raise ValueError('incomplete or mismatched reconstruction export')
    step = (0, 100, 200, 300)[rank]
    rows = []
    for arm in ('pred_joint', 'pred_recon', 'gt_recon'):
        path = args.root / 'png' / arm / f'update{step:04d}'
        result = evaluate(path, device=f'cuda:{local}', segmentation=False, depth_quality=False)
        scores = []
        for window in sorted(d for d in path.iterdir() if d.is_dir()):
            data = json.loads((window / 'render_scores.json').read_text())
            if len(data) != 4:
                raise ValueError('expected four scored views/window')
            for score in data:
                if not all(np.isfinite(score[k]) for k in ('psnr', 'ssim', 'lpips')):
                    raise ValueError('nonfinite official metric')
                scores.append(dict(arm=arm, step=step, window=window.name, **score))
        if len(scores) != 16:
            raise ValueError('expected sixteen images/condition')
        for key in ('psnr', 'ssim', 'lpips'):
            if abs(float(result[key]) - float(np.mean([s[key] for s in scores]))) > 1e-6:
                raise ValueError('official reducer differs from per-image mean')
        rows.extend(scores)
        print(json.dumps({'arm': arm, 'step': step, 'official': {k: result[k] for k in ('psnr','ssim','lpips')}}), flush=True)
    receipt = dict(status='SCORING_COMPLETED', diagnostic_only=True, rank=rank, step=step,
        official_siu3r_commit=SIU3R_COMMIT, torch=torch.__version__,
        torchmetrics=__import__('torchmetrics').__version__, rows=rows,
        metric_protocol='Unmodified official Evaluator; RGB PNG export; all four requested views, no scope subdivision',
        optimizer_updates=0, evaluation_git_sha=os.environ['TASK_CODE_SHA'])
    out = args.root / f'score_rank{rank:02d}.json'
    if out.exists():
        raise ValueError('score receipt already exists; preserve evidence')
    out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
