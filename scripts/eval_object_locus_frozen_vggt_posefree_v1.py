"""Evaluation camera protocol; scene generation receives contexts only."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))


def generate_and_render(model, context_rgb, context_plus_target_rgb):
    """Generate context-only Gaussians, independently calibrate, then request a view."""
    generated=model.generate(context_rgb)
    calibrated=model.calibrate_targets(context_plus_target_rgb,generated)
    cam_view=torch.linalg.inv(calibrated['c2w']).transpose(-1,-2)
    render=model.render_generated_at(generated,cam_view,calibrated['intrinsics'])
    return {'generated':generated,'camera_calibration':calibrated,'render':render}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',action='store_true',help='print the evaluation camera disclosure')
    args=parser.parse_args(argv)
    if args.protocol:
        print('Scene generation uses only two context views; supervision/target cameras use independent image calibration. A target camera is still required to render a specified novel view.')
    return 0


if __name__=='__main__': raise SystemExit(main())
