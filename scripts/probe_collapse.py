#!/usr/bin/env python3
"""Diagnose a collapsed reconstruction: opacity / scale / alpha of a checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--scene", default="scene0059_00")
    parser.add_argument("--root", default="/space/mawb/SIU3R/data/scannet/train")
    args = parser.parse_args()
    device = torch.device("cuda")
    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        num_input_views=2, num_views=4, batch_size=1, num_workers=0, seed=42)
    provider = SIU3RProcessedProvider(
        opt, root=args.root, subset=[args.scene], training=True, rank=0)
    provider.pair_rng.seed(1042)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in default_collate([provider[0]]).items()}
    for ck in args.checkpoints:
        model = model_registry[opt.model_type](opt)
        state = torch.load(Path(ck) / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=True)
        model = model.to(device).eval()
        with torch.no_grad():
            mi, _ = split_data(batch, opt)
            dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                    intrinsics=batch["intrinsics_all"])
            out = model.forward_reconstruction_only(
                ModelInput(mi.encoder, dec), render_decoder_input=dec)
        g = out["gaussians"][0].float()
        op, sc = g[:, 3], g[:, 4:7]
        alpha = out["render"]["alphas_pred"].float()
        dr = getattr(model.activation_head, "last_decode_radius", None)
        print(f"[probe] {ck}")
        print(f"[probe]   opacity mean/p50/max {float(op.mean()):.4f}/"
              f"{float(op.median()):.4f}/{float(op.max()):.4f}")
        print(f"[probe]   scale mean/p50/max {float(sc.mean()):.3e}/"
              f"{float(sc.median()):.3e}/{float(sc.max()):.3e}")
        print(f"[probe]   rendered alpha mean/max {float(alpha.mean()):.5f}/"
              f"{float(alpha.max()):.5f}")
        print(f"[probe]   center z mean {float(g[:, 2].mean()):.3f} "
              f"| c2w z range {float(batch['cam_to_world_input'][0,:,2,3].min()):.3f}.."
              f"{float(batch['cam_to_world_input'][0,:,2,3].max()):.3f}")
        if dr is not None:
            print(f"[probe]   decode radius {float(dr.min()):.6f}-{float(dr.max()):.6f}")
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
