#!/usr/bin/env python3
"""Diagnostic: visible-surface radius and rendered-depth stats for a checkpoint."""
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

from scripts.analyze_gaussian_locality import visible_surface_radius  # noqa: E402
from tokengs.data.scannet_raw_recon import (  # noqa: E402
    DEFAULT_SCANS_ROOT,
    ScanNetRawReconProvider,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="train_siu3r_plain_tokengs_canonical_recon")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        num_input_views=2, num_views=4, batch_size=1, num_workers=0, seed=42)
    provider = ScanNetRawReconProvider(
        opt, root=str(DEFAULT_SCANS_ROOT), scene="scene0048_01",
        context_frame_ids=(654, 664), novel_frame_ids=(655, 659), training=False)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in default_collate([provider[0]]).items()}
    for path in args.models:
        model = model_registry[opt.model_type](opt)
        state = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
        model = model.to(device).eval()
        with torch.no_grad():
            mi, _ = split_data(batch, opt)
            dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                    intrinsics=batch["intrinsics_all"])
            out = model.forward_reconstruction_only(
                ModelInput(mi.encoder, dec), render_decoder_input=dec)
        d = out["render"]["depths_pred"][0].float()
        radius, n = visible_surface_radius(out["render"], batch, 2, (256, 256))
        print(f"[scale] {path}")
        print(f"[scale]   depth min/mean/max {float(d.min()):.4f}/{float(d.mean()):.4f}/"
              f"{float(d.max()):.4f} | visible pts {n} | radius {radius:.5f}")
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
