#!/usr/bin/env python3
"""Read-only load verification for a full-data pure-reconstruction checkpoint.

Checks the checkpoint is self-contained: strict=True load into a fresh model
built from the recorded preset, SHA256 of model.pt, the frozen decode radius
(0.15) on a real 2+2 window, and the presence of a config/preset snapshot next
to the weights.  Nothing is written to the checkpoint and no optimizer is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    parser.add_argument("--expect-sha256", default=None)
    parser.add_argument("--forward-check", action="store_true",
                        help="run one real 2+2 window to read the decode radius")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    model_path = checkpoint / "model.pt"
    report = {
        "checkpoint": str(checkpoint),
        "model_sha256": sha256_file(model_path),
        "model_size": model_path.stat().st_size,
        "model_mtime": model_path.stat().st_mtime,
        "complete_marker": (checkpoint / "COMPLETE").read_text(encoding="utf-8").strip()
        if (checkpoint / "COMPLETE").is_file() else None,
        "config_snapshot": sorted(p.name for p in checkpoint.glob("config*")),
        "training_record": (checkpoint / "training_record.json").is_file(),
        "preset": args.preset,
    }
    if args.expect_sha256:
        report["sha256_matches_expected"] = report["model_sha256"] == args.expect_sha256

    torch.manual_seed(42)
    opt = config_defaults[args.preset].evolve(
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)          # raises on any mismatch
    report["strict_load"] = True
    report["model_type"] = opt.model_type
    report["parameters"] = int(sum(p.numel() for p in model.parameters()))
    report["frozen_decode_radius_config"] = bool(opt.locusgs_freeze_decode_radius)
    report["bound_delta_config"] = bool(opt.locusgs_bound_delta)
    report["frequency_grid"] = {
        "wavelength_50": None,
        "note": "frequency_grid untouched; see docs for the nyquist audit",
    }
    if "step" in payload:
        report["step_in_checkpoint"] = int(payload["step"])

    if args.forward_check:
        device = torch.device(args.device)
        from torch.utils.data import default_collate

        from tokengs.data.siu3r_processed import SIU3RProcessedProvider
        from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
        from scripts.group_eval_v2 import build_val_entries

        split = json.loads(
            Path("workspace_recon_diag/cross_scene/split.json").read_text(encoding="utf-8")
        )
        model = model.to(device).eval()
        entries = build_val_entries(opt, split, device, scenes=["scene0059_00"])
        batch = entries[0]["batch"]
        with torch.no_grad():
            model_input, _ = split_data(batch, opt)
            decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                        intrinsics=batch["intrinsics_all"])
            output = model.forward_reconstruction_only(
                ModelInput(model_input.encoder, decoder), render_decoder_input=decoder
            )
        radius = model.activation_head.last_decode_radius
        report["decode_radius_min"] = float(radius.min())
        report["decode_radius_max"] = float(radius.max())
        report["alpha_mean"] = float(output["render"]["alphas_pred"].mean())
        report["forward_window"] = {
            "scene": entries[0]["scene"],
            "context": entries[0]["context"],
            "novel": entries[0]["novel"],
        }
        del default_collate, SIU3RProcessedProvider

    print(json.dumps(report, indent=1))
    ok = report.get("strict_load", False) and report.get("config_snapshot")
    if args.expect_sha256:
        ok = ok and report["sha256_matches_expected"]
    if args.forward_check:
        ok = ok and abs(report["decode_radius_min"] - 0.15) < 1e-6
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
