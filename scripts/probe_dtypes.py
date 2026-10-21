#!/usr/bin/env python3
"""Record the actual dtypes of the backbone, Gaussian head, renderer and losses.

Configuration strings say nothing about where autocast actually applies, so this
instrumented training step records the dtype of every key tensor under
``torch.autocast(bfloat16)`` and under plain fp32.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.scannet_raw_recon import (  # noqa: E402
    DEFAULT_SCANS_ROOT,
    ScanNetRawReconProvider,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def dt(x) -> str:
    if isinstance(x, torch.Tensor):
        return str(x.dtype)
    if isinstance(x, dict):
        return "{" + ",".join(f"{k}:{dt(v)}" for k, v in list(x.items())[:6]) + "}"
    return type(x).__name__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--context", type=int, nargs="+", default=[654, 664])
    parser.add_argument("--novel", type=int, nargs="+", default=[655, 659])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        num_input_views=len(args.context), num_views=len(args.context) + len(args.novel),
        batch_size=1, num_workers=0, seed=42, dataset_kwargs=None,
    )
    model = model_registry[opt.model_type](opt).to(device).train()
    provider = ScanNetRawReconProvider(
        opt, root=str(DEFAULT_SCANS_ROOT), scene=args.scene,
        context_frame_ids=tuple(args.context), novel_frame_ids=tuple(args.novel),
        training=False,
    )
    batch = default_collate([provider[0]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    captured: dict[str, str] = {}

    def head_hook(_module, _args, output):
        captured["activation_head.output (gaussians)"] = dt(output)

    model.activation_head.register_forward_hook(head_hook)
    orig_render = model.render_reconstruction

    def render_wrapper(reconstruction, decoder_input):
        captured["render input gaussians"] = dt(reconstruction.gaussians)
        captured["render input cam_view"] = dt(decoder_input.cam_view)
        captured["render input intrinsics"] = dt(decoder_input.intrinsics)
        out = orig_render(reconstruction, decoder_input)
        captured["render images_pred"] = dt(out["images_pred"])
        captured["render alphas_pred"] = dt(out["alphas_pred"])
        captured["render means2d_pred"] = dt(out["means2d_pred"])
        captured["render depths_pred"] = dt(out["depths_pred"])
        return out

    model.render_reconstruction = render_wrapper
    captured["model parameter dtype"] = dt(next(model.parameters()))
    captured["batch images_all"] = dt(batch["images_all"])
    captured["batch input (concat rgb+plucker)"] = dt(batch["input"])

    results = {}
    for tag, enabled in (("fp32 (autocast disabled)", False), ("bf16 autocast", True)):
        captured.clear()
        captured["model parameter dtype"] = dt(next(model.parameters()))
        captured["batch images_all"] = dt(batch["images_all"])
        captured["batch input (concat rgb+plucker)"] = dt(batch["input"])
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            # what autocast actually does to an internal matmul in this mode
            _a = torch.ones(8, 8, device=device)
            _b = torch.ones(8, 8, device=device)
            captured["internal matmul result"] = dt(torch.nn.functional.linear(_a, _b))
            out, metrics = model.step_loss(batch, step=0, phase="train")
        captured["loss total"] = dt(metrics["loss"])
        captured["loss_rgb (MSE)"] = dt(metrics["loss_rgb"])
        captured["loss_ssim"] = dt(metrics["loss_ssim"])
        captured["loss_gaussian_visibility"] = dt(metrics.get("loss_gaussian_visibility"))
        # gradient dtype seen by the Gaussian head
        metrics["loss"].backward()
        gp = model.activation_head.deconv.weight.grad
        captured["activation_head weight grad"] = dt(gp)
        results[tag] = dict(captured)
        print(f"--- {tag} ---")
        for k in sorted(results[tag]):
            print(f"  {k:<38} {results[tag][k]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[dtype] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
