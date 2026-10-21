#!/usr/bin/env python3
"""Compare the bf16-autocast gradient with the fp32 gradient at the same state.

If the autocast gradient is dominated by quantisation noise its direction will
barely correlate with the fp32 gradient, which explains why an optimizer step
built from it walks the converged model off its solution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
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

CKPT = "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors"
CONTEXT = [0, 20, 40, 60, 80, 100, 120, 140]
NOVEL = [10, 30, 50, 70, 90, 110, 130]


def flat_grads(model) -> torch.Tensor:
    chunks = [p.grad.detach().float().reshape(-1) for _, p in model.named_parameters()
              if p.grad is not None]
    return torch.cat(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--scene", default="scene0059_00")
    parser.add_argument("--context", type=int, nargs="+", default=CONTEXT)
    parser.add_argument("--novel", type=int, nargs="+", default=NOVEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        num_input_views=len(args.context), num_views=len(args.context) + len(args.novel),
        batch_size=1, num_workers=0, seed=42, dataset_kwargs=None,
    )
    model = model_registry[opt.model_type](opt)
    state = load_file(args.checkpoint, device="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).train()
    provider = ScanNetRawReconProvider(
        opt, root=str(DEFAULT_SCANS_ROOT), scene=args.scene,
        context_frame_ids=tuple(args.context), novel_frame_ids=tuple(args.novel),
        training=False,
    )
    batch = default_collate([provider[0]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    grads, losses = {}, {}
    for tag, enabled in (("fp32", False), ("bf16", True)):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            _, metrics = model.step_loss(batch, step=0, phase="train")
        losses[tag] = float(metrics["loss"])
        metrics["loss"].backward()
        grads[tag] = flat_grads(model)

    g32, g16 = grads["fp32"], grads["bf16"]
    cos = float(torch.nn.functional.cosine_similarity(g32, g16, dim=0))
    out = {
        "loss_fp32": losses["fp32"],
        "loss_bf16": losses["bf16"],
        "grad_norm_fp32": float(g32.norm()),
        "grad_norm_bf16": float(g16.norm()),
        "grad_cosine_similarity": cos,
        "grad_relative_l2_diff": float((g16 - g32).norm() / g32.norm().clamp_min(1e-12)),
        "num_params": int(g32.numel()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(f"[amp] {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
