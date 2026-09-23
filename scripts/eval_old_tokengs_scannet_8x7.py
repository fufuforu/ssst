#!/usr/bin/env python3
"""Positive control: re-evaluate the old successful ScanNet TokenGS checkpoint.

Loads ``scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors``
into this repo's ``PlainTokenGSCanonicalRecon`` (the same architecture) with the
original 8-context + 7-novel ScanNet protocol and **raw .sens** frames, and
reports the load match plus PSNR/SSIM on the fixed validation windows.

Read-only with respect to the checkpoint; writes only into ``--out-dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

MANIFEST = Path("/space/mawb/tokengs/data/scannet_prompt/scannet_prompt_full_wide_8x7.json")
CKPT = Path(
    "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/"
    "tokengs_backbone_step_008000.safetensors"
)


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(move(x, device) for x in v)
    return v


def load_state_report(model, state: dict) -> dict:
    ms = model.state_dict()
    ms_keys, st_keys = set(ms), set(state)
    matched = ms_keys & st_keys
    missing = sorted(ms_keys - st_keys)
    unexpected = sorted(st_keys - ms_keys)
    mismatched = sorted(k for k in matched if tuple(state[k].shape) != tuple(ms[k].shape))
    total = sum(v.numel() for v in ms.values())
    matched_params = sum(ms[k].numel() for k in matched)
    report = {
        "model_params": int(total),
        "matched_tensors": len(matched),
        "matched_params": int(matched_params),
        "matched_param_ratio": matched_params / total,
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }
    if missing or unexpected or mismatched:
        raise RuntimeError(f"non-exact load: {report}")
    model.load_state_dict(state, strict=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CKPT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--scans-root", default=str(DEFAULT_SCANS_ROOT))
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir", default="/space/mawb/ssst/workspace_recon_diag/old_tokengs_positive_control"
    )
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        num_input_views=8, num_views=15, img_size=(256, 256),
        batch_size=1, num_workers=0, seed=42, dataset_kwargs=None,
    )
    model = model_registry[opt.model_type](opt)
    state = load_file(args.checkpoint, device="cpu")
    report = load_state_report(model, state)
    print(f"[pc] checkpoint {args.checkpoint}")
    print(f"[pc] load: matched {report['matched_tensors']} tensors / "
          f"{report['matched_params']:,} params ({report['matched_param_ratio']*100:.2f}%) "
          f"missing={len(report['missing'])} unexpected={len(report['unexpected'])} "
          f"mismatched={len(report['mismatched'])}")
    model = model.to(device).eval()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = manifest["validation_samples"]
    seen, chosen = set(), []
    for s in samples:
        if s["scene"] in seen:
            continue
        seen.add(s["scene"])
        chosen.append(s)
        if len(chosen) >= args.num_windows:
            break

    results = []
    for s in chosen:
        scene = s["scene"]
        ctx, nov = [int(x) for x in s["input_frame_ids"]], [int(x) for x in s["target_frame_ids"]]
        provider = ScanNetRawReconProvider(
            opt, root=args.scans_root, scene=scene,
            context_frame_ids=tuple(ctx), novel_frame_ids=tuple(nov), training=False,
        )
        batch = move(default_collate([provider[0]]), device)
        with torch.no_grad():
            model_input, _ = split_data(batch, opt)
            dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
            output = model.forward_reconstruction_only(
                ModelInput(model_input.encoder, dec), render_decoder_input=dec)
        pred = output["render"]["images_pred"][0].float()
        gt = batch["images_all"][0].float()
        n_in = int(opt.num_input_views)
        grey = torch.full_like(gt, 0.5)

        def psnr(a, b):
            return float(-10.0 * torch.log10((a - b).pow(2).mean().clamp_min(1e-12)))

        def ssim(a, b):
            h, w = a.shape[-2], a.shape[-1]
            return float(1.0 - 2.0 * ssim_loss(a.reshape(-1, 3, h, w), b.reshape(-1, 3, h, w)))

        row = {
            "scene": scene,
            "input_frame_ids": ctx,
            "target_frame_ids": nov,
            "context_psnr": psnr(pred[:n_in], gt[:n_in]),
            "target_psnr": psnr(pred[n_in:], gt[n_in:]),
            "all_psnr": psnr(pred, gt),
            "target_psnr_per_view": [psnr(pred[i:i + 1], gt[i:i + 1]) for i in range(n_in, len(pred))],
            "target_grey_psnr": psnr(grey[n_in:], gt[n_in:]),
            "context_ssim": ssim(pred[:n_in], gt[:n_in]),
            "target_ssim": ssim(pred[n_in:], gt[n_in:]),
            "alpha_mean": float(output["render"]["alphas_pred"][0].float().mean()),
        }
        results.append(row)
        print(f"[pc] {scene}: ctx PSNR {row['context_psnr']:.2f} | target PSNR {row['target_psnr']:.2f} "
              f"(grey {row['target_grey_psnr']:.2f}) | target SSIM {row['target_ssim']:.3f} | "
              f"alpha {row['alpha_mean']:.3f}")

        # strip: for each target view, GT | pred
        cols = []
        for i in range(n_in, len(pred)):
            cols.append(torch.cat([gt[i], pred[i]], dim=-1))
        strip = torch.cat(cols[:4], dim=-2) if len(cols) >= 4 else torch.cat(cols, dim=-2)
        Image.fromarray((strip.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        ).save(out / "images" / f"{scene}_target_gt_vs_pred.png")

    summary = {
        "checkpoint": args.checkpoint,
        "num_input_views": int(opt.num_input_views),
        "num_views": int(opt.num_views),
        "load_report": {k: v for k, v in report.items() if k not in ("missing", "unexpected", "mismatched")},
        "mean_context_psnr": float(np.mean([r["context_psnr"] for r in results])),
        "mean_target_psnr": float(np.mean([r["target_psnr"] for r in results])),
        "mean_target_ssim": float(np.mean([r["target_ssim"] for r in results])),
        "windows": results,
    }
    (out / "positive_control.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pc] mean context PSNR {summary['mean_context_psnr']:.2f} | "
          f"mean target PSNR {summary['mean_target_psnr']:.2f} | "
          f"mean target SSIM {summary['mean_target_ssim']:.3f}")
    print(f"[pc] wrote {out/'positive_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
