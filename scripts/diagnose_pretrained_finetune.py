#!/usr/bin/env python3
"""Locate the earliest degradation point of the pretrained TokenGS fine-tune.

Restarts from the strictly-loaded old ScanNet checkpoint on one fixed raw 8+7
batch and records, at each of

  after load -> after model.train() -> after the first training forward ->
  after the first backward -> after the first optimizer.step -> after steps
  2/5/10/20/50,

the *same* fixed-batch evaluation (both eval() and train() mode, fp32, fresh
forward on the current parameters): PSNR, total loss and its MSE / SSIM /
visibility terms, render alpha, predicted-RGB mean/std, per-parameter-group
gradient norms and the actual cumulative parameter displacement per group.

An ``--lr 0`` control repeats the schedule without updating to separate "just
switching train mode / running a forward" from "the optimizer step".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
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
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

CKPT = "/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors"
CONTEXT = [0, 20, 40, 60, 80, 100, 120, 140]
NOVEL = [10, 30, 50, 70, 90, 110, 130]
SCENE = "scene0059_00"

GROUPS = (
    ("encoder", ("enc_dec_backbone.encoder.", "patch_embed.", "patch_plucker_embed.")),
    ("decoder", ("enc_dec_backbone.decoder_blocks.", "enc_dec_backbone.kv_proj.")),
    ("gs_tokens", ("gs_tokens",)),
    ("gaussian_head", ("activation_head.",)),
)


def group_of(name: str) -> str:
    for group, prefixes in GROUPS:
        if name.startswith(prefixes):
            return group
    return "other"


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(move(x, device) for x in v)
    return v


def build_optimizer(model, lr, weight_decay, kind: str = "adamw"):
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or getattr(p, "_no_weight_decay", False):
            nodecay.append((name, p))
        else:
            decay.append((name, p))
    groups = [
        {"params": [p for _, p in decay], "weight_decay": weight_decay},
        {"params": [p for _, p in nodecay], "weight_decay": 0.0},
    ]
    if kind == "adamw":
        opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    elif kind == "sgd":
        opt = torch.optim.SGD(groups, lr=lr)
    else:
        raise ValueError(kind)
    names = {id(p): n for n, p in decay} | {id(p): n for n, p in nodecay}
    return opt, names, decay, nodecay


def eval_metrics(model, batch, opt, device, mode: str) -> dict:
    """Fresh fp32 forward on the *current* parameters, in the requested mode."""
    was_training = model.training
    (model.train if mode == "train" else model.eval)()
    with torch.no_grad():
        model_input, _ = split_data(batch, opt)
        dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
        output = model.forward_reconstruction_only(
            ModelInput(model_input.encoder, dec), render_decoder_input=dec)
        _, tr = model.step_loss(batch, step=0, phase="train")
    render = output["render"]
    pred = render["images_pred"][0].float()
    gt = batch["images_all"][0].float()
    n_in = int(opt.num_input_views)

    def psnr(a, b):
        return float(-10.0 * torch.log10((a - b).pow(2).mean().clamp_min(1e-12)))

    def ssim(a, b):
        h, w = a.shape[-2], a.shape[-1]
        return float(1.0 - 2.0 * ssim_loss(a.reshape(-1, 3, h, w), b.reshape(-1, 3, h, w)))

    a = render["alphas_pred"][0].float()
    model.train(was_training)
    return {
        "mode": mode,
        "ctx_psnr": psnr(pred[:n_in], gt[:n_in]),
        "target_psnr": psnr(pred[n_in:], gt[n_in:]),
        "all_psnr": psnr(pred, gt),
        "target_ssim": ssim(pred[n_in:], gt[n_in:]),
        "loss": float(tr["loss"]),
        "loss_rgb_mse": float(tr["loss_rgb"]),
        "loss_ssim_term": float(tr["loss_ssim"]),
        "loss_gaussian_visibility": float(tr.get("loss_gaussian_visibility", torch.tensor(0.0))),
        "alpha_mean": float(a.mean()),
        "alpha_std": float(a.std()),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "pred_target_mean": float(pred[n_in:].mean()),
        "pred_target_std": float(pred[n_in:].std()),
    }


def grad_norms(model) -> dict:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = group_of(name)
        out[g] = out.get(g, 0.0) + float(p.grad.detach().float().pow(2).sum())
    return {k: math.sqrt(v) for k, v in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--no-clip", action="store_true", help="disable grad clipping")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        num_input_views=len(CONTEXT), num_views=len(CONTEXT) + len(NOVEL),
        batch_size=1, num_workers=0, seed=42, dataset_kwargs=None,
    )
    model = model_registry[opt.model_type](opt)
    state = load_file(args.checkpoint, device="cpu")
    ms = model.state_dict()
    matched = set(ms) & set(state)
    missing = sorted(set(ms) - set(state))
    unexpected = sorted(set(state) - set(ms))
    mismatched = sorted(k for k in matched if tuple(state[k].shape) != tuple(ms[k].shape))
    total = sum(v.numel() for v in ms.values())
    mp = sum(ms[k].numel() for k in matched)
    load_report = {
        "matched_tensors": len(matched), "total_tensors": len(ms),
        "matched_params": int(mp), "total_params": int(total),
        "ratio": mp / total, "missing": missing, "unexpected": unexpected,
        "mismatched": mismatched,
    }
    if missing or unexpected or mismatched:
        raise RuntimeError(f"non-exact load refused: {load_report}")
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print(f"[diag] load matched {len(matched)}/{len(ms)} tensors, "
          f"{mp:,}/{total:,} params = {mp / total * 100:.2f}%")

    provider = ScanNetRawReconProvider(
        opt, root=str(DEFAULT_SCANS_ROOT), scene=SCENE,
        context_frame_ids=tuple(CONTEXT), novel_frame_ids=tuple(NOVEL), training=False,
    )
    batch = move(default_collate([provider[0]]), device)
    optimizer, pname, decay, nodecay = build_optimizer(
        model, args.lr, args.weight_decay, args.optimizer)
    print(f"[diag] optimizer: decay(wd={args.weight_decay})={sum(p.numel() for _, p in decay):,} "
          f"nodecay={sum(p.numel() for _, p in nodecay):,}")

    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    records = []

    def snapshot(tag: str, extra: dict | None = None) -> None:
        rec = {
            "tag": tag,
            "training_flag": bool(model.training),
            "eval": eval_metrics(model, batch, opt, device, "eval"),
            "train": eval_metrics(model, batch, opt, device, "train"),
            "grad_norms": grad_norms(model),
        }
        disp: dict[str, float] = {}
        for n, p in model.named_parameters():
            g = group_of(n)
            d = (p.detach().float() - init[n].float()).pow(2).sum().item()
            disp[g] = disp.get(g, 0.0) + d
        rec["param_disp"] = {k: math.sqrt(v) for k, v in disp.items()}
        rec["param_disp_total"] = math.sqrt(sum(v * v for v in rec["param_disp"].values()))
        if extra:
            rec.update(extra)
        records.append(rec)
        ev, tr = rec["eval"], rec["train"]
        print(f"[diag] {tag:<22} eval ctx {ev['ctx_psnr']:6.2f} tgt {ev['target_psnr']:6.2f} "
              f"| train ctx {tr['ctx_psnr']:6.2f} tgt {tr['target_psnr']:6.2f} "
              f"| loss {ev['loss']:.4f} rgb {ev['loss_rgb_mse']:.4f} ssim {ev['loss_ssim_term']:.4f} "
              f"vis {ev['loss_gaussian_visibility']:.4f} | alpha {ev['alpha_mean']:.3f} "
              f"| disp {rec['param_disp_total']:.3e}", flush=True)

    # ---- state transitions before any update ------------------------------ #
    snapshot("after_load")
    snapshot("after_load_repeat")          # noise floor of the fixed-batch eval
    model.train()
    snapshot("after_train_mode")

    amp_dtype = torch.bfloat16
    use_amp = not args.no_amp
    for step in range(1, args.steps + 1):
        lr_now = args.lr * (float(min(step - 1, args.warmup_steps)) / max(1, args.warmup_steps))
        if step - 1 >= args.warmup_steps:
            prog = min(1.0, (step - 1 - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps))
            lr_now = args.lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, metrics = model.step_loss(batch, step=step - 1, phase="train")
        train_loss = float(metrics["loss"])
        metrics["loss"].backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
        if not args.no_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        if step == 1:
            snapshot("after_first_forward", {"train_forward_loss": train_loss, "lr": lr_now})
            snapshot("after_first_backward", {"train_forward_loss": train_loss, "lr": lr_now})
        optimizer.step()
        if step in (1, 2, 5, 10, 20, 50):
            snapshot(f"after_step{step}",
                     {"train_forward_loss": train_loss, "lr": lr_now,
                      "grad_norm_pre_clip": pre_clip})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"load": load_report, "lr": args.lr, "amp": use_amp,
                               "clip": not args.no_clip, "optimizer": args.optimizer,
                               "records": records}, indent=2),
                   encoding="utf-8")
    print(f"[diag] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
