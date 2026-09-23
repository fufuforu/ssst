#!/usr/bin/env python3
"""Fixed-sample paired A/B: SIU3R-processed vs raw ScanNet, plain TokenGS recon.

One fixed record (``scene0048_01``, context ``[654, 664]``, novel ``[655, 659]``)
is overfit under two data sources that differ **only** in where RGB / K / pose
come from:

* ``--source processed`` -- the released SIU3R processed ScanNet tree.
* ``--source raw``       -- raw ``.sens`` frames pushed through the original
  TokenGS ``ScanNetSensReader`` + the shared ``ImageTransform``.

Everything else is held fixed: the ``PlainTokenGSCanonicalRecon`` architecture,
the initial weights (loaded from ``--init-state``), a freshly built AdamW with the
same parameter grouping, bf16 autocast, seed, loss, the 2-context + 2-novel
protocol, the fixed 4000-step learning-rate plan, and the evaluation code.
Checkpoints save both model *and* optimizer state at the mid and final steps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon_models import _full_supervision  # noqa: E402
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

TRAIN_ROOT = "/space/mawb/SIU3R/data/scannet/train"


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def build_provider(opt, source, scene, context, novel, scans_root, pair_seed=42):
    if source == "processed":
        provider = SIU3RProcessedProvider(opt, root=TRAIN_ROOT, subset="all", training=True, rank=0)
        names = [s.name for s in provider.dataset.sample_list]
        # Pin the SIU3R pair independently of the model seed so a replicate
        # varies only the initial weights.
        provider.pair_rng.seed(int(pair_seed))
        sample = provider[names.index(scene)]
        pair = provider.last_pair
    else:
        provider = ScanNetRawReconProvider(
            opt,
            root=scans_root,
            scene=scene,
            context_frame_ids=tuple(context),
            novel_frame_ids=tuple(novel),
            training=True,
        )
        sample = provider[0]
        pair = {
            "scene_id": scene,
            "context_frame_ids": list(context),
            "novel_frame_ids": list(novel),
        }
    batch = default_collate([sample])
    return provider, batch, pair


def swap_rgb(provider, batch, other_batch):
    """Replace the RGB of ``batch`` with ``other_batch``'s RGB, keeping everything else.

    Camera/K/rays of the base arm are untouched; only the supervised/encoded
    images change.  ``input`` holds normalised RGB in channels 0:3 followed by the
    Pluecker embedding, so those three channels are re-normalised in place.
    """
    other_rgb = other_batch["images_all"]
    batch["images_all"] = other_rgb.clone()
    num_in = int(batch["images_input"].shape[1])
    batch["images_input"] = other_rgb[:, :num_in].clone()
    batch["images_output"] = other_rgb[:, num_in:].clone()
    normalizer = provider.input_normalizer
    batch["input"][:, :, 0:3] = normalizer(other_rgb)
    return batch


def make_eval_fn(model, batch, opt, num_ctx):
    def evaluate():
        with torch.no_grad():
            _, metrics = model.step_loss(batch, step=0, phase="train")
            sup = _full_supervision(batch)
            model_input, _ = split_data(batch, opt)
            decoder_input = ModelInputDecoder(
                cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
            )
            output = model.forward_reconstruction_only(
                ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
            )
        render = output["render"]
        pred = render["images_pred"][0].float()
        gt = sup.images_output[0].float()

        def psnr(sl):
            return float(-10.0 * torch.log10((pred[sl] - gt[sl]).pow(2).mean().clamp_min(1e-12)))

        def ssim_v(sl):
            h, w = pred.shape[-2], pred.shape[-1]
            return float(1.0 - 2.0 * ssim_loss(
                pred[sl].reshape(-1, 3, h, w), gt[sl].reshape(-1, 3, h, w)))

        grey = torch.full_like(gt, 0.5)
        alphas = render["alphas_pred"][0].float()
        return {
            "loss": float(metrics["loss"]),
            "loss_rgb": float(metrics["loss_rgb"]),
            "loss_ssim_term": float(metrics["loss_ssim"]),
            "loss_gvis": float(metrics.get("loss_gaussian_visibility", torch.tensor(0.0))),
            "ctx_psnr": psnr(slice(0, num_ctx)),
            "novel_psnr": psnr(slice(num_ctx, None)),
            "all_psnr": psnr(slice(None)),
            "ctx_ssim": ssim_v(slice(0, num_ctx)),
            "novel_ssim": ssim_v(slice(num_ctx, None)),
            "all_ssim": ssim_v(slice(None)),
            "ctx_grey_psnr": float(-10.0 * torch.log10(
                (grey[:num_ctx] - gt[:num_ctx]).pow(2).mean().clamp_min(1e-12))),
            "novel_grey_psnr": float(-10.0 * torch.log10(
                (grey[num_ctx:] - gt[num_ctx:]).pow(2).mean().clamp_min(1e-12))),
            "alpha_mean": float(alphas.mean()),
            "alpha_gt_05": float((alphas > 0.5).float().mean()),
            "depth_nonzero": float((render["depths_pred"] > 0).float().mean()),
        }, pred, gt

    return evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["processed", "raw"])
    parser.add_argument(
        "--swap-rgb",
        action="store_true",
        help=(
            "localisation arm: keep this arm's K/C2W/rays but replace its RGB with "
            "the other source's RGB (isolates RGB from camera handling)"
        ),
    )
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--context", type=int, nargs=2, default=[654, 664])
    parser.add_argument("--novel", type=int, nargs=2, default=[655, 659])
    parser.add_argument("--scans-root", default=str(DEFAULT_SCANS_ROOT))
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--mid-step", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min-ratio", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-seed", type=int, default=42,
                        help="seed for the SIU3R pair sampler; keep fixed across replicates")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--image-every", type=int, default=0,
                        help="save a GT/pred strip every N steps (no checkpoint)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-optimizer", action="store_true",
                        help="also store optimizer state at each saved step")
    parser.add_argument("--save-steps", default="mid,final",
                        help="comma list of 'mid'/'final' checkpoints to write; 'none' to skip")
    parser.add_argument("--init-state", default=None,
                        help="model state_dict shared by both arms; created here if absent")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=args.seed, lr=args.lr,
    )
    torch.manual_seed(int(opt.seed))
    model = model_registry[opt.model_type](opt)

    # Shared initial weights: if provided, load; else generate once and persist.
    if args.init_state and Path(args.init_state).is_file():
        state = torch.load(args.init_state, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        init_source = args.init_state
    else:
        init_state = {"model": model.state_dict()}
        if args.init_state:
            Path(args.init_state).parent.mkdir(parents=True, exist_ok=True)
            torch.save(init_state, args.init_state)
        init_source = args.init_state or "<fresh seed>"
    model = model.to(device)
    model.freeze_object_queries()
    model.train()

    decay = [p for p in model.parameters() if p.requires_grad
             and p.dim() != 1 and not getattr(p, "_no_weight_decay", False)]
    nodecay = [p for p in model.parameters() if p.requires_grad
               and (p.dim() == 1 or getattr(p, "_no_weight_decay", False))]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    print(f"[ab] source={args.source} groups: decay={sum(p.numel() for p in decay):,} "
          f"nodecay={sum(p.numel() for p in nodecay):,} | init={init_source}")

    provider, batch, pair = build_provider(
        opt, args.source, args.scene, args.context, args.novel, args.scans_root,
        pair_seed=args.pair_seed)
    if args.swap_rgb:
        other_source = "raw" if args.source == "processed" else "processed"
        _, other_batch, _ = build_provider(
            opt, other_source, args.scene, args.context, args.novel, args.scans_root,
            pair_seed=args.pair_seed)
        batch = swap_rgb(provider, batch, other_batch)
        print(f"[ab] source={args.source} swap_rgb=on (RGB from {other_source}, "
              f"camera/K/rays from {args.source})")
    batch = move(batch, device)
    frames = [int(x) for x in batch["frame_ids"][0]]
    expected = [*args.context, *args.novel]
    if frames != expected:
        raise RuntimeError(f"{args.source}: frame order {frames} != requested {expected}")
    num_ctx = int(opt.num_input_views)
    print(f"[ab] source={args.source} scene={pair['scene_id']} ctx={pair['context_frame_ids']} "
          f"novel={pair['novel_frame_ids']} frames={frames} scene_scale={provider.scene_scale}")
    print(f"[ab] plan: total={args.total_steps} mid={args.mid_step} warmup={args.warmup_steps} "
          f"lr={args.lr} min_ratio={args.lr_min_ratio} wd={args.weight_decay} seed={args.seed}")

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * float(step + 1) / float(args.warmup_steps)
        progress = min(1.0, max(0.0, (step - args.warmup_steps) /
                                max(1, args.total_steps - args.warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.lr_min_ratio + (1.0 - args.lr_min_ratio) * cosine)

    evaluate = make_eval_fn(model, batch, opt, num_ctx)
    amp_dtype = torch.bfloat16
    rows = []
    for step in range(1, args.total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            _, metrics = model.step_loss(batch, step=step - 1, phase="train")
        metrics["loss"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        lr_now = lr_at(step - 1)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        optimizer.step()

        if step % args.log_every and step not in (1, args.mid_step, args.total_steps):
            continue
        row, pred, gt = evaluate()
        row.update({"source": args.source, "step": step, "lr": lr_now, "grad_norm": grad_norm})
        rows.append(row)
        print(f"[ab] {args.source} step {step:>4} lr {lr_now:.2e} loss {row['loss']:.4f} "
              f"rgb {row['loss_rgb']:.4f} ssim {row['loss_ssim_term']:.4f} gvis {row['loss_gvis']:.4f} | "
              f"ctx {row['ctx_psnr']:.2f}(g{row['ctx_grey_psnr']:.2f}) "
              f"novel {row['novel_psnr']:.2f}(g{row['novel_grey_psnr']:.2f}) | "
              f"ssim {row['ctx_ssim']:.3f}/{row['novel_ssim']:.3f} "
              f"alpha {row['alpha_mean']:.3f}>{row['alpha_gt_05']:.3f} | grad {grad_norm:.3f}",
              flush=True)

        if args.image_every and step % args.image_every == 0 and step not in (args.mid_step, args.total_steps):
            snap = torch.cat(
                [gt[:1], pred[:1], gt[num_ctx:num_ctx + 1], pred[num_ctx:num_ctx + 1]], dim=-1)[0]
            Image.fromarray((snap.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            ).save(out_dir / "images" / f"{args.source}_step{step}.png")

        save_steps = set()
        if "mid" in args.save_steps:
            save_steps.add(args.mid_step)
        if "final" in args.save_steps:
            save_steps.add(args.total_steps)
        if step in save_steps:
            tiles = [gt[:1], pred[:1], gt[num_ctx:num_ctx + 1], pred[num_ctx:num_ctx + 1]]
            grid = torch.cat(tiles, dim=-1)[0]
            Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            ).save(out_dir / "images" / f"{args.source}_step{step}.png")
            ck_dir = out_dir / f"ckpt_step{step}"
            ck_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "step": step}, ck_dir / "model.pt")
            if args.save_optimizer:
                torch.save({"optimizer": optimizer.state_dict(), "step": step}, ck_dir / "optimizer.pt")
            (ck_dir / "COMPLETE").write_text("complete\n")

    (out_dir / f"{args.source}_rows.json").write_text(
        json.dumps({"source": args.source, "pair": pair, "frames": frames,
                    "plan": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    print(f"[ab] wrote {out_dir}/{args.source}_rows.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
