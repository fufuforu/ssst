#!/usr/bin/env python3
"""Fixed-sample overfit probe for the pure TokenGS ScanNet baseline.

Structure/loss are exactly `train_siu3r_plain_tokengs_canonical_recon`
(TokenGS decoder, free-XYZ ClipActivationHead, canonical objective, 2 context +
2 novel supervision).  The `--recipe` switch only changes the learning-rate
schedule so the probe can be run with the preset's own recipe or with the
paper-like recipe used elsewhere in this repo.  Nothing else is modified.
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

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402

TRAIN_ROOT = "/space/mawb/SIU3R/data/scannet/train"
RECIPES = {"preset": {"lr": 1e-4, "warmup": 1000}, "paper_like": {"lr": 4e-4, "warmup": 2000}}


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="paper_like")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir",
                        default="/space/mawb/ssst/workspace_recon_diag/plain_tokengs_baseline")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    recipe = RECIPES[args.recipe]

    opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=args.seed, lr=recipe["lr"],
        pct_start_steps=recipe["warmup"],
    )
    torch.manual_seed(int(opt.seed))
    model = model_registry[opt.model_type](opt).to(device)
    model.freeze_object_queries()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=opt.lr, betas=(0.9, 0.95), weight_decay=0.05)
    amp_dtype = torch.bfloat16

    provider = SIU3RProcessedProvider(opt, root=TRAIN_ROOT, subset="all", training=True, rank=0)
    names = [s.name for s in provider.dataset.sample_list]
    idx = names.index(args.scene)
    provider.pair_rng.seed(int(opt.seed))
    batch = move(default_collate([provider[idx]]), device)
    pair = provider.last_pair
    frames = [int(x) for x in batch["frame_ids"][0]]
    print(f"[plain] recipe={args.recipe} lr={opt.lr} warmup={recipe['warmup']} steps={args.steps} "
          f"| scene={pair['scene_id']} ctx={pair['context_frame_ids']} novel={pair['novel_frame_ids']} frames={frames}")
    print(f"[plain] z_offset={opt.gaussian_z_offset} num_views={opt.num_views} "
          f"num_input_views={opt.num_input_views} lambda_ssim={opt.lambda_ssim} "
          f"lambda_lpips={opt.lambda_lpips} gvis_weight={opt.canonical_gaussian_visibility_weight}")

    def lr_at(step: int) -> float:
        if step < recipe["warmup"]:
            return opt.lr * float(step + 1) / float(recipe["warmup"])
        progress = min(1.0, max(0.0, (step - recipe["warmup"]) /
                                max(1, args.total_steps - recipe["warmup"])))
        return opt.lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    rows = []
    num_ctx = int(opt.num_input_views)
    gt = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output, metrics = model.step_loss(batch, step=step - 1, phase="train")
        metrics["loss"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        lr_now = lr_at(step - 1)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        optimizer.step()

        if step % args.log_every and step != 1:
            continue
        with torch.no_grad():
            render = output["render"]
            pred = render["images_pred"][0].float()
            from tokengs.models.canonical_recon_models import _full_supervision
            sup = _full_supervision(batch)
            gt_t = sup.images_output[0].float()
            def psnr(sl):
                return float(-10.0 * torch.log10((pred[sl] - gt_t[sl]).pow(2).mean().clamp_min(1e-12)))
            def ssim_v(sl):
                h, w = pred.shape[-2], pred.shape[-1]
                return float(1.0 - 2.0 * ssim_loss(
                    pred[sl].reshape(-1, 3, h, w), gt_t[sl].reshape(-1, 3, h, w)))
            grey = torch.full_like(gt_t, 0.5)
            alphas = render["alphas_pred"][0].float()
            row = {
                "recipe": args.recipe, "step": step, "lr": lr_now,
                "loss": float(metrics["loss"]), "loss_rgb": float(metrics["loss_rgb"]),
                "loss_ssim_term": float(metrics["loss_ssim"]),
                "loss_gvis": float(metrics.get("loss_gaussian_visibility", torch.tensor(0.0))),
                "ctx_psnr": psnr(slice(0, num_ctx)), "novel_psnr": psnr(slice(num_ctx, None)),
                "all_psnr": psnr(slice(None)),
                "ctx_ssim": ssim_v(slice(0, num_ctx)), "novel_ssim": ssim_v(slice(num_ctx, None)),
                "grey_psnr": float(-10.0 * torch.log10((grey - gt_t).pow(2).mean().clamp_min(1e-12))),
                "alpha_mean": float(alphas.mean()),
                "alpha_gt_05": float((alphas > 0.5).float().mean()),
                "depth_nonzero": float((render["depths_pred"] > 0).float().mean()),
                "grad_norm": grad_norm,
            }
            rows.append(row)
            print(f"[plain] {args.recipe} step {step:>4} lr {lr_now:.2e} loss {row['loss']:.4f} "
                  f"rgb {row['loss_rgb']:.4f} ssim {row['loss_ssim_term']:.4f} gvis {row['loss_gvis']:.4f} | "
                  f"ctx {row['ctx_psnr']:.2f} novel {row['novel_psnr']:.2f} (grey {row['grey_psnr']:.2f}) | "
                  f"alpha {row['alpha_mean']:.3f} >0.5 {row['alpha_gt_05']:.3f} | grad {grad_norm:.3f}", flush=True)
            if step in (250, 500, 750, args.steps):
                tiles = [gt_t[:1], pred[:1], gt_t[num_ctx:num_ctx + 1], pred[num_ctx:num_ctx + 1]]
                grid = torch.cat(tiles, dim=-1)[0]
                Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                ).save(out_dir / "images" / f"{args.recipe}_{args.scene}_step{step}.png")
                ck_dir = out_dir / f"ckpt_{args.recipe}_step{step}"
                ck_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict()}, ck_dir / "model.pt")
                (ck_dir / "COMPLETE").write_text("complete\n")
                import tyro

                (ck_dir / "config.yaml").write_text(tyro.extras.to_yaml(opt), encoding="utf-8")
        del sup

    (out_dir / f"{args.recipe}_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # per-token Gaussian dispersion on the final state (future comparison metric)
    with torch.no_grad():
        g = output["gaussians"][0].float()
        tokens = int(opt.num_gs_tokens)
        patches = g.shape[0] // tokens
        view = g[:, 0:3].reshape(tokens, patches, 3)
        centroid = view.mean(dim=1, keepdim=True)
        dist = (view - centroid).norm(dim=-1).flatten()
        s = torch.sort(dist).values
        stats = {f"token_disp_p{int(p*100)}": float(s[int(p * (s.numel() - 1))]) for p in (0.1, 0.5, 0.9, 0.95)}
        stats["token_disp_mean"] = float(dist.mean())
        stats["patches_per_token"] = int(patches)
        print(f"[plain] final per-token Gaussian dispersion: {stats}")
    (out_dir / f"{args.recipe}_summary.json").write_text(
        json.dumps({"pair": pair, "frames": frames, "rows": rows, "token_dispersion": stats,
                    "grey_psnr_final": rows[-1]["grey_psnr"]}, indent=2, default=str), encoding="utf-8")
    print(f"[plain] wrote {out_dir/(args.recipe + '_rows.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
