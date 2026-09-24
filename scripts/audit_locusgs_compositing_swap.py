#!/usr/bin/env python3
"""Read-only compositing + checkpoint-attribute-swap audit for LocusGS 250/500.

Covers: (1) actual pixel compositing (background identity, alpha statistics,
foreground-only render, constant-grey baseline), (2) same-slot attribute swaps
between the two checkpoints, (3) true anchor-centre deltas per token.
No training, no optimizer, no parameter assignment, no overwrite of existing
evaluation files.
"""
from __future__ import annotations

import argparse
import csv
import json
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

import tyro  # noqa: E402

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.models.canonical_recon_models import _full_supervision  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import Options  # noqa: E402

TRAIN_ROOT = "/space/mawb/SIU3R/data/scannet/train"
VAL_ROOT = "/space/mawb/SIU3R/data/scannet/val"
VAL_MANIFEST = ("/space/mawb/tokengs_siu3r_joint_v1/workspace/"
                "siu3r_tokengs_joint_from_scratch_text_v2/fixed_siu3r_validation16_v1.json")


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def psnr_ssim(pred: torch.Tensor, gt: torch.Tensor):
    mse = (pred - gt).pow(2).mean().clamp_min(1e-12)
    h, w = pred.shape[-2], pred.shape[-1]
    ssim = 1.0 - 2.0 * ssim_loss(pred.reshape(-1, 3, h, w), gt.reshape(-1, 3, h, w))
    return float(-10.0 * torch.log10(mse)), float(ssim)


def q(x: torch.Tensor, probs=(0.10, 0.50, 0.90)):
    s = torch.sort(x.detach().float().flatten()).values
    n = s.numel()
    return {f"p{int(p*100)}": float(s[min(n - 1, int(round(p * (n - 1))))]) for p in probs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-250", required=True)
    parser.add_argument("--checkpoint-500", required=True)
    parser.add_argument("--protocol", choices=["val16", "train80"], default="val16")
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir",
                        default="/space/mawb/ssst/workspace_recon_diag/locusgs_compositing")
    parser.add_argument("--save-images", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    ck = {250: Path(args.checkpoint_250), 500: Path(args.checkpoint_500)}
    for step, path in ck.items():
        if not (path / "COMPLETE").is_file():
            raise SystemExit(f"{path} lacks COMPLETE")
    opt = tyro.extras.from_yaml(Options, open(ck[500] / "config.yaml", encoding="utf-8"))
    opt = opt.evolve(evaluating=True, reconstruction_only=True)
    if args.protocol == "val16":
        opt = opt.evolve(num_views=6)
        provider = SIU3RProcessedProvider(opt, root=VAL_ROOT, subset="all", training=False,
                                          val_pair_json=VAL_MANIFEST, rank=0)
        records = list(provider.dataset.val_pairs)
        scenes = [(str(records[i]["scene"]), i) for i in range(len(records))][: int(args.scenes or 16)]
    else:
        provider = SIU3RProcessedProvider(opt, root=TRAIN_ROOT, subset="all", training=True, rank=0)
        names = sorted(s.name for s in provider.dataset.sample_list)
        rng = np.random.default_rng(args.seed)
        idxs = sorted(rng.choice(len(names), size=int(args.scenes or 80), replace=False).tolist())
        scenes = [(names[i], i) for i in idxs]
    print(f"[cs] protocol={args.protocol} scenes={len(scenes)} | bg_color={opt.bg_color}")

    models = {}
    for step, path in ck.items():
        m = model_registry[opt.model_type](opt).to(device).eval()
        sd = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        sd = {k: v for k, v in sd.items() if "lpips_loss" not in k}
        m.load_state_dict(sd, strict=True)
        models[step] = m
    print("[cs] both checkpoints loaded strictly")

    rows, anchor_rows = [], []
    saved = 0
    for scene, ds_index in scenes:
        if args.protocol == "val16":
            sample = provider[ds_index]
        else:
            provider.pair_rng.seed(args.seed + 1000 * ds_index + args.pair_index)
            sample = provider[ds_index]
        batch = move(default_collate([sample]), device)
        decoder_input = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                          intrinsics=batch["intrinsics_all"])
        supervision = _full_supervision(batch)
        gt = supervision.images_output[0].float()
        num_ctx = int(opt.num_input_views)
        gs, renders = {}, {}
        with torch.no_grad():
            for step, model in models.items():
                model_input, _ = split_data(batch, opt)
                states, _ = model._decode(ModelInput(model_input.encoder, decoder_input), decoder_input)
                st = states[-1]
                g = model.activation_head(st["tokens"], st["mu"], st["radii"])
                gs[step] = g
                renders[step] = model.render_reconstruction(
                    model._reconstruction_from_gaussians(g), decoder_input)
                if step == 250:
                    states250 = states
                else:
                    states500 = states
            # ---- item 1: compositing -------------------------------------- #
            bg = models[500]._background_color(torch.float32, device)
            black = torch.zeros(3, device=device)
            for step in (250, 500):
                pred = renders[step]["images_pred"][0].float()
                alphas = renders[step]["alphas_pred"][0].float()
                g_black = models[step].gs.render(gs[step], decoder_input.cam_view,
                                                 bg_color=black, intrinsics=decoder_input.intrinsics)
                fg = g_black["images_pred"][0].float()
                recon = fg + (1.0 - alphas) * bg.view(1, 3, 1, 1)
                ctx_p, ctx_s = psnr_ssim(pred[:num_ctx], gt[:num_ctx])
                nov_p, nov_s = psnr_ssim(pred[num_ctx:], gt[num_ctx:])
                grey = torch.full_like(gt, 0.5)
                rows.append({
                    "protocol": args.protocol, "scene": scene, "checkpoint_step": 250 if step == 250 else 500,
                    "gt_mean": float(gt.mean()), "gt_std": float(gt.std()),
                    "gt_spatial_var": float(gt.var()),
                    "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
                    "pred_spatial_var": float(pred.var()),
                    "pred_ch_mean": ";".join(f"{float(pred[:, i].mean()):.5f}" for i in range(3)),
                    "pred_ch_std": ";".join(f"{float(pred[:, i].std()):.5f}" for i in range(3)),
                    "alpha_mean": float(alphas.mean()), **q(alphas),
                    "alpha_gt_01": float((alphas > 0.1).float().mean()),
                    "alpha_gt_05": float((alphas > 0.5).float().mean()),
                    "alpha_gt_09": float((alphas > 0.9).float().mean()),
                    "fg_mean": float(fg.mean()), "fg_std": float(fg.std()),
                    "bg_recon_abs_err_max": float((pred - recon).abs().max()),
                    "bg_recon_abs_err_mean": float((pred - recon).abs().mean()),
                    "ctx_psnr": ctx_p, "novel_psnr": nov_p, "ctx_ssim": ctx_s, "novel_ssim": nov_s,
                    "grey_psnr": float(-10.0 * torch.log10((grey - gt).pow(2).mean().clamp_min(1e-12))),
                    "beats_grey": bool(psnr_ssim(pred, gt)[0] > float(
                        -10.0 * torch.log10((grey - gt).pow(2).mean().clamp_min(1e-12)))),
                })
            # ---- item 2: attribute swaps ---------------------------------- #
            def build(center, scale, rot, opacity, rgb):
                out = gs[500].clone()
                out[..., 0:3] = center
                out[..., 3] = opacity
                out[..., 4:7] = scale
                out[..., 7:11] = rot
                out[..., 11:14] = rgb
                return out
            G5, G2 = gs[500], gs[250]
            swaps = {
                "all250": G2,
                "all500": G5,
                "geom250_app500": build(G2[..., 0:3], G2[..., 4:7], G2[..., 7:11],
                                        G5[..., 3], G5[..., 11:14]),
                "geom500_app250": build(G5[..., 0:3], G5[..., 4:7], G5[..., 7:11],
                                        G2[..., 3], G2[..., 11:14]),
                "500swap_center": build(G2[..., 0:3], G5[..., 4:7], G5[..., 7:11],
                                        G5[..., 3], G5[..., 11:14]),
                "500swap_scale_rot": build(G5[..., 0:3], G2[..., 4:7], G2[..., 7:11],
                                           G5[..., 3], G5[..., 11:14]),
                "500swap_opacity": build(G5[..., 0:3], G5[..., 4:7], G5[..., 7:11],
                                         G2[..., 3], G5[..., 11:14]),
                "500swap_rgb": build(G5[..., 0:3], G5[..., 4:7], G5[..., 7:11],
                                     G5[..., 3], G2[..., 11:14]),
            }
            for name, g in swaps.items():
                if not torch.isfinite(g).all():
                    print(f"[cs] {name}: non-finite parameters -> skipped"); continue
                r = models[500].render_reconstruction(
                    models[500]._reconstruction_from_gaussians(g), decoder_input)
                pred = r["images_pred"][0].float()
                a = r["alphas_pred"][0].float()
                cp, cs = psnr_ssim(pred[:num_ctx], gt[:num_ctx])
                np_, ns = psnr_ssim(pred[num_ctx:], gt[num_ctx:])
                rows.append({
                    "protocol": args.protocol, "scene": scene, "checkpoint_step": f"swap:{name}",
                    "ctx_psnr": cp, "novel_psnr": np_, "ctx_ssim": cs, "novel_ssim": ns,
                    "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
                    "alpha_mean": float(a.mean()), **q(a),
                    "alpha_gt_01": float((a > 0.1).float().mean()),
                    "alpha_gt_05": float((a > 0.5).float().mean()),
                    "alpha_gt_09": float((a > 0.9).float().mean()),
                })
                # bit-exactness gate for the two pure groups
                if name in ("all250", "all500"):
                    ref = renders[250 if name == "all250" else 500]["images_pred"][0].float()
                    assert (pred - ref).abs().max() == 0.0, f"{name} does not reproduce the original render"
            # ---- item 3: true anchor deltas ------------------------------- #
            for layer in (6, 12):
                d = (states500[layer - 1]["mu"] - states250[layer - 1]["mu"]).norm(dim=-1)
                anchor_rows.append({
                    "protocol": args.protocol, "scene": scene, "layer": layer,
                    **{f"dmu_{k}": v for k, v in q(d, (0.10, 0.50, 0.90, 0.95)).items()},
                    "dmu_mean": float(d.mean()),
                    "mu250_norm_p50": float(states250[layer - 1]["mu"].norm(dim=-1).median()),
                    "mu500_norm_p50": float(states500[layer - 1]["mu"].norm(dim=-1).median()),
                })
            # ---- images ---------------------------------------------------- #
            if saved < args.save_images:
                panels = [gt[:1], renders[250]["images_pred"][0, :1].float(),
                          renders[500]["images_pred"][0, :1].float(),
                          models[500].render_reconstruction(
                              models[500]._reconstruction_from_gaussians(swaps["geom250_app500"]),
                              decoder_input)["images_pred"][0, :1].float(),
                          models[500].render_reconstruction(
                              models[500]._reconstruction_from_gaussians(swaps["geom500_app250"]),
                              decoder_input)["images_pred"][0, :1].float()]
                grid = torch.cat(panels, dim=-1)[0]
                Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                ).save(out_dir / "images" / f"{args.protocol}_{scene}_swaps.png")
                # compositing panels: GT | pred | alpha | fg(black bg)
                alpha_vis = renders[500]["alphas_pred"][0, :1].float().repeat(1, 3, 1, 1)
                fg_vis = models[500].gs.render(gs[500], decoder_input.cam_view, bg_color=black,
                                               intrinsics=decoder_input.intrinsics)["images_pred"][0, :1].float()
                cgrid = torch.cat([gt[:1], renders[500]["images_pred"][0, :1].float(), alpha_vis, fg_vis],
                                  dim=-1)[0]
                Image.fromarray((cgrid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                ).save(out_dir / "images" / f"{args.protocol}_{scene}_compositing.png")
                saved += 1
        del batch
        torch.cuda.empty_cache()

    with open(out_dir / f"{args.protocol}_rows.csv", "w", newline="") as handle:
        keys = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(out_dir / f"{args.protocol}_anchor_deltas.csv", "w", newline="") as handle:
        keys = sorted({k for r in anchor_rows for k in r})
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in anchor_rows:
            writer.writerow(r)
    (out_dir / f"{args.protocol}_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[cs] wrote {out_dir/(args.protocol + '_rows.csv')} ({len(rows)} rows) and anchor deltas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
