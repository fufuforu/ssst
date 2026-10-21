#!/usr/bin/env python3
"""Read-only audit of the full-split LocusGS run: model collapse or monitor bug?

Loads saved checkpoints, rebuilds **exactly** the four fixed monitor windows the
training job used (same provider, same `pair_rng.seed(seed + 1000 + i)`), runs an
independent forward pass and reports PSNR / SSIM / alpha / Gaussian statistics
plus GT-vs-prediction panels.  Nothing is trained and nothing is written into
the training output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(move(x, device) for x in v)
    return v


def q(x, p):
    return float(x.detach().float().flatten().quantile(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preset", default="train_siu3r_locusgs_recon_bounded_delta_frozen_radius")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="label=path (repeatable); path is a checkpoint dir with model.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    monitor = list(manifest["monitor_scenes"])
    val_root = Path(manifest["val_root"])
    print(f"[audit] monitor scenes {monitor} root {val_root}", flush=True)

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=args.seed, num_input_views=2, num_views=4)
    n_in = int(opt.num_input_views)

    # rebuild the exact monitor windows the training job evaluated
    entries = []
    for i, scene in enumerate(monitor):
        optv = config_defaults[args.preset].evolve(
            dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
            batch_size=1, num_workers=0, seed=args.seed, num_input_views=2, num_views=4)
        provider = SIU3RProcessedProvider(optv, root=str(val_root), subset=[scene],
                                          training=True, rank=0)
        provider.pair_rng.seed(int(opt.seed) + 1000 + i)
        batch = move(default_collate([provider[0]]), device)
        pair = provider.last_pair
        entries.append({"scene": scene, "batch": batch, "pair": pair})
        print(f"[audit] {scene}: ctx={pair['context_frame_ids']} "
              f"novel={pair['novel_frame_ids']}", flush=True)

    model = model_registry[opt.model_type](opt).to(device)
    report = {"manifest": args.manifest, "preset": args.preset, "seed": args.seed,
              "monitor_scenes": monitor, "checkpoints": {}}
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        ck = Path(path)
        payload = torch.load(ck / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model = model.to(device).eval()
        step = int(payload.get("step", -1))
        rows = []
        with torch.no_grad():
            for entry in entries:
                batch = entry["batch"]
                mi, _ = split_data(batch, opt)
                dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                        intrinsics=batch["intrinsics_all"])
                o = model.forward_reconstruction_only(ModelInput(mi.encoder, dec),
                                                      render_decoder_input=dec)
                pred = o["render"]["images_pred"][0].float()
                gt = batch["images_all"][0].float()
                grey = torch.full_like(gt, 0.5)
                alpha = o["render"]["alphas_pred"][0].float()
                g = o["gaussians"][0].float()
                mu = o["anchors"][0].float()
                radii = o["radii"][0].float()

                def psnr(a, b):
                    return float(-10.0 * torch.log10((a - b).pow(2).mean().clamp_min(1e-12)))

                def ssim_v(a, b):
                    h, w = a.shape[-2], a.shape[-1]
                    return float(1.0 - 2.0 * ssim_loss(a.reshape(-1, 3, h, w),
                                                       b.reshape(-1, 3, h, w)))

                rows.append({
                    "scene": entry["scene"],
                    "ctx_psnr": psnr(pred[:n_in], gt[:n_in]),
                    "novel_psnr": psnr(pred[n_in:], gt[n_in:]),
                    "ctx_grey": psnr(grey[:n_in], gt[:n_in]),
                    "novel_grey": psnr(grey[n_in:], gt[n_in:]),
                    "ctx_ssim": ssim_v(pred[:n_in], gt[:n_in]),
                    "novel_ssim": ssim_v(pred[n_in:], gt[n_in:]),
                    "alpha_mean": float(alpha.mean()),
                    "alpha_gt_05": float((alpha > 0.5).float().mean()),
                    "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
                    "scale_p50": q(g[:, 4:7], 0.5), "scale_p90": q(g[:, 4:7], 0.9),
                    "opacity_p50": q(g[:, 3], 0.5), "opacity_p90": q(g[:, 3], 0.9),
                    "center_z_p50": q(g[:, 2], 0.5), "center_z_p90": q(g[:, 2], 0.9),
                    "center_absmax": float(g[:, 0:3].abs().max()),
                    "anchor_absmax": float(mu.abs().max()),
                    "anchor_z_p50": q(mu[:, 2], 0.5),
                    "learned_radius_mean": float(radii.mean()),
                    "decode_radius": [float(model.activation_head.last_decode_radius.min()),
                                      float(model.activation_head.last_decode_radius.max())],
                })
                # panel: GT ctx | pred ctx | GT novel | pred novel
                tiles = [gt[:1], pred[:1], gt[n_in:n_in + 1], pred[n_in:n_in + 1]]
                grid = torch.cat(tiles, dim=-1)[0]
                img = Image.fromarray((grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                                       * 255).astype(np.uint8))
                d = ImageDraw.Draw(img)
                for j, t in enumerate(["GT ctx", "pred ctx", "GT novel", "pred novel"]):
                    d.text((j * 256 + 4, 4), t, fill=(255, 255, 0))
                d.text((4, 18), f"{label} step{step} {entry['scene']}", fill=(0, 255, 255))
                img.save(out / f"{label}_step{step}_{entry['scene']}.png")
        mean = lambda k: float(np.mean([r[k] for r in rows]))
        report["checkpoints"][label] = {
            "path": str(ck), "step": step, "rows": rows,
            "mean": {k: mean(k) for k in ("ctx_psnr", "novel_psnr", "ctx_grey", "novel_grey",
                                          "ctx_ssim", "novel_ssim", "alpha_mean", "alpha_gt_05",
                                          "pred_mean", "pred_std", "scale_p50", "scale_p90",
                                          "opacity_p50", "opacity_p90", "center_z_p50",
                                          "center_z_p90")}}
        m = report["checkpoints"][label]["mean"]
        print(f"[audit] {label:>14} step {step:>6} | ctx {m['ctx_psnr']:.2f} (grey {m['ctx_grey']:.2f}) "
              f"novel {m['novel_psnr']:.2f} (grey {m['novel_grey']:.2f}) "
              f"ssim {m['ctx_ssim']:.3f}/{m['novel_ssim']:.3f} alpha>0.5 {m['alpha_gt_05']:.3f} "
              f"| scale p50 {m['scale_p50']:.4f} opacity p50 {m['opacity_p50']:.3f} "
              f"z p50 {m['center_z_p50']:.2f}", flush=True)

    (out / "collapse_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[audit] wrote {out/'collapse_audit.json'} and panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
