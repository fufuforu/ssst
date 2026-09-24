#!/usr/bin/env python3
"""Per-token Gaussian locality for a trained reconstruction model.

For every token, measures the distance from the Gaussians it generates to that
token's centroid, normalised by the size of the visible scene surface (RMS
radius of the points back-projected from the model's own rendered depth).  Also
records the LocusGS anchors, ``r * delta`` offsets, opacity, scale and alpha
coverage so that "it reconstructs" and "it is actually local" can be checked
separately.
"""
from __future__ import annotations

import argparse
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

from tokengs.data.scannet_raw_recon import (  # noqa: E402
    DEFAULT_SCANS_ROOT,
    ScanNetRawReconProvider,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def visible_surface_radius(render, batch, n_in, img_size):
    """RMS radius of the visible point cloud about its centroid, in scene units."""
    depth = render["depths_pred"][0].float()                      # [V,1,H,W]
    intr = batch["intrinsics_all"][0].float()
    cam_view = batch["cam_view_all"][0].float()
    c2w = torch.inverse(cam_view.transpose(1, 2))
    H, W = int(img_size[0]), int(img_size[1])
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=depth.device),
        torch.arange(W, dtype=torch.float32, device=depth.device),
        indexing="ij",
    )
    pts = []
    for v in range(depth.shape[0]):
        z = depth[v, 0]
        valid = z > 1e-4
        if valid.sum() < 16:
            continue
        fx, fy, cx, cy = intr[v]
        x = (xs - cx) / fx * z
        y = (ys - cy) / fy * z
        cam = torch.stack([x, y, z], dim=-1)[valid]
        world = cam @ c2w[v, :3, :3].T + c2w[v, :3, 3]
        pts.append(world)
    if not pts:
        return None, None
    pts = torch.cat(pts, dim=0)
    centroid = pts.mean(dim=0, keepdim=True)
    return float((pts - centroid).norm(dim=-1).mean()), int(pts.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint directory containing model.pt")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--source", default="raw", choices=["raw", "processed"])
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--context", type=int, nargs="+", default=[654, 664])
    parser.add_argument("--novel", type=int, nargs="+", default=[655, 659])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        num_input_views=len(args.context), num_views=len(args.context) + len(args.novel),
        batch_size=1, num_workers=0, seed=42,
    )
    model = model_registry[opt.model_type](opt)
    ckpt = Path(args.model)
    state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
    state = state.get("model", state)
    missing = [k for k in model.state_dict() if k not in state]
    unexpected = [k for k in state if k not in model.state_dict()]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    print(f"[loc] {opt.model_type}: missing={len(missing)} unexpected={len(unexpected)}")

    provider = ScanNetRawReconProvider(
        opt, root=str(DEFAULT_SCANS_ROOT), scene=args.scene,
        context_frame_ids=tuple(args.context), novel_frame_ids=tuple(args.novel),
        training=False,
    )
    batch = default_collate([provider[0]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    n_in = int(opt.num_input_views)
    img_size = (int(opt.img_size[0]), int(opt.img_size[1]))

    with torch.no_grad():
        model_input, _ = split_data(batch, opt)
        dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
        out = model.forward_reconstruction_only(
            ModelInput(model_input.encoder, dec), render_decoder_input=dec)
    render = out["render"]
    g = out["gaussians"][0].float()                                   # [N*P^2, 14]
    tokens = int(opt.num_gs_tokens)
    patches = g.shape[0] // tokens
    centers = g[:, 0:3].reshape(tokens, patches, 3)
    opacity = g[:, 3].reshape(tokens, patches)
    scale = g[:, 4:7].reshape(tokens, patches, 3)

    is_locusgs = hasattr(model, "anchor_decoder")
    if is_locusgs:
        ad = model.anchor_decoder
        anchor = ad.mu.detach().float()                                # [T,3]
        radii = ad.activated_radius(ad.rho.detach().float())           # [T]
        delta = (centers - anchor.unsqueeze(1)) / radii.unsqueeze(1).unsqueeze(-1)
        anchor_norm = radii.unsqueeze(1) * delta.norm(dim=-1)          # ||r*delta||
        token_centroid = anchor
        anchor_info = {
            "mu_abs_max": float(anchor.abs().max()),
            "mu_std": float(anchor.std(dim=0).mean()),
            "mu_z_mean": float(anchor[:, 2].mean()),
            "radius_mean": float(radii.mean()),
            "radius_p50": float(radii.median()),
            "radius_p99": float(radii.quantile(0.99)),
            "r_delta_norm_mean": float(anchor_norm.mean()),
            "r_delta_norm_p99": float(anchor_norm.flatten().quantile(0.99)),
            "r_delta_norm_max": float(anchor_norm.max()),
            "delta_norm_mean": float(delta.norm(dim=-1).mean()),
            "delta_norm_p99": float(delta.norm(dim=-1).flatten().quantile(0.99)),
        }
    else:
        token_centroid = centers.mean(dim=1)                            # [T,3]
        anchor_info = {}

    dist = (centers - token_centroid.unsqueeze(1)).norm(dim=-1)         # [T,P]
    vis_r, n_vis = visible_surface_radius(render, batch, n_in, img_size)
    alpha = render["alphas_pred"][0].float()

    flat = dist.flatten()
    quant = lambda q: float(flat.quantile(q))
    result = {
        "model": opt.model_type,
        "checkpoint": str(ckpt),
        "num_tokens": tokens,
        "gaussians_per_token": patches,
        "visible_surface_rms_radius": vis_r,
        "visible_points": n_vis,
        "gaussian_to_token_centroid": {
            "mean": float(flat.mean()), "p50": quant(0.5), "p90": quant(0.9),
            "p99": quant(0.99), "max": float(flat.max()),
            "mean_over_visible": float(flat.mean() / vis_r) if vis_r else None,
            "p90_over_visible": quant(0.9) / vis_r if vis_r else None,
            "p99_over_visible": quant(0.99) / vis_r if vis_r else None,
        },
        "per_token_spread_mean": float(dist.mean(dim=1).mean()),
        "opacity": {"mean": float(opacity.mean()), "p50": float(opacity.median()),
                    "p99": float(opacity.flatten().quantile(0.99))},
        "scale": {"mean": float(scale.mean()), "p50": float(scale.median()),
                  "p99": float(scale.flatten().quantile(0.99))},
        "alpha": {"mean": float(alpha.mean()),
                  "coverage_gt_05": float((alpha > 0.5).float().mean()),
                  "coverage_gt_01": float((alpha > 0.1).float().mean())},
        "anchor": anchor_info,
    }
    # sanity: reconstruction quality on the same batch
    pred = render["images_pred"][0].float()
    gt = batch["images_all"][0].float()
    psnr = lambda a, b: float(-10.0 * torch.log10((a - b).pow(2).mean().clamp_min(1e-12)))
    result["ctx_psnr"] = psnr(pred[:n_in], gt[:n_in])
    result["novel_psnr"] = psnr(pred[n_in:], gt[n_in:])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    # histogram of normalised distances
    hist, edges = np.histogram(flat.detach().cpu().numpy(),
                               bins=60, range=(0, float(flat.max())))
    payload = " ".join(f"{edges[i]:.4f}:{hist[i]}" for i in range(len(hist)))
    (Path(args.out).with_suffix(".hist.txt")).write_text(payload, encoding="utf-8")
    for k in ("ctx_psnr", "novel_psnr", "visible_surface_rms_radius", "per_token_spread_mean"):
        print(f"[loc] {k}: {result[k]}")
    print(f"[loc] gaussian->token-centroid: {result['gaussian_to_token_centroid']}")
    if anchor_info:
        print(f"[loc] anchor: {anchor_info}")
    print(f"[loc] alpha: {result['alpha']}")
    print(f"[loc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
