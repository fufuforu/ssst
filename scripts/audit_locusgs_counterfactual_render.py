#!/usr/bin/env python3
"""Read-only counterfactual render audit for LocusGS checkpoints.

Keeps encoder / decoder / anchors / RGB / opacity / scale / rotation / cameras /
attention exactly as the checkpoint produced them and only rescales the final
local displacement:

    mu_G(alpha) = mu + alpha * (r * delta),   alpha in {0.10, 0.25, 0.50, 0.75, 1.00}

alpha = 1 must reproduce the unmodified image bit-for-bit (correctness gate).
No optimizer step, no parameter assignment, no writes into any existing
evaluation directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

import tyro  # noqa: E402
from einops import rearrange  # noqa: E402

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon import ssim_loss  # noqa: E402
from tokengs.models.canonical_recon_models import _full_supervision, patch_plucker_rays  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import Options  # noqa: E402

TRAIN_ROOT = "/space/mawb/SIU3R/data/scannet/train"
VAL_ROOT = "/space/mawb/SIU3R/data/scannet/val"
VAL_MANIFEST = ("/space/mawb/tokengs_siu3r_joint_v1/workspace/"
                "siu3r_tokengs_joint_from_scratch_text_v2/fixed_siu3r_validation16_v1.json")
ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00]


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def pstats(x: torch.Tensor) -> dict:
    flat = x.detach().float().flatten()
    s = torch.sort(flat).values
    n = s.numel()
    out = {}
    for p in (0.10, 0.50, 0.90, 0.95):
        out[f"p{int(p*100)}"] = float(s[min(n - 1, int(round(p * (n - 1))))])
    out["mean"] = float(flat.mean())
    return out


def psnr_ssim(pred: torch.Tensor, gt: torch.Tensor):
    mse = (pred - gt).pow(2).mean().clamp_min(1e-12)
    h = pred.shape[-2]
    return float(-10.0 * torch.log10(mse)), float(1.0 - 2.0 * ssim_loss(
        pred.reshape(-1, 3, h, pred.shape[-1]), gt.reshape(-1, 3, h, gt.shape[-1])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--protocol", choices=["val16", "train80"], default="val16")
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir",
                        default="/space/mawb/ssst/workspace_recon_diag/locusgs_counterfactual")
    parser.add_argument("--save-images", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    ckpts = [Path(c) for c in args.checkpoint]
    for c in ckpts:
        if not (c / "COMPLETE").is_file():
            raise SystemExit(f"{c} lacks COMPLETE")
    opt = tyro.extras.from_yaml(Options, open(ckpts[0] / "config.yaml", encoding="utf-8"))
    opt = opt.evolve(evaluating=True, reconstruction_only=True)
    if args.protocol == "val16":
        opt = opt.evolve(num_views=6)
        provider = SIU3RProcessedProvider(opt, root=VAL_ROOT, subset="all", training=False,
                                          val_pair_json=VAL_MANIFEST, rank=0)
        full_records = list(provider.dataset.val_pairs)
        # iterate by record index exactly like scripts/evaluate_ssst_validation.py
        scenes = [(str(full_records[i]["scene"]), i) for i in range(len(full_records))][: int(args.scenes or 16)]
        print(f"[cf] protocol=val16 scenes={len(scenes)} (2 context + 4 novel)")
    else:
        provider = SIU3RProcessedProvider(opt, root=TRAIN_ROOT, subset="all", training=True, rank=0)
        names = sorted(s.name for s in provider.dataset.sample_list)
        rng = np.random.default_rng(args.seed)
        idxs = sorted(rng.choice(len(names), size=int(args.scenes or 80), replace=False).tolist())
        scenes = [(names[i], i) for i in idxs]
        print(f"[cf] protocol=train80 scenes={len(scenes)} (2 context + 2 novel)")

    rows = []
    for ckpt in ckpts:
        step = json.loads((ckpt / "global_metadata.json").read_text())["global_optimizer_step"]
        model = model_registry[opt.model_type](opt).to(device).eval()
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        state = {k: v for k, v in state.items() if "lpips_loss" not in k}
        model.load_state_dict(state, strict=True)
        print(f"[cf] loaded step {step} (strict OK)")
        saved = 0
        for si, (scene, ds_index) in enumerate(scenes):
            if args.protocol == "val16":
                sample = provider[ds_index]
            else:
                provider.pair_rng.seed(args.seed + 1000 * ds_index + args.pair_index)
                sample = provider[ds_index]
            batch = move(default_collate([sample]), device)
            model_input, _ = split_data(batch, opt)
            decoder_input = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                              intrinsics=batch["intrinsics_all"])
            supervision = _full_supervision(batch)
            num_ctx = int(opt.num_input_views)
            H, W = int(opt.img_size[0]), int(opt.img_size[1])
            with torch.no_grad():
                states, _ = model._decode(ModelInput(model_input.encoder, decoder_input), decoder_input)
                st = states[-1]
                mu = st["mu"]                     # [1,N,3]
                radii = st["radii"]               # [1,N]
                g = model.activation_head(st["tokens"], mu, radii)   # [1,N*P,14]
                patches = max(1, g.shape[1] // mu.shape[1])
                mu_e = mu.repeat_interleave(patches, dim=1)
                r_e = radii.repeat_interleave(patches, dim=1)
                # the exact displacement used by the head is (centre - mu); recover it
                # directly so that alpha = 1 can reuse the original tensor bit-for-bit.
                rdelta = g[..., 0:3] - mu_e
                delta = rdelta / (r_e.unsqueeze(-1) + float(opt.locusgs_radius_epsilon))
                base_render = model.render_reconstruction(
                    model._reconstruction_from_gaussians(g), decoder_input)
                base_pred = base_render["images_pred"][0].float()
                gt = supervision.images_output[0].float()
                base_psnr = psnr_ssim(base_pred, gt)[0]
                # scene visible-surface bbox diagonal (normalized frame)
                depths = []
                pts = []
                for v in range(num_ctx):
                    d_m = torch.from_numpy(np.asarray(
                        Image.open(Path(VAL_ROOT if args.protocol == "val16" else TRAIN_ROOT)
                                   / scene / "depth" / f"{int(batch['frame_ids'][0, v])}.png")
                    ).astype(np.float32) / 1000.0).to(device) * 0.15
                    fx, fy, cx, cy = [float(x) for x in batch["intrinsics_all"][0, v]]
                    uu, vv = torch.meshgrid(torch.arange(d_m.shape[1], device=device),
                                            torch.arange(d_m.shape[0], device=device), indexing="xy")
                    valid = d_m > 1e-6
                    pc = torch.stack([(uu - cx) / fx * d_m, (vv - cy) / fy * d_m, d_m], -1)[valid]
                    c2w = torch.inverse(batch["cam_view_all"][0, v].T)
                    pts.append(pc @ c2w[:3, :3].T + c2w[:3, 3])
                pts = torch.cat(pts)
                bbox_diag = float(torch.linalg.norm(pts.max(0).values - pts.min(0).values))

                # ---- per-Gaussian statistics (item 1) -------------------- #
                stats = {
                    "delta_norm": pstats(delta.norm(dim=-1)),
                    "rdelta_norm": pstats(rdelta.norm(dim=-1)),
                    "mu_norm": pstats(mu.norm(dim=-1)),
                    "center_anchor_dist": pstats(rdelta.norm(dim=-1)),
                    "rgb_mean": pstats(g[..., 11:14].mean(-1)),
                    "opacity": pstats(g[..., 3]),
                    "scale": pstats(g[..., 4:7].mean(-1)),
                    "bbox_diag": bbox_diag,
                    "rdelta_over_bbox_p50": float(rdelta.norm(dim=-1).median()) / max(bbox_diag, 1e-9),
                }
                # ---- counterfactual alpha renders (item 2) ---------------- #
                for alpha in ALPHAS:
                    gg = g.clone()
                    if alpha != 1.0:      # alpha = 1 keeps the checkpoint's own centres
                        gg[..., 0:3] = mu_e + alpha * rdelta
                    render = model.render_reconstruction(
                        model._reconstruction_from_gaussians(gg), decoder_input)
                    pred = render["images_pred"][0].float()
                    ctx_p, ctx_s = psnr_ssim(pred[:num_ctx], gt[:num_ctx])
                    nov_p, nov_s = psnr_ssim(pred[num_ctx:], gt[num_ctx:])
                    cen = gg[..., 0:3]
                    z = (batch["cam_view_all"][0].transpose(-1, -2)[..., 2, :] @ torch.cat(
                        [cen[0], torch.ones(cen.shape[1], 1, device=device)], -1).unsqueeze(-1)).squeeze(-1)
                    # analytic visibility: z>znear (per view), counted over all views
                    visible = (z > float(opt.znear)).any(dim=0).float().mean()
                    rows.append({
                        "checkpoint_step": int(step), "scene": scene, "alpha": alpha,
                        "ctx_psnr": ctx_p, "novel_psnr": nov_p,
                        "all_psnr": psnr_ssim(pred, gt)[0],
                        "ctx_ssim": ctx_s, "novel_ssim": nov_s,
                        "alpha_nonzero": float((render["alphas_pred"] > 0).float().mean()),
                        "depth_nonzero": float((render["depths_pred"] > 0).float().mean()),
                        "gaussian_visible_frac": float(visible),
                        "center_abs_delta_vs_base": float((gg[..., 0:3] - g[..., 0:3]).abs().max()),
                        **{f"{k}_{kk}": vv for k, v in stats.items() if isinstance(v, dict)
                           for kk, vv in v.items()},
                        "bbox_diag": bbox_diag,
                        "rdelta_over_bbox_p50": stats["rdelta_over_bbox_p50"],
                    })
                # correctness gate at alpha = 1
                a1 = [r for r in rows if r["checkpoint_step"] == step and r["scene"] == scene
                      and r["alpha"] == 1.0][-1]
                assert a1["center_abs_delta_vs_base"] == 0.0, "alpha=1 changed the centres"
                assert abs(a1["all_psnr"] - base_psnr) < 1e-6, "alpha=1 did not reproduce the image"
                # ---- image grid (item 2 deliverable) ---------------------- #
                if saved < args.save_images:
                    tiles = [gt[:1], base_pred[:1]]
                    for alpha in ALPHAS[:-1]:
                        gg = g.clone()
                        gg[..., 0:3] = mu_e + alpha * rdelta
                        rr = model.render_reconstruction(
                            model._reconstruction_from_gaussians(gg), decoder_input)
                        tiles.append(rr["images_pred"][0, :1].float())
                    grid = torch.cat(tiles, dim=-1)[0]          # [3,H,W*7]
                    arr = (grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(arr).save(out_dir / "images" / f"step{step}_{scene}.png")
                    saved += 1
            del batch
        del model
        torch.cuda.empty_cache()
        print(f"[cf] step {step} done")

    keys = list(rows[0].keys())
    with open(out_dir / "counterfactual_rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    (out_dir / "counterfactual_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[cf] wrote {out_dir/'counterfactual_rows.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
