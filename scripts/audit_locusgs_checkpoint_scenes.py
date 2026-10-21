#!/usr/bin/env python3
"""Per-scene, per-checkpoint spatial diagnostics (inference only, no optimizer).

Same 80-scene sample as audit_locusgs_spatial_scale.py, one pair per scene.
Records base radius, per-layer final radius / d_rho, r*delta, bias clamp and
context/novel PSNR so that scene-to-scene differences can be separated from
checkpoint-to-checkpoint (training-time) changes.  Read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

import tyro  # noqa: E402

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon import canonical_layer_loss  # noqa: E402
from tokengs.models.canonical_recon_models import (  # noqa: E402
    _full_supervision,
    patch_plucker_rays,
)
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.models.locusgs_recon import anchor_ray_geometric_bias  # noqa: E402
from tokengs.options import Options  # noqa: E402

ROOT = "/space/mawb/SIU3R/data/scannet/train"


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
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--scenes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir",
                        default="/space/mawb/ssst/workspace_recon_diag/locusgs_spatial_scale")
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = [Path(c) for c in args.checkpoint]
    for c in ckpts:
        if not (c / "COMPLETE").is_file():
            raise SystemExit(f"checkpoint {c} has no COMPLETE marker")
    opt = tyro.extras.from_yaml(Options, open(ckpts[0] / "config.yaml", encoding="utf-8"))
    opt = opt.evolve(evaluating=True, reconstruction_only=True)
    provider = SIU3RProcessedProvider(opt, root=ROOT, subset="all", training=True, rank=0)
    names = sorted(s.name for s in provider.dataset.sample_list)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(names), size=min(args.scenes, len(names)), replace=False).tolist())
    print(f"[ckpt] {len(chosen)} scenes x {len(ckpts)} checkpoints = {len(chosen)*len(ckpts)} forwards")

    pairs = {}
    for idx in chosen:
        provider.pair_rng.seed(args.seed + 1000 * idx + args.pair_index)
        pairs[idx] = provider[idx]
    print("[ckpt] pairs sampled")

    rows = []
    for ckpt in ckpts:
        step = json.loads((ckpt / "global_metadata.json").read_text())["global_optimizer_step"]
        model = model_registry[opt.model_type](opt).to(device).eval()
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        state = {k: v for k, v in state.items() if "lpips_loss" not in k}
        model.load_state_dict(state, strict=True)
        print(f"[ckpt] loaded step {step}")
        for idx in chosen:
            batch = move(default_collate([pairs[idx]]), device)
            model_input, _ = split_data(batch, opt)
            decoder_input = ModelInputDecoder(
                cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
            )
            supervision = _full_supervision(batch)
            num_ctx = int(opt.num_input_views)
            with torch.no_grad():
                states, _ = model._decode(
                    ModelInput(model_input.encoder, decoder_input), decoder_input
                )
                radii_by_layer = [float(s["radii"].mean()) for s in states]
                d_rho_by_layer = [float(s["radius_update"]) for s in states]
                row = {
                    "checkpoint_step": int(step), "scene": names[idx],
                    "base_radius": float(model.anchor_decoder.activated_radius(
                        model.anchor_decoder.rho).mean()),
                    "radius_layer6": radii_by_layer[5],
                    "radius_layer12": radii_by_layer[11],
                    "d_rho_sum_over_layers": float(sum(d_rho_by_layer)),
                    "d_rho_layer6": d_rho_by_layer[5],
                    "d_rho_layer12": d_rho_by_layer[11],
                    "d_rho_per_layer": ";".join(f"{v:.6f}" for v in d_rho_by_layer),
                }
                last = model.supervised_layers[-1]
                st = states[last - 1]
                g = model.activation_head(st["tokens"], st["mu"], st["radii"])
                rec = model._reconstruction_from_gaussians(g)
                render = model.render_reconstruction(rec, decoder_input)
                loss = canonical_layer_loss(
                    opt=opt, img_size=model.img_size, render_results=render,
                    supervision=supervision, decoder_input=decoder_input, gaussians=g,
                    anchor_centers=st["mu"],
                    anchor_weight=float(opt.canonical_anchor_visibility_weight),
                )
                pred = render["images_pred"][0].float()
                gt = supervision.images_output[0].float()

                def psnr(sl):
                    mse = (pred[sl] - gt[sl]).pow(2).mean().clamp_min(1e-12)
                    return float(-10.0 * torch.log10(mse))

                mom, dirn = patch_plucker_rays(
                    model_input.encoder.rays_os, model_input.encoder.rays_ds,
                    patch_size=int(opt.patch_size))
                geo = anchor_ray_geometric_bias(
                    st["mu"], st["radii"], mom, dirn,
                    sigma0=float(opt.locusgs_sigma0),
                    bandwidth_floor=float(opt.locusgs_bandwidth_floor),
                    clamp_min=float(opt.locusgs_bias_clamp))
                patches = max(1, g.shape[1] // st["mu"].shape[1])
                mu_e = st["mu"].repeat_interleave(patches, dim=1)
                r_e = st["radii"].repeat_interleave(patches, dim=1)
                delta = (g[..., 0:3] - mu_e) / (r_e.unsqueeze(-1) + float(opt.locusgs_radius_epsilon))
                rdelta = (r_e * delta.norm(dim=-1)).float()
                row.update({
                    "ctx_psnr": psnr(slice(0, num_ctx)),
                    "novel_psnr": psnr(slice(num_ctx, None)),
                    "all_psnr": psnr(slice(None)),
                    "rgb": float(loss["loss_rgb"]),
                    "ssim_term": float(loss["loss_ssim"]),
                    "gvis": float(loss.get("loss_gaussian_visibility", torch.tensor(0.0))),
                    "avis": float(loss.get("loss_anchor_visibility", torch.tensor(0.0))),
                    "bias_clamp_fraction": float(
                        (geo <= float(opt.locusgs_bias_clamp) + 1e-6).float().mean()),
                    "rdelta_p50": float(rdelta.median()),
                    "rdelta_p95": float(rdelta.flatten().quantile(0.95)),
                    "alpha_nonzero": float((render["alphas_pred"] > 0).float().mean()),
                    "depth_nonzero": float((render["depths_pred"] > 0).float().mean()),
                })
            rows.append(row)
            del batch
        del model
        torch.cuda.empty_cache()

    with open(out_dir / "per_scene_checkpoints.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    (out_dir / "per_scene_checkpoints.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[ckpt] wrote {out_dir/'per_scene_checkpoints.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
