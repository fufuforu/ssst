#!/usr/bin/env python3
"""Same-tensor ray-bias / attention audit for the LocusGS V2 model at step 0.

Every statistic is taken from the *same* tensors that the forward pass feeds to
`anchor_ray_geometric_bias()` and to the cross-attention softmax, so the
all-pair distance distribution, the clamp threshold and the per-query valid-ray
counts cannot disagree by construction.  Read-only: forward only, no optimizer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.canonical_recon_models import patch_plucker_rays  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.models.locusgs_recon import (  # noqa: E402
    anchor_ray_geometric_bias,
    plucker_point_distance,
)
from tokengs.options import config_defaults  # noqa: E402

TRAIN_ROOT = "/space/mawb/SIU3R/data/scannet/train"
TARGET_SCENE = "scene0048_01"
PERCENTILES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def pct(x: torch.Tensor, extra: bool = True) -> dict:
    """Nearest-rank percentiles (sort-based, so it also works on >16M tensors)."""
    flat = x.detach().float().flatten()
    sorted_values = torch.sort(flat).values
    n = sorted_values.numel()
    out = {
        f"p{p*1000:05.1f}".replace(".0", ""): float(
            sorted_values[min(n - 1, max(0, int(round(p * (n - 1)))))]
        )
        for p in PERCENTILES
    }
    if extra:
        out.update({"min": float(flat.min()), "max": float(flat.max()),
                    "mean": float(flat.mean()), "std": float(flat.std())})
    return out


def attn_stats(logits: torch.Tensor) -> dict:
    """logits: [B,H,N,P] -> entropy / effective rays / top-k mass per (head,query)."""
    probs = logits.softmax(dim=-1)
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1)  # [B,H,N]
    eff = entropy.exp()
    topk = {}
    sorted_probs = probs.sort(dim=-1, descending=True).values
    cum = sorted_probs.cumsum(-1)
    for k in (1, 4, 8, 16, 32, 64):
        topk[f"top{k}_mass"] = float(cum[..., k - 1].mean())
    return {
        "entropy_mean": float(entropy.mean()),
        "entropy_p50": float(entropy.flatten().quantile(0.5)),
        "entropy_p95": float(entropy.flatten().quantile(0.95)),
        "effective_rays_mean": float(eff.mean()),
        **{f"{k}_mean": v for k, v in topk.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="train_siu3r_locusgs_inferred_v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)

    opt = config_defaults[args.preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        num_views=4, num_input_views=2, img_size=(256, 256), batch_size=1,
        num_workers=0, seed=42, lr=4e-4, pct_start_steps=2000,
    )
    torch.manual_seed(int(opt.seed))
    model = model_registry[opt.model_type](opt).to(device).eval()
    decoder = model.anchor_decoder

    provider = SIU3RProcessedProvider(opt, root=TRAIN_ROOT, subset="all", training=True, rank=0)
    names = [scene.name for scene in provider.dataset.sample_list]
    index = names.index(TARGET_SCENE) if TARGET_SCENE in names else 0
    provider.pair_rng.seed(int(opt.seed))
    batch = move(default_collate([provider[index]]), device)
    model_input, _ = split_data(batch, opt)

    report: dict = {"preset": args.preset, "pair": provider.last_pair,
                    "sigma0": float(opt.locusgs_sigma0),
                    "radius_init": float(opt.locusgs_radius_init),
                    "gamma_raw_init": float(opt.locusgs_gamma_raw_init)}
    print(f"[audit] preset={args.preset} impl={decoder.impl} sigma0={opt.locusgs_sigma0} "
          f"r_init={opt.locusgs_radius_init} gamma_raw_init={opt.locusgs_gamma_raw_init}")

    with torch.no_grad():
        latent = model.forward_encoder(model_input.encoder)
        rays = patch_plucker_rays(
            model_input.encoder.rays_os, model_input.encoder.rays_ds,
            patch_size=int(opt.patch_size),
        )
        moment, direction = rays
        keys = latent.keys
        num_heads = model.enc_dec_backbone.decoder_blocks[0].gs_cross_attn.num_heads
        head_dim = int(opt.enc_embed_dim) // num_heads
        scale = head_dim ** -0.5

        tokens = model.get_gs_tokens(batch_size=1)
        mu = decoder.mu.unsqueeze(0).expand(1, -1, -1).contiguous()
        rho = decoder.rho.unsqueeze(0).expand(1, -1).contiguous()
        num_patch = int(moment.shape[1])
        print(f"[audit] anchors={mu.shape[1]} patch_rays={num_patch} heads={num_heads} head_dim={head_dim}")
        report["num_anchors"] = int(mu.shape[1])
        report["num_patch_rays"] = num_patch

        layers = {}
        ratio_samples = []
        logit_layers = {}
        for idx, blk in enumerate(decoder.decoder_blocks):
            layer = idx + 1
            radii = decoder.activated_radius(rho)
            geometric = anchor_ray_geometric_bias(
                mu, radii, moment, direction,
                sigma0=float(opt.locusgs_sigma0),
                bandwidth_floor=float(opt.locusgs_bandwidth_floor),
                clamp_min=float(opt.locusgs_bias_clamp),
            )  # [B,1,N,P]
            distance = plucker_point_distance(mu, moment, direction)  # [B,N,P]
            gamma = float(F.softplus(decoder.gamma_raw[idx].detach().float()))
            clamp_min = float(opt.locusgs_bias_clamp)
            threshold = math.sqrt(-2.0 * clamp_min) * float(opt.locusgs_sigma0) * radii  # [B,N]
            manual_clamped = (distance >= threshold.unsqueeze(-1)).float().mean()
            actual_clamped = (geometric.detach() <= clamp_min + 1e-6).float().mean()
            unclamped_per_query = (geometric.detach() > clamp_min + 1e-6).float().sum(-1).squeeze(1)  # [N]
            nearest = distance.detach().amin(dim=-1)  # [B,N]
            entry = {
                "gamma": gamma,
                "radius_p01": float(radii.min()), "radius_p50": float(radii.median()),
                "radius_p95": float(torch.sort(radii.flatten()).values[int(0.95*(radii.numel()-1))]),
                "sigma0_times_r_p50": float((float(opt.locusgs_sigma0) * radii).median()),
                "D_all_pairs": pct(distance),
                "D_nearest_ray": pct(nearest),
                "raw_bias": pct(geometric),
                "gamma_bias": pct(gamma * geometric),
                "clamp_threshold_p50": float(threshold.median()),
                "manual_clamped_fraction": float(manual_clamped),
                "actual_clamped_fraction": float(actual_clamped),
                "clamp_check_abs_diff": abs(float(manual_clamped) - float(actual_clamped)),
                "unclamped_rays_per_query": pct(unclamped_per_query),
                "query_zero_unclamped_fraction": float((unclamped_per_query == 0).float().mean()),
                "query_lt_4": float((unclamped_per_query < 4).float().mean()),
                "query_lt_8": float((unclamped_per_query < 8).float().mean()),
                "query_lt_16": float((unclamped_per_query < 16).float().mean()),
                "query_lt_32": float((unclamped_per_query < 32).float().mean()),
                "query_lt_64": float((unclamped_per_query < 64).float().mean()),
                "query_lt_128": float((unclamped_per_query < 128).float().mean()),
                "query_lt_256": float((unclamped_per_query < 256).float().mean()),
            }
            # content logits from the same tokens / keys the layer uses
            attn_mod = blk.gs_cross_attn
            normed = attn_mod.gs_token_norm(tokens)
            q = rearrange(attn_mod.q_proj(normed), "b n (h d) -> b h n d", h=num_heads)
            q = attn_mod.q_norm(q)
            content = (q @ keys.transpose(-2, -1)) * scale  # [B,H,N,P]
            entry["content_logits"] = pct(content)
            entry["std_ratio"] = float(gamma * geometric.detach().std() / content.std())
            entry["meanabs_over_content_std"] = float(
                (gamma * geometric.detach()).abs().mean() / content.std()
            )
            ratio_samples.append((layer, float(content.std()), float(geometric.detach().std()), gamma))
            if layer in (1, 6, 12):
                entry["attention_content_only"] = attn_stats(content)
                entry["attention_content_plus_bias"] = attn_stats(content + gamma * geometric)
                logit_layers[layer] = entry
            layers[f"layer{layer}"] = entry

            # continue the decoder loop exactly like the model
            anchor_pe = (decoder.pe_mlp if decoder.pe_mlp is not None else decoder.pe_mlps[idx])(
                __import__("tokengs.models.locusgs_recon", fromlist=["sinusoidal_positional_encoding"])
                .sinusoidal_positional_encoding(mu, int(opt.locusgs_pe_num_freqs))
            )
            attn_bias = gamma * geometric
            tokens = tokens + blk.gs_cross_attn_scale(
                blk.gs_cross_attn(tokens, keys, latent.values, attn_bias=attn_bias)
            )
            tokens = tokens + blk.gs_self_attn_scale(blk.gs_self_attn(tokens + anchor_pe))
            tokens = tokens + blk.mlp_scale(blk.mlp(tokens))
            mu = mu + decoder.refine_mu[idx](tokens)
            rho = rho + decoder.refine_rho[idx](tokens).squeeze(-1)
        report["layers"] = layers
        report["logit_layers"] = logit_layers

    # ---- gamma selection -------------------------------------------------- #
    mean_content_std = sum(c for _, c, _, _ in ratio_samples) / len(ratio_samples)
    mean_bias_std = sum(b for _, _, b, _ in ratio_samples) / len(ratio_samples)
    current_gamma = ratio_samples[0][3]
    print()
    print(f"[audit] mean std(content logits) = {mean_content_std:.4f} | mean std(raw b) = {mean_bias_std:.4f}")
    print(f"[audit] current gamma = {current_gamma:.6f} -> ratio = {current_gamma*mean_bias_std/mean_content_std:.4f}")
    table = []
    for gamma_raw in (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0):
        g = float(F.softplus(torch.tensor(gamma_raw)))
        ratio = g * mean_bias_std / mean_content_std
        table.append({"gamma_raw": gamma_raw, "gamma": g, "ratio": ratio,
                      "abs_diff_from_0.5": abs(ratio - 0.5)})
    best = min(table, key=lambda r: r["abs_diff_from_0.5"])
    report["gamma_table"] = table
    report["gamma_choice"] = best
    report["mean_content_std"] = mean_content_std
    report["mean_raw_bias_std"] = mean_bias_std
    print(f"{'gamma_raw':>10} | {'gamma':>10} | {'std(gamma*b)/std(content)':>26}")
    for row in table:
        print(f"{row['gamma_raw']:>10.1f} | {row['gamma']:>10.5f} | {row['ratio']:>26.4f}")
    print(f"[audit] closest to 0.5 -> gamma_raw={best['gamma_raw']} gamma={best['gamma']:.5f} ratio={best['ratio']:.4f}")

    # ---- verdict ---------------------------------------------------------- #
    l6 = layers["layer6"]
    median_unclamped = l6["unclamped_rays_per_query"]["p500"]
    zero_frac = l6["query_zero_unclamped_fraction"]
    ratio_now = l6["std_ratio"]
    verdict = {
        "median_unclamped_rays_layer6": median_unclamped,
        "zero_unclamped_fraction_layer6": zero_frac,
        "ratio_layer6": ratio_now,
        "pass_geometry": bool(median_unclamped >= 32 and zero_frac < 0.05),
        "pass_dominance": bool(ratio_now > 1.0),
    }
    verdict["PASS"] = bool(verdict["pass_geometry"] and verdict["pass_dominance"])
    report["verdict"] = verdict
    print()
    print(f"[audit] layer6: median unclamped rays/query = {median_unclamped:.1f}, "
          f"zero-unclamped queries = {zero_frac*100:.2f}%, ratio = {ratio_now:.4f}")
    print(f"[audit] VERDICT: {'PASS - gamma experiment justified' if verdict['PASS'] else 'FAIL/STOP'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[wrote] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
