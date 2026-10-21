#!/usr/bin/env python3
"""One measurement standard for per-token Gaussian locality.

For every model, on the same fixed batch:

* **A** distance from each Gaussian to the centroid of the Gaussians generated
  by its own token (defined identically for plain TokenGS and LocusGS);
* **B** (LocusGS) distance from each Gaussian to its *refined* anchor;
* **C** (LocusGS) distance from the token's Gaussian centroid to the refined anchor;
* **D** (LocusGS) anchor positions and the anchor update vs initialisation;
* **E** collapse check - per-token span about the centroid, and the fraction of
  tokens whose 64 Gaussians have collapsed;
* **F** the same A/B/C restricted to the Gaussians that actually contribute to a
  rendered image (projected inside the frame in at least one view and opacity
  above a threshold), plus the selected fraction.

All distances are reported raw and divided by one shared scene scale (the
visible-surface RMS radius of a fixed reference reconstruction).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402
from scripts.analyze_gaussian_locality import visible_surface_radius  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.scannet_raw_recon import (  # noqa: E402
    DEFAULT_SCANS_ROOT,
    ScanNetRawReconProvider,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def build(preset, context, novel, device):
    opt = config_defaults[preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        num_input_views=len(context), num_views=len(context) + len(novel),
        batch_size=1, num_workers=0, seed=42,
    )
    return model_registry[opt.model_type](opt).to(device).eval(), opt


def load_into(model, ckpt_dir):
    state = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    return model


def forward(model, opt, batch):
    with torch.no_grad():
        mi, _ = split_data(batch, opt)
        dec = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
        return model.forward_reconstruction_only(
            ModelInput(mi.encoder, dec), render_decoder_input=dec)


def dist_stats(d: torch.Tensor, scale: float | None) -> dict:
    f = d.flatten()
    q = lambda p: float(f.quantile(p))
    out = {"mean": float(f.mean()), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99),
           "max": float(f.max())}
    if scale:
        out.update({f"{k}_over_scale": v / scale for k, v in
                    (("mean", out["mean"]), ("p50", out["p50"]), ("p90", out["p90"]),
                     ("p99", out["p99"]))})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint dir with model.pt")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--context", type=int, nargs="+", default=[654, 664])
    parser.add_argument("--novel", type=int, nargs="+", default=[655, 659])
    parser.add_argument("--ref-model", required=True)
    parser.add_argument("--ref-preset", default="train_siu3r_plain_tokengs_canonical_recon")
    parser.add_argument("--opacity-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    model, opt = build(args.preset, args.context, args.novel, device)
    load_into(model, args.model)

    provider = ScanNetRawReconProvider(
        opt, root=str(DEFAULT_SCANS_ROOT), scene=args.scene,
        context_frame_ids=tuple(args.context), novel_frame_ids=tuple(args.novel),
        training=False)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in default_collate([provider[0]]).items()}
    n_in = int(opt.num_input_views)
    img_size = (int(opt.img_size[0]), int(opt.img_size[1]))

    out = forward(model, opt, batch)
    render = out["render"]

    ref_model, ref_opt = build(args.ref_preset, args.context, args.novel, device)
    load_into(ref_model, args.ref_model)   # the reference must be its trained weights
    ref_render = forward(ref_model, ref_opt, batch)["render"]
    scene_scale, _ = visible_surface_radius(ref_render, batch, n_in, img_size)
    del ref_model
    print(f"[loc] shared scene scale (visible-surface RMS radius): {scene_scale:.4f}")

    g = out["gaussians"][0].float()                       # [T*P, 14]
    tokens = int(opt.num_gs_tokens)
    per_token = g.shape[0] // tokens
    centers = g[:, 0:3].reshape(tokens, per_token, 3)
    opacity = g[:, 3].reshape(tokens, per_token)
    centroid = centers.mean(dim=1)                        # [T,3]
    d_own = (centers - centroid.unsqueeze(1)).norm(dim=-1)
    per_token_span = d_own.pow(2).mean(dim=1).sqrt()

    # contributing Gaussians: inside the frame in >=1 view and opacity > threshold
    m2d = render["means2d_pred"][0].float()               # [V, T*P, 2]
    W, H = img_size[1], img_size[0]
    inside = ((m2d[..., 0] >= 0) & (m2d[..., 0] <= W) &
              (m2d[..., 1] >= 0) & (m2d[..., 1] <= H)).any(dim=0)   # [T*P]
    contrib = (inside & (g[:, 3] > args.opacity_threshold)).reshape(tokens, per_token)
    frac_contrib = float(contrib.float().mean())

    result = {
        "model": opt.model_type,
        "preset": args.preset,
        "checkpoint": str(args.model),
        "num_tokens": tokens,
        "gaussians_per_token": per_token,
        "scene_scale": scene_scale,
        "A_gaussian_to_own_token_centroid": dist_stats(d_own, scene_scale),
        "E_collapse": {
            "per_token_span_mean": float(per_token_span.mean()),
            "per_token_span_p50": float(per_token_span.median()),
            "per_token_span_p90": float(per_token_span.quantile(0.9)),
            "fraction_tokens_span_lt_1e-4": float((per_token_span < 1e-4).float().mean()),
            "fraction_tokens_span_lt_1e-3": float((per_token_span < 1e-3).float().mean()),
        },
        "F_contributing": {
            "fraction_contributing": frac_contrib,
            "A_gaussian_to_own_token_centroid": dist_stats(d_own[contrib], scene_scale),
        },
        "alpha_coverage_gt_05": float((render["alphas_pred"][0].float() > 0.5).float().mean()),
    }

    if hasattr(model, "anchor_decoder"):
        ad = model.anchor_decoder
        anchor = out["anchors"][0].detach().float()
        d_anchor = (centers - anchor.unsqueeze(1)).norm(dim=-1)
        d_centroid = (centroid - anchor).norm(dim=-1)
        dr = getattr(model.activation_head, "last_decode_radius", None)
        result.update({
            "B_gaussian_to_refined_anchor": dist_stats(d_anchor, scene_scale),
            "C_centroid_to_refined_anchor": dist_stats(d_centroid, scene_scale),
            "D_anchor": {
                "init_mu_abs_max": float(ad.mu.detach().float().abs().max()),
                "init_mu_std": float(ad.mu.detach().float().std(dim=0).mean()),
                "final_mu_abs_max": float(anchor.abs().max()),
                "final_mu_std": float(anchor.std(dim=0).mean()),
                "final_mu_z_mean": float(anchor[:, 2].mean()),
                "update_mean": float((anchor - ad.mu.detach().float()).norm(dim=-1).mean()),
                "update_max": float((anchor - ad.mu.detach().float()).norm(dim=-1).max()),
                "update_over_scale": float((anchor - ad.mu.detach().float()).norm(dim=-1).mean()) / scene_scale,
            },
            "decode_radius": {
                "mean": float(dr.float().mean()) if dr is not None else None,
                "min": float(dr.float().min()) if dr is not None else None,
                "max": float(dr.float().max()) if dr is not None else None,
                "learned_radius_mean": float(out["radii"][0].float().mean()),
                "learned_radius_over_init": float(out["radii"][0].float().mean() / opt.locusgs_radius_init),
            },
            "F_contributing_anchor": {
                "B_gaussian_to_refined_anchor": dist_stats(d_anchor[contrib], scene_scale),
            },
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    a = result["A_gaussian_to_own_token_centroid"]
    e = result["E_collapse"]
    print(f"[loc] A gs->own-centroid p50 {a['p50']:.4f} p90 {a['p90']:.4f} "
          f"({a['p50_over_scale']:.2f} / {a['p90_over_scale']:.2f} scale)")
    print(f"[loc] E span p50 {e['per_token_span_p50']:.5f} collapse<1e-3 {e['fraction_tokens_span_lt_1e-3']:.3f}")
    print(f"[loc] F contributing {result['F_contributing']['fraction_contributing']:.3f} "
          f"p50 {result['F_contributing']['A_gaussian_to_own_token_centroid']['p50']:.4f}")
    if "B_gaussian_to_refined_anchor" in result:
        b = result["B_gaussian_to_refined_anchor"]
        c = result["C_centroid_to_refined_anchor"]
        d = result["D_anchor"]
        print(f"[loc] B gs->anchor p50 {b['p50']:.4f} p90 {b['p90']:.4f} "
              f"({b['p50_over_scale']:.2f} / {b['p90_over_scale']:.2f} scale)")
        print(f"[loc] C centroid->anchor p50 {c['p50']:.4f} (over scale {c['p50_over_scale']:.2f})")
        print(f"[loc] D anchor update mean {d['update_mean']:.3f} over scale {d['update_over_scale']:.2f}"
              f" | z mean {d['final_mu_z_mean']:.3f}")
        r = result["decode_radius"]
        print(f"[loc] decode radius {r['mean']}/{r['min']}/{r['max']} "
              f"| learned {r['learned_radius_mean']:.3f} ({r['learned_radius_over_init']:.2f}x init)")
    print(f"[loc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
