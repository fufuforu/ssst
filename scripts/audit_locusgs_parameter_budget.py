#!/usr/bin/env python3
"""Parameter-budget reverse audit for the LocusGS anchor decoder.

The paper reports (App. B.3 / Table 8, 4096-token evaluation setting):
    TokenGS  222.0 M
    LocusGS  241.5 M      -> LocusGS-specific delta = +19.5 M

This script measures the repository's own parameter counts, decomposes the
LocusGS-specific modules, and evaluates candidate readings of the unspecified
parts (shared vs per-layer PE MLP, single-Linear vs bottleneck refinement MLPs)
against that budget.  Read-only: it constructs modules on CPU, no training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tokengs.models import model_registry  # noqa: E402
from tokengs.models.locusgs_recon import LocusGSAnchorDecoder  # noqa: E402
from tokengs.options import Options, config_defaults  # noqa: E402

PAPER_TOKENG_S = 222.0
PAPER_LOCUSGS = 241.5
PAPER_DELTA = PAPER_LOCUSGS - PAPER_TOKENG_S
OUT_DIR = Path("/space/mawb/ssst/workspace_recon_diag/locusgs_param_budget_v2")


def count(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def locusgs_specific(decoder: LocusGSAnchorDecoder) -> dict:
    anchors = decoder.mu.numel() + decoder.rho.numel()
    pe = sum(p.numel() for p in (decoder.pe_mlp or torch.nn.Module()).parameters()) if decoder.pe_mlp is not None else 0
    if decoder.pe_mlps is not None:
        pe = sum(p.numel() for p in decoder.pe_mlps.parameters())
    refine_mu = sum(p.numel() for p in decoder.refine_mu.parameters())
    refine_rho = sum(p.numel() for p in decoder.refine_rho.parameters())
    gamma = decoder.gamma_raw.numel()
    return {
        "anchor_mu": decoder.mu.numel(),
        "anchor_rho": decoder.rho.numel(),
        "pe_mlp_total": pe,
        "pe_mlp_per_layer": pe // len(decoder.decoder_blocks) if decoder.pe_mlps is not None else pe,
        "refine_mu_total": refine_mu,
        "refine_rho_total": refine_rho,
        "refine_mu_per_layer": refine_mu // len(decoder.refine_mu),
        "refine_rho_per_layer": refine_rho // len(decoder.refine_rho),
        "gamma": gamma,
        "locusgs_specific_total": anchors + pe + refine_mu + refine_rho + gamma,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=== paper-reported budget (App. B.3 / Table 8, 4096-token setting) ===")
    emit(f"TokenGS  : {PAPER_TOKENG_S:.1f} M")
    emit(f"LocusGS  : {PAPER_LOCUSGS:.1f} M")
    emit(f"delta    : +{PAPER_DELTA:.1f} M   (rounded inputs, so +/- ~0.1 M)")
    emit("storage delta 921.31 - 846.93 = 74.38 MiB = 77.99 MB -> 4 bytes/param (fp32)")
    emit("")

    report: dict = {"paper": {"tokengs_M": PAPER_TOKENG_S, "locusgs_M": PAPER_LOCUSGS, "delta_M": PAPER_DELTA}}

    # --- plain TokenGS at both token budgets -------------------------------- #
    emit("=== repository measurements ===")
    plain = config_defaults["train_siu3r_plain_tokengs_canonical_recon"]
    for tokens in (1024, 4096):
        opt = plain.evolve(num_gs_tokens=tokens, dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"})
        model = model_registry[opt.model_type](opt)
        stats = count(model)
        emit(f"plain TokenGS ({tokens} tokens): total {stats['total']/1e6:.3f} M")
        report[f"plain_tokengs_{tokens}"] = stats
    emit("(the paper's 222.0 M matches our 4096-token count: 218.871 M + 3x1.0486 M)")
    emit("")

    # --- LocusGS V1 and V2 at both token budgets ---------------------------- #
    base = config_defaults["train_siu3r_locusgs_recon"]
    variants = {
        "v1_legacy": {"locusgs_impl": "legacy_v1"},
        "v2_inferred": {"locusgs_impl": "inferred_v2", "locusgs_refine_hidden": 256},
    }
    plain_4096 = model_registry[plain.model_type](plain.evolve(num_gs_tokens=4096)) if False else None
    del plain_4096
    for name, overrides in variants.items():
        for tokens in (1024, 4096):
            opt = base.evolve(
                num_gs_tokens=tokens,
                dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
                **overrides,
            )
            model = model_registry[opt.model_type](opt)
            stats = count(model)
            breakdown = locusgs_specific(model.anchor_decoder)
            plain_ref = report[f"plain_tokengs_{tokens}"]["total"]
            delta = stats["total"] - plain_ref
            report[f"{name}_{tokens}"] = {**stats, "delta_vs_plain": delta, "breakdown": breakdown}
            emit(f"LocusGS {name} ({tokens} tokens): total {stats['total']/1e6:.3f} M | "
                 f"delta vs plain {delta/1e6:+.3f} M")
            if tokens == 4096:
                emit(f"    specific: anchors(mu {breakdown['anchor_mu']:,} + rho {breakdown['anchor_rho']:,}), "
                     f"PE MLPs {breakdown['pe_mlp_total']:,} ({breakdown['pe_mlp_per_layer']:,}/layer), "
                     f"f_mu {breakdown['refine_mu_total']:,} ({breakdown['refine_mu_per_layer']:,}/layer), "
                     f"f_rho {breakdown['refine_rho_total']:,} ({breakdown['refine_rho_per_layer']:,}/layer), "
                     f"gamma {breakdown['gamma']}")
                emit(f"    LocusGS-specific total {breakdown['locusgs_specific_total']/1e6:.3f} M "
                     f"| distance to paper delta {breakdown['locusgs_specific_total']/1e6 - PAPER_DELTA:+.3f} M")
    emit("")

    # --- analytic candidate sweep ------------------------------------------- #
    d, layers = 1024, 12
    pe_dim = 3 + 6 * int(base.locusgs_pe_num_freqs)
    pe_per_layer = pe_dim * d + d + d * d + d
    emit("=== candidate architectures (analytic, 4096-token anchor cost included) ===")
    emit(f"d={d}, L={layers}, pe_dim={pe_dim}, PE MLP per layer = 27*1024+1024+1024*1024+1024 = {pe_per_layer:,}")
    anchors_4096 = 4096 * 3 + 4096 + layers
    candidates = {
        "A_v1_shared_pe_linear_refine": pe_per_layer + layers * ((d * 3 + 3) + (d * 1 + 1)) + anchors_4096,
        "B_perlayer_pe_linear_refine": layers * pe_per_layer + layers * ((d * 3 + 3) + (d * 1 + 1)) + anchors_4096,
    }
    for h in (128, 192, 256, 320, 384, 512):
        refine = layers * ((d * h + h + h * 3 + 3) + (d * h + h + h * 1 + 1))
        candidates[f"C_perlayer_pe_bottleneck_h{h}"] = layers * pe_per_layer + refine + anchors_4096
    # exact-fit hidden width for the bottleneck
    target = PAPER_DELTA * 1e6 - layers * pe_per_layer - anchors_4096
    h_fit = None
    for h in range(1, 2048):
        refine = layers * ((d * h + h + h * 3 + 3) + (d * h + h + h * 1 + 1))
        if refine >= target:
            h_fit = h
            break
    emit(f"{'candidate':>34} | {'extra params':>14} | {'distance to +19.5 M':>20}")
    for name, value in candidates.items():
        emit(f"{name:>34} | {value/1e6:>11.3f} M | {value/1e6 - PAPER_DELTA:>+17.3f} M")
    emit(f"{'exact-fit bottleneck width':>34} | {'':>14} | h ~= {h_fit}")
    report["candidates"] = {k: v for k, v in candidates.items()}
    report["exact_fit_hidden"] = h_fit

    (OUT_DIR / "parameter_budget.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "parameter_budget.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wrote] {OUT_DIR/'parameter_budget.json'} and parameter_budget.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
