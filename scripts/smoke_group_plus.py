#!/usr/bin/env python3
"""Pre-run smoke for the G0+ background-slot supervision (single variable).

Checks:
1. G0+ starts from the *same* random state as G0 (loaded from G0's step-0
   checkpoint, block hashes equal) and the first forward before any update is
   bit-identical to G0 (RGB, depth, alpha, Gaussians, group logits/slot probs);
2. with the background term disabled the total loss equals G0's loss exactly;
3. with it enabled: mask/alpha conservation, `L_bg` finite, and the stuff/thing
   targets independently re-derived from the batch (including the 255 / uncovered
   exclusions) match the model's own pixel counts;
4. gradient direction on real data: stuff-only pixels push the background
   assignment up, thing-only pixels push it down, and the gradient reaches the
   group parameters (not only the background bias).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.group_eval_v2 import build_val_entries  # noqa: E402
from scripts.smoke_group_locusgs import build_batch  # noqa: E402
from scripts.train_group_locusgs import parameter_blocks  # noqa: E402
from scripts.train_object_locusgs import sha256_file, sha256_state  # noqa: E402


def build(preset: str, *, bg: bool, seed: int, init_from: str | None):
    torch.manual_seed(seed)
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm="g0", group_bg_supervision=bg, group_bg_loss_weight=1.0,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).cuda()
    provenance = "fresh seed %d" % seed
    if init_from:
        payload = torch.load(Path(init_from) / "model.pt", map_location="cpu",
                             weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        provenance = f"loaded {init_from}"
    return opt, model, provenance


def forward_all(model, batch, opt):
    model_input, _ = split_data(batch, opt)
    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                intrinsics=batch["intrinsics_all"])
    out = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder), render_decoder_input=decoder
    )
    return out, decoder


def mask_metrics(model, batch):
    opt = model.opt
    model_input, _ = split_data(batch, opt)
    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                intrinsics=batch["intrinsics_all"])
    out = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder), render_decoder_input=decoder
    )
    mask_decoder = decoder.select_batch(slice(None), slice(0, int(opt.num_input_views)))
    rendered = model.render_group_masks(out["gaussians"], model.layer10_group["slot_prob"],
                                        mask_decoder)
    return out, decoder, rendered


def background_gradient(model, batch, *, keep: str) -> dict:
    """Gradient of L_bg when only stuff or only thing pixels carry a target."""
    import torch.nn.functional as F

    altered = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    semantic = altered["semantic_label_all"].clone()
    instance = altered["instance_label_all"].clone()
    views = int(model.opt.num_input_views)
    if keep == "stuff":
        thing = (semantic[:, :views] >= 2) & (semantic[:, :views] < 20)
        semantic[:, :views][thing] = 255
    else:
        stuff = (semantic[:, :views] == 0) | (semantic[:, :views] == 1)
        semantic[:, :views][stuff] = 255
    altered["semantic_label_all"] = semantic
    altered["instance_label_all"] = instance

    model.zero_grad(set_to_none=True)
    _, decoder, rendered = mask_metrics(model, altered)
    loss, stats = model.background_slot_loss(altered, rendered, tuple(range(views)))
    loss.backward()
    bias_grad = float(model.groups.background_bias.grad) if (
        model.groups.background_bias.grad is not None
    ) else None
    group_grad = sum(
        float(p.grad.detach().float().pow(2).sum())
        for name, p in model.groups.named_parameters()
        if p.grad is not None and not name.startswith("background_bias")
    )
    geometry_grad = sum(
        float(p.grad.detach().float().pow(2).sum())
        for name, p in model.named_parameters()
        if p.grad is not None and name.startswith(("anchor_decoder", "activation_head"))
    )
    del F
    model.zero_grad(set_to_none=True)
    return {
        "loss_bg": float(loss.detach()),
        "pixels_stuff": stats["bg_pixels_stuff"],
        "pixels_thing": stats["bg_pixels_thing"],
        "background_bias_grad": bias_grad,
        "gradient_direction": (
            "increase background" if bias_grad is not None and bias_grad < 0
            else "decrease background" if bias_grad is not None else None
        ),
        "group_parameter_grad_sq": group_grad,
        "geometry_grad_sq": geometry_grad,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="workspace_group_plus/smoke")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--init-from", default="workspace_group_locusgs/arm_g0/ckpt_step0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    results: dict = {"plan_sha256": sha256_file(Path(args.plan))}

    opt_g0, g0, _ = build(args.preset, bg=False, seed=args.seed, init_from=None)
    opt_plus, plus, provenance = build(args.preset, bg=True, seed=args.seed,
                                       init_from=args.init_from)
    opt_off, plus_off, _ = build(args.preset, bg=False, seed=args.seed,
                                 init_from=args.init_from)
    entry, batch = build_batch(opt_g0, plan, 1, "cuda")
    print(f"[smoke+] step-1 window {entry}", flush=True)

    blocks = parameter_blocks(g0)
    state_g0, state_plus = g0.state_dict(), plus.state_dict()
    hashes = {
        name: (sha256_state(state_g0, keys), sha256_state(state_plus, keys))
        for name, keys in blocks.items()
    }
    results["01_initialisation"] = {
        "g0_provenance": "fresh seed 42",
        "g0plus_provenance": provenance,
        "block_hashes_g0": {k: v[0] for k, v in hashes.items()},
        "block_hashes_g0plus": {k: v[1] for k, v in hashes.items()},
        "identical": {k: v[0] == v[1] for k, v in hashes.items()},
        "status": "PASS" if all(v[0] == v[1] for v in hashes.values()) else "FAIL",
    }

    with torch.no_grad():
        out_g0, _ = forward_all(g0, batch, opt_g0)
        out_plus, _ = forward_all(plus, batch, opt_plus)
    parity = {
        "rgb": float((out_g0["render"]["images_pred"] - out_plus["render"]["images_pred"]).abs().max()),
        "depth": float((out_g0["render"]["depths_pred"] - out_plus["render"]["depths_pred"]).abs().max()),
        "alpha": float((out_g0["render"]["alphas_pred"] - out_plus["render"]["alphas_pred"]).abs().max()),
        "gaussians": float((out_g0["gaussians"] - out_plus["gaussians"]).abs().max()),
        "slot_logits": float(
            (g0.layer10_group["slot_logits"] - plus.layer10_group["slot_logits"]).abs().max()
        ),
        "group_class_logits": float(
            (g0.layer10_group["class_logits"] - plus.layer10_group["class_logits"]).abs().max()
        ),
    }
    parity["status"] = "PASS" if all(v == 0.0 for v in parity.values()) else "FAIL"
    print(f"[smoke+] 02_first_forward_parity [{'PASS' if parity['status'] == 'PASS' else 'FAIL'}] "
          f"{json.dumps(parity)}", flush=True)
    results["02_first_forward_parity"] = parity

    with torch.no_grad():
        _, metrics_g0 = g0.step_loss(batch, step=1, phase="train")
        _, metrics_off = plus_off.step_loss(batch, step=1, phase="train")
        _, metrics_plus = plus.step_loss(batch, step=1, phase="train")
    loss_gap = abs(float(metrics_g0["loss"]) - float(metrics_off["loss"]))
    expected = (
        float(metrics_off["loss"])
        + float(metrics_off["instance_weight"]) * 1.0 * float(metrics_plus["loss_bg"])
    )
    results["03_loss_terms"] = {
        "loss_g0": float(metrics_g0["loss"]),
        "loss_g0plus_with_term_disabled": float(metrics_off["loss"]),
        "gap_disabled_vs_g0": loss_gap,
        "loss_g0plus": float(metrics_plus["loss"]),
        "expected_from_identity": expected,
        "identity_gap": abs(expected - float(metrics_plus["loss"])),
        "loss_inst_original_g0": float(metrics_g0["loss_inst"]),
        "loss_inst_original_g0plus": float(metrics_plus["loss_inst"]),
        "loss_inst_total_g0plus": float(metrics_plus["loss_inst_total"]),
        "loss_bg": float(metrics_plus["loss_bg"]),
        "loss_bg_stuff": float(metrics_plus["loss_bg_stuff"]),
        "loss_bg_thing": float(metrics_plus["loss_bg_thing"]),
        "bg_pixels_stuff": float(metrics_plus["bg_pixels_stuff"]),
        "bg_pixels_thing": float(metrics_plus["bg_pixels_thing"]),
        "bg_prob_stuff_mean": float(metrics_plus["bg_prob_stuff_mean"]),
        "bg_prob_thing_mean": float(metrics_plus["bg_prob_thing_mean"]),
        "mask_alpha_max_error": float(metrics_plus["mask_alpha_max_error"]),
        "all_finite": bool(math.isfinite(float(metrics_plus["loss"]))
                           and math.isfinite(float(metrics_plus["loss_bg"]))),
    }

    # independent re-derivation of the pixel targets
    with torch.no_grad():
        _, decoder, rendered = mask_metrics(plus, batch)
        views = int(opt_plus.num_input_views)
        alpha = rendered["alpha"][:, :views]
        semantic = batch["semantic_label_all"][:, :views].long()
        instance = batch["instance_label_all"][:, :views].long()
        covered = alpha[:, :, 0] > 0.5
        stuff = ((semantic == 0) | (semantic == 1)) & covered
        thing = (semantic >= 2) & (semantic < 20) & (instance > 0) & covered
        ignored = (~covered) | (~(stuff | thing))
        counts = {
            "stuff_expected": int(stuff.sum()),
            "thing_expected": int(thing.sum()),
            "ignored_expected": int(ignored.sum()),
            "ignore_255_expected": int((semantic == 255).sum()),
            "stuff_255": int(((semantic == 255) & stuff).sum()),
            "thing_255": int(((semantic == 255) & thing).sum()),
            "stuff_uncovered": int(((semantic <= 1) & (~covered)).sum()),
            "thing_no_instance_expected": int(
                ((semantic >= 2) & (semantic < 20) & (instance <= 0) & covered).sum()
            ),
            "thing_no_instance_in_thing": int(
                (((semantic >= 2) & (semantic < 20) & (instance <= 0)) & thing).sum()
            ),
        }
    counts["model_counts_match"] = (
        int(metrics_plus["bg_pixels_stuff"]) == counts["stuff_expected"]
        and int(metrics_plus["bg_pixels_thing"]) == counts["thing_expected"]
    )
    counts["status"] = "PASS" if (
        counts["model_counts_match"] and counts["stuff_255"] == 0
        and counts["thing_255"] == 0 and counts["stuff_uncovered"] == 0
        and counts["thing_no_instance_in_thing"] == 0
        and counts["stuff_expected"] > 0 and counts["thing_expected"] > 0
    ) else "FAIL"
    print(f"[smoke+] 03_loss_terms/04_targets [{counts['status']}] {json.dumps(counts)}", flush=True)
    results["03_loss_terms"]["status"] = "PASS" if (
        loss_gap < 1e-6
        and results["03_loss_terms"]["identity_gap"] < 1e-5
        and results["03_loss_terms"]["all_finite"]
        and results["03_loss_terms"]["mask_alpha_max_error"] < 1e-4
    ) else "FAIL"
    results["04_background_targets"] = counts

    stuff_grad = background_gradient(plus, batch, keep="stuff")
    thing_grad = background_gradient(plus, batch, keep="thing")
    results["05_background_gradient"] = {
        "stuff_only": stuff_grad,
        "thing_only": thing_grad,
        "status": "PASS" if (
            stuff_grad["background_bias_grad"] is not None
            and thing_grad["background_bias_grad"] is not None
            and stuff_grad["background_bias_grad"] < 0 < thing_grad["background_bias_grad"]
            and stuff_grad["group_parameter_grad_sq"] > 0
            and thing_grad["group_parameter_grad_sq"] > 0
        ) else "FAIL",
    }
    print(f"[smoke+] 05_background_gradient [{results['05_background_gradient']['status']}] "
          f"stuff {stuff_grad['background_bias_grad']:+.3e} "
          f"thing {thing_grad['background_bias_grad']:+.3e}", flush=True)

    (out_dir / "smoke_report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    failures = [k for k, v in results.items() if isinstance(v, dict)
                and v.get("status") == "FAIL"]
    print(f"[smoke+] wrote {out_dir / 'smoke_report.json'}; failures: {failures or 'none'}",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
