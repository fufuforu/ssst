#!/usr/bin/env python3
"""Pre-run smoke suite for the from-scratch G0/G1 group experiment.

Checks (all results go to `smoke_report.json`):

1. both arms are built from random initialisation with the same seed and their
   shared reconstruction / attribute / group parameters hash identically;
2. before the first parameter update the G0 and G1 forwards are identical
   (RGB, depth, alpha, Gaussian centres, anchors, radii, decode radius) and the
   frozen decode radius is exactly 0.15;
3. label conventions (class 0 valid, 255 ignored, stuff vs thing, instance keys);
4. mask/alpha conservation `sum group masks + background = alpha`;
5. the understanding loss is non-zero from step 1 and its gradient reaches the
   group head, the semantic head, the shared decoder and (G1) the feedback
   branch; the G1 gate is not a dead end;
6. only the two context images feed the model (novel-GT tamper leaves every
   output bit-identical);
7. resumable checkpoint round-trip (model + optimizer + RNG + step);
8. informational GT-free read-out at initialisation (no threshold).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
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

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.models.object_locusgs import (  # noqa: E402
    IGNORE_SEMANTIC,
    SEMANTIC_CLASS_COUNT,
    instance_keys,
    semantic_supervision_mask,
    thing_instance_mask,
)
from tokengs.options import config_defaults  # noqa: E402
from scripts.group_locusgs_eval import (  # noqa: E402
    forward_group,
    group_view_predictions,
    summarise_group,
    build_val_entries,
    evaluate_group_entry,
)
from scripts.train_group_locusgs import parameter_blocks  # noqa: E402
from scripts.train_object_locusgs import (  # noqa: E402
    save_checkpoint,
    sha256_file,
    sha256_state,
)

SOURCE_PLAN = Path("object_locusgs/plan_6000.json")


def build_arm(preset: str, arm: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm=arm, batch_size=1, num_workers=0, num_input_views=2,
        num_views=4, dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).cuda()
    return opt, model


def build_batch(opt, plan, step: int, device):
    entry = plan["entries"][step - 1]
    provider = SIU3RProcessedProvider(
        opt, root="/space/mawb/SIU3R/data/scannet/train",
        subset=plan["train_scenes"], training=True, rank=0,
    )
    provider.pin_pair(
        scene_id=entry["scene"], context_frame_ids=entry["context"],
        novel_frame_ids=entry["novel"], pair_iou=entry["pair_iou"],
    )
    index = {p.name: i for i, p in enumerate(provider.dataset.sample_list)}[entry["scene"]]
    batch = default_collate([provider[index]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return entry, batch


def max_abs(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def forward_plain(model, batch, opt):
    model_input, _ = split_data(batch, opt)
    decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    output = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
    )
    return output, decoder_input


def grad_norms(model, prefixes) -> dict:
    out: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        for prefix in prefixes:
            if name.startswith(prefix):
                out[prefix] = out.get(prefix, 0.0) + float(
                    parameter.grad.detach().float().pow(2).sum()
                )
                break
    return {key: math.sqrt(value) for key, value in out.items()}


def understanding_backward(model, batch, opt, *, gate=None):
    """Backprop only lambda * [0.05 L_instance + 0.05 L_semantic] with lambda = 1."""
    model.train()
    model.zero_grad(set_to_none=True)
    saved = None
    if gate is not None:
        saved = float(model.feedback.gate.detach())
        with torch.no_grad():
            model.feedback.gate.fill_(gate)
    model_input, _ = split_data(batch, opt)
    decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    states, _ = model._decode(ModelInput(model_input.encoder, decoder_input), decoder_input)
    final = states[-1]
    gaussians = model.activation_head(final["tokens"], final["mu"], final["radii"])
    instance = model.group_loss_terms(batch, gaussians, decoder_input, (0, 1))
    semantic, semantic_stats, _ = model.semantic_loss_terms(
        batch, final["tokens"], gaussians, decoder_input
    )
    loss = 0.05 * instance["loss"] + 0.05 * semantic
    loss.backward()
    prefixes = [
        "groups.queries", "groups.background_bias", "groups.objectness", "groups.semantic",
        "groups.cross_attn", "groups.spatial_proj", "groups.token_proj",
        "attributes.semantic",
        "feedback.norm", "feedback.proj", "feedback.gate",
        "enc_dec_backbone.decoder_blocks.11", "enc_dec_backbone.decoder_blocks.10",
        "anchor_decoder.refine_mu.11", "anchor_decoder.refine_mu.10",
        "anchor_decoder.mu", "anchor_decoder.rho", "activation_head.deconv", "gs_tokens",
    ]
    grads = grad_norms(model, prefixes)
    gate_grad = None if model.feedback.gate.grad is None else float(model.feedback.gate.grad)
    if saved is not None:
        with torch.no_grad():
            model.feedback.gate.fill_(saved)
    model.zero_grad(set_to_none=True)
    return {
        "instance_loss": float(instance["loss"].detach()),
        "semantic_loss": float(semantic.detach()),
        "loss": float(loss.detach()),
        "grad_norms": grads,
        "gate_grad": gate_grad,
        "mask_alpha_max_error": float(instance["mask_alpha_max_error"]),
        "matched_groups": float(sum(instance["matched"])),
        "thing_targets": float(instance["thing_targets"]),
        "semantic_coverage": float(semantic_stats["coverage"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="workspace_group_locusgs/smoke")
    parser.add_argument("--plan", default=str(SOURCE_PLAN))
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results: dict = {}

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    results["00_inputs"] = {
        "plan": str(plan_path), "plan_sha256": sha256_file(plan_path),
        "steps": len(plan["entries"]), "train_scenes": len(plan["train_scenes"]),
        "val_scenes": len(plan["val_scenes"]),
        "preset_init_checkpoint": config_defaults[args.preset].init_checkpoint,
    }
    if config_defaults[args.preset].init_checkpoint:
        results["00_inputs"]["status"] = "FAIL"
        print("[smoke] preset declares init_checkpoint", flush=True)
        return 1
    results["00_inputs"]["status"] = "PASS"

    opt_g0, g0 = build_arm(args.preset, "g0", args.seed)
    opt_g1, g1 = build_arm(args.preset, "g1", args.seed)
    entry, batch = build_batch(opt_g0, plan, 1, device)
    print(f"[smoke] step-1 window {entry}", flush=True)

    # ---- 1. from-scratch initialisation parity --------------------------- #
    blocks_g0 = parameter_blocks(g0)
    blocks_g1 = parameter_blocks(g1)
    state_g0, state_g1 = g0.state_dict(), g1.state_dict()
    hashes = {
        name: (sha256_state(state_g0, blocks_g0[name]), sha256_state(state_g1, blocks_g1[name]))
        for name in blocks_g0
    }
    counts = {name: int(sum(state_g0[k].numel() for k in keys))
              for name, keys in blocks_g0.items()}
    same = {name: pair[0] == pair[1] for name, pair in hashes.items()}
    results["01_initialisation"] = {
        "parameter_counts": counts,
        "hashes_g0": {name: pair[0] for name, pair in hashes.items()},
        "hashes_g1": {name: pair[1] for name, pair in hashes.items()},
        "identical": same,
        "optimizer_parameter_count": {
            "g0": sum(p.numel() for p in g0.parameters() if p.requires_grad)
                 - sum(p.numel() for p in g0.feedback.parameters()),
            "g1": sum(p.numel() for p in g1.parameters() if p.requires_grad),
        },
        "status": "PASS" if all(same[name] for name in
                                ("reconstruction", "attributes", "groups")) else "FAIL",
    }
    print(f"[smoke] 01_initialisation counts {json.dumps(counts)} "
          f"identical {json.dumps(same)} "
          f"recon_hashes {hashes['reconstruction'][0][:16]}/{hashes['reconstruction'][1][:16]} "
          f"groups {hashes['groups'][0][:16]}/{hashes['groups'][1][:16]}", flush=True)

    # ---- 2. G0/G1 forward parity before any update ----------------------- #
    with torch.no_grad():
        out_g0, decoder_input = forward_plain(g0, batch, opt_g0)
        out_g1, _ = forward_plain(g1, batch, opt_g1)
    parity = {
        "rgb": max_abs(out_g0["render"]["images_pred"], out_g1["render"]["images_pred"]),
        "depth": max_abs(out_g0["render"]["depths_pred"], out_g1["render"]["depths_pred"]),
        "alpha": max_abs(out_g0["render"]["alphas_pred"], out_g1["render"]["alphas_pred"]),
        "gaussians": max_abs(out_g0["gaussians"], out_g1["gaussians"]),
        "centres": max_abs(out_g0["gaussians"][..., :3], out_g1["gaussians"][..., :3]),
        "anchors": max_abs(out_g0["states"][-1]["mu"], out_g1["states"][-1]["mu"]),
        "radii": max_abs(out_g0["states"][-1]["radii"], out_g1["states"][-1]["radii"]),
        "decode_radius": [
            float(g0.activation_head.last_decode_radius.min()),
            float(g0.activation_head.last_decode_radius.max()),
        ],
        "group_slot_logits": max_abs(g0.layer10_group["slot_logits"], g1.layer10_group["slot_logits"]),
        "g1_gate": float(g1.feedback.gate),
        "g1_group_update_norm": float(g1.anchor_decoder.last_group_update_norm),
    }
    parity["status"] = "PASS" if (
        all(parity[k] == 0.0 for k in ("rgb", "depth", "alpha", "gaussians", "centres",
                                       "anchors", "radii", "group_slot_logits"))
        and abs(parity["decode_radius"][0] - 0.15) < 1e-6
        and abs(parity["decode_radius"][1] - 0.15) < 1e-6
    ) else "FAIL"
    results["02_forward_parity"] = parity
    print(f"[smoke] 02_forward_parity [{'PASS' if parity['status'] == 'PASS' else 'FAIL'}] "
          f"{json.dumps(parity)}", flush=True)

    # ---- 3. labels ------------------------------------------------------- #
    semantic_gt = batch["semantic_label_all"].long()
    instance_gt = batch["instance_label_all"].long()
    keys = instance_keys(semantic_gt, instance_gt)
    supervision = semantic_supervision_mask(semantic_gt)
    thing = thing_instance_mask(semantic_gt, instance_gt)
    values, counts_ = torch.unique(semantic_gt, return_counts=True)
    per_view_keys = [
        set(int(k) for k in torch.unique(keys[0, v][thing[0, v]]).tolist())
        for v in range(semantic_gt.shape[1])
    ]
    results["03_labels"] = {
        "semantic_values": {int(v): int(c) for v, c in zip(values.tolist(), counts_.tolist())},
        "class0_pixels": int((semantic_gt == 0).sum()),
        "class0_supervised": int(((semantic_gt == 0) & supervision).sum()),
        "void_supervised": int(((semantic_gt == IGNORE_SEMANTIC) & supervision).sum()),
        "stuff_selected_as_thing": int(thing[(semantic_gt == 0) | (semantic_gt == 1)].sum()),
        "thing_pixels": int(thing.sum()),
        "thing_keys_per_view": [len(k) for k in per_view_keys],
        "context_shared_keys": len(per_view_keys[0] & per_view_keys[1]),
        "max_instance_id": int(instance_gt[thing].max()) if bool(thing.any()) else 0,
        "status": "PASS" if (
            int((semantic_gt == 0).sum()) > 0
            and int(((semantic_gt == 0) & supervision).sum()) == int((semantic_gt == 0).sum())
            and int(((semantic_gt == IGNORE_SEMANTIC) & supervision).sum()) == 0
            and int(thing[(semantic_gt == 0) | (semantic_gt == 1)].sum()) == 0
            and int(instance_gt[thing].max()) < 1000
        ) else "FAIL",
    }
    print(f"[smoke] 03_labels [{results['03_labels']['status']}] "
          f"{json.dumps({k: v for k, v in results['03_labels'].items() if k != 'status'})}", flush=True)

    # ---- 4. mask/alpha conservation + finite loss ------------------------- #
    with torch.no_grad():
        _, metrics_g0 = g0.step_loss(batch, step=1, phase="train")
        _, metrics_g1 = g1.step_loss(batch, step=1, phase="train")
        forward = forward_group(g0, batch, opt_g0)
        mass = forward["masks"]["group_mass"]
        background = forward["masks"]["background_mass"]
        alpha = forward["masks"]["alpha"]
    conservation_all = float((mass.sum(dim=2) + background[:, :, 0] - alpha[:, :, 0]).abs().max())
    results["04_masks_and_loss"] = {
        "mask_alpha_max_error_context": float(metrics_g0["mask_alpha_max_error"]),
        "mask_alpha_max_error_all_views": conservation_all,
        "alpha_mean": float(alpha.mean()),
        "group_mass_mean": float(mass.mean()),
        "background_mass_mean": float(background.mean()),
        "slot_prob_sum_error": float(
            (g0.layer10_group["slot_prob"].sum(-1) - 1).abs().max()
        ),
        "loss_g0": float(metrics_g0["loss"]),
        "loss_g1": float(metrics_g1["loss"]),
        "recon_loss": float(metrics_g0["recon_loss"]),
        "loss_inst": float(metrics_g0["loss_inst"]),
        "loss_sem": float(metrics_g0["loss_sem"]),
        "lambda_at_step1": float(metrics_g0["ramp"]),
        "instance_weight_at_step1": float(metrics_g0["instance_weight"]),
        "semantic_weight_at_step1": float(metrics_g0["semantic_weight"]),
        "semantic_pixel_alpha_gap": float(metrics_g0["semantic_pixel_alpha_gap"]),
        "all_finite": bool(all(
            math.isfinite(float(metrics_g0[key]))
            for key in ("loss", "recon_loss", "loss_inst", "loss_sem", "psnr")
        )),
    }
    results["04_masks_and_loss"]["status"] = "PASS" if (
        conservation_all < 1e-4
        and results["04_masks_and_loss"]["slot_prob_sum_error"] < 1e-5
        and results["04_masks_and_loss"]["lambda_at_step1"] > 0.0
        and results["04_masks_and_loss"]["instance_weight_at_step1"] > 0.0
        and results["04_masks_and_loss"]["all_finite"]
    ) else "FAIL"
    print(f"[smoke] 04_masks_and_loss [{results['04_masks_and_loss']['status']}] "
          f"conservation {conservation_all:.2e} lambda(1) "
          f"{results['04_masks_and_loss']['lambda_at_step1']:.2e}", flush=True)

    # ---- 5. gradient routing of the understanding loss ------------------- #
    routing = {"g0": understanding_backward(g0, batch, opt_g0)}
    routing["g1_gate0"] = understanding_backward(g1, batch, opt_g1)
    routing["g1_gate0.1"] = understanding_backward(g1, batch, opt_g1, gate=0.1)
    g0_grads = routing["g0"]["grad_norms"]
    g1_grads = routing["g1_gate0.1"]["grad_norms"]
    g0_required = (
        "groups.queries", "groups.cross_attn", "groups.token_proj", "groups.spatial_proj",
        "groups.objectness", "groups.semantic", "groups.background_bias",
        "attributes.semantic", "enc_dec_backbone.decoder_blocks.11",
        "anchor_decoder.mu", "anchor_decoder.rho", "activation_head.deconv", "gs_tokens",
    )
    g1_extra = ("feedback.proj", "anchor_decoder.refine_mu.11",
                "enc_dec_backbone.decoder_blocks.10")
    g0_ok = all(g0_grads.get(p, 0.0) > 0 for p in g0_required)
    g1_ok = all(g1_grads.get(p, 0.0) > 0 for p in g0_required + g1_extra)
    gate_dead_end = routing["g1_gate0"].get("gate_grad")
    routing["g0_status"] = "PASS" if g0_ok and not any(
        k.startswith("feedback.") and v > 0 for k, v in g0_grads.items()
    ) else "FAIL"
    routing["g1_status"] = "PASS" if g1_ok and gate_dead_end not in (None, 0.0) else "FAIL"
    print(f"[smoke] 05_gradient_routing g0 [{routing['g0_status']}] g1 [{routing['g1_status']}] "
          f"gate_grad_at_0 {gate_dead_end}", flush=True)
    results["05_gradient_routing"] = routing

    # ---- 6. context-only input (novel GT tamper) ------------------------- #
    tampered = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    tampered["images_all"][0, 2:] = torch.rand_like(tampered["images_all"][0, 2:])
    tampered["semantic_label_all"][0, 2:] = IGNORE_SEMANTIC
    tampered["instance_label_all"][0, 2:] = 0
    with torch.no_grad():
        forward_t = forward_group(g0, tampered, opt_g0)
    deltas = {
        "rgb": max_abs(forward["output"]["render"]["images_pred"],
                       forward_t["output"]["render"]["images_pred"]),
        "gaussians": max_abs(forward["output"]["gaussians"], forward_t["output"]["gaussians"]),
        "group_mass": max_abs(mass, forward_t["masks"]["group_mass"]),
        "semantic_prob": max_abs(forward["semantic_prob"], forward_t["semantic_prob"]),
    }
    results["06_context_only"] = {
        "tamper_deltas": deltas,
        "status": "PASS" if all(value == 0.0 for value in deltas.values()) else "FAIL",
    }
    print(f"[smoke] 06_context_only [{results['06_context_only']['status']}] {json.dumps(deltas)}",
          flush=True)

    # ---- 7. checkpoint round-trip ---------------------------------------- #
    g1.train()
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in g1.parameters() if p.requires_grad and p.dim() != 1
                     and not getattr(p, "_no_weight_decay", False)], "weight_decay": 0.05},
         {"params": [p for p in g1.parameters() if p.requires_grad and (p.dim() == 1
                     or getattr(p, "_no_weight_decay", False))], "weight_decay": 0.0}],
        lr=1e-4, betas=(0.9, 0.95),
    )
    _, metrics = g1.step_loss(batch, step=1, phase="train")
    metrics["loss"].backward()
    torch.nn.utils.clip_grad_norm_(g1.parameters(), 1.0)
    optimizer.step()
    # A previous smoke run may have left a COMPLETE checkpoint here; save_checkpoint
    # refuses to overwrite, so clear this smoke-only scratch directory first.
    if (out_dir / "ckpt_store").is_dir():
        shutil.rmtree(out_dir / "ckpt_store")
    ckpt = save_checkpoint(out_dir / "ckpt_store", 1, g1, optimizer, {"arm": "g1", "smoke": True},
                           {"plan_step": 1}, keep_steps=[1])
    payload = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=False)
    fresh = model_registry[opt_g1.model_type](opt_g1).cuda()
    fresh.load_state_dict(
        torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)["model"], strict=True
    )
    fresh_optimizer = torch.optim.AdamW(
        [{"params": [p for p in fresh.parameters() if p.requires_grad and p.dim() != 1
                     and not getattr(p, "_no_weight_decay", False)], "weight_decay": 0.05},
         {"params": [p for p in fresh.parameters() if p.requires_grad and (p.dim() == 1
                     or getattr(p, "_no_weight_decay", False))], "weight_decay": 0.0}],
        lr=1e-4, betas=(0.9, 0.95),
    )
    fresh_optimizer.load_state_dict(payload["optimizer"])
    same_model = sha256_state(g1.state_dict()) == sha256_state(fresh.state_dict())
    results["07_checkpoint"] = {
        "path": str(ckpt),
        "files": sorted(p.name for p in ckpt.iterdir()),
        "train_state_keys": sorted(payload.keys()),
        "has_optimizer": payload.get("optimizer") is not None,
        "optimizer_entries": len(payload["optimizer"]["state"]) if payload.get("optimizer") else 0,
        "has_torch_rng": payload.get("torch_rng") is not None,
        "has_cuda_rng": payload.get("cuda_rng") is not None,
        "has_numpy_rng": payload.get("numpy_rng") is not None,
        "step_field": int(payload["step"]),
        "meta": payload["meta"],
        "model_round_trip_identical": same_model,
        "status": "PASS" if (same_model and payload.get("optimizer") is not None
                             and payload.get("torch_rng") is not None) else "FAIL",
    }
    print(f"[smoke] 07_checkpoint [{results['07_checkpoint']['status']}]", flush=True)

    # ---- 8. informational GT-free read-out at initialisation ------------- #
    entries = build_val_entries(opt_g0, json.loads(
        Path("workspace_recon_diag/cross_scene/split.json").read_text()), device,
        scenes=["scene0059_00"],
    )
    row = evaluate_group_entry(g0, entries[0], opt_g0)
    summary = summarise_group([row])
    results["08_init_readout_info"] = {
        "note": "informational only - AP at random initialisation is NOT a smoke gate",
        "scene": entries[0]["scene"],
        "novel_ap50": summary["novel_ap50"],
        "novel_tp": summary["novel_tp"],
        "novel_fp": summary["novel_fp"],
        "novel_fn": summary["novel_fn"],
        "objectness_p90": summary["objectness_p90"],
        "slot_entropy_mean": summary["slot_entropy_mean"],
        "status": "INFO",
    }
    print(f"[smoke] 08_init_readout_info {json.dumps(results['08_init_readout_info'])}", flush=True)

    (out_dir / "smoke_report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    failures = [name for name, payload in results.items()
                if isinstance(payload, dict) and payload.get("status") == "FAIL"]
    print(f"[smoke] wrote {out_dir / 'smoke_report.json'}; failures: {failures or 'none'}",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
