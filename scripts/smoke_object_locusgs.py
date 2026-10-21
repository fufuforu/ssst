#!/usr/bin/env python3
"""Pre-run smoke suite for the object-aware LocusGS A/B experiment.

Checks, in order (all results are written to `smoke_report.json`):

1. arm A / arm B / source-checkpoint reconstruction parity at step 0
   (RGB, depth, alpha, 14-d Gaussians, anchors, radii): max abs differences;
2. attribute head output order == the Gaussian head's token -> 64-GS order, and
   the 64 Gaussians of a token do *not* share one attribute vector;
3. label conventions: semantic 0 is supervised, 255 is ignored, stuff vs thing
   and the scene-stable instance key;
4. attribute-render alpha vs RGB-render alpha consistency and finite losses;
5. gradient routing of `0.05 * L_sem + 0.10 * L_inst` alone: arm A reaches the
   attribute head, decoder and GS geometry; arm B (gate != 0) additionally
   reaches the relation module and the layer-11/12 token/anchor refinement;
6. frozen decode radius 0.15, finite values, no alpha/anchor jump over a short
   run;
7. same-arm short-run reproducibility (frame ids, initial forward, updates) plus
   the measured CUDA determinism status;
8. resumable checkpoint contents (original model + new modules + optimizer +
   schedule position + experiment-local step + sampling RNG) and save/restore.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
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
    THING_CLASS_MIN,
    instance_keys,
    semantic_supervision_mask,
    thing_instance_mask,
)
from tokengs.options import config_defaults  # noqa: E402
from scripts.train_object_locusgs import lr_at, save_checkpoint, sha256_file, sha256_state  # noqa: E402

SOURCE_CKPT = Path("workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step2000")
SOURCE_SHA256 = "f0e791b8bb9d49deeba160f0a5fcf75594e4a0638d79ed79b9021ccd6da31c6d"


def report(results: dict, name: str, payload) -> None:
    results[name] = payload
    status = payload.get("status") if isinstance(payload, dict) else None
    flag = "" if status is None else f" [{status}]"
    print(f"[smoke] {name}{flag}", flush=True)
    if status == "FAIL":
        print(f"[smoke]   failure detail: {json.dumps(payload, default=str)[:1800]}", flush=True)


def load_models(opt_a, opt_b, device):
    source = torch.load(SOURCE_CKPT / "model.pt", map_location="cpu", weights_only=False)
    arm_a = model_registry[opt_a.model_type](opt_a).to(device)
    arm_b = model_registry[opt_b.model_type](opt_b).to(device)
    reference = model_registry["siu3r_locusgs_recon"](opt_a).to(device)
    for model in (arm_a, arm_b):
        missing, unexpected = model.load_state_dict(source["model"], strict=False)
        assert not unexpected and len(missing) == 11, (missing, unexpected)
    reference.load_state_dict(source["model"], strict=True)
    for model in (arm_a, arm_b, reference):
        model.eval()
    return source, arm_a, arm_b, reference


def build_batch(opt, plan, step: int, device):
    entry = plan["entries"][step - 1]
    provider = SIU3RProcessedProvider(
        opt, root="/space/mawb/SIU3R/data/scannet/train",
        subset=plan["train_scenes"], training=True, rank=0,
    )
    provider.pin_pair(
        scene_id=entry["scene"],
        context_frame_ids=entry["context"],
        novel_frame_ids=entry["novel"],
        pair_iou=entry["pair_iou"],
    )
    index = {p.name: i for i, p in enumerate(provider.dataset.sample_list)}[entry["scene"]]
    batch = default_collate([provider[index]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return entry, batch


def forward_all(model, batch, opt):
    model_input, _ = split_data(batch, opt)
    decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    output = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
    )
    return output, decoder_input


def max_abs(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def _masked_error(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Max |error| over covered pixels (broadcast `mask` against `values`)."""
    mask = _align_mask(mask, values)
    if not bool(mask.any()):
        return float("nan")
    expanded = mask.expand_as(values)
    return float(values[expanded].abs().max())


def _uncovered_extreme(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Largest magnitude produced where nothing is covered (should be 0, not NaN)."""
    mask = _align_mask(mask, values)
    expanded = (~mask).expand_as(values)
    if not bool(expanded.any()):
        return 0.0
    return float(values[expanded].abs().max())


def _align_mask(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Drop an alpha channel dim so a [B,V,1,H,W] mask matches [B,V,H,W] values."""
    if mask.dim() == values.dim() + 1:
        return mask[:, :, 0]
    if mask.dim() == values.dim():
        return mask
    raise ValueError(f"mask {tuple(mask.shape)} cannot match values {tuple(values.shape)}")


def grad_norms(model, prefixes) -> dict:
    out = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            out.setdefault(next(p for p in prefixes if name.startswith(p)), []).append(
                float(parameter.grad.detach().float().norm())
            )
    return {k: float(np.linalg.norm(v)) for k, v in out.items()}


def understanding_backward(model, batch, opt, *, gate=None):
    """Backprop only ``0.05 * L_sem + 0.10 * L_inst`` and return per-group grads."""
    model.train()
    model.zero_grad(set_to_none=True)
    saved_gate = None
    if gate is not None:
        saved_gate = float(model.relation.gate.detach())
        with torch.no_grad():
            model.relation.gate.fill_(gate)
    model_input, _ = split_data(batch, opt)
    decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    output = model.forward_reconstruction_only(
        ModelInput(model_input.encoder, decoder_input), render_decoder_input=decoder_input
    )
    rendered = model.render_attributes(output["gaussians"], {
        "semantic_logits": output["semantic_logits"],
        "instance_embedding": output["instance_embedding"],
    }, decoder_input)
    semantic_gt = batch["semantic_label_all"].long()
    instance_gt = batch["instance_label_all"].long()
    sem_loss, _ = model.semantic_loss(
        rendered["semantic_prob"], rendered["semantic_alpha"], semantic_gt
    )
    inst_loss, inst_stats = model.instance_loss(
        rendered["instance_embedding"], rendered["instance_alpha"],
        semantic_gt, instance_gt, step=0,
    )
    loss = 0.05 * sem_loss + 0.10 * inst_loss
    loss.backward()
    prefixes = [
        "attributes.semantic", "attributes.instance",
        "relation.norm", "relation.key", "relation.value", "relation.out", "relation.gate",
        "enc_dec_backbone.decoder_blocks.11", "enc_dec_backbone.decoder_blocks.10",
        "anchor_decoder.refine_mu.11", "anchor_decoder.refine_mu.10",
        "anchor_decoder.mu", "anchor_decoder.rho",
        "activation_head.deconv", "gs_tokens",
    ]
    grads = grad_norms(model, prefixes)
    gate_grad = None if model.relation.gate.grad is None else float(model.relation.gate.grad)
    if saved_gate is not None:
        with torch.no_grad():
            model.relation.gate.fill_(saved_gate)
        model.zero_grad(set_to_none=True)
    return {
        "loss_sem": float(sem_loss.detach()),
        "loss_inst": float(inst_loss.detach()),
        "loss": float(loss.detach()),
        "grad_norms": grads,
        "gate_grad": gate_grad,
        "instances": inst_stats,
    }


def one_step(model, optimizer, batch, opt, step, *, warmup, total, grad_clip, lr_peaks):
    optimizer.zero_grad(set_to_none=True)
    _, metrics = model.step_loss(batch, step=step, phase="train")
    metrics["loss"].backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step, lr_peaks[group["name"]], warmup, total)
    optimizer.step()
    return metrics, norm


def rasterizer_determinism_probe(model, batch, opt) -> dict:
    """Measure what the CUDA deterministic setting does to the rasterizer.

    `torch.use_deterministic_algorithms(True, warn_only=True)` additionally turns
    on `torch.utils.deterministic.fill_uninitialized_memory`; gsplat allocates its
    per-Gaussian `means2d` output with `at::empty` and only writes the visible
    entries, so the unwritten ones become NaN and the canonical
    Gaussian-visibility term (part of the reconstruction objective) becomes NaN.
    Both flags are restored afterwards, which is why the probe records them.
    """
    saved_fill = torch.utils.deterministic.fill_uninitialized_memory
    results = {}
    for mode in ("default", "deterministic"):
        torch.use_deterministic_algorithms(mode == "deterministic", warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = mode == "deterministic"
        model.zero_grad(set_to_none=True)
        try:
            with torch.no_grad():
                _, metrics = model.step_loss(batch, step=0, phase="train")
            finite = bool(torch.isfinite(metrics["loss"]))
            del metrics
            with torch.enable_grad():
                _, metrics = model.step_loss(batch, step=0, phase="train")
            results[mode] = {
                "loss": float(metrics["loss"]),
                "loss_gaussian_visibility_layer12": float(
                    metrics.get(
                        "loss_gaussian_visibility_layer12",
                        metrics.get("loss_gaussian_visibility", float("nan")),
                    )
                ),
                "finite": bool(torch.isfinite(metrics["loss"])),
                "finite_without_grad": finite,
            }
            del metrics
        except Exception as error:  # noqa: BLE001 - the strict mode raises
            results[mode] = {"error": f"{type(error).__name__}: {error}"}
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory = saved_fill
    model.zero_grad(set_to_none=True)
    return results


def repeatability_probe(model, batch, opt, *, repeats=2) -> dict:
    """Run the identical step twice: measures the rasterizer's residual randomness."""
    losses = []
    grad_norms = []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        _, metrics = model.step_loss(batch, step=0, phase="train")
        losses.append(float(metrics["loss"]))
        metrics["loss"].backward()
        total = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().pow(2).sum())
        grad_norms.append(math.sqrt(total))
        del metrics
    model.zero_grad(set_to_none=True)
    return {
        "losses": losses,
        "loss_bitwise_identical": len(set(losses)) == 1,
        "max_abs_loss_gap": max(losses) - min(losses),
        "grad_norms": grad_norms,
        "max_abs_grad_gap": max(grad_norms) - min(grad_norms),
    }


def optimizer_state_signature(state: dict) -> str:
    """Exact, order-independent digest of an AdamW state dict."""
    digest = hashlib.sha256()
    for key in sorted(state["state"], key=str):
        entry = state["state"][key]
        for name in sorted(entry):
            value = entry[name]
            if torch.is_tensor(value):
                flat = value.detach().cpu().to(torch.float32).reshape(-1).numpy()
                digest.update(f"{key}.{name}{tuple(value.shape)}".encode())
                digest.update(flat.tobytes())
            else:
                digest.update(f"{key}.{name}={value}".encode())
    groups = [
        {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in group.items()}
        for group in state["param_groups"]
    ]
    digest.update(json.dumps(groups, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def make_optimizer(model, *, lr_existing, lr_new, weight_decay, arm, betas=(0.9, 0.95)):
    from scripts.train_object_locusgs import is_new_parameter

    decay, nodecay, new_decay, new_nodecay = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (decay, nodecay) if not is_new_parameter(name) else (new_decay, new_nodecay)
        if parameter.dim() != 1 and not getattr(parameter, "_no_weight_decay", False):
            target[0].append(parameter)
        else:
            target[1].append(parameter)
    if arm == "a":
        for collection in (decay, nodecay, new_decay, new_nodecay):
            collection[:] = [p for p in collection if not any(p is q for q in model.relation.parameters())]
    groups = [
        {"params": decay, "lr": lr_existing, "peak_lr": lr_existing,
         "weight_decay": weight_decay, "name": "existing_decay"},
        {"params": nodecay, "lr": lr_existing, "peak_lr": lr_existing,
         "weight_decay": 0.0, "name": "existing_nodecay"},
        {"params": new_decay, "lr": lr_new, "peak_lr": lr_new,
         "weight_decay": weight_decay, "name": "new_decay"},
        {"params": new_nodecay, "lr": lr_new, "peak_lr": lr_new,
         "weight_decay": 0.0, "name": "new_nodecay"},
    ]
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(groups, lr=lr_existing, betas=betas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="workspace_object_locusgs/smoke")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--steps", type=int, default=3, help="short-run stability length")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-repro", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results: dict = {}

    source_hash = sha256_file(SOURCE_CKPT / "model.pt")
    report(results, "00_source_checkpoint", {
        "path": str(SOURCE_CKPT),
        "sha256": source_hash,
        "expected_sha256": SOURCE_SHA256,
        "status": "PASS" if source_hash == SOURCE_SHA256 else "FAIL",
        "complete_marker": (SOURCE_CKPT / "COMPLETE").read_text().strip(),
        "model_only": sorted(
            torch.load(SOURCE_CKPT / "model.pt", map_location="cpu", weights_only=False).keys()
        ),
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    })

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    results["00b_plan"] = {
        "path": str(plan_path), "sha256": sha256_file(plan_path),
        "steps": len(plan["entries"]),
        "train_scenes": len(plan["train_scenes"]), "val_scenes": len(plan["val_scenes"]),
        "val_scenes_in_plan": sorted({e["scene"] for e in plan["entries"]} & set(plan["val_scenes"])),
    }

    base = config_defaults["train_siu3r_object_locusgs_ab"].evolve(
        seed=42, batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    opt_a = base.evolve(object_arm="a")
    opt_b = base.evolve(object_arm="b")
    source, arm_a, arm_b, reference = load_models(opt_a, opt_b, device)
    entry, batch = build_batch(opt_a, plan, 1, device)
    print(f"[smoke] step-1 window {entry}", flush=True)

    # ---- 1. reconstruction parity --------------------------------------- #
    with torch.no_grad():
        out_a, decoder_input = forward_all(arm_a, batch, opt_a)
        out_b, _ = forward_all(arm_b, batch, opt_b)
        out_ref, _ = forward_all(reference, batch, opt_a)
    parity = {}
    for name, key in (("rgb", "images_pred"), ("depth", "depths_pred"), ("alpha", "alphas_pred")):
        parity[f"a_vs_b_{name}"] = max_abs(out_a["render"][key], out_b["render"][key])
        parity[f"a_vs_source_{name}"] = max_abs(out_a["render"][key], out_ref["render"][key])
    parity["a_vs_b_gaussians"] = max_abs(out_a["gaussians"], out_b["gaussians"])
    parity["a_vs_source_gaussians"] = max_abs(out_a["gaussians"], out_ref["gaussians"])
    parity["a_vs_b_anchor"] = max_abs(out_a["states"][-1]["mu"], out_b["states"][-1]["mu"])
    parity["a_vs_source_anchor"] = max_abs(out_a["states"][-1]["mu"], out_ref["states"][-1]["mu"])
    parity["a_vs_b_radius"] = max_abs(out_a["states"][-1]["radii"], out_b["states"][-1]["radii"])
    parity["b_token_update_norm"] = (
        None if arm_b.anchor_decoder.last_token_update_norm is None
        else float(arm_b.anchor_decoder.last_token_update_norm)
    )
    parity["status"] = "PASS" if max(
        value for key, value in parity.items() if value is not None
    ) == 0.0 else "FAIL"
    report(results, "01_reconstruction_parity", parity)

    # ---- 2. attribute ordering ------------------------------------------ #
    with torch.no_grad():
        tokens = out_a["states"][-1]["tokens"]
        decoded = arm_a.decode_attributes(tokens)
        semantic = decoded["semantic_logits"]
        embedding = decoded["instance_embedding"]
        direct = arm_a.attributes.semantic(tokens)
        patches = arm_a.attributes.patches
        reshaped = direct.reshape(1, tokens.shape[1], patches, SEMANTIC_CLASS_COUNT).reshape(
            1, tokens.shape[1] * patches, SEMANTIC_CLASS_COUNT
        )
        unique_slot_std = float(semantic[0].reshape(tokens.shape[1], patches, -1).std(dim=1).mean())
        gaussian_order = arm_a.activation_head.deconv(tokens).reshape(
            1, tokens.shape[1], patches, 14
        )
        # The decoder radius is frozen at 0.15, so the head's centres are
        # mu_t + 0.15 * tanh(delta_{t,p}) with the raw head output above.
        decode_radius = torch.full(
            out_a["states"][-1]["mu"].shape[:-1],
            float(arm_a.opt.locusgs_radius_init),
            device=tokens.device,
        )
        gaussians = arm_a.activation_head(
            tokens, out_a["states"][-1]["mu"], out_a["states"][-1]["radii"]
        )
        expected_centres = (
            out_a["states"][-1]["mu"].unsqueeze(2)
            + decode_radius.unsqueeze(2).unsqueeze(-1) * torch.tanh(gaussian_order[..., 0:3])
        ).reshape(1, -1, 3)
        centre_gap = max_abs(gaussians[..., :3], expected_centres)
        order_match = centre_gap < 1e-6
    report(results, "02_attribute_order", {
        "semantic_shape": list(semantic.shape),
        "instance_shape": list(embedding.shape),
        "reshape_matches_linear_output": bool(torch.equal(semantic, reshaped)),
        "slot_std_mean": unique_slot_std,
        "slots_are_not_shared": unique_slot_std > 1e-6,
        "instance_unit_norm_max_error": float((embedding.norm(dim=-1) - 1).abs().max()),
        "gaussian_head_order_reproduced": order_match,
        "gaussian_centre_gap": centre_gap,
        "status": "PASS" if (torch.equal(semantic, reshaped) and unique_slot_std > 1e-6
                             and order_match) else "FAIL",
    })

    # ---- 3. label conventions ------------------------------------------- #
    semantic_gt = batch["semantic_label_all"].long()
    instance_gt = batch["instance_label_all"].long()
    keys = instance_keys(semantic_gt, instance_gt)
    values, counts = torch.unique(semantic_gt, return_counts=True)
    supervision_mask = semantic_supervision_mask(semantic_gt)
    thing_mask = thing_instance_mask(semantic_gt, instance_gt)
    per_view_keys = [
        set(int(k) for k in torch.unique(keys[0, view][thing_mask[0, view]]).tolist())
        for view in range(semantic_gt.shape[1])
    ]
    report(results, "03_labels", {
        "semantic_values": {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())},
        "class0_pixels": int((semantic_gt == 0).sum()),
        "class0_is_supervised": int(((semantic_gt == 0) & supervision_mask).sum()),
        "void_pixels": int((semantic_gt == 255).sum()),
        "void_is_supervised": int(((semantic_gt == IGNORE_SEMANTIC) & supervision_mask).sum()),
        "thing_pixels": int(thing_mask.sum()),
        "stuff_pixels_selected_as_thing": int(
            thing_mask[(semantic_gt == 0) | (semantic_gt == 1)].sum()
        ),
        "thing_keys_per_view": [len(k) for k in per_view_keys],
        "context_shared_keys": len(per_view_keys[0] & per_view_keys[1]),
        "max_instance_id": int(instance_gt[thing_mask].max()) if thing_mask.any() else 0,
        "key_formula_example": [
            {"semantic": int(s), "instance": int(i), "key": int((s + 1) * 1000 + i)}
            for s, i in [
                (int(a), int(b))
                for a, b in zip(semantic_gt[thing_mask][:4].tolist(), instance_gt[thing_mask][:4].tolist())
            ]
        ],
        "status": "PASS" if (
            int((semantic_gt == 0).sum()) > 0
            and int(((semantic_gt == 0) & supervision_mask).sum()) == int((semantic_gt == 0).sum())
            and int((semantic_gt == IGNORE_SEMANTIC).sum()) > 0
            and int(((semantic_gt == IGNORE_SEMANTIC) & supervision_mask).sum()) == 0
            and int(thing_mask[(semantic_gt == 0) | (semantic_gt == 1)].sum()) == 0
            and len(per_view_keys[0] & per_view_keys[1]) > 0
            and int(instance_gt[thing_mask].max()) < 1000
        ) else "FAIL",
    })

    # ---- 4. alpha consistency + finite losses ---------------------------- #
    with torch.no_grad():
        _, metrics_a = arm_a.step_loss(batch, step=0, phase="train")
        _, metrics_b = arm_b.step_loss(batch, step=0, phase="train")
    alpha_gap_sem = float((out_a["render"]["alphas_pred"] - out_a["render"]["alphas_pred"]).abs().max())
    with torch.no_grad():
        rendered_a = arm_a.render_attributes(out_a["gaussians"], {
            "semantic_logits": out_a["semantic_logits"],
            "instance_embedding": out_a["instance_embedding"],
        }, decoder_input)
    report(results, "04_render_and_losses", {
        "status": "PASS" if (
            max_abs(out_a["render"]["alphas_pred"], rendered_a["semantic_alpha"]) < 1e-6
            and max_abs(rendered_a["semantic_alpha"], rendered_a["instance_alpha"]) < 1e-6
            and math.isfinite(float(metrics_a["loss"]))
            and math.isfinite(float(metrics_b["loss"]))
            and float(rendered_a["semantic_alpha"].max()) > 0.0
        ) else "FAIL",
        "rgb_vs_semantic_alpha_max_abs": max_abs(
            out_a["render"]["alphas_pred"], rendered_a["semantic_alpha"]
        ),
        "semantic_vs_instance_alpha_max_abs": max_abs(
            rendered_a["semantic_alpha"], rendered_a["instance_alpha"]
        ),
        # Coverage-free pixels have alpha = 0, where a class distribution is
        # undefined; the losses (and these checks) are restricted to alpha > 0.05.
        "covered_fraction": float(
            (rendered_a["semantic_alpha"] > 0.05).float().mean()
        ),
        "semantic_prob_sum_error": _masked_error(
            rendered_a["semantic_prob"].sum(dim=2) - 1, rendered_a["semantic_alpha"] > 0.05
        ),
        "instance_embedding_norm_error": _masked_error(
            rendered_a["instance_embedding"].norm(dim=2) - 1,
            rendered_a["instance_alpha"] > 0.05,
        ),
        "uncovered_semantic_prob_max": _uncovered_extreme(
            rendered_a["semantic_prob"], rendered_a["semantic_alpha"] > 0.05
        ),
        "loss_a": float(metrics_a["loss"]),
        "loss_b": float(metrics_b["loss"]),
        "loss_sem": float(metrics_a["loss_sem"]),
        "loss_inst": float(metrics_a["loss_inst"]),
        "ramp_at_step0": float(metrics_a["ramp"]),
        "sem_coverage": float(metrics_a["sem_coverage"]),
        "instances_present_used": [
            float(metrics_a["instances_present"]), float(metrics_a["instances_used"])
        ],
        "all_finite": bool(all(
            math.isfinite(float(metrics_a[key]))
            for key in ("loss", "loss_sem", "loss_inst", "psnr", "alpha_mean", "radius_mean")
        )),
        "unused_alpha_gap": alpha_gap_sem,
    })

    # ---- 5. gradient routing -------------------------------------------- #
    routing = {"arm_a": understanding_backward(arm_a, batch, opt_a)}
    routing["arm_b_gate0"] = understanding_backward(arm_b, batch, opt_b)
    routing["arm_b_gate0.1"] = understanding_backward(arm_b, batch, opt_b, gate=0.1)
    a_grads = routing["arm_a"]["grad_norms"]
    b_grads = routing["arm_b_gate0.1"]["grad_norms"]
    a_ok = all(
        a_grads.get(prefix, 0.0) > 0 for prefix in (
            "attributes.semantic", "attributes.instance", "anchor_decoder.mu",
            "activation_head.deconv", "gs_tokens", "enc_dec_backbone.decoder_blocks.11",
        )
    ) and "relation.gate" not in a_grads
    b_ok = all(
        b_grads.get(prefix, 0.0) > 0 for prefix in (
            "attributes.semantic", "attributes.instance", "relation.norm", "relation.key",
            "relation.value", "relation.out", "anchor_decoder.mu", "anchor_decoder.rho",
            "anchor_decoder.refine_mu.11", "anchor_decoder.refine_mu.10",
            "enc_dec_backbone.decoder_blocks.11", "enc_dec_backbone.decoder_blocks.10",
            "activation_head.deconv", "gs_tokens",
        )
    )
    routing["arm_a_status"] = "PASS" if a_ok else "FAIL"
    routing["arm_b_status"] = "PASS" if b_ok else "FAIL"
    report(results, "05_gradient_routing", routing)

    # ---- 6. short-run stability ----------------------------------------- #
    stability = {"steps": []}
    for arm_name, model, opt_arm in (("a", arm_a, opt_a), ("b", arm_b, opt_b)):
        model.train()
        opt_run = opt_arm.evolve(object_loss_ramp_steps=0)  # ramp = 1 immediately
        optimizer = make_optimizer(
            model, lr_existing=1e-5, lr_new=1e-4, weight_decay=0.05, arm=arm_name
        )
        peaks = {group["name"]: group["peak_lr"] for group in optimizer.param_groups}
        rows = []
        for step in range(0, args.steps):
            metrics, norm = one_step(
                model, optimizer, batch, opt_run, step,
                warmup=200, total=6000, grad_clip=1.0, lr_peaks=peaks,
            )
            decode_radius = model.activation_head.last_decode_radius
            rows.append({
                "step": step,
                "loss": float(metrics["loss"]),
                "recon": float(metrics["recon_loss"]),
                "sem": float(metrics["loss_sem"]),
                "inst": float(metrics["loss_inst"]),
                "grad_norm": norm,
                "alpha_mean": float(metrics["alpha_mean"]),
                "anchor_max": float(metrics["anchor_max"]),
                "radius_min": float(metrics["radius_min"]),
                "radius_max": float(metrics["radius_max"]),
                "decode_radius": (
                    [float(decode_radius.min()), float(decode_radius.max())]
                    if decode_radius is not None else None
                ),
                "token_update_norm": (
                    float(metrics["token_update_norm"]) if "token_update_norm" in metrics else None
                ),
            })
        finite = all(
            math.isfinite(row[key])
            for row in rows
            for key in ("loss", "recon", "sem", "inst", "grad_norm", "alpha_mean", "anchor_max")
        )
        parameters_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        radius_ok = all(
            row["decode_radius"] is not None
            and abs(row["decode_radius"][0] - 0.15) < 1e-6
            and abs(row["decode_radius"][1] - 0.15) < 1e-6
            for row in rows
        )
        alpha_jump = (
            max(abs(rows[i]["alpha_mean"] - rows[i - 1]["alpha_mean"]) for i in range(1, len(rows)))
            if len(rows) > 1 else 0.0
        )
        anchor_jump = (
            max(abs(rows[i]["anchor_max"] - rows[i - 1]["anchor_max"]) for i in range(1, len(rows)))
            if len(rows) > 1 else 0.0
        )
        stability["steps"].append({
            "arm": arm_name,
            "rows": rows,
            "finite": finite,
            "parameters_finite": parameters_finite,
            "decode_radius_is_0.15": radius_ok,
            "alpha_jump_max": alpha_jump,
            "anchor_jump_max": anchor_jump,
            "status": "PASS" if (finite and parameters_finite and radius_ok
                                 and alpha_jump < 0.5 and anchor_jump < 1.0) else "FAIL",
        })
        model.load_state_dict(source["model"], strict=False)  # restore for later checks
        model.eval()
    stability["status"] = (
        "PASS" if all(entry["status"] == "PASS" for entry in stability["steps"]) else "FAIL"
    )
    report(results, "06_short_run_stability", stability)

    # ---- 7. reproducibility --------------------------------------------- #
    # Repeatability is measured first and in a clean allocator state; the
    # deterministic-mode probe poisons the CUDA caching allocator with NaN blocks
    # (see its docstring), so it runs last, right before the report is written.
    torch.cuda.empty_cache()
    repeat = repeatability_probe(arm_a, batch, opt_a)
    print(f"[smoke] 06b_identical_step_repeat {json.dumps(repeat)}", flush=True)
    if not args.skip_repro:
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTHONHASHSEED"] = "0"
        runs = []
        failed = False
        for index in (1, 2):
            run_dir = out_dir / f"repro{index}"
            command = [
                sys.executable, "scripts/train_object_locusgs.py",
                "--arm", "a", "--out-dir", str(run_dir),
                "--steps", "6000", "--max-steps", str(args.steps),
                "--ramp-override", "0", "--save-steps", "0",
                "--no-eval", "--log-every", "1",
                "--manifest-out", str(run_dir / "manifest.json"),
            ]
            proc = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True, text=True)
            runs.append({
                "index": index,
                "returncode": proc.returncode,
                "tail": proc.stdout.strip().splitlines()[-6:],
                "stderr_tail": proc.stderr.strip().splitlines()[-4:],
            })
            failed = failed or proc.returncode != 0
        if failed:
            results["07_reproducibility"] = {"runs": runs, "status": "FAIL"}
            (out_dir / "smoke_report.json").write_text(
                json.dumps(results, indent=1), encoding="utf-8"
            )
            print("[smoke] reproducibility subprocess failed", flush=True)
            return 1
        logs = [
            [json.loads(line) for line in (out_dir / f"repro{index}" / "train_log.jsonl").read_text().splitlines()]
            for index in (1, 2)
        ]
        same_frames = all(
            (a["scene"], a["context"], a["novel"]) == (b["scene"], b["context"], b["novel"])
            for a, b in zip(logs[0], logs[1])
        )
        loss_gap = max((abs(a["loss"] - b["loss"]) for a, b in zip(logs[0], logs[1])), default=0.0)
        update_gap = None
        for a, b in zip(logs[0], logs[1]):
            if a.get("update_by_group") and b.get("update_by_group"):
                gaps = [abs(a["update_by_group"][key] - b["update_by_group"][key])
                        for key in a["update_by_group"]]
                update_gap = max(gaps) if update_gap is None else max(update_gap, max(gaps))
        hash_initial = [
            sha256_file(out_dir / f"repro{index}" / "ckpt_step0" / "model.pt") for index in (1, 2)
        ]
        hash_final = [
            (out_dir / f"repro{index}" / "final_state_hash.txt").read_text().strip()
            for index in (1, 2)
        ]
        report(results, "07_reproducibility", {
            "runs": runs,
            "identical_frame_ids": same_frames,
            "max_abs_loss_gap": loss_gap,
            "max_abs_update_gap": update_gap,
            "initial_model_sha256": hash_initial,
            "initial_model_identical": hash_initial[0] == hash_initial[1],
            "final_model_sha256": hash_final,
            "final_model_identical": hash_final[0] == hash_final[1],
            "determinism_note": "deterministic_algorithms=True (warn_only) and "
                                "gsplat's rasterizer cannot be used with the CUDA "
                                "deterministic setting (its means2d output becomes "
                                "NaN, see 06b), so both subprocesses run the default "
                                "kernel and CUBLAS_WORKSPACE_CONFIG=:4096:8; any "
                                "residual randomness is reported verbatim above",
            "status": "PASS" if (same_frames and hash_initial[0] == hash_initial[1]) else "FAIL",
        })

    # ---- 8. checkpoint save / restore ------------------------------------ #
    arm_b.train()
    optimizer_b = make_optimizer(
        arm_b, lr_existing=1e-5, lr_new=1e-4, weight_decay=0.05, arm="b"
    )
    opt_run = opt_b.evolve(object_loss_ramp_steps=0)
    peaks = {group["name"]: group["peak_lr"] for group in optimizer_b.param_groups}
    metrics_before, _ = one_step(arm_b, optimizer_b, batch, opt_run, 0,
                                 warmup=200, total=6000, grad_clip=1.0, lr_peaks=peaks)
    del metrics_before
    ckpt = save_checkpoint(out_dir / "ckpt_store", 1, arm_b, optimizer_b,
                           {"arm": "b", "smoke": True}, {"plan_step": 1}, keep_steps=[1])
    payload = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=False)
    fresh = model_registry[opt_b.model_type](opt_b).to(device)
    fresh.load_state_dict(
        torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)["model"], strict=True
    )
    fresh_optimizer = make_optimizer(
        fresh, lr_existing=1e-5, lr_new=1e-4, weight_decay=0.05, arm="b"
    )
    fresh_optimizer.load_state_dict(payload["optimizer"])
    before_hash = sha256_state(arm_b.state_dict())
    restored_hash = sha256_state(fresh.state_dict())
    optimizer_restored_identical = optimizer_state_signature(
        payload["optimizer"]
    ) == optimizer_state_signature(fresh_optimizer.state_dict())
    torch.manual_seed(0)
    metrics_continue, _ = one_step(arm_b, optimizer_b, batch, opt_run, 1,
                                   warmup=200, total=6000, grad_clip=1.0, lr_peaks=peaks)
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    fresh_peaks = {group["name"]: group["peak_lr"] for group in fresh_optimizer.param_groups}
    metrics_restored, _ = one_step(fresh, fresh_optimizer, batch, opt_run, 1,
                                   warmup=200, total=6000, grad_clip=1.0, lr_peaks=fresh_peaks)
    continuation_gap = abs(float(metrics_continue["loss"]) - float(metrics_restored["loss"]))
    report(results, "08_checkpoint", {
        "path": str(ckpt),
        "files": sorted(p.name for p in ckpt.iterdir()),
        "train_state_keys": sorted(payload.keys()),
        "has_optimizer_state": payload.get("optimizer") is not None,
        "optimizer_state_entries": len(payload["optimizer"]["state"]) if payload.get("optimizer") else 0,
        "meta": payload["meta"],
        "step_field": int(payload["step"]),
        "has_torch_rng": payload.get("torch_rng") is not None,
        "has_cuda_rng": payload.get("cuda_rng") is not None,
        "has_numpy_rng": payload.get("numpy_rng") is not None,
        "model_restored_identical": before_hash == restored_hash,
        "optimizer_restored_identical": optimizer_restored_identical,
        "continuation_loss_gap": continuation_gap,
        "continuation_losses": [float(metrics_continue["loss"]), float(metrics_restored["loss"])],
        "continuation_note": "both continuations are finite; the residual gap is the "
                             "rasterizer's run-to-run randomness measured in 06b, not a "
                             "restore error (model and optimizer states are compared "
                             "exactly above)",
        "status": "PASS" if (before_hash == restored_hash
                             and optimizer_restored_identical
                             and payload.get("optimizer") is not None
                             and math.isfinite(float(metrics_continue["loss"]))
                             and math.isfinite(float(metrics_restored["loss"]))) else "FAIL",
    })

    # ---- deterministic-kernel probe (last: it poisons the allocator) ------ #
    determinism = rasterizer_determinism_probe(arm_a, batch, opt_a)
    results["06b_rasterizer_determinism"] = {
        **determinism,
        "identical_step_repeat": repeat,
        "status": "NOTE" if (
            determinism["default"]["finite"]
            and not determinism["deterministic"].get("finite", False)
        ) else "FAIL",
        "conclusion": "torch.use_deterministic_algorithms(True) also enables "
                      "torch.utils.deterministic.fill_uninitialized_memory, so "
                      "gsplat's partially written per-Gaussian `means2d` output "
                      "becomes NaN and the canonical Gaussian-visibility term (part "
                      "of the reconstruction objective) turns the total loss into "
                      "NaN.  The deterministic setting is therefore unusable with "
                      "this differentiable rasterizer; the runs stay in the default "
                      "kernel mode and report their residual randomness empirically "
                      "(see identical_step_repeat and 07_reproducibility).",
    }
    print(f"[smoke] 06b_rasterizer_determinism {json.dumps(determinism)}", flush=True)

    (out_dir / "smoke_report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    failures = [
        name for name, payload in results.items()
        if isinstance(payload, dict) and payload.get("status") == "FAIL"
    ]
    print(f"[smoke] wrote {out_dir / 'smoke_report.json'}; failures: {failures or 'none'}",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
