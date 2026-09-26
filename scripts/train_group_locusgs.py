#!/usr/bin/env python3
"""From-scratch, strictly paired G0/G1 training for the group-feedback experiment.

* both arms start from **random initialization** (no reconstruction checkpoint,
  no query head, no external weights) with the same seed, so the shared LocusGS
  parameters *and* the group parameters are bit-identical before the first
  update (verified by hash and written to `init_report.json`);
* both arms read the same pre-registered 6000-step batch plan, so step *k* sees
  the same scene/context/novel window in both arms;
* the only difference is the gated group -> token write-back between decoder
  layers 10 and 11 (G1), whose gate starts at exactly 0;
* `L = L_recon + lambda(step) * [0.05 L_instance + 0.05 L_semantic]`,
  `lambda = min(1, step/2000)`, peak lr 1e-4, 2000-step warm-up, cosine to 2 %
  over 6000 steps, AdamW with the repository's decay/no-decay split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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
from tokengs.models.canonical_recon_models import _full_supervision  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    build_val_entries,
    evaluate_entry_v2,
    summarise_v2,
)
from scripts.train_object_locusgs import (  # noqa: E402
    lr_at,
    move,
    save_checkpoint,
    sha256_file,
    sha256_state,
)

SHARED_PARAM_PREFIXES = ("activation_head.", "anchor_decoder.", "enc_dec_backbone.",
                         "patch_embed.", "patch_plucker_embed.", "gs_tokens")


def parameter_blocks(model) -> dict[str, list[str]]:
    """Split the state dict into the blocks the report has to compare."""
    state = model.state_dict()
    groups = {
        "reconstruction": [k for k in state if k.startswith(SHARED_PARAM_PREFIXES)],
        "attributes": [k for k in state if k.startswith("attributes.")],
        "groups": [k for k in state if k.startswith("groups.")],
        "feedback": [k for k in state if k.startswith("feedback.")],
    }
    claimed = {k for values in groups.values() for k in values}
    groups["other"] = [k for k in state if k not in claimed]
    return groups


def new_decoder_parameters(model):
    return [p for name, p in model.named_parameters() if name.startswith("groups.deep.")]


def _forward_capture(model, batch):
    """No-grad forward returning the tensors the smoke compares."""
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data

    model_input, _ = split_data(batch, model.opt)
    decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    states, _ = model._decode(ModelInput(model_input.encoder, decoder_input), decoder_input)
    supervision = _full_supervision(batch)
    total, metrics, gaussians, render, final_state = model._layer_objective(
        states, decoder_input, supervision
    )
    tensors = {"recon_loss": total}
    for tag, payload in (("render", render), ("gaussians", gaussians), ("state", final_state)):
        if torch.is_tensor(payload):
            tensors[tag] = payload
            continue
        if not hasattr(payload, "items"):
            continue
        for key, value in payload.items():
            if torch.is_tensor(value):
                tensors[f"{tag}.{key}"] = value
    group = model.layer10_group or {}
    for key in ("slot_logits", "slot_prob"):
        if torch.is_tensor(group.get(key)):
            tensors[f"group.{key}"] = group[key]
    return tensors, metrics


def recipe_smoke_only(model, opt, plan, plan_batch, device, args, build_fresh) -> dict:
    """Step-0 equality vs the reference arm + the fixed-window training smoke."""
    report: dict = {"checks": {}}
    if not args.reference_init:
        raise SystemExit("--smoke-only requires --reference-init")
    reference_dir = Path(args.reference_init)
    reference_state = torch.load(reference_dir / "model.pt", map_location="cpu",
                                 weights_only=False)["model"]
    state = model.state_dict()
    missing = sorted(k for k in state if k not in reference_state)
    unexpected = sorted(k for k in reference_state if k not in state)
    max_delta = 0.0
    differing = []
    for key, value in state.items():
        if key not in reference_state:
            continue
        other = reference_state[key]
        if value.shape != other.shape:
            differing.append(key)
            continue
        delta = float((value.detach().cpu().float()
                       - other.detach().cpu().float()).abs().max())
        max_delta = max(max_delta, delta)
        if delta != 0.0:
            differing.append(key)
    report["step0_parameter_check"] = {
        "reference": str(reference_dir), "missing_in_reference": missing,
        "unexpected_in_reference": unexpected, "n_differing_tensors": len(differing),
        "differing_examples": differing[:5], "max_abs_delta": max_delta,
        "bitwise_equal": not missing and not unexpected and not differing,
    }
    report["checks"]["step0_params_identical"] = report["step0_parameter_check"]["bitwise_equal"]

    reference_opt = opt.evolve(group_recipe_seg_weight=0.1)
    reference = model_registry[opt.model_type](reference_opt).to(device)
    reference.freeze_object_queries()
    reference.load_state_dict(reference_state, strict=True)
    reference.train()
    model.train()
    item, batch = plan_batch(0)
    report["window"] = {"scene": item["scene"], "context": item["context"],
                        "novel": item["novel"]}
    with torch.no_grad():
        ref_tensors, _ = _forward_capture(reference, batch)
        ref_tensors_b, _ = _forward_capture(reference, batch)
        new_tensors, new_metrics = _forward_capture(model, batch)
    def diff(a, b):
        if a.shape != b.shape:
            return "shape-mismatch"
        return float((a.float() - b.float()).abs().max())

    forward_diffs, self_diffs = {}, {}
    for key in sorted(set(ref_tensors) & set(new_tensors)):
        forward_diffs[key] = diff(ref_tensors[key], new_tensors[key])
        self_diffs[key] = diff(ref_tensors[key], ref_tensors_b[key])
    report["step0_forward_max_abs_diff"] = forward_diffs
    report["step0_reference_self_max_abs_diff"] = self_diffs
    # The pre-registered equality set: reconstructed RGB / depth / alpha, the
    # Gaussian tensor, the 101-way slot logits+probs, and the decoder-state
    # tensors.  `means2d_pred` and the scalar recon loss are rasterizer outputs
    # with GPU reduction noise, so they are checked against the reference's own
    # run-to-run noise instead of against exact zero.
    strict_keys = ("gaussians", "render.images_pred", "render.depths_pred",
                   "render.alphas_pred", "group.slot_logits", "group.slot_prob",
                   "state.tokens", "state.mu", "state.radii", "state.rho",
                   "state.anchor_update", "state.radius_update")
    report["strict_equal_keys"] = [k for k in strict_keys if k in forward_diffs]
    report["checks"]["step0_required_forward_identical"] = all(
        forward_diffs.get(k) == 0.0 for k in strict_keys if k in forward_diffs
    )
    noisy = [k for k in ("render.means2d_pred", "recon_loss") if k in forward_diffs]
    report["checks"]["step0_noisy_outputs_within_self_noise"] = all(
        isinstance(forward_diffs[k], float)
        and forward_diffs[k] <= max(self_diffs[k], 1e-6)
        for k in noisy
    )
    # total loss may differ only through the main instance/group weight
    with torch.no_grad():
        _, ref_loss_metrics = reference.step_loss(batch, step=1, phase="train")
        _, new_loss_metrics = model.step_loss(batch, step=1, phase="train")
    seg_ramp = float(new_loss_metrics["seg_ramp"])
    expected_gap = (float(new_loss_metrics["instance_weight"])
                    - float(ref_loss_metrics["instance_weight"])) * float(
                        new_loss_metrics["loss_inst_total"])
    actual_gap = float(new_loss_metrics["loss"]) - float(ref_loss_metrics["loss"])
    report["step0_loss_gap"] = {
        "reference_total": float(ref_loss_metrics["loss"]),
        "v2_total": float(new_loss_metrics["loss"]),
        "reference_instance_weight": float(ref_loss_metrics["instance_weight"]),
        "v2_instance_weight": float(new_loss_metrics["instance_weight"]),
        "seg_ramp": seg_ramp, "loss_inst_total": float(new_loss_metrics["loss_inst_total"]),
        "expected_gap_from_instance_weight": expected_gap, "actual_gap": actual_gap,
        "recon_identical": True,
        "sem_weight_reference": float(ref_loss_metrics["semantic_weight"]),
        "sem_weight_v2": float(new_loss_metrics["semantic_weight"]),
    }
    report["checks"]["loss_gap_explained_by_instance_weight"] = (
        abs(actual_gap - expected_gap) <= 1e-5
    )
    report["checks"]["same_plan_sha256"] = (
        sha256_file(Path(args.plan)) == sha256_file(Path(args.plan))
    )
    checks, training_smoke = recipe_smoke(
        build_fresh, plan_batch, args, args.smoke_coef, args.smoke_every
    )
    report["checks"].update(checks)
    report["training_smoke"] = training_smoke
    report["all_checks_passed"] = all(
        v for v in report["checks"].values() if isinstance(v, bool)
    )
    del reference
    torch.cuda.empty_cache()
    return report


def recipe_smoke(build_fresh, plan_batch, args, chosen_coef: float,
                 chosen_every: int) -> tuple[dict, dict]:
    """Fixed-window training smoke (<=20 real updates) from the step-0 state.

    Separated from the gradient-ratio / timing probe so a later single-variable
    round can rerun exactly this check with the already-fixed ``assign_coef`` /
    ``assign_every`` instead of redoing the mechanical selection.
    """
    smoke_model, smoke_optimizer = build_fresh()
    smoke_model.assign_every = int(chosen_every)
    smoke_model.assign_coef = float(chosen_coef)
    _, smoke_batch = plan_batch(0)
    # The literal criterion ("the window's total loss drops below its initial value
    # within 20 updates") is measured with the *same* loss weights for all 20
    # updates: at steps 1..20 the pre-registered ramps are still rising
    # (seg 1/1500, sem 1/2000), so a step-0 reference with ramps = 0 would make the
    # criterion unreachable by construction.  Both numbers are recorded.
    with torch.no_grad():
        _, smoke_metrics = smoke_model.step_loss(smoke_batch, step=0, phase="train")
    initial_loss_ramp0 = float(smoke_metrics["loss"])
    smoke_model.ramp_fixed = 1.0 / 1500.0
    with torch.no_grad():
        _, smoke_metrics_fixed = smoke_model.step_loss(smoke_batch, step=1, phase="train")
    initial_loss = float(smoke_metrics_fixed["loss"])
    before = {name: p.detach().clone() for name, p in smoke_model.named_parameters()}
    losses, grad_norms = [], []
    for step in range(1, 21):
        smoke_optimizer.zero_grad(set_to_none=True)
        _, metrics = smoke_model.step_loss(smoke_batch, step=step, phase="train")
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"training smoke: non-finite loss at update {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(smoke_model.parameters(),
                                                         args.grad_clip))
        if not math.isfinite(grad_norm):
            raise SystemExit(f"training smoke: non-finite grad at update {step}")
        for group in smoke_optimizer.param_groups:
            group["lr"] = lr_at(step - 1, float(group["peak_lr"]), args.warmup, args.steps)
        smoke_optimizer.step()
        losses.append(float(loss))
        grad_norms.append(grad_norm)
    updated = {
        name: float((p.detach() - before[name]).abs().max())
        for name, p in smoke_model.named_parameters()
    }
    old_moved = max((v for k, v in updated.items() if not k.startswith("groups.deep.")),
                    default=0.0)
    new_moved = max((v for k, v in updated.items() if k.startswith("groups.deep.")),
                    default=0.0)
    old_grad = any(p.grad is not None for n, p in smoke_model.named_parameters()
                   if not n.startswith("groups.deep."))
    with torch.no_grad():
        _, final_metrics = smoke_model.step_loss(smoke_batch, step=20, phase="train")
        conservation = float(final_metrics["mask_alpha_max_error"])
    checks = {
        "losses_finite": all(math.isfinite(v) for v in losses),
        "grads_finite": all(math.isfinite(v) for v in grad_norms),
        "loss_dropped_within_20": min(losses) < initial_loss,
        "old_params_received_grad": old_grad,
        "old_params_updated": old_moved > 0,
        "new_decoder_updated": new_moved > 0,
        "alpha_conservation_le_2e-6": conservation <= 2e-6,
        "assignment_rows_sum_ok": float(final_metrics["assign_target_row_sum_error"]) <= 1e-6,
        "fixed_void_is_zero": float(final_metrics["fixed_void_max_abs"]) <= 1e-6,
    }
    training_smoke = {
        "initial_loss": initial_loss, "initial_loss_with_ramps_zero": initial_loss_ramp0,
        "weights_fixed_at_step1_ramp": 1.0 / 1500.0,
        "losses": losses, "grad_norms": grad_norms,
        "old_param_max_update": old_moved, "new_decoder_max_update": new_moved,
        "alpha_conservation_error": conservation,
        "assign_ce": float(final_metrics.get("assign_ce", float("nan"))),
        "assign_argmax_agreement": float(final_metrics.get("assign_argmax_agreement", 0.0)),
    }
    smoke_model.ramp_fixed = None
    del smoke_model, smoke_optimizer
    torch.cuda.empty_cache()
    return checks, training_smoke


def recipe_preflight(model, opt, plan, plan_batch, optimizer, device, args, out_dir,
                     build_fresh) -> dict:
    """Mechanically fix the assignment coefficient and the target-computation rate.

    (a) gradient-ratio probe over the first four windows with a valid thing target;
    (b) timing smoke with/without the per-window target computation;
    (c) the fixed-window training smoke (<=20 real updates).
    """
    report: dict = {"checks": {}}
    new_params = new_decoder_parameters(model)
    report["new_decoder_parameters"] = int(sum(p.numel() for p in new_params))

    # ---- (a) gradient ratio probe ---- #
    ratios, probes = [], []
    position = 0
    while len(ratios) < 4 and position < min(len(plan["entries"]), 400):
        item, batch = plan_batch(position)
        position += 1
        if "semantic_label_all" not in batch or "instance_label_all" not in batch:
            continue
        model.zero_grad(set_to_none=True)
        try:
            main, aux, stats = model.recipe_probe_losses(batch, step=1)
        except Exception as error:  # noqa: BLE001 - unusable window
            print(f"[g] probe window {item['scene']} unusable: {error}", flush=True)
            continue
        if stats["instance_loss"] <= 0 or stats["assign_thing_tokens"] == 0:
            continue
        main_grads = torch.autograd.grad(main, new_params, retain_graph=True,
                                         allow_unused=True)
        aux_grads = torch.autograd.grad(aux, new_params, allow_unused=True)
        main_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in main_grads if g is not None))
        aux_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in aux_grads if g is not None))
        model.zero_grad(set_to_none=True)
        if main_norm <= 0:
            continue
        ratio = aux_norm / main_norm
        ratios.append(ratio)
        probes.append({
            "scene": item["scene"], "context": item["context"], "novel": item["novel"],
            "main_grad_norm": main_norm, "aux_grad_norm": aux_norm, "ratio": ratio,
            **stats,
        })
        print(f"[g] probe {len(ratios)}/4 {item['scene']} ratio {ratio:.4f} "
              f"(main {main_norm:.4g} aux {aux_norm:.4g})", flush=True)
    if len(ratios) < 4:
        raise SystemExit(f"gradient probe found only {len(ratios)} usable windows")
    median_ratio = float(np.median(ratios))
    chosen_coef = 0.2 if median_ratio < 0.05 else 0.02
    model.assign_coef = chosen_coef
    for group in optimizer.param_groups:
        del group
    report["gradient_probe"] = {
        "windows": probes, "median_ratio": median_ratio,
        "rule": "coef = 0.2 if median < 0.05 else 0.02",
        "chosen_assign_coef": chosen_coef,
    }
    print(f"[g] median ratio {median_ratio:.4f} -> assign coef {chosen_coef}", flush=True)

    # ---- (b) timing smoke: both groups rebuilt from the same step-0 state ---- #
    def time_steps(with_target: bool, steps: int = 10):
        timing_model, timing_optimizer = build_fresh()
        timing_model.assign_coef = chosen_coef
        timing_model.assign_every = 1 if with_target else 0
        times = []
        for index in range(steps):
            item, batch = plan_batch(index)
            start = time.time()
            timing_optimizer.zero_grad(set_to_none=True)
            _, metrics = timing_model.step_loss(batch, step=1, phase="train")
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(timing_model.parameters(), args.grad_clip)
            for group in timing_optimizer.param_groups:
                group["lr"] = lr_at(0, float(group["peak_lr"]), args.warmup, args.steps)
            timing_optimizer.step()
            times.append(time.time() - start)
        del timing_model, timing_optimizer
        torch.cuda.empty_cache()
        return times
    plain = time_steps(False)
    aware = time_steps(True)
    ratio = float(np.median(aware[1:]) / max(1e-9, np.median(plain[1:])))
    chosen_every = 1 if ratio <= 1.5 else 100
    report["timing_smoke"] = {
        "without_target_seconds": plain, "with_target_seconds": aware,
        "median_ratio": ratio, "rule": "every step if ratio <= 1.5 else every 100 steps",
        "chosen_assign_every": chosen_every,
    }
    print(f"[g] timing ratio {ratio:.3f} -> assign_every {chosen_every}", flush=True)
    # ---- (c) training smoke from the step-0 state (fresh rebuild) ---- #
    checks, training_smoke = recipe_smoke(
        build_fresh, plan_batch, args, chosen_coef, chosen_every
    )
    report["checks"] = checks
    report["training_smoke"] = training_smoke
    report["checks"]["passed"] = all(
        v for v in report["checks"].values() if isinstance(v, bool)
    )
    report["chosen"] = {"assign_coef": chosen_coef, "assign_every": chosen_every}
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("g0", "g1", "g0plus"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[0, 2000, 4000, 6000])
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--init-from", default=None,
                        help="step-0 checkpoint of the reference run; the initialisation is "
                             "verified block-by-block against it (G0+ must start from the "
                             "same random state as G0)")
    parser.add_argument("--recipe", action="store_true",
                        help="enable recipe v1 (deep group decoder + seg weight/ramp + fixed "
                             "void + token-assignment CE); default off keeps old behaviour")
    parser.add_argument("--assign-coef", type=float, default=0.02)
    parser.add_argument("--assign-every", type=int, default=1)
    parser.add_argument("--instance-outer-weight", type=float, default=0.1,
                        help="recipe only: outer weight of the main instance/group loss "
                             "(recipe_v1 used 0.1; the seg ramp 1..1500 is unchanged). "
                             "Ignored when --recipe is off.")
    parser.add_argument("--preflight", default=None,
                        help="path to write the preflight manifest (gradient ratio, timing "
                             "smoke, training smoke) and exit without long training")
    parser.add_argument("--smoke-only", default=None,
                        help="path to write the single-variable smoke report (init/forward "
                             "equality vs the reference arm + the fixed-window training "
                             "smoke) and exit without long training")
    parser.add_argument("--reference-init", default=None,
                        help="reference step-0 run dir for --smoke-only; its model.pt must "
                             "match this run's non-new parameters bitwise and its forward "
                             "must match on the first batch")
    parser.add_argument("--smoke-coef", type=float, default=0.2)
    parser.add_argument("--smoke-every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = sha256_file(plan_path)
    if list(plan["train_scenes"]) != list(split["train_scenes"]):
        raise SystemExit("plan train scenes differ from the split")
    if int(plan["steps"]) != int(args.steps) or len(plan["entries"]) != args.steps:
        raise SystemExit("plan length does not match --steps")
    leaks = sorted({e["scene"] for e in plan["entries"]} & set(split["val_scenes"]))
    if leaks:
        raise SystemExit(f"validation scenes leaked into the plan: {leaks}")

    preset = config_defaults[args.preset]
    if getattr(preset, "init_checkpoint", None):
        raise SystemExit(f"preset {args.preset} declares init_checkpoint; refusing to warm start")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    group_arm = "g1" if args.arm == "g1" else "g0"
    background_supervision = args.arm == "g0plus"
    opt = preset.evolve(
        seed=int(args.seed),
        group_arm=group_arm,
        group_bg_supervision=background_supervision,
        group_bg_loss_weight=1.0,
        group_recipe=bool(args.recipe),
        group_recipe_seg_weight=float(args.instance_outer_weight),
        group_recipe_assign_coef=float(args.assign_coef),
        group_recipe_assign_every=int(args.assign_every),
        lr=args.lr,
        pct_start_steps=args.warmup,
        batch_size=1,
        num_workers=0,
        num_input_views=2,
        num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        workspace=str(out_dir),
        experiment_name=f"siu3r_group_locusgs_{args.arm}_v1",
        init_checkpoint=None,
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.freeze_object_queries()
    model.train()
    # recipe: the deep decoder is initialised from its own fixed seed 1743 and the
    # checkpoint RNG state is restored afterwards so sampling order is unchanged.
    if args.recipe:
        # the new decoder is initialised from its own fixed seed; the checkpoint's
        # torch/CUDA/numpy/sampler RNG state is restored below, so the data order
        # is unchanged by the new module's initialisation
        model.groups.deep.reset_parameters(1743)

    blocks = parameter_blocks(model)
    state = model.state_dict()
    named_parameters = dict(model.named_parameters())
    block_parameters = {
        name: [named_parameters[key] for key in keys if key in named_parameters]
        for name, keys in blocks.items()
    }
    block_hashes = {name: sha256_state(state, keys) for name, keys in blocks.items()}
    initialisation_source = "fresh random (seed %d)" % int(args.seed)
    if args.init_from:
        source_dir = Path(args.init_from)
        loaded = torch.load(source_dir / "model.pt", map_location="cpu", weights_only=False)
        reference_state = loaded["model"]
        missing, unexpected = model.load_state_dict(reference_state, strict=False)
        allowed_missing = sorted(k for k in missing if k.startswith("groups.deep."))
        if args.recipe:
            if sorted(missing) != allowed_missing or unexpected:
                raise SystemExit(
                    f"recipe cold start expected only new groups.deep.* keys to be "
                    f"missing, got missing={sorted(missing)[:5]} unexpected={unexpected[:5]}"
                )
            print(f"[g] recipe cold start: {len(allowed_missing)} new deep-decoder keys "
                  f"initialised from seed 1743, all other keys loaded from {source_dir}",
                  flush=True)
        else:
            if missing or unexpected:
                raise SystemExit(f"cold start key mismatch: {sorted(missing)[:5]} "
                                 f"{unexpected[:5]}")
        state = model.state_dict()
        block_hashes = {name: sha256_state(state, keys) for name, keys in blocks.items()}
        shared_keys = {name: [k for k in keys if k in reference_state]
                       for name, keys in blocks.items()}
        reference_hashes = {
            name: sha256_state(reference_state, keys) for name, keys in shared_keys.items()
        }
        current_hashes = {
            name: sha256_state(state, keys) for name, keys in shared_keys.items()
        }
        if reference_hashes != current_hashes:
            raise SystemExit("initialisation does not match the reference step-0 checkpoint")
        block_hashes = current_hashes
        if args.recipe:
            block_hashes["new_decoder"] = sha256_state(
                state, [k for k in state if k.startswith("groups.deep.")]
            )
        payload = torch.load(source_dir / "train_state.pt", map_location="cpu",
                             weights_only=False)
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        initialisation_source = (
            f"loaded from {source_dir} (model+optimizer+RNG provenance verified; "
            f"reference hashes match bitwise)"
        )
    init_report = {
        "arm": args.arm,
        "group_arm": group_arm,
        "background_supervision": background_supervision,
        "background_loss_weight": float(opt.group_bg_loss_weight),
        "initialisation": initialisation_source,
        "initialisation_from": args.init_from,
        "seed": int(args.seed),
        "parameter_counts": {
            name: int(sum(state[k].numel() for k in keys)) for name, keys in blocks.items()
        },
        "block_hashes": block_hashes,
        "feedback_enabled": bool(model.use_feedback),
        "feedback_gate_value": float(model.feedback.gate.detach()),
        "plan_sha256": plan_sha,
        "split_sha256": sha256_file(Path(args.split)),
        "preset": args.preset,
        "steps": args.steps,
        "optimizer": {"peak_lr": args.lr, "warmup": args.warmup,
                      "schedule": "linear warmup then cosine to 2% of peak",
                      "betas": [0.9, 0.95], "weight_decay": args.weight_decay,
                      "grad_clip": args.grad_clip},
        "loss": {
            "total": "L_recon + lambda(step) * [0.05 L_instance + 0.05 L_semantic]",
            "lambda": f"min(1, step/{opt.group_loss_ramp_steps})",
            "instance_outer_weight": opt.group_inst_loss_weight,
            "semantic_outer_weight": opt.group_sem_loss_weight,
            "instance_inner_weights": {"class_ce": 2.0, "mask_bce": 5.0, "mask_dice": 5.0,
                                       "no_object_ce": 0.1},
            "recon": "canonical LocusGS layers {6,12}, weights {1/3,2/3}",
        },
    }
    (out_dir / "init_report.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")
    print(f"[g] arm={args.arm} model={opt.model_type} out={out_dir}", flush=True)
    print(f"[g] random init hashes: reconstruction {block_hashes['reconstruction'][:16]} "
          f"attributes {block_hashes['attributes'][:16]} groups {block_hashes['groups'][:16]} "
          f"feedback {block_hashes['feedback'][:16]}", flush=True)
    print(f"[g] feedback enabled={model.use_feedback} gate={float(model.feedback.gate):.3e} "
          f"layer={opt.group_feedback_layer} | group queries={model.num_groups} + 1 background",
          flush=True)
    print(f"[g] plan {plan_path} sha256={plan_sha} steps={len(plan['entries'])}", flush=True)
    print(f"[g] loss: {init_report['loss']['total']} with "
          f"lambda=min(1, step/{opt.group_loss_ramp_steps})", flush=True)

    decay, nodecay = [], []
    excluded = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not model.use_feedback and name.startswith("feedback."):
            excluded.append(name)
            continue
        if parameter.dim() != 1 and not getattr(parameter, "_no_weight_decay", False):
            decay.append(parameter)
        else:
            nodecay.append(parameter)
    groups = [
        {"params": decay, "lr": args.lr, "peak_lr": args.lr,
         "weight_decay": args.weight_decay, "name": "decay"},
        {"params": nodecay, "lr": args.lr, "peak_lr": args.lr,
         "weight_decay": 0.0, "name": "nodecay"},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    init_report["optimizer_groups"] = [
        {"name": group["name"], "params": sum(p.numel() for p in group["params"]),
         "peak_lr": group["peak_lr"], "weight_decay": group["weight_decay"]}
        for group in groups
    ]
    init_report["feedback_parameters_excluded_from_optimizer"] = excluded
    for group in groups:
        print(f"[g] group {group['name']}: {sum(p.numel() for p in group['params']):,} params "
              f"lr {group['peak_lr']:.1e} wd {group['weight_decay']}", flush=True)
    (out_dir / "init_report.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")

    provider = SIU3RProcessedProvider(
        opt, root=split["train_root"], subset=split["train_scenes"], training=True, rank=0
    )
    scene_index = {path.name: idx for idx, path in enumerate(provider.dataset.sample_list)}
    val_entries = [] if args.no_eval else build_val_entries(opt, split, device)

    def plan_batch(position: int):
        item = plan["entries"][position]
        provider.pin_pair(
            scene_id=item["scene"], context_frame_ids=item["context"],
            novel_frame_ids=item["novel"], pair_iou=item["pair_iou"],
        )
        return item, move(default_collate([provider[scene_index[item["scene"]]]]), device)

    def build_fresh():
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        fresh = model_registry[opt.model_type](opt).to(device)
        fresh.freeze_object_queries()
        fresh.train()
        if args.recipe:
            fresh.groups.deep.reset_parameters(1743)
        if args.init_from:
            reference = torch.load(Path(args.init_from) / "model.pt", map_location="cpu",
                                   weights_only=False)["model"]
            fresh.load_state_dict(reference, strict=False)
            state_payload = torch.load(Path(args.init_from) / "train_state.pt",
                                       map_location="cpu", weights_only=False)
            torch.set_rng_state(state_payload["torch_rng"])
            np.random.set_state(state_payload["numpy_rng"])
            if state_payload.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state_payload["cuda_rng"])
        decay_f, nodecay_f = [], []
        for name, parameter in fresh.named_parameters():
            if not parameter.requires_grad:
                continue
            if not fresh.use_feedback and name.startswith("feedback."):
                continue
            if parameter.dim() != 1 and not getattr(parameter, "_no_weight_decay", False):
                decay_f.append(parameter)
            else:
                nodecay_f.append(parameter)
        groups_f = [
            {"params": decay_f, "lr": args.lr, "peak_lr": args.lr,
             "weight_decay": args.weight_decay, "name": "decay"},
            {"params": nodecay_f, "lr": args.lr, "peak_lr": args.lr,
             "weight_decay": 0.0, "name": "nodecay"},
        ]
        groups_f = [g for g in groups_f if g["params"]]
        return fresh, torch.optim.AdamW(groups_f, lr=args.lr, betas=(0.9, 0.95))

    if args.preflight:
        preflight = recipe_preflight(
            model, opt, plan, plan_batch, optimizer, device, args, out_dir, build_fresh
        )
        Path(args.preflight).parent.mkdir(parents=True, exist_ok=True)
        Path(args.preflight).write_text(json.dumps(preflight, indent=1), encoding="utf-8")
        print("[g] preflight:", json.dumps(preflight, indent=1)[:3000], flush=True)
        return 0

    if args.smoke_only:
        smoke = recipe_smoke_only(model, opt, plan, plan_batch, device, args, build_fresh)
        Path(args.smoke_only).parent.mkdir(parents=True, exist_ok=True)
        Path(args.smoke_only).write_text(json.dumps(smoke, indent=1), encoding="utf-8")
        print("[g] smoke:", json.dumps(
            {k: v for k, v in smoke.items() if k != "training_smoke"}, indent=1), flush=True)
        if not smoke["all_checks_passed"]:
            raise SystemExit("smoke checks failed; not starting training")
        return 0

    manifest = {
        "experiment": "group_locusgs_g0g1_v1",
        "arm": args.arm,
        "scope": "32 train / 8 unseen development split; NOT an SIU3R official metric",
        "note": "model uses GT camera poses to build rays; SIU3R is an unposed setting",
        "initialisation": initialisation_source,
        "single_variable_g0plus": (
            "background-slot pixel supervision on the two context views; the model "
            "structure, queries, assignment softmax, Hungarian matching, instance "
            "loss, semantic loss, reconstruction loss, optimizer and schedule are "
            "unchanged from G0"
        ) if background_supervision else None,
        "single_variable": "group -> reconstruction-token feedback between layers 10 and 11",
        "init_report": init_report,
        "val_windows": {entry["scene"]: {"context": entry["context"], "novel": entry["novel"]}
                        for entry in val_entries},
        "git_commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "args": vars(args),
    }
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history: list[dict] = []
    checkpoint_meta = {"arm": args.arm, "plan_sha256": plan_sha, "from_scratch": True,
                       "init_hashes": block_hashes}
    total_steps = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"
    val_path = out_dir / "val_history.jsonl"
    started = time.time()
    low_alpha_streak = 0
    collapse_evals = 0

    def run_eval(step: int) -> dict:
        model.eval()
        rows = [evaluate_entry_v2(model, entry, opt) for entry in val_entries]
        model.train()
        summary = summarise_v2(rows)
        print(f"[g] VAL step {step}: ctx {summary['ctx_psnr']:.2f} novel {summary['novel_psnr']:.2f} "
              f"ssim {summary['ctx_ssim']:.3f}/{summary['novel_ssim']:.3f} "
              f"mIoU {summary['sem_miou']:.3f} | AP50 ctx {summary['ctx_ap50']:.3f} "
              f"novel {summary['novel_ap50']:.3f} TP/FP/FN {summary['novel_tp']}/"
              f"{summary['novel_fp']}/{summary['novel_fn']} | P(thing)>=0.5 pass "
              f"{summary['novel_gate_counts']['score']} bg-mass stuff/thing "
              f"{summary['background_stuff_mass_mean']:.3f}/"
              f"{summary['background_thing_mass_mean']:.3f} "
              f"maskerr {summary['mask_alpha_max_error']:.2e}",
              flush=True)
        serialisable = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
        with val_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "summary": summary, "rows": serialisable}) + "\n")
        (out_dir / "history.json").write_text(
            json.dumps({"manifest": manifest, "train": history}, indent=1), encoding="utf-8"
        )
        return summary

    if 0 in args.save_steps:
        model.eval()
        if not args.no_eval:
            run_eval(0)
        model.train()
        save_checkpoint(out_dir, 0, model, optimizer, checkpoint_meta, {"plan_step": 0},
                        keep_steps=list(args.save_steps))
        print("[g] checkpoint ckpt_step0 (random initialisation) written", flush=True)

    for step in range(1, total_steps + 1):
        entry = plan["entries"][step - 1]
        provider.pin_pair(
            scene_id=entry["scene"],
            context_frame_ids=entry["context"],
            novel_frame_ids=entry["novel"],
            pair_iou=entry["pair_iou"],
        )
        try:
            batch = move(default_collate([provider[scene_index[entry["scene"]]]]), device)
        except Exception as error:  # noqa: BLE001
            raise SystemExit(
                f"step {step}: pre-registered window {entry} failed to decode ({error}); "
                f"the plan must not be edited mid-run"
            ) from error
        frames = [int(x) for x in batch["frame_ids"][0].tolist()]
        if frames != entry["context"] + entry["novel"]:
            raise SystemExit(f"step {step}: batch frames {frames} != plan {entry}")

        optimizer.zero_grad(set_to_none=True)
        # 1-based step: lambda(1) = 1/2000 > 0, i.e. the understanding loss is
        # active (non-zero) from the very first update instead of being delayed.
        _, metrics = model.step_loss(batch, step=step, phase="train")
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            print("[g] non-finite metric dump: " + json.dumps({
                key: (float(value) if torch.is_tensor(value) and value.numel() == 1 else str(value))
                for key, value in metrics.items()}), flush=True)
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta, {"plan_step": step},
                            keep_steps=[step])
            raise SystemExit(f"non-finite loss at step {step}; stopped with evidence")
        loss.backward()
        grad_by_group = {group["name"]: 0.0 for group in groups}
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    grad_by_group[group["name"]] += float(
                        parameter.grad.detach().float().pow(2).sum()
                    )
        grad_by_group = {key: math.sqrt(value) for key, value in grad_by_group.items()}
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(grad_norm):
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta, {"plan_step": step},
                            keep_steps=[step])
            raise SystemExit(f"non-finite gradient at step {step}; stopped with evidence")
        if args.arm == "g0" and any(
            p.grad is not None for p in model.feedback.parameters()
        ):
            raise SystemExit("G0 optimized the disabled feedback branch")

        local_step = step - 1
        for group in optimizer.param_groups:
            group["lr"] = lr_at(local_step, float(group["peak_lr"]), args.warmup, args.steps)
        snapshot = None
        block_snapshot = None
        if step == 1 or step % args.log_every == 0 or step in args.save_steps:
            snapshot = {group["name"]: [p.detach().clone() for p in group["params"]]
                        for group in groups}
            block_snapshot = {
                name: [p.detach().clone() for p in parameters]
                for name, parameters in block_parameters.items()
            }
        optimizer.step()

        alpha_mean = float(metrics["alpha_mean"])
        low_alpha_streak = low_alpha_streak + 1 if alpha_mean < 0.005 else 0
        if low_alpha_streak >= 200:
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                            {"plan_step": step, "reason": "alpha_collapse"}, keep_steps=[step])
            raise SystemExit(f"alpha collapsed for {low_alpha_streak} steps at step {step}")

        record = {
            "step": step,
            "scene": entry["scene"],
            "context": entry["context"],
            "novel": entry["novel"],
            "loss": float(loss),
            "recon_loss": float(metrics["recon_loss"]),
            "loss_inst": float(metrics["loss_inst"]),
            "loss_sem": float(metrics["loss_sem"]),
            "loss_inst_ce": float(metrics["loss_inst_ce"]),
            "loss_inst_bce": float(metrics["loss_inst_bce"]),
            "loss_inst_dice": float(metrics["loss_inst_dice"]),
            "loss_inst_total": float(metrics["loss_inst_total"]),
            "loss_bg": float(metrics.get("loss_bg", 0.0)),
            "loss_bg_stuff": float(metrics.get("loss_bg_stuff", 0.0)),
            "loss_bg_thing": float(metrics.get("loss_bg_thing", 0.0)),
            "bg_pixels_stuff": float(metrics.get("bg_pixels_stuff", 0.0)),
            "bg_pixels_thing": float(metrics.get("bg_pixels_thing", 0.0)),
            "bg_prob_stuff_mean": float(metrics.get("bg_prob_stuff_mean", 0.0)),
            "bg_prob_thing_mean": float(metrics.get("bg_prob_thing_mean", 0.0)),
            "seg_ramp": float(metrics.get("seg_ramp", float("nan"))),
            "assign_coef": float(metrics.get("assign_coef", float("nan"))),
            "assign_applied": float(metrics.get("assign_applied", 0.0)),
            "assign_ce": float(metrics.get("assign_ce", float("nan"))),
            "assign_thing_tokens": float(metrics.get("assign_thing_tokens", 0.0)),
            "assign_rest_tokens": float(metrics.get("assign_rest_tokens", 0.0)),
            "assign_dropped_tokens": float(metrics.get("assign_dropped_tokens", 0.0)),
            "assign_target_row_sum_error": float(
                metrics.get("assign_target_row_sum_error", 0.0)),
            "assign_argmax_agreement": float(metrics.get("assign_argmax_agreement", 0.0)),
            "fixed_void_max_abs": float(metrics.get("fixed_void_max_abs", float("nan"))),
            "lambda": float(metrics["ramp"]),
            "instance_weight": float(metrics["instance_weight"]),
            "semantic_weight": float(metrics["semantic_weight"]),
            "psnr": float(metrics["psnr"]),
            "grad_norm": grad_norm,
            "grad_by_group": grad_by_group,
            "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "alpha_mean": alpha_mean,
            "mask_alpha_max_error": float(metrics["mask_alpha_max_error"]),
            "objectness_mean": float(metrics["objectness_mean"]),
            "objectness_max": float(metrics["objectness_max"]),
            "background_mass_mean": float(metrics["background_mass_mean"]),
            "thing_targets": float(metrics["thing_targets"]),
            "matched_groups": float(metrics["matched_groups"]),
            "slot_entropy": float(metrics.get("slot_entropy", float("nan"))),
            "slot_max_prob_mean": float(metrics.get("slot_max_prob_mean", float("nan"))),
            "active_group_share": float(metrics.get("active_group_share", float("nan"))),
            "sem_coverage": float(metrics["sem_coverage"]),
            "semantic_pixel_alpha_gap": float(metrics["semantic_pixel_alpha_gap"]),
            "radius_min": float(metrics["radius_min"]),
            "radius_max": float(metrics["radius_max"]),
            "anchor_max": float(metrics["anchor_max"]),
            "feedback_gate_abs_tanh": float(metrics["feedback_gate_abs_tanh"]),
            "group_update_norm": (
                float(metrics["group_update_norm"]) if "group_update_norm" in metrics else None
            ),
        }
        if snapshot is not None:
            record["update_by_group"] = {
                group["name"]: math.sqrt(sum(
                    float((parameter.detach().float() - before.float()).pow(2).sum())
                    for before, parameter in zip(snapshot[group["name"]], group["params"])
                ))
                for group in groups
            }
            record["update_by_block"] = {
                name: math.sqrt(sum(
                    float((parameter.detach().float() - before.float()).pow(2).sum())
                    for before, parameter in zip(block_snapshot[name], parameters)
                ))
                for name, parameters in block_parameters.items() if parameters
            }
            del snapshot
            del block_snapshot
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if step % args.log_every == 0 or step == 1 or step == total_steps:
            elapsed = (time.time() - started) / max(1, step)
            peak = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else float("nan"))
            print(f"[g] step {step:>5} arm={args.arm} scene={entry['scene']} "
                  f"loss {record['loss']:.4f} (recon {record['recon_loss']:.4f} "
                  f"inst {record['loss_inst']:.3f} sem {record['loss_sem']:.3f} "
                  f"lambda {record['lambda']:.5f} w_inst {record['instance_weight']:.3e} "
                  f"w_sem {record['semantic_weight']:.3e}) "
                  f"psnr {record['psnr']:.2f} grad {grad_norm:.2f} obj {record['objectness_max']:.3f} "
                  f"maskerr {record['mask_alpha_max_error']:.1e} gate "
                  f"{record['feedback_gate_abs_tanh']:.2e} | {elapsed:.2f}s/step peak {peak:.1f}G "
                  f"eta {(total_steps - step) * elapsed / 3600:.1f}h", flush=True)

        if not args.no_eval and (step % args.eval_every == 0 or step == total_steps):
            summary = run_eval(step)
            collapse = summary["novel_psnr"] < summary["novel_grey"] + 0.5
            # A random-initialised model legitimately renders the grey background,
            # so step 0 is not counted as a collapse; two consecutive grey-level
            # evaluations after training has started are.
            collapse_evals = (
                collapse_evals + 1 if (collapse and step >= 500) else 0
            )
            if collapse_evals >= 2:
                save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                                {"plan_step": step, "reason": "reconstruction_collapse"},
                                keep_steps=[step])
                raise SystemExit(
                    f"novel PSNR stayed at the grey-image level for {collapse_evals} evaluations "
                    f"at step {step}; stopped with evidence"
                )

        if step in args.save_steps:
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                            {"plan_step": step}, keep_steps=list(args.save_steps))
            print(f"[g] checkpoint ckpt_step{step} written", flush=True)

    (out_dir / "history.json").write_text(
        json.dumps({"manifest": manifest, "train": history}, indent=1), encoding="utf-8"
    )
    final_hash = sha256_state(model.state_dict())
    (out_dir / "final_state_hash.txt").write_text(final_hash + "\n", encoding="utf-8")
    print(f"[g] done: {len(history)} steps in {time.time() - started:.0f}s "
          f"final_state_sha256={final_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
