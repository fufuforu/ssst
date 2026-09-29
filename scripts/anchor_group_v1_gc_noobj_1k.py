#!/usr/bin/env python3
"""Paired 1k causal ablation for unmatched no-object classifier CE.

The architecture, Hungarian objective, optimizer, schedule and GC backward are
reused from the registered V1-GC implementation.  The only loss switch is the
unmatched no-object CE contribution (1.0 control / 0.0 ablation).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)

from scripts import anchor_group_v1_gc as gc_v1
from scripts import anchor_group_v1_endpoint_audit as endpoint_audit
from scripts.anchor_group_v1 import (
    MANIFEST, MONITORS, PLAN, PRETRAINED, PRETRAINED_SHA, SOURCE_REPORTS,
    TOTAL_STEPS, build_optimizer, build_options, evaluate_anchor_group_all,
    load_state, make_model, set_optimizer_lr, sha256, write_json,
)
from scripts.anchor_group_v1_gc import backward_gradient_controlled
from scripts.anchor_group_v1_endpoint_audit import _concentration, run_query_scope
from scripts.instance_state_generalization import _batch_for, _seen_classes
from scripts.instance_state_runtime import capture_rng, restore_rng
from tokengs.models.anchor_group_loss import (
    CLASS_CE_WEIGHT, NO_OBJECT_INDEX, UNMATCHED_CLASS_WEIGHT,
    anchor_group_losses, unified_hungarian,
)
from tokengs.models.instance_state_loss import _flat_regions

OUT = REPO / "group_plus/anchor_group_v1_gc_noobj_1k"
WORK = REPO / "workspace_group_plus/anchor_group_v1_gc_noobj_1k"
EXPECTED_MANIFEST_SHA = "1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483"
EXPECTED_PLAN_SHA = "ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323"
SCALE = 0.01
ARM_INFO = {
    "control": {"recipe": "ANCHOR_GROUP_V1_GC_CONTROL_1K", "scale": 1.0},
    "ablation": {"recipe": "ANCHOR_GROUP_V1_GC_NO_UNMATCHED_NOOBJ_1K", "scale": 0.0},
}
EVAL_STEPS = (0, 200, 500, 1000)
OPTIMIZER_EXPECTED = {
    "anchor_group_decay": (18, 2743496),
    "anchor_group_nodecay": (41, 41800),
    "reconstruction_decay": (115, 218773504),
    "reconstruction_nodecay": (335, 1229116),
}
PAIR_INIT_DIGEST = None


def _all_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _state_equal(a, b):
    ka, kb = set(a), set(b)
    mismatches = sorted(ka ^ kb)
    max_diff = 0.0
    if mismatches:
        return False, len(ka & kb), mismatches, None
    for key in sorted(ka):
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            mismatches.append(key)
            continue
        if not torch.equal(a[key].detach().cpu(), b[key].detach().cpu()):
            mismatches.append(key)
            if a[key].shape == b[key].shape and a[key].is_floating_point():
                max_diff = max(max_diff, float((a[key].detach().cpu().float() - b[key].detach().cpu().float()).abs().max()))
    return not mismatches, len(ka), mismatches, max_diff


def _state_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8")); h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _asset_audit():
    if sha256(PRETRAINED) != PRETRAINED_SHA:
        raise RuntimeError("locked pretrained checkpoint SHA mismatch")
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("locked manifest SHA mismatch")
    if sha256(PLAN) != EXPECTED_PLAN_SHA:
        raise RuntimeError("locked 5000-entry plan SHA mismatch")
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    windows, entries = manifest.get("windows", []), plan.get("entries", [])
    if len(windows) != 1024 or len({x["scene"] for x in windows}) != 128:
        raise RuntimeError("expected locked 128-scene/1024-window manifest")
    if len(entries) != 5000 or [int(x.get("step", -1)) for x in entries] != list(range(1, 5001)):
        raise RuntimeError("locked plan must contain steps 1..5000")
    for entry in entries:
        win = windows[int(entry["window_index"])]
        if any(entry.get(k) != win.get(k) for k in ("scene", "context", "novel")):
            raise RuntimeError(f"plan/manifest mismatch at step {entry['step']}")
    monitor = {}
    old_gc = REPO / "group_plus/anchor_group_v1_gc"
    for name in MONITORS:
        src = SOURCE_REPORTS / name
        v1copy = REPO / "group_plus/anchor_group_v1" / name
        gccopy = old_gc / name
        hashes = [sha256(src), sha256(v1copy), sha256(gccopy)]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"monitor source/V1/GC SHA mismatch: {name}: {hashes}")
        monitor[name] = {"sha256": hashes[0], "source_v1_gc_byte_identical": True}
    return manifest, plan, monitor


def _new_model(device, scale, source_state=None):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    opt = build_options()
    # Options permits runtime fields. It is deliberately not a model/loss
    # preset option: only anchor_group_losses reads this single ablation knob.
    opt.anchor_group_unmatched_noobj_scale = float(scale)
    if source_state is None:
        source_state = load_state(PRETRAINED)
    model, transfer = make_model(opt, device, source_state)
    if model.architecture_name != "LOCUSGS_ANCHOR_GROUP_V1":
        raise RuntimeError("unexpected architecture identity")
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("paired experiment requires every model parameter trainable")
    return model, opt, transfer


def _manual_ce_parts(pred, batch):
    targets, pairs = unified_hungarian(pred, batch)
    if len(pairs) != 1:
        raise RuntimeError("registered causal contracts require B=1")
    qi, ki = pairs[0]
    target = torch.full((100,), NO_OBJECT_INDEX, device=pred["gaussians"].device, dtype=torch.long)
    if qi.numel():
        target[qi] = targets["gt_classes"][0][ki] - 2
    logits = pred["thing_class_logits"][:, :, 2:][0].float()
    class_weight = torch.ones(19, device=logits.device, dtype=logits.dtype)
    class_weight[NO_OBJECT_INDEX] = UNMATCHED_CLASS_WEIGHT
    per_query = F.cross_entropy(logits, target, weight=class_weight, reduction="none")
    denominator = class_weight[target].sum()
    matched = torch.zeros(100, device=logits.device, dtype=torch.bool)
    matched[qi] = True
    ce_matched = per_query[matched].sum() / denominator
    ce_unmatched = per_query[~matched].sum() / denominator
    old_ce = F.cross_entropy(logits, target, weight=class_weight, reduction="mean")
    return targets, pairs, ce_matched, ce_unmatched, old_ce


def _manual_pixel_parts(pred, batch, targets, pairs):
    sem = batch["semantic_label_all"][:, :2].long()
    ins = batch["instance_label_all"][:, :2].long()
    region = _flat_regions(pred["region_mass"][:, :, :100], 100)
    qi, ki = pairs[0]
    valid = ((sem[0] >= 0) & (sem[0] <= 19) & ((sem[0] < 2) | (ins[0] > 0))).reshape(-1)
    z = torch.logit(region[0, :, valid].clamp(1e-6, 1.0 - 1e-6))
    y = targets["gt_pixel_masks"][0].flatten(1)[:, valid].float()
    if qi.numel():
        bce = F.binary_cross_entropy_with_logits(z[qi], y[ki], reduction="none").mean()
        p = torch.sigmoid(z[qi])
        inter = (p * y[ki]).sum(1)
        denom = p.sum(1) + y[ki].sum(1)
        dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
    else:
        zero = pred["gaussians"].sum() * 0.0
        bce = dice = zero
    return bce, dice


def _pair_list(pairs):
    return [[(int(q), int(k)) for q, k in zip(qi.tolist(), ki.tolist())]
            for qi, ki in pairs]


def _gradient_parity(reference, actual):
    max_abs = 0.0
    diff_sq = ref_sq = 0.0
    for a, b in zip(reference, actual):
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        d = a.double() - b.double()
        max_abs = max(max_abs, float(d.abs().max()))
        diff_sq += float(d.square().sum())
        ref_sq += float(a.double().square().sum())
    return {"max_abs_diff": max_abs,
            "relative_l2_diff": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-30)}


def _cpu_grads(values):
    return [None if g is None else g.detach().to("cpu", copy=True) for g in values]


def _run_regressions(device):
    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ag_noobj_contracts_") as tmp:
        tmp = Path(tmp)
        import scripts.check_anchor_group_v1_contract as anchor_contract
        anchor_contract.OUT = tmp / "phase_a_contracts.json"
        if anchor_contract.main() != 0:
            raise RuntimeError("Phase-A CPU contract returned nonzero")
        anchor = json.loads(anchor_contract.OUT.read_text())
        loss_path = tmp / "legacy_loss_contract.json"
        eval_path = tmp / "legacy_eval_contract.json"
        import scripts.check_instance_state_loss_contract as loss_contract
        old_argv = sys.argv
        try:
            sys.argv = [str(REPO / "scripts/check_instance_state_loss_contract.py"),
                        "--out", str(loss_path)]
            if loss_contract.main() != 0:
                raise RuntimeError("legacy loss CPU contract returned nonzero")
            import scripts.check_instance_state_eval_contract as eval_contract
            sys.argv = [str(REPO / "scripts/check_instance_state_eval_contract.py"),
                        "--out", str(eval_path)]
            if eval_contract.main() != 0:
                raise RuntimeError("legacy evaluator CPU contract returned nonzero")
        finally:
            sys.argv = old_argv
        loss = json.loads(loss_path.read_text()); evaluator = json.loads(eval_path.read_text())
        import scripts.check_anchor_group_v1_gc_contract as gc_contract_script
        if gc_contract_script.main() != 0:
            raise RuntimeError("V1-GC contract checker returned nonzero")

    # Run the real S0/S1 GPU forward regression while redirecting its artifact
    # into this experiment's comparison area; previous result directories stay read-only.
    import scripts.anchor_group_v1_legacy_forward_regression as legacy_forward
    legacy_forward.OUT = comp / "s0_s1_forward_regression.json"
    rc = legacy_forward.main()
    legacy = json.loads(legacy_forward.OUT.read_text())
    phase_a_pass = anchor.get("passed") == 18 and anchor.get("failed") == 0
    gc_pass = (json.loads((REPO / "group_plus/anchor_group_v1_gc/gc_contracts.json").read_text()).get("passed") == 15)
    result = {
        "phase_a_cpu": {"passed": anchor.get("passed"), "failed": anchor.get("failed"),
                        "total": len(anchor.get("checks", [])), "status": "pass" if phase_a_pass else "fail"},
        "legacy_loss_cpu": {"ok": loss.get("ok"), "check_count": len(loss.get("checks", []))},
        "legacy_evaluator_cpu": {"ok": evaluator.get("ok"), "check_count": len(evaluator.get("checks", []))},
        "v1_gc_contracts": {"passed": 15 if gc_pass else None, "total": 15, "status": "pass" if gc_pass else "fail"},
        "s0_s1_gpu_forward": legacy,
        "gpu": torch.cuda.get_device_name(device),
    }
    result["status"] = "pass" if phase_a_pass and loss.get("ok") and evaluator.get("ok") and gc_pass and rc == 0 and legacy.get("status") == "pass" else "fail"
    write_json(comp / "pretrain_regression_audit.json", result)
    if result["status"] != "pass":
        raise RuntimeError(f"pretrain regression gate failed: {result}")
    return result


def _run_pretrain_contracts(device):
    OUT.mkdir(parents=True, exist_ok=True)
    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    manifest, plan, monitor = _asset_audit()
    source_state = load_state(PRETRAINED)
    model, opt, transfer = _new_model(device, 1.0, source_state)
    optimizer, optimizer_audit = build_optimizer(model)
    model.eval()
    entry = plan["entries"][999]
    if int(entry["step"]) != 1000:
        raise RuntimeError("step1000 locked plan entry mismatch")
    batch = _batch_for(opt, entry, device)
    fresh_parity = gc_v1._reconstruction_parity(model, opt, source_state, device, batch)
    write_json(comp / "pretrain_reconstruction_parity.json", fresh_parity)
    if (fresh_parity.get("status") != "pass"
            or fresh_parity.get("gaussian_max_abs_diff") != 0
            or fresh_parity.get("rgb_max_abs_diff") != 0
            or fresh_parity.get("psnr_abs_diff") != 0):
        raise RuntimeError(f"fresh reconstruction parity failed: {fresh_parity}")
    output, control_metrics = model.step_loss(batch, step=1000, coupled=False,
                                             understanding_weight_override=1.0)
    pred = output["prediction"]
    opt0 = build_options()
    opt0.anchor_group_unmatched_noobj_scale = 0.0
    targets, pairs, ce_m, ce_u, old_ce = _manual_ce_parts(pred, batch)
    pixel_bce, pixel_dice = _manual_pixel_parts(pred, batch, targets, pairs)
    ablation_loss, ablation_metrics = anchor_group_losses(pred, batch, opt0)
    control_loss = control_metrics["loss_understanding"]
    control_ce_new = float(control_metrics["thing_ce"])
    # Independent old-formula reference: the scale-zero loss contains the
    # unchanged terms and matched CE; restore the old full weighted CE by
    # adding its unmatched contribution on this same graph.
    old_understanding_reference = ablation_loss + (0.1 * CLASS_CE_WEIGHT) * ce_u
    scalar_ce_diff = abs(float((control_ce_new - old_ce).detach()))
    # The default option and explicit control both use the exact legacy mean
    # reduction.  Compare those totals directly; the independently rebuilt
    # old-formula value is covered by scalar CE and gradient contracts below.
    default_loss, default_metrics = anchor_group_losses(pred, batch, build_options())
    scalar_under_diff = abs(float((default_loss - control_loss).detach()))
    ablation_delta = control_loss - ablation_loss
    expected_removed = 0.1 * CLASS_CE_WEIGHT * ce_u
    ablation_identity_error = abs(float((ablation_delta - expected_removed).detach()))
    classifier_ce_removed = (float(control_metrics["thing_ce"])
                             - float(ablation_metrics["thing_ce"]))
    classifier_ce_identity_error = abs(classifier_ce_removed - float(ce_u.detach()))

    qfinal = pred["states"][-1]["q"]
    named = dict(model.named_parameters())
    selected = (qfinal, model.anchor_group.query_init,
                named["anchor_group.thing_classifier.weight"],
                named["anchor_group.thing_classifier.bias"])
    old_grads = torch.autograd.grad(old_understanding_reference, selected,
                                    retain_graph=True, allow_unused=True)
    new_grads = torch.autograd.grad(control_loss, selected,
                                    retain_graph=True, allow_unused=True)
    grad_parity = _gradient_parity(_cpu_grads(old_grads), _cpu_grads(new_grads))

    c_pairs_a = _pair_list(unified_hungarian(pred, batch)[1])
    c_pairs_b = _pair_list(unified_hungarian(pred, batch)[1])
    cls_diff = abs(float(control_metrics["thing_ce_matched"] -
                         ablation_metrics["thing_ce_matched"]))
    unchanged_keys = ("loss_anchor_group", "anchor_ce", "anchor_dice",
                      "loss_stuff_2d", "loss_semantic", "loss_identity")
    unchanged = {k: abs(float(control_metrics[k] - ablation_metrics[k])) for k in unchanged_keys}
    pixel_component_diffs = {
        "pixel_bce": abs(float(control_metrics["pixel_bce"] - ablation_metrics["pixel_bce"])),
        "pixel_dice": abs(float(control_metrics["pixel_dice"] - ablation_metrics["pixel_dice"])),
    }
    direct_control = 0.1 * CLASS_CE_WEIGHT * ce_u
    direct_control_grad = torch.autograd.grad(direct_control, selected,
                                              retain_graph=True, allow_unused=True)
    loss_diff_grad = torch.autograd.grad(control_loss - ablation_loss, selected,
                                         retain_graph=True, allow_unused=True)
    direct_grad_parity = _gradient_parity(_cpu_grads(direct_control_grad),
                                          _cpu_grads(loss_diff_grad))

    lr_rows = {}
    for step in (0, 1, 200, 500, 1000):
        set_optimizer_lr(optimizer, step)
        lr_rows[str(step)] = {g["name"]: float(g["lr"]) for g in optimizer.param_groups}
    observed_groups = {g["name"]: (len(g["params"]), sum(p.numel() for p in g["params"]))
                       for g in optimizer.param_groups}
    if observed_groups != OPTIMIZER_EXPECTED:
        raise RuntimeError(f"optimizer groups changed: {observed_groups}")

    regression = json.loads((comp / "pretrain_regression_audit.json").read_text())
    phase_a = regression["phase_a_cpu"]
    gc_contract = json.loads((REPO / "group_plus/anchor_group_v1_gc/gc_contracts.json").read_text())
    if phase_a.get("passed") != 18 or phase_a.get("total") != 18:
        raise RuntimeError("Phase-A contract artifact is not 18/18")
    if gc_contract.get("passed") != 15 or gc_contract.get("total") != 15:
        raise RuntimeError("V1-GC contract artifact is not 15/15")
    old_parity = fresh_parity
    legacy = regression["s0_s1_gpu_forward"]

    pairs_ok = c_pairs_a == c_pairs_b
    nonclass_ok = (max(unchanged.values(), default=0.0) == 0.0
                   and max(pixel_component_diffs.values(), default=0.0) == 0.0)
    checks = {
        "NOOBJ-C1_baseline_scalar_parity": {"passed": scalar_ce_diff == 0.0 and scalar_under_diff == 0.0,
            "thing_ce_max_abs_diff": scalar_ce_diff, "loss_understanding_max_abs_diff": scalar_under_diff},
        "NOOBJ-C2_baseline_gradient_parity": {"passed": grad_parity["max_abs_diff"] <= 1e-6 and grad_parity["relative_l2_diff"] <= 1e-6,
            **grad_parity, "parameters": ["q_final", "anchor_group.query_init", "thing_classifier.weight", "thing_classifier.bias"]},
        "NOOBJ-C3_ablation_scalar_difference_identity": {"passed": ablation_identity_error <= 1e-7
            and classifier_ce_identity_error <= 1e-7,
            "control_minus_ablation": float(ablation_delta.detach()),
            "weighted_unmatched_noobj_understanding": float(expected_removed.detach()),
            "abs_diff": ablation_identity_error,
            "classifier_ce_control_minus_ablation": classifier_ce_removed,
            "classifier_ce_unmatched_component": float(ce_u.detach()),
            "classifier_ce_abs_diff": classifier_ce_identity_error,
            "direct_logit_gradient_max_abs_diff": direct_grad_parity["max_abs_diff"],
            "direct_logit_gradient_relative_l2_diff": direct_grad_parity["relative_l2_diff"]},
        "NOOBJ-C4_matched_ce_unchanged": {"passed": cls_diff == 0.0, "max_abs_diff": cls_diff},
        "NOOBJ-C5_nonclass_losses_unchanged": {"passed": nonclass_ok,
            "unchanged_loss_diffs": unchanged, "pixel_bce": float(pixel_bce.detach()),
            "pixel_dice": float(pixel_dice.detach()),
            "pixel_component_diffs": pixel_component_diffs,
            "pixel_bce_and_dice_exact": nonclass_ok},
        "NOOBJ-C6_hungarian_pairs_unchanged": {"passed": pairs_ok, "control_pairs": c_pairs_a, "ablation_pairs": c_pairs_b},
        "NOOBJ-C7_gc_backward_unchanged": {"passed": "backward_gradient_controlled" in gc_v1.__dict__
            and gc_v1.SHARED_UNDERSTANDING_GRAD_SCALE == SCALE,
            "shared_scale": gc_v1.SHARED_UNDERSTANDING_GRAD_SCALE,
            "helper": "scripts.anchor_group_v1_gc.backward_gradient_controlled"},
        "NOOBJ-C8_optimizer_groups_unchanged": {"passed": observed_groups == OPTIMIZER_EXPECTED,
            "groups": {k: {"tensor_count": v[0], "numel": v[1]} for k, v in observed_groups.items()}},
        "NOOBJ-C9_lr_schedule_unchanged": {"passed": all(float(row["anchor_group_decay"]) == float(row["anchor_group_nodecay"])
            and float(row["reconstruction_decay"]) == float(row["reconstruction_nodecay"])
            for row in lr_rows.values()), "lr_by_step": lr_rows, "total_steps": TOTAL_STEPS},
        "NOOBJ-C10_phase_a_regression": {"passed": phase_a.get("passed") == 18 and phase_a.get("total") == 18,
            "passed_count": phase_a.get("passed"), "total": phase_a.get("total")},
        "NOOBJ-C11_v1_gc_regression": {"passed": gc_contract.get("passed") == 15 and gc_contract.get("total") == 15,
            "passed_count": gc_contract.get("passed"), "total": gc_contract.get("total")},
        "NOOBJ-C12_reconstruction_and_legacy_regression": {"passed": old_parity.get("gaussian_max_abs_diff") == 0
            and old_parity.get("rgb_max_abs_diff") == 0 and old_parity.get("psnr_abs_diff") == 0
            and legacy.get("status") == "pass",
            "reconstruction_parity": old_parity, "legacy_s0_s1": legacy.get("status")},
    }
    checks["NOOBJ-C1_baseline_scalar_parity"]["default_scale_exact"] = (
        default_metrics["unmatched_noobj_scale"] == 1.0
        and abs(float(default_loss.detach() - control_loss.detach())) == 0.0)
    checks["NOOBJ-C1_baseline_scalar_parity"]["passed"] &= checks["NOOBJ-C1_baseline_scalar_parity"]["default_scale_exact"]

    mismatches = [k for k, v in checks.items() if not v["passed"]]
    result = {
        "architecture": "LOCUSGS_ANCHOR_GROUP_V1",
        "parent_recipe": "ANCHOR_GROUP_V1_GC_ALPHA001",
        "arm_scales": {k: v["scale"] for k, v in ARM_INFO.items()},
        "checkpoint_sha256": PRETRAINED_SHA,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "parent_plan_sha256": EXPECTED_PLAN_SHA,
        "plan_prefix_length": 1000,
        "monitor_sha256": monitor,
        "transfer": transfer,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "checks": checks, "passed": len(checks) - len(mismatches),
        "total": len(checks), "failed": mismatches,
        "status": "pass" if not mismatches else "fail",
    }
    write_json(comp / "pretrain_contracts.json", result)
    if mismatches:
        raise RuntimeError(f"pretrain contracts failed: {mismatches}")
    del model, optimizer, batch, output, pred, control_metrics, ablation_loss
    gc.collect(); torch.cuda.empty_cache()
    return result


def _snapshot_named(model):
    names = ("anchor_group.query_init", "anchor_decoder.mu")
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if n in names or n.startswith("enc_dec_backbone.decoder_blocks.11.")
            or n.startswith("activation_head.")}


def _prefix_delta(before, after, prefix):
    vals = [float((p.detach() - before[n].to(p.device)).abs().max())
            for n, p in after.items() if n == prefix or n.startswith(prefix)]
    return max(vals, default=0.0)


def _run_pair_smoke(device):
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("paired one-step smoke requires RTX 3090")
    if torch.cuda.get_device_properties(device).total_memory < 23 * 1024**3:
        raise RuntimeError("paired one-step smoke requires 24GB RTX 3090")
    contract = json.loads((OUT / "comparison/pretrain_contracts.json").read_text())
    if contract.get("status") != "pass" or contract.get("passed") != 12:
        raise RuntimeError("12/12 pretrain contracts must pass before paired smoke")
    source = load_state(PRETRAINED)
    models = {arm: _new_model(device, ARM_INFO[arm]["scale"], source)
              for arm in ("control", "ablation")}
    m0, o0, _ = models["control"]
    m1, _o1, _ = models["ablation"]
    equal, count, mismatch, initial_diff = _state_equal(m0.state_dict(), m1.state_dict())
    if not equal or count != 509:
        raise RuntimeError(f"paired smoke initial model state mismatch: {count}, {mismatch[:8]}")
    plan = json.loads(PLAN.read_text()); entry = plan["entries"][999]
    batch = _batch_for(o0, entry, device)
    optims = {arm: build_optimizer(model) for arm, (model, _opt, _trans) in models.items()}
    records, predictions, metrics_by_arm, smoke_pairs = {}, {}, {}, {}
    snapshots = {}
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for arm in ("control", "ablation"):
        model, opt, _ = models[arm]
        torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.cuda.manual_seed_all(42)
        model.train()
        optimizer, optimizer_audit = optims[arm]
        set_optimizer_lr(optimizer, 1000)
        optimizer.zero_grad(set_to_none=True)
        snapshots[arm] = _snapshot_named(model)
        output, metrics = model.step_loss(batch, step=1000, coupled=False)
        raw_pred = output["prediction"]
        predictions[arm] = {
            "gaussians": raw_pred["gaussians"].detach().cpu().clone(),
            "render": {"images_pred": raw_pred["render"]["images_pred"].detach().cpu().clone()},
            "states": [{"A_post": raw_pred["states"][-1]["A_post"].detach().cpu().clone()}],
            "region_mass": raw_pred["region_mass"].detach().cpu().clone(),
            "semantic_scores": raw_pred["semantic_scores"].detach().cpu().clone(),
            "thing_class_logits": raw_pred["thing_class_logits"].detach().cpu().clone(),
        }
        metrics_by_arm[arm] = {
            k: (float(v.detach()) if torch.is_tensor(v) else v)
            for k, v in metrics.items()
            if k in ("loss", "loss_recon", "loss_understanding", "loss_anchor_group",
                     "weighted_unmatched_noobj_understanding", "thing_ce_matched",
                     "thing_ce_unmatched")
        }
        online_pairs = unified_hungarian(output["prediction"], batch)[1]
        smoke_pairs[arm] = _pair_list(online_pairs)
        current = {int(q) for qi, _ki in online_pairs for q in qi.tolist()}
        if len(current) == 0:
            raise RuntimeError("step1000 smoke unexpectedly has no Hungarian matches")
        if not all(torch.isfinite(metrics[k]).all() for k in ("loss", "loss_recon", "loss_understanding")):
            raise RuntimeError(f"nonfinite smoke loss in {arm}")
        backward_gradient_controlled(model, metrics["loss_recon"], metrics["loss_understanding"],
                                     1.0, shared_scale=SCALE)
        preclip = float(clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        if not math.isfinite(preclip) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError(f"nonfinite paired smoke gradient in {arm}")
        optimizer.step()
        if not _all_finite(optimizer.state):
            raise RuntimeError(f"nonfinite optimizer state after paired smoke {arm}")
        after = dict(model.named_parameters())
        delta = {
            "query_init": float((after["anchor_group.query_init"].detach() - snapshots[arm]["anchor_group.query_init"].to(device)).abs().max()),
            "late_decoder": _prefix_delta(snapshots[arm], after, "enc_dec_backbone.decoder_blocks.11."),
            "anchor_mu": float((after["anchor_decoder.mu"].detach() - snapshots[arm]["anchor_decoder.mu"].to(device)).abs().max()),
            "activation_head": _prefix_delta(snapshots[arm], after, "activation_head."),
        }
        if any(v <= 0 for v in delta.values()):
            raise RuntimeError(f"paired smoke parameter update missing for {arm}: {delta}")
        records[arm] = {
            "recipe": ARM_INFO[arm]["recipe"], "unmatched_noobj_scale": ARM_INFO[arm]["scale"],
            "step": 1000, "understanding_weight": float(metrics["understanding_weight"]),
            "loss": float(metrics["loss"].detach()), "loss_recon": float(metrics["loss_recon"].detach()),
            "loss_understanding": float(metrics["loss_understanding"].detach()),
            "loss_anchor_group": float(metrics["loss_anchor_group"]),
            "online_matched_query_ids": sorted(current),
            "group_lr": next(float(g["lr"]) for g in optimizer.param_groups if g["name"].startswith("anchor_group_")),
            "reconstruction_lr": next(float(g["lr"]) for g in optimizer.param_groups if g["name"].startswith("reconstruction_")),
            "optimizer_groups": optimizer_audit["groups"], "preclip_global_norm": preclip,
            "parameter_delta_max_abs": delta, "optimizer_step_finite": True,
        }
        model.zero_grad(set_to_none=True)
        del output, metrics
        del raw_pred
        gc.collect(); torch.cuda.empty_cache()
    tensor_getters = {
        "gaussians": lambda p: p["gaussians"],
        "rgb": lambda p: p["render"]["images_pred"],
        "A_post": lambda p: p["states"][-1]["A_post"],
        "region_mass": lambda p: p["region_mass"],
        "semantic_scores": lambda p: p["semantic_scores"],
        "thing_class_logits": lambda p: p["thing_class_logits"],
    }
    tensor_parity = {k: {"exact": torch.equal(fn(predictions["control"]), fn(predictions["ablation"])),
                         "max_abs_diff": float((fn(predictions["control"]).float() - fn(predictions["ablation"]).float()).abs().max())}
                     for k, fn in tensor_getters.items()}
    pairs_a = smoke_pairs["control"]
    pairs_b = smoke_pairs["ablation"]
    m_a, m_b = metrics_by_arm["control"], metrics_by_arm["ablation"]
    loss_diff = float(m_a["loss_understanding"] - m_b["loss_understanding"])
    expected_diff = float(m_a["weighted_unmatched_noobj_understanding"])
    result = {
        "gpu": torch.cuda.get_device_name(device), "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "initial_model_state": {"tensor_count": count, "exact": equal, "mismatched_keys": mismatch, "max_abs_diff": initial_diff},
        "same_batch_object": True, "step": 1000,
        "forward_tensor_parity": tensor_parity,
        "forward_tensor_all_exact": all(v["exact"] for v in tensor_parity.values()),
        "hungarian_pairs": {"control": pairs_a, "ablation": pairs_b, "exact": pairs_a == pairs_b},
        "loss_component_parity_and_identity": {
            "control_loss_recon": float(m_a["loss_recon"]),
            "ablation_loss_recon": float(m_b["loss_recon"]),
            "loss_recon_abs_diff": abs(float(m_a["loss_recon"] - m_b["loss_recon"])),
            "control_loss_understanding": float(m_a["loss_understanding"]),
            "ablation_loss_understanding": float(m_b["loss_understanding"]),
            "control_minus_ablation_understanding": loss_diff,
            "weighted_unmatched_noobj_understanding": expected_diff,
            "ablation_identity_abs_error": abs(loss_diff - expected_diff),
            "control_matched_ce": m_a["thing_ce_matched"], "ablation_matched_ce": m_b["thing_ce_matched"],
            "control_unmatched_ce": m_a["thing_ce_unmatched"], "ablation_unmatched_ce": m_b["thing_ce_unmatched"],
        },
        "arms": records,
        "optimizer_step_count": 2,
        "status": "pass" if equal and all(v["exact"] for v in tensor_parity.values()) and pairs_a == pairs_b
            and abs(loss_diff - expected_diff) <= 1e-7 and all(r["optimizer_step_finite"] for r in records.values()) else "fail",
    }
    # Capture GPU memory after both paired one-step updates.
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
    write_json(OUT / "comparison/paired_one_step_smoke.json", result)
    if result["status"] != "pass":
        raise RuntimeError("paired one-step RTX3090 smoke failed")
    return result


def _summary_distribution(values):
    # Callers include generator expressions for the ever/never matched query
    # subsets; materialize once before NumPy conversion.
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "count": 0}
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "count": int(x.size)}


def _manifest_utilization(model, opt, arm, step, device):
    manifest = json.loads(MANIFEST.read_text())
    valmon = json.loads((OUT / arm / "monitor_32pairs.json").read_text())
    train_windows = manifest["windows"]
    val_windows = valmon["pairs"]
    model.eval()
    train_result = run_query_scope(model, opt, train_windows, device, f"{arm}-step{step}-train1024")
    val_result = run_query_scope(model, opt, val_windows, device, f"{arm}-step{step}-val32")

    def add_query_diagnostics(result):
        rows = result["queries"]
        matched = [r for r in rows if r["n_hungarian_matches"] > 0]
        never = [r for r in rows if r["n_hungarian_matches"] == 0]
        all_noobj = [r["mean_no_object_probability"] for r in rows]
        all_maxthing = [r["mean_max_thing_class_probability"] for r in rows]
        summary = result["summary"]
        summary["mean_no_object_probability_across_queries"] = _summary_distribution(all_noobj)
        summary["mean_max_thing_class_probability_across_queries"] = _summary_distribution(all_maxthing)
        summary["ever_matched_no_object_probability"] = _summary_distribution(
            r["mean_no_object_probability"] for r in matched)
        summary["never_matched_no_object_probability"] = _summary_distribution(
            r["mean_no_object_probability"] for r in never)
        summary["ever_matched_max_thing_probability"] = _summary_distribution(
            r["mean_max_thing_class_probability"] for r in matched)
        summary["never_matched_max_thing_probability"] = _summary_distribution(
            r["mean_max_thing_class_probability"] for r in never)
        summary["manifest_wide_utilization"] = True
    add_query_diagnostics(train_result)
    add_query_diagnostics(val_result)
    result = {
        "kind": "manifest_wide_utilization",
        "step": int(step), "arm": arm, "recipe": ARM_INFO[arm]["recipe"],
        "unmatched_noobj_scale": ARM_INFO[arm]["scale"],
        "train1024": train_result["summary"],
        "val32_context": val_result["summary"],
        "train1024_queries": train_result["queries"],
        "val32_queries": val_result["queries"],
    }
    write_json(OUT / arm / f"manifest_utilization_step{step}.json", result)
    model.train()
    return result


def _online_summary(counts, first_steps, scene_sets, step):
    conc = _concentration(counts, fail_zero=True)
    firsted = [x for x in first_steps if x is not None]
    return {
        "step": int(step),
        "total_online_gt_matches": int(np.sum(counts)),
        "unique_queries_ever_matched": int(np.sum(np.asarray(counts) > 0)),
        "never_matched_queries": int(np.sum(np.asarray(counts) == 0)),
        "top1_share": conc["top1_share"], "top5_share": conc["top5_share"],
        "top10_share": conc["top10_share"], "top20_share": conc["top20_share"],
        "match_gini": conc["gini"],
        "normalized_entropy": conc["normalized_entropy"],
        "effective_query_count": conc["effective_query_count"],
        "first_positive_coverage": {
            "by_step200": sum(x is not None and x <= 200 for x in first_steps),
            "by_step500": sum(x is not None and x <= 500 for x in first_steps),
            "by_step1000": sum(x is not None and x <= 1000 for x in first_steps),
            "newly_matched_by_current_step": sum(x == step for x in first_steps),
            "ever_matched": len(firsted),
        },
        "query_rows": [
            {"query_id": q, "online_match_count": int(counts[q]),
             "first_positive_step": first_steps[q],
             "n_distinct_scenes_matched": len(scene_sets[q])}
            for q in range(100)
        ],
        "top20": sorted(
            [{"query_id": q, "online_match_count": int(counts[q]),
              "first_positive_step": first_steps[q],
              "n_distinct_scenes_matched": len(scene_sets[q])} for q in range(100)],
            key=lambda x: (-x["online_match_count"], x["query_id"]))[:20],
    }


def _save_checkpoint(model, optimizer, arm, step, rng):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": int(step), "architecture": "LOCUSGS_ANCHOR_GROUP_V1",
        "recipe": ARM_INFO[arm]["recipe"], "joint": True, "beta": 0,
        "shared_understanding_grad_scale": SCALE,
        "unmatched_noobj_scale": ARM_INFO[arm]["scale"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "parent_plan_sha256": EXPECTED_PLAN_SHA, "plan_prefix_length": 1000,
        "pretrained_sha256": PRETRAINED_SHA, "rng": rng,
    }
    torch.save(payload, WORK / arm / f"checkpoint_step{step}.pt")
    return payload


def _endpoint_audit(payload, model, arm, step):
    optimizer_finite = _all_finite(payload["optimizer"])
    model_finite = _all_finite(payload["model"])
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    req = {
        "step": step, "architecture": "LOCUSGS_ANCHOR_GROUP_V1",
        "recipe": ARM_INFO[arm]["recipe"], "joint": True, "beta": 0,
        "shared_understanding_grad_scale": SCALE,
        "unmatched_noobj_scale": ARM_INFO[arm]["scale"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "parent_plan_sha256": EXPECTED_PLAN_SHA, "plan_prefix_length": 1000,
        "pretrained_sha256": PRETRAINED_SHA,
        "rng_present": isinstance(payload.get("rng"), dict)
            and all(payload["rng"].get(k) is not None for k in ("python", "numpy", "torch", "cuda")),
        "model_tensors_finite": model_finite, "optimizer_tensors_finite": optimizer_finite,
        "trainable_reconstruction_numel": sum(p.numel() for n, p in model.named_parameters()
                                                 if p.requires_grad and not n.startswith("anchor_group.")),
        "trainable_anchor_group_numel": sum(p.numel() for n, p in model.named_parameters()
                                               if p.requires_grad and n.startswith("anchor_group.")),
        "frozen_numel": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "frozen_parameter_names": frozen,
    }
    valid = (req["step"] == 1000 and req["architecture"] == "LOCUSGS_ANCHOR_GROUP_V1"
             and req["recipe"] == ARM_INFO[arm]["recipe"] and req["joint"] is True
             and req["beta"] == 0 and req["shared_understanding_grad_scale"] == SCALE
             and req["unmatched_noobj_scale"] == ARM_INFO[arm]["scale"]
             and req["manifest_sha256"] == EXPECTED_MANIFEST_SHA
             and req["parent_plan_sha256"] == EXPECTED_PLAN_SHA
             and req["pretrained_sha256"] == PRETRAINED_SHA
             and req["rng_present"] and model_finite and optimizer_finite and not frozen)
    req["status"] = "pass" if valid else "fail"
    write_json(OUT / arm / "endpoint_step1000_audit.json", req)
    if not valid:
        raise RuntimeError(f"{arm} endpoint audit failed: {req}")
    return req


def _drift_from_initial(model, initial_state):
    from scripts.anchor_group_v1_gc import _drift_category
    sums = {k: {"tensor_count": 0, "numel": 0, "delta_sq": 0.0,
                "base_sq": 0.0, "max_abs": 0.0}
            for k in ("other_reconstruction", "decoder", "activation_head", "anchor_geometry")}
    for name, param in model.named_parameters():
        if name.startswith("anchor_group."):
            continue
        if name not in initial_state:
            raise RuntimeError(f"missing initialization parameter in drift audit: {name}")
        base = initial_state[name].double()
        delta = param.detach().cpu().double() - base
        item = sums[_drift_category(name)]
        item["tensor_count"] += 1; item["numel"] += param.numel()
        item["delta_sq"] += float(delta.square().sum())
        item["base_sq"] += float(base.square().sum())
        item["max_abs"] = max(item["max_abs"], float(delta.abs().max()))
    categories = {}
    for name, item in sums.items():
        categories[name] = {
            "tensor_count": item["tensor_count"], "numel": item["numel"],
            "l2_delta": math.sqrt(item["delta_sq"]),
            "relative_l2_delta": math.sqrt(item["delta_sq"]) / max(math.sqrt(item["base_sq"]), 1e-30),
            "max_abs_delta": item["max_abs"],
        }
    return {"baseline_checkpoint_sha256": PRETRAINED_SHA, "endpoint_step": 1000,
            "categories": categories, "status": "pass"}


def _copy_monitors(arm):
    (OUT / arm).mkdir(parents=True, exist_ok=True)
    for name in MONITORS:
        src = SOURCE_REPORTS / name
        dst = OUT / arm / name
        shutil.copyfile(src, dst)
        if sha256(src) != sha256(dst):
            raise RuntimeError(f"{arm} monitor copy SHA mismatch: {name}")


def _train_arm(arm, device, manifest, plan, source_state):
    global PAIR_INIT_DIGEST
    arm_out, arm_work = OUT / arm, WORK / arm
    arm_out.mkdir(parents=True, exist_ok=True); arm_work.mkdir(parents=True, exist_ok=True)
    _copy_monitors(arm)
    model, opt, transfer = _new_model(device, ARM_INFO[arm]["scale"], source_state)
    optimizer, optimizer_audit = build_optimizer(model)
    observed = {g["name"]: (len(g["params"]), sum(p.numel() for p in g["params"]))
                for g in optimizer.param_groups}
    if observed != OPTIMIZER_EXPECTED:
        raise RuntimeError(f"{arm} optimizer groups differ from registered V1-GC: {observed}")
    if len(model.state_dict()) != 509 or any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError(f"{arm} fresh model state/trainability mismatch")
    fresh_digest = _state_digest(model.state_dict())
    if PAIR_INIT_DIGEST is None:
        PAIR_INIT_DIGEST = fresh_digest
    elif fresh_digest != PAIR_INIT_DIGEST:
        raise RuntimeError(f"{arm} fresh initialization does not match paired control state")
    initial_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                     if not n.startswith("anchor_group.")}
    initial_rng = capture_rng()
    model.train()
    if optimizer.state:
        raise RuntimeError("fresh paired optimizer must start empty")

    # Registered step-zero evaluation must be read-only and precede training.
    state0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    evaluate_anchor_group_all(model, opt, arm_out, 0, device, _seen_classes(arm_out))
    model.train()
    changed0 = [k for k, v in state0.items() if not torch.equal(v, model.state_dict()[k].detach().cpu())]
    if changed0 or optimizer.state:
        raise RuntimeError(f"{arm} step0 evaluation changed state/optimizer: {changed0[:6]}")
    step0_util = _manifest_utilization(model, opt, arm, 0, device)
    restore_rng(initial_rng)
    del state0

    counts = np.zeros(100, dtype=np.int64)
    first_steps = [None] * 100
    scene_sets = [set() for _ in range(100)]
    online_snapshots = {}
    log_rows = []
    endpoint_audit_result = None
    for entry in plan["entries"][:1000]:
        step = int(entry["step"])
        if step > 1000:
            raise RuntimeError("paired plan prefix escaped step1000")
        model.train()
        set_optimizer_lr(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        batch = _batch_for(opt, entry, device)
        output, metrics = model.step_loss(batch, step=step, coupled=False)
        pred = output["prediction"]
        # The required replay uses this prediction/batch only, no forward and
        # no gradient; it cannot influence loss construction or matching used
        # internally by step_loss.
        with torch.no_grad():
            targets, pairs = unified_hungarian(pred, batch)
        for qi, ki in pairs:
            for q in qi.tolist():
                q = int(q); counts[q] += 1
                if first_steps[q] is None:
                    first_steps[q] = step
                scene_sets[q].add(str(entry["scene"]))
        uweight = float(metrics["understanding_weight"])
        expected_u = 0.0 if step <= 200 else ((step - 200) / 800.0 if step < 1000 else 1.0)
        if uweight != expected_u or metrics["unmatched_noobj_scale"] != ARM_INFO[arm]["scale"]:
            raise RuntimeError(f"{arm} recipe coefficient mismatch at {step}")
        if not all(torch.isfinite(metrics[k]).all() for k in ("loss", "loss_recon", "loss_understanding")):
            raise RuntimeError(f"nonfinite {arm} loss at step {step}")
        backward_gradient_controlled(model, metrics["loss_recon"], metrics["loss_understanding"],
                                     uweight, shared_scale=SCALE)
        preclip = clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        preclip_value = float(preclip)
        clip_coef = min(1.0, 1.0 / (preclip_value + 1e-6))
        optimizer.step()
        if not math.isfinite(preclip_value) or not _all_finite(optimizer.state):
            raise RuntimeError(f"nonfinite {arm} optimizer update at step {step}")
        if step % 100 == 0:
            fields = ("loss", "loss_recon", "loss_understanding", "understanding_weight",
                      "loss_thing_2d", "thing_ce_matched", "thing_ce_unmatched",
                      "weighted_unmatched_noobj_understanding", "anchor_ce", "anchor_dice",
                      "loss_anchor_group", "loss_stuff_2d", "loss_semantic", "loss_identity",
                      "unmatched_noobj_scale")
            row = {k: float(metrics[k].detach()) if torch.is_tensor(metrics[k]) else float(metrics[k])
                   for k in fields}
            row.update({
                "step": step, "event": "train_step", "recipe": ARM_INFO[arm]["recipe"],
                "shared_understanding_grad_scale": SCALE, "preclip_global_norm": preclip_value,
                "clip_coefficient": clip_coef,
                "group_lr": next(float(g["lr"]) for g in optimizer.param_groups if g["name"].startswith("anchor_group_")),
                "reconstruction_lr": next(float(g["lr"]) for g in optimizer.param_groups if g["name"].startswith("reconstruction_")),
            })
            if not _all_finite(row):
                raise RuntimeError(f"nonfinite {arm} train log at step {step}")
            log_rows.append(row)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        del batch, output, pred, metrics, targets, pairs
        if step in (200, 500, 1000):
            online_snapshots[str(step)] = _online_summary(counts.copy(), list(first_steps),
                                                          [set(s) for s in scene_sets], step)
            checkpoint_rng = capture_rng()
            metrics_at_step = evaluate_anchor_group_all(model, opt, arm_out, step,
                                                        device, _seen_classes(arm_out))
            util = _manifest_utilization(model, opt, arm, step, device)
            checkpoint_payload = _save_checkpoint(model, optimizer, arm, step, checkpoint_rng)
            if step == 1000:
                endpoint_audit_result = _endpoint_audit(checkpoint_payload, model, arm, step)
            del checkpoint_payload, metrics_at_step, util
            restore_rng(checkpoint_rng)
            gc.collect(); torch.cuda.empty_cache()
        if step % 100 == 0:
            print(f"[{arm}] completed step {step}/1000", flush=True)

    write_json(arm_out / "online_match_exposure.json", {
        "definition": "online positive exposure from per-step no-grad unified_hungarian replay on the training prediction/batch",
        "plan_prefix_length": 1000, "parent_plan_sha256": EXPECTED_PLAN_SHA,
        "snapshots": online_snapshots,
        "final_first_positive_coverage": {
            str(cut): sum(x is not None and x <= cut for x in first_steps)
            for cut in (200, 500, 1000)},
        "queries": _online_summary(counts, first_steps, scene_sets, 1000)["query_rows"],
        "top20": _online_summary(counts, first_steps, scene_sets, 1000)["top20"],
    })
    drift = _drift_from_initial(model, initial_state)
    write_json(arm_out / "reconstruction_drift_step1000.json", drift)
    log_norms = np.asarray([r["preclip_global_norm"] for r in log_rows], dtype=np.float64)
    clips = np.asarray([r["clip_coefficient"] for r in log_rows], dtype=np.float64)
    if len(log_rows) != 10 or [r["step"] for r in log_rows] != list(range(100, 1001, 100)):
        raise RuntimeError(f"{arm} requires exactly ten finite 100-step log rows")
    summary = {
        "completed_steps": 1000, "logged_train_steps": len(log_rows),
        "logged_steps": [r["step"] for r in log_rows],
        "all_logged_metrics_finite": all(_all_finite(r) for r in log_rows),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "parent_plan_sha256": EXPECTED_PLAN_SHA, "plan_prefix_length": 1000,
        "pretrained_sha256": PRETRAINED_SHA,
        "recipe": ARM_INFO[arm]["recipe"], "shared_understanding_grad_scale": SCALE,
        "unmatched_noobj_scale": ARM_INFO[arm]["scale"],
        "trainable_reconstruction_numel": sum(p.numel() for n, p in model.named_parameters()
                                                 if p.requires_grad and not n.startswith("anchor_group.")),
        "trainable_anchor_group_numel": sum(p.numel() for n, p in model.named_parameters()
                                               if p.requires_grad and n.startswith("anchor_group.")),
        "frozen_numel": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "preclip_norm": {"median": float(np.median(log_norms)),
                         "p10": float(np.percentile(log_norms, 10)),
                         "p90": float(np.percentile(log_norms, 90)),
                         "max": float(np.max(log_norms))},
        "clip_coefficient": {"median": float(np.median(clips)),
                             "fraction_actually_clipped": float(np.mean(clips < 1.0))},
        "fresh_initial_state_sha256": fresh_digest,
        "status": "pass" if all(_all_finite(r) for r in log_rows)
                  and len(log_rows) == 10 and summary_if_true(model) else "fail",
    }
    write_json(arm_out / "training_log_summary.json", summary)
    completed = {"arm": arm, "transfer": transfer, "endpoint_audit": endpoint_audit_result,
                 "drift": drift, "summary": summary, "online": online_snapshots,
                 "util_step0": step0_util}
    del model, optimizer, opt, initial_state
    gc.collect(); torch.cuda.empty_cache()
    return completed


def summary_if_true(model):
    return all(p.requires_grad for p in model.parameters())


def _load_step(path):
    return json.loads(path.read_text())


def _paired_comparison():
    compdir = OUT / "comparison"
    paired_curves, slot = {}, {}
    for step in EVAL_STEPS:
        row = {"step": step, "arms": {}, "v1_gc_reference": {}}
        for arm in ("control", "ablation"):
            curves = _load_step(OUT / arm / f"curves_{step}.json")
            direct = _load_step(OUT / arm / f"anchor_group_diagnostics_step{step}.json")
            ctx, tgt = curves["val32_context"], curves["val32_target"]
            d = direct["val32_context"]
            row["arms"][arm] = {
                "val32_context": ctx, "val32_target": tgt,
                "direct_anchor_context": {
                    k: d[k] for k in ("anchor_ownership_accuracy", "thing_anchor_correct_fraction",
                                      "anchor_group_gt_recall50", "active_thing_queries_mean",
                                      "assignment_entropy_mean", "mechanism_means",
                                      "anchor_valid_count", "anchor_thing_count",
                                      "gt_with_anchor_support", "anchor_group_gt_success50")
                },
            }
            util = _load_step(OUT / arm / f"manifest_utilization_step{step}.json")
            online_file = _load_step(OUT / arm / "online_match_exposure.json") if step != 0 else None
            summary = util["train1024"]
            slot.setdefault(str(step), {})[arm] = {
                "online": online_file["snapshots"].get(str(step)) if online_file else None,
                "manifest_train1024": {
                    "total_gt_matches": summary["total_gt_matches"],
                    "unique_queries_ever_matched": summary["unique_queries_ever_matched"],
                    "queries_never_matched": summary["queries_never_matched"],
                    "top5_share": summary["match_concentration"]["top5_share"],
                    "top10_share": summary["match_concentration"]["top10_share"],
                    "gini": summary["match_concentration"]["gini"],
                    "effective_query_count": summary["match_concentration"]["effective_query_count"],
                    "ownership_gini": summary["ownership_concentration"]["gini"],
                    "mean_active_output_queries": summary["mean_active_output_queries"],
                    "noobj_never": summary["never_matched_no_object_probability"],
                    "max_thing_never_median": summary["never_matched_max_thing_probability"]["median"],
                },
                "manifest_val32": {
                    "total_gt_matches": util["val32_context"]["total_gt_matches"],
                    "unique_queries_ever_matched": util["val32_context"]["unique_queries_ever_matched"],
                    "queries_never_matched": util["val32_context"]["queries_never_matched"],
                    "top5_share": util["val32_context"]["match_concentration"]["top5_share"],
                    "top10_share": util["val32_context"]["match_concentration"]["top10_share"],
                    "gini": util["val32_context"]["match_concentration"]["gini"],
                    "effective_query_count": util["val32_context"]["match_concentration"]["effective_query_count"],
                    "ownership_gini": util["val32_context"]["ownership_concentration"]["gini"],
                },
            }
        v1dir = REPO / "group_plus/anchor_group_v1_gc"
        oldcurves = _load_step(v1dir / f"curves_{step}.json")
        olddiag = _load_step(v1dir / f"anchor_group_diagnostics_step{step}.json")
        row["v1_gc_reference"] = {
            "val32_context": oldcurves["val32_context"],
            "val32_target": oldcurves["val32_target"],
            "direct_anchor_context": {
                k: olddiag["val32_context"][k] for k in (
                    "anchor_ownership_accuracy", "thing_anchor_correct_fraction",
                    "anchor_group_gt_recall50", "active_thing_queries_mean",
                    "assignment_entropy_mean", "mechanism_means")
            },
        }
        for arm in ("control", "ablation"):
            row["arms"][arm]["delta_vs_v1_gc"] = {
                scope: {
                    key: (row["arms"][arm][scope][key] - row["v1_gc_reference"][scope][key]
                          if isinstance(row["arms"][arm][scope].get(key), (int, float))
                          and isinstance(row["v1_gc_reference"][scope].get(key), (int, float)) else None)
                    for key in ("mIoU_thing", "class_agnostic_recall50", "class_aware_recall50",
                                "psnr", "active_thing_queries")
                }
                for scope in ("val32_context", "val32_target")
            }
        paired_curves[str(step)] = row
    result = {
        "registered_steps": list(EVAL_STEPS), "paired_curves": paired_curves,
        "slot_formation": slot,
        "old_v1_gc_control_sanity": "same recipe/scales; GPU differences reported without bit-exact endpoint assertion",
    }
    write_json(compdir / "paired_curves.json", result)
    return result


def _write_causal_report(paired):
    lines = [
        "# Unmatched No-Object CE Causal Ablation — Paired 1k",
        "",
        "## Completion and provenance",
        "",
        "- Architecture LOCUSGS_ANCHOR_GROUP_V1; parent recipe ANCHOR_GROUP_V1_GC_ALPHA001.",
        "- Control unmatched no-object CE scale: 1.0.",
        "- Ablation unmatched no-object CE scale: 0.0.",
        "- Shared understanding-to-reconstruction gradient scale: 0.01.",
        f"- Pretrained SHA: {PRETRAINED_SHA}; manifest SHA: {EXPECTED_MANIFEST_SHA}.",
        f"- Parent 5000-step plan SHA: {EXPECTED_PLAN_SHA}; only its first 1000 entries were used.",
        "- Both arms start fresh from the same pretrained reconstruction and seed-31415 Anchor-Group initialization. No V1-GC endpoint or smoke checkpoint was used.",
        "",
        "## Registered task and direct grouping curves",
        "",
        "| Step | Arm | thing mIoU ctx/tgt | ca-R50 ctx/tgt | class-aware R50 ctx/tgt | ctx TP/FP/FN | target TP/FP/FN | PSNR ctx/tgt | anchor ownership acc | thing-anchor correct | supported-GT recall50 | active queries |",
        "|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for step in EVAL_STEPS:
        row = paired["paired_curves"][str(step)]
        for arm in ("control", "ablation"):
            c, t = row["arms"][arm]["val32_context"], row["arms"][arm]["val32_target"]
            d = row["arms"][arm]["direct_anchor_context"]
            lines.append(
                f"| {step} | {arm} | {c['mIoU_thing']:.6f}/{t['mIoU_thing']:.6f} | "
                f"{c['class_agnostic_recall50']:.6f}/{t['class_agnostic_recall50']:.6f} | "
                f"{c['class_aware_recall50']:.6f}/{t['class_aware_recall50']:.6f} | "
                f"{c['tp_class_agnostic']}/{c['fp_class_agnostic']}/{c['fn_class_agnostic']} | "
                f"{t['tp_class_agnostic']}/{t['fp_class_agnostic']}/{t['fn_class_agnostic']} | "
                f"{c['psnr']:.5f}/{t['psnr']:.5f} | {d['anchor_ownership_accuracy']:.5f} | "
                f"{d['thing_anchor_correct_fraction']:.5f} | {d['anchor_group_gt_recall50']:.5f} | "
                f"{d['active_thing_queries_mean']:.2f} |")
    lines.extend(["", "## Online training positive exposure",
                  "",
                  "| Step | Arm | matches | unique matched | never matched | Top5 | Top10 | Gini | effective queries | first-positive coverage |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for step in (200, 500, 1000):
        for arm in ("control", "ablation"):
            x = paired["slot_formation"][str(step)][arm]["online"]
            lines.append(f"| {step} | {arm} | {x['total_online_gt_matches']} | {x['unique_queries_ever_matched']} | {x['never_matched_queries']} | {x['top5_share']:.5f} | {x['top10_share']:.5f} | {x['match_gini']:.5f} | {x['effective_query_count']:.3f} | {x['first_positive_coverage']['ever_matched']}/100 |")
    lines.extend(["", "## Manifest-wide train1024 utilization",
                  "",
                  "| Step | Arm | GT matches | unique | never | Top5 | Top10 | Gini | effective queries | ownership Gini |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for step in EVAL_STEPS:
        for arm in ("control", "ablation"):
            x = paired["slot_formation"][str(step)][arm]["manifest_train1024"]
            lines.append(f"| {step} | {arm} | {x['total_gt_matches']} | {x['unique_queries_ever_matched']} | {x['queries_never_matched']} | {x['top5_share']:.5f} | {x['top10_share']:.5f} | {x['gini']:.5f} | {x['effective_query_count']:.3f} | {x['ownership_gini']:.5f} |")
    lines.extend(["", "## Step1000 no-object and maximum thing probability",
                  "",
                  "| Arm | never-matched P(no-object) median | never-matched max thing probability median |",
                  "|---|---:|---:|"])
    for arm in ("control", "ablation"):
        x = paired["slot_formation"]["1000"][arm]["manifest_train1024"]
        lines.append(f"| {arm} | {x['noobj_never']['median']:.6f} | {x['max_thing_never_median']:.6f} |")
    lines.extend(["", "## Reconstruction drift at step1000", "",
                  "| Arm | category | relative L2 drift | max absolute drift |",
                  "|---|---|---:|---:|"])
    for arm in ("control", "ablation"):
        drift = _load_step(OUT / arm / "reconstruction_drift_step1000.json")
        for category, value in drift["categories"].items():
            lines.append(f"| {arm} | {category} | {value['relative_l2_delta']:.6g} | {value['max_abs_delta']:.6g} |")
    lines.extend(["", "## Direct grouping mechanism diagnostics", "",
                  "| Step | Arm | assignment entropy | thing mass median | mass max | max/median | query cosine mean/p90 | no-object mean |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for step in EVAL_STEPS:
        for arm in ("control", "ablation"):
            d = paired["paired_curves"][str(step)]["arms"][arm]["direct_anchor_context"]
            mass = d["mechanism_means"]["thing_ownership_mass"]
            cosine = d["mechanism_means"]["query_cosine"]
            noobj = d["mechanism_means"]["no_object_probability"]
            lines.append(f"| {step} | {arm} | {d['assignment_entropy_mean']:.5f} | "
                         f"{mass['median']:.5f} | {mass['max']:.5f} | {mass['max_over_median']:.2f} | "
                         f"{cosine['offdiag_mean']:.5f}/{cosine['p90']:.5f} | {noobj['mean']:.5f} |")
    lines.extend(["", "## Control sanity against committed V1-GC", "",
                  "Renderer-level outputs can have intrinsic GPU numerical noise; this table reports the measured values without a bit-exact claim.",
                  "",
                  "| Step | Recipe | thing mIoU ctx/tgt | ca-R50 ctx/tgt | class-aware R50 ctx/tgt | PSNR ctx/tgt |",
                  "|---:|---|---:|---:|---:|---:|"])
    for step in EVAL_STEPS:
        row = paired["paired_curves"][str(step)]
        for label, item in [("V1-GC", row["v1_gc_reference"]),
                            ("Control", row["arms"]["control"]),
                            ("Ablation", row["arms"]["ablation"])]:
            c, t = item["val32_context"], item["val32_target"]
            lines.append(f"| {step} | {label} | {c['mIoU_thing']:.6f}/{t['mIoU_thing']:.6f} | "
                         f"{c['class_agnostic_recall50']:.6f}/{t['class_agnostic_recall50']:.6f} | "
                         f"{c['class_aware_recall50']:.6f}/{t['class_aware_recall50']:.6f} | "
                         f"{c['psnr']:.5f}/{t['psnr']:.5f} |")
    lines.extend(["", "## Causal interpretation"])
    online = paired["slot_formation"]["1000"]
    c_online, a_online = online["control"]["online"], online["ablation"]["online"]
    c_manifest, a_manifest = online["control"]["manifest_train1024"], online["ablation"]["manifest_train1024"]
    c_direct = paired["paired_curves"]["1000"]["arms"]["control"]["direct_anchor_context"]
    a_direct = paired["paired_curves"]["1000"]["arms"]["ablation"]["direct_anchor_context"]
    lines.append(
        f"- Online step1000 exposure: unique matched queries control/ablation "
        f"{c_online['unique_queries_ever_matched']}/{a_online['unique_queries_ever_matched']}; "
        f"effective count {c_online['effective_query_count']:.3f}/{a_online['effective_query_count']:.3f}; "
        f"Top10 share {c_online['top10_share']:.5f}/{a_online['top10_share']:.5f}.")
    lines.append(
        f"- Full train1024 checkpoint replay: unique "
        f"{c_manifest['unique_queries_ever_matched']}/{a_manifest['unique_queries_ever_matched']}, "
        f"never {c_manifest['queries_never_matched']}/{a_manifest['queries_never_matched']}, "
        f"effective count {c_manifest['effective_query_count']:.3f}/{a_manifest['effective_query_count']:.3f}, "
        f"Top10 share {c_manifest['top10_share']:.5f}/{a_manifest['top10_share']:.5f}.")
    lines.append(
        f"- Val32 context direct grouping: ownership accuracy "
        f"{c_direct['anchor_ownership_accuracy']:.5f}/{a_direct['anchor_ownership_accuracy']:.5f}, "
        f"thing-anchor fraction {c_direct['thing_anchor_correct_fraction']:.5f}/"
        f"{a_direct['thing_anchor_correct_fraction']:.5f}, supported-GT recall50 "
        f"{c_direct['anchor_group_gt_recall50']:.5f}/{a_direct['anchor_group_gt_recall50']:.5f}.")
    c_noobj = c_manifest["noobj_never"]["median"]
    a_noobj = a_manifest["noobj_never"]["median"]
    a_iou = paired["paired_curves"]["1000"]["arms"]["ablation"]["val32_context"]["class_agnostic_recall50"]
    c_iou = paired["paired_curves"]["1000"]["arms"]["control"]["val32_context"]["class_agnostic_recall50"]
    if (a_online["unique_queries_ever_matched"] > c_online["unique_queries_ever_matched"]
            and a_manifest["effective_query_count"] > c_manifest["effective_query_count"]
            and (a_direct["anchor_group_gt_recall50"] > c_direct["anchor_group_gt_recall50"] or a_iou > c_iou)):
        outcome = "Outcome A"
        interpretation = "slot utilization and object grouping/overlap both moved in the favorable direction, evidence that unmatched no-object supervision is an important causal driver of slot death in this registered 1k comparison."
    elif a_noobj < c_noobj and a_manifest["effective_query_count"] <= c_manifest["effective_query_count"]:
        outcome = "Outcome B"
        interpretation = "never-matched no-object probability fell without an increase in effective train1024 query utilization; removing negative classification pressure was insufficient to create object-slot representations in this comparison."
    else:
        outcome = "Outcome C"
        interpretation = "manifest-wide slot utilization did not show the combined improvement required for Outcome A; the observed 1k data do not support unmatched no-object CE as the dominant cause of representation-level dead slots."
    lines.extend(["", f"Classification: {outcome}. {interpretation}",
                  "",
                  "No arbitrary success threshold was introduced; raw values and deltas are reported for review. No corrective model or follow-on experiment was implemented.",
                  ""])
    (OUT / "comparison/causal_1k_report.md").write_text("\n".join(lines))


def _train_paired(device):
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("paired 1000-step experiment requires RTX 3090")
    if torch.cuda.get_device_properties(device).total_memory < 23 * 1024**3:
        raise RuntimeError("paired experiment requires 24GB RTX 3090")
    pre = _load_step(OUT / "comparison/pretrain_contracts.json")
    smoke = _load_step(OUT / "comparison/paired_one_step_smoke.json")
    if pre.get("status") != "pass" or pre.get("passed") != 12:
        raise RuntimeError("12/12 pretrain contracts are a hard gate")
    if smoke.get("status") != "pass" or smoke.get("optimizer_step_count") != 2:
        raise RuntimeError("paired RTX3090 smoke is a hard gate")
    manifest, plan, monitor = _asset_audit()
    for arm in ("control", "ablation"):
        if ((WORK / arm / "checkpoint_step200.pt").exists()
                or (WORK / arm / "checkpoint_step500.pt").exists()):
            raise RuntimeError(f"{arm} workspace contains prior step checkpoints; fresh-start run refused")
    source_state = load_state(PRETRAINED)
    results = {}
    for arm in ("control", "ablation"):
        print(json.dumps({"event": "paired_arm_start", "arm": arm,
                          "recipe": ARM_INFO[arm]["recipe"], "scale": ARM_INFO[arm]["scale"]}), flush=True)
        results[arm] = _train_arm(arm, device, manifest, plan, source_state)
        print(json.dumps({"event": "paired_arm_complete", "arm": arm,
                          "completed_steps": results[arm]["summary"]["completed_steps"],
                          "endpoint_status": results[arm]["endpoint_audit"]["status"]}), flush=True)
    if results["control"]["summary"]["fresh_initial_state_sha256"] != results["ablation"]["summary"]["fresh_initial_state_sha256"]:
        raise RuntimeError("paired formal arms did not share exact fresh initialization")
    paired = _paired_comparison()
    paired["online_exposure"] = {
        arm: _load_step(OUT / arm / "online_match_exposure.json")
        for arm in ("control", "ablation")}
    write_json(OUT / "comparison/paired_curves.json", paired)
    write_json(OUT / "comparison/slot_formation_comparison.json", {
        "registered_steps": list(EVAL_STEPS),
        "online_exposure_definition": "cumulative per-step no-grad unified Hungarian replay on each training prediction and batch",
        "manifest_wide_definition": "independent full train1024/val32-context replay at each checkpoint",
        "by_step": paired["slot_formation"],
        "status": "pass",
    })
    _write_causal_report(paired)
    write_json(OUT / "comparison/paired_training_audit.json", {
        "completed_steps_per_arm": {arm: results[arm]["summary"]["completed_steps"] for arm in results},
        "fresh_initial_state_sha256": {arm: results[arm]["summary"]["fresh_initial_state_sha256"] for arm in results},
        "initial_state_equal": results["control"]["summary"]["fresh_initial_state_sha256"] == results["ablation"]["summary"]["fresh_initial_state_sha256"],
        "endpoint_audits": {arm: results[arm]["endpoint_audit"] for arm in results},
        "train_summaries": {arm: results[arm]["summary"] for arm in results},
        "manifest_sha256": EXPECTED_MANIFEST_SHA, "parent_plan_sha256": EXPECTED_PLAN_SHA,
        "plan_prefix_length": 1000, "pretrained_sha256": PRETRAINED_SHA,
        "monitor_sha256": monitor, "status": "pass",
    })


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("audit", "smoke", "train-paired"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this registered experiment requires a working CUDA device")
    device = torch.device("cuda")
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("Stage-2 gates require NVIDIA GeForce RTX 3090")
    if torch.cuda.get_device_properties(device).total_memory < 23 * 1024**3:
        raise RuntimeError("Stage-2 run requires 24GB RTX3090")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        _asset_audit()
        regression = _run_regressions(device)
        print(json.dumps({"event": "pretrain_regressions", "status": regression["status"],
                          "phase_a": regression["phase_a_cpu"],
                          "legacy_loss": regression["legacy_loss_cpu"],
                          "legacy_eval": regression["legacy_evaluator_cpu"],
                          "s0_s1": regression["s0_s1_gpu_forward"]["status"]}, sort_keys=True), flush=True)
        result = _run_pretrain_contracts(device)
        print(json.dumps({"event": "pretrain_contracts", "passed": result["passed"],
                          "total": result["total"], "status": result["status"]}, sort_keys=True), flush=True)
    elif args.phase == "smoke":
        result = _run_pair_smoke(device)
        print(json.dumps({"event": "paired_one_step_smoke", "status": result["status"],
                          "initial_state_tensors": result["initial_model_state"]["tensor_count"],
                          "peak_allocated_gib": result["peak_allocated_gib"],
                          "peak_reserved_gib": result["peak_reserved_gib"]}, sort_keys=True), flush=True)
    else:
        _train_paired(device)


if __name__ == "__main__":
    _main()
