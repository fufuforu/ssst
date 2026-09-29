#!/usr/bin/env python3
"""Anchor-Group V1-GC: shared-gradient-controlled joint recipe.

The architecture and losses are the unchanged V1 implementation.  This file
changes only how the two already-computed task losses contribute gradients.
There is deliberately no adaptive control and no train phase is run by audit.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)

from scripts.anchor_group_v1 import (
    EVAL_STEPS, GRAD_CLIP, GROUP_PEAK_LR, MANIFEST, MANIFEST_SHA, MONITORS,
    OUT as V1_OUT, PLAN, PRETRAINED, PRETRAINED_SHA, RECON_PEAK_LR,
    SOURCE_REPORTS, TOTAL_STEPS, WEIGHT_DECAY, build_optimizer, build_options,
    evaluate_anchor_group_all, load_state, make_model, sha256,
    set_optimizer_lr, transfer_anchor_group_reconstruction_weights, write_json,
)
from scripts.instance_state_generalization import _batch_for, _seen_classes
from scripts.instance_state_runtime import capture_rng, restore_rng
from tokengs.models import model_registry
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data

OUT = REPO / "group_plus/anchor_group_v1_gc"
WORKSPACE = REPO / "workspace_group_plus/anchor_group_v1_gc"
EXPECTED_PLAN_SHA = "ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323"
SHARED_UNDERSTANDING_GRAD_SCALE = 0.01
GRAD_WINDOW_INDICES = [0, 64, 128, 192, 256, 320, 384, 448,
                       512, 576, 640, 704, 768, 832, 896, 960]
CATEGORY_NAMES = ("all_reconstruction", "encoder", "decoder", "anchor_geometry",
                  "activation_head", "other_reconstruction")


def _write(path, payload):
    write_json(path, payload)


def validate_locked_recipe():
    if sha256(PRETRAINED) != PRETRAINED_SHA:
        raise RuntimeError("pretrained checkpoint SHA256 mismatch")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("locked train manifest SHA256 mismatch")
    if sha256(PLAN) != EXPECTED_PLAN_SHA:
        raise RuntimeError("locked 5000-step plan SHA256 mismatch")
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    windows, entries = manifest.get("windows", []), plan.get("entries", [])
    if len(windows) != 1024 or len({x["scene"] for x in windows}) != 128:
        raise RuntimeError("locked manifest must retain 128 scenes / 1024 windows")
    if len(entries) != 5000:
        raise RuntimeError("locked plan must contain exactly 5000 entries")
    for i, entry in enumerate(entries, 1):
        if int(entry.get("step", -1)) != i:
            raise RuntimeError(f"plan step numbering differs at {i}")
        wi = int(entry["window_index"])
        if not 0 <= wi < len(windows):
            raise RuntimeError(f"invalid plan window index at step {i}")
        if any(entry.get(k) != windows[wi].get(k) for k in ("scene", "context", "novel")):
            raise RuntimeError(f"plan entry differs from locked manifest at step {i}")
    monitor_rows = {}
    for name in MONITORS:
        source = SOURCE_REPORTS / name
        old_copy = V1_OUT / name
        src_sha = sha256(source)
        if not old_copy.is_file() or sha256(old_copy) != src_sha:
            raise RuntimeError(f"registered monitor is not byte-identical to V1: {name}")
        monitor_rows[name] = {"source_sha256": src_sha, "v1_sha256": sha256(old_copy), "equal": True}
    return manifest, plan, monitor_rows


def _unique_parameters(model, *, group=None):
    seen, out = set(), []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_group = name.startswith("anchor_group.")
        if group is True and not is_group:
            continue
        if group is False and is_group:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        out.append((name, param))
    return out


def backward_gradient_controlled(model, loss_recon, loss_understanding,
                                  understanding_weight, shared_scale=SHARED_UNDERSTANDING_GRAD_SCALE,
                                  audit_capture=None):
    """Accumulate gR + alpha*w*gU on reconstruction and w*gU on group params.

    Caller owns ``zero_grad`` and clipping/optimizer operations. Temporary
    hooks exist only during the understanding backward and are always removed.
    """
    weight = float(understanding_weight)
    scale = float(shared_scale)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("understanding_weight must be finite and nonnegative")
    if scale != SHARED_UNDERSTANDING_GRAD_SCALE:
        raise ValueError("shared gradient scale is a fixed recipe constant (0.01)")
    if not torch.is_tensor(loss_recon) or not loss_recon.requires_grad:
        raise ValueError("loss_recon must be a graph-connected scalar")
    loss_recon.backward(retain_graph=True)
    if audit_capture is not None:
        audit_capture["g_recon_before_understanding"] = {
            name: (None if param.grad is None else param.grad.detach().to("cpu", copy=True))
            for name, param in _unique_parameters(model, group=False)}
        audit_capture["group_after_reconstruction"] = {
            name: (None if param.grad is None else param.grad.detach().to("cpu", copy=True))
            for name, param in _unique_parameters(model, group=True)}
    if weight == 0.0:
        return {"registered_hook_count": 0, "removed_hook_count": 0,
                "reconstruction_capture_hook_count": 0,
                "group_capture_hook_count": 0,
                "understanding_backward_skipped": True}
    handles = []
    seen = set()
    registered = 0
    try:
        for name, param in model.named_parameters():
            if not param.requires_grad or name.startswith("anchor_group.") or id(param) in seen:
                continue
            seen.add(id(param))
            if audit_capture is None:
                hook = lambda grad, alpha=scale: grad * alpha
            else:
                def hook(grad, *, param_name=name, alpha=scale):
                    audit_capture.setdefault("g_under_raw", {})[param_name] = grad.detach().to("cpu", copy=True)
                    return grad * alpha
            handles.append(param.register_hook(hook))
            registered += 1
        group_registered = 0
        if audit_capture is not None:
            for name, param in _unique_parameters(model, group=True):
                def group_hook(grad, *, param_name=name):
                    audit_capture.setdefault("g_group_raw", {})[param_name] = grad.detach().to("cpu", copy=True)
                    return grad
                handles.append(param.register_hook(group_hook))
                group_registered += 1
        (weight * loss_understanding).backward()
    finally:
        for handle in handles:
            handle.remove()
    return {"registered_hook_count": len(handles),
            "reconstruction_hook_count": registered,
            "removed_hook_count": len(handles),
            "reconstruction_capture_hook_count": registered if audit_capture is not None else 0,
            "group_capture_hook_count": group_registered if audit_capture is not None else 0,
            "understanding_backward_skipped": False}


def _category(name):
    if name.startswith("anchor_group."):
        return None
    if name.startswith("enc_dec_backbone.decoder_blocks."):
        return "decoder"
    if name.startswith("anchor_decoder.") and (
        name in ("anchor_decoder.mu", "anchor_decoder.rho", "anchor_decoder.gamma_raw")
        or name.startswith(("anchor_decoder.refine_mu.", "anchor_decoder.refine_rho.",
                            "anchor_decoder.pe_mlp", "anchor_decoder.pe_mlps"))):
        return "anchor_geometry"
    if name.startswith("activation_head."):
        return "activation_head"
    if name.startswith("enc_dec_backbone."):
        return "encoder"
    return "other_reconstruction"


def _clone_grads(model):
    return {name: (param.grad.detach().to("cpu", copy=True) if param.grad is not None else None)
            for name, param in model.named_parameters()}


def _grad_diff(expected, actual, names):
    max_abs, diff_sq, ref_sq, tensors = 0.0, 0.0, 0.0, 0
    for name in names:
        a, b = expected.get(name), actual.get(name)
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        delta = a.double() - b.double()
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max()))
            diff_sq += float(delta.square().sum())
            ref_sq += float(a.double().square().sum())
            tensors += 1
    return {"max_abs_diff": max_abs,
            "relative_l2_diff": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-30),
            "tensor_count_checked": tensors}


def _linear_gradient_combo(a, b, b_scale):
    if a is None and b is None:
        return None
    if a is None:
        a = torch.zeros_like(b)
    if b is None:
        b = torch.zeros_like(a)
    return a + float(b_scale) * b


def _get_losses(model, batch, step, rng_state):
    restore_rng(rng_state)
    output, metrics = model.step_loss(batch, step=step, coupled=False,
                                      understanding_weight_override=1.0)
    return output, metrics


def _raw_task_grad(model, batch, step, key, rng_state):
    model.zero_grad(set_to_none=True)
    _out, metrics = _get_losses(model, batch, step, rng_state)
    loss = metrics[key]
    if not torch.isfinite(loss) or not loss.requires_grad:
        raise RuntimeError(f"invalid graph/value for {key}")
    scalar = float(loss.detach())
    loss.backward()
    grads = _clone_grads(model)
    model.zero_grad(set_to_none=True)
    del _out, metrics, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scalar, grads


def _max_abs_tensor(a, b):
    if a is None and b is None:
        return 0.0
    if a is None:
        a = torch.zeros_like(b)
    if b is None:
        b = torch.zeros_like(a)
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def _gradient_window_stats(g_recon, g_under, named_parameters):
    per_category = {}
    for category in CATEGORY_NAMES:
        selected = [n for n, _p in named_parameters.items()
                    if category == "all_reconstruction" or _category(n) == category]
        nr2 = nu2 = ne2 = nc2 = dot = combined_dot = 0.0
        for name in selected:
            gr, gu = g_recon.get(name), g_under.get(name)
            if gr is None and gu is None:
                continue
            if gr is None:
                gr = torch.zeros_like(gu)
            if gu is None:
                gu = torch.zeros_like(gr)
            grd, gud = gr.double(), gu.double()
            ge = gu * SHARED_UNDERSTANDING_GRAD_SCALE
            gcmb = gr + ge
            nr2 += float(grd.square().sum())
            nu2 += float(gud.square().sum())
            ne2 += float(ge.double().square().sum())
            nc2 += float(gcmb.double().square().sum())
            dot += float((grd * gud).sum())
            combined_dot += float((gr.double() * gcmb.double()).sum())
        nr, nu, ne, nc = map(math.sqrt, (nr2, nu2, ne2, nc2))
        raw_ratio = nu / max(nr, 1e-30)
        eff_ratio = ne / max(nr, 1e-30)
        raw_cos = dot / max(nr * nu, 1e-30)
        eff_cos = (SHARED_UNDERSTANDING_GRAD_SCALE * dot) / max(nr * ne, 1e-30)
        comb_cos = combined_dot / max(nr * nc, 1e-30)
        theoretical_ratio = SHARED_UNDERSTANDING_GRAD_SCALE * raw_ratio
        ratio_abs_error = abs(eff_ratio - theoretical_ratio)
        ratio_rel_error = ratio_abs_error / max(abs(theoretical_ratio), 1e-30)
        per_category[category] = {
            "norm_recon": nr, "norm_under_raw": nu,
            "norm_under_effective": ne, "norm_combined": nc,
            "raw_ratio": raw_ratio, "effective_ratio": eff_ratio,
            "expected_effective_ratio": theoretical_ratio,
            "effective_ratio_abs_error": ratio_abs_error,
            "effective_ratio_relative_error": ratio_rel_error,
            "raw_cosine": raw_cos, "effective_cosine": eff_cos,
            "cosine_invariance_abs_error": abs(raw_cos - eff_cos),
            "combined_over_recon_norm": nc / max(nr, 1e-30),
            "combined_vs_recon_cosine": comb_cos,
        }
    return per_category


def _summary_gradient_rows(rows):
    summary = {}
    for category in CATEGORY_NAMES:
        vals = [row["categories"][category] for row in rows]
        eff = np.asarray([v["effective_ratio"] for v in vals])
        comb_ratio = np.asarray([v["combined_over_recon_norm"] for v in vals])
        comb_cos = np.asarray([v["combined_vs_recon_cosine"] for v in vals])
        summary[category] = {
            "raw_ratio_median": float(np.median([v["raw_ratio"] for v in vals])),
            "effective_ratio_median": float(np.median(eff)),
            "effective_ratio_p10": float(np.quantile(eff, .10)),
            "effective_ratio_p90": float(np.quantile(eff, .90)),
            "combined_over_recon_norm_median": float(np.median(comb_ratio)),
            "combined_over_recon_norm_p10": float(np.quantile(comb_ratio, .10)),
            "combined_over_recon_norm_p90": float(np.quantile(comb_ratio, .90)),
            "combined_vs_recon_cosine_median": float(np.median(comb_cos)),
            "combined_vs_recon_cosine_p10": float(np.quantile(comb_cos, .10)),
            "combined_vs_recon_cosine_p90": float(np.quantile(comb_cos, .90)),
            "fraction_combined_cosine_lt_0_9": float(np.mean(comb_cos < .9)),
            "fraction_combined_cosine_lt_0_5": float(np.mean(comb_cos < .5)),
            "fraction_combined_cosine_lt_0": float(np.mean(comb_cos < 0)),
            "max_effective_ratio_relative_error": float(max(v["effective_ratio_relative_error"] for v in vals)),
            "max_cosine_invariance_abs_error": float(max(v["cosine_invariance_abs_error"] for v in vals)),
        }
    return summary


def _report_stats(grads, predicate):
    nr2 = 0.0
    for name, grad in grads.items():
        if grad is not None and predicate(name):
            nr2 += float(grad.double().square().sum())
    return math.sqrt(nr2)


def _compare_forward_outputs(a, b):
    # GC has no alternate forward branch: this function compares the complete
    # readouts and losses emitted by repeated V1 step_loss calls.
    ap, bp = a["prediction"], b["prediction"]
    pairs = {
        "gaussians": (ap["gaussians"], bp["gaussians"]),
        "rgb": (ap["render"]["images_pred"], bp["render"]["images_pred"]),
        "A_post": (ap["states"][-1]["A_post"], bp["states"][-1]["A_post"]),
        "region_mass": (ap["region_mass"], bp["region_mass"]),
        "semantic_scores": (ap["semantic_scores"], bp["semantic_scores"]),
        "thing_class_logits": (ap["thing_class_logits"], bp["thing_class_logits"]),
    }
    diffs = {k: float((x - y).abs().max()) for k, (x, y) in pairs.items()}
    for key in ("loss_recon", "loss_understanding", "loss_anchor_group", "loss"):
        av, bv = a["metrics"][key], b["metrics"][key]
        if torch.is_tensor(av) and torch.is_tensor(bv):
            diffs[key] = float((av - bv).abs().max())
        else:
            diffs[key] = abs(float(av) - float(bv))
    return diffs


def _state_tensor_parity(reference, candidate):
    a = reference.state_dict() if hasattr(reference, "state_dict") else reference
    b = candidate.state_dict() if hasattr(candidate, "state_dict") else candidate
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    mismatches = []
    for name in sorted(set(a) & set(b)):
        if (a[name].shape != b[name].shape or a[name].dtype != b[name].dtype or
                not torch.equal(a[name].detach().cpu(), b[name].detach().cpu())):
            mismatches.append(name)
    return {"model_state_equal": not missing and not extra and not mismatches,
            "tensor_count": len(a), "missing_keys": missing, "extra_keys": extra,
            "mismatched_keys": mismatches}


def _batch_tensor_manifest(batch):
    rows = {}
    def visit(path, value):
        if torch.is_tensor(value):
            cpu = value.detach().contiguous().cpu()
            raw = cpu.view(torch.uint8).numpy().tobytes()
            rows[path] = {"shape": list(value.shape), "dtype": str(value.dtype),
                          "device": str(value.device), "sha256": hashlib.sha256(raw).hexdigest(),
                          "sum": float(cpu.double().sum()) if cpu.numel() else 0.0}
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(f"{path}.{key}" if path else str(key), item)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(f"{path}[{i}]", item)
    visit("", batch)
    return rows


def _scalar_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if not key.startswith("loss"):
            continue
        if torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach())
        elif isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def _tensor_cpu(value):
    if not torch.is_tensor(value):
        return None
    return value.detach().to("cpu", copy=True)


def _capture_forward_tensors(output):
    pred = output["prediction"]
    return {"gaussians": _tensor_cpu(pred["gaussians"]),
            "rgb": _tensor_cpu(pred["render"]["images_pred"]),
            "A_post": _tensor_cpu(pred["states"][-1]["A_post"]),
            "region_mass": _tensor_cpu(pred["region_mass"]),
            "semantic_scores": _tensor_cpu(pred["semantic_scores"]),
            "thing_class_logits": _tensor_cpu(pred["thing_class_logits"])}


def _forward_tensor_diffs(a, b):
    diffs = {key: float((a[key] - b[key]).abs().max()) if a[key].numel() else 0.0
             for key in a}
    return {"max_abs_diff_by_tensor": diffs,
            "all_exact": all(value == 0.0 for value in diffs.values())}


def _pairwise_abs(values):
    return max((abs(a - b) for i, a in enumerate(values) for b in values[i + 1:]), default=0.0)


def _distribution(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(arr)), "max": float(np.max(arr)),
            "mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "std": float(np.std(arr)), "range": float(np.max(arr) - np.min(arr)),
            "max_pairwise_abs_diff": _pairwise_abs([float(x) for x in arr])}


def _measure_repeat(model, batch, rng):
    module = importlib.import_module("tokengs.models.anchor_group_locusgs")
    original_loss = module.canonical_layer_loss
    layer_calls = []
    def capture_layer_loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        render = kwargs["render_results"]
        layer_calls.append({"gaussians": _tensor_cpu(kwargs["gaussians"]),
                            **{key: _tensor_cpu(render[key]) for key in
                               ("images_pred", "alphas_pred", "means2d_pred") if key in render},
                            "objective_loss": float(result["loss"].detach()),
                            "components": {key: float(value.detach()) for key, value in result.items()
                                           if torch.is_tensor(value) and value.numel() == 1}})
        return result
    module.canonical_layer_loss = capture_layer_loss
    try:
        restore_rng(rng)
        output, metrics = model.step_loss(batch, step=1000, coupled=False)
        scalar = _scalar_metrics(metrics)
        forward = _capture_forward_tensors(output)
        # Layer ordering comes from the model's registered supervised layer tuple.
        layers = {str(layer): row for layer, row in zip(model.supervised_layers, layer_calls)}
        if len(layer_calls) != len(model.supervised_layers):
            raise RuntimeError(f"captured {len(layer_calls)} supervised renders, expected {len(model.supervised_layers)}")
        if not {"6", "12"}.issubset(layers):
            raise RuntimeError(f"expected supervised layer 6 and 12 captures, got {sorted(layers)}")
        return {"scalars": scalar, "forward": forward, "layers": layers}
    finally:
        module.canonical_layer_loss = original_loss


def _run_parity_noise_root_cause(device, model_v1, model_gc, batch):
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("parity noise audit requires the registered RTX 3090 GPU")
    state_parity = _state_tensor_parity(model_v1, model_gc)
    batch_manifest = _batch_tensor_manifest(batch)
    batch_shared = {"same_batch_object_reused": True,
                    "tensor_field_count": len(batch_manifest),
                    "tensor_manifest": batch_manifest}
    model_v1.eval(); model_gc.eval()
    rng = capture_rng()
    repeats = {"v1": [], "gc": []}
    for arm, model in (("v1", model_v1), ("gc", model_gc)):
        for i in range(10):
            repeats[arm].append(_measure_repeat(model, batch, rng))
            model.zero_grad(set_to_none=True)
            print(f"[GC parity repeat] {arm} {i + 1}/10", flush=True)
    if _batch_tensor_manifest(batch) != batch_manifest:
        raise RuntimeError("forward mutated the shared audit batch")

    metric_names = sorted(set.intersection(*(set(x["scalars"]) for arm in repeats for x in repeats[arm])))
    metric_rows, noise_envelopes = {}, {}
    for metric in metric_names:
        vals = {arm: [row["scalars"][metric] for row in repeats[arm]] for arm in repeats}
        stats = {arm: _distribution(vals[arm]) for arm in repeats}
        cross = [abs(a - b) for a in vals["v1"] for b in vals["gc"]]
        med_diff = abs(stats["v1"]["median"] - stats["gc"]["median"])
        envelope = stats["v1"]["max_pairwise_abs_diff"] + stats["gc"]["max_pairwise_abs_diff"] + 1e-7
        overlap = max(stats["v1"]["min"], stats["gc"]["min"]) <= min(stats["v1"]["max"], stats["gc"]["max"]) + 1e-7
        metric_rows[metric] = {"v1": stats["v1"], "gc": stats["gc"],
                               "median_abs_diff": med_diff,
                               "cross_arm_10x10_abs_diff": {"min": min(cross),
                                   "median": float(np.median(cross)), "max": max(cross)},
                               "noise_envelope": envelope, "distributions_overlap": overlap,
                               "corrected_parity_pass": med_diff <= envelope and overlap}
        noise_envelopes[metric] = envelope

    layer_rows = {}
    layer_keys = sorted(set.intersection(*(set(x["layers"]) for arm in repeats for x in repeats[arm])))
    for layer in layer_keys:
        layer_rows[layer] = {}
        fields = sorted(set.intersection(*(set(x["layers"][layer]) for arm in repeats for x in repeats[arm])))
        for field in fields:
            if field == "components":
                component_keys = sorted(set.intersection(*(set(x["layers"][layer][field]) for arm in repeats for x in repeats[arm])))
                for component in component_keys:
                    vals = {arm: [row["layers"][layer][field][component] for row in repeats[arm]] for arm in repeats}
                    layer_rows[layer][f"component:{component}"] = {
                        "v1": _distribution(vals["v1"]), "gc": _distribution(vals["gc"]),
                        "exact_within_v1": _pairwise_abs(vals["v1"]) == 0,
                        "exact_within_gc": _pairwise_abs(vals["gc"]) == 0}
                continue
            if field == "objective_loss":
                vals = {arm: [row["layers"][layer][field] for row in repeats[arm]] for arm in repeats}
                layer_rows[layer][field] = {"v1": _distribution(vals["v1"]), "gc": _distribution(vals["gc"])}
                continue
            vals = {arm: [row["layers"][layer][field] for row in repeats[arm]] for arm in repeats}
            diffs = []
            for arm in repeats:
                for i, a in enumerate(vals[arm]):
                    for b in vals[arm][i + 1:]:
                        diffs.append(float((a - b).abs().max()) if a.numel() else 0.0)
            layer_rows[layer][field] = {"v1_max_pairwise_repeat_diff": max(
                                                (float((a - b).abs().max()) if a.numel() else 0.0
                                                 for i, a in enumerate(vals["v1"]) for b in vals["v1"][i + 1:]), default=0.0),
                                        "gc_max_pairwise_repeat_diff": max(
                                                (float((a - b).abs().max()) if a.numel() else 0.0
                                                 for i, a in enumerate(vals["gc"]) for b in vals["gc"][i + 1:]), default=0.0),
                                        "exact_within_both_arms": all(x == 0.0 for x in diffs)}

    component_candidates = [name for name in metric_names if name.startswith("loss_") and "_layer" in name]
    noisy_components = sorted(component_candidates,
                              key=lambda n: max(metric_rows[n][arm]["range"] for arm in ("v1", "gc")),
                              reverse=True)
    first_noisy = next((n for n in noisy_components
                        if max(metric_rows[n][arm]["range"] for arm in ("v1", "gc")) > 0), None)
    source = "no_repeat_variation_detected"
    if first_noisy:
        component = first_noisy.rsplit("_layer", 1)[0]
        layer = first_noisy.rsplit("_layer", 1)[1]
        layer_diag = layer_rows.get(layer, {})
        render_nonexact = any(layer_diag.get(field, {}).get("v1_max_pairwise_repeat_diff", 0.0) > 0 or
                              layer_diag.get(field, {}).get("gc_max_pairwise_repeat_diff", 0.0) > 0
                              for field in ("gaussians", "images_pred", "alphas_pred", "means2d_pred"))
        source = ("render_level_numerical_nondeterminism" if render_nonexact
                  else f"layer_{layer}_{component}_calculation_or_reduction")
    first_v1 = repeats["v1"][0]; first_gc = repeats["gc"][0]
    c1a_tensor = _forward_tensor_diffs(first_v1["forward"], first_gc["forward"])
    c1a = {**state_parity, **batch_shared, **c1a_tensor,
           "passed": state_parity["model_state_equal"] and c1a_tensor["all_exact"]}
    c1b = {"metrics": metric_rows,
           "passed": bool(metric_rows) and all(row["corrected_parity_pass"] for row in metric_rows.values())}
    result = {"gpu": torch.cuda.get_device_name(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "dtype": "FP32", "seed": 42, "anchor_group_init_seed": 31415,
              "step": 1000, "repeat_count_per_arm": 10, "backend_settings": {
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "cudnn_deterministic": torch.backends.cudnn.deterministic,
                  "cudnn_benchmark": torch.backends.cudnn.benchmark,
                  "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                  "tf32_cudnn": torch.backends.cudnn.allow_tf32},
              "model_state": state_parity, "batch": batch_shared,
              "same_rng_state_restored_before_every_repeat": True,
              "metric_statistics": metric_rows, "noise_envelopes": noise_envelopes,
              "layer_render_repeat_diagnostics": layer_rows,
              "root_cause": {"noisy_reconstruction_components_ranked": noisy_components,
                             "first_nonexact_reconstruction_component": first_noisy,
                             "source": source},
              "C1a_forward_tensor_parity": c1a,
              "C1b_empirical_loss_parity": c1b,
              "historical_independent_bit_exact_gate": {
                  "prior_status": "failed",
                  "interpretation": "independent CUDA reduction/rasterization executions are evaluated against measured same-arm repeat noise; forward tensors remain exact-gated"},
              "status": "pass" if c1a["passed"] and c1b["passed"] else "fail",
              "repeats": {arm: [{"scalars": row["scalars"]} for row in repeats[arm]] for arm in repeats}}
    _write(OUT / "parity_noise_root_cause.json", result)
    return result


def _forward_parity(model, opt, batch):
    model.eval()
    rng = capture_rng()
    with torch.no_grad():
        restore_rng(rng)
        out_v1, metrics_v1 = model.step_loss(batch, step=1000, coupled=False,
                                             understanding_weight_override=1.0)
        restore_rng(rng)
        out_gc, metrics_gc = model.step_loss(batch, step=1000, coupled=False,
                                             understanding_weight_override=1.0)
    diff = _compare_forward_outputs(
        {"prediction": out_v1["prediction"], "metrics": metrics_v1},
        {"prediction": out_gc["prediction"], "metrics": metrics_gc})
    return {"max_abs_diff_by_tensor_or_loss": diff,
            "overall_max_abs_diff": max(diff.values()),
            "all_exact": all(v == 0.0 for v in diff.values()),
            "same_model_state_same_batch_rng_restored": True}


def _optimizer_parity(model):
    optimizer, audit = build_optimizer(model)
    expected = {
        "anchor_group_decay": (18, 2743496),
        "anchor_group_nodecay": (41, 41800),
        "reconstruction_decay": (115, 218773504),
        "reconstruction_nodecay": (335, 1229116),
    }
    groups = {row["name"]: row for row in audit["groups"]}
    exact = set(groups) == set(expected)
    for name, (count, numel) in expected.items():
        exact = exact and groups[name]["tensor_count"] == count and groups[name]["numel"] == numel
    audit["registered_v1_group_expectations"] = {
        k: {"tensor_count": v[0], "numel": v[1]} for k, v in expected.items()}
    audit["optimizer_group_parity"] = bool(exact and audit["all_trainable_parameters_exactly_once"])
    return optimizer, audit


def _phase_a_contract_replay():
    from scripts import check_anchor_group_v1_contract as contract
    with tempfile.TemporaryDirectory(prefix="agv1gc_phasea_") as tmp:
        temp_out = Path(tmp) / "contracts.json"
        contract.OUT = temp_out
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = contract.main()
        payload = json.loads(temp_out.read_text())
        return {"exit_code": int(code), "passed": payload["passed"],
                "failed": payload["failed"], "total": len(payload["checks"]),
                "failed_contracts": [x["contract"] for x in payload["checks"] if not x["passed"]],
                "status": "pass" if code == 0 and payload["passed"] == 18 else "fail"}


def _reconstruction_parity(model, opt, source_state, device, batch):
    from tokengs.models.canonical_recon_models import LocusGSRecon
    model.eval()
    mi, _ = split_data(batch, opt)
    dec = ModelInputDecoder(cam_view=batch["cam_view_all"],
                            intrinsics=batch["intrinsics_all"])
    with torch.no_grad():
        ag = model.forward_anchor_group(ModelInput(mi.encoder, dec),
                                        render_decoder_input=dec,
                                        coupled=False, step=0)
    baseline_opt = opt.evolve(model_type="siu3r_locusgs_recon", workspace="",
                              experiment_name="anchor_group_v1_gc_parity")
    baseline = LocusGSRecon(baseline_opt)
    baseline.load_state_dict(source_state, strict=True)
    baseline = baseline.to(device).eval()
    with torch.no_grad():
        base = baseline.forward_reconstruction_only(ModelInput(mi.encoder, dec),
                                                    render_decoder_input=dec)
    gs_diff = float((ag["gaussians"].float() - base["gaussians"].float()).abs().max())
    rgb_diff = float((ag["render"]["images_pred"].float() - base["render"]["images_pred"].float()).abs().max())
    gt = batch["images_all"].float()
    def psnr(x):
        return float((-10 * torch.log10((x - gt).square().mean(dim=(-1, -2, -3)))).mean())
    p0, p1 = psnr(base["render"]["images_pred"].float()), psnr(ag["render"]["images_pred"].float())
    return {"gpu": torch.cuda.get_device_name(device), "cuda_available": True,
            "checkpoint_sha256": sha256(PRETRAINED), "scene": str(batch["scene_id"][0]) if "scene_id" in batch else None,
            "gaussian_max_abs_diff": gs_diff, "rgb_max_abs_diff": rgb_diff,
            "psnr_baseline": p0, "psnr_anchor_group_v1_gc": p1,
            "psnr_abs_diff": abs(p0 - p1),
            "status": "pass" if gs_diff == 0 and rgb_diff == 0 and p0 == p1 else "fail"}


def _legacy_regression_replay():
    from scripts import anchor_group_v1_legacy_forward_regression as legacy
    with tempfile.TemporaryDirectory(prefix="agv1gc_legacy_") as tmp:
        legacy.OUT = Path(tmp) / "legacy_forward_regression.json"
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = legacy.main()
        result = json.loads(legacy.OUT.read_text())
        return {"exit_code": int(code), "status": result["status"],
                "gpu": result.get("gpu"), "models": result["models"]}


def _run_raw_gc_contracts(model, opt, device, batch):
    model.eval()
    names_all = dict(model.named_parameters())
    rec_names = [n for n, p in names_all.items() if p.requires_grad and not n.startswith("anchor_group.")]
    group_names = [n for n, p in names_all.items() if p.requires_grad and n.startswith("anchor_group.")]
    rng = capture_rng()
    l_r, g_r = _raw_task_grad(model, batch, 1000, "loss_recon", rng)
    l_u, g_u = _raw_task_grad(model, batch, 1000, "loss_understanding", rng)
    c2 = {"passed": math.isfinite(l_r) and math.isfinite(l_u),
          "loss_recon": l_r, "loss_understanding": l_u,
          "reconstruction_gradient_tensors": sum(g_r.get(n) is not None for n in rec_names),
          "understanding_gradient_tensors": sum(g_u.get(n) is not None for n in names_all)}

    # C3/C4: the exact same forward graph supplies the production gR pass,
    # raw gU hook captures, and the accumulated GC gradient.  This avoids using
    # a second rasterization as a gradient reference.
    before = {n: p.detach().cpu().clone() for n, p in names_all.items() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    _o, m = _get_losses(model, batch, 1000, rng)
    capture1000 = {}
    hooks = backward_gradient_controlled(model, m["loss_recon"], m["loss_understanding"], 1.0,
                                         audit_capture=capture1000)
    gc1000 = _clone_grads(model)
    g_r1000 = capture1000["g_recon_before_understanding"]
    g_u1000 = capture1000.get("g_under_raw", {})
    group_u1000 = capture1000.get("g_group_raw", {})
    rec_expected = {n: _linear_gradient_combo(g_r1000.get(n), g_u1000.get(n),
                                              SHARED_UNDERSTANDING_GRAD_SCALE)
                    for n in rec_names}
    rec_actual = {n: gc1000.get(n) for n in rec_names}
    rec_diff = _grad_diff(rec_expected, rec_actual, rec_names)
    group_diff = _grad_diff(group_u1000, gc1000, group_names)
    def nonzero_count(grads, names):
        return sum(grads.get(name) is not None and bool(torch.count_nonzero(grads[name])) for name in names)
    g_r_nz = nonzero_count(g_r1000, rec_names)
    g_u_nz = nonzero_count(g_u1000, rec_names)
    c3 = {**rec_diff,
          "tensor_count_total": len(rec_names),
          "gR_nonzero_count": g_r_nz,
          "gU_nonzero_count": g_u_nz,
          "both_nonzero_count": sum(g_r1000.get(n) is not None and bool(torch.count_nonzero(g_r1000[n])) and
                                     g_u1000.get(n) is not None and bool(torch.count_nonzero(g_u1000[n]))
                                     for n in rec_names),
          "only_gR_count": sum(g_r1000.get(n) is not None and bool(torch.count_nonzero(g_r1000[n])) and
                               not (g_u1000.get(n) is not None and bool(torch.count_nonzero(g_u1000[n])))
                               for n in rec_names),
          "only_gU_count": sum(g_u1000.get(n) is not None and bool(torch.count_nonzero(g_u1000[n])) and
                               not (g_r1000.get(n) is not None and bool(torch.count_nonzero(g_r1000[n])))
                               for n in rec_names),
          "group_grad_after_reconstruction_none_or_zero": all(
              capture1000["group_after_reconstruction"].get(n) is None or
              not bool(torch.count_nonzero(capture1000["group_after_reconstruction"][n])) for n in group_names),
          "hooks": {"registered": hooks["registered_hook_count"], "removed": hooks["removed_hook_count"],
                    "reconstruction_capture": hooks["reconstruction_capture_hook_count"],
                    "group_capture": hooks["group_capture_hook_count"]},
          "method": "single_forward_graph_production_backward_with_raw_gradient_capture",
          "passed": rec_diff["max_abs_diff"] <= 1e-6 and rec_diff["relative_l2_diff"] <= 1e-6 and
                    hooks["registered_hook_count"] == hooks["removed_hook_count"]}
    c4 = {**group_diff,
          "tensor_count": len(group_names),
          "nonzero_count": nonzero_count(group_u1000, group_names),
          "query_init_max_abs_diff": _max_abs_tensor(
              group_u1000.get("anchor_group.query_init"), gc1000.get("anchor_group.query_init")),
          "hooks_registered_removed_equal": hooks["registered_hook_count"] == hooks["removed_hook_count"],
          "passed": group_diff["max_abs_diff"] <= 1e-6 and group_diff["relative_l2_diff"] <= 1e-6 and
                    hooks["registered_hook_count"] == hooks["removed_hook_count"]}
    unchanged = all(torch.equal(before[n], p.detach().cpu()) for n, p in names_all.items() if p.requires_grad)
    c8 = {"parameters_exactly_unchanged": unchanged,
          "parameter_tensor_count": len(before), "passed": unchanged}
    model.zero_grad(set_to_none=True)
    del _o, m, gc1000
    _, metrics_next = _get_losses(model, batch, 1000, rng)
    metrics_next["loss_recon"].backward(retain_graph=True)
    raw_after_hook_cleanup = _clone_grads(model)
    model.zero_grad(set_to_none=True)
    active_handles = [n for n, p in names_all.items()
                      if getattr(p, "_backward_hooks", None)]
    metrics_next["loss_recon"].backward()
    next_recon = _clone_grads(model)
    c7_diff = _grad_diff(raw_after_hook_cleanup, next_recon, rec_names)
    c7 = {**c7_diff, "registered_hook_count": hooks["registered_hook_count"],
          "removed_hook_count": hooks["removed_hook_count"],
          "remaining_parameter_hooks": active_handles,
          "next_reconstruction_unscaled": c7_diff["max_abs_diff"] <= 1e-6 and c7_diff["relative_l2_diff"] <= 1e-6,
          "passed": hooks["registered_hook_count"] == hooks["removed_hook_count"] and
                    not active_handles and
                    c7_diff["max_abs_diff"] <= 1e-6 and c7_diff["relative_l2_diff"] <= 1e-6}
    model.zero_grad(set_to_none=True)

    # C5 step 600: raw hook capture already includes the 0.5 objective weight.
    rng600 = capture_rng()
    model.zero_grad(set_to_none=True)
    _, m600 = _get_losses(model, batch, 600, rng600)
    capture600 = {}
    h600 = backward_gradient_controlled(model, m600["loss_recon"], m600["loss_understanding"], .5,
                                        audit_capture=capture600)
    actual600 = _clone_grads(model)
    g_r600 = capture600["g_recon_before_understanding"]
    g_u600 = capture600.get("g_under_raw", {})
    group_u600 = capture600.get("g_group_raw", {})
    expected600 = {n: _linear_gradient_combo(g_r600.get(n), g_u600.get(n), .01)
                   for n in rec_names}
    diff600_r = _grad_diff(expected600, actual600, rec_names)
    expected600_g = group_u600
    diff600_g = _grad_diff(expected600_g, actual600, group_names)
    c5 = {"understanding_weight": .5, "hook_capture_includes_weight": True,
          "reconstruction": diff600_r, "group": diff600_g,
          "hook_counts": h600,
          "passed": diff600_r["max_abs_diff"] <= 1e-6 and diff600_r["relative_l2_diff"] <= 1e-6 and
                    diff600_g["max_abs_diff"] <= 1e-6 and diff600_g["relative_l2_diff"] <= 1e-6}
    model.zero_grad(set_to_none=True)

    # C6 step 200: w=0 runs only reconstruction backward, with no temp hooks.
    rng200 = capture_rng()
    model.zero_grad(set_to_none=True)
    _, m200 = _get_losses(model, batch, 200, rng200)
    capture200 = {}
    h200 = backward_gradient_controlled(model, m200["loss_recon"], m200["loss_understanding"], 0.0,
                                        audit_capture=capture200)
    actual200 = _clone_grads(model)
    g_r200 = capture200["g_recon_before_understanding"]
    c6r = _grad_diff(g_r200, actual200, rec_names)
    group_zero = all(actual200.get(n) is None or torch.count_nonzero(actual200[n]) == 0 for n in group_names)
    c6 = {"understanding_weight": 0.0, "reconstruction": c6r,
          "group_none_or_zero": bool(group_zero), "hook_counts": h200,
          "group_grad_is_none_or_zero": bool(group_zero),
          "temporary_hook_count": h200["registered_hook_count"],
          "understanding_backward_skipped": h200["understanding_backward_skipped"],
          "passed": c6r["max_abs_diff"] <= 1e-6 and c6r["relative_l2_diff"] <= 1e-6 and
                    group_zero and h200["understanding_backward_skipped"]}
    model.zero_grad(set_to_none=True)

    optimizer, optimizer_audit = _optimizer_parity(model)
    c9 = {"optimizer_groups": optimizer_audit["groups"],
          "all_trainable_parameters_exactly_once": optimizer_audit["all_trainable_parameters_exactly_once"],
          "registered_four_group_counts_and_numel_match": optimizer_audit["optimizer_group_parity"],
          "passed": optimizer_audit["optimizer_group_parity"]}
    del optimizer
    del before, g_r, g_u, rec_expected, rec_actual, g_r1000, g_u1000, group_u1000
    del g_r600, g_u600, group_u600, actual600, expected600, expected600_g
    del g_r200, actual200, next_recon
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"GC-C2_raw_gradient_reference": c2,
            "GC-C3_reconstruction_formula_step1000": c3,
            "GC-C4_group_unscaled_step1000": c4,
            "GC-C5_warmup_composition_step600": c5,
            "GC-C6_zero_understanding_step200": c6,
            "GC-C7_hook_cleanup_and_next_recon": c7,
            "GC-C8_no_parameter_mutation": c8,
            "GC-C9_optimizer_parity": c9}, optimizer_audit


def _gradient_scale_audit(model, opt, device, manifest):
    model.eval()
    named = {n: p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("anchor_group.")}
    rows = []
    rng_outer = capture_rng()
    try:
        for index in GRAD_WINDOW_INDICES:
            if index >= len(manifest["windows"]):
                raise RuntimeError(f"gradient audit window {index} absent")
            window = manifest["windows"][index]
            batch = _batch_for(opt, window, device)
            rng = capture_rng()
            _, gr = _raw_task_grad(model, batch, 0, "loss_recon", rng)
            _, gu = _raw_task_grad(model, batch, 0, "loss_understanding", rng)
            categories = _gradient_window_stats(gr, gu, named)
            rows.append({"scene": window["scene"], "window_index": index,
                         "categories": categories})
            del gr, gu, batch
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GC gradient scale] {index}/1024 complete", flush=True)
    finally:
        restore_rng(rng_outer)
        model.zero_grad(set_to_none=True)
    summary = _summary_gradient_rows(rows)
    errors_ratio = [v["effective_ratio_relative_error"]
                    for row in rows for v in row["categories"].values()]
    errors_cos = [v["cosine_invariance_abs_error"]
                  for row in rows for v in row["categories"].values()]
    passed = max(errors_ratio, default=0.0) <= 1e-6 and max(errors_cos, default=0.0) <= 1e-6
    return {"model_state": "fresh_step0", "shared_understanding_grad_scale": SHARED_UNDERSTANDING_GRAD_SCALE,
            "window_indices": GRAD_WINDOW_INDICES,
            "interpretation": "positive scaling changes gradient scale, not gradient direction; it does not resolve direction conflict",
            "summary": summary, "windows": rows,
            "max_effective_ratio_relative_error": max(errors_ratio, default=0.0),
            "max_cosine_invariance_abs_error": max(errors_cos, default=0.0),
            "effective_ratio_and_cosine_contract_pass": passed}


def _run_audit(device):
    manifest, plan, monitor_audit = validate_locked_recipe()
    opt = build_options()
    source = load_state(PRETRAINED)
    model, transfer = make_model(opt, device, source)
    model.eval()
    if any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("GC audit expects all canonical model parameters trainable")
    _write(OUT / "pretrained_transfer_audit.json", transfer)
    window0 = manifest["windows"][0]
    batch0 = _batch_for(opt, window0, device)
    # The V1 and GC arms share architecture, forward and losses.  Compare two
    # independently fresh seeded states, then measure each loss's own repeat
    # envelope before judging scalar parity.
    gc_model, transfer_gc = make_model(opt, device, source)
    step1000_batch = _batch_for(opt, plan["entries"][999], device)
    parity_noise = _run_parity_noise_root_cause(device, model, gc_model, step1000_batch)
    c1 = {"C1a_forward_tensor_parity": parity_noise["C1a_forward_tensor_parity"],
          "C1b_empirical_loss_parity": parity_noise["C1b_empirical_loss_parity"],
          "passed": parity_noise["C1a_forward_tensor_parity"]["passed"] and
                    parity_noise["C1b_empirical_loss_parity"]["passed"]}
    if transfer != transfer_gc:
        c1["passed"] = False
        c1["transfer_audits_equal"] = False
    del gc_model, transfer_gc
    gc.collect(); torch.cuda.empty_cache()

    # Fresh reconstruction output remains exactly the canonical pretrained baseline.
    reconstruction_parity = _reconstruction_parity(model, opt, source, device, batch0)
    _write(OUT / "reconstruction_parity.json", reconstruction_parity)

    raw_gc, optimizer_audit = _run_raw_gc_contracts(model, opt, device, step1000_batch)
    del step1000_batch
    # C10/C11: unweighted task gradient scale on exactly the registered fresh 16 windows.
    scale_audit = _gradient_scale_audit(model, opt, device, manifest)
    _write(OUT / "gradient_scale_audit_fresh16.json", scale_audit)
    c10 = {"shared_scale": SHARED_UNDERSTANDING_GRAD_SCALE,
           "max_effective_ratio_relative_error": scale_audit["max_effective_ratio_relative_error"],
           "passed": scale_audit["max_effective_ratio_relative_error"] <= 1e-6}
    c11 = {"max_cosine_invariance_abs_error": scale_audit["max_cosine_invariance_abs_error"],
           "passed": scale_audit["max_cosine_invariance_abs_error"] <= 1e-6}

    phase_a = _phase_a_contract_replay()
    c12 = {**phase_a, "passed": phase_a["status"] == "pass"}
    legacy = _legacy_regression_replay()
    c14 = {"passed": legacy["status"] == "pass", **legacy}
    c13 = {"passed": reconstruction_parity["status"] == "pass", **reconstruction_parity}
    contracts = {
        "GC-C1_forward_parity": c1,
        **raw_gc,
        "GC-C10_effective_ratio_0_01x_raw": c10,
        "GC-C11_cosine_invariant_positive_scale": c11,
        "GC-C12_phase_a_18_of_18_regression": c12,
        "GC-C13_reconstruction_parity": c13,
        "GC-C14_s0_s1_legacy_regression": c14,
        "GC-C15_rtx3090_v1_vs_gc_one_step": {"passed": False, "status": "pending_smoke"},
    }
    total = len(contracts)
    passed = sum(bool(v["passed"]) for v in contracts.values())
    payload = {"experiment": "ANCHOR_GROUP_V1_GC_ALPHA001",
               "architecture": "LOCUSGS_ANCHOR_GROUP_V1",
               "shared_understanding_grad_scale": SHARED_UNDERSTANDING_GRAD_SCALE,
               "passed": passed, "total": total,
               "checks": contracts,
               "optimizer_audit": optimizer_audit,
               "locked_inputs": {"manifest_sha256": sha256(MANIFEST),
                                 "plan_sha256": sha256(PLAN),
                                 "pretrained_sha256": sha256(PRETRAINED),
                                 "monitor_sources_byte_identical_to_v1": monitor_audit},
               "status": "pass" if passed == total else "fail",
               "formal_training_started": False}
    _write(OUT / "gc_contracts.json", payload)
    _write_implementation_report(payload, scale_audit, None, parity_noise)
    del model, source, batch0
    gc.collect(); torch.cuda.empty_cache()
    non_smoke_checks = [v["passed"] for k, v in contracts.items()
                        if k != "GC-C15_rtx3090_v1_vs_gc_one_step"]
    if not all(non_smoke_checks):
        raise RuntimeError(f"GC audit contracts failed: {passed}/{total}")
    return payload, scale_audit


def _run_parity_noise_only(device):
    manifest, plan, _ = validate_locked_recipe()
    del manifest
    if int(plan["entries"][999].get("step", -1)) != 1000:
        raise RuntimeError("locked plan has no step-1000 entry at index 999")
    opt = build_options()
    source = load_state(PRETRAINED)
    model_v1, transfer_v1 = make_model(opt, device, source)
    model_gc, transfer_gc = make_model(opt, device, source)
    if transfer_v1 != transfer_gc:
        raise RuntimeError("fresh V1/GC transfer audits differ")
    batch = _batch_for(opt, plan["entries"][999], device)
    result = _run_parity_noise_root_cause(device, model_v1, model_gc, batch)
    del model_v1, model_gc, transfer_v1, transfer_gc, source, batch
    gc.collect(); torch.cuda.empty_cache()
    return result


def _parameter_vector_stats(model):
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    def norm(pred):
        return _report_stats(grads, pred)
    return {
        "global": norm(lambda _n: True),
        "reconstruction": norm(lambda n: not n.startswith("anchor_group.")),
        "anchor_group": norm(lambda n: n.startswith("anchor_group.")),
        "decoder": norm(lambda n: _category(n) == "decoder"),
        "activation_head": norm(lambda n: _category(n) == "activation_head"),
        "anchor_geometry": norm(lambda n: _category(n) == "anchor_geometry"),
    }


def _module_grad_norm(grads, name):
    g = grads.get(name)
    return 0.0 if g is None else float(g.double().norm())


def _choose_nonzero_name(grads, predicate):
    for n, g in grads.items():
        if predicate(n) and g is not None and torch.count_nonzero(g):
            return n
    raise RuntimeError("could not find a finite nonzero gradient parameter for smoke diagnostic")


def _cos_grad(a, b):
    if a is None or b is None:
        return None
    af, bf = a.double().reshape(-1), b.double().reshape(-1)
    den = float(af.norm() * bf.norm())
    return float(torch.dot(af, bf) / max(den, 1e-30))


def _run_one_model_smoke(model, optimizer, batch, rng, mode):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    restore_rng(rng)
    output, metrics = model.step_loss(batch, step=1000, coupled=False)
    losses = _scalar_metrics(metrics)
    for k, v in losses.items():
        if not math.isfinite(v):
            raise RuntimeError(f"nonfinite smoke loss {k}")
    forward_tensors = _capture_forward_tensors(output)
    if mode == "v1":
        metrics["loss"].backward()
        hooks = {"registered_hook_count": 0, "removed_hook_count": 0}
    else:
        hooks = backward_gradient_controlled(model, metrics["loss_recon"],
                                             metrics["loss_understanding"], 1.0)
    grads = _clone_grads(model)
    stats = _parameter_vector_stats(model)
    query_grad = grads["anchor_group.query_init"]
    late_name = _choose_nonzero_name(grads, lambda n: n.startswith("enc_dec_backbone.decoder_blocks.11."))
    activation_name = _choose_nonzero_name(grads, lambda n: n.startswith("activation_head."))
    selected = {"decoder": late_name, "anchor_mu": "anchor_decoder.mu",
                "activation_head": activation_name}
    selected_grad = {k: grads.get(n) for k, n in selected.items()}
    clip_norm = float(clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    after_clip = _parameter_vector_stats(model)
    coefficient = min(1.0, 1.0 / max(clip_norm, 1e-30))
    q0 = model.anchor_group.query_init.detach().cpu().clone()
    selected0 = {k: dict(model.named_parameters())[n].detach().cpu().clone()
                 for k, n in selected.items()}
    optimizer.step()
    qdelta = float((model.anchor_group.query_init.detach().cpu() - q0).abs().max())
    selected_delta = {k: float((dict(model.named_parameters())[n].detach().cpu() - selected0[k]).abs().max())
                      for k, n in selected.items()}
    return {"losses": losses, "preclip_gradient_norms": stats,
            "postclip_gradient_norms": after_clip,
            "pre_clip_global_norm": clip_norm,
            "clip_coefficient": coefficient,
            "post_clip_global_norm": after_clip["global"],
            "query_init_gradient": query_grad,
            "selected_parameter_names": selected,
            "selected_parameter_grad_norms": {k: _module_grad_norm(grads, name) for k, name in selected.items()},
            "selected_parameter_grads": selected_grad,
            "query_init_delta": qdelta,
            "selected_parameter_deltas": selected_delta,
            "forward_tensors": forward_tensors,
            "hook_counts": hooks,
            "forward_finite": True,
            "backward_finite": all(g is None or bool(torch.isfinite(g).all()) for g in grads.values()),
            "optimizer_step_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "_grad_map": grads}


def _run_one_step_smoke(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V1-vs-GC one-step smoke requires CUDA")
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("V1-vs-GC one-step smoke requires NVIDIA GeForce RTX 3090")
    if torch.cuda.get_device_properties(device).total_memory < 23 * 1024**3:
        raise RuntimeError("V1-vs-GC smoke requires a 24GB-class RTX 3090")
    manifest, plan, _ = validate_locked_recipe()
    entry = plan["entries"][999]
    if entry["step"] != 1000:
        raise RuntimeError("plan step 1000 entry mismatch")
    opt = build_options()
    source = load_state(PRETRAINED)
    base, transfer_base = make_model(opt, device, source)
    base_initial_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    base_optimizer, base_audit = _optimizer_parity(base)
    set_optimizer_lr(base_optimizer, 1000)
    batch = _batch_for(opt, entry, device)
    batch_manifest = _batch_tensor_manifest(batch)
    rng = capture_rng()
    v1 = _run_one_model_smoke(base, base_optimizer, batch, rng, "v1")
    base_query_grad = v1["query_init_gradient"]
    base_selected_grad = v1["selected_parameter_grads"]
    base_selected_norms = v1["selected_parameter_grad_norms"]
    base_deltas = {"query_init": v1["query_init_delta"], **v1["selected_parameter_deltas"]}
    base_losses = v1["losses"]
    base_gradnorms = v1["preclip_gradient_norms"]
    base_clip = {"pre_clip_global_norm": v1["pre_clip_global_norm"],
                 "clip_coefficient": v1["clip_coefficient"],
                 "post_clip_global_norm": v1["post_clip_global_norm"]}
    base_names = v1["selected_parameter_names"]
    base_forward_tensors = v1["forward_tensors"]
    base_all_losses = v1["losses"]
    del base, base_optimizer, v1
    gc.collect(); torch.cuda.empty_cache()

    gc_model, transfer_gc = make_model(opt, device, source)
    model_state_parity = _state_tensor_parity(base_initial_state, gc_model)
    gc_optimizer, gc_audit = _optimizer_parity(gc_model)
    set_optimizer_lr(gc_optimizer, 1000)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device) / 1024**3
    reserved_before = torch.cuda.memory_reserved(device) / 1024**3
    controlled = _run_one_model_smoke(gc_model, gc_optimizer, batch, rng, "gc")
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    qdiff = _max_abs_tensor(base_query_grad, controlled["query_init_gradient"])
    grad_compare = {}
    for key, name in base_names.items():
        bg, gg = base_selected_grad[key], controlled["selected_parameter_grads"][key]
        grad_compare[key] = {"parameter_name": name,
                             "v1_grad_norm": base_selected_norms[key],
                             "gc_grad_norm": controlled["selected_parameter_grad_norms"][key],
                             "v1_vs_gc_cosine": _cos_grad(bg, gg),
                             "max_abs_diff": _max_abs_tensor(bg, gg)}
    forward_parity = _forward_tensor_diffs(base_forward_tensors, controlled["forward_tensors"])
    root_noise = json.loads((OUT / "parity_noise_root_cause.json").read_text())
    envelopes = root_noise["noise_envelopes"]
    all_loss_names = sorted(set(base_all_losses) | set(controlled["losses"]))
    loss_diffs = {k: abs(base_all_losses[k] - controlled["losses"][k])
                  for k in all_loss_names if k in base_all_losses and k in controlled["losses"]}
    loss_gate = {}
    for key in all_loss_names:
        if key not in base_all_losses or key not in controlled["losses"] or key not in envelopes:
            loss_gate[key] = {"passed": False, "reason": "missing arm metric or empirical envelope"}
        else:
            diff = abs(base_all_losses[key] - controlled["losses"][key])
            loss_gate[key] = {"v1": base_all_losses[key], "gc": controlled["losses"][key],
                              "abs_diff": diff, "noise_envelope": envelopes[key],
                              "passed": diff <= envelopes[key]}
    loss_parity_pass = bool(loss_gate) and all(x.get("passed", False) for x in loss_gate.values())
    loss_exact = all(v == 0.0 for v in loss_diffs.values())
    batch_exact = _batch_tensor_manifest(batch) == batch_manifest
    optimizer_exact = (base_audit["groups"] == gc_audit["groups"] and
                       base_audit["betas"] == gc_audit["betas"] and
                       base_audit["all_trainable_parameters_exactly_once"] and
                       gc_audit["all_trainable_parameters_exactly_once"])
    lr_exact = all(float(a["lr"]) == float(b["lr"])
                   for a, b in zip(base_audit["groups"], gc_audit["groups"]))
    hooks_clean = controlled["hook_counts"]["registered_hook_count"] == controlled["hook_counts"]["removed_hook_count"]
    query_updated = base_deltas["query_init"] > 0 and controlled["query_init_delta"] > 0
    recon_updated = all(base_deltas[k] > 0 and controlled["selected_parameter_deltas"][k] > 0
                        for k in base_names)
    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "step": 1000, "understanding_weight": 1.0,
        "shared_understanding_grad_scale": SHARED_UNDERSTANDING_GRAD_SCALE,
        "losses_v1": base_losses, "losses_gc": controlled["losses"],
        "all_scalar_loss_components_v1": base_all_losses,
        "all_scalar_loss_components_gc": controlled["losses"],
        "loss_abs_diff": loss_diffs, "losses_exact": loss_exact,
        "empirical_loss_parity": {"metric_gates": loss_gate,
                                   "all_components_pass": loss_parity_pass},
        "model_state_exact_before_forward": model_state_parity,
        "batch_exact_and_shared": {"same_batch_object": True,
                                    "tensor_manifest_before": batch_manifest,
                                    "tensor_manifest_after_equal": batch_exact},
        "forward_tensor_parity": forward_parity,
        "baseline_v1": {"preclip_gradient_norms": base_gradnorms, **base_clip,
                        "query_init_delta": base_deltas["query_init"],
                        "selected_parameter_deltas": {k: base_deltas[k] for k in base_names}},
        "gc": {"preclip_gradient_norms": controlled["preclip_gradient_norms"],
               "postclip_gradient_norms": controlled["postclip_gradient_norms"],
               "pre_clip_global_norm": controlled["pre_clip_global_norm"],
               "clip_coefficient": controlled["clip_coefficient"],
               "post_clip_global_norm": controlled["post_clip_global_norm"],
               "query_init_delta": controlled["query_init_delta"],
               "selected_parameter_deltas": controlled["selected_parameter_deltas"],
               "allocated_before_gib": allocated_before, "reserved_before_gib": reserved_before,
               "peak_memory_allocated_gib": peak_alloc, "peak_memory_reserved_gib": peak_reserved},
        "query_init_gradient_max_abs_diff_before_clip": qdiff,
        "selected_parameter_gradient_comparison": grad_compare,
        "optimizer_groups_identical": optimizer_exact,
        "learning_rates_identical": lr_exact,
        "pretrained_transfer_audits_equal": transfer_base == transfer_gc,
        "hook_registered_removed_equal": hooks_clean,
        "query_parameter_updated_both_arms": query_updated,
        "selected_reconstruction_parameters_updated_both_arms": recon_updated,
        "finite": all(controlled[k] for k in ("forward_finite", "backward_finite", "optimizer_step_finite")),
        "optimizer_step_count": 1,
        "status": "pass" if model_state_parity["model_state_equal"] and batch_exact and
                  forward_parity["all_exact"] and loss_parity_pass and optimizer_exact and lr_exact and
                  qdiff <= 1e-6 and hooks_clean and query_updated and recon_updated and
                  all(controlled[k] for k in ("forward_finite", "backward_finite", "optimizer_step_finite")) else "fail",
    }
    _write(OUT / "v1_vs_gc_one_step_smoke.json", payload)
    del gc_model, gc_optimizer, source, batch, base_initial_state, base_selected_grad, controlled
    gc.collect(); torch.cuda.empty_cache()
    return payload


def _write_implementation_report(contracts_payload, scale_audit, smoke, parity_noise=None):
    checks = contracts_payload.get("checks", {})
    lines = [
        "# Anchor-Group V1-GC Implementation Audit", "",
        "Experiment identity: `ANCHOR_GROUP_V1_GC_ALPHA001`; architecture remains `LOCUSGS_ANCHOR_GROUP_V1` / model type `siu3r_anchor_group_locusgs`.", "",
        "## Gradient routing", "",
        "Forward and loss math use the unchanged V1 model/loss implementation. The logged total remains `L_recon + understanding_weight * L_understanding`; only backward routing changes. Reconstruction loss is unscaled. During understanding backward only, reconstruction-parameter gradient contributions receive fixed factor `0.01`; `anchor_group.*` receives the full understanding contribution. Warm-up remains the canonical V1 helper, the four-group AdamW is reused, and global clip remains 1.0 after both backward contributions.", "",
        "Step 200 has weight 0, skips the understanding backward and matches V1 reconstruction-only gradients. Step 600 has weight 0.5 and applies `gR + 0.005*gU` to reconstruction, `0.5*gU` to group. Step 1000 applies `gR + 0.01*gU` to reconstruction and `1.0*gU` to group.", "",
        "Hooks are registered only on unique, trainable, non-`anchor_group.*` parameters around the understanding backward and are removed in `finally`. The helper contains no optimizer step. The future train driver is registered in this script but was not invoked.", "",
        "## Scalar parity noise audit", "",
        (f"The C1 gate uses exact equality for model state, reused batch, Gaussian/RGB/A_post/region_mass/semantic/class outputs, and empirical same-arm repeat envelopes for scalar losses. GPU reduction/render repeat noise is measured independently per scalar; no fixed numeric tolerance replaces that envelope. Root cause: `{parity_noise['root_cause']['source']}`; first non-exact component: `{parity_noise['root_cause']['first_nonexact_reconstruction_component']}`." if parity_noise else "C1 scalar parity uses the recorded same-arm empirical repeat-noise envelope."), "",
        "## Contracts", "",
        f"GC contracts: **{contracts_payload.get('passed', 0)}/{contracts_payload.get('total', 0)} PASS** before the optional smoke entry is added.", "",
        "| Contract | Result |", "|---|---|"]
    for name, result in checks.items():
        lines.append(f"| {name} | {'PASS' if result.get('passed') else 'FAIL'} |")
    if scale_audit:
        lines.extend(["", "## Fresh step0 gradient scale", "",
                      "| Category | raw median ratio | effective median ratio | combined/recon norm median | combined vs recon cosine median | frac cosine < 0.9 | frac < 0.5 | frac < 0 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for cat in CATEGORY_NAMES:
            s = scale_audit["summary"][cat]
            lines.append(f"| {cat} | {s['raw_ratio_median']:.6g} | {s['effective_ratio_median']:.6g} | {s['combined_over_recon_norm_median']:.6g} | {s['combined_vs_recon_cosine_median']:.6f} | {s['fraction_combined_cosine_lt_0_9']:.4f} | {s['fraction_combined_cosine_lt_0_5']:.4f} | {s['fraction_combined_cosine_lt_0']:.4f} |")
        lines.append("")
        lines.append(f"Maximum relative effective/raw ratio error: `{scale_audit['max_effective_ratio_relative_error']:.3g}`; maximum raw/effective cosine difference: `{scale_audit['max_cosine_invariance_abs_error']:.3g}`. Positive scaling controls magnitude, not direction conflict.")
    if smoke:
        lines.extend(["", "## RTX 3090 one-step A/B smoke", "",
                      f"Status: **{smoke['status']}**; step 1000, one optimizer step per independent fresh model. Empirical component-loss gate: `{smoke['empirical_loss_parity']['all_components_pass']}`; strict forward tensor parity: `{smoke['forward_tensor_parity']['all_exact']}`. Query gradient max diff before clip: `{smoke['query_init_gradient_max_abs_diff_before_clip']:.3g}`. GC peak allocated/reserved: `{smoke['gc']['peak_memory_allocated_gib']:.3f}/{smoke['gc']['peak_memory_reserved_gib']:.3f} GiB.", ""])
    lines.extend(["", "No formal V1-GC 5000-step training was started. No query-starvation fix was introduced. No model, loss, Hungarian definition, optimizer, or LR was changed.", ""])
    (OUT / "implementation_audit.md").write_text("\n".join(lines))


def _train_formal(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("future formal V1-GC training requires CUDA")
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("future formal V1-GC training is registered for RTX 3090")
    manifest, plan, _ = validate_locked_recipe()
    OUT.mkdir(parents=True, exist_ok=True)
    monitor_audit = {}
    for name in MONITORS:
        src, dst = SOURCE_REPORTS / name, OUT / name
        shutil.copyfile(src, dst)
        monitor_audit[name] = {"source_sha256": sha256(src), "copy_sha256": sha256(dst),
                               "equal": sha256(src) == sha256(dst)}
        if not monitor_audit[name]["equal"]:
            raise RuntimeError(f"monitor byte identity failure: {name}")
    opt = build_options()
    model, transfer = make_model(opt, device)
    if model.architecture_name != "LOCUSGS_ANCHOR_GROUP_V1" or any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("future GC run requires unchanged architecture with every V1 parameter trainable")
    optimizer, optimizer_audit = build_optimizer(model)
    if not optimizer_audit["all_trainable_parameters_exactly_once"]:
        raise RuntimeError("V1 optimizer parameter coverage failed")
    model.train()
    for entry in plan["entries"]:
        step = int(entry["step"])
        set_optimizer_lr(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        batch = _batch_for(opt, entry, device)
        _output, metrics = model.step_loss(batch, step=step, coupled=False)
        if not all(bool(torch.isfinite(metrics[k]).all()) for k in ("loss", "loss_recon", "loss_understanding")):
            raise RuntimeError(f"nonfinite loss at formal step {step}")
        backward_gradient_controlled(model, metrics["loss_recon"], metrics["loss_understanding"],
                                     metrics["understanding_weight"])
        clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            keys = ("loss", "loss_recon", "loss_understanding", "understanding_weight",
                    "loss_thing_2d", "loss_stuff_2d", "loss_semantic", "loss_identity",
                    "loss_anchor_group", "anchor_ce", "anchor_dice")
            print(json.dumps({"event": "train_step", "step": step,
                              **{k: float(metrics[k].detach()) if torch.is_tensor(metrics[k]) else float(metrics[k]) for k in keys},
                              "shared_understanding_grad_scale": SHARED_UNDERSTANDING_GRAD_SCALE,
                              "group_lr": next(g["lr"] for g in optimizer.param_groups if g["name"].startswith("anchor_group_")),
                              "reconstruction_lr": next(g["lr"] for g in optimizer.param_groups if g["name"].startswith("reconstruction_"))}, sort_keys=True), flush=True)
        if step in EVAL_STEPS[1:]:
            evaluate_anchor_group_all(model, opt, OUT, step, device,
                                      _seen_classes(OUT))
        del batch, _output, metrics
    endpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": TOTAL_STEPS, "architecture": "LOCUSGS_ANCHOR_GROUP_V1",
                "recipe": "ANCHOR_GROUP_V1_GC_ALPHA001", "joint": True, "beta": 0,
                "shared_understanding_grad_scale": SHARED_UNDERSTANDING_GRAD_SCALE,
                "manifest_sha256": sha256(MANIFEST), "plan_sha256": sha256(PLAN),
                "pretrained_sha256": sha256(PRETRAINED),
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    torch.save(endpoint, WORKSPACE / "formal_endpoint_step5000.pt")
    del transfer, monitor_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "parity-noise", "smoke", "train"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.phase in ("audit", "parity-noise", "smoke") and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("GC audit and one-step smoke require CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        _run_audit(device)
    elif args.phase == "parity-noise":
        _run_parity_noise_only(device)
    elif args.phase == "smoke":
        smoke = _run_one_step_smoke(device)
        contracts = json.loads((OUT / "gc_contracts.json").read_text())
        contracts["checks"]["GC-C15_rtx3090_v1_vs_gc_one_step"] = {
            "passed": smoke["status"] == "pass", "status": smoke["status"],
            "losses_exact": smoke["losses_exact"],
            "empirical_loss_components_pass": smoke["empirical_loss_parity"]["all_components_pass"],
            "forward_tensors_exact": smoke["forward_tensor_parity"]["all_exact"],
            "model_state_exact": smoke["model_state_exact_before_forward"]["model_state_equal"],
            "batch_exact": smoke["batch_exact_and_shared"]["tensor_manifest_after_equal"],
            "optimizer_and_lr_exact": smoke["optimizer_groups_identical"] and smoke["learning_rates_identical"],
            "query_init_gradient_max_abs_diff_before_clip": smoke["query_init_gradient_max_abs_diff_before_clip"],
            "hook_cleanup_pass": smoke["hook_registered_removed_equal"],
            "peak_memory_allocated_gib": smoke["gc"]["peak_memory_allocated_gib"],
            "peak_memory_reserved_gib": smoke["gc"]["peak_memory_reserved_gib"]}
        contracts["total"] = len(contracts["checks"])
        contracts["passed"] = sum(bool(x["passed"]) for x in contracts["checks"].values())
        contracts["status"] = "pass" if contracts["passed"] == contracts["total"] == 15 else "fail"
        _write(OUT / "gc_contracts.json", contracts)
        scale = json.loads((OUT / "gradient_scale_audit_fresh16.json").read_text())
        root_noise = json.loads((OUT / "parity_noise_root_cause.json").read_text())
        _write_implementation_report(contracts, scale, smoke, root_noise)
        if contracts["status"] != "pass":
            raise RuntimeError(f"V1-GC contracts failed after smoke: {contracts['passed']}/{contracts['total']}")
    else:
        _train_formal(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
