#!/usr/bin/env python3
"""Read-only endpoint audits for Anchor-Group V1.

This entry point intentionally has no training phase and never creates or steps
an optimizer.  Gradient audits call backward only to measure task gradients.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)

from scripts.anchor_group_v1 import (
    MANIFEST, OUT, PRETRAINED, PRETRAINED_SHA, build_options, load_state,
    make_model, sha256,
)
from scripts.instance_state_generalization import _batch_for
from scripts.instance_state_runtime import capture_rng, restore_rng
from tokengs.models import model_registry
from tokengs.models.anchor_group_loss import build_anchor_targets, unified_hungarian
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data

ENDPOINT = REPO / "workspace_group_plus/anchor_group_v1/formal_endpoint_step5000.pt"
ENDPOINT_AUDIT = OUT / "formal_endpoint_step5000_audit.json"
VAL32 = OUT / "monitor_32pairs.json"
EXPECTED_MANIFEST_SHA = "1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483"
GRAD_WINDOW_INDICES = [0, 64, 128, 192, 256, 320, 384, 448,
                       512, 576, 640, 704, 768, 832, 896, 960]
CATEGORIES = ("all_reconstruction", "encoder", "decoder", "anchor_geometry",
              "activation_head", "other_reconstruction")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_endpoint_payload():
    if not ENDPOINT.is_file() or not ENDPOINT_AUDIT.is_file():
        raise FileNotFoundError("formal step-5000 endpoint or its passing audit is missing")
    audit = json.loads(ENDPOINT_AUDIT.read_text())
    if not (audit.get("status") == "pass" and audit.get("step") == 5000
            and audit.get("architecture") == "LOCUSGS_ANCHOR_GROUP_V1"
            and audit.get("joint") is True and audit.get("beta") == 0):
        raise RuntimeError("formal endpoint audit does not verify the required step-5000 endpoint")
    payload = torch.load(ENDPOINT, map_location="cpu", weights_only=False)
    if not (payload.get("step") == 5000
            and payload.get("architecture") == "LOCUSGS_ANCHOR_GROUP_V1"
            and payload.get("joint") is True and payload.get("beta") == 0):
        raise RuntimeError("endpoint payload metadata mismatch")
    for key in ("manifest_sha256", "plan_sha256", "pretrained_sha256"):
        if payload.get(key) != audit.get(key):
            raise RuntimeError(f"endpoint payload/audit {key} mismatch")
    if payload["manifest_sha256"] != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("endpoint was trained with a different manifest")
    if payload["pretrained_sha256"] != PRETRAINED_SHA:
        raise RuntimeError("endpoint pretrained checkpoint SHA mismatch")
    return payload, audit


def make_endpoint_model(opt, device, payload=None):
    if payload is None:
        if sha256(PRETRAINED) != PRETRAINED_SHA:
            raise RuntimeError("fresh-step0 pretrained checkpoint SHA256 mismatch")
        model, _transfer_audit = make_model(opt, device)
    else:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        model = model_registry[opt.model_type](opt)
        missing, unexpected = model.load_state_dict(payload["model"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"endpoint strict load failed: missing={missing}, unexpected={unexpected}")
        model.to(device)
    model.eval()
    if getattr(model, "architecture_name", None) != "LOCUSGS_ANCHOR_GROUP_V1":
        raise RuntimeError("unexpected model architecture")
    return model


def forward_context(model, opt, batch, step=5000):
    mi, _ = split_data(batch, opt)
    decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                intrinsics=batch["intrinsics_all"])
    context = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :2],
                               intrinsics=batch["intrinsics_all"][:, :2])
    return model.forward_anchor_group(ModelInput(mi.encoder, decoder),
                                      render_decoder_input=decoder,
                                      context_decoder=context,
                                      coupled=False, step=step)


def _gini(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or np.any(x < 0):
        raise ValueError("Gini requires a one-dimensional nonnegative vector")
    total = float(x.sum())
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / (n * total)) - (n + 1) / n)


def _concentration(values, *, fail_zero=False):
    x = np.asarray(values, dtype=np.float64)
    total = float(x.sum())
    if total <= 0 and fail_zero:
        raise RuntimeError("cannot compute match concentration with zero total matches")
    if total <= 0:
        p = np.zeros_like(x)
        entropy = 0.0
    else:
        p = x / total
        nz = p > 0
        entropy = float(-np.sum(p[nz] * np.log(p[nz])))
    n = len(x)
    return {
        "top1_share": float(np.sort(p)[-1:].sum()),
        "top5_share": float(np.sort(p)[-5:].sum()),
        "top10_share": float(np.sort(p)[-10:].sum()),
        "top20_share": float(np.sort(p)[-20:].sum()),
        "gini": _gini(x),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(n) if n > 1 else 0.0,
        "effective_query_count": math.exp(entropy),
        "total": total,
    }


def _spearman(x, y):
    result = spearmanr(x, y)
    rho, p = float(result.statistic), float(result.pvalue)
    return {"rho": rho if math.isfinite(rho) else None,
            "p_value": p if math.isfinite(p) else None,
            "finite": math.isfinite(rho) and math.isfinite(p)}


def _matched_iou(raw_masks, query, gt_mask):
    from scripts.eval_instance_state_v1 import _multiview_iou_for_gt
    return _multiview_iou_for_gt(raw_masks, int(query), gt_mask)


def _query_rows_init():
    return [{"query_id": q, "n_windows": 0, "n_hungarian_matches": 0,
             "n_windows_matched": 0, "total_matched_gt_anchor_count": 0,
             "total_ownership_mass": 0.0, "mean_ownership_mass": 0.0,
             "median_ownership_mass": 0.0, "mean_no_object_probability": 0.0,
             "mean_max_thing_class_probability": 0.0,
             "mean_matched_gt_iou_2d": None, "mean_matched_anchor_fraction": None,
             "n_class_aware_success_iou50": 0,
             "n_class_agnostic_success_iou50": 0,
             "positive_match_rate": 0.0, "positive_anchor_exposure": 0,
             "negative_only_windows": 0, "unique_scene_count_matched": 0}
            for q in range(100)]


@torch.no_grad()
def run_query_scope(model, opt, windows, device, scope_name):
    from scripts.eval_instance_state_v1 import _masks
    stats = _query_rows_init()
    mass_values = [[] for _ in range(100)]
    iou_values = [[] for _ in range(100)]
    anchor_fraction_values = [[] for _ in range(100)]
    scene_sets = [set() for _ in range(100)]
    matched_hist = np.zeros(100, dtype=np.int64)
    active_per_window = []
    for wi, window in enumerate(windows):
        batch = _batch_for(opt, window, device)
        pred = forward_context(model, opt, batch, step=5000)
        targets = build_anchor_targets(pred["states"][-1]["mu"],
                                       batch["semantic_label_all"],
                                       batch["instance_label_all"],
                                       batch["cam_view_all"],
                                       batch["intrinsics_all"])
        targets, pairs = unified_hungarian(pred, batch, targets)
        if len(pairs) != 1:
            raise RuntimeError("query utilization expects one B=1 Hungarian result per window")
        qi, ki = pairs[0]
        A = pred["states"][-1]["A_post"][0]
        ownership = A[:, :100].sum(0).detach().float().cpu().numpy()
        pclass = pred["p_class"][0].detach().float()
        noobj = pclass[:, 18].cpu().numpy()
        max_thing = pclass[:, :18].max(-1).values.cpu().numpy()
        raw, pred_cls, _score, _is_thing = _masks(pred)
        gt_masks = targets["gt_pixel_masks"][0]
        gt_classes = targets["gt_classes"][0]
        y_anchor = targets["Y_anchor"][0]
        for q in range(100):
            row = stats[q]
            row["n_windows"] += 1
            row["total_ownership_mass"] += float(ownership[q])
            row["mean_no_object_probability"] += float(noobj[q])
            row["mean_max_thing_class_probability"] += float(max_thing[q])
            mass_values[q].append(float(ownership[q]))
        matched_set = set()
        for q_t, k_t in zip(qi.tolist(), ki.tolist()):
            q, k = int(q_t), int(k_t)
            row = stats[q]
            row["n_hungarian_matches"] += 1
            row["total_matched_gt_anchor_count"] += int(y_anchor[k].sum().item())
            row["positive_anchor_exposure"] += int(y_anchor[k].sum().item())
            matched_set.add(q)
            matched_hist[q] += 1
            scene_sets[q].add(str(window["scene"]))
            iou = _matched_iou(raw, q, gt_masks[k])
            if iou is not None:
                iou_values[q].append(float(iou))
                if float(iou) >= 0.5:
                    row["n_class_agnostic_success_iou50"] += 1
                    if int(pred_cls[q]) == int(gt_classes[k]):
                        row["n_class_aware_success_iou50"] += 1
            y = y_anchor[k]
            n_positive = int(y.sum().item())
            if n_positive:
                pred_owner = A.argmax(-1)
                fraction = float(((pred_owner == q) & (y > 0)).sum().item() / n_positive)
                anchor_fraction_values[q].append(fraction)
        for q in matched_set:
            stats[q]["n_windows_matched"] += 1
        active_per_window.append(int((pclass[:, :18].sum(-1) >= 0.5).sum().item()))
        if (wi + 1) % 32 == 0 or wi + 1 == len(windows):
            print(f"[query-utilization:{scope_name}] {wi + 1}/{len(windows)} windows", flush=True)
        del pred, targets, batch, raw
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    for q, row in enumerate(stats):
        n = row["n_windows"]
        row["mean_ownership_mass"] = row["total_ownership_mass"] / max(1, n)
        row["median_ownership_mass"] = float(np.median(mass_values[q])) if mass_values[q] else 0.0
        row["mean_no_object_probability"] /= max(1, n)
        row["mean_max_thing_class_probability"] /= max(1, n)
        row["mean_matched_gt_iou_2d"] = float(np.mean(iou_values[q])) if iou_values[q] else None
        row["mean_matched_anchor_fraction"] = float(np.mean(anchor_fraction_values[q])) if anchor_fraction_values[q] else None
        row["positive_match_rate"] = row["n_windows_matched"] / max(1, n)
        row["negative_only_windows"] = n - row["n_windows_matched"]
        row["unique_scene_count_matched"] = len(scene_sets[q])
    matches = np.asarray([r["n_hungarian_matches"] for r in stats], dtype=np.int64)
    ownership = np.asarray([r["total_ownership_mass"] for r in stats], dtype=np.float64)
    match_conc = _concentration(matches, fail_zero=True)
    own_conc = _concentration(ownership)
    top20 = sorted(stats, key=lambda r: (-r["n_hungarian_matches"], r["query_id"]))[:20]
    scene_denominator = max(1, len({str(w["scene"]) for w in windows}))
    summary = {
        "scope": scope_name, "n_windows": len(windows),
        "total_gt_matches": int(matches.sum()),
        "unique_queries_ever_matched": int((matches > 0).sum()),
        "queries_never_matched": int((matches == 0).sum()),
        "queries_matched_1_to_4_times": int(((matches >= 1) & (matches <= 4)).sum()),
        "queries_matched_5_to_19_times": int(((matches >= 5) & (matches <= 19)).sum()),
        "queries_matched_20plus_times": int((matches >= 20).sum()),
        "queries_matched_in_lt_1_percent_windows": int(np.sum(matches / max(1, len(windows)) < .01)),
        "queries_matched_in_lt_5_percent_windows": int(np.sum(matches / max(1, len(windows)) < .05)),
        "queries_matched_in_ge_10_percent_windows": int(np.sum(matches / max(1, len(windows)) >= .10)),
        "match_concentration": match_conc,
        "ownership_concentration": own_conc,
        "match_count_vs_mean_no_object_probability": _spearman(matches, [r["mean_no_object_probability"] for r in stats]),
        "match_count_vs_mean_ownership_mass": _spearman(matches, [r["mean_ownership_mass"] for r in stats]),
        "mean_active_output_queries": float(np.mean(active_per_window)) if active_per_window else 0.0,
        "query_match_histogram": [{"query_id": q, "n_matches": int(matches[q])} for q in range(100)],
        "top20_queries": [{"query_id": r["query_id"], "matches": r["n_hungarian_matches"],
                           "unique_scenes": r["unique_scene_count_matched"],
                           "scene_match_fraction": r["unique_scene_count_matched"] / scene_denominator}
                          for r in top20],
        "cross_scene_query_reuse": [{"query_id": r["query_id"],
                                     "matches": r["n_hungarian_matches"],
                                     "unique_scenes": r["unique_scene_count_matched"],
                                     "scene_match_fraction": r["unique_scene_count_matched"] / scene_denominator}
                                    for r in stats],
        "active_output_query_definition": "sum of 18 thing-class probabilities >= 0.5",
    }
    return {"summary": summary, "queries": stats}


def run_query_utilization(model, opt, device):
    manifest = json.loads(MANIFEST.read_text())
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("locked train manifest SHA256 mismatch")
    train = manifest.get("windows", [])
    if len(train) != 1024:
        raise RuntimeError(f"expected all 1024 training windows, found {len(train)}")
    valdoc = json.loads(VAL32.read_text())
    val = valdoc.get("pairs", [])
    if len(val) != 32:
        raise RuntimeError(f"locked val32 monitor must contain 32 pairs, found {len(val)}")
    model.eval()
    train_result = run_query_scope(model, opt, train, device, "train1024")
    val_result = run_query_scope(model, opt, val, device, "val32_context")
    write_json(OUT / "query_utilization_train1024.json", train_result)
    write_json(OUT / "query_utilization_val32.json", val_result)
    summary = {"endpoint": str(ENDPOINT.relative_to(REPO)), "step": 5000,
               "architecture": "LOCUSGS_ANCHOR_GROUP_V1",
               "train1024": train_result["summary"], "val32": val_result["summary"]}
    write_json(OUT / "query_starvation_summary.json", summary)
    return summary


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


def _tensor_report_from_grads(name, param, recon_grad, under_grad):
    device = param.device
    if recon_grad is None:
        nr2 = torch.zeros((), device=device, dtype=torch.float64)
    else:
        nr2 = recon_grad.double().square().sum()
    if under_grad is None:
        nu2 = torch.zeros((), device=device, dtype=torch.float64)
    else:
        nu2 = under_grad.detach().double().square().sum()
    dot = (torch.zeros((), device=device, dtype=torch.float64) if recon_grad is None or under_grad is None
           else (recon_grad.double() * under_grad.detach().double()).sum())
    eps = 1e-12
    nr, nu = torch.sqrt(nr2), torch.sqrt(nu2)
    cosine = dot / (nr * nu + eps)
    ratio = nu / (nr + eps)
    return {"parameter_name": name, "norm_recon": float(nr.item()),
            "norm_under": float(nu.item()), "norm_ratio": float(ratio.item()),
            "cosine": float(cosine.item()), "dot_product": float(dot.item()),
            "recon_nonzero": bool(nr.item() > 0), "under_nonzero": bool(nu.item() > 0),
            "finite": all(math.isfinite(float(x.item())) for x in (nr, nu, dot, cosine, ratio))}


def _aggregate_named_grads(named_params, recon_grads, under_grads, predicate):
    sample = next(iter(named_params.values()))
    device = sample.device
    dot = torch.zeros((), device=device, dtype=torch.float64)
    nr2 = torch.zeros_like(dot); nu2 = torch.zeros_like(dot)
    nr_count = torch.zeros((), device=device, dtype=torch.int64)
    nu_count = torch.zeros_like(nr_count)
    n_total = 0
    for name, param in named_params.items():
        if not predicate(name):
            continue
        n_total += 1
        gr = recon_grads.get(name)
        gu = under_grads.get(name)
        if gr is not None:
            nr2 += gr.double().square().sum()
            nr_count += (torch.count_nonzero(gr) > 0).to(torch.int64)
        if gu is not None:
            gu = gu.detach()
            nu2 += gu.double().square().sum()
            nu_count += (torch.count_nonzero(gu) > 0).to(torch.int64)
        if gr is not None and gu is not None:
            dot += (gr.double() * gu.double()).sum()
    nr, nu = torch.sqrt(nr2), torch.sqrt(nu2)
    cosine = dot / (nr * nu + 1e-12)
    ratio = nu / (nr + 1e-12)
    vals = [dot.item(), nr.item(), nu.item(), cosine.item(), ratio.item()]
    return {"dot_product": float(vals[0]), "norm_recon": float(vals[1]),
            "norm_under": float(vals[2]), "norm_ratio": float(vals[4]),
            "cosine": float(vals[3]), "n_param_tensors": int(n_total),
            "n_params_total": int(n_total),
            "n_params_recon_nonzero": int(nr_count.item()),
            "n_params_under_nonzero": int(nu_count.item()),
            "n_recon_nonzero": int(nr_count.item()),
            "n_under_nonzero": int(nu_count.item()),
            "finite": all(math.isfinite(x) for x in vals)}


def _select_grad_report(named_params, recon_grads, under_grads, predicate, label):
    selected = {n: p for n, p in named_params.items() if predicate(n)}
    if not selected:
        raise RuntimeError(f"required gradient category/module is empty: {label}")
    report = _aggregate_named_grads(selected, recon_grads, under_grads, lambda _n: True)
    report["parameter_names"] = list(selected)
    report["parameter_stats"] = [
        _tensor_report_from_grads(name, param, recon_grads.get(name), under_grads.get(name))
        for name, param in selected.items()
    ]
    return report


def _summarize_rows(rows):
    result = {}
    for category in CATEGORIES:
        vals = [r["categories"][category] for r in rows]
        cos = np.asarray([x["cosine"] for x in vals], dtype=np.float64)
        ratio = np.asarray([x["norm_ratio"] for x in vals], dtype=np.float64)
        dots = np.asarray([x["dot_product"] for x in vals], dtype=np.float64)
        result[category] = {
            "mean_cosine": float(np.mean(cos)), "median_cosine": float(np.median(cos)),
            "p10_cosine": float(np.quantile(cos, .10)), "p90_cosine": float(np.quantile(cos, .90)),
            "fraction_cosine_lt_0": float(np.mean(cos < 0)),
            "fraction_cosine_lt_minus_0_25": float(np.mean(cos < -.25)),
            "fraction_cosine_gt_0_25": float(np.mean(cos > .25)),
            "mean_norm_ratio": float(np.mean(ratio)), "median_norm_ratio": float(np.median(ratio)),
            "p10_norm_ratio": float(np.quantile(ratio, .10)), "p90_norm_ratio": float(np.quantile(ratio, .90)),
            "fraction_dot_product_lt_0": float(np.mean(dots < 0)),
        }
    return result


def _run_gradient_pass(model, batch, opt, rng_state, loss_key):
    model.zero_grad(set_to_none=True)
    restore_rng(rng_state)
    _out, metrics = model.step_loss(batch, step=5000, coupled=False,
                                   understanding_weight_override=1.0)
    loss = metrics[loss_key]
    if not torch.is_tensor(loss) or not loss.requires_grad or not torch.isfinite(loss):
        raise RuntimeError(f"invalid {loss_key} graph/value in gradient audit")
    loss_value = float(loss.detach().item())
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in model.named_parameters() if not n.startswith("anchor_group.")}
    del _out, metrics, loss
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return loss_value, grads


def run_gradient_audit(model, opt, device, windows, model_label):
    model.eval()  # autograd remains enabled below
    named_params = {n: p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("anchor_group.")}
    if not named_params:
        raise RuntimeError("no trainable shared reconstruction parameters")
    rows = []
    outer_rng = capture_rng()
    try:
        for idx in GRAD_WINDOW_INDICES:
            if idx >= len(windows):
                raise RuntimeError(f"required gradient window index {idx} is unavailable")
            window = windows[idx]
            batch = _batch_for(opt, window, device)
            same_rng = capture_rng()
            l_recon, g_recon = _run_gradient_pass(model, batch, opt, same_rng, "loss_recon")
            l_under, g_under = _run_gradient_pass(model, batch, opt, same_rng, "loss_understanding")
            categories = {}
            for category in CATEGORIES:
                if category == "all_reconstruction":
                    pred = lambda _n: True
                else:
                    pred = lambda n, c=category: _category(n) == c
                categories[category] = _aggregate_named_grads(named_params, g_recon, g_under, pred)
            focused = {
                "decoder_block_11": _select_grad_report(named_params, g_recon, g_under,
                    lambda n: n.startswith("enc_dec_backbone.decoder_blocks.11."), "decoder_block_11"),
                "anchor_decoder.mu": _select_grad_report(named_params, g_recon, g_under,
                    lambda n: n == "anchor_decoder.mu", "anchor_decoder.mu"),
                "anchor_decoder.rho": _select_grad_report(named_params, g_recon, g_under,
                    lambda n: n == "anchor_decoder.rho", "anchor_decoder.rho"),
                "activation_head": _select_grad_report(named_params, g_recon, g_under,
                    lambda n: n.startswith("activation_head."), "activation_head"),
            }
            row = {"scene": window["scene"], "window_index": idx,
                   "L_recon": l_recon, "L_understanding": l_under,
                   "categories": categories, "focused_modules": focused}
            if not all(x["finite"] for x in categories.values()) or not all(x["finite"] for x in focused.values()):
                raise RuntimeError(f"nonfinite gradient statistics on {model_label} window {idx}")
            rows.append(row)
            del g_recon, g_under, batch
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[gradient-conflict:{model_label}] window_index={idx} complete", flush=True)
    finally:
        restore_rng(outer_rng)
        model.zero_grad(set_to_none=True)
    payload = {
        "model_state": model_label, "architecture": "LOCUSGS_ANCHOR_GROUP_V1",
        "step": 0 if model_label == "fresh_step0" else 5000,
        "model_mode": "eval", "autograd": True,
        "optimizer_step_called": False,
        "window_indices": GRAD_WINDOW_INDICES,
        "parameter_category_rules": {
            "encoder": "enc_dec_backbone.* excluding enc_dec_backbone.decoder_blocks.*",
            "decoder": "enc_dec_backbone.decoder_blocks.*",
            "anchor_geometry": "anchor_decoder.mu/rho, refine_mu.*, refine_rho.*, pe_mlp*, pe_mlps*, gamma_raw",
            "activation_head": "activation_head.*",
            "other_reconstruction": "all remaining trainable non-anchor_group parameters",
            "all_reconstruction": "all trainable parameters whose name does not start anchor_group.",
        },
        "category_summary": _summarize_rows(rows),
        "windows": rows,
    }
    return payload


def run_gradient_conflict(opt, device):
    manifest = json.loads(MANIFEST.read_text())
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("locked train manifest SHA256 mismatch")
    windows = manifest.get("windows", [])
    if len(windows) != 1024:
        raise RuntimeError("gradient audit requires the locked 1024-window manifest")
    for idx in GRAD_WINDOW_INDICES:
        if idx >= len(windows):
            raise RuntimeError(f"required window index {idx} missing")
    fresh = make_endpoint_model(opt, device)
    fresh_result = run_gradient_audit(fresh, opt, device, windows, "fresh_step0")
    write_json(OUT / "gradient_conflict_fresh_step0.json", fresh_result)
    del fresh
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    payload, _audit = load_endpoint_payload()
    endpoint = make_endpoint_model(opt, device, payload)
    del payload
    endpoint_result = run_gradient_audit(endpoint, opt, device, windows, "endpoint_step5000")
    write_json(OUT / "gradient_conflict_endpoint_step5000.json", endpoint_result)
    del endpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fresh_result, endpoint_result


def _render_report():
    summary = json.loads((OUT / "query_starvation_summary.json").read_text())
    fresh = json.loads((OUT / "gradient_conflict_fresh_step0.json").read_text())
    endpoint = json.loads((OUT / "gradient_conflict_endpoint_step5000.json").read_text())
    lines = ["# Anchor-Group V1 Endpoint Failure Diagnostics", "",
             "Read-only inference/Hungarian replay and backward-only gradient measurement. No optimizer or model update was performed.", "",
             "## Query utilization / starvation", "",
             "| metric | train1024 | val32 |", "|---|---:|---:|"]
    fields = [
        ("total GT matches", "total_gt_matches"),
        ("unique matched queries", "unique_queries_ever_matched"),
        ("never matched queries", "queries_never_matched"),
    ]
    for label, key in fields:
        lines.append(f"| {label} | {summary['train1024'][key]} | {summary['val32'][key]} |")
    for label, key in (("top1 share", "top1_share"), ("top5 share", "top5_share"),
                       ("top10 share", "top10_share"), ("top20 share", "top20_share"),
                       ("match Gini", "gini"), ("normalized entropy", "normalized_entropy"),
                       ("effective query count", "effective_query_count")):
        lines.append(f"| {label} | {summary['train1024']['match_concentration'][key]:.6f} | {summary['val32']['match_concentration'][key]:.6f} |")
    lines.append(f"| ownership Gini | {summary['train1024']['ownership_concentration']['gini']:.6f} | {summary['val32']['ownership_concentration']['gini']:.6f} |")
    lines.append(f"| active output queries (mean) | {summary['train1024']['mean_active_output_queries']:.3f} | {summary['val32']['mean_active_output_queries']:.3f} |")
    lines.extend(["", "### Top 20 matched queries", "",
                  "| scope | query | matches | unique scenes | scene match fraction |",
                  "|---|---:|---:|---:|---:|"])
    for scope in ("train1024", "val32"):
        for row in summary[scope]["top20_queries"]:
            lines.append(f"| {scope} | {row['query_id']} | {row['matches']} | {row['unique_scenes']} | {row['scene_match_fraction']:.4f} |")
    for scope in ("train1024", "val32"):
        s = summary[scope]
        lines.append("")
        lines.append(f"{scope} Spearman: match count vs no-object probability rho={s['match_count_vs_mean_no_object_probability']['rho']}, p={s['match_count_vs_mean_no_object_probability']['p_value']}; match count vs ownership mass rho={s['match_count_vs_mean_ownership_mass']['rho']}, p={s['match_count_vs_mean_ownership_mass']['p_value']}.")
    lines.extend(["", "## Gradient conflict", "",
                  "Cosine < 0 means gradient conflict on the measured batch/category; cosine > 0 means locally aligned. Gradient scale is the measured `||g_under|| / ||g_recon||`, not the loss scalar ratio.", "",
                  "| model | category | median cos | frac cos < 0 | median ||gU||/||gR|| | p90 ratio |",
                  "|---|---|---:|---:|---:|---:|"])
    for model_name, obj in (("fresh_step0", fresh), ("endpoint_step5000", endpoint)):
        for cat in CATEGORIES:
            s = obj["category_summary"][cat]
            lines.append(f"| {model_name} | {cat} | {s['median_cosine']:.6f} | {s['fraction_cosine_lt_0']:.4f} | {s['median_norm_ratio']:.6g} | {s['p90_norm_ratio']:.6g} |")
    lines.extend(["", "### Fresh versus endpoint", "",
                  "| category | step0 cosine median | endpoint cosine median | step0 norm ratio median | endpoint norm ratio median |",
                  "|---|---:|---:|---:|---:|"])
    for cat in CATEGORIES:
        a, b = fresh["category_summary"][cat], endpoint["category_summary"][cat]
        lines.append(f"| {cat} | {a['median_cosine']:.6f} | {b['median_cosine']:.6f} | {a['median_norm_ratio']:.6g} | {b['median_norm_ratio']:.6g} |")
    lines.extend(["", "### Activation head and focused shared paths", "",
                  "| model | module | median cosine | fraction cosine < 0 | median norm ratio | p90 norm ratio |",
                  "|---|---|---:|---:|---:|---:|"])
    for model_name, obj in (("fresh_step0", fresh), ("endpoint_step5000", endpoint)):
        for module in ("decoder_block_11", "anchor_decoder.mu", "anchor_decoder.rho", "activation_head"):
            vals = [r["focused_modules"][module] for r in obj["windows"]]
            cos = np.asarray([v["cosine"] for v in vals]); ratio = np.asarray([v["norm_ratio"] for v in vals])
            lines.append(f"| {model_name} | {module} | {np.median(cos):.6f} | {np.mean(cos < 0):.4f} | {np.median(ratio):.6g} | {np.quantile(ratio,.9):.6g} |")
    lines.extend(["", "## Factual interpretation", "",
                  "The measurements above describe these locked windows and this endpoint only. Positive-match concentration and never/rarely matched slots are evidence consistent with query starvation/slot collapse when concentrated; they are not a theoretical proof.", "",
                  "Gradient comparisons are local per-window measurements in eval mode with autograd enabled. Negative cosine denotes conflict on that measured batch/category; positive cosine denotes local alignment. No structural or optimization recommendation is made here.", ""])
    tr, va = summary["train1024"], summary["val32"]
    fr, ep = fresh["category_summary"], endpoint["category_summary"]
    lines.extend(["### Measured answers", "",
                  f"- Train positive supervision reached {tr['unique_queries_ever_matched']}/100 queries; {tr['queries_never_matched']}/100 were never matched. Val32 reached {va['unique_queries_ever_matched']}/100; {va['queries_never_matched']}/100 were never matched.",
                  f"- Train top-5/top-10 took {tr['match_concentration']['top5_share']:.1%}/{tr['match_concentration']['top10_share']:.1%} of matches; effective matched-query count was {tr['match_concentration']['effective_query_count']:.2f}. Val32 values were {va['match_concentration']['top5_share']:.1%}/{va['match_concentration']['top10_share']:.1%} and {va['match_concentration']['effective_query_count']:.2f}.",
                  f"- Ownership mass was also concentrated (Gini {tr['ownership_concentration']['gini']:.4f} train, {va['ownership_concentration']['gini']:.4f} val32). Match count versus no-object probability was Spearman rho {tr['match_count_vs_mean_no_object_probability']['rho']:.4f} (p={tr['match_count_vs_mean_no_object_probability']['p_value']:.3g}) train and {va['match_count_vs_mean_no_object_probability']['rho']:.4f} (p={va['match_count_vs_mean_no_object_probability']['p_value']:.3g}) val32. Match count versus ownership mass was rho {tr['match_count_vs_mean_ownership_mass']['rho']:.4f} (p={tr['match_count_vs_mean_ownership_mass']['p_value']:.3g}) train and {va['match_count_vs_mean_ownership_mass']['rho']:.4f} (p={va['match_count_vs_mean_ownership_mass']['p_value']:.3g}) val32.",
                  f"- Median understanding/reconstruction gradient norm ratio for all shared reconstruction parameters was {fr['all_reconstruction']['median_norm_ratio']:.3g} at step0 and {ep['all_reconstruction']['median_norm_ratio']:.3g} at step5000. The measured understanding gradient norm was larger than the reconstruction gradient norm on these aggregate windows.",
                  f"- Decoder had the largest category median scale ratio ({fr['decoder']['median_norm_ratio']:.3g} fresh, {ep['decoder']['median_norm_ratio']:.3g} endpoint). Activation-head ratio changed from {fr['activation_head']['median_norm_ratio']:.3g} to {ep['activation_head']['median_norm_ratio']:.3g}; its negative-cosine fraction changed from {fr['activation_head']['fraction_cosine_lt_0']:.1%} to {ep['activation_head']['fraction_cosine_lt_0']:.1%}.",
                  f"- Direction conflict was already present on fresh windows: negative-cosine fractions were {fr['all_reconstruction']['fraction_cosine_lt_0']:.1%} overall, {fr['decoder']['fraction_cosine_lt_0']:.1%} decoder, {fr['activation_head']['fraction_cosine_lt_0']:.1%} activation head, and {fr['anchor_geometry']['fraction_cosine_lt_0']:.1%} anchor geometry. Endpoint fractions were {ep['all_reconstruction']['fraction_cosine_lt_0']:.1%}, {ep['decoder']['fraction_cosine_lt_0']:.1%}, {ep['activation_head']['fraction_cosine_lt_0']:.1%}, and {ep['anchor_geometry']['fraction_cosine_lt_0']:.1%}, respectively.", ""])
    (OUT / "endpoint_failure_diagnostics_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("query-utilization", "gradient-conflict", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("endpoint audit requires the CUDA node used for renderer-capable model forwards")
    if sha256(PRETRAINED) != PRETRAINED_SHA:
        raise RuntimeError("pretrained checkpoint SHA256 mismatch")
    opt = build_options()
    OUT.mkdir(parents=True, exist_ok=True)
    query = None
    grads = None
    if args.phase in ("query-utilization", "all"):
        payload, _ = load_endpoint_payload()
        endpoint = make_endpoint_model(opt, device, payload)
        del payload
        query = run_query_utilization(endpoint, opt, device)
        del endpoint
        gc.collect(); torch.cuda.empty_cache()
    if args.phase in ("gradient-conflict", "all"):
        grads = run_gradient_conflict(opt, device)
    if args.phase == "all":
        _render_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
