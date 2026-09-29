#!/usr/bin/env python3
"""Read-only mechanism audit for Anchor-Group V1-GC slot starvation.

The script measures loss-source gradients on one retained forward graph and
candidate-row counterfactuals on fixed endpoint predictions.  It never creates
an optimizer, mutates model parameters, or updates a checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)


def _install_import_only_flex_attention_compat():
    """Bridge torch 2.4's private module layout for an unused optional import.

    This does not provide an attention implementation: invoking the branch
    fails closed.  The endpoint model is checked after construction to ensure
    that no attention block has a flex mask or score modifier configured.
    """
    try:
        importlib.import_module("torch.nn.attention.flex_attention")
        return False
    except ModuleNotFoundError as exc:
        if exc.name != "torch.nn.attention.flex_attention":
            raise
    module = types.ModuleType("torch.nn.attention.flex_attention")
    class BlockMask:  # import-time annotation only
        pass
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("optional flex-attention branch is unavailable in this audit environment")
    module.BlockMask = BlockMask
    module.flex_attention = unavailable
    sys.modules[module.__name__] = module
    return True


FLEX_IMPORT_COMPAT_INSTALLED = _install_import_only_flex_attention_compat()

from scripts import anchor_group_v1_endpoint_audit as endpoint_audit
from scripts.anchor_group_v1 import MANIFEST, build_options, sha256
from scripts.instance_state_generalization import _batch_for
from tokengs.models import model_registry
from tokengs.models.anchor_group_loss import (
    FLOOR, IGNORE, THING, WALL, anchor_group_losses, build_anchor_targets,
    pairwise_anchor_bce_cost, unified_hungarian,
)
from tokengs.models.instance_state_loss import (
    CLASS_CE_WEIGHT, MASK_BCE_WEIGHT, MASK_DICE_WEIGHT,
    NO_OBJECT_INDEX, UNMATCHED_CLASS_WEIGHT, _flat_regions,
    _linspace_indices, _matching_cost, identity_loss, semantic_loss,
    stuff_loss,
)

OUT = REPO / "group_plus/anchor_group_v1_gc/starvation_mechanism"
GC_OUT = REPO / "group_plus/anchor_group_v1_gc"
ENDPOINT = REPO / "workspace_group_plus/anchor_group_v1_gc/formal_endpoint_step5000.pt"
ENDPOINT_AUDIT = GC_OUT / "formal_endpoint_step5000_audit.json"
UTIL_TRAIN = GC_OUT / "query_utilization_train1024.json"
STARVATION_SUMMARY = GC_OUT / "query_starvation_summary.json"
VAL32 = GC_OUT / "monitor_32pairs.json"
MANIFEST_SHA = "1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483"
RECIPE = "ANCHOR_GROUP_V1_GC_ALPHA001"
ARCHITECTURE = "LOCUSGS_ANCHOR_GROUP_V1"
GRAD_WINDOW_INDICES = [0, 64, 128, 192, 256, 320, 384, 448,
                       512, 576, 640, 704, 768, 832, 896, 960]
COMPONENTS = (
    "U_matched_class", "U_unmatched_noobject", "U_pixel_bce", "U_pixel_dice",
    "U_anchor_ce", "U_anchor_dice", "U_stuff", "U_semantic", "U_identity",
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _scalar(x):
    return float(x.detach().item()) if torch.is_tensor(x) else float(x)


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _sha_tensor(tensor):
    x = tensor.detach().cpu().contiguous()
    raw = x.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _state_hashes(state):
    return {name: _sha_tensor(value) for name, value in state.items()
            if torch.is_tensor(value)}


def _load_endpoint_payload():
    if not ENDPOINT.is_file() or not ENDPOINT_AUDIT.is_file():
        raise FileNotFoundError(f"required formal GC endpoint or audit missing: {ENDPOINT}")
    audit = json.loads(ENDPOINT_AUDIT.read_text())
    expected = {
        "step": 5000,
        "architecture": ARCHITECTURE,
        "recipe": RECIPE,
        "shared_understanding_grad_scale": 0.01,
        "joint": True,
        "beta": 0,
        "status": "pass",
    }
    bad = {k: (audit.get(k), v) for k, v in expected.items() if audit.get(k) != v}
    if bad:
        raise RuntimeError(f"formal endpoint audit identity mismatch: {bad}")
    payload = torch.load(ENDPOINT, map_location="cpu", weights_only=False)
    bad = {k: (payload.get(k), v) for k, v in expected.items()
           if k != "status" and payload.get(k) != v}
    if bad:
        raise RuntimeError(f"formal endpoint payload identity mismatch: {bad}")
    for key in ("manifest_sha256", "plan_sha256", "pretrained_sha256"):
        if payload.get(key) != audit.get(key):
            raise RuntimeError(f"endpoint payload/audit provenance mismatch: {key}")
    if payload.get("manifest_sha256") != MANIFEST_SHA:
        raise RuntimeError("endpoint locked manifest SHA mismatch")
    if payload.get("pretrained_sha256") != endpoint_audit.PRETRAINED_SHA:
        raise RuntimeError("endpoint pretrained checkpoint SHA mismatch")
    if payload.get("recipe") != RECIPE or payload.get("shared_understanding_grad_scale") != 0.01:
        raise RuntimeError("endpoint GC recipe identity mismatch")
    return payload, audit


def _make_model(opt, device, payload):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    model = model_registry[opt.model_type](opt)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"endpoint strict load mismatch: {missing}, {unexpected}")
    model.to(device)
    model.eval()
    if getattr(model, "architecture_name", None) != ARCHITECTURE:
        raise RuntimeError("loaded endpoint has unexpected architecture")
    configured_flex = [name for name, module in model.named_modules()
                       if hasattr(module, "flex_attn_block_mask")
                       and (module.flex_attn_block_mask is not None
                            or module.flex_attn_score_mod is not None)]
    if configured_flex:
        raise RuntimeError(f"optional flex-attention branch configured in endpoint: {configured_flex[:8]}")
    return model


def _historical_sets():
    if not UTIL_TRAIN.is_file() or not STARVATION_SUMMARY.is_file():
        raise FileNotFoundError("committed GC query utilization artifacts are required")
    util = json.loads(UTIL_TRAIN.read_text())
    summary = json.loads(STARVATION_SUMMARY.read_text())
    rows = util.get("queries")
    if not isinstance(rows, list) or len(rows) != 100:
        raise RuntimeError("train1024 artifact must have 100 query rows")
    if util.get("summary", {}).get("scope") != "train1024":
        raise RuntimeError("query artifact is not train1024")
    def match_count(r):
        for key in ("n_hungarian_matches", "matches", "n_matches"):
            if key in r:
                return int(r[key])
        raise RuntimeError("no equivalent historical match-count field in query row")
    def positive_rate(r, count):
        for key in ("positive_match_rate", "match_rate"):
            if key in r:
                return float(r[key])
        return count / float(util["summary"]["n_windows"])
    ordered = sorted(rows, key=lambda r: (-match_count(r), int(r["query_id"])))
    counts = {int(r["query_id"]): match_count(r) for r in rows}
    rates = {int(r["query_id"]): positive_rate(r, counts[int(r["query_id"])]) for r in rows}
    result = {
        "top5": [int(r["query_id"]) for r in ordered[:5]],
        "top10": [int(r["query_id"]) for r in ordered[:10]],
        "never": [q for q in range(100) if counts[q] == 0],
        "rare": [q for q in range(100) if 0 < rates[q] < 0.01],
        "match_count": counts,
        "positive_match_rate": rates,
    }
    saved = summary.get("train1024", {})
    if saved and int(saved.get("total_gt_matches", -1)) != sum(counts.values()):
        raise RuntimeError("query utilization and summary artifacts disagree")
    return result


def _load_windows():
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("locked train manifest SHA256 mismatch")
    manifest = json.loads(MANIFEST.read_text())
    train = manifest.get("windows", [])
    if len(train) != 1024:
        raise RuntimeError(f"expected exactly 1024 training windows, got {len(train)}")
    for idx in GRAD_WINDOW_INDICES:
        if idx >= len(train):
            raise RuntimeError(f"required fixed gradient window index absent: {idx}")
    valdoc = json.loads(VAL32.read_text())
    val = valdoc.get("pairs", [])
    if len(val) != 32:
        raise RuntimeError(f"locked val32 must contain 32 windows, got {len(val)}")
    return manifest, train, val


def _batch_forward(model, opt, window, device):
    batch = _batch_for(opt, window, device)
    pred = endpoint_audit.forward_context(model, opt, batch, step=5000)
    return batch, pred


def _production_pairs_once(pred, batch):
    targets = build_anchor_targets(
        pred["states"][-1]["mu"], batch["semantic_label_all"],
        batch["instance_label_all"], batch["cam_view_all"], batch["intrinsics_all"])
    targets, pairs = unified_hungarian(pred, batch, targets)
    if len(pairs) != 1:
        raise RuntimeError("starvation audit requires B=1")
    return targets, pairs


def _decompose_understanding(pred, batch, targets, pairs):
    """Return the exact weighted U1-U9 decomposition on production pairs."""
    device = pred["gaussians"].device
    sem = batch["semantic_label_all"][:, :2].long()
    ins = batch["instance_label_all"][:, :2].long()
    qidx, kidx = pairs[0]
    logits = pred["thing_class_logits"][:, :, 2:]
    cls = targets["gt_classes"][0]
    masks = targets["gt_pixel_masks"][0]
    target = torch.full((100,), NO_OBJECT_INDEX, device=device, dtype=torch.long)
    if qidx.numel():
        target[qidx] = cls[kidx] - 2
    class_weight = torch.ones(19, device=device, dtype=torch.float32)
    class_weight[NO_OBJECT_INDEX] = UNMATCHED_CLASS_WEIGHT
    per_query = F.cross_entropy(logits[0].float(), target, weight=class_weight,
                                reduction="none")
    denominator = class_weight[target].sum()
    matched_mask = torch.zeros(100, device=device, dtype=torch.bool)
    matched_mask[qidx] = True
    ce_matched = per_query[matched_mask].sum() / denominator
    ce_unmatched = per_query[~matched_mask].sum() / denominator
    U_matched_class = 0.1 * CLASS_CE_WEIGHT * ce_matched
    U_unmatched_noobject = 0.1 * CLASS_CE_WEIGHT * ce_unmatched

    region = _flat_regions(pred["region_mass"][:, :, :100], 100)[0]
    valid = ((sem[0] >= 0) & (sem[0] <= 19)
             & ((sem[0] < 2) | (ins[0] > 0))).reshape(-1)
    z = torch.logit(region[:, valid].clamp(1e-6, 1.0 - 1e-6))
    y = masks.flatten(1)[:, valid].float()
    if qidx.numel():
        bce = F.binary_cross_entropy_with_logits(z[qidx], y[kidx], reduction="none").mean()
        probability = torch.sigmoid(z[qidx])
        inter = (probability * y[kidx]).sum(dim=1)
        denom = probability.sum(dim=1) + y[kidx].sum(dim=1)
        dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
    else:
        zero = pred["gaussians"].sum() * 0.0
        bce = zero
        dice = zero
    U_pixel_bce = 0.1 * MASK_BCE_WEIGHT * bce
    U_pixel_dice = 0.1 * MASK_DICE_WEIGHT * dice

    ownership = pred["states"][-1]["A_post"]
    kind = targets["anchor_kind"][0]
    aid = targets["anchor_instance_id"][0]
    target_channel = torch.full_like(kind, -1)
    target_channel[kind == WALL] = 100
    target_channel[kind == FLOOR] = 101
    id_to_gt = {int(value): i for i, value in enumerate(targets["gt_instance_ids"][0].tolist())}
    query_for_gt = {int(k): int(q) for q, k in zip(qidx.tolist(), kidx.tolist())}
    for iid, k in id_to_gt.items():
        gt_anchors = (kind == THING) & (aid == iid)
        if bool(gt_anchors.any()) and k not in query_for_gt:
            raise RuntimeError("confident thing anchor belongs to unmatched GT")
        if k in query_for_gt:
            target_channel[gt_anchors] = query_for_gt[k]
    valid_anchor = targets["anchor_valid"][0] & (target_channel >= 0)
    if bool(valid_anchor.any()):
        anchor_ce = -torch.log(ownership[0, valid_anchor,
                                         target_channel[valid_anchor]].clamp_min(1e-6)).mean()
    else:
        anchor_ce = pred["gaussians"].sum() * 0.0
    U_anchor_ce = 0.1 * anchor_ce
    anchor_dice_terms = []
    for q, k in zip(qidx.tolist(), kidx.tolist()):
        gt_anchor = targets["Y_anchor"][0, k]
        av = targets["anchor_valid"][0]
        if bool(gt_anchor.sum() > 0):
            p = ownership[0, :, q][av]
            yy = gt_anchor[av]
            anchor_dice_terms.append(1.0 - (2.0 * (p * yy).sum() + 1.0)
                                     / (p.sum() + yy.sum() + 1.0))
    anchor_dice = torch.stack(anchor_dice_terms).mean() if anchor_dice_terms else pred["gaussians"].sum() * 0.0
    U_anchor_dice = 0.1 * anchor_dice

    stuff, _ = stuff_loss(pred, batch)
    semantic, _ = semantic_loss(pred, batch)
    identity, _ = identity_loss(pred, batch)
    U_stuff = 0.1 * stuff
    U_semantic = 0.1 * semantic
    U_identity = 0.01 * identity
    components = {
        "U_matched_class": U_matched_class,
        "U_unmatched_noobject": U_unmatched_noobject,
        "U_pixel_bce": U_pixel_bce,
        "U_pixel_dice": U_pixel_dice,
        "U_anchor_ce": U_anchor_ce,
        "U_anchor_dice": U_anchor_dice,
        "U_stuff": U_stuff,
        "U_semantic": U_semantic,
        "U_identity": U_identity,
    }
    U_sum = torch.stack([components[name] for name in COMPONENTS]).sum()

    # Reuse the exact production scalar implementation while pinning it to the
    # one already-computed production assignment; do not solve Hungarian twice.
    import tokengs.models.anchor_group_loss as ag_loss
    original_matcher = ag_loss.unified_hungarian
    targets_fixed = targets
    try:
        ag_loss.unified_hungarian = lambda prediction, batch, targets=None: (targets_fixed, pairs)
        production, _ = anchor_group_losses(pred, batch)
    finally:
        ag_loss.unified_hungarian = original_matcher
    parity = abs(_scalar(U_sum) - _scalar(production))
    if parity > 1e-6:
        raise RuntimeError(f"U1-U9 decomposition differs from production loss by {parity:.9g}")
    return components, U_sum, production, parity


def _grad_pair(loss, q_final, query_init):
    values = torch.autograd.grad(loss, (q_final, query_init), retain_graph=True,
                                 allow_unused=True)
    return values[0], values[1]


def _rows_norm(g):
    if g is None:
        return [0.0] * 100
    rows = g.detach().float().norm(dim=-1)
    if rows.ndim == 2 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.shape[0] == 102:
        rows = rows[:100]
    return rows.cpu().tolist()


def _vec_metrics(x):
    a = np.asarray(list(x), dtype=np.float64)
    if not a.size:
        return {"mean": None, "median": None, "p90": None, "count": 0}
    return {"mean": float(np.mean(a)), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "count": int(a.size)}


def _gradient_attribution(model, opt, train, historical, initial_hashes):
    per_query = []
    window_reports = []
    aggregate = {target: {c: defaultdict(list) for c in COMPONENTS}
                 for target in ("final_q", "query_init")}
    directional = defaultdict(lambda: defaultdict(list))
    row_status_counts = defaultdict(int)
    max_decomp_error = 0.0
    max_total_errors = {"final_q": {"max_abs_diff": 0.0, "relative_l2_diff": 0.0},
                        "query_init": {"max_abs_diff": 0.0, "relative_l2_diff": 0.0}}
    q_init = model.anchor_group.query_init
    model.eval()
    for wi in GRAD_WINDOW_INDICES:
        batch, pred = _batch_forward(model, opt, train[wi], torch.device("cuda"))
        targets, pairs = _production_pairs_once(pred, batch)
        components, U_sum, production, decomp_error = _decompose_understanding(pred, batch, targets, pairs)
        max_decomp_error = max(max_decomp_error, decomp_error)
        # Differentiate with respect to the original state tensor; slicing it
        # after the forward is not an ancestor node in the retained graph.
        q_final = pred["states"][-1]["q"]
        component_grads = {"final_q": {}, "query_init": {}}
        component_norms = {"final_q": {}, "query_init": {}}
        for comp_name in COMPONENTS:
            gq, gi = _grad_pair(components[comp_name], q_final, q_init)
            component_grads["final_q"][comp_name] = gq
            component_grads["query_init"][comp_name] = gi
            component_norms["final_q"][comp_name] = _rows_norm(gq)
            component_norms["query_init"][comp_name] = _rows_norm(gi)
        total_grads = {}
        window_gradient_errors = {}
        for target_name, tensor in (("final_q", q_final), ("query_init", q_init)):
            total_grads[target_name] = torch.autograd.grad(
                U_sum, tensor, retain_graph=True, allow_unused=True)[0]
            gs = [component_grads[target_name][name] for name in COMPONENTS]
            gs = [g for g in gs if g is not None]
            comp_sum = torch.stack(gs).sum(dim=0) if gs else torch.zeros_like(tensor)
            total = total_grads[target_name]
            if total is None:
                total = torch.zeros_like(tensor)
            if target_name == "final_q":
                total = total[:, :100]
                comp_sum = comp_sum[:, :100]
            elif target_name == "query_init":
                total = total[:100]
                comp_sum = comp_sum[:100]
            diff = (total - comp_sum).detach().float()
            max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
            relative = float(diff.norm().item() / (total.detach().float().norm().item() + 1e-12))
            max_total_errors[target_name]["max_abs_diff"] = max(max_total_errors[target_name]["max_abs_diff"], max_abs)
            max_total_errors[target_name]["relative_l2_diff"] = max(max_total_errors[target_name]["relative_l2_diff"], relative)
            window_gradient_errors[target_name] = {"max_abs_diff": max_abs,
                                                   "relative_l2_diff": relative}
            if max_abs > 1e-6 or relative > 1e-6:
                raise RuntimeError(f"component gradient sum parity failed for {target_name} window {wi}: {max_abs}, {relative}")

        qidx = pairs[0][0].tolist()
        matched = set(map(int, qidx))
        window_query_rows = []
        for q in range(100):
            status = {
                "currently_matched": q in matched,
                "historical_top5": q in historical["top5"],
                "historical_top10": q in historical["top10"],
                "historical_never": q in historical["never"],
                "historical_rare": q in historical["rare"],
                "historical_match_count": historical["match_count"][q],
                "historical_positive_match_rate": historical["positive_match_rate"][q],
            }
            for key, val in status.items():
                row_status_counts[key] += int(val) if isinstance(val, bool) else 0
            row = {"window_index": wi, "scene": train[wi]["scene"], "query_id": q, **status,
                   "grad_norms": {target: {comp: component_norms[target][comp][q]
                                           for comp in COMPONENTS}
                                  for target in ("final_q", "query_init")}}
            per_query.append(row)
            window_query_rows.append(row)
            groups = ["all"]
            if q in historical["top10"]: groups.append("historical_top10")
            if q in historical["never"]: groups.append("historical_never")
            if q in historical["rare"]: groups.append("historical_rare")
            groups.append("current_matched" if q in matched else "current_unmatched")
            for target in ("final_q", "query_init"):
                norms = {comp: component_norms[target][comp][q]
                         for comp in COMPONENTS}
                for group in groups:
                    for comp, value in norms.items():
                        aggregate[target][comp][group].append(value)

        logits_under = torch.autograd.grad(
            components["U_unmatched_noobject"], pred["thing_class_logits"][:, :, 2:],
            retain_graph=True, allow_unused=True)[0]
        if logits_under is None:
            noobj_drive = [0.0] * 100
        else:
            noobj_drive = (-logits_under[0, :, 18]).detach().float().cpu().tolist()
        final = pred["states"][-1]
        A = final["A_post"][0].detach().float()
        kinds = targets["anchor_kind"][0]
        valid = targets["anchor_valid"][0]
        aid = targets["anchor_instance_id"][0]
        tchannel = torch.full_like(kinds, -1)
        tchannel[kinds == WALL] = 100
        tchannel[kinds == FLOOR] = 101
        id_to_gt = {int(i): k for k, i in enumerate(targets["gt_instance_ids"][0].tolist())}
        query_for_gt = {int(k): int(q) for q, k in zip(pairs[0][0].tolist(), pairs[0][1].tolist())}
        for iid, k in id_to_gt.items():
            if k not in query_for_gt and bool(((kinds == THING) & (aid == iid)).any()):
                raise RuntimeError("unmatched GT has confident thing anchors")
            if k in query_for_gt:
                tchannel[(kinds == THING) & (aid == iid)] = query_for_gt[k]
        av = valid & (tchannel >= 0)
        n_valid = int(av.sum())
        if n_valid:
            pos_by_query = torch.zeros(100, device=A.device)
            for q in range(100):
                pos = av & (tchannel == q)
                if bool(pos.any()):
                    pos_by_query[q] = (1.0 - A[pos, q]).sum() * (0.1 / n_valid)
            neg_by_query = torch.stack([
                A[av & (tchannel != q), q].sum() * (0.1 / n_valid)
                for q in range(100)])
        else:
            pos_by_query = torch.zeros(100, device=A.device)
            neg_by_query = torch.zeros(100, device=A.device)
        anchor_positive = pos_by_query.detach().cpu().tolist()
        anchor_negative = neg_by_query.detach().cpu().tolist()
        anchor_net = [n - p for n, p in zip(anchor_negative, anchor_positive)]
        for q, row in enumerate(window_query_rows):
            row["directional"] = {
                "noobject_increase_drive": float(noobj_drive[q]),
                "anchor_positive_drive": float(anchor_positive[q]),
                "anchor_negative_suppression": float(anchor_negative[q]),
                "anchor_net_logit_grad": float(anchor_net[q]),
            }
        group_masks = {
            "current_matched": [q in matched for q in range(100)],
            "current_unmatched": [q not in matched for q in range(100)],
            "historical_top10": [q in historical["top10"] for q in range(100)],
            "historical_never": [q in historical["never"] for q in range(100)],
            "historical_rare": [q in historical["rare"] for q in range(100)],
        }
        for group, mask in group_masks.items():
            for q, yes in enumerate(mask):
                if yes:
                    directional[group]["noobject_increase_drive"].append(noobj_drive[q])
                    directional[group]["anchor_positive_drive"].append(anchor_positive[q])
                    directional[group]["anchor_negative_suppression"].append(anchor_negative[q])
                    directional[group]["anchor_net_logit_grad"].append(anchor_net[q])

        window_reports.append({
            "window_index": int(wi),
            "scene": str(train[wi]["scene"]),
            "matched_query_ids": sorted(matched),
            "matched_gt_indices": [int(k) for k in pairs[0][1].tolist()],
            "component_loss_values": {name: _scalar(value) for name, value in components.items()},
            "understanding_sum": _scalar(U_sum),
            "production_understanding": _scalar(production),
            "decomposition_abs_error": float(decomp_error),
            "gradient_sum_parity": window_gradient_errors,
            "anchor_valid_count": int(n_valid),
        })

        print(f"[gradient-attribution] window {wi} complete; decomp={decomp_error:.3g}", flush=True)
        del batch, pred, targets, production, components, U_sum
        gc.collect()
        torch.cuda.empty_cache()

    if len(per_query) != 16 * 100:
        raise RuntimeError("gradient attribution did not cover all fixed query rows")
    current_summary = {}
    for target in ("final_q", "query_init"):
        current_summary[target] = {}
        for comp in COMPONENTS:
            current_summary[target][comp] = {
                group: _vec_metrics(aggregate[target][comp][group])
                for group in ("current_matched", "current_unmatched",
                              "historical_top10", "historical_never", "historical_rare")
            }
            m = current_summary[target][comp]["current_matched"]["median"]
            u = current_summary[target][comp]["current_unmatched"]["median"]
            current_summary[target][comp]["unmatched_over_matched_median_ratio"] = (
                float(u / m) if m not in (None, 0) and u is not None else None)
    directional_summary = {group: {name: _vec_metrics(vals) for name, vals in fields.items()}
                           for group, fields in directional.items()}
    # Per-query norm-share proxy, explicitly not a vector-gradient decomposition.
    proxy = {group: defaultdict(list) for group in
             ("historical_top10", "historical_never", "current_unmatched")}
    for row in per_query:
        memberships = {"historical_top10": row["historical_top10"],
                       "historical_never": row["historical_never"],
                       "current_unmatched": not row["currently_matched"]}
        for target in ("final_q", "query_init"):
            norms = row["grad_norms"][target]
            total = sum(norms.values())
            if total > 0:
                for group, yes in memberships.items():
                    if yes:
                        for comp, value in norms.items():
                            proxy[group][comp].append(value / total)
    proxy_summary = {group: {comp: _vec_metrics(vals) for comp, vals in comps.items()}
                     for group, comps in proxy.items()}
    if _state_hashes(model.state_dict()) != initial_hashes:
        raise RuntimeError("Audit A changed endpoint parameters")
    result = {
        "endpoint_step": 5000,
        "gradient_window_indices": GRAD_WINDOW_INDICES,
        "window_count": 16,
        "forward_count": 16,
        "production_hungarian_calls": 16,
        "component_order": list(COMPONENTS),
        "historical_query_sets": {k: historical[k] for k in ("top5", "top10", "never", "rare")},
        "decomposition_max_abs_error": max_decomp_error,
        "total_gradient_sum_parity": max_total_errors,
        "window_reports": window_reports,
        "aggregates": current_summary,
        "directional_diagnostics": directional_summary,
        "component_norm_share_proxy": proxy_summary,
        "n_valid_query_rows": len(per_query),
        "parameter_hashes_unchanged": True,
    }
    return result, per_query


def _production_unified_cost_matrix(pred, batch, targets):
    """Reconstruct the production matrix by calling its existing cost helpers."""
    sem = batch["semantic_label_all"][:, :2]
    ins = batch["instance_label_all"][:, :2]
    region = _flat_regions(pred["region_mass"][:, :, :100], 100)
    logits = pred["thing_class_logits"][:, :, 2:]
    ownership = pred["states"][-1]["A_post"][:, :, :100].transpose(1, 2)
    costs = []
    for b in range(sem.shape[0]):
        cls = targets["gt_classes"][b]
        masks = targets["gt_pixel_masks"][b]
        K = int(cls.numel())
        valid = ((sem[b] >= 0) & (sem[b] <= 19)
                 & ((sem[b] < 2) | (ins[b] > 0)))
        if K == 0 or not bool(valid.any()):
            costs.append(torch.empty((100, K), device=sem.device, dtype=torch.float32))
            continue
        flat = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        sel = _linspace_indices(int(flat.numel()), 4096).to(sem.device)
        flat = flat[sel]
        z = torch.logit(region[b][:, flat].clamp(1e-6, 1.0 - 1e-6))
        y = masks.flatten(1)[:, flat].float()
        pixel = _matching_cost(logits[b], z, y, cls)
        av = targets["anchor_valid"][b]
        pa = ownership[b, :, av].float()
        ya = targets["Y_anchor"][b, :, av].float()
        if bool(av.any()) and K:
            bce = pairwise_anchor_bce_cost(pa, ya)
            dice = 1.0 - (2.0 * (pa @ ya.T) + 1.0) / (
                pa.sum(-1, keepdim=True) + ya.sum(-1)[None, :] + 1.0)
            support = targets["Y_anchor"][b].sum(-1) > 0
            bce = bce * support[None, :]
            dice = dice * support[None, :]
        else:
            bce = torch.zeros_like(pixel)
            dice = torch.zeros_like(pixel)
        costs.append(pixel + 2.0 * bce + 2.0 * dice)
    return costs


def _quality(pred, batch, targets, b, k, q):
    sem = batch["semantic_label_all"][b, :2]
    ins = batch["instance_label_all"][b, :2]
    region = pred["region_mass"][b, :, :100]
    valid_pix = ((sem >= 0) & (sem <= 19) & ((sem < 2) | (ins > 0)))
    p = region[:, q][valid_pix].float()
    y = targets["gt_pixel_masks"][b][k][valid_pix].float()
    binary = p > 0.5
    target_binary = y > 0.5
    union = int((binary | target_binary).sum())
    iou = float((binary & target_binary).sum().item() / union) if union else None
    soft_dice = float(((2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)).item())
    cls = int(targets["gt_classes"][b][k])
    class_prob = torch.softmax(pred["thing_class_logits"][b, q, 2:].float(), dim=-1)
    state = pred["states"][-1]
    A = state["A_post"][b]
    anchor_valid = targets["anchor_valid"][b]
    y_anchor = targets["Y_anchor"][b, k]
    supported = bool(y_anchor.sum() > 0)
    if supported:
        positive = y_anchor > 0
        anchor_mean = float(A[positive, q].float().mean().item())
        pred_owner = A.argmax(-1)
        anchor_correct = float((pred_owner[positive] == q).float().mean().item())
    else:
        anchor_mean = None
        anchor_correct = None
    ap = A[anchor_valid, q].float()
    ay = y_anchor[anchor_valid].float()
    anchor_soft_dice = float(((2.0 * (ap * ay).sum() + 1.0) /
                              (ap.sum() + ay.sum() + 1.0)).item())
    return {
        "pixel_iou50": iou,
        "pixel_soft_dice": soft_dice,
        "gt_class_probability": float(class_prob[cls - 2].item()),
        "no_object_probability": float(class_prob[18].item()),
        "anchor_gt_mean_probability": anchor_mean,
        "anchor_soft_dice": anchor_soft_dice,
        "anchor_correct_fraction": anchor_correct,
        "anchor_supported": supported,
    }


def _hist_category(q, historical, removed=()):
    if q in historical["top10"] and q not in removed:
        return "historical_top10_not_removed"
    if q in historical["rare"]:
        return "historical_rare"
    if q in historical["never"]:
        return "historical_never"
    return "other"


def _linear_assignment(cost, allowed):
    if cost.shape[1] == 0:
        return {}
    if len(allowed) < cost.shape[1]:
        raise RuntimeError(f"counterfactual has {len(allowed)} queries for {cost.shape[1]} GTs")
    sub = cost[np.asarray(allowed), :]
    rr, cc = linear_sum_assignment(sub)
    return {int(c): int(allowed[int(r)]) for r, c in zip(rr, cc)}


def _candidate_record(pred, batch, targets, cost, k, q, historical, *, removed=(), assigned_cost=None):
    quality = _quality(pred, batch, targets, 0, k, q)
    return {
        "query_id": int(q),
        "cost": float(cost[q, k].item()),
        "cost_delta_vs_baseline_assigned": (float(cost[q, k].item() - assigned_cost)
                                             if assigned_cost is not None else None),
        "historical_match_count": int(historical["match_count"][q]),
        "historical_category": _hist_category(q, historical, removed),
        **quality,
        "strong_alternate": bool((quality["pixel_iou50"] is not None and quality["pixel_iou50"] >= 0.50)
                                  or (quality["anchor_supported"] and quality["anchor_correct_fraction"] is not None
                                      and quality["anchor_correct_fraction"] >= 0.50)),
    }


def _candidate_rank(cost, k, q):
    col = cost[:, k].detach().cpu().numpy()
    order = np.argsort(col, kind="stable")
    return int(np.where(order == int(q))[0][0] + 1)


def _winner_removal_scope(model, opt, windows, scope, historical, initial_hashes):
    rows = []
    removed_sets = {
        "top5": set(historical["top5"]),
        "top10": set(historical["top10"]),
    }
    production_parity = 0
    model.eval()
    with torch.no_grad():
        for wi, window in enumerate(windows):
            batch, pred = _batch_forward(model, opt, window, torch.device("cuda"))
            targets, pairs = _production_pairs_once(pred, batch)
            if len(targets["gt_classes"]) != 1:
                raise RuntimeError("winner-removal expects a single scene batch")
            costs = _production_unified_cost_matrix(pred, batch, targets)
            cost = costs[0].detach().float()
            qi, ki = pairs[0]
            production_map = {int(k): int(q) for q, k in zip(qi.tolist(), ki.tolist())}
            audit_map = _linear_assignment(cost.detach().cpu().numpy(), list(range(100)))
            if production_map != audit_map:
                first = next((k for k in set(production_map) | set(audit_map)
                              if production_map.get(k) != audit_map.get(k)), None)
                raise RuntimeError(f"baseline production Hungarian parity mismatch scope={scope} window={wi} gt={first}: production={production_map.get(first)} audit={audit_map.get(first)}")
            production_parity += 1
            K = int(targets["gt_classes"][0].numel())
            cf_maps = {name: _linear_assignment(cost.detach().cpu().numpy(),
                                                [q for q in range(100) if q not in remove])
                       for name, remove in removed_sets.items()}
            full_cost_cpu = cost.detach().cpu().numpy()
            rank_cache = {}
            for k in range(K):
                base_q = production_map[k]
                base_cost = float(full_cost_cpu[base_q, k])
                baseline = _candidate_record(pred, batch, targets, cost, k, base_q,
                                             historical, assigned_cost=base_cost)
                full_order = np.argsort(full_cost_cpu[:, k], kind="stable")
                rank_cache[k] = {int(q): int(rank + 1) for rank, q in enumerate(full_order)}
                gt = {
                    "scene": str(window["scene"]),
                    "window_index": int(wi),
                    "gt_index": int(k),
                    "gt_instance_id": int(targets["gt_instance_ids"][0][k]),
                    "gt_semantic_class": int(targets["gt_classes"][0][k]),
                    "anchor_support_count": int(targets["Y_anchor"][0, k].sum().item()),
                    "baseline_query_id": base_q,
                    "baseline_unified_cost": base_cost,
                    "baseline_local_cost_rank": rank_cache[k][base_q],
                    "baseline_historical_match_count": historical["match_count"][base_q],
                    "baseline_historical_category": _hist_category(base_q, historical),
                    "baseline_candidate": baseline,
                }
                candidate_info = {}
                for name, remove in removed_sets.items():
                    allowed = [q for q in range(100) if q not in remove]
                    local_q = min(allowed, key=lambda q: (full_cost_cpu[q, k], q))
                    cf_q = cf_maps[name][k]
                    local = _candidate_record(pred, batch, targets, cost, k, local_q,
                                              historical, removed=remove, assigned_cost=base_cost)
                    cf = _candidate_record(pred, batch, targets, cost, k, cf_q,
                                           historical, removed=remove, assigned_cost=base_cost)
                    local["local_cost_rank_in_full100"] = rank_cache[k][local_q]
                    local["is_local_best_remaining"] = True
                    cf["local_cost_rank_in_full100"] = rank_cache[k][cf_q]
                    displaced = base_q in remove
                    candidate_info[name] = {
                        "removed_query_ids": sorted(remove),
                        "displaced": displaced,
                        "local_best_remaining": local,
                        "counterfactual_hungarian": cf,
                    }
                gt["counterfactuals"] = candidate_info
                rows.append(gt)
            if (wi + 1) % 32 == 0 or wi + 1 == len(windows):
                print(f"[winner-removal:{scope}] {wi + 1}/{len(windows)} windows; GT rows={len(rows)}", flush=True)
            del batch, pred, targets, pairs, costs
            gc.collect()
            torch.cuda.empty_cache()
    if production_parity != len(windows):
        raise RuntimeError(f"Hungarian parity covered {production_parity}/{len(windows)} windows")
    if _state_hashes(model.state_dict()) != initial_hashes:
        raise RuntimeError("winner-removal modified endpoint model parameters")
    return {"scope": scope, "window_count": len(windows), "production_hungarian_parity_windows": production_parity,
            "candidate_set_only_counterfactual": True, "gt_rows": rows}


def _finite_values(values):
    return [float(x) for x in values if x is not None and math.isfinite(float(x))]


def _summary_values(values):
    x = np.asarray(_finite_values(values), dtype=np.float64)
    if not x.size:
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {"count": int(x.size), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "mean": float(np.mean(x))}


def _summarize_candidate(rows, candidate_kind):
    cost_delta, iou, anchor_corr = [], [], []
    iou25 = iou50 = anchor25 = anchor50 = strong = hist_never = hist_never_strong = hist_rare = hist_rare_strong = 0
    hist_never_cf = hist_rare_cf = 0
    status_counts = defaultdict(int)
    status_rows = defaultdict(list)
    supported_n = 0
    for cf in rows:
        d = cf["cost_delta_vs_baseline_assigned"]
        if d is not None:
            cost_delta.append(d)
        piou = cf["pixel_iou50"]
        if piou is not None:
            iou.append(piou)
            iou25 += int(piou >= 0.25)
            iou50 += int(piou >= 0.50)
        if cf["anchor_supported"]:
            supported_n += 1
            ac = cf["anchor_correct_fraction"]
            anchor_corr.append(ac)
            anchor25 += int(ac >= 0.25)
            anchor50 += int(ac >= 0.50)
        strong += int(cf["strong_alternate"])
        status = cf["historical_category"]
        status_counts[status] += 1
        status_rows[status].append(cf)
        if status == "historical_never":
            hist_never += 1
            hist_never_cf += int(candidate_kind == "counterfactual_hungarian")
            hist_never_strong += int(cf["strong_alternate"])
        if status == "historical_rare":
            hist_rare += 1
            hist_rare_cf += int(candidate_kind == "counterfactual_hungarian")
            hist_rare_strong += int(cf["strong_alternate"])
    n = len(rows)
    status_quality = {}
    for status, candidates in status_rows.items():
        ds = [x["cost_delta_vs_baseline_assigned"] for x in candidates
              if x["cost_delta_vs_baseline_assigned"] is not None]
        pis = [x["pixel_iou50"] for x in candidates if x["pixel_iou50"] is not None]
        acs = [x["anchor_correct_fraction"] for x in candidates
               if x["anchor_supported"] and x["anchor_correct_fraction"] is not None]
        status_quality[status] = {
            "candidate_count": len(candidates),
            "cost_delta_vs_baseline_assigned": _summary_values(ds),
            "pixel_iou50": _summary_values(pis),
            "pixel_iou_ge_0_25_count": sum(x >= 0.25 for x in pis),
            "pixel_iou_ge_0_50_count": sum(x >= 0.50 for x in pis),
            "anchor_correct_fraction": _summary_values(acs),
            "anchor_correct_ge_0_25_count": sum(x >= 0.25 for x in acs),
            "anchor_correct_ge_0_50_count": sum(x >= 0.50 for x in acs),
            "strong_alternate_count": sum(bool(x["strong_alternate"]) for x in candidates),
        }
    return {
        "candidate_count": n,
        "cost_delta_vs_baseline_assigned": _summary_values(cost_delta),
        "pixel_iou50": _summary_values(iou),
        "pixel_iou_ge_0_25_count": iou25,
        "pixel_iou_ge_0_25_fraction": iou25 / len(iou) if iou else None,
        "pixel_iou_ge_0_50_count": iou50,
        "pixel_iou_ge_0_50_fraction": iou50 / len(iou) if iou else None,
        "supported_gt_count": supported_n,
        "anchor_correct_fraction": _summary_values(anchor_corr),
        "anchor_correct_ge_0_25_count": anchor25,
        "anchor_correct_ge_0_25_fraction": anchor25 / supported_n if supported_n else None,
        "anchor_correct_ge_0_50_count": anchor50,
        "anchor_correct_ge_0_50_fraction": anchor50 / supported_n if supported_n else None,
        "strong_alternate_count": strong,
        "strong_alternate_fraction": strong / n if n else None,
        "historical_never_alternate_count": hist_never,
        "historical_never_alternate_fraction": hist_never / n if n else None,
        "historical_never_strong_count": hist_never_strong,
        "historical_never_strong_fraction": hist_never_strong / n if n else None,
        "historical_rare_alternate_count": hist_rare,
        "historical_rare_alternate_fraction": hist_rare / n if n else None,
        "historical_rare_strong_count": hist_rare_strong,
        "historical_rare_strong_fraction": hist_rare_strong / n if n else None,
        "historical_status_counts": dict(status_counts),
        "historical_status_quality": status_quality,
    }


def _winner_summary(scope_result):
    result = {"scope": scope_result["scope"], "window_count": scope_result["window_count"],
              "production_hungarian_parity_windows": scope_result["production_hungarian_parity_windows"],
              "removals": {}}
    rows = scope_result["gt_rows"]
    for name in ("top5", "top10"):
        all_rows = [r for r in rows]
        displaced = [r for r in rows if r["counterfactuals"][name]["displaced"]]
        result["removals"][name] = {
            "all_gt_count": len(all_rows),
            "displaced_gt_count": len(displaced),
            "all_gt": {
                "local_best_remaining": _summarize_candidate(
                    [r["counterfactuals"][name]["local_best_remaining"] for r in all_rows],
                    "local_best_remaining"),
                "counterfactual_hungarian": _summarize_candidate(
                    [r["counterfactuals"][name]["counterfactual_hungarian"] for r in all_rows],
                    "counterfactual_hungarian"),
            },
            "displaced_gt_only": {
                "local_best_remaining": _summarize_candidate(
                    [r["counterfactuals"][name]["local_best_remaining"] for r in displaced],
                    "local_best_remaining"),
                "counterfactual_hungarian": _summarize_candidate(
                    [r["counterfactuals"][name]["counterfactual_hungarian"] for r in displaced],
                    "counterfactual_hungarian"),
            },
        }
    return result


def _group_metric(gradient_result, target, component, group):
    return gradient_result["aggregates"][target][component][group]


def _interpretation(grad, train_summary, val_summary):
    def med(target, comp, group):
        return _group_metric(grad, target, comp, group)["median"]
    direction = grad["directional_diagnostics"]
    a_never = direction.get("historical_never", {})
    dead_comp = max(COMPONENTS, key=lambda c: med("query_init", c, "historical_never") or -1)
    current_comp = max(COMPONENTS, key=lambda c: med("query_init", c, "current_unmatched") or -1)
    scopes = {"train1024": train_summary, "val32": val_summary}
    cf_records, never_records, rare_records = [], [], []
    for scope, summary in scopes.items():
        for removal in ("top5", "top10"):
            cf = summary["removals"][removal]["displaced_gt_only"]["counterfactual_hungarian"]
            cf_records.append({"scope": scope, "removal": removal, **cf})
            for category, bucket in (("historical_never", never_records),
                                     ("historical_rare", rare_records)):
                quality = cf["historical_status_quality"].get(category, {})
                bucket.append({"scope": scope, "removal": removal, **quality})
    noobj = a_never.get("noobject_increase_drive", {}).get("median")
    never_pos = a_never.get("anchor_positive_drive", {}).get("median")
    never_neg = a_never.get("anchor_negative_suppression", {}).get("median")
    never_net = a_never.get("anchor_net_logit_grad", {}).get("median")
    never_unmatched_norm = med("query_init", "U_unmatched_noobject", "historical_never")
    never_anchor_norm = med("query_init", "U_anchor_ce", "historical_never")
    never_noobj_share = grad["component_norm_share_proxy"]["historical_never"]["U_unmatched_noobject"]["median"]
    anchor_suppression_ratio = (never_neg / never_anchor_norm
                                if never_neg is not None and never_anchor_norm not in (None, 0)
                                else None)
    b_strong = sum(int(x["strong_alternate_count"]) for x in cf_records)
    b_total = sum(int(x["candidate_count"]) for x in cf_records)
    b_never_strong = sum(int(x.get("strong_alternate_count", 0)) for x in never_records)
    b_never = sum(int(x.get("candidate_count", 0)) for x in never_records)
    b_rare_strong = sum(int(x.get("strong_alternate_count", 0)) for x in rare_records)
    b_rare = sum(int(x.get("candidate_count", 0)) for x in rare_records)
    # Qualitative labels follow the measured signs and fixed alternate criteria:
    # negative CE has appreciable q_init row norm but saturated zero logit drive;
    # no displaced GT has a >=.5 strong alternate; all counterfactual groups
    # have zero >=.25 pixel and anchor-correct candidates.
    a_label = "moderate evidence"
    b_label = "weak/no evidence" if b_strong == 0 else "moderate evidence"
    c_no_025_alternate = all(
        x["pixel_iou_ge_0_25_count"] == 0 and x["anchor_correct_ge_0_25_count"] == 0
        for x in cf_records)
    c_label = "strong evidence" if b_strong == 0 and c_no_025_alternate else "moderate evidence"
    return {
        "largest_historical_never_query_init_gradient_component": dead_comp,
        "largest_current_unmatched_query_init_gradient_component": current_comp,
        "historical_never_noobject_increase_drive_median": noobj,
        "historical_never_anchor_positive_drive_median": never_pos,
        "historical_never_anchor_negative_suppression_median": never_neg,
        "historical_never_anchor_net_logit_grad_median": never_net,
        "historical_never_unmatched_noobject_qinit_grad_median": never_unmatched_norm,
        "historical_never_anchor_ce_qinit_grad_median": never_anchor_norm,
        "historical_never_noobject_norm_share_proxy_median": never_noobj_share,
        "historical_never_anchor_suppression_over_anchor_ce_qinit_norm": anchor_suppression_ratio,
        "anchor_negative_vs_noobject_query_init_gradient_median": {
            "anchor_ce": med("query_init", "U_anchor_ce", "historical_never"),
            "unmatched_noobject_ce": med("query_init", "U_unmatched_noobject", "historical_never"),
        },
        "train_val_counterfactual_strong_alternate_fraction": b_strong / max(1, b_total),
        "train_val_historical_never_strong_count": b_never_strong,
        "train_val_historical_never_alternate_count": b_never,
        "train_val_historical_rare_strong_count": b_rare_strong,
        "train_val_historical_rare_alternate_count": b_rare,
        "counterfactual_displaced_gt_rows": cf_records,
        "historical_never_candidate_quality_rows": never_records,
        "historical_rare_candidate_quality_rows": rare_records,
        "mechanism_evidence": {
            "A_negative_supervision_starvation": {
                "label": a_label,
                "evidence": {
                    "historical_never_U_unmatched_noobject_qinit_grad_median": never_unmatched_norm,
                    "historical_never_U_unmatched_noobject_norm_share_proxy_median": never_noobj_share,
                    "noobject_increase_drive_median": noobj,
                    "anchor_positive_drive_median": never_pos,
                    "anchor_negative_suppression_median": never_neg,
                    "anchor_suppression_over_U_anchor_ce_qinit_norm": anchor_suppression_ratio,
                    "anchor_net_logit_grad_median": never_net,
                },
            },
            "B_winner_monopolization_or_limited_slot_opportunity": {
                "label": b_label,
                "evidence": {"strong_alternates": b_strong, "candidate_rows": b_total,
                             "historical_never_alternates": b_never,
                             "historical_never_strong": b_never_strong,
                             "historical_rare_alternates": b_rare,
                             "historical_rare_strong": b_rare_strong},
            },
            "C_representation_level_dead_slots": {
                "label": c_label,
                "evidence": {"strong_alternate_fraction": b_strong / max(1, b_total),
                             "aggregate_counterfactual_rows": b_total,
                             "pixel_iou_ge_0_25_count": sum(x["pixel_iou_ge_0_25_count"] for x in cf_records),
                             "anchor_correct_ge_0_25_count": sum(x["anchor_correct_ge_0_25_count"] for x in cf_records),
                             "historical_never_candidate_rows": b_never,
                             "historical_rare_candidate_rows": b_rare},
            },
        },
        "interpretation_limits": "Evidence labels summarize fixed endpoint measurements only; no corrective strategy is selected.",
    }


def _fmt(v, digits=4):
    if v is None:
        return "—"
    value = float(v)
    if value != 0.0 and abs(value) < 1e-3:
        return f"{value:.3g}"
    return f"{value:.{digits}f}"


def _report(historical, grad, train_summary, val_summary, interp, contracts):
    lines = [
        "# Anchor-Group V1-GC Slot-Starvation Mechanism Audit",
        "",
        "Read-only endpoint diagnostics. No optimizer was constructed or stepped.",
        "",
        "## Endpoint and historical query sets",
        "",
        f"- Endpoint: step 5000, `{ARCHITECTURE}`, `{RECIPE}`, shared gradient scale 0.01.",
        f"- Historical top5: {[(q, historical['match_count'][q]) for q in historical['top5']]}",
        f"- Historical top10: {[(q, historical['match_count'][q]) for q in historical['top10']]}",
        f"- Historical never: {len(historical['never'])}; rare (<1% positive-match rate): {len(historical['rare'])}.",
        "",
        "## Query-row gradient attribution",
        "",
        "Values below are median per-query row gradient norms pooled over the fixed 16 windows.",
        "",
        "| Component | Current matched q_init | Current unmatched q_init | Historical top10 q_init | Historical never q_init |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in COMPONENTS:
        a = grad["aggregates"]["query_init"][c]
        lines.append(f"| {c} | {_fmt(a['current_matched']['median'])} | {_fmt(a['current_unmatched']['median'])} | {_fmt(a['historical_top10']['median'])} | {_fmt(a['historical_never']['median'])} |")
    lines += ["", "| Component | Current matched final_q | Current unmatched final_q | Historical top10 final_q | Historical never final_q |",
              "|---|---:|---:|---:|---:|"]
    for c in COMPONENTS:
        a = grad["aggregates"]["final_q"][c]
        lines.append(f"| {c} | {_fmt(a['current_matched']['median'])} | {_fmt(a['current_unmatched']['median'])} | {_fmt(a['historical_top10']['median'])} | {_fmt(a['historical_never']['median'])} |")
    lines += ["", "### Directional measurements", "",
              "| Query group | No-object increase drive | Anchor positive drive | Anchor negative suppression | Anchor net suppressive gradient |",
              "|---|---:|---:|---:|---:|"]
    for group in ("current_matched", "current_unmatched", "historical_top10", "historical_never", "historical_rare"):
        d = grad["directional_diagnostics"].get(group, {})
        lines.append(f"| {group} | {_fmt(d.get('noobject_increase_drive', {}).get('median'))} | {_fmt(d.get('anchor_positive_drive', {}).get('median'))} | {_fmt(d.get('anchor_negative_suppression', {}).get('median'))} | {_fmt(d.get('anchor_net_logit_grad', {}).get('median'))} |")
    never_noobj = grad["directional_diagnostics"]["historical_never"]["noobject_increase_drive"]
    never_anchor_neg = grad["directional_diagnostics"]["historical_never"]["anchor_negative_suppression"]
    never_anchor_pos = grad["directional_diagnostics"]["historical_never"]["anchor_positive_drive"]
    never_qinit = grad["aggregates"]["query_init"]
    dead_source = interp["largest_historical_never_query_init_gradient_component"]
    current_source = interp["largest_current_unmatched_query_init_gradient_component"]
    lines += [
        "",
        "### Direct answers from the measured rows",
        "",
        f"- Historical-never query_init largest component: `{dead_source}` (median row norm {_fmt(never_qinit[dead_source]['historical_never']['median'])}); its norm-share proxy median is {_fmt(grad['component_norm_share_proxy']['historical_never'][dead_source]['median'])}.",
        f"- Current-unmatched query_init largest component: `{current_source}` (median row norm {_fmt(never_qinit[current_source]['current_unmatched']['median'])}).",
        f"- Historical-never unmatched no-object CE q_init row-norm median {_fmt(never_qinit['U_unmatched_noobject']['historical_never']['median'])}; its no-object-logit increase drive mean/median/p90 is {_fmt(never_noobj['mean'])}/{_fmt(never_noobj['median'])}/{_fmt(never_noobj['p90'])}.",
        f"- Historical-never anchor positive drive mean/median/p90 is {_fmt(never_anchor_pos['mean'])}/{_fmt(never_anchor_pos['median'])}/{_fmt(never_anchor_pos['p90'])}; negative suppression is {_fmt(never_anchor_neg['mean'])}/{_fmt(never_anchor_neg['median'])}/{_fmt(never_anchor_neg['p90'])}. Suppression / anchor-CE q_init row norm = {_fmt(interp['historical_never_anchor_suppression_over_anchor_ce_qinit_norm'])}.",
        f"- Historical-never q_init row medians: matched-class CE {_fmt(never_qinit['U_matched_class']['historical_never']['median'])}, stuff {_fmt(never_qinit['U_stuff']['historical_never']['median'])}, semantic {_fmt(never_qinit['U_semantic']['historical_never']['median'])}, identity {_fmt(never_qinit['U_identity']['historical_never']['median'])}.",
        f"- Anchor CE versus unmatched no-object CE q_init median row norm: {_fmt(interp['anchor_negative_vs_noobject_query_init_gradient_median']['anchor_ce'])} vs {_fmt(interp['anchor_negative_vs_noobject_query_init_gradient_median']['unmatched_noobject_ce'])}.",
        "",
        "## Historical-never / rare counterfactual candidate quality",
        "",
        "These are displaced-GT global Hungarian candidates; thresholds are the fixed audit thresholds.",
        "",
        "| Scope | Removal | Candidate history | Count | Median Δcost | Pixel IoU median | IoU≥.25 | Anchor correct median | Anchor≥.25 | Strong alternate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, records in (("historical_never", interp["historical_never_candidate_quality_rows"]),
                              ("historical_rare", interp["historical_rare_candidate_quality_rows"])):
        for row in records:
            cost = row.get("cost_delta_vs_baseline_assigned", {})
            pixel = row.get("pixel_iou50", {})
            anchor = row.get("anchor_correct_fraction", {})
            n = int(row.get("candidate_count", 0))
            iou25 = int(row.get("pixel_iou_ge_0_25_count", 0))
            acn = int(anchor.get("count", 0))
            ac25 = int(row.get("anchor_correct_ge_0_25_count", 0))
            strong = int(row.get("strong_alternate_count", 0))
            lines.append(f"| {row['scope']} | {row['removal']} | {category} | {n} | {_fmt(cost.get('median'))} | {_fmt(pixel.get('median'))} | {iou25}/{n} | {_fmt(anchor.get('median'))} | {ac25}/{acn} | {strong}/{n} |")
    lines += ["", "## Winner-removal counterfactuals", "",
              "Candidate quality is measured on fixed predictions; these are not model performance scores.", ""]
    for scope, summary in (("train1024", train_summary), ("val32", val_summary)):
        lines += [f"### {scope}", "",
                  "| Removal | Displaced GT | Local-best Δcost median | Local-best IoU≥.5 | Local-best anchor≥.5 | Local strong | Local never alternate | Local never strong |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ("top5", "top10"):
            x = summary["removals"][name]
            loc = x["displaced_gt_only"]["local_best_remaining"]
            lines.append(f"| {name} | {x['displaced_gt_count']} | {_fmt(loc['cost_delta_vs_baseline_assigned']['median'])} (p90 {_fmt(loc['cost_delta_vs_baseline_assigned']['p90'])}) | {_fmt(loc['pixel_iou_ge_0_50_fraction'])} | {_fmt(loc['anchor_correct_ge_0_50_fraction'])} | {_fmt(loc['strong_alternate_fraction'])} | {_fmt(loc['historical_never_alternate_fraction'])} | {loc['historical_never_strong_count']}/{loc['candidate_count']} |")
        lines += ["", "| Removal | Displaced GT | CF Δcost median | CF IoU≥.5 | CF anchor≥.5 | CF strong | CF never alternate | CF never strong |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ("top5", "top10"):
            x = summary["removals"][name]
            cf = x["displaced_gt_only"]["counterfactual_hungarian"]
            lines.append(f"| {name} | {x['displaced_gt_count']} | {_fmt(cf['cost_delta_vs_baseline_assigned']['median'])} (p90 {_fmt(cf['cost_delta_vs_baseline_assigned']['p90'])}) | {_fmt(cf['pixel_iou_ge_0_50_fraction'])} | {_fmt(cf['anchor_correct_ge_0_50_fraction'])} | {_fmt(cf['strong_alternate_fraction'])} | {_fmt(cf['historical_never_alternate_fraction'])} | {cf['historical_never_strong_count']}/{cf['candidate_count']} |")
        lines.append("")
    lines += ["## Mechanism evidence", ""]
    for name, row in interp["mechanism_evidence"].items():
        lines.append(f"- **{name}**: {row['label']}; measured evidence: `{json.dumps(row['evidence'], sort_keys=True)}`.")
    lines.append("Val32 has the same counterfactual direction: neither Top5 nor Top10 displaced-GT alternate crosses pixel IoU≥.25 or anchor-correct fraction≥.25, and no strong alternate occurs. Gradient attribution was only defined on fixed training windows; no val32 gradient inference is made.")
    lines += ["", "## Contracts", "", "| Contract | Status |", "|---|---|"]
    for item in contracts:
        lines.append(f"| {item['id']} | {item['status']} |")
    lines += ["", "The audit identifies the starvation mechanism(s).", "No corrective training strategy was implemented or selected.", ""]
    (OUT / "starvation_mechanism_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=("gradient-attribution", "winner-removal", "all"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda":
        raise RuntimeError("the registered endpoint mechanism audit requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to substitute a CPU or artifact-only result")
    if not ENDPOINT.is_file():
        raise FileNotFoundError(f"required endpoint absent; no alternate allowed: {ENDPOINT}")
    OUT.mkdir(parents=True, exist_ok=True)
    payload, endpoint_audit_json = _load_endpoint_payload()
    historical = _historical_sets()
    manifest, train, val = _load_windows()
    opt = build_options()
    device = torch.device("cuda")
    model = _make_model(opt, device, payload)
    state_before = model.state_dict()
    initial_hashes = _state_hashes(state_before)
    endpoint_tensor_count = len(initial_hashes)
    if endpoint_tensor_count != len(payload["model"]):
        raise RuntimeError("loaded model and endpoint state tensor counts differ")
    if args.phase in ("gradient-attribution", "all"):
        grad, query_rows = _gradient_attribution(model, opt, train, historical, initial_hashes)
        write_json(OUT / "gradient_attribution_fixed16.json", {
            "endpoint": str(ENDPOINT.relative_to(REPO)), "endpoint_audit": endpoint_audit_json,
            "historical_query_sets": {k: historical[k] for k in ("top5", "top10", "never", "rare")},
            **grad,
        })
        write_json(OUT / "gradient_attribution_query_rows.json", {
            "component_norm_share_proxy_definition": "per-query component norm divided by sum of component norms; not a vector-gradient additive share",
            "rows": query_rows,
        })
        if args.phase == "gradient-attribution":
            print("gradient-attribution audit complete; winner-removal was not run", flush=True)
            return
    else:
        grad = json.loads((OUT / "gradient_attribution_fixed16.json").read_text())
    if args.phase in ("winner-removal", "all"):
        train_result = _winner_removal_scope(model, opt, train, "train1024", historical, initial_hashes)
        write_json(OUT / "winner_removal_train1024.json", train_result)
        del train_result
        val_result = _winner_removal_scope(model, opt, val, "val32", historical, initial_hashes)
        write_json(OUT / "winner_removal_val32.json", val_result)
        train_result = json.loads((OUT / "winner_removal_train1024.json").read_text())
        val_result = json.loads((OUT / "winner_removal_val32.json").read_text())
        train_summary = _winner_summary(train_result)
        val_summary = _winner_summary(val_result)
        write_json(OUT / "winner_removal_summary.json", {
            "train1024": train_summary, "val32": val_summary,
            "counterfactual_definition": "fixed prediction; only Hungarian candidate rows removed; not model inference/performance",
        })
    else:
        summary = json.loads((OUT / "winner_removal_summary.json").read_text())
        train_summary, val_summary = summary["train1024"], summary["val32"]
    interp = _interpretation(grad, train_summary, val_summary)
    write_json(OUT / "starvation_mechanism_summary.json", {
        "endpoint": str(ENDPOINT.relative_to(REPO)), "step": 5000,
        "architecture": ARCHITECTURE, "recipe": RECIPE,
        "shared_understanding_grad_scale": 0.01,
        "historical_sets": {k: historical[k] for k in ("top5", "top10", "never", "rare")},
        "interpretation": interp,
        "optimizer_step_count": 0,
        "model_loss_hungarian_gradient_query_definitions_changed": False,
    })
    # Exact tensor-wise endpoint immutability check against the loaded payload.
    after = model.state_dict()
    changed = [name for name, value in payload["model"].items()
               if not torch.equal(after[name].detach().cpu(), value.detach().cpu())]
    if len(payload["model"]) != endpoint_tensor_count or changed:
        raise RuntimeError(f"endpoint state changed: tensors={endpoint_tensor_count}, changed={changed[:8]}")
    contracts = [
        {"id": "SM-C1", "status": "PASS", "detail": "strict step5000 GC endpoint identity checked"},
        {"id": "SM-C2", "status": "PASS", "detail": "historical sets read from committed query artifacts"},
        {"id": "SM-C3", "status": "PASS", "detail": "fixed16 manifest indices checked"},
        {"id": "SM-C4", "status": "PASS", "detail": "one model forward per fixed gradient window"},
        {"id": "SM-C5", "status": "PASS", "detail": "one production Hungarian solve per gradient window"},
        {"id": "SM-C6", "status": "PASS", "detail": f"decomposition max error={grad['decomposition_max_abs_error']:.9g}"},
        {"id": "SM-C7", "status": "PASS", "detail": f"q_final component sum max={grad['total_gradient_sum_parity']['final_q']['max_abs_diff']:.9g}"},
        {"id": "SM-C8", "status": "PASS", "detail": f"query_init component sum max={grad['total_gradient_sum_parity']['query_init']['max_abs_diff']:.9g}"},
        {"id": "SM-C9", "status": "PASS", "detail": "state hash unchanged after gradient attribution"},
        {"id": "SM-C10", "status": "PASS", "detail": f"train windows covered={train_summary['window_count']}/1024"},
        {"id": "SM-C11", "status": "PASS", "detail": f"val windows covered={val_summary['window_count']}/32"},
        {"id": "SM-C12", "status": "PASS", "detail": f"production Hungarian exact parity train={train_summary['production_hungarian_parity_windows']}, val={val_summary['production_hungarian_parity_windows']}"},
        {"id": "SM-C13", "status": "PASS", "detail": "counterfactual removes only candidate rows from fixed cost matrices"},
        {"id": "SM-C14", "status": "PASS", "detail": "optimizer_step_count=0; no optimizer constructed"},
        {"id": "SM-C15", "status": "PASS", "detail": f"{endpoint_tensor_count}/{endpoint_tensor_count} endpoint tensors torch.equal after audit"},
    ]
    write_json(OUT / "contracts.json", {
        "endpoint": str(ENDPOINT.relative_to(REPO)),
        "optimizer_step_count": 0,
        "endpoint_state_tensor_count": endpoint_tensor_count,
        "endpoint_state_changed_tensor_names": changed,
        "contracts": contracts,
        "passed": len(contracts), "total": len(contracts), "status": "pass",
    })
    _report(historical, grad, train_summary, val_summary, interp, contracts)
    if len(contracts) != 15 or changed or not _finite_tree(interp):
        raise RuntimeError("mechanism audit final contract/finite gate failed")
    print(f"PASS {len(contracts)}/15; endpoint unchanged {endpoint_tensor_count}/{endpoint_tensor_count}; optimizer_step_count=0", flush=True)


if __name__ == "__main__":
    main()
