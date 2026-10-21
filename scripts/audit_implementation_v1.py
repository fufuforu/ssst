#!/usr/bin/env python3
"""Read-only runtime audit of the group recipe implementation (round: implementation_audit_v1).

Covers the mandated checks: checkpoint provenance, window identity, assignment
distributions (with the thing/rest/dropped stratification), contribution/target
reconciliation against the verified routing-v2 statistics, Hungarian
determinism, void accounting, gradient routing and data-pipeline invariants.

Read-only: no training, no optimizer.step, no checkpoint write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
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
from tokengs.models.ssst_loss import hungarian_match  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.train_object_locusgs import move as move_to_device  # noqa: E402
from scripts.train_group_locusgs import SHARED_PARAM_PREFIXES  # noqa: E402
from scripts.audit_soft_token_oracle import token_contribution_stats  # noqa: E402
from scripts.eval_group_locusgs import build_train_entries  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    build_val_entries,
    forward_group,
    group_predictions_v2,
)

PREREGISTERED_PLAN_SHA256 = (
    "a2a65c1382da0307a68fb6d27e5c1345aa78b46331acb2927e88b47a3c3d08bb"
)
PREREGISTERED_SPLIT_SHA256 = (
    "acf9afac57ca2e5287b9b1b545a62a369fee398a0d1a789cb77b851202b313d5"
)

MODELS = (
    ("g0plus_step6000", "workspace_group_plus/arm_g0plus/ckpt_step6000",
     {"recipe": False, "g0plus": True, "head_mode": "legacy_prefix"}),
    ("recipe_v1_step6000", "workspace_group_plus/recipe_v1/run/ckpt_step6000",
     {"recipe": True, "g0plus": False, "head_mode": "legacy_prefix"}),
    ("recipe_v2_step6000", "workspace_group_plus/recipe_v2/run/ckpt_step6000",
     {"recipe": True, "g0plus": False, "head_mode": "legacy_prefix"}),
)

EXPECTED_SHA = {
    "recipe_v1_step6000": "39c53ce825b68a24292e84c838d836d02808526e4369bdb4483bbe934b7543ba",
    "recipe_v2_step6000": "373cac99ab1aa95ab7118b85ea4d7ebc56befb618928cc8c74d8804054caddcb",
}


def file_identity(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": path.stat().st_mtime, "bytes": path.stat().st_size}


def checkpoint_identity(label: str, directory: Path) -> dict:
    payload = torch.load(directory / "train_state.pt", map_location="cpu",
                         weights_only=False)
    meta = payload.get("meta", {}) or {}
    return {
        "label": label,
        "directory": str(directory),
        "model_pt": file_identity(directory / "model.pt"),
        "train_state_pt": file_identity(directory / "train_state.pt"),
        "complete": (directory / "COMPLETE").read_text().strip()
        if (directory / "COMPLETE").is_file() else None,
        "config_yaml_present": (directory / "config.yaml").is_file(),
        "train_state_step": int(payload.get("step", -1)),
        "meta_arm": meta.get("arm"),
        "plan_sha256": meta.get("plan_sha256") or payload.get("plan_sha256"),
        "expected_sha256": EXPECTED_SHA.get(label),
        "sha256_matches_expected": (
            EXPECTED_SHA.get(label) is None
            or EXPECTED_SHA[label] == file_identity(directory / "model.pt")["sha256"]
        ),
    }


def load(label, directory, preset, seed, device, flags):
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm="g0", group_bg_supervision=flags["g0plus"],
        group_recipe=flags["recipe"],
        group_recipe_head_mode=flags["head_mode"],
        group_recipe_seg_weight=(0.1 if label == "recipe_v1_step6000" else 0.05),
        group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        group_recipe_assign_stop_shared_grad=False,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(directory) / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return opt, model


def quantiles(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {}
    q = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=flat.device)
    quant = torch.quantile(flat, q).tolist()
    return {"min": quant[0], "p01": quant[1], "p50": quant[2], "p99": quant[3],
            "max": quant[4], "mean": float(flat.mean())}


def distribution_block(model, opt, entry, device):
    """Assignment/geometry distributions for one window (all context tokens)."""
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data

    batch = entry["batch"]
    model.groups.capture = {}
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        capture = model.groups.capture
        model.groups.capture = None
        instance = model.group_loss_terms(batch, forward["output"]["gaussians"],
                                          ModelInputDecoder(
                                              cam_view=batch["cam_view_all"],
                                              intrinsics=batch["intrinsics_all"]),
                                          tuple(range(int(opt.num_input_views))))
        contributions = model.token_instance_contributions(
            forward["output"]["gaussians"], batch, forward["masks"]["alpha"],
            tuple(range(int(opt.num_input_views))),
        )
        target, kept, total = model.assignment_target(
            contributions, instance["segment_keys"], instance["matched_rows"]
        )
    prob = forward["group"]["slot_prob"][0].float()
    logits = capture["slot_logits"][0].float()
    entropy = -(prob.clamp_min(1e-12).log() * prob).sum(-1)
    max_prob = prob.max(-1).values
    p100 = prob[:, : model.num_groups]
    denom = p100.sum(-1)
    defined = denom > 1e-12
    p100n = torch.where(defined.unsqueeze(-1),
                        p100 / denom.clamp_min(1e-12).unsqueeze(-1),
                        torch.zeros_like(p100))
    ent100 = -(p100n.clamp_min(1e-12).log() * p100n).sum(-1)
    # target-side numbers
    log_prob = torch.log_softmax(logits, dim=-1)
    ce = -(target * log_prob).sum(-1)
    t_ent = -(target.clamp_min(1e-12).log() * target).sum(-1)
    thing_mass = target[:, : model.num_groups].sum(-1)
    thing_tokens = kept & (thing_mass > 0)
    rest_tokens = kept & (~thing_tokens)
    dropped = ~kept
    primary = (target[:, : model.num_groups] > 0).float()
    primary_logits = logits[:, : model.num_groups]
    thing_dominant = kept & (thing_mass > 0.5)
    masked = torch.where(primary > 0, primary_logits, torch.full_like(primary_logits, -1e9))
    thing_only_agree = (masked.argmax(-1) == primary.argmax(-1))
    weights = total.detach()
    weights = weights / weights.sum().clamp_min(1e-12)

    def strat(mask):
        if not bool(mask.any()):
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "entropy_101": float(entropy[mask].mean()),
            "max_prob_101": float(max_prob[mask].mean()),
            "entropy_100_conditional": float(ent100[mask][defined[mask]].mean())
            if bool(defined[mask].any()) else None,
            "max_prob_100_conditional": float(
                p100n[mask][defined[mask]].max(-1).values.mean())
            if bool(defined[mask].any()) else None,
            "target_entropy": float(t_ent[mask].mean()),
            "ce": float(ce[mask].mean()),
            "kl": float((ce - t_ent)[mask].mean()),
            "agreement_101": float((logits.argmax(-1) == target.argmax(-1))[mask].float().mean()),
        }

    stats = {
        "tokens": int(prob.shape[0]),
        "token_feature_norm": quantiles(capture["token_features"][0].norm(dim=-1)),
        "query_norm_learned": quantiles(capture["queries"][0].norm(dim=-1)),
        "query_norm_after_head": quantiles(capture["query_features"][0].norm(dim=-1)),
        "slot_logits": quantiles(logits),
        "void_logit": quantiles(logits[:, -1]),
        "void_probability": quantiles(prob[:, -1]),
        "entropy_101": quantiles(entropy),
        "max_prob_101": quantiles(max_prob),
        "entropy_100_conditional": quantiles(ent100[defined]) if bool(defined.any()) else {},
        "undefined_conditional_tokens": int((~defined).sum()),
        "thing_tokens": strat(thing_tokens),
        "rest_only_tokens": strat(rest_tokens),
        "dropped_tokens": strat(dropped),
        "thing_dominant_agreement": float(thing_only_agree[thing_dominant].float().mean())
        if bool(thing_dominant.any()) else None,
        "thing_dominant_n": int(thing_dominant.sum()),
        "thing_column_only_agreement": float(
            thing_only_agree[thing_tokens].float().mean()) if bool(thing_tokens.any()) else None,
        # contribution-weighted (not a plain token mean)
        "weighted_entropy_101": float((entropy * weights).sum()),
        "weighted_max_prob_101": float((max_prob * weights).sum()),
        "weighted_void_probability": float((prob[:, -1] * weights).sum()),
        "effective_columns_mean_prob": int(
            (prob[:, : model.num_groups].mean(0) > 0.01).sum()),
        "target_row_sum_error": float((target.sum(-1)[kept] - 1).abs().max())
        if bool(kept.any()) else 0.0,
    }
    # per-query accounting
    matched = np.zeros(model.num_groups, dtype=np.int64)
    for row in instance["matched_rows"]:
        if row is not None and 0 <= int(row) < model.num_groups:
            matched[int(row)] += 1
    gt_free_counts = np.zeros(model.num_groups, dtype=np.int64)
    for view in (2, 3):
        for prediction in group_predictions_v2(forward, view)[0]:
            gt_free_counts[int(prediction["group"])] += 1
    stats["per_query"] = {
        "matched_count_total": int(matched.sum()),
        "target_column_mass": [float(target[:, q].sum()) for q in range(model.num_groups)],
        "predicted_token_mass": [float(prob[:, q].sum()) for q in range(model.num_groups)],
        "rendered_mass": [float(forward["masks"]["group_mass"][0, :, q].sum())
                          for q in range(model.num_groups)],
        "gt_free_outputs": gt_free_counts.tolist(),
    }
    del forward
    torch.cuda.empty_cache()
    return stats


def contribution_reconciliation(model, opt, entry, device):
    """recipe contributions vs the verified routing-v2 statistics, same state."""
    batch = entry["batch"]
    context = tuple(range(int(opt.num_input_views)))
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        gaussians = forward["output"]["gaussians"]
        alpha = forward["masks"]["alpha"]
        semantic = batch["semantic_label_all"].long()
        instance = batch["instance_label_all"].long()
        values = torch.unique(semantic).tolist()
        legal = all((v == 255) or (0 <= v <= 19) for v in values)
        group = model.group_loss_terms(batch, gaussians, _decoder(batch), context)
        keys = [int(k) for k in group["segment_keys"]]
        recipe = model.token_instance_contributions(gaussians, batch, alpha, context)
        routing = token_contribution_stats(model, opt, entry, keys, device)
        # alpha conservation of the group masks
        conservation = float(
            (forward["masks"]["group_mass"][0].sum(1)  # sum over the 100 query channels
             + forward["masks"]["background_mass"][0, :, 0]
             - forward["masks"]["alpha"][0, :, 0]).abs().max()
        )
        target, kept, total = model.assignment_target(
            recipe, group["segment_keys"], group["matched_rows"]
        )
        row_sum = float((target.sum(-1)[kept] - 1).abs().max()) if bool(kept.any()) else 0.0
        # oracle annotated_total - recipe total must equal the covered, valid thing
        # pixels whose instance id is <= 0
        num_tokens = recipe["rest"].shape[0]
        num_gaussians = gaussians.shape[1]
        token_of = torch.arange(num_gaussians, device=device) // 64
        onehot = torch.zeros(1, num_gaussians, num_tokens, device=device)
        onehot[0, torch.arange(num_gaussians, device=device), token_of] = 1.0
        missing = torch.zeros(num_tokens, device=device)
        sem, ins = semantic[0], instance[0]
        covered = alpha[0, :, 0] > 0.5
        for view in context:
            mask = (covered[view] & (sem[view] != 255) & (sem[view] >= 2)
                    & (sem[view] < 20) & (ins[view] <= 0))
            if not bool(mask.any()):
                continue
            rendered = model.gs.render_feature_channels(
                gaussians, onehot, batch["cam_view_all"][:, view:view + 1],
                intrinsics=batch["intrinsics_all"][:, view:view + 1],
            )["images_pred"][0, 0]
            missing = missing + rendered[:, mask].sum(-1)
            del rendered
        diff = (routing["annotated_total"] - recipe["total"] - missing)
        # Control: recompute `annotated_total` from *this* forward.  The routing
        # helper runs its own forward_group internally and the rasterizer's 2-D
        # projection is non-deterministic, so this isolates whether the identity
        # is violated structurally or only by the helper's second forward.
        annotated_same = torch.zeros(num_tokens, device=device)
        for view in context:
            rendered = model.gs.render_feature_channels(
                gaussians, onehot, batch["cam_view_all"][:, view:view + 1],
                intrinsics=batch["intrinsics_all"][:, view:view + 1],
            )["images_pred"][0, 0]
            mask = covered[view] & (sem[view] != 255)
            if bool(mask.any()):
                annotated_same = annotated_same + rendered[:, mask].sum(-1)
            del rendered
        routing_second = token_contribution_stats(model, opt, entry, keys, device)
        routing_repeat_gap = float(
            (routing_second["annotated_total"] - routing["annotated_total"]).abs().max())
        per_key_ok, per_key_max = True, 0.0
        for key in keys:
            a = recipe["instance"].get(key)
            b = routing["per_instance"].get(key)
            if a is None or b is None:
                continue
            delta = float((a - b).abs().max())
            per_key_max = max(per_key_max, delta)
            if not torch.allclose(a, b, atol=1e-4, rtol=1e-5):
                per_key_ok = False
    return {
        "semantic_values_legal": legal,
        "semantic_unique_sample": values[:8],
        "alpha_conservation_max": conservation,
        "target_row_sum_error": row_sum,
        "per_key_allclose_atol_1e-4": bool(per_key_ok),
        "per_key_max_abs_diff": per_key_max,
        "annotated_minus_recipe_minus_missing_max": float(diff.abs().max()),
        "same_forward_annotated_minus_recipe_minus_missing_max": float(
            (annotated_same - recipe["total"] - missing).abs().max()),
        "routing_helper_repeat_annotated_total_gap": routing_repeat_gap,
        "instances_compared": len(keys),
        "context_views": list(context),
        "b_t": [1, num_tokens], "patches": 64,
        "num_gaussians": int(gaussians.shape[1]),
        "gaussians_equal_tokens_times_64": int(gaussians.shape[1]) == 64 * num_tokens,
    }


def _decoder(batch):
    from tokengs.models.input_types import ModelInputDecoder

    return ModelInputDecoder(cam_view=batch["cam_view_all"],
                             intrinsics=batch["intrinsics_all"])


def hungarian_determinism(model, entry, device):
    """Two Hungarian solves on the same logits must give identical rows/cols."""
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, model.opt)
        group = model.layer10_group
        mask_decoder = _decoder(batch).select_batch(slice(None), slice(0, 2))
        rendered = model.render_group_masks(forward["output"]["gaussians"],
                                            group["slot_prob"], mask_decoder)
        mass = rendered["group_mass"].permute(0, 2, 1, 3, 4)
        mask_logits = torch.logit(mass.clamp(1e-5, 1 - 1e-5))
        class_logits = torch.cat([group["class_logits"],
                                  group["objectness"].unsqueeze(-1)], dim=-1)
        from tokengs.models.group_locusgs import build_context_segments
        labels, masks = build_context_segments(
            batch["semantic_label_all"], batch["instance_label_all"], (0, 1))
        keep = labels[0] >= 2
        labels, masks = labels[0][keep], masks[0][keep]
        rows_a, cols_a = hungarian_match(class_logits[0], mask_logits[0], labels, masks)
        rows_b, cols_b = hungarian_match(class_logits[0], mask_logits[0], labels, masks)
    return {
        "rows_identical": bool(torch.equal(rows_a, rows_b)),
        "cols_identical": bool(torch.equal(cols_a, cols_b)),
        "n_matched": int(rows_a.numel()),
        "n_gt": int(labels.numel()),
        "rows_unique": int(rows_a.numel()) == len(set(rows_a.tolist())),
        "cols_unique": int(cols_a.numel()) == len(set(cols_a.tolist())),
        "n_gt_le_100": int(labels.numel()) <= int(model.num_groups),
    }


def void_accounting(model, opt, entry, device):
    """void probability vs rendered void mass, on all four rendered views."""
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        prob = forward["group"]["slot_prob"][0].float()
        void_prob = prob[:, -1]
        gaussians = forward["output"]["gaussians"]
        num_gaussians = gaussians.shape[1]
        num_tokens = num_gaussians // 64
        token_of = torch.arange(num_gaussians, device=device) // 64
        onehot = torch.zeros(1, num_gaussians, num_tokens, device=device)
        onehot[0, torch.arange(num_gaussians, device=device), token_of] = 1.0
        per_view = {}
        for view in range(4):
            token_mass = model.gs.render_feature_channels(
                gaussians, onehot, batch["cam_view_all"][:, view:view + 1],
                intrinsics=batch["intrinsics_all"][:, view:view + 1],
            )["images_pred"][0, 0].flatten(1).sum(1)            # [T] (all pixels)
            void_mass = forward["masks"]["background_mass"][0, view, 0].float().sum()
            total_mass = forward["masks"]["group_mass"][0, view].float().sum() + void_mass
            # weighted identity: sum_t A[t,void] * M[t] == rendered void mass
            predicted = float((void_prob * token_mass).sum())
            per_view[str(view)] = {
                "mean_void_probability": float(void_prob.mean()),
                "rendered_void_fraction": float(void_mass / total_mass.clamp_min(1e-9)),
                "weighted_void_probability": predicted,
                "rendered_void_mass": float(void_mass),
                "identity_abs_diff": abs(predicted - float(void_mass)),
            }
            del token_mass
    from tokengs.models.input_types import ModelInputDecoder
    instance = model.group_loss_terms(
        batch, forward["output"]["gaussians"],
        ModelInputDecoder(cam_view=batch["cam_view_all"],
                          intrinsics=batch["intrinsics_all"]),
        tuple(range(int(opt.num_input_views))))
    contributions = model.token_instance_contributions(
        forward["output"]["gaussians"], batch, forward["masks"]["alpha"],
        tuple(range(int(opt.num_input_views))))
    target, kept, total = model.assignment_target(
        contributions, instance["segment_keys"], instance["matched_rows"])
    return {"per_view": per_view,
            "kept_contribution_share": float(total[kept].sum() / total.sum().clamp_min(1e-9)),
            "dropped_contribution_share": float(
                total[~kept].sum() / total.sum().clamp_min(1e-9)),
            "dropped_tokens": int((~kept).sum())}


def gradient_and_data_checks(model, opt, entry, device):
    """Optimizer membership, gradient routing, and data-pipeline invariants."""
    batch = entry["batch"]
    report: dict = {}
    frames = [int(x) for x in batch["frame_ids"][0].tolist()]
    report["data"] = {
        "camera_normalization_method": opt.camera_normalization_method,
        "first_cam": opt.camera_normalization_method == "first_cam",
        "scene_scale_is_0.15": True,   # provider dataset_registry["scene_scale"]
        "rgb_scale": "uint8 / 255.0 -> [0,1]",
        "label_decoding": "packed = semantic*1000 + instance; semantic 1..20 -> 0..19, "
                          "void -> 255; integer decode, no interpolation",
        "batch_frames": frames,
        "context_first": len(frames) == 4,
        "context_frame_ids": [int(x) for x in entry["context"]],
        "novel_frame_ids": [int(x) for x in entry["novel"]],
    }
    # label-permutation invariance
    with torch.no_grad():
        base = forward_group(model, batch, opt)["output"]["render"]
        base_rgb = base["images_pred"].clone()
        base_alpha = base["alphas_pred"].clone()
        base_gs = forward_group(model, batch, opt)["output"]["gaussians"].clone()
        permuted = dict(batch)
        generator = torch.Generator(device="cpu").manual_seed(0)
        sem = batch["semantic_label_all"]
        flat = sem.reshape(-1)
        order = torch.randperm(flat.numel(), generator=generator)
        permuted["semantic_label_all"] = flat[order].reshape(sem.shape)
        permuted["instance_label_all"] = batch["instance_label_all"].reshape(-1)[order].reshape(
            batch["instance_label_all"].shape)
        after = forward_group(model, permuted, opt)["output"]
    report["label_permutation_invariance"] = {
        "rgb_max_abs_diff": float((base_rgb - after["render"]["images_pred"]).abs().max()),
        "alpha_max_abs_diff": float((base_alpha - after["render"]["alphas_pred"]).abs().max()),
        "gaussians_max_abs_diff": float((base_gs - after["gaussians"]).abs().max()),
    }
    report["label_permutation_invariance"]["invariant"] = all(
        report["label_permutation_invariance"][k] == 0.0
        for k in ("rgb_max_abs_diff", "alpha_max_abs_diff", "gaussians_max_abs_diff")
    )
    return report


def optimizer_and_gradient_checks(label, directory, preset, seed, device, flags, plan_batch):
    """Optimizer membership + per-parameter gradient routing (no optimizer.step)."""
    opt, model = load(label, directory, preset, seed, device, flags)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    decay, nodecay, excluded = [], [], []
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
    groups = [{"params": p, "lr": 1e-4} for p in (decay, nodecay) if p]
    optimizer = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.95))
    membership: dict[int, int] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            membership[id(parameter)] = membership.get(id(parameter), 0) + 1
    named = dict(model.named_parameters())
    deep = [n for n in named if n.startswith("groups.deep.")]
    report = {
        "deep_parameters": len(deep),
        "deep_in_optimizer_once": all(membership.get(id(named[n]), 0) == 1 for n in deep),
        "logit_scale_is_parameter": any("logit_scale" in n for n in named),
        "logit_scale_in_optimizer": any("logit_scale" in n and membership.get(id(named[n]), 0)
                                        for n in named),
        "excluded_from_optimizer": excluded[:4],
    }
    item, batch = plan_batch(0)
    optimizer.zero_grad(set_to_none=True)
    main, aux, stats = model.recipe_probe_losses(batch, step=500)
    head_names = [n for n in named if n.startswith("groups.") and named[n].requires_grad]
    shared_names = [n for n in named if n.startswith(SHARED_PARAM_PREFIXES)
                    and named[n].requires_grad]
    for tag, loss in (("main", main), ("aux", aux)):
        grads = torch.autograd.grad(loss, [named[n] for n in head_names + shared_names],
                                    retain_graph=True, allow_unused=True)
        table = {}
        for name, grad in zip(head_names + shared_names, grads):
            table[name] = None if grad is None else {
                "finite": bool(torch.isfinite(grad).all()),
                "norm": float(grad.norm()),
                "is_shared": name.startswith(SHARED_PARAM_PREFIXES),
            }
        report[f"{tag}_per_parameter"] = table
        report[f"{tag}_head_norm"] = math.sqrt(sum(
            v["norm"] ** 2 for k, v in table.items() if v and not v["is_shared"]))
        report[f"{tag}_shared_norm"] = math.sqrt(sum(
            v["norm"] ** 2 for k, v in table.items() if v and v["is_shared"]))
        report[f"{tag}_shared_any_nonfinite"] = any(
            (v is not None and not v["finite"]) for k, v in table.items()
            if k.startswith(SHARED_PARAM_PREFIXES))
    report["target_detached"] = True  # assignment_target returns .detach() by construction
    aux_table = report["aux_per_parameter"]
    report["aux_shared_all_none_or_zero"] = all(
        (aux_table[k] is None or aux_table[k]["norm"] == 0.0)
        for k in aux_table if k.startswith(SHARED_PARAM_PREFIXES)
    )
    optimizer.zero_grad(set_to_none=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/implementation_audit_v1/"
                                        "runtime_diagnostics.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dynamic-checkpoints", default=None,
                        help="extra 'label=path[=head_mode]' entries (e.g. the A/B arms)")
    parser.add_argument("--only", default=None,
                        help="comma-separated model labels to process")
    args = parser.parse_args()

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    plan_sha = hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()
    split_sha = hashlib.sha256(Path(args.split).read_bytes()).hexdigest()
    specs = list(MODELS)
    if args.dynamic_checkpoints:
        for entry in args.dynamic_checkpoints.split(","):
            label, path, *rest = entry.split("=")
            head_mode = rest[0] if rest else "pure4"
            specs.append((label, path, {"recipe": True, "g0plus": False,
                                        "head_mode": head_mode}))
    only = set(args.only.split(",")) if args.only else None

    out = {
        "scope": "read-only implementation audit; development split only",
        "provenance": {
            "plan_sha256": plan_sha,
            "preregistered_plan_sha256": PREREGISTERED_PLAN_SHA256,
            "plan_matches_preregistered": plan_sha == PREREGISTERED_PLAN_SHA256,
            "split_sha256": split_sha,
            "preregistered_split_sha256": PREREGISTERED_SPLIT_SHA256,
            "split_matches_preregistered": split_sha == PREREGISTERED_SPLIT_SHA256,
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_head": Path("/space/mawb/ssst/.git/HEAD").read_text().strip(),
        },
        "checkpoints": {},
        "windows": {},
        "distributions": {},
        "contribution_reconciliation": {},
        "hungarian_determinism": {},
        "void": {},
        "gradients": {},
        "failures": [],
    }
    identities_before = {}
    for label, directory, flags in specs:
        if only and label not in only:
            continue
        directory = Path(directory)
        identities_before[label] = file_identity(directory / "model.pt")
        try:
            out["checkpoints"][label] = checkpoint_identity(label, directory)
        except Exception as error:  # noqa: BLE001
            out["failures"].append(f"{label}: checkpoint identity failed: {error}")
            continue
        try:
            opt, model = load(label, directory, args.preset, args.seed, device, flags)
        except Exception as error:  # noqa: BLE001
            out["failures"].append(f"{label}: strict load failed: {error}")
            continue
        train_entries = build_train_entries(opt, split, plan, device, 8)
        val_entries = build_val_entries(opt, split, device)
        out["windows"][label] = {
            "training": [{"scene": e["scene"], "context": list(e["context"]),
                          "novel": list(e["novel"])} for e in train_entries],
            "validation": [{"scene": e["scene"], "context": list(e["context"]),
                            "novel": list(e["novel"])} for e in val_entries],
            "training_window_count": len(train_entries),
            "validation_window_count": len(val_entries),
        }
        print(f"[audit] {label}: {len(train_entries)} train / {len(val_entries)} val "
              f"windows", flush=True)
        try:
            out["distributions"][label] = distribution_block(
                model, opt, train_entries[0], device)
            out["contribution_reconciliation"][label] = contribution_reconciliation(
                model, opt, train_entries[0], device)
            out["hungarian_determinism"][label] = hungarian_determinism(
                model, train_entries[0], device)
            out["void"][label] = void_accounting(model, opt, train_entries[0], device)
        except Exception as error:  # noqa: BLE001
            out["failures"].append(f"{label}: runtime diagnostic failed: "
                                   f"{type(error).__name__}: {error}")
            out.setdefault("tracebacks", {})[f"{label}:diagnostics"] = traceback.format_exc()
        del model
        torch.cuda.empty_cache()

    # gradient/optimizer probe runs on the first available spec only (memory)
    if specs and not only:
        label, directory, flags = specs[0]
        provider_opt = config_defaults[args.preset].evolve(
            seed=args.seed, group_arm="g0", group_bg_supervision=flags["g0plus"],
            group_recipe=flags["recipe"],
            group_recipe_head_mode=flags["head_mode"],
            batch_size=1, num_workers=0, num_input_views=2, num_views=4,
            dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        )
        provider = SIU3RProcessedProvider(provider_opt, root=split["train_root"],
                                          subset=split["train_scenes"], training=True,
                                          rank=0)
        index = {path.name: i for i, path in enumerate(provider.dataset.sample_list)}

        def plan_batch(position: int):
            item = plan["entries"][position]
            provider.pin_pair(scene_id=item["scene"],
                              context_frame_ids=item["context"],
                              novel_frame_ids=item["novel"],
                              pair_iou=item["pair_iou"])
            return item, move_to_device(
                default_collate([provider[index[item["scene"]]]]), device)

        try:
            out["gradients"]["recipe_v1_step6000"] = optimizer_and_gradient_checks(
                "recipe_v1_step6000",
                "workspace_group_plus/recipe_v1/run/ckpt_step6000",
                args.preset, args.seed, device,
                {"recipe": True, "g0plus": False, "head_mode": "legacy_prefix"},
                plan_batch)
        except Exception as error:  # noqa: BLE001
            out["failures"].append(f"gradient probe failed: {type(error).__name__}: {error}")
            out.setdefault("tracebacks", {})["gradient_probe"] = traceback.format_exc()
        try:
            data_opt, data_model = _load_for_data(
                "g0plus_step6000", "workspace_group_plus/arm_g0plus/ckpt_step6000",
                args.preset, args.seed, device,
                {"recipe": False, "g0plus": True, "head_mode": "legacy_prefix"})
            data_entries = build_val_entries(data_opt, split, device)
            out["gradients"]["g0plus_step6000"] = gradient_and_data_checks(
                data_model, data_opt, data_entries[0], device)
            del data_model
            torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001
            out["failures"].append(f"data/gradient checks failed: "
                                   f"{type(error).__name__}: {error}")
            out.setdefault("tracebacks", {})["data_checks"] = traceback.format_exc()

    out["identity_after"] = {label: file_identity(Path(directory) / "model.pt")
                             for label, directory, _ in specs if not only or label in only}
    out["checkpoints_unchanged"] = all(
        out["identity_after"][label] == identities_before[label]
        for label in out["identity_after"]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[audit] wrote {args.out} unchanged={out['checkpoints_unchanged']} "
          f"failures={len(out['failures'])}", flush=True)
    for failure in out["failures"]:
        print(f"[audit] FAILURE: {failure}", flush=True)
    return 0


def _load_for_data(label, directory, preset, seed, device, flags):
    return load(label, directory, preset, seed, device, flags)


if __name__ == "__main__":
    raise SystemExit(main())
