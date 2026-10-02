"""Locked data, transfer, optimizer, GC and utility runtime for Object-Locus V3-Set."""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)

from tokengs.data.provider import Provider
from tokengs.data.registry import dataset_registry
from tokengs.data.siu3r_processed import DATASET_NAME, SIU3RProcessedProvider
from tokengs.models import model_registry


PRETRAINED = Path("/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt")
PRETRAINED_SHA = "5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f"
PRETRAINED_STEP = 47500
ASSET_ROOT = Path("/space/mawb/ssst/group_plus/instance_state_v1_generalization")
REPORTS_DEFAULT = Path("/space/mawb/ssst/group_plus/object_locus_joint_v1")
RUN_ROOT_DEFAULT = Path("/space/mawb/ssst/workspace_group_plus/object_locus_joint_v1")
MANIFEST = ASSET_ROOT / "train128_windows1024.json"
PLAN = ASSET_ROOT / "plan_C_frozen_5000.json"
MANIFEST_SHA = "1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483"
PLAN_SHA = "ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323"
MONITOR_SHA = {
    "monitor_train16.json": "133811557d13dca81864b9843b831f89df9226a899c381bfb79e0131991a5961",
    "monitor_8pairs.json": "5dae7077779dd398ba97851f0259aef2df7de083df80f5a6bea4b5f5e731c321",
    "monitor_32pairs.json": "af51dfe52d8cf31140f028a5c3b1d1bc5402fd524805d241940f314fad87bd36",
}
MONITOR_FILES = (*MONITOR_SHA, "train128_class_coverage.json")
TOTAL_STEPS = 3584
WARMUP_STEPS = 200
OBJECT_PEAK_LR = 1e-4
RECON_PEAK_LR = 1e-6
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
GC_ALPHA = 0.01
SEED = 42


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")


def jsonable(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist() if x.ndim else x.detach().cpu().item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x


def move_to(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to(v, device) for v in value)
    return value


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def understanding_weight(step):
    return min(max(float(step), 0.0) / 200.0, 1.0)


def lr_multiplier(step):
    if step <= 0: return 0.0
    if step <= 200: return step / 200.0
    return 0.1 + 0.9 * (1 + math.cos(math.pi * (step - 200) / (3584 - 200))) / 2


def build_options():
    from tokengs.options import config_defaults
    return config_defaults["train_siu3r_object_locus_joint_v1"]


def load_checkpoint_state(path=PRETRAINED):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["model"] if isinstance(blob, dict) and "model" in blob else blob


def transfer_object_locus_reconstruction_weights(model, opt, source_state):
    from tokengs.models.canonical_recon_models import LocusGSRecon
    if sha256_file(PRETRAINED) != PRETRAINED_SHA:
        raise RuntimeError("pretrained reconstruction checkpoint SHA256 mismatch")
    canonical_opt = opt.evolve(model_type="siu3r_locusgs_recon", workspace="",
                               experiment_name="object_locus_transfer_reference")
    canonical = LocusGSRecon(canonical_opt)
    canonical.load_state_dict(source_state, strict=True)
    source = canonical.state_dict()
    target = model.state_dict()
    new_keys = {k for k in target if k.startswith("object_locus_v3_set.")}
    injection_keys = {k for k in target if k.startswith("object_locus_joint_injection.")}
    recon_keys = set(target) - new_keys - injection_keys
    if (len(recon_keys), len(new_keys), len(injection_keys)) != (450, 78, 4):
        raise RuntimeError("450/78/4 transfer contract violated")
    if any(torch.count_nonzero(target[k]) for k in injection_keys):
        raise RuntimeError("injection must initialize to zero")
    missing = sorted(recon_keys - set(source))
    unexpected = sorted(set(source) - recon_keys)
    shape = sorted(k for k in (recon_keys & set(source)) if target[k].shape != source[k].shape or target[k].dtype != source[k].dtype)
    if missing or unexpected or shape:
        raise RuntimeError(f"strict object-locus transfer failed: missing={missing[:10]}, unexpected={unexpected[:10]}, shape={shape[:10]}")
    with torch.no_grad():
        for key in recon_keys:
            target[key].copy_(source[key])
    return {
        "source_sha256": PRETRAINED_SHA,
        "pretrained_step": PRETRAINED_STEP,
        "matched_reconstruction_tensor_count": len(recon_keys),
        "new_object_locus_tensor_count": len(new_keys),
        "new_injection_tensor_count": len(injection_keys),
        "missing_non_object_keys": missing,
        "unexpected_source_keys": unexpected,
        "shape_mismatch_keys": shape,
    }


def build_model(device="cpu", *, opt=None, arm="control"):
    if arm not in ("control", "joint"): raise ValueError(arm)
    opt = build_options() if opt is None else opt
    seed_everything(SEED)
    model = model_registry["siu3r_object_locus_joint_v1"](opt)
    model.inject_enabled = arm == "joint"
    source = load_checkpoint_state()
    transfer = transfer_object_locus_reconstruction_weights(model, opt, source)
    model = model.to(device)
    model.float()
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"Object-Locus V3-Set must be fully trainable, frozen={frozen[:12]}")
    if model.reconstruction_only:
        raise RuntimeError("Object-Locus model incorrectly exposes reconstruction_only=True")
    return model, opt, transfer


def trainability_counts(model):
    return {
        "total_numel": sum(p.numel() for p in model.parameters()),
        "trainable_numel": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_numel": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_reconstruction_numel": sum(p.numel() for n, p in model.named_parameters()
                                               if p.requires_grad and not n.startswith(("object_locus_v3_set.", "object_locus_joint_injection."))),
        "trainable_object_locus_numel": sum(p.numel() for n, p in model.named_parameters()
                                             if p.requires_grad and n.startswith(("object_locus_v3_set.", "object_locus_joint_injection."))),
        "frozen_names": [n for n, p in model.named_parameters() if not p.requires_grad],
    }


def build_optimizer(model):
    buckets = {name: [] for name in ("object_decay", "object_nodecay",
                                      "reconstruction_decay", "reconstruction_nodecay")}
    ids = set()
    names = {}
    for name, p in model.named_parameters():
        if id(p) in ids:
            continue
        ids.add(id(p))
        names[id(p)] = name
        if not p.requires_grad:
            continue
        branch = "object" if name.startswith(("object_locus_v3_set.", "object_locus_joint_injection.")) else "reconstruction"
        no_decay = (p.ndim == 1 or name.endswith(".bias") or name.endswith("stuff_seed")
                    or bool(getattr(p, "_no_weight_decay", False)))
        buckets[f"{branch}_{'nodecay' if no_decay else 'decay'}"].append(p)
    groups = []
    for name, params in buckets.items():
        object_branch = name.startswith("object_")
        nodecay = name.endswith("nodecay")
        groups.append({"params": params, "name": name,
                       "lr": OBJECT_PEAK_LR if object_branch else RECON_PEAK_LR,
                       "weight_decay": 0.0 if nodecay else WEIGHT_DECAY})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8,
                                  amsgrad=False, foreach=False, fused=False)
    locations = {}
    for index, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            locations.setdefault(id(p), []).append(index)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    missing_ids = expected - set(locations)
    duplicate_ids = [ident for ident, groups_for_param in locations.items() if len(groups_for_param) != 1]
    if missing_ids or duplicate_ids or set(locations) != expected:
        raise RuntimeError("optimizer does not cover each unique trainable parameter exactly once")
    info = {"betas": [0.9, 0.95], "eps": 1e-8, "groups": [
        {"name": group["name"], "tensor_count": len(group["params"]),
         "numel": sum(p.numel() for p in group["params"]), "lr": group["lr"],
         "weight_decay": group["weight_decay"]} for group in optimizer.param_groups],
        "unique_trainable_tensors": len(expected), "missing": [], "duplicates": [],
        "all_trainable_once": True}
    return optimizer, info


def set_optimizer_lr(optimizer, step):
    multiplier = lr_multiplier(int(step))
    for group in optimizer.param_groups:
        peak = OBJECT_PEAK_LR if group["name"].startswith("object_") else RECON_PEAK_LR
        group["lr"] = peak * multiplier


def backward_gradient_controlled(model, loss_recon, loss_understanding, weight,
                                 shared_scale=GC_ALPHA):
    weight = float(weight)
    if shared_scale != GC_ALPHA:
        raise ValueError("Object-Locus V3-Set GC alpha is fixed at 0.01")
    if not (torch.isfinite(loss_recon).all() and torch.isfinite(loss_understanding).all()):
        raise FloatingPointError("nonfinite loss before GC backward")
    loss_recon.backward(retain_graph=True)
    if weight == 0.0:
        return {"registered_hook_count": 0, "removed_hook_count": 0,
                "understanding_backward_skipped": True}
    handles = []
    seen = set()
    try:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name.startswith(("object_locus_v3_set.", "object_locus_joint_injection.")) or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            handles.append(parameter.register_hook(lambda grad: grad * GC_ALPHA))
        (weight * loss_understanding).backward()
    finally:
        for handle in handles:
            handle.remove()
    return {"registered_hook_count": len(handles), "removed_hook_count": len(handles),
            "understanding_backward_skipped": False}


def _all_finite_tensors(value, path="root"):
    bad = []
    if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
        bad.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            bad.extend(_all_finite_tensors(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            bad.extend(_all_finite_tensors(child, f"{path}[{index}]"))
    return bad


def _failure_json_value(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _failure_json_value(value.detach().item())
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(k): _failure_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_failure_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "NaN" if math.isnan(float(value)) else ("+Inf" if float(value) > 0 else "-Inf")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _failure_tensor_stats(tensor):
    """Summarize a tensor with temporary finite masks but retain no tensor copy."""
    row = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
           "device": str(tensor.device), "numel": int(tensor.numel())}
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return row
    finite = torch.isfinite(tensor)
    row["nan_count"] = int(torch.isnan(tensor).sum().item())
    row["posinf_count"] = int(torch.isposinf(tensor).sum().item())
    row["neginf_count"] = int(torch.isneginf(tensor).sum().item())
    row["finite_count"] = int(finite.sum().item())
    bad = torch.nonzero(~finite, as_tuple=False)
    row["first_nonfinite_index"] = bad[0].detach().cpu().tolist() if bad.numel() else None
    if row["finite_count"]:
        values = tensor[finite]
        row["finite_min"] = float(values.min().item())
        row["finite_max"] = float(values.max().item())
        row["finite_mean"] = float(values.float().mean().item())
    return row


def _failure_prediction_stats(output):
    prediction = (output or {}).get("prediction", {}) if isinstance(output, dict) else {}
    result = {}
    for key in ("gaussians", "region_mass", "semantic_scores", "pixel_void_mass", "identity_render",
                "alpha", "p_class", "conditional_class_prob", "objectness_prob",
                "pooled_feature", "anchor_pool_mass", "gaussian_pool_mass",
                "gaussian_membership", "gaussian_mask_logits", "anchor_membership",
                "thing_class_logits"):
        value = prediction.get(key)
        if torch.is_tensor(value): result[key] = _failure_tensor_stats(value.detach())
    render = prediction.get("render", {})
    result["render"] = {k: _failure_tensor_stats(v.detach()) for k, v in render.items()
                        if torch.is_tensor(v)} if isinstance(render, dict) else {}
    states = []
    for state in prediction.get("states", []):
        layer = int(state.get("layer", -1))
        if layer not in (6, 8, 10, 12): continue
        selected = {k: _failure_tensor_stats(v.detach()) for k, v in state.items()
                    if torch.is_tensor(v) and k in (
                        "tokens", "mu", "radii", "anchor_embedding", "q", "c", "s", "ell",
                        "scene_origin", "evidence_logits", "R", "R_bar", "anchor_assignment",
                        "anchor_mask_logits", "anchor_membership", "thing_logits19",
                        "thing_class_logits", "category_logits18", "objectness_logits",
                        "conditional_class_prob", "objectness_prob", "u_anchor",
                        "anchor_pool_mass", "u_gaussian", "gaussian_pool_mass",
                        "c_displacement", "s_displacement", "joint_T", "joint_u", "joint_raw", "joint_delta")}
        states.append({"layer": layer, "tensors": selected})
    result["registered_states"] = states
    return result


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _write_step_failure(path, *, step, stage, error, batch, metrics=None, output=None,
                        model=None, optimizer=None, failure_context=None,
                        optimizer_updated=False):
    if path is None:
        return
    identity = {}
    for key in ("scene_name", "frame_ids"):
        if key in batch:
            value = batch[key]
            identity[key] = (value.detach().cpu().tolist() if key == "frame_ids" and torch.is_tensor(value)
                             else _failure_json_value(value))
    batch_stats = {}
    for key in ("images_all", "semantic_label_all", "instance_label_all", "depth_gt_m_all",
                "depth_gt_scene_all", "depth_gt_valid_all", "cam_view_all", "intrinsics_all"):
        value = batch.get(key)
        if torch.is_tensor(value): batch_stats[key] = _failure_tensor_stats(value.detach())
    scalar_metrics = {}
    for key, value in (metrics or {}).items():
        if torch.is_tensor(value) and value.ndim == 0:
            scalar_metrics[key] = _failure_json_value(value.detach().item())
        elif isinstance(value, (int, float, bool, str, np.generic)):
            scalar_metrics[key] = _failure_json_value(value)
    target = Path(path) / f"failure_step_{int(step):08d}.json"
    stack = traceback.format_exc()
    if stack.strip() == "NoneType: None":
        stack = "".join(traceback.format_stack()[:-1])
    payload = {"step": int(step), "stage": stage, "exception": repr(error),
               "traceback": stack, "batch_identity": identity,
               "batch_tensor_stats": batch_stats, "scalar_metrics": scalar_metrics,
               "prediction_tensor_stats": _failure_prediction_stats(output),
               "nonfinite_gradient_stats": {n: _failure_tensor_stats(p.grad.detach())
                   for n,p in model.named_parameters() if p.grad is not None and
                   not torch.isfinite(p.grad).all()} if model is not None else {}}
    write_json(target, payload)
    if model is not None:
        state = {
            "model": _cpu_tree(model.state_dict()),
            "optimizer": _cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
            "rng": capture_rng(), "pre_forward_rng": getattr(model, "_joint_pre_forward_rng", None),
            "batch": _cpu_tree(batch), "arm": "joint" if model.inject_enabled else "control", "step": int(step),
            "failure_context": _failure_json_value(failure_context or {}),
            "optimizer_updated": bool(optimizer_updated),
        }
        torch.save(state, target.with_name(target.stem + "_state.pt"))


def train_one_step(model, optimizer, batch, step, *, precomputed=None, gradient_audit=None,
                   failure_capture_dir=None, failure_context=None,
                   understanding_weight_value=None, lr_values=None):
    model.train()
    model.understanding_step = int(step)
    pre_forward_rng = capture_rng()
    model._joint_pre_forward_rng = pre_forward_rng
    if understanding_weight_value is None:
        understanding_weight_value = understanding_weight(step)
    if lr_values is None:
        set_optimizer_lr(optimizer, step)
    else:
        obj_lr, recon_lr = map(float, lr_values)
        for group in optimizer.param_groups:
            group["lr"] = obj_lr if group["name"].startswith("object_") else recon_lr
    optimizer.zero_grad(set_to_none=True)
    try:
        output, metrics = (model.step_loss(batch, step=step, phase="train", coupled=None,
                                           understanding_weight=understanding_weight_value)
                           if precomputed is None else precomputed)
    except Exception as error:
        _write_step_failure(failure_capture_dir, step=step, stage="step_loss_forward",
                            error=error, batch=batch, model=model, optimizer=optimizer,
                            failure_context=failure_context)
        raise
    prediction = output["prediction"]
    loss_recon = metrics["loss_recon"]
    loss_understanding = metrics["loss_understanding"]
    weight = float(metrics["understanding_weight"])
    for name, value in metrics.items():
        nonfinite = (torch.is_tensor(value) and value.is_floating_point()
                     and not torch.isfinite(value).all())
        if isinstance(value, (float, np.floating)):
            nonfinite = not math.isfinite(float(value))
        if nonfinite:
            error = FloatingPointError(f"nonfinite metric {name} at step {step}")
            _write_step_failure(failure_capture_dir, step=step, stage=f"metric_guard:{name}",
                                error=error, batch=batch, metrics=metrics, output=output,
                                model=model, optimizer=optimizer, failure_context=failure_context)
            raise error
    forward_bad = _all_finite_tensors({"prediction": prediction})
    if forward_bad:
        error = FloatingPointError(f"nonfinite forward tensors at step {step}: {forward_bad[:12]}")
        _write_step_failure(failure_capture_dir, step=step, stage="forward_tensor_guard",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context)
        raise error
    observation_handles = []
    upstream = {}
    for state in prediction['states']:
        delta = state.get('joint_delta')
        layer = state['layer']
        if delta is not None and delta.requires_grad and model.inject_enabled:
            def observe(grad, layer=layer):
                upstream[layer] = upstream.get(layer, 0.0) + float(grad.detach().norm())
            observation_handles.append(delta.register_hook(observe))
    try:
        hooks = backward_gradient_controlled(model, loss_recon, loss_understanding, weight)
    except Exception as error:
        _write_step_failure(failure_capture_dir, step=step, stage="backward",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context)
        raise
    finally:
        for handle in observation_handles: handle.remove()
    joint_diagnostics = injection_diagnostics(model, prediction, upstream)
    if hooks["registered_hook_count"] != hooks["removed_hook_count"]:
        error = RuntimeError("temporary GC hook cleanup mismatch")
        _write_step_failure(failure_capture_dir, step=step, stage="gc_hook_cleanup",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context)
        raise error
    grad_bad = [n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()]
    if grad_bad:
        error = FloatingPointError(f"nonfinite gradients at step {step}: {grad_bad[:12]}")
        _write_step_failure(failure_capture_dir, step=step, stage="gradient_guard",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context)
        raise error
    named_parameters = dict(model.named_parameters())
    watched = ("object_locus_v3_set.W_Q.weight", "object_locus_v3_set.class_head.weight",
               "object_locus_v3_set.cls_fuse.weight", "object_locus_v3_set.mask_q_mlp.2.weight",
               "object_locus_v3_set.child_mlp.2.weight", "object_locus_v3_set.W_c.weight",
               "object_locus_v3_set.W_s.weight", "anchor_decoder.mu",
               "activation_head.deconv.weight")
    gradient_report = {}
    for name in watched:
        parameter = named_parameters.get(name)
        if parameter is None:
            gradient_report[name] = {"present": False, "grad_norm": 0.0, "finite": False}
            continue
        grad = parameter.grad
        gradient_report[name] = {
            "present": grad is not None,
            "grad_norm": float(grad.detach().float().norm()) if grad is not None else 0.0,
            "max_abs": float(grad.detach().float().abs().max()) if grad is not None else 0.0,
            "finite": bool(torch.isfinite(grad).all()) if grad is not None else True,
            "nonzero": bool(grad is not None and torch.count_nonzero(grad).item() > 0),
        }
    if gradient_audit is not None:
        for name, parameter in named_parameters.items():
            if not name.startswith(("object_locus_v3_set.", "object_locus_joint_injection.")):
                continue
            grad = parameter.grad
            gradient_audit[name] = {
                "present": grad is not None,
                "norm": float(grad.detach().float().norm()) if grad is not None else 0.0,
                "finite": bool(torch.isfinite(grad).all()) if grad is not None else True,
                "nonzero": bool(grad is not None and torch.count_nonzero(grad).item() > 0),
            }
    try:
        preclip = clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
    except Exception as error:
        _write_step_failure(failure_capture_dir, step=step, stage="gradient_clip",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context)
        raise
    preclip_value = float(preclip.detach())
    clip_coefficient = min(1.0, GRAD_CLIP / (preclip_value + 1e-6))
    try:
        optimizer.step()
    except Exception as error:
        _write_step_failure(failure_capture_dir, step=step, stage="optimizer_step",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context,
                            optimizer_updated=False)
        raise
    param_bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if param_bad:
        error = FloatingPointError(f"nonfinite model parameters after step {step}: {param_bad[:12]}")
        _write_step_failure(failure_capture_dir, step=step, stage="post_step_parameter_guard",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context,
                            optimizer_updated=True)
        raise error
    state_bad = []
    for index, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                state_bad.append(key)
    if state_bad:
        error = FloatingPointError(f"nonfinite optimizer state at step {step}")
        _write_step_failure(failure_capture_dir, step=step, stage="post_step_optimizer_state_guard",
                            error=error, batch=batch, metrics=metrics, output=output,
                            model=model, optimizer=optimizer, failure_context=failure_context,
                            optimizer_updated=True)
        raise error
    group_lr = next(g["lr"] for g in optimizer.param_groups if g["name"].startswith("object_"))
    recon_lr = next(g["lr"] for g in optimizer.param_groups if g["name"].startswith("reconstruction_"))
    detached = {k: (float(v.detach()) if torch.is_tensor(v) and v.ndim == 0 else v)
                for k, v in metrics.items() if k not in ("loss_recon", "loss_understanding", "loss", "loss_total")}
    detached.update({"step": int(step), "loss": float(metrics["loss"].detach()),
                     "loss_total": float(metrics["loss_total"].detach()),
                     "loss_recon": float(loss_recon.detach()),
                     "loss_understanding": float(loss_understanding.detach()),
                     "understanding_weight": weight, "object_locus_lr": group_lr,
                     "reconstruction_lr": recon_lr, "shared_understanding_grad_scale": GC_ALPHA,
                     "pre_clip_global_grad_norm": preclip_value,
                     "clip_coefficient": clip_coefficient,
                     "clip_applied": clip_coefficient < 1.0,
                     "gc_registered_hooks": hooks["registered_hook_count"],
                     "gc_removed_hooks": hooks["removed_hook_count"]})
    detached["gradient_report_before_clip"] = gradient_report
    detached["injection"] = joint_diagnostics
    detached["beta"] = prediction["beta"]
    if torch.cuda.is_available():
        detached["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
        detached["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
    return output, detached



# The provider and all frame/crop/depth behavior remain exactly the V3 implementation.
from scripts.object_locus_v3_set_runtime import build_batch


def injection_diagnostics(model, prediction, upstream=None):
    rows = {}
    for state in prediction['states']:
        if 'joint_delta' not in state: continue
        layer = state['layer']
        w = model.object_locus_joint_injection[f'L{layer}'].weight
        delta = state['joint_delta'].detach()
        relative = delta.norm(dim=-1) / (state['joint_h_norm'] + 1e-6)
        route = state['joint_T'].detach()
        row = {'relative_delta_mean': float(relative.mean()), 'relative_delta_max': float(relative.max()),
               'direct_delta_mu_over_ell': 0.0, 'direct_delta_rho': 0.0, 'direct_delta_radius': 0.0,
               'T_row_sum_min': float(route.sum(-1).min()), 'T_row_sum_max': float(route.sum(-1).max()),
               'T_entropy_mean': float(-(route * route.clamp_min(1e-6).log()).sum(-1).mean()),
               'thing_route_mass': float(route[..., :100].sum(-1).mean()),
               'stuff_route_mass': float(route[..., 100:].sum(-1).mean()),
               'W_norm': float(w.detach().norm()),
               'W_gradient_norm': float(w.grad.detach().norm()) if w.grad is not None else None,
               'W_gradient_status': 'PRESENT' if w.grad is not None else 'NONE_expected',
               'delta_upstream_gradient_norm': (upstream or {}).get(layer, 0.0)}
        for key in ('joint_u', 'joint_raw', 'joint_delta'):
            norm = state[key].detach().norm(dim=-1)
            row[key + '_norm_mean'] = float(norm.mean()); row[key + '_norm_max'] = float(norm.max())
        for key in ('c', 's'):
            value = state[key].detach()
            row[key + '_finite'] = bool(torch.isfinite(value).all())
            row[key + '_min'] = float(value.min()); row[key + '_max'] = float(value.max())
        rows[f'L{layer}'] = row
    return rows

SOURCE_MANIFEST = Path('/space/mawb/ssst/group_plus/object_locus_v3_set/data_manifest.json')
SOURCE_MANIFEST_SHA = 'c6c1a0dbfb5c88745a9f633c93bd0bb513a946717ceca34a41ad5802cc3f9b35'
SPLITS = ('train_all56', 'same_scene_holdout8', 'dev8', 'val32')
EVAL_EPOCHS = (0, 8, 16, 32, 64)


def build_manifest():
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA:
        raise RuntimeError('locked V3 manifest SHA mismatch')
    return json.loads(SOURCE_MANIFEST.read_text())


def build_plan(manifest):
    from scripts.train_object_locus_v3_set import build_plan as original
    return original(manifest)


def scientific_state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


IMPLEMENTATION_FILES = (
    'tokengs/models/object_locus_joint_v1.py',
    'scripts/object_locus_joint_v1_runtime.py',
    'scripts/train_object_locus_joint_v1.py',
    'scripts/eval_object_locus_joint_v1.py',
    'scripts/smoke_object_locus_joint_v1.py',
    'scripts/submit_object_locus_joint_v1.sh',
    'scripts/report_object_locus_joint_v1.py',
    'scripts/replay_object_locus_joint_v1_failure.py',
    'tests/test_object_locus_joint_v1_contracts.py',
    'tokengs/rendering/gs.py', 'tokengs/models/canonical_recon.py',
    'tokengs/models/__init__.py', 'tokengs/options.py')


def implementation_hashes():
    return {name: sha256_file(REPO / name) for name in (*IMPLEMENTATION_FILES[:-2], 'tokengs/rendering/gs.py', 'tokengs/models/canonical_recon.py')}


def assert_joint_gpu():
    import os
    if not os.uname().nodename.startswith('3dimage-11'):
        raise RuntimeError('Joint V1 requires node 3dimage-11')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Joint V1 requires exactly one visible CUDA GPU')
    if torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3090':
        raise RuntimeError('Joint V1 requires RTX3090')
    if torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
        raise RuntimeError('Joint V1 requires 24GB class memory')
    extension = Path(sys.modules['gsplat_cuda'].__file__)
    return {'node': os.uname().nodename, 'gpu': torch.cuda.get_device_name(0),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gsplat_extension': str(extension), 'gsplat_sha256': sha256_file(extension),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}
