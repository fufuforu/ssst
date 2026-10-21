#!/usr/bin/env python3
"""Isolated, provenance-locked replay to capture Object-Locus V1 step-4090 failure."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v1_runtime import (
    MANIFEST, MANIFEST_SHA, PLAN, PLAN_SHA, PRETRAINED_SHA,
    REPORTS_DEFAULT, RUN_ROOT_DEFAULT, build_batch, build_model, build_optimizer,
    capture_rng, restore_rng, sha256_file, train_one_step,
)

OUT = Path("/space/mawb/ssst/group_plus/object_locus_v1_failure_analysis")
FAIL_STEP = 4090
SOURCE_COMMIT = "61b22322e79e973cdf7ff868afb6242f4253f3bb"
BASELINE_COMMIT = "85cad0d163eafc0e197a79bb374d39f5cfed3814"
ALLOWED_CONTROLLER_OBSERVATION_DIFF_SHA = "46f0eb35d92f85c7e258cdab7ce92c947ead02078fd822075df2d9f9f196421b"
CHECKPOINT = RUN_ROOT_DEFAULT / "checkpoints/step_00003500/train_state.pt"
REPLAY_LOG = OUT / "replay_log.jsonl"


def append_replay_row(record):
    with REPLAY_LOG.open("a") as f:
        f.write(json.dumps(json_safe(record), allow_nan=False) + "\n")


def json_safe(x):
    if torch.is_tensor(x):
        if x.ndim == 0:
            return json_safe(x.detach().cpu().item())
        return {"tensor_shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    if isinstance(x, dict): return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return json_safe(x.tolist())
    if isinstance(x, (np.integer, np.floating, np.bool_)): return json_safe(x.item())
    if isinstance(x, float) and not math.isfinite(x): return "NaN" if math.isnan(x) else ("+Inf" if x > 0 else "-Inf")
    if isinstance(x, (str, int, float, bool)) or x is None: return x
    return repr(x)


def tensor_stats(value):
    if not torch.is_tensor(value): return {"value": json_safe(value)}
    t = value.detach()
    row = {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device), "numel": int(t.numel())}
    if t.is_floating_point() or t.is_complex():
        finite = torch.isfinite(t)
        row["nan_count"] = int(torch.isnan(t).sum().item())
        row["posinf_count"] = int(torch.isposinf(t).sum().item())
        row["neginf_count"] = int(torch.isneginf(t).sum().item())
        row["finite_count"] = int(finite.sum().item())
        first = torch.nonzero(~finite, as_tuple=False)
        row["first_nonfinite_index"] = first[0].detach().cpu().tolist() if first.numel() else None
        if row["finite_count"]:
            vals = t[finite]
            row.update({"finite_min": json_safe(vals.min().item()),
                        "finite_max": json_safe(vals.max().item()),
                        "finite_mean": json_safe(vals.float().mean().item())})
        else:
            row.update({"finite_min": None, "finite_max": None, "finite_mean": None})
    elif t.dtype == torch.bool:
        row["true_count"] = int(t.sum().item())
        row["false_count"] = int(t.numel() - row["true_count"])
    return row


def state_stats(state):
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value): out[key] = tensor_stats(value)
        elif isinstance(value, (float, int, np.number)): out[key] = {"value": json_safe(value)}
    return out


def simplex_sum_stats(tensor, dim):
    if not torch.is_tensor(tensor): return None
    sums = tensor.detach().sum(dim=dim)
    return {"sum": tensor_stats(sums), "abs_error_from_one": tensor_stats((sums-1.0).abs()),
            "max_abs_error_from_one": float((sums-1.0).abs().max().item()) if sums.numel() else None}


def tree_tensor_stats(value, path="root"):
    if torch.is_tensor(value): return tensor_stats(value)
    if isinstance(value, dict): return {str(k):tree_tensor_stats(v,f"{path}.{k}") for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [tree_tensor_stats(v,f"{path}[{i}]") for i,v in enumerate(value)]
    return None


def registered_layer_stats(states):
    result={}
    for state in states:
        layer=int(state.get("layer",-1))
        if layer not in (6,8,10,12): continue
        row=state_stats(state)
        if torch.is_tensor(state.get("evidence_attention")):
            row["evidence_attention_anchor_sum"] = simplex_sum_stats(state["evidence_attention"],-1)
        if torch.is_tensor(state.get("evidence_attention_mean")):
            row["evidence_attention_mean_anchor_sum"] = simplex_sum_stats(state["evidence_attention_mean"],-1)
        if torch.is_tensor(state.get("anchor_assignment")):
            row["anchor_assignment_channel_sum"] = simplex_sum_stats(state["anchor_assignment"],-1)
        if torch.is_tensor(state.get("c_displacement")):
            row["c_displacement_norm"] = tensor_stats(torch.linalg.vector_norm(state["c_displacement"].detach(),dim=-1))
        result[str(layer)]=row
    return result


def rng_equal(a, b):
    if a["python"] != b["python"]: return False
    na, nb = a["numpy"], b["numpy"]
    if na[0] != nb[0] or na[2:] != nb[2:] or not np.array_equal(na[1], nb[1]): return False
    if not torch.equal(a["torch"], b["torch"]): return False
    if a["cuda"] is None or b["cuda"] is None: return a["cuda"] is b["cuda"]
    return len(a["cuda"]) == len(b["cuda"]) and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))


def source_hashes(payload):
    expected = payload["source_file_hashes"]
    current = {name: sha256_file(REPO / name) for name in expected}
    mismatches = {k: {"checkpoint": v, "current": current.get(k)} for k, v in expected.items()
                  if current.get(k) != v}
    permitted_current_differences = {"tokengs/models/object_locus_v1_controller.py",
                                     "docs/object_locus_v1_codex_spec.md"}
    if not set(mismatches).issubset(permitted_current_differences) or \
            "tokengs/models/object_locus_v1_controller.py" not in mismatches:
        raise RuntimeError(f"unexpected calculation source hash drift: {mismatches}")
    # Verify pre-instrumentation source from the recorded implementation commit exactly.
    for name, digest in expected.items():
        content = subprocess.check_output(["git", "-C", str(REPO), "show", f"{SOURCE_COMMIT}:{name}"])
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"checkpoint source hash does not match implementation commit for {name}")
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff", SOURCE_COMMIT, "--",
                                    "tokengs/models/object_locus_v1_controller.py"], text=True)
    if hashlib.sha256(diff.encode()).hexdigest() != ALLOWED_CONTROLLER_OBSERVATION_DIFF_SHA:
        raise RuntimeError("controller diagnostic instrumentation differs from reviewed observation-only patch")
    doc_diff = subprocess.check_output(["git", "-C", str(REPO), "diff", SOURCE_COMMIT, "--",
                                       "docs/object_locus_v1_codex_spec.md"], text=True)
    return {"checkpoint_source_hashes": expected, "current_source_hashes": current,
            "only_current_hash_difference": list(mismatches),
            "baseline_source_verified_from_git_commit": SOURCE_COMMIT,
            "non_calculation_specification_diff_since_source_commit": doc_diff,
            "controller_diff_classification": "opt-in detached-tensor observation callback only; original forward formulas and control flow retained when callback is None",
            "controller_diff": diff}


def summarize_batch(batch):
    keys = ("images_all", "semantic_label_all", "instance_label_all", "depth_gt_m_all",
            "depth_gt_scene_all", "depth_gt_valid_all", "cam_view_all", "intrinsics_all", "frame_ids")
    result = {key: tensor_stats(batch[key]) if key in batch else "MISSING" for key in keys}
    sem, inst = batch.get("semantic_label_all"), batch.get("instance_label_all")
    if sem is not None:
        result["semantic_unique"] = torch.unique(sem.detach().cpu()).tolist()
        result["semantic_valid_pixel_count"] = int(((sem >= 0) & (sem <= 19)).sum().item())
    if inst is not None:
        areas = []
        frame_ids = batch.get("frame_ids")
        for b in range(inst.shape[0]):
            for v in range(inst.shape[1]):
                ids, counts = torch.unique(inst[b, v][inst[b, v] > 0], return_counts=True)
                frame = int(frame_ids[b, v]) if frame_ids is not None else v
                areas.append({"batch": b, "view_index": v, "frame_id": frame,
                              "instances": [{"instance_id": int(i), "pixel_area": int(c)}
                                            for i, c in zip(ids.cpu(), counts.cpu())]})
        result["instance_positive_pixel_count"] = int((inst > 0).sum().item())
        result["instance_areas_by_frame"] = areas
    return result


class Capture:
    def __init__(self):
        self.layer = None
        self.layer_cursor = 0
        self.controller = {}
        self.canonical = []
        self.match_costs = []

    def observe(self, stage, values):
        if stage == "registered_layer_input":
            self.layer = (6, 8, 10, 12)[self.layer_cursor]
            self.layer_cursor += 1
        layer = self.layer if self.layer is not None else "unknown"
        self.controller.setdefault(str(layer), {}).setdefault(stage, {}).update(values)

    def serialized_controller(self):
        return {layer: {stage: state_stats(values) for stage, values in stages.items()}
                for layer, stages in self.controller.items()}


def patch_observers(current_capture):
    from tokengs.models import canonical_recon_models as recon_module
    from tokengs.models import canonical_recon as canonical_module
    from tokengs.models import object_locus_v1_loss as object_loss
    original_layer = recon_module.canonical_layer_loss
    original_compute = canonical_module.compute_tokengs_loss
    original_assignment = object_loss.linear_sum_assignment

    def compute_wrapper(*args, **kwargs):
        out = original_compute(*args, **kwargs)
        capture = current_capture()
        capture.canonical[-1]["compute_tokengs_loss"] = {
            k: tensor_stats(v) for k, v in out.items() if torch.is_tensor(v)
        }
        # Exact detached diagnostic decomposition of the production visibility expression.
        render = capture.canonical[-1].get("render_tensors", {})
        means = render.get("means2d_pred")
        if means is not None:
            capture.canonical[-1]["means2d_visibility_input"] = tensor_stats(means)
        return out

    def layer_wrapper(*args, **kwargs):
        capture = current_capture()
        layer_index = len(capture.canonical)
        row = {"layer": (6, 12)[layer_index] if layer_index < 2 else "extra"}
        gaussians = kwargs.get("gaussians")
        render = kwargs.get("render_results")
        row["gaussians"] = gaussians
        row["render_tensors"] = render
        row["gaussian_stats"] = tensor_stats(gaussians) if torch.is_tensor(gaussians) else None
        row["render_stats"] = {k: tensor_stats(v) for k, v in render.items() if torch.is_tensor(v)} if isinstance(render, dict) else {}
        capture.canonical.append(row)
        result = original_layer(*args, **kwargs)
        row["loss_stats"] = {k: tensor_stats(v) for k, v in result.items() if torch.is_tensor(v)}
        # visibility expression inputs/intermediates, recomputed on detached data only for diagnostics.
        means = render.get("means2d_pred") if isinstance(render, dict) else None
        if means is not None:
            height, width = (int(x) for x in kwargs["img_size"])
            uv = torch.stack([means.detach()[..., 0] / width * 2 - 1,
                              means.detach()[..., 1] / height * 2 - 1], dim=-1)
            oob = torch.relu(uv.abs() - 1.0).sum(-1)
            clip = float(getattr(kwargs["opt"], "visibility_distance_threshold", 0.0))
            clipped = oob.clamp(max=clip) if clip > 0 else oob
            minimum = clipped.min(dim=1).values
            row["gaussian_visibility_recomputed_detached"] = {
                "means2d": tensor_stats(means), "normalized_uv": tensor_stats(uv),
                "per_view_point_oob": tensor_stats(oob), "post_clip": tensor_stats(clipped),
                "min_over_views": tensor_stats(minimum), "mean": tensor_stats(minimum.mean()),
                "clip_threshold": clip}
        capture.canonical[-1] = row
        return result

    def assignment_wrapper(cost):
        capture = current_capture()
        capture.match_costs.append(tensor_stats(torch.as_tensor(cost)))
        return original_assignment(cost)

    recon_module.canonical_layer_loss = layer_wrapper
    canonical_module.compute_tokengs_loss = compute_wrapper
    object_loss.linear_sum_assignment = assignment_wrapper


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n")


def optimizer_finite(opt_state):
    bad = []
    for i, st in opt_state.get("state", {}).items():
        for k, v in st.items():
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all(): bad.append(f"{i}.{k}")
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-pre-step", default=None,
                        help="continue the same isolated replay from its already-saved exact pre-step state")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("isolated replay requires an available CUDA device")
    initial_checkpoint = Path(args.resume_pre_step) if args.resume_pre_step else CHECKPOINT
    payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
    if args.resume_pre_step:
        start_step = int(payload.get("target_step", -1))
        if start_step <= 3500 or int(payload.get("step", -1)) != start_step-1 or \
                int(payload.get("plan_position", -1)) != start_step-1:
            raise RuntimeError("diagnostic resume payload is not a complete pre-step state")
    else:
        if int(payload.get("step", -1)) != 3500 or int(payload.get("plan_position", -1)) != 3500:
            raise RuntimeError("latest complete replay checkpoint is not exactly step/position 3500")
        start_step = 3501
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    if sha256_file(MANIFEST) != MANIFEST_SHA or sha256_file(PLAN) != PLAN_SHA:
        raise RuntimeError("locked manifest/plan SHA mismatch")
    if sha256_file(Path("/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt")) != PRETRAINED_SHA:
        raise RuntimeError("locked pretrained SHA mismatch")
    src = source_hashes(payload)
    target = plan["entries"][FAIL_STEP-1]
    fixed = {"step": 4090, "scene": "scene0014_00", "window_index": 236,
             "context": [2055, 2067], "novel": [2057, 2061]}
    if any(target.get(k) != v for k, v in fixed.items()):
        raise RuntimeError(f"locked step4090 plan mismatch: {target}")
    if payload.get("git_commit") != SOURCE_COMMIT:
        raise RuntimeError(f"checkpoint git commit differs from source commit: {payload.get('git_commit')}")
    if payload.get("source_file_hashes") != json.loads((RUN_ROOT_DEFAULT / "run_manifest.json").read_text())["source_file_hashes"]:
        raise RuntimeError("checkpoint/run manifest source hashes differ")
    if payload.get("manifest_sha256") != MANIFEST_SHA or payload.get("plan_sha256") != PLAN_SHA or payload.get("pretrained_sha256") != PRETRAINED_SHA:
        raise RuntimeError("checkpoint locked data provenance mismatch")

    opt_cfg = __import__("scripts.object_locus_v1_runtime", fromlist=["build_options"]).build_options()
    model, opt, transfer = build_model(args.device, opt=opt_cfg)
    optimizer, optimizer_info = build_optimizer(model)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    model.train()
    if len(optimizer.state) != len(payload["optimizer"].get("state", {})):
        raise RuntimeError("optimizer moment state was not completely restored")
    if optimizer_finite(payload["optimizer"]): raise RuntimeError("checkpoint optimizer contains nonfinite state")
    ctrl = model.object_locus
    provenance = {
        "status": "replay running", "execution_worktree": str(REPO),
        "original_worktree": "/space/mawb/ssst", "head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "origin_main": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "origin/main"], text=True).strip(),
        "baseline_failure_commit": BASELINE_COMMIT, "implementation_commit": SOURCE_COMMIT,
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "diagnostic_resume_checkpoint": str(initial_checkpoint) if args.resume_pre_step else None,
        "formal_checkpoint_step": 3500, "replay_restore_step": payload["step"],
        "checkpoint_plan_position": payload["plan_position"],
        "checkpoint_git_commit": payload["git_commit"], "checkpoint_config": payload["config"],
        "optimizer_state_count": len(payload["optimizer"]["state"]),
        "optimizer_groups": optimizer_info["groups"], "optimizer_state_finite": True,
        "checkpoint_rng_keys": sorted(payload["rng"]), "restored_rng": True,
        "plan_step4090": target, "fixed_window_match": True,
        "manifest_sha256": sha256_file(MANIFEST), "plan_sha256": sha256_file(PLAN),
        "pretrained_sha256": sha256_file(Path("/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt")),
        "source_verification": src, "transfer_audit": transfer,
        "gpu": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_device_properties": {"name": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "backend": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "allow_fp16_reduced_precision_reduction": getattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", None)},
        "restored_from_intermediate_diagnostic_snapshot": bool(args.resume_pre_step),
        "isolation": True, "formal_run_paths_written": False,
        "step_range_intended": "3501..4090, halt at first nonfinite; no automatic continuation"}
    write_json(OUT / "replay_provenance.json", provenance)
    # Runtime callbacks are inactive by default and only retain detached values.
    collector = Capture()
    ctrl._failure_diagnostic_callback = lambda stage, values: collector.observe(
        stage, {k: (v.detach() if torch.is_tensor(v) else v) for k, v in values.items()})
    patch_observers(lambda: collector)

    for step in range(start_step, FAIL_STEP + 1):
        entry = plan["entries"][step-1]
        rng_pre_step = capture_rng()
        pre_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        if step == FAIL_STEP:
            # The saved state is pre-batch: restoring it replays the same provider calls/RNG.
            rng_pre = rng_pre_step
            pre = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": step-1, "plan_position": step-1, "target_step": step,
                   "rng": rng_pre, "git_commit": payload["git_commit"],
                   "source_file_hashes": payload["source_file_hashes"],
                   "manifest_sha256": MANIFEST_SHA, "plan_sha256": PLAN_SHA,
                   "pretrained_sha256": PRETRAINED_SHA, "config": payload["config"],
                   "optimizer_groups": payload["optimizer_groups"], "architecture_name": payload["architecture_name"]}
            path = OUT / f"replay_pre_step_{step:05d}.pt"
            torch.save(pre, path)
            if not rng_equal(rng_pre, capture_rng()):
                raise RuntimeError("saving pre-step checkpoint changed RNG state")
            del pre
            print(f"[saved-pre-step] {step} {path}", flush=True)
        collector = Capture()
        ctrl._failure_diagnostic_callback = lambda stage, values: collector.observe(
            stage, {k: (v.detach() if torch.is_tensor(v) else v) for k, v in values.items()})
        t0 = time.time()
        record = None
        nonfinite_metrics = []
        try:
            batch = build_batch(opt, entry, args.device)
            model.train()
            output, metrics = model.step_loss(batch, step=step, phase="train", coupled=False)
            # Capture detached evidence before the production scalar guard executes.
            scalar_metrics = {k: (v.detach() if torch.is_tensor(v) else v)
                              for k, v in metrics.items()}
            nonfinite_metrics = [k for k, v in metrics.items() if torch.is_tensor(v) and v.numel()
                                 and v.is_floating_point() and not torch.isfinite(v).all()]
            record = {"step": step, "entry": entry, "batch": summarize_batch(batch),
                      "metrics": json_safe(scalar_metrics), "nonfinite_metric_names": nonfinite_metrics,
                      "controller": collector.serialized_controller(),
                      "registered_state_stats": registered_layer_stats(output["prediction"]["states"]),
                      "canonical_layers": [{k: v for k, v in layer.items() if k not in ("gaussians", "render_tensors")}
                                           for layer in collector.canonical],
                      "hungarian_cost_stats": collector.match_costs,
                      "final_pairs": [[q.detach().cpu().tolist(), k.detach().cpu().tolist()]
                                      for q,k in output["prediction"].get("final_pairs", [])],
                      "final_targets": tree_tensor_stats(output["prediction"].get("final_targets", {})),
                      "elapsed_forward_seconds": time.time()-t0}
            if nonfinite_metrics:
                append_replay_row(record)
                state_stats_failure = {
                    "step": step, "entry": entry, "batch": record["batch"],
                    "metrics": record["metrics"], "nonfinite_metric_names": nonfinite_metrics,
                    "controller": record["controller"], "registered_state_stats": record["registered_state_stats"],
                    "canonical_layers": record["canonical_layers"], "hungarian_cost_stats": collector.match_costs,
                    "final_pairs": record["final_pairs"], "final_targets": record["final_targets"],
                    "prediction_tensor_stats": tree_tensor_stats(output["prediction"]),
                    "first_nonfinite_operation": None,
                    "classification": "aggregate scalar guard preempts prediction/gradient checks; component evidence captured before throw"}
                write_json(OUT / "failure_tensor_stats.json", state_stats_failure)
                (OUT / "failure_context.json").write_text(json.dumps(json_safe({
                    "step": step, "entry": entry, "exception_expected_at_existing_scalar_guard": True,
                    "already_computed": sorted(metrics.keys()),
                    "not_executed": ["prediction tensor finite guard", "backward", "gradient finite guard", "gradient clipping", "optimizer.step"],
                    "all_loss_components": record["metrics"],
                    "source": "scripts/object_locus_v1_runtime.py:312"}), indent=2, allow_nan=False)+"\n")
            # Run the unmodified production finite guards/backward/update on this exact graph.
            result = train_one_step(model, optimizer, batch, step, precomputed=(output, metrics))
            record["production_train_one_step_metrics"] = json_safe(result[1])
            append_replay_row(record)
            del result, output, metrics, batch, record, scalar_metrics
            if step % 100 == 0:
                print(f"[replay-step] {step} elapsed={time.time()-t0:.3f}s", flush=True)
        except Exception as exc:
            # The exception is not suppressed: save evidence and re-raise to stop immediately.
            if step < FAIL_STEP:
                # Forward/backward has not reached optimizer.step on failure. Restore any
                # mutable buffers to the pre-step snapshot and preserve pre-batch RNG.
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        if name in pre_buffers:
                            value.copy_(pre_buffers[name])
                restore_rng(rng_pre_step)
                early_pre = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "step": step-1, "plan_position": step-1, "target_step": step,
                             "rng": rng_pre_step, "git_commit": payload["git_commit"],
                             "source_file_hashes": payload["source_file_hashes"],
                             "manifest_sha256": MANIFEST_SHA, "plan_sha256": PLAN_SHA,
                             "pretrained_sha256": PRETRAINED_SHA, "config": payload["config"],
                             "optimizer_groups": payload["optimizer_groups"],
                             "architecture_name": payload["architecture_name"]}
                early_path = OUT / f"replay_pre_step_{step:05d}.pt"
                torch.save(early_pre, early_path)
                if not rng_equal(rng_pre_step, capture_rng()):
                    raise RuntimeError("saving early-failure pre-step checkpoint changed RNG state")
                failure_pre_path = early_path
            else:
                failure_pre_path = OUT / f"replay_pre_step_{step:05d}.pt"
            failure = {"step": step, "entry": entry, "exception_type": type(exc).__name__,
                       "exception": str(exc), "traceback": traceback.format_exc(),
                       "phase": "forward_or_production_guard", "optimizer_step_completed": False,
                       "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                       "gpu_reserved_bytes": torch.cuda.memory_reserved()}
            if "record" in locals() and isinstance(record, dict) and record.get("step") == step and not nonfinite_metrics:
                record["exception"] = {"type":type(exc).__name__,"message":str(exc)}
                append_replay_row(record)
            write_json(OUT / "replay_failure.json", failure)
            if not (OUT / "failure_tensor_stats.json").exists():
                write_json(OUT / "failure_tensor_stats.json", {"step": step, "entry": entry,
                    "batch_stats": summarize_batch(locals().get("batch", {})),
                    "controller": collector.serialized_controller(),
                    "canonical_layers": [{k: v for k,v in layer.items() if k not in ("gaussians", "render_tensors")}
                                         for layer in collector.canonical],
                    "first_nonfinite_operation": "not localized by aggregate output; see captured stage values",
                    "nonfinite_inputs_or_outputs": True})
            try:
                initial_context = json.loads((OUT / "failure_context_initial.json").read_text())
            except FileNotFoundError:
                initial_context = {}
            evidence = json.loads((OUT / "failure_tensor_stats.json").read_text())
            write_json(OUT / "failure_context.json", {
                "formal_run_snapshot": initial_context,
                "isolated_replay_failure": failure,
                "captured_failure_evidence": {
                    "step": step, "plan_entry": entry,
                    "loss_components": evidence.get("metrics"),
                    "nonfinite_metric_names": evidence.get("nonfinite_metric_names"),
                    "saved_pre_step_checkpoint": str(failure_pre_path),
                    "optimizer_step_completed": False,
                },
            })
            provenance["status"] = "failed as captured" if "nonfinite" in str(exc).lower() else "stopped on replay exception"
            provenance["first_failure_step"] = step
            provenance["failure_exception"] = f"{type(exc).__name__}: {exc}"
            provenance["saved_pre_step_checkpoint"] = str(failure_pre_path)
            provenance["restoration_completeness"] = f"complete from formal step3500 checkpoint, then exact isolated pre-step state at step {step} with model, optimizer moments, Python/NumPy/CPU/CUDA RNG, plan_position and config; no scheduler existed in checkpoint/driver"
            write_json(OUT / "replay_provenance.json", provenance)
            raise
    provenance["status"] = "not_reproduced through step4090"
    provenance["first_failure_step"] = None
    provenance["restoration_completeness"] = f"complete; formal checkpoint step3500 plus exact replay continuation from step {start_step}; no scheduler existed in checkpoint/driver"
    write_json(OUT / "replay_provenance.json", provenance)


if __name__ == "__main__":
    main()
