#!/usr/bin/env python3
"""RTX 3090 contract smoke using the exact production model/provider/optimizer path."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v1_runtime import (
    MANIFEST, PLAN, REPORTS_DEFAULT, build_batch, build_model, build_optimizer,
    jsonable, locked_assets, seed_everything, sha256_file, train_one_step,
    trainability_counts, write_json,
)

REPORTS = REPORTS_DEFAULT
SOURCES = (
    "tokengs/models/__init__.py", "tokengs/options.py",
    "tokengs/models/object_locus_v1_controller.py", "tokengs/models/object_locus_v1.py",
    "tokengs/models/object_locus_v1_loss.py", "scripts/object_locus_v1_runtime.py",
    "scripts/train_object_locus_v1.py", "scripts/eval_object_locus_v1.py",
    "scripts/export_object_locus_v1_official.py", "scripts/smoke_object_locus_v1.py",
    "scripts/submit_object_locus_v1.sh", "tests/test_object_locus_v1_contracts.py",
    "tests/test_object_locus_v1_1_gradients.py", "docs/Object_Locus_V1_1.md",
)
SPEC_SHA = sha256_file(REPO / "docs/Object_Locus_V1_1.md")
EXPECTED_ENTRY = {
    "step": 1000, "window_index": 241, "scene": "scene0016_00",
    "context": [1506, 1517], "novel": [1509, 1516],
}
VAL_WINDOW = {"scene": "scene0011_00", "context": [68, 87],
              "novel": [69, 70, 71, 72], "window_index": 0}


def _hashes():
    return {name: sha256_file(REPO / name) for name in SOURCES}


def _module_paths():
    import tokengs.models.object_locus_v1 as model_mod
    import tokengs.models.object_locus_v1_controller as controller_mod
    import tokengs.models.object_locus_v1_loss as loss_mod
    import scripts.object_locus_v1_runtime as runtime_mod
    paths = {"model": model_mod.__file__, "controller": controller_mod.__file__,
             "loss": loss_mod.__file__, "runtime": runtime_mod.__file__}
    for key, value in paths.items():
        if not Path(value).resolve().is_relative_to(REPO):
            raise RuntimeError(f"{key} module resolved outside task worktree: {value}")
    return paths


def _finite_tree(value, prefix="root"):
    bad = []
    if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
        bad.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            bad.extend(_finite_tree(child, f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            bad.extend(_finite_tree(child, f"{prefix}[{index}]"))
    return bad


def _gradient(grad):
    if grad is None:
        return {"is_none": True, "norm": 0.0, "max_abs": 0.0, "finite": True, "nonzero": False}
    return {"is_none": False, "norm": float(grad.detach().float().norm()),
            "max_abs": float(grad.detach().float().abs().max()),
            "finite": bool(torch.isfinite(grad).all()),
            "nonzero": bool(torch.count_nonzero(grad).item() > 0)}


def _step0_entry(plan):
    entries = plan["entries"]
    if len(entries) != 5000:
        raise RuntimeError(f"locked plan has {len(entries)} entries")
    entry = entries[999]
    normalized = {key: (int(value) if key in ("step", "window_index") else value)
                  for key, value in entry.items() if key in EXPECTED_ENTRY}
    if normalized != EXPECTED_ENTRY:
        raise RuntimeError(f"fixed smoke plan entry mismatch: {normalized}")
    return entry


def run(device="cuda"):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Object-Locus smoke requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"smoke requires exactly one visible GPU, found {torch.cuda.device_count()}")
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 3090":
        raise RuntimeError(f"smoke is registered for RTX 3090; got {gpu}")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 23 * 1024**3:
        raise RuntimeError(f"3090-class 24GB memory required, got {props.total_memory} bytes")

    start = time.time()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    payload = {
        "status": "RUNNING", "version": "Object-Locus V1.1",
        "git_head_before_smoke": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "git_status_before_smoke": subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--short"], text=True),
        "gpu": gpu, "gpu_total_memory_bytes": props.total_memory,
        "gpu_total_memory_gib": props.total_memory / 1024**3,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "hostname": os.uname().nodename, "device": str(device),
        "source_hashes": _hashes(), "spec_sha256": SPEC_SHA,
        "module_paths": _module_paths(),
        "memory": {"before_allocated_bytes": allocated_before,
                   "before_reserved_bytes": reserved_before},
        "oom": False,
    }
    stage = "startup"
    REPORTS.mkdir(parents=True, exist_ok=True)
    payload["version"] = "Object-Locus V1.1"
    write_json(REPORTS / "smoke_object_locus_v1_1.json", payload)
    try:
        stage = "assets_and_model_initialization"
        seed_everything(42)
        manifest, plan = locked_assets(REPORTS)
        entry = _step0_entry(plan)
        if int(entry["window_index"]) >= len(manifest["windows"]):
            raise RuntimeError("locked smoke window index is missing")
        model, opt, transfer = build_model(device)
        optimizer, optimizer_audit = build_optimizer(model)
        trainability = trainability_counts(model)
        if trainability["frozen_numel"] != 0:
            raise RuntimeError(f"frozen parameters found: {trainability['frozen_names']}")
        batch = build_batch(opt, entry, device)
        expected_frames = [1506, 1517, 1509, 1516]
        actual_frames = batch["frame_ids"][0].detach().cpu().tolist()
        if actual_frames != expected_frames:
            raise RuntimeError(f"smoke frame IDs mismatch: {actual_frames}")
        stage = "training_forward"
        model.train()
        torch.cuda.synchronize()
        forward_start = time.time()
        output, metrics = model.step_loss(batch, step=1000, phase="train", coupled=False)
        torch.cuda.synchronize()
        payload["train_forward_seconds"] = time.time() - forward_start
        prediction = output["prediction"]
        finite_bad = _finite_tree({"output": output, "metrics": metrics})
        if finite_bad:
            raise FloatingPointError(f"nonfinite training forward values: {finite_bad[:30]}")
        required_loss_keys = ("loss_recon", "loss_understanding", "loss_anchor_group", "loss_total",
                              "loss_thing_2d", "loss_stuff_2d", "loss_semantic", "loss_identity",
                              "anchor_ce", "anchor_dice")
        losses = {key: float(metrics[key].detach()) for key in required_loss_keys}
        losses.update({"understanding_weight": float(metrics["understanding_weight"]),
                       "object_locus_loss": float(metrics["loss"].detach())})
        if losses["understanding_weight"] != 1.0:
            raise RuntimeError("step 1000 must have full understanding weight")
        payload["losses"] = losses
        payload["transfer_audit"] = transfer
        payload["optimizer_audit"] = optimizer_audit
        payload["trainability"] = trainability
        payload["smoke_training_window"] = entry
        payload["fixed_frame_ids"] = actual_frames
        payload["batch"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                    "device": str(value.device)}
                            for key, value in batch.items() if torch.is_tensor(value)}

        stage = "same_graph_gradient_probe"
        # Probe both task graphs on this same forward; autograd.grad leaves .grad untouched.
        model.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        late_decoder_candidates = [name for name, parameter in parameters.items()
                                   if name.startswith("enc_dec_backbone.decoder_blocks.11.")
                                   and parameter.ndim == 2]
        if not late_decoder_candidates:
            raise RuntimeError("no matrix parameter found in decoder block 11")
        watched_names = (late_decoder_candidates[0], "anchor_decoder.mu",
                         "activation_head.deconv.weight")
        missing = [name for name in watched_names if name not in parameters]
        if missing:
            raise RuntimeError(f"registered representative parameters are missing: {missing}")
        selected = [parameters[name] for name in watched_names]
        grads_u = torch.autograd.grad(metrics["loss_understanding"], selected,
                                      retain_graph=True, allow_unused=True)
        grads_r = torch.autograd.grad(metrics["loss_recon"], selected,
                                      retain_graph=True, allow_unused=True)
        gradient_probe = {}
        for name, gu, gr in zip(watched_names, grads_u, grads_r):
            gradient_probe[name] = {"understanding": _gradient(gu), "reconstruction": _gradient(gr)}
        decoder_name = watched_names[0]
        if not gradient_probe[decoder_name]["understanding"]["nonzero"]:
            raise RuntimeError("understanding gradient did not reach late decoder block 11")
        if not gradient_probe["anchor_decoder.mu"]["understanding"]["nonzero"]:
            raise RuntimeError("understanding gradient did not reach anchor geometry mu")
        if not gradient_probe["activation_head.deconv.weight"]["reconstruction"]["nonzero"]:
            raise RuntimeError("reconstruction gradient did not reach RGB activation head")
        payload["same_graph_gradient_probe"] = gradient_probe

        final = prediction["states"][-1]
        checks = {
            "anchor_embedding_shape": list(final["anchor_embedding"].shape),
            "q_shape": list(final["q"].shape), "c_shape": list(final["c"].shape),
            "s_shape": list(final["s"].shape),
            "evidence_shape": list(final["evidence_attention"].shape),
            "anchor_assignment_shape": list(final["anchor_assignment"].shape),
            "gaussian_shape": list(output["prediction"]["gaussians"].shape),
            "semantic_scores_shape": list(output["prediction"]["semantic_scores"].shape),
            "ownership_simplex_max_abs_error": float((final["anchor_assignment"].sum(-1)-1).abs().max()),
            "evidence_simplex_max_abs_error": float((final["evidence_attention"].sum(-1)-1).abs().max()),
            "min_s_over_ell": float((final["s"] / final["ell"][:, None, None]).min()),
            "max_s_over_ell": float((final["s"] / final["ell"][:, None, None]).max()),
            "max_c_displacement_over_ell": float(
                torch.linalg.vector_norm(final["c_displacement"], dim=-1).max() / final["ell"].max()),
        }
        if final["anchor_embedding"].shape != (1, 1024, 256) or final["q"].shape != (1, 102, 256):
            raise RuntimeError(f"registered state shapes mismatch: {checks}")
        if final["anchor_assignment"].shape != (1, 1024, 103):
            raise RuntimeError(f"ownership shape mismatch: {checks}")
        if checks["ownership_simplex_max_abs_error"] > 1e-5 or checks["evidence_simplex_max_abs_error"] > 1e-5:
            raise RuntimeError(f"probability simplex check failed: {checks}")
        if checks["min_s_over_ell"] < .05 - 1e-6 or checks["max_s_over_ell"] > 2 + 1e-6:
            raise RuntimeError(f"scale bounds failed: {checks}")
        if checks["max_c_displacement_over_ell"] > .25 + 1e-6:
            raise RuntimeError(f"geometry displacement failed: {checks}")
        payload["forward_numeric_contracts"] = checks

        stage = "production_backward_clip_optimizer_step"
        gc_gradient_audit = {}
        param_before = {name: parameters[name].detach().clone() for name in (
            "object_locus.W_Q.weight", "anchor_decoder.mu",
            "activation_head.deconv.weight")}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        backward_start = time.time()
        _step_output, step_metrics = train_one_step(
            model, optimizer, batch, 1000, precomputed=(output, metrics),
            gradient_audit=gc_gradient_audit,
        )
        torch.cuda.synchronize()
        payload["backward_clip_step_seconds"] = time.time() - backward_start
        if not all(row["finite"] and row["nonzero"] for name, row in gc_gradient_audit.items()
                   if name.endswith(("W_V.weight", "W_own_e.weight", "thing_classifier.weight",
                                     "W_c.weight", "W_s.weight"))):
            raise RuntimeError("one or more required evidence/ownership/class/geometry heads had no gradient")
        required_head_names = ("object_locus.W_V.weight", "object_locus.W_own_e.weight",
                               "object_locus.thing_classifier.weight", "object_locus.W_c.weight",
                               "object_locus.W_s.weight")
        missing_heads = [name for name in required_head_names
                         if name not in gc_gradient_audit or not gc_gradient_audit[name]["nonzero"]]
        if missing_heads:
            raise RuntimeError(f"required Object-Locus heads did not receive nonzero gradients: {missing_heads}")
        if not all(row["finite"] for row in gc_gradient_audit.values()):
            raise FloatingPointError("nonfinite Object-Locus parameter gradient")
        del output, metrics, prediction
        gc.collect()
        torch.cuda.empty_cache()
        updated = {}
        for name, before in param_before.items():
            delta = parameters[name].detach() - before
            updated[name] = {"max_abs_delta": float(delta.abs().max()),
                             "changed": bool(torch.count_nonzero(delta).item() > 0)}
        if not all(row["changed"] for row in updated.values()):
            raise RuntimeError(f"smoke optimizer step failed to update required branches: {updated}")
        payload["production_step"] = {"step": 1000, "losses": losses,
                                      "gradient_report_before_clip": step_metrics["gradient_report_before_clip"],
                                      "object_locus_gradient_audit": gc_gradient_audit,
                                      "pre_clip_global_grad_norm": step_metrics["pre_clip_global_grad_norm"],
                                      "clip_coefficient": step_metrics["clip_coefficient"],
                                      "registered_hooks": step_metrics["gc_registered_hooks"],
                                      "removed_hooks": step_metrics["gc_removed_hooks"],
                                      "parameter_updates": updated}
        payload["memory"].update({
            "training_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "training_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "training_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "training_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        })
        payload["status"] = "TRAIN_SMOKE_PASS"
        write_json(REPORTS / "smoke_object_locus_v1_1.json", payload)

        stage = "val32_evaluator_interface"
        # Keep this same post-step model only for the required single val32 pair interface smoke.
        from scripts.eval_instance_state_v1 import evaluate_windows
        from scripts.export_object_locus_v1_official import export_windows
        from scripts.eval_object_locus_v1 import _official_run
        model.eval()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        eval_start = time.time()
        val_context = evaluate_windows(model, opt, [VAL_WINDOW], 1000, "context",
                                      REPORTS / "smoke_val32_local", arm="C", device=device,
                                      batch_builder=build_batch)
        val_target = evaluate_windows(model, opt, [VAL_WINDOW], 1000, "target",
                                     REPORTS / "smoke_val32_local", arm="C", device=device,
                                     batch_builder=build_batch)
        all_export = export_windows(model, opt, [VAL_WINDOW], REPORTS / "smoke_val32_official_all",
                                    device=device, batch_builder=build_batch, target_frames="all")
        novel_export = export_windows(model, opt, [VAL_WINDOW], REPORTS / "smoke_val32_official_novel",
                                      device=device, batch_builder=build_batch, target_frames="novel")
        official = _official_run(REPORTS / "smoke_val32_official_all",
                                 REPORTS / "smoke_val32_official_all_result.json")
        torch.cuda.synchronize()
        payload["evaluator_smoke"] = {
            "status": "PASS", "window": VAL_WINDOW,
            "frame_ids": [68, 87, 69, 70, 71, 72],
            "context_local": val_context, "target_all_local": val_target,
            "official_all_export": all_export, "official_novel_export": novel_export,
            "official_all_eval": official,
            "official_all_result": official.get("result"),
            "seconds": time.time() - eval_start,
        }
        payload["memory"].update({"eval_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "eval_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                  "eval_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                                  "eval_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
        overall_allocated = max(payload["memory"]["training_peak_allocated_bytes"],
                                payload["memory"]["eval_peak_allocated_bytes"])
        overall_reserved = max(payload["memory"]["training_peak_reserved_bytes"],
                               payload["memory"]["eval_peak_reserved_bytes"])
        payload["memory"].update({"overall_peak_allocated_bytes": overall_allocated,
                                  "overall_peak_reserved_bytes": overall_reserved,
                                  "overall_peak_allocated_gib": overall_allocated / 1024**3,
                                  "overall_peak_reserved_gib": overall_reserved / 1024**3})
        payload["elapsed_seconds"] = time.time() - start
        payload["status"] = "PASS"
        write_json(REPORTS / "smoke_object_locus_v1_1.json", payload)
        return payload
    except torch.cuda.OutOfMemoryError as exc:
        payload["status"] = "FAIL_OOM"
        payload["oom"] = True
        payload["failed_stage"] = stage
        payload["exception"] = repr(exc)
        payload["traceback"] = traceback.format_exc()
        payload["memory"].update({"failure_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "failure_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        write_json(REPORTS / "smoke_object_locus_v1_1.json", payload)
        raise
    except Exception as exc:
        payload["status"] = "FAIL"
        payload["failed_stage"] = stage
        payload["exception"] = repr(exc)
        payload["traceback"] = traceback.format_exc()
        try:
            torch.cuda.synchronize()
            payload["memory"].update({"failure_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "failure_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        except Exception:
            pass
        write_json(REPORTS / "smoke_object_locus_v1_1.json", payload)
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.device)
    print(json.dumps(jsonable(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
