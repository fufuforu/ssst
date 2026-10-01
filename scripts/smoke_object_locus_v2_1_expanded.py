"""One isolated continuation update and single-pair evaluator/export smoke."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v2_1_runtime import build_model, build_optimizer, build_batch, train_one_step, write_json, trainability_counts
from scripts.train_object_locus_v2_1_expanded import (
    SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA, SOURCE_GIT_SHA, SOURCE_DATA_MANIFEST_SHA,
    SOURCE_MANIFEST_PATH, SOURCE_REPORTS, REPORTS, INITIAL_GLOBAL_STEP,
    EXPECTED_MODEL, _assert_source_checkpoint, _assert_optimizer_state_exact,
    expanded_lrs, _assert_backend_settings,
)
from scripts.object_locus_v2_1_runtime import sha256_file, capture_rng, restore_rng


def _finite(value):
    if torch.is_tensor(value): return bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(_finite(v) for v in value)
    if isinstance(value, float): return value == value and abs(value) != float("inf")
    return True


def _exercise_bundle_schema(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rows = [{"step": 1, "scalar": 0.5, "identity": {"scene": "scene"}},
            {"step": 2, "scalar": 0.25, "flag": True}]
    csv_path = directory / "heterogeneous.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    json_path = directory / "smoke_schema.json"
    json_path.write_text(json.dumps({"rows": rows}, allow_nan=False))
    archive = directory / "smoke_schema.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, csv_path.name); z.write(json_path, json_path.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        import io
        assert len(list(csv.DictReader(io.StringIO(z.read(csv_path.name).decode())))) == 2
        assert json.loads(z.read(json_path.name))["rows"][0]["identity"]["scene"] == "scene"
    return {"archive": str(archive), "bytes": archive.stat().st_size, "schema_check": "PASS"}


def main():
    source_sha = sha256_file(SOURCE_CHECKPOINT)
    if source_sha != SOURCE_CHECKPOINT_SHA: raise RuntimeError("source checkpoint SHA mismatch")
    if sha256_file(SOURCE_MANIFEST_PATH) != SOURCE_DATA_MANIFEST_SHA:
        raise RuntimeError("source Stage S manifest changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expanded smoke requires exactly one CUDA GPU")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090" or not os.uname().nodename.startswith("3dimage-13"):
        raise RuntimeError("expanded smoke must run on 3dimage-13 RTX3090")
    if torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
        raise RuntimeError("expanded smoke GPU is not 24GB class")
    backend_settings = _assert_backend_settings()
    execution_sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    _assert_source_checkpoint(checkpoint)
    model, opt, transfer = build_model("cuda")
    optimizer, optimizer_info = build_optimizer(model)
    trainability = trainability_counts(model)
    if trainability["frozen_numel"] != 0:
        raise RuntimeError("smoke found frozen model parameters")
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.equal(p.detach().cpu(), checkpoint["model"][k])
           for k, p in model.state_dict().items()):
        raise RuntimeError("smoke source model did not restore exactly")
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_restore = _assert_optimizer_state_exact(checkpoint["optimizer"], optimizer)
    if optimizer_restore["state_entries"] != 540 or optimizer_restore["internal_step_values"] != [1792.0]:
        raise RuntimeError(f"smoke optimizer continuation state mismatch: {optimizer_restore}")
    restore_rng(checkpoint["rng"])
    del checkpoint
    window = {"scene": "scene0016_00", "context": [1506, 1517], "novel": [1509, 1516]}
    batch = build_batch(opt, window, "cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    output, metrics = train_one_step(model, optimizer, batch, INITIAL_GLOBAL_STEP + 1,
        understanding_weight_value=1.0, lr_values=expanded_lrs(1),
        failure_capture_dir=REPORTS / "smoke/failures",
        failure_context={"stage": "expanded_smoke", "expanded_step": 1,
                         "global_optimizer_step": INITIAL_GLOBAL_STEP + 1, "window": window,
                         "optimizer_updated": False})
    finite = _finite(metrics) and _finite(output["prediction"])
    post_finite = all(torch.isfinite(p).all() for p in model.parameters()) and all(
        torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values()
        if torch.is_tensor(v) and v.is_floating_point())
    key = metrics["gradient_report_before_clip"]
    required = ("object_locus_v2_1.category_head.weight", "object_locus_v2_1.objectness_head.weight",
        "object_locus_v2_1.cls_fuse.weight", "object_locus_v2_1.child_mlp.2.weight",
        "activation_head.deconv.weight")
    grad_ok = all(key.get(name, {}).get("finite", False) and key.get(name, {}).get("nonzero", False)
                  for name in required)
    key_outputs = {}
    for name in ("category_logits18", "objectness_logits", "anchor_membership",
                 "gaussian_membership", "gaussians", "semantic_scores", "identity_render"):
        value = output["prediction"].get(name)
        key_outputs[name] = {"finite": bool(torch.isfinite(value).all()), "shape": list(value.shape)} if torch.is_tensor(value) else None
    smoke_dir = REPORTS / "smoke"; smoke_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "PASS" if finite and post_finite and grad_ok else "FAIL",
        "git_sha": execution_sha, "source_checkpoint_sha256": source_sha,
        "source_git_sha": SOURCE_GIT_SHA, "source_data_manifest_sha256": SOURCE_DATA_MANIFEST_SHA,
        "backend_settings": backend_settings, "trainability": trainability,
        "node": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
        "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "window": window, "global_optimizer_step": INITIAL_GLOBAL_STEP + 1,
        "object_lr": expanded_lrs(1)[0], "reconstruction_lr": expanded_lrs(1)[1],
        "understanding_weight": 1.0, "gc_alpha": 0.01, "metrics": metrics,
        "key_output_tensors": key_outputs, "required_gradients": {n: key.get(n) for n in required},
        "optimizer_restore": optimizer_restore, "all_finite": finite,
        "post_step_parameters_optimizer_finite": bool(post_finite), "required_gradients_pass": grad_ok,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "optimizer_mapping": optimizer_info}
    write_json(smoke_dir / "smoke_result.json", result)
    del output, metrics, batch
    model.eval()
    from scripts.eval_object_locus_v2_1 import evaluate_windows
    val32 = json.loads((SOURCE_REPORTS / "monitor_32pairs.json").read_text())["pairs"]
    result["minimal_evaluator"] = evaluate_windows(model, opt, val32[:1], INITIAL_GLOBAL_STEP + 1,
        "val32_expanded_smoke", smoke_dir, "cuda", build_batch, official=True, panels=True)
    result["bundle_schema_check"] = _exercise_bundle_schema(smoke_dir / "bundle_schema")
    write_json(smoke_dir / "smoke_result.json", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if result["status"] != "PASS": raise RuntimeError("expanded continuation smoke failed")


if __name__ == "__main__": main()
