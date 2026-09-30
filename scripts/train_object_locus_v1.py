#!/usr/bin/env python3
"""Fresh-start or exact-resume the registered Object-Locus V1 5000-step run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v1_runtime import (
    GC_ALPHA, GRAD_CLIP, MANIFEST_SHA, MONITOR_SHA, PLAN_SHA,
    PRETRAINED_SHA, REPORTS_DEFAULT, RUN_ROOT_DEFAULT, TOTAL_STEPS,
    WARMUP_STEPS, build_batch, build_model, build_optimizer, capture_rng,
    jsonable, locked_assets, restore_rng, seed_everything, sha256_file,
    train_one_step, trainability_counts, write_json,
)
from scripts.eval_object_locus_v1 import evaluate_dataset

EVAL_STEPS = (0, 200, 500, 1000, 2000, 3500, 5000)
TRAIN_LOG_STEPS = 100
SPEC_SHA = "1bf3dc7c0affceaff9f6fac3299004f33f1eca33e1efb5b2a1c0fd0d18a1d395"
SOURCE_FILES = (
    "tokengs/models/__init__.py", "tokengs/options.py",
    "tokengs/models/object_locus_v1_controller.py",
    "tokengs/models/object_locus_v1.py", "tokengs/models/object_locus_v1_loss.py",
    "scripts/object_locus_v1_runtime.py", "scripts/train_object_locus_v1.py",
    "scripts/eval_object_locus_v1.py", "scripts/export_object_locus_v1_official.py",
    "scripts/smoke_object_locus_v1.py", "scripts/submit_object_locus_v1.sh",
    "tests/test_object_locus_v1_contracts.py", "docs/object_locus_v1_codex_spec.md",
)


def _source_hashes():
    return {name: sha256_file(REPO / name) for name in SOURCE_FILES}


def _assert_committed_and_pushed():
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain=v1"], text=True).strip():
        raise RuntimeError("formal driver refuses a dirty implementation worktree")
    try:
        subprocess.run(["git", "-C", str(REPO), "fetch", "origin", "main"], check=True)
        subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit,
                        "origin/main"], check=True)
        remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                         cwd=REPO, text=True).split()[0]
        verification = "live_fetch_and_ls_remote"
    except subprocess.CalledProcessError as exc:
        # Compute nodes can inherit a login-node-only proxy. The run is submitted
        # only after the caller pushed and live-verified main; accept the cached
        # remote-tracking ref only when it is an exact HEAD match.
        cached_remote = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "refs/remotes/origin/main"], text=True
        ).strip()
        if cached_remote != commit:
            raise RuntimeError(
                f"remote verification network failure ({exc}); cached origin/main "
                f"{cached_remote} does not exactly match HEAD {commit}"
            ) from exc
        remote = cached_remote
        verification = "exact_cached_origin_main_after_prior_live_push_verification"
    if remote != commit:
        raise RuntimeError(f"formal run requires verified remote main==HEAD, got {remote} vs {commit}")
    print(f"[remote-check] main={remote} method={verification}", flush=True)
    return commit

def _optimizer_audit(optimizer):
    return [{"name": g["name"], "tensor_count": len(g["params"]),
             "numel": sum(p.numel() for p in g["params"]),
             "weight_decay": g["weight_decay"], "peak_lr": g["lr"]}
            for g in optimizer.param_groups]


def _snapshot_parameters(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _assert_snapshot(model, snapshot, where):
    current = model.state_dict()
    changed = [name for name, value in snapshot.items() if not torch.equal(value, current[name].detach().cpu())]
    if changed:
        raise RuntimeError(f"evaluation mutated model tensors at {where}: {changed[:12]}")


def _checkpoint_payload(model, optimizer, step, plan_position, opt, commit, source_hashes,
                        transfer, trainability, optimizer_groups, online_matches):
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": int(step), "plan_position": int(plan_position),
        "total_steps": TOTAL_STEPS, "warmup_steps": WARMUP_STEPS,
        "architecture_name": model.architecture_name,
        "config": {"model_type": opt.model_type, "seed": 42,
                   "instance_state_layers": list(model.state_layers),
                   "state_dim": 256, "num_thing": 100, "num_stuff": 2,
                   "void_index": 102, "num_region_channels": 103,
                   "precision": "fp32", "batch_size": 1,
                   "gradient_accumulation_steps": 1},
        "spec_sha256": SPEC_SHA, "git_commit": commit,
        "source_file_hashes": source_hashes,
        "pretrained_sha256": PRETRAINED_SHA, "manifest_sha256": MANIFEST_SHA,
        "plan_sha256": PLAN_SHA, "monitor_sha256": MONITOR_SHA,
        "transfer_audit": transfer, "trainability": trainability,
        "optimizer_groups": optimizer_groups, "shared_understanding_grad_scale": GC_ALPHA,
        "joint": True, "beta": 0.0, "rng": capture_rng(),
        "online_matched_query_counts": online_matches,
    }


def _save_checkpoint(run_root, payload, step):
    root = Path(run_root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{step:08d}"
    if final.exists():
        if (final / "COMPLETE").is_file():
            raise RuntimeError(f"refusing to overwrite completed checkpoint {final}")
        raise RuntimeError(f"incomplete checkpoint directory exists; manual provenance review required: {final}")
    temp = Path(tempfile.mkdtemp(prefix=f".step_{step:08d}.", dir=root))
    try:
        torch.save(payload, temp / "train_state.pt")
        with (temp / "train_state.pt").open("rb") as f:
            os.fsync(f.fileno())
        (temp / "COMPLETE").write_text(json.dumps({"step": step, "git_commit": payload["git_commit"]}) + "\n")
        os.replace(temp, final)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return final / "train_state.pt"


def _verify_resume(payload, commit, source_hashes):
    expected = {
        "total_steps": TOTAL_STEPS, "warmup_steps": WARMUP_STEPS,
        "architecture_name": "LOCUSGS_OBJECT_LOCUS_V1", "spec_sha256": SPEC_SHA,
        "git_commit": commit, "source_file_hashes": source_hashes,
        "pretrained_sha256": PRETRAINED_SHA, "manifest_sha256": MANIFEST_SHA,
        "plan_sha256": PLAN_SHA, "monitor_sha256": MONITOR_SHA,
        "shared_understanding_grad_scale": GC_ALPHA, "joint": True, "beta": 0.0,
    }
    differences = {key: (payload.get(key), value) for key, value in expected.items()
                   if payload.get(key) != value}
    required = ("model", "optimizer", "step", "plan_position", "config", "rng", "optimizer_groups")
    missing = [key for key in required if key not in payload]
    if differences or missing:
        raise RuntimeError(f"checkpoint provenance mismatch={differences}, missing={missing}")


def _latest_checkpoint(run_root):
    root = Path(run_root) / "checkpoints"
    if not root.exists():
        return None
    complete = sorted(p for p in root.glob("step_*") if (p / "COMPLETE").is_file())
    return complete[-1] / "train_state.pt" if complete else None


def _eval_node(model, opt, step, reports, device, windows, run_root, checkpoint_path):
    snapshot = _snapshot_parameters(model)
    mode = model.training
    rng = capture_rng()
    try:
        model.eval()
        split_results = {}
        for split, rows in (("train16", windows["train16"]),
                            ("val8", windows["val8"]),
                            ("val32", windows["val32"])):
            split_results[split] = evaluate_dataset(
                model, opt, rows, split, step, reports, device, build_batch,
                panels=True, official=True,
            )
    finally:
        restore_rng(rng)
        model.train(mode)
    _assert_snapshot(model, snapshot, f"step {step} evaluation")
    row = {"step": step, "checkpoint": str(checkpoint_path), "git_commit": subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "splits": split_results,
        "camera_pose_note": "ssst uses GT camera poses; SIU3R is unposed; same evaluator does not equal same input condition",
        "benchmark_scope_note": "128-scene monitor structural validation, not the complete SIU3R 1860-pair benchmark",
    }
    write_json(Path(reports) / f"curves_{step}.json", row)
    return row


def _drift_category(name):
    if name.startswith("enc_dec_backbone.decoder_blocks."):
        return "decoder"
    if name.startswith("anchor_decoder.") and (name in {
        "anchor_decoder.mu", "anchor_decoder.rho", "anchor_decoder.gamma_raw"
    } or name.startswith(("anchor_decoder.refine_mu.", "anchor_decoder.refine_rho.",
                           "anchor_decoder.pe_mlp", "anchor_decoder.pe_mlps"))):
        return "anchor_geometry"
    if name.startswith("activation_head."):
        return "activation_head"
    if name.startswith("enc_dec_backbone."):
        return "encoder"
    return "other_reconstruction"


def _write_endpoint_audits(model, opt, optimizer, reports, run_root, step0_state,
                           payload, log_rows):
    state = payload
    requirements = {
        "step": state["step"] == 5000,
        "architecture": state["architecture_name"] == "LOCUSGS_OBJECT_LOCUS_V1",
        "joint": state["joint"] is True, "beta": state["beta"] == 0.0,
        "shared_scale": state["shared_understanding_grad_scale"] == GC_ALPHA,
        "manifest": state["manifest_sha256"] == MANIFEST_SHA,
        "plan": state["plan_sha256"] == PLAN_SHA,
        "pretrained": state["pretrained_sha256"] == PRETRAINED_SHA,
        "rng": all(k in state["rng"] for k in ("python", "numpy", "torch", "cuda")),
        "trainability": state["trainability"]["frozen_numel"] == 0,
    }
    bad_model = [key for key, value in state["model"].items()
                 if value.is_floating_point() and not torch.isfinite(value).all()]
    bad_optimizer = []
    for group_i, group in enumerate(state["optimizer"].get("state", {}).values()):
        for key, value in group.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                bad_optimizer.append(f"{group_i}.{key}")
    requirements["model_finite"] = not bad_model
    requirements["optimizer_finite"] = not bad_optimizer
    audit = {"status": "PASS" if all(requirements.values()) else "FAIL",
             "step": state["step"], "architecture": state["architecture_name"],
             "recipe": "OBJECT_LOCUS_V1_GC_ALPHA001", "joint": state["joint"],
             "beta": state["beta"], "shared_scale": GC_ALPHA,
             "manifest_sha256": MANIFEST_SHA, "plan_sha256": PLAN_SHA,
             "pretrained_sha256": PRETRAINED_SHA,
             "trainability": state["trainability"], "requirements": requirements,
             "nonfinite_model_tensors": bad_model, "nonfinite_optimizer_tensors": bad_optimizer,
             "rng_present": list(state["rng"])}
    write_json(Path(reports) / "formal_endpoint_step5000_audit.json", audit)
    if not all(requirements.values()):
        raise RuntimeError(f"formal endpoint audit failed: {audit}")

    drift = {}
    totals = {name: {"ref_sq": 0.0, "delta_sq": 0.0, "max_abs": 0.0, "tensors": 0}
              for name in ("encoder", "decoder", "anchor_geometry", "activation_head", "other_reconstruction")}
    endpoint = model.state_dict()
    for name, before in step0_state.items():
        if name.startswith("object_locus."):
            continue
        after = endpoint[name].detach().cpu()
        delta = after.double() - before.double()
        category = _drift_category(name)
        row = totals[category]
        row["ref_sq"] += float(before.double().square().sum())
        row["delta_sq"] += float(delta.square().sum())
        row["max_abs"] = max(row["max_abs"], float(delta.abs().max()))
        row["tensors"] += 1
    for name, values in totals.items():
        l2 = math.sqrt(values["delta_sq"])
        drift[name] = {"tensor_count": values["tensors"], "l2_delta": l2,
                       "relative_l2_delta": l2 / max(math.sqrt(values["ref_sq"]), 1e-30),
                       "max_abs_delta": values["max_abs"]}
    write_json(Path(reports) / "formal_reconstruction_drift_audit.json", drift)
    norms = np.asarray([r["pre_clip_global_grad_norm"] for r in log_rows], dtype=np.float64)
    clips = np.asarray([r["clip_coefficient"] for r in log_rows], dtype=np.float64)
    summary = {"completed_steps": 5000, "logged_train_steps": len(log_rows),
               "all_logged_metrics_finite": all(all(math.isfinite(float(v)) for v in row.values()
                                                     if isinstance(v, (float, int))) for row in log_rows),
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "cuda": torch.version.cuda, "manifest_sha256": MANIFEST_SHA,
               "plan_sha256": PLAN_SHA, "pretrained_sha256": PRETRAINED_SHA,
               "shared_understanding_grad_scale": GC_ALPHA,
               "trainability": state["trainability"],
               "preclip_global_grad_norm": {"median": float(np.median(norms)),
                                             "p10": float(np.quantile(norms, .1)),
                                             "p90": float(np.quantile(norms, .9)),
                                             "max": float(norms.max())},
               "clip_coefficient": {"median": float(np.median(clips)),
                                    "fraction_below_one": float(np.mean(clips < 1.0)),
                                    "min": float(clips.min())},
               "logs": str(Path(run_root) / "train.log"),
               "structured_metrics_jsonl": str(Path(reports) / "training_log_metrics.jsonl"),
               "driver_events": str(Path(run_root) / "train_events.log")}
    write_json(Path(reports) / "training_log_summary.json", summary)
    return audit, drift, summary


def _read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def _fmt(value, digits=4):
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "N/A" if not math.isfinite(number) else f"{number:.{digits}f}"


def _curves_local(curves, split, scope):
    if not curves:
        return None
    # New task nodes wrap evaluator outputs; S0/S1 artifacts retain the flat legacy schema.
    node = curves.get("splits", {}).get(split)
    if node:
        return node.get("local", {}).get(scope)
    return curves.get(f"{split}_{scope}")


def _curves_psnr(curves, split, scope):
    if not curves:
        return None
    node = curves.get("splits", {}).get(split)
    if node:
        key = {"context": "context", "novel": "novel", "target": "target_all"}[scope]
        return node.get("float_psnr_db", {}).get(key)
    return curves.get(f"{split}_{scope}", {}).get("psnr")


def _write_formal_report(reports, drift, summary, online_matches):
    reports = Path(reports)
    steps = (0, 200, 500, 1000, 2000, 3500, 5000)
    current = {step: _read_json(reports / f"curves_{step}.json") for step in steps}
    v1_root = Path("/space/mawb/ssst/group_plus/anchor_group_v1")
    s0_root = Path("/space/mawb/ssst/group_plus/instance_state_v1_generalization")
    s1_root = Path("/space/mawb/ssst/group_plus/instance_state_v2_s1_local3d")
    baselines = {
        "S0@5k": _read_json(s0_root / "curves_5000.json"),
        "S1@5k": _read_json(s1_root / "curves_5000.json"),
        "Anchor-Group V1@5k": _read_json(v1_root / "curves_5000.json"),
    }
    v1_drift = _read_json(v1_root / "formal_reconstruction_drift_audit.json")
    rows = ["# Object-Locus V1 Formal 5k Report", "", "## Completion and provenance", "",
            "- Completed steps: 5000/5000; endpoint audit: PASS.",
            "- Architecture: `LOCUSGS_OBJECT_LOCUS_V1`; recipe: `OBJECT_LOCUS_V1_GC_ALPHA001`.",
            "- Fresh initialization uses the locked pretrained reconstruction checkpoint and fresh object-locus seed; no smoke checkpoint was used.",
            f"- Manifest SHA256: `{MANIFEST_SHA}`; plan SHA256: `{PLAN_SHA}`; pretrained SHA256: `{PRETRAINED_SHA}`.",
            "- Model was fully trainable, FP32, batch 1, AdamW; GC coefficient 0.01; no AMP or accumulation.",
            "- Evaluation nodes: 0, 200, 500, 1000, 2000, 3500, 5000 on locked train16/val8/val32 monitors.",
            "- SSST evaluation inputs use GT camera poses; SIU3R is unposed. This 128-scene monitor is structural validation, not the complete 1860-pair benchmark.", "",
            "## Registered curves", "",
             "Metrics below are local evaluator values in [0,1] (not percentages); PSNR is dB. Target means target-all views.", ""]
    rows.append("The official target column is read from the separately exported novel-only target subset; it is not the target-all aggregate.")
    rows.append("")
    for split in ("train16", "val8", "val32"):
        rows += [f"### {split}", "",
                 "| Step | Scope | thing mIoU | all mIoU | ca-R50 TP/FP/FN (GT) | class-aware R50 | local PQ | PSNR dB | official mIoU/PQ/mAP/AP50 |",
                 "|---:|:---|---:|---:|:---|---:|---:|---:|:---|"]
        for step in steps:
            data = current[step]
            for scope, official_scope in (("context", "context_from_official_all"),
                                          ("target", "novel_from_novel_only_subset")):
                local = _curves_local(data, split, scope) if data else None
                if not local:
                    rows.append(f"| {step} | {scope} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                    continue
                tp, fp, fn = (local.get("tp_class_agnostic"), local.get("fp_class_agnostic"),
                              local.get("fn_class_agnostic"))
                official = (data.get("splits", {}).get(split, {}).get("official") or {}).get(official_scope, {})
                official_name = "context" if scope == "context" else "target"
                miou_key = f"{official_name}_miou"
                pq_key = f"{official_name}_pq"
                map_key = f"{official_name}_map"
                ap = official.get(f"{map_key}")
                ap50 = official.get(f"{map_key}_50")
                official_text = (f"{_fmt(official.get(miou_key))}/"
                                 f"{_fmt(official.get(pq_key))}/"
                                 f"{_fmt(ap)}/ {_fmt(ap50)}")
                rows.append(
                    f"| {step} | {scope} | {_fmt(local.get('mIoU_thing'))} | "
                    f"{_fmt(local.get('mIoU_all_nonempty'))} | {tp}/{fp}/{fn} ({local.get('n_gt_instances')}) | "
                    f"{_fmt(local.get('class_aware_recall50'))} | {_fmt(local.get('local_pq'))} | "
                    f"{_fmt(_curves_psnr(data, split, scope), 3)} | {official_text} |"
                )
        rows.append("")
    rows += ["## Same-endpoint local comparison on val32", "",
             "| Model | Scope | thing mIoU | ca-R50 TP/FP/FN (GT) | class-aware R50 | local PQ | PSNR dB |", "|:---|:---|---:|:---|---:|---:|---:|"]
    for model_name, curves in baselines.items():
        for scope in ("context", "target"):
            local = _curves_local(curves, "val32", scope)
            if not local:
                rows.append(f"| {model_name} | {scope} | N/A | N/A | N/A | N/A | N/A |")
                continue
            rows.append(f"| {model_name} | {scope} | {_fmt(local.get('mIoU_thing'))} | "
                        f"{local.get('tp_class_agnostic')}/{local.get('fp_class_agnostic')}/"
                        f"{local.get('fn_class_agnostic')} ({local.get('n_gt_instances')}) | "
                        f"{_fmt(local.get('class_aware_recall50'))} | {_fmt(local.get('local_pq'))} | "
                        f"{_fmt(_curves_psnr(curves, 'val32', scope), 3)} |")
    rows += ["", "Historical S0/S1 curve artifacts expose local evaluator metrics; archived official all/novel metrics are not present in those curve JSONs, so their official comparison is N/A rather than reconstructed.", "",
             "## Anchor and slot diagnostics", "",
             "Per-window query/anchor diagnostics, including centered covariance participation ratio, query cosine, evidence overlap, ownership concentration, best visible-anchor Dice, matched query IDs and active query count, are retained in each `eval_<split>/step_*/evaluation_summary.json` and per-window files.",
             "The local class-agnostic/class-aware instance TP counts use GT-free predicted query masks and one-to-one IoU>=0.5 matching; `n_gt_instances` is the matched GT denominator.",
             f"Online training Hungarian matches: {json.dumps(online_matches)}; total matches={sum(online_matches)}, unique winning queries={sum(x > 0 for x in online_matches)}.",
             "Evaluation utilization is reported separately from online matches; no extra 1024-window scan was run.", "",
             "## Endpoint parameter drift", "",
             "| Reconstruction category | L2 delta | Relative L2 | Max absolute delta |",
             "|:---|---:|---:|---:|"]
    for category, item in drift.items():
        old = (v1_drift or {}).get(category, {})
        rows.append(f"| {category} | {_fmt(item['l2_delta'], 6)} | {_fmt(item['relative_l2_delta'], 6)} | {_fmt(item['max_abs_delta'], 6)} | "
                    f"{_fmt(old.get('relative_l2_delta'), 6)} |")
    # Amend the header with the historical V1 comparison column.
    header_at = rows.index("| Reconstruction category | L2 delta | Relative L2 | Max absolute delta |")
    rows[header_at] = "| Reconstruction category | L2 delta | Relative L2 | Max absolute delta | V1 relative L2 |"
    rows[header_at + 1] = "|:---|---:|---:|---:|---:|"
    rows += ["", "## Reconstruction preservation", "",
             "Fixed registered criterion: context and novel PSNR must each decline by no more than 0.5 dB from this run's step 0. The local evaluator records target-all PSNR; official task evaluation separately records all and novel-only segmentation scopes."]
    for scope in ("context", "novel"):
        p0 = _curves_psnr(current[0], "val32", scope) if current[0] else None
        p5 = _curves_psnr(current[5000], "val32", scope) if current[5000] else None
        delta = (p5 - p0) if p0 is not None and p5 is not None else None
        label = "context" if scope == "context" else "novel"
        verdict = "within 0.5 dB" if delta is not None and delta >= -0.5 else "exceeds 0.5 dB decline" if delta is not None else "N/A"
        rows.append(f"- Val32 {label}: step0={_fmt(p0, 3)} dB, step5000={_fmt(p5, 3)} dB, delta={_fmt(delta, 3)} dB; {verdict}.")
    train_psnr0 = _curves_psnr(current[0], "train16", "context") if current[0] else None
    train_psnr5 = _curves_psnr(current[5000], "train16", "context") if current[5000] else None
    rows += ["", "## Effective object masks and endpoint deltas", "",
             "The local evaluator forms predictions from GT-free thing-query ownership masks and applies its registered score/IoU thresholds; TP is one-to-one matched predicted masks at IoU>=0.5. These counts are diagnostic and are not an official mAP/PQ substitute.", "",
             "| Comparison | Scope | thing mIoU delta | ca-R50 delta | class-aware R50 delta | PSNR delta dB |", "|:---|:---|---:|---:|---:|---:|"]
    current_end = current[5000]
    for baseline_name in ("S0@5k", "S1@5k", "Anchor-Group V1@5k"):
        base = baselines.get(baseline_name)
        for scope in ("context", "target"):
            now = _curves_local(current_end, "val32", scope) if current_end else None
            before = _curves_local(base, "val32", scope)
            if not now or not before:
                rows.append(f"| Object-Locus V1 vs {baseline_name} | {scope} | N/A | N/A | N/A | N/A |")
                continue
            def delta(key):
                left, right = now.get(key), before.get(key)
                return left - right if left is not None and right is not None else None
            now_psnr, before_psnr = (_curves_psnr(current_end, "val32", scope),
                                     _curves_psnr(base, "val32", scope))
            psnr_delta = now_psnr - before_psnr if now_psnr is not None and before_psnr is not None else None
            rows.append(f"| Object-Locus V1 vs {baseline_name} | {scope} | "
                        f"{_fmt(delta('mIoU_thing'), 5)} | "
                        f"{_fmt(delta('class_agnostic_recall50'), 5)} | "
                        f"{_fmt(delta('class_aware_recall50'), 5)} | {_fmt(psnr_delta, 3)} |")
    rows += ["", "### Val32 step-5000 GT-free mask matches", "",
             "| Scope | Class-agnostic TP/GT | Class-aware TP/GT | Active thing queries |", "|:---|:---|:---|---:|"]
    for scope in ("context", "target"):
        local = _curves_local(current_end, "val32", scope) if current_end else None
        if not local:
            rows.append(f"| {scope} | N/A | N/A | N/A |")
        else:
            rows.append(f"| {scope} | {local['tp_class_agnostic']}/{local['n_gt_instances']} | "
                        f"{local['tp_class_aware']}/{local['n_gt_instances']} | {_fmt(local.get('active_thing_queries'), 2)} |")
    rows += ["", "### Val32 step-5000 mechanism diagnostics", "",
             "| Scope | mean assignment entropy | ownership median / p90 / max / Gini | q cosine off-diagonal mean / p90 / max | evidence overlap mean / p90 / max | covariance PR q / projected-u | matched queries | supported GT | mean best anchor Dice |", "|:---|---:|:---|:---|:---|:---|---:|---:|---:|"]
    for scope in ("context", "target"):
        scope_rows = (current_end or {}).get("splits", {}).get("val32", {}).get("local_rows", {}).get(scope, [])
        diags = [r.get("anchor_group_diagnostics", {}) for r in scope_rows]
        if not diags:
            rows.append(f"| {scope} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        mean = lambda key: float(np.mean([d[key] for d in diags if d.get(key) is not None])) if any(d.get(key) is not None for d in diags) else None
        ownership = [d.get("ownership_mass", {}) for d in diags]
        qcos = [d.get("q_cosine", {}) for d in diags]
        overlap = [d.get("evidence_overlap", {}) for d in diags]
        own_txt = " / ".join(_fmt(np.mean([x[k] for x in ownership if k in x]), 4)
                             for k in ("median", "p90", "max", "gini"))
        q_txt = " / ".join(_fmt(np.mean([x[k] for x in qcos if k in x]), 4)
                           for k in ("offdiag_mean", "p90", "max"))
        ev_txt = " / ".join(_fmt(np.mean([x[k] for x in overlap if k in x]), 4)
                            for k in ("offdiag_mean", "p90", "max"))
        matched = float(np.mean([d.get("matched_query_count", 0) for d in diags]))
        rows.append(f"| {scope} | {_fmt(mean('assignment_entropy'),4)} | {own_txt} | {q_txt} | {ev_txt} | "
                    f"{_fmt(mean('q_covariance_participation_ratio'),3)} / {_fmt(mean('projected_u_covariance_participation_ratio'),3)} | "
                    f"{matched:.2f} | {sum(d.get('supported_gt_count',0) for d in diags)} | {_fmt(mean('gt_best_anchor_dice_mean'),4)} |")
    rows += ["", "## Interpretation", "",
             "Interpretation is limited to the registered monitor metrics. Direct anchor diagnostics and per-query output should be read alongside 2D mask matching; low utilization is diagnostic, not a training stop condition.",
             "No unregistered model, loss, threshold or training changes were introduced. No next experiment was started.", "",
             "## Artifact paths", "",
             f"- Reports: `{reports}`", f"- Run/checkpoints/log: `{RUN_ROOT_DEFAULT}`", "- Per-window evaluations and panels are stored under `eval_train16`, `eval_val8`, and `eval_val32`."]
    path = reports / "formal_5k_report.md"
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def run_train(device, reports, run_root, until_step):
    if until_step != TOTAL_STEPS:
        raise RuntimeError("the registered first-run endpoint is fixed at 5000 steps")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("formal Object-Locus V1 training requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"formal run requires one GPU, found {torch.cuda.device_count()}")
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA GeForce RTX 3090":
        raise RuntimeError(f"formal run is registered for RTX 3090, got {gpu}")
    reports, run_root = Path(reports), Path(run_root)
    if reports != REPORTS_DEFAULT or run_root != RUN_ROOT_DEFAULT:
        raise RuntimeError("formal artifact roots are fixed by the registered recipe")
    commit = _assert_committed_and_pushed()
    manifest, plan = locked_assets(reports)
    source_hashes = _source_hashes()
    if (run_root.exists() and not (run_root / "run_manifest.json").is_file()):
        leftovers = list(run_root.iterdir())
        # Failed preflight attempts may leave only the tee'd stdout log before
        # the run manifest is created. Preserve it, but never infer a resumable
        # training state from a log: any training marker or other file fails.
        unexpected = [path for path in leftovers if path.name != "train.log"]
        startup_log = run_root / "train.log"
        has_training_marker = False
        if startup_log.is_file():
            text = startup_log.read_text(errors="replace")
            has_training_marker = ("[start]" in text or "[resume]" in text or
                                  any('"step"' in line for line in text.splitlines()))
        if unexpected or has_training_marker:
            raise RuntimeError(f"run root exists without this task manifest: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = run_root / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text())
        if run_manifest.get("git_commit") != commit or run_manifest.get("source_file_hashes") != source_hashes:
            raise RuntimeError("run manifest belongs to different source/commit; refusing resume")
    else:
        run_manifest = {"task": "Object-Locus V1", "git_commit": commit,
                        "source_file_hashes": source_hashes, "spec_sha256": SPEC_SHA,
                        "gpu": gpu, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "manifest_sha256": MANIFEST_SHA, "plan_sha256": PLAN_SHA,
                        "pretrained_sha256": PRETRAINED_SHA,
                        "created_at_unix": time.time()}
        write_json(run_manifest_path, run_manifest)
    write_json(reports / "formal_run_provenance.json", run_manifest)

    seed_everything(42)
    model, opt, transfer = build_model(device)
    optimizer, optimizer_audit = build_optimizer(model)
    step0_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                   if not k.startswith("object_locus.")}
    trainability = trainability_counts(model)
    if trainability["frozen_numel"] != 0:
        raise RuntimeError("canonical pretrained model unexpectedly has frozen parameters")
    write_json(reports / "pretrained_transfer_audit.json", transfer)
    write_json(reports / "optimizer_audit.json", optimizer_audit)
    plan_windows = json.loads(MANIFEST.read_text())["windows"]
    monitors = {
        "train16": json.loads((reports / "monitor_train16.json").read_text())["windows"],
        "val8": json.loads((reports / "monitor_8pairs.json").read_text())["pairs"],
        "val32": json.loads((reports / "monitor_32pairs.json").read_text())["pairs"],
    }
    start_step, online_matches = 0, [0] * 100
    log_jsonl = reports / "training_log_metrics.jsonl"
    log_by_step = {}
    if log_jsonl.exists():
        for line in log_jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                log_by_step[int(row["step"])] = row
    checkpoint = _latest_checkpoint(run_root)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _verify_resume(state, commit, source_hashes)
        saved_groups = [{k: row[k] for k in ("name", "tensor_count", "numel", "weight_decay")}
                        for row in state["optimizer_groups"]]
        current_groups = [{k: row[k] for k in ("name", "tensor_count", "numel", "weight_decay")}
                          for row in _optimizer_audit(optimizer)]
        if saved_groups != current_groups:
            raise RuntimeError("resume optimizer groups differ from checkpoint provenance")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        if int(state["plan_position"]) != start_step:
            raise RuntimeError("resume plan position differs from checkpoint step")
        online_matches = list(state.get("online_matched_query_counts", online_matches))
        restore_rng(state["rng"])
        # A SLURM timeout can leave logs beyond the last atomic checkpoint. Those steps
        # are replayed from the checkpoint, so truncate only this task's own structured
        # JSONL records to keep the final per-step series unique and exact.
        log_by_step = {s: row for s, row in log_by_step.items() if s <= start_step}
        if log_jsonl.exists():
            log_jsonl.write_text("".join(json.dumps(log_by_step[s], allow_nan=False) + "\n"
                                                for s in sorted(log_by_step)))
        online_path = reports / "online_match_log.jsonl"
        if online_path.exists():
            online_rows = [json.loads(line) for line in online_path.read_text().splitlines() if line.strip()]
            online_rows = [row for row in online_rows if int(row["step"]) <= start_step]
            online_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in online_rows))
        del state
        print(f"[resume] verified step {start_step} from {checkpoint}", flush=True)
    log_path = run_root / "train_events.log"
    log_file = log_path.open("a", buffering=1)
    start_message = (f"[start] commit={commit} GPU={gpu} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                     f"torch={torch.__version__} CUDA={torch.version.cuda} start={start_step}")
    print(start_message, flush=True)
    print(start_message, file=log_file, flush=True)
    evaluations = {}
    try:
        if start_step in EVAL_STEPS and start_step > 0 and not (reports / f"curves_{start_step}.json").is_file():
            # Checkpoint publication precedes node evaluation. If a job ends inside
            # evaluation, replay that evaluation from the exact checkpoint before
            # consuming the next locked plan entry.
            pending_eval_checkpoint = _latest_checkpoint(run_root)
            evaluations[start_step] = _eval_node(
                model, opt, start_step, reports, device, monitors, run_root,
                pending_eval_checkpoint,
            )
        if start_step == 0:
            payload = _checkpoint_payload(model, optimizer, 0, 0, opt, commit, source_hashes,
                                          transfer, trainability, _optimizer_audit(optimizer), online_matches)
            path0 = _latest_checkpoint(run_root)
            if path0 is None:
                path0 = _save_checkpoint(run_root, payload, 0)
            if not (Path(reports) / "curves_0.json").is_file():
                state_count = len(optimizer.state)
                evaluations[0] = _eval_node(model, opt, 0, reports, device, monitors,
                                            run_root, path0)
                if len(optimizer.state) != state_count:
                    raise RuntimeError("step0 evaluation mutated optimizer state")
            else:
                existing = torch.load(path0, map_location="cpu", weights_only=False)
                _verify_resume(existing, commit, source_hashes)
                if int(existing["step"]) != 0 or int(existing["plan_position"]) != 0:
                    raise RuntimeError("step0 curve exists but latest checkpoint is not the initial state")
                del existing
        for step in range(start_step + 1, TOTAL_STEPS + 1):
            entry = plan["entries"][step - 1]
            batch = build_batch(opt, entry, device)
            t0 = time.time()
            output, metrics = train_one_step(model, optimizer, batch, step)
            for qi, _ki in output["prediction"].get("final_pairs", []):
                for query in qi.detach().cpu().tolist():
                    online_matches[int(query)] += 1
            if step % TRAIN_LOG_STEPS == 0:
                # The runtime returns detached scalar diagnostics for every component;
                # retain both unweighted terms and their registered weighted values.
                row = {key: value for key, value in metrics.items()
                       if isinstance(value, (int, float, np.integer, np.floating))}
                row.update({"step_seconds": time.time() - t0,
                            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                            "gpu_reserved_bytes": torch.cuda.memory_reserved()})
                if not all(math.isfinite(float(v)) for v in row.values() if isinstance(v, (int, float))):
                    raise FloatingPointError(f"nonfinite train log at step {step}: {row}")
                log_by_step[step] = row
                line = json.dumps(row, allow_nan=False)
                print(line, flush=True)
                print(line, file=log_file, flush=True)
                with log_jsonl.open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                with (reports / "online_match_log.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, "scene": entry["scene"],
                                        "window_index": entry["window_index"],
                                        "matched_gt_count": sum(len(x[0]) for x in output["prediction"].get("final_pairs", [])),
                                        "query_counts": online_matches}, allow_nan=False) + "\n")
            # Release the just-backpropagated graph before a registered evaluation node.
            del output, metrics, batch
            if step in EVAL_STEPS:
                payload = _checkpoint_payload(model, optimizer, step, step, opt, commit,
                                              source_hashes, transfer, trainability,
                                              _optimizer_audit(optimizer), online_matches)
                path = _save_checkpoint(run_root, payload, step)
                evaluations[step] = _eval_node(model, opt, step, reports, device,
                                               monitors, run_root, path)
                del payload
        torch.cuda.synchronize()
    finally:
        log_file.close()
    endpoint_path = _latest_checkpoint(run_root)
    endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    if int(endpoint["step"]) != 5000:
        raise RuntimeError(f"formal training ended at {endpoint['step']}, expected 5000")
    endpoint["trainability"] = trainability
    log_rows = [log_by_step[s] for s in sorted(log_by_step)]
    endpoint_audit, drift, summary = _write_endpoint_audits(
        model, opt, optimizer, reports, run_root, step0_state, endpoint, log_rows
    )
    write_json(reports / "online_match_utilization.json", {
        "scope": "online training Hungarian wins, not single-checkpoint evaluation utilization",
        "total_matches": sum(online_matches), "query_counts": online_matches,
        "unique_queries": sum(x > 0 for x in online_matches),
        "never_matched": sum(x == 0 for x in online_matches),
    })
    report_path = _write_formal_report(reports, drift, summary, online_matches)
    return {"commit": commit, "endpoint_audit": endpoint_audit,
            "drift": drift, "training_summary": summary,
            "evaluations": evaluations, "endpoint_path": str(endpoint_path),
            "formal_report": report_path}


def _optimizer_audit(optimizer):
    return [{"name": g["name"], "tensor_count": len(g["params"]),
             "numel": sum(p.numel() for p in g["params"]),
             "lr": g["lr"], "weight_decay": g["weight_decay"]}
            for g in optimizer.param_groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reports", type=Path, default=REPORTS_DEFAULT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--until-step", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()
    if args.until_step != TOTAL_STEPS:
        raise SystemExit("--until-step is fixed at 5000 for the registered first run")
    result = run_train(args.device, args.reports, args.run_root, args.until_step)
    print(json.dumps(jsonable(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
