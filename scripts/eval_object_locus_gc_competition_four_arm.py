"""Single-GPU one-pass inference and pinned-SIU3R four-arm endpoint export."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.object_locus_competition_gc001_runtime import (
    ARMS, REPORT_ROOT, RUN_ROOT, SOURCE_CHECKPOINT, SOURCE_EXPOSURES,
    SOURCE_MANIFEST, SOURCE_SHA256, build_batch,
    rank_world, sha256, write_json,
)
from scripts.object_locus_panoptic_v1_runtime import build_model as build_registered_model
from scripts.eval_object_locus_v3_set import _candidate_stats
from scripts.export_object_locus_v3_set_official import write_official_pair
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data


GC_REPORT = Path("/space/mawb/ssst/group_plus/object_locus_gc_sweep_v1")
GC_RUN = Path("/space/mawb/ssst/workspace_group_plus/object_locus_gc_sweep_v1")
ARMS_FOUR = ("gc001", "gc010", "gc100", "comp_gc001")
SPLITS = ("expanded_train_probe32", "train_all56", "same_scene_holdout8", "dev8", "val32")
SCOPES = ("context", "target-all", "true-novel")
SIU3R_COMMIT = "8ea80166be76854f938e90521f1a5b688b755c87"
OFFICIAL_PYTHON = os.environ.get(
    "TASK_OFFICIAL_PYTHON", "/space/mawb/SIU3R/.venv_gpu_v4/bin/python")


def tensor_state_sha(model) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8")); h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        t = value.detach().contiguous().cpu()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def checkpoint_rows(eval_root: Path):
    src_hash = sha256(SOURCE_CHECKPOINT)
    if src_hash != SOURCE_SHA256:
        raise RuntimeError(f"Full1201 source checkpoint SHA mismatch: {src_hash}")
    plan_path = REPORT_ROOT / "training_plan.json"
    source_manifest = REPORT_ROOT / "source_manifest.json"
    plan_sha, manifest_sha = sha256(plan_path), sha256(source_manifest)
    plan = json.loads(plan_path.read_text())
    manifest = json.loads(source_manifest.read_text())
    source_blob = torch.load(SOURCE_CHECKPOINT, map_location="cpu",
                             weights_only=False, mmap=True)
    source_config = source_blob.get("config")
    if not isinstance(source_config, dict):
        raise RuntimeError("source checkpoint does not contain the registered config")
    del source_blob
    rows = []
    for arm in ARMS_FOUR:
        path = (RUN_ROOT if arm == "comp_gc001" else GC_RUN) / arm / "checkpoint_epoch8.pt"
        digest = sha256(path)
        blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        alpha = 0.01 if arm in ("gc001", "comp_gc001") else (0.1 if arm == "gc010" else 1.0)
        expected_code = (json.loads((REPORT_ROOT / "git_provenance.json").read_text())["training_sha"]
                         if arm == "comp_gc001" else "9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0")
        checks = {
            "alpha": blob.get("alpha") == alpha,
            "epoch": blob.get("epoch") == 8,
            "completed_updates": blob.get("completed_updates") == 1008,
            "new_exposures": blob.get("new_exposures") == 8064,
            "source_exposure": blob.get("source_exposure") == 50064,
            "model_exposure": blob.get("model_exposure") == 58128,
            "plan_sha256": blob.get("plan_sha256") == plan_sha,
            "source_checkpoint_sha256": blob.get("source_checkpoint", {}).get("sha256") == SOURCE_SHA256,
            "code_sha": blob.get("code_sha") == expected_code,
            "world_size": blob.get("world_size") == 8,
            "config_matches_source": blob.get("config") == source_config,
        }
        if arm == "comp_gc001":
            checks["competition_lambda"] = blob.get("competition_lambda") == 2.0
            checks["arm"] = blob.get("arm") == "comp_gc001"
        if not all(checks.values()):
            raise RuntimeError(f"invalid {arm} endpoint: {checks}")
        rows.append({"arm": arm, "checkpoint": str(path), "sha256": digest,
                     "alpha": alpha, "checks": checks,
                     "config": blob["config"], "plan_sha256": plan_sha,
                     "source_manifest_sha256": manifest_sha,
                     "code_sha": blob["code_sha"]})
        del blob
    write_json(eval_root / "checkpoint_manifest.json", {
        "source_checkpoint": str(SOURCE_CHECKPOINT), "source_sha256": src_hash,
        "plan_sha256": plan_sha, "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": manifest_sha, "endpoints": rows,
    })
    return manifest, plan, rows


def set_metric_payload_cpu(payload, nviews):
    for branch in ("pred", "target"):
        masks = payload[branch]["masks"]
        if masks.shape[0]:
            payload[branch]["masks"] = masks.reshape(masks.shape[0], -1, masks.shape[-1]).cpu()
        else:
            payload[branch]["masks"] = torch.zeros((0, nviews * 256, 256), dtype=torch.bool)
        for key, value in payload[branch].items():
            if torch.is_tensor(value):
                payload[branch][key] = value.cpu()


def invoke_official(eval_path: Path, out_path: Path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([
        OFFICIAL_PYTHON,
        str(Path(__file__).resolve().parents[1] / "scripts/invoke_siu3r_official_evaluator.py"),
        "--eval-path", str(eval_path), "--output", str(out_path),
        "--device", "cpu", "--no-image-depth",
    ], check=True, env=env)
    envelope = json.loads(out_path.read_text())
    if (envelope.get("official_evaluator_used") is not True or
            envelope.get("siu3r_commit") != SIU3R_COMMIT):
        raise RuntimeError(f"official evaluator provenance mismatch: {out_path}")


def run_interface_smoke(device, smoke_root: Path):
    if smoke_root.exists() and any(smoke_root.iterdir()):
        raise RuntimeError(f"smoke output already exists; refusing overwrite: {smoke_root}")
    smoke_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPORT_ROOT / "source_manifest.json").read_text())
    dev = {str(w["scene"]) for w in manifest["dev8"]}
    windows = [w for w in manifest["val32"] if str(w["scene"]) not in dev]
    if len(windows) != 24:
        raise RuntimeError("interface smoke requires the registered val32-excluding-dev8 cohort")
    window = windows[0]
    _, _, endpoints = checkpoint_rows(smoke_root)
    smoke_rows = []
    for endpoint in endpoints:
        arm = endpoint["arm"]
        model, opt = build_registered_model(device, report=False)
        blob = torch.load(endpoint["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(blob["model"], strict=True)
        del blob
        model.understanding_step = 58128
        model.eval()
        before = tensor_state_sha(model)
        batch = build_batch(opt, window, device)
        model_input, _ = split_data(batch, opt)
        decoder = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
        with torch.no_grad():
            out = model.forward_object_locus(ModelInput(model_input.encoder, decoder),
                    render_decoder_input=decoder, context_decoder=decoder, step=58128)
        beta = float(torch.as_tensor(out["beta"]).detach().cpu())
        if not math.isclose(beta, 0.1, abs_tol=1e-6, rel_tol=1e-6):
            raise RuntimeError(f"{arm} interface smoke beta mismatch: {beta}")
        for key in ("gaussians", "region_mass", "alpha", "p_class", "semantic_scores"):
            value = out.get(key)
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"{arm} interface smoke nonfinite {key}")
        for key in ("images_pred", "depths_pred"):
            value = out["render"].get(key)
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"{arm} interface smoke nonfinite render {key}")
        if "images_pred" not in out["render"] or out["render"]["images_pred"].shape[0] != 1:
            raise RuntimeError(f"{arm} interface smoke missing batch-one RGB render")
        frame_ids = [int(v) for v in batch["frame_ids"][0].detach().cpu().tolist()]
        novel_ids = set(map(int, window["novel"]))
        scopes = {"context": [0, 1], "target-all": list(range(len(frame_ids))),
                  "true-novel": [i for i, fid in enumerate(frame_ids) if fid in novel_ids]}
        candidate = {}
        for scope, ids in scopes.items():
            stats = _candidate_stats(out, batch, ids)
            payload = stats.pop("_map_payload")
            if not payload or "pred" not in payload or "target" not in payload:
                raise RuntimeError(f"{arm} candidate interface failed for {scope}")
            candidate[scope] = {"candidate_count": stats["candidate_count"],
                                "gt_count": stats["gt_count"]}
        arm_root = smoke_root / arm
        write_official_pair(out, batch, window, arm_root / "official_all", target_frames="all")
        write_official_pair(out, batch, window, arm_root / "official_novel", target_frames="novel")
        after = tensor_state_sha(model)
        if after != before:
            raise RuntimeError(f"{arm} interface smoke mutated model state")
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError(f"{arm} interface smoke populated model gradients")
        smoke_rows.append({"arm": arm, "checkpoint": endpoint["checkpoint"],
                           "window": {"scene": window["scene"], "context": window["context"],
                                      "novel": window["novel"]}, "beta": beta,
                           "context_indices": [0, 1], "candidate_scopes": candidate,
                           "state_sha256_before_after": before, "state_unchanged": True,
                           "gradients_none": True, "official_all_export": True,
                           "official_novel_export": True})
        del model, opt, batch, out, decoder, model_input
        torch.cuda.empty_cache()
    write_json(smoke_root / "interface_smoke.json", {
        "status": "PASS", "four_endpoints": smoke_rows,
        "no_single_window_metric_interpretation": True})


def run_inference(device, eval_root: Path, evaluation_code_sha: str):
    if eval_root.exists() and any(eval_root.iterdir()):
        raise RuntimeError(f"evaluation output already exists; refusing overwrite: {eval_root}")
    manifest, plan, checkpoints = checkpoint_rows(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    dev_scenes = {str(w["scene"]) for w in manifest["dev8"]}
    val_scenes = {str(w["scene"]) for w in manifest["val32"]}
    if len(dev_scenes) != 8 or not dev_scenes.issubset(val_scenes):
        raise RuntimeError("dev8 is not exactly an 8-scene subset of val32")
    train_scenes = {str(w["scene"]) for w in manifest["expanded_train_windows"]}
    cohort24 = [w for w in manifest["val32"] if str(w["scene"]) not in dev_scenes]
    if len(cohort24) != 24 or len({str(w["scene"]) for w in cohort24}) != 24:
        raise RuntimeError("val32 excluding dev8 must be exactly 24 scene/windows")
    if {str(w["scene"]) for w in cohort24} & train_scenes:
        raise RuntimeError("primary val32 cohort overlaps training scenes")
    write_json(eval_root / "cohort_manifest.json", {
        "dev8_scenes": sorted(dev_scenes), "val32_scenes": sorted(val_scenes),
        "val32_excluding_dev8_scenes": [
            {"scene": w["scene"], "context": w["context"], "novel": w["novel"]}
            for w in cohort24],
        "val32_excluding_dev8_scenes_sha256": hashlib.sha256(
            json.dumps(cohort24, sort_keys=True).encode()).hexdigest(),
        "source_manifest_sha256": sha256(REPORT_ROOT / "source_manifest.json"),
    })
    split_windows = {s: manifest[s] for s in SPLITS}
    split_windows["val32_excluding_dev8_scenes"] = cohort24
    all_window_ids = {}
    per_window, per_gt, per_query = [], [], []
    local_summary = {}
    for endpoint in checkpoints:
        arm = endpoint["arm"]
        model, opt = build_registered_model(device, report=False)
        blob_path = Path(endpoint["checkpoint"])
        blob = torch.load(blob_path, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(blob["model"], strict=True)
        del blob
        model.understanding_step = 58128
        model.eval()
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError(f"{arm} has nonempty gradients before evaluation")
        before_hash = tensor_state_sha(model)
        arm_results = {}
        for split, windows in split_windows.items():
            split_root = eval_root / arm / split
            all_root, novel_root = split_root / "official_all", split_root / "official_novel"
            all_root.mkdir(parents=True, exist_ok=True)
            novel_root.mkdir(parents=True, exist_ok=True)
            split_rows = []
            ap_metrics = {}
            try:
                from torchmetrics.detection.mean_ap import MeanAveragePrecision
                ap_metrics = {s: MeanAveragePrecision(iou_type="segm", sync_on_compute=False)
                              for s in ("context", "target-all", "true-novel")}
            except Exception as exc:
                raise RuntimeError(f"candidate MeanAveragePrecision unavailable: {exc}") from exc
            ids_for_split = []
            for wi, window in enumerate(windows):
                batch = build_batch(opt, window, device)
                model_input, _ = split_data(batch, opt)
                decoder = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                            intrinsics=batch["intrinsics_all"])
                with torch.no_grad():
                    out = model.forward_object_locus(
                        ModelInput(model_input.encoder, decoder),
                        render_decoder_input=decoder, context_decoder=decoder,
                        step=58128)
                for key in ("gaussians", "region_mass", "alpha", "p_class", "semantic_scores"):
                    value = out.get(key)
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise FloatingPointError(f"{arm} produced nonfinite {key} in {split}")
                for key in ("images_pred", "depths_pred"):
                    value = out["render"].get(key)
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise FloatingPointError(f"{arm} produced nonfinite rendered {key} in {split}")
                beta = float(torch.as_tensor(out["beta"]).detach().cpu())
                if not math.isfinite(beta) or not math.isclose(beta, 0.1, abs_tol=1e-6, rel_tol=1e-6):
                    raise RuntimeError(f"{arm} forward beta/exposure mismatch")
                for state in out["states"]:
                    if not torch.isclose(torch.as_tensor(state["beta"]).float(),
                                         torch.tensor(0.1), atol=1e-6, rtol=1e-6):
                        raise RuntimeError(f"{arm} feedback beta mismatch")
                frame_ids = [int(v) for v in batch["frame_ids"][0].detach().cpu().tolist()]
                novel_ids = set(map(int, window["novel"]))
                view_sets = {"context": [0, 1],
                             "target-all": list(range(len(frame_ids))),
                             "true-novel": [i for i, fid in enumerate(frame_ids) if fid in novel_ids]}
                scope_rows = {}
                for scope, view_ids in view_sets.items():
                    row = _candidate_stats(out, batch, view_ids)
                    payload = row.pop("_map_payload")
                    set_metric_payload_cpu(payload, len(view_ids))
                    ap_metrics[scope].update([payload["pred"]], [payload["target"]])
                    rec = {
                        "arm": arm, "split": split, "scene": str(window["scene"]),
                        "context_ids": json.dumps(list(map(int, window["context"]))),
                        "novel_ids": json.dumps(list(map(int, window["novel"]))),
                        "scope": scope, "view_count": len(view_ids),
                        "miou": row["semantic_miou"], "pq": row["panoptic_pq"],
                        "candidate_count": row["candidate_count"], "gt_count": row["gt_count"],
                        "candidate_ca": json.dumps(row["candidate_ca"]),
                        "candidate_cw": json.dumps(row["candidate_cw"]),
                        "raw_mask_iou_ge_0_5_fraction": row["raw_best_iou_ge_0_5_fraction"],
                        "raw_mask_iou_ge_0_75_count": sum(v >= .75 for v in row["raw_best_ious"]),
                        "raw_mask_iou_gt_count": len(row["raw_best_ious"]),
                        "packed_candidate_ca": json.dumps(row["panoptic_ca"]),
                        "packed_candidate_cw": json.dumps(row["panoptic_cw"]),
                        "eligible_query_count": sum(bool(q["joint_eligible"]) for q in row["query_rows"]),
                        "packed_instance_count": len({q["query_id"] for q in row["query_rows"] if q["panoptic_retained"]}),
                        "context_matched_class_accuracy": row["matched_19_class_accuracy"],
                        "matched_gt_count": row["matched_gt_count"],
                        "candidate_ap": "aggregated_per_split",
                    }
                    scope_rows[scope] = rec
                    per_window.append(rec)
                    per_gt.extend({"arm": arm, "split": split, "scene": str(window["scene"]),
                                   "scope": scope, **g} for g in row["per_gt"])
                    per_query.extend({"arm": arm, "split": split, "scene": str(window["scene"]),
                                      "scope": scope, **q} for q in row["query_rows"])
                packed_all = write_official_pair(out, batch, window, all_root,
                                                 target_frames="all")
                packed_novel = write_official_pair(out, batch, window, novel_root,
                                                   target_frames="novel")
                cache_rel = Path("reconstruction_cache") / arm / split / (
                    str(window["scene"]) + "_context" + "_".join(map(str, window["context"])) + ".npz")
                cache_path = eval_root / cache_rel
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                render_depth = out["render"]["depths_pred"]
                if render_depth.ndim == 5:
                    render_depth = render_depth[:, :, 0]
                np.savez_compressed(
                    cache_path,
                    frame_ids=np.asarray(frame_ids, dtype=np.int64),
                    context_ids=np.asarray(window["context"], dtype=np.int64),
                    novel_ids=np.asarray(window["novel"], dtype=np.int64),
                    pred_rgb=out["render"]["images_pred"][0].detach().float().clamp(0, 1).cpu().numpy(),
                    gt_rgb=batch["images_all"][0].detach().float().clamp(0, 1).cpu().numpy(),
                    pred_depth=render_depth[0].detach().float().cpu().numpy(),
                    gt_depth_m=batch["depth_gt_m_all"][0, :, 0].detach().float().cpu().numpy(),
                    depth_valid=batch["depth_gt_valid_all"][0, :, 0].detach().bool().cpu().numpy(),
                )
                ids_for_split.append({"scene": str(window["scene"]),
                                      "context": list(map(int, window["context"])),
                                      "novel": list(map(int, window["novel"])),
                                      "cache": str(cache_path),
                                      "cache_sha256": sha256(cache_path)})
                split_rows.append(scope_rows)
                del batch, out, decoder, model_input
                torch.cuda.empty_cache()
            local = {}
            for scope, metric in ap_metrics.items():
                value = metric.compute()
                local[scope] = {"map": float(value["map"]),
                                "map_50": float(value["map_50"]),
                                "candidate_ap_is_local_not_official": True}
                metric.reset()
            arm_results[split] = {"window_count": len(windows), "local_candidate_ap": local,
                                  "windows": ids_for_split}
            all_official = split_root / "official_all.json"
            novel_official = split_root / "official_novel.json"
            invoke_official(all_root, all_official)
            invoke_official(novel_root, novel_official)
            arm_results[split]["official_all"] = json.loads(all_official.read_text())
            arm_results[split]["official_novel"] = json.loads(novel_official.read_text())
            write_json(split_root / "split_result.json", arm_results[split])
        after_hash = tensor_state_sha(model)
        if after_hash != before_hash:
            raise RuntimeError(f"{arm} model state changed during inference")
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError(f"{arm} evaluation populated model gradients")
        arm_results["state_dict_sha256_before_after"] = {"before": before_hash,
                                                           "after": after_hash,
                                                           "equal": True}
        write_json(eval_root / arm / "evaluation_arm.json", arm_results)
        del model, opt
        torch.cuda.empty_cache()
    identity_path = eval_root / "window_identities.json"
    write_json(identity_path, {arm: json.loads((eval_root / arm / "evaluation_arm.json").read_text())
                               for arm in ARMS_FOUR})
    for name, rows in (("per_window.csv", per_window), ("per_gt.csv", per_gt),
                       ("per_query.csv", per_query)):
        path = eval_root / name
        keys = sorted({k for row in rows for k in row})
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    provenance = {
        "status": "INFERENCE_AND_OFFICIAL_EXPORT_COMPLETE",
        "arms": list(ARMS_FOUR), "splits": list(SPLITS) + ["val32_excluding_dev8_scenes"],
        "scopes": list(SCOPES), "one_forward_per_window": True,
        "exposure": 58128, "feedback_beta": 0.1,
        "official_evaluator_commit": SIU3R_COMMIT,
        "official_python": OFFICIAL_PYTHON,
        "task_python": sys.executable, "tf32": False,
        "evaluation_code_sha": evaluation_code_sha,
        "checkpoint_training_code_sha": {r["arm"]: r["code_sha"] for r in checkpoints},
        "source_manifest_sha256": sha256(REPORT_ROOT / "source_manifest.json"),
        "plan_sha256": sha256(REPORT_ROOT / "training_plan.json"),
        "window_identity_file": str(identity_path), "window_identity_sha256": sha256(identity_path),
    }
    write_json(eval_root / "provenance.json", provenance)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--interface-smoke", action="store_true")
    parser.add_argument("--smoke-root", type=Path)
    args = parser.parse_args()
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) != 1:
        raise RuntimeError("four-arm evaluation is single-process, single-GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("four-arm model inference requires the assigned GPU")
    siu3r_head = subprocess.check_output(
        ["git", "-C", "/space/mawb/SIU3R", "rev-parse", "HEAD"], text=True).strip()
    if siu3r_head != SIU3R_COMMIT:
        raise RuntimeError(f"SIU3R checkout mismatch: {siu3r_head}")
    if Path(OFFICIAL_PYTHON).resolve() != Path(
            "/space/mawb/SIU3R/.venv_gpu_v4/bin/python").resolve():
        raise RuntimeError(f"wrong pinned SIU3R interpreter: {OFFICIAL_PYTHON}")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.interface_smoke:
        if args.smoke_root is None:
            raise RuntimeError("--interface-smoke requires --smoke-root")
        run_interface_smoke(torch.device("cuda", 0), args.smoke_root)
    else:
        sha = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                       "rev-parse", "HEAD"], text=True).strip()
        run_inference(torch.device("cuda", 0), args.eval_root, sha)


if __name__ == "__main__":
    main()
