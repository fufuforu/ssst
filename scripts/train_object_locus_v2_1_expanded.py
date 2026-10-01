"""Fixed expanded-data continuation of Object-Locus V2.1 Stage S."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.object_locus_v2_1_runtime import (
    ASSET_ROOT, MANIFEST, MANIFEST_SHA, MONITOR_SHA, OBJECT_PEAK_LR, PRETRAINED_SHA,
    RECON_PEAK_LR, REPORTS_DEFAULT, RUN_ROOT_DEFAULT, build_batch, build_model,
    build_optimizer, capture_rng, jsonable, restore_rng, sha256_file, trainability_counts,
    train_one_step, write_json,
)

SOURCE_STAGE_S = Path("/space/mawb/ssst/group_plus/object_locus_v2_1")
SOURCE_CHECKPOINT = Path("/space/mawb/ssst/workspace_group_plus/object_locus_v2_1/stage_s/checkpoint_S_epoch_16.pt")
SOURCE_CHECKPOINT_SHA = "8461937d92965ad10bf56b401a12d9012862b2eeb5330113a15ac9998419450d"
SOURCE_DATA_MANIFEST_SHA = "ebed1a133d64ed38ef7afce17aaaf27bbe65c6b0edea950c229b5d1e4ff77bf0"
SOURCE_GIT_SHA = "d9a5cef3263dc7b16ad79784560b3c30d2045b09"
REPORTS = Path("/space/mawb/ssst/group_plus/object_locus_v2_1_expanded")
RUN = Path("/space/mawb/ssst/workspace_group_plus/object_locus_v2_1_expanded")
SOURCE_RUN = Path("/space/mawb/ssst/workspace_group_plus/object_locus_v2_1")
SOURCE_REPORTS = Path("/space/mawb/ssst/group_plus/object_locus_v2_1")
SOURCE_MANIFEST_PATH = SOURCE_REPORTS / "data_manifest.json"
SOURCE_GATE = SOURCE_REPORTS / "stage_s_expansion_gate.json"
SOURCE_EPOCH16 = SOURCE_REPORTS / "stage_s/stage_s_epoch16.json"
EXPECTED_MODEL = "LOCUSGS_OBJECT_LOCUS_V2_1"
EXPECTED_RECIPE = "OBJECT_LOCUS_V2_1_EXPANDED_RECON_LR_1E6"
EPOCHS = 16
WINDOWS_PER_EPOCH = 1008
TOTAL_NEW_UPDATES = EPOCHS * WINDOWS_PER_EPOCH
INITIAL_GLOBAL_STEP = 1792
FINAL_GLOBAL_STEP = INITIAL_GLOBAL_STEP + TOTAL_NEW_UPDATES
EVAL_EPOCHS = (0, 2, 4, 8, 16)
EVAL_SPLITS = ("expanded_train_probe32", "train_probe16", "same_scene_holdout16", "dev8",
               "legacy_train16", "val8", "val32")
PANEL_SPLITS = set(EVAL_SPLITS)


def expanded_lr_multiplier(step: int) -> float:
    if not 1 <= int(step) <= TOTAL_NEW_UPDATES:
        raise ValueError(f"expanded step must be within [1,{TOTAL_NEW_UPDATES}]")
    u = (int(step) - 1) / (TOTAL_NEW_UPDATES - 1)
    return 0.1 + 0.9 * (1.0 + math.cos(math.pi * u)) / 2.0


def expanded_lrs(step: int) -> tuple[float, float]:
    multiplier = expanded_lr_multiplier(step)
    return OBJECT_PEAK_LR * multiplier, 1e-6 * multiplier


def window_identity(row):
    return (str(row["scene"]), tuple(map(int, row["context"])), tuple(map(int, row["novel"])))


def _probe32(windows):
    by_scene = {}
    for index, row in enumerate(windows):
        by_scene.setdefault(row["scene"], []).append((index, row))
    scenes = sorted(by_scene)
    chosen = []
    for scene_pos in range(0, len(scenes), 4):
        scene = scenes[scene_pos]
        candidates = sorted(by_scene[scene], key=lambda pair: (
            tuple(pair[1]["context"]), tuple(pair[1]["novel"]), pair[0]))
        row = dict(candidates[0][1])
        row["expanded_pool_index"] = candidates[0][0]
        chosen.append(row)
    if len(chosen) != 32:
        raise RuntimeError(f"expanded_train_probe32 must contain 32 windows, got {len(chosen)}")
    return chosen


def expanded_epoch_order(epoch_index, n=WINDOWS_PER_EPOCH):
    return np.random.default_rng(10042 + int(epoch_index)).permutation(int(n)).tolist()


def build_expanded_plan(manifest):
    windows = manifest["expanded_train_windows"]
    if len(windows) != WINDOWS_PER_EPOCH or len({w["scene"] for w in windows}) != 128:
        raise RuntimeError("locked expanded pool must be 1008 windows / 128 scenes")
    epochs = []
    entries = []
    step = 0
    for epoch in range(EPOCHS):
        order = expanded_epoch_order(epoch, len(windows))
        epochs.append({"epoch_index": epoch, "permutation": order})
        for position, pool_index in enumerate(order):
            win = windows[pool_index]
            step += 1
            entries.append({"expanded_step": step, "epoch_index": epoch,
                "epoch": epoch + 1, "position": position, "pool_index": pool_index,
                "window_identity": {"scene": win["scene"], "context": win["context"], "novel": win["novel"]},
                "scene": win["scene"], "context": win["context"], "novel": win["novel"]})
    if step != TOTAL_NEW_UPDATES:
        raise RuntimeError(f"plan updates mismatch: {step}")
    return {"recipe": EXPECTED_RECIPE, "source_stage_s_global_step": INITIAL_GLOBAL_STEP,
        "epochs": EPOCHS, "windows_per_epoch": len(windows), "new_updates": step,
        "initial_global_optimizer_step": INITIAL_GLOBAL_STEP,
        "final_global_optimizer_step": FINAL_GLOBAL_STEP,
        "permutation_seed": "10042 + epoch_index", "epoch_permutations": epochs,
        "entries": entries}


def _assert_source_checkpoint(state):
    required = {"stage": "S", "epoch": 16, "stage_step": 1792,
                "global_optimizer_step": 1792, "git_sha": SOURCE_GIT_SHA,
                "architecture_name": EXPECTED_MODEL}
    for key, value in required.items():
        if state.get(key) != value:
            raise RuntimeError(f"source checkpoint {key} mismatch: {state.get(key)!r} != {value!r}")
    if state.get("source_hashes", {}).get("pretrained") != PRETRAINED_SHA:
        raise RuntimeError("source checkpoint pretrained asset SHA mismatch")
    if state.get("source_hashes", {}).get("manifest") != MANIFEST_SHA:
        raise RuntimeError("source checkpoint training-manifest SHA mismatch")
    if state.get("source_hashes", {}).get("monitor") != dict(MONITOR_SHA):
        raise RuntimeError("source checkpoint monitor asset SHA mismatch")
    if state.get("data_manifest_sha256") != SOURCE_DATA_MANIFEST_SHA:
        raise RuntimeError("source checkpoint data manifest SHA mismatch")
    if state.get("training_plan_sha256") != sha256_file(SOURCE_REPORTS / "training_plan.json"):
        raise RuntimeError("source checkpoint source-plan SHA mismatch")
    if not isinstance(state.get("rng"), dict) or not all(k in state["rng"] for k in ("python", "numpy", "torch", "cuda")):
        raise RuntimeError("source checkpoint lacks complete RNG state")


def _equal_value(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal_value(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    try:
        return a == b
    except Exception:
        return False


def _assert_optimizer_state_exact(source_optimizer, optimizer):
    actual = optimizer.state_dict()
    expected = copy.deepcopy(source_optimizer)
    if len(actual["param_groups"]) != len(expected["param_groups"]):
        raise RuntimeError("optimizer group count changed while restoring Stage S")
    for old, new in zip(expected["param_groups"], actual["param_groups"]):
        if old.get("name") != new.get("name"):
            raise RuntimeError("optimizer parameter-group mapping changed")
        for key in set(old) | set(new):
            if key == "lr":
                continue
            if key not in old or key not in new or not _equal_value(old[key], new[key]):
                raise RuntimeError(f"optimizer group field changed during restore: {old.get('name')}:{key}")
    if not _equal_value(expected["state"], actual["state"]):
        raise RuntimeError("optimizer moments/internal steps differ from source checkpoint")
    step_values = sorted({float(v["step"].item()) for v in actual["state"].values()
                          if torch.is_tensor(v.get("step"))})
    return {"state_entries": len(actual["state"]),
            "internal_step_values": step_values,
            "groups": [{"name": g["name"], "tensor_count": len(g["params"])}
                       for g in actual["param_groups"]]}


def _assert_optimizer_layout_same(source_audit, new_audit):
    old = source_audit.get("groups", []); new = new_audit.get("groups", [])
    if len(old) != 4 or len(new) != 4:
        raise RuntimeError("Stage S optimizer must have exactly four parameter groups")
    for a, b in zip(old, new):
        for key in ("name", "tensor_count", "numel", "weight_decay"):
            if a.get(key) != b.get(key):
                raise RuntimeError(f"optimizer mapping differs at {a.get('name')}:{key}: {a.get(key)} != {b.get(key)}")


def _assert_rng_exact(expected, actual):
    if not _equal_value(expected, actual):
        raise RuntimeError("RNG state differs immediately after source checkpoint restore")


def _set_lrs(optimizer, values):
    object_lr, recon_lr = values
    for group in optimizer.param_groups:
        if group["name"].startswith("object_locus_v2_1_"):
            group["lr"] = object_lr
        elif group["name"].startswith("reconstruction_"):
            group["lr"] = recon_lr
        else:
            raise RuntimeError(f"unexpected optimizer group: {group['name']}")


def _assert_backend_settings():
    source = json.loads((SOURCE_REPORTS / "run_manifest.json").read_text())
    registered = source.get("tf32", {})
    actual = {"cudnn": bool(torch.backends.cudnn.allow_tf32),
              "matmul": bool(torch.backends.cuda.matmul.allow_tf32)}
    if registered != {"cudnn": True, "matmul": False} or actual != registered:
        raise RuntimeError(f"TF32 backend settings differ from Stage S: source={registered}, current={actual}")
    if torch.are_deterministic_algorithms_enabled() is not False or source.get("deterministic_algorithms") is not False:
        raise RuntimeError("deterministic-algorithm setting differs from Stage S")
    return {"tf32": actual, "deterministic_algorithms": False}


def _checkpoint(path, model, optimizer, *, cursor, plan_sha, source_sha, exec_sha,
                data_manifest_sha, exposure, optimizer_info):
    rng_before = capture_rng()
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": rng_before, "stage": "E", "epoch": int(cursor["completed_epochs"]),
        "stage_step": int(cursor["expanded_step"]), "expanded_step": int(cursor["expanded_step"]),
        "global_optimizer_step": INITIAL_GLOBAL_STEP + int(cursor["expanded_step"]),
        "next_epoch": int(cursor["next_epoch"]), "next_position": int(cursor["next_position"]),
        "plan_position": int(cursor["next_position"]), "git_sha": exec_sha,
        "architecture_name": EXPECTED_MODEL, "recipe": EXPECTED_RECIPE,
        "source_checkpoint": str(SOURCE_CHECKPOINT), "source_checkpoint_sha256": source_sha,
        "source_git_sha": SOURCE_GIT_SHA, "execution_git_sha": exec_sha,
        "data_manifest_sha256": data_manifest_sha, "training_plan_sha256": plan_sha,
        "optimizer_config": optimizer_info, "optimizer_current_lrs": [g["lr"] for g in optimizer.param_groups],
        "exposure_stats": exposure}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp); os.replace(temp, path)
    _assert_rng_exact(rng_before, capture_rng())


def _checkpoint_and_latest(model, optimizer, run, *, endpoint_epoch, cursor, **kwargs):
    latest = Path(run) / "latest_recovery.pt"
    if endpoint_epoch is not None:
        endpoint = Path(run) / f"checkpoint_E_epoch_{endpoint_epoch:02d}.pt"
        _checkpoint(endpoint, model, optimizer, cursor=cursor, **kwargs)
        tmp_link = latest.with_suffix(".tmp")
        if tmp_link.exists() or tmp_link.is_symlink(): tmp_link.unlink()
        tmp_link.symlink_to(endpoint.name); os.replace(tmp_link, latest)
        return endpoint
    _checkpoint(latest, model, optimizer, cursor=cursor, **kwargs)
    return latest


def _restore_own_checkpoint(path, model, optimizer, *, exec_sha, plan_sha, data_manifest_sha, source_sha):
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"git_sha": exec_sha, "execution_git_sha": exec_sha,
        "architecture_name": EXPECTED_MODEL, "recipe": EXPECTED_RECIPE,
        "training_plan_sha256": plan_sha, "data_manifest_sha256": data_manifest_sha,
        "source_checkpoint_sha256": source_sha, "source_git_sha": SOURCE_GIT_SHA}
    for key, value in expected.items():
        if state.get(key) != value: raise RuntimeError(f"expanded checkpoint resume mismatch: {key}")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state


def _source_eval_path(split_name, global_step):
    return SOURCE_REPORTS / "stage_s" / f"eval_{split_name}_step{int(global_step):04d}.json"


def _stage_s_reference(split_name):
    path = _source_eval_path(split_name, INITIAL_GLOBAL_STEP)
    if not path.is_file(): raise RuntimeError(f"missing registered Stage S endpoint artifact: {path}")
    return json.loads(path.read_text()), {"path": str(path), "sha256": sha256_file(path), "source_step": 1792}


def _copy_source_panels(reports, split_names):
    source_root = SOURCE_REPORTS / "stage_s/qualitative/step_1792"
    target_root = Path(reports) / "stage_e/qualitative/step_1792"
    copied = []
    for split in split_names:
        src_dir = source_root / split
        if not src_dir.is_dir():
            raise RuntimeError(f"missing Stage S16 fixed-window panels for {split}: {src_dir}")
        dst_dir = target_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.png"):
            dst = dst_dir / src.name
            if dst.exists() and sha256_file(dst) != sha256_file(src):
                raise RuntimeError(f"E0 qualitative collision: {dst}")
            if not dst.exists(): shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def _eval_split_set(model, opt, splits, reports, global_step, *, official, panels):
    from scripts.eval_object_locus_v2_1 import evaluate_windows
    results = {}
    signatures = {}
    aliases = {}
    was_training = model.training
    for name in EVAL_SPLITS:
        windows = splits[name]
        signature = tuple(window_identity(w) for w in windows)
        if signature in signatures:
            source_name = signatures[signature]
            result = copy.deepcopy(results[source_name]); result["split"] = name
            write_json(Path(reports) / f"eval_{name}_step{global_step:04d}.json", result)
            aliases[name] = {"identical_to": source_name, "window_identities_exact": True}
        else:
            result = evaluate_windows(model, opt, windows, global_step, name, reports,
                "cuda", build_batch, official=official, panels=panels and name in PANEL_SPLITS)
            signatures[signature] = name
        results[name] = result
    model.train(was_training)
    return {"stage": "E", "global_optimizer_step": global_step, "official": bool(official),
            "splits": results, "identical_split_aliases": aliases}


def _write_e0_reference(model, opt, splits, reports, run, *, source_sha, exec_sha,
                        data_manifest_sha, plan_sha, optimizer, optimizer_info, exposure):
    from scripts.eval_object_locus_v2_1 import evaluate_windows
    references = {}
    split_results = {}
    for name in ("train_probe16", "same_scene_holdout16", "dev8", "legacy_train16", "val8", "val32"):
        split_results[name], references[name] = _stage_s_reference(name)
        write_json(Path(reports) / "stage_e" / f"eval_{name}_step{INITIAL_GLOBAL_STEP:04d}.json",
                   split_results[name])
    # This newly registered split has no Stage S counterpart and receives an E0 local + official evaluation.
    probe = evaluate_windows(model, opt, splits["expanded_train_probe32"], INITIAL_GLOBAL_STEP,
        "expanded_train_probe32", Path(reports) / "stage_e", "cuda", build_batch,
        official=True, panels=True)
    split_results["expanded_train_probe32"] = probe
    write_json(Path(reports) / "stage_e/curves_e_epoch00.json", {
        "stage": "E", "epoch": 0, "expanded_step": 0,
        "global_optimizer_step": INITIAL_GLOBAL_STEP, "official": True,
        "splits": split_results, "source_stage_s_references": references,
        "expanded_train_probe32_e0_new_evaluation": True})
    _copy_source_panels(Path(reports), ("train_probe16", "same_scene_holdout16", "dev8", "val32"))
    endpoint = Path(reports) / "stage_e/expanded_train_probe32_e0_reference.json"
    write_json(endpoint, {"stage": "E", "epoch": 0, "global_optimizer_step": INITIAL_GLOBAL_STEP,
        "new_split": "expanded_train_probe32", "new_evaluation": True,
        "source_stage_s_references": references})
    _checkpoint_and_latest(model, optimizer, run, endpoint_epoch=0,
        cursor={"completed_epochs": 0, "expanded_step": 0, "next_epoch": 0, "next_position": 0},
        plan_sha=plan_sha, source_sha=source_sha, exec_sha=exec_sha,
        data_manifest_sha=data_manifest_sha, exposure=exposure, optimizer_info=optimizer_info)


def _write_run_manifest(reports, run, *, exec_sha, source_sha, plan_sha,
                        data_manifest_sha, split_manifest, optimizer_info, trainability, gpu, node):
    payload = {"architecture_name": EXPECTED_MODEL, "recipe": EXPECTED_RECIPE,
        "source_checkpoint": str(SOURCE_CHECKPOINT), "source_checkpoint_sha256": source_sha,
        "source_git_sha": SOURCE_GIT_SHA, "execution_git_sha": exec_sha,
        "data_manifest_sha256": data_manifest_sha, "plan_sha256": plan_sha,
        "source_data_manifest_sha256": SOURCE_DATA_MANIFEST_SHA,
        "source_stage_s_gate_passed": False, "expanded_training_authorized_by_this_spec": True,
        "model_and_loss_unchanged": True, "optimizer_state_continued": True,
        "reconstruction_peak_lr_changed_from": 1e-5, "reconstruction_peak_lr": 1e-6,
        "object_peak_lr": 1e-4, "gc_alpha": 0.01, "understanding_weight": 1.0,
        "expanded_windows": WINDOWS_PER_EPOCH, "expanded_scenes": 128, "epochs": EPOCHS,
        "new_updates": TOTAL_NEW_UPDATES, "initial_global_step": INITIAL_GLOBAL_STEP,
        "final_global_step": FINAL_GLOBAL_STEP, "optimizer": optimizer_info,
        "trainability": trainability,
        "gpu": gpu, "node": node, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32": {"cudnn": torch.backends.cudnn.allow_tf32,
                 "matmul": torch.backends.cuda.matmul.allow_tf32},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "splits": split_manifest}
    for root in (Path(reports), Path(run)):
        root.mkdir(parents=True, exist_ok=True)
        target = root / "run_manifest.json"
        if target.exists() and json.loads(target.read_text()) != jsonable(payload):
            prior = json.loads(target.read_text()); current = jsonable(payload)
            prior_comparable = dict(prior); current_comparable = dict(current)
            prior_comparable.pop("execution_git_sha", None); current_comparable.pop("execution_git_sha", None)
            has_expanded_state = (any(Path(run).glob("checkpoint_E_epoch_*.pt")) or
                                  (Path(run)/"latest_recovery.pt").exists() or
                                  ((Path(reports)/"training_metrics.jsonl").exists() and
                                   (Path(reports)/"training_metrics.jsonl").stat().st_size > 0))
            if prior_comparable != current_comparable or prior.get("execution_git_sha") == current.get("execution_git_sha") or has_expanded_state:
                raise RuntimeError(f"expanded run manifest conflicts with existing run: {target}")
            # A prior startup failed before E0 or any optimizer update. Rebind only
            # the code SHA after the zero-update implementation fix; all recipe/assets are exact.
            write_json(target, payload)
        if not target.exists(): write_json(target, payload)
    return payload


def _flatten_task_metrics(curves, target_csv):
    rows = []
    confusion = {}
    gt_rows = []
    waterfalls = []
    for curve_path in sorted(Path(target_csv).parent.glob("stage_e/eval_*_step*.json")):
        try: payload = json.loads(curve_path.read_text())
        except Exception: continue
        if "local" not in payload: continue
        split = payload.get("split", curve_path.name.split("eval_", 1)[-1].split("_step", 1)[0])
        official = payload.get("official") or {}
        for scope, local in payload.get("local", {}).items():
            all_metrics = official.get("all") or {}; novel_metrics = official.get("novel") or {}
            mapkey = "context_map" if scope == "context" else "target_map"
            all_map = all_metrics.get(mapkey) or {}; novel_map = novel_metrics.get(mapkey) or {}
            metric_scope = "context" if scope == "context" else "target"
            rows.append({"stage":"E", "global_optimizer_step":payload.get("step"),
                "split":split, "scope":scope, "windows":local.get("windows"),
                "mIoU_all":local.get("mIoU_all_nonempty"), "mIoU_thing":local.get("mIoU_thing"),
                "mIoU_stuff":local.get("mIoU_stuff"), "local_PQ":local.get("local_pq"),
                "official_all_mIoU":all_metrics.get(f"{metric_scope}_miou"),
                "official_all_PQ":all_metrics.get(f"{metric_scope}_pq"),
                "official_all_mAP":all_map.get("map"), "official_all_AP50":all_map.get("map_50"),
                "official_novel_mIoU":novel_metrics.get(f"{metric_scope}_miou"),
                "official_novel_PQ":novel_metrics.get(f"{metric_scope}_pq"),
                "official_novel_mAP":novel_map.get("map"), "official_novel_AP50":novel_map.get("map_50"),
                "PSNR_scope":local.get("psnr"), "true_novel_PSNR":payload.get("true_novel_psnr") if scope=="target_all" else None,
                "raw_mask_IoU50_GT_fraction":local.get("raw_best_iou_ge_0_5_fraction"),
                "conditional_class_accuracy":local.get("conditional_classification_accuracy"),
                "joint_class_accuracy":local.get("joint_classification_accuracy"),
                "matched_objectness_recall_p50":local.get("matched_objectness_recall_p50"),
                "per_class_iou":local.get("per_class_iou"),
                "class_agnostic_TP":local.get("class_agnostic_tp"), "class_agnostic_FP":local.get("class_agnostic_fp"),
                "class_agnostic_FN":local.get("class_agnostic_fn"), "class_aware_TP":local.get("class_aware_tp"),
                "class_aware_FP":local.get("class_aware_fp"), "class_aware_FN":local.get("class_aware_fn"),
                "GT_count":local.get("gt_count"), "GT_with_anchor_support_fraction":local.get("gt_support_fraction"),
                "filtering_waterfall":local.get("filtering_waterfall")})
            confusion[f"E:{split}:{payload.get('step')}:{scope}"] = local.get("conditional_class_confusion")
            waterfall = local.get("filtering_waterfall") or {}
            waterfalls.append({"stage":"E", "split":split, "global_optimizer_step":payload.get("step"),
                               "scope":scope, **waterfall})
            scope_key = "context" if scope == "context" else "target"
            for window_index, row in enumerate(payload.get("local_rows", {}).get(scope_key, [])):
                for gt in row.get("raw_best_iou", []):
                    gt_rows.append({"stage":"E", "split":split, "global_optimizer_step":payload.get("step"),
                                    "scope":scope, "window_index":window_index, **gt})
    for name, data in ((target_csv, rows), (Path(target_csv).parent/"per_gt_mask_and_classification.csv",gt_rows),
                       (Path(target_csv).parent/"filtering_waterfall.csv",waterfalls)):
        with Path(name).open("w", newline="") as f:
            fields = sorted({k for row in data for k in row}) or ["status"]
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for row in data:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                                 for k,v in row.items()})
    write_json(Path(target_csv).parent/"classification_confusion.json", confusion)
    compact_nodes=[]
    for path in sorted((Path(target_csv).parent/"stage_e").glob("curves_e_epoch*.json")):
        node=json.loads(path.read_text())
        compact_nodes.append({"path":str(path.relative_to(Path(target_csv).parent)),
            "epoch":node.get("epoch"),"global_optimizer_step":node.get("global_optimizer_step"),
            "source_stage_s_references":node.get("source_stage_s_references")})
    write_json(Path(target_csv).parent/"task_metrics.json", {"rows":rows,"evaluation_nodes":compact_nodes})
    return rows, gt_rows


def _finalize_report_bundle():
    curves_path = REPORTS / "stage_e/curves_e_epoch00.json"
    if not curves_path.is_file(): raise RuntimeError("cannot finalize without the E0 endpoint reference")
    curve_paths = [curves_path] + [REPORTS / f"stage_e/curves_e_epoch{e:02d}.json" for e in (2,4,8,16)]
    curves = [json.loads(p.read_text()) for p in curve_paths if p.is_file()]
    task_csv = REPORTS / "task_metrics.csv"
    rows, gt_rows = _flatten_task_metrics(curves, task_csv)
    training_source = REPORTS / "training_metrics.jsonl"
    if not training_source.exists(): training_source.write_text("")
    status = json.loads((REPORTS/"final_status.json").read_text())
    lines = ["# Object-Locus V2.1 Expanded Training — 报告\n\n",
      f"- 状态：`{status['task_status']}`；新增更新 `{status['expanded_updates_completed']}/{TOTAL_NEW_UPDATES}`；global step `{status['global_optimizer_step']}`。\n",
      f"- 来源：Stage S epoch16 checkpoint `{SOURCE_CHECKPOINT}`，SHA256 `{SOURCE_CHECKPOINT_SHA}`，source git `{SOURCE_GIT_SHA}`。\n",
      "- Stage S expansion gate仍为FAIL；本轮由单独规格授权，不回写旧gate。\n",
      "- 唯一配方差异：reconstruction peak LR从1e-5降至1e-6；object peak LR=1e-4、GC alpha=0.01、understanding weight=1，model/loss未改。\n",
      "- 每个窗口一轮一次、共16 epochs；跨提交恢复原model、AdamW moments及RNG。\n",
      "- 输入使用GT相机位姿，本结果不等同于完整unposed SIU3R benchmark。\n\n",
      "## E0 / E2 / E4 / E8 / E16 指标摘录\n\n",
      "完整 context/target-all/novel、local/official 表见 `task_metrics.csv` 与 `task_metrics.json`。缺少official结果的中间节点按MISSING处理。\n\n",
      "| Epoch | Global step | Split | Scope | thing/stuff mIoU | local PQ | official all mIoU/PQ | all mAP/AP50 | novel mIoU/PQ | novel mAP/AP50 | PSNR | true novel PSNR | raw IoU≥0.5 fraction | conditional/joint acc | CA TP/FP/FN |\n",
      "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"]
    for r in rows:
        step = r.get("global_optimizer_step")
        epoch = next((c.get("epoch") for c in curves if any(x.get("step")==step for x in c.get("splits",{}).values())), None)
        # Evaluation split JSON carries the global step; infer registered E epochs if the enclosing payload did not.
        epoch = epoch if epoch is not None else {1792:0,3808:2,5824:4,9856:8,17920:16}.get(step)
        vals = (epoch, step, r.get("split"), r.get("scope"), f"{r.get('mIoU_thing')}/{r.get('mIoU_stuff')}",
                r.get("local_PQ"),f"{r.get('official_all_mIoU')}/{r.get('official_all_PQ')}",
                f"{r.get('official_all_mAP')}/{r.get('official_all_AP50')}",
                f"{r.get('official_novel_mIoU')}/{r.get('official_novel_PQ')}",
                f"{r.get('official_novel_mAP')}/{r.get('official_novel_AP50')}",r.get("PSNR_scope"),
                r.get("true_novel_PSNR"),r.get("raw_mask_IoU50_GT_fraction"),
                f"{r.get('conditional_class_accuracy')}/{r.get('joint_class_accuracy')}",
                f"{r.get('class_agnostic_TP')}/{r.get('class_agnostic_FP')}/{r.get('class_agnostic_FN')}")
        lines.append("| " + " | ".join("MISSING" if v is None or (isinstance(v,str) and v.startswith("None/")) else str(v) for v in vals) + " |\n")
    lines.extend(["\n## Reconstruction comparison\n\n",
        "E0是Stage S16，不是原始pretrained S0。下面分别列出E0→E16与源V2.1 S0→E16。\n\n",
        "| Split | Scope | E0 PSNR | E16 PSNR | E16−E0 | Source S0 PSNR | E16−S0 | E0 true novel PSNR | E16 true novel PSNR | E16−E0 novel |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"])
    eval_index={}
    for p in sorted((REPORTS/"stage_e").glob("eval_*_step*.json")):
        data=json.loads(p.read_text()); split=data.get("split")
        if split: eval_index[(split,int(data.get("step",-1)))]=data
    for split in EVAL_SPLITS:
        e0=eval_index.get((split,INITIAL_GLOBAL_STEP)); e16=eval_index.get((split,FINAL_GLOBAL_STEP))
        s0_path=_source_eval_path(split, 0)
        s0=json.loads(s0_path.read_text()) if s0_path.is_file() else None
        if e0 is None or e16 is None: continue
        for scope in ("context","target_all"):
            p0=e0.get("local",{}).get(scope,{}).get("psnr")
            p16=e16.get("local",{}).get(scope,{}).get("psnr")
            ps0=s0.get("local",{}).get(scope,{}).get("psnr") if s0 else None
            n0=e0.get("true_novel_psnr") if scope=="target_all" else None
            n16=e16.get("true_novel_psnr") if scope=="target_all" else None
            vals=(split,scope,p0,p16,(p16-p0) if p0 is not None and p16 is not None else None,
                  ps0,(p16-ps0) if ps0 is not None and p16 is not None else None,n0,n16,
                  (n16-n0) if n0 is not None and n16 is not None else None)
            lines.append("| "+" | ".join("MISSING" if x is None else str(x) for x in vals)+" |\n")
    lines.extend(["\n## 文件\n\n",
        "- [任务指标CSV](task_metrics.csv)\n- [逐GT结果](per_gt_mask_and_classification.csv)\n",
        "- [分类混淆](classification_confusion.json)\n- [训练指标](training_metrics.jsonl)\n",
        "- [数据manifest](data_manifest.json)\n- [完整训练plan](training_plan.json)\n",
        "- [运行manifest](run_manifest.json)\n- [源码diff](source.patch)\n"])
    (REPORTS/"analysis_report.md").write_text("".join(lines))
    bundle = REPORTS/"review_bundle"
    if bundle.exists(): shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    file_list = [REPORTS/"analysis_report.md",task_csv,REPORTS/"task_metrics.json",
      training_source,REPORTS/"per_gt_mask_and_classification.csv",REPORTS/"classification_confusion.json",
      REPORTS/"filtering_waterfall.csv",REPORTS/"data_manifest.json",REPORTS/"training_plan.json",
      REPORTS/"run_manifest.json",REPORTS/"source_provenance.json",REPORTS/"continuation_audit.json",
      REPORTS/"source_stage_s_expansion_gate.json"]
    for p in file_list:
        if p.exists(): shutil.copy2(p, bundle/p.name)
    code_files = [REPO/"docs/object_locus_v2_1_expanded_spec.md",
      REPO/"scripts/train_object_locus_v2_1_expanded.py",REPO/"scripts/smoke_object_locus_v2_1_expanded.py",
      REPO/"scripts/submit_object_locus_v2_1_expanded.sh",REPO/"tests/test_object_locus_v2_1_expanded.py",
      REPO/"scripts/object_locus_v2_1_runtime.py",REPO/"scripts/eval_object_locus_v2_1.py",
      REPO/"scripts/export_object_locus_v2_1_official.py"]
    for src in code_files:
        dst=bundle/"code"/src.relative_to(REPO); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    for src in sorted((REPORTS/"slurm").glob("*.out")):
        dst=bundle/"key_logs"/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    diff=subprocess.check_output(["git","-C",str(REPO),"diff","d9a5cef3263dc7b16ad79784560b3c30d2045b09..HEAD"],text=True)
    (bundle/"source.patch").write_text(diff)
    (REPORTS/"source.patch").write_text(diff)
    (bundle/"git_status.txt").write_text(subprocess.check_output(["git","-C",str(REPO),"status","--short"],text=True))
    for src in (REPORTS/"stage_e/qualitative").rglob("*.png"):
        dst=bundle/"qualitative"/src.relative_to(REPORTS/"stage_e/qualitative")
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    (bundle/"README.md").write_text("# Object-Locus V2.1 Expanded Training\n\n"
       "See [analysis report](analysis_report.md), [metrics](task_metrics.csv), and [source diff](source.patch).\n"
       "The archive may have companion qualitative ZIP files; upload all named parts. Checkpoints and datasets are excluded.\n")
    shutil.copy2(REPORTS/"analysis_report.md",bundle/"analysis_report.md")
    limit=28*1024*1024; main=REPORTS/"result_bundle.zip"
    def zip_all(include_images):
        with zipfile.ZipFile(main,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in bundle.rglob("*"):
                if p.is_file() and (include_images or "qualitative" not in p.parts): z.write(p,p.relative_to(bundle))
    zip_all(True)
    parts=[]
    if main.stat().st_size>limit:
        zip_all(False)
        imgs=sorted((bundle/"qualitative").rglob("*.png")); chunks=[]; cur=[]; size=0
        for p in imgs:
            if cur and size+p.stat().st_size>24*1024*1024: chunks.append(cur);cur=[];size=0
            cur.append(p);size+=p.stat().st_size
        if cur:chunks.append(cur)
        for i,group in enumerate(chunks,1):
            part=REPORTS/f"result_bundle_qualitative_{i:03d}.zip"
            with zipfile.ZipFile(part,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in group:z.write(p,p.relative_to(bundle))
            parts.append(str(part))
    for archive in [main,*[Path(p) for p in parts]]:
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None: raise RuntimeError(f"bundle zip integrity failure: {archive}")
            import io
            names=set(z.namelist())
            for name in names:
                if name.endswith(".json"): json.loads(z.read(name))
                elif name.endswith(".csv"): list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8"))))
            if "README.md" in names:
                readme=z.read("README.md").decode("utf-8")
                import re
                for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)",readme):
                    if "://" not in target and target not in names:
                        raise RuntimeError(f"review-bundle README link missing: {target}")
            for name in names:
                if name.lower().endswith((".png",".jpg",".jpeg")):
                    from PIL import Image
                    with Image.open(io.BytesIO(z.read(name))) as image: image.verify()
    if main.stat().st_size>limit or any(Path(p).stat().st_size>limit for p in parts):
        raise RuntimeError("expanded result bundle exceeds the registered 28 MiB per-archive limit")
    result={"main_bundle":str(main),"main_bytes":main.stat().st_size,"qualitative_parts":parts,
            "task_metric_rows":len(rows),"per_gt_rows":len(gt_rows)}
    write_json(REPORTS/"bundle_manifest.json",result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--package-only", action="store_true",
                        help="rebuild report archives from an already completed expanded run")
    args = parser.parse_args()
    if args.package_only:
        bundle = _finalize_report_bundle()
        final_path = REPORTS / "final_status.json"
        final = json.loads(final_path.read_text())
        final["result_bundle"] = bundle
        write_json(final_path, final)
        write_json(RUN / "final_status.json", final)
        print(json.dumps(final, indent=2), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expanded training requires exactly one CUDA device")
    gpu = torch.cuda.get_device_name(0); node = os.uname().nodename
    if gpu != "NVIDIA GeForce RTX 3090" or torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
        raise RuntimeError(f"expanded training requires an RTX3090 24GB, got {gpu}")
    if not node.startswith("3dimage-13"):
        raise RuntimeError(f"expanded training is registered for 3dimage-13, got {node}")
    backend_settings = _assert_backend_settings()
    exec_sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    ancestry = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                               "d9a5cef3263dc7b16ad79784560b3c30d2045b09", exec_sha])
    if ancestry.returncode != 0:
        raise RuntimeError(f"execution commit is not based on the registered V2.1 source: {exec_sha}")
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError(f"execution worktree must be clean after push; found modifications: {dirty}")
    source_sha = sha256_file(SOURCE_CHECKPOINT)
    if source_sha != SOURCE_CHECKPOINT_SHA:
        raise RuntimeError(f"source checkpoint SHA mismatch: {source_sha}")
    if sha256_file(SOURCE_MANIFEST_PATH) != SOURCE_DATA_MANIFEST_SHA:
        raise RuntimeError("source split manifest differs from registered Stage S data")
    if sha256_file(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("locked training manifest SHA mismatch")
    for monitor_name, expected_sha in MONITOR_SHA.items():
        if sha256_file(SOURCE_REPORTS / monitor_name) != expected_sha:
            raise RuntimeError(f"source monitor SHA mismatch: {monitor_name}")
    if not SOURCE_GATE.is_file() or json.loads(SOURCE_GATE.read_text()).get("passed") is not False:
        raise RuntimeError("source Stage S gate evidence missing or no longer marked FAIL")
    if REPORTS.exists() and any(REPORTS.iterdir()) and not (REPORTS / "run_manifest.json").exists():
        unexpected = [p.name for p in REPORTS.iterdir() if p.name not in ("smoke", "slurm")]
        if unexpected:
            raise RuntimeError(f"expanded reports directory contains unregistered files: {unexpected}")
    if RUN.exists() and any(RUN.iterdir()) and not (RUN / "run_manifest.json").exists():
        raise RuntimeError(f"expanded checkpoint directory contains unregistered files: {RUN}")
    REPORTS.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    if len(source_manifest["expanded_train_windows"]) != WINDOWS_PER_EPOCH:
        raise RuntimeError("expanded source split no longer has exactly 1008 windows")
    if source_manifest.get("window_counts", {}).get("expanded_scenes") != 128:
        raise RuntimeError("expanded source split no longer has exactly 128 scenes")
    windows = source_manifest["expanded_train_windows"]
    if {w["scene"] for w in source_manifest["dev8"]} & {w["scene"] for w in windows}:
        raise RuntimeError("registered cross-scene dev8 overlaps expanded training scenes")
    expanded_probe = _probe32(windows)
    old_small = source_manifest["small_train_windows"]
    identity_counts = {window_identity(w): 16 for w in old_small}
    expanded_identities = {window_identity(w) for w in windows}
    exposure_rows = []
    for win in windows:
        old_s = identity_counts.get(window_identity(win), 0)
        exposure_rows.append({"scene": win["scene"], "context": win["context"], "novel": win["novel"],
                              "stage_s_exposures": old_s, "expanded_exposures": 16,
                              "cumulative_exposures": old_s + 16})
    legacy_exposure = []
    for row in source_manifest["legacy_train16_exposure"]:
        ident = (row["scene"], tuple(row["context"]), tuple(row["novel"]))
        legacy_exposure.append({**row, "expanded_exposures": 16 if ident in expanded_identities else 0,
                                "cumulative_exposures": int(row.get("stage_s_exposures", 0)) +
                                    (16 if ident in expanded_identities else 0)})
    split_payload = {"source_data_manifest": str(SOURCE_MANIFEST_PATH),
        "source_data_manifest_sha256": SOURCE_DATA_MANIFEST_SHA,
        "expanded_train_windows": windows, "expanded_train_probe32": expanded_probe,
        "train_probe16": source_manifest["train_probe16"],
        "same_scene_holdout16": source_manifest["same_scene_holdout16"],
        "dev8": source_manifest["dev8"], "legacy_train16": source_manifest["legacy_train16"],
        "val8": source_manifest["fixed_val8"], "val32": source_manifest["fixed_val32"],
        "small_train_windows": old_small,
        "expanded_excluded_windows": source_manifest["expanded_excluded_windows"],
        "holdout_selection": source_manifest["holdout_selection"],
        "legacy_train16_exposure": legacy_exposure,
        "expanded_window_exposure": exposure_rows,
        "window_counts": {"expanded_train": len(windows), "expanded_scenes": len({w["scene"] for w in windows}),
                          "expanded_train_probe32": len(expanded_probe), "small_train": len(old_small)}}
    data_path = REPORTS / "data_manifest.json"
    if data_path.exists() and json.loads(data_path.read_text()) != split_payload:
        raise RuntimeError(f"existing expanded data manifest differs: {data_path}")
    write_json(data_path, split_payload)
    plan = build_expanded_plan(split_payload)
    plan_path = REPORTS / "training_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError(f"existing expanded plan differs: {plan_path}")
    write_json(plan_path, plan)
    plan_sha = sha256_file(plan_path); data_sha = sha256_file(data_path)
    source_gate = json.loads(SOURCE_GATE.read_text())
    write_json(REPORTS / "source_stage_s_expansion_gate.json", source_gate)
    write_json(REPORTS / "source_provenance.json", {"source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": source_sha, "source_git_sha": SOURCE_GIT_SHA,
        "source_data_manifest_sha256": SOURCE_DATA_MANIFEST_SHA,
        "source_gate_passed": False, "source_gate_path": str(SOURCE_GATE),
        "source_gate_sha256": sha256_file(SOURCE_GATE)})

    from scripts.object_locus_v2_1_runtime import build_options
    model, opt, transfer = build_model("cuda")
    optimizer, optimizer_info = build_optimizer(model)
    trainability = trainability_counts(model)
    if trainability["frozen_numel"] or any(not p.requires_grad for p in model.parameters()):
        raise RuntimeError("expanded continuation unexpectedly froze a parameter")
    names_by_id = {id(p): n for n, p in model.named_parameters()}
    for group in optimizer.param_groups:
        if not group.get("name") or not group["params"]:
            raise RuntimeError("optimizer group mapping is malformed")
    source_state = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    _assert_source_checkpoint(source_state)
    model.load_state_dict(source_state["model"], strict=True)
    model_mismatch = [name for name, tensor in model.state_dict().items()
                      if not torch.equal(tensor.detach().cpu(), source_state["model"][name])]
    if model_mismatch: raise RuntimeError(f"source model did not load exactly: {model_mismatch[:8]}")
    optimizer.load_state_dict(source_state["optimizer"])
    optimizer_restore = _assert_optimizer_state_exact(source_state["optimizer"], optimizer)
    if optimizer_restore["state_entries"] != 540 or optimizer_restore["internal_step_values"] != [1792.0]:
        raise RuntimeError(f"unexpected source AdamW state/step: {optimizer_restore}")
    source_rng = source_state["rng"]
    restore_rng(source_rng); _assert_rng_exact(source_rng, capture_rng())
    del source_state
    gc_alpha = 0.01
    source_optimizer_info = json.loads((SOURCE_REPORTS / "stage_s/optimizer_audit.json").read_text())
    _assert_optimizer_layout_same(source_optimizer_info, optimizer_info)
    if optimizer_info["unique_trainable_tensors"] != source_optimizer_info["unique_trainable_tensors"]:
        raise RuntimeError("trainable optimizer tensor count differs from Stage S")
    optimizer_info = copy.deepcopy(optimizer_info)
    for group in optimizer_info["groups"]:
        group["lr"] = 1e-4 if group["name"].startswith("object_locus_v2_1_") else 1e-6
    optimizer_info["expanded_schedule"] = {"object_peak_lr": 1e-4, "reconstruction_peak_lr": 1e-6,
        "lr_floor_multiplier": 0.1, "gc_alpha": 0.01, "understanding_weight": 1.0}
    commit_payload = _write_run_manifest(REPORTS, RUN, exec_sha=exec_sha,
        source_sha=source_sha, plan_sha=plan_sha, data_manifest_sha=data_sha,
        split_manifest={"expanded_train": 1008, "expanded_scenes": 128,
                        "expanded_train_probe32": 32, "same_scene_holdout16": 16,
                        "dev8": 8, "val8": 8, "val32": 32},
        optimizer_info=optimizer_info, trainability=trainability, gpu=gpu, node=node)
    write_json(REPORTS / "continuation_audit.json", {
        "source_checkpoint_sha256": source_sha, "source_git_sha": SOURCE_GIT_SHA,
        "execution_git_sha": exec_sha, "architecture": EXPECTED_MODEL,
        "model_tensor_count": len(model.state_dict()), "model_state_exact": not model_mismatch,
        "optimizer_state_exact_before_lr_change": True, "optimizer_restore": optimizer_restore,
        "optimizer_mapping": optimizer_info, "rng_exact_after_restore": True,
        "backend_settings": backend_settings, "trainability": trainability,
        "loaded_source_pretrained_transfer": transfer,
        "lr_change_only": {"old_reconstruction_peak": 1e-5, "new_reconstruction_peak": 1e-6,
                           "object_peak_unchanged": 1e-4}})

    splits = {"expanded_train_probe32": expanded_probe,
        "train_probe16": split_payload["train_probe16"],
        "same_scene_holdout16": split_payload["same_scene_holdout16"],
        "dev8": split_payload["dev8"], "legacy_train16": split_payload["legacy_train16"],
        "val8": split_payload["val8"], "val32": split_payload["val32"]}
    exposure = {"source_stage_s_updates": INITIAL_GLOBAL_STEP,
        "expanded_updates_completed": 0, "expanded_window_exposures": 0,
        "stage_s_small_windows": len(old_small), "expanded_pool_windows": len(windows),
        "expanded_epochs_completed": 0}
    latest = RUN / "latest_recovery.pt"
    resume_state = None
    if latest.exists():
        resume_state = _restore_own_checkpoint(latest, model, optimizer, exec_sha=exec_sha,
            plan_sha=plan_sha, data_manifest_sha=data_sha, source_sha=source_sha)
        next_epoch = int(resume_state["next_epoch"]); next_position = int(resume_state["next_position"])
        expanded_done = int(resume_state["expanded_step"])
        if not (0 <= next_epoch <= EPOCHS and 0 <= next_position < WINDOWS_PER_EPOCH):
            if next_epoch == EPOCHS and next_position == 0: pass
            else: raise RuntimeError("invalid expanded resume cursor")
        exposure = resume_state.get("exposure_stats", exposure)
        write_json(REPORTS / "resume_audit.json", {"resumed": True, "checkpoint": str(latest),
            "next_epoch": next_epoch, "next_position": next_position, "expanded_step": expanded_done,
            "global_optimizer_step": INITIAL_GLOBAL_STEP + expanded_done})
    else:
        if any(RUN.iterdir()) and not (RUN / "run_manifest.json").exists():
            raise RuntimeError("expanded run directory contains unregistered recovery files")
        if any(p.name.startswith("checkpoint_E_epoch_") for p in RUN.iterdir()):
            raise RuntimeError("registered endpoint checkpoints exist but latest_recovery.pt is absent")
        _set_lrs(optimizer, expanded_lrs(1))
        model.train()
        _write_e0_reference(model, opt, splits, REPORTS, RUN, source_sha=source_sha,
            exec_sha=exec_sha, data_manifest_sha=data_sha, plan_sha=plan_sha,
            optimizer=optimizer, optimizer_info=optimizer_info, exposure=exposure)
        next_epoch = 0; next_position = 0; expanded_done = 0

    # If a job ended after a registered endpoint checkpoint but before its evaluation
    # finished, replay only that missing fixed evaluation from the exact checkpoint.
    if next_epoch in EVAL_EPOCHS[1:] and not (REPORTS / f"stage_e/curves_e_epoch{next_epoch:02d}.json").is_file():
        pending_step = INITIAL_GLOBAL_STEP + expanded_done
        pending = _eval_split_set(model, opt, splits, REPORTS / "stage_e", pending_step,
            official=next_epoch in (8, 16), panels=next_epoch in (8, 16))
        pending["epoch"] = next_epoch; pending["expanded_step"] = expanded_done
        write_json(REPORTS / f"stage_e/curves_e_epoch{next_epoch:02d}.json", pending)

    log_path = REPORTS / "training_metrics.jsonl"
    from scripts.eval_object_locus_v2_1 import evaluate_windows
    eval_root = REPORTS / "stage_e"
    for epoch_index in range(next_epoch, EPOCHS):
        permutation = plan["epoch_permutations"][epoch_index]["permutation"]
        start = next_position if epoch_index == next_epoch else 0
        model.train()
        for position in range(start, WINDOWS_PER_EPOCH):
            expanded_step = epoch_index * WINDOWS_PER_EPOCH + position + 1
            if expanded_step != expanded_done + 1:
                raise RuntimeError(f"resume plan cursor discontinuity at expanded step {expanded_step}")
            pool_index = permutation[position]
            window = windows[pool_index]
            batch = build_batch(opt, window, "cuda")
            global_step = INITIAL_GLOBAL_STEP + expanded_step
            lr_values = expanded_lrs(expanded_step)
            output, metrics = train_one_step(model, optimizer, batch, global_step,
                understanding_weight_value=1.0, lr_values=lr_values,
                failure_capture_dir=RUN / "failures",
                failure_context={"stage": "E", "expanded_step": expanded_step,
                    "global_optimizer_step": global_step, "epoch_index": epoch_index,
                    "position": position, "pool_index": pool_index, "window": window,
                    "optimizer_updated": False})
            expanded_done = expanded_step
            exposure["expanded_updates_completed"] = expanded_done
            exposure["expanded_window_exposures"] = expanded_done
            if expanded_step % 20 == 0 or expanded_step == TOTAL_NEW_UPDATES:
                row = {k: v for k, v in metrics.items() if k != "classification_confusion"}
                row.update({"stage": "E", "epoch": epoch_index + 1,
                    "epoch_index": epoch_index, "expanded_step": expanded_step,
                    "stage_step": expanded_step, "global_optimizer_step": global_step,
                    "plan_position": position + 1, "pool_index": pool_index,
                    "window_identity": {"scene": window["scene"], "context": window["context"], "novel": window["novel"]},
                    "object_lr": lr_values[0], "reconstruction_lr": lr_values[1],
                    "understanding_weight": 1.0, "shared_understanding_grad_scale": 0.01,
                    "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                    "reserved_gib": torch.cuda.memory_reserved() / 1024**3})
                with log_path.open("a") as f: f.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            del output, metrics, batch
        completed_epochs = epoch_index + 1
        exposure["expanded_epochs_completed"] = completed_epochs
        exposure["stage_s_window_exposures"] = 16
        exposure["expanded_window_exposure_each"] = completed_epochs
        cursor = {"completed_epochs": completed_epochs, "expanded_step": expanded_done,
                  "next_epoch": completed_epochs, "next_position": 0}
        endpoint_epoch = completed_epochs if completed_epochs in EVAL_EPOCHS[1:] else None
        _checkpoint_and_latest(model, optimizer, RUN, endpoint_epoch=endpoint_epoch,
            cursor=cursor, plan_sha=plan_sha, source_sha=source_sha, exec_sha=exec_sha,
            data_manifest_sha=data_sha, exposure=exposure, optimizer_info=optimizer_info)
        if endpoint_epoch is not None:
            if endpoint_epoch in (2, 4):
                payload = _eval_split_set(model, opt, splits, eval_root,
                    INITIAL_GLOBAL_STEP + expanded_done, official=False, panels=False)
            else:
                payload = _eval_split_set(model, opt, splits, eval_root,
                    INITIAL_GLOBAL_STEP + expanded_done, official=True, panels=True)
            payload["epoch"] = endpoint_epoch
            payload["expanded_step"] = expanded_done
            write_json(eval_root / f"curves_e_epoch{endpoint_epoch:02d}.json", payload)
            # Evaluation restores model mode and RNG through the registered evaluator.
            model.train()
            for opt_group in optimizer.param_groups:
                if not math.isfinite(float(opt_group["lr"])):
                    raise RuntimeError("nonfinite optimizer LR after evaluation")
    final = {"stage": "E", "recipe": EXPECTED_RECIPE, "completed_epochs": EPOCHS,
        "expanded_updates_completed": expanded_done, "total_new_updates": TOTAL_NEW_UPDATES,
        "global_optimizer_step": INITIAL_GLOBAL_STEP + expanded_done,
        "expected_final_global_step": FINAL_GLOBAL_STEP, "complete": expanded_done == TOTAL_NEW_UPDATES,
        "source_stage_s_gate_passed": False,
        "task_status": "expanded training completed; report task metrics independently"}
    write_json(REPORTS / "final_status.json", final); write_json(RUN / "final_status.json", final)
    bundle = _finalize_report_bundle()
    final["result_bundle"] = bundle
    write_json(REPORTS / "final_status.json", final); write_json(RUN / "final_status.json", final)
    print(json.dumps(final, indent=2), flush=True)


EXPECTED_EXEC_SHA = ""
if __name__ == "__main__":
    main()
