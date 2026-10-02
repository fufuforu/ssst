"""Continue V3-Set from its registered epoch-64 checkpoint over the locked 128-scene pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import zipfile
import zlib
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.object_locus_v3_set_runtime import (
    PRETRAINED_SHA, REPORTS_DEFAULT, RUN_ROOT_DEFAULT, build_model, build_optimizer,
    build_batch, train_one_step, capture_rng, restore_rng, sha256_file, write_json,
    jsonable, trainability_counts, GC_ALPHA, OBJECT_PEAK_LR, RECON_PEAK_LR,
)
from scripts.eval_object_locus_v3_set import evaluate_windows

SOURCE_CKPT = Path("/space/mawb/ssst/workspace_group_plus/object_locus_v3_set/checkpoint_epoch_64.pt")
SOURCE_CKPT_SHA = "485dfbbd9497ccc756b3ba54b42b41a01ee1bcea35fb609ca0c2536b3c648120"
SOURCE_MANIFEST = Path("/space/mawb/ssst/group_plus/object_locus_v2_1_expanded/data_manifest.json")
SOURCE_MANIFEST_SHA = "8b9f42584159cbcebf93fcb635e55b0f96c5aae5436dff87a0aaa2f4e947f430"
V3_MANIFEST = Path("/space/mawb/ssst/group_plus/object_locus_v3_set/data_manifest.json")
V3_MANIFEST_SHA = "c6c1a0dbfb5c88745a9f633c93bd0bb513a946717ceca34a41ad5802cc3f9b35"
V3_EVALFIX = Path("/space/mawb/ssst/group_plus/object_locus_v3_set_evalfix")
SOURCE_GIT_SHA = "3079b02a67b06141f8d5e068f7cbd12c81a6bc9d"
EPOCHS, WINDOWS_PER_EPOCH = 32, 1008
NEW_UPDATES = EPOCHS * WINDOWS_PER_EPOCH
INITIAL_GLOBAL_STEP = 3584
FINAL_GLOBAL_STEP = INITIAL_GLOBAL_STEP + NEW_UPDATES
REGISTERED_EPOCHS = (0, 2, 4, 8, 16, 32)
LOCAL_EPOCHS = REGISTERED_EPOCHS
VAL32_EPOCHS = (0, 4, 8, 16, 32)
OFFICIAL_EPOCHS = (0, 8, 16, 32)
REPORTS = Path("/space/mawb/ssst/group_plus/object_locus_v3_set_expanded")
RUN = Path("/space/mawb/ssst/workspace_group_plus/object_locus_v3_set_expanded")
RECIPE = "OBJECT_LOCUS_V3_SET_EXPANDED_32E_1008W"
TRAIN_CONFIG = {
    "recipe": RECIPE, "batch_size": 1, "precision": "FP32", "amp": False,
    "epochs": EPOCHS, "windows_per_epoch": WINDOWS_PER_EPOCH, "new_updates": NEW_UPDATES,
    "initial_global_optimizer_step": INITIAL_GLOBAL_STEP, "final_global_optimizer_step": FINAL_GLOBAL_STEP,
    "object_peak_lr": OBJECT_PEAK_LR, "reconstruction_peak_lr": RECON_PEAK_LR,
    "lr_warmup_updates": 200, "understanding_weight": 1.0, "gc_alpha": GC_ALPHA,
    "global_grad_clip": 1.0,
    "lr_schedule": "t<=200: t/200; else 0.1+0.9*(1+cos(pi*(t-200)/(32256-200)))/2",
    "optimizer": "AdamW betas=(0.9,0.95), eps=1e-8; inherited 0.05/0 WD groups and moments",
}
SCIENCE_SHA256 = {
    "tokengs/models/object_locus_v3_set.py": "355e138e7ede35fb43f314133e452b47a44ab8ba2bb1d7bb626df44643418d2e",
    "tokengs/models/object_locus_v3_set_controller.py": "df7094adc4e2493a9cda6f130d18c6e420c0da423c403269ae719be8f452ca51",
    "tokengs/models/object_locus_v3_set_loss.py": "74e41a42a55d7caa7fcebc5c048e2ceccfbbbd825a2de736a96114058be803ad",
    "scripts/object_locus_v3_set_runtime.py": "eb4d5b56f8fd603f72e52e5854f070acca962c231386c91f330c934bee0a3ac0",
}


def window_key(w):
    return (str(w["scene"]), tuple(map(int, w["context"])), tuple(map(int, w["novel"])))


def frames(w):
    return set(map(int, w["context"] + w["novel"]))


def science_module_hashes():
    return {name: sha256_file(REPO / name) for name in SCIENCE_SHA256}


def expanded_lr(t):
    t = int(t)
    if not 1 <= t <= NEW_UPDATES:
        raise ValueError(f"expanded local step out of range: {t}")
    if t <= 200:
        multiplier = t / 200.0
    else:
        u = (t - 200) / (NEW_UPDATES - 200)
        multiplier = 0.1 + 0.9 * (1.0 + math.cos(math.pi * u)) / 2.0
    return OBJECT_PEAK_LR * multiplier, RECON_PEAK_LR * multiplier


def build_manifest_and_plan():
    for path, digest in ((SOURCE_MANIFEST, SOURCE_MANIFEST_SHA), (V3_MANIFEST, V3_MANIFEST_SHA)):
        if sha256_file(path) != digest:
            raise RuntimeError(f"locked source manifest SHA mismatch: {path}")
    source = json.loads(SOURCE_MANIFEST.read_text())
    old = json.loads(V3_MANIFEST.read_text())
    windows = source["expanded_train_windows"]
    scenes = sorted({w["scene"] for w in windows})
    if len(windows) != WINDOWS_PER_EPOCH or len(scenes) != 128:
        raise RuntimeError(f"expanded pool identity mismatch: {len(windows)} windows / {len(scenes)} scenes")
    # Fixed deterministic monitor: one earliest ordered window at scene positions 0,4,...,124.
    by_scene = {}
    for wi, w in enumerate(windows):
        by_scene.setdefault(w["scene"], []).append((wi, w))
    probe = []
    for pos in range(0, 128, 4):
        scene = scenes[pos]
        wi, w = sorted(by_scene[scene], key=lambda x: (tuple(x[1]["context"]), tuple(x[1]["novel"]), x[0]))[0]
        probe.append({**w, "expanded_pool_index": wi})
    manifest = {
        "recipe": RECIPE,
        "source_expanded_manifest": str(SOURCE_MANIFEST),
        "source_expanded_manifest_sha256": SOURCE_MANIFEST_SHA,
        "source_v3_manifest": str(V3_MANIFEST),
        "source_v3_manifest_sha256": V3_MANIFEST_SHA,
        "expanded_train_windows": windows,
        "expanded_train_probe32": probe,
        "original_train_all56": old["train_all56"],
        "same_scene_holdout16": source["same_scene_holdout16"],
        "dev8": source["dev8"],
        "val32": source["val32"],
        "legacy_train16": source["legacy_train16"],
        "small_train_windows": source["small_train_windows"],
        "window_counts": {"expanded_train_windows": len(windows), "expanded_scenes": len(scenes),
                           "expanded_train_probe32": len(probe), "original_train_all56": len(old["train_all56"]),
                           "same_scene_holdout16": len(source["same_scene_holdout16"]),
                           "dev8": len(source["dev8"]), "val32": len(source["val32"])},
        "context_class_window_counts": {str(c): sum(c in set(map(int, w.get("semantic_classes_context", []))) for w in windows)
                                         for c in range(2, 20)},
        "total_manifest_thing_instances": sum(int(w.get("thing_instance_count", 0)) for w in windows),
        "scene_counts": {"expanded_train": len(scenes),
                          "same_scene_holdout_scenes": len({w["scene"] for w in source["same_scene_holdout16"]}),
                          "dev8_scenes": len({w["scene"] for w in source["dev8"]}),
                          "val32_scenes": len({w["scene"] for w in source["val32"]})},
        "epochs": EPOCHS, "windows_per_epoch": WINDOWS_PER_EPOCH, "new_updates": NEW_UPDATES,
        "initial_global_optimizer_step": INITIAL_GLOBAL_STEP, "final_global_optimizer_step": FINAL_GLOBAL_STEP,
        "old_train56_additional_exposures": 32, "old_train56_prior_exposures": 64,
        "old_train56_final_exposures": 96, "other_expanded_window_final_exposures": 32,
    }
    entries = []
    for epoch in range(EPOCHS):
        permutation = np.random.default_rng(42 + epoch).permutation(WINDOWS_PER_EPOCH)
        for position, wi in enumerate(permutation):
            w = windows[int(wi)]
            entries.append({"expanded_step": len(entries) + 1, "global_optimizer_step": INITIAL_GLOBAL_STEP + len(entries) + 1,
                            "epoch_index": epoch, "epoch": epoch + 1, "position": position,
                            "window_index": int(wi), "identity": {"scene": w["scene"], "context": w["context"], "novel": w["novel"]},
                            "scene": w["scene"], "context": w["context"], "novel": w["novel"]})
    plan = {"recipe": RECIPE, "permutation_seed_base": 42, "epochs": EPOCHS,
            "windows_per_epoch": WINDOWS_PER_EPOCH, "new_updates": NEW_UPDATES,
            "initial_global_optimizer_step": INITIAL_GLOBAL_STEP,
            "final_global_optimizer_step": FINAL_GLOBAL_STEP, "entries": entries}
    return manifest, plan


def validate_assets(manifest, plan):
    windows = manifest["expanded_train_windows"]
    keys = [window_key(w) for w in windows]
    if len(set(keys)) != 1008 or len({w["scene"] for w in windows}) != 128:
        raise RuntimeError("expanded training window identities/scenes are not locked as specified")
    old_keys = {window_key(w) for w in manifest["original_train_all56"]}
    if len(old_keys) != 56 or not old_keys.issubset(set(keys)):
        raise RuntimeError("all original 56 V3-Set windows must be present in expanded pool")
    train_scenes = {w["scene"] for w in windows}
    hold = manifest["same_scene_holdout16"]
    leaks = []
    for h in hold:
        for w in windows:
            if w["scene"] == h["scene"] and frames(w) & frames(h):
                leaks.append((window_key(h), window_key(w)))
    if leaks:
        raise RuntimeError(f"same-scene holdout frame leak: {leaks[:3]}")
    for name in ("dev8", "val32"):
        overlap = train_scenes & {w["scene"] for w in manifest[name]}
        if overlap:
            raise RuntimeError(f"{name} overlaps expanded train scenes: {sorted(overlap)}")
    if len(plan["entries"]) != NEW_UPDATES:
        raise RuntimeError("expanded plan update count mismatch")
    seen = {}
    for step, e in enumerate(plan["entries"], 1):
        if e["expanded_step"] != step or e["global_optimizer_step"] != INITIAL_GLOBAL_STEP + step:
            raise RuntimeError(f"expanded/global step mismatch at {step}")
        wi = int(e["window_index"])
        if not 0 <= wi < len(windows) or window_key(e["identity"]) != window_key(windows[wi]):
            raise RuntimeError(f"expanded plan identity mismatch at {step}")
        seen[window_key(e["identity"])] = seen.get(window_key(e["identity"]), 0) + 1
    if len(seen) != 1008 or set(seen.values()) != {32}:
        raise RuntimeError("every expanded window must occur exactly once per epoch")
    context_classes = sorted({int(c) for w in windows for c in w.get("semantic_classes_context", []) if 2 <= int(c) <= 19})
    if context_classes != list(range(2, 20)):
        raise RuntimeError(f"manifest context class coverage incomplete: {context_classes}")
    return {"expanded_windows": 1008, "expanded_scenes": 128, "original_train56_subset": True,
            "holdout_frame_disjoint": True, "dev8_train_scene_disjoint": True,
            "val32_train_scene_disjoint": True, "context_classes": context_classes,
            "plan_updates": NEW_UPDATES, "exposures_per_window": 32,
            "old_train56_final_exposures": 96}


def equal_tree(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    try:
        return a == b
    except Exception:
        return False


def _write_checkpoint(path, model, optimizer, rng, *, epoch, expanded_step, next_epoch,
                      next_position, plan_sha, manifest_sha, exposures, source_sha, execution_sha,
                      config, optimizer_audit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng,
               "stage": "EXPANDED_V3_SET", "epoch": epoch, "stage_step": expanded_step,
               "expanded_local_step": expanded_step,
               "global_optimizer_step": INITIAL_GLOBAL_STEP + expanded_step,
               "completed_updates": expanded_step, "next_epoch": next_epoch,
               "next_position": next_position, "plan_position": expanded_step,
               "architecture_name": model.architecture_name, "recipe": RECIPE,
               "source_checkpoint": str(SOURCE_CKPT), "source_checkpoint_sha256": source_sha,
               "source_git_sha": SOURCE_GIT_SHA, "execution_git_sha": execution_sha,
               "source_manifest_sha256": SOURCE_MANIFEST_SHA, "v3_manifest_sha256": V3_MANIFEST_SHA,
               "plan_sha256": plan_sha, "manifest_sha256": manifest_sha,
               "optimizer_config": optimizer_audit, "current_lr": [float(g["lr"]) for g in optimizer.param_groups],
               "exposure_stats": exposures, "config": config}
    tmp = path.with_suffix(path.suffix + ".tmp")
    before = capture_rng()
    torch.save(payload, tmp)
    after = capture_rng()
    if not equal_tree(before, after):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("checkpoint save changed training RNG")
    os.replace(tmp, path)


def _gpu_assert():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expanded formal run requires exactly one CUDA GPU")
    name = torch.cuda.get_device_name(0)
    node = os.uname().nodename
    if name != "NVIDIA GeForce RTX 3090" or torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
        raise RuntimeError(f"expected 24GB RTX3090, got {name}")
    if not node.startswith("3dimage-13"):
        raise RuntimeError(f"formal run is fixed to 3dimage-13, got {node}")
    backend = {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if backend != {"cudnn_allow_tf32": True, "matmul_allow_tf32": False, "deterministic_algorithms": False}:
        raise RuntimeError(f"source checkpoint CUDA numerical settings differ: {backend}")
    return {"gpu": name, "node": node, "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
            "cuda": torch.version.cuda, **backend}


def _official_scope(official, scope):
    arm = "novel" if scope == "novel" else "all"
    view = "context" if scope == "context" else "target"
    block = official.get(arm, {}) if isinstance(official, dict) else {}
    map_block = block.get(f"{view}_map", {}) if isinstance(block, dict) else {}
    def get(obj, key):
        value = obj.get(key) if isinstance(obj, dict) else None
        if value is None: return "MISSING"
        if isinstance(value, (int, float)) and value == -1: return "UNDEFINED"
        return value
    raw = {"scope_official_miou_raw": block.get(f"{view}_miou") if isinstance(block, dict) else None,
           "scope_official_pq_raw": block.get(f"{view}_pq") if isinstance(block, dict) else None,
           "scope_official_map_raw": map_block.get("map") if isinstance(map_block, dict) else None,
           "scope_official_ap50_raw": map_block.get("map_50") if isinstance(map_block, dict) else None}
    provenance = official.get("_provenance", {}).get(arm, {}) if isinstance(official, dict) else {}
    source = f'official["{arm}"]["{view}_map"] / [{view}_miou] / [{view}_pq]'
    if isinstance(provenance, dict) and provenance.get("path"):
        source = f"{provenance['path']} sha256={provenance.get('sha256', 'MISSING')} ({source})"
    return ({"scope_official_miou": get(block, f"{view}_miou"),
            "scope_official_pq": get(block, f"{view}_pq"),
            "scope_official_map": get(map_block, "map"),
            "scope_official_ap50": get(map_block, "map_50"),
            "scope_official_source": source}) | raw


def _attach_official_provenance(row, reports, step, split):
    official = row.get("official")
    if not isinstance(official, dict):
        return row
    # Iterate over a stable snapshot because provenance is stored in the same dictionary.
    for arm in tuple(official.keys()):
        if arm not in ("all", "novel"):
            continue
        filename = "official_all.json" if arm == "all" else "official_novel.json"
        json_path = Path(reports) / "official" / f"step_{step:04d}" / split / filename
        official.setdefault("_provenance", {})[arm] = {
            "path": str(json_path), "sha256": sha256_file(json_path) if json_path.exists() else "MISSING"}
    return row


def _eval_node(model, opt, manifest, epoch, reports, *, official, panels=False):
    step = INITIAL_GLOBAL_STEP + epoch * WINDOWS_PER_EPOCH
    split_map = {
        "expanded_train_probe32": manifest["expanded_train_probe32"],
        "original_train_all56": manifest["original_train_all56"],
        "same_scene_holdout16": manifest["same_scene_holdout16"],
        "dev8": manifest["dev8"],
    }
    if epoch in VAL32_EPOCHS:
        split_map["val32"] = manifest["val32"]
    node = {"epoch": epoch, "expanded_local_step": epoch * WINDOWS_PER_EPOCH,
            "global_optimizer_step": step, "splits": {}}
    per_gt, query_rows = [], []
    rng = capture_rng()
    try:
        for split, windows in split_map.items():
            row, gt, queries = evaluate_windows(model, opt, windows, step, split, reports,
                "cuda", build_batch, official=official, panels=panels)
            _attach_official_provenance(row, reports, step, split)
            write_json(reports / f"eval_{split}_step{step:05d}.json", row)
            node["splits"][split] = row
            per_gt.extend(gt)
            query_rows.extend(queries)
    finally:
        restore_rng(rng)
        model.train()
    return node, per_gt, query_rows


def _compare_e0(node):
    comparisons = []
    map_old = {"original_train_all56": "train_all56", "dev8": "dev8", "val32": "val32"}
    for new_name, old_name in map_old.items():
        old_path = V3_EVALFIX / f"eval_{old_name}_step3584.json"
        if not old_path.exists():
            raise RuntimeError(f"required E0 parity source missing: {old_path}")
        old = json.loads(old_path.read_text())
        new = node["splits"][new_name]
        def visit(scope, path, a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in sorted(set(a) & set(b)):
                    visit(scope, f"{path}.{key}" if path else key, a[key], b[key])
            elif isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    comparisons.append({"split": new_name, "scope": scope, "field": path,
                                        "source": len(a), "expanded": len(b), "abs_diff": abs(len(a)-len(b)), "pass": False})
                else:
                    for i, (av, bv) in enumerate(zip(a, b)):
                        visit(scope, f"{path}[{i}]", av, bv)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diff = abs(float(a) - float(b))
                comparisons.append({"split": new_name, "scope": scope, "field": path,
                                    "source": a, "expanded": b, "abs_diff": diff,
                                    "pass": diff <= 1e-5})
            elif type(a) is type(b) and isinstance(a, (str, bool, type(None))):
                comparisons.append({"split": new_name, "scope": scope, "field": path,
                                    "source": a, "expanded": b, "abs_diff": 0 if a == b else None,
                                    "pass": a == b})
        for scope in ("context", "target_all", "novel"):
            visit(scope, "", old["local"][scope], new["local"][scope])
        if "official" in old:
            # The official output comes from the fixed model state; this run freshly evaluates it.
            pass
    failed = [x for x in comparisons if not x["pass"]]
    if failed:
        raise RuntimeError(f"E0 evaluation parity failed (tolerance 1e-5): {failed[:8]}")
    return comparisons


def _write_node_tables(reports, nodes, per_gt, query_rows):
    rows = []
    for node in nodes:
        for split, result in node["splits"].items():
            for scope, metric in result["local"].items():
                row = {"epoch": node["epoch"], "expanded_local_step": node["expanded_local_step"],
                       "global_optimizer_step": node["global_optimizer_step"], "split": split, "scope": scope}
                for key, value in metric.items():
                    if isinstance(value, (str, int, float)):
                        row[key] = value
                    elif isinstance(value, dict):
                        for sk, sv in value.items():
                            if isinstance(sv, (str, int, float)):
                                row[f"{key}_{sk}"] = sv
                row.update(_official_scope(result.get("official", {}), scope))
                rows.append(row)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with (reports / "task_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
        write_json(reports / "task_metrics.json", rows)
    for name, values in (("per_gt_candidate_and_panoptic.csv", per_gt), ("candidate_panoptic_queries.csv", query_rows)):
        if values:
            keys = sorted({k for r in values for k in r})
            with (reports / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(values)
    confusion = {}
    for n in nodes:
        for split, result in n["splits"].items():
            for scope, metric in result["local"].items():
                confusion[f"E{n['epoch']}:{split}:{scope}"] = metric["classification_confusion"]
    write_json(reports / "classification_confusion.json", confusion)


def _final_report(reports, nodes, parity, run_manifest):
    by_epoch = {n["epoch"]: n for n in nodes}
    s0 = by_epoch[0]["splits"]
    e32 = by_epoch[32]["splits"]
    probe = e32["expanded_train_probe32"]["local"]["context"]
    hold = e32["same_scene_holdout16"]["local"]["context"]
    val = e32["val32"]["local"]["context"]
    offprobe = e32["expanded_train_probe32"].get("official", {}).get("all", {}).get("context_map", {}).get("map_50", 0.)
    offhold = e32["same_scene_holdout16"].get("official", {}).get("all", {}).get("context_map", {}).get("map_50", 0.)
    offval = e32["val32"].get("official", {}).get("all", {}).get("context_map", {}).get("map_50", 0.)
    goals = {
        "expanded_train_probe32_candidate_cw_recall_ge_0_60": probe["candidate_cw"]["recall"] >= .60,
        "expanded_train_probe32_candidate_cw_precision_ge_0_60": probe["candidate_cw"]["precision"] >= .60,
        "expanded_train_probe32_candidate_ap50_ge_0_60": probe.get("candidate_ap", {}).get("map_50", 0.) >= .60,
        "expanded_train_probe32_official_ap50_ge_0_40": offprobe >= .40,
        "same_scene_holdout16_official_ap50_ge_0_15": offhold >= .15,
        "same_scene_holdout16_candidate_cw_recall_ge_0_25": hold["candidate_cw"]["recall"] >= .25,
        "val32_official_ap50_ge_0_05": offval >= .05,
        "val32_candidate_ap50_ge_0_10": val.get("candidate_ap", {}).get("map_50", 0.) >= .10,
        "val32_candidate_cw_recall_ge_0_10": val["candidate_cw"]["recall"] >= .10,
    }
    psnr = {}
    for split in ("expanded_train_probe32", "original_train_all56", "same_scene_holdout16", "dev8", "val32"):
        if split not in s0 or split not in e32: continue
        for scope in ("context", "novel"):
            p0 = s0[split]["local"][scope]["psnr"]
            p32 = e32[split]["local"][scope]["psnr"]
            psnr[f"{split}:{scope}"] = {"E0": p0, "E32": p32, "drop_db": p0-p32,
                                         "within_0_5db": p0-p32 <= .5}
    # Count windows/scenes showing at least one candidate CW true positive at E32.
    spread = {}
    for split in ("expanded_train_probe32", "original_train_all56", "same_scene_holdout16", "dev8", "val32"):
        result = e32[split]
        rows = result["windows"]
        positive = [w for w in rows if w["scopes"]["context"]["candidate_cw"]["tp"] > 0]
        spread[split] = {"windows_with_candidate_cw_tp": len(positive),
                         "scenes_with_candidate_cw_tp": len({w["scene"] for w in positive}),
                         "window_rows": [{"scene": w["scene"], "context": w["context"],
                                          "tp": w["scopes"]["context"]["candidate_cw"]["tp"],
                                          "fp": w["scopes"]["context"]["candidate_cw"]["fp"],
                                          "fn": w["scopes"]["context"]["candidate_cw"]["fn"]} for w in rows]}
    classification = {
        "expanded_train_pool_learned": all(goals[k] for k in list(goals)[:4]),
        "same_scene_new_window_improved": goals[list(goals)[4]] and goals[list(goals)[5]],
        "cross_scene_initial_improvement": all(goals[k] for k in list(goals)[6:]),
    }
    result = {"goals": goals, "conclusion": classification, "psnr": psnr, "candidate_cw_tp_spread": spread,
              "E0_evaluation_parity_passed": all(x["pass"] for x in parity), "gate_is_report_only": True}
    write_json(reports / "expanded_training_assessment.json", result)
    lines = ["# Object-Locus V3-Set Expanded：结果报告", "",
             "本实验从 V3-Set 训练窗口成功 checkpoint 继续，新增32轮×1008窗口（32256 updates），不是fresh模型、也不是与旧训练的等曝光配对比较。输入沿用GT camera poses，不代表完整unposed SIU3R benchmark。", "",
             f"- 完成：{run_manifest.get('completed_updates', 'MISSING')}/32256新增更新；global step {run_manifest.get('final_global_optimizer_step', 'MISSING')}。",
             f"- 数据：1008窗口、128场景；旧train56窗口新增32次曝光、累计96次，其它窗口新增32次。", "",
             "## E0→E32主要结果", "",
             "| Split | Scope | E0 candidate CW P/R | E32 candidate CW P/R | E32 candidate AP50 | E32 official AP50 | E0 PSNR | E32 PSNR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("expanded_train_probe32", "original_train_all56", "same_scene_holdout16", "dev8", "val32"):
        if split not in e32: continue
        for scope in ("context", "target_all", "novel"):
            m0=s0[split]["local"][scope];m=e32[split]["local"][scope]
            om=e32[split].get("official",{})
            ap=om.get("all",{}).get("context_map" if scope=="context" else "target_map",{}).get("map_50","MISSING") if scope!="novel" else om.get("novel",{}).get("target_map",{}).get("map_50","MISSING")
            lines.append(f"| {split} | {scope} | {m0['candidate_cw']['precision']:.3f}/{m0['candidate_cw']['recall']:.3f} | {m['candidate_cw']['precision']:.3f}/{m['candidate_cw']['recall']:.3f} | {m.get('candidate_ap',{}).get('map_50','MISSING')} | {ap} | {m0['psnr']:.3f} | {m['psnr']:.3f} |")
    lines += ["", "## 固定工程目标（仅endpoint判读，不影响训练推进）", ""]
    lines += [f"- {'PASS' if value else 'FAIL'} — {key}" for key,value in goals.items()]
    lines += ["", "## 结论分层", "", f"- 扩展训练 probe：{'达到本规格训练probe目标' if classification['expanded_train_pool_learned'] else '尚未达到本规格训练probe目标'}。",
              f"- 同场景保留窗口：{'达到明显改善目标' if classification['same_scene_new_window_improved'] else '尚未达到明显改善目标'}。",
              f"- 跨场景 val32：{'出现初步有效改善目标信号' if classification['cross_scene_initial_improvement'] else '尚未达到初步跨场景目标'}；dev8是补充监测，且dev8与val32允许重叠，不作为独立证据。",
              "- 通过raw mask、资格/分类、candidate及panoptic指标分别判断候选形成和输出转换；不把单项指标称为任务成功。", "",
              "## 重建保持", ""]
    for key, value in psnr.items():
        lines.append(f"- {key}: E0 {value['E0']:.4f} dB → E32 {value['E32']:.4f} dB，下降 {value['drop_db']:.4f} dB；{'≤0.5 dB' if value['within_0_5db'] else '>0.5 dB'}。")
    lines += ["", "## E32候选CW TP窗口/场景覆盖", "", "详见 `expanded_training_assessment.json` 中每个split的逐窗口记录。", "",
              "## 评估口径", "", "local candidate、local panoptic、local semantic和official packed-panoptic分别报告。novel Hungarian classification沿用context配对，不代表novel独立匹配。Official指标按context→all/context、target-all→all/target、novel→novel/target读取。", ""]
    (reports / "analysis_report.md").write_text("\n".join(lines))


def _package(reports):
    execution_sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    identity = {"architecture": "LOCUSGS_OBJECT_LOCUS_V3_SET", "recipe": RECIPE,
                "source_checkpoint": str(SOURCE_CKPT), "source_checkpoint_sha256": SOURCE_CKPT_SHA,
                "source_git_sha": SOURCE_GIT_SHA, "execution_git_sha": execution_sha,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA, "source_v3_manifest_sha256": V3_MANIFEST_SHA,
                "new_updates": NEW_UPDATES, "global_start": INITIAL_GLOBAL_STEP, "global_end": FINAL_GLOBAL_STEP,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "58063"),
                "training_scenes": 128, "training_windows": WINDOWS_PER_EPOCH, "training_epochs": EPOCHS}
    write_json(reports / "bundle_identity.json", identity)
    main_zip = reports / "object_locus_v3_set_expanded_result_bundle.zip"
    qzip = reports / "object_locus_v3_set_expanded_qualitative.zip"
    for old in reports.glob("object_locus_v3_set_expanded_result_bundle_part*.zip"):
        old.unlink()
    main_zip.unlink(missing_ok=True)
    qzip.unlink(missing_ok=True)

    # Include all report artifacts and official JSON metrics, but no model, data,
    # rendered export PNG, or qualitative panel in the evidence archives.
    excluded_suffixes = {".pt", ".pth", ".ckpt", ".png", ".jpg", ".jpeg", ".webp", ".zip"}
    files = [p for p in reports.rglob("*") if p.is_file()
             and "qualitative" not in p.relative_to(reports).parts
             and p.suffix.lower() not in excluded_suffixes]
    weighed = []
    for p in sorted(files, key=lambda x: str(x.relative_to(reports))):
        compressor = zlib.compressobj(level=6, wbits=-15)
        compressed_bytes = 0
        with p.open("rb") as f:
            while True:
                block = f.read(1024 * 1024)
                if not block: break
                compressed_bytes += len(compressor.compress(block))
        compressed_bytes += len(compressor.flush())
        weighed.append((p, compressed_bytes + 128))
    budget = 25 * 1024 * 1024
    groups, group, used = [], [], 0
    for p, cost in weighed:
        if group and used + cost > budget:
            groups.append(group); group, used = [], 0
        group.append(p); used += cost
    if group: groups.append(group)
    part_paths = [main_zip] + [reports / f"object_locus_v3_set_expanded_result_bundle_part{i:02d}.zip"
                               for i in range(2, len(groups) + 1)]
    identity["bundle_parts"] = [p.name for p in part_paths]
    write_json(reports / "bundle_identity.json", identity)
    readme_lines = ["# Object-Locus V3-Set Expanded result bundle", "",
                    "该交付由多个 ZIP 组成，请将所有列出的证据包放在同一目录并一并上传。", "",
                    "- [分析报告](analysis_report.md)", "- [任务指标 CSV](task_metrics.csv)",
                    "- [任务指标 JSON](task_metrics.json)", "- [训练指标](training_metrics.jsonl)",
                    "- [运行清单](run_manifest.json)", "- [数据清单](data_manifest.json)",
                    "- [训练计划](training_plan.json)", "- [验收评估](expanded_training_assessment.json)", "",
                    "## Evidence ZIP parts", ""]
    readme_lines.extend(f"- `{p.name}`" for p in part_paths)
    readme_lines += ["- `object_locus_v3_set_expanded_qualitative.zip`：固定窗口可视化。", "",
                     "本实验从 V3-Set epoch64 checkpoint 继续，不是 fresh 或等曝光配对实验；输入沿用 GT camera poses，不等同于完整 unposed SIU3R benchmark。ZIP 不含 checkpoint、数据集或官方 PNG 导出。"]
    (reports / "README.md").write_text("\n".join(readme_lines) + "\n")
    core = [reports / "README.md", reports / "bundle_identity.json"]
    groups[0] = [p for p in groups[0] if p not in core]
    groups[0] = core + groups[0]
    for zp, entries in zip(part_paths, groups):
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in entries: z.write(p, p.relative_to(reports))
        if zp.stat().st_size >= 28 * 1024 * 1024:
            raise RuntimeError(f"result ZIP part exceeds 28 MiB: {zp} ({zp.stat().st_size} bytes)")
    qfiles = [p for p in (reports / "qualitative").rglob("*") if p.is_file()]
    if qfiles:
        with zipfile.ZipFile(qzip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in qfiles: z.write(p, p.relative_to(reports))
            z.writestr("README.md", "Fixed panels at registered E0/E8/E32 nodes for the deterministic selected windows.\n")
        if qzip.stat().st_size >= 28 * 1024 * 1024:
            raise RuntimeError("qualitative result ZIP exceeds 28 MiB")
    for zp in (*part_paths, qzip):
        if zp.exists():
            with zipfile.ZipFile(zp) as z:
                if z.testzip(): raise RuntimeError(f"invalid ZIP {zp}")


def _resume_state(model, optimizer, path, *, plan_sha, manifest_sha):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("recipe") != RECIPE or ck.get("execution_git_sha") != subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip():
        raise RuntimeError("expanded resume checkpoint recipe/code identity mismatch")
    if ck.get("plan_sha256") != plan_sha or ck.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("expanded resume checkpoint plan/manifest mismatch")
    model.load_state_dict(ck["model"], strict=True)
    optimizer.load_state_dict(ck["optimizer"])
    restore_rng(ck["rng"])
    for group, lr in zip(optimizer.param_groups, ck["current_lr"]): group["lr"] = float(lr)
    return ck


def _source_optimizer_exact(optimizer, source_state):
    actual = optimizer.state_dict()
    expected_groups = source_state["param_groups"]
    if len(actual["param_groups"]) != len(expected_groups): raise RuntimeError("source optimizer group count mismatch")
    for ag, eg in zip(actual["param_groups"], expected_groups):
        if len(ag["params"]) != len(eg["params"]): raise RuntimeError("source optimizer parameter mapping mismatch")
        if ag.get("name") != eg.get("name"): raise RuntimeError("source optimizer group name mismatch")
    if actual["state"].keys() != source_state["state"].keys(): raise RuntimeError("source optimizer parameter state IDs mismatch")
    for pid, state in source_state["state"].items():
        actual_state = actual["state"][pid]
        if actual_state.keys() != state.keys(): raise RuntimeError(f"optimizer state key mismatch at {pid}")
        for key, value in state.items():
            av = actual_state[key]
            if torch.is_tensor(value):
                if not torch.equal(av.detach().cpu(), value.detach().cpu()):
                    raise RuntimeError(f"optimizer state mismatch: parameter {pid} field {key}")
            elif av != value:
                raise RuntimeError(f"optimizer scalar state mismatch: parameter {pid} field {key}")
    return {"state_entries": len(actual["state"]), "moments_exact": True,
            "optimizer_steps": sorted({int(st["step"].item()) if torch.is_tensor(st["step"]) else int(st["step"])
                                        for st in actual["state"].values() if "step" in st})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train",), default="train")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpu = _gpu_assert()
    science_hashes = science_module_hashes()
    if science_hashes != SCIENCE_SHA256:
        raise RuntimeError(f"protected V3-Set science modules differ from baseline: {science_hashes}")
    REPORTS.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    manifest, plan = build_manifest_and_plan()
    split_audit = validate_assets(manifest, plan)
    data_path = REPORTS / "data_manifest.json"; plan_path = REPORTS / "training_plan.json"
    for path, payload in ((data_path, manifest), (plan_path, plan)):
        if path.exists():
            tmp = path.with_suffix(path.suffix + ".check")
            write_json(tmp, payload)
            same = sha256_file(path) == sha256_file(tmp)
            tmp.unlink()
            if not same: raise RuntimeError(f"refusing to overwrite mismatched existing artifact: {path}")
        else:
            write_json(path, payload)
    manifest_sha, plan_sha = sha256_file(data_path), sha256_file(plan_path)
    if sha256_file(SOURCE_CKPT) != SOURCE_CKPT_SHA: raise RuntimeError("source epoch64 checkpoint SHA mismatch")
    opt = None
    model, opt, transfer = build_model("cuda")
    optimizer, optimizer_audit = build_optimizer(model)
    execution_sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    start_epoch, pergt_all, qrows_all, nodes = 0, [], [], []
    existing_log = REPORTS / "training_metrics.jsonl"
    exposure = {window_key(w): 64 if window_key(w) in {window_key(x) for x in manifest["original_train_all56"]} else 0
                for w in manifest["expanded_train_windows"]}
    if args.resume:
        ckpath = RUN / "checkpoint_latest.pt"
        if not ckpath.exists(): raise RuntimeError("--resume requested but no latest checkpoint exists")
        ck = _resume_state(model, optimizer, ckpath, plan_sha=plan_sha, manifest_sha=manifest_sha)
        start_epoch = int(ck["next_epoch"]); exposure = ck["exposure_stats"]
        if start_epoch < 0 or start_epoch > EPOCHS or int(ck["expanded_local_step"]) != start_epoch * WINDOWS_PER_EPOCH:
            raise RuntimeError("resume checkpoint position is inconsistent")
        # Registered evaluations are loaded from their saved result artifacts.
        for epoch in REGISTERED_EPOCHS:
            if epoch <= start_epoch:
                path = REPORTS / f"eval_node_epoch_{epoch:02d}.json"
                if not path.exists() and epoch < start_epoch:
                    raise RuntimeError(f"registered prior evaluation is missing on resume: {path}")
                if path.exists(): nodes.append(json.loads(path.read_text()))
                details = REPORTS / f"eval_details_epoch_{epoch:02d}.json"
                if details.exists():
                    saved = json.loads(details.read_text())
                    pergt_all.extend(saved.get("per_gt", [])); qrows_all.extend(saved.get("queries", []))
    else:
        if any(RUN.iterdir()): raise RuntimeError(f"run directory not empty; use --resume for a registered run: {RUN}")
        if sha256_file(SOURCE_CKPT) != SOURCE_CKPT_SHA: raise RuntimeError("source checkpoint SHA mismatch")
        source = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
        required = {"architecture_name": "LOCUSGS_OBJECT_LOCUS_V3_SET", "stage": "V3_SET", "epoch": 64,
                    "stage_step": 3584, "global_optimizer_step": 3584, "git_sha": SOURCE_GIT_SHA,
                    "manifest_sha256": V3_MANIFEST_SHA}
        mismatches = {k: (source.get(k), v) for k, v in required.items() if source.get(k) != v}
        if mismatches: raise RuntimeError(f"source checkpoint metadata mismatch: {mismatches}")
        model.load_state_dict(source["model"], strict=True)
        model_mismatch = [k for k, v in source["model"].items() if not torch.equal(model.state_dict()[k].detach().cpu(), v.detach().cpu())]
        if model_mismatch: raise RuntimeError(f"model differs from source before first update: {model_mismatch[:5]}")
        optimizer.load_state_dict(source["optimizer"])
        opt_exact = _source_optimizer_exact(optimizer, source["optimizer"])
        restore_rng(source["rng"])
        lr1, lrrecon1 = expanded_lr(1)
        for group in optimizer.param_groups:
            group["lr"] = lr1 if group["name"].startswith("object_locus_v3_set_") else lrrecon1
        start_state = {"model_equal": True, "model_tensor_count": len(source["model"]),
                       "optimizer": opt_exact, "rng_restored": equal_tree(capture_rng(), source["rng"]),
                       "optimizer_mapping": optimizer_audit}
        if not start_state["rng_restored"]: raise RuntimeError("source RNG was not restored exactly")
        write_json(REPORTS / "continuation_restore_audit.json", start_state)
        run_manifest = {"architecture": "LOCUSGS_OBJECT_LOCUS_V3_SET", "recipe": RECIPE,
                        "git_sha": execution_sha, "source_git_sha": SOURCE_GIT_SHA,
                        "source_checkpoint": str(SOURCE_CKPT), "source_checkpoint_sha256": SOURCE_CKPT_SHA,
                        "source_global_optimizer_step": INITIAL_GLOBAL_STEP,
                        "source_manifest_sha256": SOURCE_MANIFEST_SHA, "v3_manifest_sha256": V3_MANIFEST_SHA,
                        "manifest_sha256": manifest_sha, "plan_sha256": plan_sha,
                        "source_v3_set_epoch64": True, "source_small_run_metrics": "V3-Set epoch64 endpoint",
                        "new_updates": NEW_UPDATES, "epochs": EPOCHS, "windows_per_epoch": WINDOWS_PER_EPOCH,
                        "training_config": TRAIN_CONFIG,
                        "final_global_optimizer_step": FINAL_GLOBAL_STEP, "optimizer_continued": True,
                        "optimizer_reset": False, "rng_continued": True, "understanding_weight": 1.0,
                        "gc_alpha": GC_ALPHA, "object_peak_lr": OBJECT_PEAK_LR,
                        "reconstruction_peak_lr": RECON_PEAK_LR, "lr_warmup_updates": 200,
                        "source_tf32_settings": {"cudnn_allow_tf32": True, "matmul_allow_tf32": False,
                                                  "deterministic_algorithms": False},
                        "gpu": gpu, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "MISSING"),
                        "protected_science_module_sha256": science_hashes,
                        "data_split_audit": split_audit,
                        "exposure_policy": {"original_train56_prior": 64, "expanded_additional": 32,
                                            "original_train56_final": 96, "other_expanded_final": 32},
                        "optimizer_config": optimizer_audit, "trainability": trainability_counts(model),
                        "unregistered_changes": []}
        write_json(REPORTS / "run_manifest.json", run_manifest)
        # Epoch zero is evaluated from the source checkpoint before any optimizer update.
        names0 = ("expanded_train_probe32", "original_train_all56", "same_scene_holdout16", "dev8", "val32")
        node, pergt, qrows = _eval_node(model, opt, manifest, 0, REPORTS, official=True, panels=True)
        parity = _compare_e0(node)
        write_json(REPORTS / "epoch0_source_parity.json", parity)
        nodes.append(node); pergt_all.extend(pergt); qrows_all.extend(qrows)
        write_json(REPORTS / "eval_node_epoch_00.json", node)
        write_json(REPORTS / "eval_details_epoch_00.json", {"per_gt": pergt, "queries": qrows})
        _write_checkpoint(RUN / "checkpoint_latest.pt", model, optimizer, capture_rng(), epoch=0,
                          expanded_step=0, next_epoch=0, next_position=0, plan_sha=plan_sha,
                          manifest_sha=manifest_sha, exposures=exposure, source_sha=SOURCE_CKPT_SHA,
                          execution_sha=execution_sha, config=TRAIN_CONFIG, optimizer_audit=optimizer_audit)
        os.link(RUN / "checkpoint_latest.pt", RUN / "checkpoint_epoch_00.pt")
    # If a time limit interrupted an evaluation after its epoch checkpoint, finish that registered
    # evaluation before applying another optimizer update.
    if args.resume and start_epoch in LOCAL_EPOCHS and (
            not (REPORTS / f"eval_node_epoch_{start_epoch:02d}.json").exists() or
            not (REPORTS / f"eval_details_epoch_{start_epoch:02d}.json").exists()):
        node, pergt, qrows = _eval_node(model, opt, manifest, start_epoch, REPORTS,
                                        official=start_epoch in OFFICIAL_EPOCHS,
                                        panels=start_epoch in (0, 8, 32))
        if start_epoch == 0:
            parity = _compare_e0(node); write_json(REPORTS / "epoch0_source_parity.json", parity)
        nodes = [n for n in nodes if n["epoch"] != start_epoch] + [node]
        pergt_all.extend(pergt); qrows_all.extend(qrows)
        write_json(REPORTS / f"eval_node_epoch_{start_epoch:02d}.json", node)
        write_json(REPORTS / f"eval_details_epoch_{start_epoch:02d}.json", {"per_gt": pergt, "queries": qrows})
    if args.resume and start_epoch in REGISTERED_EPOCHS:
        registered = RUN / f"checkpoint_epoch_{start_epoch:02d}.pt"
        if not registered.exists():
            tmp_link = registered.with_suffix(".pt.tmp")
            tmp_link.unlink(missing_ok=True)
            os.link(RUN / "checkpoint_latest.pt", tmp_link)
            os.replace(tmp_link, registered)
    manifest_keys = {window_key(w): i for i, w in enumerate(manifest["expanded_train_windows"])}
    plan_entries = plan["entries"]
    log = REPORTS / "training_metrics.jsonl"
    if existing_log.exists() and not args.resume: raise RuntimeError("training log exists but run was not resumed")
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        permutation = np.random.default_rng(42 + epoch).permutation(WINDOWS_PER_EPOCH)
        for pos, wi in enumerate(permutation):
            expanded_step = epoch * WINDOWS_PER_EPOCH + pos + 1
            e = plan_entries[expanded_step - 1]
            if e["window_index"] != int(wi) or e["epoch_index"] != epoch or e["position"] != pos:
                raise RuntimeError(f"runtime plan diverged at expanded step {expanded_step}")
            window = manifest["expanded_train_windows"][int(wi)]
            batch = build_batch(opt, window, "cuda")
            lr_values = expanded_lr(expanded_step)
            output, metrics = train_one_step(model, optimizer, batch, expanded_step,
                understanding_weight_value=1.0, lr_values=lr_values,
                failure_capture_dir=RUN / "failures",
                failure_context={"epoch": epoch + 1, "epoch_index": epoch, "position": pos,
                                 "expanded_step": expanded_step, "global_optimizer_step": INITIAL_GLOBAL_STEP+expanded_step,
                                 "window": window, "recipe": RECIPE})
            key = window_key(window); exposure[key] = int(exposure.get(key, 0)) + 1
            if expanded_step % 50 == 0 or expanded_step == NEW_UPDATES:
                row = {k: v for k, v in metrics.items() if k != "classification_confusion"}
                row.update({"epoch": epoch + 1, "epoch_index": epoch, "position": pos,
                            "expanded_local_step": expanded_step,
                            "global_optimizer_step": INITIAL_GLOBAL_STEP + expanded_step,
                            "scene": window["scene"], "context": window["context"], "novel": window["novel"],
                            "gc_alpha": GC_ALPHA, "allocated_gib": torch.cuda.memory_allocated()/1024**3,
                            "reserved_gib": torch.cuda.memory_reserved()/1024**3})
                with log.open("a") as f: f.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            del output, metrics, batch
        done_epoch = epoch + 1
        expanded_step = done_epoch * WINDOWS_PER_EPOCH
        rng = capture_rng()
        target = RUN / "checkpoint_latest.pt"
        _write_checkpoint(target, model, optimizer, rng, epoch=done_epoch, expanded_step=expanded_step,
                          next_epoch=done_epoch, next_position=0, plan_sha=plan_sha, manifest_sha=manifest_sha,
                          exposures=exposure, source_sha=SOURCE_CKPT_SHA, execution_sha=execution_sha,
                          config=TRAIN_CONFIG, optimizer_audit=optimizer_audit)
        if done_epoch in REGISTERED_EPOCHS:
            registered = RUN / f"checkpoint_epoch_{done_epoch:02d}.pt"
            tmp_link = registered.with_suffix(".pt.tmp")
            tmp_link.unlink(missing_ok=True); os.link(target, tmp_link); os.replace(tmp_link, registered)
        # Non-registered epoch states are represented by the atomically replaced latest file only.
        if done_epoch in LOCAL_EPOCHS:
            official = done_epoch in OFFICIAL_EPOCHS
            panel = done_epoch in (8, 32)
            node, pergt, qrows = _eval_node(model, opt, manifest, done_epoch, REPORTS, official=official, panels=panel)
            nodes.append(node); pergt_all.extend(pergt); qrows_all.extend(qrows)
            write_json(REPORTS / f"eval_node_epoch_{done_epoch:02d}.json", node)
            write_json(REPORTS / f"eval_details_epoch_{done_epoch:02d}.json", {"per_gt": pergt, "queries": qrows})
            _write_node_tables(REPORTS, nodes, pergt_all, qrows_all)
        run_manifest_path = REPORTS / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text())
        run_manifest.update({"completed_updates": expanded_step, "final_global_optimizer_step": INITIAL_GLOBAL_STEP+expanded_step,
                             "completed_epoch": done_epoch, "latest_checkpoint": str(RUN / "checkpoint_latest.pt"),
                             "exposure_stats": {"unique_windows": len(exposure), "min_exposures_added": min(exposure.values()),
                                                "max_exposures_added": max(exposure.values()), "total_exposures_added": sum(exposure.values())}})
        write_json(run_manifest_path, run_manifest)
        del rng
    if len(nodes) != len(REGISTERED_EPOCHS):
        # On resume, epoch zero/earlier nodes are already on disk and loaded above.
        nodes = [json.loads((REPORTS / f"eval_node_epoch_{e:02d}.json").read_text()) for e in REGISTERED_EPOCHS]
        pergt_all, qrows_all = [], []
        for node in nodes:
            e = node["epoch"]
            details_path = REPORTS / f"eval_details_epoch_{e:02d}.json"
            if not details_path.exists(): raise RuntimeError(f"registered evaluation details missing: {details_path}")
            saved = json.loads(details_path.read_text())
            pergt_all.extend(saved.get("per_gt", [])); qrows_all.extend(saved.get("queries", []))
    run_manifest = json.loads((REPORTS / "run_manifest.json").read_text())
    run_manifest.update({"completed_updates": NEW_UPDATES, "final_global_optimizer_step": FINAL_GLOBAL_STEP,
                         "completed_epochs": EPOCHS, "completed": True})
    write_json(REPORTS / "run_manifest.json", run_manifest)
    # Tables and report use eval-node aggregates; detailed eval rows are also persisted per split/eval call.
    _write_final_artifacts(REPORTS, manifest, plan, run_manifest, nodes, pergt_all, qrows_all)
    _package(REPORTS)
    print(json.dumps({"complete": True, "new_updates": NEW_UPDATES, "global_optimizer_step": FINAL_GLOBAL_STEP,
                      "reports": str(REPORTS), "bundle": str(REPORTS / "object_locus_v3_set_expanded_result_bundle.zip")}, allow_nan=False), flush=True)


def _write_final_artifacts(reports, manifest, plan, run_manifest, nodes, pergt, qrows):
    # The local evaluator has already emitted per-split full JSON; flatten those records for final delivery.
    _write_node_tables(reports, nodes, pergt, qrows)
    parity_path = reports / "epoch0_source_parity.json"
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else []
    _final_report(reports, nodes, parity, run_manifest)
    final_hashes = science_module_hashes()
    if final_hashes != SCIENCE_SHA256:
        raise RuntimeError(f"protected V3-Set science modules changed during run: {final_hashes}")
    write_json(reports / "protected_science_module_hashes_final.json", final_hashes)
    (reports / "source.patch").write_text(subprocess.check_output(
        ["git", "-C", str(REPO), "diff", "b00b94fdc45af6d2f78c55f05671aaa75906204f..HEAD"], text=True))
    (reports / "git_status.txt").write_text(subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--short"], text=True) or "clean\n")
    write_json(reports / "training_plan.json", plan)
    # plan hashes in the run manifest refer to the exact serialized plan; caller writes the file before training.


if __name__ == "__main__":
    main()
