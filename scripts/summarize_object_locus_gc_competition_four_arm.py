"""Pinned SIU3R metric reduction and preregistered paired scene bootstrap."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/space/mawb/SIU3R")
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME, STUFF_CLASSES, THING_CLASSES

REPORT = Path("/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1")
EVAL = REPORT / "four_arm_evaluation"
ARMS = ("gc001", "gc010", "gc100", "comp_gc001")
SPLITS = ("expanded_train_probe32", "train_all56", "same_scene_holdout8",
          "dev8", "val32", "val32_excluding_dev8_scenes")
SCOPES = ("context", "target-all", "true-novel")


def metric_evaluator():
    cfg = EvaluatorCfg(dataset_name="scannet", eval_context_miou=False,
        eval_context_pq=False, eval_context_map=False, eval_target_miou=False,
        eval_target_pq=False, eval_target_map=False, eval_image_quality=True,
        eval_depth_quality=False, id2label=PANOPTIC_SEMANTIC2NAME,
        stuffs=STUFF_CLASSES, things=THING_CLASSES, device="cpu", eval_path=str(EVAL))
    obj = Evaluator(cfg); obj.setup(); return obj


def scope_ids(cache, scope):
    frame_ids = cache["frame_ids"].tolist()
    context = set(cache["context_ids"].tolist())
    novel = set(cache["novel_ids"].tolist())
    if scope == "context": return [i for i, f in enumerate(frame_ids) if f in context]
    if scope == "target-all": return list(range(len(frame_ids)))
    return [i for i, f in enumerate(frame_ids) if f in novel and f not in context]


def reconstruction_metrics():
    torch.set_num_threads(4)
    evaluator = metric_evaluator()
    aggregates = {}
    image_rows = []
    per_window = {}
    for arm in ARMS:
        arm_root = EVAL / "reconstruction_cache" / arm
        for split in SPLITS:
            caches = sorted((arm_root / split).glob("*.npz"))
            aggregates[(arm, split)] = {s: {k: [] for k in
                ("psnr", "ssim", "lpips", "absrel", "rmse")} | {"undefined_depth": 0,
                 "depth_images": 0} for s in SCOPES}
            for path in caches:
                with np.load(path) as z:
                    data = {k: z[k] for k in z.files}
                scene = path.stem.split("_context")[0]
                for scope in SCOPES:
                    acc = aggregates[(arm, split)][scope]
                    for vi in scope_ids(data, scope):
                        pred = torch.from_numpy(data["pred_rgb"][vi]).float()[None]
                        truth = torch.from_numpy(data["gt_rgb"][vi]).float()[None]
                        rgb_scores = {
                            "psnr": float(evaluator.psnr(pred, truth).item()),
                            "ssim": float(evaluator.ssim(pred, truth).item()),
                            "lpips": float(evaluator.lpips(pred, truth).item()),
                        }
                        evaluator.psnr.reset(); evaluator.ssim.reset(); evaluator.lpips.reset()
                        for key, value in rgb_scores.items():
                            if not math.isfinite(value):
                                raise FloatingPointError(f"nonfinite {key}: {path}")
                            acc[key].append(value)
                        pred_m = torch.from_numpy(data["pred_depth"][vi]).float() / 0.15
                        gt = torch.from_numpy(data["gt_depth_m"][vi]).float()
                        valid = torch.from_numpy(data["depth_valid"][vi]).bool() & (gt > 0)
                        depth = {"absrel": None, "rmse": None, "scale": None, "shift": None}
                        if valid.any():
                            gt_masked = torch.where(valid, gt, torch.zeros_like(gt))
                            scale, shift = evaluator.fit_scale_and_shift(pred_m, gt_masked)
                            scaled = pred_m * scale + shift
                            err = scaled[valid] - gt[valid]
                            absrel = (err.abs() / gt[valid]).mean()
                            rmse = err.square().mean().sqrt()
                            values = (float(absrel), float(rmse), float(scale), float(shift))
                            if not all(math.isfinite(v) for v in values):
                                raise FloatingPointError(f"nonfinite fitted depth: {path}")
                            acc["absrel"].append(values[0]); acc["rmse"].append(values[1])
                            acc["depth_images"] += 1
                            depth = dict(absrel=values[0], rmse=values[1],
                                         scale=values[2], shift=values[3])
                        else:
                            acc["undefined_depth"] += 1
                        image_rows.append({"arm": arm, "split": split, "scene": scene,
                            "context_ids": json.dumps(data["context_ids"].tolist()),
                            "frame_id": int(data["frame_ids"][vi]), "scope": scope,
                            **rgb_scores, **depth, "depth_defined": bool(valid.any())})
                        key = (arm, split, scene, tuple(data["context_ids"].tolist()), scope)
                        target = per_window.setdefault(key, {"psnr": [], "ssim": [], "lpips": [],
                                                             "absrel": [], "rmse": [],
                                                             "undefined_depth": 0})
                        for metric in ("psnr", "ssim", "lpips"):
                            target[metric].append(rgb_scores[metric])
                        if depth["absrel"] is None:
                            target["undefined_depth"] += 1
                        else:
                            target["absrel"].append(depth["absrel"]); target["rmse"].append(depth["rmse"])
            for scope, values in aggregates[(arm, split)].items():
                for metric in ("psnr", "ssim", "lpips", "absrel", "rmse"):
                    rows = values[metric]
                    values[metric] = float(np.mean(rows)) if rows else None
            write_json_path = EVAL / arm / split / "reconstruction_metrics.json"
            write_json_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_path.write_text(json.dumps(aggregates[(arm, split)], indent=2) + "\n")
    with (EVAL / "reconstruction_per_image.csv").open("w", newline="") as f:
        fields = sorted({k for row in image_rows for k in row})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(image_rows)
    compact = {f"{a}/{s}": values for (a, s), values in aggregates.items()}
    (EVAL / "reconstruction_metrics.json").write_text(json.dumps({
        "metric_protocol": "SIU3R Evaluator.setup PSNR/SSIM/LPIPS; per-image float RGB; render depth divided by 0.15 to meters; provider valid and GT>0 mask; SIU3R Evaluator.fit_scale_and_shift per image",
        "lpips": "vgg, normalize=True", "depth_alignment": "per-image scale and shift",
        "scope_metrics": compact, "per_image_count": len(image_rows),
        "undefined_depth_images_excluded_and_counted": True,
    }, indent=2) + "\n")
    return aggregates, per_window


def flat_official(arm, split):
    root = json.loads((EVAL / arm / "evaluation_arm.json").read_text())[split]
    allx = root["official_all"]["result"]
    nov = root["official_novel"]["result"]
    def block(src, prefix):
        mp = src.get(prefix + "_map") or {}
        return {"mIoU": src.get(prefix + "_miou"),
                "PQ": src.get(prefix + "_pq"),
                "mAP": mp.get("map"), "AP50": mp.get("map_50")}
    return {"context": block(allx, "context"),
            "target-all": block(allx, "target"),
            "true-novel": block(nov, "target")}


def gt_file_integrity():
    reference = EVAL / "gc001"
    failures = []
    for split in SPLITS:
        for export in ("official_all", "official_novel"):
            ref_root = reference / split / export
            if not ref_root.exists():
                failures.append(f"missing {ref_root}"); continue
            for pair in sorted(p for p in ref_root.iterdir() if p.is_dir()):
                for gt_dirname in ("context_seg_gt", "target_seg_gt"):
                    ref_dir = pair / gt_dirname
                    if not ref_dir.exists(): continue
                    ref_names = {p.name for p in ref_dir.glob("*.png")}
                    for arm in ARMS[1:]:
                        other = EVAL / arm / split / export / pair.name / gt_dirname
                        if not other.exists(): failures.append(f"missing {other}"); continue
                        other_names = {p.name for p in other.glob("*.png")}
                        if other_names != ref_names:
                            failures.append(f"GT filename set mismatch: {ref_dir} vs {other}")
                        for p in ref_dir.glob("*.png"):
                            q = other / p.name
                            if not q.is_file() or sha256_file(p) != sha256_file(q):
                                failures.append(f"GT mismatch: {p} vs {q}")
    return failures


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def bootstrap():
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    cfg = EvaluatorCfg(dataset_name="scannet", eval_context_miou=False,
        eval_context_pq=False, eval_context_map=False, eval_target_miou=False,
        eval_target_pq=False, eval_target_map=True, eval_image_quality=False,
        eval_depth_quality=False, id2label=PANOPTIC_SEMANTIC2NAME,
        stuffs=STUFF_CLASSES, things=THING_CLASSES, device="cpu",
        eval_path=str(EVAL / "gc001/val32/official_novel"))
    evaluator = Evaluator(cfg); evaluator.setup()
    cohorts = ("val32_excluding_dev8_scenes", "val32")
    output = {}
    for cohort in cohorts:
        split = cohort
        control_root = EVAL / "gc001" / split / "official_novel"
        pair_dirs = sorted(p for p in control_root.iterdir() if p.is_dir())
        scene_to_pairs = {}
        for pair in pair_dirs:
            scene = pair.name.split("_context")[0]
            scene_to_pairs.setdefault(scene, []).append(pair.name)
        scenes = sorted(scene_to_pairs)
        expected = 24 if cohort.startswith("val32_excluding") else 32
        if len(scenes) != expected or any(len(scene_to_pairs[s]) != 1 for s in scenes):
            raise RuntimeError(f"{cohort} scene/window grouping mismatch: {len(scenes)}")
        caches = {}
        for arm in ARMS:
            arm_root = EVAL / arm / split / "official_novel"
            dirs = sorted(p for p in arm_root.iterdir() if p.is_dir())
            if [p.name for p in dirs] != [p.name for p in pair_dirs]:
                raise RuntimeError(f"{arm} window identities mismatch in {cohort}")
            rows = {scene: [] for scene in scenes}
            for pair in dirs:
                parsed = evaluator.process_segmentation(pair / "target_seg_pred",
                                                        pair / "target_seg_gt")
                scene = pair.name.split("_context")[0]
                rows[scene].append((parsed["map_pred"], parsed["map_gt"]))
            caches[arm] = rows
        matrix_path = EVAL / f"bootstrap_indices_{cohort}_seed2026.npy"
        indices = np.random.default_rng(2026).choice(len(scenes),
            size=(2000, len(scenes)), replace=True)
        np.save(matrix_path, indices)
        mAPs = {arm: np.empty(2000, dtype=np.float64) for arm in ARMS}
        ap50s = {arm: np.empty(2000, dtype=np.float64) for arm in ARMS}
        for sample_i, selection in enumerate(indices):
            for arm in ARMS:
                pairs = [pair for scene_i in selection
                         for pair in caches[arm][scenes[int(scene_i)]]]
                metric = MeanAveragePrecision(iou_type="segm", class_metrics=True,
                                              sync_on_compute=False)
                metric.update([p for p, _ in pairs], [g for _, g in pairs])
                result = metric.compute()
                mAPs[arm][sample_i] = float(result["map"])
                ap50s[arm][sample_i] = float(result["map_50"])
                metric.reset()
        comparisons = {}
        for arm in ("comp_gc001", "gc010", "gc100"):
            dmap = mAPs[arm] - mAPs["gc001"]
            dap50 = ap50s[arm] - ap50s["gc001"]
            comparisons[arm] = {
                "against": "gc001", "actual_mAP_difference":
                    flat_official(arm, split)["true-novel"]["mAP"] -
                    flat_official("gc001", split)["true-novel"]["mAP"],
                "actual_AP50_difference":
                    flat_official(arm, split)["true-novel"]["AP50"] -
                    flat_official("gc001", split)["true-novel"]["AP50"],
                "mAP_mean_delta": float(dmap.mean()),
                "mAP_difference_ci95": np.percentile(dmap, [2.5, 97.5]).tolist(),
                "AP50_mean_delta": float(dap50.mean()),
                "AP50_difference_ci95": np.percentile(dap50, [2.5, 97.5]).tolist(),
                "mAP_bootstrap_differences": dmap.tolist(),
                "AP50_bootstrap_differences": dap50.tolist(),
            }
        output[cohort] = {
            "seed": 2026, "resamples": 2000, "scene_names": scenes,
            "shared_scene_indices_path": str(matrix_path),
            "shared_scene_indices_sha256": sha256_file(matrix_path),
            "comparisons": comparisons,
        }
    (EVAL / "paired_scene_bootstrap.json").write_text(json.dumps({
        "status": "COMPLETE", "official_metric": "global packed segmentation mAP/AP50",
        "same_scene_indices_shared_across_all_four_arms": True,
        "bootstrap_is_exploratory_not_multiplicity_adjusted": True,
        "cohorts": output,
    }, indent=2) + "\n")
    return output


def summarize(recon, per_window, boot):
    official = {}
    candidate_ap = {}
    evaluations = {}
    for arm in ARMS:
        evaluations[arm] = json.loads((EVAL / arm / "evaluation_arm.json").read_text())
        official[arm] = {}
        candidate_ap[arm] = {}
        for split in SPLITS:
            official[arm][split] = flat_official(arm, split)
            local = evaluations[arm][split]["local_candidate_ap"]
            candidate_ap[arm][split] = local
    (EVAL / "official_results.json").write_text(json.dumps(official, indent=2) + "\n")
    rows = []
    for arm in ARMS:
        for split in SPLITS:
            for scope in SCOPES:
                off = official[arm][split][scope]
                rec = recon[(arm, split)][scope]
                local = candidate_ap[arm][split][scope]
                rows.append({"arm": arm, "split": split, "scope": scope,
                    **{f"official_{k}": v for k, v in off.items()},
                    "candidate_mAP": local["map"], "candidate_AP50": local["map_50"],
                    **{f"{k}": rec.get(k) for k in ("psnr", "ssim", "lpips", "absrel", "rmse")},
                    "undefined_depth_images": rec["undefined_depth"],
                    "depth_metric_images": rec["depth_images"]})
    with (EVAL / "summary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    integrity_failures = gt_file_integrity()
    if integrity_failures:
        raise RuntimeError("official GT input mismatch: " + "; ".join(integrity_failures[:10]))
    control = official["gc001"]
    comp_boot = boot["val32_excluding_dev8_scenes"]["comparisons"]["comp_gc001"]
    main_split = "val32_excluding_dev8_scenes"
    main_diff = (official["comp_gc001"][main_split]["true-novel"]["mAP"] -
                 control[main_split]["true-novel"]["mAP"])
    main_ap50 = (official["comp_gc001"][main_split]["true-novel"]["AP50"] -
                 control[main_split]["true-novel"]["AP50"])
    main_pq = (official["comp_gc001"][main_split]["true-novel"]["PQ"] -
               control[main_split]["true-novel"]["PQ"])
    base_recon = recon[("gc001", main_split)]["true-novel"]
    comp_recon = recon[("comp_gc001", main_split)]["true-novel"]
    psnr_delta = comp_recon["psnr"] - base_recon["psnr"]
    absrel_ratio = (comp_recon["absrel"] / base_recon["absrel"]
                    if base_recon["absrel"] not in (None, 0) else float("nan"))
    ci = comp_boot["mAP_difference_ci95"]
    ap_ci = comp_boot["AP50_difference_ci95"]
    integrity = json.loads((EVAL / "provenance.json").read_text())
    integrity_ok = integrity.get("status") == "INFERENCE_AND_OFFICIAL_EXPORT_COMPLETE"
    finite_values = [main_diff, main_ap50, main_pq, psnr_delta, absrel_ratio, *ci, *ap_ci]
    invalid = (not integrity_ok or not all(math.isfinite(float(x)) for x in finite_values)
               or any(not x.get("equal") for x in [
                   json.loads((EVAL / a / "evaluation_arm.json").read_text())["state_dict_sha256_before_after"]
                   for a in ARMS]))
    success_checks = {
        "actual_delta_mAP_ge_0.01": main_diff >= .01,
        "mAP_CI95_lower_gt_0": ci[0] > 0,
        "actual_delta_AP50_ge_minus_0.01": main_ap50 >= -.01,
        "actual_delta_PQ_ge_minus_0.01": main_pq >= -.01,
        "true_novel_PSNR_drop_le_0.5dB": psnr_delta >= -.5,
        "true_novel_AbsRel_relative_increase_le_5pct": absrel_ratio <= 1.05,
        "integrity_and_budget_protocol_pass": integrity_ok and not invalid,
    }
    if invalid:
        status = "INVALID"
    elif ci[1] < 0 or ap_ci[1] < -.01:
        status = "FAILURE_RECONSTRAINT" if (main_diff >= .01 and
             (psnr_delta < -.5 or absrel_ratio > 1.05)) else "FAILURE"
    elif all(success_checks.values()):
        status = "SUCCESS"
    elif main_diff >= .01 and (psnr_delta < -.5 or absrel_ratio > 1.05):
        status = "FAILURE_RECONSTRAINT"
    else:
        status = "INCONCLUSIVE"
    gc_checks = {}
    for arm in ("gc010", "gc100"):
        b = boot["val32"]["comparisons"][arm]
        values = {}
        vdiff = official[arm]["val32"]["true-novel"]
        bdiff = control["val32"]["true-novel"]
        values["mAP_CI95_lower_gt_0"] = b["mAP_difference_ci95"][0] > 0
        values["AP50_drop_le_0.01"] = vdiff["AP50"] - bdiff["AP50"] >= -.01
        values["PQ_drop_le_0.01"] = vdiff["PQ"] - bdiff["PQ"] >= -.01
        for split in ("expanded_train_probe32", "train_all56", "same_scene_holdout8", "dev8", "val32"):
            for scope in ("context", "true-novel"):
                key = f"{split}_{scope}_PSNR_drop_le_0.5dB"
                values[key] = recon[(arm, split)][scope]["psnr"] - recon[("gc001", split)][scope]["psnr"] >= -.5
        base_abs = recon[("gc001", "val32")]["true-novel"]["absrel"]
        cand_abs = recon[(arm, "val32")]["true-novel"]["absrel"]
        values["val32_true_novel_AbsRel_increase_le_5pct"] = cand_abs <= base_abs * 1.05
        gc_checks[arm] = {"checks": values, "all_registered_conditions_met": all(values.values()),
                          "mAP_CI95": b["mAP_difference_ci95"]}
    result = {
        "status": status, "primary_comparison": "comp_gc001 - gc001",
        "primary_cohort": main_split, "primary_scope": "true-novel",
        "actual_delta_mAP": main_diff, "actual_delta_AP50": main_ap50,
        "actual_delta_PQ": main_pq, "mAP_difference_CI95": ci,
        "AP50_difference_CI95": ap_ci, "true_novel_PSNR_delta_dB": psnr_delta,
        "true_novel_AbsRel_relative_ratio": absrel_ratio,
        "success_checks": success_checks,
        "gc_registered_comparisons": gc_checks,
        "bootstrap_note": "Exploratory paired scene intervals; no multiplicity-adjusted four-arm claim; training seed uncertainty is not represented.",
        "no_automatic_model_selection_or_follow_on_training": True,
        "gc_four_arm_official_and_reconstruction_table": rows,
    }
    (EVAL / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Object-Locus competition GC four-arm endpoint summary", "",
        f"Primary result: **{status}**", "",
        "Primary comparison: `comp_gc001 - gc001`, val32 excluding dev8 scenes, true-novel.", "",
        f"- Actual ΔmAP: {main_diff:.6f}; paired scene bootstrap 95% CI [{ci[0]:.6f}, {ci[1]:.6f}].",
        f"- Actual ΔAP50: {main_ap50:.6f}; PQ: {main_pq:.6f}.",
        f"- True-novel PSNR change: {psnr_delta:.4f} dB.",
        f"- True-novel AbsRel ratio: {absrel_ratio:.5f}.", "",
        "Bootstrap intervals are exploratory and are not multiplicity adjusted. They do not include training seed uncertainty.",
        "GC0.1/GC1.0 registered comparisons are reported separately; no automatic model selection or follow-on training was performed.", "",
        "See `summary_table.csv` for all arms, splits, scopes, local candidate metrics, official metrics, and reconstruction measures."]
    (EVAL / "summary.md").write_text("\n".join(lines) + "\n")
    complete = {"status": status, "four_arm_endpoints": True, "official_and_bootstrap_complete": True,
                "reconstruction_metrics_complete": True,
                "primary_comparison": "comp_gc001 - gc001",
                "no_full1860_or_additional_training": True,
                "complete_sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}
    (EVAL / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")


def main():
    if Path(sys.executable).resolve() != Path(
            "/space/mawb/SIU3R/.venv_gpu_v4/bin/python").resolve():
        raise RuntimeError(f"summary requires pinned SIU3R interpreter, got {sys.executable}")
    siu3r_head = __import__("subprocess").check_output(
        ["git", "-C", "/space/mawb/SIU3R", "rev-parse", "HEAD"], text=True).strip()
    if siu3r_head != "8ea80166be76854f938e90521f1a5b688b755c87":
        raise RuntimeError(f"SIU3R checkout mismatch: {siu3r_head}")
    provenance = json.loads((EVAL / "provenance.json").read_text())
    if provenance.get("status") != "INFERENCE_AND_OFFICIAL_EXPORT_COMPLETE":
        raise RuntimeError("model inference/export stage is incomplete")
    recon, per_window = reconstruction_metrics()
    boot = bootstrap()
    summarize(recon, per_window, boot)


if __name__ == "__main__":
    main()
