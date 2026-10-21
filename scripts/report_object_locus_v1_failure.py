#!/usr/bin/env python3
"""Read-only assembly of Object-Locus V1 failure evidence from existing artifacts."""
from __future__ import annotations

import csv
import argparse
import colorsys
import hashlib
import json
import math
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/space/mawb/ssst")
REPORTS = ORIGINAL / "group_plus/object_locus_v1"
RUN = ORIGINAL / "workspace_group_plus/object_locus_v1"
OUT = ORIGINAL / "group_plus/object_locus_v1_failure_analysis"
STEPS = (0, 200, 500, 1000, 2000, 3500)


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def git_text(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def mark(value):
    if value is None:
        return "MISSING"
    if number(value) and float(value) == -1.0:
        return "UNDEFINED"
    return "PRESENT"


def _metric_row(step, split, protocol, scope, metric, value, field_path, source):
    return {"step": step, "split": split, "protocol": protocol, "scope": scope,
            "metric": metric, "value": value, "status": mark(value),
            "field_path": field_path, "source_file": source}


def _nested(value, dotted):
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict): return None
        current = current.get(part)
    return current


def task_metrics():
    rows = []
    for step in STEPS:
        path = REPORTS / f"curves_{step}.json"
        curves = read_json(path)
        for split in ("train16", "val8", "val32"):
            if not curves or split not in curves.get("splits", {}):
                for protocol, scope in (("local", "context"), ("local", "target_all"),
                                        ("official_all", "context"), ("official_all", "target_all"),
                                        ("official_novel", "context"), ("official_novel", "novel")):
                    for metric in ("mIoU_all", "mIoU_thing", "mIoU_stuff", "PQ", "mAP", "AP50", "PSNR"):
                        rows.append(_metric_row(step, split, protocol, scope, metric, None,
                                                "MISSING", str(path)))
                continue
            split_node = curves["splits"][split]
            source = str(path)
            for scope, key in (("context", "context"), ("target_all", "target")):
                node = split_node.get("local", {}).get(key, {})
                scope_tag = "context" if scope == "context" else "target"
                raw_local_path = (REPORTS / f"eval_{split}" / f"step_{step:08d}" /
                                  f"local_{scope_tag}" / f"eval_step{step}_{scope_tag}.json")
                raw_local = read_json(raw_local_path)
                eval_summary_path = REPORTS / f"eval_{split}" / f"step_{step:08d}" / "evaluation_summary.json"
                eval_summary = read_json(eval_summary_path)
                pq_rows = ((eval_summary or {}).get("local_rows", {}).get(scope_tag, []))
                pq_vals = [r.get("local_panoptic", {}).get("mean_pq") for r in pq_rows
                           if number(r.get("local_panoptic", {}).get("mean_pq"))]
                local_pq = sum(pq_vals) / len(pq_vals) if pq_vals else None
                local_fields = {
                    "mIoU_all": (node.get("mIoU_all_nonempty"), f"splits.{split}.local.{key}.mIoU_all_nonempty"),
                    "mIoU_thing": (node.get("mIoU_thing"), f"splits.{split}.local.{key}.mIoU_thing"),
                    "mIoU_stuff": (node.get("mIoU_stuff"), f"splits.{split}.local.{key}.mIoU_stuff"),
                    "PQ": (local_pq, "evaluation_summary.local_rows." + scope_tag + "[*].local_panoptic.mean_pq averaged over windows (local per-class thing PQ diagnostic; definition in task_metrics.md)"),
                    "mAP": (None, f"splits.{split}.local.{key}.mAP (field absent; local evaluator does not emit mAP)"),
                    "AP50": (None, f"splits.{split}.local.{key}.AP50 (field absent; local evaluator does not emit AP50)"),
                    "PSNR": (node.get("psnr"), f"splits.{split}.local.{key}.psnr"),
                }
                for metric, (value, field) in local_fields.items():
                    rows.append(_metric_row(step, split, "local", scope, metric, value, field,
                                            str(eval_summary_path) if metric == "PQ" else source))
            # Official all-scope: context and target-all values, using the raw COCO result.
            raw = split_node.get("official", {}).get("target_all_raw", {})
            raw_path = split_node.get("official", {}).get("all_result_path")
            raw_node = read_json(raw_path) if raw_path and Path(raw_path).is_file() else None
            # Curves keeps summary mIoU/PQ for context; target-all raw contains COCO fields.
            all_ctx = split_node.get("official", {}).get("context_from_official_all", {})
            for metric, key, field in (
                ("mIoU_all", "context_miou", "result.context_miou"),
                ("PQ", "context_pq", "result.context_pq"),
                ("mAP", "context_map", "result.context_map.map"),
                ("AP50", "context_map_50", "result.context_map.map_50"),
            ):
                value = all_ctx.get(key)
                cand = _nested(raw_node, field) if raw_node else None
                if number(cand): value = cand
                rows.append(_metric_row(step, split, "official_all", "context", metric, value,
                                        field, str(raw_path or path)))
            for metric, key, field in (
                ("mIoU_all", "target_miou", "result.target_miou"),
                ("PQ", "target_pq", "result.target_pq"),
                ("mAP", "map", "result.target_map.map"),
                ("AP50", "map_50", "result.target_map.map_50"),
            ):
                value = (raw.get(key) if key in raw else None)
                cand = _nested(raw_node, field) if raw_node else None
                if number(cand): value = cand
                rows.append(_metric_row(step, split, "official_all", "target_all", metric, value,
                                        field, str(raw_path or path)))
            # Novel-only evaluator: context and its novel-only target subset.
            nov = split_node.get("official", {}).get("novel_from_novel_only_subset", {})
            nov_path = split_node.get("official", {}).get("novel_result_path")
            nov_node = read_json(nov_path) if nov_path and Path(nov_path).is_file() else None
            for scope, key_prefix in (("context", "context"), ("novel", "target")):
                for metric, key, field in (("mIoU_all", f"{key_prefix}_miou", f"result.{key_prefix}_miou"),
                                           ("PQ", f"{key_prefix}_pq", f"result.{key_prefix}_pq"),
                                           ("mAP", f"{key_prefix}_map", f"result.{key_prefix}_map.map"),
                                           ("AP50", f"{key_prefix}_map_50", f"result.{key_prefix}_map.map_50")):
                    value = nov.get(key)
                    cand = _nested(nov_node, field) if nov_node else None
                    if number(cand): value = cand
                    rows.append(_metric_row(step, split, "official_novel", scope, metric, value,
                                            field, str(nov_path or path)))
            # Frame-level PSNR is emitted by registered evaluator for context, true novel, target-all.
            psnr = split_node.get("float_psnr_db", {})
            for scope, key in (("context", "context"), ("novel", "novel"), ("target_all", "target_all")):
                rows.append(_metric_row(step, split, "render", scope, "PSNR", psnr.get(key),
                                        f"splits.{split}.float_psnr_db.{key}", source))
    for split in ("train16", "val8", "val32"):
        for protocol, scope in (("local", "context"), ("local", "target_all"),
                                ("official_all", "context"), ("official_all", "target_all"),
                                ("official_novel", "context"), ("official_novel", "novel"),
                                ("render", "context"), ("render", "novel"), ("render", "target_all")):
            for metric in ("mIoU_all", "mIoU_thing", "mIoU_stuff", "PQ", "mAP", "AP50", "PSNR"):
                rows.append(_metric_row(5000, split, protocol, scope, metric, None,
                                        "NOT_RUN", "no step-5000 evaluation was run"))
                rows[-1]["status"] = "NOT_RUN"
    return rows


def training_trace():
    path = REPORTS / "training_log_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    wanted = ["step", "loss_total", "loss", "loss_recon", "loss_understanding", "understanding_weight",
              "loss_thing_2d", "thing_ce", "pixel_bce", "pixel_dice", "loss_stuff_2d", "stuff_bce",
              "stuff_dice", "loss_semantic", "sem_nll", "loss_identity", "id_pull", "id_push",
              "loss_anchor_group", "anchor_ce", "anchor_dice", "loss_final_understanding",
              "loss_aux_mean", "loss_aux_weighted", "aux_ce_layer6", "aux_anchor_ce_layer6",
              "aux_anchor_dice_layer6", "aux_ce_layer8", "aux_anchor_ce_layer8",
              "aux_anchor_dice_layer8", "aux_ce_layer10", "aux_anchor_ce_layer10",
              "aux_anchor_dice_layer10", "object_locus_lr", "reconstruction_lr",
              "pre_clip_global_grad_norm", "clip_coefficient", "gc_registered_hooks", "gc_removed_hooks",
              "gpu_allocated_bytes", "gpu_reserved_bytes", "step_seconds"]
    for row in rows:
        for key in row:
            if key.startswith(("loss_", "aux_")) and key not in wanted:
                wanted.append(key)
        report = row.get("gradient_report_before_clip", {})
        watched_gradients = ("object_locus.W_Q.weight", "object_locus.W_own_e.weight",
                             "object_locus.thing_classifier.weight", "object_locus.W_c.weight",
                             "object_locus.W_s.weight", "anchor_decoder.mu")
        for name in watched_gradients:
            item = report.get(name)
            row[f"grad.{name}.norm"] = item.get("grad_norm", "MISSING") if item else "MISSING"
            row[f"grad.{name}.finite"] = item.get("finite", "MISSING") if item else "MISSING"
    fields = list(dict.fromkeys(wanted + [k for row in rows for k in row if k not in wanted]))
    normalized = [{field: row.get(field, "MISSING") for field in fields} for row in rows]
    with (OUT / "training_trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(normalized)
    return rows, fields


def _find_bad_stats(value, path="root"):
    found=[]
    if isinstance(value,dict):
        if any(int(value.get(k,0) or 0)>0 for k in ("nan_count","posinf_count","neginf_count")):
            found.append({"path":path,"stats":value})
        for k,v in value.items():
            if isinstance(v,(dict,list)): found.extend(_find_bad_stats(v,f"{path}.{k}"))
    elif isinstance(value,list):
        for i,v in enumerate(value):
            if isinstance(v,(dict,list)): found.extend(_find_bad_stats(v,f"{path}[{i}]"))
    return found


def _replay_summary():
    prov=read_json(OUT/"replay_provenance.json")
    stats=read_json(OUT/"failure_tensor_stats.json")
    fail=read_json(OUT/"replay_failure.json")
    if not prov: return None
    if prov.get("status") == "not_reproduced through step4090":
        # Earlier instrumentation attempts stopped on diagnostic-code issues at
        # step3672. Do not present those as the formal failure. The final exact
        # replay record is the evidence for the fixed step4090 batch.
        log_path=OUT/"replay_log.jsonl"
        rec=None
        if log_path.is_file():
            for line in reversed(log_path.read_text().splitlines()):
                candidate=json.loads(line)
                if candidate.get("step") == 4090:
                    rec=candidate; break
        if rec is not None:
            stats={"step":4090,"status":"finite replay; original failure not reproduced",
                   "entry":rec.get("entry"),"batch_stats":rec.get("batch"),
                   "controller":rec.get("controller"),
                   "registered_state_stats":rec.get("registered_state_stats"),
                   "canonical_layers":rec.get("canonical_layers"),
                   "hungarian_cost_stats":rec.get("hungarian_cost_stats"),
                   "final_pairs":rec.get("final_pairs"),"final_targets":rec.get("final_targets"),
                   "metrics":rec.get("metrics"),
                   "replay_step_execution_metrics":rec.get("production_train_one_step_metrics"),
                   "nonfinite_metric_names":[],
                   "metric_scalar_metadata_status":{
                       k:("MISSING: replay JSON retained the scalar value but not tensor shape/dtype/device")
                       for k,v in (rec.get("metrics") or {}).items() if isinstance(v,(int,float)) and not isinstance(v,bool)},
                   "original_failed_step_component_values":"MISSING: the formal run guard aborted before logging or snapshotting these values; the replay values below are not a substitute",
                   "first_nonfinite_operation":None,
                   "interpretation":"All captured tensors and scalar loss metrics are finite in the replay at the locked step4090 batch; this does not identify the original failed execution's first nonfinite operation."}
            (OUT/"failure_tensor_stats.json").write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n")
            fail={"status":"not reproduced","step":4090,"plan_entry":rec.get("entry"),
                  "replay_exception":None,"original_formal_exception":"FloatingPointError: nonfinite loss at step 4090",
                  "original_failed_batch_metrics_available":False,
                  "replay_metrics_finite":True,
                  "prior_diagnostic_attempts":"See diagnostic_attempt_history.json; no prior diagnostic exception is the formal training failure."}
            (OUT/"replay_failure.json").write_text(json.dumps(fail,indent=2,allow_nan=False)+"\n")
            initial=read_json(OUT/"failure_context_initial.json") or {}
            context={"formal_run_snapshot":initial,"isolated_replay_failure":fail,
                     "captured_step4090_replay_evidence":{
                         "step":4090,"plan_entry":rec.get("entry"),"loss_components":rec.get("metrics"),
                         "replay_step_execution_metrics":rec.get("production_train_one_step_metrics"),
                         "nonfinite_metric_names":[],
                         "saved_pre_step_checkpoint":prov.get("saved_pre_step_checkpoint"),
                         "pre_step_state_saved_before_step4090":True,
                         "optimizer_step_completed_for_step4090":True,
                         "meaning":"Replay step4090 was finite, so the production replay optimizer update completed. This is isolated evidence and did not alter the original run."}}
            (OUT/"failure_context.json").write_text(json.dumps(context,indent=2,allow_nan=False)+"\n")
        return {"provenance":prov,"failure":fail,"stats":stats,"first_bad":None}
    bad=_find_bad_stats(stats or {})
    first=bad[0] if bad else None
    if stats is not None:
        stats["nonfinite_locations_in_capture_order"] = bad
        if first:
            stats["first_nonfinite_operation"] = {
                "captured_at": first["path"], "stats": first["stats"],
                "interpretation": "first captured nonfinite tensor in observation sequence; operation-level origin follows from exact layer/stage ordering; see section below"}
        else:
            stats["first_nonfinite_operation"] = {
                "captured_at": "aggregate scalar metrics only" if stats.get("nonfinite_metric_names") else None,
                "interpretation": "No captured intermediate tensor statistic contains NaN/Inf; use nonfinite scalar component metrics to identify first aggregate operation."}
        (OUT/"failure_tensor_stats.json").write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n")
    return {"provenance":prov,"failure":fail,"stats":stats,"first_bad":first}


def _task_metric_table(rows, split="val32"):
    by={(r["step"],r["split"],r["protocol"],r["scope"],r["metric"]):r["value"] for r in rows}
    lines=["| Step | local ctx mIoU all/thing/stuff/PQ | local target-all mIoU all/thing/stuff/PQ | official-all ctx mIoU/PQ/mAP/AP50 | official-all target-all mIoU/PQ/mAP/AP50 | official-novel ctx mIoU/PQ/mAP/AP50 | official-novel target mIoU/PQ/mAP/AP50 | context PSNR | true novel PSNR |",
           "|---:|---|---|---|---|---|---|---:|---:|"]
    for st in (0,200,500,1000,2000,3500):
        def vals(protocol,scope,metrics):
            return "/".join("MISSING" if by.get((st,split,protocol,scope,m)) is None else f"{by[(st,split,protocol,scope,m)]:.6g}"
                           for m in metrics)
        lines.append(f"| {st} | {vals('local','context',['mIoU_all','mIoU_thing','mIoU_stuff','PQ'])} | {vals('local','target_all',['mIoU_all','mIoU_thing','mIoU_stuff','PQ'])} | {vals('official_all','context',['mIoU_all','PQ','mAP','AP50'])} | {vals('official_all','target_all',['mIoU_all','PQ','mAP','AP50'])} | {vals('official_novel','context',['mIoU_all','PQ','mAP','AP50'])} | {vals('official_novel','novel',['mIoU_all','PQ','mAP','AP50'])} | {by.get((st,split,'render','context','PSNR'),'MISSING')} | {by.get((st,split,'render','novel','PSNR'),'MISSING')} |")
    return lines


def _write_chatgpt_summary(rows, logs):
    replay=_replay_summary()
    out=["# Object-Locus V1 failure analysis evidence", "",
         "## Task metrics: val32 (raw ratios 0–1; PSNR in dB)", "", *_task_metric_table(rows), "",
         "Local/official `target-all` runs every frame in the batch: train16 has 2 context + 2 novel = 4 frames/window; val8 and val32 have 2 context + 4 novel = 6 frames/window. It is not novel-only. `official-novel` target uses exactly each monitor row's `novel` frame IDs. True novel-frame PSNR is sourced from `float_psnr_db.novel`. Official `-1` values remain `-1` and are marked UNDEFINED in task_metrics.csv/json. Step5000 is NOT_RUN.", "",
         "Local PQ here is the stored evaluator diagnostic: final ownership/alpha winner masks, best-overlap predicted query per GT thing, TP at IoU>0.5, then per-class sum(TP IoU)/(TP+0.5*FN), averaged over GT-present classes and windows. It does not count unmatched prediction FP and is not official PQ.", "",
         "Full train16/val8/val32 × step × protocol table: `task_metrics.md` / `task_metrics.csv` / `task_metrics.json`.", "",
         "Complete original `train.log`, `train_events.log`, and job `slurm-56991.out` are copied verbatim into `formal_logs_context.txt`; no source log was edited.", "",
         "## Training log: last ten available records", "",
          "These are 100-step records through step4000 and do not represent step4089.", "", "```json",
         json.dumps(logs[-10:],indent=2,allow_nan=False), "```", "",
         "## Stop exception", "", "```text"]
    slurm=(REPORTS/"slurm-56991.out").read_text(errors="replace")
    idx=slurm.rfind("Traceback (most recent call last):")
    out.append(slurm[idx:idx+6000] if idx>=0 else "traceback marker missing from slurm log")
    out += ["```", "", "## Isolated replay result", ""]
    if replay:
        out += [f"- Status: **{replay['provenance'].get('status')}**",
                f"- Start checkpoint: step {replay['provenance'].get('checkpoint_step')} at `{replay['provenance'].get('checkpoint')}`",
                f"- Restoration completeness: {replay['provenance'].get('restoration_completeness','pending')}",
                f"- GPU / job: {replay['provenance'].get('gpu','pending')} / {replay['provenance'].get('slurm_job_id','56996')}",
                f"- First failure step: {replay['provenance'].get('first_failure_step')}",
                f"- Exception: {replay['provenance'].get('failure_exception')}"]
        if replay.get("first_bad"):
            out += [f"- Earliest captured nonfinite: `{replay['first_bad']['path']}`; counts and inputs are in `failure_tensor_stats.json`."]
        if replay.get("stats"):
            out += ["", "Step-4090 replay metrics (the original failed-run batch metrics were not persisted; these are finite replay values, not original failure values):", "", "```json",
                    json.dumps(replay["stats"].get("metrics",{}),indent=2,allow_nan=False), "```"]
            exec_metrics=replay["stats"].get("replay_step_execution_metrics") or {}
            exec_keys=("object_locus_lr","reconstruction_lr","pre_clip_global_grad_norm","clip_coefficient",
                       "gc_registered_hooks","gc_removed_hooks","gpu_allocated_bytes","gpu_reserved_bytes","step_seconds")
            out += ["", "Step-4090 isolated replay optimizer/gradient telemetry:", "", "```json",
                    json.dumps({k:exec_metrics.get(k,"MISSING") for k in exec_keys},indent=2,allow_nan=False), "```"]
    else:
        out.append("Replay has not produced a provenance artifact yet.")
    out += ["", "## Source locations", "",
            "- `scripts/object_locus_v1_runtime.py:297-312`: production step order and scalar finite guard before prediction finite validation, backward, clip, and optimizer.step.",
            "- `scripts/train_object_locus_v1.py:648-678`: model/optimizer/plan/RNG checkpoint restore semantics.",
            "- `tokengs/models/object_locus_v1.py:step_loss`: reconstruction objective, object losses, and aggregate loss construction.",
            "- `tokengs/models/canonical_recon.py:canonical_layer_loss`: RGB/SSIM/visibility component combination.",
            "- `tokengs/models/object_locus_v1_loss.py:final_hungarian` and `object_locus_v1_losses`: matching and understanding objective.",
            "", "## Fixed qualitative panels", "",
            "The first three locked val32 pairs are rendered at steps 0, 1000, and 3500. See `qualitative/index.json` and `qualitative/step_00000000/`, `qualitative/step_00001000/`, `qualitative/step_00003500/`. Panels include the same context/novel frames, GT/reconstruction RGB, semantic GT/prediction, GT/predicted instance maps, raw matched-slot masks, postprocessed matched-object masks and projected center/support overlays.",
            ""]
    (OUT/"analysis_for_chatgpt.md").write_text("\n".join(out))


def _write_full_analysis_report(rows, logs):
    replay=_replay_summary()
    git_info=read_json(OUT/"git_provenance.json") or {}
    lines=["# Object-Locus V1 failure analysis report", "",
           "## Status and scope", "",
           "Formal Object-Locus V1 job 56991 stopped at step4090 after the runtime aggregate-loss finite guard raised. This report uses the existing evaluation outputs plus one isolated replay from the full step3500 checkpoint. The original run tree, checkpoints, recipe, manifest, and evaluation artifacts were not modified. The isolated replay reached step4090 with finite values and did not reproduce the failure; no later steps were run.", "",
           "## Repository provenance", "",
           f"- Execution worktree HEAD/origin: `{git_info.get('execution_head')}` / `{git_info.get('execution_origin_main')}`.",
           f"- Original worktree HEAD/origin: `{git_info.get('original_head')}` / `{git_info.get('original_origin_main')}`.",
           f"- Implementation baseline: `61b22322e79e973cdf7ff868afb6242f4253f3bb`; failure-record baseline: `85cad0d163eafc0e197a79bb374d39f5cfed3814`.",
           "- Calculation-code change: only an opt-in controller observer in `tokengs/models/object_locus_v1_controller.py`; its exact diff is recorded in replay provenance and does not alter tensor formulas when disabled. New diagnostic/report scripts are under `scripts/`. No reconstruction, loss, optimizer, LR, dtype, seed, data-order, or training calculation was changed.",
           f"- Execution worktree status at report generation: `{git_info.get('execution_status_short')}`.",
           f"- Commit-range file difference 61b2232..85cad0d: `{git_info.get('diff_source_commit_to_head_names') or '(none)'}`. Commit-range difference 85cad0d..current HEAD: `{git_info.get('diff_failure_commit_to_head_names') or '(none)'}`. The remaining tracked worktree diff is the opt-in controller observer shown in `replay_provenance.json`; diagnostic scripts are new untracked files.",
           "- Original worktree dirty status was read and preserved; diagnostics wrote only the requested new analysis directory there.", "",
           "## Existing task metric results", "",
           "Full all-split/all-step raw-scale table: `task_metrics.md`, `task_metrics.csv`, `task_metrics.json`. `target-all` means all batch views (train16 2 context + 2 novel; val8/val32 2 context + 4 novel). Novel-only metrics and PSNR are kept separate.", "",
           *_task_metric_table(rows), "",
           "Val32 step0→3500 per-class IoU, class-aware/agnostic TP/FP/FN/GT counts, recall50, raw recall50, active slots and PSNR deltas are in `val32_step0_to_3500.json`.", "",
           "Local `mean_pq` definition: final winner map uses ownership >0.5, alpha >0.05 and score-weighted ownership; per GT thing instance, best-overlap query is TP at IoU >0.5, otherwise FN. Per-class `sum(TP IoU)/(TP+0.5*FN)` is averaged across GT-present thing classes, then per-window scores are averaged. This diagnostic does not accumulate unmatched-prediction FP or enforce predicted class agreement; it is not official PQ. The source implementation is `_panoptic_pq` in the read-only legacy evaluator.", "",
           "## Training records and stop guard", "",
           f"The structured training log contains {len(logs)} records at steps {logs[0]['step']}..{logs[-1]['step']} (100-step cadence). The last ten exact rows are `last10_training_records.json`; the complete union-field trace is `training_trace.csv`. Step4089 has no formal telemetry record.",
           "The production order is: `step_loss` computes forward, reconstruction objective, final unified Hungarian and final/aux understanding losses; then `train_one_step` checks scalar names in order `loss`, `loss_total`, `loss_recon`, `loss_understanding`, `loss_anchor_group`, `anchor_ce`, `anchor_dice`; then checks prediction tensors; then GC backward; gradient finite check; clipping; optimizer.step. Formal exception was `FloatingPointError: nonfinite loss at step 4090`, so the first scalar guard actually reached was `metrics['loss']`; later prediction guard/backward/gradient check/clip/optimizer step were skipped.", "",
           "Full traceback excerpt: `formal_error_excerpt.txt`; raw untouched logs remain at the original train.log/train_events.log/slurm paths listed in `failure_context_initial.json`.", "",
           "## Isolated replay and failure evidence", ""]
    if replay:
        p=replay["provenance"]
        lines += [f"- Replay result: **{p.get('status')}**; first stopped step {p.get('first_failure_step')}.",
                  f"- Restore path: formal step3500 checkpoint SHA `{p.get('checkpoint_sha256')}`, followed by diagnostic resume state `{p.get('diagnostic_resume_checkpoint')}` when applicable.",
                  f"- Restore completeness: {p.get('restoration_completeness')}.",
                  f"- Hardware: {p.get('gpu')}; CUDA_VISIBLE_DEVICES={p.get('cuda_visible_devices')}; torch {p.get('torch')}; CUDA {p.get('torch_cuda')}. SLURM job id: {p.get('slurm_job_id','56999')} (resumed diagnostic job).",
                  f"- Determinism flags: `{json.dumps(p.get('backend',{}),sort_keys=True)}`.",
                  f"- Locked step4090 plan row: `{json.dumps(p.get('plan_step4090'),sort_keys=True)}`; fixed-window match={p.get('fixed_window_match')}.",
                  f"- Original checkpoint/manifest/plan/pretrained hashes were verified; no scheduler state existed in the formal driver/checkpoint.", ""]
        if replay.get("failure", {}).get("exception"):
            lines += ["Replay exception:", "", "```text", replay["failure"].get("exception"), replay["failure"].get("traceback"), "```", ""]
        elif replay.get("provenance", {}).get("status") == "not_reproduced through step4090":
            lines += ["The isolated replay raised no exception at step4090; all checked scalar losses were finite and the replay optimizer update completed.", ""]
        stats=replay.get("stats") or {}
        if stats.get("metrics"):
            lines += ["Step-4090 replay loss/component scalars (not original failed-run values; full per-tensor ranges, finite counts and indices: `failure_tensor_stats.json`):", "", "```json", json.dumps(stats["metrics"],indent=2,allow_nan=False), "```", ""]
        lines += [f"- Nonfinite metric names: `{stats.get('nonfinite_metric_names')}`.",
                  f"- First captured nonfinite location: `{replay.get('first_bad',{}).get('path') if replay.get('first_bad') else 'none among captured floating tensors'}`.",
                  f"- Pre-step state for step4090 was saved at `{p.get('saved_pre_step_checkpoint')}` with save-time RNG equality checked. The original failed run did not update at step4090; the isolated replay was finite and did complete that isolated step update.", ""]
        if replay.get("first_bad"):
            lines += ["First captured nonfinite tensor statistics:", "", "```json", json.dumps(replay["first_bad"],indent=2,allow_nan=False), "```", ""]
        else:
            lines += ["The isolated step4090 replay capture contains no NaN/Inf in captured tensors or loss components. The original failed execution did not persist its batch/component values, so the exact first nonfinite operation and direct input remain undetermined.", ""]
    else:
        lines.append("Replay provenance is not available yet.")
    lines += ["## Root-cause status", "",
              "The formal log proves that `metrics['loss']` was the first scalar guard to raise at `scripts/object_locus_v1_runtime.py:312`, but the formal run did not persist that batch's component metrics or tensors. The exact replay reached the same step4090 plan row with finite captured batch/controller/reconstruction/matching/loss values and completed the isolated optimizer update. Therefore this replay does not establish the first nonfinite operation, its direct input, or an upstream cause. The original failure remains unlocalized; no numerical patch or auto-repair was applied. `failure_tensor_stats.json` contains finite replay evidence, not fabricated values for the original failed execution. Understanding scalar replay values were JSON-scalarized, so their original tensor shape/dtype/device are marked MISSING; reconstruction and intermediate tensor metadata are recorded. Earlier diagnostic OOM/serialization attempts are separated in `diagnostic_attempt_history.json` and are not the formal failure.", "",
              "## Fixed qualitative panels", "",
              "Panels use the first three locked val32 pairs at steps 0,1000,3500, with the same frame lists and stable GT-instance colors. Each panel distinguishes raw slot ownership masks (>0.5) from current evaluator postprocessed instance masks; sidecars include class, GT class, confidence, mask IoU and projected center/c±s support. See `qualitative/index.json` and `qualitative/step_*/pair_*.png`.", "",
              "No step5000 result is claimed. No repair, formal resume, or additional scientific experiment was started.", ""]
    (OUT/"analysis_report.md").write_text("\n".join("" if x is None else str(x) for x in lines))


def generate_panels(device):
    """Targeted read-only forwards for first three locked val32 pairs only."""
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from scripts.runtime_bootstrap import prepare_runtime
    prepare_runtime(REPO)
    from scripts.object_locus_v1_runtime import build_batch, build_model
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    from tokengs.models.object_locus_v1_loss import final_hungarian
    from tokengs.models.canonical_recon import project_points_means2d
    from scripts.export_object_locus_v1_official import assemble_panoptic

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixed qualitative supplementation requires CUDA")
    monitor = read_json(ORIGINAL / "group_plus/instance_state_v1_generalization/monitor_32pairs.json")
    pairs = monitor["pairs"][:3]
    expected = [("scene0011_00", [68, 87], [69, 70, 71, 72]),
                ("scene0011_01", [134, 160], [137, 152, 153, 158]),
                ("scene0015_00", [91, 105], [93, 100, 101, 103])]
    got = [(x["scene"], x["context"], x["novel"]) for x in pairs]
    if got != expected:
        raise RuntimeError(f"val32 fixed first-three monitor rows differ: {got}")
    step_meta = {}
    for step in (0, 1000, 3500):
        checkpoint = RUN / f"checkpoints/step_{step:08d}/train_state.pt"
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(blob.get("step", -1)) != step:
            raise RuntimeError(f"checkpoint step mismatch at {checkpoint}")
        model, opt, transfer = build_model(device)
        model.load_state_dict(blob["model"], strict=True)
        model.eval()
        step_meta[str(step)] = {"checkpoint": str(checkpoint), "transfer_audit": transfer,
                                "git_commit": blob.get("git_commit"), "pairs": []}
        for pi, window in enumerate(pairs):
            batch = build_batch(opt, window, device)
            mi, _ = split_data(batch, opt)
            decoder = ModelInputDecoder(cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"])
            with torch.no_grad():
                out = model.forward_object_locus(ModelInput(mi.encoder, decoder), render_decoder_input=decoder,
                                                 context_decoder=decoder, coupled=False, step=step)
            targets, match_pairs = final_hungarian(out, batch)
            semantic_pred, instance_pred, semantic_raw = assemble_panoptic(out)
            q_to_gt = {}
            match_rows = []
            for qids, gids in match_pairs:
                for q, g in zip(qids.tolist(), gids.tolist()):
                    q_to_gt[int(q)] = int(g)
                    iid = int(targets["gt_instance_ids"][0][g])
                    gt_cls = int(targets["gt_classes"][0][g])
                    probs = out["p_class"][0, q]
                    pred_cls = int(probs[:18].argmax()) + 2
                    cls_conf = float(probs[:18].max())
                    eval_score = float(probs[:18].sum())
                    ious = []
                    for view in (0, 1):
                        gtmask = targets["gt_pixel_masks"][0][g, view]
                        postmask = instance_pred[view] == (q + 1)
                        inter = int((gtmask & postmask).sum())
                        union = int((gtmask | postmask).sum())
                        ious.append(inter / union if union else None)
                    match_rows.append({"query": q, "gt_index": g, "gt_instance_id": iid,
                                       "gt_class": gt_cls, "predicted_class": pred_cls,
                                       "class_confidence_max_thing_probability": cls_conf,
                                       "evaluator_score_sum_thing_probability": eval_score,
                                       "class_aware_correct": pred_cls == gt_cls,
                                       "postprocessed_instance_mask_iou_context_views": ious,
                                       "raw_slot_mask_definition": "region_mass[:,q] > 0.5",
                                       "postprocessed_mask_definition": "current assemble_panoptic output instance_id=q+1; score>=0.5, ownership>0.5, alpha>0.05, ascending query winner"})

            def color_for(key):
                digest = hashlib.sha256(str(key).encode()).digest()
                hue = int.from_bytes(digest[:4], "little") / 2**32
                r, g, b = colorsys.hsv_to_rgb(hue, .78, .98)
                return tuple(int(x * 255) for x in (r, g, b))

            def semantic_color(arr):
                result = np.zeros((*arr.shape, 3), np.uint8)
                for c in range(21):
                    result[arr == c] = ((37*c+53)%255, (97*c+31)%255, (173*c+71)%255)
                return result

            def id_color_map(ids, key_prefix):
                h, w = ids.shape
                arr = np.zeros((h,w,3),np.uint8)
                for iid in np.unique(ids):
                    if iid <= 0: continue
                    arr[ids == iid] = color_for(f"{key_prefix}:{int(iid)}")
                return arr

            frame_ids = [int(x) for x in batch["frame_ids"][0].cpu().tolist()]
            frames = []
            draw_meta = []
            H, W = batch["images_all"].shape[-2:]
            for v, frame_id in enumerate(frame_ids):
                rgb_gt = (batch["images_all"][0,v].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                rgb_pred = (out["render"]["images_pred"][0,v].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                sem_gt = batch["semantic_label_all"][0,v].detach().cpu().numpy().astype(np.int32)
                sem_pred = semantic_pred[v].detach().cpu().numpy().astype(np.int32)
                ins_gt = batch["instance_label_all"][0,v].detach().cpu().numpy().astype(np.int64)
                ins_pred = instance_pred[v].detach().cpu().numpy().astype(np.int64)
                gt_inst_rgb = id_color_map(ins_gt, f"{window['scene']}:instance")
                pred_inst_rgb = np.zeros((H,W,3),np.uint8)
                for q in np.unique(ins_pred):
                    if q <= 0: continue
                    query = int(q-1)
                    gidx = q_to_gt.get(query)
                    key = (f"{window['scene']}:instance:{int(targets['gt_instance_ids'][0][gidx])}"
                           if gidx is not None else f"{window['scene']}:query:{query}")
                    pred_inst_rgb[ins_pred == q] = color_for(key)
                # Raw slot ownership masks, shown separately from post-processed instance output.
                raw_overlay = rgb_gt.copy()
                raw_mass = out["region_mass"][0,v,:100].detach().cpu()
                for query, gidx in q_to_gt.items():
                    iid = int(targets["gt_instance_ids"][0][gidx])
                    color = np.asarray(color_for(f"{window['scene']}:instance:{iid}"),np.float32)
                    mask = raw_mass[query].numpy() > 0.5
                    raw_overlay[mask] = (0.55*raw_overlay[mask] + 0.45*color).astype(np.uint8)
                # Object center and c±s support projection, using the evaluator camera convention.
                projection = rgb_pred.copy()
                canvas = Image.fromarray(projection); draw = ImageDraw.Draw(canvas)
                projected = []
                if v < 2:
                    state = out["states"][-1]
                    for query, gidx in q_to_gt.items():
                        iid = int(targets["gt_instance_ids"][0][gidx])
                        color = color_for(f"{window['scene']}:instance:{iid}")
                        c = state["c"][0,query:query+1]
                        s = state["s"][0,query]
                        corners = torch.stack([c[0] + torch.tensor([sx,sy,sz],device=c.device,dtype=c.dtype)*s
                                               for sx in (-1.,1.) for sy in (-1.,1.) for sz in (-1.,1.)])[None]
                        center_uv = project_points_means2d(c[None], batch["cam_view_all"][0:1,v:v+1],
                                                           batch["intrinsics_all"][0:1,v:v+1])[0,0,0]
                        corner_uv = project_points_means2d(corners, batch["cam_view_all"][0:1,v:v+1],
                                                           batch["intrinsics_all"][0:1,v:v+1])[0,0]
                        cx,cy = [float(x) for x in center_uv]
                        xy = corner_uv.detach().cpu().numpy()
                        xy = xy[np.isfinite(xy).all(-1)]
                        if len(xy):
                            box = (max(0,int(xy[:,0].min())),max(0,int(xy[:,1].min())),
                                   min(W-1,int(xy[:,0].max())),min(H-1,int(xy[:,1].max())))
                            if box[2] >= box[0] and box[3] >= box[1]:
                                draw.rectangle(box,outline=color,width=2)
                        draw.ellipse((cx-4,cy-4,cx+4,cy+4),fill=color,outline="white")
                        projected.append({"query":query,"gt_instance_id":iid,"center_uv": [cx,cy],
                                          "support_bbox_uv": [float(xy[:,0].min()),float(xy[:,1].min()),
                                                              float(xy[:,0].max()),float(xy[:,1].max())] if len(xy) else None})
                else:
                    canvas = Image.fromarray(projection)
                frames.append([rgb_gt,(rgb_pred),(semantic_color(np.where(sem_gt<=20,sem_gt,20))),
                               semantic_color(np.where(sem_pred<=20,sem_pred,20)),gt_inst_rgb,
                               pred_inst_rgb,raw_overlay,np.asarray(canvas)])
                draw_meta.append({"frame_id":frame_id,"is_context":v<2,"projection":projected})

            tile=224; header=86; nrows=len(frames); ncols=8
            sheet=Image.new("RGB",(tile*ncols,header+tile*nrows),"white"); d=ImageDraw.Draw(sheet)
            titles=["RGB GT","RGB reconstruction","semantic GT","semantic prediction",
                    "GT instances (stable colors)","postprocessed instances","raw matched slot masks (>0.5)","projected c / c±s support"]
            for col,title in enumerate(titles): d.text((col*tile+4,3),title,fill="black")
            d.text((4,22),f"{window['scene']} context={window['context']} novel={window['novel']} step={step}",fill="black")
            for fi, row in enumerate(frames):
                y=header+fi*tile
                for col,arr in enumerate(row):
                    im=Image.fromarray(arr).resize((tile,tile))
                    sheet.paste(im,(col*tile,y))
                d.text((3,y+3),f"frame {frame_ids[fi]} {'context' if fi<2 else 'novel'}",fill="yellow")
            outdir=OUT/"qualitative"/f"step_{step:08d}"
            outdir.mkdir(parents=True,exist_ok=True)
            png=outdir/f"pair_{pi:02d}.png"; sheet.save(png)
            metadata={"step":step,"pair_index":pi,"window":window,"frame_ids":frame_ids,
                      "predicted_instances_from_current_evaluator":True,
                      "matching":"current final_hungarian; one unified production matcher",
                      "matched_objects":match_rows,"frames":draw_meta,
                      "raw_slot_mask":"anchor ownership rendered to pixels; region_mass[q] > 0.5, pre-panoptic suppression",
                      "postprocessed_mask":"assemble_panoptic (official evaluator), includes score>=0.5 and alpha>0.05 unchanged",
                      "ap_ranking_score":"sum of 18 thing class probabilities; no AP ranking performed in panels",
                      "panel":str(png)}
            (outdir/f"pair_{pi:02d}.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
            step_meta[str(step)]["pairs"].append(metadata)
            print(f"[panel] step={step} pair={pi} {png}",flush=True)
        del model, blob
    (OUT/"qualitative"/"index.json").write_text(json.dumps(step_meta,indent=2,allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("reports", "panels"), default="reports")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "panels":
        generate_panels(args.device)
        return
    rows = task_metrics()
    (OUT / "task_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (OUT / "task_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    logs, fields = training_trace()
    manifest = read_json(RUN / "run_manifest.json") or {}
    plan = read_json(ORIGINAL / "group_plus/instance_state_v1_generalization/plan_C_frozen_5000.json") or {}
    entry = plan.get("entries", [])[4089] if len(plan.get("entries", [])) > 4089 else None
    failure = {
        "formal_job_id": "56991", "failed_step": 4090,
        "guard_source": "scripts/object_locus_v1_runtime.py:312 train_one_step scalar finite guard",
        "guard_order": ["step_loss computes prediction, reconstruction metrics, final Hungarian, understanding losses, aggregate loss",
                        "scalar finite checks in order: loss, loss_total, loss_recon, loss_understanding, loss_anchor_group, anchor_ce, anchor_dice",
                        "prediction tensor finite check", "GC backward and gradient check", "clip", "optimizer.step"],
        "first_named_failed_scalar": "loss (per recorded exception)",
        "plan_entry_step4090": entry,
        "plan_entry_matches_fixed_window": bool(entry and entry.get("scene") == "scene0014_00" and
             entry.get("window_index") == 236 and entry.get("context") == [2055, 2067] and
             entry.get("novel") == [2057, 2061]),
        "last_training_metrics_step": logs[-1].get("step") if logs else None,
        "last10_training_metrics": logs[-10:],
        "run_manifest": manifest,
        "checkpoint_inventory": [],
        "log_trace_fields": fields,
        "complete_log_excerpt_files": {
            "train_log": str(RUN / "train.log"),
            "train_events_log": str(RUN / "train_events.log"),
            "slurm_log": str(REPORTS / "slurm-56991.out"),
        },
        "input_artifact_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (REPORTS / "training_log_metrics.jsonl", RUN / "train.log",
                         RUN / "train_events.log", REPORTS / "slurm-56991.out",
                         RUN / "run_manifest.json", *[REPORTS / f"curves_{s}.json" for s in STEPS])
            if path.is_file()
        },
    }
    previous_context=read_json(OUT/"failure_context_initial.json") or {}
    if previous_context.get("checkpoint_inventory"):
        failure["checkpoint_inventory"] = previous_context["checkpoint_inventory"]
    else:
        ckroot = RUN / "checkpoints"
        for path in sorted(ckroot.glob("step_*/train_state.pt")):
            blob = __import__("torch").load(path, map_location="cpu", weights_only=False)
            complete = read_json(path.parent / "COMPLETE") or {}
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            failure["checkpoint_inventory"].append({"path": str(path), "step": blob.get("step"),
                "sha256": digest, "complete_marker": complete,
                "plan_position": blob.get("plan_position"), "git_commit": blob.get("git_commit"),
                "has_model": "model" in blob, "has_optimizer": "optimizer" in blob,
                "optimizer_state_tensors": len(blob.get("optimizer", {}).get("state", {})),
                "rng_keys": sorted(blob.get("rng", {}).keys()) if isinstance(blob.get("rng"), dict) else [],
                "source_file_hashes_match_manifest": blob.get("source_file_hashes") == manifest.get("source_file_hashes")})
    (OUT / "failure_context_initial.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
    prior_stats=read_json(OUT/"failure_tensor_stats.json")
    prior_fail=read_json(OUT/"replay_failure.json")
    if (prior_stats or {}).get("step") != 4090 and ((prior_stats is not None) or (prior_fail is not None)):
        (OUT/"diagnostic_attempt_history.json").write_text(json.dumps({
            "note":"These are failed diagnostic-instrumentation attempts, not the formal Object-Locus failure.",
            "prior_failure_tensor_stats":prior_stats,"prior_replay_failure":prior_fail,
            "jobs":[56993,56996,56997,56998,56999]},indent=2,allow_nan=False)+"\n")
    git_info = {
        "execution_worktree": str(REPO), "execution_head": git_text(REPO,"rev-parse","HEAD"),
        "execution_origin_main": git_text(REPO,"rev-parse","origin/main"),
        "execution_status_short": git_text(REPO,"status","--short"),
        "execution_diff_stat_worktree": git_text(REPO,"diff","--stat"),
        "diff_source_commit_to_head_names": git_text(REPO,"diff","--name-only","61b22322e79e973cdf7ff868afb6242f4253f3bb..HEAD"),
        "diff_failure_commit_to_head_names": git_text(REPO,"diff","--name-only","85cad0d163eafc0e197a79bb374d39f5cfed3814..HEAD"),
        "original_worktree": str(ORIGINAL),
        "original_head": git_text(ORIGINAL,"rev-parse","HEAD"),
        "original_origin_main": git_text(ORIGINAL,"rev-parse","origin/main"),
        "original_status_short": git_text(ORIGINAL,"status","--short"),
        "original_diff_stat_worktree": git_text(ORIGINAL,"diff","--stat"),
        "original_dirty_tree_touched_by_diagnostics": False,
        "baseline_commits": {"implementation":"61b22322e79e973cdf7ff868afb6242f4253f3bb",
                             "failure_record":"85cad0d163eafc0e197a79bb374d39f5cfed3814"},
    }
    (OUT/"git_provenance.json").write_text(json.dumps(git_info,indent=2,allow_nan=False)+"\n")
    (OUT / "last10_training_records.json").write_text(json.dumps(logs[-10:],indent=2,allow_nan=False)+"\n")
    slurm_text=(REPORTS/"slurm-56991.out").read_text(errors="replace")
    log_parts=[]
    for label, path in (("workspace train.log", RUN/"train.log"),
                        ("workspace train_events.log", RUN/"train_events.log"),
                        ("SLURM job 56991 stdout/stderr", REPORTS/"slurm-56991.out")):
        log_parts += [f"===== {label}: {path} =====\n", path.read_text(errors="replace"), "\n"]
    (OUT/"formal_logs_context.txt").write_text("".join(log_parts))
    trace_pos=slurm_text.rfind("Traceback (most recent call last):")
    (OUT/"formal_error_excerpt.txt").write_text(slurm_text[trace_pos:trace_pos+8000] if trace_pos>=0 else "Traceback marker not found\n")

    # Val32 step0 -> 3500 focused diagnostic, retaining the raw 0..1 scale.
    by = {(r["step"], r["split"], r["protocol"], r["scope"], r["metric"]): r for r in rows}
    key_metrics = ("mIoU_all", "mIoU_thing", "mIoU_stuff", "PQ", "mAP", "AP50", "PSNR")
    delta = {"split": "val32", "steps": [0, 3500], "metrics": {}}
    for protocol, scopes in (("local", ("context", "target_all")),
                             ("official_all", ("context", "target_all")),
                             ("official_novel", ("context", "novel")),
                             ("render", ("context", "novel", "target_all"))):
        for scope in scopes:
            for metric in key_metrics:
                a = by.get((0, "val32", protocol, scope, metric), {}).get("value")
                b = by.get((3500, "val32", protocol, scope, metric), {}).get("value")
                if number(a) and number(b) and (a == -1 or b == -1):
                    delta["metrics"][f"{protocol}.{scope}.{metric}"] = {
                        "step0": a, "step3500": b,
                        "step0_status": "UNDEFINED" if a == -1 else "PRESENT",
                        "step3500_status": "UNDEFINED" if b == -1 else "PRESENT",
                        "delta": "UNDEFINED: at least one official endpoint is -1"}
                elif number(a) and number(b):
                    delta["metrics"][f"{protocol}.{scope}.{metric}"] = {"step0": a, "step3500": b, "delta": b-a}
                else:
                    delta["metrics"][f"{protocol}.{scope}.{metric}"] = {"step0": a or "MISSING", "step3500": b or "MISSING"}
    per_class = {}
    grouping_existing = {}
    for scope in ("context", "target"):
        vals = []
        scope_tag = scope
        for st in (0, 3500):
            c = read_json(REPORTS / f"curves_{st}.json")
            item = c["splits"]["val32"]["local"][scope]
            vals.append({"step": st, "per_class_iou": item.get("per_class_iou"),
                         **{k: item.get(k) for k in ("tp_class_aware", "fp_class_aware", "fn_class_aware",
                               "tp_class_agnostic", "fp_class_agnostic", "fn_class_agnostic", "n_gt_instances",
                               "class_aware_recall50", "class_agnostic_recall50", "raw_recall50", "active_thing_queries")}})
            summary = read_json(REPORTS / f"eval_val32/step_{st:08d}/evaluation_summary.json") or {}
            window_rows = summary.get("local_rows", {}).get(scope_tag, [])
            dg = [r.get("anchor_group_diagnostics", {}) for r in window_rows]
            scalar_keys = ("assignment_entropy", "q_covariance_participation_ratio",
                           "projected_u_covariance_participation_ratio", "supported_gt_count",
                           "gt_best_anchor_dice_mean", "active_thing_queries")
            diag_summary = {}
            for k in scalar_keys:
                numbers = [x[k] for x in dg if number(x.get(k))]
                diag_summary[k + "_mean_over_windows"] = sum(numbers) / len(numbers) if numbers else "MISSING"
                diag_summary[k + "_window_values"] = numbers if numbers else "MISSING"
            for k in ("ownership_mass", "q_cosine", "evidence_overlap"):
                metric_keys = ("mean", "median", "p10", "p90", "max", "gini") if k == "ownership_mass" else ("offdiag_mean", "p90", "max")
                for mk in metric_keys:
                    numbers = [x.get(k, {}).get(mk) for x in dg if number(x.get(k, {}).get(mk))]
                    diag_summary[f"{k}.{mk}_mean_over_windows"] = sum(numbers) / len(numbers) if numbers else "MISSING"
            diag_summary["matched_classification_accuracy"] = "MISSING: not emitted by the stored val32 evaluator rows"
            vals[-1]["existing_grouping_diagnostics"] = diag_summary
        per_class[scope] = vals
        grouping_existing[scope] = {str(x["step"]): x.get("existing_grouping_diagnostics") for x in vals}
    delta["per_class_and_instance_counts"] = per_class
    delta["existing_val32_grouping_diagnostics_no_new_metric_computation"] = grouping_existing
    (OUT / "val32_step0_to_3500.json").write_text(json.dumps(delta, indent=2, allow_nan=False) + "\n")

    md = ["# Object-Locus V1 task metrics (existing evaluations)", "",
          "All ratio metrics are shown on their original 0–1 scale. `target_all` includes the registered target-all frames (context + novel); it is not novel-only. `official_novel` target is novel-only. `float_psnr_db.novel` is computed on actual novel frames. Local mAP/AP50 are MISSING because the local evaluator does not emit them.", "",
          "Local `mean_pq` is from the existing evaluator aggregate. Exact implementation: `/space/mawb/ssst/scripts/eval_instance_state_v1.py::_panoptic_pq`. It forms the final per-pixel winner query using ownership >0.5, alpha > `ALPHA_MIN`, and score-weighted ownership; for each GT thing instance it uses the best-overlap predicted query and counts TP when IoU > `IOU_TP`, otherwise FN. It computes per-class `sum(IoU for TP)/(TP + 0.5*FN)` over GT-present thing classes, then averages classes per window and averages windows. It does not accumulate unmatched-prediction FP or require predicted class agreement, so this local mean_pq diagnostic is not official COCO/PQ.", "",
          "Official evaluator values are retained as stored. A value `-1` is marked UNDEFINED, not converted.", "",
          "| step | split | protocol | scope | mIoU all | mIoU thing | mIoU stuff | PQ | mAP | AP50 | PSNR dB |", "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for step in STEPS:
        for split in ("train16", "val8", "val32"):
            for protocol, scopes in (("local", ("context", "target_all")),
                                     ("official_all", ("context", "target_all")),
                                     ("official_novel", ("context", "novel")),
                                     ("render", ("context", "novel", "target_all"))):
                for scope in scopes:
                    vals=[]
                    for metric in key_metrics:
                        v=by.get((step,split,protocol,scope,metric),{}).get("value")
                        if v is None: vals.append("MISSING")
                        elif number(v) and v == -1: vals.append("-1 (UNDEFINED)")
                        elif number(v): vals.append(f"{v:.6f}")
                        else: vals.append(str(v))
                    md.append(f"| {step} | {split} | {protocol} | {scope} | " + " | ".join(vals) + " |")
    md += ["", "## Val32 step 0 → 3500", "", "```json", json.dumps(delta, indent=2, allow_nan=False), "```", ""]
    (OUT / "task_metrics.md").write_text("\n".join(md))

    # Main trace markdown excerpt and known guard ordering; replay later appends actual failure values.
    trace_md = ["# Object-Locus V1 failure analysis (pre-replay evidence)", "",
                f"- Existing logged points: {len(logs)} (steps {logs[0]['step']}–{logs[-1]['step']}); per-100-step logs do not provide step 4089 telemetry.",
                "- Last ten rows are embedded in `failure_context.json`; full union-schema trace is `training_trace.csv`.",
                "- At step 4090 the first scalar guard is `metrics['loss']` in `scripts/object_locus_v1_runtime.py:312`; named exception: `nonfinite loss at step 4090`.",
                "- Code order: all step_loss forward/reconstruction/understanding terms are computed before this scalar guard. If that guard raises, prediction tensor finite validation, backward, gradient finite check, clipping and optimizer.step do not run.",
                "- Checkpoint step3500 includes model/optimizer/RNG/plan position; no step4089 state was saved by the formal run.",
                "- This is pre-replay evidence; root cause and actual first nonfinite op require isolated replay.", ""]
    (OUT / "analysis_report.md").write_text("\n".join(trace_md))
    _write_chatgpt_summary(rows,logs)
    _replay_summary()
    _write_full_analysis_report(rows,logs)
    print(f"wrote {OUT}; rows={len(rows)} trace={len(logs)}")


if __name__ == "__main__":
    main()
