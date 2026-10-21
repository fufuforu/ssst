"""Registered Object-Locus V2.1 small-sample gate followed by expansion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from scripts.object_locus_v2_1_runtime import (
    ASSET_ROOT, MANIFEST, MANIFEST_SHA, MONITOR_SHA, OBJECT_PEAK_LR, PRETRAINED_SHA,
    REPORTS_DEFAULT, RUN_ROOT_DEFAULT, RECON_PEAK_LR, SEED, TRAIN_ROOT, VAL_ROOT,
    build_batch, build_model, build_optimizer, build_v2_1_splits, capture_rng,
    load_v2_1_assets, restore_rng, seed_everything, train_one_step, trainability_counts,
    write_json, jsonable,
)
from scripts.object_locus_v2_1_runtime import sha256_file

S_EVAL_EPOCHS = (0, 2, 4, 8, 16)
PANEL_SPLITS = ("train_probe16", "same_scene_holdout16", "dev8", "val32")


def _check_cuda():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Object-Locus V2.1 requires exactly one CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 3090": raise RuntimeError(f"registered run requires RTX3090, got {gpu}")
    if torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
        raise RuntimeError("registered GPU is not 24GB-class")
    node = os.uname().nodename
    if not (node.startswith("3dimage-13") or node.startswith("3dimage-11")):
        raise RuntimeError(f"unregistered GPU node {node}")
    return gpu, node


def _stable_commit():
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _verify_execution_source():
    import inspect
    from tokengs.models.object_locus_v2_1 import LocusGSObjectLocusV2_1Recon
    source=Path(inspect.getfile(LocusGSObjectLocusV2_1Recon)).resolve()
    if not source.is_relative_to(REPO.resolve()):
        raise RuntimeError(f"V2.1 model imported outside execution worktree: {source}")
    payload={"git_sha":_stable_commit(),"model_module_file":str(source),
             "worktree":str(REPO.resolve())}
    print(json.dumps(payload),flush=True)
    return payload


def _truncate_training_log(path, completed_global_step):
    path=Path(path)
    if not path.exists():return
    kept=[]
    for line in path.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if int(row.get("global_optimizer_step",-1))<=int(completed_global_step):
            kept.append(json.dumps(row,allow_nan=False))
    path.write_text("".join(line+"\n" for line in kept))


def _seen_scenes(windows): return {x["scene"] for x in windows}


def _eval_splits(model, opt, stage, epoch, global_step, splits, reports, *, official, panels):
    from scripts.eval_object_locus_v2_1 import evaluate_windows
    results = {}
    model.eval()
    for name, windows in splits.items():
        result = evaluate_windows(model, opt, windows, global_step, name, reports,
                                  "cuda", build_batch, official=official,
                                  panels=panels and name in PANEL_SPLITS)
        results[name] = result
    payload = {"stage": stage, "epoch": int(epoch), "global_optimizer_step": int(global_step),
               "official": bool(official), "splits": results}
    write_json(reports / f"curves_{stage.lower()}_epoch{epoch:02d}.json", payload)
    model.train()
    return payload


def _save_checkpoint(path, model, optimizer, *, stage, stage_step, epoch,
                     plan_position, global_step, commit, data_manifest, training_plan,
                     opt_audit, exposure):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": capture_rng(), "stage": stage, "stage_step": int(stage_step),
        "global_optimizer_step": int(global_step), "epoch": int(epoch),
        "plan_position": int(plan_position), "git_sha": commit,
        "architecture_name": model.architecture_name,
        "source_hashes": {"pretrained": PRETRAINED_SHA, "manifest": MANIFEST_SHA,
                          "monitor": dict(MONITOR_SHA)},
        "data_manifest_sha256": sha256_file(data_manifest),
        "training_plan_sha256": sha256_file(training_plan),
        "optimizer_config": opt_audit, "config": jsonable(vars(model.opt)),
        "exposure_stats": exposure}
    temp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(payload, temp); os.replace(temp, path)
    latest = Path(path).parent / "latest_recovery.pt"
    link = latest.with_suffix(".tmp")
    try:
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(Path(path).name)
        os.replace(link, latest)
    except OSError:
        if link.exists() or link.is_symlink(): link.unlink()
        shutil.copy2(path, latest)


def _restore_checkpoint(path, model, optimizer, commit, data_manifest, training_plan):
    state = torch.load(path, map_location="cpu", weights_only=False)
    checks = {"git_sha": commit, "architecture_name": model.architecture_name,
              "source_hashes": {"pretrained": PRETRAINED_SHA, "manifest": MANIFEST_SHA,
                                "monitor": dict(MONITOR_SHA)},
              "data_manifest_sha256": sha256_file(data_manifest),
              "training_plan_sha256": sha256_file(training_plan)}
    for key, expected in checks.items():
        if state.get(key) != expected: raise RuntimeError(f"checkpoint resume mismatch: {key}")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state


def _run_epoch(model, optimizer, windows, *, stage, epoch, start_position,
               global_start, reports, run, commit, data_manifest, training_plan,
               opt_audit, exposure, log_path):
    if stage == "S":
        perm = np.random.default_rng(42 + epoch).permutation(len(windows)).tolist()
    else:
        perm = np.random.default_rng(10042 + epoch).permutation(len(windows)).tolist()
    if start_position and start_position > len(perm): raise RuntimeError("bad epoch plan_position")
    model.train()
    for pos in range(start_position, len(perm)):
        win = windows[perm[pos]]
        batch = build_batch(model.opt, win, "cuda")
        stage_step = epoch * len(windows) + pos + 1
        global_step = global_start + pos + 1
        if stage == "S":
            lr_mult = min(stage_step / 200.0, 1.0)
            uweight = min(stage_step / 200.0, 1.0)
            lrs = (OBJECT_PEAK_LR * lr_mult, RECON_PEAK_LR * lr_mult)
        else:
            t = stage_step
            u = (t - 1) / max(1, (16 * len(windows) - 1))
            lr_mult = 0.1 + 0.9 * (1.0 + math.cos(math.pi * u)) / 2.0
            uweight = 1.0
            lrs = (OBJECT_PEAK_LR * lr_mult, RECON_PEAK_LR * lr_mult)
        output, metrics = train_one_step(model, optimizer, batch, global_step,
            understanding_weight_value=uweight, lr_values=lrs,
            failure_capture_dir=run / "failures",
            failure_context={"stage": stage, "stage_epoch": epoch, "stage_step": stage_step,
                             "plan_position": pos, "window": win, "optimizer_updated": False})
        if stage_step % 20 == 0:
            row = {k: v for k, v in metrics.items() if k != "classification_confusion"}
            row.update({"stage": stage, "epoch": epoch + 1, "stage_step": stage_step,
                        "global_optimizer_step": global_step, "plan_position": pos + 1,
                        "window_identity": {"scene": win["scene"], "context": win["context"], "novel": win["novel"]},
                        "object_lr": lrs[0], "reconstruction_lr": lrs[1],
                        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                        "reserved_gib": torch.cuda.memory_reserved() / 1024**3})
            with Path(log_path).open("a") as f: f.write(json.dumps(row, allow_nan=False) + "\n")
        del output, metrics, batch
    return len(perm)


def _metric(splits, name, scope="context"):
    return splits[name]["local"][scope]


def _official_ap50(split_result, scope="context"):
    official = split_result.get("official") or {}
    all_result = official.get("all") or {}
    key = "context_map" if scope == "context" else "target_map"
    metric = all_result.get(key) or {}
    return metric.get("map_50")


def _stage_s_gate(endpoint, baseline):
    train = _metric(endpoint["splits"], "train_probe16")
    hold = _metric(endpoint["splits"], "same_scene_holdout16")
    dev = _metric(endpoint["splits"], "dev8")
    train_gt = max(1, int(train.get("raw_gt_count", 0)))
    cond_train = train.get("conditional_classification_accuracy", 0.0)
    joint_train = train.get("joint_classification_accuracy", 0.0)
    train_ca = train.get("class_agnostic_tp", 0) / max(1, train.get("gt_count", 0))
    hold_ca = hold.get("class_agnostic_tp", 0) / max(1, hold.get("gt_count", 0))
    drops = {}
    for name in ("train_probe16", "same_scene_holdout16", "dev8"):
        cur = _metric(endpoint["splits"], name)
        base = _metric(baseline["splits"], name)
        drops[name] = {"context_drop_db": base["psnr"] - cur["psnr"],
                       "true_novel_drop_db": baseline["splits"][name].get("true_novel_psnr", 0.0)
                                              - endpoint["splits"][name].get("true_novel_psnr", 0.0)}
    checks = {
      "train_raw_mask_iou50_fraction_ge_0_40": train.get("raw_best_iou_ge_0_5_fraction", 0) >= .40,
      "train_conditional_accuracy_ge_0_60": cond_train >= .60,
      "train_joint_accuracy_ge_0_50": joint_train >= .50,
      "train_matched_objectness_recall_ge_0_70": train.get("matched_objectness_recall_p50", 0) >= .70,
      "train_final_ca_recall_ge_0_20": train_ca >= .20,
      "holdout_raw_mask_iou50_fraction_ge_0_20": hold.get("raw_best_iou_ge_0_5_fraction", 0) >= .20,
      "holdout_conditional_accuracy_ge_0_30": hold.get("conditional_classification_accuracy", 0) >= .30,
      "holdout_final_ca_recall_ge_0_10": hold_ca >= .10,
      "holdout_official_context_ap50_gt_0": (_official_ap50(endpoint["splits"]["same_scene_holdout16"]) or 0) >= .01,
      "dev_conditional_accuracy_ge_0_10": dev.get("conditional_classification_accuracy", 0) >= .10,
      "dev_final_class_aware_tp_ge_1": dev.get("class_aware_tp", 0) >= 1,
      "dev_official_context_ap50_gt_0": (_official_ap50(endpoint["splits"]["dev8"]) or 0) > 0,
    }
    for name, values in drops.items():
        checks[f"{name}_context_psnr_drop_le_0_5db"] = values["context_drop_db"] <= .5
        checks[f"{name}_true_novel_psnr_drop_le_0_5db"] = values["true_novel_drop_db"] <= .5
    return {"passed": all(checks.values()), "checks": checks,
            "train_probe16_context": {"raw_best_iou50_fraction": train.get("raw_best_iou_ge_0_5_fraction"),
              "raw_gt_count": train_gt, "conditional_accuracy": cond_train,
              "joint_accuracy": joint_train,"matched_objectness_recall_p50":train.get("matched_objectness_recall_p50"),
              "class_agnostic_recall50":train_ca},
            "same_scene_holdout16_context": {"raw_best_iou50_fraction":hold.get("raw_best_iou_ge_0_5_fraction"),
              "conditional_accuracy":hold.get("conditional_classification_accuracy"),
              "class_agnostic_recall50":hold_ca,"official_ap50":_official_ap50(endpoint["splits"]["same_scene_holdout16"])},
            "dev8_context":{"conditional_accuracy":dev.get("conditional_classification_accuracy"),
              "class_aware_tp":dev.get("class_aware_tp"),"official_ap50":_official_ap50(endpoint["splits"]["dev8"])},
            "psnr_drops":drops}


def _stage_s(reports, run, splits, commit, data_manifest, training_plan):
    reports.mkdir(parents=True, exist_ok=True); run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()) and not (run / "stage_s_manifest.json").exists():
        raise RuntimeError(f"Stage S run directory has unregistered files: {run}")
    model,opt,transfer=build_model("cuda")
    optimizer,opt_audit=build_optimizer(model)
    counts=trainability_counts(model)
    if counts["frozen_numel"]: raise RuntimeError("V2.1 requires all parameters trainable")
    small=splits["small_train_windows"]; n=len(small); total=16*n
    if total != json.loads(training_plan.read_text())["stage_s_updates"]:
        raise RuntimeError("Stage-S total updates differ from saved deterministic plan")
    manifest={"architecture":"LOCUSGS_OBJECT_LOCUS_V2_1","recipe":"OBJECT_LOCUS_V2_1_MASK_CONDITIONED_CLASSIFIER",
      "git_sha":commit,"pretrained_sha256":PRETRAINED_SHA,"manifest_sha256":MANIFEST_SHA,
      "data_manifest_sha256":sha256_file(data_manifest),"training_plan_sha256":sha256_file(training_plan),
      "stage_s_windows":n,"stage_s_epochs":16,"stage_s_updates":total,"legacy_no_object_ce_used":False,
      "global_seed":42,"object_seed":31415,
      "optimizer":opt_audit,"trainability":counts}
    mpath=run/"stage_s_manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest: raise RuntimeError("Stage S run manifest conflict")
    write_json(mpath,manifest); write_json(reports/"stage_s_manifest.json",manifest)
    write_json(reports/"transfer_audit.json",transfer); write_json(reports/"optimizer_audit.json",opt_audit)
    log_path=reports/"training_metrics.jsonl"
    checkpoint=max(run.glob("checkpoint_S_epoch_*.pt"),key=lambda p:int(p.stem.split("_")[-1]),default=None)
    start_epoch=0; global_start=0; baseline=None; endpoint=None
    if checkpoint:
        st=_restore_checkpoint(checkpoint,model,optimizer,commit,data_manifest,training_plan)
        start_epoch=int(st["epoch"]); global_start=int(st["global_optimizer_step"])
        _truncate_training_log(log_path,global_start)
        if st.get("stage")!="S": raise RuntimeError("latest Stage S checkpoint has incorrect stage")
        if start_epoch==16:
            baseline=json.loads((reports/"curves_s_epoch00.json").read_text())
        if start_epoch in S_EVAL_EPOCHS[1:] and not (reports/f"curves_s_epoch{start_epoch:02d}.json").exists():
            chosen={k:splits[k] for k in ("train_probe16","same_scene_holdout16","dev8","legacy_train16")}
            if start_epoch==16:chosen.update({"val8":splits["val8"],"val32":splits["val32"]})
            endpoint=_eval_splits(model,opt,"S",start_epoch,global_start,chosen,reports,
                official=start_epoch in (8,16),panels=start_epoch in (8,16))
            if start_epoch==16:write_json(reports/"stage_s_epoch16.json",endpoint)
        if start_epoch==16:
            endpoint=json.loads((reports/"curves_s_epoch16.json").read_text())
            if not (reports/"stage_s_epoch16.json").exists():write_json(reports/"stage_s_epoch16.json",endpoint)
            gate_path=reports/"stage_s_expansion_gate.json"
            gate=json.loads(gate_path.read_text()) if gate_path.exists() else _stage_s_gate(endpoint,baseline)
            if not gate_path.exists():write_json(gate_path,gate)
    if start_epoch==0:
        base_splits={k:splits[k] for k in ("train_probe16","same_scene_holdout16","dev8","legacy_train16","val8","val32")}
        baseline=_eval_splits(model,opt,"S",0,0,base_splits,reports,official=True,panels=True)
        _save_checkpoint(run/"checkpoint_S_epoch_00.pt",model,optimizer,stage="S",stage_step=0,epoch=0,
          plan_position=0,global_step=0,commit=commit,data_manifest=data_manifest,training_plan=training_plan,
          opt_audit=opt_audit,exposure={"stage_s_epochs_completed":0,"small_train_window_count":n})
        start_epoch=0
    for epoch in range(start_epoch,16):
        if epoch < start_epoch: continue
        _run_epoch(model,optimizer,small,stage="S",epoch=epoch,start_position=0,global_start=global_start,
          reports=reports,run=run,commit=commit,data_manifest=data_manifest,training_plan=training_plan,
          opt_audit=opt_audit,exposure={"stage_s_window_count":n},log_path=log_path)
        global_start=(epoch+1)*n
        epoch_num=epoch+1
        _save_checkpoint(run/f"checkpoint_S_epoch_{epoch_num:02d}.pt",model,optimizer,stage="S",
          stage_step=epoch_num*n,epoch=epoch_num,plan_position=n,global_step=global_start,
          commit=commit,data_manifest=data_manifest,training_plan=training_plan,opt_audit=opt_audit,
          exposure={"stage_s_epochs_completed":epoch_num,"small_train_window_count":n,
                    "each_window_exposures":epoch_num})
        if epoch_num in S_EVAL_EPOCHS[1:]:
            names=("train_probe16","same_scene_holdout16","dev8","legacy_train16")
            chosen={k:splits[k] for k in names}
            official=epoch_num in (8,16)
            if epoch_num==16: chosen.update({"val8":splits["val8"],"val32":splits["val32"]})
            endpoint=_eval_splits(model,opt,"S",epoch_num,global_start,chosen,reports,
                                  official=official,panels=epoch_num in (8,16))
            if epoch_num==16: write_json(reports/"stage_s_epoch16.json",endpoint)
        if epoch_num==16:
            # Epoch-16 eval contains S endpoint metrics. Ensure the S0 baseline includes matching splits.
            if not (reports/"curves_s_epoch00.json").exists(): raise RuntimeError("missing S0 baseline result")
            baseline=json.loads((reports/"curves_s_epoch00.json").read_text())
            gate=_stage_s_gate(endpoint,baseline)
            write_json(reports/"stage_s_expansion_gate.json",gate)
            write_json(reports/"stage_s_status.json",{"stage_s_completed_epochs":16,
              "stage_s_optimizer_updates":total,"stage_e_started":bool(gate["passed"]),
              "task_status":"Stage S passed; Stage E authorized" if gate["passed"] else "training completed; expansion gate failed"})
            if not gate["passed"]:
                write_json(run/"final_status.json",{"stage":"S","epochs":16,"optimizer_updates":total,
                  "stage_e_started":False,"task_status":"training completed; expansion gate failed"})
                return model,optimizer,opt,opt_audit,transfer,baseline,endpoint,gate,global_start
    return model,optimizer,opt,opt_audit,transfer,baseline,endpoint,gate,global_start


def _stage_e(reports,run,model,optimizer,opt,opt_audit,splits,commit,data_manifest,training_plan,
             global_start,baseline_s,endpoint_s):
    expanded=splits["expanded_train_windows"]; n=len(expanded); total=16*n
    em=run/"stage_e"; er=reports/"stage_e"; em.mkdir(parents=True,exist_ok=True); er.mkdir(parents=True,exist_ok=True)
    manifest={"architecture":"LOCUSGS_OBJECT_LOCUS_V2_1","stage":"E","git_sha":commit,
      "training_plan_sha256":sha256_file(training_plan),"stage_e_windows":n,"stage_e_epochs":16,
      "stage_e_updates":total,"total_updates":global_start+total,"starts_from_stage_s_global_step":global_start,
      "optimizer_state_continued":True,"understanding_weight":1.0,"optimizer":opt_audit}
    write_json(em/"stage_e_manifest.json",manifest);write_json(er/"stage_e_manifest.json",manifest)
    # E0 is exactly the S endpoint and reuses its already-completed evaluation outputs.
    write_json(er/"stage_e_epoch00_reference.json",{"stage":"E","epoch":0,
      "global_optimizer_step":global_start,"source_stage_s_epoch16":str(reports/"stage_s/stage_s_epoch16.json"),
      "reused":True})
    checkpoint=max(em.glob("checkpoint_E_epoch_*.pt"),key=lambda p:int(p.stem.split("_")[-1]),default=None)
    start_epoch=0
    log_path=reports/"training_metrics.jsonl"
    if checkpoint:
        st=_restore_checkpoint(checkpoint,model,optimizer,commit,data_manifest,training_plan)
        if st.get("stage")!="E":raise RuntimeError("Stage E resume checkpoint stage mismatch")
        start_epoch=int(st["epoch"]);global_start=int(st["global_optimizer_step"])
        _truncate_training_log(log_path,global_start)
        if start_epoch in S_EVAL_EPOCHS[1:] and not (er/f"curves_e_epoch{start_epoch:02d}.json").exists():
            _eval_splits(model,opt,"E",start_epoch,global_start,
                {k:splits[k] for k in ("train_probe16","same_scene_holdout16","dev8","legacy_train16","val8","val32")},
                er,official=start_epoch in (8,16),panels=start_epoch in (8,16))
    all_splits={k:splits[k] for k in ("train_probe16","same_scene_holdout16","dev8","legacy_train16","val8","val32")}
    for epoch in range(start_epoch,16):
        _run_epoch(model,optimizer,expanded,stage="E",epoch=epoch,start_position=0,
          global_start=global_start,reports=er,run=em,commit=commit,data_manifest=data_manifest,
          training_plan=training_plan,opt_audit=opt_audit,exposure={"stage_e_window_count":n},log_path=log_path)
        global_start += n
        epoch_num=epoch+1
        _save_checkpoint(em/f"checkpoint_E_epoch_{epoch_num:02d}.pt",model,optimizer,stage="E",
          stage_step=epoch_num*n,epoch=epoch_num,plan_position=n,global_step=global_start,
          commit=commit,data_manifest=data_manifest,training_plan=training_plan,opt_audit=opt_audit,
          exposure={"stage_s_window_count":len(splits["small_train_windows"]),
                    "stage_e_window_count":n,"each_stage_s_window_exposure":16,
                    "each_expanded_window_exposure":epoch_num})
        if epoch_num in S_EVAL_EPOCHS[1:]:
            official=epoch_num in (8,16)
            _eval_splits(model,opt,"E",epoch_num,global_start,all_splits,er,
                         official=official,panels=epoch_num in (8,16))
    write_json(em/"final_status.json",{"stage":"E","epochs":16,"stage_e_updates":total,
      "total_optimizer_updates":global_start,"stage_e_completed":True,"task_status":"training completed; evaluate task metrics independently"})
    return global_start


def _write_result_bundle(reports, repo=REPO):
    """Compact user-facing review bundle; smoke can exercise the same schemas."""
    reports=Path(reports); bundle=reports/"review_bundle"
    data_manifest_path=reports/"data_manifest.json"
    data_manifest=json.loads(data_manifest_path.read_text()) if data_manifest_path.exists() else {}
    if bundle.exists(): shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    curves=[]
    for p in sorted(reports.rglob("eval_*_step*.json")):
        try:
            d=json.loads(p.read_text())
            if "local" in d: curves.append({"path":str(p.relative_to(reports)),"payload":d})
        except Exception: continue
    metric_rows=[]; gt_rows=[]; confusion={}; waterfall=[]
    for item in curves:
        p=item["payload"]; stage="E" if "/stage_e/" in item["path"] else "S"
        for scope,local in p.get("local",{}).items():
            off=p.get("official",{}) or {}; allr=off.get("all") or {}; nov=off.get("novel") or {}
            mapkey="context_map" if scope=="context" else "target_map"
            am=allr.get(mapkey) or {}; nm=(nov.get(mapkey) or {})
            metric_rows.append({"stage":stage,"eval_path":item["path"],"split":p.get("split"),"scope":scope,
              "semantic_miou":local.get("mIoU_all_nonempty"),"thing_miou":local.get("mIoU_thing"),
              "stuff_miou":local.get("mIoU_stuff"),"local_pq":local.get("local_pq"),
              "official_all_miou":allr.get("context_miou" if scope=="context" else "target_miou"),
              "official_all_pq":allr.get("context_pq" if scope=="context" else "target_pq"),
              "official_all_map":am.get("map"),"official_all_ap50":am.get("map_50"),
              "official_novel_miou":nov.get("context_miou" if scope=="context" else "target_miou"),
              "official_novel_pq":nov.get("context_pq" if scope=="context" else "target_pq"),
              "official_novel_map":nm.get("map"),"official_novel_ap50":nm.get("map_50"),
              "psnr":local.get("psnr"),"true_novel_psnr":p.get("true_novel_psnr") if scope=="target_all" else None,
              "raw_mask_iou50_fraction":local.get("raw_best_iou_ge_0_5_fraction"),
              "conditional_classification_accuracy":local.get("conditional_classification_accuracy"),
              "joint_classification_accuracy":local.get("joint_classification_accuracy"),
              "matched_objectness_recall_p50":local.get("matched_objectness_recall_p50"),
              "unmatched_objectness_p50_fraction":local.get("unmatched_objectness_p50_fraction"),
              "ca_tp":local.get("class_agnostic_tp"),"ca_fp":local.get("class_agnostic_fp"),"ca_fn":local.get("class_agnostic_fn"),
              "cw_tp":local.get("class_aware_tp"),"cw_fp":local.get("class_aware_fp"),"cw_fn":local.get("class_aware_fn"),
              "gt_count":local.get("gt_count"),"filtering_waterfall":local.get("filtering_waterfall")})
            confusion[f"{stage}:{p.get('split')}:{p.get('step')}:{scope}"]=local.get("conditional_class_confusion")
            for row in p.get("local_rows",{}).get("context" if scope=="context" else "target",[]):
                for gt in row.get("raw_best_iou",[]):gt_rows.append({"stage":stage,"split":p.get("split"),"scope":scope,"step":p.get("step"),**gt})
            waterfall.append({"stage":stage,"split":p.get("split"),"step":p.get("step"),"scope":scope,**(local.get("filtering_waterfall") or {})})
    for name,rows in (("task_metrics.csv",metric_rows),("per_gt_mask_and_classification.csv",gt_rows),("filtering_waterfall.csv",waterfall)):
        with (bundle/name).open("w",newline="") as f:
            fields=sorted({key for row in rows for key in row})
            if not fields: fields=["status"]
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader()
            for row in rows:writer.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in row.items()})
    write_json(bundle/"task_metrics.json",curves);write_json(bundle/"classification_confusion.json",confusion)
    for name in ("data_manifest.json","training_plan.json","run_manifest.json","stage_s_expansion_gate.json","smoke_result.json"):
        src=reports/name
        if src.exists():shutil.copy2(src,bundle/name)
    all_training_rows=[]
    for logs in sorted(reports.rglob("training_metrics.jsonl")):
        all_training_rows.extend(json.loads(line) for line in logs.read_text().splitlines() if line.strip())
    (bundle/"training_metrics.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in all_training_rows))
    for name in ("stage_s","stage_e"):
        for src in reports.glob(f"{name}/**/*.out"):
            dst=bundle/"key_logs"/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in reports.glob("slurm/**/*.out"):
        dst=bundle/"key_logs"/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    code_names=["docs/object_locus_v2_1_codex_spec.md","tokengs/models/object_locus_v2_1.py",
      "tokengs/models/object_locus_v2_1_controller.py","tokengs/models/object_locus_v2_1_loss.py",
      "scripts/object_locus_v2_1_runtime.py","scripts/train_object_locus_v2_1.py",
      "scripts/eval_object_locus_v2_1.py","scripts/export_object_locus_v2_1_official.py",
      "scripts/smoke_object_locus_v2_1.py","scripts/submit_object_locus_v2_1.sh",
      "tests/test_object_locus_v2_1_contracts.py","tests/test_object_locus_v2_1_gradients.py",
      "tokengs/models/__init__.py","tokengs/options.py"]
    for rel in code_names:
        src=repo/rel
        if src.exists():
            dst=bundle/"code"/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    patch=subprocess.check_output(["git","-C",str(repo),"diff","1f8ddf2333a4ad54f541e23574719c52f4d87d5f..HEAD"],text=True)
    (bundle/"source.patch").write_text(patch)
    (bundle/"git_status.txt").write_text(subprocess.check_output(["git","-C",str(repo),"status","--short"],text=True))
    for src in sorted(reports.rglob("qualitative/**/*.png")):
        dst=bundle/"qualitative"/src.relative_to(reports);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    status_path=reports/"final_status.json"
    gate_path=reports/"stage_s_expansion_gate.json"
    status=json.loads(status_path.read_text()) if status_path.exists() else {"task_status":"implementation/smoke only; formal training not started"}
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else None
    summary_rows=[]
    for item in curves:
        payload=item["payload"];stage="E" if "/stage_e/" in item["path"] else "S"
        for split,split_result in payload.get("splits",{}).items():
            ctx=split_result.get("local",{}).get("context",{})
            official=(split_result.get("official") or {}).get("all") or {}
            cmap=official.get("context_map") or {}
            summary_rows.append((stage,payload.get("epoch"),split,ctx.get("mIoU_all_nonempty"),
                ctx.get("local_pq"),cmap.get("map"),cmap.get("map_50"),ctx.get("psnr"),
                ctx.get("raw_best_iou_ge_0_5_fraction"),ctx.get("conditional_classification_accuracy"),
                ctx.get("joint_classification_accuracy")))
    report=["# Object-Locus V2.1 阶段报告\n\n",
      "本轮采用mask-conditioned分类和独立objectness；保留V2 scene states、independent masks与child residual。该设计是待验证方案，不代表已证明跨场景泛化。\n\n",
      f"- 当前状态：`{status.get('task_status')}`；Stage E started：`{status.get('stage_e_started', False)}`。\n",
      f"- 数据训练计划：Stage S {len(data_manifest.get('small_train_windows', []))}窗 / {data_manifest.get('window_counts', {}).get('small_train', 0) * 16} updates；Stage E {data_manifest.get('window_counts', {}).get('expanded_train', 0)}窗 / {data_manifest.get('window_counts', {}).get('expanded_train', 0) * 16} updates（仅 gate 通过时）。\n",
      "- 任务训练状态见 `run_manifest.json` 与 `stage_s_expansion_gate.json`。\n",
      "- 指标表分开记录local semantic/panoptic、official all/novel、raw masks、条件分类/joint分类、objectness、filtering waterfall和PSNR。\n",
      "- true novel PSNR独立列出；target-all PSNR没有冒充novel指标。\n",
      "- 本轮使用GT camera poses，不等同于完整unposed SIU3R benchmark。\n\n",
      "## Registered evaluation summary\n\n",
      "| Stage | Epoch | Split | context mIoU | local PQ | official all mAP | AP50 | PSNR | raw IoU≥0.5 GT fraction | conditional acc | joint acc |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for row in summary_rows:
        report.append("| "+" | ".join("MISSING" if x is None else str(x) for x in row)+" |\n")
    if gate is not None:
        report.extend(["\n## Stage S expansion gate\n\n",
                       f"Passed: `{gate.get('passed')}`. Individual fixed checks: `{json.dumps(gate.get('checks',{}),ensure_ascii=False)}`\n\n"])
    report.append("详见 [task_metrics.csv](task_metrics.csv)、[per_gt_mask_and_classification.csv](per_gt_mask_and_classification.csv)、[training_metrics.jsonl](training_metrics.jsonl)、[task_metrics.json](task_metrics.json)。图片在qualitative目录。checkpoint与数据集未打包。\n")
    (bundle/"analysis_report.md").write_text("".join(report))
    (bundle/"README.md").write_text("# Object-Locus V2.1 review bundle\n\nSee [report](analysis_report.md), [metrics](task_metrics.csv), [training metrics](training_metrics.jsonl). Upload every `result_bundle_qualitative_*.zip` with this main archive.\n")
    main=reports/"result_bundle.zip"
    for old in reports.glob("result_bundle_qualitative_*.zip"):old.unlink()
    limit=28*1024*1024
    with zipfile.ZipFile(main,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in bundle.rglob("*"):
            if p.is_file():z.write(p,p.relative_to(bundle))
    if main.stat().st_size>limit:
        with zipfile.ZipFile(main,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in bundle.rglob("*"):
                if p.is_file() and "qualitative" not in p.parts:z.write(p,p.relative_to(bundle))
        images=[p for p in (bundle/"qualitative").rglob("*.png")]
        groups=[];cur=[];size=0
        for p in images:
            if cur and size+p.stat().st_size>24*1024*1024:groups.append(cur);cur=[];size=0
            cur.append(p);size+=p.stat().st_size
        if cur:groups.append(cur)
        for i,group in enumerate(groups,1):
            with zipfile.ZipFile(reports/f"result_bundle_qualitative_{i:03d}.zip","w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in group:z.write(p,p.relative_to(bundle))
    with zipfile.ZipFile(main) as z:
        if z.testzip() is not None:raise RuntimeError("V2.1 bundle integrity failure")
        names=set(z.namelist())
        for name in names:
            if name.endswith(".json"):
                json.loads(z.read(name))
            elif name.endswith(".csv"):
                import io
                list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8"))))
        readme=z.read("README.md").decode("utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)",readme):
            if "://" not in target and target not in names:
                raise RuntimeError(f"bundle README link is missing: {target}")
    from PIL import Image
    for image in (bundle/"qualitative").rglob("*.png"):
        with Image.open(image) as opened: opened.verify()
    if main.stat().st_size > limit:
        raise RuntimeError("main Object-Locus V2.1 bundle exceeds 28 MiB")
    for part in reports.glob("result_bundle_qualitative_*.zip"):
        if part.stat().st_size > limit:
            raise RuntimeError(f"qualitative bundle exceeds 28 MiB: {part}")
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None:raise RuntimeError(f"qualitative bundle integrity failure: {part}")
            for name in z.namelist():
                if name.endswith(".png"):
                    from io import BytesIO
                    with Image.open(BytesIO(z.read(name))) as opened:opened.verify()
    return str(main)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("train",),default="train")
    parser.add_argument("--device",default="cuda");args=parser.parse_args()
    if args.device!="cuda":raise RuntimeError("registered V2.1 run is CUDA-only")
    gpu,node=_check_cuda();source_info=_verify_execution_source();reports=REPORTS_DEFAULT;run=RUN_ROOT_DEFAULT
    if run.exists() and any(run.iterdir()) and not (run/"run_manifest.json").exists():
        raise RuntimeError(f"V2.1 checkpoint directory already contains unregistered content: {run}")
    manifest=load_v2_1_assets(reports)
    legacy=json.loads((reports/"monitor_train16.json").read_text())["windows"]
    monitor8=json.loads((reports/"monitor_8pairs.json").read_text())["pairs"]
    monitor32=json.loads((reports/"monitor_32pairs.json").read_text())["pairs"]
    splits=build_v2_1_splits(manifest,legacy,monitor8,monitor32)
    data_manifest={"source_manifest":str(MANIFEST),"source_sha256":MANIFEST_SHA,
      "small_stage_scenes":splits["small_stage_scenes"],"same_scene_holdout16":splits["same_scene_holdout16"],
      "small_train_windows":splits["small_train_windows"],"train_probe16":splits["train_probe16"],
      "legacy_train16":splits["legacy_train16"],"dev8":splits["dev8"],"dev8_source":splits["dev8_source"],
      "fixed_val8":splits["val8"],"fixed_val32":splits["val32"],
      "expanded_train_windows":splits["expanded_train_windows"],"expanded_excluded_windows":splits["expanded_excluded_windows"],
      "holdout_selection":splits["holdout_selection"],"legacy_train16_exposure":splits["legacy_train16_exposure"],
      "window_exposure_contract":{"small_train_each_window":16,"expanded_train_each_window_if_stage_e":16,
        "small_train_identity_counts":[{"scene":w["scene"],"context":w["context"],"novel":w["novel"],"stage_s_exposures":16}
          for w in splits["small_train_windows"]],
        "expanded_train_identity_counts":[{"scene":w["scene"],"context":w["context"],"novel":w["novel"],"stage_e_exposures_if_started":16}
          for w in splits["expanded_train_windows"]]},
      "window_counts":{"small_train":len(splits["small_train_windows"]),"expanded_train":len(splits["expanded_train_windows"]),
                       "expanded_scenes":len({x["scene"] for x in splits["expanded_train_windows"]})},
      "scene_window_counts":{sc:sum(x["scene"]==sc for x in splits["expanded_train_windows"])
                             for sc in sorted({x["scene"] for x in splits["expanded_train_windows"]})}}
    plan={"stage_s_epochs":16,"stage_s_epoch_permutations":[np.random.default_rng(42+e).permutation(len(splits["small_train_windows"])).tolist() for e in range(16)],
      "stage_e_epochs":16,"stage_e_epoch_permutations":[np.random.default_rng(10042+e).permutation(len(splits["expanded_train_windows"])).tolist() for e in range(16)],
      "stage_s_updates":splits["stage_s_updates"],"stage_e_updates_if_gate_passes":splits["stage_e_updates"],
      "maximum_total_updates":splits["total_updates"],"plan_identity":"(scene, tuple(context), tuple(novel))"}
    data_path=reports/"data_manifest.json";plan_path=reports/"training_plan.json"
    for path,value in ((data_path,data_manifest),(plan_path,plan)):
        if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError(f"generated V2.1 asset conflict: {path}")
        write_json(path,value)
    commit=_stable_commit()
    run_manifest={"architecture":"LOCUSGS_OBJECT_LOCUS_V2_1","recipe":"OBJECT_LOCUS_V2_1_MASK_CONDITIONED_CLASSIFIER",
      "git_sha":commit,"gpu":gpu,"node":node,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
      "torch":torch.__version__,"cuda":torch.version.cuda,"tf32":{"cudnn":torch.backends.cudnn.allow_tf32,
      "matmul":torch.backends.cuda.matmul.allow_tf32},"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
      "pretrained_sha256":PRETRAINED_SHA,"manifest_sha256":MANIFEST_SHA,
      "monitor_sha256":MONITOR_SHA,"global_seed":42,"object_init_seed":31415,"gc_alpha":.01,
      "batch_size":1,"precision":"fp32","legacy_no_object_ce_used":False,
      "stage_s_epochs":16,"stage_s_windows":len(splits["small_train_windows"]),
      "stage_s_updates":splits["stage_s_updates"],"stage_e_epochs":16,"stage_e_windows":len(splits["expanded_train_windows"]),
      "stage_e_updates_if_gate_passes":splits["stage_e_updates"],"maximum_total_updates":splits["total_updates"],
      "source_workspace":"/space/mawb/ssst_object_locus_v2_1"}
    run_manifest["execution_source"]=source_info
    rm=reports/"run_manifest.json"
    if rm.exists():
        prior=json.loads(rm.read_text())
        for key in ("architecture","recipe","git_sha","pretrained_sha256","manifest_sha256","global_seed",
                    "object_init_seed","gc_alpha","batch_size","precision","stage_s_updates",
                    "stage_e_updates_if_gate_passes"):
            if prior.get(key)!=run_manifest.get(key):raise RuntimeError(f"run_manifest conflicts at {key}")
    else:write_json(rm,run_manifest)
    write_json(run/"run_manifest.json",json.loads(rm.read_text()))
    stage_s_reports=reports/"stage_s";stage_s_run=run/"stage_s"
    result=_stage_s(stage_s_reports,stage_s_run,splits,commit,data_path,plan_path)
    model,optimizer,opt,opt_audit,transfer,baseline,endpoint,gate,global_step=result
    write_json(reports/"stage_s_expansion_gate.json",gate)
    if endpoint is not None:write_json(reports/"stage_s_epoch16.json",endpoint)
    if gate["passed"]:
        global_step=_stage_e(reports,run,model,optimizer,opt,opt_audit,splits,commit,data_path,plan_path,
                             global_step,baseline,endpoint)
        stage_e_started=True
    else:stage_e_started=False
    final={"stage_s_updates":splits["stage_s_updates"],"stage_e_started":stage_e_started,
      "stage_e_updates":splits["stage_e_updates"] if stage_e_started else 0,
      "total_optimizer_updates":global_step,"gate":gate,
      "task_status":"training completed; expansion gate failed" if not stage_e_started else "training completed; evaluate task metrics independently"}
    write_json(reports/"final_status.json",final);write_json(run/"final_status.json",final)
    write_json(reports/"trainability_audit.json",trainability_counts(model))
    _write_result_bundle(reports)
    print(json.dumps({"final_status":final,"data_counts":data_manifest["window_counts"],
                      "result_bundle":str(reports/"result_bundle.zip")},indent=2,allow_nan=False),flush=True)


if __name__=="__main__":main()
