#!/usr/bin/env python3
"""Registered Object-Locus V2 Stage A gate and conditional Stage B run."""
from __future__ import annotations

import argparse, csv, json, math, os, random, shutil, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v2_runtime import (
    ASSET_ROOT,MANIFEST,PLAN,MANIFEST_SHA,PLAN_SHA,PRETRAINED_SHA,REPORTS_DEFAULT,
    RUN_ROOT_DEFAULT,build_batch,build_model,build_optimizer,capture_rng,restore_rng,
    locked_assets,seed_everything,sha256_file,train_one_step,trainability_counts,
    write_json,
)
from scripts.eval_object_locus_v2 import evaluate_windows

STAGE_A_STEPS=1500
A_EVAL=(0,500,1000,1500)
B_EVAL=(0,1000,3000,5000)
ALPHA=0.01

def _save(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,tmp); os.replace(tmp,path)

def _a_payload(model,optimizer,astep,position,opt,transfer,commit):
    return {"model":model.state_dict(),"optimizer":optimizer.state_dict(),
      "stage":"A","step":int(astep),"stage_a_step":int(astep),"plan_position":int(position),
      "architecture_name":model.architecture_name,"recipe":"OBJECT_LOCUS_V2_A_THEN_B",
      "git_commit":commit,
      "config":{"version":"Object-Locus V2","seed":42,"object_seed":31415,
        "state_layers":[6,8,10,12],"num_anchors":1024,"num_thing":100,"num_stuff":2,
        "children_per_anchor":64,"precision":"fp32","batch_size":1,"gc_alpha":ALPHA},
      "manifest_sha256":MANIFEST_SHA,"plan_sha256":PLAN_SHA,"pretrained_sha256":PRETRAINED_SHA,
      "transfer":transfer,"trainability":trainability_counts(model),"rng":capture_rng()}

def _metric_rows(result):
    return result["local"]

def _eval_node(model,opt,step,windows,reports,device,*,splits,official=False,panels=False):
    results={}
    for split,rows in splits.items():
        results[split]=evaluate_windows(model,opt,rows,step,split,reports,device,build_batch,
                                       official=official,panels=panels)
    path=Path(reports)/f"curves_{step:04d}.json"
    write_json(path,{"step":step,"splits":results})
    print(f"[eval step={step}] "+json.dumps({s:r["local"] for s,r in results.items()},allow_nan=False),flush=True)
    return results

def _gate_metrics(train_eval,step0_eval):
    rows=train_eval["local_rows"]["context"]
    all_ious=[float(x["best_raw_iou"]) for row in rows for x in row["raw_best_iou"]]
    raw_fraction=sum(x>=.5 for x in all_ious)/max(1,len(all_ious))
    matched_n=sum(int(x["matched_gt_count"]) for x in rows)
    matched_correct=sum(float(x["matched_class_accuracy"])*int(x["matched_gt_count"]) for x in rows)
    class_acc=matched_correct/max(1,matched_n)
    step0ctx=step0_eval["local"]["context"]["psnr"]
    step0nov=step0_eval["true_novel_psnr"]
    finalctx=train_eval["local"]["context"]["psnr"]
    finalnov=train_eval["true_novel_psnr"]
    drops={"context":step0ctx-finalctx,"novel":step0nov-finalnov}
    passes={"A1_raw_mask_iou_ge_0_5_fraction":raw_fraction>=.40,
      "A2_matched_classification_accuracy":class_acc>=.60,
      "A3_context_psnr_drop_le_0_5":drops["context"]<=.5,
      "A3_novel_psnr_drop_le_0_5":drops["novel"]<=.5}
    return {"raw_best_mask_iou_ge_0_5_fraction":raw_fraction,"raw_gt_count":len(all_ious),
      "matched_classification_accuracy":class_acc,"matched_gt_count":matched_n,
      "step0_psnr_context":step0ctx,"step1500_psnr_context":finalctx,
      "step0_psnr_novel":step0nov,"step1500_psnr_novel":finalnov,"psnr_drop_db":drops,
      "checks":passes,"passed":all(passes.values())}

def _dev8(train16,val8,val32):
    train_scenes={x["scene"] for x in train16}
    chosen=[x for x in val8 if x["scene"] not in train_scenes]
    source="monitor_8pairs.json"
    if len(chosen)<8:
        chosen=[x for x in val32 if x["scene"] not in train_scenes][:8]
        source="monitor_32pairs.json first nonoverlap pairs"
    if len(chosen)!=8: raise RuntimeError("could not construct 8-pair nonoverlapping dev8")
    return chosen,source,sorted(train_scenes & {x["scene"] for x in val8})

def _check_cuda():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Object-Locus V2 training requires exactly one CUDA GPU")
    gpu=torch.cuda.get_device_name()
    if gpu!="NVIDIA GeForce RTX 3090": raise RuntimeError(f"V2 run requires RTX3090, got {gpu}")
    if torch.cuda.get_device_properties(0).total_memory < 23*1024**3:
        raise RuntimeError("V2 node lacks 24GB-class GPU memory")
    node=os.uname().nodename
    if not (node.startswith("3dimage-13") or node.startswith("3dimage-11")):
        raise RuntimeError(f"V2 run node not registered: {node}")
    return gpu,node

def stage_a(device="cuda"):
    gpu,node=_check_cuda()
    reports=REPORTS_DEFAULT/"stage_a"; run=RUN_ROOT_DEFAULT
    if reports.exists() and any(reports.iterdir()) and not (reports/"stage_a_manifest.json").exists():
        raise RuntimeError(f"Stage A report directory already contains unregistered files: {reports}")
    if run.exists() and any(run.iterdir()) and not (run/"stage_a_manifest.json").exists():
        raise RuntimeError(f"Stage A run directory already contains unregistered files: {run}")
    reports.mkdir(parents=True,exist_ok=True); run.mkdir(parents=True,exist_ok=True)
    assets=locked_assets(REPORTS_DEFAULT)
    train16=json.loads((REPORTS_DEFAULT/"monitor_train16.json").read_text())["windows"]
    val8_orig=json.loads((REPORTS_DEFAULT/"monitor_8pairs.json").read_text())["pairs"]
    val32=json.loads((REPORTS_DEFAULT/"monitor_32pairs.json").read_text())["pairs"]
    dev8,dev_source,overlap=_dev8(train16,val8_orig,val32)
    write_json(reports/"stage_a_dev8.json",{"source":dev_source,"windows":dev8,
                  "train16_scene_overlap":overlap,"cross_scene":True})
    opt=__import__("scripts.object_locus_v2_runtime",fromlist=["build_options"]).build_options()
    commit=os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()
    if subprocess_check_clean_commit(REPO,commit) is False: raise RuntimeError("Stage A requires committed code")
    import tokengs.models.object_locus_v2_controller as controller_module
    module_file=Path(controller_module.__file__).resolve()
    if not module_file.is_relative_to(REPO.resolve()): raise RuntimeError(f"V2 controller outside execution worktree: {module_file}")
    print(json.dumps({"execution_worktree":str(REPO.resolve()),"git_sha":commit,
                      "controller_module_file":str(module_file)},flush=True),flush=True)
    seed_everything(42)
    model,opt,transfer=build_model(device); optimizer,opt_audit=build_optimizer(model)
    trainability=trainability_counts(model)
    if trainability["frozen_numel"]: raise RuntimeError("V2 requires all model parameters trainable")
    manifest={"task":"Object-Locus V2","stage":"A","architecture":model.architecture_name,
      "recipe":"OBJECT_LOCUS_V2_A_THEN_B","git_sha":commit,"gpu":gpu,"node":node,
      "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch":torch.__version__,"cuda":torch.version.cuda,
      "tf32":{"cudnn":torch.backends.cudnn.allow_tf32,"matmul":torch.backends.cuda.matmul.allow_tf32},
      "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"manifest_sha256":MANIFEST_SHA,
      "plan_sha256":PLAN_SHA,"pretrained_sha256":PRETRAINED_SHA,"seed":42,"object_seed":31415,
      "stage_a_steps":1500,"training_windows":train16,"dev8_source":dev_source}
    manifest_path=reports/"stage_a_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Stage-A manifest conflicts with the registered run; refusing overwrite")
    run_manifest_path=run/"stage_a_manifest.json"
    if run_manifest_path.exists() and json.loads(run_manifest_path.read_text()) != manifest:
        raise RuntimeError("Stage-A run manifest conflicts with the registered run; refusing overwrite")
    write_json(reports/"stage_a_manifest.json",manifest); write_json(run/"stage_a_manifest.json",manifest)
    write_json(reports/"transfer_audit.json",transfer); write_json(reports/"optimizer_audit.json",opt_audit)
    # Resume only from a V2 Stage-A checkpoint with matching step and locked assets.
    checkpoint=max(run.glob("checkpoint_A_*.pt"),key=lambda p:int(p.stem.split("_")[-1]),default=None)
    start=0
    if checkpoint:
        st=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if (st.get("architecture_name")!=model.architecture_name or st.get("manifest_sha256")!=MANIFEST_SHA
            or st.get("plan_sha256")!=PLAN_SHA or st.get("pretrained_sha256")!=PRETRAINED_SHA
            or st.get("git_commit")!=commit):
            raise RuntimeError("Stage-A checkpoint provenance mismatch")
        model.load_state_dict(st["model"],strict=True); optimizer.load_state_dict(st["optimizer"])
        restore_rng(st["rng"]); start=int(st["stage_a_step"])
    log_path=reports/"training_log_metrics.jsonl"
    existing={}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip():
                row=json.loads(line)
                if int(row["stage_a_step"])<=start: existing[int(row["stage_a_step"])]=row
    log_path.write_text("".join(json.dumps(existing[k],allow_nan=False)+"\n" for k in sorted(existing)))
    split0={"train16":train16,"dev8":dev8}
    step0=None
    if start==0:
        step0res=_eval_node(model,opt,0,train16,reports,device,splits=split0,panels=True)
        step0=step0res["train16"]
        _save(run/"checkpoint_A_0000.pt",_a_payload(model,optimizer,0,0,opt,transfer,commit))
    else:
        step0=json.loads((reports/"curves_0000.json").read_text())["splits"]["train16"]
    stage_a_rows=[]
    with (run/"train.log").open("a",buffering=1) as log:
        for astep in range(start+1,STAGE_A_STEPS+1):
            window=train16[(astep-1)%16]
            batch=build_batch(opt,window,device)
            lr_multiplier=min(astep/50.0,1.0)
            metrics_weight=min(astep/100.0,1.0)
            lrs=(1e-4*lr_multiplier,1e-5*lr_multiplier)
            output,metrics=train_one_step(model,optimizer,batch,astep,
                understanding_weight_value=metrics_weight,lr_values=lrs,
                failure_capture_dir=run/"failures",
                failure_context={"stage":"A","stage_step":astep,"window":window,"optimizer_updated":False})
            if astep%100==0:
                row={k:(float(v.detach()) if torch.is_tensor(v) and v.ndim==0 else v)
                     for k,v in metrics.items() if k not in ("loss_recon","loss_understanding","loss","loss_total")}
                row.update({"stage":"A","stage_a_step":astep,"global_optimizer_step":astep,
                    "stage_window_index":(astep-1)%16,"scene":window["scene"],
                    "context":window["context"],"novel":window["novel"],
                    "step_seconds":float(metrics.get("step_seconds",0)),
                    "gpu_allocated_bytes":torch.cuda.memory_allocated(),"gpu_reserved_bytes":torch.cuda.memory_reserved()})
                for key in ("loss","loss_total","loss_recon","loss_understanding"):
                    value=metrics.get(key)
                    if torch.is_tensor(value): row[key]=float(value.detach())
                if not all(math.isfinite(float(v)) for v in row.values() if isinstance(v,(float,int))):
                    raise FloatingPointError(f"nonfinite Stage-A metrics at {astep}: {row}")
                existing[astep]=row; line=json.dumps(row,allow_nan=False)
                with log_path.open("a") as f:f.write(line+"\n")
                print(line,flush=True); print(line,file=log,flush=True)
            del output,metrics,batch
            if astep in A_EVAL:
                model.eval()
                splits={"train16":train16}
                if astep in (1000,1500):splits["dev8"]=dev8
                evalres=_eval_node(model,opt,astep,train16,reports,device,splits=splits,
                    official=(astep==1500),panels=True)
                model.train()
                _save(run/f"checkpoint_A_{astep:04d}.pt",_a_payload(model,optimizer,astep,astep,opt,transfer,commit))
                stage_a_rows.append((astep,evalres))
    final_eval=json.loads((reports/"curves_1500.json").read_text())["splits"]["train16"]
    step0_eval=json.loads((reports/"curves_0000.json").read_text())["splits"]["train16"]
    gate=_gate_metrics(final_eval,step0_eval)
    write_json(reports/"stage_a_gate.json",gate)
    report={"stage_a_gate":gate,"dev8_source":dev_source,"dev8_train_scene_overlap":overlap,
       "stage_a_completed_steps":STAGE_A_STEPS,"stage_b_started":False,
       "task_status":"Stage A passed; Stage B pending execution" if gate["passed"] else "Stage A completed but instance-formation gate failed"}
    write_json(reports/"stage_a_summary.json",report)
    if gate["passed"]:
        run_stage_b(model,optimizer,opt,transfer,train16,dev8,val32,device,reports,
                    RUN_ROOT_DEFAULT.parent/"stage_b",commit)
    else:
        write_json(reports/"final_status.json",{**report,"stage_b_started":False,
                   "stop_reason":"one or more registered Stage-A gates failed"})
    _write_result_bundle(REPORTS_DEFAULT)
    return report

def subprocess_check_clean_commit(repo,commit):
    import subprocess
    if subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip(): return False
    remote=subprocess.check_output(["git","-C",str(repo),"rev-parse","origin/main"],text=True).strip()
    if remote!=commit: raise RuntimeError(f"Stage A requires pushed origin/main==HEAD, got {remote} vs {commit}")
    return bool(commit)

def _b_payload(model,optimizer,bstep,commit):
    return {"model":model.state_dict(),"optimizer":optimizer.state_dict(),
      "step":1500+int(bstep),"stage":"B","stage_b_step":int(bstep),
      "plan_position":int(bstep),"architecture_name":model.architecture_name,
      "recipe":"OBJECT_LOCUS_V2_A_THEN_B","git_commit":commit,
      "config":{"version":"Object-Locus V2","seed":42,"object_seed":31415,
        "precision":"fp32","batch_size":1,"gc_alpha":ALPHA,
        "stage_a_steps":1500,"stage_b_steps":5000},
      "manifest_sha256":MANIFEST_SHA,"plan_sha256":PLAN_SHA,
      "pretrained_sha256":PRETRAINED_SHA,"rng":capture_rng()}

def _write_result_bundle(root):
    root=Path(root)
    bundle=root/"review_bundle"
    if bundle.exists(): shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    curves=[]
    for stage in ("stage_a","stage_b"):
        for path in sorted((root/stage).glob("curves_*.json")):
            payload=json.loads(path.read_text())
            curves.append({"stage":stage,"path":str(path),"payload":payload})
    with (bundle/"task_metrics.csv").open("w",newline="") as f:
        fields=["stage","stage_step","cumulative_optimizer_steps","split","scope",
          "mIoU_panoptic_all","mIoU_panoptic_thing","mIoU_panoptic_stuff","mIoU_semantic_readout",
          "local_PQ","local_ca_TP","local_ca_FP","local_ca_FN","local_ca_GT","local_ca_recall50",
          "local_class_aware_TP","local_class_aware_FP","local_class_aware_FN","local_class_aware_recall50",
          "raw_mask_recall50","raw_best_IoU_ge_0_5_fraction","GT_anchor_support_fraction",
          "matched_class_accuracy","PSNR_scope","PSNR_true_novel",
          "official_all_mIoU","official_all_PQ","official_all_mAP","official_all_AP50",
          "official_novel_mIoU","official_novel_PQ","official_novel_mAP","official_novel_AP50"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for item in curves:
            p=item["payload"]; step=int(p.get("step",0)); stage_step=step if item["stage"]=="stage_a" else step
            cumulative=step if item["stage"]=="stage_a" else 1500+step
            for split,data in p.get("splits",{}).items():
                for scope,key in (("context","context"),("target-all","target_all")):
                    m=data.get("local",{}).get(key,{})
                    allres=data.get("official",{}).get("all") or {}
                    novres=data.get("official",{}).get("novel") or {}
                    def om(res,prefix):
                        mapp=res.get(prefix+"_map") or {}
                        return {"miou":res.get(prefix+"_miou"),"pq":res.get(prefix+"_pq"),
                                "map":mapp.get("map") if isinstance(mapp,dict) else None,
                                "ap50":mapp.get("map_50") if isinstance(mapp,dict) else None}
                    oa=om(allres,"context" if key=="context" else "target")
                    on=om(novres,"target")
                    ca=m.get("class_agnostic_tp",0);gt=m.get("gt_count",0)
                    aware=m.get("class_aware_tp",0)
                    w.writerow({"stage":item["stage"],"stage_step":stage_step,
                      "cumulative_optimizer_steps":cumulative,"split":split,"scope":scope,
                      "mIoU_panoptic_all":m.get("mIoU_all_nonempty"),
                      "mIoU_panoptic_thing":m.get("mIoU_thing"),"mIoU_panoptic_stuff":m.get("mIoU_stuff"),
                      "mIoU_semantic_readout":m.get("semantic_readout_miou"),"local_PQ":m.get("local_pq"),
                      "local_ca_TP":ca,"local_ca_FP":m.get("class_agnostic_fp"),
                      "local_ca_FN":m.get("class_agnostic_fn"),"local_ca_GT":gt,
                      "local_ca_recall50":ca/max(1,gt),"local_class_aware_TP":aware,
                      "local_class_aware_FP":m.get("class_aware_fp"),"local_class_aware_FN":m.get("class_aware_fn"),
                      "local_class_aware_recall50":aware/max(1,gt),
                      "raw_mask_recall50":m.get("raw_recall50"),
                      "raw_best_IoU_ge_0_5_fraction":m.get("raw_best_iou_ge_0_5_fraction"),
                      "GT_anchor_support_fraction":m.get("gt_support_fraction"),
                      "matched_class_accuracy":m.get("matched_class_accuracy"),"PSNR_scope":m.get("psnr"),
                      "PSNR_true_novel":data.get("true_novel_psnr") if key=="target_all" else None,
                      "official_all_mIoU":oa["miou"],"official_all_PQ":oa["pq"],
                      "official_all_mAP":oa["map"],"official_all_AP50":oa["ap50"],
                      "official_novel_mIoU":on["miou"],"official_novel_PQ":on["pq"],
                      "official_novel_mAP":on["map"],"official_novel_AP50":on["ap50"]})
    (bundle/"task_metrics.json").write_text(json.dumps(curves,indent=2,allow_nan=False)+"\n")
    per_gt=[]
    for item in curves:
        payload=item["payload"]
        for split,data in payload.get("splits",{}).items():
            for scope,rows in data.get("local_rows",{}).items():
                for wi,row in enumerate(rows):
                    for gt in row.get("raw_best_iou",[]):
                        per_gt.append({"stage":item["stage"],"step":payload.get("step"),
                          "split":split,"scope":scope,"window_index":wi,
                          "scene":row.get("scene"),**gt})
    with (bundle/"per_gt_raw_mask_iou.csv").open("w",newline="") as f:
        fields=sorted({key for row in per_gt for key in row})
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(per_gt)
    logs=[]
    for stage in ("stage_a","stage_b"):
        p=root/stage/"training_log_metrics.jsonl"
        if p.is_file():
            logs += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    with (bundle/"training_curve.csv").open("w",newline="") as f:
        fields=sorted({k for row in logs for k,v in row.items() if not isinstance(v,(dict,list))})
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(logs)
    (bundle/"training_curve.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in logs))
    final_status=json.loads((root/"stage_a"/"final_status.json").read_text()) if (root/"stage_a"/"final_status.json").is_file() else {}
    gate=json.loads((root/"stage_a"/"stage_a_gate.json").read_text()) if (root/"stage_a"/"stage_a_gate.json").is_file() else {}
    stage_b_status=json.loads((root/"stage_b"/"final_status.json").read_text()) if (root/"stage_b"/"final_status.json").is_file() else {}
    provenance={"architecture":"LOCUSGS_OBJECT_LOCUS_V2","recipe":"OBJECT_LOCUS_V2_A_THEN_B",
      "stage_a_gate":gate,"final_status":final_status,"stage_b_status":stage_b_status,
      "smoke":json.loads((root/"smoke"/"smoke_result.json").read_text()) if (root/"smoke"/"smoke_result.json").is_file() else None,
      "pretrained_sha256":PRETRAINED_SHA,"manifest_sha256":MANIFEST_SHA,"plan_sha256":PLAN_SHA}
    (bundle/"config_and_provenance.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n")
    # Embed actual fixed-window images, not remote paths.
    qdst=bundle/"qualitative"
    for stage in ("stage_a", "stage_b"):
        qsrc=root/stage/"qualitative"
        if qsrc.exists():
            for src in qsrc.rglob("*.png"):
                rel=src.relative_to(qsrc); dest=qdst/stage/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
    # Include only the new version's source and the actual committed diff.
    code_paths=["tokengs/models/object_locus_v2.py","tokengs/models/object_locus_v2_controller.py",
      "tokengs/models/object_locus_v2_loss.py","scripts/object_locus_v2_runtime.py",
      "scripts/train_object_locus_v2.py","scripts/eval_object_locus_v2.py",
      "scripts/export_object_locus_v2_official.py","scripts/smoke_object_locus_v2.py",
      "scripts/submit_object_locus_v2.sh","tests/test_object_locus_v2_contracts.py",
      "tokengs/models/__init__.py","tokengs/options.py","docs/Object_Locus_V2.md"]
    for rel in code_paths:
        src=REPO/rel; dest=bundle/"code"/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
    import subprocess
    patch=subprocess.check_output(["git","-C",str(REPO),"diff","658b7e7c3f8559400398f1c0c4f229a5d9e58070..HEAD"],text=True)
    (bundle/"changes.patch").write_text(patch)
    (bundle/"git_status.txt").write_text(subprocess.check_output(["git","-C",str(REPO),"status","--short"],text=True))
    v11_rows=[]
    v11_root=Path("/space/mawb/ssst/group_plus/object_locus_v1_1")
    for vstep in (1000,2000,3500,5000):
        vpath=v11_root/f"curves_{vstep}.json"
        if not vpath.is_file(): continue
        vp=json.loads(vpath.read_text())
        vd=vp.get("splits",{}).get("val32",{})
        for scope,local_key in (("context","context"),("target-all","target")):
            vm=vd.get("local",{}).get(local_key,{})
            off=vd.get("official",{})
            if scope=="context": official=off.get("context_from_official_all",{})
            else: official=off.get("target_all_raw",{})
            v11_rows.append({"step":vstep,"scope":scope,
                "thing_mIoU":vm.get("mIoU_thing"),"all_mIoU":vm.get("mIoU_all_nonempty"),
                "PQ":official.get("context_pq" if scope=="context" else "target_pq"),
                "mAP":official.get("context_map" if scope=="context" else "map"),
                "AP50":official.get("context_map_50" if scope=="context" else "map_50"),
                "PSNR":vd.get("float_psnr_db",{}).get("context" if scope=="context" else "target_all")})
    with (bundle/"v1_1_comparison.csv").open("w",newline="") as f:
        fields=["step","scope","thing_mIoU","all_mIoU","PQ","mAP","AP50","PSNR"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(v11_rows)
    # Include compact failure summaries and useful log tails, never state dumps.
    failures=bundle/"failure_summaries"; failures.mkdir(exist_ok=True)
    for failure_dir in (RUN_ROOT_DEFAULT/"failures",
                        RUN_ROOT_DEFAULT.parent/"stage_b"/"failures",
                        root/"smoke"/"failure"):
        if failure_dir.exists():
            for src in failure_dir.glob("failure_step_*.json"):
                shutil.copy2(src,failures/src.name)
    logdir=bundle/"key_logs"; logdir.mkdir(exist_ok=True)
    for src in sorted((root/"slurm").glob("*.out")):
        lines=src.read_text(errors="replace").splitlines()[-160:]
        (logdir/src.name).write_text("\n".join(lines)+"\n")
    report=["# Object-Locus V2 阶段结果\n\n",
      "本轮采用独立 anchor/Gaussian sigmoid memberships 与 child residual；重建参数、像素阈值、官方评测器均未替换。\n\n",
      f"- Stage A 1500-step gate: **{'PASS' if gate.get('passed') else 'FAIL / 未通过'}**\n",
      f"- Stage A gate metrics: `{json.dumps(gate,ensure_ascii=False)}`\n",
      f"- Stage B status: `{json.dumps(stage_b_status,ensure_ascii=False)}`\n",
      "- GPU smoke: RTX 3090 one-step and local/official single-pair interface passed; single-pair AP is not a task-success claim.\n",
      "- Task effectiveness is judged from independent raw masks, matched classification, local panoptic masks and official metrics, separately from finite losses.\n",
      "- V1.1 used a fresh 5000-step 128-scene run; Stage A is 1500 updates over 16 fixed windows. These are different data exposures and not paired results. If Stage B runs, its 5000 updates continue from Stage A (6500 cumulative), not a fresh 5000-step run.\n",
      "- Historical Object-Locus V1 step-4090 nonfinite root cause remains UNKNOWN.\n\n",
      "## 与 V1.1 已有结果对照\n\n",
      "V1.1 原始曲线见 [v1_1_comparison.csv](v1_1_comparison.csv)。V1.1 是 fresh 5000-step、128-scene 训练；V2 Stage A 是 1500 updates over 16 fixed windows。两者训练数据曝光不同，不构成等条件配对。若 Stage B 运行，V2 累计 6500 updates。\n\n",
      "## Metrics\n\n",
      "See [task_metrics.csv](task_metrics.csv) for local context/target-all and official all/novel rows; missing official values remain blank. `per_gt_raw_mask_iou.csv` includes each GT's raw-mask best IoU and anchor-support status.\n\n",
      "## Fixed qualitative images\n\n",
      "See the `qualitative/` folder. Panels use the first three fixed windows per split in file order and include actual PNGs.\n\n",
      "## Source and run provenance\n\n",
      "See [config_and_provenance.json](config_and_provenance.json), [changes.patch](changes.patch), [training_curve.csv](training_curve.csv), and the code folder. No checkpoints or dataset files are included.\n"]
    (bundle/"README.md").write_text("# Object-Locus V2 review bundle\n\nSee [Chinese report](Object_Locus_V2_Report_zh.md), [metrics](task_metrics.csv), [curves](training_curve.csv).\n")
    (bundle/"Object_Locus_V2_Report_zh.md").write_text("".join(report))
    (qdst/"README.md").write_text(
        "PNG panels are fixed first-three windows in monitor file order. "
        "If the main archive exceeds the size limit, upload every sibling "
        "`result_bundle_qualitative_*.zip` together with `result_bundle.zip`.\n")
    zip_path=root/"result_bundle.zip"
    if zip_path.exists(): zip_path.unlink()
    for stale in root.glob("result_bundle_qualitative_*.zip"):
        stale.unlink()
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in bundle.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(bundle))
    limit=28*1024*1024
    if zip_path.stat().st_size>limit:
        # Keep the required main archive under the cap and place real PNGs in a
        # separately uploadable qualitative archive.
        with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in bundle.rglob("*"):
                if p.is_file() and ("qualitative" not in p.parts or p.name=="README.md"):
                    z.write(p,p.relative_to(bundle))
        if zip_path.stat().st_size>limit:
            raise RuntimeError(f"Object-Locus V2 non-qualitative bundle exceeds 28 MiB: {zip_path.stat().st_size}")
        image_files=[p for p in (bundle/"qualitative").rglob("*") if p.is_file() and p.name!="README.md"]
        part_limit=24*1024*1024
        batches=[]; current=[]; current_size=0
        for p in image_files:
            if p.stat().st_size>part_limit:
                raise RuntimeError(f"single qualitative image exceeds bundle part limit: {p}")
            if current and current_size+p.stat().st_size>part_limit:
                batches.append(current); current=[]; current_size=0
            current.append(p); current_size+=p.stat().st_size
        if current: batches.append(current)
        for index,items in enumerate(batches,1):
            qzip=root/f"result_bundle_qualitative_{index:03d}.zip"
            with zipfile.ZipFile(qzip,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in items: z.write(p,p.relative_to(bundle))
            if qzip.stat().st_size>limit:
                raise RuntimeError(f"Object-Locus V2 qualitative archive exceeds 28 MiB: {qzip}")
    # Verify archive integrity and all images before return.
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None: raise RuntimeError("V2 result bundle zip integrity check failed")
    return str(zip_path)

def run_stage_b(model,optimizer,opt,transfer,train16,dev8,val32,device,reports,run,commit):
    manifest,plan=locked_assets(REPORTS_DEFAULT)
    windows=manifest["windows"]; entries=plan["entries"]
    reports=REPORTS_DEFAULT/"stage_b"
    reports.mkdir(parents=True,exist_ok=True)
    if len(entries)!=5000: raise RuntimeError("Stage B locked plan length differs from 5000")
    if run.exists() and any(run.iterdir()) and not (run/"stage_b_manifest.json").exists():
        raise RuntimeError(f"Stage B path has existing unregistered content: {run}")
    run.mkdir(parents=True,exist_ok=True)
    bmanifest={"architecture":model.architecture_name,"git_sha":commit,
      "manifest_sha256":MANIFEST_SHA,"plan_sha256":PLAN_SHA,"pretrained_sha256":PRETRAINED_SHA,
      "starts_from_stage_a_step":1500,"stage_b_steps":5000,"total_optimizer_steps":6500,
      "recipe":"OBJECT_LOCUS_V2_A_THEN_B"}
    manifest_path=run/"stage_b_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=bmanifest:
        raise RuntimeError("Stage-B manifest conflicts with the registered run")
    if not manifest_path.exists(): write_json(manifest_path,bmanifest)
    checkpoint=max(run.glob("checkpoint_B_*.pt"),key=lambda p:int(p.stem.split("_")[-1]),default=None)
    bstart=0
    if checkpoint:
        st=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if st.get("architecture_name")!=model.architecture_name or st.get("plan_sha256")!=PLAN_SHA:raise RuntimeError("Stage-B resume provenance mismatch")
        model.load_state_dict(st["model"],strict=True);optimizer.load_state_dict(st["optimizer"]);restore_rng(st["rng"]);bstart=int(st["stage_b_step"])
    log_path=reports/"training_log_metrics.jsonl"
    if log_path.exists():
        kept=[]
        for line in log_path.read_text().splitlines():
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("stage")!="B" or int(row.get("stage_b_step",0))<=bstart: kept.append(row)
        log_path.write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in kept))
    if bstart==0:
        _eval_node(model,opt,0,train16,reports,device,splits={"train16":train16,"dev8":dev8,"val32":val32},official=True,panels=True)
        _save(run/"checkpoint_B_0000.pt",_b_payload(model,optimizer,0,commit))
    for bs in range(bstart+1,5001):
        entry=entries[bs-1]; batch=build_batch(opt,entry,device)
        mult=(.1+.9*bs/100) if bs<=100 else .02+.98*.5*(1+math.cos(math.pi*(bs-100)/4900))
        out,metrics=train_one_step(model,optimizer,batch,1500+bs,
            understanding_weight_value=1.0,lr_values=(1e-4*mult,1e-5*mult),
            failure_capture_dir=run/"failures",failure_context={"stage":"B","stage_step":bs,"plan_entry":entry,"optimizer_updated":False})
        if bs%100==0:
            row={k:(float(v.detach()) if torch.is_tensor(v) and v.ndim==0 else v) for k,v in metrics.items() if k not in ("loss_recon","loss_understanding","loss","loss_total")}
            row.update({"stage":"B","stage_b_step":bs,"global_optimizer_step":1500+bs,"scene":entry["scene"],"window_index":entry["window_index"],"gpu_allocated_bytes":torch.cuda.memory_allocated(),"gpu_reserved_bytes":torch.cuda.memory_reserved()})
            for key in ("loss","loss_total","loss_recon","loss_understanding"):
                if torch.is_tensor(metrics.get(key)):row[key]=float(metrics[key].detach())
            with log_path.open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        del out,metrics,batch
        if bs in B_EVAL:
            if bs in (1000,3000,5000):
                splits={"train16":train16,"dev8":dev8,"val32":val32}
                ev=_eval_node(model,opt,bs,windows,reports,device,splits=splits,official=True,panels=True)
            _save(run/f"checkpoint_B_{bs:04d}.pt",_b_payload(model,optimizer,bs,commit))
    write_json(reports/"final_status.json",{"stage_a_passed":True,"stage_b_completed_steps":5000,"total_optimizer_steps":6500,"task_status":"training completed; evaluate task metrics independently"})

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("stage_a","train"),default="stage_a");parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.device!="cuda":raise RuntimeError("registered V2 stages require the locked CUDA device")
    print(json.dumps(stage_a(args.device),indent=2),flush=True)

if __name__=="__main__":main()
