#!/usr/bin/env python3
"""Single RTX3090 one-step + one-window evaluator/export smoke for V2."""
from __future__ import annotations

import json, os, sys, time, subprocess, inspect
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v2_runtime import (
    build_model,build_optimizer,build_batch,PLAN,REPORTS_DEFAULT,write_json,
    train_one_step,sha256_file,PRETRAINED_SHA,GC_ALPHA,locked_assets,
)
import tokengs.models.object_locus_v2_controller as v2_controller

def main():
    source_sha=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    module_path=Path(inspect.getfile(v2_controller)).resolve()
    if not module_path.is_relative_to(REPO.resolve()):
        raise RuntimeError(f"V2 controller imported outside execution worktree: {module_path}")
    print(json.dumps({"execution_worktree":str(REPO.resolve()),"git_sha":source_sha,
                      "controller_module_file":str(module_path)}),flush=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!="NVIDIA GeForce RTX 3090":
        raise RuntimeError("V2 smoke requires the registered RTX3090")
    if torch.cuda.get_device_properties(0).total_memory < 23*1024**3:
        raise RuntimeError("V2 smoke GPU does not provide the registered 24GB class")
    plan=json.loads(PLAN.read_text()); entry=plan["entries"][999]
    if (entry["scene"],int(entry["window_index"])) != ("scene0016_00",241):
        raise RuntimeError(f"locked step1000 batch mismatch: {entry}")
    if [int(x) for x in entry["context"]] != [1506,1517] or [int(x) for x in entry["novel"]] != [1509,1516]:
        raise RuntimeError(f"locked step1000 frame mismatch: {entry}")
    outdir=REPORTS_DEFAULT/"smoke"
    outdir.mkdir(parents=True,exist_ok=True)
    locked_assets(REPORTS_DEFAULT)
    model,opt,transfer=build_model("cuda"); model.train()
    optimizer,optinfo=build_optimizer(model)
    batch=build_batch(opt,entry,"cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0=time.time()
    output,metrics=train_one_step(model,optimizer,batch,1000,understanding_weight_value=1.0,
                                  lr_values=(1e-4,1e-5),failure_capture_dir=outdir/"failure",
                                  failure_context={"stage":"smoke","entry":entry})
    scalar={k:(float(v.detach()) if torch.is_tensor(v) else float(v))
            for k,v in metrics.items() if torch.is_tensor(v) or isinstance(v,(int,float))}
    finite=all(torch.isfinite(torch.tensor(v)) for v in scalar.values())
    gradients={"object":0,"shared":0,"finite":True}
    for name,p in model.named_parameters():
        if p.grad is not None:
            gradients["object" if name.startswith("object_locus_v2.") else "shared"]+=1
            gradients["finite"] &= bool(torch.isfinite(p.grad).all())
    params_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
    state_finite=all(bool(torch.isfinite(v).all()) for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v) and v.is_floating_point())
    result={"status":"PASS" if finite and gradients["finite"] and params_finite and state_finite else "FAIL",
      "job_node":os.uname().nodename,"gpu":torch.cuda.get_device_name(),
      "git_sha":source_sha,
      "controller_module_file":inspect.getfile(v2_controller),
      "total_memory_gib":torch.cuda.get_device_properties(0).total_memory/1024**3,
      "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch":torch.__version__,"cuda":torch.version.cuda,
      "step":1000,"entry":entry,"pretrained_sha256":PRETRAINED_SHA,
      "transfer":transfer,"optimizer":optinfo,"metrics":scalar,"gradient_counts":gradients,
      "parameters_finite":params_finite,"optimizer_state_finite":state_finite,
      "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
      "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"step_seconds":time.time()-t0}
    del output,metrics,batch
    model.eval()
    from scripts.eval_object_locus_v2 import evaluate_windows
    val32=json.loads((REPORTS_DEFAULT/"monitor_32pairs.json").read_text())["pairs"]
    result["minimal_eval_export"]=evaluate_windows(model,opt,val32[:1],1000,"val32_smoke",outdir,"cuda",build_batch,official=True,panels=True)
    write_json(outdir/"smoke_result.json",result)
    print(json.dumps(result,indent=2,allow_nan=False))
    if result["status"]!="PASS": raise RuntimeError("Object-Locus V2 smoke finite gate failed")

if __name__=="__main__": main()
