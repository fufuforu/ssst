"""One temporary continuation update plus local/official evaluator smoke on RTX3090."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from scripts.object_locus_v3_set_runtime import build_model, build_optimizer, build_batch, train_one_step, write_json
from scripts.train_object_locus_v3_set_expanded import (
    SOURCE_CKPT, SOURCE_CKPT_SHA, REPORTS, RUN, build_manifest_and_plan, expanded_lr,
    _gpu_assert, _source_optimizer_exact, science_module_hashes, SCIENCE_SHA256,
)
from scripts.object_locus_v3_set_runtime import sha256_file, restore_rng, capture_rng
from scripts.train_object_locus_v3_set_expanded import equal_tree
from scripts.eval_object_locus_v3_set import evaluate_windows


def _finite(value, prefix="root"):
    bad=[]
    if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all(): bad.append(prefix)
    elif isinstance(value, dict):
        for k,v in value.items(): bad.extend(_finite(v, f"{prefix}.{k}"))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value): bad.extend(_finite(v, f"{prefix}[{i}]"))
    return bad


def main():
    gpu = _gpu_assert()
    if science_module_hashes()!=SCIENCE_SHA256: raise RuntimeError("protected V3-Set scientific modules differ")
    if sha256_file(SOURCE_CKPT) != SOURCE_CKPT_SHA: raise RuntimeError("source checkpoint hash mismatch")
    audit_path=REPORTS/"smoke"/"expanded_smoke_audit_precommit.json"
    if audit_path.exists(): raise RuntimeError(f"refusing to overwrite prior smoke {audit_path}")
    manifest, _ = build_manifest_and_plan()
    window = manifest["expanded_train_windows"][0]
    source = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    model,opt,_ = build_model("cuda")
    optimizer,optimizer_audit=build_optimizer(model)
    model.load_state_dict(source["model"],strict=True)
    mismatch=[k for k,v in source["model"].items() if not torch.equal(model.state_dict()[k].detach().cpu(),v.detach().cpu())]
    if mismatch: raise RuntimeError(f"smoke model does not exactly restore source checkpoint: {mismatch[:5]}")
    optimizer.load_state_dict(source["optimizer"])
    optimizer_restore=_source_optimizer_exact(optimizer,source["optimizer"])
    restore_rng(source["rng"])
    rng_exact=equal_tree(capture_rng(),source["rng"])
    if not rng_exact: raise RuntimeError("smoke failed to restore source RNG exactly")
    batch=build_batch(opt,window,"cuda")
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    out,metrics=train_one_step(model,optimizer,batch,1,understanding_weight_value=1.0,
        lr_values=expanded_lr(1),failure_capture_dir=RUN/"smoke_failures",
        failure_context={"smoke":True,"window":window,"source_checkpoint_sha256":SOURCE_CKPT_SHA})
    finite_bad=_finite({"prediction":out.get("prediction"),"metrics":metrics})
    if finite_bad: raise RuntimeError(f"smoke nonfinite outputs: {finite_bad[:12]}")
    grads=metrics.get("gradient_report_before_clip",{})
    watched=("object_locus_v3_set.class_head.weight","object_locus_v3_set.cls_fuse.weight",
             "object_locus_v3_set.mask_q_mlp.2.weight","object_locus_v3_set.child_mlp.2.weight",
             "object_locus_v3_set.W_Q.weight","anchor_decoder.mu","activation_head.deconv.weight")
    for name in watched:
        g=grads.get(name)
        if not g or not g.get("present") or not g.get("finite") or g.get("grad_norm",0.)<=0:
            raise RuntimeError(f"required finite nonzero smoke gradient missing: {name}: {g}")
    if any(not torch.isfinite(p).all() for p in model.parameters()): raise RuntimeError("smoke model parameters became nonfinite")
    if any(torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()
           for state in optimizer.state.values() for v in state.values()): raise RuntimeError("smoke optimizer state became nonfinite")
    eval_dir=REPORTS/"smoke"/"eval_precommit"
    result,pergt,queries=evaluate_windows(model,opt,[window],1,"expanded_first_window",eval_dir,
        "cuda",build_batch,official=True,panels=True)
    if len(result["windows"])!=1: raise RuntimeError("single-window evaluator output malformed")
    scopes=result["local"]
    for scope in ("context","target_all","novel"):
        if scope not in scopes: raise RuntimeError(f"missing local scope {scope}")
        if not torch.isfinite(torch.tensor(float(scopes[scope]["psnr"]))): raise RuntimeError("nonfinite evaluator metric")
    torch.cuda.synchronize()
    report={"status":"PASS","gpu":gpu,"node":os.uname().nodename,"window":window,
        "source_checkpoint_sha256":SOURCE_CKPT_SHA,"source_global_step":3584,
        "model_tensor_count":len(source["model"]),"model_state_exact":True,
        "optimizer_restore":optimizer_restore,"optimizer_groups":optimizer_audit,"rng_restored_exact":rng_exact,
        "protected_science_module_sha256":science_module_hashes(),
        "lr_first_step":{"object":expanded_lr(1)[0],"reconstruction":expanded_lr(1)[1]},
        "losses":{k:(float(v.detach()) if torch.is_tensor(v) and v.ndim==0 else v)
                  for k,v in metrics.items() if torch.is_tensor(v) and v.ndim==0 or isinstance(v,(float,int))},
        "gradients":{k:grads[k] for k in watched},"local_eval":scopes,
        "official_evaluator":"single-window all and novel export/evaluator completed; undefined metrics retained",
        "per_gt_rows":len(pergt),"candidate_query_rows":len(queries),
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
        "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3}
    audit_path.parent.mkdir(parents=True,exist_ok=True);write_json(audit_path,report)
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=="__main__": main()
