#!/usr/bin/env python3
"""Post-run endpoint integrity, reconstruction drift, and registered comparison."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from tokengs.options import config_defaults
from tokengs.models import model_registry
from scripts.anchor_group_v1 import PRETRAINED, PRETRAINED_SHA, MANIFEST, MANIFEST_SHA, PLAN, sha256, write_json

OUT=REPO/"group_plus/anchor_group_v1"
S0=REPO/"group_plus/instance_state_v1_generalization"
S1=REPO/"group_plus/instance_state_v2_s1_local3d"
EXPECTED_STEPS=(0,200,500,1000,2000,3500,5000)

def _finite_tree(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(_finite_tree(x) for x in value.values())
    if isinstance(value,(list,tuple)):return all(_finite_tree(x) for x in value)
    return True

def _endpoint_and_drift(path):
    if sha256(PRETRAINED)!=PRETRAINED_SHA:raise RuntimeError("pretrained SHA mismatch during endpoint audit")
    if sha256(MANIFEST)!=MANIFEST_SHA:raise RuntimeError("manifest SHA mismatch during endpoint audit")
    obj=torch.load(path,map_location="cpu",weights_only=False)
    required=("model","optimizer","step","architecture","joint","beta","manifest_sha256","plan_sha256","pretrained_sha256","rng")
    missing=[k for k in required if k not in obj]
    if missing:raise RuntimeError(f"endpoint payload keys missing: {missing}")
    checks={"step":obj["step"]==5000,"architecture":obj["architecture"]=="LOCUSGS_ANCHOR_GROUP_V1","joint":obj["joint"] is True,"beta":obj["beta"]==0,"manifest_sha256":obj["manifest_sha256"]==MANIFEST_SHA,"plan_sha256":obj["plan_sha256"]==sha256(PLAN),"pretrained_sha256":obj["pretrained_sha256"]==PRETRAINED_SHA,"model_finite":_finite_tree(obj["model"]),"optimizer_finite":_finite_tree(obj["optimizer"]),"rng_present":all(k in obj["rng"] and obj["rng"][k] is not None for k in ("python","numpy","torch","cuda"))}
    if not all(checks.values()):raise RuntimeError(f"endpoint integrity checks failed: {checks}")
    opt=config_defaults["train_siu3r_anchor_group_v1"].evolve(dataset_kwargs={"data_root":"/space/mawb/SIU3R/data/scannet"},batch_size=1,num_workers=0,seed=42,num_input_views=2,num_views=4)
    model=model_registry[opt.model_type](opt)
    model.load_state_dict(obj["model"],strict=True)
    rec=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith("anchor_group."))
    grp=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith("anchor_group."))
    frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    checks.update({"reconstruction_trainable":rec==220002620,"anchor_group_trainable":grp==2785296,"frozen_numel_zero":frozen==0})
    if not all(checks.values()):raise RuntimeError(f"final trainability audit failed: {checks}")
    write_json(OUT/"formal_endpoint_step5000_audit.json",{"endpoint":str(path.relative_to(REPO)),"checkpoint_bytes":path.stat().st_size,"step":obj["step"],"architecture":obj["architecture"],"joint":obj["joint"],"beta":obj["beta"],"manifest_sha256":obj["manifest_sha256"],"plan_sha256":obj["plan_sha256"],"pretrained_sha256":obj["pretrained_sha256"],"rng_present":sorted(obj["rng"]),"all_model_tensors_finite":checks["model_finite"],"all_optimizer_tensors_finite":checks["optimizer_finite"],"trainable_reconstruction_numel":rec,"trainable_anchor_group_numel":grp,"frozen_numel":frozen,"checks":checks,"status":"pass"})
    base_obj=torch.load(PRETRAINED,map_location="cpu",weights_only=False);base=base_obj.get("model",base_obj)
    final=obj["model"];names=dict(model.named_parameters());stats={}
    for name,param in names.items():
        if name.startswith("anchor_group."):continue
        if name not in base or base[name].shape!=param.shape:raise RuntimeError(f"reconstruction drift baseline key mismatch: {name}")
        if name.startswith("enc_dec_backbone.encoder_blocks."):cat="encoder"
        elif name.startswith("enc_dec_backbone.decoder_blocks."):cat="decoder"
        elif name.startswith("anchor_decoder."):cat="anchor_geometry"
        elif name.startswith("activation_head."):cat="activation_head"
        else:cat="other_reconstruction"
        x=param.detach().double();y=base[name].double();delta=x-y
        r=stats.setdefault(cat,{"tensor_count":0,"numel":0,"delta_sq":0.0,"base_sq":0.0,"max_abs_delta":0.0})
        r["tensor_count"]+=1;r["numel"]+=param.numel();r["delta_sq"]+=float(delta.square().sum());r["base_sq"]+=float(y.square().sum());r["max_abs_delta"]=max(r["max_abs_delta"],float(delta.abs().max()))
    for r in stats.values():
        r["l2_delta"]=math.sqrt(r.pop("delta_sq"));base_norm=math.sqrt(r.pop("base_sq"));r["baseline_l2"]=base_norm;r["relative_l2_delta"]=r["l2_delta"]/max(base_norm,1e-30)
    write_json(OUT/"formal_reconstruction_drift_audit.json",{"baseline_checkpoint_sha256":PRETRAINED_SHA,"endpoint_step":5000,"categories":stats,"status":"pass"})
    del obj,base_obj,base,final,model
    return json.loads((OUT/"formal_endpoint_step5000_audit.json").read_text()),stats

def _read_curve(root,step):return json.loads((root/f"curves_{step}.json").read_text())
def _num(x):return "NA" if x is None else f"{x:.6f}"
def _curve_table(curves,diagnostics):
    rows=[]
    for step in EXPECTED_STEPS:
        c=curves[str(step)] if str(step) in curves else _read_curve(OUT,step)
        d=json.loads((OUT/f"anchor_group_diagnostics_step{step}.json").read_text())
        vc=c["val32_context"];vt=c["val32_target"];dc=d["val32_context"]
        rows.append({"step":step,"val32_ctx_thing_miou":vc["mIoU_thing"],"val32_target_thing_miou":vt["mIoU_thing"],"val32_ctx_ca_r50":vc["class_agnostic_recall50"],"val32_target_ca_r50":vt["class_agnostic_recall50"],"ctx_tpf":[vc["tp_class_agnostic"],vc["fp_class_agnostic"],vc["fn_class_agnostic"]],"target_tpf":[vt["tp_class_agnostic"],vt["fp_class_agnostic"],vt["fn_class_agnostic"]],"ctx_psnr":vc["psnr"],"target_psnr":vt["psnr"],"anchor_ownership_accuracy":dc["anchor_ownership_accuracy"],"thing_anchor_correct_fraction":dc["thing_anchor_correct_fraction"],"anchor_group_gt_recall50":dc["anchor_group_gt_recall50"]})
    return rows

def _report(endpoint,drift):
    ag_curves={str(s):_read_curve(OUT,s) for s in EXPECTED_STEPS}
    direct={str(s):json.loads((OUT/f"anchor_group_diagnostics_step{s}.json").read_text()) for s in EXPECTED_STEPS}
    rows=_curve_table(ag_curves,direct)
    s0=_read_curve(S0,5000);s1=_read_curve(S1,5000);ag=ag_curves["5000"]
    scopes=("val32_context","val32_target")
    endpoint_rows=[]
    for label,c in (("S0@5k",s0),("S1@5k",s1),("Anchor-Group@5k",ag)):
        for scope in scopes:
            x=c[scope];endpoint_rows.append((label,scope,x["mIoU_thing"],x["mIoU_all_nonempty"],x["mIoU_stuff"],x["class_agnostic_recall50"],x["tp_class_agnostic"],x["fp_class_agnostic"],x["fn_class_agnostic"],x["class_aware_recall50"],x["active_thing_queries"],x["psnr"]))
    a_train=ag["train16_context"];a_val=ag["val32_context"]
    gaps={k:{"train16":a_train[k],"val32":a_val[k],"val32_minus_train16":a_val[k]-a_train[k]} for k in ("mIoU_thing","class_agnostic_recall50","class_aware_recall50")}
    direct5=direct["5000"]["val32_context"];mechanism=direct5["mechanism_means"]
    table="| Step | val32 ctx thing mIoU | val32 target thing mIoU | ctx ca-R50 | target ca-R50 | ctx TP/FP/FN | target TP/FP/FN | ctx/target PSNR | anchor acc | thing-anchor correct | GT recall50 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for x in rows:table+=f"| {x['step']} | {_num(x['val32_ctx_thing_miou'])} | {_num(x['val32_target_thing_miou'])} | {_num(x['val32_ctx_ca_r50'])} | {_num(x['val32_target_ca_r50'])} | {'/'.join(map(str,x['ctx_tpf']))} | {'/'.join(map(str,x['target_tpf']))} | {_num(x['ctx_psnr'])} / {_num(x['target_psnr'])} | {_num(x['anchor_ownership_accuracy'])} | {_num(x['thing_anchor_correct_fraction'])} | {_num(x['anchor_group_gt_recall50'])} |\n"
    endpoints="| Model | Scope | Thing mIoU | all-nonempty mIoU | stuff mIoU | ca-R50 | TP/FP/FN | class-aware R50 | active queries | PSNR |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in endpoint_rows:endpoints+=f"| {r[0]} | {r[1]} | {_num(r[2])} | {_num(r[3])} | {_num(r[4])} | {_num(r[5])} | {r[6]}/{r[7]}/{r[8]} | {_num(r[9])} | {_num(r[10])} | {_num(r[11])} |\n"
    text=f'''# Anchor-Group V1 Formal 5k Result

## Completion and provenance

- Completed: **5000/5000 registered training steps**.
- Endpoint audit: **{endpoint['status'].upper()}**; architecture `LOCUSGS_ANCHOR_GROUP_V1`, joint=true, beta=0.
- Fresh initialization: pretrained reconstruction checkpoint SHA `{PRETRAINED_SHA}`, step 47500; Anchor-Group seed 31415; global seed 42. The Phase-B1 smoke model/optimizer state was not loaded.
- Manifest SHA `{MANIFEST_SHA}`; plan SHA `{sha256(PLAN)}`; 128 scenes / 1024 windows, 5000 exact plan entries.
- Recipe: FP32, batch 1, AdamW betas (0.9, 0.95), LR peaks 1e-4/1e-5, differential ratio 10, WD 0.05 for decay groups, grad clip 1.0, all model parameters trainable.
- Evaluations ran at steps 0, 200, 500, 1000, 2000, 3500, 5000 on locked train16/val8/val32, context and target scopes.

## Registered val32 curves

{table}

Step 200 has understanding weight 0 and is a reconstruction-only adaptation checkpoint; its grouping readout is untrained. Step 500 uses weight 0.375 and remains in the ramp. Step 1000 is the first full-joint checkpoint; steps 2000–5000 are the primary structural comparison interval.

## S0/S1/Anchor-Group endpoint comparison

{endpoints}

## Direct anchor-group diagnostics at step 5000

On val32 context: anchor ownership accuracy `{_num(direct5['anchor_ownership_accuracy'])}`, thing-anchor correct fraction `{_num(direct5['thing_anchor_correct_fraction'])}`, and supported-GT recall50 `{_num(direct5['anchor_group_gt_recall50'])}` ({direct5['anchor_group_gt_success50']}/{direct5['gt_with_anchor_support']} supported GTs). Counts: valid/thing/wall/floor/ignore anchors = {direct5['anchor_valid_count']}/{direct5['anchor_thing_count']}/{direct5['anchor_wall_count']}/{direct5['anchor_floor_count']}/{direct5['anchor_ignore_count']}; GTs with/without support = {direct5['gt_with_anchor_support']}/{direct5['gt_without_anchor_support']}.

Final-layer mechanism diagnostics: assignment entropy `{_num(direct5['assignment_entropy_mean'])}`; thing ownership mass mean/median/p10/p90/max/max-over-median `{_num(mechanism['thing_ownership_mass']['mean'])}/{_num(mechanism['thing_ownership_mass']['median'])}/{_num(mechanism['thing_ownership_mass']['p10'])}/{_num(mechanism['thing_ownership_mass']['p90'])}/{_num(mechanism['thing_ownership_mass']['max'])}/{_num(mechanism['thing_ownership_mass']['max_over_median'])}`; query cosine off-diagonal mean/p90/max `{_num(mechanism['query_cosine']['offdiag_mean'])}/{_num(mechanism['query_cosine']['p90'])}/{_num(mechanism['query_cosine']['max'])}`; no-object probability mean/max `{_num(mechanism['no_object_probability']['mean'])}/{_num(mechanism['no_object_probability']['max'])}`; active queries mean `{_num(direct5['active_thing_queries_mean'])}`.

## Train16 vs val32 gap at step 5000

| Metric | train16 context | val32 context | val32 minus train16 |
|---|---:|---:|---:|
'''
    for k,v in gaps.items():text+=f"| {k} | {_num(v['train16'])} | {_num(v['val32'])} | {_num(v['val32_minus_train16'])} |\n"
    d0=json.loads((OUT/"anchor_group_diagnostics_step0.json").read_text())
    psnr0=ag_curves["0"]["val32_context"]["psnr"];psnr5=ag["val32_context"]["psnr"];psnr_s1=s1["val32_context"]["psnr"]
    text+=f'''
## Reconstruction tradeoff

Val32 context PSNR: step0 `{_num(psnr0)}`, Anchor-Group step5000 `{_num(psnr5)}`, frozen S1@5k `{_num(psnr_s1)}`. Val32 target PSNR at step0/step5000: `{_num(ag_curves['0']['val32_target']['psnr'])}` / `{_num(ag['val32_target']['psnr'])}`. Relative to pretrained step0, the context PSNR change is `{_num(psnr5-psnr0)}` dB.

The trained model parameter drift from the pretrained reconstruction checkpoint is saved in `formal_reconstruction_drift_audit.json`; category L2 / relative L2 / max absolute deltas:

| Category | L2 delta | Relative L2 delta | Max abs delta |
|---|---:|---:|---:|
'''
    for name,x in drift.items():text+=f"| {name} | {_num(x['l2_delta'])} | {_num(x['relative_l2_delta'])} | {_num(x['max_abs_delta'])} |\n"
    ctx0=d0["val32_context"]
    text+=f'''

## Interpretation

- Direct grouping is assessed by ownership accuracy, thing-anchor correct fraction, and GT recall50 above; these measure anchor-domain grouping using the registered unified Hungarian assignment.
- Val32 instance change versus frozen S1: context ca-R50 `{_num(ag['val32_context']['class_agnostic_recall50']-s1['val32_context']['class_agnostic_recall50'])}`; target ca-R50 `{_num(ag['val32_target']['class_agnostic_recall50']-s1['val32_target']['class_agnostic_recall50'])}`. Thing mIoU changes: context `{_num(ag['val32_context']['mIoU_thing']-s1['val32_context']['mIoU_thing'])}`, target `{_num(ag['val32_target']['mIoU_thing']-s1['val32_target']['mIoU_thing'])}`.
- Cross-scene generalization is summarized by the train16/val32 gap table. A train improvement without a val32 improvement indicates a remaining cross-scene generalization bottleneck.
- If anchor-domain grouping scores are strong while 2D ca-R50 remains low, the failure lies downstream in Gaussian rendering, query classification, 2D region readout, or cross-view projection; no architecture change is made here.
- Reconstruction change is quantified by PSNR and parameter drift above; the experiment did not alter the registered recipe in response to intermediate metrics.

All per-window standard evaluator rows, per-scope curve JSONs, direct grouping diagnostic curves, endpoint audit, and parameter drift audit are retained under `group_plus/anchor_group_v1/`. Formal training ended at step 5000. No next experiment was started.
'''
    (OUT/"formal_5k_report.md").write_text(text)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--endpoint",default="/space/mawb/ssst/workspace_group_plus/anchor_group_v1/formal_endpoint_step5000.pt");args=ap.parse_args()
    ep,drift=_endpoint_and_drift(Path(args.endpoint));_report(ep,drift)
    print(json.dumps({"endpoint_status":ep["status"],"drift_categories":list(drift),"report":str(OUT/"formal_5k_report.md")},indent=2))
if __name__=="__main__":main()
