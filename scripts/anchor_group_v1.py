#!/usr/bin/env python3
"""Anchor-Group V1 joint training driver and Phase-B1 audits.

The ``train`` phase implements the registered 5000-step recipe but is not run
by Phase-B1 verification. ``audit`` and ``smoke`` are separate explicit phases.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, shutil, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from tokengs.options import config_defaults
from tokengs.models import model_registry
from tokengs.models.input_types import ModelInput,ModelInputDecoder,split_data
from tokengs.models.anchor_group_locusgs import anchor_group_lr_multiplier,anchor_group_understanding_weight
from scripts.instance_state_generalization import _batch_for,_seen_classes,layered
from scripts.run_instance_state_v1 import PRETRAINED,PRETRAINED_SHA,PRETRAINED_STEP,BASE_PRESET,SEED

OUT=REPO/"group_plus/anchor_group_v1"
MANIFEST=REPO/"group_plus/instance_state_v1_generalization/train128_windows1024.json"
PLAN=REPO/"group_plus/instance_state_v1_generalization/plan_C_frozen_5000.json"
SOURCE_REPORTS=REPO/"group_plus/instance_state_v1_generalization"
MANIFEST_SHA="1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483"
MONITORS=("monitor_train16.json","monitor_8pairs.json","monitor_32pairs.json","train128_class_coverage.json")
EVAL_STEPS=(0,200,500,1000,2000,3500,5000)
TOTAL_STEPS=5000; GROUP_PEAK_LR=1e-4; RECON_PEAK_LR=1e-5; WEIGHT_DECAY=.05; GRAD_CLIP=1.0

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")

def locked_assets():
    manifest=json.loads(MANIFEST.read_text());plan=json.loads(PLAN.read_text())
    if sha256(MANIFEST)!=MANIFEST_SHA:raise RuntimeError("locked training manifest SHA256 mismatch")
    if len(manifest.get("windows",[]))!=1024 or len({w["scene"] for w in manifest["windows"]})!=128:raise RuntimeError("locked manifest must contain 128 scenes and 1024 windows")
    if len(plan.get("entries",[]))!=TOTAL_STEPS:raise RuntimeError("locked 5000-step plan length mismatch")
    for n,entry in enumerate(plan["entries"],1):
        if entry.get("step")!=n:raise RuntimeError(f"locked plan step mismatch at {n}")
        wi=int(entry["window_index"])
        if not 0<=wi<len(manifest["windows"]):raise RuntimeError(f"plan window index invalid at step {n}")
        win=manifest["windows"][wi]
        if any(entry.get(k)!=win.get(k) for k in ("scene","context","novel")):raise RuntimeError(f"locked plan entry differs from manifest at step {n}")
    monitor_hashes={}
    OUT.mkdir(parents=True,exist_ok=True)
    for name in MONITORS:
        src=SOURCE_REPORTS/name;dst=OUT/name
        shutil.copyfile(src,dst)
        hs,hd=sha256(src),sha256(dst)
        if hs!=hd:raise RuntimeError(f"monitor copy SHA mismatch: {name}")
        monitor_hashes[name]={"source_sha256":hs,"copied_sha256":hd,"equal":True}
    return manifest,plan,monitor_hashes

def load_state(path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    return obj["model"] if isinstance(obj,dict) and "model" in obj else obj

def transfer_anchor_group_reconstruction_weights(model,opt,source_state,source_sha=PRETRAINED_SHA):
    """Strictly transfer every canonical reconstruction tensor, preserving AG init."""
    if source_sha!=PRETRAINED_SHA:raise RuntimeError(f"pretrained SHA mismatch: {source_sha}")
    canonical_opt=opt.evolve(model_type="siu3r_locusgs_recon",workspace="",experiment_name="anchor_group_transfer_baseline")
    baseline=model_registry["siu3r_locusgs_recon"](canonical_opt)
    baseline.load_state_dict(source_state,strict=True)
    source=baseline.state_dict();target=model.state_dict()
    group_keys={k for k in target if k.startswith("anchor_group.")}
    non_group={k for k in target if not k.startswith("anchor_group.")}
    missing=sorted(non_group-set(source));extra=sorted(set(source)-non_group)
    shape=[k for k in sorted(non_group & set(source)) if target[k].shape!=source[k].shape]
    if missing or extra or shape:raise RuntimeError(f"strict reconstruction transfer failure missing={missing[:8]} extra={extra[:8]} shape={shape[:8]}")
    group_before={k:target[k].detach().clone() for k in group_keys}
    with torch.no_grad():
        for k in non_group:target[k].copy_(source[k])
    changed=[k for k in group_keys if not torch.equal(target[k],group_before[k])]
    if changed:raise RuntimeError(f"transfer changed seeded anchor_group values: {changed[:8]}")
    audit={"source_sha256":source_sha,"pretrained_step":PRETRAINED_STEP,"matched_reconstruction_tensor_count":len(non_group),"new_anchor_group_tensor_count":len(group_keys),"missing_non_group_keys":missing,"unexpected_source_reconstruction_keys":extra,"shape_mismatch_keys":shape,"group_seed_values_preserved":not changed,"group_init_changed_keys":changed}
    del baseline
    return audit

def build_options():
    return config_defaults["train_siu3r_anchor_group_v1"].evolve(dataset_kwargs={"data_root":"/space/mawb/SIU3R/data/scannet"},batch_size=1,num_workers=0,seed=SEED,num_input_views=2,num_views=4)

def make_model(opt,device,source_state=None):
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    model=model_registry[opt.model_type](opt)
    if source_state is None:source_state=load_state(PRETRAINED)
    transfer=transfer_anchor_group_reconstruction_weights(model,opt,source_state,sha256(PRETRAINED))
    model=model.to(device)
    return model,transfer

def trainability_audit(model,canonical_frozen_names=()):
    rows=list(model.named_parameters());frozen=[n for n,p in rows if not p.requires_grad]
    group_names={n for n,p in rows if n.startswith("anchor_group.")}
    rec_names={n for n,p in rows if not n.startswith("anchor_group.")}
    illegal_group=[n for n,p in rows if n.startswith("anchor_group.") and not p.requires_grad]
    unexpected_frozen=sorted(set(frozen)-set(canonical_frozen_names))
    if illegal_group or unexpected_frozen:raise RuntimeError(f"unexpected frozen parameters: group={illegal_group}, others={unexpected_frozen[:8]}")
    return {"total_parameter_numel":sum(p.numel() for _,p in rows),"trainable_parameter_numel":sum(p.numel() for _,p in rows if p.requires_grad),"frozen_parameter_numel":sum(p.numel() for _,p in rows if not p.requires_grad),"trainable_reconstruction_numel":sum(p.numel() for n,p in rows if p.requires_grad and not n.startswith("anchor_group.")),"trainable_anchor_group_numel":sum(p.numel() for n,p in rows if p.requires_grad and n.startswith("anchor_group.")),"frozen_parameter_names":frozen,"anchor_group_trainable":not illegal_group,"canonical_frozen_names":sorted(canonical_frozen_names)}

def build_optimizer(model):
    params={"anchor_group_decay":[],"anchor_group_nodecay":[],"reconstruction_decay":[],"reconstruction_nodecay":[]}
    names_by_id={};seen=set()
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if id(p) in seen:continue
        seen.add(id(p));names_by_id[id(p)]=name
        group="anchor_group" if name.startswith("anchor_group.") else "reconstruction"
        no_decay=p.ndim==1 or name.endswith(".bias") or name.endswith("query_init") or bool(getattr(p,"_no_weight_decay",False))
        params[f"{group}_{'nodecay' if no_decay else 'decay'}"].append(p)
    specs=[]
    for name,ps in params.items():
        is_group=name.startswith("anchor_group_");no_decay=name.endswith("nodecay")
        specs.append({"params":ps,"lr":GROUP_PEAK_LR if is_group else RECON_PEAK_LR,"weight_decay":0.0 if no_decay else WEIGHT_DECAY,"name":name})
    optimizer=torch.optim.AdamW(specs,betas=(.9,.95))
    locations={}
    for gi,g in enumerate(optimizer.param_groups):
        for p in g["params"]:locations.setdefault(id(p),[]).append(gi)
    expected={id(p) for p in model.parameters() if p.requires_grad}
    actual=set(locations)
    missing=sorted(names_by_id[x] for x in expected-actual if x in names_by_id)
    duplicates=sorted(names_by_id[x] for x,gs in locations.items() if len(gs)>1 and x in names_by_id)
    multi_group=duplicates.copy()
    if missing or duplicates or actual!=expected:raise RuntimeError(f"optimizer coverage failure missing={missing} duplicates={duplicates}")
    rows=[]
    for g in optimizer.param_groups:
        rows.append({"name":g["name"],"tensor_count":len(g["params"]),"numel":sum(p.numel() for p in g["params"]),"lr":g["lr"],"weight_decay":g["weight_decay"]})
    audit={"betas":[.9,.95],"groups":rows,"duplicates":duplicates,"missing":missing,"multi_group":multi_group,"unique_trainable_parameter_count":len(expected),"optimizer_parameter_count":len(actual),"all_trainable_parameters_exactly_once":actual==expected}
    return optimizer,audit

def set_optimizer_lr(optimizer,step):
    mult=anchor_group_lr_multiplier(step,TOTAL_STEPS)
    for g in optimizer.param_groups:g["lr"]=(GROUP_PEAK_LR if g["name"].startswith("anchor_group_") else RECON_PEAK_LR)*mult

def tensor_grad_report(param,name):
    grad=param.grad
    if grad is None:return {"parameter_name":name,"grad_norm":0.0,"max_abs_grad":0.0,"finite":True,"nonzero":False,"is_none":True}
    g=grad.detach().float();return {"parameter_name":name,"grad_norm":float(g.norm()),"max_abs_grad":float(g.abs().max()),"finite":bool(torch.isfinite(g).all()),"nonzero":bool(torch.isfinite(g).all() and g.norm()>0),"is_none":False}

def find_grad(model,predicate):
    for n,p in model.named_parameters():
        if predicate(n,p) and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.detach().float().norm()>0:return tensor_grad_report(p,n)
    # Report the first matching name even when absent/zero.
    for n,p in model.named_parameters():
        if predicate(n,p):return tensor_grad_report(p,n)
    return {"parameter_name":None,"grad_norm":0.0,"max_abs_grad":0.0,"finite":False,"nonzero":False,"is_none":True}

def finite_tensor(x):return bool(torch.isfinite(x).all())

def fresh_real_batch(opt,device,window_index=0,plan_entry=None):
    m=json.loads(MANIFEST.read_text());window=m["windows"][window_index]
    if plan_entry is not None:window=plan_entry
    return _batch_for(opt,window,device),window

def post_transfer_reconstruction_parity(model,opt,source_state,device):
    from tokengs.models.canonical_recon_models import LocusGSRecon
    batch,window=fresh_real_batch(opt,device,0);mi,_=split_data(batch,opt)
    dec=ModelInputDecoder(cam_view=batch["cam_view_all"],intrinsics=batch["intrinsics_all"])
    with torch.no_grad():
        ag=model.forward_anchor_group(ModelInput(mi.encoder,dec),render_decoder_input=dec,coupled=False,step=0)
        ag_gs=ag["gaussians"].float().cpu();ag_rgb=ag["render"]["images_pred"].float().cpu()
    del ag;gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()
    baseline_opt=opt.evolve(model_type="siu3r_locusgs_recon",workspace="",experiment_name="anchor_group_post_transfer_parity")
    baseline=LocusGSRecon(baseline_opt);baseline.load_state_dict(source_state,strict=True);baseline=baseline.to(device).eval()
    with torch.no_grad():base=baseline.forward_reconstruction_only(ModelInput(mi.encoder,dec),render_decoder_input=dec)
    base_gs=base["gaussians"].float().cpu();base_rgb=base["render"]["images_pred"].float().cpu();gt=batch["images_all"].float().cpu()
    rgbdiff=(ag_rgb-base_rgb).abs();gd=float((ag_gs-base_gs).abs().max())
    def psnr(x):return float((-10*torch.log10((x-gt).square().mean(dim=(-1,-2,-3)))).mean())
    pb,pa=psnr(base_rgb),psnr(ag_rgb);pd=abs(pb-pa)
    payload={"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU","cuda_available":torch.cuda.is_available(),"checkpoint":str(PRETRAINED),"checkpoint_sha256":sha256(PRETRAINED),"scene":window["scene"],"context_frames":window["context"],"after_strict_transfer":True,"coupled":False,"beta":0.0,"gaussian_max_abs_diff":gd,"rgb_max_abs_diff":float(rgbdiff.max()),"rgb_mean_abs_diff":float(rgbdiff.mean()),"baseline_psnr":pb,"anchor_group_psnr":pa,"psnr_abs_diff":pd,"status":"pass" if gd==0 and float(rgbdiff.max())==0 and pd==0 else "fail","pass_target":gd==0 and float(rgbdiff.max())==0 and pd==0}
    write_json(OUT/"reconstruction_parity.json",payload)
    del base,baseline,ag_gs,ag_rgb,base_gs,base_rgb,gt,batch;gc.collect()
    if not payload["pass_target"]:raise RuntimeError(f"post-transfer reconstruction parity failed: {payload}")
    return payload

def audit_warmup_lr():
    uw={s:anchor_group_understanding_weight(s) for s in (0,1,199,200,201,600,999,1000,5000)}
    expected={0:0.,1:0.,199:0.,200:0.,201:.00125,600:.5,999:.99875,1000:1.,5000:1.}
    ue={str(k):abs(uw[k]-v) for k,v in expected.items()}
    steps=(0,1,100,200,201,1000,2500,5000);mult={s:anchor_group_lr_multiplier(s,TOTAL_STEPS) for s in steps}
    lr={str(s):{"anchor_group":GROUP_PEAK_LR*mult[s],"reconstruction":RECON_PEAK_LR*mult[s],"ratio":10.0 if s else None} for s in steps}
    for s in steps:
        if s and abs(lr[str(s)]["anchor_group"]/lr[str(s)]["reconstruction"]-10)>1e-10:raise RuntimeError("LR ratio contract failed")
    if any(x>1e-12 for x in ue.values()) or abs(mult[5000]-.02)>1e-12:raise RuntimeError("warmup/LR numerical contract failed")
    return {"understanding_weight":{str(k):uw[k] for k in uw},"understanding_expected_abs_error":ue,"lr_multiplier":{str(k):mult[k] for k in mult},"lr_by_step":lr,"step5000_floor":{"anchor_group":2e-6,"reconstruction":2e-7},"pass":True}

def canonical_frozen_names(opt):
    base_opt=opt.evolve(model_type="siu3r_locusgs_recon",experiment_name="anchor_group_trainability_reference")
    baseline=model_registry["siu3r_locusgs_recon"](base_opt)
    names=[n for n,p in baseline.named_parameters() if not p.requires_grad];del baseline;gc.collect();return names

def run_grad_audits(model,opt,device):
    batch,_=fresh_real_batch(opt,device,0);optimizer,_=build_optimizer(model)
    optimizer.zero_grad(set_to_none=True)
    _,metrics=model.step_loss(batch,step=1000,coupled=False,understanding_weight_override=1.0)
    loss_u=metrics["loss_understanding"]
    if not torch.is_tensor(loss_u) or not loss_u.requires_grad:raise RuntimeError("understanding loss graph missing")
    loss_u.backward()
    query=find_grad(model,lambda n,p:n=="anchor_group.query_init")
    late=find_grad(model,lambda n,p:n.startswith("enc_dec_backbone.decoder_blocks.11."))
    geom=find_grad(model,lambda n,p:n=="anchor_decoder.mu")
    rho=find_grad(model,lambda n,p:n=="anchor_decoder.rho")
    refine_mu=find_grad(model,lambda n,p:n.startswith("anchor_decoder.refine_mu."))
    refine_rho=find_grad(model,lambda n,p:n.startswith("anchor_decoder.refine_rho."))
    feature=find_grad(model,lambda n,p:n.startswith("enc_dec_backbone.decoder_blocks.11.") and ".gs_" in n)
    activation=find_grad(model,lambda n,p:n.startswith("activation_head."))
    joint={"understanding_weight":1.0,"loss_understanding":float(loss_u.detach()),"query_init":query,"late_decoder":late,"anchor_mu":geom,"anchor_rho":rho,"anchor_refine_mu":refine_mu,"anchor_refine_rho":refine_rho,"reconstruction_feature":feature,"activation_head":activation}
    joint_gates={"query_init":query["nonzero"],"late_decoder":late["nonzero"],"anchor_mu":geom["nonzero"],"reconstruction_feature":feature["nonzero"],"activation_head":activation["nonzero"]}
    write_json(OUT/"phase_b1_joint_gradient_audit.json",{"items":joint,"gates":joint_gates,"pass":all(joint_gates.values())})
    if not all(joint_gates.values()):raise RuntimeError(f"understanding gradient gate failed: {joint_gates}")
    optimizer.zero_grad(set_to_none=True);del metrics,loss_u;gc.collect();torch.cuda.empty_cache()
    _,recon_metrics=model.step_loss(batch,step=1000,coupled=False,understanding_weight_override=1.0)
    loss_r=recon_metrics["loss_recon"]
    if not torch.is_tensor(loss_r) or not loss_r.requires_grad:raise RuntimeError("reconstruction loss graph missing")
    loss_r.backward()
    recon_late=find_grad(model,lambda n,p:n.startswith("enc_dec_backbone.decoder_blocks.11."))
    recon_mu=find_grad(model,lambda n,p:n=="anchor_decoder.mu")
    recon_act=find_grad(model,lambda n,p:n.startswith("activation_head."))
    qrecon=tensor_grad_report(model.anchor_group.query_init,"anchor_group.query_init")
    q_allowed=qrecon["is_none"] or (qrecon["finite"] and not qrecon["nonzero"] and qrecon["max_abs_grad"]==0)
    recon_gates={"late_decoder":recon_late["nonzero"],"anchor_mu":recon_mu["nonzero"],"activation_head":recon_act["nonzero"],"query_init_none_or_zero":q_allowed}
    write_json(OUT/"phase_b1_recon_gradient_audit.json",{"loss_recon":float(loss_r.detach()),"late_decoder":recon_late,"anchor_mu":recon_mu,"activation_head":recon_act,"anchor_group_query_init":qrecon,"gates":recon_gates,"pass":all(recon_gates.values())})
    if not all(recon_gates.values()):raise RuntimeError(f"reconstruction gradient gate failed: {recon_gates}")
    return joint,recon_gates

def eval_interface_smoke(model,opt,device):
    from scripts.eval_instance_state_v1 import evaluate_windows
    item=json.loads((OUT/"monitor_32pairs.json").read_text())["pairs"][0]
    result=evaluate_windows(model,opt,[item],step=0,scope="context",output_dir=OUT/"eval_interface",arm="C",device=device,batch_builder=_batch_for)
    row=result["windows"][0]; inst=row["instance_class_aware"]
    finite=all(math.isfinite(float(row[k])) for k in ("semantic_miou","psnr")) and all(isinstance(inst[k],(int,float)) and math.isfinite(float(inst[k])) for k in ("n_gt","tp","fp","fn","precision","recall")) and inst["n_gt"]>=0
    payload={"status":"pass" if finite else "fail","scope":"context","window":item,"semantic_miou":row["semantic_miou"],"instance_class_aware":inst,"instance_class_agnostic":row["instance_class_agnostic"],"psnr":row["psnr"],"finite":finite}
    write_json(OUT/"phase_b1_eval_interface_smoke.json",payload)
    if not finite:raise RuntimeError("one-window evaluator smoke produced nonfinite metrics")
    return payload

@torch.no_grad()
def anchor_group_window_diagnostics(prediction,batch):
    """Evaluation-only ownership/matching diagnostics; never contributes to loss."""
    from tokengs.models.anchor_group_loss import FLOOR, IGNORE, THING, WALL, unified_hungarian
    final=prediction["states"][-1]; A_all=final["A_post"].float(); A=A_all[0]; q=final["q"][0].float()
    mass=A[:,:100].sum(0); qn=torch.nn.functional.normalize(q,dim=-1,eps=1e-6)
    cos=qn@qn.T; off=cos[~torch.eye(cos.shape[0],dtype=torch.bool,device=cos.device)]
    ent=-(A.clamp_min(1e-9)*A.clamp_min(1e-9).log()).sum(-1).mean()
    targets,pairs=unified_hungarian(prediction,batch)
    ownership_correct=thing_correct=valid_count=thing_count=supported=success=0
    wall_count=floor_count=ignore_count=without_support=0; ce_rows=[]; dice_rows=[]
    per_gt=[]
    for b,(qi,ki) in enumerate(pairs):
        kinds=targets["anchor_kind"][b]; aid=targets["anchor_instance_id"][b]
        valid=targets["anchor_valid"][b]
        tgt=torch.full_like(kinds,-1);tgt[kinds==WALL]=100;tgt[kinds==FLOOR]=101
        matched={int(k):int(j) for j,k in zip(qi.tolist(),ki.tolist())}
        ids=targets["gt_instance_ids"][b].tolist()
        id_to_gt={int(iid):k for k,iid in enumerate(ids)}
        for iid,k in id_to_gt.items():
            own=(kinds==THING)&(aid==iid)
            if own.any():
                if k not in matched: raise RuntimeError("diagnostic found supported GT without unified Hungarian match")
                tgt[own]=matched[k]
        use=valid&(tgt>=0); pred_chan=A_all[b].argmax(-1)
        valid_count+=int(use.sum()); ownership_correct+=int((pred_chan[use]==tgt[use]).sum())
        wall_count+=int((kinds==WALL).sum());floor_count+=int((kinds==FLOOR).sum());ignore_count+=int((kinds==IGNORE).sum())
        thing=(kinds==THING);thing_count+=int(thing.sum());thing_correct+=int((pred_chan[thing]==tgt[thing]).sum())
        ce_rows.extend((-torch.log(A_all[b,use,tgt[use]].clamp_min(1e-6))).tolist())
        Y=targets["Y_anchor"][b]
        support=Y.sum(-1)>0;supported+=int(support.sum());without_support+=int((~support).sum())
        for j,k in zip(qi.tolist(),ki.tolist()):
            y=Y[k];
            if float(y.sum())>0:
                pp=A_all[b,valid,j]; yy=y[valid]
                dice=1-(2*(pp*yy).sum()+1)/(pp.sum()+yy.sum()+1)
                dice_rows.append(float(dice))
                frac=float((pred_chan[(kinds==THING)&(aid==ids[k])]==j).float().mean())
                ok=frac>0.5;success+=int(ok)
                per_gt.append({"instance_id":int(ids[k]),"query":int(j),"anchor_count":int(y.sum()),"matched_query_fraction":frac,"success_gt_recall50":ok})
    ce=float(torch.tensor(ce_rows).mean()) if ce_rows else 0.0
    dice=float(np.mean(dice_rows)) if dice_rows else 0.0
    p=prediction["p_class"][0].float();pthing=p[:,:18].sum(-1)
    med=float(mass.median())
    return {
        "assignment_entropy":float(ent),
        "thing_ownership_mass":{"mean":float(mass.mean()),"median":med,"p10":float(mass.quantile(.10)),"p90":float(mass.quantile(.90)),"max":float(mass.max()),"max_over_median":float(mass.max()/max(med,1e-12))},
        "query_cosine":{"offdiag_mean":float(off.mean()),"p90":float(off.quantile(.90)),"max":float(off.max())},
        "no_object_probability":{"mean":float(p[:,18].mean()),"max":float(p[:,18].max())},
        "active_thing_queries":int((pthing>=.5).sum()),
        "anchor_valid_count":valid_count,"anchor_thing_count":thing_count,"anchor_wall_count":wall_count,"anchor_floor_count":floor_count,"anchor_ignore_count":ignore_count,
        "gt_with_anchor_support":supported,"gt_without_anchor_support":without_support,
        "anchor_ownership_correct":ownership_correct,"anchor_ownership_accuracy":ownership_correct/max(1,valid_count),
        "thing_anchor_correct":thing_correct,"thing_anchor_correct_fraction":thing_correct/max(1,thing_count),
        "anchor_group_gt_success50":success,"anchor_group_gt_recall50":success/max(1,supported),
        "anchor_ce":ce,"anchor_dice":dice,"loss_anchor_group":ce+dice,
        "supported_gt_details":per_gt,
    }

def _aggregate_anchor_group_diagnostics(rows):
    ds=[r["anchor_group_diagnostics"] for r in rows]
    if not ds:return {}
    sums={k:sum(d[k] for d in ds) for k in ("anchor_valid_count","anchor_thing_count","anchor_wall_count","anchor_floor_count","anchor_ignore_count","gt_with_anchor_support","gt_without_anchor_support","anchor_ownership_correct","thing_anchor_correct","anchor_group_gt_success50")}
    return {
        **sums,
        "anchor_ownership_accuracy":sums["anchor_ownership_correct"]/max(1,sums["anchor_valid_count"]),
        "thing_anchor_correct_fraction":sums["thing_anchor_correct"]/max(1,sums["anchor_thing_count"]),
        "anchor_group_gt_recall50":sums["anchor_group_gt_success50"]/max(1,sums["gt_with_anchor_support"]),
        "anchor_ce":float(np.mean([d["anchor_ce"] for d in ds])),"anchor_dice":float(np.mean([d["anchor_dice"] for d in ds])),"loss_anchor_group":float(np.mean([d["loss_anchor_group"] for d in ds])),
        "mechanism_means":{k:{stat:float(np.mean([d[k][stat] for d in ds])) for stat in stats} for k,stats in (("thing_ownership_mass",("mean","median","p10","p90","max","max_over_median")),("query_cosine",("offdiag_mean","p90","max")),("no_object_probability",("mean","max")))},
        "assignment_entropy_mean":float(np.mean([d["assignment_entropy"] for d in ds])),"active_thing_queries_mean":float(np.mean([d["active_thing_queries"] for d in ds])),
        "windows":ds,
    }

def evaluate_anchor_group_all(model,opt,reports,step,device,seen):
    from scripts.eval_instance_state_v1 import evaluate_windows
    train16=json.loads((reports/"monitor_train16.json").read_text())["windows"]
    val8=json.loads((reports/"monitor_8pairs.json").read_text())["pairs"]
    val32=json.loads((reports/"monitor_32pairs.json").read_text())["pairs"]
    aggregate_metrics={};direct={}
    for name,windows in (("train16",train16),("val8",val8),("val32",val32)):
        for scope in ("context","target"):
            res=evaluate_windows(model,opt,windows,step,scope,reports/f"eval_{name}",arm="C",device=str(device),batch_builder=_batch_for,row_diagnostic_fn=anchor_group_window_diagnostics)
            aggregate_metrics[f"{name}_{scope}"]=layered(res["windows"],seen)
            direct[f"{name}_{scope}"]=_aggregate_anchor_group_diagnostics(res["windows"])
    write_json(reports/f"curves_{step}.json",aggregate_metrics)
    write_json(reports/f"anchor_group_diagnostics_step{step}.json",direct)
    return aggregate_metrics

def setup_audit(device):
    manifest,plan,monitor_hashes=locked_assets();
    if sha256(PRETRAINED)!=PRETRAINED_SHA:raise RuntimeError("pretrained checkpoint SHA256 mismatch")
    opt=build_options();source=load_state(PRETRAINED);model,transfer=make_model(opt,device,source)
    write_json(OUT/"phase_b1_pretrained_transfer_audit.json",transfer)
    post_transfer_reconstruction_parity(model,opt,source,device)
    frozen_ref=canonical_frozen_names(opt);trainability=trainability_audit(model,frozen_ref)
    optimizer,opt_audit=build_optimizer(model);set_optimizer_lr(optimizer,1)
    opt_audit.update(trainability);opt_audit["monitor_sha256"]=monitor_hashes;opt_audit["manifest_sha256"]=MANIFEST_SHA;opt_audit["plan_sha256"]=sha256(PLAN);opt_audit["pretrained_sha256"]=sha256(PRETRAINED)
    write_json(OUT/"phase_b1_optimizer_audit.json",opt_audit)
    warm=audit_warmup_lr();write_json(OUT/"phase_b1_warmup_contract.json",warm)
    run_grad_audits(model,opt,device)
    eval_interface_smoke(model,opt,device)
    del optimizer,model,source;gc.collect()
    return 0

def smoke_one_step(device):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!="NVIDIA GeForce RTX 3090":raise RuntimeError("Phase-B1 optimizer smoke requires NVIDIA GeForce RTX 3090")
    prop=torch.cuda.get_device_properties(0)
    if prop.total_memory < 23*1024**3:raise RuntimeError(f"3090 memory gate expects 24GB class device, found {prop.total_memory/1024**3:.2f} GiB")
    manifest,plan,_=locked_assets();entry=plan["entries"][999]
    if entry["step"]!=1000:raise RuntimeError("locked plan entry for global step 1000 not found")
    opt=build_options();source=load_state(PRETRAINED);model,transfer=make_model(opt,device,source);optm,opt_audit=build_optimizer(model);model.train();set_optimizer_lr(optm,1000)
    window={k:entry[k] for k in ("scene","context","novel")};batch=_batch_for(opt,window,device)
    grad_audit=json.loads((OUT/"phase_b1_joint_gradient_audit.json").read_text());recon_name=grad_audit["items"]["late_decoder"]["parameter_name"]
    named=dict(model.named_parameters());p_recon=named[recon_name];q=model.anchor_group.query_init
    q0=q.detach().clone();r0=p_recon.detach().clone()
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();allocated_before=torch.cuda.memory_allocated();reserved_before=torch.cuda.memory_reserved()
    optm.zero_grad(set_to_none=True);forward_ok=backward_ok=step_ok=False;phase="forward"
    try:
        _,metrics=model.step_loss(batch,step=1000,coupled=False)
        loss=metrics["loss"];u=anchor_group_understanding_weight(1000)
        losses={k:(float(metrics[k].detach()) if torch.is_tensor(metrics[k]) else float(metrics[k])) for k in ("loss","loss_recon","loss_understanding","loss_anchor_group")}
        forward_ok=all(math.isfinite(v) for v in losses.values()) and u==1.0
        phase="backward";loss.backward()
        grad_norm=clip_grad_norm_(model.parameters(),GRAD_CLIP,error_if_nonfinite=True)
        backward_ok=bool(torch.isfinite(grad_norm)) and all(p.grad is None or finite_tensor(p.grad) for p in model.parameters())
        phase="optimizer_step";optm.step();step_ok=all(finite_tensor(p) for p in (q,p_recon))
    except torch.cuda.OutOfMemoryError as e:
        payload={"status":"OOM","oom_phase":phase,"error":str(e),"gpu":torch.cuda.get_device_name(0),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"total_memory_gib":prop.total_memory/1024**3,"allocated_before_gib":allocated_before/1024**3,"reserved_before_gib":reserved_before/1024**3,"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,"peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3}
        write_json(OUT/"phase_b1_gpu_one_step_smoke.json",payload);return 1
    qdelta=float((q.detach()-q0).abs().max());rdelta=float((p_recon.detach()-r0).abs().max())
    payload={"status":"pass" if forward_ok and backward_ok and step_ok and qdelta>0 and rdelta>0 else "fail","gpu":torch.cuda.get_device_name(0),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"total_memory_gib":prop.total_memory/1024**3,"step":1000,"understanding_weight":u,"losses":losses,"group_lr":next(g["lr"] for g in optm.param_groups if g["name"].startswith("anchor_group_")),"reconstruction_lr":next(g["lr"] for g in optm.param_groups if g["name"].startswith("reconstruction_")),"query_init_delta":qdelta,"reconstruction_parameter_name":recon_name,"reconstruction_param_delta":rdelta,"allocated_before_gib":allocated_before/1024**3,"reserved_before_gib":reserved_before/1024**3,"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,"peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"forward_finite":forward_ok,"backward_finite":backward_ok,"optimizer_step_finite":step_ok,"transfer":transfer,"optimizer_group_audit":opt_audit}
    write_json(OUT/"phase_b1_gpu_one_step_smoke.json",payload);return int(payload["status"]!="pass")

def train_formal(device):
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("formal Anchor-Group run requires CUDA")
    if torch.cuda.get_device_name(device)!="NVIDIA GeForce RTX 3090":raise RuntimeError("formal Anchor-Group run requires NVIDIA GeForce RTX 3090")
    if torch.cuda.get_device_properties(device).total_memory<23*1024**3:raise RuntimeError("formal Anchor-Group run requires a 24GB-class RTX 3090")
    manifest,plan,_=locked_assets();opt=build_options();model,transfer=make_model(opt,device);optimizer,opt_audit=build_optimizer(model)
    if model.architecture_name!="LOCUSGS_ANCHOR_GROUP_V1":raise RuntimeError(f"unexpected architecture: {model.architecture_name}")
    if any(not p.requires_grad for p in model.parameters()):raise RuntimeError("formal joint run requires all parameters trainable")
    runtime={"event":"train_start","gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else str(device),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"seed":SEED,"anchor_group_init_seed":int(getattr(opt,"anchor_group_init_seed",31415)),"pretrained_sha256":sha256(PRETRAINED),"manifest_sha256":sha256(MANIFEST),"plan_sha256":sha256(PLAN),"trainable_reconstruction_numel":sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith("anchor_group.")),"trainable_anchor_group_numel":sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith("anchor_group.")),"frozen_numel":sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    print(json.dumps(runtime,sort_keys=True),flush=True)
    model.train();seen=_seen_classes(OUT);evaluate_anchor_group_all(model,opt,OUT,0,device,seen)
    provider_cache=None
    for entry in plan["entries"]:
        step=int(entry["step"]);set_optimizer_lr(optimizer,step);optimizer.zero_grad(set_to_none=True)
        batch=_batch_for(opt,entry,device)
        _,metrics=model.step_loss(batch,step=step,coupled=False)
        for key in ("loss","loss_recon","loss_understanding"):
            if not torch.isfinite(metrics[key]).all():raise RuntimeError(f"nonfinite {key} at step {step}")
        metrics["loss"].backward();clip_grad_norm_(model.parameters(),GRAD_CLIP,error_if_nonfinite=True);optimizer.step()
        if step % 100 == 0:
            log_keys=("loss","loss_recon","loss_understanding","understanding_weight","loss_thing_2d","loss_stuff_2d","loss_semantic","loss_identity","loss_anchor_group","anchor_ce","anchor_dice")
            row={k:(float(metrics[k].detach()) if torch.is_tensor(metrics[k]) else float(metrics[k])) for k in log_keys}
            row.update(event="train_step",step=step,group_lr=next(g["lr"] for g in optimizer.param_groups if g["name"].startswith("anchor_group_")),reconstruction_lr=next(g["lr"] for g in optimizer.param_groups if g["name"].startswith("reconstruction_")))
            print(json.dumps(row,sort_keys=True),flush=True)
        if step in EVAL_STEPS[1:]:evaluate_anchor_group_all(model,opt,OUT,step,device,seen)
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":TOTAL_STEPS,"architecture":"LOCUSGS_ANCHOR_GROUP_V1","joint":True,"beta":0,"manifest_sha256":MANIFEST_SHA,"plan_sha256":sha256(PLAN),"pretrained_sha256":sha256(PRETRAINED),"rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    target=Path(opt.workspace)/"formal_endpoint_step5000.pt";target.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,target)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=("audit","smoke","train"),required=True);ap.add_argument("--device",default="cuda");args=ap.parse_args()
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    if args.phase=="audit":return setup_audit(device)
    if args.phase=="smoke":return smoke_one_step(device)
    return train_formal(device)
if __name__=="__main__":raise SystemExit(main())
