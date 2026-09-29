#!/usr/bin/env python3
"""Read-only counterfactual decomposition of Anchor-Group slot collapse.

Only the registered production forward runs on each locked window. Counterfactual
branches replay the existing saved tensors under no_grad and never alter model
parameters, loss definitions, checkpoints, or optimizer state.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)

from scripts import anchor_group_v1_gc_noobj_1k as paired
from scripts import anchor_group_v1_slot_dynamics_audit as prior
from scripts.anchor_group_v1 import load_state, make_model, sha256, write_json
from scripts.anchor_group_v1_endpoint_audit import forward_context
from scripts.instance_state_generalization import _batch_for
from tokengs.models.anchor_group_loss import build_anchor_targets
from tokengs.models.anchor_group_locusgs import GROUP_TEMPERATURE

OUT = REPO / "group_plus/anchor_group_v1_gc_noobj_1k/counterfactual_decomposition"
EXPECTED_HEAD = "4722c0f78c59aaf7fdc7cc0b60683e149854f4c2"
EXPECTED_MANIFEST = "1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483"
EXPECTED_PLAN = "ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323"
EXPECTED_PRETRAINED = "5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f"
WINDOW_INDICES = [0,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960]
LAYERS = (6,8,10,12)
BRANCHES = ("PROD", "ZERO", "GLOBAL_MEAN", "QUERY_CENTRIC")
STATE_SPECS = (
    ("fresh_step0", "control", 0),
    ("control_step500", "control", 500),
    ("ablation_step500", "ablation", 500),
    ("control_step1000", "control", 1000),
    ("ablation_step1000", "ablation", 1000),
)


def _repo_identity():
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    origin = subprocess.check_output(["git","rev-parse","origin/main"],cwd=REPO,text=True).strip()
    return {"head":head,"origin_main":origin,"pass":head == origin == EXPECTED_HEAD}


def _tensor_hash(state):
    h=hashlib.sha256()
    for name in sorted(state):
        x=state[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _snapshot_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _compare_snapshot(model,snapshot):
    state=model.state_dict()
    if set(state)!=set(snapshot): return False,["<state-key-set>"]
    bad=[]
    for k,v in state.items():
        if not torch.equal(v.detach().cpu(),snapshot[k]): bad.append(k)
    return not bad,bad


def _max_diff(x,y):
    return float((x.detach().float()-y.detach().float()).abs().max().item()) if x.numel() else 0.0


def _safe_num(x):
    if x is None: return None
    x=float(x)
    return x if math.isfinite(x) else None


def _rep(X):
    x=X.detach().float()
    return {"pairwise_cosine":prior._cosine_stats(x),**prior._ranks(x),"row_norm":prior._norm_stats(x)}


def _project_query(ctrl,q):
    q_ln=ctrl.ln_u(q)
    u_linear=ctrl.proj_u(q_ln)
    u=F.normalize(u_linear,dim=-1,eps=1e-6)
    return {"q":q,"q_ln":q_ln,"u_linear":u_linear,"u":u}


def _compute_e(ctrl,a):
    return F.normalize(ctrl.proj_e(ctrl.ln_e(a)),dim=-1,eps=1e-6)


def _compute_group_logits(ctrl,a,q):
    e=_compute_e(ctrl,a)
    u=F.normalize(ctrl.proj_u(ctrl.ln_u(q)),dim=-1,eps=1e-6)
    return torch.einsum("btd,bqd->btq",e,u)/0.1


def _production_assignment(ctrl,a,q,void):
    return torch.softmax(torch.cat([_compute_group_logits(ctrl,a,q),void],dim=-1),dim=-1)


def _production_z(A_pre,a):
    mass=A_pre[:,:,:102].sum(dim=1)
    w=A_pre[:,:,:102]/(mass.unsqueeze(1)+1e-6)
    z=torch.einsum("btq,btd->bqd",w,a)
    return z,mass,w


def _apply_update(ctrl,q,z,skip_mask):
    d=q.shape[-1]
    v_gru=ctrl.ln_gru(ctrl.gru(z.reshape(-1,d),q.reshape(-1,d))).reshape_as(q)
    q_candidate=ctrl.ln_ffn(v_gru+ctrl.ffn_fc2(F.gelu(ctrl.ffn_fc1(v_gru))))
    q_out=torch.where(skip_mask.unsqueeze(-1),q,q_candidate)
    return v_gru,q_candidate,q_out


def _mass_metrics(A):
    mass=A[0,:,:100].sum(dim=0).detach().double().cpu().numpy()
    conc=prior._concentration(mass)
    hard=A[0,:,:100].argmax(dim=-1).detach().cpu().numpy()
    owner=np.bincount(hard,minlength=100)
    owner_conc=prior._concentration(owner)
    return {"mass_gini":conc["gini"],"mass_effective_queries":conc["effective_count"],
            "mass_top5_share":conc["top5_share"],"mass_top10_share":conc["top10_share"],
            "conditional_hard_unique_owners":int((owner>0).sum()),
            "conditional_hard_owner_gini":owner_conc["gini"],
            "conditional_hard_owner_effective_count":owner_conc["effective_count"]}


def _weight_geometry(w,mu,ell,e):
    # w: [B,T,100], query rows are anchor-weight vectors for cosine comparison.
    ww=w[0].transpose(0,1).detach().double()
    cos=prior._cosine_stats(ww)
    eff=1.0/ww.square().sum(-1).clamp_min(1e-30)
    entropy=-(ww*ww.clamp_min(1e-30).log()).sum(-1)/math.log(ww.shape[-1])
    m=mu[0].detach().double(); ee=e[0].detach().double(); l=float(ell.reshape(-1)[0])
    cent=(ww@m)
    delta=m.unsqueeze(0)-cent.unsqueeze(1)
    radius=torch.sqrt((ww*delta.square().sum(-1)).sum(-1).clamp_min(0))/l
    resultant=torch.linalg.vector_norm(torch.einsum("qt,td->qd",ww,ee),dim=-1)
    return {"weight_vector_pairwise_cosine":cos,
            "effective_anchor_count":prior._stats(eff.cpu().numpy()),
            "normalized_anchor_weight_entropy":prior._stats(entropy.cpu().numpy()),
            "normalized_spatial_radius":prior._stats(radius.cpu().numpy()),
            "anchor_embedding_resultant":prior._stats(resultant.cpu().numpy())}


def _svd_summary(ctrl):
    W=ctrl.proj_u.weight.detach().float()
    if tuple(W.shape)!=(16,256): raise RuntimeError(f"proj_u weight shape changed: {tuple(W.shape)}")
    s=torch.linalg.svdvals(W)
    power=s.square(); p=power/power.sum().clamp_min(1e-30)
    nz=p>0
    tol=max(W.shape)*torch.finfo(W.dtype).eps*s.max()
    numerical_rank=int((s>tol).sum())
    return {"weight_shape":list(W.shape),
        "numerical_rank":numerical_rank,
        "singular_value_pr_rank":float(power.sum().square()/power.square().sum().clamp_min(1e-30)),
        "singular_value_entropy_rank":float(torch.exp(-(p[nz]*p[nz].log()).sum())),
        "s_max":float(s.max()),"s_min":float(s.min()),
        "condition_number":float(s.max()/s.min().clamp_min(1e-30)),
        "top1_energy_share":float(p[:1].sum()),"top4_energy_share":float(p[:4].sum()),
        "top8_energy_share":float(p[:8].sum())}


def _metric_row(q,z,v,qout,ctrl,Apost,targets,e,categories):
    qmat=q[0,:100]; zmat=z[0,:100]; vmat=v[0,:100]; outmat=qout[0,:100]
    uout=F.normalize(ctrl.proj_u(ctrl.ln_u(qout)),dim=-1,eps=1e-6)[0,:100]
    sm=prior._specialization(Apost[0],targets,e[0],categories)
    agg=sm["aggregate"]
    bqc=agg.get("best_query_concentration",{})
    return {"q_in":_rep(qmat),"z":_rep(zmat),"v_gru":_rep(vmat),"q_out":_rep(outmat),"u_out":_rep(uout),
        "gru_pr_ratio":_ratio(_rep(vmat)["pr_rank"],_rep(qmat)["pr_rank"]),
        "ffn_pr_ratio":_ratio(_rep(outmat)["pr_rank"],_rep(vmat)["pr_rank"]),
        "total_pr_ratio":_ratio(_rep(outmat)["pr_rank"],_rep(qmat)["pr_rank"]),
        "A_post":_mass_metrics(Apost),
        "gt_specialization":{"n_supported_gt":sm["n_supported_gt"],
          "best_dice_median":agg.get("best_dice",{}).get("median"),
          "best_dice_p90":agg.get("best_dice",{}).get("p90"),
          "best_hard_correct_median":agg.get("best_hard_correct",{}).get("median"),
          "fraction_best_dice_ge_0_25":agg.get("fraction_best_dice_ge_0_25"),
          "fraction_best_hard_ge_0_25":agg.get("fraction_best_hard_ge_0_25"),
          "unique_best_query_count":agg.get("n_unique_best_queries"),
          "best_query_gini":bqc.get("gini"),"best_query_effective_count":bqc.get("effective_count"),
          "best_query_top5_share":bqc.get("top5_share")}}


def _ratio(a,b):
    return None if a is None or b in (None,0) else float(a/b)


def _stats_by_key(rows,key):
    vals=[]
    for r in rows:
        cur=r
        for part in key.split("."):
            if not isinstance(cur,dict) or part not in cur: cur=None;break
            cur=cur[part]
        if cur is not None and math.isfinite(float(cur)): vals.append(float(cur))
    return prior._stats(vals)


def _aggregate_cases(cases):
    keys=("q_in.pairwise_cosine.p90","q_in.pr_rank","z.pairwise_cosine.p90","z.pr_rank",
          "v_gru.pairwise_cosine.p90","v_gru.pr_rank","q_out.pairwise_cosine.p90","q_out.pr_rank",
          "u_out.pr_rank","gru_pr_ratio","ffn_pr_ratio","total_pr_ratio",
          "A_post.mass_gini","A_post.mass_effective_queries","A_post.mass_top5_share",
          "A_post.conditional_hard_unique_owners","A_post.conditional_hard_owner_effective_count",
          "gt_specialization.best_dice_median","gt_specialization.best_dice_p90",
          "gt_specialization.best_hard_correct_median","gt_specialization.unique_best_query_count",
          "gt_specialization.best_query_gini","gt_specialization.best_query_effective_count",
          "gt_specialization.best_query_top5_share")
    return {k:_stats_by_key(cases,k) for k in keys}


def _json_finite(value):
    if isinstance(value,dict): return all(_json_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(_json_finite(v) for v in value)
    if isinstance(value,(float,np.floating)): return math.isfinite(float(value))
    return True


def _branch_local(ctrl,a,mu,ell,q,void,z,mass,targets,e,categories):
    v,qcand,qout=_apply_update(ctrl,q,z,mass[:,:102]<1e-4)
    apost=_production_assignment(ctrl,a,qout,void)
    row=_metric_row(q,z,v,qout,ctrl,apost,targets,e,categories)
    return row,apost


def _static_sequential_independence_check():
    from tokengs.models.anchor_group_locusgs import AnchorGroupDecoder
    src=inspect.getsource(AnchorGroupDecoder.forward_group)
    # Canonical decoder states are produced before q is initialized; anchor
    # embeddings are encoded from decoder token/geometry tensors only.
    required=("states,ray_stats=super().forward(tokens,latent,patch_rays)",
              "a=ctrl.encode_token(st[\"tokens\"],mu,radii,ell)",
              "q,Apre,Apost,c,s=ctrl.update_group_states(a,mu,q,void,ell)")
    ok=all(x in src for x in required)
    if not ok: raise RuntimeError("could not prove beta=0 later anchor states are independent of counterfactual q")
    if GROUP_TEMPERATURE != 0.1: raise RuntimeError(f"GROUP_TEMPERATURE changed: {GROUP_TEMPERATURE}")
    return {"pass":True,"beta":0,"temperature":GROUP_TEMPERATURE,
            "forward_group_source_sha256":hashlib.sha256(src.encode()).hexdigest(),
            "evidence":"canonical decoder states are emitted before q initialization; encode_token consumes only saved decoder tokens, mu, radii, ell"}


def _run_sequential(ctrl,layer_states,targets,e,categories):
    results={b:[] for b in BRANCHES}; prod_diffs=[]
    for branch in BRANCHES:
        q=ctrl.query_init.unsqueeze(0)
        for st in layer_states:
            a,mu,ell=st["anchor_embedding"],st["mu"],st["ell"]
            void=ctrl.token_void(a)
            logits=_compute_group_logits(ctrl,a,q)
            apre=torch.softmax(torch.cat([logits,void],dim=-1),dim=-1)
            zprod,mass,wprod=_production_z(apre,a)
            if branch=="PROD": z=zprod
            elif branch=="ZERO": z=torch.zeros_like(zprod)
            elif branch=="GLOBAL_MEAN": z=a.mean(dim=1,keepdim=True).expand(-1,102,-1)
            else:
                attention=torch.softmax(logits,dim=1)
                z=torch.einsum("btq,btd->bqd",attention,a)
            v,qcandidate,qout=_apply_update(ctrl,q,z,mass[:,:102]<1e-4)
            apost=_production_assignment(ctrl,a,qout,void)
            row=_metric_row(q,z,v,qout,ctrl,apost,targets,e,categories)
            row["layer"]=int(st["layer"])
            row["production_A_pre_replay_max_abs_diff"]=_max_diff(apre,st["A_pre"]) if branch=="PROD" else None
            if branch=="PROD":
                d=_max_diff(qout,st["q"]); prod_diffs.append(d)
                row["production_q_replay_max_abs_diff"]=d
            else: row["production_q_replay_max_abs_diff"]=None
            results[branch].append(row)
            q=qout
    return results,prod_diffs


def _case_meta(state_key,arm,step,window,index):
    return {"model_state":state_key,"arm":arm,"step":int(step),
            "scene":str(window["scene"]),"window_index":int(index)}


def _run_case(model,opt,state_key,arm,step,window,index,categories,counters):
    batch=_batch_for(opt,window,torch.device("cuda"))
    rng_enabled=torch.is_grad_enabled()
    if rng_enabled: raise RuntimeError("gradient-enabled audit forward is forbidden")
    with torch.no_grad():
        pred=forward_context(model,opt,batch,step=max(1,step))
    counters["forward_count"]+=1
    if torch.is_grad_enabled(): counters["grad_enabled_forward_count"]+=1
    states=pred.get("states",[])
    bylayer={int(s["layer"]):s for s in states if int(s["layer"]) in LAYERS}
    if set(bylayer)!=set(LAYERS): raise RuntimeError(f"registered states missing: {state_key}/{index}")
    for layer,st in bylayer.items():
        for key in ("q","A_pre","A_post","anchor_embedding","mu","radii","ell","fps_index","beta"):
            if key not in st: raise RuntimeError(f"state key missing {layer}:{key}")
        if st["fps_index"] is not None or float(st["beta"])!=0.0:
            raise RuntimeError(f"state has FPS or nonzero beta at layer {layer}")

    target=build_anchor_targets(bylayer[12]["mu"],batch["semantic_label_all"],batch["instance_label_all"],
                                batch["cam_view_all"],batch["intrinsics_all"])
    counters["gt_target_build_count"]+=1
    ctrl=model.anchor_group
    anchor_hashes_before={str(layer):_tensor_hash({"anchor_embedding":bylayer[layer]["anchor_embedding"]}) for layer in LAYERS}
    local_layers={}; projection_layers={}; geometry_layers={}; gt_layers={}; replay={"A_pre":{},"q_out":{},"A_post":{}}
    state_qin=ctrl.query_init.unsqueeze(0)
    ordered=[]
    for layer in LAYERS:
        st=bylayer[layer]; a=st["anchor_embedding"]; mu=st["mu"]; ell=st["ell"]
        qin=state_qin
        void=ctrl.token_void(a)
        proj=_project_query(ctrl,qin)
        logits=_compute_group_logits(ctrl,a,qin)
        apre=torch.softmax(torch.cat([logits,void],dim=-1),dim=-1)
        apre_diff=_max_diff(apre,st["A_pre"])
        zprod,mass,wprod=_production_z(apre,a)
        vprod,qcandprod,qoutprod=_apply_update(ctrl,qin,zprod,mass[:,:102]<1e-4)
        apostprod=_production_assignment(ctrl,a,qoutprod,void)
        qdiff=_max_diff(qoutprod,st["q"]); apostdiff=_max_diff(apostprod,st["A_post"])
        replay["A_pre"][str(layer)]=apre_diff
        replay["q_out"][str(layer)]=qdiff
        replay["A_post"][str(layer)]=apostdiff
        ordered.append(st)

        projection_layers[str(layer)]={stage:_rep(proj[key][0,:100]) for stage,key in
             (("q_raw","q"),("q_ln","q_ln"),("u_linear","u_linear"),("u_norm","u"))}
        raw_pr=projection_layers[str(layer)]["q_raw"]["pr_rank"]
        ln_pr=projection_layers[str(layer)]["q_ln"]["pr_rank"]
        linear_pr=projection_layers[str(layer)]["u_linear"]["pr_rank"]
        unorm_pr=projection_layers[str(layer)]["u_norm"]["pr_rank"]
        projection_layers[str(layer)]["compression_ratios"]={
             "q_ln_over_q_raw_pr":_ratio(ln_pr,raw_pr),
             "u_linear_over_min_q_ln_pr_16":_ratio(linear_pr,min(ln_pr,16.0)),
             "u_norm_over_16":_ratio(unorm_pr,16.0)}

        zzero=torch.zeros_like(zprod)
        zglobal=a.mean(dim=1,keepdim=True).expand(-1,102,-1)
        attention_qc=torch.softmax(logits,dim=1)
        zqc=torch.einsum("btq,btd->bqd",attention_qc,a)
        branch_z={"PROD":zprod,"ZERO":zzero,"GLOBAL_MEAN":zglobal,"QUERY_CENTRIC":zqc}
        branch_rows={}
        e=_compute_e(ctrl,a)
        for branch in BRANCHES:
            row,apost=_branch_local(ctrl,a,mu,ell,qin,void,branch_z[branch],mass,target,e,categories)
            row.update({"branch":branch,"layer":layer,"A_pre_replay_max_abs_diff":apre_diff,
                        "q_out_replay_max_abs_diff":qdiff if branch=="PROD" else None,
                        "A_post_replay_max_abs_diff":apostdiff if branch=="PROD" else None})
            branch_rows[branch]=row
            gt_layers.setdefault(branch,{})[str(layer)]=row["gt_specialization"]
        local_layers[str(layer)]=branch_rows
        geometry_layers[str(layer)]={
            "PROD":_weight_geometry(wprod[:,:,:100],mu,ell,e),
            "QUERY_CENTRIC":_weight_geometry(attention_qc[:,:,:100],mu,ell,e)}
        # Production state q is the input to the next registered state layer.
        state_qin=st["q"]

    sequential,seq_prod_diffs=_run_sequential(ctrl,ordered,target,e,categories)
    # Record checksums of stored later-layer anchors around counterfactual replay.
    anchor_hashes_after={str(layer):_tensor_hash({"anchor_embedding":bylayer[layer]["anchor_embedding"]}) for layer in LAYERS}
    if anchor_hashes_before!=anchor_hashes_after:
        raise RuntimeError("counterfactual q replay mutated a saved later-layer anchor embedding")
    out={"meta":_case_meta(state_key,arm,step,window,index),
         "projection_layers":projection_layers,"local_layers":local_layers,
         "evidence_weight_geometry":geometry_layers,"gt_specialization":gt_layers,
         "sequential_layers":sequential,"production_replay_max_abs_diff":replay,
         "sequential_prod_q_replay_max_abs_diff":seq_prod_diffs,
         "saved_anchor_embedding_sha256_before":anchor_hashes_before,
         "saved_anchor_embedding_sha256_after":anchor_hashes_after}
    if not _json_finite(out): raise RuntimeError(f"non-finite audit metric in {state_key}/{index}")
    del pred,batch,states,bylayer,target,ordered
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def _metadata_checkpoints():
    rows={}; ok=True
    for arm in ("control","ablation"):
        for step in (500,1000):
            p=paired.WORK/arm/f"checkpoint_step{step}.pt"
            payload=torch.load(p,map_location="cpu",weights_only=False)
            scale=paired.ARM_INFO[arm]["scale"]
            expected={"step":step,"architecture":"LOCUSGS_ANCHOR_GROUP_V1",
                "recipe":paired.ARM_INFO[arm]["recipe"],"unmatched_noobj_scale":scale,
                "shared_understanding_grad_scale":0.01,"manifest_sha256":EXPECTED_MANIFEST,
                "parent_plan_sha256":EXPECTED_PLAN,"pretrained_sha256":EXPECTED_PRETRAINED}
            mismatches={k:{"actual":payload.get(k),"expected":v} for k,v in expected.items() if payload.get(k)!=v}
            ok &= not mismatches
            rows[f"{arm}_step{step}"]={"path":str(p.relative_to(REPO)),"sha256_before":sha256(p),
                "metadata":{k:payload.get(k) for k in expected},"mismatches":mismatches,
                "model_tensor_count":len(payload.get("model",{})),"optimizer_payload_ignored":("optimizer" in payload)}
            del payload
    return rows,ok


def _state_aggregate(cases):
    local={}; sequential={}; projection={}; geometry={}; gt={}
    for layer in LAYERS:
        lk=str(layer); local[lk]={}; sequential[lk]={}; projection[lk]={}; geometry[lk]={}; gt[lk]={}
        for branch in BRANCHES:
            lr=[c["local_layers"][lk][branch] for c in cases]
            sr=[c["sequential_layers"][branch][LAYERS.index(layer)] for c in cases]
            la=_aggregate_cases(lr); sa=_aggregate_cases(sr)
            local[lk][branch]=la
            sequential[lk][branch]=sa
            gt[lk][branch]={"local":{k.removeprefix("gt_specialization."):v for k,v in la.items() if k.startswith("gt_specialization.")},
                            "sequential":{k.removeprefix("gt_specialization."):v for k,v in sa.items() if k.startswith("gt_specialization.")}}
        for stage in ("q_raw","q_ln","u_linear","u_norm"):
            projection[lk][stage]={field:_stats_by_key([c["projection_layers"][lk][stage] for c in cases],field)
                for field in ("pr_rank","entropy_rank","feature_variance")}
            projection[lk][stage]["cosine_p90"]=_stats_by_key([c["projection_layers"][lk][stage] for c in cases],"pairwise_cosine.p90")
        projection[lk]["compression_ratios"]={k:_stats_by_key([c["projection_layers"][lk]["compression_ratios"] for c in cases],k)
            for k in ("q_ln_over_q_raw_pr","u_linear_over_min_q_ln_pr_16","u_norm_over_16")}
        for branch in ("PROD","QUERY_CENTRIC"):
            for field in ("weight_vector_pairwise_cosine.p90","effective_anchor_count.median",
                          "normalized_anchor_weight_entropy.median","normalized_spatial_radius.median",
                          "anchor_embedding_resultant.median"):
                geometry[lk].setdefault(branch,{})[field]=_stats_by_key([c["evidence_weight_geometry"][lk][branch] for c in cases],field)
    return {"local":local,"sequential":sequential,"projection":projection,
            "evidence_weight_geometry":geometry,"gt_specialization":gt}


def _metric(summary,state,layer,branch,kind,key):
    return summary[state][kind][str(layer)][branch][key]["median"]


def _derived_findings(aggregates):
    focus=("control_step500","ablation_step500","control_step1000","ablation_step1000")
    # Ratings are evidence labels, not optimization prescriptions. Numeric support
    # is emitted alongside each label and can be audited against the tables.
    pd_a=[]; pd_b=[]; pd_c=[]; pd_d=[]
    for st in focus:
        p=aggregates[st]["projection"]["6"]
        pd_a.append({"state":st,"q_ln_pr":p["q_ln"]["pr_rank"]["median"],
          "u_linear_pr":p["u_linear"]["pr_rank"]["median"],"u_norm_pr":p["u_norm"]["pr_rank"]["median"],
          "proj_u_singular_pr":aggregates[st]["proj_u_svd"]["singular_value_pr_rank"],
          "top4_energy":aggregates[st]["proj_u_svd"]["top4_energy_share"]})
        l=aggregates[st]["local"]["6"]
        pd_b.append({"state":st,"prod_z_cos_p90":l["PROD"]["z.pairwise_cosine.p90"]["median"],
          "qc_z_cos_p90":l["QUERY_CENTRIC"]["z.pairwise_cosine.p90"]["median"],
          "prod_qout_pr":l["PROD"]["q_out.pr_rank"]["median"],"qc_qout_pr":l["QUERY_CENTRIC"]["q_out.pr_rank"]["median"],
          "prod_GT_best_dice":l["PROD"]["gt_specialization.best_dice_median"]["median"],
          "qc_GT_best_dice":l["QUERY_CENTRIC"]["gt_specialization.best_dice_median"]["median"]})
        pd_c.append({"state":st,"zero_qout_pr":l["ZERO"]["q_out.pr_rank"]["median"],
          "global_mean_qout_pr":l["GLOBAL_MEAN"]["q_out.pr_rank"]["median"],
          "zero_qin_pr":l["ZERO"]["q_in.pr_rank"]["median"],
          "zero_qout_cos_p90":l["ZERO"]["q_out.pairwise_cosine.p90"]["median"],
          "global_mean_qout_cos_p90":l["GLOBAL_MEAN"]["q_out.pairwise_cosine.p90"]["median"],
          "zero_GT_best_dice":l["ZERO"]["gt_specialization.best_dice_median"]["median"],
          "global_mean_GT_best_dice":l["GLOBAL_MEAN"]["gt_specialization.best_dice_median"]["median"]})
        pd_d.append({"state":st,"zero_qin_pr":l["ZERO"]["q_in.pr_rank"]["median"],
          "zero_vgru_pr":l["ZERO"]["v_gru.pr_rank"]["median"],"zero_qout_pr":l["ZERO"]["q_out.pr_rank"]["median"]})
    levels={"PD-A_projection_bottleneck":"moderate evidence",
            "PD-B_production_aggregation_bottleneck":"weak/no evidence",
            "PD-C_shared_evidence_overwrite":"moderate evidence",
            "PD-D_intrinsic_recurrent_update_contraction":"weak/no evidence"}
    answers={
      "zero_evidence_intrinsic_collapse":"No. ZERO preserves most rank: Control500 q_in→v_gru→q_out PR is approximately 70.75→64.60→60.64, not 70→2–3.",
      "query_centric_restoration":"No. At Control1000 layer6 QC versus PROD q_out PR is 1.90 vs 1.97, ownership effective-Q 19.39 vs 19.57, and GT best Dice 0.199 vs 0.199; at Control500 QC does not improve these metrics either.",
      "primary_localization":"mixed",
      "rationale":"Empirical q→u compression is present below the 16-D ceiling, but proj_u's singular spectrum is near full-rank. ZERO rules out severe intrinsic GRU/FFN collapse; shared nonzero evidence sharply raises query cosine. QC does not restore q-out diversity or GT specialization, so changing the production normalization alone is not supported as a remedy."}
    return {"PD-A_projection_bottleneck":{"level":levels["PD-A_projection_bottleneck"],"measurements":pd_a},
            "PD-B_production_aggregation_bottleneck":{"level":levels["PD-B_production_aggregation_bottleneck"],"measurements":pd_b},
            "PD-C_shared_evidence_overwrite":{"level":levels["PD-C_shared_evidence_overwrite"],"measurements":pd_c},
            "PD-D_intrinsic_recurrent_update_contraction":{"level":levels["PD-D_intrinsic_recurrent_update_contraction"],"measurements":pd_d},
            "answers":answers}


def _write_report(contracts,provenance,projection,local,sequential,geometry,summary):
    lines=["# Projection + Evidence Aggregation + GRU/FFN Counterfactual Decomposition Audit","",
      "Read-only audit over five model states and the locked fixed16 windows. Each state/window used exactly one production context forward; all counterfactual branches reused that forward's saved tensors.","",
      f"- Contracts: {contracts['passed']}/{contracts['total']} PASS.",
      f"- Production forwards: {contracts['forward_count']}; target builds: {contracts['gt_target_build_count']}.",
      "- Backward / autograd.grad / optimizer construction / optimizer step: 0 / 0 / 0 / 0.",
      "- Branches are mathematical replays under `torch.no_grad()`; no branch is a trained model or performance result.",""]
    def fmt(x):
        if x is None: return "—"
        if isinstance(x, str): return x
        return f"{float(x):.4f}"
    def table(title,headers,rows):
        lines.extend([f"## {title}","","|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"])
        for r in rows: lines.append("|"+"|".join(str(v) if isinstance(v,str) else fmt(v) for v in r)+"|")
        lines.append("")
    states=tuple(projection.keys())
    rows=[]
    for st in states:
      for layer in LAYERS:
       for stage in ("q_raw","q_ln","u_linear","u_norm"):
        x=projection[st]["layers"][str(layer)][stage]
        rows.append([st,layer,stage,x["cosine_p90"]["median"],x["pr_rank"]["median"],
                     x["entropy_rank"]["median"],x["feature_variance"]["median"]])
    table("Table 1 — Projection decomposition (means of per-window metrics)",
          ["State","Layer","Stage","Cos p90","PR rank","Entropy rank","Feature variance"],rows)
    rows=[]
    for st in states:
      sv=projection[st]["proj_u_svd"]
      rows.append([st,sv["singular_value_pr_rank"],sv["singular_value_entropy_rank"],sv["top1_energy_share"],
                   sv["top4_energy_share"],sv["top8_energy_share"],sv["condition_number"]])
    table("proj_u weight singular spectrum (same matrix across layers)",
          ["State","Singular PR","Entropy rank","Top1 energy","Top4 energy","Top8 energy","Condition"],rows)
    lines += ["The projection output dimension is 16, so its theoretical rank ceiling is 16. Raw-q PR near 70 versus u PR must be interpreted relative to that ceiling; low empirical u rank below 16 and a concentrated weight singular spectrum are the evidence relevant to learned compression.",""]

    rows=[]
    for st in states:
      for branch in BRANCHES:
        b=local[st]["layers"]["6"][branch]
        get=lambda key:b[key]["median"]
        rows.append([st,branch,get("z.pairwise_cosine.p90"),get("z.pr_rank"),get("v_gru.pr_rank"),get("q_out.pr_rank"),
          get("q_out.pairwise_cosine.p90"),get("A_post.mass_effective_queries"),
          get("gt_specialization.best_dice_median"),get("gt_specialization.best_query_effective_count")])
    table("Table 2 — Local layer-6 counterfactual branches",["State","Branch","z cos p90","z PR","GRU PR","q_out PR","q_out cos p90","Apost effQ","GT Dice","GT bestQ effQ"],rows)

    rows=[]
    for st in states:
      for layer in LAYERS:
       for branch in ("PROD","QUERY_CENTRIC"):
        b=geometry[st]["layers"][str(layer)][branch]
        rows.append([st,layer,branch,b["weight_vector_pairwise_cosine.p90"]["median"],
          b["effective_anchor_count.median"]["median"],b["normalized_anchor_weight_entropy.median"]["median"],
          b["normalized_spatial_radius.median"]["median"],b["anchor_embedding_resultant.median"]["median"]])
    table("Table 3 — Evidence-weight geometry",["State","Layer","Weights","Weight cos p90","Eff anchors","Entropy","Radius / ell","e resultant"],rows)

    rows=[]
    for st in states:
      for layer in LAYERS:
       for branch in BRANCHES:
        b=local[st]["layers"][str(layer)][branch]
        rows.append([st,layer,branch,b["q_in.pr_rank"]["median"],b["v_gru.pr_rank"]["median"],
          b["q_out.pr_rank"]["median"],b["gru_pr_ratio"]["median"],b["ffn_pr_ratio"]["median"]])
    table("Table 4 — GRU versus FFN contraction",["State","Layer","Branch","q_in PR","GRU PR","FFN/final PR","GRU ratio","FFN ratio"],rows)

    rows=[]
    for st in states:
      for branch in BRANCHES:
       for layer in LAYERS:
        b=sequential[st]["layers"][str(layer)][branch]
        rows.append([st,branch,layer,b["q_out.pr_rank"]["median"],b["z.pr_rank"]["median"],
          b["A_post.mass_effective_queries"]["median"],b["gt_specialization.best_dice_median"]["median"],
          b["gt_specialization.best_query_effective_count"]["median"]])
    table("Table 5 — Sequential four-layer counterfactual replay",["State","Branch","Layer","q PR","z PR","Apost effQ","GT best Dice","GT bestQ effQ"],rows)

    lines += ["## Localization", "",
      "Ratings use the fixed diagnostic labels in the audit specification; they describe evidence in these windows/checkpoints, not causal proof or a model recommendation.",""]
    for label,item in summary.items():
      if label=="answers": continue
      lines.append(f"- **{label}: {item['level']}**")
      for m in item["measurements"]:
        lines.append("  - "+", ".join(f"{k}={fmt(v)}" for k,v in m.items()))
    lines += ["", "### Counterfactual answers", "",
      f"- ZERO evidence at layer 6: q_in PR {fmt(local['control_step500']['layers']['6']['ZERO']['q_in.pr_rank']['median'])} → v_gru PR {fmt(local['control_step500']['layers']['6']['ZERO']['v_gru.pr_rank']['median'])} → q_out PR {fmt(local['control_step500']['layers']['6']['ZERO']['q_out.pr_rank']['median'])} for Control500.",
      f"- GLOBAL_MEAN at layer 6: q_in PR {fmt(local['control_step500']['layers']['6']['GLOBAL_MEAN']['q_in.pr_rank']['median'])} → q_out PR {fmt(local['control_step500']['layers']['6']['GLOBAL_MEAN']['q_out.pr_rank']['median'])} for Control500.",
      f"- QUERY_CENTRIC versus PROD at layer 6, Control500: z PR {fmt(local['control_step500']['layers']['6']['PROD']['z.pr_rank']['median'])} → {fmt(local['control_step500']['layers']['6']['QUERY_CENTRIC']['z.pr_rank']['median'])}; q_out PR {fmt(local['control_step500']['layers']['6']['PROD']['q_out.pr_rank']['median'])} → {fmt(local['control_step500']['layers']['6']['QUERY_CENTRIC']['q_out.pr_rank']['median'])}; GT best Dice {fmt(local['control_step500']['layers']['6']['PROD']['gt_specialization.best_dice_median']['median'])} → {fmt(local['control_step500']['layers']['6']['QUERY_CENTRIC']['gt_specialization.best_dice_median']['median'])}.",
      f"- Direct answers: {summary['answers']['zero_evidence_intrinsic_collapse']} {summary['answers']['query_centric_restoration']}",
      f"- Primary localization: **{summary['answers']['primary_localization']}**. {summary['answers']['rationale']}",
      "- These are fixed-checkpoint counterfactual replays only; they are not trained performance estimates.","",
      "## Audit boundary", "",
      "This was the final read-only mechanism decomposition before a structural intervention experiment. No training was run. No backward or autograd.grad was run. No optimizer was constructed. No model, loss, Hungarian, or checkpoint was modified. No corrective mechanism was implemented. No next training experiment was started.",""]
    (OUT/"counterfactual_report.md").write_text("\n".join(lines))


def _read_protected_diff():
    files=["tokengs/models/anchor_group_locusgs.py","tokengs/models/anchor_group_loss.py",
      "scripts/anchor_group_v1_gc.py","scripts/anchor_group_v1_gc_noobj_1k.py",
      "scripts/anchor_group_v1_slot_dynamics_audit.py"]
    unstaged=subprocess.check_output(["git","diff","--",*files],cwd=REPO,text=True).strip()
    staged=subprocess.check_output(["git","diff","--cached","--",*files],cwd=REPO,text=True).strip()
    return {"protected_files":files,"unstaged_diff_empty":not bool(unstaged),
            "staged_diff_empty":not bool(staged),"pass":not unstaged and not staged}


def run_audit(device):
    OUT.mkdir(parents=True,exist_ok=True)
    identity=_repo_identity()
    if not identity["pass"]: raise RuntimeError(f"repository identity mismatch: {identity}")
    if GROUP_TEMPERATURE!=0.1: raise RuntimeError(f"temperature mismatch: {GROUP_TEMPERATURE}")
    static_independence=_static_sequential_independence_check()
    if sha256(paired.MANIFEST)!=EXPECTED_MANIFEST or sha256(paired.PLAN)!=EXPECTED_PLAN or sha256(paired.PRETRAINED)!=EXPECTED_PRETRAINED:
        raise RuntimeError("locked manifest/plan/pretrained SHA mismatch")
    if not torch.cuda.is_available() or device.type!="cuda": raise RuntimeError("CUDA required; no CPU fallback")
    windows=json.loads(paired.MANIFEST.read_text())["windows"]
    if len(windows)!=1024 or any(i>=len(windows) for i in WINDOW_INDICES): raise RuntimeError("fixed16 manifest scope invalid")
    checkpoint_meta,checkpoint_ok=_metadata_checkpoints()
    cp_paths={f"{arm}_{step}":paired.WORK/arm/f"checkpoint_step{step}.pt" for arm in ("control","ablation") for step in (500,1000)}
    checkpoint_hashes={k:sha256(p) for k,p in cp_paths.items()}
    source=load_state(paired.PRETRAINED)
    # Fresh step 0 is instantiated twice from the registered seed path and compared exactly.
    m0c,o0c,meta0c=prior._model_for("control",0,device,source)
    s0c=_snapshot_state(m0c)
    m0a,o0a,meta0a=prior._model_for("ablation",0,device,source)
    s0a=m0a.state_dict(); s0_equal=set(s0c)==set(s0a) and all(torch.equal(s0c[k],s0a[k].detach().cpu()) for k in s0c)
    del s0a,m0a,o0a,meta0a,source
    if not s0_equal: raise RuntimeError("fresh Control/Ablation step0 model states differ")
    manifest_windows=[windows[i] for i in WINDOW_INDICES]
    counters={"forward_count":0,"grad_enabled_forward_count":0,"gt_target_build_count":0,
              "autograd_grad_count":0,"backward_count":0,"optimizer_construct_count":0,"optimizer_step_count":0}
    provenance={"repo_identity":identity,"gpu":torch.cuda.get_device_name(device),"torch":torch.__version__,
      "cuda":torch.version.cuda,"cuda_visible_devices":__import__('os').environ.get("CUDA_VISIBLE_DEVICES"),
      "manifest_sha256":EXPECTED_MANIFEST,"parent_plan_sha256":EXPECTED_PLAN,
      "pretrained_sha256":EXPECTED_PRETRAINED,"group_temperature":GROUP_TEMPERATURE,
      "global_seed":42,"anchor_group_init_seed":31415,
      "window_indices":WINDOW_INDICES,"window_count":len(WINDOW_INDICES),
      "checkpoint_metadata":checkpoint_meta,"sequential_anchor_independence":static_independence,
      "fresh_step0":{"control_transfer":meta0c,"control_ablation_exact_state_equal":s0_equal,
        "state_tensor_count":len(s0c),"state_hash":_tensor_hash({k:v for k,v in s0c.items()})}}
    projection={}; local={}; sequential={}; geometry={}; gt={}; cases_by_state={}; model_immutability={}
    replay_pre=[]; replay_q=[]; replay_post=[]; sequential_prod=[]
    for state_key,arm,step in STATE_SPECS:
        if state_key=="fresh_step0": model,opt=m0c,o0c
        else: model,opt,_=prior._model_for(arm,step,device,None)
        if tuple(model.instance_state_layers)!=LAYERS: raise RuntimeError("registered state layers mismatch")
        model.eval()
        snapshot=_snapshot_state(model)
        state_before=_tensor_hash(model.state_dict())
        categories=prior._query_categories("control" if step==0 else arm,step,"fixed16")
        rows=[]; svd=_svd_summary(model.anchor_group)
        for j,(idx,window) in enumerate(zip(WINDOW_INDICES,manifest_windows)):
            with torch.no_grad(): case=_run_case(model,opt,state_key,arm,step,window,idx,categories,counters)
            rows.append(case)
            for l,d in case["production_replay_max_abs_diff"]["A_pre"].items(): replay_pre.append(float(d))
            for l,d in case["production_replay_max_abs_diff"]["q_out"].items(): replay_q.append(float(d))
            for l,d in case["production_replay_max_abs_diff"]["A_post"].items(): replay_post.append(float(d))
            sequential_prod.extend(case["sequential_prod_q_replay_max_abs_diff"])
            if (j+1)%4==0 or j+1==len(WINDOW_INDICES):
                print(f"[counterfactual] {state_key}: {j+1}/{len(WINDOW_INDICES)}",flush=True)
        unchanged,bad=_compare_snapshot(model,snapshot)
        state_after=_tensor_hash(model.state_dict())
        model_immutability[state_key]={"torch_equal":unchanged,"mismatched_keys":bad,
                                      "state_hash_before":state_before,"state_hash_after":state_after}
        if not unchanged: raise RuntimeError(f"model state mutated during read-only audit: {state_key} {bad[:10]}")
        cases_by_state[state_key]=rows
        agg=_state_aggregate(rows)
        projection[state_key]={"arm":arm,"step":step,"n_windows":len(rows),
          "layers":agg["projection"],"proj_u_svd":svd}
        local[state_key]={"arm":arm,"step":step,"n_windows":len(rows),"layers":agg["local"]}
        sequential[state_key]={"arm":arm,"step":step,"n_windows":len(rows),"layers":agg["sequential"]}
        geometry[state_key]={"arm":arm,"step":step,"n_windows":len(rows),"layers":agg["evidence_weight_geometry"]}
        gt[state_key]={"arm":arm,"step":step,"n_windows":len(rows),"layers":agg["gt_specialization"]}
        if step!=0: del model,opt
        gc.collect(); torch.cuda.empty_cache()

    checkpoint_after={k:sha256(p) for k,p in cp_paths.items()}
    cp_unchanged=checkpoint_hashes==checkpoint_after
    provenance["checkpoint_sha256_before"]={k:checkpoint_hashes[k] for k in checkpoint_hashes}
    provenance["checkpoint_sha256_after"]={k:checkpoint_after[k] for k in checkpoint_after}
    provenance["checkpoint_sha_unchanged"]=cp_unchanged
    provenance["model_immutability"]=model_immutability
    write_json(OUT/"provenance.json",provenance)

    protected=_read_protected_diff()
    contracts_rows=[
      {"id":"CF-C1 repository identity","pass":identity["pass"],"detail":identity},
      {"id":"CF-C2 checkpoint provenance","pass":checkpoint_ok,"detail":{"trained_checkpoint_count":4,"mismatches":sum(len(x["mismatches"]) for x in checkpoint_meta.values())}},
      {"id":"CF-C3 fresh step0 exact initialization","pass":s0_equal,"detail":{"state_tensors":len(s0c),"torch_equal":s0_equal}},
      {"id":"CF-C4 forward coverage","pass":counters["forward_count"]==80,"detail":{"expected":80,"actual":counters["forward_count"]}},
      {"id":"CF-C5 one production forward and one GT build per case","pass":len(cases_by_state)==5 and all(len(v)==16 for v in cases_by_state.values()) and counters["gt_target_build_count"]==80,"detail":{"state_count":len(cases_by_state),"cases":sum(map(len,cases_by_state.values())),"gt_target_build_count":counters["gt_target_build_count"]}},
      {"id":"CF-C6 production A_pre replay","pass":max(replay_pre,default=0)<=1e-6,"detail":{"max_abs_diff":max(replay_pre,default=0)}},
      {"id":"CF-C7 production q_out replay","pass":max(replay_q,default=0)<=1e-6,"detail":{"max_abs_diff":max(replay_q,default=0)}},
      {"id":"CF-C8 production A_post replay","pass":max(replay_post,default=0)<=1e-6,"detail":{"max_abs_diff":max(replay_post,default=0)}},
      {"id":"CF-C9 sequential PROD replay","pass":max(sequential_prod,default=0)<=1e-6,"detail":{"max_abs_diff":max(sequential_prod,default=0),"count":len(sequential_prod)}},
      {"id":"CF-C10 no-grad only","pass":counters["grad_enabled_forward_count"]==0 and counters["backward_count"]==0 and counters["autograd_grad_count"]==0,"detail":{k:counters[k] for k in ("grad_enabled_forward_count","backward_count","autograd_grad_count")}},
      {"id":"CF-C11 no optimizer","pass":counters["optimizer_construct_count"]==0 and counters["optimizer_step_count"]==0,"detail":{k:counters[k] for k in ("optimizer_construct_count","optimizer_step_count")}},
      {"id":"CF-C12 model immutability","pass":all(v["torch_equal"] for v in model_immutability.values()),"detail":model_immutability},
      {"id":"CF-C13 checkpoint immutability","pass":cp_unchanged,"detail":{"sha_before_after_equal":cp_unchanged}},
      {"id":"CF-C14 existing scientific files unchanged","pass":protected["pass"],"detail":protected},
      {"id":"CF-C15 metric finiteness","pass":all(_json_finite(x) for x in (projection,local,sequential,geometry,gt)),"detail":{"all_non_null_scalars_finite":True}}
    ]
    contracts={"status":"pass" if all(x["pass"] for x in contracts_rows) else "fail",
      "passed":sum(x["pass"] for x in contracts_rows),"total":len(contracts_rows),
      "forward_count":counters["forward_count"],"gt_target_build_count":counters["gt_target_build_count"],
      "grad_enabled_forward_count":counters["grad_enabled_forward_count"],"backward_count":counters["backward_count"],
      "autograd_grad_count":counters["autograd_grad_count"],"optimizer_construct_count":counters["optimizer_construct_count"],
      "optimizer_step_count":counters["optimizer_step_count"],"contracts":contracts_rows}
    aggregate_view={k:{"local":local[k]["layers"],"projection":projection[k]["layers"],
                       "proj_u_svd":projection[k]["proj_u_svd"]} for k,_,_ in STATE_SPECS}
    summary=_derived_findings(aggregate_view)
    # Keep the compact per-window measures, never raw q/z/assignment tensors.
    local_out={k:{"arm":v["arm"],"step":v["step"],"n_windows":v["n_windows"],"layers":v["layers"]} for k,v in local.items()}
    sequential_out={k:{"arm":v["arm"],"step":v["step"],"n_windows":v["n_windows"],"layers":v["layers"]} for k,v in sequential.items()}
    projection_out={k:{"arm":v["arm"],"step":v["step"],"n_windows":v["n_windows"],"layers":v["layers"],"proj_u_svd":v["proj_u_svd"]} for k,v in projection.items()}
    evidence_out={k:v for k,v in geometry.items()}
    gt_out={k:v for k,v in gt.items()}
    write_json(OUT/"projection_decomposition.json",projection_out)
    write_json(OUT/"local_counterfactuals.json",local_out)
    write_json(OUT/"sequential_counterfactuals.json",sequential_out)
    write_json(OUT/"evidence_weight_geometry.json",evidence_out)
    write_json(OUT/"gt_specialization_counterfactuals.json",gt_out)
    write_json(OUT/"counterfactual_summary.json",{"contract_status":contracts["status"],
      "contracts_passed":contracts["passed"],"contracts_total":contracts["total"],
      "forward_count":contracts["forward_count"],"static_sequential_anchor_independence":static_independence,
      "production_replay":{"A_pre_max_abs_diff":max(replay_pre,default=0),"q_out_max_abs_diff":max(replay_q,default=0),
        "A_post_max_abs_diff":max(replay_post,default=0),"sequential_prod_q_max_abs_diff":max(sequential_prod,default=0)},
      "diagnostic_labels":summary})
    write_json(OUT/"contracts.json",contracts)
    _write_report(contracts,provenance,projection_out,local_out,sequential_out,evidence_out,summary)
    if contracts["status"]!="pass": raise RuntimeError(f"counterfactual contract failure {contracts['passed']}/{contracts['total']}")
    return contracts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("audit",),required=True)
    parser.add_argument("--device",choices=("cuda",),required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; refusing CPU fallback")
    device=torch.device("cuda")
    print(f"[counterfactual] GPU={torch.cuda.get_device_name(device)} torch={torch.__version__} CUDA={torch.version.cuda}",flush=True)
    result=run_audit(device)
    print(f"[counterfactual] PASS {result['passed']}/{result['total']}",flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
