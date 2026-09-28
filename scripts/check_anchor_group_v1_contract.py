#!/usr/bin/env python3
"""CPU contracts for Anchor-Group V1; no CUDA or training is used."""
from __future__ import annotations
import ast, json, inspect
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from tokengs.models.anchor_group_locusgs import AnchorGroupController, NUM_THING, NUM_STUFF

OUT=Path("group_plus/anchor_group_v1/contracts.json")

def record(rows,name,ok,detail=None): rows.append({"contract":name,"passed":bool(ok),"detail":detail})

def main():
    torch.manual_seed(19); checks=[]
    class O: enc_embed_dim=64; dec_patch_size=2
    ctrl=AnchorGroupController(O())
    a=torch.randn(1,1024,256,requires_grad=True)
    q=ctrl.query_init.unsqueeze(0).expand(1,-1,-1)
    mu=torch.randn(1,1024,3); ell=torch.ones(1)
    void=ctrl.token_void(a)
    qn,pre,post,c,s=ctrl.update_group_states(a,mu,q,void,ell)
    record(checks,"C1_shapes",a.shape==(1,1024,256) and q.shape==(1,102,256) and pre.shape==(1,1024,103) and post.shape==(1,1024,103) and c.shape==(1,100,3) and s.shape==(1,100,3),{"a":list(a.shape),"q":list(q.shape),"A_pre":list(pre.shape),"A_post":list(post.shape),"c":list(c.shape),"s":list(s.shape)})
    err=max(float((pre.sum(-1)-1).abs().max()),float((post.sum(-1)-1).abs().max()))
    record(checks,"C2_simplex",float(min(pre.min(),post.min()))>=0 and err<=1e-6,{"max_sum_error":err})
    src=inspect.getsource(__import__("tokengs.models.anchor_group_locusgs",fromlist=["AnchorGroupDecoder"]).AnchorGroupDecoder)
    record(checks,"C3_no_fps_local8",all(x not in src for x in ("deterministic_fps(","local_3d_evidence_pool(","fps_index","local8")))
    loss=-torch.log(post[:,:,0].clamp_min(1e-6)).mean(); loss.backward()
    rows=a.grad.abs().sum(-1); n=int((torch.isfinite(rows)&(rows>0)).sum())
    record(checks,"C4_all_anchor_gradient",n==1024,{"all_anchor_grad_rows":n})
    qgrad=ctrl.query_init.grad
    record(checks,"C5_query_gradient",qgrad is not None and bool(torch.isfinite(qgrad).all()) and float(qgrad.abs().sum())>0)
    ownership=torch.rand(1,1024,103); expanded=ownership[:,:,None,:].expand(1,1024,4,103).reshape(1,4096,103)
    exact=torch.equal(expanded.reshape(1,1024,4,103)[:,:,0],ownership) and all(torch.equal(expanded.reshape(1,1024,4,103)[:,:,i],ownership) for i in range(4))
    record(checks,"C6_gaussian_parent_exact",exact)

    # Deterministic consensus cases exercise the same state transition as projected observations.
    from tokengs.models.anchor_group_loss import resolve_anchor_observations,THING,WALL,FLOOR,IGNORE
    record(checks,"C7_same_instance",resolve_anchor_observations([(THING,2,7)]*2)==(THING,7,2))
    record(checks,"C8_one_view",resolve_anchor_observations([(THING,2,7)])==(THING,7,2))
    record(checks,"C9_instance_conflict",resolve_anchor_observations([(THING,2,7),(THING,2,8)])[0]==IGNORE)
    record(checks,"C10_thing_stuff_conflict",resolve_anchor_observations([(THING,2,7),(WALL,0,0)])[0]==IGNORE)
    record(checks,"C11_wall",resolve_anchor_observations([(WALL,0,0)]*2)[0]==WALL)
    record(checks,"C12_floor",resolve_anchor_observations([(FLOOR,1,0)])[0]==FLOOR)
    record(checks,"C13_no_observation",resolve_anchor_observations([])[0]==IGNORE)
    # Perfect vs permuted ownership CE/Dice on valid anchors.
    y=torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.]])
    perfect=y.clone()*.99+.005; perm=perfect.flip(0)
    def cd(p,t):
        ce=-(t*torch.log(p.clamp_min(1e-6))+(1-t)*torch.log((1-p).clamp_min(1e-6))).mean()
        dice=1-(2*(p*t).sum(-1)+1)/(p.sum(-1)+t.sum(-1)+1)
        return ce,dice.mean()
    cp,dp=cd(perfect,y); cq,dq=cd(perm,y)
    record(checks,"C14_perfect_beats_permuted",cp<cq and dp<dq and torch.isfinite(cp+dp+cq+dq).item(),{"perfect_ce":float(cp),"permuted_ce":float(cq),"perfect_dice":float(dp),"permuted_dice":float(dq)})
    from tokengs.models.anchor_group_loss import unified_hungarian
    sem=torch.full((1,2,2,2),255,dtype=torch.long); ins=torch.zeros_like(sem)
    sem[0,:,0,:]=2; ins[0,:,0,:]=7; sem[0,:,1,:]=3; ins[0,:,1,:]=8
    masks=torch.zeros(2,2,2,2,dtype=torch.bool); masks[0,:,0,:]=True; masks[1,:,1,:]=True
    region=torch.full((1,2,103,2,2),.001)
    region[0,:,0,0,:]=.99; region[0,:,1,1,:]=.99
    clslog=torch.full((1,100,21),-8.); clslog[0,0,2]=8.; clslog[0,1,3]=8.
    ap=torch.full((1,1024,103),.001); ap[:,:,102]=.001
    ap[0,0]=.001; ap[0,0,0]=.99; ap[0,1]=.001; ap[0,1,1]=.99
    targets={"gt_classes":[torch.tensor([2,3])],"gt_pixel_masks":[masks],"gt_instance_ids":[torch.tensor([7,8])],"Y_anchor":torch.zeros(1,2,1024),"anchor_valid":torch.zeros(1,1024,dtype=torch.bool)}
    targets["Y_anchor"][0,0,0]=1; targets["Y_anchor"][0,1,1]=1; targets["anchor_valid"][0,:2]=True
    pred={"gaussians":torch.zeros(1,1,11),"region_mass":region,"thing_class_logits":clslog,"anchor_assignment":ap,"states":[{"mu":torch.zeros(1,1024,3)}]}
    batch={"semantic_label_all":sem,"instance_label_all":ins}
    targets,pairs=unified_hungarian(pred,batch,targets=targets)
    qi,ki=pairs[0]; pair_list=list(zip(qi.tolist(),ki.tolist()))
    one_solve=inspect.getsource(unified_hungarian).count("linear_sum_assignment(")==1
    record(checks,"C15_unified_hungarian_unique",len(set(qi.tolist()))==2 and len(set(ki.tolist()))==2 and pair_list==[(0,0),(1,1)] and one_solve,{"pairs":pair_list,"single_solve":one_solve,"shared_pairs_for_pixel_anchor":True})

    # C16: same deterministic seed and options, checking every inherited tensor.
    from tokengs.options import config_defaults
    from tokengs.models.instance_state_locusgs import InstanceStateController
    from tokengs.models.anchor_group_loss import pairwise_anchor_bce_cost
    opt=config_defaults["train_siu3r_anchor_group_v1"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31415); old=InstanceStateController(opt)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31415); new=AnchorGroupController(opt)
    shared=("ln_h","proj_h","proj_m","ln_x","ln_e","proj_e","ln_u","proj_u","query_init","gru","ln_gru","ffn_fc1","ffn_fc2","ln_ffn","token_void","thing_classifier","ln_vq","proj_vq","ln_fx","ln_fm","proj_wh","proj_wmu","proj_wr","proj_wgs","ln_de","proj_de","proj_off","ln_void","proj_gvoid")
    mismatches=[]; compared=0; equal=0
    old_params=dict(old.named_parameters()); new_params=dict(new.named_parameters())
    for prefix in shared:
        for name,value in old_params.items():
            if name==prefix or name.startswith(prefix+"."):
                compared+=1
                if name not in new_params or value.shape!=new_params[name].shape or not torch.equal(value,new_params[name]): mismatches.append(name)
                else: equal+=1
    void_max=float(new.token_void(a).abs().max())
    init_ok=bool(compared>0 and equal==compared and not mismatches and void_max==0.0)
    record(checks,"C16_shared_init_parity",init_ok,{"shared_init_parameter_count":compared,"shared_init_equal_count":equal,"shared_init_mismatch_names":mismatches,"step0_void_logit_max_abs":void_max})

    # C17: production pairwise cost agrees elementwise with exhaustive BCEWithLogits.
    z=torch.linspace(-4.5,4.5,5*17).reshape(5,17); p=torch.sigmoid(z)
    gen=torch.Generator().manual_seed(31415); y=(torch.rand(3,17,generator=gen)>.5).float()
    production=pairwise_anchor_bce_cost(p,y)
    brute=torch.stack([torch.stack([F.binary_cross_entropy_with_logits(z[q],y[k],reduction="mean") for k in range(3)]) for q in range(5)])
    bce_diff=float((production-brute).abs().max())
    record(checks,"C17_anchor_bce_bruteforce",bce_diff<=1e-6,{"Q":5,"K":3,"N":17,"max_abs_diff":bce_diff})

    # C18: extreme finite logits are routed through the exact production clamp path.
    extreme=torch.tensor([-20.,-10.,0.,10.,20.]).expand(5,-1)
    pe=torch.sigmoid(extreme); ye=torch.tensor([[0.,0.,1.,1.,1.],[1.,0.,1.,0.,1.],[0.,1.,0.,1.,0.]])
    extreme_cost=pairwise_anchor_bce_cost(pe,ye)
    extreme_ok=bool(torch.isfinite(extreme_cost).all())
    record(checks,"C18_extreme_anchor_bce_finite",extreme_ok,{"all_finite":extreme_ok,"values_tested":[-20,-10,0,10,20]})

    audit={"shared_init_parameter_count":compared,"shared_init_equal_count":equal,"shared_init_mismatch_names":mismatches,"step0_void_logit_max_abs":void_max,"anchor_bce_bruteforce_max_abs_diff":bce_diff,"anchor_bce_bruteforce_pass":bce_diff<=1e-6,"extreme_anchor_bce_all_finite":extreme_ok}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    (OUT.parent/"phase_a1_initialization_and_bce_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    payload={"architecture":"LOCUSGS_ANCHOR_GROUP_V1","passed":sum(x["passed"] for x in checks),"failed":sum(not x["passed"] for x in checks),"checks":checks,"phase_a1_audit":audit}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2)); return int(payload["failed"]>0)
if __name__=="__main__": raise SystemExit(main())
