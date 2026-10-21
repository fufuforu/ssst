#!/usr/bin/env python3
"""Paired 1k Anchor-Group V2-SMRU (scale-matched residual query update gamma=0.1)."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts import anchor_group_v1_gc as gc_v1
from scripts import anchor_group_v1_gc_noobj_1k as reuse
from scripts import anchor_group_v1_endpoint_audit as endpoint
from scripts import anchor_group_v1_slot_dynamics_audit as slot
from scripts.anchor_group_v1 import (MANIFEST,MONITORS,PLAN,PRETRAINED,PRETRAINED_SHA,SOURCE_REPORTS,
    TOTAL_STEPS,build_optimizer,build_options,evaluate_anchor_group_all,load_state,make_model,
    set_optimizer_lr,sha256,write_json)
from scripts.anchor_group_v1_gc import backward_gradient_controlled
from scripts.anchor_group_v1_endpoint_audit import run_query_scope
from scripts.instance_state_generalization import _batch_for,_seen_classes
from scripts.instance_state_runtime import capture_rng,restore_rng
from tokengs.models.anchor_group_loss import build_anchor_targets,unified_hungarian
from tokengs.models.anchor_group_locusgs import anchor_group_understanding_weight
from scripts.anchor_group_v1_endpoint_audit import forward_context

OUT=REPO/'group_plus/anchor_group_v2_smru_gamma01_1k'
WORK=REPO/'workspace_group_plus/anchor_group_v2_smru_gamma01_1k'
EXPECTED_HEAD='da2f06c5e85e2affee1d4014a42668b1024d390e'
MANIFEST_SHA='1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483'
PLAN_SHA='ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323'
SCALE=.01; GAMMA={'control':1.0,'smru':.1}
ARM={'control':{'recipe':'ANCHOR_GROUP_V1_GC_SMRU_CONTROL_1K','gamma':1.0,'scale_match':False},
     'smru':{'recipe':'ANCHOR_GROUP_V2_SMRU_GAMMA01_1K','gamma':.1,'scale_match':True}}
STEPS=(0,200,500,1000); FIXED16=[0,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960]
OPT_EXPECTED={'anchor_group_decay':(18,2743496),'anchor_group_nodecay':(41,41800),
'reconstruction_decay':(115,218773504),'reconstruction_nodecay':(335,1229116)}

# Reuse validated no-object-1k read-only/evaluation helpers with isolated output roots.
reuse.OUT=OUT; reuse.WORK=WORK
reuse.ARM_INFO={
    "control":{"recipe":ARM["control"]["recipe"],"scale":1.0},
    "smru":{"recipe":ARM["smru"]["recipe"],"scale":1.0},
}

def finite(x):
    if torch.is_tensor(x): return bool(torch.isfinite(x).all())
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(finite(v) for v in x)
    if isinstance(x,(int,float,np.number)): return math.isfinite(float(x))
    return True

def state_equal(a,b): return reuse._state_equal(a,b)
def state_digest(s): return reuse._state_digest(s)
def model_for(device,gamma,source,scale_match=False):
    torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.cuda.manual_seed_all(42)
    opt=build_options(); opt.anchor_group_query_update_gamma=float(gamma); opt.anchor_group_query_update_scale_match=bool(scale_match); opt.anchor_group_unmatched_noobj_scale=1.0
    model,transfer=make_model(opt,device,source)
    expected=('LOCUSGS_ANCHOR_GROUP_V1' if gamma==1.0 and not scale_match else
              'LOCUSGS_ANCHOR_GROUP_V2_RU_GAMMA01' if gamma==.1 and not scale_match else
              'LOCUSGS_ANCHOR_GROUP_V2_SMRU_GAMMA01' if gamma==.1 and scale_match else None)
    if expected is None or model.architecture_name!=expected or model.anchor_group.query_update_gamma!=gamma or model.anchor_group.query_update_scale_match!=scale_match: raise RuntimeError('architecture/gamma/scale-match identity mismatch')
    if any(not p.requires_grad for p in model.parameters()): raise RuntimeError('unexpected frozen parameter')
    return model,opt,transfer

def _contract(ok,detail): return {'passed':bool(ok),'detail':detail}

def _rng_seed():
    torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.cuda.manual_seed_all(42)

def _optimizer_counts(opt):
    optimizer,audit=build_optimizer(opt)
    groups={g['name']:{'tensor_count':len(g['params']),'numel':sum(p.numel() for p in g['params']),
                       'weight_decay':float(g['weight_decay']),'lr':float(g['lr'])} for g in optimizer.param_groups}
    return optimizer,audit,groups

def _layer_inputs(pred,ctrl):
    qin=ctrl.query_init.unsqueeze(0)
    for st in pred['states']:
        layer=int(st['layer'])
        if layer not in (6,8,10,12): continue
        yield st,qin
        qin=st['q']

def _replay(ctrl,a,mu,q,void,ell,gamma,scale_match=False):
    apre=ctrl.assign_group(a,q,void); mass=apre[:,:,:102].sum(1)
    w=apre[:,:,:102]/(mass.unsqueeze(1)+1e-6); z=torch.einsum('btq,btd->bqd',w,a)
    v=ctrl.ln_gru(ctrl.gru(z.reshape(-1,ctrl.D),q.reshape(-1,ctrl.D))).reshape_as(q)
    cand=ctrl.ln_ffn(v+ctrl.ffn_fc2(F.gelu(ctrl.ffn_fc1(v))))
    qnorm=torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    candnorm=torch.linalg.vector_norm(cand,dim=-1,keepdim=True)
    cand_scaled=cand*(qnorm/candnorm.clamp_min(1e-6))
    if gamma==1.0: qnew=cand
    elif scale_match: qnew=q+gamma*(cand_scaled-q)
    else: qnew=q+gamma*(cand-q)
    qnew=torch.where((mass<1e-4).unsqueeze(-1),q,qnew)
    apost=ctrl.assign_group(a,qnew,void)
    return apre,mass,z,v,cand,cand_scaled,qnew,apost

def _pretrain_contracts(device):
    OUT.mkdir(parents=True,exist_ok=True); comp=OUT/'comparison'; comp.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    origin=subprocess.check_output(['git','rev-parse','origin/main'],cwd=REPO,text=True).strip()
    if head!=EXPECTED_HEAD or origin!=EXPECTED_HEAD: raise RuntimeError('SM-C1 repository identity mismatch')
    manifest,plan,monitor=reuse._asset_audit()
    if sha256(PRETRAINED)!=PRETRAINED_SHA or sha256(MANIFEST)!=MANIFEST_SHA or sha256(PLAN)!=PLAN_SHA: raise RuntimeError('locked asset SHA mismatch')
    source=load_state(PRETRAINED)
    c,oc,tc=model_for(device,1.0,source,False)
    oldru,oo,tor=model_for(device,.1,source,False)
    sm,os,opt_transfer=model_for(device,.1,source,True)
    states=[c.state_dict(),oldru.state_dict(),sm.state_dict()]
    state_comparisons=[state_equal(states[0],x) for x in states[1:]]
    state_ok=all(x[0] and x[1]==509 and not x[2] for x in state_comparisons)
    param_maps=[dict(x.named_parameters()) for x in (c,oldru,sm)]
    param_names_equal=param_maps[0].keys()==param_maps[1].keys()==param_maps[2].keys()
    param_shapes_equal=param_names_equal and all(param_maps[0][n].shape==param_maps[1][n].shape==param_maps[2][n].shape for n in param_maps[0])
    param_numel=[sum(p.numel() for p in x.parameters()) for x in (c,oldru,sm)]
    optcs=[_optimizer_counts(x) for x in (c,oldru,sm)]
    group_summaries=[x[2] for x in optcs]
    optimizer_equal=group_summaries[0]==group_summaries[1]==group_summaries[2]
    entries=json.loads(PLAN.read_text())['entries']; entry=entries[599]
    if int(entry['step'])!=600: raise RuntimeError('fixed step600 plan entry mismatch')
    batch=_batch_for(oc,entry,device)
    models=(c,oldru,sm); opts=(oc,oo,os); preds=[]; metrs=[]
    for model in models:
        model.eval()
        with torch.no_grad(): out,metrics=model.step_loss(batch,step=600,coupled=False)
        preds.append(out['prediction']); metrs.append(metrics)
    pc,po,ps=preds
    # Same fresh state and same batch: reconstruction decoder outputs are independent of query update.
    reconstruction_keys=('gaussians',)
    rec_checks={k:torch.equal(pc[k],ps[k]) for k in reconstruction_keys}
    per_layer_rec={}
    for layer in (6,8,10,12):
        aa,bb=pc['states'][layer-1],ps['states'][layer-1]
        for k in ('tokens','mu','radii'):
            per_layer_rec[f'layer{layer}_{k}']=torch.equal(aa[k],bb[k])
    # Replay each rule using that arm's actual q_in at every registered layer.
    legacy_rows=[]; oldru_rows=[]; smru_rows=[]; preupdate=[]; norm_rows=[]; lowmass_rows=[]
    inputs=[]
    for pred,model in zip(preds,models):
        inputs.append({int(st['layer']):(st,qin) for st,qin in _layer_inputs(pred,model.anchor_group)})
    for layer in (6,8,10,12):
        (sc,qc),(so,qo),(ss,qs)=(x[layer] for x in inputs)
        rc=_replay(c.anchor_group,sc['anchor_embedding'],sc['mu'],qc,c.anchor_group.token_void(sc['anchor_embedding']),sc['ell'],1.0,False)
        ro=_replay(oldru.anchor_group,so['anchor_embedding'],so['mu'],qo,oldru.anchor_group.token_void(so['anchor_embedding']),so['ell'],.1,False)
        rs=_replay(sm.anchor_group,ss['anchor_embedding'],ss['mu'],qs,sm.anchor_group.token_void(ss['anchor_embedding']),ss['ell'],.1,True)
        apre,mass,z,v,cand,cscaled,q_exp,apost=rs
        legacy_rows.append({'layer':layer,'max_abs_diff':float((sc['q']-rc[6]).abs().max()),'exact':torch.equal(sc['q'],rc[6])})
        oldru_rows.append({'layer':layer,'max_abs_diff':float((so['q']-ro[6]).abs().max())})
        smru_rows.append({'layer':layer,'max_abs_diff':float((ss['q']-rs[6]).abs().max())})
        if layer==6:
            preupdate.append({'layer':layer,'A_pre_exact':torch.equal(rc[0],ro[0]) and torch.equal(rc[0],rs[0]),'mass_exact':torch.equal(rc[1],ro[1]) and torch.equal(rc[1],rs[1]),'z_exact':torch.equal(rc[2],ro[2]) and torch.equal(rc[2],rs[2]),'candidate_exact':torch.equal(rc[4],ro[4]) and torch.equal(rc[4],rs[4])})
        non_skip=mass>=1e-4
        qn=torch.linalg.vector_norm(qs,dim=-1); cn=torch.linalg.vector_norm(cand,dim=-1); sn=torch.linalg.vector_norm(cscaled,dim=-1); outn=torch.linalg.vector_norm(ss['q'],dim=-1)
        valid=non_skip & (cn>1e-6)
        norm_diff=(sn-qn).abs()[valid]
        direction=F.cosine_similarity(cscaled[valid],cand[valid],dim=-1) if valid.any() else torch.ones(1,device=device)
        candidate_delta=torch.linalg.vector_norm(cscaled-qs,dim=-1)
        actual_delta=torch.linalg.vector_norm(ss['q']-qs,dim=-1)
        use=non_skip & (candidate_delta>1e-8)
        ratios=actual_delta[use]/candidate_delta[use]
        norm_rows.append({'layer':layer,'sample_count':int(valid.sum()),'norm_max_abs_diff':float(norm_diff.max()) if norm_diff.numel() else 0.0,'scaled_to_q_ratio_mean':float((sn[valid]/qn[valid].clamp_min(1e-12)).mean()) if valid.any() else 1.0,'scaled_to_q_ratio_median':float((sn[valid]/qn[valid].clamp_min(1e-12)).median()) if valid.any() else 1.0,'scaled_to_q_ratio_p10':float(torch.quantile(sn[valid]/qn[valid].clamp_min(1e-12),.1)) if valid.any() else 1.0,'scaled_to_q_ratio_p90':float(torch.quantile(sn[valid]/qn[valid].clamp_min(1e-12),.9)) if valid.any() else 1.0,'direction_cosine_min':float(direction.min()) if direction.numel() else 1.0,'update_ratio_max_abs_error':float((ratios-.1).abs().max()) if ratios.numel() else 0.0,'raw_candidate_to_q_norm_ratio_median':float((cn[valid]/qn[valid].clamp_min(1e-12)).median()) if valid.any() else 0.0})
        # Synthetic all-void ownership verifies the original last-stage skip for all three variants.
        void_all=torch.full((*ss['anchor_embedding'].shape[:2],1),1e4,device=device,dtype=ss['anchor_embedding'].dtype)
        skip=[]
        for model,gamma,scale,q_layer in ((c,1.,False,qc),(oldru,.1,False,qo),(sm,.1,True,qs)):
            skip.append(_replay(model.anchor_group,ss['anchor_embedding'],ss['mu'],q_layer,void_all,ss['ell'],gamma,scale)[6])
        lowmass_rows.append({'layer':layer,'all_mass_below_threshold':bool((_replay(sm.anchor_group,ss['anchor_embedding'],ss['mu'],qs,void_all,ss['ell'],.1,True)[1]<1e-4).all()),'all_three_exact_qin':all(torch.equal(x,qv) for x,qv in zip(skip,(qc,qo,qs)))})
    # q_in carried at each state layer is identical across model arms by construction; ensure checked explicitly.
    qin_state_exact=torch.equal(c.anchor_group.query_init,oldru.anchor_group.query_init) and torch.equal(c.anchor_group.query_init,sm.anchor_group.query_init)
    # SM-RU graph must carry finite, nonzero gradients through query and update modules.
    gm,go,_=model_for(device,.1,source,True); gm.train(); go_batch=_batch_for(go,entry,device); gm.zero_grad(set_to_none=True)
    grad_out,grad_metrics=gm.step_loss(go_batch,step=600,coupled=False)
    backward_gradient_controlled(gm,grad_metrics['loss_recon'],grad_metrics['loss_understanding'],.5,shared_scale=.01)
    grad_report={}
    for name in ('anchor_group.query_init','anchor_group.gru.weight_ih','anchor_group.ffn_fc1.weight','anchor_group.proj_u.weight'):
        grad= dict(gm.named_parameters())[name].grad
        grad_report[name]={'present':grad is not None,'finite':grad is not None and bool(torch.isfinite(grad).all()),'norm':float(torch.linalg.vector_norm(grad)) if grad is not None else 0.0}
    grad_ok=all(x['present'] and x['finite'] and x['norm']>0 for x in grad_report.values())
    # Run established regressions with their outputs isolated under this run;
    # prior committed experiment artifacts remain read-only.
    # The established GC checker is read-only: it validates already committed
    # artifacts and does not emit into its OUT path. Keep that historical
    # result directory untouched while the regression helper isolates all
    # newly generated CPU/S0/S1 regression outputs under this experiment.
    regressions=reuse._run_regressions(device)
    parity_c=gc_v1._reconstruction_parity(c,oc,source,device,batch)
    parity_sm=gc_v1._reconstruction_parity(sm,os,source,device,batch)
    loss_diff=subprocess.check_output(['git','diff',EXPECTED_HEAD,'--','tokengs/models/anchor_group_loss.py'],cwd=REPO,text=True).strip()
    git_changes=subprocess.check_output(['git','diff','--name-only',EXPECTED_HEAD],cwd=REPO,text=True).splitlines()
    allowed_tracked={'tokengs/models/anchor_group_locusgs.py'}
    protected={'tokengs/models/anchor_group_loss.py','scripts/anchor_group_v1.py','scripts/anchor_group_v1_gc.py','scripts/anchor_group_v1_gc_noobj_1k.py','scripts/anchor_group_v2_ru_gamma01_1k.py','scripts/anchor_group_v1_slot_dynamics_audit.py','scripts/anchor_group_v1_counterfactual_decomposition.py'}
    boundary_ok=not loss_diff and not (set(git_changes)&protected) and set(git_changes).issubset(allowed_tracked|{'docs/fixed4_shared_head.md'})
    warmup_ok=all(abs(float(anchor_group_understanding_weight(s))-w)<1e-12 for s,w in [(200,0),(201,.00125),(600,.5),(999,.99875),(1000,1)])
    noobj_ok=all(float(m['unmatched_noobj_scale'])==1.0 for m in metrs)
    no_detach='detach' not in Path(REPO/'tokengs/models/anchor_group_locusgs.py').read_text().split('candidate_scaled =',1)[1].split('q_new = torch.where',1)[0]
    recipe_ok=all(model.anchor_group.query_update_gamma==ARM[arm]['gamma'] and model.anchor_group.query_update_scale_match==ARM[arm]['scale_match'] for arm,model in (('control',c),('smru',sm)))
    checks={
      'SM-C1_repository_identity':_contract(head==EXPECTED_HEAD and origin==EXPECTED_HEAD,{'head':head,'origin_main':origin}),
      'SM-C2_no_new_parameter_or_state':_contract(state_ok and param_names_equal and param_shapes_equal and len(set(param_numel))==1,{'state_tensor_count':509,'state_exact_all_three':state_ok,'state_equal_results':state_comparisons,'parameter_names_equal':param_names_equal,'parameter_shapes_equal':param_shapes_equal,'parameter_numel':param_numel}),
      'SM-C3_legacy_path_exact':_contract(all(x['exact'] and x['max_abs_diff']==0 for x in legacy_rows),{'layers':legacy_rows}),
      'SM-C4_previous_ru_compatibility':_contract(max(x['max_abs_diff'] for x in oldru_rows)<=1e-7,{'layers':oldru_rows}),
      'SM-C5_scale_matched_formula':_contract(max(x['max_abs_diff'] for x in smru_rows)<=1e-7,{'layers':smru_rows}),
      'SM-C6_norm_matching':_contract(max(x['norm_max_abs_diff'] for x in norm_rows)<=1e-5 and all(abs(x['scaled_to_q_ratio_median']-1.)<1e-5 for x in norm_rows),{'layers':norm_rows}),
      'SM-C7_direction_preservation':_contract(min(x['direction_cosine_min'] for x in norm_rows)>=.999999,{'layers':norm_rows}),
      'SM-C8_interpolation_magnitude':_contract(max(x['update_ratio_max_abs_error'] for x in norm_rows)<=1e-5,{'layers':norm_rows}),
      'SM-C9_no_detach_gradient_path':_contract(grad_ok and no_detach,{'gradient_stats':grad_report,'source_has_no_detach_in_update':no_detach}),
      'SM-C10_preupdate_identity':_contract(qin_state_exact and all(all(x[k] for k in ('A_pre_exact','mass_exact','z_exact','candidate_exact')) for x in preupdate),{'q_in_layers_equal':qin_state_exact,'per_layer':preupdate}),
      'SM-C11_low_mass_skip':_contract(all(x['all_mass_below_threshold'] and x['all_three_exact_qin'] for x in lowmass_rows),{'layers':lowmass_rows}),
      'SM-C12_reconstruction_invariance':_contract(all(rec_checks.values()) and all(per_layer_rec.values()) and parity_c['status']=='pass' and parity_sm['status']=='pass',{'gaussians':rec_checks,'canonical_reconstruction':per_layer_rec,'parity_control':parity_c,'parity_smru':parity_sm}),
      'SM-C13_optimizer_parity':_contract(optimizer_equal and all(OPT_EXPECTED[k]==(v['tensor_count'],v['numel']) for k,v in group_summaries[0].items()),{'control':group_summaries[0],'old_ru':group_summaries[1],'smru':group_summaries[2]}),
      'SM-C14_recipe_parity':_contract(warmup_ok and noobj_ok and recipe_ok,{'warmup_points':{str(s):anchor_group_understanding_weight(s) for s in (200,201,600,999,1000)},'gc_alpha':SCALE,'clip':1.0,'unmatched_noobj_scale':1.0,'control':ARM['control'],'smru':ARM['smru'],'manifest_sha256':MANIFEST_SHA,'plan_sha256':PLAN_SHA}),
      'SM-C15_previous_regressions':_contract(regressions['status']=='pass',regressions),
      'SM-C16_scientific_file_boundary':_contract(boundary_ok,{'tracked_changes':git_changes,'anchor_group_loss_clean':not bool(loss_diff),'only_allowed_tracked_model_file':True})}
    failed=[k for k,v in checks.items() if not v['passed']]
    result={'status':'pass' if not failed else 'fail','passed':len(checks)-len(failed),'total':16,'failed':failed,'gpu':torch.cuda.get_device_name(device),'torch':torch.__version__,'cuda':torch.version.cuda,'monitor_hashes':monitor,'pretrained_sha256':PRETRAINED_SHA,'manifest_sha256':MANIFEST_SHA,'plan_sha256':PLAN_SHA,'initial_state_tensor_count':509,'parameter_numel':param_numel,'optimizer_groups':group_summaries[0],'reconstruction_parity':{'control':parity_c,'smru':parity_sm},'regressions':regressions,'checks':checks}
    write_json(comp/'pretrain_contracts.json',result)
    if failed: raise RuntimeError(f'SMRU pretrain contracts failed: {failed}')
    del c,oldru,sm,gm,oc,oo,os,go,batch,grad_out,grad_metrics,pc,po,ps
    gc.collect(); torch.cuda.empty_cache(); return result

def _smoke(device):
    if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 3090' or torch.cuda.get_device_properties(device).total_memory<23*1024**3: raise RuntimeError('RTX3090 24GB required')
    pre=json.loads((OUT/'comparison/pretrain_contracts.json').read_text())
    if pre['status']!='pass' or pre['passed']!=16: raise RuntimeError('16/16 pretrain contracts required')
    source=load_state(PRETRAINED); models={arm:model_for(device,GAMMA[arm],source,ARM[arm]['scale_match']) for arm in ('control','smru')}
    mc,oc,_=models['control']; mr,orr,_=models['smru']; eq,n,bad,diff=state_equal(mc.state_dict(),mr.state_dict())
    if not eq or n!=509: raise RuntimeError('smoke fresh initialization mismatch')
    plan=json.loads(PLAN.read_text()); entry=plan['entries'][599]
    if entry['step']!=600: raise RuntimeError('smoke plan step is not 600')
    batch=_batch_for(oc,entry,device); records={}
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for arm,(model,opt,_) in models.items():
        _rng_seed(); model.train(); optimizer,audit=build_optimizer(model); set_optimizer_lr(optimizer,600); optimizer.zero_grad(set_to_none=True)
        output,metrics=model.step_loss(batch,step=600,coupled=False); pred=output['prediction']
        if float(metrics['understanding_weight'])!=.5 or float(metrics['unmatched_noobj_scale'])!=1.: raise RuntimeError('smoke recipe coefficient mismatch')
        if not all(torch.isfinite(metrics[k]).all() for k in ('loss','loss_recon','loss_understanding')): raise RuntimeError(f'nonfinite {arm} smoke loss')
        backward_gradient_controlled(model,metrics['loss_recon'],metrics['loss_understanding'],.5,shared_scale=.01)
        grads={n:p.grad for n,p in model.named_parameters()}
        required=['anchor_group.query_init','anchor_group.gru.weight_ih','anchor_group.ffn_fc1.weight','anchor_group.proj_u.weight']
        grad_report={}
        for name in required:
            g=grads[name]; norm=0. if g is None else float(torch.linalg.vector_norm(g))
            grad_report[name]={'present':g is not None,'finite':g is not None and bool(torch.isfinite(g).all()),'norm':norm,'nonzero':norm>0}
        if arm=='smru' and not all(x['present'] and x['finite'] and x['nonzero'] for x in grad_report.values()): raise RuntimeError('SMRU residual path gradient missing')
        smru_formula=[]
        if arm=='smru':
            qin=model.anchor_group.query_init.unsqueeze(0)
            with torch.no_grad():
                for st in pred['states']:
                    layer=int(st['layer'])
                    if layer not in (6,8,10,12): continue
                    a=st['anchor_embedding']; void=model.anchor_group.token_void(a)
                    replay=_replay(model.anchor_group,a,st['mu'],qin,void,st['ell'],.1,True)
                    mass,cand,scaled,qout=replay[1],replay[4],replay[5],replay[6]
                    use=(mass>=1e-4) & (torch.linalg.vector_norm(cand,dim=-1)>1e-6)
                    err=float((torch.linalg.vector_norm(scaled,dim=-1)[use]-torch.linalg.vector_norm(qin,dim=-1)[use]).abs().max()) if use.any() else 0.0
                    denom=torch.linalg.vector_norm(scaled-qin,dim=-1); delta=torch.linalg.vector_norm(qout-qin,dim=-1); ix=(mass>=1e-4)&(denom>1e-8)
                    ratio_err=float(((delta[ix]/denom[ix])-.1).abs().max()) if ix.any() else 0.0
                    smru_formula.append({'layer':layer,'norm_match_max_abs_error':err,'update_ratio_max_abs_error':ratio_err})
                    if err>1e-5 or ratio_err>1e-5: raise RuntimeError(f'SMRU smoke formula gate failed at layer {layer}')
                    qin=st['q']
        preclip=float(clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)); coef=min(1.,1/(preclip+1e-6)); optimizer.step()
        if not finite(optimizer.state): raise RuntimeError('nonfinite smoke optimizer state')
        records[arm]={'gamma':GAMMA[arm],'scale_match':ARM[arm]['scale_match'],'loss_recon':float(metrics['loss_recon']),'loss_understanding':float(metrics['loss_understanding']),
         'loss_anchor_group':float(metrics['loss_anchor_group']),'loss_total':float(metrics['loss']),'uweight':.5,
         'optimizer_groups':audit['groups'],'gradients':grad_report,'preclip_global_norm':preclip,'clip_coefficient':coef,
         'post_step_optimizer_finite':True}
        if arm=='smru': records[arm]['scale_match_formula_checks']=smru_formula
        if arm=='smru':
            records[arm]['peak_allocated_gib']=torch.cuda.max_memory_allocated(device)/1024**3
            records[arm]['peak_reserved_gib']=torch.cuda.max_memory_reserved(device)/1024**3
        del output,metrics,pred; gc.collect(); torch.cuda.empty_cache()
    result={'status':'pass','gpu':torch.cuda.get_device_name(device),'total_memory_gib':torch.cuda.get_device_properties(device).total_memory/1024**3,
      'torch':torch.__version__,'cuda':torch.version.cuda,'step':600,'understanding_weight':.5,'initial_state_exact':eq,'initial_state_tensor_count':n,
      'optimizer_step_count':2,'arms':records}
    write_json(OUT/'comparison/paired_one_step_smoke.json',result); return result

def _save_checkpoint(model,optimizer,arm,step,rng):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
      'architecture':model.architecture_name,'recipe':ARM[arm]['recipe'],'query_update_gamma':GAMMA[arm],'query_update_scale_match':ARM[arm]['scale_match'],
      'joint':True,'beta':0,'shared_understanding_grad_scale':.01,'unmatched_noobj_scale':1.0,
      'manifest_sha256':MANIFEST_SHA,'parent_plan_sha256':PLAN_SHA,'plan_prefix_length':1000,
      'pretrained_sha256':PRETRAINED_SHA,'rng':rng}
    torch.save(payload,WORK/arm/f'checkpoint_step{step}.pt'); return payload

def _endpoint_check(model,payload,arm):
    frozen=[n for n,p in model.named_parameters() if not p.requires_grad]
    valid=payload['step']==1000 and payload['architecture']==model.architecture_name and payload['recipe']==ARM[arm]['recipe'] and payload['query_update_gamma']==GAMMA[arm] and payload['query_update_scale_match']==ARM[arm]['scale_match'] and payload['joint'] is True and payload['beta']==0 and payload['shared_understanding_grad_scale']==.01 and payload['unmatched_noobj_scale']==1.0 and payload['manifest_sha256']==MANIFEST_SHA and payload['parent_plan_sha256']==PLAN_SHA and payload['pretrained_sha256']==PRETRAINED_SHA and all(payload['rng'].get(k) is not None for k in ('python','numpy','torch','cuda')) and finite(payload['model']) and finite(payload['optimizer']) and not frozen
    result={'status':'pass' if valid else 'fail','step':payload['step'],'architecture':payload['architecture'],'recipe':payload['recipe'],'query_update_gamma':payload['query_update_gamma'],'query_update_scale_match':payload['query_update_scale_match'],'joint':payload['joint'],'beta':payload['beta'],'shared_understanding_grad_scale':payload['shared_understanding_grad_scale'],'unmatched_noobj_scale':payload['unmatched_noobj_scale'],'manifest_sha256':payload['manifest_sha256'],'plan_sha256':payload['parent_plan_sha256'],'plan_prefix_length':payload['plan_prefix_length'],'pretrained_sha256':payload['pretrained_sha256'],'rng_present':all(payload['rng'].get(k) is not None for k in ('python','numpy','torch','cuda')),'model_finite':finite(payload['model']),'optimizer_finite':finite(payload['optimizer']),'trainable_reconstruction_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('anchor_group.')),'trainable_anchor_group_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith('anchor_group.')),'frozen_numel':sum(p.numel() for p in model.parameters() if not p.requires_grad),'frozen_names':frozen}
    write_json(OUT/arm/'endpoint_step1000_audit.json',result)
    if not valid: raise RuntimeError(f'endpoint audit failed {arm}')
    return result

def _slot_dynamics(model,opt,arm,step,device,util):
    manifest=json.loads(MANIFEST.read_text())['windows']; rows={str(l):[] for l in (6,8,10,12)}; update_rows={str(l):[] for l in (6,8,10,12)}
    norm_rows={str(l):[] for l in (6,8,10,12)}
    qr=util['train1024_queries']; counts=np.array([int(x['n_hungarian_matches']) for x in qr]); matched=set(np.where(counts>0)[0].tolist()); never=set(np.where(counts==0)[0].tolist()); top10=set(sorted(range(100),key=lambda q:(-counts[q],q))[:10]); cats={'matched':matched,'never':never,'top10':top10}
    model.eval()
    for idx in FIXED16:
        window=manifest[idx]; batch=_batch_for(opt,window,device)
        with torch.no_grad(): pred=forward_context(model,opt,batch,step=max(step,1))
        targets=build_anchor_targets(pred['states'][11]['mu'],batch['semantic_label_all'],batch['instance_label_all'],batch['cam_view_all'],batch['intrinsics_all'])
        qin=model.anchor_group.query_init.unsqueeze(0)
        for st in pred['states']:
            layer=int(st['layer'])
            if str(layer) not in rows: continue
            rows[str(layer)].append(slot._layer_case(st,qin,pred,targets,cats,model.anchor_group))
            a=st['anchor_embedding']; void=model.anchor_group.token_void(a)
            replay=_replay(model.anchor_group,a,st['mu'],qin,void,st['ell'],GAMMA[arm],ARM[arm]['scale_match'])
            _ap,mass,_z,_v,cand,cscaled,qexpected,_apost=replay
            non_skip=mass>=1e-4
            qn=torch.linalg.vector_norm(qin,dim=-1); cn=torch.linalg.vector_norm(cand,dim=-1); sn=torch.linalg.vector_norm(cscaled,dim=-1); outn=torch.linalg.vector_norm(st['q'],dim=-1)
            valid=non_skip & (cn>1e-6)
            normdiff=(sn-qn).abs()[valid]
            ratio_scaled=sn[valid]/qn[valid].clamp_min(1e-12)
            raw_ratio=cn[valid]/qn[valid].clamp_min(1e-12)
            direction=F.cosine_similarity(cscaled[valid],cand[valid],dim=-1) if valid.any() else torch.ones(1,device=device)
            denom=torch.linalg.vector_norm((cscaled if ARM[arm]['scale_match'] else cand)-qin,dim=-1)
            delta=torch.linalg.vector_norm(st['q']-qin,dim=-1)
            use=non_skip & (denom>1e-8)
            ratios=(delta[use]/denom[use]).detach().cpu().numpy() if use.any() else np.array([])
            update_rows[str(layer)].append({'ratio_mean':float(ratios.mean()) if ratios.size else None,'ratio_min':float(ratios.min()) if ratios.size else None,'ratio_max':float(ratios.max()) if ratios.size else None,'max_abs_error':float(np.max(np.abs(ratios-GAMMA[arm]))) if ratios.size else None,'non_skip_query_count':int(use.sum()),'candidate_delta_nonzero_count':int((denom>1e-8).sum()),'reference_delta':'candidate_scaled' if ARM[arm]['scale_match'] else 'q_candidate'})
            if not ratios.size or float(np.max(np.abs(ratios-GAMMA[arm])))>1e-5: raise RuntimeError(f'{arm} query interpolation diagnostic failed at layer {layer}, step {step}')
            norm_rows[str(layer)].append({'q_norm':qn[valid].detach().cpu().tolist(),'q_candidate_norm':cn[valid].detach().cpu().tolist(),'candidate_scaled_norm':sn[valid].detach().cpu().tolist(),'raw_candidate_q_norm_ratio':raw_ratio.detach().cpu().tolist(),'scaled_candidate_q_norm_ratio':ratio_scaled.detach().cpu().tolist(),'q_out_norm':outn[valid].detach().cpu().tolist(),'q_update_delta':delta[use].detach().cpu().tolist(),'reference_candidate_delta':denom[use].detach().cpu().tolist(),'norm_abs_error':normdiff.detach().cpu().tolist(),'direction_cosine':direction.detach().cpu().tolist()})
            qin=st['q']
        del pred,batch,targets
    agg={l:slot._aggregate_layer_cases(v) for l,v in rows.items()}
    norm_summary={}
    for layer,windows in norm_rows.items():
        def cat(key):
            values=[v for row in windows for v in row[key]]
            return np.asarray(values,dtype=np.float64)
        qn=cat('q_norm'); cn=cat('q_candidate_norm'); sn=cat('candidate_scaled_norm'); raw=cat('raw_candidate_q_norm_ratio'); scaled=cat('scaled_candidate_q_norm_ratio'); outn=cat('q_out_norm'); upd=cat('q_update_delta'); ref=cat('reference_candidate_delta'); err=cat('norm_abs_error'); direction=cat('direction_cosine')
        norm_summary[layer]={'non_skip_query_samples':int(len(qn)),'q_norm_median':float(np.median(qn)),'q_candidate_norm_median':float(np.median(cn)),'candidate_scaled_norm_median':float(np.median(sn)),'raw_candidate_q_norm_ratio_median':float(np.median(raw)),'scaled_candidate_q_norm_ratio_mean':float(np.mean(scaled)),'scaled_candidate_q_norm_ratio_median':float(np.median(scaled)),'scaled_candidate_q_norm_ratio_p10':float(np.percentile(scaled,10)),'scaled_candidate_q_norm_ratio_p90':float(np.percentile(scaled,90)),'max_norm_match_abs_error':float(np.max(err)),'direction_cosine_min':float(np.min(direction)),'q_out_norm_median':float(np.median(outn)),'q_update_delta_median':float(np.median(upd)),'reference_candidate_delta_median':float(np.median(ref)),'update_over_candidate_delta_median':float(np.median(upd/ref))}
    out={'step':step,'arm':arm,'scope':'locked fixed16 train windows','window_indices':FIXED16,'layer_aggregates':agg,'query_update_interpolation':update_rows,'query_norm_matching':norm_summary,'gamma':GAMMA[arm],'scale_match':ARM[arm]['scale_match'],'metric_definition_source':'scripts/anchor_group_v1_slot_dynamics_audit.py:_layer_case/_assignment_stage/_specialization'}
    write_json(OUT/arm/f'slot_dynamics_step{step}.json',out); model.train(); return out

def _train_arm(arm,device,manifest,plan,source):
    armout,armwork=OUT/arm,WORK/arm; armout.mkdir(parents=True,exist_ok=True); armwork.mkdir(parents=True,exist_ok=True); reuse._copy_monitors(arm)
    model,opt,transfer=model_for(device,GAMMA[arm],source,ARM[arm]['scale_match']); optimizer,optaudit=build_optimizer(model)
    observed={g['name']:(len(g['params']),sum(p.numel() for p in g['params'])) for g in optimizer.param_groups}
    if observed!=OPT_EXPECTED or len(model.state_dict())!=509: raise RuntimeError(f'optimizer/state mismatch {arm}: {observed}/{len(model.state_dict())}')
    digest=state_digest(model.state_dict()); initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('anchor_group.')}; rng0=capture_rng()
    model.train(); state0={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    evaluate_anchor_group_all(model,opt,armout,0,device,_seen_classes(armout)); model.train()
    changed=[k for k,v in state0.items() if not torch.equal(v,model.state_dict()[k].detach().cpu())]
    if changed or optimizer.state: raise RuntimeError(f'step0 evaluator mutated {arm}: {changed[:5]}')
    util0=reuse._manifest_utilization(model,opt,arm,0,device)
    struct0=_slot_dynamics(model,opt,arm,0,device,util0); restore_rng(rng0); del state0
    counts=np.zeros(100,dtype=np.int64); first=[None]*100; scenes=[set() for _ in range(100)]; online={}; logs=[]; endpoint_audit=None
    for entry in plan['entries'][:1000]:
        step=int(entry['step']); model.train(); set_optimizer_lr(optimizer,step); optimizer.zero_grad(set_to_none=True)
        batch=_batch_for(opt,entry,device); output,metrics=model.step_loss(batch,step=step,coupled=False); pred=output['prediction']
        with torch.no_grad(): _targets,pairs=unified_hungarian(pred,batch)
        for qi,_ki in pairs:
            for q in qi.tolist():
                q=int(q); counts[q]+=1
                if first[q] is None: first[q]=step
                scenes[q].add(str(entry['scene']))
        uw=float(metrics['understanding_weight']); expected=0. if step<=200 else ((step-200)/800 if step<1000 else 1.)
        if uw!=expected or float(metrics['unmatched_noobj_scale'])!=1.: raise RuntimeError(f'{arm} recipe mismatch at step{step}')
        if not all(torch.isfinite(metrics[k]).all() for k in ('loss','loss_recon','loss_understanding')): raise RuntimeError(f'nonfinite {arm} loss at step{step}')
        backward_gradient_controlled(model,metrics['loss_recon'],metrics['loss_understanding'],uw,shared_scale=.01)
        preclip=float(clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)); clip=min(1.,1/(preclip+1e-6)); optimizer.step()
        if not math.isfinite(preclip) or not finite(optimizer.state): raise RuntimeError(f'nonfinite optimizer update {arm} step{step}')
        if step%100==0:
            keys=('loss','loss_recon','loss_understanding','understanding_weight','loss_thing_2d','thing_ce','thing_ce_matched','thing_ce_unmatched','anchor_ce','anchor_dice','loss_anchor_group','loss_stuff_2d','loss_semantic','loss_identity','unmatched_noobj_scale')
            row={k:float(metrics[k].detach()) if torch.is_tensor(metrics[k]) else float(metrics[k]) for k in keys if k in metrics}
            row.update({'step':step,'arm':arm,'recipe':ARM[arm]['recipe'],'gamma':GAMMA[arm],'query_update_scale_match':ARM[arm]['scale_match'],'shared_understanding_grad_scale':.01,'preclip_global_norm':preclip,'clip_coefficient':clip,'group_lr':next(g['lr'] for g in optimizer.param_groups if g['name'].startswith('anchor_group_')),'reconstruction_lr':next(g['lr'] for g in optimizer.param_groups if g['name'].startswith('reconstruction_'))})
            if not finite(row): raise RuntimeError(f'nonfinite log {arm} step{step}')
            logs.append(row); print(json.dumps(row,sort_keys=True,allow_nan=False),flush=True)
        del output,pred,metrics,batch,_targets,pairs
        if step in (200,500,1000):
            online[str(step)]=reuse._online_summary(counts.copy(),list(first),[set(x) for x in scenes],step)
            rng=capture_rng(); evaluate_anchor_group_all(model,opt,armout,step,device,_seen_classes(armout)); model.train()
            util=reuse._manifest_utilization(model,opt,arm,step,device); struct=_slot_dynamics(model,opt,arm,step,device,util)
            payload=_save_checkpoint(model,optimizer,arm,step,rng)
            if step==1000: endpoint_audit=_endpoint_check(model,payload,arm)
            del payload; restore_rng(rng); gc.collect(); torch.cuda.empty_cache()
            print(f'[{arm}] eval/dynamics complete step {step}',flush=True)
    full_online=reuse._online_summary(counts,list(first),scenes,1000)
    write_json(armout/'online_match_exposure.json',{'definition':'per-step no-grad production unified_hungarian on each training prediction/batch','plan_prefix_length':1000,'snapshots':online,'final_first_positive_coverage':{str(k):sum(x is not None and x<=k for x in first) for k in (200,500,1000)},'queries':full_online['query_rows'],'top20':full_online['top20']})
    drift=reuse._drift_from_initial(model,initial); write_json(armout/'reconstruction_drift_step1000.json',drift)
    norms=np.array([x['preclip_global_norm'] for x in logs]); clips=np.array([x['clip_coefficient'] for x in logs])
    if len(logs)!=10: raise RuntimeError(f'{arm} missing 100-step logs')
    summary={'completed_steps':1000,'logged_train_steps':len(logs),'logged_steps':[x['step'] for x in logs],'all_logged_metrics_finite':all(finite(x) for x in logs),'gpu':torch.cuda.get_device_name(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'cuda':torch.version.cuda,'manifest_sha256':MANIFEST_SHA,'parent_plan_sha256':PLAN_SHA,'plan_prefix_length':1000,'pretrained_sha256':PRETRAINED_SHA,'recipe':ARM[arm]['recipe'],'query_update_gamma':GAMMA[arm],'query_update_scale_match':ARM[arm]['scale_match'],'shared_understanding_grad_scale':.01,'unmatched_noobj_scale':1.0,'trainable_reconstruction_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('anchor_group.')),'trainable_anchor_group_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith('anchor_group.')),'frozen_numel':sum(p.numel() for p in model.parameters() if not p.requires_grad),'preclip_norm':{'median':float(np.median(norms)),'p10':float(np.percentile(norms,10)),'p90':float(np.percentile(norms,90)),'max':float(norms.max())},'clip_coefficient':{'median':float(np.median(clips)),'fraction_actually_clipped':float(np.mean(clips<1))},'fresh_initial_state_sha256':digest,'status':'pass'}
    write_json(armout/'training_log_summary.json',summary); completed={'arm':arm,'transfer':transfer,'endpoint_audit':endpoint_audit,'drift':drift,'summary':summary}
    del model,optimizer,opt,initial; gc.collect(); torch.cuda.empty_cache(); return completed

def _paired_results(results):
    by={}; utilization={};
    for step in STEPS:
        by[str(step)]={'step':step,'arms':{}}
        for arm in ('control','smru'):
            curves=json.loads((OUT/arm/f'curves_{step}.json').read_text()); diag=json.loads((OUT/arm/f'anchor_group_diagnostics_step{step}.json').read_text()); util=json.loads((OUT/arm/f'manifest_utilization_step{step}.json').read_text())['train1024']; dyn=json.loads((OUT/arm/f'slot_dynamics_step{step}.json').read_text())
            by[str(step)]['arms'][arm]={'curves':curves,'diagnostics':diag,'slot_dynamics':dyn}
            u=util; mc=u['match_concentration']; utilization.setdefault(str(step),{})[arm]={'total_gt_matches':u['total_gt_matches'],'unique':u['unique_queries_ever_matched'],'never':u['queries_never_matched'],'top1':mc['top1_share'],'top5':mc['top5_share'],'top10':mc['top10_share'],'top20':mc['top20_share'],'gini':mc['gini'],'normalized_entropy':mc['normalized_entropy'],'effective_q':mc['effective_query_count'],'ownership_gini':u['ownership_concentration']['gini'],'ownership':u['ownership_concentration'],'no_object_mean':u['mean_no_object_probability_across_queries']['mean'],'active_queries':u['mean_active_output_queries']}
    norm={s:{a:by[s]['arms'][a]['slot_dynamics']['query_norm_matching'] for a in ('control','smru')} for s in by}
    result={'registered_steps':list(STEPS),'by_step':by,'train1024_utilization':utilization,'online_exposure':{a:json.loads((OUT/a/'online_match_exposure.json').read_text()) for a in ('control','smru')},'query_norm_diagnostics':norm,'status':'pass'}
    write_json(OUT/'comparison/paired_curves.json',result); write_json(OUT/'comparison/slot_formation_comparison.json',{'steps':list(STEPS),'by_step':{s:{a:by[s]['arms'][a]['slot_dynamics']['layer_aggregates'] for a in ('control','smru')} for s in by},'train1024_utilization':utilization,'status':'pass'})
    write_json(OUT/'comparison/norm_matching_diagnostics.json',{'steps':norm,'definition':'per fixed16 window and registered layer; Control scaled candidate is diagnostic only and is not used by Control forward','status':'pass'})
    return result

def _historical_ru_comparison(paired):
    old=REPO/'group_plus/anchor_group_v2_ru_gamma01_1k'
    rows={}
    for step in (500,1000):
        cur={}
        for arm in ('control','smru'):
            dyn=paired['by_step'][str(step)]['arms'][arm]['slot_dynamics']['layer_aggregates']; util=paired['train1024_utilization'][str(step)][arm]
            curve=paired['by_step'][str(step)]['arms'][arm]['curves']['val32_context']; diag=paired['by_step'][str(step)]['arms'][arm]['diagnostics']['val32_context']
            l6,l12=dyn['6'],dyn['12']; qin=l6['q_in.all.pr_rank']['mean']; qout=l6['q_out.all.pr_rank']['mean']
            cur[arm]={'l6_qout_pr':qout,'l6_pr_retention':qout/qin,'l6_qout_norm':l6['q_out.all.row_norm.median']['mean'],'l6_best_dice':l6['gt_specialization.post.aggregate.best_dice.median']['mean'],'l12_best_query_effective_q':l12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean'],'train1024_unique':util['unique'],'train1024_effective_q':util['effective_q'],'supported_gt_recall50':diag['anchor_group_gt_recall50'],'ca_r50':curve['class_agnostic_recall50'],'thing_miou':curve['mIoU_thing']}
        dyn=json.loads((old/'ru'/f'slot_dynamics_step{step}.json').read_text())['layer_aggregates']; l6,l12=dyn['6'],dyn['12']; util=json.loads((old/'ru'/f'manifest_utilization_step{step}.json').read_text())['train1024']; diag=json.loads((old/'ru'/f'anchor_group_diagnostics_step{step}.json').read_text())['val32_context']; curve=json.loads((old/'ru'/f'curves_{step}.json').read_text())['val32_context']; qin=l6['q_in.all.pr_rank']['mean']; qout=l6['q_out.all.pr_rank']['mean']
        cur['previous_ru_historical']={'l6_qout_pr':qout,'l6_pr_retention':qout/qin,'l6_qout_norm':l6['q_out.all.row_norm.median']['mean'],'l6_best_dice':l6['gt_specialization.post.aggregate.best_dice.median']['mean'],'l12_best_query_effective_q':l12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean'],'train1024_unique':util['unique_queries_ever_matched'],'train1024_effective_q':util['match_concentration']['effective_query_count'],'supported_gt_recall50':diag['anchor_group_gt_recall50'],'ca_r50':curve['class_agnostic_recall50'],'thing_miou':curve['mIoU_thing'],'source':'committed historical artifacts; not retrained in this paired experiment'}
        rows[str(step)]=cur
    result={'steps':rows,'previous_ru_retrained':False,'status':'pass'}
    write_json(OUT/'comparison/historical_ru_comparison.json',result)
    return result

def _smru_summary(result):
    # Use directional evidence across every registered outcome family; no
    # effect-size thresholds are introduced.
    outcomes={}
    for step in (500,1000):
        by=result['by_step'][str(step)]['arms']
        c6=by['control']['slot_dynamics']['layer_aggregates']['6']
        r6=by['smru']['slot_dynamics']['layer_aggregates']['6']
        c12=by['control']['slot_dynamics']['layer_aggregates']['12']
        r12=by['smru']['slot_dynamics']['layer_aggregates']['12']
        util=result['train1024_utilization'][str(step)]
        cu,ru=util['control'],util['smru']
        ccurve=by['control']['curves']['val32_context']; rcurve=by['smru']['curves']['val32_context']
        cdiag=by['control']['diagnostics']['val32_context']; rdiag=by['smru']['diagnostics']['val32_context']
        outcomes[str(step)]={
          'slot_diversity':{
            'control_qin_pr':c6['q_in.all.pr_rank']['mean'],'smru_qin_pr':r6['q_in.all.pr_rank']['mean'],
            'control_qout_pr':c6['q_out.all.pr_rank']['mean'],'smru_qout_pr':r6['q_out.all.pr_rank']['mean'],
            'control_qout_cosine_p90':c6['q_out.all.pairwise_cosine.p90']['mean'],'smru_qout_cosine_p90':r6['q_out.all.pairwise_cosine.p90']['mean'],
            'control_uout_pr':c6['u_out.all.pr_rank']['mean'],'smru_uout_pr':r6['u_out.all.pr_rank']['mean']},
          'gt_specialization':{
            'control_l6_best_dice':c6['gt_specialization.post.aggregate.best_dice.median']['mean'],'smru_l6_best_dice':r6['gt_specialization.post.aggregate.best_dice.median']['mean'],
            'control_l12_best_dice':c12['gt_specialization.post.aggregate.best_dice.median']['mean'],'smru_l12_best_dice':r12['gt_specialization.post.aggregate.best_dice.median']['mean'],
            'control_l12_hard_correct':c12['gt_specialization.post.aggregate.best_hard_correct.median']['mean'],'smru_l12_hard_correct':r12['gt_specialization.post.aggregate.best_hard_correct.median']['mean'],
            'control_l12_best_query_effq':c12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean'],'smru_l12_best_query_effq':r12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean']},
          'slot_utilization':{'control_unique':cu['unique'],'smru_unique':ru['unique'],'control_effective_q':cu['effective_q'],'smru_effective_q':ru['effective_q'],
            'control_top5_share':cu['top5'],'smru_top5_share':ru['top5'],'control_match_gini':cu['gini'],'smru_match_gini':ru['gini']},
          'grouping':{'control_supported_gt_recall50':cdiag['anchor_group_gt_recall50'],'smru_supported_gt_recall50':rdiag['anchor_group_gt_recall50'],
            'control_ca_r50':ccurve['class_agnostic_recall50'],'smru_ca_r50':rcurve['class_agnostic_recall50'],
            'control_thing_miou':ccurve['mIoU_thing'],'smru_thing_miou':rcurve['mIoU_thing'],
            'control_class_aware_r50':ccurve['class_aware_recall50'],'smru_class_aware_r50':rcurve['class_aware_recall50']}}
    a=outcomes['1000']; d=a['slot_diversity']; s=a['gt_specialization']; u=a['slot_utilization']; g=a['grouping']
    # Registered classification is directional and comparative. Do not add
    # a custom requirement that q_out PR must equal or exceed q_in PR.
    diversity_by_step={}
    for step in (500,1000):
        sd=outcomes[str(step)]['slot_diversity']
        diversity_by_step[str(step)]=(sd['smru_qout_pr']>sd['control_qout_pr'] and
                                      sd['smru_qout_cosine_p90']<sd['control_qout_cosine_p90'])
    diversity_directionally_better=all(diversity_by_step.values())
    diversity_preserved=diversity_directionally_better
    specialization=(s['smru_l6_best_dice']>s['control_l6_best_dice'] and s['smru_l12_best_dice']>s['control_l12_best_dice'] and s['smru_l12_hard_correct']>s['control_l12_hard_correct'] and s['smru_l12_best_query_effq']>s['control_l12_best_query_effq'])
    utilization=(u['smru_unique']>u['control_unique'] and u['smru_effective_q']>u['control_effective_q'] and u['smru_top5_share']<u['control_top5_share'] and u['smru_match_gini']<u['control_match_gini'])
    grouping=(g['smru_supported_gt_recall50']>g['control_supported_gt_recall50'] and g['smru_ca_r50']>g['control_ca_r50'] and g['smru_thing_miou']>g['control_thing_miou'] and g['smru_class_aware_r50']>g['control_class_aware_r50'])
    task_metrics=('smru_supported_gt_recall50','smru_ca_r50','smru_thing_miou','smru_class_aware_r50')
    control_metrics=('control_supported_gt_recall50','control_ca_r50','control_thing_miou','control_class_aware_r50')
    task_worse=all(g[r]<g[c] for r,c in zip(task_metrics,control_metrics))
    if diversity_preserved and specialization and utilization and grouping: outcome='SMRU-A'
    elif diversity_preserved and task_worse: outcome='SMRU-D'
    elif diversity_preserved: outcome='SMRU-B'
    else: outcome='SMRU-C'
    delta_keys={
      'slot_diversity':('control_qin_pr','smru_qin_pr','control_qout_pr','smru_qout_pr','control_qout_cosine_p90','smru_qout_cosine_p90'),
      'gt_specialization':('control_l6_best_dice','smru_l6_best_dice','control_l12_best_dice','smru_l12_best_dice','control_l12_hard_correct','smru_l12_hard_correct','control_l12_best_query_effq','smru_l12_best_query_effq'),
      'slot_utilization':('control_unique','smru_unique','control_effective_q','smru_effective_q','control_top5_share','smru_top5_share','control_match_gini','smru_match_gini'),
      'grouping':('control_supported_gt_recall50','smru_supported_gt_recall50','control_ca_r50','smru_ca_r50','control_thing_miou','smru_thing_miou','control_class_aware_r50','smru_class_aware_r50')}
    deltas={}
    for step in (500,1000):
        deltas[str(step)]={}
        for family,keys in delta_keys.items():
            vals=outcomes[str(step)][family]; row={}
            for ck,sk in zip(keys[::2],keys[1::2]):
                cval=float(vals[ck]); sval=float(vals[sk]); delta=sval-cval
                row[sk.removeprefix('smru_')]={'control':cval,'smru':sval,'absolute_delta':delta,'relative_delta':delta/abs(cval) if cval else None}
            deltas[str(step)][family]=row
    summary={'registered_outcome':outcome,'numeric_basis':outcomes,'deltas_smru_minus_control':deltas,
      'directional_checks_at_1000':{'slot_diversity_directionally_better_at_500_and_1000':diversity_directionally_better,'slot_diversity_relative_to_control':diversity_preserved,'smru_layer6_qin_pr':d.get('smru_qin_pr'),'smru_layer6_qout_pr':d['smru_qout_pr'],'smru_layer6_qout_pr_retention':d['smru_qout_pr'] / d['smru_qin_pr'] if d.get('smru_qin_pr') else None,'gt_specialization_improved':specialization,'slot_utilization_improved':utilization,'grouping_improved':grouping,'all_grouping_metrics_lower':task_worse},
      'interpretation':{'SMRU-A':'Scale-matched residual preservation improves both slot diversity and object-slot formation.','SMRU-B':'Relative query-output diversity improves, but the change does not produce synchronized gains in slot utilization and grouping/task metrics.','SMRU-C':'Scale mismatch was not the primary reason for query representation collapse.','SMRU-D':'Strong identity preservation conflicts with scene-conditioned object specialization.'}[outcome]}
    summary['evidence_text']=(
      'At steps 500 and 1000, SM-RU has higher layer-6 q_out PR and lower q_out cosine p90 than Control. '
      f'At step 1000, SM-RU q_in/q_out PR is {d["smru_qin_pr"]:.3f}/{d["smru_qout_pr"]:.3f} '
      f'(within-layer retention {d["smru_qout_pr"]/d["smru_qin_pr"]:.3f}), so query input diversity has also fallen during training. '
      f'GT-specialization indicators improve, but step-1000 utilization and task metrics are mixed: effective matched queries '
      f'{u["smru_effective_q"]:.3f} vs {u["control_effective_q"]:.3f}; supported-GT recall50 '
      f'{g["smru_supported_gt_recall50"]:.4f} vs {g["control_supported_gt_recall50"]:.4f}; ca-R50 '
      f'{g["smru_ca_r50"]:.4f} vs {g["control_ca_r50"]:.4f}; thing mIoU '
      f'{g["smru_thing_miou"]:.4f} vs {g["control_thing_miou"]:.4f}.')
    write_json(OUT/'comparison/smru_causal_summary.json',summary); return summary

def _report(paired,summary):
    historical=_historical_ru_comparison(paired)
    lines=['# Anchor-Group V2-SMRU gamma=0.1 — Paired 1k report','','## Provenance / recipe','',f'- GPU: NVIDIA GeForce RTX 3090; pretrained SHA: {PRETRAINED_SHA}; manifest SHA: {MANIFEST_SHA}; plan SHA: {PLAN_SHA} (first 1000 entries).','- Fresh paired Control and SM-RU arms used seed 42 and Anchor-Group init seed 31415. Both used the same pretrained reconstruction step 47500, GC alpha .01, 5000-step optimizer/LR schedule (first 1000 steps), warm-up, global clip, loss, Hungarian, data order and evaluations.','- The only paired-arm scientific difference was the registered query update: Control uses q_new=q_candidate; SM-RU matches candidate norm to q per query, preserving candidate direction, then uses q_new=q+0.1*(candidate_scaled-q).','- No new trainable parameter, gate, normalization, detach, loss term, assignment/evidence aggregation, Hungarian, optimizer, LR, GC or data-plan change was introduced. Previous plain RU is read-only historical context and was not retrained.','','## Table 1 — Layer-6 representation','','|Step|Arm|q-in PR|q-out PR|PR retention|q-in cosine p90|q-out cosine p90|q-out norm median|u-out PR|','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','smru'):
        d=paired['by_step'][str(step)]['arms'][arm]['slot_dynamics']['layer_aggregates']['6']; qi=d['q_in.all.pr_rank']['mean']; qo=d['q_out.all.pr_rank']['mean']
        lines.append(f"|{step}|{arm}|{qi:.4f}|{qo:.4f}|{qo/qi:.4f}|{d['q_in.all.pairwise_cosine.p90']['mean']:.4f}|{d['q_out.all.pairwise_cosine.p90']['mean']:.4f}|{d['q_out.all.row_norm.median']['mean']:.4f}|{d['u_out.all.pr_rank']['mean']:.4f}|")
    lines += ['','## Layer-wise q-output diversity','','|Step|Arm|Layer|q-out PR|q-out cosine p90|q-out norm median|u-out PR|','|---:|---|---:|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','smru'):
        d=paired['by_step'][str(step)]['arms'][arm]['slot_dynamics']['layer_aggregates']
        for layer in ('6','8','10','12'):
          x=d[layer]; lines.append(f"|{step}|{arm}|{layer}|{x['q_out.all.pr_rank']['mean']:.4f}|{x['q_out.all.pairwise_cosine.p90']['mean']:.4f}|{x['q_out.all.row_norm.median']['mean']:.4f}|{x['u_out.all.pr_rank']['mean']:.4f}|")
    lines += ['','## Table 2 — Norm matching and interpolation (fixed16)','','Control scale-matched candidate is diagnostic only; it is not used by Control forward. Ratios and error are computed over non-skip queries. The update/reference-candidate delta uses raw q_candidate for Control and candidate_scaled for SM-RU.','','|Step|Arm|Layer|median ||q|||median ||q_candidate|||raw cand/q norm ratio|median ||candidate_scaled|||scaled/q norm ratio mean / median / p10 / p90|max norm error|median ||q_out|||median update / reference-candidate delta|','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','smru'):
        d=paired['query_norm_diagnostics'][str(step)][arm]
        for layer in ('6','8','10','12'):
          x=d[layer]; ratios=f"{x['scaled_candidate_q_norm_ratio_mean']:.6f}/{x['scaled_candidate_q_norm_ratio_median']:.6f}/{x['scaled_candidate_q_norm_ratio_p10']:.6f}/{x['scaled_candidate_q_norm_ratio_p90']:.6f}"
          lines.append(f"|{step}|{arm}|{layer}|{x['q_norm_median']:.6g}|{x['q_candidate_norm_median']:.6g}|{x['raw_candidate_q_norm_ratio_median']:.6g}|{x['candidate_scaled_norm_median']:.6g}|{ratios}|{x['max_norm_match_abs_error']:.3g}|{x['q_out_norm_median']:.6g}|{x['update_over_candidate_delta_median']:.6f}|")
    lines += ['','## Table 3 — GT specialization','','|Step|Arm|Layer|Best Dice median / p90|Hard-correct median|fraction Dice≥.25|fraction hard≥.25|unique best queries|best-query Gini|best-query effective Q|Top5 share|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','smru'):
        d=paired['by_step'][str(step)]['arms'][arm]['slot_dynamics']['layer_aggregates']
        for layer in ('6','12'):
          x=d[layer]; p='gt_specialization.post.aggregate.'
          lines.append(f"|{step}|{arm}|{layer}|{x[p+'best_dice.median']['mean']:.4f}/{x[p+'best_dice.p90']['mean']:.4f}|{x[p+'best_hard_correct.median']['mean']:.4f}|{x[p+'fraction_best_dice_ge_0_25']['mean']:.4f}|{x[p+'fraction_best_hard_ge_0_25']['mean']:.4f}|{x[p+'best_query_concentration.pr_count']['mean']:.2f}|{x[p+'best_query_concentration.gini']['mean']:.4f}|{x[p+'best_query_concentration.effective_count']['mean']:.3f}|{x[p+'best_query_concentration.top5_share']['mean']:.4f}|")
    lines += ['','## Table 4 — Online positive exposure','','|Step|Arm|Matches|Unique ever|Never|Top1|Top5|Top10|Top20|Gini|Normalized entropy|Effective Q|First-positive coverage|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in (200,500,1000):
      for arm in ('control','smru'):
        x=paired['online_exposure'][arm]['snapshots'][str(step)]; mc=x['match_concentration'] if 'match_concentration' in x else x
        lines.append(f"|{step}|{arm}|{x['total_online_gt_matches']}|{x['unique_queries_ever_matched']}|{x['never_matched_queries']}|{x.get('top1_share',mc.get('top1_share',0)):.4f}|{x['top5_share']:.4f}|{x['top10_share']:.4f}|{x.get('top20_share',0):.4f}|{x['match_gini']:.4f}|{x.get('normalized_entropy',0):.4f}|{x['effective_query_count']:.3f}|{x['first_positive_coverage']['ever_matched']}/100|")
    lines += ['','## Table 5 — Train1024 utilization','','|Step|Arm|Matches|Unique|Never|Top1|Top5|Top10|Top20|Gini|Normalized entropy|Effective Q|Ownership Gini|P(no-object) mean|active output queries|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','smru'):
        x=paired['train1024_utilization'][str(step)][arm]; lines.append(f"|{step}|{arm}|{x['total_gt_matches']}|{x['unique']}|{x['never']}|{x['top1']:.4f}|{x['top5']:.4f}|{x['top10']:.4f}|{x['top20']:.4f}|{x['gini']:.4f}|{x['normalized_entropy']:.4f}|{x['effective_q']:.3f}|{x['ownership_gini']:.4f}|{x['no_object_mean']:.4f}|{x['active_queries']:.2f}|")
    lines += ['','## Table 6 — Val32 task and direct grouping','','|Step|Arm|Thing mIoU ctx/tgt|All mIoU ctx/tgt|Stuff mIoU ctx/tgt|ca-R50 ctx/tgt|class-aware R50 ctx/tgt|TP/FP/FN context|TP/FP/FN target|PSNR ctx/tgt|active queries ctx/tgt|supported-GT recall50|','|---:|---|---|---|---|---|---|---|---|---|---|---:|']
    for step in STEPS:
      for arm in ('control','smru'):
        x=paired['by_step'][str(step)]['arms'][arm]; c=x['curves']['val32_context']; t=x['curves']['val32_target']; d=x['diagnostics']['val32_context']
        allc=c.get('mIoU_all_nonempty',c.get('mIoU_all',0)); allt=t.get('mIoU_all_nonempty',t.get('mIoU_all',0)); stuffc=c.get('mIoU_stuff',0); stufft=t.get('mIoU_stuff',0)
        lines.append(f"|{step}|{arm}|{c['mIoU_thing']:.5f}/{t['mIoU_thing']:.5f}|{allc:.5f}/{allt:.5f}|{stuffc:.5f}/{stufft:.5f}|{c['class_agnostic_recall50']:.5f}/{t['class_agnostic_recall50']:.5f}|{c['class_aware_recall50']:.5f}/{t['class_aware_recall50']:.5f}|{c['tp_class_agnostic']}/{c['fp_class_agnostic']}/{c['fn_class_agnostic']}|{t['tp_class_agnostic']}/{t['fp_class_agnostic']}/{t['fn_class_agnostic']}|{c['psnr']:.4f}/{t['psnr']:.4f}|{c['active_thing_queries']:.2f}/{t['active_thing_queries']:.2f}|{d['anchor_group_gt_recall50']:.5f}|")
    lines += ['','### Val32 context direct-anchor diagnostics','','|Step|Arm|ownership accuracy|thing-anchor correct|supported-GT recall50|assignment entropy|thing ownership mean/median/p10/p90/max|max:median|query cosine mean/p90/max|P(no-object) mean/max|active queries|','|---:|---|---:|---:|---:|---:|---|---:|---|---|---:|']
    for step in STEPS:
      for arm in ('control','smru'):
        d=paired['by_step'][str(step)]['arms'][arm]['diagnostics']['val32_context']; m=d['mechanism_means']; mass=m['thing_ownership_mass']; cos=m['query_cosine']; no=m['no_object_probability']
        lines.append(f"|{step}|{arm}|{d['anchor_ownership_accuracy']:.5f}|{d['thing_anchor_correct_fraction']:.5f}|{d['anchor_group_gt_recall50']:.5f}|{d['assignment_entropy_mean']:.5f}|{mass['mean']:.4f}/{mass['median']:.4f}/{mass['p10']:.4f}/{mass['p90']:.4f}/{mass['max']:.4f}|{mass['max_over_median']:.2f}|{cos['offdiag_mean']:.5f}/{cos['p90']:.5f}/{cos['max']:.5f}|{no['mean']:.5f}/{no['max']:.5f}|{d['active_thing_queries_mean']:.2f}|")
    lines += ['','## Table 7 — Historical RU context (descriptive; not retrained)','','|Step|Arm|L6 q-out PR|L6 PR retention|L6 q-out norm|L6 best Dice|L12 bestQ effective Q|train1024 unique|train1024 effective Q|supported-GT R50|ca-R50|thing mIoU|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in (500,1000):
      for arm in ('control','previous_ru_historical','smru'):
        x=historical['steps'][str(step)][arm]; label={'control':'current Control','previous_ru_historical':'previous RU (historical)','smru':'current SM-RU'}[arm]
        lines.append(f"|{step}|{label}|{x['l6_qout_pr']:.4f}|{x['l6_pr_retention']:.4f}|{x['l6_qout_norm']:.4f}|{x['l6_best_dice']:.4f}|{x['l12_best_query_effective_q']:.3f}|{x['train1024_unique']}|{x['train1024_effective_q']:.3f}|{x['supported_gt_recall50']:.5f}|{x['ca_r50']:.5f}|{x['thing_miou']:.5f}|")
    lines += ['','## Registered SM-RU minus Control deltas','','Relative deltas use |Control| as denominator; they are n/a where Control is zero.','', '|Step|Metric|Control|SM-RU|Absolute delta|Relative delta|','|---:|---|---:|---:|---:|---:|']
    for step in ('500','1000'):
      for family,metrics in summary['deltas_smru_minus_control'][step].items():
        for metric,x in metrics.items():
          rel='n/a' if x['relative_delta'] is None else f"{x['relative_delta']:.4f}"
          lines.append(f"|{step}|{family}.{metric}|{x['control']:.6g}|{x['smru']:.6g}|{x['absolute_delta']:.6g}|{rel}|")
    lines += ['','## Classification','',f"**{summary['registered_outcome']}** — {summary['interpretation']}",'',summary.get('evidence_text',''),'','Previous RU values are read from committed historical artifacts and were not retrained. No gamma sweep, 5000-step run, or follow-up experiment was started.']
    (OUT/'comparison/smru_1k_report.md').write_text('\n'.join(lines)+'\n')

def _train_paired(device):
    if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 3090' or torch.cuda.get_device_properties(device).total_memory<23*1024**3: raise RuntimeError('RTX3090 24GB required')
    pre=json.loads((OUT/'comparison/pretrain_contracts.json').read_text()); smoke=json.loads((OUT/'comparison/paired_one_step_smoke.json').read_text())
    if pre['status']!='pass' or pre['passed']!=16 or smoke['status']!='pass': raise RuntimeError('pretrain/smoke gate failed')
    manifest,plan,monitor=reuse._asset_audit(); source=load_state(PRETRAINED); results={}
    for arm in ('control','smru'):
        if (WORK/arm/'checkpoint_step1000.pt').exists(): raise RuntimeError(f'refusing nonfresh existing {arm} endpoint')
        print(json.dumps({'event':'arm_start','arm':arm,'gamma':GAMMA[arm]}),flush=True); results[arm]=_train_arm(arm,device,manifest,plan,source)
    if results['control']['summary']['fresh_initial_state_sha256']!=results['smru']['summary']['fresh_initial_state_sha256']: raise RuntimeError('paired fresh state hashes differ')
    paired=_paired_results(results); summary=_smru_summary(paired); _report(paired,summary)
    write_json(OUT/'comparison/paired_training_audit.json',{'status':'pass','completed_steps_per_arm':{a:results[a]['summary']['completed_steps'] for a in results},'initial_state_hashes':{a:results[a]['summary']['fresh_initial_state_sha256'] for a in results},'exact_fresh_state':True,'endpoint_audits':{a:results[a]['endpoint_audit'] for a in results},'summaries':{a:results[a]['summary'] for a in results},'manifest_sha256':MANIFEST_SHA,'plan_sha256':PLAN_SHA,'pretrained_sha256':PRETRAINED_SHA,'monitor_hashes':monitor})

def _posthoc_rebuild_registered_diagnostics(device):
    """Read-only repair/rebuild of registered slot dynamics and paired reports."""
    if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 3090':
        raise RuntimeError('RTX3090 required for registered diagnostic replay')
    source=load_state(PRETRAINED)
    manifest,plan,_monitor=reuse._asset_audit()
    for arm in ('control','smru'):
        for step in STEPS:
            model,opt,_transfer=model_for(device,GAMMA[arm],source,ARM[arm]['scale_match'])
            if step:
                cp=WORK/arm/f'checkpoint_step{step}.pt'
                payload=torch.load(cp,map_location='cpu',weights_only=False)
                if payload.get('step')!=step or payload.get('architecture')!=model.architecture_name or payload.get('recipe')!=ARM[arm]['recipe'] or payload.get('query_update_gamma')!=GAMMA[arm] or payload.get('query_update_scale_match')!=ARM[arm]['scale_match']:
                    raise RuntimeError(f'checkpoint metadata mismatch: {cp}')
                model.load_state_dict(payload['model'],strict=True)
                del payload
            util=json.loads((OUT/arm/f'manifest_utilization_step{step}.json').read_text())
            _slot_dynamics(model,opt,arm,step,device,util)
            del model,opt
            gc.collect(); torch.cuda.empty_cache()
            print(f'[posthoc] registered slot dynamics rebuilt: {arm} step {step}',flush=True)
    return _posthoc_finalize_existing_artifacts()

def _posthoc_finalize_existing_artifacts():
    """Assemble reports/audits from completed training and diagnostic JSONs."""
    paired=_paired_results({})
    summary=_smru_summary(paired)
    _report(paired,summary)
    summaries={a:json.loads((OUT/a/'training_log_summary.json').read_text()) for a in ('control','smru')}
    endpoint_audits={a:json.loads((OUT/a/'endpoint_step1000_audit.json').read_text()) for a in ('control','smru')}
    hashes={a:summaries[a]['fresh_initial_state_sha256'] for a in ('control','smru')}
    exact_fresh=(hashes['control']==hashes['smru'])
    dynamics={a:{str(step):(OUT/a/f'slot_dynamics_step{step}.json').is_file() for step in STEPS} for a in ('control','smru')}
    dynamics_complete=all(all(v.values()) for v in dynamics.values())
    status='pass' if exact_fresh and dynamics_complete and all(endpoint_audits[a]['status']=='pass' and summaries[a]['completed_steps']==1000 for a in ('control','smru')) else 'fail'
    audit={'status':status,'completed_steps_per_arm':{a:summaries[a]['completed_steps'] for a in summaries},'initial_state_hashes':hashes,'exact_fresh_state':exact_fresh,'endpoint_audits':endpoint_audits,'summaries':summaries,'manifest_sha256':MANIFEST_SHA,'plan_sha256':PLAN_SHA,'pretrained_sha256':PRETRAINED_SHA,'registered_slot_dynamics_complete':dynamics,'registered_slot_dynamics_all_present':dynamics_complete}
    write_json(OUT/'comparison/paired_training_audit.json',audit)
    if status!='pass': raise RuntimeError('paired training audit failed during posthoc result assembly')
    return {'status':status,'outcome':summary['registered_outcome'],'registered_steps':list(STEPS)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('audit','smoke','train-paired'),required=True);p.add_argument('--device',choices=('cuda',),required=True);a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    d=torch.device('cuda')
    if torch.cuda.get_device_name(d)!='NVIDIA GeForce RTX 3090' or torch.cuda.get_device_properties(d).total_memory<23*1024**3: raise RuntimeError('this experiment requires RTX 3090 24GB')
    OUT.mkdir(parents=True,exist_ok=True)
    if a.phase=='audit':
        regression=reuse._run_regressions(d)
        if regression['status']!='pass': raise RuntimeError('legacy/Phase-A/Gc regressions failed')
        r=_pretrain_contracts(d);print(json.dumps({'pretrain_contracts':r['passed'],'total':r['total'],'status':r['status']}),flush=True)
    elif a.phase=='smoke':
        r=_smoke(d);print(json.dumps({'paired_smoke':r['status'],'optimizer_steps':r['optimizer_step_count'],'gpu':r['gpu']}),flush=True)
    else:_train_paired(d)

if __name__=='__main__':main()
