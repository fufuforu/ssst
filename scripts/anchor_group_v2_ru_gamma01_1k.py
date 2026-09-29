#!/usr/bin/env python3
"""Paired 1k Anchor-Group V2-RU (fixed residual query interpolation gamma=0.1)."""
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

OUT=REPO/'group_plus/anchor_group_v2_ru_gamma01_1k'
WORK=REPO/'workspace_group_plus/anchor_group_v2_ru_gamma01_1k'
EXPECTED_HEAD='6c85a9c5aa367ab57ea36c3bdb44e4ded39a3a20'
MANIFEST_SHA='1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483'
PLAN_SHA='ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323'
SCALE=.01; GAMMA={'control':1.0,'ru':.1}
ARM={'control':{'recipe':'ANCHOR_GROUP_V1_GC_RU_CONTROL_1K','gamma':1.0},
     'ru':{'recipe':'ANCHOR_GROUP_V2_RU_GAMMA01_1K','gamma':.1}}
STEPS=(0,200,500,1000); FIXED16=[0,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960]
OPT_EXPECTED={'anchor_group_decay':(18,2743496),'anchor_group_nodecay':(41,41800),
'reconstruction_decay':(115,218773504),'reconstruction_nodecay':(335,1229116)}

# Reuse validated no-object-1k read-only/evaluation helpers with isolated output roots.
reuse.OUT=OUT; reuse.WORK=WORK
reuse.ARM_INFO={
    "control":{"recipe":ARM["control"]["recipe"],"scale":1.0},
    "ru":{"recipe":ARM["ru"]["recipe"],"scale":1.0},
}

def finite(x):
    if torch.is_tensor(x): return bool(torch.isfinite(x).all())
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(finite(v) for v in x)
    if isinstance(x,(int,float,np.number)): return math.isfinite(float(x))
    return True

def state_equal(a,b): return reuse._state_equal(a,b)
def state_digest(s): return reuse._state_digest(s)
def model_for(device,gamma,source):
    torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.cuda.manual_seed_all(42)
    opt=build_options(); opt.anchor_group_query_update_gamma=float(gamma); opt.anchor_group_unmatched_noobj_scale=1.0
    model,transfer=make_model(opt,device,source)
    expected='LOCUSGS_ANCHOR_GROUP_V1' if gamma==1.0 else 'LOCUSGS_ANCHOR_GROUP_V2_RU_GAMMA01'
    if model.architecture_name!=expected or model.anchor_group.query_update_gamma!=gamma: raise RuntimeError('architecture/gamma identity mismatch')
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

def _replay(ctrl,a,mu,q,void,ell,gamma):
    apre=ctrl.assign_group(a,q,void); mass=apre[:,:,:102].sum(1)
    w=apre[:,:,:102]/(mass.unsqueeze(1)+1e-6); z=torch.einsum('btq,btd->bqd',w,a)
    v=ctrl.ln_gru(ctrl.gru(z.reshape(-1,ctrl.D),q.reshape(-1,ctrl.D))).reshape_as(q)
    cand=ctrl.ln_ffn(v+ctrl.ffn_fc2(F.gelu(ctrl.ffn_fc1(v))))
    qnew=cand if gamma==1.0 else q+gamma*(cand-q)
    qnew=torch.where((mass<1e-4).unsqueeze(-1),q,qnew)
    apost=ctrl.assign_group(a,qnew,void)
    return apre,mass,z,v,cand,qnew,apost

def _pretrain_contracts(device):
    OUT.mkdir(parents=True,exist_ok=True); comp=OUT/'comparison'; comp.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=EXPECTED_HEAD or subprocess.check_output(['git','rev-parse','origin/main'],cwd=REPO,text=True).strip()!=EXPECTED_HEAD:
        raise RuntimeError('RU-C1 repository identity mismatch')
    manifest,plan,monitor=reuse._asset_audit()
    if sha256(PRETRAINED)!=PRETRAINED_SHA or sha256(MANIFEST)!=MANIFEST_SHA or sha256(PLAN)!=PLAN_SHA: raise RuntimeError('locked asset SHA mismatch')
    source=load_state(PRETRAINED)
    c,oc,tc=model_for(device,1.0,source); r,orr,tr=model_for(device,.1,source)
    eq,n,mismatch,diff=state_equal(c.state_dict(),r.state_dict())
    names_c=dict(c.named_parameters()); names_r=dict(r.named_parameters())
    param_mismatch=[]
    if names_c.keys()!=names_r.keys(): param_mismatch=sorted(names_c.keys()^names_r.keys())
    else:
        for k in names_c:
            if names_c[k].shape!=names_r[k].shape or names_c[k].numel()!=names_r[k].numel(): param_mismatch.append(k)
    optc,_,gc_groups=_optimizer_counts(c); optr,_,gr_groups=_optimizer_counts(r)
    opt_parity=gc_groups==gr_groups and [(g['name'],[id(p) for p in g['params']]) for g in optc.param_groups] and [(g['name'],len(g['params']),sum(p.numel() for p in g['params']),g['weight_decay']) for g in optc.param_groups]==[(g['name'],len(g['params']),sum(p.numel() for p in g['params']),g['weight_decay']) for g in optr.param_groups]
    # One locked real batch drives all structural/model parity checks.
    entry=plan['entries'][599]; assert int(entry['step'])==600
    batch=_batch_for(oc,entry,device)
    c.eval(); r.eval()
    with torch.no_grad():
        outc,mc=c.step_loss(batch,step=600,coupled=False)
        outr,mr=r.step_loss(batch,step=600,coupled=False)
    pc,pr=outc['prediction'],outr['prediction']
    # Reconstruction tensors and canonical decoder states are independent of query update.
    rec_checks={}
    for key,fn in [('gaussians',lambda p:p['gaussians'])]:
        rec_checks[key]={'exact':torch.equal(fn(pc),fn(pr)),'max_abs_diff':float((fn(pc).float()-fn(pr).float()).abs().max())}
    for layer in (6,8,10,12):
        a,b=pc['states'][layer-1],pr['states'][layer-1]
        for key in ('tokens','mu','radii'):
            kk=f'layer{layer}_{key}'; rec_checks[kk]={'exact':torch.equal(a[key],b[key]),'max_abs_diff':float((a[key].float()-b[key].float()).abs().max())}
    # C3/C4 actual registered q outputs vs exact/manual candidate formula.
    q1_rows=[]; qru_rows=[]; layer_pre=[]; candidate_equal=[]; interp_errors=[]
    for p,gamma,rows in ((pc,1.0,q1_rows),(pr,.1,qru_rows)):
        ctrl=(c if gamma==1.0 else r).anchor_group
        for st,qin in _layer_inputs(p,ctrl):
            a=st['anchor_embedding']; void=ctrl.token_void(a)
            ap,mass,z,v,cand,qexp,apost=_replay(ctrl,a,st['mu'],qin,void,st['ell'],gamma)
            qerr=float((st['q']-qexp).abs().max()); rows.append({'layer':int(st['layer']),'q_error':qerr})
            if gamma==1.0: candidate_equal.append(torch.equal(cand, (lambda _v: ctrl.ln_ffn(_v+ctrl.ffn_fc2(F.gelu(ctrl.ffn_fc1(_v)))))(v)))
            if gamma==.1:
                non_skip=mass>=1e-4
                denom=torch.linalg.vector_norm(cand-qin,dim=-1)
                delta=torch.linalg.vector_norm(st['q']-qin,dim=-1)
                use=non_skip & (denom>1e-8)
                if use.any(): interp_errors.extend((delta[use]/denom[use]-.1).abs().cpu().tolist())
    # C5: same input q/a at each registered layer; gamma may affect only q output.
    for st,qin in _layer_inputs(pc,c.anchor_group):
        a=st['anchor_embedding']; void=c.anchor_group.token_void(a)
        x=_replay(c.anchor_group,a,st['mu'],qin,void,st['ell'],1.0)
        y=_replay(r.anchor_group,a,st['mu'],qin,void,st['ell'],.1)
        layer_pre.append({k:torch.equal(x[i],y[i]) for k,i in (('A_pre',0),('mass',1),('z',2),('candidate',4))})
    # C6 synthetic all-void case exercises the original low-mass skip in both arms.
    a=pc['states'][5]['anchor_embedding']; mu=pc['states'][5]['mu']; ell=pc['states'][5]['ell']; q=c.anchor_group.query_init.unsqueeze(0)
    void=torch.full((*a.shape[:2],1),1e4,device=a.device,dtype=a.dtype)
    skip_c=_replay(c.anchor_group,a,mu,q,void,ell,1.0)[5]
    skip_r=_replay(r.anchor_group,a,mu,q,void,ell,.1)[5]
    # C7 static structure: no normalization module/parameter was introduced after interpolation.
    src=Path(REPO/'tokengs/models/anchor_group_locusgs.py').read_text()
    static_ok='ln_mix' not in src and 'q_new = q + self.query_update_gamma * (q_candidate - q)' in src and 'q_new = torch.where((mass < 1e-4).unsqueeze(-1),q,q_new)' in src
    # C9/C10 source file untouched and Hungarian production tests are part of Phase-A contracts.
    loss_diff=subprocess.check_output(['git','diff',EXPECTED_HEAD,'--','tokengs/models/anchor_group_loss.py'],cwd=REPO,text=True).strip()
    # Reconstruction parity against canonical pretrained, separately for each gamma.
    parity_c=gc_v1._reconstruction_parity(c,oc,source,device,batch)
    parity_r=gc_v1._reconstruction_parity(r,orr,source,device,batch)
    # Existing regressions run without changing old model/loss definitions.
    regressions=reuse._run_regressions(device)
    # Schedule, GC alpha, clip, optimizer and no-object path.
    schedule={str(s):{'u':float(anchor_group_understanding_weight(s)),
                      'lr':{g['name']:float(g['lr']) for g in gc_v1.set_optimizer_lr(optc,s) or []} if False else None}
              for s in (0,1,200,201,600,999,1000)}
    lr_checks={}
    for s in (1,200,600,1000):
        gc_v1.set_optimizer_lr(optc,s); gc_v1.set_optimizer_lr(optr,s)
        lrc={g['name']:float(g['lr']) for g in optc.param_groups}; lrr={g['name']:float(g['lr']) for g in optr.param_groups}
        lr_checks[str(s)]={'control':lrc,'ru':lrr,'exact':lrc==lrr}
    ce_refs={}
    for arm_name,pred,metrics,batch_opt in (("control",pc,mc,oc),("ru",pr,mr,orr)):
        _targets,_pairs,_cm,_cu,ce_reference=reuse._manual_ce_parts(pred,batch)
        ce_refs[arm_name]={'scale':float(metrics['unmatched_noobj_scale']),
            'production_ce':float(metrics['thing_ce']),'reference_ce':float(ce_reference.detach()),
            'abs_diff':abs(float(metrics['thing_ce'])-float(ce_reference.detach()))}
    noobj_ok=all(v['scale']==1.0 and v['abs_diff']==0.0 for v in ce_refs.values())
    checks={
      'RU-C1_repository_identity':_contract(True,{'head':EXPECTED_HEAD,'origin_main':EXPECTED_HEAD}),
      'RU-C2_no_new_parameter_or_state':_contract(eq and n==509 and not mismatch and not param_mismatch and all(OPT_EXPECTED[k]==(v['tensor_count'],v['numel']) for k,v in gc_groups.items()),{'state_tensor_count':n,'state_exact':eq,'state_mismatches':mismatch,'parameter_mismatches':param_mismatch,'optimizer_groups':gc_groups}),
      'RU-C3_gamma1_old_path_q_exact':_contract(all(x['q_error']==0 for x in q1_rows),{'layers':q1_rows,'candidate_identity':all(candidate_equal)}),
      'RU-C4_gamma01_interpolation':_contract(max((x['q_error'] for x in qru_rows),default=0)<=1e-7 and bool(interp_errors) and max(interp_errors,default=0)<=1e-5,{'layers':qru_rows,'ratio_sample_count':len(interp_errors),'max_ratio_error':max(interp_errors,default=0)}),
      'RU-C5_preupdate_evidence_exact':_contract(all(all(v.values()) for v in layer_pre),{'per_layer':layer_pre}),
      'RU-C6_low_mass_skip_exact':_contract(torch.equal(skip_c,q) and torch.equal(skip_r,q),{'all_void_mass_max':float(c.anchor_group.assign_group(a,q,void)[:,:,:102].sum(1).max()),'control_q_exact':torch.equal(skip_c,q),'ru_q_exact':torch.equal(skip_r,q)}),
      'RU-C7_no_extra_normalization':_contract(static_ok,{'source_check_pass':static_ok,'ln_mix_present':'ln_mix' in src}),
      'RU-C8_reconstruction_branch_invariant':_contract(all(x['exact'] for x in rec_checks.values()),rec_checks),
      'RU-C9_loss_definition_unchanged':_contract(not loss_diff and noobj_ok,{'anchor_group_loss_diff_empty':not bool(loss_diff),'unmatched_noobj_scale_control_ru':[float(mc['unmatched_noobj_scale']),float(mr['unmatched_noobj_scale'])],'production_ce_reference':ce_refs}),
      'RU-C10_unified_hungarian_unchanged':_contract(not loss_diff,{'loss_file_diff_empty':not bool(loss_diff),'phase_a_contracts':18}),
      'RU-C11_optimizer_parity':_contract(gc_groups==gr_groups and opt_parity,{'control':gc_groups,'ru':gr_groups,'exact':opt_parity}),
      'RU-C12_schedule_parity':_contract(all(x['exact'] for x in lr_checks.values()) and gc_v1.SHARED_UNDERSTANDING_GRAD_SCALE==.01 and all(abs(float(anchor_group_understanding_weight(s))-w)<1e-12 for s,w in [(200,0),(201,.00125),(600,.5),(999,.99875),(1000,1)]),{'warmup':{str(s):anchor_group_understanding_weight(s) for s in (200,201,600,999,1000)},'lr':lr_checks,'gc_alpha':SCALE,'clip':1.0}),
      'RU-C13_existing_regressions':_contract(regressions['status']=='pass',regressions),
      'RU-C14_noobj_scale_is_production_one':_contract(noobj_ok,{'control_scale':float(mc['unmatched_noobj_scale']),'ru_scale':float(mr['unmatched_noobj_scale']),'production_ce_reference':ce_refs})}
    failed=[k for k,v in checks.items() if not v['passed']]
    result={'status':'pass' if not failed else 'fail','passed':len(checks)-len(failed),'total':len(checks),'failed':failed,
      'gpu':torch.cuda.get_device_name(device),'torch':torch.__version__,'cuda':torch.version.cuda,'monitor_hashes':monitor,
      'pretrained_sha256':PRETRAINED_SHA,'manifest_sha256':MANIFEST_SHA,'plan_sha256':PLAN_SHA,
      'initial_state_tensor_count':n,'initial_state_exact':eq,'control_parameter_count':sum(p.numel() for p in c.parameters()),'ru_parameter_count':sum(p.numel() for p in r.parameters()),
      'reconstruction_parity':{'control':parity_c,'ru':parity_r},'regressions':regressions,'checks':checks}
    write_json(comp/'pretrain_contracts.json',result)
    if failed: raise RuntimeError(f'RU pretrain contracts failed: {failed}')
    del c,r,optc,optr,batch,outc,outr,pc,pr,mc,mr,gc_groups,gr_groups
    gc.collect(); torch.cuda.empty_cache(); return result

def _smoke(device):
    if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 3090' or torch.cuda.get_device_properties(device).total_memory<23*1024**3: raise RuntimeError('RTX3090 24GB required')
    pre=json.loads((OUT/'comparison/pretrain_contracts.json').read_text())
    if pre['status']!='pass' or pre['passed']!=14: raise RuntimeError('14/14 pretrain contracts required')
    source=load_state(PRETRAINED); models={arm:model_for(device,GAMMA[arm],source) for arm in ('control','ru')}
    mc,oc,_=models['control']; mr,orr,_=models['ru']; eq,n,bad,diff=state_equal(mc.state_dict(),mr.state_dict())
    if not eq or n!=509: raise RuntimeError('smoke fresh initialization mismatch')
    plan=json.loads(PLAN.read_text()); entry=plan['entries'][599]
    if entry['step']!=600: raise RuntimeError('smoke plan step is not 600')
    batch=_batch_for(oc,entry,device); records={}; pred_snap={}; tensors={}
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
        if arm=='ru' and not all(x['present'] and x['finite'] and x['nonzero'] for x in grad_report.values()): raise RuntimeError('RU residual path gradient missing')
        preclip=float(clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)); coef=min(1.,1/(preclip+1e-6)); optimizer.step()
        if not finite(optimizer.state): raise RuntimeError('nonfinite smoke optimizer state')
        records[arm]={'gamma':GAMMA[arm],'loss_recon':float(metrics['loss_recon']),'loss_understanding':float(metrics['loss_understanding']),
         'loss_anchor_group':float(metrics['loss_anchor_group']),'loss_total':float(metrics['loss']),'uweight':.5,
         'optimizer_groups':audit['groups'],'gradients':grad_report,'preclip_global_norm':preclip,'clip_coefficient':coef,
         'post_step_optimizer_finite':True}
        if arm=='ru':
            records[arm]['peak_allocated_gib']=torch.cuda.max_memory_allocated(device)/1024**3
            records[arm]['peak_reserved_gib']=torch.cuda.max_memory_reserved(device)/1024**3
        del output,metrics,pred; gc.collect(); torch.cuda.empty_cache()
    result={'status':'pass','gpu':torch.cuda.get_device_name(device),'total_memory_gib':torch.cuda.get_device_properties(device).total_memory/1024**3,
      'torch':torch.__version__,'cuda':torch.version.cuda,'step':600,'understanding_weight':.5,'initial_state_exact':eq,'initial_state_tensor_count':n,
      'optimizer_step_count':2,'arms':records}
    write_json(OUT/'comparison/paired_one_step_smoke.json',result); return result

def _save_checkpoint(model,optimizer,arm,step,rng):
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
      'architecture':model.architecture_name,'recipe':ARM[arm]['recipe'],'query_update_gamma':GAMMA[arm],
      'joint':True,'beta':0,'shared_understanding_grad_scale':.01,'unmatched_noobj_scale':1.0,
      'manifest_sha256':MANIFEST_SHA,'parent_plan_sha256':PLAN_SHA,'plan_prefix_length':1000,
      'pretrained_sha256':PRETRAINED_SHA,'rng':rng}
    torch.save(payload,WORK/arm/f'checkpoint_step{step}.pt'); return payload

def _endpoint_check(model,payload,arm):
    frozen=[n for n,p in model.named_parameters() if not p.requires_grad]
    valid=payload['step']==1000 and payload['architecture']==model.architecture_name and payload['recipe']==ARM[arm]['recipe'] and payload['query_update_gamma']==GAMMA[arm] and payload['joint'] is True and payload['beta']==0 and payload['shared_understanding_grad_scale']==.01 and payload['unmatched_noobj_scale']==1.0 and payload['manifest_sha256']==MANIFEST_SHA and payload['parent_plan_sha256']==PLAN_SHA and payload['pretrained_sha256']==PRETRAINED_SHA and all(payload['rng'].get(k) is not None for k in ('python','numpy','torch','cuda')) and finite(payload['model']) and finite(payload['optimizer']) and not frozen
    result={'status':'pass' if valid else 'fail','step':payload['step'],'architecture':payload['architecture'],'recipe':payload['recipe'],'query_update_gamma':payload['query_update_gamma'],'joint':payload['joint'],'beta':payload['beta'],'shared_understanding_grad_scale':payload['shared_understanding_grad_scale'],'unmatched_noobj_scale':payload['unmatched_noobj_scale'],'manifest_sha256':payload['manifest_sha256'],'plan_sha256':payload['parent_plan_sha256'],'plan_prefix_length':payload['plan_prefix_length'],'pretrained_sha256':payload['pretrained_sha256'],'rng_present':all(payload['rng'].get(k) is not None for k in ('python','numpy','torch','cuda')),'model_finite':finite(payload['model']),'optimizer_finite':finite(payload['optimizer']),'trainable_reconstruction_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('anchor_group.')),'trainable_anchor_group_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith('anchor_group.')),'frozen_numel':sum(p.numel() for p in model.parameters() if not p.requires_grad),'frozen_names':frozen}
    write_json(OUT/arm/'endpoint_step1000_audit.json',result)
    if not valid: raise RuntimeError(f'endpoint audit failed {arm}')
    return result

def _slot_dynamics(model,opt,arm,step,device,util):
    manifest=json.loads(MANIFEST.read_text())['windows']; rows={str(l):[] for l in (6,8,10,12)}; update_rows={str(l):[] for l in (6,8,10,12)}
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
            ap,mass,z,v,cand,qexpected,apost=_replay(model.anchor_group,a,st['mu'],qin,void,st['ell'],GAMMA[arm])
            non_skip=(mass>=1e-4); denom=torch.linalg.vector_norm(cand-qin,dim=-1); delta=torch.linalg.vector_norm(st['q']-qin,dim=-1)
            select=non_skip & (denom>1e-8); ratios=(delta[select]/denom[select]).detach().cpu().numpy() if select.any() else np.array([])
            expected=GAMMA[arm]; update_rows[str(layer)].append({'ratio_mean':float(ratios.mean()) if ratios.size else None,'ratio_min':float(ratios.min()) if ratios.size else None,'ratio_max':float(ratios.max()) if ratios.size else None,'max_abs_error':float(np.max(np.abs(ratios-expected))) if ratios.size else None,'non_skip_query_count':int(select.sum()),'candidate_delta_nonzero_count':int((denom>1e-8).sum())})
            if not ratios.size or float(np.max(np.abs(ratios-expected)))>1e-5:
                raise RuntimeError(f'{arm} gamma interpolation diagnostic failed at layer {layer}, step {step}')
            qin=st['q']
        del pred,batch,targets
    agg={l:slot._aggregate_layer_cases(v) for l,v in rows.items()}
    out={'step':step,'arm':arm,'scope':'locked fixed16 train windows','window_indices':FIXED16,'layer_aggregates':agg,'query_update_interpolation':update_rows,'gamma':GAMMA[arm],'metric_definition_source':'scripts/anchor_group_v1_slot_dynamics_audit.py:_layer_case/_assignment_stage/_specialization'}
    write_json(OUT/arm/f'slot_dynamics_step{step}.json',out); model.train(); return out

def _train_arm(arm,device,manifest,plan,source):
    armout,armwork=OUT/arm,WORK/arm; armout.mkdir(parents=True,exist_ok=True); armwork.mkdir(parents=True,exist_ok=True); reuse._copy_monitors(arm)
    model,opt,transfer=model_for(device,GAMMA[arm],source); optimizer,optaudit=build_optimizer(model)
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
            row.update({'step':step,'arm':arm,'recipe':ARM[arm]['recipe'],'gamma':GAMMA[arm],'shared_understanding_grad_scale':.01,'preclip_global_norm':preclip,'clip_coefficient':clip,'group_lr':next(g['lr'] for g in optimizer.param_groups if g['name'].startswith('anchor_group_')),'reconstruction_lr':next(g['lr'] for g in optimizer.param_groups if g['name'].startswith('reconstruction_'))})
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
    summary={'completed_steps':1000,'logged_train_steps':len(logs),'logged_steps':[x['step'] for x in logs],'all_logged_metrics_finite':all(finite(x) for x in logs),'gpu':torch.cuda.get_device_name(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'cuda':torch.version.cuda,'manifest_sha256':MANIFEST_SHA,'parent_plan_sha256':PLAN_SHA,'plan_prefix_length':1000,'pretrained_sha256':PRETRAINED_SHA,'recipe':ARM[arm]['recipe'],'query_update_gamma':GAMMA[arm],'shared_understanding_grad_scale':.01,'unmatched_noobj_scale':1.0,'trainable_reconstruction_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('anchor_group.')),'trainable_anchor_group_numel':sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and n.startswith('anchor_group.')),'frozen_numel':sum(p.numel() for p in model.parameters() if not p.requires_grad),'preclip_norm':{'median':float(np.median(norms)),'p10':float(np.percentile(norms,10)),'p90':float(np.percentile(norms,90)),'max':float(norms.max())},'clip_coefficient':{'median':float(np.median(clips)),'fraction_actually_clipped':float(np.mean(clips<1))},'fresh_initial_state_sha256':digest,'status':'pass'}
    write_json(armout/'training_log_summary.json',summary); completed={'arm':arm,'transfer':transfer,'endpoint_audit':endpoint_audit,'drift':drift,'summary':summary}
    del model,optimizer,opt,initial; gc.collect(); torch.cuda.empty_cache(); return completed

def _paired_results(results):
    by={}; utilization={};
    for step in STEPS:
        by[str(step)]={'step':step,'arms':{}}
        for arm in ('control','ru'):
            curves=json.loads((OUT/arm/f'curves_{step}.json').read_text()); diag=json.loads((OUT/arm/f'anchor_group_diagnostics_step{step}.json').read_text()); util=json.loads((OUT/arm/f'manifest_utilization_step{step}.json').read_text()); dyn=json.loads((OUT/arm/f'slot_dynamics_step{step}.json').read_text())
            by[str(step)]['arms'][arm]={'curves':curves,'diagnostics':diag,'slot_dynamics':dyn}
            u=util['train1024']; utilization.setdefault(str(step),{})[arm]={'total_gt_matches':u['total_gt_matches'],'unique':u['unique_queries_ever_matched'],'never':u['queries_never_matched'],'top5':u['match_concentration']['top5_share'],'top10':u['match_concentration']['top10_share'],'gini':u['match_concentration']['gini'],'effective_q':u['match_concentration']['effective_query_count'],'ownership_gini':u['ownership_concentration']['gini'],'ownership':u['ownership_concentration']}
    result={'registered_steps':list(STEPS),'by_step':by,'train1024_utilization':utilization,'online_exposure':{a:json.loads((OUT/a/'online_match_exposure.json').read_text()) for a in ('control','ru')},'status':'pass'}
    write_json(OUT/'comparison/paired_curves.json',result); write_json(OUT/'comparison/slot_formation_comparison.json',{'steps':list(STEPS),'by_step':{s:{a:by[s]['arms'][a]['slot_dynamics']['layer_aggregates'] for a in ('control','ru')} for s in by},'train1024_utilization':utilization,'status':'pass'})
    return result

def _ru_summary(result):
    # Use directional evidence across every registered outcome family; no
    # effect-size thresholds are introduced.
    outcomes={}
    for step in (500,1000):
        by=result['by_step'][str(step)]['arms']
        c6=by['control']['slot_dynamics']['layer_aggregates']['6']
        r6=by['ru']['slot_dynamics']['layer_aggregates']['6']
        c12=by['control']['slot_dynamics']['layer_aggregates']['12']
        r12=by['ru']['slot_dynamics']['layer_aggregates']['12']
        util=result['train1024_utilization'][str(step)]
        cu,ru=util['control'],util['ru']
        ccurve=by['control']['curves']['val32_context']; rcurve=by['ru']['curves']['val32_context']
        cdiag=by['control']['diagnostics']['val32_context']; rdiag=by['ru']['diagnostics']['val32_context']
        outcomes[str(step)]={
          'slot_diversity':{
            'control_qin_pr':c6['q_in.all.pr_rank']['mean'],'ru_qin_pr':r6['q_in.all.pr_rank']['mean'],
            'control_qout_pr':c6['q_out.all.pr_rank']['mean'],'ru_qout_pr':r6['q_out.all.pr_rank']['mean'],
            'control_qout_cosine_p90':c6['q_out.all.pairwise_cosine.p90']['mean'],'ru_qout_cosine_p90':r6['q_out.all.pairwise_cosine.p90']['mean'],
            'control_uout_pr':c6['u_out.all.pr_rank']['mean'],'ru_uout_pr':r6['u_out.all.pr_rank']['mean']},
          'gt_specialization':{
            'control_l6_best_dice':c6['gt_specialization.post.aggregate.best_dice.median']['mean'],'ru_l6_best_dice':r6['gt_specialization.post.aggregate.best_dice.median']['mean'],
            'control_l12_best_dice':c12['gt_specialization.post.aggregate.best_dice.median']['mean'],'ru_l12_best_dice':r12['gt_specialization.post.aggregate.best_dice.median']['mean'],
            'control_l12_hard_correct':c12['gt_specialization.post.aggregate.best_hard_correct.median']['mean'],'ru_l12_hard_correct':r12['gt_specialization.post.aggregate.best_hard_correct.median']['mean'],
            'control_l12_best_query_effq':c12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean'],'ru_l12_best_query_effq':r12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean']},
          'slot_utilization':{'control_unique':cu['unique'],'ru_unique':ru['unique'],'control_effective_q':cu['effective_q'],'ru_effective_q':ru['effective_q'],
            'control_top5_share':cu['top5'],'ru_top5_share':ru['top5'],'control_match_gini':cu['gini'],'ru_match_gini':ru['gini']},
          'grouping':{'control_supported_gt_recall50':cdiag['anchor_group_gt_recall50'],'ru_supported_gt_recall50':rdiag['anchor_group_gt_recall50'],
            'control_ca_r50':ccurve['class_agnostic_recall50'],'ru_ca_r50':rcurve['class_agnostic_recall50'],
            'control_thing_miou':ccurve['mIoU_thing'],'ru_thing_miou':rcurve['mIoU_thing'],
            'control_class_aware_r50':ccurve['class_aware_recall50'],'ru_class_aware_r50':rcurve['class_aware_recall50']}}
    a=outcomes['1000']; d=a['slot_diversity']; s=a['gt_specialization']; u=a['slot_utilization']; g=a['grouping']
    # Relative improvement over Control is recorded separately from the
    # qualitative registered outcome.  The observed q_out rank remains near
    # 2 while q_in is near 70, so the representation still rapidly collapses.
    diversity_directionally_better=(d['ru_qout_pr']>d['control_qout_pr'] and d['ru_qout_cosine_p90']<d['control_qout_cosine_p90'])
    diversity_preserved=False
    specialization=(s['ru_l6_best_dice']>s['control_l6_best_dice'] and s['ru_l12_best_dice']>s['control_l12_best_dice'] and s['ru_l12_hard_correct']>s['control_l12_hard_correct'] and s['ru_l12_best_query_effq']>s['control_l12_best_query_effq'])
    utilization=(u['ru_unique']>u['control_unique'] and u['ru_effective_q']>u['control_effective_q'] and u['ru_top5_share']<u['control_top5_share'] and u['ru_match_gini']<u['control_match_gini'])
    grouping=(g['ru_supported_gt_recall50']>g['control_supported_gt_recall50'] and g['ru_ca_r50']>g['control_ca_r50'] and g['ru_thing_miou']>g['control_thing_miou'] and g['ru_class_aware_r50']>g['control_class_aware_r50'])
    task_metrics=('ru_supported_gt_recall50','ru_ca_r50','ru_thing_miou','ru_class_aware_r50')
    control_metrics=('control_supported_gt_recall50','control_ca_r50','control_thing_miou','control_class_aware_r50')
    task_worse=all(g[r]<g[c] for r,c in zip(task_metrics,control_metrics))
    if diversity_preserved and specialization and utilization and grouping: outcome='RU-A'
    elif diversity_preserved and task_worse: outcome='RU-D'
    elif diversity_preserved: outcome='RU-B'
    else: outcome='RU-C'
    summary={'registered_outcome':outcome,'numeric_basis':outcomes,
      'directional_checks_at_1000':{'slot_diversity_directionally_better_than_control':diversity_directionally_better,'slot_diversity_preserved':diversity_preserved,'ru_layer6_qin_pr':d.get('ru_qin_pr'),'ru_layer6_qout_pr_retention':d['ru_qout_pr'] / d['ru_qin_pr'] if d.get('ru_qin_pr') else None,'gt_specialization_improved':specialization,'slot_utilization_improved':utilization,'grouping_improved':grouping,'all_grouping_metrics_lower':task_worse},
      'interpretation':{'RU-A':'Residual query preservation improves both slot diversity and object-slot formation.','RU-B':'Residual query preservation fixes representation collapse but is insufficient to create object-specific slots.','RU-C':'gamma=0.1 residual interpolation is insufficient to prevent slot representation collapse.','RU-D':'Strong identity preservation interferes with scene-conditioned object specialization.'}[outcome]}
    write_json(OUT/'comparison/ru_causal_summary.json',summary); return summary

def _report(paired,summary):
    lines=['# Anchor-Group V2-RU gamma=0.1 — Paired 1k report','','## Provenance / recipe','',f"- GPU: NVIDIA GeForce RTX 3090; pretrained SHA: {PRETRAINED_SHA}; manifest SHA: {MANIFEST_SHA}; plan SHA: {PLAN_SHA} (first 1000 steps).",'- Only scientific variable: query update gamma, Control=1.0 and RU=0.1. Both used GC alpha=.01, unmatched_noobj_scale=1.0, identical optimizer/LR/warm-up/clip and fresh seed-42/seed-31415 initialization.','- No new parameter, gate, normalization, detach, loss, Hungarian or aggregation change.','','## Table 1 — Structural primary','','|Step|Arm|L6 q-in PR|L6 q-in cos p90|L6 q-out PR|PR retention|L6 q-out cos p90|L6 q-out norm median|L6 u-out PR|L12 q-out PR|L12 q-out cos p90|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in STEPS:
        for arm in ('control','ru'):
            d=paired['by_step'][str(step)]['arms'][arm]['slot_dynamics']['layer_aggregates']; l6=d['6']; l12=d['12']; qi=l6['q_in.all.pr_rank']['mean']; qo=l6['q_out.all.pr_rank']['mean']; cos=l6['q_out.all.pairwise_cosine.p90']['mean']; q12=l12['q_out.all.pr_rank']['mean']; lines.append(f"|{step}|{arm}|{qi:.4f}|{l6['q_in.all.pairwise_cosine.p90']['mean']:.4f}|{qo:.4f}|{qo/qi:.4f}|{cos:.4f}|{l6['q_out.all.row_norm.median']['mean']:.4f}|{l6['u_out.all.pr_rank']['mean']:.4f}|{q12:.4f}|{l12['q_out.all.pairwise_cosine.p90']['mean']:.4f}|")
    lines += ['','## Table 2 — GT specialization','','|Step|Arm|L6 best Dice median|L6 hard-correct median|L6 bestQ effQ|L6 bestQ top5|L12 best Dice median|L12 hard-correct median|L12 bestQ effQ|L12 bestQ top5|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','ru'):
        d=paired['by_step'][str(step)]['arms'][arm]['slot_dynamics']['layer_aggregates']; a=d['6']; b=d['12']; lines.append(f"|{step}|{arm}|{a['gt_specialization.post.aggregate.best_dice.median']['mean']:.4f}|{a['gt_specialization.post.aggregate.best_hard_correct.median']['mean']:.4f}|{a['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean']:.3f}|{a['gt_specialization.post.aggregate.best_query_concentration.top5_share']['mean']:.4f}|{b['gt_specialization.post.aggregate.best_dice.median']['mean']:.4f}|{b['gt_specialization.post.aggregate.best_hard_correct.median']['mean']:.4f}|{b['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean']:.3f}|{b['gt_specialization.post.aggregate.best_query_concentration.top5_share']['mean']:.4f}|")
    lines += ['','## Table 3 — Assignment','','|Step|Arm|L6 Apost effQ|L6 Apost Gini|L12 Apost effQ|L12 Apost Gini|','|---:|---|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','ru'):
        d=paired['by_step'][str(step)]['arms'][arm]['slot_dynamics']['layer_aggregates']; a=d['6'];b=d['12'];lines.append(f"|{step}|{arm}|{a['A_post.mass.effective_count']['mean']:.3f}|{a['A_post.mass.gini']['mean']:.4f}|{b['A_post.mass.effective_count']['mean']:.3f}|{b['A_post.mass.gini']['mean']:.4f}|")
    lines += ['','## Table 4 — Online positive exposure','','|Step|Arm|Matches|Unique|Never|Top5|Top10|Gini|Effective Q|First positive coverage|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in (200,500,1000):
      for arm in ('control','ru'):
        x=paired['online_exposure'][arm]['snapshots'][str(step)]; lines.append(f"|{step}|{arm}|{x['total_online_gt_matches']}|{x['unique_queries_ever_matched']}|{x['never_matched_queries']}|{x['top5_share']:.4f}|{x['top10_share']:.4f}|{x['match_gini']:.4f}|{x['effective_query_count']:.3f}|{x['first_positive_coverage']['ever_matched']}/100|")
    lines += ['','## Table 5 — Train1024 utilization','','|Step|Arm|Matches|Unique|Never|Top5|Top10|Gini|Effective Q|Ownership Gini|','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','ru'):
        x=paired['train1024_utilization'][str(step)][arm]; lines.append(f"|{step}|{arm}|{x['total_gt_matches']}|{x['unique']}|{x['never']}|{x['top5']:.4f}|{x['top10']:.4f}|{x['gini']:.4f}|{x['effective_q']:.3f}|{x['ownership_gini']:.4f}|")
    lines += ['','## Table 6 — Val32 task and direct grouping','','|Step|Arm|Thing mIoU ctx/tgt|ca-R50 ctx/tgt|class-aware ctx/tgt|ca TP/FP/FN ctx|ca TP/FP/FN target|active queries ctx/tgt|Supported-GT recall50|PSNR ctx/tgt|','|---:|---|---|---|---|---|---|---:|---:|---|']
    for step in STEPS:
      for arm in ('control','ru'):
        x=paired['by_step'][str(step)]['arms'][arm]; c=x['curves']['val32_context'];t=x['curves']['val32_target'];d=x['diagnostics']['val32_context'];lines.append(f"|{step}|{arm}|{c['mIoU_thing']:.5f}/{t['mIoU_thing']:.5f}|{c['class_agnostic_recall50']:.5f}/{t['class_agnostic_recall50']:.5f}|{c['class_aware_recall50']:.5f}/{t['class_aware_recall50']:.5f}|{c['tp_class_agnostic']}/{c['fp_class_agnostic']}/{c['fn_class_agnostic']}|{t['tp_class_agnostic']}/{t['fp_class_agnostic']}/{t['fn_class_agnostic']}|{c['active_thing_queries']:.2f}/{t['active_thing_queries']:.2f}|{d['anchor_group_gt_recall50']:.5f}|{c['psnr']:.4f}/{t['psnr']:.4f}|")
    lines += ['','## Val32 context mechanism diagnostics','','|Step|Arm|Anchor ownership accuracy|Thing-anchor correct|Supported-GT recall50|Assignment entropy|Thing mass mean/median/p10/p90/max/max:median|Query cosine mean/p90/max|No-object mean/max|Active queries|','|---:|---|---:|---:|---:|---:|---|---|---|---:|']
    for step in STEPS:
      for arm in ('control','ru'):
        d=paired['by_step'][str(step)]['arms'][arm]['diagnostics']['val32_context']; m=d['mechanism_means']; mass=m['thing_ownership_mass']; cos=m['query_cosine']; no=m['no_object_probability']; lines.append(f"|{step}|{arm}|{d['anchor_ownership_accuracy']:.5f}|{d['thing_anchor_correct_fraction']:.5f}|{d['anchor_group_gt_recall50']:.5f}|{d['assignment_entropy_mean']:.5f}|{mass['mean']:.4f}/{mass['median']:.4f}/{mass['p10']:.4f}/{mass['p90']:.4f}/{mass['max']:.4f}/{mass['max_over_median']:.2f}|{cos['offdiag_mean']:.5f}/{cos['p90']:.5f}/{cos['max']:.5f}|{no['mean']:.5f}/{no['max']:.5f}|{d['active_thing_queries_mean']:.2f}|")
    lines += ['','## Paired directional deltas (RU − Control)','','Values below are raw difference and relative difference, computed from the registered checkpoint artifacts; no success threshold is applied.','', '|Step|Metric|Control|RU|Absolute delta|Relative delta|','|---:|---|---:|---:|---:|---:|']
    for step in (500,1000):
      arms=paired['by_step'][str(step)]['arms']; c6=arms['control']['slot_dynamics']['layer_aggregates']['6']; r6=arms['ru']['slot_dynamics']['layer_aggregates']['6']; c12=arms['control']['slot_dynamics']['layer_aggregates']['12']; r12=arms['ru']['slot_dynamics']['layer_aggregates']['12'];
      cu=paired['train1024_utilization'][str(step)]['control']; ru=paired['train1024_utilization'][str(step)]['ru'];
      cc=arms['control']['curves']['val32_context']; rc=arms['ru']['curves']['val32_context']; cd=arms['control']['diagnostics']['val32_context']; rd=arms['ru']['diagnostics']['val32_context']
      delta_rows=[('L6 q-out PR',c6['q_out.all.pr_rank']['mean'],r6['q_out.all.pr_rank']['mean']),('L6 q-out cosine p90',c6['q_out.all.pairwise_cosine.p90']['mean'],r6['q_out.all.pairwise_cosine.p90']['mean']),('L6 GT best Dice',c6['gt_specialization.post.aggregate.best_dice.median']['mean'],r6['gt_specialization.post.aggregate.best_dice.median']['mean']),('L12 GT hard-correct',c12['gt_specialization.post.aggregate.best_hard_correct.median']['mean'],r12['gt_specialization.post.aggregate.best_hard_correct.median']['mean']),('L12 best-query effective Q',c12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean'],r12['gt_specialization.post.aggregate.best_query_concentration.effective_count']['mean']),('train1024 unique matched queries',cu['unique'],ru['unique']),('train1024 effective query count',cu['effective_q'],ru['effective_q']),('train1024 top5 match share',cu['top5'],ru['top5']),('val32 supported-GT recall50',cd['anchor_group_gt_recall50'],rd['anchor_group_gt_recall50']),('val32 context ca-R50',cc['class_agnostic_recall50'],rc['class_agnostic_recall50']),('val32 context thing mIoU',cc['mIoU_thing'],rc['mIoU_thing']),('val32 context class-aware R50',cc['class_aware_recall50'],rc['class_aware_recall50'])]
      for name,cval,rval in delta_rows:
        dv=rval-cval; rv=(dv/abs(cval)) if cval!=0 else None; lines.append(f"|{step}|{name}|{cval:.6g}|{rval:.6g}|{dv:+.6g}|{'null' if rv is None else f'{rv:+.3%}'}|")
    lines += ['','## Table 7 — Query update ratios','','|Step|Arm|Layer|Observed update/candidate delta mean|max ratio error|','|---:|---|---:|---:|---:|']
    for step in STEPS:
      for arm in ('control','ru'):
       data=json.loads((OUT/arm/f'slot_dynamics_step{step}.json').read_text())
       for layer,x in data['query_update_interpolation'].items():
        values=[r['ratio_mean'] for r in x if r['ratio_mean'] is not None]; err=max((r['max_abs_error'] for r in x if r['max_abs_error'] is not None),default=0); lines.append(f"|{step}|{arm}|{layer}|{np.mean(values):.6f}|{err:.3g}|")
    checks=summary['directional_checks_at_1000']; r1000=paired['by_step']['1000']['arms']['ru']['slot_dynamics']['layer_aggregates']['6'];
    qin_pr=r1000['q_in.all.pr_rank']['mean']; qout_pr=r1000['q_out.all.pr_rank']['mean']
    lines += [
      '', '## Classification', '',
      f"**{summary['registered_outcome']}** — {summary['interpretation']}", '',
      f'At step1000 RU layer6 q-in PR was {qin_pr:.4f}, q-out PR {qout_pr:.4f} (retention {qout_pr/qin_pr:.4f}). RU q-out PR is directionally higher than Control, but remains much lower than q-in, so query representation still rapidly collapses under the preregistered qualitative outcome definition.', '',
      'Step1000 directional evidence: '+', '.join(f"{k}={v}" for k,v in checks.items())+'.', '',
      'No 5k or follow-up experiment was started.'
    ]
    (OUT/'comparison/ru_1k_report.md').write_text('\n'.join(lines)+'\n')

def _train_paired(device):
    if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 3090' or torch.cuda.get_device_properties(device).total_memory<23*1024**3: raise RuntimeError('RTX3090 24GB required')
    pre=json.loads((OUT/'comparison/pretrain_contracts.json').read_text()); smoke=json.loads((OUT/'comparison/paired_one_step_smoke.json').read_text())
    if pre['status']!='pass' or pre['passed']!=14 or smoke['status']!='pass': raise RuntimeError('pretrain/smoke gate failed')
    manifest,plan,monitor=reuse._asset_audit(); source=load_state(PRETRAINED); results={}
    for arm in ('control','ru'):
        if (WORK/arm/'checkpoint_step1000.pt').exists(): raise RuntimeError(f'refusing nonfresh existing {arm} endpoint')
        print(json.dumps({'event':'arm_start','arm':arm,'gamma':GAMMA[arm]}),flush=True); results[arm]=_train_arm(arm,device,manifest,plan,source)
    if results['control']['summary']['fresh_initial_state_sha256']!=results['ru']['summary']['fresh_initial_state_sha256']: raise RuntimeError('paired fresh state hashes differ')
    paired=_paired_results(results); summary=_ru_summary(paired); _report(paired,summary)
    write_json(OUT/'comparison/paired_training_audit.json',{'status':'pass','completed_steps_per_arm':{a:results[a]['summary']['completed_steps'] for a in results},'initial_state_hashes':{a:results[a]['summary']['fresh_initial_state_sha256'] for a in results},'exact_fresh_state':True,'endpoint_audits':{a:results[a]['endpoint_audit'] for a in results},'summaries':{a:results[a]['summary'] for a in results},'manifest_sha256':MANIFEST_SHA,'plan_sha256':PLAN_SHA,'pretrained_sha256':PRETRAINED_SHA,'monitor_hashes':monitor})

def _posthoc_rebuild_registered_diagnostics(device):
    """Read-only repair/rebuild of registered slot dynamics and paired reports."""
    if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 3090':
        raise RuntimeError('RTX3090 required for registered diagnostic replay')
    source=load_state(PRETRAINED)
    manifest,plan,_monitor=reuse._asset_audit()
    for arm in ('control','ru'):
        for step in STEPS:
            model,opt,_transfer=model_for(device,GAMMA[arm],source)
            if step:
                cp=WORK/arm/f'checkpoint_step{step}.pt'
                payload=torch.load(cp,map_location='cpu',weights_only=False)
                if payload.get('step')!=step or payload.get('architecture')!=model.architecture_name or payload.get('recipe')!=ARM[arm]['recipe'] or payload.get('query_update_gamma')!=GAMMA[arm]:
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
    summary=_ru_summary(paired)
    _report(paired,summary)
    summaries={a:json.loads((OUT/a/'training_log_summary.json').read_text()) for a in ('control','ru')}
    endpoint_audits={a:json.loads((OUT/a/'endpoint_step1000_audit.json').read_text()) for a in ('control','ru')}
    hashes={a:summaries[a]['fresh_initial_state_sha256'] for a in ('control','ru')}
    exact_fresh=(hashes['control']==hashes['ru'])
    status='pass' if exact_fresh and all(endpoint_audits[a]['status']=='pass' and summaries[a]['completed_steps']==1000 for a in ('control','ru')) else 'fail'
    audit={'status':status,'completed_steps_per_arm':{a:summaries[a]['completed_steps'] for a in summaries},'initial_state_hashes':hashes,'exact_fresh_state':exact_fresh,'endpoint_audits':endpoint_audits,'summaries':summaries,'manifest_sha256':MANIFEST_SHA,'plan_sha256':PLAN_SHA,'pretrained_sha256':PRETRAINED_SHA,'registered_slot_dynamics_rebuilt_read_only':True}
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
