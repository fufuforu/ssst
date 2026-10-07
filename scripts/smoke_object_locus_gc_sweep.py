"""Discarded real-data one-card and global-eight smoke gates."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from scripts.object_locus_gc_sweep_runtime import *

def digest_state(model,optimizer=None):
    h=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):
        h.update(n.encode());h.update(memoryview(t.detach().cpu().contiguous().numpy()).cast('B'))
    if optimizer:
        for p,s in optimizer.state.items():
            for k,v in sorted(s.items()):
                h.update(str(k).encode())
                if torch.is_tensor(v):h.update(memoryview(v.detach().cpu().contiguous().numpy()).cast('B'))
                else:h.update(str(v).encode())
    return h.hexdigest()

def digest_optimizer(model,optimizer):
    h=hashlib.sha256();states=optimizer.state
    for name,p in sorted(model.named_parameters()):
        if p not in states:continue
        h.update(name.encode())
        for key,value in sorted(states[p].items()):
            h.update(str(key).encode())
            if torch.is_tensor(value):h.update(memoryview(value.detach().cpu().contiguous().numpy()).cast('B'))
            else:h.update(str(value).encode())
    return h.hexdigest()

def finite_prediction(out):
    p=out['prediction']
    keys=('gaussians','RGB','gaussian_membership','membership_mass','region_mass','semantic_scores','alpha','p_class')
    found=[]
    for k in keys:
        v=p.get(k)
        if torch.is_tensor(v):
            if not torch.isfinite(v).all():raise FloatingPointError('nonfinite output '+k)
            found.append(k)
    for state in p['states']:
        for k in ('q','c','s','route'):
            if k in state and not torch.isfinite(state[k]).all():raise FloatingPointError(f'nonfinite state {k}')
    return found

def single():
    device=init_distributed();rank,_=rank_world()
    manifest,plan,_=ensure_shared_plan();first=manifest['expanded_train_windows'][plan['entries'][0]['rank_windows'][0]]
    base_model=None;batch=None;all_logs=[];initial_hashes=[]
    for arm,alpha in ARMS.items():
        model,opt,source=build_model(device);initial_hashes.append(digest_state(model));optimizer=build_optimizer(model)
        if batch is None:batch=build_batch(opt,first,device)
        logs=[];torch.cuda.reset_peak_memory_stats()
        # Smoke updates exercise the stated 0.04/0.08 local warm-up weights.
        for update in (1,2):
            out,row=train_step(model,opt,optimizer,batch,update,alpha,diagnose=(update==2))
            found=finite_prediction(out)
            if not np.isfinite(row['loss_recon']) or not np.isfinite(row['loss_understanding']) or not np.isfinite(row['preclip_global_grad_norm']):raise FloatingPointError('nonfinite smoke scalar')
            expected_weight=(.04,.08)[update-1];expected_step=SOURCE_EXPOSURES+8*update
            if abs(row['understanding_weight']-expected_weight)>1e-12 or row['model_exposure']!=expected_step:raise RuntimeError('smoke warm-up/model-step mismatch')
            row.update(outputs_checked=found);logs.append(row);del out
        row=logs[-1];row.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        all_logs.append({'arm':arm,'alpha':alpha,'steps':logs,'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved()})
        if arm=='gc001':base_model=(model,opt)
        else:del model,opt,optimizer
    if len(set(initial_hashes))!=1:raise RuntimeError('arms did not share identical initial parameters/buffers')
    model,opt=base_model
    from scripts.eval_object_locus_panoptic_v1 import evaluate_windows
    local,_,_=evaluate_windows(model,opt,[first],0,'single_smoke',REPORT_ROOT/'smoke_eval',device,build_batch,official=True,panels=False)
    write_json(REPORT_ROOT/'single_smoke.json',{'status':'PASS','discarded':True,'updates_per_arm':2,'source_checkpoint':SOURCE_SHA256,'same_initial_state':True,'model_steps':[50072,50080],'understanding_weights':[.04,.08],'smokes':all_logs,'local_eval_and_single_pair_official_export_interface':{'status':'PASS','result':local}})
    del model,opt

def eight():
    device=init_distributed();rank,world=rank_world()
    if world!=8:raise RuntimeError('eight smoke needs eight ranks')
    manifest,plan,_=ensure_shared_plan();results=[]
    for arm,alpha in ARMS.items():
        model,opt,source=build_model(device);optimizer=build_optimizer(model);logs=[];windows=[];torch.cuda.reset_peak_memory_stats()
        for u,entry in enumerate(plan['entries'][:40]):
            wi=entry['rank_windows'][rank];batch=build_batch(opt,manifest['expanded_train_windows'][wi],device)
            out,row=train_step(model,opt,optimizer,batch,u,alpha,diagnose=(u+1)%100==0)
            finite_prediction(out);windows.append(wi)
            if row['new_exposures']!=(u+1)*8 or row['model_exposure']!=SOURCE_EXPOSURES+8*u or row['endpoint_model_exposure']!=SOURCE_EXPOSURES+(u+1)*8:raise RuntimeError('smoke exposure/plan mismatch')
            logs.append(row);del out,batch
        model_sha=digest_state(model);optimizer_sha=digest_optimizer(model,optimizer)
        local_counts=torch.tensor([windows.count(i) for i in range(WINDOWS)],device=device,dtype=torch.int32);dist.all_reduce(local_counts)
        expected=np.zeros(WINDOWS,dtype=np.int32)
        for entry in plan['entries'][:40]:expected[entry['rank_windows']]+=1
        if not np.array_equal(local_counts.cpu().numpy(),expected):raise RuntimeError('eight smoke rank exposure plan mismatch')
        state_steps=[float(s['step']) for s in optimizer.state.values() if 'step' in s]
        step_values=torch.tensor([len(logs),len(state_steps)],device=device,dtype=torch.int64)
        min_steps=step_values.clone();max_steps=step_values.clone()
        dist.all_reduce(min_steps,op=dist.ReduceOp.MIN);dist.all_reduce(max_steps,op=dist.ReduceOp.MAX)
        steps_ok=len(logs)==40 and (not state_steps or (all(1<=v<=40 for v in state_steps) and max(state_steps)==40))
        payload={'arm':arm,'alpha':alpha,'rank':rank,'windows':windows,'model_sha256':model_sha,'optimizer_sha256':optimizer_sha,'optimizer_state_tensors':len(state_steps),'optimizer_state_step_min':min(state_steps) if state_steps else None,'optimizer_state_step_max':max(state_steps) if state_steps else None,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'clip_count':sum(r['clipped'] for r in logs)}
        gathered=[None]*world;dist.all_gather_object(gathered,payload)
        model_sync=len({p['model_sha256'] for p in gathered})==1;optimizer_sync=len({p['optimizer_sha256'] for p in gathered})==1
        counts_sync=int(min_steps[0])==int(max_steps[0])==40 and int(min_steps[1])==int(max_steps[1])
        if not model_sync or not optimizer_sync or not steps_ok or not counts_sync:
            if rank==0:write_json(REPORT_ROOT/f'eight_smoke_failure_{arm}.json',{'ranks':gathered,'model_sync':model_sync,'optimizer_sync':optimizer_sync,'steps_ok':steps_ok,'rank_counts_sync':counts_sync})
            raise RuntimeError('rank parameter/optimizer state sync or forty optimizer updates failed')
        if rank==0:results.append({'arm':arm,'alpha':alpha,'ranks':gathered,'updates':40,'exposures':320,'rank_sync':True,'optimizer_sync':True,'exposure_plan_match':True})
        del model,opt,optimizer
        dist.barrier()
    if rank==0:write_json(REPORT_ROOT/'eight_smoke.json',{'status':'PASS','discarded':True,'updates_per_arm':40,'global_exposures_per_arm':320,'arms':results})
    dist.barrier();dist.destroy_process_group()

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--single',action='store_true');g.add_argument('--eight',action='store_true');a=ap.parse_args()
    REPORT_ROOT.mkdir(parents=True,exist_ok=True)
    if a.single:single()
    else:eight()
if __name__=='__main__':main()
