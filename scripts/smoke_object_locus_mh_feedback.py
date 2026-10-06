#!/usr/bin/env python3
"""Temporary contract/single-rank/eight-rank MH smoke; never writes checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from scripts import object_locus_mh_feedback_runtime as runtime
from scripts.object_locus_v3_set_runtime import write_json, jsonable


def model_digest(model):
    return runtime.model_state_sha(model.state_dict())


def optimizer_digest(optimizer):
    h=hashlib.sha256()
    names={id(p):n for group in optimizer.param_groups for n,p in zip(group.get('param_names',()),group['params'])}
    for key in sorted(optimizer.state,key=lambda p:names.get(id(p),'')):
        h.update(names.get(id(key),'').encode())
        for name,value in sorted(optimizer.state[key].items()):
            h.update(name.encode())
            if torch.is_tensor(value):
                h.update(str((tuple(value.shape),str(value.dtype))).encode())
                if value.numel():
                    h.update(str((float(value.float().sum()),float(value.float().square().sum()))).encode())
            else:h.update(str(value).encode())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('contracts','single','eight'),required=True)
    ap.add_argument('--updates',type=int)
    args=ap.parse_args()
    if args.mode=='contracts':
        import pytest
        code=pytest.main(['-q','tests/test_object_locus_mh_feedback_contracts.py'])
        if code==0:
            runtime.REPORT_ROOT.mkdir(parents=True,exist_ok=True)
            write_json(runtime.REPORT_ROOT/'contracts_smoke.json',dict(status='PASS',cpu_contracts=2,
                route='[B,8,1024,103] softmax over channels',mask='[B,1024,102]',
                plan='56 windows, 448 updates, 3584 exposures'))
        raise SystemExit(code)
    world_expected=1 if args.mode=='single' else 8
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world!=world_expected:raise RuntimeError(f'{args.mode} smoke expected {world_expected} ranks, got {world}')
    rank=0
    if world==8:
        local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
        dist.init_process_group('nccl');device=torch.device('cuda',local);rank=dist.get_rank()
    else:
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        if not socket.gethostname().startswith('3dimage-11'):raise RuntimeError('GPU smoke is fixed to 3dimage-11')
        if torch.cuda.get_device_name(device)!='NVIDIA GeForce RTX 3090':raise RuntimeError('RTX3090 required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    steps=2 if args.mode=='single' else 40
    if args.updates is not None and args.updates!=steps:raise RuntimeError('smoke update count is fixed')
    manifest,plan=runtime.manifest_and_plan()
    model,opt=runtime.build_model(device,report=(rank==0))
    optimizer=runtime.build_optimizer(model)
    initial={n:p.detach().clone() for n,p in model.named_parameters() if '.feedback_' in n or '.W_inject.' in n}
    inject_norms={}
    start=time.monotonic()
    for update in range(steps):
        wi=plan['entries'][update]['rank_windows'][rank]
        batch=runtime.build_batch(opt,manifest['train_all56'][wi],device)
        output,row=runtime.train_one_step(model,optimizer,batch,update)
        if not all(math.isfinite(float(v)) for k,v in row.items() if isinstance(v,(int,float))):
            raise FloatingPointError(f'non-finite smoke metric at update {update+1}')
        pred=output['prediction']
        for key in ('gaussians',):
            if not torch.isfinite(pred[key]).all():raise FloatingPointError(f'nonfinite {key}')
        if not torch.isfinite(pred['render']['images_pred']).all():raise FloatingPointError('nonfinite rendered RGB')
        for state in pred['states']:
            if 'feedback_attention' in state and not torch.isfinite(state['feedback_attention']).all():
                raise FloatingPointError('nonfinite feedback attention')
        if update == steps-1:
            groups={'reconstruction':False,'understanding':False,'object':False}
            for name,param in model.named_parameters():
                if param.grad is None:continue
                if not torch.isfinite(param.grad).all():raise FloatingPointError(f'nonfinite critical gradient: {name}')
                group='understanding' if name.startswith('understanding.') else 'object' if name.startswith('panoptic.') else 'reconstruction'
                if param.grad.norm()>0:groups[group]=True
            if not all(groups.values()):raise RuntimeError(f'missing reconstruction/understanding/object gradient path: {groups}')
            for lid in ('L6','L8','L10','L12'):
                layer=model.panoptic.layers[lid]
                for name in ('feedback_q','feedback_k','feedback_v','feedback_o'):
                    grad=getattr(layer,name).weight.grad
                    if grad is None or not torch.isfinite(grad).all():raise RuntimeError(f'missing/nonfinite {lid}.{name} feedback gradient')
        if update==steps-1:
            for lid in ('L6','L8','L10','L12'):
                layer=model.panoptic.layers[lid]
                inject_norms[lid]=float(layer.W_inject.weight.detach().norm())
                if not torch.isfinite(layer.W_inject.weight).all():raise FloatingPointError(f'{lid} inject nonfinite')
        del output,batch
    changed={name:bool(not torch.equal(param.detach(),initial[name])) for name,param in model.named_parameters() if name in initial}
    if args.mode=='single' and not all(changed.values()):
        missing=[n for n,v in changed.items() if not v]
        raise RuntimeError(f'expected every MH projection/W_inject to update after two steps: {missing}')
    if args.mode=='eight':
        if set(inject_norms)!={'L6','L8','L10','L12'} or not all(math.isfinite(v) and v>0 for v in inject_norms.values()):
            raise RuntimeError('all four feedback injections must be active after 40 updates')
        for name,param in model.named_parameters():
            if '.feedback_' in name and not torch.isfinite(param).all():raise FloatingPointError(f'nonfinite {name}')
        inactive=[name for name,changed_value in changed.items() if '.feedback_' in name and not changed_value]
        if inactive:raise RuntimeError(f'feedback projection weights did not update during 40-step smoke: {inactive}')
    row=dict(mode=args.mode,status='PASS',updates=steps,rank=rank,world_size=world,
        completed_exposures=8*steps,all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
        model_sha256=model_digest(model),optimizer_sha256=optimizer_digest(optimizer),
        injection_weight_norms=inject_norms,changed_feedback_or_inject=changed,
        data_plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
        elapsed_seconds=time.monotonic()-start)
    if world==8:
        local_row=dict(rank=rank,model_sha256=row['model_sha256'],optimizer_sha256=row['optimizer_sha256'])
        gathered=[None]*world if rank==0 else None
        dist.gather_object(local_row,gathered,dst=0)
        if rank==0:
            if len({x['model_sha256'] for x in gathered})!=1 or len({x['optimizer_sha256'] for x in gathered})!=1:
                raise RuntimeError('model or optimizer diverged across smoke ranks')
            row['ranks']=gathered
    if rank==0:
        runtime.REPORT_ROOT.mkdir(parents=True,exist_ok=True)
        path=runtime.REPORT_ROOT/f'{args.mode}_smoke.json'
        write_json(path,row)
        print(json.dumps(row,indent=2))
    if world==8:
        dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
