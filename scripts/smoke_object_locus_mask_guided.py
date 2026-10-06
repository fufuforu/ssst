"""Temporary matched single-GPU and eight-GPU update smoke; saves no state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
import torch.distributed as dist

from scripts.object_locus_mask_guided_runtime import (
    REPORT_ROOT, arm_dirs, build_model, build_optimizer, build_batch,
    manifest_and_plan, rank_world, train_one_step, write_json,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('single','eight'),required=True)
    parser.add_argument('--arm',choices=('control','mask_guided','both'),default='both')
    args=parser.parse_args()
    world=int(os.environ.get('WORLD_SIZE','1'))
    if args.mode=='single' and world!=1:raise RuntimeError('single smoke expects one process')
    if args.mode=='eight' and world!=8:raise RuntimeError('eight smoke expects eight processes')
    if not torch.cuda.is_available():raise RuntimeError('smoke requires CUDA')
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local)
    if args.mode=='eight':
        import datetime
        dist.init_process_group('nccl',timeout=datetime.timedelta(hours=4))
    device=torch.device('cuda',local)
    if not os.uname().nodename.startswith('3dimage-11') or torch.cuda.get_device_name(local)!='NVIDIA GeForce RTX 3090':
        raise RuntimeError('fixed smoke node/device is 3dimage-11 RTX3090')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest,plan=manifest_and_plan()
    plan_sha=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    if args.mode=='eight':
        all_shas=[None]*world
        dist.all_gather_object(all_shas,plan_sha)
        if len(set(all_shas))!=1:raise RuntimeError('rank data-plan SHA mismatch')
    arms=('control','mask_guided') if args.arm=='both' else (args.arm,)
    zero_injection_reference=None
    for arm in arms:
        torch.cuda.reset_peak_memory_stats(device)
        model,opt=build_model(arm,device,report=False);optimizer=build_optimizer(model)
        if args.arm=='both':
            for lid in ('L6','L8','L10','L12'):
                if torch.count_nonzero(model.panoptic.layers[lid].W_inject.weight):
                    raise RuntimeError(f'{lid} W_inject is not fresh zero initialization')
            probe=build_batch(opt,manifest['train_all56'][plan['entries'][0]['rank_windows'][rank]],device)
            model.eval()
            with torch.no_grad():
                first=model.step_loss(probe,step=0,understanding_weight=0)[0]['prediction']
                gauss=first['gaussians'].detach().cpu().clone()
                rgb=first['render']['images_pred'].detach().cpu().clone()
                del first
                if arm=='control':
                    repeated=model.step_loss(probe,step=0,understanding_weight=0)[0]['prediction']
                    g_noise=float((gauss-repeated['gaussians'].detach().cpu()).abs().max())
                    rgb_noise=float((rgb-repeated['render']['images_pred'].detach().cpu()).abs().max())
                    del repeated
                    zero_injection_reference=(gauss,rgb,g_noise,rgb_noise)
                else:
                    if zero_injection_reference is None:raise RuntimeError('control zero-injection baseline missing')
                    ref_g,ref_rgb,g_noise,rgb_noise=zero_injection_reference
                    g_diff=float((gauss-ref_g).abs().max())
                    rgb_diff=float((rgb-ref_rgb).abs().max())
                    if g_diff>g_noise+1e-6 or rgb_diff>rgb_noise+1e-6:
                        raise RuntimeError(f'zero-injection output mismatch exceeds measured repeat envelope: {g_diff}/{g_noise}; {rgb_diff}/{rgb_noise}')
                    zero_injection_reference=dict(gaussian_repeat_noise=g_noise,rgb_repeat_noise=rgb_noise,
                        control_vs_mask_guided_gaussian=g_diff,control_vs_mask_guided_rgb=rgb_diff)
            model.train();del probe
        updates=2 if args.mode=='single' else 40
        started=time.monotonic();local_rows=[]
        for update in range(updates):
            entry=plan['entries'][update]
            wi=entry['rank_windows'][rank]
            batch=build_batch(opt,manifest['train_all56'][wi],device)
            output,row=train_one_step(model,optimizer,batch,update,check_rec_under=True)
            required=('loss','loss_recon','loss_understanding','preclip_norm')
            if not all(k in row for k in required):raise RuntimeError(f'{arm} missing smoke metrics')
            if not all(torch.isfinite(torch.tensor(row[k])) for k in required):raise FloatingPointError('nonfinite smoke metric')
            if update==updates-1:
                rec=[p.grad for n,p in model.named_parameters() if not n.startswith('understanding.') and p.grad is not None]
                under=[p.grad for n,p in model.named_parameters() if n.startswith('understanding.') and p.grad is not None]
                if not rec or not under or not any(float(g.norm())>0 for g in rec) or not any(float(g.norm())>0 for g in under):
                    raise RuntimeError('reconstruction/understanding gradient path absent')
            if (update+1)%100==0:
                row['route_injection']={}
                for state in output['prediction']['states']:
                    if 'route' in state:
                        layer=f"L{state['layer']}"
                        inj=model.panoptic.layers[layer].W_inject.weight
                        row['route_injection'][layer]=dict(route_min=float(state['route'].min()),
                            route_max=float(state['route'].max()),route_entropy=float(-(state['route'].clamp_min(1e-12).log()*state['route']).sum(-1).mean()),
                            injection_norm=float(inj.norm()),delta_norm=float(state['joint_delta'].norm()))
            local_rows.append(row)
            del output,batch
        # All ranks must hold equal model values and AdamW state after each explicit mean.
        if args.mode=='eight':
            with torch.no_grad():
                for p in model.parameters():
                    lo=p.detach().clone();hi=lo.clone()
                    dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX)
                    if not torch.allclose(lo,hi,rtol=0,atol=1e-6):raise RuntimeError('rank parameter drift')
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value) and value.numel()>1:
                        lo=value.clone();hi=value.clone();dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX)
                        if not torch.allclose(lo,hi,rtol=0,atol=1e-6):raise RuntimeError('rank optimizer state drift')
        if args.mode=='eight':
            for lid in ('L6','L8','L10','L12'):
                value=model.panoptic.layers[lid].W_inject.weight
                low=value.detach().clone();high=low.clone();dist.all_reduce(low,op=dist.ReduceOp.MIN);dist.all_reduce(high,op=dist.ReduceOp.MAX)
                if not torch.isfinite(value).all() or float(value.norm())<=0:raise RuntimeError(f'{lid} injection inactive/nonfinite after smoke')
        if rank==0:
            reports,_=arm_dirs(arm)
            injection_norms={lid:float(model.panoptic.layers[lid].W_inject.weight.detach().norm())
                             for lid in ('L6','L8','L10','L12')}
            result=dict(status='PASS',mode=args.mode,arm=arm,updates=updates,
                data_plan_sha256=plan_sha,
                parameter_numel=sum(p.numel() for p in model.parameters()),
                state_keys=len(model.state_dict()),all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
                injection_weight_norms=injection_norms,
                synchronized_parameters_and_optimizer=(args.mode=='eight'),
                first_loss=local_rows[0],last_loss=local_rows[-1],
                zero_injection_comparison=zero_injection_reference if arm=='mask_guided' else None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                elapsed_seconds=time.monotonic()-started,checkpoint_state='DISCARDED')
            write_json(reports/f'{args.mode}_smoke.json',result)
        del optimizer,model
        torch.cuda.empty_cache()
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
