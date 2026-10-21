"""Required real 3090 single-card and forty-update global8 gates only."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import torch
import torch.distributed as dist
from scripts.object_locus_panoptic_v1_runtime import *
from tokengs.models.input_types import split_data
from tokengs.models.canonical_recon_models import LocusGSRecon
from tokengs.models.object_locus_panoptic_v1_lift import RendererTranspose


def adjoint_check(model,prediction,batch):
    from tokengs.models.input_types import ModelInputDecoder
    g=prediction['gaussians'].detach();decoder=ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2])
    x=torch.randn((*g.shape[:2],3),device=g.device)
    f=torch.randn((1,2,3,256,256),device=g.device,requires_grad=True)
    sx=model.gs.render_feature_channels(g,x,decoder.cam_view,decoder.intrinsics)['images_pred'].detach()
    lf=RendererTranspose.apply(f,g,decoder.cam_view,decoder.intrinsics,model.gs)
    lhs=(sx*f).sum();rhs=(x*lf).sum()
    scale=(sx*f).abs().sum()+(x*lf).abs().sum()
    error=float((lhs-rhs).abs()/scale.clamp_min(1e-12))
    lf.square().mean().backward()
    if error>1e-4 or not torch.isfinite(f.grad).all() or f.grad.norm()==0:raise RuntimeError('lifting adjoint/backward failed')
    return dict(normalized_error=error,feature_grad_norm=float(f.grad.norm()),geometry_grad_from_reading=None)


def parity_check(model,opt,batch,device):
    mi,_=split_data(batch,opt);model.eval()
    with torch.no_grad():
        reference1=LocusGSRecon.forward_reconstruction_only(model,mi)
        reference2=LocusGSRecon.forward_reconstruction_only(model,mi)
        actual=model.forward_object_locus(mi,step=0)
    report={}
    for key in ('tokens','mu','rho'):
        noise=max(float((a[key]-b[key]).abs().max()) for a,b in zip(reference1['states'],reference2['states']))
        delta=max(float((a[key]-b[key]).abs().max()) for a,b in zip(reference1['states'],actual['states']))
        report[key]=dict(repeat_noise=noise,delta=delta)
        if delta>noise+1e-6:raise RuntimeError(f'reconstruction skeleton differs: {key} {delta}/{noise}')
    for key in ('gaussians','RGB'):
        a,b,c=(o['gaussians'] if key=='gaussians' else o['render']['images_pred'] for o in (reference1,reference2,actual))
        noise=float((a-b).abs().max());delta=float((a-c).abs().max())
        report[key]=dict(repeat_noise=noise,delta=delta)
        if delta>noise+1e-6:raise RuntimeError(f'reconstruction parity {key}: {delta}/{noise}')
    model.train();return report


def state_digest(model,optimizer):
    digest=hashlib.sha256()
    for n,p in sorted(model.state_dict().items()):
        digest.update(n.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    for p,state in optimizer.state.items():
        for key in ('step','exp_avg','exp_avg_sq'):
            if key in state:digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--single',action='store_true');group.add_argument('--eight',action='store_true');args=parser.parse_args()
    device=init_distributed();rank,world=rank_world()
    if world!=(1 if args.single else 8):raise RuntimeError('smoke world mismatch')
    if args.eight and json.loads((REPORTS/'single_smoke.json').read_text())['status']!='PASS':raise RuntimeError('single smoke must pass first')
    manifest,_,_,_=assets();model,opt=build_model(device);optimizer=build_optimizer(model)
    first=manifest['expanded_train_windows'][sample_index(0,0,rank)]
    batch=build_batch(opt,first,device)
    parity=parity_check(model,opt,batch,device) if args.single else None
    logs=[];identities=[];adjoint=None
    torch.cuda.reset_peak_memory_stats()
    for update in range(2 if args.single else 40):
        wi=sample_index(0,update,rank) if args.eight else sample_index(0,0,rank)
        if args.eight:batch=build_batch(opt,manifest['expanded_train_windows'][wi],device)
        out,row=train_one_step(model,opt,optimizer,batch,update,check_rec_under=args.eight and update==39)
        prediction=out['prediction']
        for key in ('F_m','q_pre','gaussians','gaussian_membership','region_mass','alpha'):
            if not torch.isfinite(prediction[key]).all():raise FloatingPointError(f'nonfinite smoke {key}')
        for s in prediction['states']:
            if 'q' in s and not all(torch.isfinite(s[k]).all() for k in ('q','c','s','route')):raise FloatingPointError('nonfinite object state')
        if args.single and update==1:
            if row['gradient_norms']['pretrained']['under']<=0 or row['gradient_norms']['reconstruction']['under']<=0:raise RuntimeError('missing under gradient')
            if not all(v>0 for k,v in row['selected_under_gradients'].items()):raise RuntimeError('missing class/child/mask gradient')
            adjoint=adjoint_check(model,prediction,batch)
        row.update(rank=rank,window_index=wi,identity=manifest['expanded_train_windows'][wi]);logs.append(row);identities.append(wi)
        with (REPORTS/f'{"single" if args.single else "eight"}_smoke_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:row[k] for k in ('update','exposure','loss_recon','loss_understanding','peak_allocated','seconds')}),flush=True)
        del out,prediction
    if args.single:
        from scripts.eval_object_locus_v3_set import evaluate_windows
        result,_,_=evaluate_windows(model,opt,[first],0,'single_smoke',REPORTS/'smoke_eval',device,build_batch,official=True,panels=True)
        write_json(REPORTS/'single_smoke.json',dict(status='PASS',discarded=True,optimizer_steps=2,parity=parity,adjoint=adjoint,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),eval=result,steps=logs))
    else:
        activity=logs[-1]['activity']
        error=None
        if any(a['weight_norm']<=0 or a['delta_norm']<=0 for a in activity.values()):error=RuntimeError('inactive injection at forty updates')
        if not logs[-1]['rec_to_pretrained']:error=RuntimeError('rec→pretrained path has no finite nonzero gradient')
        synchronized_check(error,device,'activity40',40,batch)
        digest=state_digest(model,optimizer)
        digest_all=[None]*8;dist.all_gather_object(digest_all,digest)
        if len(set(digest_all))!=1:raise RuntimeError('rank model/optimizer mismatch')
        payload=dict(rank=rank,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),windows=identities,activity=activity,digest=digest)
        all_payload=[None]*8;dist.all_gather_object(all_payload,payload)
        if rank==0:
            seen=[wi for p in all_payload for wi in p['windows']]
            expected=np.random.default_rng(42).permutation(1008)[:320].tolist()
            if sorted(seen)!=sorted(expected):raise RuntimeError('global exposure set mismatch')
            write_json(REPORTS/'eight_smoke.json',dict(status='PASS',updates=40,exposures=320,discarded=True,ranks=all_payload,rec_to_pretrained=logs[-1]['rec_to_pretrained']))
        dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
