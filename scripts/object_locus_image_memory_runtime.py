"""Fixed full-data C32/U128 continuation, four ranks and two microsteps."""
import dataclasses, datetime, gc as python_gc, json, math, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from scripts import object_locus_gc_sweep_runtime as base
from tokengs.models.object_locus_panoptic_v1 import object_image_memory

ROOT = Path('/space/mawb/ssst/group_plus/object_locus_image_memory_full128_v1')
RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_image_memory_full128_v1')
SOURCE = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
ARMS = {'c32':32, 'u128':128}
write_json, sha256, build_batch = base.write_json, base.sha256, base.build_batch


def prepare_plan():
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    old=json.loads((SOURCE/'training_plan.json').read_text())
    windows=manifest['windows']; n=len(windows); u=(n+7)//8; p=8*u-n
    if (len({w['scene'] for w in windows}),n)!=(1191,8337): raise RuntimeError('full data identity differs')
    if sha256(SOURCE/'manifest.json')!=old['manifest_sha256']: raise RuntimeError('manifest SHA differs')
    identities=[(w['scene'],w['window_index'],tuple(w['context']),tuple(w['target'])) for w in windows]
    if len(set(identities))!=n or any(len(w['context'])!=2 for w in windows): raise RuntimeError('window protocol differs')
    orders=[];padding=[];counts=np.zeros(n,dtype=np.int64)
    for e in range(8):
        perm=np.random.default_rng(42+e).permutation(n).tolist(); order=perm+perm[:p]
        if order!=old['orders'][e]: raise RuntimeError('source deterministic order differs')
        orders.append(order);padding.append(perm[:p]);np.add.at(counts,order,1)
    plan={'S':1191,'N':n,'U':u,'P':p,'epochs':8,'world_size':4,'microsteps':2,'global_batch':8,
          'total_updates':8*u,'total_exposures':64*u,'extra_padding_exposures':8*p,
          'source_updates':6258,'source_exposures':50064,'endpoint_updates':6258+8*u,'endpoint_exposures':50064+64*u,
          'orders':orders,'padding_window_ids':padding,'window_exposure_counts':counts.tolist(),
          'source_manifest_path':str(SOURCE/'manifest.json'),'source_manifest_sha256':sha256(SOURCE/'manifest.json'),
          'source_plan_path':str(SOURCE/'training_plan.json'),'source_plan_sha256':sha256(SOURCE/'training_plan.json'),
          'source_checkpoint':str(base.SOURCE_CHECKPOINT),'source_checkpoint_sha256':base.SOURCE_SHA256,
          'checkpoint_epochs':[4,8],'latest_each_epoch':True,'seed':42,'alpha':0.01,
          'arms':ARMS,'evaluation_this_turn':False}
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,data in [('manifest.json',manifest),('training_plan.json',plan),('weights_provenance.json',json.loads((SOURCE/'weights_provenance.json').read_text()))]:
        path=ROOT/name
        if path.exists() and json.loads(path.read_text())!=data: raise RuntimeError(f'fixed asset differs: {path}')
        if not path.exists():write_json(path,data)
    for arm,size in ARMS.items():
        (ROOT/arm).mkdir(exist_ok=True);(RUN/arm).mkdir(parents=True,exist_ok=True)
        write_json(ROOT/arm/'run_manifest.json',{'arm':arm,'object_image_memory_size':size,'plan_sha256':sha256(ROOT/'training_plan.json'),
            'source_checkpoint':plan['source_checkpoint'],'source_checkpoint_sha256':base.SOURCE_SHA256,
            'initial_optimizer':'fresh','activation_checkpointing_object_layers':False,'scientific_variable':'object image memory resolution only'})
    return manifest,plan,sha256(ROOT/'training_plan.json')


def init_device():
    world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world not in (1,4) or not os.uname().nodename.startswith('3dimage-11'):raise RuntimeError('requires node11, world1/4')
    torch.cuda.set_device(local)
    if torch.cuda.get_device_name(local)!='NVIDIA GeForce RTX 3090':raise RuntimeError('requires RTX3090')
    if world==4:dist.init_process_group('nccl',timeout=datetime.timedelta(hours=4))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda',local)


def construct(arm,device):
    model,opt,identity=base.build_model(device)
    opt.object_image_memory_size=ARMS[arm];model.opt.object_image_memory_size=ARMS[arm]
    model.anchor_decoder.opt.object_image_memory_size=ARMS[arm]
    model.understanding_step=base.SOURCE_EXPOSURES
    if dist.is_initialized():
        for value in model.state_dict().values():dist.broadcast(value,src=0)
    optimizer=base.build_optimizer(model)
    return model,opt,optimizer,identity


def multiplier(update,total):
    t=update+1
    if not 1<=t<=total:raise ValueError('LR update outside plan')
    return t/200 if t<=200 else .1+.9*(1+math.cos(math.pi*(t-200)/(total-200)))/2


def accumulate(acc, names, rec, under, divisor):
    """Sum GC sample gradients / local microstep count; rank mean follows once."""
    for i,(name,gr,gu) in enumerate(zip(names,rec,under)):
        if gr is None and gu is None:continue
        alpha=.01 if base.family(name)=='reconstruction' else 1.
        if acc[i] is None:
            acc[i]=(gr if gr is not None else gu).detach().clone().zero_()
        if gr is not None:acc[i].add_(gr,alpha=1/divisor)
        if gu is not None:acc[i].add_(gu,alpha=alpha/divisor)


def update(model,opt,optimizer,windows,ids,step,total,single=False):
    rank,world=base.rank_world();device=next(model.parameters()).device
    if not single and world!=4:raise RuntimeError('formal effective batch requires four ranks')
    micros=1 if single else 2; named=sorted(model.named_parameters());names=[n for n,_ in named];params=[p for _,p in named]
    optimizer.zero_grad(set_to_none=True);acc=[None]*len(params);uw=min(8*step/200,1.);exposure=50064+8*step
    for group in optimizer.param_groups:group['lr']=group['peak_lr']*multiplier(step,total)
    logs=[];used=[];begin=time.perf_counter()
    for micro in range(micros):
        wi=ids[0] if single else ids[4*micro+rank];used.append(wi);error=None;metrics=None
        try:
            batch=build_batch(opt,windows[wi],device)
            output,metrics=model.step_loss(batch,step=exposure,understanding_weight=uw)
            from scripts.smoke_object_locus_gc_sweep import finite_prediction
            finite_prediction(output)
            if not all(torch.isfinite(metrics[k]).all() for k in ('loss_recon','loss_understanding')):raise FloatingPointError('nonfinite losses')
        except Exception as exc:error=exc
        base.synchronized_error(error,device,'forward',step,metrics)
        error=None
        try:
            rec=torch.autograd.grad(metrics['loss_recon'],params,allow_unused=True,retain_graph=uw>0)
            under=torch.autograd.grad(uw*metrics['loss_understanding'],params,allow_unused=True) if uw>0 else (None,)*len(params)
            if any(g is not None and not torch.isfinite(g).all() for g in (*rec,*under)):raise FloatingPointError('nonfinite gradient')
            accumulate(acc,names,rec,under,micros)
            logs.append({k:float(v.detach()) for k,v in metrics.items() if torch.is_tensor(v) and v.numel()==1})
            del rec,under,output,metrics,batch
        except Exception as exc:error=exc
        base.synchronized_error(error,device,'autograd',step)
    base.average_gradients_into_params(params,acc) # local /2 and rank /4 => eight-sample mean
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('nonfinite updated parameter')
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    losses={k:sum(row[k] for row in logs)/micros for k in logs[0]}
    if world>1:
        keys=sorted(losses);v=torch.tensor([losses[k] for k in keys],device=device,dtype=torch.float64)
        dist.all_reduce(v);v/=world;losses=dict(zip(keys,v.tolist()))
    return dict(losses=losses,completed_new_updates=step+1,completed_new_exposures=8*(step+1),
        completed_updates=6258+step+1,completed_exposures=50064+8*(step+1),model_exposure=exposure,
        understanding_weight=uw,beta=.1,global_grad_norm=float(norm),rank_windows=used,
        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},seconds=elapsed,
        windows_per_second=(1 if single else 8)/elapsed,peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved())


def save_checkpoint(model,opt,optimizer,arm,epoch,step,plan_sha,code_sha,counts):
    rank,world=base.rank_world();rank_state={'rng':base.capture_rng(),'window_counts':counts}
    states=[None]*world
    if world>1:dist.all_gather_object(states,rank_state)
    else:states=[rank_state]
    if rank==0:
        folder=RUN/arm;path=folder/f'latest_epoch_{epoch:02d}.pt'
        payload={'model':model.state_dict(),'config':dict(dataclasses.asdict(opt),object_image_memory_size=ARMS[arm]),
            'object_image_memory_size':ARMS[arm],'arm':arm,'epoch':epoch,'completed_new_updates':step,
            'completed_new_exposures':8*step,'completed_updates':6258+step,'completed_exposures':50064+8*step,
            'optimizer':optimizer.state_dict(),'scheduler':{'new_update':step,'total_updates':8344,'warmup':200},
            'rank_states':states,'plan_sha256':plan_sha,'git_sha':code_sha,'source_checkpoint_sha256':base.SOURCE_SHA256,
            'world_size':4,'microsteps':2,'alpha':.01}
        atomic_checkpoint(path,payload)
        if epoch==4:
            only={k:v for k,v in payload.items() if k not in ('optimizer','scheduler','rank_states')}
            atomic_checkpoint(folder/'checkpoint_epoch_04_model_only.pt',only)
        if epoch==8:
            endpoint=folder/'checkpoint_epoch_08.pt'
            os.link(path,endpoint)
        for old in folder.glob('latest_epoch_*.pt'):
            if old!=path:old.unlink()
    if world>1:dist.barrier()


def atomic_checkpoint(path,payload):
    tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp)
    check=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False)
    for key in ('completed_new_updates','plan_sha256','git_sha','object_image_memory_size','arm'):
        if check[key]!=payload[key]:raise RuntimeError('checkpoint write verification failed')
    if check['model'].keys()!=payload['model'].keys():raise RuntimeError('checkpoint state keys differ')
    del check;os.replace(tmp,path)


def restore(path,model,optimizer,arm,plan_sha):
    blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if (blob['arm'],blob['object_image_memory_size'],blob['plan_sha256'],blob['world_size'],blob['microsteps'],blob['alpha'])!=(arm,ARMS[arm],plan_sha,4,2,.01):raise RuntimeError('cross-arm/recipe restore forbidden')
    from scripts.object_locus_image_memory_checkpoint import restore_image_memory_configuration
    restore_image_memory_configuration(model,blob)
    model.load_state_dict(blob['model'],strict=True);optimizer.load_state_dict(blob['optimizer'])
    rank,_=base.rank_world();base.restore_rng(blob['rank_states'][rank]['rng'])
    model.understanding_step=blob['completed_exposures']
    return blob['completed_new_updates'],blob['rank_states'][rank]['window_counts']
