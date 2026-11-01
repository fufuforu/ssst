"""Fresh native reconstruction + baseline understanding, 8337 windows x4 epochs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,datetime,hashlib,json,math,os,random,subprocess,time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer
from native_scannet.runtime import renderer
from native_scannet.render_contract import verify as render_contract
from tokengs.models.querysplat_native_joint4 import QuerySplatNativeJoint4,ARCH
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order,lr_factor
from scripts.dual_stream_frozen_cache import cpu_tree,atomic_torch_save
from scripts.native_joint4_support import task_state,initialize_zero,install_caches,tensor_sha,verify_online_native_inputs,verify_native_unit_bridge,clip_independent_branches,FrozenStateAudit,FrozenStateViolation
from native_scannet.core import schedule


def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)


class Task(torch.nn.Module):
    def __init__(self,model,perceptual):
        super().__init__();self.model=model;self.perceptual=perceptual
        object.__setattr__(model,'perceptual',perceptual)
    def forward(self,batch,step):
        _,m=self.model.step_loss(batch,step=step,understanding_weight=min(1.,(step+1)/1043))
        numbers={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in m.items()
                 if isinstance(v,(int,float)) or torch.is_tensor(v) and v.numel()==1}
        if not all(math.isfinite(v) for v in numbers.values()):raise FloatingPointError('Nonfinite metrics')
        return m['loss'],numbers,m['loss_recon'],m['loss_understanding']


def isolation(lr,lu,model):
    rec=list(model.core.reconstruction.parameters());under=model.parameter_groups()['understanding']
    cross_u=torch.autograd.grad(lu,rec,retain_graph=True,allow_unused=True)
    cross_r=torch.autograd.grad(lr,under,retain_graph=True,allow_unused=True)
    assert all(g is None or not g.count_nonzero() for g in cross_u+cross_r),'Cross-branch gradient leakage'
    return dict(status='PASSED',LU_to_reconstruction_zero=True,LR_to_understanding_zero=True,
                reconstruction_tensors=len(rec),understanding_tensors=len(under))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run=a.run_root;cfg=json.loads((run/'configuration.json').read_text());out=run/('smoke' if a.smoke else 'train')
    rank,local,world=(int(os.environ[n]) for n in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    assert world==8;torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45));renderer()
    assert cfg['code_sha']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not (out/f'STARTED_rank{rank:02d}.json').exists() and not (out/'checkpoint_latest.pt').exists()
    if not a.smoke:
        gate=json.loads((run/'smoke/SMOKE_PASSED.json').read_text())
        assert gate['status']=='PASSED' and gate['world_size']==8 and gate['updates']==2 and gate['code_sha']==cfg['code_sha']
    # Hash the large source once per node, then share proof; it is a ZERO source.
    if rank==0:
        for path,key in [(cfg['source_checkpoint'],'source_sha256'),(MANIFEST,'manifest_sha256'),(HELDOUT,'heldout_sha256'),(cfg['cohort'],'cohort_sha256')]:assert digest(path)==cfg[key]
    dist.barrier()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    saved=torch.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    opt=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig(**saved['meta']['config'])
    model=QuerySplatNativeJoint4(opt,config=config).to(device)
    proof=initialize_zero(model,saved,cfg);del saved
    write(out/f'INITIALIZATION_rank{rank:02d}.json',proof)
    if a.smoke and rank==0:
        write(out/'NATIVE_RENDER_CONTRACT.json',render_contract(device))
        write(out/'UNIT_BRIDGE_GPU_CONTRACT.json',verify_native_unit_bridge(device))
    dist.barrier()
    cache_counts=install_caches(model,run/'frozen_cache')
    required=cfg['required_update_parameters'];initial={n:tensor_sha(p) for n,p in model.named_parameters() if n in required}
    assert set(initial)==set(required)
    names={n:p for n,p in model.named_parameters() if p.requires_grad};groups=[]
    for family in ['reconstruction','understanding']:
        for wd in [0.,.05]:
            selected=[n for n,p in names.items() if ('reconstruction' if n.startswith('core.reconstruction.') else 'understanding')==family and (0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05)==wd]
            groups.append(dict(params=[names[n] for n in selected],parameter_names=selected,weight_decay=wd,lr=1e-4,family=family))
    assert sum(len(g['params']) for g in groups)==len(names)
    adam=ZeroRedundancyOptimizer(groups,optimizer_class=torch.optim.AdamW,lr=1e-4,betas=(.9,.95),eps=1e-8,foreach=False)
    import lpips
    perceptual=lpips.LPIPS(net='vgg',verbose=False).to(device).eval().requires_grad_(False)
    task=Task(model,perceptual)
    frozen_audit=FrozenStateAudit(model)
    ddp=DistributedDataParallel(task,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    write(out/f'FROZEN_DDP_INIT_rank{rank:02d}.json',frozen_audit.after_ddp())
    meta=dict(cfg,architecture=ARCH,config=config.to_dict(),omega_identity=model.frozen_omega.source_identity,
              sam_identity=model.sam_proposals.source_identity,epochs=4,world_size=8,global_batch=8)
    write(out/f'STARTED_rank{rank:02d}.json',meta)
    windows,_,_=cohorts();orders=[epoch_order(len(windows),e) for e in range(4)]
    assert all(len(v)==8344 for v in orders)
    # EMA lives on CPU: no extra native-width model on the 24GB training GPU.
    ema={n:p.detach().cpu().clone() for n,p in names.items()} if rank==0 and not a.smoke else None
    if rank==0 and not a.smoke:atomic_torch_save(dict(model=cpu_tree(task_state(model)),progress=dict(epoch=0,update_in_epoch=0,full_updates=0),meta=meta),out/'checkpoint_initial.pt')
    dist.barrier()
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(run/('smoke_tensorboard' if a.smoke else 'tensorboard'))) if rank==0 else None
    log=(out/f'updates_rank{rank:02d}.jsonl').open('x');limit=2 if a.smoke else 4172;started=time.monotonic();active=set()
    for step in range(limit):
        tick=time.monotonic();model.train();adam.zero_grad(set_to_none=True)
        index=orders[step//1043][(step%1043)*8+rank];batch=build_batch(opt,windows[index],device)
        if a.smoke and step==0:
            from scripts.smoke_v4init_isolated import validate_batch_contract
            write(out/f'BATCH_CONTRACT_rank{rank:02d}.json',validate_batch_contract(batch,windows[index]))
            write(out/f'ONLINE_NATIVE_INPUT_rank{rank:02d}.json',verify_online_native_inputs(model,batch['images_input']))
        weights=schedule(step,4172)
        for group in adam.param_groups:group['lr']=weights['lr'] if group['family']=='reconstruction' else 1e-4*lr_factor('full',step,4172)
        loss,metrics,lr,lu=ddp(batch,step)
        if step==0:
            write(out/f'ISOLATION_rank{rank:02d}.json',isolation(lr,lu,model))
        loss.backward()
        if step in (0,limit-1):
            grads={n:float(names[n].grad.norm()) if names[n].grad is not None else 0. for n in required}
            assert all(math.isfinite(v) and v>0 for v in grads.values()),str(grads)
            write(out/f'GRADIENTS_{step+1}_rank{rank:02d}.json',dict(status='PASSED',gradients=grads))
        active.update(n for n,p in names.items() if p.grad is not None)
        branch_norms=clip_independent_branches(adam)
        norm=math.sqrt(sum(v*v for v in branch_norms.values()));assert math.isfinite(norm)
        adam.step()
        if ema is not None:
            with torch.no_grad():
                for n,p in names.items():ema[n].lerp_(p.detach().cpu(),1-.9995)
        try:frozen_audit.verify_step()
        except FrozenStateViolation as error:
            write(out/f'FROZEN_FAILURE_rank{rank:02d}.json',dict(status='FAILED',update=step+1,details=error.details));raise
        del loss,lr,lu,batch;torch.cuda.synchronize()
        metrics.update(update=step+1,epoch=step//1043,source_index=index,seconds=time.monotonic()-tick,gradient_norm=norm,
                       branch_gradient_norms=branch_norms,understanding_weight=min(1.,(step+1)/1043),peak_allocated_bytes=torch.cuda.max_memory_allocated(),schedule=weights)
        log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
        keys=['loss','loss_recon','loss_understanding','psnr','l1','seconds','gradient_norm']
        mean=torch.tensor([metrics[k] for k in keys],dtype=torch.float64,device=device);dist.all_reduce(mean);mean/=world
        if writer is not None:
            for k,v in zip(keys,mean.tolist()):writer.add_scalar('train/'+k,v,step+1)
            writer.add_scalar('lr/native',weights['lr'],step+1)
            writer.add_scalar('lr/understanding',1e-4*lr_factor('full',step,4172),step+1)
            if (step+1)%10==0:writer.flush()
        if rank==0 and (a.smoke or (step+1)%10==0):print(json.dumps(metrics),flush=True)
        if not a.smoke and (step+1)%1043==0:
            # Consolidate and save at epoch boundaries only, rolling atomic
            # checkpoint; all ranks participate in optimizer-state gathering.
            adam.consolidate_state_dict(to=0)
            if rank==0:
                optim=adam.state_dict()
                atomic_torch_save(dict(model=cpu_tree(task_state(model)),optimizer=optim,ema=ema,
                    progress=dict(epoch=(step+1)//1043,update_in_epoch=0,full_updates=step+1),meta=meta),out/'checkpoint_latest.pt')
                del optim
            dist.barrier()
    log.close()
    write(out/f'FROZEN_ENDPOINT_rank{rank:02d}.json',frozen_audit.finish())
    changed={n:tensor_sha(names[n])!=initial[n] for n in required};assert all(changed.values())
    digest_model=hashlib.sha256()
    for n,p in model.named_parameters():
        if n in names:digest_model.update(n.encode());digest_model.update(p.detach().cpu().contiguous().numpy().tobytes())
    hashes=[None]*8;dist.all_gather_object(hashes,digest_model.hexdigest());assert len(set(hashes))==1
    write(out/f'UPDATE_AUDIT_rank{rank:02d}.json',dict(status='PASSED',updates=limit,required_changed=changed,
        all_parameter_sha256=hashes[0],DDP_replicas_equal=True,frozen_versions_equal=True,active_parameter_names=sorted(active)))
    if rank==0:
        if not a.smoke:
            saved=torch.load(out/'checkpoint_latest.pt',map_location='cpu',mmap=True,weights_only=False)
            recids=[i for g in saved['optimizer']['param_groups'] for n,i in zip(g['parameter_names'],g['params']) if n.startswith('core.reconstruction.')]
            assert all(int(saved['optimizer']['state'][i]['step'])==4172 for i in recids)
        write(out/('SMOKE_PASSED.json' if a.smoke else 'TRAINING_COMPLETE.json'),dict(status='PASSED',code_sha=cfg['code_sha'],updates=limit,
            world_size=8,global_batch=8,full_updates=limit,elapsed_seconds=time.monotonic()-started,
            frozen_updates=0,gradient_isolation_verified=True,cache_counts_rank0=cache_counts,
            smoke_weights_used_for_training=False,progress=dict(epoch=0 if a.smoke else 4,update_in_epoch=limit if a.smoke else 0,full_updates=limit)))
        writer.close()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
