"""Paired QPanchor1 warm-start finetuning:8337 windows, three additional epochs; save each epoch."""
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
from tokengs.models.superpoint_structural_variants import MODELS
from scripts.membership_geometry_sdpa import install as install_sdpa
from scripts.qp_anchor_diagnostics import verify_anchor_output
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,digest,MANIFEST,HELDOUT
from scripts.view24_data import build_batch
from scripts.verified_runtime_stage import load_run_config
from scripts.continuation_support import initialize,install_caches,optimizer_audit
from scripts.dual_stream_training_policy import epoch_order,lr_factor
from scripts.dual_stream_frozen_cache import cpu_tree
from scripts.native_joint4_support import task_state,tensor_sha,verify_online_native_inputs,verify_native_unit_bridge,clip_independent_branches,FrozenStateAudit,FrozenStateViolation
from native_scannet.core import schedule


def write(p,d):
    p=Path(p)
    if p.exists():p=p.with_name(p.stem+'_attempt'+os.environ.get('SLURM_JOB_ID','local')+p.suffix)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)


class Task(torch.nn.Module):
    def __init__(self,model,perceptual):
        super().__init__();self.model=model;self.perceptual=perceptual
        object.__setattr__(model,'perceptual',perceptual)
    def forward(self,batch,step,audit=True):
        output,m=self.model.step_loss(batch,step=5215+step,understanding_weight=1.)
        if step<2 and audit:verify_anchor_output(self.model,output['prediction'],step)
        numbers={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in m.items()
                 if isinstance(v,(int,float)) or torch.is_tensor(v) and v.numel()==1}
        if not all(math.isfinite(v) for v in numbers.values()):raise FloatingPointError('Nonfinite metrics')
        # Audit raw per-branch graphs before DDP's multi-output _DDPSink
        # adds scheduling edges between otherwise independent output graphs.
        if step==0 and audit:
            cfg=self.model._anchor_diagnostic_config
            out=Path(cfg['run_root'])/('smoke' if self.model._anchor_smoke else 'train')
            proof=isolation(m['loss_recon'],m['loss_understanding'],self.model)
            proof['location']='inside_Task_forward_before_DDP_output_wrapping'
            write(out/f'ISOLATION_rank{int(os.environ["RANK"]):02d}.json',proof)
        return m['loss'],numbers,m['loss_recon'],m['loss_understanding']


def isolation(lr,lu,model):
    return graph_isolation(lr,lu,list(model.core.reconstruction.parameters()),model.parameter_groups()['understanding'])

from scripts.factorial_contract import graph_isolation


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run=a.run_root;cfg=load_run_config(run);out=run/('smoke' if a.smoke else 'train')
    from scripts.scannet_data_snapshot import verify_job_snapshot
    verify_job_snapshot(cfg)
    rank,local,world=(int(os.environ[n]) for n in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    assert world==cfg['world_size']==4
    accumulation=cfg['gradient_accumulation_steps'];assert accumulation==2 and world*accumulation==cfg['global_batch']==8
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45));renderer();install_sdpa()
    assert cfg['code_sha']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not (out/f'STARTED_rank{rank:02d}.json').exists(), 'final-only policy requires a fresh run directory'
    if not a.smoke:
        gate=json.loads((run/'smoke/SMOKE_PASSED.json').read_text())
        assert gate['checkpoint_policy']=='each_epoch_live_only'
        assert gate['smoke_epoch_offsets']==cfg['smoke_epoch_offsets']
        assert gate['batch_updates_zero_based']==cfg['smoke_batch_updates'] and len(cfg['smoke_batch_updates'])==3 and gate['max_superpoints']==512
        assert gate['status']=='PASSED' and gate['world_size']==world and gate['gradient_accumulation_steps']==accumulation and gate['updates']==3 and gate['code_sha']==cfg['code_sha']
    # Hash the large source once per node, then share proof; it is the common trained four-epoch source.
    if rank==0:
        for path,key in [(cfg['source_checkpoint'],'source_sha256'),(MANIFEST,'manifest_sha256'),(HELDOUT,'heldout_sha256'),(cfg['cohort'],'cohort_sha256')]:assert digest(path)==cfg[key]
        assert digest(os.environ.get('SSST_RUNTIME_VIEW_MANIFEST',cfg['view_manifest']))==cfg['view_manifest_sha256']
    dist.barrier()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    saved=torch.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    opt=config_defaults['train_siu3r_unified_query_v1'].evolve(num_input_views=cfg['context_views'],num_views=cfg['context_views']+2);config=UnifiedQueryConfig(**dict(saved['meta']['config'],max_superpoints=cfg['max_superpoints']))
    from native_scannet.core import options as native_options
    ARCH=cfg['architecture'];model=MODELS[cfg['model_key']](opt,config=config,native_options=native_options().evolve(num_input_views=cfg['context_views'],num_gs_tokens=cfg['token_count'])).to(device)
    model.context_views=cfg['context_views']
    proof=initialize(model,saved,cfg);source_rank_rng=saved['rank_rng'];del saved
    model._anchor_diagnostic_config=cfg;model._anchor_smoke=a.smoke
    write(out/f'INITIALIZATION_rank{rank:02d}.json',proof)
    if a.smoke and rank==0:
        write(out/'NATIVE_RENDER_CONTRACT.json',render_contract(device))
        write(out/'UNIT_BRIDGE_GPU_CONTRACT.json',verify_native_unit_bridge(device))
    dist.barrier()
    cache_counts=install_caches(model,Path(cfg['readonly_frozen_cache']))
    required=cfg['required_update_parameters'];initial={n:tensor_sha(p) for n,p in model.named_parameters() if n in required}
    assert set(initial)==set(required)
    names={n:p for n,p in model.named_parameters() if p.requires_grad};groups=[]
    for family in ['reconstruction','understanding']:
        for wd in [0.,.05]:
            selected=[n for n,p in names.items() if ('reconstruction' if n.startswith('core.reconstruction.') else 'understanding')==family and (0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05)==wd]
            groups.append(dict(params=[names[n] for n in selected],parameter_names=selected,weight_decay=wd,lr=1e-5,family=family))
    assert sum(len(g['params']) for g in groups)==len(names)
    adam=ZeroRedundancyOptimizer(groups,optimizer_class=torch.optim.AdamW,lr=1e-5,betas=(.9,.95),eps=1e-8,foreach=False)
    import lpips
    perceptual=lpips.LPIPS(net='vgg',verbose=False).to(device).eval().requires_grad_(False)
    task=Task(model,perceptual)
    frozen_audit=FrozenStateAudit(model)
    ddp=DistributedDataParallel(task,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    write(out/f'FROZEN_DDP_INIT_rank{rank:02d}.json',frozen_audit.after_ddp())
    meta=dict(cfg,architecture=ARCH,config=config.to_dict(),omega_identity=model.frozen_omega.source_identity,
              sam_identity=model.sam_proposals.source_identity,epochs=3,world_size=world,gradient_accumulation_steps=accumulation,global_batch=8,source_updates=5215,source_epochs=5,finetune_updates=3129)
    write(out/f'STARTED_rank{rank:02d}.json',meta)
    windows,_,_=cohorts();orders=[epoch_order(len(windows),e) for e in (5,6,7)]
    assert all(len(v)==8344 for v in orders)
    # EMA lives on CPU: no extra native-width model on the 24GB training GPU.
    ema=None # Storage-only policy: live final weights scored; no EMA affects optimization.
    if rank==0:write(out/'INITIALIZATION_IDENTITY.json',dict(seed=42,source_sha256=cfg['source_sha256'],required_parameter_sha256=initial,zero_optimizer=True,architecture=ARCH))
    dist.barrier()
    from torch.utils.tensorboard import SummaryWriter
    from scripts.factorial_resume import rng as checkpoint_rng,restore_rng
    before_writer=checkpoint_rng()
    writer=SummaryWriter(str(run/('smoke_tensorboard' if a.smoke else 'tensorboard'))) if rank==0 else None
    restore_rng(source_rank_rng[rank]);del source_rank_rng
    write(out/f'SOURCE_RNG_RESTORED_rank{rank:02d}.json',dict(status='PASSED',source_cumulative_epochs=5,rank=rank,restored_after_initialization=True))
    log=(out/f'updates_rank{rank:02d}.jsonl').open('x');limit=3 if a.smoke else 3129;started=time.monotonic();active=set()
    def snapshot(completed):
        local_epoch=completed//1043
        epoch_out=out/('smoke_endpoint' if a.smoke else f'epoch{5+local_epoch:02d}')
        epoch_out.mkdir(parents=True,exist_ok=True)
        progress=dict(epoch=local_epoch,update_in_epoch=0 if not a.smoke else completed,full_updates=completed)
        write(epoch_out/f'FROZEN_ENDPOINT_rank{rank:02d}.json',frozen_audit.finish())
        changed={n:tensor_sha(names[n])!=initial[n] for n in required};assert all(changed.values())
        digest_model=hashlib.sha256()
        for n,p in model.named_parameters():
            if n in names:digest_model.update(n.encode());digest_model.update(p.detach().cpu().contiguous().numpy().tobytes())
        hashes=[None]*world;dist.all_gather_object(hashes,digest_model.hexdigest());assert len(set(hashes))==1
        write(epoch_out/f'UPDATE_AUDIT_rank{rank:02d}.json',dict(status='PASSED',updates=completed,required_changed=changed,
            all_parameter_sha256=hashes[0],DDP_replicas_equal=True,frozen_versions_equal=True,active_parameter_names=sorted(active)))
        optproof=optimizer_audit(adam,completed)
        write(epoch_out/f'OPTIMIZER_ENDPOINT_rank{rank:02d}.json',optproof)
        from scripts.train_unified_query_v1 import rng
        states=[None]*world;dist.all_gather_object(states,rng())
        if rank==0 and not a.smoke:
            import fcntl
            with open(cfg['checkpoint_write_lock'],'a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                from scripts.factorial_final_checkpoint import save_final
                file_record=save_final(dict(model=cpu_tree(task_state(model)),progress=progress,meta=dict(meta,epochs=local_epoch,cumulative_epochs=5+local_epoch,cumulative_updates=5215+completed,finetune_updates=completed),rank_rng=states,optimizer_audit=dict(status='PASSED',step=completed,finite=True,optimizer_saved=False)),epoch_out/'checkpoint_live.pt')
                write(epoch_out/'FINAL_CHECKPOINT_STORAGE.json',dict(status='PASSED',policy='each_epoch_live_only',optimizer_saved=False,**file_record))
            final=torch.load(epoch_out/'checkpoint_live.pt',map_location='cpu',mmap=True,weights_only=False)
            verify_hash=hashlib.sha256()
            for n in names:verify_hash.update(n.encode());verify_hash.update(final['model'][n].contiguous().numpy().tobytes())
            assert verify_hash.hexdigest()==hashes[0]
            write(epoch_out/'OPTIMIZER_ENDPOINT_AUDIT.json',dict(status='PASSED',step=completed,finite=True,all_parameter_sha256=hashes[0],optimizer_saved=False))
            del final
        if rank==0:
            write(epoch_out/'EPOCH_COMPLETE.json',dict(status='PASSED',progress=progress,cumulative_epochs=5+local_epoch,cumulative_updates=5215+completed,optimizer_reset_after_epoch5=True,optimizer_continuous_across_added_epochs=True,code_sha=cfg['code_sha']))
        dist.barrier()
    for step in range(limit):
        tick=time.monotonic();model.train();adam.zero_grad(set_to_none=True)
        from scripts.factorial_accumulation import sample_indices,accumulate
        batch_update=cfg['smoke_batch_updates'][step] if a.smoke else step%1043
        indices=sample_indices(orders[cfg['smoke_epoch_offsets'][step] if a.smoke else step//1043],batch_update,rank,world,accumulation)
        index=indices[0]
        def first_batch(batch):
            if a.smoke and step==0:
                from scripts.smoke_v4init_isolated import validate_batch_contract
                assert batch['images_input'].shape==(1,cfg['context_views'],3,256,256)
                assert batch['images_all'].shape==(1,cfg['training_supervised_views'],3,256,256)
                assert batch['frame_ids'][0].cpu().tolist()==batch['generation_frame_ids'][0].cpu().tolist()+windows[index]['novel']
                write(out/f'BATCH_CONTRACT_rank{rank:02d}.json',dict(status='PASSED',generation_frames=batch['generation_frame_ids'][0].cpu().tolist(),supervised_frames=batch['frame_ids'][0].cpu().tolist(),context_views=cfg['context_views'],extra_context_GT_supervised=cfg['context_views']==4,understanding_supervised_views=batch['understanding_context_views']))
                write(out/f'ONLINE_NATIVE_INPUT_rank{rank:02d}.json',verify_online_native_inputs(model,batch['images_input']))
        def release_graph():
            model.core.reconstruction.anchor_final=None;model.core.reconstruction.anchor_seed=None
        weights=dict(schedule(4171,4172),lr=1e-5)
        for group in adam.param_groups:group['lr']=1e-5
        batches=(build_batch(opt,windows[i],device) for i in indices)
        metrics=accumulate(ddp,batches,step,accumulation,first_batch=first_batch,after_backward=release_graph)
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
        model.core.reconstruction.anchor_final=None;model.core.reconstruction.anchor_seed=None
        torch.cuda.synchronize()
        metrics.update(update=step+1,epoch=step//1043,source_index=index,source_indices=indices,batch_update_zero_based=batch_update,gradient_accumulation_steps=accumulation,seconds=time.monotonic()-tick,gradient_norm=norm,
                       branch_gradient_norms=branch_norms,understanding_weight=1.,peak_allocated_bytes=torch.cuda.max_memory_allocated(),schedule=weights)
        log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
        keys=['loss','loss_recon','loss_understanding','psnr','l1','seconds','gradient_norm']
        mean=torch.tensor([metrics[k] for k in keys],dtype=torch.float64,device=device);dist.all_reduce(mean);mean/=world
        if writer is not None:
            for k,v in zip(keys,mean.tolist()):writer.add_scalar('train/'+k,v,step+1)
            writer.add_scalar('lr/native',weights['lr'],step+1)
            writer.add_scalar('lr/understanding',1e-5,step+1)
            if (step+1)%10==0:writer.flush()
        if rank==0 and (a.smoke or (step+1)%10==0):print(json.dumps(metrics),flush=True)
        if not a.smoke and (step+1)%1043==0:snapshot(step+1)
    log.close()
    if a.smoke:snapshot(limit)
    write(out/f'CACHE_AUDIT_rank{rank:02d}.json',dict(status='PASSED',counts=cache_counts,shared_cache_modified=False,smoke=a.smoke))
    if a.smoke:
        assert cache_counts['sam_parity_verified']+cache_counts['sam_online_determinism_verified']>=1, 'GPU SAM consistency not verified'
    if rank==0:
        write(out/('SMOKE_PASSED.json' if a.smoke else 'TRAINING_COMPLETE.json'),dict(status='PASSED',code_sha=cfg['code_sha'],updates=limit,
            world_size=world,gradient_accumulation_steps=accumulation,global_batch=8,full_updates=limit,max_superpoints=config.max_superpoints,
            smoke_epoch_offsets=cfg['smoke_epoch_offsets'] if a.smoke else None,batch_updates_zero_based=cfg['smoke_batch_updates'] if a.smoke else list(range(limit)),elapsed_seconds=time.monotonic()-started,
            checkpoint_policy='each_epoch_live_only',optimizer_saved=False,mid_epoch_checkpoints=False,
            frozen_updates=0,gradient_isolation_verified=True,cache_counts_rank0=cache_counts,
            smoke_weights_used_for_training=False,progress=dict(epoch=0 if a.smoke else 3,update_in_epoch=limit if a.smoke else 0,full_updates=limit)))
        writer.close()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
