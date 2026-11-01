"""Eight-rank zero-only full4 training; separate two-step gate never supplies weights."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json,math,os,random,subprocess,time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.reconstruction4096_scratch_policy import initialize,optimizer,KEY
from scripts.reconstruction_abcd_policy import task_state,tensor_sha
from scripts.train_unified_query_v1 import rng,restore_rng
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order,lr_factor
from scripts.dual_stream_frozen_cache import atomic_torch_save,cpu_tree
from scripts.membership_geometry_sdpa import install,evidence
from scripts.smoke_v4init_isolated import validate_batch_contract
from tokengs.options import config_defaults
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def fingerprints(model,trainable):
    return {n:tensor_sha(p) for n,p in model.named_parameters()
            if not n.startswith(('frozen_omega.','sam_proposals.')) and p.requires_grad==trainable}


class Task(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model;self.last=None
    def forward(self,batch,step,weight):
        output,metrics=self.model.step_loss(batch,step=step,understanding_weight=weight)
        pred=output['prediction'];self.last=pred
        if pred['gaussians'].shape!=(1,262144,14):raise ValueError('Wrong Gaussian shape')
        parent=pred['gaussian_parent_anchor']
        if not torch.equal(parent,torch.arange(4096,device=parent.device).repeat_interleave(64)):
            raise ValueError('Parent index mismatch')
        if pred['anchor_point_indices'].unique().numel()!=4096:raise ValueError('Duplicated FPS sources')
        numbers={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items()
                 if isinstance(v,(int,float)) or torch.is_tensor(v) and v.numel()==1}
        if not all(math.isfinite(v) for v in numbers.values()):raise FloatingPointError('Nonfinite metrics')
        return metrics['loss'],numbers,metrics['loss_recon'],metrics['loss_understanding']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run=args.run_root;cfg=json.loads((run/'configuration.json').read_text())
    out=run/('smoke' if args.smoke else 'train');out.mkdir(exist_ok=True)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8:raise ValueError('Exactly eight ranks required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    install();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if cfg['code_sha']!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip():
        raise ValueError('Code identity differs')
    if (out/'STARTED.json').exists() or (out/'checkpoint_latest.pt').exists():raise ValueError('Refusing duplicate start')
    if not args.smoke:
        gate=json.loads((run/'smoke/SMOKE_PASSED.json').read_text())
        if gate['status']!='PASSED' or gate['code_sha']!=cfg['code_sha'] or gate['updates']!=2 or gate['world_size']!=8:
            raise ValueError('Smoke gate differs')
    if digest(cfg['source_checkpoint'])!=cfg['source_sha256']:raise ValueError('Source SHA differs')
    for key,path in [('manifest_sha256',MANIFEST),('heldout_sha256',HELDOUT),('cohort_sha256',cfg['cohort'])]:
        if digest(path)!=cfg[key]:raise ValueError('Data identity differs: '+key)
    loaded=Path(sys.modules['gsplat_cuda'].__file__).resolve()
    if loaded!=Path(cfg['renderer']).resolve() or digest(loaded)!=cfg['renderer_sha256']:
        raise ValueError('Pinned renderer differs')
    saved=torch.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    opt=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig(**saved['meta']['config'])
    model=build_structural_model(KEY,opt,config).to(device)
    init=initialize(model,saved,cfg);adam=optimizer(model)
    write(out/f'INITIALIZATION_rank{rank:02d}.json',dict(init,source_sha256=cfg['source_sha256'],optimizer_empty=not adam.state,
          frozen_omega=model.frozen_omega.source_identity,sam=model.sam_proposals.source_identity))
    restore_rng(saved['rank_rng'][rank]);del saved
    initial=fingerprints(model,True);protected=fingerprints(model,False)
    if rank==0:write(out/'STARTED.json',cfg)
    task=Task(model);preserved=rng();ddp=DDP(task,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True);restore_rng(preserved)
    windows,_,_=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    total=4172;limit=2 if args.smoke else total;added=0;start=time.monotonic()
    meta=dict(cfg,architecture=model.architecture_name,config=config.to_dict(),omega_identity=model.frozen_omega.source_identity,
              sam_identity=model.sam_proposals.source_identity,epochs=4,short_validation=False,initial_updates=0,smoke_only=args.smoke)
    writer=None
    if rank==0 and not args.smoke:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(run/'tensorboard'),flush_secs=15)
        writer.add_text('run/config',json.dumps(meta,indent=1),0);writer.flush()
    def save():
        states=[None]*8;dist.all_gather_object(states,rng())
        if rank==0:
            progress=dict(epoch=added//1043,update_in_epoch=added%1043,full_updates=added)
            atomic_torch_save(dict(model=cpu_tree(task_state(model)),optimizer=cpu_tree(adam.state_dict()),
                                  progress=progress,rank_rng=states,meta=meta),out/'checkpoint_latest.pt')
        dist.barrier()
    if not args.smoke:
        save()
        if rank==0:os.link(out/'checkpoint_latest.pt',out/'checkpoint_initial.pt')
        dist.barrier()
    log=(out/f'updates_rank{rank:02d}.jsonl').open('x')
    for step in range(limit):
        started=time.monotonic();torch.cuda.reset_peak_memory_stats();model.train();adam.zero_grad(set_to_none=True)
        for group in adam.param_groups:group['lr']=group['base_lr']*lr_factor('full',step,total)
        order=epoch_order(len(windows),step//1043);index=order[(step%1043)*8+rank]
        batch=build_batch(opt,windows[index],device)
        if step==0:write(out/f'BATCH_CONTRACT_rank{rank:02d}.json',validate_batch_contract(batch,windows[index]))
        loss,row,lr,lu=ddp(batch,step*8,min(1.,(step+1)/1043))
        pred=task.last
        if args.smoke and step==0:
            groups=model.parameter_groups();rec=groups['reconstruction'];other=groups['understanding']+groups['shared']
            def leak(objective,params):
                gradients=torch.autograd.grad(objective,params,retain_graph=True,allow_unused=True)
                return sum(int(torch.count_nonzero(g)) for g in gradients if g is not None)
            lu_rec=leak(lu,rec);lr_other=leak(lr,other)
            if lu_rec or lr_other:raise ValueError('Gradient isolation violation')
            write(out/f'ISOLATION_rank{rank:02d}.json',dict(status='PASSED',LU_reconstruction_nonzero=lu_rec,LR_understanding_shared_nonzero=lr_other))
            write(out/f'SHAPES_rank{rank:02d}.json',dict(gaussians=list(pred['gaussians'].shape),unique_sources=4096,
                  gpu=torch.cuda.get_device_name(local),understanding_queries=pred['num_thing_queries']))
        loss.backward()
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen tensor received gradient')
        if args.smoke or step==0:
            norms={}
            for family,params in model.parameter_groups().items():
                value=sum(float(p.grad.detach().float().square().sum()) for p in params if p.grad is not None)**.5
                if not math.isfinite(value) or value<=0:raise ValueError('Missing gradient: '+family)
                norms[family]=value
            grad=model.core.v4_initialization.gs_tokens.grad
            norms['new_token_rows']=float(grad[1024:].norm()) if grad is not None else 0.
            if not math.isfinite(norms['new_token_rows']) or norms['new_token_rows']<=0:raise ValueError('New rows have no gradient')
            write(out/f'GRADIENTS_{step+1}_rank{rank:02d}.json',dict(status='PASSED',norms=norms))
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        if not math.isfinite(norm) or norm<=0:raise ValueError('Invalid gradient norm')
        adam.step();added=step+1;task.last=None;del pred,loss,lr,lu,batch
        torch.cuda.synchronize()
        row.update(added_updates=added,epoch=step//1043,source_index=index,seconds=time.monotonic()-started,
                   gradient_norm=norm,understanding_weight=min(1.,added/1043),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),learning_rates={g['family']:g['lr'] for g in adam.param_groups})
        log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
        # All eight ranks contribute to TensorBoard; JSONL retains individual rank evidence.
        keys=('loss','loss_recon','loss_understanding','psnr','psnr_layer6','psnr_layer12',
              'gradient_norm','seconds','understanding_weight','num_reconstruction_anchors','num_gaussians','peak_allocated_bytes')
        values=torch.tensor([row[k] for k in keys],dtype=torch.float64,device=device);dist.all_reduce(values);values/=world
        if writer is not None:
            for k,v in zip(keys,values.tolist()):writer.add_scalar('train/'+k,v,added)
            for g in adam.param_groups:writer.add_scalar('lr/'+g['family'],g['lr'],added)
            if added%10==0:writer.flush()
        if rank==0 and (args.smoke or step%10==0):print(json.dumps(row),flush=True)
        if step==0 or added==limit:
            current=fingerprints(model,True);changed=[n for n in current if current[n]!=initial[n]]
            if not changed or fingerprints(model,False)!=protected:raise ValueError('No updates or protected parameters changed')
            hashes=[None]*8;dist.all_gather_object(hashes,current)
            if any(h!=current for h in hashes):raise ValueError('DDP replicas differ')
            write(out/f'UPDATE_AUDIT_{added}_rank{rank:02d}.json',dict(status='PASSED',changed_names=changed,
                  protected_tensors_equal=True,ddp_replicas_equal=True,parameter_fingerprints=current))
        if not args.smoke and (added%100==0 or added%1043==0):save()
    log.close()
    ranks=[None]*8;dist.all_gather_object(ranks,dict(rank=rank,updates=added,finite=True,protected_unchanged=True))
    if rank==0:write(out/('SMOKE_PASSED.json' if args.smoke else 'TRAINING_COMPLETE.json'),
                     dict(status='PASSED',code_sha=cfg['code_sha'],world_size=8,global_batch=8,updates=added,full_updates=added,
                          ranks=ranks,elapsed_seconds=time.monotonic()-start,attention_backend=evidence(),smoke_weights_used_for_formal=False))
    if writer is not None:writer.flush();writer.close()
    dist.destroy_process_group()


if __name__=='__main__':main()
