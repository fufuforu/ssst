"""Independent unified-query training entry. Does not submit jobs.

Requires an explicit understanding loss weight; the former dual-stream gradient
isolation recipe is intentionally not silently reused for the shared trunk.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,contextlib,json,math,os,random,subprocess,time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tokengs.options import config_defaults
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches,lr_factor
from scripts.dual_stream_frozen_cache import atomic_torch_save,cpu_tree


def trainable_state(model):return cpu_tree({k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))})


def restore_state(model,state):
    target=model.state_dict();keys={k for k in target if not k.startswith(('frozen_omega.','sam_proposals.'))}
    if set(state)!=keys or any(target[k].shape!=state[k].shape for k in keys):raise ValueError('Unified checkpoint architecture differs')
    with torch.no_grad():
        for k in keys:target[k].copy_(state[k])


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


class Task(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,batch,step,weight):
        _,metrics=self.model.step_loss(batch,step=step,understanding_weight=weight)
        numbers={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items() if isinstance(v,(int,float)) or torch.is_tensor(v) and v.numel()==1}
        if not all(math.isfinite(v) for v in numbers.values()):raise FloatingPointError('Nonfinite unified loss/metric')
        return metrics['loss'],numbers


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--understanding-weight',type=float,required=True)
    parser.add_argument('--architecture',choices=('unified','dual_superpoint','unified_direct'),default='unified')
    parser.add_argument('--epochs',type=int,default=4);parser.add_argument('--max-updates',type=int)
    parser.add_argument('--learning-rate',type=float,default=1e-4);parser.add_argument('--understanding-warmup',type=int,default=1043)
    parser.add_argument('--short-validation',action='store_true',help='16 fixed training scenes, 256 updates, initial/final probes')
    parser.add_argument('--epoch-probes',action='store_true',help='Fixed train/heldout probes initially and at each full epoch')
    parser.add_argument('--initial-checkpoint',type=Path);parser.add_argument('--initial-sha256')
    parser.add_argument('--config',type=Path);parser.add_argument('--resume',action='store_true');parser.add_argument('--checkpoint-every',type=int,default=50)
    args=parser.parse_args()
    if args.understanding_weight<0 or not math.isfinite(args.understanding_weight) or args.epochs<1 or args.learning_rate<=0 or args.understanding_warmup<0 or args.checkpoint_every<1 or args.max_updates is not None and args.max_updates<1:parser.error('Invalid training settings')
    if bool(args.initial_checkpoint)!=bool(args.initial_sha256) or args.initial_checkpoint and (args.resume or args.short_validation):parser.error('Zero-step initialization requires SHA and fresh full training')
    if args.short_validation:
        if args.resume or args.max_updates is not None:parser.error('Short validation uses a fresh bounded run')
        args.epochs=128;args.max_updates=256;args.understanding_warmup=20
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world not in (1,2,4,8):raise ValueError('World size must divide fixed global batch8')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=UnifiedQueryConfig(**json.loads(args.config.read_text())) if args.config else UnifiedQueryConfig()
    opt=config_defaults['train_siu3r_unified_query_v1'];windows,chosen,validation=cohorts()
    if args.short_validation:windows=[windows[i] for i in chosen]
    from tokengs.models.superpoint_structural_variants import build_structural_model
    model=build_structural_model(args.architecture,opt,config).to(device)
    if args.initial_checkpoint:
        if digest(args.initial_checkpoint)!=args.initial_sha256:raise ValueError('Zero-step checkpoint SHA differs')
        from scripts.unified_query_initial_state import validate_zero_initial
        initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        validate_zero_initial(initial,model,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=world)
        restore_state(model,initial['model']);restore_rng(initial['rank_rng'][rank]);del initial
    grouped={}
    for name,p in model.named_parameters():
        if p.requires_grad:grouped.setdefault(0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,weight_decay=wd) for wd,ps in grouped.items()],lr=args.learning_rate,betas=(.9,.95),eps=1e-8)
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    meta=dict(architecture=model.architecture_name,config=config.to_dict(),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),omega_identity=model.frozen_omega.source_identity,sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None,
        initial_checkpoint=str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None,initial_sha256=args.initial_sha256,initial_updates=0,epoch_probes=args.epoch_probes,short_validation=args.short_validation,training_windows=windows if args.short_validation else None,world_size=world,global_batch=8,epochs=args.epochs,understanding_weight=args.understanding_weight,understanding_warmup=args.understanding_warmup,learning_rate=args.learning_rate)
    progress=dict(epoch=0,update_in_epoch=0,full_updates=0)
    if args.resume:
        saved=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
        for key in ('initial_checkpoint','initial_sha256','initial_updates'):meta[key]=saved['meta'][key]
        if saved['meta']!=meta:raise ValueError('Resume source/config/code/data differ')
        restore_state(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);progress=saved['progress'];restore_rng(saved['rank_rng'][rank]);del saved
    else:
        refusal=['Refusing duplicate run; use own --resume' if rank==0 and ((root/'STARTED.json').exists() or (root/'checkpoint_latest.pt').exists()) else None]
        if world>1:dist.broadcast_object_list(refusal,src=0)
        if refusal[0]:raise ValueError(refusal[0])
        if rank==0:(root/'STARTED.json').write_text(json.dumps(meta,indent=2)+'\n')
        if world>1:dist.barrier()
    if args.max_updates is not None and progress['full_updates']>=args.max_updates:raise ValueError('Requested update limit already reached; no updates replayed')
    if rank==0:(root/'last_request.json').write_text(json.dumps(dict(resume=args.resume,max_updates=args.max_updates),indent=2)+'\n')
    from scripts.unified_query_validation import run_probes,parameter_digest,synchronized_first_update
    probe_train=windows if args.short_validation else [windows[i] for i in chosen]
    if not args.resume and (args.short_validation or args.epoch_probes):run_probes(model,opt,probe_train,validation,root,'initial',rank,world,device,args.understanding_weight)
    initial_digest=parameter_digest(model)
    before_ddp=rng()
    task=Task(model);task=DDP(task,device_ids=[local],find_unused_parameters=True) if world>1 else task
    restore_rng(before_ddp)
    updates_per_epoch=len(epoch_order(len(windows),0))//8;total=args.epochs*updates_per_epoch
    if args.resume and progress['full_updates']>=total:raise ValueError('Training already complete; no updates replayed')
    def save():
        local_rng=rng();states=[local_rng]
        if world>1:states=[None]*world;dist.all_gather_object(states,local_rng)
        if rank==0:atomic_torch_save(dict(model=trainable_state(model),optimizer=cpu_tree(optimizer.state_dict()),progress=dict(progress),rank_rng=states,meta=meta),root/'checkpoint_latest.pt')
        if world>1:dist.barrier()
    if not args.resume:
        save()
        if rank==0:os.link(root/'checkpoint_latest.pt',root/'checkpoint_initial.pt')
        if world>1:dist.barrier()
    stop=False
    for epoch in range(progress['epoch'],args.epochs):
        order=epoch_order(len(windows),epoch)
        for update in range(progress['update_in_epoch'],updates_per_epoch):
            started=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);rows=[]
            step=progress['full_updates'];beta=min(1.,(step+1)/max(1,args.understanding_warmup))
            for group in optimizer.param_groups:group['lr']=args.learning_rate*lr_factor('short' if args.short_validation else 'full',step,total)
            indices=rank_microbatches(order,update,rank,world)
            for micro,index in enumerate(indices):
                batch=build_batch(opt,windows[index],device)
                sync=task.no_sync() if world>1 and micro<len(indices)-1 else contextlib.nullcontext()
                with sync:
                    loss,row=task(batch,step*8,args.understanding_weight*beta);(loss/len(indices)).backward()
                rows.append(row)
            if step==0:
                gradient_evidence={}
                for name,parameters in model.parameter_groups().items():
                    value=float(torch.stack([p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]).sum().sqrt())
                    if not math.isfinite(value) or value<=0:raise ValueError('Missing/nonfinite first gradient: '+name)
                    gradient_evidence[name]=value
                if any(p.grad is not None for module in (model.frozen_omega,model.sam_proposals) if module is not None for p in module.parameters()):raise ValueError('Frozen source received gradients')
                (root/f'FIRST_GRADIENTS_rank{rank:02}.json').write_text(json.dumps(gradient_evidence,indent=2)+'\n')
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step()
            if step==0:synchronized_first_update(model,initial_digest,root,rank,world)
            progress.update(epoch=epoch+int(update+1==updates_per_epoch),update_in_epoch=0 if update+1==updates_per_epoch else update+1,full_updates=step+1)
            row={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]};row.update(update=step+1,epoch=epoch,grad_norm=float(norm),understanding_weight=args.understanding_weight*beta,local_indices=indices,seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
            with (root/f'train_rank{rank:02}.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            if rank==0 and (step<3 or (step+1)%10==0):print(json.dumps(row),flush=True)
            if (step+1)%args.checkpoint_every==0:save()
            if args.max_updates is not None and step+1>=args.max_updates:stop=True;break
        if not stop:progress.update(epoch=epoch+1,update_in_epoch=0)
        if not args.short_validation or stop or progress["full_updates"]==total:save()
        if args.epoch_probes and not args.short_validation and progress['update_in_epoch']==0:
            if rank==0:
                epoch_checkpoint=root/f'checkpoint_epoch{progress["epoch"]}.pt'
                if not epoch_checkpoint.exists():os.link(root/'checkpoint_latest.pt',epoch_checkpoint)
            run_probes(model,opt,probe_train,validation,root,'final' if progress['epoch']==args.epochs else f'epoch{progress["epoch"]}',rank,world,device,args.understanding_weight)
        if stop:break
    if args.short_validation:run_probes(model,opt,windows,validation,root,'final',rank,world,device,args.understanding_weight)
    if rank==0:
        complete=progress['epoch']==args.epochs
        (root/('TRAINING_COMPLETE.json' if complete else 'LIMIT_REACHED.json')).write_text(json.dumps(dict(status='TRAINING_COMPLETE' if complete else 'UPDATE_LIMIT_REACHED',short_validation=args.short_validation,official_128=False if args.short_validation else None,progress=progress,architecture=model.architecture_name),indent=2)+'\n')
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
