"""Matched four-rank geometry-only continuation arms, 1043 updates each."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,contextlib,json,os,time,subprocess,math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.membership_geometry_core import *
from scripts.train_unified_query_v1 import restore_state,trainable_state,rng,restore_rng
from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT,digest
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
from scripts.dual_stream_frozen_cache import atomic_torch_save,cpu_tree
from scripts.v4init_isolated_recipe import optimizer_groups
from tokengs.options import config_defaults
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from scripts.membership_geometry_sdpa import install,evidence

class Task(torch.nn.Module):
    def __init__(self,model,arm):super().__init__();self.model=model;self.arm=arm
    def forward(self,batch,record,step):
        pred=self.model.generate(batch['images_input'])
        recon=reconstruction(self.model,pred,batch)
        mask,ce,dice=geometry_mask_loss(self.model,pred,record)
        weight=coefficient(self.arm,step)
        # G1 traverses the same forward; its mask graph is intentionally unused.
        total=recon+weight*mask
        diag=(pred['gaussians'][...,[0,1,2,4,5,6]].detach()-record['source_xyz_scale'].to(pred['gaussians']).float()).abs().mean((0,1))
        numbers=dict(loss=float(total.detach()),reconstruction=float(recon.detach()),mask=float(mask.detach()),ce=float(ce.detach()),dice=float(dice.detach()),weight=weight,
                     xyz_displacement=float(diag[:3].mean()),scale_displacement=float(diag[3:].mean()))
        if not all(math.isfinite(v) for v in numbers.values()):raise FloatingPointError(numbers)
        return total,numbers

def main():
    install()
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--arm',choices=['G1','G2'],required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    run=a.run_root;limit=1 if a.gate else 1043;out=run/('gate' if a.gate else a.arm)/(a.arm if a.gate else 'train');out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('Exactly four ranks required')
    torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((run/'configuration.json').read_text());source=torch.load(run/'source_checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);validate_source(source)
    if sha(run/'source_checkpoint.pt')!=SOURCE_SHA or source['meta']['manifest_sha256']!=digest(MANIFEST) or source['meta']['heldout_sha256']!=digest(HELDOUT):raise ValueError('Source/data identity differs')
    renderer=verify_teacher_renderer(cfg)
    write(out/f'RUNTIME_rank{rank:02d}.json',renderer)
    if sha(run/'teacher/export_complete.json')!=cfg['teacher_manifest_sha256']:raise ValueError('Teacher manifest changed')
    cache=json.loads((run/'teacher/export_complete.json').read_text())
    if cache['source_sha256']!=SOURCE_SHA or cache['windows']!=8337 or cache['gpu_gradient_smoke_ranks']!=4:raise ValueError('Teacher not complete')
    for r in range(4):
        smoke=json.loads((run/f'teacher/GPU_GRADIENT_SMOKE_rank{r:02d}.json').read_text())
        if smoke['status']!='PASSED' or not smoke.get('reconstruction_backward') or smoke.get('attention_backend',{}).get('math_bias_calls',0)<=0:raise ValueError('GPU engineering check missing')
    opt=config_defaults['train_siu3r_unified_query_v1']
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=build_structural_model('dual_v4init_isolated_noca',opt,UnifiedQueryConfig(**source['meta']['config'])).to(device)
    if model.frozen_omega.source_identity!=source['meta']['omega_identity']:raise ValueError('Frozen Omega identity differs')
    restore_state(model,source['model'])
    optimizer=torch.optim.AdamW(optimizer_groups(model),lr=1e-4,betas=(.9,.95),eps=1e-8)
    optimizer.load_state_dict(source['optimizer'])
    names=freeze(model)
    if not a.gate:
        for gate_arm in ('G1','G2'):
            proof=json.loads((run/'gate'/gate_arm/'GATE_COMPLETE.json').read_text())
            if proof['status']!='PASSED' or not proof['ddp_replicas_equal'] or proof['new_updates']!=1:raise ValueError('Real DDP gate incomplete')
    for group in optimizer.param_groups:group['lr']=1e-6
    restore_rng(source['rank_rng'][rank])
    windows,_,_=cohorts()
    if windows!=json.loads((run/'training_cohort.json').read_text())['records'] or sha(run/'training_cohort.json')!=cfg['training_cohort_sha256']:raise ValueError('Training data differs')
    order=epoch_order(len(windows),4)
    if len(order)//8!=1043:raise ValueError('Update count differs')
    write(out/f'RESTORE_rank{rank:02d}.json',dict(status='PASSED',source_sha256=SOURCE_SHA,rank_rng_source=rank,source_world=8,target_world=4,
         source_progress=source['progress'],trainable_names=names,adam_states=len(optimizer.state),adam_steps=sorted({float(s['step']) for s in optimizer.state.values()})))
    before_ddp=rng()
    ddp=DDP(Task(model,a.arm),device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    restore_rng(before_ddp)
    start=time.time();log=(out/f'updates_rank{rank:02d}.jsonl').open('x')
    def save(step):
        states=[None]*world;dist.all_gather_object(states,rng())
        if rank==0:
            meta=dict(source['meta'],epochs=5,world_size=4,code_sha=cfg['code_sha'],pilot_policy=POLICY,pilot_arm=a.arm,pilot_source_sha256=SOURCE_SHA,
                      attention_backward_policy=evidence()['policy'],pilot_geometry_lr=1e-6,pilot_lambda=.01 if a.arm=='G2' else 0.,pilot_warmup=100,pilot_trainable_names=names,
                      rank_rng_mapping=[0,1,2,3],effective_global_batch=8,teacher_cache_sha256=sha(run/'teacher/export_complete.json'))
            progress=dict(epoch=5 if step==1043 else 4,update_in_epoch=0 if step==1043 else step,full_updates=4172+step)
            atomic_torch_save(dict(model=trainable_state(model),optimizer=cpu_tree(optimizer.state_dict()),rank_rng=states,progress=progress,meta=meta),out/'checkpoint_latest.pt')
        dist.barrier()
    for step in range(limit):
        optimizer.zero_grad(set_to_none=True);numbers=[]
        for micro,index in enumerate(rank_microbatches(order,step,rank,world)):
            record_path=run/'teacher'/cache['records'][index]['file']
            if sha(record_path)!=cache['records'][index]['sha256']:raise ValueError('Teacher cache changed')
            record=torch.load(record_path,map_location='cpu',weights_only=False)
            batch=build_batch(opt,windows[index],device)
            if record['window_index']!=index or record['source_sha256']!=SOURCE_SHA or tensor_sha(batch['images_input'])!=record['rgb_sha256']:raise ValueError('Cache/input pairing differs')
            with (ddp.no_sync() if micro==0 else contextlib.nullcontext()):
                loss,row=ddp(batch,record,step);(loss/2).backward()
            numbers.append(row);del loss,batch,record
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        if step==0:
            if any(p.grad is None for p in model.parameters() if p.requires_grad):raise ValueError('Missing geometry gradient')
            write(out/f'FIRST_BACKWARD_rank{rank:02d}.json',dict(status='PASSED',gradient_norm=float(norm),trainable_tensors=48,accumulation=2))
        optimizer.step()
        log.write(json.dumps(dict(step=step+1,full_updates=4173+step,elapsed=time.time()-start,gradient_norm=float(norm),microbatches=numbers))+'\n');log.flush()
        if not a.gate and ((step+1)%100==0 or step==1042):save(step+1)
    log.close()
    state=trainable_state(model);changed=[]
    for n,v in state.items():
        if not torch.isfinite(v).all():raise ValueError('Nonfinite weight '+n)
        if not torch.equal(v,source['model'][n]):changed.append(n)
    if set(changed)!=set(names) or any(not n.startswith(PREFIXES) for n in changed):raise ValueError('Frozen weight changed or geometry did not update')
    original_by_name={n:s for g in source['optimizer']['param_groups'] for n,s in zip(g['parameter_names'],g['params'])}
    for g in optimizer.param_groups:
        for n,param in zip(g['parameter_names'],g['params']):
            st=optimizer.state[param];old=source['optimizer']['state'][original_by_name[n]]
            if n.startswith(PREFIXES):
                if float(st['step'])!=4172+limit or any(not torch.isfinite(st[k]).all() for k in ('exp_avg','exp_avg_sq')):raise ValueError('Geometry Adam invalid')
            elif any(not torch.equal(st[k].cpu(),old[k].cpu()) for k in ('step','exp_avg','exp_avg_sq')):raise ValueError('Frozen Adam changed')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen parameter gradient present')
    if evidence()['math_bias_calls']<=0:raise ValueError('MATH bias guard not exercised')
    proof=dict(status='PASSED',changed_names=changed,frozen_weights_bitwise_unchanged=True,frozen_adam_bitwise_unchanged=True,
        geometry_adam_step=4172+limit,frozen_adam_step=4172,new_updates=limit,source_sha256=SOURCE_SHA)
    write(out/f'FINAL_AUDIT_rank{rank:02d}.json',proof)
    # All replicas must finish the weight and optimizer audits before a receipt.
    fingerprints=[None]*world;dist.all_gather_object(fingerprints,dict(weights={n:tensor_sha(v) for n,v in state.items()},adam={n:{k:tensor_sha(optimizer.state[p][k]) for k in ('step','exp_avg','exp_avg_sq')} for n,p in model.named_parameters() if p.requires_grad}))
    if any(f!=fingerprints[0] for f in fingerprints):raise ValueError('DDP replicas differ')
    if rank==0:write(out/('GATE_COMPLETE.json' if a.gate else 'TRAINING_COMPLETE.json'),dict(proof,world_size=4,global_batch=8,source_rng_mapping=[0,1,2,3],ddp_replicas_equal=True,elapsed_seconds=time.time()-start,attention_backend=evidence(),checkpoint_sha256=None if a.gate else sha(out/'checkpoint_latest.pt')))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
