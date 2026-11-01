"""Eight-rank gated training. Smoke runs never supply formal initial weights."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,contextlib,json,math,os,random,subprocess,time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.reconstruction_abcd_policy import initialize,optimizer,task_state,tensor_sha
from scripts.train_unified_query_v1 import rng,restore_rng
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches,lr_factor
from scripts.dual_stream_frozen_cache import atomic_torch_save,cpu_tree
from scripts.membership_geometry_sdpa import install,evidence
from tokengs.options import config_defaults
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def fingerprints(model,trainable=None):
    return {n:tensor_sha(p) for n,p in model.named_parameters() if not n.startswith(('frozen_omega.','sam_proposals.')) and (trainable is None or p.requires_grad==trainable)}


class Task(torch.nn.Module):
    def __init__(self,model,arm):super().__init__();self.model=model;self.arm=arm;self.last=None
    def forward(self,batch,step,weight):
        output,metrics=self.model.step_loss(batch,step=step,understanding_weight=weight)
        pred=output['prediction'];self.last=pred
        n=4096 if self.arm=='B' else 1024
        if pred['gaussians'].shape!=(1,n*64,23 if self.arm=='D' else 14):raise ValueError('Wrong GS shape')
        parent=pred['gaussian_parent_anchor']
        if parent.shape!=(n*64,) or not torch.equal(parent,torch.arange(n,device=parent.device).repeat_interleave(64)):raise ValueError('Parent index mismatch')
        if self.arm!='D' and pred['anchor_point_indices'].unique().numel()!=n:raise ValueError('Duplicated FPS anchors')
        numbers={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items() if isinstance(v,(int,float)) or torch.is_tensor(v) and v.numel()==1}
        if not all(math.isfinite(v) for v in numbers.values()):raise FloatingPointError('Nonfinite loss/metric')
        # ABC do not update understanding. D uses original warmup.
        loss=metrics['loss'] if self.arm=='D' else metrics['loss_recon']
        return loss,numbers,metrics['loss_understanding']


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',dest='run',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=json.loads((a.run/'configuration.json').read_text());arm=cfg['arm'];out=a.run/('smoke' if a.smoke else 'train');out.mkdir(exist_ok=True)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8:raise ValueError('Exactly eight ranks required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    install();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if cfg['code_sha']!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip():raise ValueError('Code SHA differs')
    if not a.smoke:
        from scripts.reconstruction_abcd_gate_reuse import validate
        validate(a.run,cfg,arm)
    if (out/'STARTED.json').exists() or (out/'checkpoint_latest.pt').exists():raise ValueError('Refusing duplicate start; recovery requires explicit audit')
    if digest(cfg['source_checkpoint'])!=cfg['source_sha256']:raise ValueError('Source file SHA differs')
    for k,path in [('manifest_sha256',MANIFEST),('heldout_sha256',HELDOUT),('cohort_sha256',cfg['cohort'])]:
        if digest(path)!=cfg[k]:raise ValueError('Data SHA differs '+k)
    loaded_renderer=Path(sys.modules['gsplat_cuda'].__file__).resolve()
    if loaded_renderer!=Path(cfg['renderer']).resolve() or digest(loaded_renderer)!=cfg['renderer_sha256']:raise ValueError('Pinned renderer differs')
    source=torch.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    if source['meta']['manifest_sha256']!=cfg['manifest_sha256'] or source['meta']['heldout_sha256']!=cfg['heldout_sha256']:raise ValueError('Source data differs')
    opt=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig(**source['meta']['config'])
    model=build_structural_model('reconstruction_'+arm.lower(),opt,config).to(device)
    init=initialize(model,source,arm);adam,adam_proof=optimizer(model,source,arm)
    write(out/f'INITIALIZATION_rank{rank:02d}.json',dict(init,adam=adam_proof,source_sha256=cfg['source_sha256'],frozen_omega=model.frozen_omega.source_identity,sam=model.sam_proposals.source_identity))
    restore_rng(source['rank_rng'][rank]);del source
    protected=fingerprints(model,False);initial=fingerprints(model,True)
    if rank==0:write(out/'STARTED.json',cfg)
    task=Task(model,arm);before_ddp=rng();ddp=DDP(task,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True);restore_rng(before_ddp)
    windows,_,_=cohorts()
    if len(windows)!=8337 or len({w['scene'] for w in windows})!=1191:raise ValueError('Training cohort differs')
    total=4172 if arm=='D' else 1043;limit=2 if a.smoke else total
    added=0;start=time.monotonic();log=(out/f'updates_rank{rank:02d}.jsonl').open('x')
    meta=dict(cfg,architecture=model.architecture_name,config=config.to_dict(),omega_identity=model.frozen_omega.source_identity,sam_identity=model.sam_proposals.source_identity,
        epochs=4 if arm=='D' else 9,short_validation=False,global_batch=8,world_size=8,initial_updates=0 if arm=='D' else 8344,
        supervised_layers=[6,12],layer_weights=[1/3,2/3],gradient_isolation=True,smoke_only=a.smoke)
    def save():
        states=[None]*world;dist.all_gather_object(states,rng())
        if rank==0:
            absolute=added+(0 if arm=='D' else 8344)
            progress=dict(epoch=absolute//1043,update_in_epoch=absolute%1043,full_updates=absolute)
            atomic_torch_save(dict(model=cpu_tree(task_state(model)),optimizer=cpu_tree(adam.state_dict()),progress=progress,rank_rng=states,meta=meta),out/'checkpoint_latest.pt')
        dist.barrier()
    if not a.smoke:
        save()
        if rank==0:os.link(out/'checkpoint_latest.pt',out/'checkpoint_initial.pt')
        dist.barrier()
    baseline_c=None
    # C's unchanged semantic/geometry outputs compared against identical seeded A
    # using the same source/candidate pool, not a second trained weight source.
    for step in range(limit):
        started=time.monotonic();torch.cuda.reset_peak_memory_stats();model.train();adam.zero_grad(set_to_none=True)
        for group in adam.param_groups:
            if arm=='D':group['lr']=group['base_lr']*lr_factor('full',step,total)
            elif group['family']=='appearance':group['lr']=3e-5*min(1.,(step+1)/100)
            # ABC original Adam/rates remain exactly at source endpoint.
        epoch=step//1043+(0 if arm=='D' else 8);order=epoch_order(len(windows),epoch)
        index=order[(step%1043)*8+rank];batch=build_batch(opt,windows[index],device)
        loss,row,lu=ddp(batch,step*8,min(1.,(step+1)/1043) if arm=='D' else 0.)
        pred=task.last
        if a.smoke and step==0:
            if arm=='D':
                # No LU -> reconstruction leakage, even through GS alpha/geometry.
                rec=model.parameter_groups()['reconstruction']
                grads=torch.autograd.grad(lu,[p for p in rec if p.requires_grad],retain_graph=True,allow_unused=True)
                nonzero=sum(int(torch.count_nonzero(g)) for g in grads if g is not None)
                if nonzero:raise ValueError('D understanding gradients enter reconstruction')
                write(out/f'ISOLATION_rank{rank:02d}.json',dict(status='PASSED',LU_reconstruction_nonzero=nonzero))
            if arm=='C':
                with torch.no_grad():
                    state=pred['states'][-1]
                    base=model.core.activation_head(state['tokens'],state['mu'],state['radii'])
                    if not torch.equal(base,pred['gaussians']):raise ValueError('C zero residual not identical')
                    from scripts.reconstruction_c_protection import capture
                    preserved_rng=rng();rgb=batch['images_input'].detach().clone()
                    reference=capture(model,rgb);restore_rng(preserved_rng)
                    baseline_c=dict(reference=reference,repeat=capture(model,rgb),rgb=rgb,rng=preserved_rng)
                    restore_rng(preserved_rng)
            write(out/f'SHAPES_rank{rank:02d}.json',dict(gaussians=list(pred['gaussians'].shape),parent_count=pred['gaussian_parent_anchor'].numel(),unique_sources=None if arm=='D' else pred['anchor_point_indices'].unique().numel(),gpu=torch.cuda.get_device_name(local)))
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        if not math.isfinite(norm) or norm<=0:raise ValueError('No finite gradients')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen parameter has gradients')
        if a.smoke:
            targeted={}
            def required_gradient(name,p):
                value=float(p.grad.norm()) if p.grad is not None else 0.
                if not math.isfinite(value) or value<=0:raise ValueError('Missing targeted gradient '+name)
                targeted[name]=value
            if arm=='D':
                required_gradient('geometry_head',model.core.geometry_head.weight)
                required_gradient('appearance_head',model.core.appearance.output.weight)
                required_gradient('geometry_query',model.core.gs_tokens)
            if arm=='B':
                grad=model.core.v4_initialization.gs_tokens.grad
                value=float(grad[1024:].norm()) if grad is not None else 0.
                if not math.isfinite(value) or value<=0:raise ValueError('New B rows have no gradient')
                targeted['new_B_rows']=value
            if arm=='C' and step==1:required_gradient('RGB_patch_after_zero_head_update',model.core.appearance.rgb.weight)
            write(out/f'TARGETED_GRADIENTS_{step+1}_rank{rank:02d}.json',dict(status='PASSED',norms=targeted))
        adam.step();added=step+1;task.last=None
        del pred,loss,lu,batch
        torch.cuda.synchronize()
        row.update(added_updates=added,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gradient_norm=norm,learning_rates={str(i):g['lr'] for i,g in enumerate(adam.param_groups)})
        log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
        if rank==0 and (a.smoke or step%10==0):print(json.dumps(row),flush=True)
        if step==0 or added==limit:
            current=fingerprints(model,True);changed=[n for n in current if current[n]!=initial[n]]
            if not changed:raise ValueError('No parameter updates')
            if fingerprints(model,False)!=protected:raise ValueError('Frozen task tensors changed')
            all_hashes=[None]*8;dist.all_gather_object(all_hashes,current)
            if any(x!=current for x in all_hashes):raise ValueError('DDP replicas differ')
            write(out/f'UPDATE_AUDIT_{added}_rank{rank:02d}.json',dict(status='PASSED',changed_names=changed,protected_tensors_equal=True,ddp_replicas_equal=True,parameter_fingerprints=current))
        if not a.smoke and (added%100==0 or added%1043==0):save()
    if a.smoke and arm=='C':
        from scripts.reconstruction_c_protection import capture,verify
        current_rng=rng();restore_rng(baseline_c['rng'])
        after=capture(model,baseline_c['rgb']);restore_rng(current_rng)
        proof=verify(baseline_c['reference'],baseline_c['repeat'],after)
        write(out/f'C_OUTPUT_PROTECTION_rank{rank:02d}.json',proof)
        if proof['status']!='PASSED':raise ValueError('C protected semantic output exceeds measured roundoff; see receipt')
    if not a.smoke and added!=total:raise ValueError('Wrong endpoint')
    receipts=[None]*8;dist.all_gather_object(receipts,dict(rank=rank,updates=added,finite=True,protected_unchanged=True))
    if rank==0:
        write(out/('SMOKE_PASSED.json' if a.smoke else 'TRAINING_COMPLETE.json'),dict(status='PASSED',arm=arm,code_sha=cfg['code_sha'],world_size=8,global_batch=8,added_updates=added,full_updates=added+(0 if arm=='D' else 8344),elapsed_seconds=time.monotonic()-start,ranks=receipts,attention_backend=evidence(),smoke_weights_used_for_formal=False))
    dist.destroy_process_group()

if __name__=='__main__':main()
