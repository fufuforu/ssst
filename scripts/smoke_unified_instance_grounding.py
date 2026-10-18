"""Real four-GPU signed feature/auxiliary gradient and two-update DDP gate."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,contextlib,gc,json,math,os,random,subprocess
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.unified_instance_grounding import rendered_instance_grounding_loss
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.options import config_defaults
from scripts.train_unified_query_v1 import Task
from scripts.unified_query_validation import parameter_digest
from scripts.train_dual_stream_v1 import build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
from scripts.instance_grounding_smoke_policy import validate_smoke


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();root=args.run_root
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('This bounded smoke requires four ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    preflight=json.loads((root/'preflight.json').read_text());code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if digest(MANIFEST)!=preflight['manifest_sha256'] or digest(HELDOUT)!=preflight['heldout_sha256'] or code_sha!=preflight['code_sha']:
        raise ValueError('Smoke data/code differs from preflight')
    smoke=root/'smoke';smoke.mkdir(exist_ok=True)
    if rank==0:
        if (smoke/'STARTED.json').exists() or (root/'STARTED.json').exists():raise ValueError('Refusing duplicate smoke/training')
        (smoke/'STARTED.json').write_text(json.dumps(dict(world_size=world,code_sha=code_sha,optimizer_updates_discarded=2))+'\n')
    dist.barrier()
    opt=config_defaults['train_siu3r_unified_query_v1'];cfg=UnifiedQueryConfig(**preflight['config'])
    model=build_structural_model(preflight.get('architecture_key','unified_instance_grounding'),opt,cfg).to(device).train()
    if model.frozen_omega.source_identity!=preflight['omega_identity'] or model.sam_proposals.source_identity!=preflight['sam_identity']:
        raise ValueError('Smoke assets differ')
    if model.extra_checkpoint_metadata!=preflight['extra_checkpoint_metadata']:raise ValueError('Smoke recipe differs')
    initial=parameter_digest(model);head_before={k:p.detach().clone() for k,p in model.core.instance_head.named_parameters()}
    lift_before={k:p.detach().clone() for k,p in model.core.lift_residual.named_parameters()} if hasattr(model.core,'lift_residual') else {}
    batch=build_batch(opt,preflight['training_windows'][rank],device)
    prediction=model.generate(batch['images_input'])
    # Independent read-only CUDA oracle with signed channels and mixed signs.
    with torch.no_grad():
        g=prediction['gaussians'].detach();positive=torch.ones_like(prediction['gaussian_instance_features'])
        camera=torch.linalg.inv(prediction['predicted_context_c2w']).transpose(-1,-2)
        intrinsics=prediction['predicted_context_intrinsics']
        one=model.gs.render_feature_channels(g,positive,camera,intrinsics)
        signs=torch.tensor([1.,-1.]*4,device=device)
        mixed=model.gs.render_feature_channels(g,positive*signs,camera,intrinsics)
        torch.testing.assert_close(one['images_pred'],one['alphas_pred'].expand_as(one['images_pred']),rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(mixed['images_pred'],one['images_pred']*signs[None,None,:,None,None],rtol=1e-4,atol=1e-5)
    coupling_checks={}
    recipe=getattr(model,'coupling_recipe',None)
    if recipe is not None and recipe.lift:
        lift_gradient,geometry_gradient=torch.autograd.grad(prediction['lift_residual'].square().mean(),
            (model.gfm_projection[1].weight,prediction['gaussians']),allow_unused=True,retain_graph=True)
        coupling_checks.update(lift_feature_gradient_norm=float(lift_gradient.norm()),lift_geometry_gradient_absent=geometry_gradient is None,
            lift_residual_norm=float(prediction['lift_residual'].norm()),lift_visible_gaussians=int((prediction['lift_mass']>0).sum()))
        if coupling_checks['lift_feature_gradient_norm']<=0 or not coupling_checks['lift_geometry_gradient_absent']:raise ValueError('Lift gradient contract failed')
        del lift_gradient,geometry_gradient
    if recipe is not None and recipe.boundary:
        from tokengs.models.unified_instance_coupling import boundary_rgb_loss
        boundary,weights=boundary_rgb_loss(prediction['render']['images_pred'],batch['images_input'],prediction['rendered_instance_features'],recipe)
        rgb_gradient,instance_gradient=torch.autograd.grad(boundary,
            (prediction['render']['images_pred'],prediction['rendered_instance_features']),allow_unused=True,retain_graph=True)
        coupling_checks.update(boundary_rgb_gradient_norm=float(rgb_gradient.norm()),boundary_instance_gradient_absent=instance_gradient is None,
            boundary_weights_detached=not weights.requires_grad)
        if coupling_checks['boundary_rgb_gradient_norm']<=0 or not coupling_checks['boundary_instance_gradient_absent'] or weights.requires_grad:raise ValueError('Boundary gradient contract failed')
        del boundary,weights,rgb_gradient,instance_gradient
    aux,_=rendered_instance_grounding_loss(prediction['rendered_instance_features'],build_context_instance_targets(batch))
    if not torch.isfinite(aux):raise FloatingPointError('Nonfinite smoke auxiliary loss')
    values=torch.autograd.grad(aux,(model.core.instance_head[1].weight,prediction['gaussians'],model.gfm_projection[1].weight),allow_unused=True)
    norms={name:float(value.float().norm()) if value is not None else 0. for name,value in zip(('instance_head','gaussian_geometry','shared_projection'),values)}
    if not all(math.isfinite(v) for v in norms.values()):raise FloatingPointError('Nonfinite auxiliary gradient')
    del prediction,batch,values,aux,g,positive,one,mixed;gc.collect();torch.cuda.empty_cache()
    grouped={}
    for p in model.parameters():
        if p.requires_grad:grouped.setdefault(0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,weight_decay=wd) for wd,ps in grouped.items()],lr=1e-4,betas=(.9,.95),eps=1e-8)
    task=DDP(Task(model),device_ids=[local],find_unused_parameters=True)
    order=epoch_order(16,0);steps=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);rate=1e-4*(step+1)/20
        for group in optimizer.param_groups:group['lr']=rate
        for micro,index in enumerate(rank_microbatches(order,step,rank,world)):
            batch=build_batch(opt,preflight['training_windows'][index],device)
            sync=task.no_sync() if micro==0 else contextlib.nullcontext()
            with sync:
                loss,numbers=task(batch,step*8,.05*(step+1)/20);(loss/2).backward()
            del batch,loss
        head_norm=float(torch.stack([p.grad.float().square().sum() for p in model.core.instance_head.parameters() if p.grad is not None]).sum().sqrt())
        if not math.isfinite(head_norm) or head_norm<=0:raise ValueError('Smoke instance head did not receive a gradient')
        if any(p.grad is not None for module in (model.frozen_omega,model.sam_proposals) for p in module.parameters()):raise ValueError('Frozen source received gradients')
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        optimizer.step()
        steps.append(dict(step=step+1,loss=numbers['loss'],auxiliary_loss=numbers['loss_instance_grounding'],grad_norm=norm,
            instance_head_grad_norm=head_norm,parameter_sha256=parameter_digest(model)))
        print(json.dumps(dict(rank=rank,smoke_step=step+1,**steps[-1])),flush=True)
    row=dict(rank=rank,initial_parameter_sha256=initial,auxiliary_gradient_norms=norms,coupling_checks=coupling_checks,optimizer_updates=2,steps=steps,
        frozen_without_gradients=not any(p.grad is not None for module in (model.frozen_omega,model.sam_proposals) for p in module.parameters()),
        parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad),
        instance_head_changed=any(not torch.equal(head_before[k],p) for k,p in model.core.instance_head.named_parameters()),
        lift_head_changed=any(not torch.equal(lift_before[k],p) for k,p in model.core.lift_residual.named_parameters()) if lift_before else False,
        adam_steps=sorted({float(state['step']) for state in optimizer.state.values()}),signed_render_replay_verified=True,embedding_channels=8,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
    rows=[None]*world;dist.all_gather_object(rows,row);receipt=validate_smoke(rows,world)
    receipt.update(code_sha=code_sha,preflight_sha256=digest(root/'preflight.json'),fresh_short_training_seed=42,parameter_names=preflight['trainable_parameter_names'])
    if recipe is not None:
        from scripts.instance_coupling_smoke_policy import validate_coupling_smoke
        validate_coupling_smoke(receipt,preflight)
    if rank==0:(root/'SMOKE_PASSED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
