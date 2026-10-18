"""All 16 real windows, signed CUDA rendering, independent gradients, 2 DDP updates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,contextlib,gc,json,math,os,random,subprocess
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.direct_instance_attributes import RECIPE
from tokengs.models.unified_query_loss import unified_instance_loss
from tokengs.models.unified_instance_grounding import rendered_instance_grounding_loss
from tokengs.options import config_defaults
from scripts.train_unified_query_v1 import Task
from scripts.unified_query_validation import parameter_digest
from scripts.train_dual_stream_v1 import build_batch,digest,MANIFEST,HELDOUT
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
from scripts.direct_instance_short_policy import HEADS,validate_attribute_smoke


def norm(value):return float(value.float().norm()) if value is not None else 0.


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('Four CUDA workers required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',device_id=device)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pre=json.loads((root/'preflight.json').read_text());sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if sha!=pre['code_sha'] or digest(MANIFEST)!=pre['manifest_sha256'] or digest(HELDOUT)!=pre['heldout_sha256']:raise ValueError('Source/data differs')
    smoke=root/'smoke';smoke.mkdir(exist_ok=True)
    if rank==0:
        if (smoke/'STARTED.json').exists() or (root/'STARTED.json').exists():raise ValueError('Duplicate smoke/training')
        (smoke/'STARTED.json').write_text(json.dumps(dict(code_sha=sha,discarded_optimizer_updates=2))+'\n')
    dist.barrier();opt=config_defaults['train_siu3r_unified_query_v1']
    model=build_structural_model(pre['architecture_key'],opt,UnifiedQueryConfig(**pre['config'])).to(device).train()
    if model.extra_checkpoint_metadata!=pre['extra_checkpoint_metadata'] or model.frozen_omega.source_identity!=pre['omega_identity'] or model.sam_proposals.source_identity!=pre['sam_identity']:raise ValueError('Recipe/assets differ')
    initial=parameter_digest(model);heads_before={h:{k:v.detach().clone() for k,v in getattr(model.core,h).named_parameters()} for h in HEADS}
    checks=[];signed=False
    for index in range(rank,16,world):
        batch=build_batch(opt,pre['training_windows'][index],device);p=model.generate(batch['images_input'])
        q=p['unified_query'];attributes=p['gaussian_instance_attributes'];prototypes=p['query_instance_prototypes']
        torch.testing.assert_close(model.core.instance_attribute_head(q).reshape_as(attributes),attributes,rtol=0,atol=0)
        expected=model.core.instance_logit_scale.exp().clamp(RECIPE.membership_scale_min,RECIPE.membership_scale_max)*(F.normalize(attributes.float(),dim=-1,eps=RECIPE.normalization_epsilon)@F.normalize(prototypes.float(),dim=-1,eps=RECIPE.normalization_epsilon).transpose(1,2))+model.core.instance_logit_bias
        torch.testing.assert_close(expected,p['gaussian_mask_logits'][...,:p['num_thing_queries']],rtol=0,atol=0)
        assert attributes is p['gaussian_instance_features'] and attributes.shape[1]==p['gaussians'].shape[1]
        if not signed:
            with torch.no_grad():
                g=p['gaussians'].detach();one=torch.ones_like(attributes);camera=torch.linalg.inv(p['predicted_context_c2w']).transpose(-1,-2);intrinsics=p['predicted_context_intrinsics']
                positive=model.gs.render_feature_channels(g,one,camera,intrinsics);signs=torch.tensor([1.,-1.]*4,device=device)
                mixed=model.gs.render_feature_channels(g,one*signs,camera,intrinsics)
                torch.testing.assert_close(positive['images_pred'],positive['alphas_pred'].expand_as(positive['images_pred']),rtol=1e-4,atol=1e-5)
                torch.testing.assert_close(mixed['images_pred'],positive['images_pred']*signs[None,None,:,None,None],rtol=1e-4,atol=1e-5)
                replay=model.gs.render_feature_channels(g,attributes.detach(),camera,intrinsics)['images_pred']
                torch.testing.assert_close(replay,p['rendered_instance_features'],rtol=1e-4,atol=1e-5)
            signed=True;del g,one,camera,intrinsics,positive,mixed,signs,replay
        membership,_=unified_instance_loss(p,batch,model.config)
        aux,terms=rendered_instance_grounding_loss(p['rendered_instance_features'],p['final_targets'])
        attr_weight=model.core.instance_attribute_head[1].weight;proto_weight=model.core.instance_prototype_head[1].weight
        aux_g=torch.autograd.grad(aux,[attr_weight,proto_weight,p['gaussians'],model.gfm_projection[1].weight],allow_unused=True,retain_graph=True)
        mask_g=torch.autograd.grad(membership,[attr_weight,proto_weight,model.core.stuff_attribute_head[1].weight,model.gfm_projection[1].weight],allow_unused=True,retain_graph=True)
        forbidden=torch.autograd.grad(attributes.square().mean(),[p['gaussians'],p['states'][-1]['mu'],p['states'][-1]['radii']]+list(model.core.activation_head.parameters())+list(model.core.geometry.parameters()),allow_unused=True)
        check=dict(window_index=index,scene=pre['training_windows'][index]['scene'],direct_geometry_independent=all(g is None for g in forbidden),
            direct_values_equal=True,all_query_dot_equal=True,shared_attribute_tensor=True,aux_prototype_gradient_absent=aux_g[1] is None,
            aux_gradients=dict(attributes=norm(aux_g[0]),geometry=norm(aux_g[2]),shared=norm(aux_g[3])),
            mask_gradients=dict(attributes=norm(mask_g[0]),prototypes=norm(mask_g[1]),stuff=norm(mask_g[2]),shared=norm(mask_g[3])),
            supervised_pixels=int(terms['instance_feature_supervised_pixels']))
        checks.append(check);print(json.dumps(dict(rank=rank,attribute_probe=check)),flush=True)
        del batch,p,q,attributes,prototypes,expected,membership,aux,terms,aux_g,mask_g,forbidden,attr_weight,proto_weight
        gc.collect();torch.cuda.empty_cache()
    if parameter_digest(model)!=initial or any(p.grad is not None for p in model.parameters()):raise ValueError('Read-only checks changed model/gradients')
    grouped={}
    for p in model.parameters():
        if p.requires_grad:grouped.setdefault(0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,weight_decay=wd) for wd,ps in grouped.items()],lr=1e-4,betas=(.9,.95),eps=1e-8)
    task=DDP(Task(model),device_ids=[local],find_unused_parameters=False);order=epoch_order(16,0);steps=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=1e-4*(step+1)/20
        for micro,index in enumerate(rank_microbatches(order,step,rank,world)):
            batch=build_batch(opt,pre['training_windows'][index],device)
            with task.no_sync() if micro==0 else contextlib.nullcontext():
                loss,numbers=task(batch,step*8,.05*(step+1)/20);(loss/2).backward()
            del batch,loss
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise ValueError('Missing/nonfinite trainable gradient')
        grads={h:float(torch.stack([p.grad.float().square().sum() for p in getattr(model.core,h).parameters()]).sum().sqrt()) for h in HEADS}
        grad_norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        optimizer.step();steps.append(dict(step=step+1,loss=numbers['loss'],grad_norm=grad_norm,head_gradient_norms=grads,parameter_sha256=parameter_digest(model)))
        print(json.dumps(dict(rank=rank,smoke_step=steps[-1])),flush=True)
    row=dict(rank=rank,initial_parameter_sha256=initial,checks=checks,steps=steps,optimizer_updates=2,
        adam_steps=sorted({float(v['step']) for v in optimizer.state.values()}),signed_render_verified=signed,
        heads_changed={h:any(not torch.equal(heads_before[h][k],v) for k,v in getattr(model.core,h).named_parameters()) for h in HEADS},
        parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad),
        frozen_without_gradients=not any(p.grad is not None for module in (model.frozen_omega,model.sam_proposals) for p in module.parameters()))
    rows=[None]*world;dist.all_gather_object(rows,row)
    receipt=dict(status='PASSED',code_sha=sha,preflight_sha256=digest(root/'preflight.json'),ranks=rows,
        initial_parameter_sha256=initial,parameter_names=pre['trainable_parameter_names'],fresh_training_seed=42,smoke_weights_discarded=True)
    validate_attribute_smoke(receipt,preflight_sha256=digest(root/'preflight.json'),code_sha=sha)
    if rank==0:(root/'SMOKE_PASSED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
