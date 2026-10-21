"""Dedicated one-window, one-GPU smoke. Two updates, global8 by accumulation.

Never submits another job or writes full model weights. Receipts contain actual
GPU shapes, timing/memory, gradients, directional isolation and weight hashes.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,math,os,random,subprocess,time,traceback


def write(root,name,value):
    target=root/name;tmp=target.with_suffix(target.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(target)



def batch_shapes(batch):
    return {key:list(batch[key].shape) for key in ('images_input','images_all','images_output','frame_ids') if key in batch}


def validate_batch_contract(batch,window):
    context=list(window['context']);novel=list(window['novel']);frames=context+novel
    actual=batch_shapes(batch)
    expected=dict(images_input=[1,2,3,256,256],images_all=[1,4,3,256,256],images_output=[1,2,3,256,256],frame_ids=[1,4])
    if len(context)!=2 or len(novel)!=2 or len(set(frames))!=4 or actual!=expected:
        raise ValueError(f'Original A4 four-view contract differs: actual={actual}, expected={expected}, frames={frames}')
    actual_frames=batch['frame_ids'][0].tolist()
    if actual_frames!=frames:raise ValueError(f'Batch frame order differs: actual={actual_frames}, expected={frames}')
    return dict(status='PASSED',shapes=actual,frame_ids=actual_frames,context_views=2,novel_views=2,supervised_views=4)


def main(root):
    stage='runtime_import';completed=[]
    try:
        from scripts.runtime_bootstrap import prepare_runtime
        prepare_runtime(ROOT)
        import numpy as np
        import torch
        from tokengs.options import config_defaults
        from tokengs.models.unified_superpoints import UnifiedQueryConfig
        from tokengs.models.superpoint_structural_variants import build_structural_model
        from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
        from scripts.dual_stream_training_policy import epoch_order,lr_factor
        from scripts.train_unified_query_v1 import restore_rng
        from scripts.point1024_isoadapter_initial_state import initialize_point1024_isoadapter_from_legacy_zero
        from scripts.unified_query_validation import parameter_digest
        from scripts.verify_point1024_fmquery_smoke import verify
        if torch.cuda.device_count()!=1:raise ValueError('Exactly one allocated GPU required')
        if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Dedicated smoke supports one rank only')
        torch.cuda.set_device(0);device=torch.device('cuda',0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if sha!=(root/'code.sha').read_text().strip():raise ValueError('Snapshot identity differs')
        subprocess.run(['git','diff','--exit-code','HEAD','--','tokengs','scripts','tests'],cwd=ROOT,check=True)
        stage='source_and_data'
        source=root/'checkpoint_untrained_initial.pt'
        if digest(source)!=(root/'untrained_initial.sha256').read_text().strip():raise ValueError('Source SHA differs')
        saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        config=UnifiedQueryConfig(**saved['meta']['config'])
        windows,_,_=cohorts();index=epoch_order(len(windows),0)[0];window=windows[index]
        if len(windows)!=8337 or len({w['scene'] for w in windows})!=1191:raise ValueError('Training cohort differs')
        stage='model_and_zero_migration'
        opt=config_defaults['train_siu3r_unified_query_v1']
        model=build_structural_model('dual_point1024_isoadapter',opt,config).to(device)
        evidence=initialize_point1024_isoadapter_from_legacy_zero(saved,model,
            manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=1)
        write(root,'POINT1024_ISO_INITIALIZATION.json',evidence)
        restore_rng(saved['rank_rng'][0]);del saved
        grouped={}
        for name,p in model.named_parameters():
            if p.requires_grad:grouped.setdefault(0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05,[]).append(p)
        optimizer=torch.optim.AdamW([dict(params=ps,weight_decay=wd) for wd,ps in grouped.items()],lr=1e-4,betas=(.9,.95),eps=1e-8)
        groups=model.parameter_groups()
        adapters=list(model.core.understanding_adapters.parameters());adapter_ids={id(p) for p in adapters}
        diagnostics=dict(reconstruction=groups['reconstruction'],shared=groups['shared'],
            understanding=[p for p in groups['understanding'] if id(p) not in adapter_ids],understanding_adapters=adapters)
        reconstruction=diagnostics['reconstruction']+diagnostics['shared']
        understanding=diagnostics['understanding']+adapters
        all_params=[p for p in model.parameters() if p.requires_grad]
        if len({id(p) for ps in diagnostics.values() for p in ps})!=len(all_params):raise ValueError('Parameter partition incomplete')
        def norm(params):
            sums=[p.grad.detach().float().square().sum() for p in params if p.grad is not None]
            return float(torch.stack(sums).sum().sqrt()) if sums else 0.
        def leakage(params):
            nonzero=sum(int(torch.count_nonzero(p.grad)) for p in params if p.grad is not None)
            return dict(parameter_count=len(params),nonzero_gradient_elements=nonzero,norm=norm(params))
        before=parameter_digest(model)
        budget=dict(architecture=model.architecture_name,code_sha=sha,world_size=1,global_batch=8,
            accumulation_steps=8,unique_windows=1,window_index=index,window=window,
            training_windows=8337,training_scenes=1191,epochs=4,schedule_total_updates=4172,max_updates=2,
            lr_base=1e-4,lr_warmup=100,lr_floor=1e-5,understanding_weight=.05,understanding_warmup=1043,
            supervised_layers=list(model.supervised_layers),layer_weights=list(model.layer_weights),image_hw=list(opt.img_size),
            gpu_name=torch.cuda.get_device_name(0),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            color_prior=False,write_strength=0.,full_training_submitted=False)
        write(root,'BUDGET.json',budget)
        stage='batch_load';batch=build_batch(opt,window,device)
        write(root,'BATCH_SHAPES.json',dict(status='OBSERVED',shapes=batch_shapes(batch)))
        contract=validate_batch_contract(batch,window)
        write(root,'BATCH_SHAPES.json',contract);print(json.dumps(contract),flush=True)
        budget.update(context_views=2,novel_views=2,supervised_views=4,batch_contract=contract)
        write(root,'BUDGET.json',budget)
        model.train()
        for update in range(2):
            stage=f'update_{update+1}';optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=1e-4*lr_factor('full',update,4172)
            weight=.05*min(1.,(update+1)/1043)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic();losses=[]
            shapes=None
            for micro in range(8):
                output,metrics=model.step_loss(batch,step=update*8,understanding_weight=weight)
                prediction=output['prediction'];g=prediction['gaussians'];pa=prediction['gaussian_parent_anchor'];pq=prediction['gaussian_parent_query']
                if tuple(g.shape)!=(1,65536,14) or pa.shape!=pq.shape or pa.numel()!=65536 or not torch.equal(pa,pq):raise ValueError('GS shape/legacy parent alias differs')
                shapes=dict(gaussians=list(g.shape),gaussian_parent_anchor=list(pa.shape),gaussian_parent_query=list(pq.shape),parent_alias_equal=True)
                if update==0 and micro==0:
                    stage='directional_LU'
                    optimizer.zero_grad(set_to_none=True);metrics['loss_understanding'].backward(retain_graph=True)
                    lu=leakage(reconstruction);lu_adapters=norm(adapters)
                    if lu['nonzero_gradient_elements']!=0 or lu_adapters<=0:raise ValueError('LU isolation failed or adapter has no gradient')
                    optimizer.zero_grad(set_to_none=True);stage='directional_LR'
                    metrics['loss_recon'].backward(retain_graph=True);lr=leakage(understanding)
                    if lr['nonzero_gradient_elements']!=0:raise ValueError('LR isolation failed')
                    write(root,'DIRECTIONAL_GRADIENTS.json',dict(status='PASSED',LU_to_reconstruction_including_shared=lu,
                        LR_to_understanding_including_adapters=lr,LU_adapter_norm=lu_adapters,method='separate_backward_retained_graph_read_parameter_grad'))
                    optimizer.zero_grad(set_to_none=True);stage=f'update_{update+1}'
                loss=metrics['loss']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss')
                losses.append(float(loss.detach()));(loss/8).backward()
                del output,prediction,metrics,g,pa,pq,loss
            grad_norms={name:norm(ps) for name,ps in diagnostics.items()}
            adapter_norms={name:norm(module.parameters()) for name,module in model.core.understanding_adapters.items()}
            if any(not math.isfinite(x) or x<=0 for x in grad_norms.values()):raise ValueError('Missing/nonfinite group gradient')
            if any(not math.isfinite(x) for x in adapter_norms.values()) or adapter_norms['L12']<=0:raise ValueError('Nonfinite adapter or missing L12 gradient')
            if any(p.grad is not None for module in (model.frozen_omega,model.sam_proposals) if module is not None for p in module.parameters()):raise ValueError('Frozen backbone has gradients')
            clip=float(torch.nn.utils.clip_grad_norm_(all_params,1.,error_if_nonfinite=True));optimizer.step()
            torch.cuda.synchronize();seconds=time.monotonic()-started
            row=dict(update=update+1,microbatches=8,unique_windows=1,loss_mean=sum(losses)/8,
                learning_rate=optimizer.param_groups[0]['lr'],understanding_weight=weight,
                seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gradient_norms_before_clip=grad_norms,adapter_gradient_norms_before_clip=adapter_norms,clip_input_norm=clip,shapes=shapes)
            completed.append(row);write(root,'STEP_RESULTS.json',completed);print(json.dumps(row,allow_nan=False),flush=True)
            if update==0:
                after=parameter_digest(model)
                if before==after:raise ValueError('First update left all weights unchanged')
                write(root,'FIRST_UPDATE_VERIFIED.json',dict(status='PASSED',ranks=[dict(rank=0,sha256=after,changed=True)],before_sha256=before,
                    optimizer_updates=1,accumulated_microbatches=8,global_batch=8,schedule_total_updates=4172))
        stage='verify_receipts'
        result=verify(root);write(root,'SMOKE_PASSED.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        write(root,'JOB_FAILED.json',dict(status='FAILED',stage=stage,error_type=type(exc).__name__,error=str(exc),completed_updates=len(completed),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'SMOKE_STARTED.json').open('x') as stream:json.dump(dict(status='STARTED',pid=os.getpid()),stream)
    main(args.output_dir)
