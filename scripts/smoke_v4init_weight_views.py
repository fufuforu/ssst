"""Dedicated one-window, one-GPU smoke. Two updates, global8 by accumulation.

Never submits another job or writes full model weights. Receipts contain actual
GPU shapes, timing/memory, gradients, isolated-gradient routing and weight hashes.
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
    actual=batch_shapes(batch);nv=len(novel);total=2+nv
    expected=dict(images_input=[1,2,3,256,256],images_all=[1,total,3,256,256],images_output=[1,nv,3,256,256],frame_ids=[1,total])
    if len(context)!=2 or nv not in (2,4) or len(set(frames))!=total or actual!=expected:
        raise ValueError(f'Declared two-context training view contract differs: actual={actual}, expected={expected}, frames={frames}')
    actual_frames=batch['frame_ids'][0].tolist()
    if actual_frames!=frames:raise ValueError(f'Batch frame order differs: actual={actual_frames}, expected={frames}')
    return dict(status='PASSED',shapes=actual,frame_ids=actual_frames,context_views=2,novel_views=nv,supervised_views=total)


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
        from scripts.v4init_isolated_recipe import initialize_from_v4init_zero as initialize_olda4_ablation,recipe_identity,optimizer_groups
        from scripts.unified_query_validation import parameter_digest
        from scripts.v4init_weight_views_pipeline import verify_smoke as verify
        if torch.cuda.device_count()!=1:raise ValueError('Exactly one allocated GPU required')
        if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Dedicated smoke supports one rank only')
        torch.cuda.set_device(0);device=torch.device('cuda',0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if sha!=(root/'code.sha').read_text().strip():raise ValueError('Snapshot identity differs')
        subprocess.run(['git','diff','--exit-code','HEAD','--','tokengs','scripts','tests'],cwd=ROOT,check=True)
        run_cfg=json.loads((root/'configuration.json').read_text());variant=run_cfg['model_key']
        stage='source_and_data'
        source=root/'checkpoint_untrained_initial.pt'
        if digest(source)!=(root/'untrained_initial.sha256').read_text().strip():raise ValueError('Source SHA differs')
        saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        config=UnifiedQueryConfig(**saved['meta']['config'])
        windows,_,_=cohorts()
        training_identity={}
        if run_cfg['novel_views']==4:
            from scripts.training_windows_2ctx4novel import load_training_windows,DEFAULT_MANIFEST
            windows,training_identity=load_training_windows(run_cfg['training_window_manifest'],windows,digest(MANIFEST))
        index=epoch_order(len(windows),0)[0];window=windows[index]
        if len(windows)!=8337 or len({w['scene'] for w in windows})!=1191:raise ValueError('Training cohort differs')
        stage='model_and_zero_migration'
        opt=config_defaults['train_siu3r_unified_query_v1']
        if run_cfg['novel_views']==4:opt=opt.evolve(num_views=6)
        variant=json.loads((root/'configuration.json').read_text())['model_key']
        model=build_structural_model(variant,opt,config).to(device)
        evidence=initialize_olda4_ablation(saved,model,
            manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=1)
        evidence['training_data_extension']=training_identity
        write(root,'OLDA4_INITIALIZATION.json',evidence)
        restore_rng(saved['rank_rng'][0]);del saved
        optimizer=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.95),eps=1e-8)
        groups=model.parameter_groups()
        diagnostics=groups
        reconstruction=diagnostics['reconstruction']
        understanding=diagnostics['understanding']
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
            lr_base=1e-4,lr_warmup=100,lr_floor=1e-5,understanding_weight=1.,understanding_warmup=1043,
            supervised_layers=list(model.supervised_layers),layer_weights=list(model.layer_weights),image_hw=list(opt.img_size),
            gpu_name=torch.cuda.get_device_name(0),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            gs_color_initialization_prior=False,write_strength=0.,formal_weights_from_smoke=False)
        budget.update(recipe_identity(variant));budget.update(training_identity)
        write(root,'BUDGET.json',budget)
        stage='batch_load';batch=build_batch(opt,window,device)
        write(root,'BATCH_SHAPES.json',dict(status='OBSERVED',shapes=batch_shapes(batch)))
        contract=validate_batch_contract(batch,window)
        write(root,'BATCH_SHAPES.json',contract);print(json.dumps(contract),flush=True)
        budget.update(context_views=2,novel_views=run_cfg['novel_views'],supervised_views=2+run_cfg['novel_views'],batch_contract=contract)
        write(root,'BUDGET.json',budget)
        model.train()
        observed_fm=[]
        def observe_fm(module,args,out):
            if tuple(out.shape)!=(1,2,256,128,128):raise ValueError('Original A4 native FM128 changed')
            observed_fm.append(list(out.shape))
        model.core.semantic_memory.register_forward_hook(observe_fm)
        expected_ca=False
        assert all(hasattr(l,'anchor_attention')==expected_ca and hasattr(l,'image_attention') for l in model.core.panoptic.layers.values())
        for update in range(2):
            stage=f'update_{update+1}';optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=group['base_lr']*lr_factor('full',update,4172)
            weight=run_cfg['understanding_weight']*min(1.,(update+1)/1043)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic();losses=[]
            shapes=None
            for micro in range(8):
                output,metrics=model.step_loss(batch,step=update*8,understanding_weight=weight)
                prediction=output['prediction']
                if prediction['target_camera_calibration']['c2w'].shape[1]!=2+run_cfg['novel_views']:raise ValueError('Calibration view count differs')
                g=prediction['gaussians'];pa=prediction['gaussian_parent_anchor'];pq=prediction['gaussian_parent_query']
                expected_anchors=1024
                if tuple(g.shape)!=(1,expected_anchors*64,14) or pa.shape!=pq.shape or pa.numel()!=expected_anchors*64 or not torch.equal(pa,pq):raise ValueError('GS shape/parent aliases differ')
                if prediction['unified_query'].shape[-1]!=256 or prediction['num_thing_queries']!=prediction['q_pre'].shape[1]:raise ValueError('Understanding count/dimension changed')
                shapes=dict(gaussians=list(g.shape),gaussian_parent_anchor=list(pa.shape),gaussian_parent_query=list(pq.shape),parent_alias_equal=True,
                    reconstruction_anchors=expected_anchors,object_queries=prediction['num_thing_queries'],semantic_fm_resolution=128)
                if update==0 and micro==0:
                    stage='directional_LU';optimizer.zero_grad(set_to_none=True)
                    metrics['loss_understanding'].backward(retain_graph=True)
                    lu={k:leakage(ps) for k,ps in diagnostics.items()}
                    if lu['reconstruction']['nonzero_gradient_elements']!=0 or lu['reconstruction']['norm']!=0 or lu['understanding']['norm']<=0 or lu['shared']['norm']<=0:raise ValueError('LU isolation failed')
                    optimizer.zero_grad(set_to_none=True);stage='directional_LR'
                    metrics['loss_recon'].backward(retain_graph=True)
                    lr={k:leakage(ps) for k,ps in diagnostics.items()}
                    if lr['reconstruction']['norm']<=0 or any(lr[g]['nonzero_gradient_elements']!=0 for g in ('understanding','shared')):raise ValueError('LR isolation failed')
                    write(root,'DIRECTIONAL_GRADIENTS.json',dict(status='PASSED',LU=lu,LR=lr,gradient_isolation=True,
                        method='separate_backward_retained_graph_read_parameter_grad',note='LU reconstruction gradients exactly zero; LR understanding/shared gradients exactly zero'))
                    optimizer.zero_grad(set_to_none=True);stage=f'update_{update+1}'
                loss=metrics['loss']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss')
                losses.append(float(loss.detach()));(loss/8).backward()
                del output,prediction,metrics,g,pa,pq,loss
            grad_norms={name:norm(ps) for name,ps in diagnostics.items()}
            if any(not math.isfinite(x) or x<=0 for x in grad_norms.values()):raise ValueError('Missing/nonfinite group gradient')
            if any(p.grad is not None for module in (model.frozen_omega,model.sam_proposals) if module is not None for p in module.parameters()):raise ValueError('Frozen backbone has gradients')
            init_norms={name:float(p.grad.detach().float().norm()) if p.grad is not None else 0. for name,p in model.core.v4_initialization.named_parameters()}
            if any(not math.isfinite(x) or x<=0 for x in init_norms.values()):raise ValueError('V4 initialization missing/nonfinite gradient')
            clip=float(torch.nn.utils.clip_grad_norm_(all_params,1.,error_if_nonfinite=True));optimizer.step()
            torch.cuda.synchronize();seconds=time.monotonic()-started
            row=dict(update=update+1,microbatches=8,unique_windows=1,loss_mean=sum(losses)/8,
                learning_rate=1e-4*lr_factor('full',update,4172),learning_rates={g['family']:g['lr'] for g in optimizer.param_groups},understanding_weight=weight,
                seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gradient_norms_before_clip=grad_norms,clip_input_norm=clip,shapes=shapes)
            row['observed_native_fm_shape']=observed_fm[-1]
            row['v4_initialization_gradient_norms_before_clip']=init_norms
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
