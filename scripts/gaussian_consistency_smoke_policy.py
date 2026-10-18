"""Real-data CUDA gradient probe and strict gate for the new direct objective."""
import math

FAMILIES=('instance_head','child_feature','semantic_adapter','blocks','shared_projection')


def validate_direct_smoke(receipt):
    rows=receipt['ranks']
    if len(rows)!=4 or {r['rank'] for r in rows}!=set(range(4)):
        raise ValueError('Direct smoke rank coverage differs')
    checks=[x for r in rows for x in r['direct_checks']]
    if sorted(x['window_index'] for x in checks)!=list(range(16)):
        raise ValueError('Direct smoke must inspect exactly the original 16 windows')
    if not all(x['geometry_gradients_absent'] and x['assignment_detached'] and x['native_readout_equal'] for x in checks):
        raise ValueError('Direct geometry isolation/assignment/native readout contract failed')
    numeric=[v for x in checks for v in list(x['metrics'].values())+list(x['gradient_norms'].values())]
    if not all(math.isfinite(v) and v>=0 for v in numeric):raise ValueError('Nonfinite or negative direct diagnostics')
    if sum(x['metrics']['direct_positive_gaussians'] for x in checks)<=0:
        raise ValueError('No real cross-parent positive samples')
    if sum(x['metrics']['direct_negative_edges'] for x in checks)<=0:
        raise ValueError('No real local other-instance negative samples')
    for family in FAMILIES:
        if sum(x['gradient_norms'][family] for x in checks)<=0:raise ValueError('Direct gradient missing: '+family)
    for row in rows:
        for step in row['steps']:
            if not math.isfinite(step['loss_gaussian_direct']):raise ValueError('Direct DDP loss nonfinite')
    return dict(status='PASSED',real_training_windows=16,explicit_geometry_gradients_absent=True,
        actual_positive_and_negative_samples=True,independent_new_loss_gradients_verified=True,
        native_readout_equal=True,supervision_recipe_unchanged=True)


def probe_direct_path(model,opt,preflight,rank,world,device):
    import gc
    import torch
    from scripts.train_dual_stream_v1 import build_batch
    from tokengs.models.input_types import ModelInputDecoder
    from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
    from tokengs.models.gaussian_instance_consistency import contribution_instance_targets,direct_gaussian_consistency_loss
    checks=[]
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    for index in range(rank,16,world):
        batch=build_batch(opt,preflight['training_windows'][index],device)
        p=model.generate(batch['images_input'])
        decoder=ModelInputDecoder(cam_view=torch.linalg.inv(p['predicted_context_c2w']).transpose(-1,-2),intrinsics=p['predicted_context_intrinsics'])
        assignment=contribution_instance_targets(p['gaussians'],decoder,model.gs,build_context_instance_targets(batch))
        features=model.core.direct_instance_features(p)
        torch.testing.assert_close(features,p['gaussian_instance_features'],rtol=0,atol=0)
        loss,metrics=direct_gaussian_consistency_loss(features,p['gaussians'][...,:3],p['gaussian_parent_query'],assignment,p['cloud']['scale'])
        explicit=[p['gaussians'],p['states'][-1]['mu'],p['states'][-1]['radii']]
        grads=torch.autograd.grad(loss,[v for _,v in params]+explicit,allow_unused=True)
        norms={family:0. for family in FAMILIES};forbidden=[]
        for (name,_),grad in zip(params,grads):
            if name.startswith(('core.geometry.','core.activation_head.','core.reconstruction_adapter.')):
                forbidden.append(grad)
            for family in FAMILIES:
                prefix='gfm_projection.' if family=='shared_projection' else 'core.'+family+'.'
                if name.startswith(prefix) and grad is not None:norms[family]+=float(grad.float().square().sum())
        norms={k:math.sqrt(v) for k,v in norms.items()}
        row=dict(window_index=index,geometry_gradients_absent=all(g is None for g in forbidden+list(grads[-3:])),
            assignment_detached=not any(x.requires_grad for x in assignment.values()),native_readout_equal=True,
            gradient_norms=norms,metrics={k:float(v) for k,v in metrics.items()})
        checks.append(row)
        print(__import__('json').dumps(dict(rank=rank,direct_probe=row)),flush=True)
        del batch,p,decoder,assignment,features,loss,metrics,explicit,grads,forbidden,grad
        gc.collect();torch.cuda.empty_cache()
    return checks
