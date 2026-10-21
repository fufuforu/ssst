"""Strict real-input M1, complete-step smoke, and one fixed 40-update M2."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import gc
import torch
from scripts.object_locus_joint_v1_runtime import (
    build_model, build_optimizer, build_batch, build_manifest, build_plan, capture_rng,
    restore_rng, train_one_step, write_json, REPORTS_DEFAULT, seed_everything,
    scientific_state_sha, backward_gradient_controlled, understanding_weight, implementation_hashes)
from scripts.object_locus_joint_v1_runtime import assert_joint_gpu
from scripts.eval_object_locus_v3_set import evaluate_windows


def cpu_tensors(value, path='root'):
    rows = {}
    if torch.is_tensor(value): rows[path] = value.detach().cpu().clone()
    elif isinstance(value, dict):
        for key, child in value.items():
            if key.startswith('joint_'): continue
            rows.update(cpu_tensors(child, f'{path}.{key}'))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value): rows.update(cpu_tensors(child, f'{path}[{i}]'))
    return rows


def loss_forward(model, batch, step):
    model.eval(); model.understanding_step = step
    with torch.no_grad():
        out, metrics = model.step_loss(batch, step=step,
            understanding_weight=understanding_weight(step))
        return cpu_tensors({'output': out, 'metrics': metrics})


def envelope(reference, repeat, actual):
    result = {}; first = None
    for key, a in reference.items():
        b, c = repeat[key], actual[key]
        if a.shape != c.shape or a.dtype != c.dtype:
            raise AssertionError(f'shape/dtype diverged at {key}')
        if a.is_floating_point():
            e = float((a-b).abs().max()) if a.numel() else 0.0
            d = float((a-c).abs().max()) if a.numel() else 0.0
            passed = d <= e
        else:
            e = 0.0; d = float(not torch.equal(a,c)); passed = d == 0
        result[key] = {'E_X': e, 'difference': d, 'passed': passed}
        if not passed and first is None: first = {'tensor': key, **result[key]}
    return {'passed': first is None, 'first_divergence': first, 'tensors': result}



def check_image_invariance(gaussians, batch, opt, root):
    from tokengs.rendering.gs import GaussianRenderer, rasterization
    camera=batch['cam_view_all'];intr=batch['intrinsics_all']
    viewmat=camera.float().transpose(-1,-2)
    ks=torch.zeros((*intr.shape[:2],3,3),device=gaussians.device,dtype=torch.float32)
    ks[...,0,0]=intr[...,0];ks[...,1,1]=intr[...,1]
    ks[...,0,2]=intr[...,2];ks[...,1,2]=intr[...,3];ks[...,2,2]=1
    height,width=opt.img_size
    images=[];alphas=[];depths=[]
    for b in range(gaussians.shape[0]):
        rgbd,alpha,_=rasterization(means=gaussians[b,:,:3].contiguous(),
            quats=gaussians[b,:,7:11].contiguous(),scales=gaussians[b,:,4:7].contiguous(),
            opacities=gaussians[b,:,3].contiguous(),colors=gaussians[b,:,11:].contiguous(),
            viewmats=viewmat[b],Ks=ks[b],width=width,height=height,
            near_plane=opt.znear,far_plane=opt.zfar,packed=False,
            backgrounds=torch.ones(camera.shape[1],3,device=gaussians.device),render_mode='RGB+ED')
        images.append(rgbd[...,:3].permute(0,3,1,2));depths.append(rgbd[...,3:].permute(0,3,1,2))
        alphas.append(alpha.permute(0,3,1,2))
    raw={'images_pred':torch.stack(images),'alphas_pred':torch.stack(alphas),'depths_pred':torch.stack(depths)}
    fixed=GaussianRenderer(opt).render(gaussians,camera,intrinsics=intr)
    report={'window':{'scene':'scene0018_00','context':[63,75],'novel':[69,72]},
            'fields':{k:{'exact':torch.equal(v,fixed[k]),'max_abs_difference':float((v-fixed[k]).abs().max())} for k,v in raw.items()}}
    report['passed']=all(v['exact'] for v in report['fields'].values())
    write_json(root/'common_fix_image_invariance.json',report)
    if not report['passed']:raise AssertionError('Common fix changed image rendering')

def m1(root, plan):
    from scripts.object_locus_v3_set_runtime import build_model as build_v3
    reference, _, _ = build_v3('cpu')
    control, opt, transfer = build_model('cpu', arm='control')
    for key, value in reference.state_dict().items():
        if not torch.equal(value, control.state_dict()[key]):
            raise AssertionError(f'common initialization differs: {key}')
    del control; gc.collect()
    reports = []
    for entry in plan['entries'][:3]:
        batch = build_batch(opt, entry['identity'], 'cuda')
        reference.cuda(); rng = capture_rng()
        restore_rng(rng); a = loss_forward(reference, batch, 0)
        restore_rng(rng); b = loss_forward(reference, batch, 0)
        if entry['step']==1 and not (root/'common_fix_image_invariance.json').exists():
            check_image_invariance(a['root.output.prediction.gaussians'].cuda(),batch,opt,root)
        reference.cpu(); torch.cuda.empty_cache()
        for arm in ('control', 'joint'):
            model, _, _ = build_model('cuda', arm=arm)
            restore_rng(rng); c = loss_forward(model, batch, 0)
            report = envelope(a,b,c)
            report.update(reference_name='V3 control + common analytic visibility correctness fix',
                          arm=arm, window=entry['identity'],
                          gpu=assert_joint_gpu(),
                          peak_allocated=torch.cuda.max_memory_allocated(),
                          peak_reserved=torch.cuda.max_memory_reserved(),
                          implementation_hashes=implementation_hashes())
            reports.append(report); write_json(root/'m1.json', reports)
            del model, c; gc.collect(); torch.cuda.empty_cache()
            if not report['passed']:
                print(report['first_divergence'], flush=True)
                raise AssertionError('M1 failed; no tolerance floor permitted')
        del a,b,batch
    del reference; gc.collect()
    return transfer


def intervention(model, opt, window, step):
    from tokengs.models.input_types import split_data, ModelInput, ModelInputDecoder
    batch = build_batch(opt, window, 'cuda'); mi,_ = split_data(batch,opt)
    dec = ModelInputDecoder(cam_view=batch['cam_view_all'], intrinsics=batch['intrinsics_all'])
    was, rng = model.training, capture_rng(); model.eval(); model.understanding_step=step
    def forward(disabled=(), mean=False):
        restore_rng(rng)
        model.anchor_decoder.disabled_layers = disabled
        model.anchor_decoder.object_mean_message = mean
        with torch.no_grad():
            out = model.forward_object_locus(ModelInput(mi.encoder,dec), context_decoder=dec)
            return {'gaussians':out['gaussians'].detach().cpu(),
                    'mu':out['states'][-1]['mu'].detach().cpu(),
                    'ell':float(out['states'][-1]['ell']),
                    'layers':{f'L{s["layer"]}':{
                        'u_norm':float(s['joint_u'].norm()),
                        'delta_norm':float(s['joint_delta'].norm()),
                        'relative':(s['joint_delta'].norm(dim=-1)/(s['joint_h_norm']+1e-6)).cpu()
                    } for s in out['states'] if 'joint_delta' in s}}
    try:
        a,b = forward(),forward(); noise=(a['gaussians']-b['gaussians']).abs()
        rows={}
        for layer in (6,8,10,12):
            x=forward((layer,)); difference=(a['gaussians']-x['gaussians']).abs()
            key=f'L{layer}'; rel=a['layers'][key]['relative']
            rel_noise=(rel-b['layers'][key]['relative']).abs()
            rows[key]={
                'XYZ_max_over_ell':float(difference[...,:3].max())/a['ell'],
                'attributes_max':float(difference[...,3:].max()),
                'mu12_max_over_ell':float((a['mu']-x['mu']).abs().max())/a['ell'],
                'XYZ_exceeds_envelope':bool((difference[...,:3]>noise[...,:3]).any()),
                'attributes_exceed_envelope':bool((difference[...,3:]>noise[...,3:]).any()),
                'u_norm':a['layers'][key]['u_norm'], 'delta_norm':a['layers'][key]['delta_norm'],
                'relative_max':float(rel.max()), 'relative_noise_max':float(rel_noise.max()),
                'relative_exceeds_envelope':bool((rel>rel_noise).any())}
        x=forward(mean=True); diff=(a['gaussians']-x['gaussians']).abs()
        return {'layers':rows,'object_mean_exceeds_envelope':bool((diff>noise).any()),
                'object_mean_XYZ_max_over_ell':float(diff[...,:3].max())/a['ell'],
                'object_mean_attributes_max':float(diff[...,3:].max()),
                'enabled_XYZ_noise_max':float(noise[...,:3].max()),
                'enabled_attributes_noise_max':float(noise[...,3:].max())}
    finally:
        model.anchor_decoder.disabled_layers=();model.anchor_decoder.object_mean_message=False
        restore_rng(rng);model.train(was)


def gradient_probe(model, batch, step, rng):
    restore_rng(rng); model.zero_grad(set_to_none=True);model.train()
    out, metrics=model.step_loss(batch,step=step,understanding_weight=understanding_weight(step))
    backward_gradient_controlled(model,metrics['loss_recon'],metrics['loss_understanding'],understanding_weight(step))
    grads={key:p.weight.grad.detach().cpu().clone() for key,p in model.object_locus_joint_injection.items()}
    del out,metrics
    return grads



def diagnose_m1(root, plan):
    """Inspect the failed registered M1 window; preserve all scientific operations."""
    from scripts.object_locus_v3_set_runtime import build_model as build_v3
    model,opt,_=build_v3('cuda')
    batch=build_batch(opt,plan['entries'][0]['identity'],'cuda')
    rng=capture_rng(); original=model.render_reconstruction
    captured=[]
    def observe(reconstruction, decoder):
        render=original(reconstruction,decoder)
        captured.append({k:v.detach().cpu().clone() for k,v in render.items()
                         if torch.is_tensor(v)})
        return render
    model.render_reconstruction=observe
    restore_rng(rng);a=loss_forward(model,batch,0);ra=captured;captured=[]
    restore_rng(rng);b=loss_forward(model,batch,0);rb=captured
    model.render_reconstruction=original
    rows=[]
    # First render is the final forward; remaining calls are the unchanged layer objective.
    layers=['final_forward',*model.supervised_layers]
    for layer,x,y in zip(layers,ra,rb):
        fields={}
        for key in x:
            diff=(x[key]-y[key]).abs()
            fields[key]={'max_abs_difference':float(diff.max()),
                         'different_elements':int(torch.count_nonzero(diff)),
                         'finite_run1':bool(torch.isfinite(x[key]).all()),
                         'finite_run2':bool(torch.isfinite(y[key]).all())}
        rows.append({'layer':layer,'render_fields':fields})
    report={'window':plan['entries'][0]['identity'],
        'purpose':'M1 failure diagnosis only; no new experiment or changed scientific path',
        'original_V3_gaussians_exact':torch.equal(a['root.output.prediction.gaussians'],
                                                b['root.output.prediction.gaussians']),
        'all_original_V3_decoder_states_exact':all(torch.equal(v,b[k]) for k,v in a.items()
                                                   if '.states[' in k),
        'renders':rows,
        'installation_source_evidence':{
            'allocation':'gsplat/cuda/csrc/Projection.cpp uses at::empty for means2d',
            'early_return':'ProjectionEWA3DGSFused.cu returns before writing means2d when culled',
            'loss_read':'canonical_recon.py:186-195 reads all means2d without culling validity mask'},
        'limitation':'Installed source is evidence; cached binary build provenance is not established.',
        'status':'M1_FAILED; M2 and formal runs remain forbidden'}
    write_json(root/'m1_failure_diagnosis.json',report)
    print(report,flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('m1','m2','all','diagnose'),default='all');args=p.parse_args()
    gpu=assert_joint_gpu(); root=REPORTS_DEFAULT/'validation';root.mkdir(parents=True,exist_ok=True)
    plan=build_plan(build_manifest()); seed_everything(42)
    torch.cuda.reset_peak_memory_stats()
    if args.phase == 'diagnose':
        diagnose_m1(root,plan);return
    if args.phase in ('m1','all'):
        transfer=m1(root,plan)
        model,opt,_=build_model('cuda',arm='control');optimizer,opta=build_optimizer(model)
        batch=build_batch(opt,plan['entries'][0]['identity'],'cuda')
        out,metrics=train_one_step(model,optimizer,batch,1,failure_capture_dir=root/'failure')
        grads=metrics['gradient_report_before_clip']
        required=('object_locus_v3_set.class_head.weight','object_locus_v3_set.cls_fuse.weight',
          'object_locus_v3_set.mask_q_mlp.2.weight','object_locus_v3_set.child_mlp.2.weight',
          'anchor_decoder.mu','activation_head.deconv.weight')
        if not all(grads[n]['nonzero'] and grads[n]['finite'] for n in required):
            raise AssertionError('complete-step required gradient failure')
        del out;gc.collect();torch.cuda.empty_cache()
        model.understanding_step=1
        ev,_,_=evaluate_windows(model,opt,[plan['entries'][0]['identity']],1,'smoke',root,
            'cuda',build_batch,official=True,panels=True)
        write_json(root/'smoke.json',{'passed':True,'gpu':gpu,'transfer':transfer,
            'optimizer':opta,'metrics':metrics,'evaluation':ev,
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        import subprocess
        write_json(root/'final_tree_validation.json',{'passed':True,
            'git_sha':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            'implementation_hashes':implementation_hashes()})
        del model,optimizer,batch;gc.collect();torch.cuda.empty_cache()
    if args.phase in ('m2','all'):
        if not (root/'smoke.json').exists():raise RuntimeError('M1 smoke required before M2')
        import json
        if not json.loads((root/'smoke.json').read_text())['passed']:raise RuntimeError('M1 smoke failed')
        seed_everything(42);model,opt,transfer=build_model('cuda',arm='joint');init_sha=scientific_state_sha(model)
        optimizer,opta=build_optimizer(model)
        for entry in plan['entries'][:40]:
            batch=build_batch(opt,entry['identity'],'cuda')
            out,metrics=train_one_step(model,optimizer,batch,entry['step'],
                failure_capture_dir=root/'failure',failure_context={'entry':entry,'arm':'M2'})
            del out,batch
            if entry['step'] in (20,40):write_json(root/f'm2_step{entry["step"]}.json',metrics)
        probe=intervention(model,opt,plan['entries'][0]['identity'],40)
        batch=build_batch(opt,plan['entries'][0]['identity'],'cuda');rng=capture_rng()
        a=gradient_probe(model,batch,40,rng);b=gradient_probe(model,batch,40,rng)
        for key,row in probe['layers'].items():
            row.update(W_norm=float(model.object_locus_joint_injection[key].weight.detach().norm()),
                       W_gradient_norm=float(a[key].norm()),W_gradient_noise_norm=float((a[key]-b[key]).norm()))
            row['passed']=all((row['W_norm']>0,row['u_norm']>0,row['delta_norm']>0,
                row['relative_exceeds_envelope'],row['W_gradient_norm']>row['W_gradient_noise_norm'],
                row['XYZ_exceeds_envelope'],row['attributes_exceed_envelope']))
        probe.update(passed=all(r['passed'] for r in probe['layers'].values()) and probe['object_mean_exceeds_envelope'],
                     updates=40,init_sha=init_sha,gpu=gpu,transfer=transfer,optimizer=opta,
                     implementation_hashes=implementation_hashes(),
                     peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        write_json(root/'m2.json',probe)
        if not probe['passed']:raise RuntimeError('未形成有效干预；STOP，不启动正式训练')
    print('Validation passed',flush=True)


if __name__=='__main__':main()
