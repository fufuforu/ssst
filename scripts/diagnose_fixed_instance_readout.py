"""Two fixed models, matched 4+4 windows; GPU export and mechanism diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,gc,hashlib,json,os,random,subprocess,time
import numpy as np
import torch
import torch.distributed as dist
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.unified_query_loss import unified_instance_loss
from tokengs.models.gaussian_instance_consistency import contribution_instance_targets,direct_gaussian_consistency_loss,local_different_instance_neighbors
from tokengs.options import config_defaults
from scripts.train_unified_query_v1 import restore_state
from scripts.train_dual_stream_v1 import build_batch,digest,MANIFEST,HELDOUT
from scripts.unified_query_validation import parameter_digest
from scripts.export_object_locus_v3_set_official import assemble_panoptic
from scripts.instance_boundary_diagnostic_math import gaussian_pair_probe,distribution


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def query_readout(p,assignment):
    """GT only labels the frozen response; never affects generated predictions."""
    qi,ki=p['final_pairs'][0];targets=p['final_targets'];n=p['num_thing_queries']
    masks=p['region_mass'][0,:2,:n].permute(1,0,2,3).float()
    gt=targets['gt_pixel_masks'][0];valid=targets['valid_pixels'][0]
    labels=assignment['labels'][0];gmem=p['gaussian_membership'][0,:,:n]
    reliable=labels>=0
    near,within=local_different_instance_neighbors(p['gaussians'][0,reliable,:3],labels[reliable],p['cloud']['scale'])
    rel_indices=reliable.nonzero().flatten();rows=[];native_ious=[]
    for k in range(len(gt)):
        assigned=qi[ki==k];truth=gt[k]&valid;area=int(truth.sum())
        def best(q,threshold):
            if not len(q):return 0.
            pred=(masks[q]>=threshold)&valid
            return float(((pred&truth).sum((1,2,3))/(pred|truth).sum((1,2,3)).clamp_min(1)).max())
        threshold_iou={str(t):best(assigned,t) for t in (.1,.3,.5,.7)}
        if len(assigned):native_ious.append(threshold_iou['0.5'])
        own=(labels==k);negative=rel_indices[near[(labels[reliable]==k)]][within[(labels[reliable]==k)]] if len(rel_indices) else rel_indices
        negative=negative.unique()
        response=gmem[:,assigned].max(1).values if len(assigned) else None
        prob=p['p_class'][0,assigned]
        rows.append(dict(gt_index=k,gt_instance_id=int(targets['gt_instance_ids'][0][k]),gt_class=int(targets['gt_classes'][0][k]),
            gt_pixels=area,assigned_queries=assigned.tolist(),assigned_count=len(assigned),
            assigned_best_iou_thresholds=threshold_iou,all_query_best_iou=best(torch.arange(n,device=masks.device),.5),
            assigned_correct_class_count=int((prob.argmax(-1)==targets['gt_classes'][0][k]-2).sum()),
            assigned_non_background_count=int((prob.argmax(-1)!=18).sum()),
            assigned_nms_eligible_count=int(p['instance_eligible'][assigned].sum()),
            own_gaussians=int(own.sum()),local_other_gaussians=len(negative),
            assigned_max_membership_on_own=distribution(response[own]) if response is not None else dict(count=0),
            assigned_max_membership_on_local_other=distribution(response[negative]) if response is not None else dict(count=0)))
    return dict(gt_rows=rows,native_supported_gt_iou=sum(native_ious)/len(native_ious) if native_ious else 0.,
        native_supported_gt_count=len(native_ious),queries=n,nms_kept=int(p['instance_eligible'].sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('Four independent inference workers required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pre=json.loads((root/'preflight.json').read_text());spec=pre['models'][rank//2]
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=pre['code_sha']:raise ValueError('Code differs')
    if digest(MANIFEST)!=pre['manifest_sha256'] or digest(HELDOUT)!=pre['heldout_sha256']:raise ValueError('Manifest differs')
    if rank==0:
        if (root/'STARTED.json').exists():raise ValueError('Refusing duplicate diagnostic')
        write(root/'STARTED.json',dict(code_sha=pre['code_sha'],parameter_updates=0,workers=4,models=2,windows=16))
    dist.barrier()
    if digest(Path(spec['checkpoint']))!=spec['sha256']:raise ValueError('Checkpoint changed')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    opt=config_defaults['train_siu3r_unified_query_v1']
    model=build_structural_model(spec['architecture_key'],opt,UnifiedQueryConfig(**meta['config'])).to(device).eval()
    if model.architecture_name!=meta['architecture']:raise ValueError('Architecture differs')
    if any(meta.get(k)!=v for k,v in model.extra_checkpoint_metadata.items()):raise ValueError('Recipe differs')
    if meta['omega_identity']!=model.frozen_omega.source_identity or meta['sam_identity']!=model.sam_proposals.source_identity:raise ValueError('Assets differ')
    restore_state(model,saved['model']);source=hashlib.sha256()
    for name,param in model.named_parameters():
        if param.requires_grad:
            if not torch.equal(param.detach().cpu(),saved['model'][name]):raise ValueError('Loaded parameter differs')
            source.update(name.encode());source.update(saved['model'][name].contiguous().numpy().tobytes())
    before=parameter_digest(model)
    if before!=source.hexdigest():raise ValueError('Loaded source SHA differs')
    versions={name:p._version for name,p in model.named_parameters()}
    del saved;gc.collect();folder=root/spec['name'];folder.mkdir(exist_ok=True)
    indices=list(range(rank%2,8,2))
    with torch.no_grad():
        for index in indices:
            started=time.monotonic();window=pre['windows'][index]
            random.seed(42+index);np.random.seed(42+index);torch.manual_seed(42+index);torch.cuda.manual_seed_all(42+index)
            batch=build_batch(opt,window,device);p=model.generate(batch['images_input'])
            cache=torch.load(Path(spec['run'])/'probes/final'/f'{index:02d}.pt',map_location='cpu',weights_only=False)
            if not torch.equal(batch['images_input'].cpu(),cache['context_rgb']) or cache['scene']!=window['scene']:raise ValueError('Probe inputs differ')
            for name in ('gaussians','gaussian_features','gaussian_instance_features','rendered_instance_features','region_mass'):
                if not torch.isfinite(p[name]).all():raise ValueError('Nonfinite '+name)
            decoder=ModelInputDecoder(cam_view=torch.linalg.inv(p['predicted_context_c2w']).transpose(-1,-2),intrinsics=p['predicted_context_intrinsics'])
            replay=model.render_instance_features(p,decoder)['rendered_instance_features']
            torch.testing.assert_close(replay,p['rendered_instance_features'],rtol=1e-4,atol=1e-5)
            # Matching is diagnostic only, after generation; same native assignment as historical probes.
            unified_instance_loss(p,batch,model.config);target=p['final_targets']
            gt_ref=torch.load(Path(pre['gt_cache_root'])/'probes/final'/f'{index:02d}_coupling.pt',map_location='cpu',weights_only=False)
            if not torch.equal(batch['semantic_label_all'][:,:2].cpu(),gt_ref['gt_semantic']) or not torch.equal(batch['instance_label_all'][:,:2].cpu(),gt_ref['gt_instance']):raise ValueError('GT changed')
            assignment=contribution_instance_targets(p['gaussians'],decoder,model.gs,target)
            direct,terms=direct_gaussian_consistency_loss(p['gaussian_instance_features'],p['gaussians'][...,:3],p['gaussian_parent_query'],assignment,p['cloud']['scale'])
            pair={name:gaussian_pair_probe(f,p['gaussians'][0,:,:3],assignment['labels'][0],p['gaussian_parent_query'])
                for name,f in [('child256',p['gaussian_features'][0]),('instance8d',p['gaussian_instance_features'][0])]}
            readout=query_readout(p,assignment)
            semantic,instance,_=assemble_panoptic(p)
            raw=p['region_mass'][0,:2,:p['num_thing_queries']].permute(1,0,2,3)>=.5
            supported=torch.zeros(len(target['gt_pixel_masks'][0]),dtype=torch.bool,device=device);supported[p['final_pairs'][0][1].unique()]=True
            rgb_difference=float((p['render']['images_pred'].cpu()-cache['rgb']).abs().max())
            if rgb_difference>.02:raise ValueError('Unexpected native RGB replay difference')
            row=dict(model=spec['name'],index=index,scope='train' if index<4 else 'heldout',scene=window['scene'],context=window['context'],
                gaussian_features=pair,direct_metrics={k:float(v) for k,v in terms.items()},readout=readout,
                cached_rgb_max_difference=rgb_difference,cached_supported_iou=pre['models'][rank//2]['probes'][index]['best_supported_gt_mask_iou'],
                native_feature_render_replay=True,seconds=time.monotonic()-started)
            write(folder/f'{index:02d}_mechanism.json',row)
            torch.save(dict(scene=window['scene'],context=window['context'],context_rgb=batch['images_input'].cpu(),
                features=p['rendered_instance_features'][0].float().cpu(),alpha=p['alpha'][0,:,0].float().cpu(),
                raw_membership_masks=raw.cpu(),nms_eligible=p['instance_eligible'].cpu(),assembled_instance=instance.cpu(),
                predicted_semantic=p['semantic_scores'][0].argmax(1).cpu(),
                gt_masks=target['gt_pixel_masks'][0].cpu(),gt_ids=target['gt_instance_ids'][0].cpu(),
                gt_semantic=batch['semantic_label_all'][0,:2].cpu(),gt_instance=batch['instance_label_all'][0,:2].cpu(),
                valid=target['valid_pixels'][0].cpu(),supported_gt=supported.cpu(),
                gaussian_instance_features=p['gaussian_instance_features'].cpu(),gaussian_child_features=p['gaussian_features'].cpu(),
                gaussians=p['gaussians'].cpu(),parent=p['gaussian_parent_query'].cpu(),assignment={k:v.cpu() for k,v in assignment.items()},
                gaussian_membership=p['gaussian_membership'].cpu(),scene_scale=p['cloud']['scale'].cpu()),folder/f'{index:02d}_inference.pt')
            if index==indices[0]:write(folder/f'FIRST_WINDOW_GATE_rank{rank:02d}.json',dict(status='PASSED',source_parameter_sha=before,native_feature_replay=True,gt_and_rgb_cache_equal=True))
            print(json.dumps(dict(rank=rank,model=spec['name'],window=index,seconds=row['seconds'])),flush=True)
            del p,batch,cache,replay,target,gt_ref,assignment,pair,raw;gc.collect();torch.cuda.empty_cache()
    after=parameter_digest(model)
    if before!=after or any(p._version!=versions[name] for name,p in model.named_parameters()):raise ValueError('Fixed parameters changed')
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Gradients accumulated')
    write(folder/f'FIXED_WEIGHTS_rank{rank:02d}.json',dict(status='PASSED',source=source.hexdigest(),before=before,after=after,
        all_parameter_versions_unchanged=True,no_parameter_gradients=True,parameter_updates=0,optimizer_created=False,windows=indices))
    dist.barrier()
    if rank==0:write(root/'GPU_COMPLETE.json',dict(status='FIXED_WEIGHT_EXPORT_COMPLETE',models=2,windows=16,parameter_updates=0))
    dist.destroy_process_group()


if __name__=='__main__':main()
