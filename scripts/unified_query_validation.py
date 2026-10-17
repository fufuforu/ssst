"""Bounded trainability probes; no official full-cohort claims or loss-drop gate."""
import hashlib,json,math,random
import torch
import torch.distributed as dist
from scripts.train_dual_stream_v1 import build_batch
from scripts.export_object_locus_v3_set_official import assemble_panoptic


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def synchronized_first_update(model,before,root,rank,world):
    after=parameter_digest(model);rows=[None]*world
    if world>1:dist.all_gather_object(rows,dict(rank=rank,sha256=after,changed=after!=before))
    else:rows=[dict(rank=rank,sha256=after,changed=after!=before)]
    if len({r['sha256'] for r in rows})!=1 or not all(r['changed'] for r in rows):raise ValueError('First update unchanged or ranks diverged')
    if rank==0:(root/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(dict(status='PASSED',all_trainable_parameters=True,ranks=rows),indent=2)+'\n')


@torch.no_grad()
def run_probes(model,opt,train,validation,root,label,rank,world,device,weight):
    py=random.getstate();was_training=model.training;model.eval();rows=[]
    with torch.random.fork_rng(devices=[device.index]):
        for index,window in enumerate(train[:4]+validation[:4]):
            if index%world!=rank:continue
            output,metrics=model.step_loss(build_batch(opt,window,device),understanding_weight=weight)
            p=output['prediction'];qi,ki=p['final_pairs'][0];t=p['final_targets'];n=p['num_thing_queries']
            sem,ins,_=assemble_panoptic(p)
            valid=t['valid_pixels'][0].flatten();mask=p['region_mass'][0,:2,:n].permute(1,0,2,3).flatten(1)[:,valid]>=.5
            target=t['gt_pixel_masks'][0].flatten(1)[:,valid].bool()
            ious=[]
            for gt in ki.unique():
                pred=mask[qi[ki==gt]];truth=target[gt]
                iou=(pred&truth).sum(1)/(pred|truth).sum(1).clamp_min(1);ious.append(float(iou.max()))
            scalars={k:float(v) for k,v in metrics.items() if isinstance(v,(int,float)) or torch.is_tensor(v) and v.numel()==1}
            if not all(math.isfinite(v) for v in scalars.values()):raise FloatingPointError('Nonfinite probe')
            semantic_target=batch['semantic_label_all'][:,:2]
            semantic_valid=(semantic_target>=0)&(semantic_target<20)
            semantic_accuracy=float((p['semantic_scores'].argmax(2)[semantic_valid]==semantic_target[semantic_valid]).float().mean())
            row=dict(raw_semantic_pixel_accuracy=semantic_accuracy,scope='train' if index<4 else 'heldout',scene=window['scene'],context=window['context'],metrics=scalars,
                best_supported_gt_mask_iou=sum(ious)/max(1,len(ious)),supported_gt_count=len(ious),
                nms_kept=int(p['instance_eligible'].sum()),final_instances=int(ins.unique().numel()-int((ins==0).any())),
                source_observations=len(p['cloud']['source_indices']),fused_points=len(p['cloud']['points']),gaussians=p['gaussians'].shape[1])
            rows.append(row)
            folder=root/'probes'/label;folder.mkdir(parents=True,exist_ok=True)
            torch.save(dict(scene=window['scene'],context_rgb=p['generation_context_rgb'].cpu(),rgb=p['render']['images_pred'].cpu(),
                semantic=sem.cpu(),raw_semantic=p['semantic_scores'].argmax(2).cpu(),instance=ins.cpu(),p_class=p['p_class'].cpu(),eligible=p['instance_eligible'].cpu()),folder/f'{index:02d}.pt')
            del output,p,mask,target
    random.setstate(py);model.train(was_training)
    allrows=[None]*world
    if world>1:dist.all_gather_object(allrows,rows)
    else:allrows=[rows]
    if rank==0:(root/f'PROBES_{label.upper()}.json').write_text(json.dumps(dict(parameter_updates=0,official_128=False,rows=[r for group in allrows for r in group]),indent=2)+'\n')
