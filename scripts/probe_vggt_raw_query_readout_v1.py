"""Frozen forwards: locate mask loss before versus during panoptic assembly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.diagnose_vggt_joint_gradient_spikes_v1 import restored
from scripts.eval_vggt_feedback_expanded_v1 import windows
from scripts.eval_vggt_joint_short_controls_v1 import write
from scripts.analyze_vggt_instance_bottlenecks_v1 import mask_matrix,best_masks
from scripts.export_object_locus_v3_set_official import assemble_panoptic


def packed(path):
    x=np.asarray(Image.open(path),dtype=np.int64)
    return x[:,:,0]+256*x[:,:,1]+65536*x[:,:,2]


def encode_tensor(masks):
    # [V,Q,H,W] -> official lexicographic-frame concatenation [V*H,W,Q].
    v,q,h,w=masks.shape
    array=masks.permute(0,2,3,1).reshape(v*h,w,q).cpu().numpy().astype(np.uint8)
    return coco_mask.encode(np.asfortranarray(array))


def describe(masks,labels,gt,gtlabels):
    encoded=encode_tensor(masks);iou,pa,ga,_=mask_matrix(encoded,gt)
    keep=pa>0;iou=iou[keep];labels=np.asarray(labels)[keep]
    best,same=best_masks(iou,labels,gtlabels)
    return dict(gt=len(gt),nonempty_queries=int(keep.sum()),best50=int((best>=.5).sum()),best75=int((best>=.75).sum()),
        same50=int((same>=.5).sum()),same75=int((same>=.75).sum()),best_iou_sum=float(best.sum()),
        gt_best_iou=best.tolist(),gt_same_class_best_iou=same.tolist())


def raw_stages(render):
    member=render['region_mass'][0];alpha=render['alpha'][0,:,0];pc=render['p_class'][0]
    prob,cls=pc[:,:18].max(-1);eligible=(pc.argmax(-1)!=18)&(prob>=.05)
    raw=(member[:,:100]>=.5)&(alpha[:,None]>.05)
    candidates=torch.cat((raw&eligible[None,:,None,None],(member[:,100:102]>=.5)&(alpha[:,None]>.05)),1)
    scores=torch.cat((member[:,:100]*prob[None,:,None,None],member[:,100:102]),1)
    scores=torch.where(candidates,scores,torch.full_like(scores,-1.))
    maximum,winner=scores.max(1)
    ids=torch.arange(100,device=member.device)[None,:,None,None]
    won=(winner[:,None]==ids)&(maximum[:,None]>=0)
    _,instances,_=assemble_panoptic(render)
    final=instances[:,None]==(ids+1)
    return dict(raw_all=raw,eligible_raw=raw&eligible[None,:,None,None],winner_before_area_filter=won,final=final),cls.cpu().numpy()+2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expanded-root',type=Path,required=True)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=4:raise ValueError('four independent inference workers required')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
    if '4090' not in torch.cuda.get_device_name(local):raise ValueError('matched official4090 inference required')
    rows=json.loads(a.cohort.read_text())['records']
    if a.smoke:rows=rows[:1]
    choices=rows if a.smoke and rank<3 else [] if a.smoke else rows[rank::world]
    letters=['ABC'[rank]] if a.smoke and rank<3 else [] if a.smoke else list('ABC')
    all_windows,_=windows();root=a.output/f'rank{rank:02d}';root.mkdir(parents=True,exist_ok=False)
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render
    results=[]
    for arm in letters:
        model,opt,optimizer,_,identity=restored(arm,500,device,collective_hash=False,restore_optimizer=False);del optimizer
        model.eval().requires_grad_(False);model.understanding_step=4000
        versions={n:p._version for n,p in model.named_parameters()}
        for index,record in enumerate(choices):
            w=all_windows[record['source_index']];batch=build_batch(opt,w,device)
            rgb_hash=hashlib.sha256(batch['images_all'].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
            if rgb_hash!=record['input_rgb_sha256']:raise ValueError('provider RGB differs from official export')
            with torch.no_grad():output=generate_and_render(model,batch['images_input'],batch['images_all'])
            render=output['render'];generated=output['generated'];stages,labels=raw_stages(render)
            sem,ins,_=assemble_panoptic(render);frame_ids=batch['frame_ids'][0].cpu().tolist()
            pair=a.expanded_root/arm/f"rank{record['original_rank']:02d}"/'all'/record['name']
            views={}
            for view,ids in [('context',[0,1]),('target',list(range(6)))]:
                ids=sorted(ids,key=lambda i:f"{w['scene']}_pred{frame_ids[i]}.png")
                target=[]
                for i in ids:
                    # Native packed PNGs are the authoritative GT/mask ordering.
                    native_pred=packed(pair/f'{view}_seg_pred'/f"{w['scene']}_pred{frame_ids[i]}.png")
                    current=(sem[i].cpu().numpy()+1)*1000+ins[i].cpu().numpy()
                    current=np.where(sem[i].cpu().numpy()==20,ins[i].cpu().numpy(),current)
                    if not np.array_equal(native_pred,current):raise ValueError('native endpoint panoptic PNG not reproduced')
                    target.append(packed(pair/f'{view}_seg_gt'/f"{w['scene']}_gt{frame_ids[i]}.png"))
                target=np.concatenate(target,axis=0);gtids=np.unique(target%1000);gtids=gtids[gtids!=0]
                gtlabels=np.asarray([target[target%1000==g][0]//1000-1 for g in gtids])
                gt=coco_mask.encode(np.asfortranarray(np.stack([target%1000==g for g in gtids],-1).astype(np.uint8))) if len(gtids) else []
                views[view]={name:describe(mask[ids],labels,gt,gtlabels) for name,mask in stages.items()}
                if view=='context':
                    fm=generated['F_m'][0];mq=model.understanding.mask_embedder(generated['q_pre'])[0]
                    visual=F.interpolate(fm,size=(256,256),mode='bilinear',align_corners=False)
                    visual_mask=(torch.einsum('vchw,qc->vqhw',visual,mq).sigmoid()>=.5)
                    proxy=describe(visual_mask[ids],np.full(100,-1),gt,gtlabels)
                    proxy['label']='2D q_pre mask proxy, class-agnostic only; not a native trained segmentation result'
                    views[view]['visual_qpre_proxy']=proxy
            for view in views:
                if views[view]['raw_all']['best50']<views[view]['eligible_raw']['best50']:raise ValueError('mask eligibility ceiling violated')
            results.append(dict(arm=arm,name=record['name'],source_index=record['source_index'],selection_stratum=record['selection_stratum'],
                views=views,endpoint_pngs_reproduced=True,provider_rgb_reproduced=True,source=identity))
            write(root/'progress.json',dict(arm=arm,completed=len(results),expected=len(choices)*len(letters),optimizer_updates=0))
            print('RAW_READOUT_PROBED',rank,arm,index+1,record['name'],flush=True)
            del output,render,generated,stages,batch
        if any(p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('inference mutated model parameters')
        del model
        torch.cuda.empty_cache()
    write(root/'smoke_report.json' if a.smoke else root/'COMPLETE.json',dict(status='RAW_READOUT_SMOKE_PASSED' if a.smoke else 'RAW_READOUT_PROBED',
        records=results,new_optimizer_updates=0,model_parameters_unchanged=True,
        cohort_selection='failure-stratified diagnostic; not validation scores or generalization estimates',
        native_stage_math_unchanged=True,gt_only_used_after_context_generation=True))


if __name__=='__main__':main()
