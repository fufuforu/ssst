#!/usr/bin/env python3
"""Real two-step context smoke; fixed to a single RTX3090 on 3dimage-11."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import default_collate

from object_locus_text_refer.adapter import load_full1201_frozen, validate_visual_outputs
from object_locus_text_refer.evaluation import evaluate_records
from object_locus_text_refer.head import ObjectLocusTextReferHead, hard_gaussian_membership, soft_gaussian_membership
from object_locus_text_refer.loss import refer_loss, resolve_slot_target
from object_locus_text_refer.text_encoder import encode_text, load_frozen_clip_text
from scripts import object_locus_v3_set_runtime as data_runtime
from tokengs.models.input_types import ModelInputDecoder, split_data
from tokengs.models.object_locus_v3_set_loss import final_hungarian


OUT_DIR=Path('/space/mawb/ssst/group_plus/object_locus_text_refer_v1')
VAL_ROOT=Path('/space/mawb/SIU3R/data/scannet/val')
REFER=Path('/space/mawb/SIU3R/data/scannet/val_refer_seg_data.json')
PAIRS=Path('/space/mawb/SIU3R/data/scannet/val_refer_pair.json')


def digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_real_batch(opt, pair):
    scene=pair['scene_name']; context=[int(x) for x in pair['context_views_id']]
    frame_dir=VAL_ROOT/scene/'depth'
    available=sorted(int(p.stem) for p in frame_dir.glob('*.png'))
    novel=[x for x in available if x not in context][:2]
    if len(novel)!=2: raise RuntimeError(f'not enough extra provider views for {scene}')
    provider=data_runtime.ObjectLocusV1Provider(opt,root=str(VAL_ROOT),subset=[scene],training=True,rank=0)
    provider.pin_pair(scene_id=scene,context_frame_ids=context,novel_frame_ids=novel)
    batch=data_runtime.move_to(default_collate([provider[0]]),'cuda:0')
    if batch['frame_ids'][0].detach().cpu().tolist()!=context+novel:
        raise RuntimeError('provider changed the requested official context frame order')
    return batch,novel


def main():
    if not torch.cuda.is_available() or not os.uname().nodename.startswith('3dimage-11'):
        raise RuntimeError('real smoke is fixed to one RTX3090 on 3dimage-11')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 3090':
        raise RuntimeError('Slurm job must expose exactly one RTX3090')
    torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    refer=json.loads(REFER.read_text()); pairs=json.loads(PAIRS.read_text())
    model,opt=load_full1201_frozen('cuda:0')
    model.eval()
    visual_sha=digest(model)
    tokenizer,encoder,text_prov=load_frozen_clip_text(cache_dir=str(OUT_DIR/'hf_cache'),provenance_path=str(OUT_DIR/'text_encoder_provenance.json'))
    encoder=encoder.to('cuda:0').eval()
    text_sha=digest(encoder)
    head=ObjectLocusTextReferHead().cuda().float()
    wd=[]; no_wd=[]
    for n,p in head.named_parameters():
        (wd if p.ndim>1 and not n.endswith('.bias') else no_wd).append(p)
    optimizer=torch.optim.AdamW([{'params':wd,'weight_decay':.05},{'params':no_wd,'weight_decay':0.0}],lr=1e-4,betas=(.9,.95),eps=1e-8)
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in head.parameters()}:
        raise RuntimeError('optimizer does not contain exactly the new head')
    skip=[];selected=None
    for pair_i,pair in enumerate(pairs):
        scene=pair['scene_name']; oid=int(pair['context_objects']); text=pair['texts']
        if not isinstance(text,str) or not text.strip(): skip.append({'pair_index':pair_i,'reason':'empty text'});continue
        try: batch,extra_views=build_real_batch(opt,pair)
        except Exception as exc: skip.append({'pair_index':pair_i,'reason':f'provider: {exc}'});continue
        ins=batch['instance_label_all'][:,:2].long();sem=batch['semantic_label_all'][:,:2].long()
        valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0))
        target=valid & (ins==oid)
        if not target.any(): skip.append({'pair_index':pair_i,'reason':'zero target pixels in context valid domain'});continue
        selected=(pair_i,pair,batch,extra_views,target,valid);break
    if selected is None: raise RuntimeError(f'no visible official val pair; skipped={skip}')
    pair_i,pair,batch,extra_views,gt,valid=selected
    mi,_=split_data(batch,opt)
    context_decoder=ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2])
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        visual=model.forward_object_locus(mi,render_decoder_input=context_decoder,read_context_decoder=context_decoder,context_decoder=context_decoder,step=0)
        q,P=validate_visual_outputs(visual['states'][-1]['q'],visual['gaussian_membership'])
        targets,pairs_h=final_hungarian(visual,batch)
    object_ids=targets['gt_instance_ids'][0].detach().cpu().tolist()
    visible=bool(gt.any())
    if visible:
        matched_slots={int(ki):int(qi) for qi,ki in zip(*[x.tolist() for x in pairs_h[0]])}
        slot_target=resolve_slot_target(int(pair['context_objects']),object_ids,matched_slots,True)
    else:
        frames=refer[pair['scene_name']]['frame2object']
        listed=any(str(pair['context_objects']) in frames.get(str(frame),[]) for frame in pair['context_views_id'])
        slot_target=None if listed else 100
    ids,attn,T=encode_text(tokenizer,encoder,[pair['texts']],'cuda:0')
    P_render=visual['gaussian_membership'][:,:,:100].detach()
    gaussians=visual['gaussians'].detach()
    ctx=ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2])
    logs=[]; head_initial=[p.detach().clone() for p in head.parameters()]
    frozen_before=(visual_sha,text_sha)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        out=head(T,ids,attn,q,tokenizer.eos_token_id)
        M=soft_gaussian_membership(out['pi'],P_render)
        rendered=model.gs.render_feature_channels(gaussians,M.unsqueeze(-1),ctx.cam_view,ctx.intrinsics)
        probability=model.normalize_membership(rendered['images_pred'],rendered['alphas_pred']) if hasattr(model,'normalize_membership') else None
        if probability is None:
            from tokengs.models.object_locus_v3_set import alpha_normalize_membership
            probability=alpha_normalize_membership(rendered['images_pred'],rendered['alphas_pred'])[:,:,0]
        loss,parts=refer_loss(out['scores'],probability,gt,valid,slot_target=slot_target,visible=visible,matched=slot_target is not None)
        values=(loss,out['scores'],out['pi'],M,probability)
        if not all(torch.isfinite(v).all() for v in values): raise FloatingPointError('nonfinite smoke output')
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True)
        grads=[p.grad for p in head.parameters() if p.grad is not None]
        grad_norm=sum(float(g.norm()) for g in grads)
        if not grads or grad_norm<=0 or any(not torch.isfinite(g).all() for g in grads): raise RuntimeError('new head gradient invalid')
        optimizer.step()
        logs.append({'update':update+1,'loss':float(loss.detach()),'grad_norm_sum':grad_norm,'parts':{k:float(v) for k,v in parts.items()},'slot_target':slot_target})
    # Hard inference selects one 3D slot once for both context cameras.
    with torch.no_grad():
        pred=head(T,ids,attn,q,tokenizer.eos_token_id)
        hard,slot=hard_gaussian_membership(pred['scores'],P_render)
        hard_render=model.gs.render_feature_channels(gaussians,hard.unsqueeze(-1),ctx.cam_view,ctx.intrinsics)
        from tokengs.models.object_locus_v3_set import alpha_normalize_membership
        hard_masks=alpha_normalize_membership(hard_render['images_pred'],hard_render['alphas_pred'])[:,:,0]
        # Extra explicit camera API check, using one of this same sample's cameras.
        one_cam=ModelInputDecoder(cam_view=batch['cam_view_all'][:,2:3],intrinsics=batch['intrinsics_all'][:,2:3])
        novel_render=model.gs.render_feature_channels(gaussians,hard.unsqueeze(-1),one_cam.cam_view,one_cam.intrinsics)
        novel_mask=alpha_normalize_membership(novel_render['images_pred'],novel_render['alphas_pred'])[:,:,0]
        eval_rows=[]
        for view in range(2):
            eval_rows.append({'text_key':f"{pair['scene_name']}/{pair['context_objects']}/0",
                'scene':pair['scene_name'],'object_id':int(pair['context_objects']),'text':pair['texts'],
                'view_id':int(pair['context_views_id'][view]),'pred_mask':hard_masks[0,view]>.5,
                'gt_mask':gt[0,view],'valid_mask':valid[0,view],'selected_slot':int(slot[0])})
        eval_summary=evaluate_records(eval_rows)
    if digest(model)!=frozen_before[0] or digest(encoder)!=frozen_before[1]: raise RuntimeError('frozen state changed')
    if any(p.grad is not None for p in model.parameters()) or any(p.grad is not None for p in encoder.parameters()): raise RuntimeError('frozen model/text encoder received gradients')
    if not any(not torch.equal(p.detach(),initial) for p,initial in zip(head.parameters(),head_initial)): raise RuntimeError('head parameters did not update')
    torch.cuda.synchronize()
    report={'status':'PASS','node':os.uname().nodename,'gpu':torch.cuda.get_device_name(0),'pair_index':pair_i,
      'scene':pair['scene_name'],'object_id':int(pair['context_objects']),'text':pair['texts'],'context_frame_ids':pair['context_views_id'],
      'extra_camera_frame_id':extra_views[0],'skipped_pairs':skip,'steps':logs,'selected_slot':int(slot[0]),
      'context_mask_shapes':list(hard_masks.shape),'explicit_extra_view_mask_shape':list(novel_mask.shape),
      'context_evaluation':eval_summary,
      'visual_and_text_frozen':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
      'text_revision':text_prov['revision'],'protocol_note':'extra view is an interface smoke, not an official novel benchmark'}
    out_path=OUT_DIR/'rtx3090_smoke.json';out_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
