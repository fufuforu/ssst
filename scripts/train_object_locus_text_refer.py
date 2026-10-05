#!/usr/bin/env python3
"""Independent frozen-visual, head-only trainer; requires explicit update budget."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import default_collate

from object_locus_text_refer.adapter import forward_frozen_visual, load_full1201_frozen, render_refer_membership, validate_visual_outputs
from object_locus_text_refer.data import NoVisibleReferent, choose_context_candidate, sample_context_referent
from object_locus_text_refer.head import ObjectLocusTextReferHead, build_head_optimizer, soft_gaussian_membership
from object_locus_text_refer.loss import refer_loss, resolve_slot_target
from object_locus_text_refer.text_encoder import encode_text, load_frozen_clip_text
from scripts import object_locus_v3_set_runtime as runtime
from tokengs.models.input_types import ModelInputDecoder, split_data
from tokengs.models.object_locus_v3_set_loss import final_hungarian


GLOBAL_SEED=42
HEAD_SEED=31415
MAX_CONTEXT_ATTEMPTS=128


def initialize_global_seed(seed=GLOBAL_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def draw_visible_context(provider, scene_to_index, train_refs, scene_names, rng, *, attempts=MAX_CONTEXT_ATTEMPTS, device='cuda:0'):
    skipped=[]
    for attempt in range(attempts):
        scene=choose_context_candidate(scene_names,rng)
        raw=default_collate([provider[scene_to_index[scene]]])
        batch=runtime.move_to(raw,device)
        try:
            sample=sample_context_referent(train_refs,scene,batch,rng)
        except NoVisibleReferent as exc:
            skipped.append({'attempt':attempt+1,'scene':scene,'reason':str(exc)})
            del batch,raw
            continue
        sample['skipped_before_selection']=skipped
        return sample
    raise RuntimeError(f'no described visible context target after {attempts} provider batches; skipped={skipped}')


def run_train_update(model, tokenizer, encoder, head, optimizer, sample, update):
    """Single implementation shared by the formal entry and RTX3090 smoke."""
    batch=sample['batch']; opt=model.opt
    mi,_=split_data(batch,opt)
    decoder=ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2])
    visual=forward_frozen_visual(model,mi,decoder)
    q,P=validate_visual_outputs(visual['states'][-1]['q'],visual['gaussian_membership'])
    targets,pairs=final_hungarian(visual,batch)
    gt_ids=[int(value) for value in targets['gt_instance_ids'][0].detach().cpu().tolist()]
    matched={int(ki):int(qi) for qi,ki in zip(pairs[0][0].detach().cpu().tolist(),pairs[0][1].detach().cpu().tolist())}
    slot_target=resolve_slot_target(sample['object_id'],gt_ids,matched,visible=True)
    ids,attention,text_features=encode_text(tokenizer,encoder,[sample['text']],device=q.device)
    out=head(text_features,ids,attention,q,tokenizer.eos_token_id)
    gaussian_mask=soft_gaussian_membership(out['pi'],P)
    rendered,_alpha=render_refer_membership(model.gs,gaussian_mask,
        {'gaussians':visual['gaussians'].detach(),'decoder':decoder})
    loss,parts=refer_loss(out['scores'],rendered,sample['context_target_mask'].unsqueeze(0),
        sample['context_valid_mask'].unsqueeze(0),slot_target,visible=True,matched=slot_target is not None)
    tensors=(loss,out['scores'],out['pi'],gaussian_mask,rendered)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError(f"nonfinite text refer value at update {update}")
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grad_norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True)
    grads=[p.grad for p in head.parameters() if p.grad is not None]
    if not grads or any(not torch.isfinite(grad).all() for grad in grads) or float(grad_norm)<=0:
        raise RuntimeError(f"invalid/nonzero text head gradient failure at update {update}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    log={'update':int(update),'scene':sample['scene'],'context_frame_ids':list(sample['context_frame_ids']),
         'object_id':int(sample['object_id']),'text_index':int(sample['text_index']),'text':sample['text'],
         'slot_target':slot_target,'matched':slot_target is not None,'loss':float(loss.detach()),
         'gradient_norm_before_clip':float(grad_norm.detach()),'visual_beta':[float(x['beta']) for x in visual['states']],
         'parts':{key:float(value) for key,value in parts.items()}}
    del mi,decoder,visual,q,P,targets,pairs,ids,attention,text_features,out,gaussian_mask,rendered,_alpha
    del loss,parts,tensors,grad_norm,grads
    # Return CPU scalars only; all CUDA intermediates die with this call frame.
    return log


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-updates',type=int,required=True)
    parser.add_argument('--checkpoint',default='/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')
    parser.add_argument('--data-root',default='/space/mawb/SIU3R/data/scannet')
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=1)
    args=parser.parse_args()
    if args.max_updates<1: parser.error('--max-updates must be positive')
    if args.batch_size!=1: parser.error('this version fixes batch size to 1')
    if not torch.cuda.is_available(): raise RuntimeError('text refer training requires CUDA')
    initialize_global_seed()
    root=Path(args.data_root); train_json=root/'train_refer_seg_data.json'; val_json=root/'val_refer_seg_data.json'
    train_refs=json.loads(train_json.read_text()); val_scenes=set(json.loads(val_json.read_text()))
    overlap=set(train_refs)&val_scenes
    if overlap: raise RuntimeError(f'official train/val scene leakage: {sorted(overlap)[:5]}')

    model,opt,visual_exposure=load_full1201_frozen('cuda:0',args.checkpoint)
    tokenizer,encoder,text_provenance=load_frozen_clip_text(
        cache_dir='/space/mawb/ssst/group_plus/object_locus_text_refer_v1/hf_cache',
        provenance_path='/space/mawb/ssst/group_plus/object_locus_text_refer_v1/text_encoder_provenance.json')
    encoder=encoder.cuda().eval()
    head=ObjectLocusTextReferHead(HEAD_SEED).cuda().float()
    optimizer=build_head_optimizer(head)
    provider=runtime.ObjectLocusV1Provider(opt,root=str(root/'train'),subset='all',training=True,rank=0)
    scene_to_index={path.name:index for index,path in enumerate(provider.dataset.sample_list)}
    scene_names=sorted(set(train_refs)&set(scene_to_index))
    if not scene_names: raise RuntimeError('official train refer/provider scene intersection is empty')
    rng=random.Random(GLOBAL_SEED); history=[]
    for update in range(1,args.max_updates+1):
        sample=draw_visible_context(provider,scene_to_index,train_refs,scene_names,rng)
        history.append(run_train_update(model,tokenizer,encoder,head,optimizer,sample,update))
        del sample
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'head':head.state_dict(),'completed_updates':args.max_updates,
        'visual_checkpoint':args.checkpoint,'visual_exposure':visual_exposure,
        'text_encoder':text_provenance,'optimizer':optimizer.state_dict(),
        'train_scene_count':len(scene_names),'validation_scene_count':len(val_scenes),
        'train_val_disjoint':True,'seed':GLOBAL_SEED,'head_seed':HEAD_SEED,'history':history},output)


if __name__=='__main__': main()
