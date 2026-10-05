#!/usr/bin/env python3
"""Independent frozen-visual, head-only text refer trainer; not run by smoke."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import default_collate

from object_locus_text_refer.adapter import load_full1201_frozen, render_refer_membership, validate_visual_outputs
from object_locus_text_refer.data import SIU3RReferDataset
from object_locus_text_refer.head import ObjectLocusTextReferHead, soft_gaussian_membership
from object_locus_text_refer.loss import refer_loss, resolve_slot_target
from object_locus_text_refer.text_encoder import encode_text, load_frozen_clip_text
from scripts import object_locus_v3_set_runtime as runtime
from tokengs.models.input_types import ModelInputDecoder, split_data
from tokengs.models.object_locus_v3_set_loss import final_hungarian


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--max-updates',type=int,required=True)
    p.add_argument('--checkpoint',default='/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')
    p.add_argument('--data-root',default='/space/mawb/SIU3R/data/scannet')
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=.05)
    p.add_argument('--seed',type=int,default=31415)
    a=p.parse_args()
    if a.max_updates<1: p.error('--max-updates must be positive')
    if a.batch_size!=1: p.error('this first version fixes batch size to 1')
    if not torch.cuda.is_available(): raise RuntimeError('head training requires CUDA; this entry never substitutes a random visual/text encoder')
    root=Path(a.data_root); train_json=root/'train_refer_seg_data.json'; val_json=root/'val_refer_seg_data.json'
    train_refs=json.loads(train_json.read_text()); val_refs=json.loads(val_json.read_text())
    overlap=set(train_refs)&set(val_refs)
    if overlap: raise RuntimeError(f'official train/val scene leakage: {sorted(overlap)[:5]}')

    model,opt=load_full1201_frozen('cuda:0',a.checkpoint); model.eval()
    tokenizer,encoder,text_prov=load_frozen_clip_text(cache_dir='/space/mawb/ssst/group_plus/object_locus_text_refer_v1/hf_cache',
      provenance_path='/space/mawb/ssst/group_plus/object_locus_text_refer_v1/text_encoder_provenance.json')
    encoder=encoder.cuda().eval()
    head=ObjectLocusTextReferHead(a.seed).cuda().float()
    decay=[];nodecay=[]
    for name,param in head.named_parameters(): (decay if param.ndim>1 and not name.endswith('.bias') else nodecay).append(param)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':a.weight_decay},{'params':nodecay,'weight_decay':0.0}],
                                lr=a.lr,betas=(.9,.95),eps=1e-8)
    provider=runtime.ObjectLocusV1Provider(opt,root=str(root/'train'),subset='all',training=True,rank=0)
    scene_to_index={p.name:i for i,p in enumerate(provider.dataset.sample_list)}
    batch_cache={}
    def load_train_arrays(scene, frame_ids):
        del frame_ids
        if scene not in scene_to_index: raise KeyError(f'train scene missing from provider: {scene}')
        batch=runtime.move_to(default_collate([provider[scene_to_index[scene]]]),'cuda:0')
        batch_cache[scene]=batch
        sem=batch['semantic_label_all'][0,:2].detach().cpu().long()
        ins=batch['instance_label_all'][0,:2].detach().cpu().long()
        packed=sem*1000+ins
        valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0))
        return {'packed_panoptic':packed.numpy(),'valid_mask':valid.numpy(),
                'context_frame_ids':batch['frame_ids'][0,:2].detach().cpu().tolist()}
    dataset=SIU3RReferDataset(train_json,split='train',load_arrays=load_train_arrays,seed=a.seed)
    if not len(dataset): raise RuntimeError('SIU3R train refer data has no text records')
    order_rng=random.Random(a.seed); losses=[]
    from tokengs.models.object_locus_v3_set import alpha_normalize_membership
    for update in range(a.max_updates):
        index=order_rng.randrange(len(dataset)); item=dataset[index]; batch=batch_cache[item['scene']]
        if tuple(batch['images_all'].shape[:2])!=(1,4): raise RuntimeError('provider batch must contain 2 context plus 2 other views')
        mi,_=split_data(batch,opt)
        decoder=ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2])
        with torch.no_grad():
            visual=model.forward_object_locus(mi,render_decoder_input=decoder,read_context_decoder=decoder,context_decoder=decoder,step=0)
            q,P=validate_visual_outputs(visual['states'][-1]['q'],visual['gaussian_membership'])
            target_rows,pairs=final_hungarian(visual,batch)
        object_ids=target_rows['gt_instance_ids'][0].detach().cpu().tolist()
        matched_slots={int(ki):int(qi) for qi,ki in zip(*[v.tolist() for v in pairs[0]])}
        scene_refs=train_refs[item['scene']]
        frames=item['context_frame_ids']
        listed=any(str(item['object_id']) in scene_refs['frame2object'].get(str(f),[]) for f in frames)
        gt=torch.as_tensor(item['context_target_mask'],device='cuda:0').unsqueeze(0)
        valid=torch.as_tensor(item['context_valid_mask'],device='cuda:0').unsqueeze(0)
        visible=bool(gt.any())
        if visible:
            slot_target=resolve_slot_target(item['object_id'],object_ids,matched_slots,True)
        elif listed:
            # Inconsistent/missing pixel annotation: no slot/null or pixel negative.
            slot_target=None; valid=torch.zeros_like(valid)
        else:
            slot_target=100; visible=False
        ids,attention,T=encode_text(tokenizer,encoder,[item['text']],'cuda:0')
        out=head(T,ids,attention,q,tokenizer.eos_token_id)
        M=soft_gaussian_membership(out['pi'],P)
        rendered,alpha=render_refer_membership(model.gs,M,{'gaussians':visual['gaussians'].detach(),'decoder':decoder})
        loss,parts=refer_loss(out['scores'],rendered,gt,valid,slot_target,
                              visible=visible,matched=slot_target is not None)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite text refer loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
        losses.append({'update':update+1,'scene':item['scene'],'object_id':item['object_id'],'text_index':item['text_index'],
                       'loss':float(loss.detach()),'slot_target':slot_target,'visible':visible,
                       'parts':{k:float(v) for k,v in parts.items()}})
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'head':head.state_dict(),'completed_updates':a.max_updates,'base_checkpoint':a.checkpoint,
      'text_encoder':text_prov,'optimizer':optimizer.state_dict(),'train_scenes':len(train_refs),
      'validation_scenes':len(val_refs),'official_split_disjoint':True,'config':vars(a),'history':losses},output)


if __name__=='__main__': main()
