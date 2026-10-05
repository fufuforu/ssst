#!/usr/bin/env python3
"""Generate and score deterministic SIU3R context text refer predictions."""
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

from object_locus_text_refer.adapter import forward_frozen_visual, load_full1201_frozen, validate_visual_outputs
from object_locus_text_refer.evaluation import aggregate_expressions, binarize_probability, masked_iou
from object_locus_text_refer.head import ObjectLocusTextReferHead, hard_gaussian_membership
from object_locus_text_refer.text_encoder import encode_text, load_frozen_clip_text
from scripts import object_locus_v3_set_runtime as runtime
from tokengs.models.input_types import ModelInputDecoder, split_data
from tokengs.models.object_locus_v3_set import alpha_normalize_membership


def set_global_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_head(path, device):
    blob=torch.load(path,map_location='cpu',weights_only=False)
    state=blob.get('head',blob.get('state_dict',blob)) if isinstance(blob,dict) else blob
    head=ObjectLocusTextReferHead(seed=31415)
    head.load_state_dict(state,strict=True)
    return head.to(device).float().eval()


def available_extra_frames(root, scene, context):
    files=sorted((Path(root)/scene/'depth').glob('*.png'),key=lambda p:int(p.stem))
    extras=[int(path.stem) for path in files if int(path.stem) not in set(map(int,context))]
    if len(extras)<2: raise RuntimeError(f'{scene} has fewer than two non-context provider frames')
    return extras[:2]


def run_expression(model,opt,head,tokenizer,encoder,refer,pair,text,root,device):
    scene=str(pair['scene_name']); context=[int(value) for value in pair['context_views_id']]
    object_id=int(pair['context_objects'])
    extras=available_extra_frames(root/'val',scene,context)
    provider=runtime.ObjectLocusV1Provider(opt,root=str(root/'val'),subset=[scene],training=True,rank=0)
    provider.pin_pair(scene_id=scene,context_frame_ids=context,novel_frame_ids=extras)
    batch=runtime.move_to(default_collate([provider[0]]),device)
    if [int(v) for v in batch['frame_ids'][0,:2].detach().cpu().tolist()]!=context:
        raise RuntimeError('provider changed official context frame IDs/order')
    mi,_=split_data(batch,opt)
    decoder=ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2])
    visual=forward_frozen_visual(model,mi,decoder)
    q,P=validate_visual_outputs(visual['states'][-1]['q'],visual['gaussian_membership'])
    ids,attention,text_features=encode_text(tokenizer,encoder,[text],device)
    scores=head(text_features,ids,attention,q,tokenizer.eos_token_id)['scores']
    hard,selected=hard_gaussian_membership(scores,P)
    rendered=model.gs.render_feature_channels(visual['gaussians'].detach(),hard.unsqueeze(-1),decoder.cam_view,decoder.intrinsics)
    probability=alpha_normalize_membership(rendered['images_pred'],rendered['alphas_pred'])[:,:,0]
    predictions=binarize_probability(probability)
    sem=batch['semantic_label_all'][0,:2].long()
    ins=batch['instance_label_all'][0,:2].long()
    valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0))
    target=valid & (ins==object_id)
    view_ious=[]; view_rows=[]; reasons=[]
    for view in range(2):
        iou,reason=masked_iou(predictions[0,view],target[view],valid[view])
        view_ious.append(float(iou))
        if reason: reasons.append(f'view {context[view]}: {reason}')
        view_rows.append({'context_frame_id':context[view],'iou':float(iou)})
    failure='; '.join(reasons) if reasons else None
    record={'scene':scene,'object_id':object_id,'text':text,'context_frame_ids':context,
            'selected_slot':int(selected[0]),'view_results':view_rows,'view_ious':view_ious,
            'expression_iou':sum(view_ious)/2,'failure_reason':failure,
            'provider_extra_frames':extras}
    if failure: record['expression_iou']=0.0
    return record


def run_evaluation(args):
    if args.max_records is not None and args.max_records<1:
        raise ValueError('--max-records must be positive')
    if not torch.cuda.is_available(): raise RuntimeError('context text refer evaluation requires CUDA')
    set_global_seed(42); device='cuda:0'; root=Path(args.data_root)
    refer_path=Path(args.refer_json) if args.refer_json else root/'val_refer_seg_data.json'
    pair_path=Path(args.pair_json) if args.pair_json else root/'val_refer_pair.json'
    refer=json.loads(refer_path.read_text()); pairs=json.loads(pair_path.read_text())
    model,opt,visual_exposure=load_full1201_frozen(device,args.visual_checkpoint); model.eval()
    head=load_head(args.head_checkpoint,device)
    tokenizer,encoder,text_provenance=load_frozen_clip_text(
        cache_dir='/space/mawb/ssst/group_plus/object_locus_text_refer_v1/hf_cache',
        provenance_path='/space/mawb/ssst/group_plus/object_locus_text_refer_v1/text_encoder_provenance.json')
    encoder=encoder.to(device).eval()
    records=[]
    for pair_index,pair in enumerate(pairs):
        if args.max_records is not None and len(records)>=args.max_records: break
        raw_texts=pair.get('texts','')
        texts=raw_texts if isinstance(raw_texts,list) else [raw_texts]
        # Pair-level object and frames are normalized once. Descriptions stay in source order.
        pair={**pair,'context_views_id':[int(x) for x in pair['context_views_id']],
              'context_objects':int(pair['context_objects'])}
        frame_objects={int(x) for frame in pair['context_views_id']
                       for x in refer[pair['scene_name']]['frame2object'].get(str(int(frame)),[])}
        # Keep normalized frame2object IDs for the protocol record; GT never affects slot selection.
        pair['frame2object_context_ids']=sorted(frame_objects)
        for text_index,text in enumerate(texts):
            if args.max_records is not None and len(records)>=args.max_records: break
            base={'pair_index':pair_index,'text_index':text_index,'scene':pair['scene_name'],
                  'object_id':int(pair['context_objects']),'text':text,
                  'context_frame_ids':pair['context_views_id'],'selected_slot':None,
                  'view_results':[{'context_frame_id':int(f),'iou':0.0} for f in pair['context_views_id']],
                  'view_ious':[0.0,0.0],'expression_iou':0.0,'failure_reason':None}
            try:
                if not isinstance(text,str) or not text.strip(): raise ValueError('empty raw expression')
                result=run_expression(model,opt,head,tokenizer,encoder,refer,pair,text,root,device)
                result.update(pair_index=pair_index,text_index=text_index,frame2object_context_ids=sorted(frame_objects))
                records.append(result)
            except Exception as exc:
                base['failure_reason']=f'{type(exc).__name__}: {exc}'
                records.append(base)
    metrics=aggregate_expressions(records)
    metrics.update({'visual_checkpoint':args.visual_checkpoint,'visual_exposure':visual_exposure,
      'head_checkpoint':args.head_checkpoint,'text_model':'openai/clip-vit-base-patch32',
      'text_revision':text_provenance['revision'],'refer_json':str(refer_path),'pair_json':str(pair_path),
      'processed_pairs':len({r['pair_index'] for r in records}),'max_records':args.max_records,
      'scope':'context only','novel_protocol_status':'not aligned; no official novel target-view list'})
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    (output/'context_refer_records.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    (output/'context_refer_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k!='context_refer_records'},indent=2),flush=True)
    del records,metrics,encoder,head,model


def build_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--visual-checkpoint',default='/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')
    parser.add_argument('--head-checkpoint',required=True)
    parser.add_argument('--data-root',default='/space/mawb/SIU3R/data/scannet')
    parser.add_argument('--refer-json')
    parser.add_argument('--pair-json')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-records',type=int,default=None)
    return parser


def main():
    args=build_parser().parse_args()
    run_evaluation(args)


if __name__=='__main__': main()
