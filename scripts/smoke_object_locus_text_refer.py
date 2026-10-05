#!/usr/bin/env python3
"""Fixed two-update train-entry smoke plus one-record actual eval-entry smoke."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import default_collate

from object_locus_text_refer.adapter import assert_visual_beta, forward_frozen_visual, load_full1201_frozen
from object_locus_text_refer.data import NoVisibleReferent, sample_context_referent
from object_locus_text_refer.head import build_head_optimizer
from object_locus_text_refer.text_encoder import load_frozen_clip_text
from scripts import object_locus_v3_set_runtime as runtime
from scripts.train_object_locus_text_refer import (
    GLOBAL_SEED, HEAD_SEED, initialize_global_seed, run_train_update,
)
from scripts.eval_object_locus_text_refer import run_evaluation
from tokengs.models.input_types import ModelInputDecoder, split_data


ROOT=Path('/space/mawb/SIU3R/data/scannet')
OUTPUT=Path('/space/mawb/ssst/group_plus/object_locus_text_refer_v1')
VISUAL_CHECKPOINT=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt')


def digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def scene_names_and_provider(opt,train_refs):
    provider=runtime.ObjectLocusV1Provider(opt,root=str(ROOT/'train'),subset='all',training=True,rank=0)
    provider_index={path.name:index for index,path in enumerate(provider.dataset.sample_list)}
    names=sorted(set(train_refs)&set(provider_index))
    if not names: raise RuntimeError('official train/provider scene intersection is empty')
    return provider,provider_index,names


def select_first_two(provider,provider_index,scene_names,train_refs,rng):
    samples=[]; skipped=[]
    for scene in scene_names:
        provider.pinned_pair=None
        raw=default_collate([provider[provider_index[scene]]])
        try:
            selected=sample_context_referent(train_refs,scene,raw,rng)
        except NoVisibleReferent as exc:
            skipped.append({'scene':scene,'reason':str(exc)}); del raw; continue
        metadata={k:selected[k] for k in ('scene','context_frame_ids','object_id','text','text_index','candidate_object_ids')}
        selected['context_frame_ids']=metadata['context_frame_ids']
        # Pixel masks are recalculated from the pinned provider batch at update time.
        del selected,raw
        samples.append({**metadata,'skipped_before_selection':list(skipped)})
        if len(samples)==2: break
    if len(samples)!=2: raise RuntimeError(f'found only {len(samples)} fixed smoke samples; skipped={skipped}')
    return samples,skipped


def reload_pinned_sample(provider,provider_index,metadata):
    scene=metadata['scene']; context=[int(x) for x in metadata['context_frame_ids']]
    frame_dir=ROOT/'train'/scene/'depth'
    available=sorted(int(p.stem) for p in frame_dir.glob('*.png'))
    novel=[frame for frame in available if frame not in set(context)][:2]
    if len(novel)!=2: raise RuntimeError(f'{scene} has fewer than two extra provider frames')
    provider.pin_pair(scene_id=scene,context_frame_ids=context,novel_frame_ids=novel)
    raw=default_collate([provider[provider_index[scene]]])
    batch=runtime.move_to(raw,'cuda:0')
    sem=batch['semantic_label_all'][0,:2].long(); ins=batch['instance_label_all'][0,:2].long()
    valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0))
    gt=valid&(ins==int(metadata['object_id']))
    if not gt.any(): raise RuntimeError('fixed smoke target disappeared from pinned context batch')
    return {**metadata,'batch':batch,'context_target_mask':gt,'context_valid_mask':valid,'extra_frame_ids':novel}


def main():
    if not torch.cuda.is_available() or not os.uname().nodename.startswith('3dimage-11'):
        raise RuntimeError('smoke requires one Slurm RTX3090 on 3dimage-11')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 3090':
        raise RuntimeError('Slurm must expose exactly one RTX3090')
    torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    initialize_global_seed(GLOBAL_SEED)
    train_refs=json.loads((ROOT/'train_refer_seg_data.json').read_text())
    model,opt,visual_exposure=load_full1201_frozen('cuda:0',VISUAL_CHECKPOINT)
    tokenizer,encoder,text_provenance=load_frozen_clip_text(
        cache_dir=str(OUTPUT/'hf_cache'),provenance_path=str(OUTPUT/'text_encoder_provenance.json'))
    encoder=encoder.cuda().eval()
    from object_locus_text_refer.head import ObjectLocusTextReferHead
    head=ObjectLocusTextReferHead(HEAD_SEED).cuda().float()
    optimizer=build_head_optimizer(head)
    visual_before=digest(model); text_before=digest(encoder); head_before=[p.detach().clone() for p in head.parameters()]
    provider,provider_index,scene_names=scene_names_and_provider(opt,train_refs)
    samples,scan_skips=select_first_two(provider,provider_index,scene_names,train_refs,random.Random(GLOBAL_SEED))

    # Compare direct source-model forward with the exposure adapter on identical input.
    first=reload_pinned_sample(provider,provider_index,samples[0])
    mi,_=split_data(first['batch'],opt)
    decoder=ModelInputDecoder(cam_view=first['batch']['cam_view_all'][:,:2],intrinsics=first['batch']['intrinsics_all'][:,:2])
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        direct=model.forward_object_locus(mi,render_decoder_input=decoder,read_context_decoder=decoder,
            context_decoder=decoder,step=visual_exposure)
    adapted=forward_frozen_visual(model,mi,decoder,visual_exposure)
    direct_states={int(state['layer']):state for state in direct['states'] if int(state['layer']) in (6,8,10,12)}
    adapted_states={int(state['layer']):state for state in adapted['states'] if int(state['layer']) in (6,8,10,12)}
    if set(direct_states)!={6,8,10,12} or set(adapted_states)!={6,8,10,12}:
        raise RuntimeError('Full1201 output lacks registered L6/L8/L10/L12 states')
    for layer in (6,8,10,12):
        layer_a,layer_b=direct_states[layer],adapted_states[layer]
        if abs(float(layer_a['beta'])-.1)>1e-8 or abs(float(layer_b['beta'])-.1)>1e-8: raise RuntimeError('visual beta is not 0.1')
        torch.testing.assert_close(layer_a['q'],layer_b['q'],rtol=1e-6,atol=1e-5)
    for key in ('gaussians','gaussian_membership','region_mass','semantic_scores'):
        torch.testing.assert_close(direct[key],adapted[key],rtol=1e-6,atol=1e-5)
    assert_visual_beta(adapted['states'])
    del direct,adapted,mi,decoder,first

    logs=[]; selected=[]
    for update,metadata in enumerate(samples,1):
        sample=reload_pinned_sample(provider,provider_index,metadata)
        selected.append({k:metadata[k] for k in ('scene','context_frame_ids','object_id','text','text_index','candidate_object_ids')}
            | {'extra_frame_ids':sample['extra_frame_ids'],'skipped_before_selection':metadata['skipped_before_selection']})
        log=run_train_update(model,tokenizer,encoder,head,optimizer,sample,update)
        logs.append(log)
        del sample,log
        gc.collect()
    if digest(model)!=visual_before or digest(encoder)!=text_before:
        raise RuntimeError('frozen visual/text parameters or buffers changed')
    if any(p.grad is not None for p in model.parameters()) or any(p.grad is not None for p in encoder.parameters()):
        raise RuntimeError('frozen visual/text model received gradients')
    if not any(not torch.equal(p.detach(),start) for p,start in zip(head.parameters(),head_before)):
        raise RuntimeError('text head parameters did not change')
    if any(any(abs(beta-.1)>1e-8 for beta in row['visual_beta']) for row in logs):
        raise RuntimeError('train entry visual beta mismatch')
    torch.cuda.synchronize()
    temp_head=OUTPUT/'temporary_two_update_head.pt'
    torch.save({'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
        'completed_updates':2,'visual_exposure':visual_exposure,'text_revision':text_provenance['revision']},temp_head)
    train_report={'status':'PASS','job_node':os.uname().nodename,'gpu':torch.cuda.get_device_name(0),
        'visual_exposure':visual_exposure,'visual_beta':[row['visual_beta'][:4] for row in logs],
        'selected_samples':selected,'scene_scan_skips':scan_skips,'updates':logs,
        'head_updated':True,'visual_state_unchanged':True,'text_state_unchanged':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'temporary_head_checkpoint':str(temp_head)}
    train_report_path=OUTPUT/'rtx3090_train_smoke.json'
    train_report_path.write_text(json.dumps(train_report,indent=2,ensure_ascii=False)+'\n')

    # Drop every parent-process GPU reference before actual eval CLI launches its own model.
    del provider,provider_index,scene_names,samples,head,optimizer,encoder,model,train_refs
    gc.collect(); torch.cuda.synchronize()
    eval_dir=OUTPUT/'rtx3090_eval_smoke'
    eval_args=argparse.Namespace(visual_checkpoint=str(VISUAL_CHECKPOINT),head_checkpoint=str(temp_head),
        data_root=str(ROOT),refer_json=None,pair_json=None,output_dir=str(eval_dir),max_records=1)
    run_evaluation(eval_args)
    metrics=json.loads((eval_dir/'context_refer_metrics.json').read_text())
    records=json.loads((eval_dir/'context_refer_records.json').read_text())
    if metrics['context_refer_expression_count']!=1 or len(records)!=1:
        raise RuntimeError('actual eval CLI did not process exactly the requested first record')
    eval_report={'status':'PASS','records':len(records),'metrics':{k:v for k,v in metrics.items() if k!='context_refer_records'},
        'first_record':records[0]}
    (OUTPUT/'rtx3090_train_smoke.json').write_text(json.dumps(train_report,indent=2,ensure_ascii=False)+'\n')
    (OUTPUT/'rtx3090_eval_smoke.json').write_text(json.dumps(eval_report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'train':train_report,'eval':eval_report},indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
