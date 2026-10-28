"""Strict source loader + frozen context forward, cached labels from train only."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
import torch.distributed as dist
from scripts.frozen_iou_head_core import SOURCE_SHA,TARGET,context_iou_targets,digest,write_json,atomic_save


def export(model,opt,windows,root,rank,world,device,*,expected_windows):
    from scripts.train_dual_stream_v1 import build_batch
    root=Path(root);run=root.parent
    config=json.loads((run/'configuration.json').read_text())
    if digest(run/'training_cohort.json')!=config['training_cohort_sha256'] or digest(run/'scene_split.json')!=config['scene_split_sha256']:
        raise ValueError('Frozen head data/split identity differs')
    split=json.loads((run/'scene_split.json').read_text())
    forbidden=set(split['official_eval']);allowed=set(split['train'])|set(split['dev'])
    if len(windows)!=8337 or expected_windows!=8337 or len({w['scene'] for w in windows})!=1191:
        raise ValueError('Frozen head cache must use original 8337 train windows only')
    if {w['scene'] for w in windows}!=allowed or allowed&forbidden:
        raise ValueError('Head cache scene identity differs')
    model.requires_grad_(False);model.eval()
    versions={n:p._version for n,p in model.named_parameters()}
    features=[];targets=[];chunk_records=[];all_records=[];chunks=[];offset=0;chunk_index=0
    def flush():
        nonlocal features,targets,chunk_records,offset,chunk_index
        if not chunk_records:return
        path=root/f'rank{rank:02d}_chunk{chunk_index:03d}.pt'
        value=dict(features=torch.cat(features),targets=torch.cat(targets),records=chunk_records,
                   source_sha256=SOURCE_SHA,target_definition=TARGET,feature_key='unified_query_L12_updated',feature_dim=256)
        atomic_save(path,value)
        chunks.append(dict(file=path.name,sha256=digest(path),rows=offset,windows=len(chunk_records)))
        features=[];targets=[];chunk_records=[];offset=0;chunk_index+=1
    with torch.no_grad():
        for i in range(rank,len(windows),world):
            window=windows[i];batch=build_batch(opt,window,device)
            generated=model.generate(batch['images_input'])
            tokens=generated['unified_query'][0]
            if tokens.shape!=(generated['num_thing_queries'],256) or not torch.isfinite(tokens).all():
                raise ValueError('Missing/nonfinite final updated understanding tokens')
            y,keep=context_iou_targets(generated,batch['semantic_label_all'][0,:2],batch['instance_label_all'][0,:2])
            if not torch.isfinite(y).all() or bool(((y<0)|(y>1)).any()):raise ValueError('Invalid context IoU target')
            x=tokens[keep].float().detach().cpu();labels=y[keep].cpu()
            record=dict(window_index=i,scene=window['scene'],context=window['context'],start=offset,end=offset+len(x),
                        raw_queries=len(tokens),selected_query_ids=(keep.nonzero().flatten()+1).cpu().tolist())
            features.append(x);targets.append(labels);chunk_records.append(record)
            all_records.append(dict(window_index=i,scene=window['scene'],selected_queries=len(x)))
            offset+=len(x)
            if len(chunk_records)==100:flush()
            if len(all_records)%25==0:
                write_json(root/f'progress_rank{rank:02d}.json',dict(completed=len(all_records),total=len(windows[rank::world]),backbone_updates=0))
            del batch,generated,tokens,y,keep,x,labels
        flush()
    write_json(root/f'progress_rank{rank:02d}.json',dict(completed=len(all_records),total=len(windows[rank::world]),backbone_updates=0))
    if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.named_parameters()):
        raise ValueError('Backbone was not fully frozen during token extraction')
    write_json(root/f'BACKBONE_FROZEN_rank{rank:02d}.json',dict(status='VERIFIED',trainable_backbone_parameters=0,
               backbone_updates=0,parameter_versions_unchanged=True,all_parameter_grads_none=True,windows=len(all_records)))
    gathered=[None]*world;dist.all_gather_object(gathered,dict(records=all_records,chunks=chunks))
    if rank==0:
        records=sorted([x for g in gathered for x in g['records']],key=lambda x:x['window_index'])
        if [x['window_index'] for x in records]!=list(range(8337)):raise ValueError('Duplicate/missing training cache windows')
        write_json(root/'export_complete.json',dict(status='FROZEN_IOU_TRAIN_CACHE_COMPLETE',records=records,
                   chunks=[x for g in gathered for x in g['chunks']],windows=8337,scenes=1191,world_size=world,
                   selected_queries=sum(x['selected_queries'] for x in records),source_sha256=SOURCE_SHA,
                   target_definition=TARGET,feature_key='unified_query_L12_updated',parameter_updates=0,backbone_frozen=True))
    dist.barrier()


if __name__=='__main__':
    from scripts import export_unified_query_checkpoint as loader
    loader.export=export;loader.main()
