"""Aggregate persisted fixed evaluations; never instantiate a trained model."""
import argparse, collections, csv, json, math, shutil, subprocess
from pathlib import Path
import numpy as np
import torch
from scripts.eval_object_locus_panoptic_full1201 import EVAL, EPOCHS, SCOPES, csvout, read, write, full_windows
from scripts import object_locus_panoptic_full1201_runtime as rt

def metric(v):
    if v is None:return 'MISSING'
    if isinstance(v,(float,int)) and (not math.isfinite(v) or v<0):return 'UNDEFINED'
    return v

def rows_for(epoch,split,result,cohort='fixed'):
    rows=[]
    for scope in SCOPES:
        l=result['local'][scope];arm,view=('novel','target') if scope=='novel' else ('all','target' if scope=='target_all' else 'context')
        off=result.get('official',{}).get(arm,{}) or {};ap=off.get(view+'_map',{}) or {};c=np.asarray(l['classification_confusion']);den=int(c.sum());no=int(c[:,18].sum())
        row=dict(epoch=epoch,split=split,cohort=cohort,scope=scope,windows=l['windows'],gt_count=l['gt_count'],
                 semantic_miou=metric(l.get('semantic_miou')),thing_miou=metric(l.get('mIoU_thing')),stuff_miou=metric(l.get('mIoU_stuff')),
                 panoptic_miou=metric(l.get('panoptic_semantic_miou')),panoptic_pq=metric(l.get('panoptic_pq')),
                 candidate_map=metric(l.get('candidate_ap',{}).get('map')),candidate_ap50=metric(l.get('candidate_ap',{}).get('map_50')),
                 official_miou=metric(off.get(view+'_miou')),official_pq=metric(off.get(view+'_pq')),official_map=metric(ap.get('map')),official_ap50=metric(ap.get('map_50')),
                 psnr=metric(l.get('psnr')),ssim=metric(l.get('ssim')),lpips=metric(l.get('lpips')),
                 raw_mask_coverage=l['raw_best_iou_ge_0_5_fraction'] if l['gt_count'] else 'UNDEFINED',
                 matched_class_accuracy=l['matched_19_class_accuracy'] if den else 'UNDEFINED',matched_no_object_count=no,matched_denominator=den,
                 matched_no_object_fraction=no/den if den else 'UNDEFINED',matching_source='context final_hungarian; NOT novel-only classification')
        for b in ('candidate_ca','candidate_cw','panoptic_ca','panoptic_cw'):
            d=l[b]
            for key in ('tp','fp','fn'):row[b+'_'+key]=d[key]
            row[b+'_precision']=d['tp']/(d['tp']+d['fp']) if d['tp']+d['fp'] else 'UNDEFINED'
            row[b+'_recall']=d['tp']/(d['tp']+d['fn']) if d['tp']+d['fn'] else 'UNDEFINED'
        rows.append(row)
    return rows

def monitor():
    manifest=read(rt.REPORT/'manifest.json');rows=[];inventory=[]
    for epoch in EPOCHS:
        for split,windows in manifest['monitor_splits'].items():
            root=EVAL/f'epoch{epoch:02}'/split;done=read(root/'complete.json');assert done['status']=='COMPLETE' and done['windows']==len(windows)
            for name,h in done['files'].items():assert rt.sha(root/name)==h
            result=read(root/'result.json')
            assert all(all(k in result['local'][s]['candidate_ap'] for k in ('map','map_50')) for s in SCOPES),'missing candidate AP execution result'
            assert all(all(k in result['official'][arm] for k in ('context_miou','context_pq','context_map','target_miou','target_pq','target_map')) for arm in ('all','novel')),'missing official result'
            rows.extend(rows_for(epoch,split,result));inventory.append(dict(epoch=epoch,split=split,source=done['source'],result_sha256=done['files']['result.json']))
    write(rt.REPORT/'metrics_all_nodes.json',rows);csvout(rt.REPORT/'metrics_all_nodes.csv',rows);write(rt.REPORT/'evaluation_sources.json',inventory)
    candidates=[r for r in rows if r['split']=='dev8' and r['scope']=='novel'];assert len(candidates)==6 and all(isinstance(r['official_ap50'],(float,int)) for r in candidates)
    chosen=min(candidates,key=lambda r:(-r['official_ap50'],r['epoch']))
    write(rt.REPORT/'checkpoint_selection.json',dict(best_epoch=chosen['epoch'],value=chosen['official_ap50'],metric='dev8 true-novel official packed AP50',
        exact_tie='earlier epoch',candidates=[dict(epoch=r['epoch'],ap50=r['official_ap50']) for r in candidates],
        dev8_used_for_selection=True,dev8_subset_of_val32=True,val32_independent_of_selection=False,endpoint_epoch=8,
        full_evaluation_epochs=sorted({chosen['epoch'],8})))
    print('SELECTED',chosen['epoch'],chosen['official_ap50'],flush=True)

def aggregate_local(windows,states,keep):
    from scripts.eval_object_locus_panoptic_v1 import local_ap_metric
    result={}
    for scope in SCOPES:
        rr=[w['scopes'][scope] for i,w in enumerate(windows) if i in keep];conf=np.sum([r['semantic_confusion'] for r in rr],axis=0);pc=np.sum([r['panoptic_semantic_confusion'] for r in rr],axis=0)
        def miou(mat,classes):
            vals=[]
            for c in classes:
                den=mat[c,:].sum()+mat[:,c].sum()-mat[c,c]
                if den:vals.append(mat[c,c]/den)
            return float(np.mean(vals)) if vals else 'UNDEFINED'
        m=local_ap_metric()
        for key in m._defaults:
            original=states[scope][key]
            setattr(m,key,[x for i,x in enumerate(original) if i in keep] if len(original)==len(windows) else original)
        m._update_count=len(keep)
        ap=m.compute();raw=[v for r in rr for v in r['raw_best_ious']];den=sum(r['matched_gt_count'] for r in rr)
        l=dict(windows=len(rr),valid_pixels=sum(r['valid_pixels'] for r in rr),windows_without_thing_gt=sum(r['gt_count']==0 for r in rr),gt_count=sum(r['gt_count'] for r in rr),semantic_confusion=conf.tolist(),panoptic_semantic_confusion=pc.tolist(),
               semantic_miou=miou(conf,range(20)),mIoU_thing=miou(conf,range(2,20)),mIoU_stuff=miou(conf,(0,1)),panoptic_semantic_miou=miou(pc,range(20)),
               panoptic_pq=float(np.mean([r['panoptic_pq'] for r in rr])),candidate_ap={k:float(ap[k]) for k in ('map','map_50')},
               raw_best_iou_ge_0_5_fraction=sum(v>=.5 for v in raw)/len(raw) if raw else 'UNDEFINED',matched_gt_count=den,
               classification_confusion=np.sum([r['classification_confusion'] for r in rr],axis=0).tolist(),
               matched_19_class_accuracy=sum(r['matched_19_class_accuracy']*r['matched_gt_count'] for r in rr)/den if den else 'UNDEFINED')
        for key in ('psnr','ssim','lpips'):l[key]=float(np.mean([r[key] for r in rr]))
        for b in ('candidate_ca','candidate_cw','panoptic_ca','panoptic_cw'):l[b]={k:sum(r[b][k] for r in rr) for k in ('tp','fp','fn')}
        result[scope]=l
    return result

def prepare_full(epoch):
    root=EVAL/f'full_epoch{epoch:02}';allrows=[];states={s:collections.defaultdict(list) for s in SCOPES};sources=[]
    expected=full_windows();expected_ids={(w['scene'],tuple(w['context'])) for w in expected}
    for shard in range(8):
        p=root/f'shard{shard:02}';done=read(p/'complete.json');assert done['status']=='COMPLETE'
        for name,h in done['files'].items():assert rt.sha(p/name)==h
        result=read(p/'result.json');allrows.extend(result['windows']);sources.append(done)
        state=torch.load(p/'candidate_ap_states.pt',map_location='cpu',weights_only=False)
        for s in SCOPES:
            for k,v in state[s].items():states[s][k].extend(v)
        for arm in ('all','novel'):
            export=p/f'official/step_{epoch*1043:04d}/full_{shard:02}'/arm
            target=root/'aggregated_exports'/arm;target.mkdir(parents=True,exist_ok=True)
            for scene in export.iterdir():
                if scene.is_dir():
                    link=target/scene.name
                    if not link.exists():link.symlink_to(scene.resolve(),target_is_directory=True)
    assert len(allrows)==len(expected) and {(w['scene'],tuple(w['context'])) for w in allrows}==expected_ids
    original={(w['scene'],tuple(w['context'])):i for i,w in enumerate(allrows)}
    order=[original[(w['scene'],tuple(w['context']))] for w in expected]
    allrows=[allrows[i] for i in order]
    for s in SCOPES:
        for k,v in states[s].items():
            if len(v)==len(order):states[s][k]=[v[i] for i in order]
    dev={w['scene'] for w in read(rt.REPORT/'manifest.json')['monitor_splits']['dev8']}
    results={}
    for cohort in ('all','excluding_dev8_scenes'):
        keep={i for i,w in enumerate(allrows) if cohort=='all' or w['scene'] not in dev}
        results[cohort]=dict(local=aggregate_local(allrows,states,keep),window_count=len(keep),scenes=sorted({allrows[i]['scene'] for i in keep}))
    write(root/'aggregated_local.json',results);write(root/'full_sources.json',sources);write(root/'all_windows.json',allrows)
    torch.save(dict(states=states,windows=[{k:v for k,v in w.items() if k!='scopes'} for w in allrows]),root/'all_candidate_ap_states.pt')
    # Per-scene metrics are computed from the same cached raw predictions; AP is
    # recomputed across its windows rather than averaged window AP.
    per_scene=[]
    for scene in sorted({w['scene'] for w in allrows}):
        keep={i for i,w in enumerate(allrows) if w['scene']==scene};local=aggregate_local(allrows,states,keep)
        per_scene.append(dict(scene=scene,windows=len(keep),local=local))
    write(root/'per_scene_local.json',per_scene)
    print('FULL_LOCAL_COMPLETE',epoch,flush=True)

def finish_full(epoch):
    root=EVAL/f'full_epoch{epoch:02}';local=read(root/'aggregated_local.json');official=read(root/'official_aggregated.json');rows=[]
    for cohort,r in local.items():
        r['official']=official['cohorts'][cohort];rows.extend(rows_for(epoch,'full_validation',r,cohort))
    write(root/'metrics.json',rows);csvout(root/'metrics.csv',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['monitor','prepare_full','finish_full']);p.add_argument('--epoch',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(4)
    {'monitor':monitor,'prepare_full':lambda:prepare_full(a.epoch),'finish_full':lambda:finish_full(a.epoch)}[a.mode]()
