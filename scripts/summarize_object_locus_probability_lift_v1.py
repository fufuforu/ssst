#!/usr/bin/env python3
"""Summarize official paired exports and compute the registered scene bootstrap."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

REPORT=Path('/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1')
SIU=Path('/space/mawb/SIU3R')

def main():
    import sys
    sys.path.insert(0,str(SIU))
    from src.config import EvaluatorCfg
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
    cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=True,eval_context_pq=True,eval_context_map=True,
      eval_target_miou=True,eval_target_pq=True,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
      id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(REPORT))
    official=Evaluator(cfg); official.setup()
    cohort='val32_excluding_dev8_scenes'; root=REPORT/'C'/cohort/'official'/'novel'
    scenes=sorted(p for p in root.iterdir() if p.is_dir())
    payload={arm:[] for arm in ('C','P')}
    for scene_dir in scenes:
      for arm in payload:
        d=REPORT/arm/cohort/'official'/'novel'/scene_dir.name
        x=official.process_segmentation(d/'target_seg_pred',d/'target_seg_gt')
        payload[arm].append((x['map_pred'],x['map_gt']))
    rng=np.random.default_rng(42); delta=[]; reps=2000
    metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
    for _ in range(reps):
      ids=rng.integers(0,len(scenes),len(scenes)); vals={}
      for arm in ('C','P'):
        metric.reset();metric.update([payload[arm][i][0] for i in ids],[payload[arm][i][1] for i in ids])
        result=metric.compute();vals[arm]=(float(result['map']),float(result['map_50']))
      delta.append({'map':vals['P'][0]-vals['C'][0],'map_50':vals['P'][1]-vals['C'][1]})
    ci={k:[float(np.quantile([r[k] for r in delta],.025)),float(np.quantile([r[k] for r in delta],.975))] for k in ('map','map_50')}
    bootstrap={'scope':'main cohort true-novel official packed','seed':42,'resamples':reps,'scene_count':len(scenes),
      'delta_CI95':ci,'mean_delta':{k:float(np.mean([r[k] for r in delta])) for k in ('map','map_50')}}
    (REPORT/'paired_bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    rows=[]
    for cohort in ('val32_all','val32_excluding_dev8_scenes'):
      for scope in ('all','novel'):
       results={arm:json.loads((REPORT/arm/cohort/f'official_{scope}.json').read_text()) for arm in ('C','P')}
       for key in ('context_miou','target_miou','context_pq','target_pq','context_map','target_map'):
        a=results['C'].get(key);b=results['P'].get(key)
        if isinstance(a,dict):a=a.get('map')
        if isinstance(b,dict):b=b.get('map')
        if isinstance(a,(int,float)) and isinstance(b,(int,float)): rows.append({'cohort':cohort,'scope':scope,'metric':key,'C':a,'P':b,'P-C':b-a})
    cmain=json.loads((REPORT/'C'/cohort/'official_novel.json').read_text())
    pmain=json.loads((REPORT/'P'/cohort/'official_novel.json').read_text())
    def metric(obj,key,sub=None):
      x=obj.get(key)
      if isinstance(x,dict) and sub is not None: x=x.get(sub)
      return float(x) if isinstance(x,(int,float)) else float('nan')
    dm=metric(pmain,'target_map','map')-metric(cmain,'target_map','map')
    da=metric(pmain,'target_map','map_50')-metric(cmain,'target_map','map_50')
    dpq=metric(pmain,'target_pq')-metric(cmain,'target_pq')
    if bootstrap['delta_CI95']['map'][0]>0 and dm>=.01 and da>=-.01 and dpq>=-.01: status='SUCCESS'
    elif bootstrap['delta_CI95']['map'][1]<0 or bootstrap['delta_CI95']['map_50'][1]<-.01: status='FAILURE'
    else: status='INCONCLUSIVE'
    summary={'status':status,'primary':bootstrap,'primary_point_estimates':{'delta_map':dm,'delta_ap50':da,'delta_pq':dpq},'official':rows,
      'interpretation':'single checkpoint inference-only readout comparison; does not establish training efficacy'}
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Probability lift paired evaluation','',f"Pre-registered status: **{summary['status']}**",'',
      'Primary true-novel scene bootstrap (P−C):',f"- mAP CI95: {ci['map']}",f"- AP50 CI95: {ci['map_50']}",'',
      '| Cohort | Export | Official metric | C | P | P−C |','|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['cohort']} | {r['scope']} | {r['metric']} | {r['C']:.4f} | {r['P']:.4f} | {r['P-C']:+.4f} |" for r in rows]
    lines += ['', 'This is a same-checkpoint, inference-only comparison. It does not establish that a retrained model would improve.']
    (REPORT/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
