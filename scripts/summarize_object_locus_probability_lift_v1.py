#!/usr/bin/env python3
"""Summarize official paired exports and compute the registered scene bootstrap."""
import argparse, json, hashlib, math
from pathlib import Path
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

DEFAULT_REPORT=Path('/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1')
SIU=Path('/space/mawb/SIU3R')

def main():
    import sys
    parser=argparse.ArgumentParser();parser.add_argument('--report-root',type=Path,default=DEFAULT_REPORT);args=parser.parse_args()
    global REPORT
    REPORT=args.report_root.resolve()
    sys.path.insert(0,str(SIU))
    from src.config import EvaluatorCfg
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
    cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=True,eval_context_pq=True,eval_context_map=True,
      eval_target_miou=True,eval_target_pq=True,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
      id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(REPORT))
    official=Evaluator(cfg); official.setup()
    cohort='val32_excluding_dev8_scenes'; root=REPORT/'C'/cohort/'official'/'novel'
    manifests=json.loads((REPORT/'manifests.json').read_text())
    def invalidate(reason):
      summary={'status':'INVALID','reason':reason,'official_metrics':None,'bootstrap':None}
      (REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
      (REPORT/'summary.md').write_text('# Probability lift paired evaluation\n\n**Status: INVALID.**\n\n'+reason+'\n')
      raise SystemExit(2)
    for cohort_name,manifest_key in (('val32_all','val32_all'),('val32_excluding_dev8_scenes',cohort)):
      expected={w['scene'] for w in manifests[manifest_key]}
      if len(expected)!=(32 if cohort_name=='val32_all' else 24): invalidate(f'Unexpected registered window count for {cohort_name}')
      for export_scope in ('all','novel'):
        roots={arm:REPORT/arm/cohort_name/'official'/export_scope for arm in ('C','P')}
        arm_dirs={}
        for arm,root_dir in roots.items():
          dirs=[p for p in root_dir.iterdir() if p.is_dir()]
          found={}
          for d in dirs:
            hits=[scene for scene in expected if d.name.startswith(scene+'_context')]
            if len(hits)!=1 or hits[0] in found: invalidate(f'Unexpected/duplicate export identity {d} in {cohort_name}/{export_scope}')
            found[hits[0]]=d
          if set(found)!=expected or len(dirs)!=len(expected): invalidate(f'Incomplete {arm} export identities in {cohort_name}/{export_scope}')
          arm_dirs[arm]=found
        for scene in sorted(expected):
          cgt=sorted((arm_dirs['C'][scene]/'target_seg_gt').glob('*.png'))
          pgt=sorted((arm_dirs['P'][scene]/'target_seg_gt').glob('*.png'))
          if [p.name for p in cgt]!=[p.name for p in pgt] or not cgt: invalidate(f'GT identity mismatch for {cohort_name}/{export_scope}/{scene}')
          for cf,pf in zip(cgt,pgt):
            if hashlib.sha256(cf.read_bytes()).digest()!=hashlib.sha256(pf.read_bytes()).digest(): invalidate(f'GT content mismatch: {scene}/{cf.name}')
    scenes=sorted(p for p in root.iterdir() if p.is_dir())
    payload={arm:[] for arm in ('C','P')}
    for scene_dir in scenes:
      cdir=REPORT/'C'/cohort/'official'/'novel'/scene_dir.name
      pdir=REPORT/'P'/cohort/'official'/'novel'/scene_dir.name
      if not pdir.is_dir(): invalidate(f'Missing P export for {scene_dir.name}')
      cgt=sorted((cdir/'target_seg_gt').glob('*.png'));pgt=sorted((pdir/'target_seg_gt').glob('*.png'))
      if [p.name for p in cgt]!=[p.name for p in pgt] or not cgt: invalidate(f'GT identity mismatch for {scene_dir.name}')
      for cf,pf in zip(cgt,pgt):
        if hashlib.sha256(cf.read_bytes()).digest()!=hashlib.sha256(pf.read_bytes()).digest(): invalidate(f'GT content mismatch: {scene_dir.name}/{cf.name}')
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
    for cohort_name in ('val32_all','val32_excluding_dev8_scenes'):
      all_results={arm:json.loads((REPORT/arm/cohort_name/'official_all.json').read_text()) for arm in ('C','P')}
      novel_results={arm:json.loads((REPORT/arm/cohort_name/'official_novel.json').read_text()) for arm in ('C','P')}
      for arm in ('C','P'):
        for result in (all_results[arm],novel_results[arm]):
          if not isinstance(result,dict): invalidate('Official result file is not an object.')
      for scope,source,keys in (
        ('context',all_results,('context_miou','context_pq','context_map')),
        ('target-all',all_results,('target_miou','target_pq','target_map')),
        ('true-novel',novel_results,('target_miou','target_pq','target_map'))):
        for metric_name,key in zip(('mIoU','PQ','mAP'),keys):
          def get(arm,sub=None):
            x=source[arm].get(key)
            return x.get(sub) if sub and isinstance(x,dict) else (None if sub else x)
          a=get('C','map') if metric_name=='mAP' else get('C')
          b=get('P','map') if metric_name=='mAP' else get('P')
          aa=get('C','map_50') if metric_name=='mAP' else None
          bb=get('P','map_50') if metric_name=='mAP' else None
          if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isfinite(a) or not math.isfinite(b):
            invalidate(f'Missing/nonfinite official {metric_name} for {cohort_name}/{scope}')
          rows.append({'cohort':cohort_name,'scope':scope,'metric':metric_name,'C':a,'P':b,'P-C':b-a})
          if metric_name=='mAP':
            if not isinstance(aa,(int,float)) or not isinstance(bb,(int,float)) or not math.isfinite(aa) or not math.isfinite(bb): invalidate(f'Missing/nonfinite AP50 for {cohort_name}/{scope}')
            rows.append({'cohort':cohort_name,'scope':scope,'metric':'AP50','C':aa,'P':bb,'P-C':bb-aa})
    cmain=json.loads((REPORT/'C'/cohort/'official_novel.json').read_text())
    pmain=json.loads((REPORT/'P'/cohort/'official_novel.json').read_text())
    def metric(obj,key,sub=None):
      x=obj.get(key)
      if isinstance(x,dict) and sub is not None: x=x.get(sub)
      return float(x) if isinstance(x,(int,float)) else float('nan')
    dm=metric(pmain,'target_map','map')-metric(cmain,'target_map','map')
    da=metric(pmain,'target_map','map_50')-metric(cmain,'target_map','map_50')
    dpq=metric(pmain,'target_pq')-metric(cmain,'target_pq')
    primary_values=[dm,da,dpq,*bootstrap['delta_CI95']['map'],*bootstrap['delta_CI95']['map_50']]
    if not all(math.isfinite(x) for x in primary_values): invalidate('Primary metric or paired confidence interval is missing or nonfinite.')
    parity=json.loads((REPORT/'parity.json').read_text())
    all_scenes={w['scene'] for w in manifests['val32_all']}
    if len(parity.get('windows',[]))!=32 or {x.get('scene') for x in parity['windows']}!=all_scenes:
      invalidate('Paired parity report is missing windows or has mismatched identities.')
    if parity.get('unchanged') is not True:
      invalidate('Model state_dict changed during inference.')
    for row in parity['windows']:
      vals=row.get('diff',{})
      if not vals or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v>1e-6 for v in vals.values()):
        invalidate(f'Parity failed or has nonfinite values for {row.get("scene")}')
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
