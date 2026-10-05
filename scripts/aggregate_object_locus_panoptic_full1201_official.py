"""Exact state aggregation using only pinned official processing/metric classes."""
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0,'/space/mawb/SIU3R')
import numpy as np
import torch
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
METRICS=('context_miou','context_pq','context_mAP','target_miou','target_pq','target_mAP')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')
def itemize(x):
 if isinstance(x,torch.Tensor):return x.tolist() if x.ndim else x.item()
 if isinstance(x,dict):return {k:itemize(v) for k,v in x.items()}
 return x

def create(path):
 cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=True,eval_context_pq=True,eval_context_map=True,
  eval_target_miou=True,eval_target_pq=True,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
  id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(path))
 e=Evaluator(cfg);e.setup();return e

def update(e,data,view):
 getattr(e,view+'_miou').update(data['pred_semantics'],data['gt_semantics'])
 getattr(e,view+'_pq').update(torch.stack([data['pred_semantics'],data['pred_instances']],dim=-1),torch.stack([data['gt_semantics'],data['gt_instances']],dim=-1))
 getattr(e,view+'_mAP').update([data['map_pred']],[data['map_gt']])

def result(e):
 r={}
 for view in ('context','target'):
  iou=itemize(getattr(e,view+'_miou').compute());pq=itemize(getattr(e,view+'_pq').compute())
  r[view+'_ious_per_class']=iou;r[view+'_miou']=float(np.mean(iou));r[view+'_pqs_per_class']=pq;r[view+'_pq']=float(np.mean(pq))
  r[view+'_map']=itemize(getattr(e,view+'_mAP').compute())
 return r

def states(e):return {name:{k:getattr(getattr(e,name),k) for k in getattr(e,name)._defaults} for name in METRICS}

def merge(blobs,path):
 e=create(path);names=[n for b in blobs for n in b['names']];order=sorted(range(len(names)),key=lambda i:names[i])
 for name in METRICS:
  m=getattr(e,name)
  for k in m._defaults:
   values=[b['states'][name][k] for b in blobs]
   if isinstance(values[0],list):
    values=[v for group in values for v in group]
    if len(values)==len(names):values=[values[i] for i in order]
    setattr(m,k,values)
   else:setattr(m,k,sum(values))
  m._update_count=len(names)
 return result(e)

def shard(root,shard_index,total):
 exports=root/'aggregated_exports';names=sorted(p.name for p in (exports/'all').iterdir() if p.is_dir());assert len(names)==1860
 scenes=sorted({n.split('_context')[0] for n in names});selected=set(scenes[shard_index::total]);names=[n for n in names if n.split('_context')[0] in selected]
 dev={w['scene'] for w in read(REPORT/'manifest.json')['monitor_splits']['dev8']}
 per_scene={s:dict(scene=s,windows=sum(n.split('_context')[0]==s for n in names),official={}) for s in selected};saved={}
 for arm in ('all','novel'):
  totals={c:create(exports/arm) for c in ('all','excluding_dev8_scenes')};cohort_names={c:[] for c in totals}
  scene_metrics={s:create(exports/arm) for s in selected}
  for index,name in enumerate(names):
   scene=name.split('_context')[0];pair=exports/arm/name
   for view in ('context','target'):
    data=totals['all'].process_segmentation(pair/f'{view}_seg_pred',pair/f'{view}_seg_gt')
    update(totals['all'],data,view);update(scene_metrics[scene],data,view)
    if scene not in dev:update(totals['excluding_dev8_scenes'],data,view)
   cohort_names['all'].append(name)
   if scene not in dev:cohort_names['excluding_dev8_scenes'].append(name)
   print('OFFICIAL_RAW',arm,index+1,len(names),name,flush=True)
  saved[arm]={c:dict(names=cohort_names[c],states=states(e)) for c,e in totals.items()}
  for scene,e in scene_metrics.items():per_scene[scene]['official'][arm]=result(e)
 torch.save(saved,root/f'official_states_shard{shard_index:02}.pt')
 write(root/f'per_scene_official_shard{shard_index:02}.json',[per_scene[s] for s in sorted(per_scene)])
 write(root/f'official_shard{shard_index:02}_complete.json',dict(status='COMPLETE',windows=len(names),scenes=len(selected),official_commit=SIU3R_COMMIT,optimizer_updates=0))

def reduce(root):
 blobs=[torch.load(root/f'official_states_shard{s:02}.pt',map_location='cpu',weights_only=False) for s in range(8)];cohorts={}
 for cohort in ('all','excluding_dev8_scenes'):
  cohorts[cohort]={arm:merge([b[arm][cohort] for b in blobs],root) for arm in ('all','novel')}
 write(root/'official_aggregated.json',dict(cohorts=cohorts,official_commit=SIU3R_COMMIT,source_count=1860,
  aggregation='Pinned process_segmentation and official metric updates; summed sufficient states, canonical pair order for globally recomputed AP; no scene AP mean',optimizer_updates=0,job_id=os.environ.get('SLURM_JOB_ID')))

def smoke():
 root=REPORT/'evaluation/smoke';base=root/'official/step_8344/evaluator_smoke';checked=[]
 for arm in ('all','novel'):
  e=create(base/arm);names=sorted(p.name for p in (base/arm).iterdir() if p.is_dir())
  for name in names:
   pair=base/arm/name
   for view in ('context','target'):update(e,e.process_segmentation(pair/f'{view}_seg_pred',pair/f'{view}_seg_gt'),view)
  actual=merge([dict(names=names,states=states(e))],base/arm);expected=read(base/f'official_{arm}.json')['result']
  for view in ('context','target'):
   for key in ('miou','pq'):assert abs(actual[view+'_'+key]-expected[view+'_'+key])<1e-7,(arm,view,key)
   for key in ('map','map_50'):assert abs(actual[view+'_map'][key]-expected[view+'_map'][key])<1e-7,(arm,view,key)
  checked.append(arm)
 write(REPORT/'official_aggregation_contract.json',dict(status='PASS',arms=checked,method='saved real-window official exports reproduce all context/target mIoU/PQ/mAP/AP50 via exact state merge',optimizer_updates=0))
 print('OFFICIAL_STATE_MERGE_CONTRACT_PASS',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,default=8);p.add_argument('--scene-shard',type=int);p.add_argument('--shards',type=int,default=8);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 assert subprocess.check_output(['git','-C','/space/mawb/SIU3R','rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
 torch.set_num_threads(4)
 if a.smoke:smoke()
 elif a.scene_shard is None:reduce(REPORT/'evaluation'/f'full_epoch{a.epoch:02}')
 else:shard(REPORT/'evaluation'/f'full_epoch{a.epoch:02}',a.scene_shard,a.shards)
if __name__=='__main__':main()
