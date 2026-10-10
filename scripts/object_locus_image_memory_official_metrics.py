"""Unchanged Full1201 official metric create/update/state-merge helpers.

Copied verbatim from the completed Full1201 evaluator; no metric changes.
"""
import sys
sys.path.insert(0, '/space/mawb/SIU3R')
import numpy as np
import torch
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME, STUFF_CLASSES, THING_CLASSES
METRICS=('context_miou','context_pq','context_mAP','target_miou','target_pq','target_mAP')
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
