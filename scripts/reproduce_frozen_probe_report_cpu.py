#!/usr/bin/env python3
"""Recompute classifier metrics and cohort point deltas from packaged caches.

No renderer, original model, training windows, optimizer, or CUDA is used.
Official packed AP can be reproduced from the packaged prediction trees with
the registered SIU3R evaluator; this script recomputes classification and
per-window point aggregates from dev/test q/z and labels.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from scripts.object_locus_probe_metrics import classification_summary
from scripts.train_object_locus_frozen_probe import Readout

def rows(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,data):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def validate_packed_window_identity(root,window_ids):
 idx_path=root/'packed_window_index.json';manifest_path=root/'cohort_manifest.json'
 if not idx_path.is_file() or not manifest_path.is_file():raise FileNotFoundError('CPU report reproduction requires packed_window_index.json and cohort_manifest.json')
 index=json.loads(idx_path.read_text());manifest=json.loads(manifest_path.read_text())['test']
 h0=[x for x in index['entries'] if x['readout']=='H0']
 observed=[(x['scene'],x['context_frame_ids'],x['true_novel_frame_ids'],x['pair_name']) for x in h0]
 expected=[(w['scene'],list(map(int,w['context'])),list(map(int,w['novel'])),f"{w['scene']}_context{'_'.join(map(str,w['context']))}") for w in manifest]
 if observed!=expected:raise RuntimeError('packed_window_index H0 identities differ from locked test cohort_manifest order')
 expected_ids=[f"test_{i:04d}_{w['scene']}_c{'_'.join(map(str,w['context']))}" for i,w in enumerate(manifest)]
 observed_ids=[str(x) for x in window_ids if str(x).startswith('test_')]
 if observed_ids!=expected_ids:raise RuntimeError('cached test feature window order differs from packed_window_index/cohort_manifest')
 return {'status':'PASS','test_windows':len(expected),'scene_order':[w['scene'] for w in manifest],
   'packed_window_index_sha256':__import__('hashlib').sha256(idx_path.read_bytes()).hexdigest()}
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root
 if torch.cuda.is_available():raise RuntimeError('reproduction must be CPU-only')
 with np.load(r/'features/dev_test_q_z.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 packed_window_identity=validate_packed_window_identity(r,data['window_ids'].tolist())
 idx={str(w):i for i,w in enumerate(data['window_ids'].tolist())};fixed=rows(r/'labels/dev_test_scope_labels.csv');r3d=rows(r/'labels/r3d_dev_test_scope_labels.csv') if (r/'labels/r3d_dev_test_scope_labels.csv').exists() else []
 labelmap={(x['split'],x['scope'],x['window_id']):None for x in fixed}
 for x in fixed:labelmap[(x['split'],x['scope'],x['window_id'])]=[]
 for x in fixed:labelmap[(x['split'],x['scope'],x['window_id'])].append(int(x['label']))
 rmap={}
 for x in r3d:rmap.setdefault((x['split'],x['scope'],x['window_id']),[]).append(int(x['label']))
 heads=[('H0',None)]+[(h,s) for h in ('H1','H2','H3') for s in (20261,20262,20263)]
 predictions={}
 for h,s in heads:
  model=None
  if h!='H0':
   cp=torch.load(r/f'heads/{h}/seed_{s}/best.pt',map_location='cpu',weights_only=False);model=Readout(h).cpu().float();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
  per={}
  for i,w in enumerate(data['window_ids'].tolist()):
   with torch.no_grad():prob=data['pclass'][i] if h=='H0' else torch.softmax(model(torch.from_numpy(data['q'][i]),torch.from_numpy(data['z'][i])),-1).numpy()
   per[str(w)]=prob
  predictions[(h,s)]=per
 metrics=[];deltas=[]
 for (h,s),per in predictions.items():
  for scope in ('context','true-novel'):
   for split in ('dev','test'):
    yy=[];pp=[];correct=cond_correct=positives=0
    for wid,prob in per.items():
     if not wid.startswith(split+'_'):continue
     y=np.asarray(labelmap[(split,scope,wid)],np.int64);m=classification_summary(prob,y)
     yy.extend(y.tolist());pp.extend(prob.tolist());correct+=int(round((m['joint19_accuracy'] or 0)*m['positive_count']));cond_correct+=int(round((m['conditional18_accuracy'] or 0)*m['positive_count']));positives+=m['positive_count']
    agg=classification_summary(np.asarray(pp),np.asarray(yy));metrics.append({'cohort':split,'scope':scope,'head':h,'seed':'' if s is None else s,
      'joint19_accuracy':agg['joint19_accuracy'],'conditional18_accuracy':agg['conditional18_accuracy'],'macro_f1_supported_classes':agg['macro_f1_supported_classes'],
      'conditional_macro_f1_supported_classes':agg['conditional_macro_f1_supported_classes'],'positive_count':agg['positive_count'],'negative_count':agg['negative_count'],'ambiguous_count':agg['ambiguous_count']})
 write(r/'reproduced_feature_metrics.csv',metrics)
 lookup={(x['cohort'],x['scope'],x['head'],x['seed']):x for x in metrics}
 for split in ('dev','test'):
  for scope in ('context','true-novel'):
   r_y=[];r_p=[]
   for w in data['window_ids'].tolist():
    if not w.startswith(split+'_'):continue
    with np.load(r/f'r3d/features/{w}.npz') as rz:r_p.append(rz['pclass'].copy())
    r_y.extend(rmap[(split,scope,w)])
   r3=classification_summary(np.concatenate(r_p),np.asarray(r_y))
   metrics.append({'cohort':split,'scope':scope,'head':'R3D','seed':'','joint19_accuracy':r3['joint19_accuracy'],'conditional18_accuracy':r3['conditional18_accuracy'],
    'macro_f1_supported_classes':r3['macro_f1_supported_classes'],'conditional_macro_f1_supported_classes':r3['conditional_macro_f1_supported_classes'],
    'positive_count':r3['positive_count'],'negative_count':r3['negative_count'],'ambiguous_count':r3['ambiguous_count']})
   lookup[(split,scope,'R3D','')]=metrics[-1]
   for h,s in [('H1',20261),('H2',20261),('H3',20261),('R3D',None)]:
    left=lookup[(split,scope,h,'' if s is None else str(s))];base=lookup[(split,scope,'H0','')]
    deltas.append({'cohort':split,'scope':scope,'comparison':h+'_seed_'+str(s)+'-H0','joint19_accuracy_delta':None if left['joint19_accuracy'] is None else left['joint19_accuracy']-base['joint19_accuracy'],
      'conditional18_accuracy_delta':None if left['conditional18_accuracy'] is None else left['conditional18_accuracy']-base['conditional18_accuracy']})
 write(r/'reproduced_point_deltas.csv',deltas)
 print(json.dumps({'status':'PASS','readouts':11,'classifier_rows':len(metrics),'point_deltas':len(deltas),'cuda_used':False,
  'packed_window_identity':packed_window_identity,
  'scope':'R3D class metrics use R3D-native GT matching; probe metrics use the fixed GC001 mask labels'}))
if __name__=='__main__':main()
