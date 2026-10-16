import pathlib,json,numpy as np
from PIL import Image
p=pathlib.Path('/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt02/full')
b=pathlib.Path('/space/mawb/ssst/group_plus/dual_stream_training_v1')
results={}
def read(f):
 a=np.asarray(Image.open(f),dtype=np.int64);return a[:,:,0]+256*a[:,:,1]+65536*a[:,:,2]
for arm,name in [('isolated','attempt12_geometric_sam_isolated_node13_8gpu'),('weight005','attempt13_geometric_sam_weight005_node18_8gpu')]:
 rr=[];detail=[]
 for c in sorted((p/arm).glob('rank*/COMPLETE.json')):
  d=json.loads(c.read_text())
  for idx in d['records']:
   r=json.loads((c.parent/f'{idx}.json').read_text());pair=b/name/'evaluation/all'/(r['scene']+'_context'+'_'.join(map(str,r['context'])))
   gold=np.stack([read(pair/'context_seg_gt'/f"{r['scene']}_gt{f}.png") for f in r['context']]);pred=np.stack([read(pair/'context_seg_pred'/f"{r['scene']}_pred{f}.png") for f in r['context']])
   sem=gold//1000-1;ins=gold%1000;pid=pred%1000;valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0));pa=np.bincount(pid[valid],minlength=101)[1:101]
   n=len(r['gt_ids']);m=np.zeros((100,n));labels=np.full(100,-1)
   for record in json.loads((pair/'context_seg_pred/pred.json').read_text()):labels[record['id']-1]=record['label_id']-1
   for k,iid in enumerate(r['gt_ids']):
    mask=valid&(sem>=2)&(ins==iid);inter=np.bincount(pid[mask],minlength=101)[1:101];m[:,k]=inter/np.maximum(1,pa+mask.sum()-inter)
   a=np.asarray(r['mask_IoU']['final']).reshape(100,n)
   for k in range(n):
    rr.append(dict(src=idx,gt=r['gt_ids'][k],pixels=r['gt_pixels'][k],gold=float(m[:,k].max()),diagnostic=float(a[:,k].max()),gold_correct=float(np.where(labels==r['gt_classes'][k],m[:,k],0).max()),diag_correct=float(np.where(np.asarray(r['query_labels'])==r['gt_classes'][k],a[:,k],0).max())))
   detail.append(dict(source_index=idx,max_matrix_diff=float(np.abs(m-a).max()) if n else 0,max_best_diff=float(np.abs(m.max(0)-a.max(0)).max()) if n else 0))
 print(arm,'N',len(rr),'goldrec50',sum(x['gold']>=.5 for x in rr),'diagrec50',sum(x['diagnostic']>=.5 for x in rr),'gold_correct50',sum(x['gold_correct']>=.5 for x in rr),'diag_correct50',sum(x['diag_correct']>=.5 for x in rr),'mean_gold',np.mean([x['gold'] for x in rr]),'mean_diag',np.mean([x['diagnostic'] for x in rr]))
 print('changes',sum((x['gold']>=.5)!=(x['diagnostic']>=.5) for x in rr),'top',sorted(detail,key=lambda x:-x['max_best_diff'])[:6])
 results[arm]=dict(instances=rr,scene_differences=detail)
(p/'golden_output_audit.json').write_text(json.dumps(results,indent=2)+'\n')
