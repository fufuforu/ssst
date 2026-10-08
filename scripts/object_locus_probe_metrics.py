"""Single source of truth for frozen-probe classification summaries."""
from __future__ import annotations
import numpy as np,hashlib,json
from pathlib import Path

def _file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def verify_gc_cache_manifest(root):
    root=Path(root);m=json.loads((root/'cache_manifest.json').read_text());checked=0
    for row in m['files']:
        for pk,hk,sk in [('path','sha256','size'),('feature_path','feature_sha256','feature_size'),('iou_path','iou_sha256','iou_size'),('reconstruction_cache','reconstruction_sha256','reconstruction_size')]:
            if pk not in row:continue
            p=Path(row[pk])
            if not p.is_file() or p.stat().st_size!=row[sk] or _file_sha(p)!=row[hk]:raise RuntimeError(f'GC001 cache file SHA/size mismatch: {p}')
            checked+=1
    c=m['combined_feature_bundle'];p=Path(c['path'])
    if not p.is_file() or p.stat().st_size!=c['size'] or _file_sha(p)!=c['sha256']:raise RuntimeError('combined q/z feature cache SHA mismatch')
    return {'status':'PASS','files':checked+1}

def verify_r3d_cache_manifest(root):
    root=Path(root);m=json.loads((root/'r3d_cache_manifest.json').read_text());checked=0
    for row in m['records']:
        for pkey,hkey in [('cache','cache_sha256'),('features','features_sha256'),('reconstruction_cache','reconstruction_sha256')]:
            p=Path(row[pkey]);
            if not p.is_file() or _file_sha(p)!=row[hkey]:raise RuntimeError(f'R3D cache file SHA mismatch: {p}')
            checked+=1
        for x in row['scope_iou'].values():
            p=Path(x['path'])
            if not p.is_file() or p.stat().st_size!=x['size'] or _file_sha(p)!=x['sha256']:raise RuntimeError(f'R3D IoU SHA mismatch: {p}')
            checked+=1
    return {'status':'PASS','files':checked}

def gt_rows(sem,ins):
    import torch
    valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0));ids=torch.unique(ins[valid&(sem>=2)&(ins>0)],sorted=True);rows=[]
    for iid in ids:
        mask=valid&(sem>=2)&(ins==iid);counts=torch.bincount(sem[mask],minlength=20)
        cls=int(torch.nonzero(counts==counts.max(),as_tuple=False)[0,0]);rows.append((int(iid),cls,mask))
    return valid,rows

def raw_iou(region,alpha,sem,ins):
    valid,rows=gt_rows(sem,ins);pred=(region[:,:100]>=.5)&(alpha[:,None]>.05)
    iou=np.zeros((len(rows),100),np.float64);inter=np.zeros((len(rows),100),np.int64);union=np.zeros_like(inter)
    for gi,(_,_,gm) in enumerate(rows):
        for q in range(100):
            pm=pred[:,q]&valid;iv=int((pm&gm).sum());uv=int(pm.sum()+gm.sum()-iv)
            inter[gi,q]=iv;union[gi,q]=uv;iou[gi,q]=iv/uv if uv else 0.
    return rows,iou,inter,union

def binary_ranking(labels, scores):
    y=np.asarray(labels,dtype=np.int64);s=np.asarray(scores,dtype=np.float64)
    pos=y==1;neg=y==0
    if not pos.any() or not neg.any(): return None,None
    order=np.argsort(s,kind='mergesort'); ranks=np.empty(len(s),np.float64)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[order[j]]==s[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2.;i=j
    auc=float((ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))
    desc=np.argsort(-s,kind='mergesort'); yy=pos[desc].astype(np.int64)
    tp=np.cumsum(yy); fp=np.cumsum(1-yy)
    ap=float(np.sum((tp/np.maximum(tp+fp,1))*yy)/pos.sum())
    return auc,ap

def classification_summary(pclass, labels):
    p=np.asarray(pclass,dtype=np.float64);y=np.asarray(labels,dtype=np.int64)
    if p.shape!=(len(y),19): raise ValueError('pclass must be [N,19]')
    positive=(y>=0)&(y<18);negative=y==18;ambiguous=y==-1;known=positive|negative
    joint=p.argmax(-1);conditional=p[:,:18].argmax(-1);conf=np.zeros((18,19),np.int64)
    for yy,pp in zip(y[positive],joint[positive]):conf[yy,pp]+=1
    active=np.flatnonzero(conf.sum(1)>0); per=[];cconf=np.zeros((18,18),np.int64)
    for yy,pp in zip(y[positive],conditional[positive]):cconf[yy,pp]+=1
    for c in range(18):
        tp=int(conf[c,c]);fp=int(conf[:,c].sum()-tp);fn=int(conf[c].sum()-tp)
        per.append({'class_internal':c+2,'support':int(conf[c].sum()),'precision':tp/(tp+fp) if tp+fp else 0.,
                    'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.})
    auc,ap=binary_ranking(positive[known].astype(np.int64),1-p[known,18])
    return {'joint19_accuracy':float(np.mean(joint[positive]==y[positive])) if positive.any() else None,
      'conditional18_accuracy':float(np.mean(conditional[positive]==y[positive])) if positive.any() else None,
      'known_query19_accuracy':float(np.mean(joint[known]==y[known])) if known.any() else None,
      'macro_f1_supported_classes':float(np.mean([per[c]['f1'] for c in active])) if len(active) else None,
      'conditional_macro_f1_supported_classes':float(np.mean([2*cconf[c,c]/(2*cconf[c,c]+cconf[:,c].sum()-cconf[c,c]+cconf[c,:].sum()-cconf[c,c]) if 2*cconf[c,c]+cconf[:,c].sum()+cconf[c,:].sum()-2*cconf[c,c] else 0. for c in active])) if len(active) else None,
      'support_classes':(active+2).tolist(),'no_support_classes':(np.flatnonzero(conf.sum(1)==0)+2).tolist(),
      'confusion_18x19':conf.tolist(),'conditional_confusion_18x18':cconf.tolist(),'per_class':per,'positive_count':int(positive.sum()),
      'negative_count':int(negative.sum()),'ambiguous_count':int(ambiguous.sum()),
      'objectness_auroc':auc,'objectness_auprc':ap}

def normalize_official_result(result):
    def block(prefix):
        m=result.get(prefix+'_map') or {}
        return {'mIoU':result.get(prefix+'_miou'),'PQ':result.get(prefix+'_pq'),'mAP':m.get('map'),'AP50':m.get('map_50')}
    if 'context_miou' in result or 'target_miou' in result:
        return {'context':block('context'),'true-novel':block('target')}
    if 'context' in result and 'target' in result:
        return {'context':result['context'],'true-novel':result['target']}
    if 'context' in result and 'true-novel' in result:return result
    raise ValueError(f'unrecognized SIU3R official result schema: {sorted(result)}')

def validate_gc001_endpoint_metadata(blob, source_checkpoint_sha):
    expected={'alpha':.01,'epoch':8,'completed_updates':1008,'new_exposures':8064,
      'source_exposure':50064,'model_exposure':58128,'code_sha':'9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0',
      'plan_sha256':'0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8'}
    got={k:blob.get(k) for k in expected}
    if got!=expected:raise ValueError(f'GC001 endpoint metadata mismatch: {got}')
    if source_checkpoint_sha!='68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a':
        raise ValueError('GC001 endpoint source checkpoint identity mismatch')
    if not isinstance(blob.get('config'),dict) or not isinstance(blob.get('model'),dict):raise ValueError('GC001 endpoint needs complete dict config and model state')
    return True

def _self_check():
    p=np.zeros((3,19),np.float64);p[0,18]=.8;p[0,3]=.2
    p[1,3]=.9;p[1,18]=.1;p[2,18]=.9;p[2,3]=.1
    y=np.array([3,3,18]);a=classification_summary(p,y)
    assert a['joint19_accuracy']==.5 and a['conditional18_accuracy']==1
    assert a['objectness_auroc']==1 and a['objectness_auprc']==1
    changed=classification_summary(np.concatenate([p,np.tile(p[0],(10,1))]),np.r_[y,np.full(10,-1)])
    assert changed['joint19_accuracy']==a['joint19_accuracy'] and changed['objectness_auroc']==a['objectness_auroc']
    reverse=classification_summary(np.array([p[2],p[1]]),np.array([3,18]))
    assert reverse['objectness_auroc']==0
    assert classification_summary(p,np.array([-1,-1,-1]))['joint19_accuracy'] is None
    return True

if __name__=='__main__': print({'passed':_self_check()})
