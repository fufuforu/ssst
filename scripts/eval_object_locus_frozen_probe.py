#!/usr/bin/env python3
"""Replay all readouts from frozen CPU caches; export and score packed masks."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.object_locus_frozen_probe_contract import labels_from_context,diagnostic_max_cardinality
from scripts.train_object_locus_frozen_probe import Readout,load_split
from scripts.export_object_locus_v3_set_official import write_official_pair
from scripts.eval_object_locus_v3_set import _candidate_stats
from scripts.object_locus_probe_metrics import classification_summary
from scripts.object_locus_probe_metrics import gt_rows,raw_iou
from scripts.object_locus_probe_metrics import normalize_official_result
from scripts.object_locus_probe_metrics import verify_gc_cache_manifest,verify_r3d_cache_manifest

ATTEMPT=Path(os.environ.get('TASK_ATTEMPT_ROOT','/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02'))
OFFICIAL_PY='/space/mawb/SIU3R/.venv_gpu_v4/bin/python'
INVOKE=ROOT/'scripts/invoke_siu3r_official_evaluator.py'

def dump(path,obj): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
def rows_to_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields:fields.append(key)
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:(json.dumps(v,ensure_ascii=False,separators=(',',':'),default=lambda x:x.item() if hasattr(x,'item') else str(x)) if isinstance(v,(dict,list,tuple)) else v) for k,v in row.items()})

def probs(logits):
    with torch.no_grad(): return torch.softmax(torch.as_tensor(logits,dtype=torch.float32).detach(),-1).cpu().numpy()

def cls_summary(pclass,labels):
    return classification_summary(pclass,labels)

def flatten_aligned_batch(pclass, labels, *, queries=100, classes=19):
    """Flatten window-major/query-major classifier batches after strict alignment checks."""
    p=np.asarray(pclass); y=np.asarray(labels)
    if p.ndim not in (2,3) or p.shape[-1] != classes:
        raise ValueError(f'pclass must be [N,{classes}] or [B,Q,{classes}] before flattening; got {p.shape}')
    if p.ndim == 3 and p.shape[1] != queries:
        raise ValueError(f'batched pclass query dimension must be {queries}; got {p.shape}')
    if p.shape[:-1] != y.shape:
        raise ValueError(f'pclass leading dimensions {p.shape[:-1]} do not match labels {y.shape}')
    flat_p=p.reshape((-1,classes),order='C')
    flat_y=y.reshape(-1,order='C')
    if flat_p.ndim != 2 or flat_p.shape[1] != classes or flat_y.ndim != 1 or len(flat_p) != len(flat_y):
        raise ValueError(f'flattened p/y contract failed: {flat_p.shape}/{flat_y.shape}')
    return flat_p,flat_y

def summarize_train_readout(train_data, head, model=None, batch_size=16):
    """Use the same ordered batch aggregation for formal evaluation and preflight."""
    p_batches=[];y_batches=[];batch_records=[]
    for start in range(0,len(train_data),batch_size):
        rows=train_data[start:start+batch_size]
        raw_y=np.stack([r['labels'] for r in rows])
        if head=='H0':
            raw_p=np.stack([r['pclass'] for r in rows])
        else:
            if model is None: raise ValueError(f'{head} requires a loaded readout')
            q=torch.from_numpy(np.stack([r['q'] for r in rows])).float()
            z=torch.from_numpy(np.stack([r['z'] for r in rows])).float()
            with torch.no_grad(): raw_p=probs(model(q,z))
        expected=(len(rows),100,19)
        if raw_p.shape != expected:
            raise ValueError(f'{head} batch pclass must be {expected}; got {raw_p.shape}')
        if raw_y.shape != expected[:-1]:
            raise ValueError(f'{head} batch labels must be {expected[:-1]}; got {raw_y.shape}')
        p,y=flatten_aligned_batch(raw_p,raw_y)
        p_batches.append(p);y_batches.append(y)
        batch_records.append({'start_window_index':start,'window_count':len(rows),'p_shape':list(raw_p.shape),
            'y_shape':list(raw_y.shape),'flat_p_shape':list(p.shape),'flat_y_shape':list(y.shape)})
    p=np.concatenate(p_batches,axis=0);y=np.concatenate(y_batches,axis=0)
    if p.shape!=(len(train_data)*100,19) or y.shape!=(len(train_data)*100,):
        raise ValueError(f'{head} final train aggregation mismatch: {p.shape}/{y.shape}')
    return p,y,batch_records

def ranking_metrics(binary,score):
    y=np.asarray(binary,dtype=bool);s=np.asarray(score,dtype=np.float64)
    pos=int(y.sum());neg=int((~y).sum())
    if not pos or not neg:return None,None
    order=np.argsort(s,kind='mergesort'); sorted_s=s[order]; sorted_y=y[order]
    # Average ranks for tied scores give the standard Mann-Whitney AUROC.
    ranks=np.empty(len(s),dtype=np.float64);i=0
    while i<len(s):
        j=i+1
        while j<len(s) and sorted_s[j]==sorted_s[i]:j+=1
        ranks[order[i:j]]=(i+1+j)/2.0;i=j
    auc=(ranks[y].sum()-pos*(pos+1)/2)/(pos*neg)
    # Average precision integrates precision over recall jumps at each unique threshold.
    order=np.argsort(-s,kind='mergesort');ss=s[order];yy=y[order];tp=fp=0;ap=0.0;i=0
    while i<len(ss):
        j=i+1
        while j<len(ss) and ss[j]==ss[i]:j+=1
        group_pos=int(yy[i:j].sum());tp+=group_pos;fp+=(j-i-group_pos)
        if group_pos:ap+=(group_pos/pos)*(tp/(tp+fp))
        i=j
    return float(auc),float(ap)

def fixed_threshold(y,pclass):
    y=np.asarray(y); known=(y>=0); positive=(y<18)&known; negative=(y==18)&known
    if not positive.any() or not negative.any():return None
    s=1-np.asarray(pclass)[:,18]; candidates=np.unique(s[known]); valid=[]
    for t in candidates:
        recall=float((s[positive]>=t).mean())
        if recall>=.90:valid.append((float(t),recall))
    return max(valid,key=lambda x:x[0]) if valid else None

def make_out(region,alpha,pclass,logits):
    region=torch.as_tensor(region,dtype=torch.float32);alpha=torch.as_tensor(alpha,dtype=torch.float32)
    p=torch.as_tensor(pclass,dtype=torch.float32); logits=torch.as_tensor(logits,dtype=torch.float32)
    scores=region.new_zeros((1,region.shape[0],20,256,256))
    scores[:,:,0:2]=region[None,:,100:102]
    scores[:,:,2:20]=torch.einsum('bvqhw,bqc->bvchw',region[None,:,:100],p[None,:,:18])
    scores=scores/(scores.sum(2,keepdim=True)+1e-6)
    return {'region_mass':region[None], 'alpha':alpha[None,:,None], 'p_class':p[None],
        'semantic_scores':scores,'states':[{'thing_logits19':logits[None]}]}

def metric_batch(data,ids):
    return {'semantic_label_all':torch.as_tensor(data['sem'][None],dtype=torch.long),
            'instance_label_all':torch.as_tensor(data['ins'][None],dtype=torch.long),
            'frame_ids':torch.as_tensor(data['frame_ids'][None],dtype=torch.long)}

def flatten_map_payload(payload):
    for branch in ('pred','target'):
        for key,value in list(payload[branch].items()):
            if torch.is_tensor(value):
                if key=='masks' and value.ndim==4:value=value.reshape(value.shape[0],value.shape[1]*value.shape[2],value.shape[3])
                payload[branch][key]=value.cpu()
    return payload

def compare_replay(gpu_root,cpu_root,expected_names,cache_root,feature_root,windows,cohort):
    failures=[];checked=0;candidate_rows=[]
    gpu_pairs={p.name:p for p in gpu_root.iterdir() if p.is_dir() and p.name in expected_names}
    cpu_pairs={p.name:p for p in cpu_root.iterdir() if p.is_dir()}
    if set(gpu_pairs)!=set(cpu_pairs):raise RuntimeError('H0 GPU/CPU window pair set differs')
    from PIL import Image
    for name,src in gpu_pairs.items():
        dst=cpu_pairs[name]
        wi=next(i for i,w in enumerate(windows) if f'{w["scene"]}_context{"_".join(map(str,w["context"]))}'==name)
        prefix=f'{cohort}_{wi:04d}_{windows[wi]["scene"]}_c{"_".join(map(str,windows[wi]["context"]))}'
        with np.load(cache_root/f'{prefix}.npz') as d:region=d['region'].copy();alpha=d['alpha'].copy();frame_ids=d['frame_ids'].copy()
        with np.load(feature_root/f'{prefix}.npz') as d:pclass=d['pclass'].copy()
        clsprob=pclass[:,:18].max(-1);eligible=(pclass.argmax(-1)!=18)&(clsprob>=.05)
        packed_maps={}
        for split in ('context','target'):
            a=src/f'{split}_seg_pred';b=dst/f'{split}_seg_pred'
            pa=json.loads((a/'pred.json').read_text());pb=json.loads((b/'pred.json').read_text())
            if len(pa)!=len(pb):failures.append({'pair':name,'scope':split,'field':'pred_count'})
            if [x['id'] for x in pa]!=[x['id'] for x in pb] or [x['label_id'] for x in pa]!=[x['label_id'] for x in pb]:
                failures.append({'pair':name,'scope':split,'field':'pred_ids_labels'})
            for x,y in zip(pa,pb):
                if not np.isclose(x['score'],y['score'],atol=1e-6,rtol=1e-6):failures.append({'pair':name,'scope':split,'field':'score','id':x['id']})
            for path in sorted(a.glob('*.png')):
                other=b/path.name
                gpu_rgb=np.asarray(Image.open(path).convert('RGB'),dtype=np.int64)
                if not other.exists():
                    failures.append({'pair':name,'scope':split,'field':'packed_png','file':path.name});continue
                cpu_rgb=np.asarray(Image.open(other).convert('RGB'),dtype=np.int64)
                if not np.array_equal(gpu_rgb,cpu_rgb):
                    failures.append({'pair':name,'scope':split,'field':'packed_png','file':path.name})
                code=gpu_rgb[:,:,0]+256*gpu_rgb[:,:,1]+65536*gpu_rgb[:,:,2]
                packed_maps[int(path.stem.split('pred')[-1])]=(code%1000).astype(np.int32)
            ga=src/f'{split}_seg_gt';gb=dst/f'{split}_seg_gt'
            for path in sorted(ga.glob('*.png')):
                other=gb/path.name
                if not other.exists() or not np.array_equal(np.asarray(Image.open(path)),np.asarray(Image.open(other))):
                    failures.append({'pair':name,'scope':split,'field':'gt_png','file':path.name})
            checked+=1
        # Exact candidate eligibility/area and packed winner/removal audit.
        for v,fid in enumerate(frame_ids):
            inst=packed_maps.get(int(fid))
            if inst is None:
                failures.append({'pair':name,'scope':'candidate','field':'missing_frame_map','frame_id':int(fid)});continue
            for q in range(100):
                mask=(region[v,q]>=.5)&(alpha[v]>.05);raw_area=int(mask.sum())
                won=int((inst==q+1).sum());is_eligible=bool(eligible[q])
                if not is_eligible:reason='ineligible'
                elif raw_area==0:reason='empty_raw_mask'
                elif won/raw_area<.5:reason='lost_or_removed_by_panoptic_competition'
                else:reason='retained'
                candidate_rows.append((name,int(fid),q,is_eligible,raw_area,won,reason))
    if len(candidate_rows)!=len(gpu_pairs)*6*100:
        failures.append({'field':'candidate_audit_row_count','actual':len(candidate_rows),'expected':len(gpu_pairs)*6*100})
    candidate_hash=hashlib.sha256(json.dumps(candidate_rows,separators=(',',':')).encode()).hexdigest()
    return {'passed':not failures,'windows':len(gpu_pairs),'scopes_checked':checked,'candidate_query_views_checked':len(candidate_rows),
        'candidate_eligibility_raw_area_won_area_removal_reason_exact':not any(x.get('scope')=='candidate' or x.get('field')=='candidate_audit_row_count' for x in failures),
        'candidate_audit_sha256':candidate_hash,'failures':failures}

def cached_window(root,cohort,i,w,head):
    prefix=f'{cohort}_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
    base=root/'r3d' if head=='R3D' else root
    with np.load(base/'features'/f'{prefix}.npz') as f:feat={k:f[k].copy() for k in f.files}
    with np.load(base/'cache/dev_test'/f'{prefix}.npz' if head!='R3D' else base/'cache'/f'{prefix}.npz') as d:data={k:d[k].copy() for k in d.files}
    return prefix,data,feat

def labels_for_scope(root,cohort,i,w,scope,head,data):
    if head=='R3D':
        prefix=f'{cohort}_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
        with np.load(root/'r3d'/'cache'/f'{prefix}_{scope}_iou.npz') as z:
            return labels_from_context(z['iou'],z['gt_classes'].tolist())['labels']
    if head!='R3D' and scope=='context':
        prefix=f'{cohort}_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
        with np.load(root/'cache/dev_test'/f'{prefix}_iou.npz') as z:return labels_from_context(z['iou'],z['gt_classes'].tolist())['labels']
    ids=[0,1] if scope=='context' else [j for j,f in enumerate(data['frame_ids']) if int(f) in set(map(int,w['novel']))]
    _,matrix,_,_=raw_iou(torch.from_numpy(data['region'][ids]),torch.from_numpy(data['alpha'][ids]),torch.from_numpy(data['sem'][ids]),torch.from_numpy(data['ins'][ids]))
    _,rows=gt_rows(torch.from_numpy(data['sem'][ids]),torch.from_numpy(data['ins'][ids]))
    return labels_from_context(matrix,[r[1] for r in rows])['labels']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,default=ATTEMPT);args=ap.parse_args();root=args.attempt
    if (root/'extraction_complete.json').exists() is False or json.loads((root/'extraction_complete.json').read_text()).get('complete') is not True:raise RuntimeError('missing extraction receipt')
    cache_receipt=verify_gc_cache_manifest(root);r3d_cache_receipt=verify_r3d_cache_manifest(root)
    dump(root/'cached_eval_cache_sha_verification.json',{'GC001':cache_receipt,'R3D':r3d_cache_receipt})
    cohorts=json.loads((root/'cohort_manifest.json').read_text()); test=cohorts['test'];dev=cohorts['dev']
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    readouts=[('H0',None)]+[(h,s) for h in ('H1','H2','H3') for s in (20261,20262,20263)]+[('R3D',None)]
    funnel=[]; clsrows=[]; feature=[]; official=[]; pergt=[]; perquery=[]; perwindow=[]; oracle=[];scope_labels=[];r3d_scope_labels=[]
    fixed_hungarian={}
    with (root/'labels/original_context_hungarian.csv').open(newline='') as f:
        for r in csv.DictReader(f):fixed_hungarian.setdefault(r['window_id'],{})[int(r['gt_id'])]=int(r['query_id'])
    parity_roots=[]; thresholds={}
    # Training is aggregated for classifier diagnostics only; no full train
    # q/z or soft-mask prediction table is exported.
    train_data=load_split(root,'train')
    for head,seed in readouts:
        if head=='R3D':continue
        name=head if head=='H0' else f'{head}_seed_{seed}'
        if head=='H0':
            train_p,train_y,batch_shapes=summarize_train_readout(train_data,head)
        else:
            cp=torch.load(root/'heads'/head/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False)
            model=Readout(head).cpu().float();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
            train_p,train_y,batch_shapes=summarize_train_readout(train_data,head,model)
        sm=cls_summary(train_p,train_y)
        feature.append({'cohort':'train','scope':'context','head':head,'seed':seed,'threshold':None,
            **{k:v for k,v in sm.items() if k not in ('confusion_18x19','per_class')},
            'confusion_18x19':sm['confusion_18x19'],'per_class':sm['per_class'],
            'aggregation_batch_shapes':batch_shapes})
    del train_data
    # A0/A1 training context is computed from the already saved complete GT x
    # query IoU matrices; no train region images or additional model forward.
    for i,w in enumerate(cohorts['train']):
        with np.load(root/'cache/train_iou'/f'{i:04d}.npz') as d:
            im=d['iou'];m05=diagnostic_max_cardinality(im,.5);m075=diagnostic_max_cardinality(im,.75)
            oracle.append({'cohort':'train','scope':'context','window_id':f'train_{i:04d}_{w["scene"]}',
                'scene':w['scene'],'gt_count':int(im.shape[0]),'a0_ge_05':int((im.max(1)>=.5).sum()) if len(im) else 0,
                'a0_ge_075':int((im.max(1)>=.75).sum()) if len(im) else 0,'a1_ge_05':m05['objective'][0],
                'a1_ge_075':m075['objective'][0],'a1_matches':json.dumps(m05['matches'])})
    for cohort,windows in (('dev',dev),('test',test)):
      for head,seed in readouts:
        name=head if head in ('H0','R3D') else f'{head}_seed_{seed}'
        predroot=root/'predictions'/cohort/name/'official'; predroot.mkdir(parents=True,exist_ok=True)
        evaluator_root=predroot
        gpu_root=root/'r3d'/'reference_gpu' if head=='R3D' else root/'reference_gpu'
        if head in ('H1','H2','H3'):
            cp=torch.load(root/'heads'/head/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False)
            model=Readout(head).cpu().float();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        candidate_metric={scope:MeanAveragePrecision(iou_type='segm',sync_on_compute=False) for scope in ('context','true-novel')}
        cls_scope={'context':{'p':[],'y':[]},'true-novel':{'p':[],'y':[]}}
        threshold=None; raw_rows=[]
        if cohort=='dev':
            for i,w in enumerate(windows):
                prefix,data,features=cached_window(root,cohort,i,w,head);p0=features['pclass'];q=features['q'];z=features['z']
                if head in ('H0','R3D'):pclass=p0
                else:
                    with torch.no_grad():pclass=probs(model(torch.from_numpy(q),torch.from_numpy(z)))
                labels=labels_for_scope(root,cohort,i,w,'context',head,data)
                cls_scope['context']['p'].append(pclass);cls_scope['context']['y'].append(labels)
            ps=np.concatenate(cls_scope['context']['p']);ys=np.concatenate(cls_scope['context']['y']);threshold=fixed_threshold(ys,ps)
            thresholds[name]=threshold
            cls_scope['context']={'p':[],'y':[]}
        else:
            threshold=thresholds.get(name)
        for i,w in enumerate(windows):
            prefix,data,features=cached_window(root,cohort,i,w,head)
            logits0=features['logits'];p0=features['pclass'];q=features['q'];z=features['z']
            pclass=p0 if head in ('H0','R3D') else probs(model(torch.from_numpy(q),torch.from_numpy(z)))
            out=make_out(data['region'],data['alpha'],pclass,logits0);batch=metric_batch(data,list(range(len(data['frame_ids']))))
            write_official_pair(out,batch,w,evaluator_root,target_frames='novel')
            if head in ('H0','R3D'): parity_roots.append((cohort,gpu_root,evaluator_root,head))
            for scope,ids in (('context',[0,1]),('true-novel',[j for j,fid in enumerate(data['frame_ids']) if int(fid) in set(map(int,w['novel']))])):
                stat=_candidate_stats(out,batch,ids)
                payload=flatten_map_payload(stat.pop('_map_payload'))
                candidate_metric[scope].update([payload['pred']],[payload['target']])
                if head!='R3D':
                    for gr in stat['per_gt']:
                        qid=fixed_hungarian.get(prefix,{}).get(int(gr['instance_id'])) if scope=='context' else None
                        if head=='H0' and qid is not None and gr.get('matched_query') not in (None,qid):raise RuntimeError(f'H0 fixed context Hungarian query mismatch for {prefix}/{gr["instance_id"]}')
                        gr['matched_query']=qid
                        if qid is None:
                            gr.update({'matched_class':None,'matched_class_correct':None,'matched_conditional_class':None,'matched_conditional_class_correct':None})
                        else:
                            pred19=int(np.argmax(pclass[qid]));pred18=int(np.argmax(pclass[qid,:18]));truth=int(gr['class'])-2
                            gr.update({'matched_class':pred19+2 if pred19<18 else None,'matched_class_correct':bool(pred19==truth),
                                'matched_conditional_class':pred18+2,'matched_conditional_class_correct':bool(pred18==truth)})
                # Fixed mask-set A0/A1 reference.
                scope_gt,sub_iou,_,_=raw_iou(torch.from_numpy(data['region'][ids]),torch.from_numpy(data['alpha'][ids]),
                    torch.from_numpy(data['sem'][ids]),torch.from_numpy(data['ins'][ids]))
                a1=diagnostic_max_cardinality(sub_iou,.5);a175=diagnostic_max_cardinality(sub_iou,.75)
                if head=='H0':
                    oracle.append({'cohort':cohort,'scope':scope,'window_id':prefix,'scene':w['scene'],'gt_count':int(sub_iou.shape[0]),
                        'a0_ge_05':int((sub_iou.max(axis=1)>=.5).sum()) if len(sub_iou) else 0,
                        'a0_ge_075':int((sub_iou.max(axis=1)>=.75).sum()) if len(sub_iou) else 0,
                        'a1_ge_05':a1['objective'][0],'a1_ge_075':a175['objective'][0],
                        'a1_matches':json.dumps([{'gt_id':scope_gt[g][0],'query_id':q,'iou':v} for g,q,v in a1['matches']])})
                yy=labels_for_scope(root,cohort,i,w,scope,head,data)
                if head in ('H0','R3D'):
                    dst=scope_labels if head=='H0' else r3d_scope_labels
                    for qid,label in enumerate(yy):dst.append({'split':cohort,'scope':scope,'window_id':prefix,'scene':w['scene'],'query_id':qid,
                        'label':int(label),'role':'POSITIVE' if 0<=label<18 else 'NEGATIVE' if label==18 else 'AMBIGUOUS'})
                cls_scope[scope]['p'].append(pclass);cls_scope[scope]['y'].append(yy)
                csummary=cls_summary(pclass,yy)
                clsrows.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'window_id':prefix,
                    'scene':w['scene'],'joint19_accuracy':csummary['joint19_accuracy'],'conditional18_accuracy':csummary['conditional18_accuracy'],
                    'macro_f1_supported_classes':csummary['macro_f1_supported_classes'],'positive_count':csummary['positive_count'],
                    'negative_count':csummary['negative_count'],'ambiguous_count':csummary['ambiguous_count'],
                    'objectness_auroc':csummary['objectness_auroc'],'objectness_auprc':csummary['objectness_auprc']})
                funnel.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'window_id':prefix,'scene':w['scene'],
                    'gt_count':stat['gt_count'],'eligible_query_count':stat['candidate_count'],
                    'candidate_ca':stat['candidate_ca'],'candidate_cw':stat['candidate_cw'],
                    'eligible_coverage_ge_05':sum(float(r['best_eligible_candidate_iou'])>=.5 for r in stat['per_gt']),
                    'panoptic_ca':stat['panoptic_ca'],'panoptic_cw':stat['panoptic_cw'],
                    'raw_best_iou_ge_0_5_fraction':stat['raw_best_iou_ge_0_5_fraction'],
                    'panoptic_pq':stat['panoptic_pq'],
                    'semantic_miou':stat['semantic_miou']})
                for r in stat['per_gt']:pergt.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'window_id':prefix,'scene':w['scene'],**r})
                for r in stat['query_rows']:
                    qi=int(r['query_id']);r['pclass_json']=pclass[qi].tolist();r['h0_pclass_json']=p0[qi].tolist()
                    r['joint_class_0_18']=int(np.argmax(pclass[qi]))
                    r['conditional_thing_class_internal']=int(np.argmax(pclass[qi,:18])+2)
                    r['diagnostic_label']=int(yy[qi]);r['diagnostic_role']='POSITIVE' if 0<=yy[qi]<18 else 'NEGATIVE' if yy[qi]==18 else 'AMBIGUOUS'
                    perquery.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'window_id':prefix,'scene':w['scene'],**r})
                perwindow.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'window_id':prefix,'scene':w['scene'],
                    'gt_count':stat['gt_count'],'candidate_count':stat['candidate_count'],'packed_ca':stat['panoptic_ca'],'packed_cw':stat['panoptic_cw'],
                    'pq':stat['panoptic_pq'],'miou':stat['panoptic_semantic_miou'],
                    'classification_confusion_18x19':csummary['confusion_18x19'],
                    'conditional_confusion_18x18':csummary['conditional_confusion_18x18'],
                    'joint_correct':int(((np.asarray(pclass).argmax(-1)==np.asarray(yy))&(np.asarray(yy)>=0)&(np.asarray(yy)<18)).sum()),
                    'joint_total':csummary['positive_count'],
                    'conditional_correct':int(((np.asarray(pclass)[:,:18].argmax(-1)==np.asarray(yy))&(np.asarray(yy)<18)).sum()),
                    'conditional_total':csummary['positive_count']})
                hmatched=[r for r in stat['per_gt'] if r.get('matched_query') is not None]
                hcond=[r for r in hmatched if r.get('matched_conditional_class') is not None]
                perwindow[-1].update({'h0_context_hungarian_joint_correct':sum(bool(r.get('matched_class_correct')) for r in hmatched),
                    'h0_context_hungarian_joint_total':len(hmatched),
                    'h0_context_hungarian_conditional_correct':sum(bool(r.get('matched_conditional_class_correct')) for r in hcond),
                    'h0_context_hungarian_conditional_total':len(hcond)})
                if cohort=='test' and scope=='context':
                    obj=1-np.asarray(pclass)[:,18];known=np.asarray(yy)>=0;positive=(np.asarray(yy)<18)&known;negative=np.asarray(yy)==18
                    th=threshold
                    if th is not None and positive.any() and negative.any():
                        selected=obj>=th[0];perwindow[-1].update({'objectness_threshold':th[0],
                            'objectness_tp':int((selected&positive).sum()),'objectness_positive':int(positive.sum()),
                            'objectness_fp':int((selected&negative).sum()),'objectness_negative':int(negative.sum())})
                    else:perwindow[-1].update({'objectness_threshold':None,'objectness_tp':None,'objectness_positive':int(positive.sum()),
                            'objectness_fp':None,'objectness_negative':int(negative.sum())})
            del out,batch
        official_dir=predroot
        result_path=root/'official'/cohort/f'{name}.json';result_path.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        subprocess.run([OFFICIAL_PY,str(INVOKE),'--eval-path',str(official_dir),'--output',str(result_path),'--device','cpu','--no-image-depth'],check=True,env=env)
        envelope=json.loads(result_path.read_text())
        if envelope.get('official_evaluator_used') is not True or envelope.get('siu3r_commit')!='8ea80166be76854f938e90521f1a5b688b755c87':raise RuntimeError('official evaluator provenance mismatch')
        metric_res=envelope['result'];normalized=normalize_official_result(metric_res)
        official.append({'cohort':cohort,'head':head,'seed':seed,'readout':name,'result':normalized,'raw_result_schema':sorted(metric_res)})
        for scope in ('context','true-novel'):
            met=candidate_metric[scope].compute();candidate_metric[scope].reset()
            # Local map is a distinct candidate metric; official values come from SIU3R.
            mkey='context' if scope=='context' else 'target'
            candidate_ap={'map':float(met['map']),'map_50':float(met['map_50'])}
            official[-1].setdefault('local_candidate_ap',{})[scope]=candidate_ap
            for fr in funnel:
                if fr['cohort']==cohort and fr['scope']==scope and fr['head']==head and fr['seed']==seed:fr['candidate_ap']=candidate_ap
        for scope in ('context','true-novel'):
            summary=cls_summary(np.concatenate(cls_scope[scope]['p']),np.concatenate(cls_scope[scope]['y']))
            threshold_detail=None
            if scope=='context' and cohort=='dev' and threshold is not None:
                threshold_detail={'value':threshold[0],'dev_positive_recall':threshold[1]}
            elif scope=='context' and cohort=='test':
                tw=[r for r in perwindow if r['cohort']=='test' and r['head']==head and r['seed']==seed and r['scope']=='context']
                poss=sum(r['objectness_positive'] or 0 for r in tw);negs=sum(r['objectness_negative'] or 0 for r in tw)
                tps=sum(r['objectness_tp'] or 0 for r in tw);fps=sum(r['objectness_fp'] or 0 for r in tw)
                threshold_detail={'value':threshold[0] if threshold else None,'test_positive_recall':tps/poss if poss else None,
                    'test_false_positive_count':fps if threshold else None,'test_false_positive_rate':fps/negs if negs and threshold else None,
                    'test_false_positive_per_window':fps/max(1,len(tw)) if threshold else None,
                    'positive_count':poss,'negative_count':negs}
            feature.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'threshold':threshold_detail,
                            **{k:v for k,v in summary.items() if k!='confusion_18x19' and k!='per_class'},
                            'confusion_18x19':summary['confusion_18x19'],'per_class':summary['per_class']})
        print(f'evaluated {cohort} {name}',flush=True)
        if cohort=='test' and head=='H0':
            tr=[r for r in funnel if r['cohort']=='test' and r['scope']=='true-novel' and r['head']=='H0']
            oo=[r for r in oracle if r['cohort']=='test' and r['scope']=='true-novel']
            actual={'test_windows':len(tr),'gt_count':sum(int(r['gt_count']) for r in tr),
                'raw_iou_ge_05':sum(int(r['a0_ge_05']) for r in oo),'raw_iou_ge_075':sum(int(r['a0_ge_075']) for r in oo),
                'candidate_ca_tp':sum(int(r['candidate_ca']['tp']) for r in tr),
                'candidate_cw_tp':sum(int(r['candidate_cw']['tp']) for r in tr),
                'packed_ca_tp':sum(int(r['panoptic_ca']['tp']) for r in tr),
                'packed_cw_tp':sum(int(r['panoptic_cw']['tp']) for r in tr)}
            expected={'test_windows':24,'gt_count':104,'raw_iou_ge_05':88,'raw_iou_ge_075':63,
                'candidate_ca_tp':68,'candidate_cw_tp':61,'packed_ca_tp':62,'packed_cw_tp':56}
            old=json.loads(Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_evaluation_retry01/official_results.json').read_text())['gc001']['val32_excluding_dev8_scenes']['true-novel']
            dump(root/'h0_against_previous_precheck.json',{'counts':actual,'expected':expected,'counts_exact_match':actual==expected,
                'current_official_point':normalized.get('true-novel'),'previous_official_point':old,
                'official_point_difference':{k:normalized.get('true-novel',{}).get(k)-old.get(k) for k in ('mAP','AP50','PQ','mIoU')
                    if normalized.get('true-novel',{}).get(k) is not None and old.get(k) is not None}})
            if actual!=expected:raise RuntimeError(f'prior GC001 fixed-count precheck failed: {actual}')
    parity=[]
    for cohort,gpu,cpu,head in parity_roots:
        # each cohort gets an isolated CPU output tree at predictions/<cohort>/H0/official
        windows=cohorts[cohort]
        expected={f'{w["scene"]}_context{"_".join(map(str,w["context"]))}' for w in windows}
        cache=root/'r3d'/'cache' if head=='R3D' else root/'cache/dev_test'
        features=root/'r3d'/'features' if head=='R3D' else root/'features'
        parity.append({'cohort':cohort,'readout':head,'result':compare_replay(gpu,cpu,expected,cache,features,windows,cohort)})
    if not all(x['result']['passed'] for x in parity):
        dump(root/'h0_cache_replay_parity.json',{'status':'INVALID','cohorts':parity});raise RuntimeError('H0 GPU->CPU packed replay parity failed')
    dump(root/'h0_cache_replay_parity.json',{'status':'PASS','cohorts':parity})
    for filename,data in [('feature_probe_metrics.csv',feature),('official_metrics.csv',official),('funnel_metrics.csv',funnel),('per_gt.csv',pergt),('per_query.csv',perquery),('per_window.csv',perwindow),('oracle_metrics.csv',oracle)]:rows_to_csv(root/filename,data)
    unique={}
    for r in scope_labels:
        key=(r['split'],r['scope'],r['window_id'],r['query_id'])
        if key in unique and unique[key]!=r:raise RuntimeError(f'fixed labels disagree across heads: {key}')
        unique[key]=r
    if len(unique)!=32*2*100:raise RuntimeError(f'scope labels must contain one row per window/query/scope, got {len(unique)}')
    rows_to_csv(root/'labels/dev_test_scope_labels.csv',list(unique.values()))
    rows_to_csv(root/'labels/r3d_dev_test_scope_labels.csv',r3d_scope_labels)
    dump(root/'objectness_thresholds.json',thresholds)
    forbidden=[m for m in sys.modules if m in ('scripts.object_locus_gc_sweep_runtime','scripts.object_locus_panoptic_v1_runtime')]
    if forbidden:raise RuntimeError(f'cached CPU evaluator imported a model runtime: {forbidden}')
    dump(root/'cached_eval_provenance.json',{'cached_feature_only':True,'model_checkpoint_loaded':False,
        'GC001_model_constructed':False,'R3D_model_constructed':False,'model_forward_calls':0,'distributed_initialized':False,
        'optimizer_or_backward_on_model':False,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_runtime_modules_imported':forbidden,'head_state_dicts_loaded':9,'readouts_evaluated':11})
    # one pre-fixed dev-derived objectness threshold per readout; caller applies on test
    dump(root/'eval_complete.json',{'status':'PASS','readouts':len(readouts),'cohorts':['dev','test'],'packed_parity':'PASS','official_results':len(official)})

if __name__=='__main__':main()
