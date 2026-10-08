#!/usr/bin/env python3
"""Replay all readouts from frozen CPU caches; export and score packed masks."""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.object_locus_frozen_probe_contract import labels_from_context,diagnostic_max_cardinality
from scripts.train_object_locus_frozen_probe import Readout,load_split
from scripts.export_object_locus_v3_set_official import write_official_pair
from scripts.eval_object_locus_v3_set import _candidate_stats

ATTEMPT=Path('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt00')
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

def probs(logits): return torch.softmax(torch.as_tensor(logits,dtype=torch.float32),-1).cpu().numpy()

def cls_summary(pclass,labels):
    y=np.asarray(labels,dtype=np.int64); p=np.asarray(pclass).argmax(-1); pos=y<18; valid=y>=0
    confusion=np.zeros((18,19),np.int64)
    for yy,pp in zip(y[pos],p[pos]): confusion[yy,pp]+=1
    supports=confusion.sum(1); active=np.where(supports>0)[0]
    per=[]
    for c in range(18):
        tp=int(confusion[c,c]); fp=int(confusion[:,c].sum()-tp); fn=int(confusion[c].sum()-tp)
        per.append({'class_internal':c+2,'support':int(supports[c]),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn) if 2*tp+fp+fn else 0.})
    conditional=np.asarray(pclass)[:,:18].argmax(-1)
    binary=np.isin(y,[18]); known=np.isin(y,[*range(18),18]); score=1-np.asarray(pclass)[:,18]
    auc,auprc=ranking_metrics(binary[known],score[known])
    return {'joint19_accuracy':float((p[valid]==y[valid]).mean()) if valid.any() else None,
        'conditional18_accuracy':float((conditional[pos]==y[pos]).mean()) if pos.any() else None,
        'macro_f1_supported_classes':float(np.mean([per[c]['f1'] for c in active])) if len(active) else None,
        'support_classes':(active+2).tolist(),'no_support_classes':(np.where(supports==0)[0]+2).tolist(),
        'confusion_18x19':confusion.tolist(),'per_class':per,'positive_count':int(pos.sum()),
        'negative_count':int((y==18).sum()),'ambiguous_count':int((y==-1).sum()),
        'objectness_auroc':None if auc is None else float(auc),'objectness_auprc':None if auprc is None else float(auprc)}

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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,default=ATTEMPT);args=ap.parse_args();root=args.attempt
    if (root/'extraction_complete.json').exists() is False or json.loads((root/'extraction_complete.json').read_text()).get('complete') is not True:raise RuntimeError('missing extraction receipt')
    cohorts=json.loads((root/'cohort_manifest.json').read_text()); test=cohorts['test'];dev=cohorts['dev']
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    readouts=[('H0',None)]+[(h,s) for h in ('H1','H2','H3') for s in (20261,20262,20263)]
    funnel=[]; clsrows=[]; feature=[]; official=[]; pergt=[]; perquery=[]; perwindow=[]; oracle=[];scope_labels=[]
    parity_roots=[]; thresholds={}
    # Training is aggregated for classifier diagnostics only; no full train
    # q/z or soft-mask prediction table is exported.
    train_data=load_split(root,'train')
    for head,seed in readouts:
        name=head if head=='H0' else f'{head}_seed_{seed}'
        if head=='H0':train_p=np.concatenate([r['pclass'] for r in train_data])
        else:
            cp=torch.load(root/'heads'/head/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False)
            model=Readout(head).cpu().float();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
            train_p=[]
            with torch.no_grad():
                for start in range(0,len(train_data),16):
                    q=torch.from_numpy(np.stack([r['q'] for r in train_data[start:start+16]])).float()
                    z=torch.from_numpy(np.stack([r['z'] for r in train_data[start:start+16]])).float()
                    train_p.append(probs(model(q,z)))
            train_p=np.concatenate(train_p)
        train_y=np.concatenate([r['labels'] for r in train_data])
        sm=cls_summary(train_p,train_y)
        feature.append({'cohort':'train','scope':'context','head':head,'seed':seed,'threshold':None,
            **{k:v for k,v in sm.items() if k not in ('confusion_18x19','per_class')},
            'confusion_18x19':sm['confusion_18x19'],'per_class':sm['per_class']})
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
        name=head if head=='H0' else f'{head}_seed_{seed}'
        predroot=root/'predictions'/cohort/name/'official'; predroot.mkdir(parents=True,exist_ok=True)
        evaluator_root=predroot
        if head!='H0':
            cp=torch.load(root/'heads'/head/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False)
            model=Readout(head).cpu().float();model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        candidate_metric={scope:MeanAveragePrecision(iou_type='segm',sync_on_compute=False) for scope in ('context','true-novel')}
        cls_scope={'context':{'p':[],'y':[]},'true-novel':{'p':[],'y':[]}}
        threshold=None; raw_rows=[]
        if cohort=='dev':
            for i,w in enumerate(windows):
                prefix=f'dev_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
                with np.load(root/'features'/f'{prefix}.npz') as f: logits0=f['logits'].copy();p0=f['pclass'].copy()
                with np.load(root/'cache/dev_test'/f'{prefix}.npz') as d: data={k:d[k].copy() for k in d.files}
                if head=='H0': pclass=p0
                else:
                    with np.load(root/'features'/f'{prefix}.npz') as f: q,z=f['q'].copy(),f['z'].copy()
                    with torch.no_grad():pclass=probs(model(torch.from_numpy(q),torch.from_numpy(z)))
                # Recompute fixed context labels and choose this head's threshold only on dev context.
                # labels require GT classes from the exact IoU archive.
                with np.load(root/'cache/dev_test'/f'{prefix}_iou.npz') as x: labels=labels_from_context(x['iou'],x['gt_classes'].tolist())['labels']
                cls_scope['context']['p'].append(pclass);cls_scope['context']['y'].append(labels)
            ps=np.concatenate(cls_scope['context']['p']);ys=np.concatenate(cls_scope['context']['y']);threshold=fixed_threshold(ys,ps)
            thresholds[name]=threshold
            cls_scope['context']={'p':[],'y':[]}
        else:
            threshold=thresholds.get(name)
        for i,w in enumerate(windows):
            prefix=f'{cohort}_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
            with np.load(root/'features'/f'{prefix}.npz') as f: logits0=f['logits'].copy();p0=f['pclass'].copy();q=f['q'].copy();z=f['z'].copy()
            with np.load(root/'cache/dev_test'/f'{prefix}.npz') as d: data={k:d[k].copy() for k in d.files}
            pclass=p0 if head=='H0' else probs(model(torch.from_numpy(q),torch.from_numpy(z)))
            out=make_out(data['region'],data['alpha'],pclass,logits0);batch=metric_batch(data,list(range(len(data['frame_ids']))))
            write_official_pair(out,batch,w,evaluator_root,target_frames='novel')
            if head=='H0': parity_roots.append((cohort,root/'reference_gpu',evaluator_root))
            for scope,ids in (('context',[0,1]),('true-novel',[j for j,fid in enumerate(data['frame_ids']) if int(fid) in set(map(int,w['novel']))])):
                stat=_candidate_stats(out,batch,ids)
                payload=flatten_map_payload(stat.pop('_map_payload'))
                candidate_metric[scope].update([payload['pred']],[payload['target']])
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
                with np.load(root/'cache/dev_test'/f'{prefix}_iou.npz') as x: yctx=labels_from_context(x['iou'],x['gt_classes'].tolist())['labels']
                if scope=='context': yy=yctx
                else:
                    from scripts.extract_object_locus_frozen_probe import raw_iou
                    niou=raw_iou(torch.from_numpy(data['region'][ids]),torch.from_numpy(data['alpha'][ids]),torch.from_numpy(data['sem'][ids]),torch.from_numpy(data['ins'][ids]))[1]
                    with np.load(root/'cache/dev_test'/f'{prefix}_iou.npz') as x:
                        sem_gt,ins_gt=torch.from_numpy(data['sem'][ids]),torch.from_numpy(data['ins'][ids])
                        _,novel_rows=__import__('scripts.extract_object_locus_frozen_probe',fromlist=['gt_rows']).gt_rows(sem_gt,ins_gt)
                        yy=labels_from_context(niou,[r[1] for r in novel_rows])['labels']
                for qid,label in enumerate(yy):
                    scope_labels.append({'split':cohort,'scope':scope,'window_id':prefix,'scene':w['scene'],'query_id':qid,
                        'label':int(label),'role':'POSITIVE' if label<18 else 'NEGATIVE' if label==18 else 'AMBIGUOUS'})
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
                    r['diagnostic_label']=int(yy[qi]);r['diagnostic_role']='POSITIVE' if yy[qi]<18 else 'NEGATIVE' if yy[qi]==18 else 'AMBIGUOUS'
                    perquery.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'window_id':prefix,'scene':w['scene'],**r})
                perwindow.append({'cohort':cohort,'scope':scope,'head':head,'seed':seed,'window_id':prefix,'scene':w['scene'],
                    'gt_count':stat['gt_count'],'candidate_count':stat['candidate_count'],'packed_ca':stat['panoptic_ca'],'packed_cw':stat['panoptic_cw'],
                    'pq':stat['panoptic_pq'],'miou':stat['panoptic_semantic_miou'],
                    'classification_confusion_18x19':csummary['confusion_18x19'],
                    'joint_correct':int(((np.asarray(pclass).argmax(-1)==np.asarray(yy))&(np.asarray(yy)>=0)).sum()),
                    'joint_total':int((np.asarray(yy)>=0).sum()),
                    'conditional_correct':int(((np.asarray(pclass)[:,:18].argmax(-1)==np.asarray(yy))&(np.asarray(yy)<18)).sum()),
                    'conditional_total':csummary['positive_count']})
                hmatched=[r for r in stat['per_gt'] if r.get('matched_query') is not None]
                hcond=[r for r in hmatched if r.get('matched_class') is not None]
                perwindow[-1].update({'h0_context_hungarian_joint_correct':sum(bool(r.get('matched_class_correct')) for r in hmatched),
                    'h0_context_hungarian_joint_total':len(hmatched),
                    'h0_context_hungarian_conditional_correct':sum(bool(r.get('matched_class_correct')) for r in hcond),
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
        metric_res=envelope['result'];official.append({'cohort':cohort,'head':head,'seed':seed,'readout':name,'result':metric_res})
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
                'current_official_point':metric_res.get('target'),'previous_official_point':old,
                'official_point_difference':{k:metric_res.get('target',{}).get(k)-old.get(k) for k in ('mAP','AP50','PQ','mIoU')
                    if metric_res.get('target',{}).get(k) is not None and old.get(k) is not None}})
            if actual!=expected:raise RuntimeError(f'prior GC001 fixed-count precheck failed: {actual}')
    parity=[]
    for cohort,gpu,cpu in parity_roots:
        # each cohort gets an isolated CPU output tree at predictions/<cohort>/H0/official
        windows=cohorts[cohort]
        expected={f'{w["scene"]}_context{"_".join(map(str,w["context"]))}' for w in windows}
        parity.append({'cohort':cohort,'result':compare_replay(gpu,cpu,expected,root/'cache/dev_test',root/'features',windows,cohort)})
    if not all(x['result']['passed'] for x in parity):
        dump(root/'h0_cache_replay_parity.json',{'status':'INVALID','cohorts':parity});raise RuntimeError('H0 GPU->CPU packed replay parity failed')
    dump(root/'h0_cache_replay_parity.json',{'status':'PASS','cohorts':parity})
    for filename,data in [('feature_probe_metrics.csv',feature),('official_metrics.csv',official),('funnel_metrics.csv',funnel),('per_gt.csv',pergt),('per_query.csv',perquery),('per_window.csv',perwindow),('oracle_metrics.csv',oracle)]:rows_to_csv(root/filename,data)
    rows_to_csv(root/'labels/dev_test_scope_labels.csv',scope_labels)
    dump(root/'objectness_thresholds.json',thresholds)
    forbidden=[m for m in sys.modules if m in ('scripts.object_locus_gc_sweep_runtime','scripts.object_locus_panoptic_v1_runtime')]
    if forbidden:raise RuntimeError(f'cached CPU evaluator imported a model runtime: {forbidden}')
    dump(root/'cached_eval_provenance.json',{'cached_feature_only':True,'model_checkpoint_loaded':False,
        'GC001_model_constructed':False,'model_forward_calls':0,'distributed_initialized':False,
        'optimizer_or_backward_on_model':False,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_runtime_modules_imported':forbidden,'head_state_dicts_loaded':9})
    # one pre-fixed dev-derived objectness threshold per readout; caller applies on test
    dump(root/'eval_complete.json',{'status':'PASS','readouts':len(readouts),'cohorts':['dev','test'],'packed_parity':'PASS','official_results':len(official)})

if __name__=='__main__':main()
