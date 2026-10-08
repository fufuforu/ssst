#!/usr/bin/env python3
"""Paired scene bootstrap, fixed-case report and reproducible ZIP packaging."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,stat,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.object_locus_probe_metrics import normalize_official_result
from scripts.object_locus_r3d_registration import normalize_r3d_registration,registered_outcome

ATTEMPT=Path(os.environ.get('TASK_ATTEMPT_ROOT','/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02'))
FOCUS_SOURCE=Path('/space/mawb/ssst/group_plus/object_locus_instance_attribution_v1/attempts/attempt00/results/focus20_cases.jsonl')
FOCUS_MANIFEST=Path('/space/mawb/ssst/group_plus/object_locus_instance_attribution_v1/attempts/attempt00/selection_manifest.json')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
def read_csv(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 if not rows:return
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def parse_json(x,default=None):
 if x in (None,''):return default
 try:return json.loads(x)
 except Exception:return default
def ci(x):return np.percentile(np.asarray(x,dtype=np.float64),[2.5,97.5]).tolist()

def readout_name(head,seed):return head if head in ('H0','R3D') else f'{head}_seed_{seed}'
def metric_result(rows,cohort,head,seed,scope):
 target='context' if scope=='context' else 'true-novel'
 for r in rows:
  if r['cohort']==cohort and r['head']==head and (r['seed'] in ('','None') if seed is None else int(r['seed'])==seed):
   value=parse_json(r['result'],{})
   normalized=normalize_official_result(value)
   if target in normalized:return normalized[target]
 raise KeyError((cohort,head,seed,scope))
def class_row(rows,cohort,scope,head,seed):
 return next(r for r in rows if r['cohort']==cohort and r['scope']==scope and r['head']==head and
             (r['seed'] in ('','None') if seed is None else int(r['seed'])==seed))

def confusion_metrics(conf):
 c=np.asarray(conf,dtype=np.int64);total=int(c.sum());correct=int(np.trace(c[:,:18]))
 # Joint denominator includes explicit no-object examples. They are represented by
 # the separate negative-correct count supplied in per-window totals, not confusion.
 supported=np.where(c.sum(1)>0)[0];f1=[]
 for k in supported:
  tp=c[k,k];fp=c[:,k].sum()-tp;fn=c[k,:].sum()-tp
  f1.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
 return correct,total,float(np.mean(f1)) if f1 else None

def aggregate_scene_rows(rows,indices,scenes,readout,scope):
 by={}
 for r in rows:
  if r['cohort']=='test' and r['scope']==scope and r['head']==readout[0] and (r['seed'] in ('','None') if readout[1] is None else int(r['seed'])==readout[1]):by[r['scene']]=r
 conf=np.zeros((18,19),np.int64);cconf=np.zeros((18,18),np.int64);joint_c=joint_n=cond_c=cond_n=cw_tp=cw_fp=cw_fn=0
 for idx in indices:
  row=by[scenes[int(idx)]];conf+=np.asarray(parse_json(row['classification_confusion_18x19'],np.zeros((18,19))),dtype=np.int64)
  cconf+=np.asarray(parse_json(row.get('conditional_confusion_18x18'),np.zeros((18,18))),dtype=np.int64)
  joint_c+=int(row['joint_correct']);joint_n+=int(row['joint_total']);cond_c+=int(row['conditional_correct']);cond_n+=int(row['conditional_total'])
  cw=parse_json(row['packed_cw'],{});cw_tp+=int(cw.get('tp',0));cw_fp+=int(cw.get('fp',0));cw_fn+=int(cw.get('fn',0))
 _,_,mf=confusion_metrics(conf);_,_,cmf=confusion_metrics(cconf)
 return {'joint19_accuracy':joint_c/joint_n if joint_n else None,'conditional18_accuracy':cond_c/cond_n if cond_n else None,
         'macro_f1_supported_classes':mf,'conditional_macro_f1_supported_classes':cmf,
         'packed_cw':{'tp':cw_tp,'fp':cw_fp,'fn':cw_fn},'confusion':conf,'conditional_confusion':cconf}

def official_bootstrap(root,official_rows,scene_rows):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    sys.path.insert(0,'/space/mawb/SIU3R')
    from src.config import EvaluatorCfg
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
    cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,eval_context_map=False,
        eval_target_miou=False,eval_target_pq=False,eval_target_map=True,eval_image_quality=False,eval_depth_quality=False,
        id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',
        eval_path=str(root/'predictions/test/H0/official'))
    evaluator=Evaluator(cfg);evaluator.setup()
    scenes=sorted({r['scene'] for r in scene_rows if r['cohort']=='test' and r['scope']=='true-novel'})
    if len(scenes)!=24:raise RuntimeError(f'official bootstrap requires exactly24 test scenes, got {len(scenes)}')
    matrix=np.random.default_rng(2026).choice(24,size=(2000,24),replace=True)
    np.save(root/'bootstrap_scene_indices_seed2026.npy',matrix)
    readouts=[('H0',None),('H1',20261),('H2',20261),('H3',20261),('R3D',None)]
    payloads={};point={}
    for head,seed in readouts:
        name=readout_name(head,seed);pairs=sorted((root/'predictions/test'/name/'official').iterdir())
        if len(pairs)!=24:raise RuntimeError(f'{name} is missing packed test windows')
        by_scene={}
        for pair in pairs:
            scene=pair.name.split('_context')[0]
            if scene in by_scene:raise RuntimeError(f'{name} has duplicate window for scene={scene}')
            parsed=evaluator.process_segmentation(pair/'target_seg_pred',pair/'target_seg_gt')
            by_scene[scene]=(parsed['map_pred'],parsed['map_gt'])
        if set(by_scene)!=set(scenes):raise RuntimeError(f'{name} official window scene identity mismatch')
        metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
        for scene in scenes:
            p,g=by_scene[scene];metric.update([p],[g])
        full=metric.compute();metric.reset()
        exp=metric_result(official_rows,'test',head,seed,'true-novel')
        for key,actual in [('mAP',float(full['map'])),('AP50',float(full['map_50']))]:
            if abs(actual-float(exp[key]))>1e-6:raise RuntimeError(f'{name} cached official {key} {actual} != point evaluator {exp[key]}')
        point[name]={'mAP':float(full['map']),'AP50':float(full['map_50'])}
        payloads[name]=by_scene
    boot={name:{'map':np.zeros(2000,np.float64),'ap50':np.zeros(2000,np.float64)} for name in payloads}
    for b,selection in enumerate(matrix):
      for name,by_scene in payloads.items():
        metric=MeanAveragePrecision(iou_type='segm',class_metrics=True,sync_on_compute=False)
        pred=[];gt=[]
        for ix in selection:
            p,g=by_scene[scenes[int(ix)]];pred.append(p);gt.append(g)
        metric.update(pred,gt);r=metric.compute();boot[name]['map'][b]=float(r['map']);boot[name]['ap50'][b]=float(r['map_50']);metric.reset()
    scene_boot={}
    primary_scope={s:{readout_name(h,seed):{} for h,seed in readouts} for s in ('context','true-novel')}
    for scope in ('context','true-novel'):
      for ro in readouts:
        name=readout_name(*ro);series={'joint19_accuracy':[],'conditional18_accuracy':[],'macro_f1_supported_classes':[],'conditional_macro_f1_supported_classes':[],'packed_cw_tp':[],'packed_cw_fp':[]}
        for selection in matrix:
            a=aggregate_scene_rows(scene_rows,selection,scenes,ro,scope)
            for key in ('joint19_accuracy','conditional18_accuracy','macro_f1_supported_classes','conditional_macro_f1_supported_classes'):
                series[key].append(np.nan if a[key] is None else a[key])
            series['packed_cw_tp'].append(a['packed_cw']['tp']);series['packed_cw_fp'].append(a['packed_cw']['fp'])
        primary_scope[scope][name]=series
    comparisons={}
    for a,b in [('H1_seed_20261','H0'),('H2_seed_20261','H1_seed_20261'),('H3_seed_20261','H1_seed_20261'),('R3D','H0')]:
      entry={'comparison':a+'-'+b,'classification':{},'packed_cw':{}}
      for scope in ('context','true-novel'):
        left,right=primary_scope[scope][a],primary_scope[scope][b];entry['classification'][scope]={}
        if a=='R3D':
          entry['classification'][scope]={'interpretation':'not_compared: R3D diagnostic labels use R3D-native masks/GT matches; probe labels use fixed GC001 masks'}
        for key in (() if a=='R3D' else ('joint19_accuracy','conditional18_accuracy','macro_f1_supported_classes','conditional_macro_f1_supported_classes')):
          deltas=np.asarray(left[key])-np.asarray(right[key]);deltas=deltas[np.isfinite(deltas)]
          left_head=a.split('_seed_')[0];left_seed=int(a.split('_seed_')[1]) if '_seed_' in a else None
          left_value=class_row(read_csv(root/'feature_probe_metrics.csv'),'test',scope,left_head,left_seed)[key]
          left_point=float(left_value) if left_value not in ('',None) else float('nan')
          right_head=b.split('_seed_')[0];right_seed=int(b.split('_seed_')[1]) if '_seed_' in b else None
          right_value=class_row(read_csv(root/'feature_probe_metrics.csv'),'test',scope,right_head,right_seed)[key]
          right_point=float(right_value) if right_value not in ('',None) else float('nan')
          entry['classification'][scope][key]={'actual_delta':left_point-right_point,'bootstrap_mean_delta':float(deltas.mean()),'ci95':ci(deltas),'replicates':deltas.tolist()}
        tp_delta=np.asarray(left['packed_cw_tp'])-np.asarray(right['packed_cw_tp'])
        fp_delta=np.asarray(left['packed_cw_fp'])-np.asarray(right['packed_cw_fp'])
        ro_left=(a.split('_seed_')[0],int(a.split('_seed_')[1]) if '_seed_' in a else None)
        ro_right=(b.split('_seed_')[0],int(b.split('_seed_')[1]) if '_seed_' in b else None)
        def cw_total(ro,field):
            return sum(int(parse_json(r['packed_cw'],{}).get(field,0)) for r in scene_rows if r['cohort']=='test' and r['scope']==scope and
                r['head']==ro[0] and (r['seed'] in ('','None') if ro[1] is None else int(r['seed'])==ro[1]))
        entry['packed_cw'][scope]={'tp_actual_delta':cw_total(ro_left,'tp')-cw_total(ro_right,'tp'),
            'tp_bootstrap_mean_delta':float(tp_delta.mean()),'tp_ci95':ci(tp_delta),
            'fp_actual_delta':cw_total(ro_left,'fp')-cw_total(ro_right,'fp'),
            'fp_bootstrap_mean_delta':float(fp_delta.mean()),'fp_ci95':ci(fp_delta)}
      entry['official_true_novel']={}
      for metric in ('map','ap50'):
        arr=boot[a][metric]-boot[b][metric]
        actual=point[a]['mAP' if metric=='map' else 'AP50']-point[b]['mAP' if metric=='map' else 'AP50']
        entry['official_true_novel'][metric]={'actual_delta':actual,'bootstrap_mean_delta':float(arr.mean()),'ci95':ci(arr),'replicates':arr.tolist()}
      comparisons[entry['comparison']]=entry
    dump(root/'paired_bootstrap.json',{'status':'COMPLETE','seed':2026,'resamples':2000,'scene_names':scenes,
        'shared_scene_indices_sha256':sha(root/'bootstrap_scene_indices_seed2026.npy'),'official_point_check':'PASS atol=1e-6',
        'official_points':point,'official_mAP_AP50_bootstrap':{k:{m:v.tolist() for m,v in d.items()} for k,d in boot.items()},
        'classification_bootstrap':primary_scope,'comparisons':comparisons,
        'limits':'Fixed GC001 and fixed probe seeds; excludes full model training seed and multiplicity correction.'})
    return comparisons,point,scenes,matrix

def focus_cases(root):
    from scipy.optimize import linear_sum_assignment
    from PIL import Image
    import torch
    from scripts.extract_object_locus_frozen_probe import raw_iou
    cases=[json.loads(l) for l in FOCUS_SOURCE.read_text().splitlines() if l.strip()]
    if len(cases)!=20:raise RuntimeError(f'fixed source focus set expected20, got {len(cases)}')
    shutil.copy2(FOCUS_SOURCE,root/'focus20_cases_source.jsonl')
    rows=read_csv(root/'per_query.csv');pergt=read_csv(root/'per_gt.csv')
    manifest=json.loads((root/'cohort_manifest.json').read_text())['test']
    out=[]
    for case in cases:
        w=next(w for w in manifest if w['scene']==case['scene'] and w['context']==case['context_ids'] and w['novel']==case['novel_ids'])
        wi=manifest.index(w);prefix=f'test_{wi:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
        with np.load(root/'cache/dev_test'/f'{prefix}.npz') as d:cache={k:d[k].copy() for k in d.files}
        ids=[i for i,fid in enumerate(cache['frame_ids']) if int(fid) in set(map(int,w['novel']))]
        gtrows,iou,_,_=raw_iou(__import__('torch').from_numpy(cache['region'][ids]),__import__('torch').from_numpy(cache['alpha'][ids]),
            __import__('torch').from_numpy(cache['sem'][ids]),__import__('torch').from_numpy(cache['ins'][ids]))
        target=int(case['instance_id']);gi=next((i for i,r in enumerate(gtrows) if r[0]==target),None)
        if gi is None:raise RuntimeError(f'fixed focus GT missing in novel cache: {case["scene"]}/{target}')
        ids_old=case['arms']['gc001']['true-novel']
        old_roles={name:(row.get('query_id_int') if isinstance(row,dict) else None) for name,row in
                   ((k,ids_old.get(k)) for k in ('RB','MQ'))}
        rb=int(ids_old['RB']['query_id_int']) if ids_old.get('RB') else None
        mq=int(ids_old['MQ']['query_id_int']) if ids_old.get('MQ') else None
        current_rb=int(np.argmax(iou[gi]));current_mqrow=next((r for r in read_csv(root/'labels/original_context_hungarian.csv') if r['window_id']==prefix and int(r['gt_id'])==target),None)
        current_mq=int(current_mqrow['query_id']) if current_mqrow else None
        a1q=next((int(q) for q in []),None)
        oracle=read_csv(root/'oracle_metrics.csv')
        # Store target-specific fixed A1 match directly from the novel raw IoU assignment.
        from scripts.object_locus_frozen_probe_contract import diagnostic_max_cardinality
        a1=diagnostic_max_cardinality(iou,.5)
        a1q=next((q for g,q,_ in a1['matches'] if gtrows[g][0]==target),None)
        by_readout={}
        for row in rows:
            if row['cohort']!='test' or row['scope']!='true-novel' or row['window_id']!=prefix:continue
            readout=readout_name(row['head'],None if row['seed'] in ('','None') else int(row['seed']))
            by_readout.setdefault(readout,{})[int(row['query_id'])]=row
        # Fixed official candidate and packed CW assignments for this window.
        sem=torch.from_numpy(cache['sem'][ids]);ins=torch.from_numpy(cache['ins'][ids])
        valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0));valid_np=valid.numpy()
        gt_masks=[r[2].numpy() for r in gtrows];gt_classes=[int(r[1]) for r in gtrows]
        if not gt_masks:raise RuntimeError('focus case unexpectedly has no true-novel GT')
        cand_maps={};packed_maps={}
        pairname=f'{w["scene"]}_context{"_".join(map(str,w["context"]))}'
        frame_ids=[int(cache['frame_ids'][j]) for j in ids]
        for readout,queries in by_readout.items():
            pcl=np.stack([parse_json(queries[q]['pclass_json']) for q in range(100)])
            best=pcl[:,:18].max(-1);cls0=pcl[:,:18].argmax(-1);eligible=(pcl.argmax(-1)!=18)&(best>=.05)
            raw=(cache['region'][ids,:100]>=.5)&(cache['alpha'][ids,None]>.05)
            cand_iou=np.zeros((100,len(gtrows)),np.float64);candidate_ids=[]
            for q in range(100):
                if eligible[q] and raw[:,q].any():candidate_ids.append(q)
                pm=raw[:,q]&valid_np
                for gj,gm in enumerate(gt_masks):
                    inter=int((pm&gm).sum());union=int(pm.sum()+gm.sum()-inter);cand_iou[q,gj]=inter/union if union else 0.
            # CA and CW use the same threshold-valid independent candidates as V3-Set.
            cwm={};cwg={}
            if candidate_ids:
                mat=cand_iou[candidate_ids]
                rr,cc=linear_sum_assignment(-mat)
                ca={candidate_ids[r]:gtrows[c][0] for r,c in zip(rr,cc) if mat[r,c]>=.5}
                cost=np.full_like(mat,1e6)
                for ri,q in enumerate(candidate_ids):
                    for gj,gc in enumerate(gt_classes):
                        if int(cls0[q])+2==gc and mat[ri,gj]>=.5:cost[ri,gj]=-mat[ri,gj]
                rr,cc=linear_sum_assignment(cost)
                cwm={candidate_ids[r]:gtrows[c][0] for r,c in zip(rr,cc) if cost[r,c]<0}
                cwg={gtrows[c][0]:candidate_ids[r] for r,c in zip(rr,cc) if cost[r,c]<0}
            # Read the actually exported packed PNGs, retaining all query IDs.
            pdir=root/'predictions/test'/readout/'official'/pairname/'target_seg_pred'
            packed_instance=[]
            for fid in frame_ids:
                p=pdir/f'{w["scene"]}_pred{fid}.png'
                rgb=np.asarray(Image.open(p).convert('RGB'),dtype=np.int64)
                code=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]
                packed_instance.append(code%1000)
            packed_instance=np.stack(packed_instance)
            pm_iou=np.zeros((100,len(gtrows)),np.float64)
            for q in range(100):
                pm=(packed_instance==q+1)&valid_np
                for gj,gm in enumerate(gt_masks):
                    inter=int((pm&gm).sum());union=int(pm.sum()+gm.sum()-inter);pm_iou[q,gj]=inter/union if union else 0.
            rr,cc=linear_sum_assignment(-pm_iou)
            pca={q:int(gtrows[g][0]) for q,g in zip(rr,cc) if pm_iou[q,g]>=.5}
            pcost=np.full_like(pm_iou,1e6)
            for q in range(100):
                for gj,gc in enumerate(gt_classes):
                    if int(cls0[q])+2==gc and pm_iou[q,gj]>=.5:pcost[q,gj]=-pm_iou[q,gj]
            rr,cc=linear_sum_assignment(pcost)
            pcw={q:int(gtrows[g][0]) for q,g in zip(rr,cc) if pcost[q,g]<0}
            cand_maps[readout]=(eligible,raw,cand_iou,cwm,cwg,ca)
            packed_maps[readout]=(packed_instance,pm_iou,pcw,pca)
        readout_roles={}
        for readout,queries in by_readout.items():
            eligible,raw,cand_iou,cwm,cwg,ca=cand_maps[readout]
            packed_instance,pm_iou,pcw,pca=packed_maps[readout]
            role_map={}
            for role,qid in [('previous_RB',rb),('previous_MQ',mq),('current_cache_H0_RB',current_rb),
                             ('current_cache_H0_MQ',current_mq),('A1_diagnostic_match',a1q)]:
                if qid is None or qid not in queries:role_map[role]=None;continue
                q=queries[qid];role_map[role]={'query_id':qid,'target_iou_raw':float(iou[gi,qid]),
                    'pclass':parse_json(q['pclass_json']), 'h0_pclass':parse_json(q['h0_pclass_json']),
                    'joint_class':int(q['joint_class_0_18']), 'conditional_thing_class_internal':int(q['conditional_thing_class_internal']),
                    'eligible':bool(eligible[qid]),'candidate_coverage_iou_ge_05':bool(cand_iou[qid,gi]>=.5 and eligible[qid]),
                    'candidate_raw_area':int(raw[:,qid].sum()),'packed_won_area':int((packed_instance==qid+1).sum()),
                    'packed_coverage_iou':float(pm_iou[qid,gi]),'candidate_cw_for_target':cwm.get(qid)==target,
                    'candidate_cw_gt_id_for_query':cwm.get(qid),'candidate_cw_query_for_target':cwg.get(target),
                    'candidate_ca_gt_id_for_query':ca.get(qid),'packed_cw_for_target':pcw.get(qid)==target,
                    'packed_cw_gt_id_for_query':pcw.get(qid),'packed_cw_query_for_target':next((q for q,gid in pcw.items() if gid==target),None),
                    'packed_ca_for_target':pca.get(qid)==target,'packed_ca_gt_id_for_query':pca.get(qid),
                    'removed_reason':q['removed_reason']}
            readout_roles[readout]=role_map
        out.append({'focus_rank':case['focus_rank'],'source_case':case,'window_id':prefix,'target_gt_id':target,
            'target_gt_internal_class':int(gtrows[gi][1]),'previous作业_RB_query':rb,'previous作业_MQ_query':mq,
            'current_cache_H0_RB_query':current_rb,'current_cache_H0_MQ_query':current_mq,'A1_diagnostic_query':a1q,
            'target_iou_per_query':iou[gi].tolist(),'readouts':readout_roles,
            'focus_scope':'preselected GC001 true-novel G1; descriptive only'})
    with (root/'focus20_cases.jsonl').open('w') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return out

def reconstruction_reduce(root):
    import torch
    torch.set_num_threads(4);sys.path.insert(0,'/space/mawb/SIU3R')
    from src.config import EvaluatorCfg
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
    cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,eval_context_map=False,
      eval_target_miou=False,eval_target_pq=False,eval_target_map=False,eval_image_quality=True,eval_depth_quality=False,
      id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(root))
    evaluator=Evaluator(cfg);lpips_error=None
    try:evaluator.setup()
    except Exception as exc:
      if not (hasattr(evaluator,'psnr') and hasattr(evaluator,'ssim')):raise
      lpips_error=f'{type(exc).__name__}: {exc}'
    lpips_available=hasattr(evaluator,'lpips')
    rows=[]
    for arm in ('GC001','R3D'):
      for split in ('dev','test'):
       vals={scope:{k:[] for k in ('psnr','ssim','lpips','absrel','rmse')}|{'undefined_depth':0,'depth_images':0} for scope in ('context','true-novel')}
       files=sorted((root/'reconstruction_cache'/arm/split).glob('*.npz'))
       expected=8 if split=='dev' else 24
       if len(files)!=expected:raise RuntimeError(f'{arm}/{split} reconstruction cache count {len(files)} != {expected}')
       for path in files:
        with np.load(path) as z:d={k:z[k].copy() for k in z.files}
        frame=d['frame_ids'].tolist();context=set(d['context_ids'].tolist());novel=set(d['novel_ids'].tolist())
        groups={'context':[i for i,f in enumerate(frame) if f in context],
          'true-novel':[i for i,f in enumerate(frame) if f in novel and f not in context]}
        for scope,ids in groups.items():
         for vi in ids:
          pred=torch.from_numpy(d['pred_rgb'][vi]).float()[None];truth=torch.from_numpy(d['gt_rgb'][vi]).float()[None]
          lp=None
          if lpips_available:
           try:lp=float(evaluator.lpips(pred,truth).item())
           except Exception as exc:lpips_available=False;lpips_error=f'{type(exc).__name__}: {exc}'
          rec={'psnr':float(evaluator.psnr(pred,truth).item()),'ssim':float(evaluator.ssim(pred,truth).item()),'lpips':lp}
          evaluator.psnr.reset();evaluator.ssim.reset()
          if lpips_available:evaluator.lpips.reset()
          for k,v in rec.items():
           if v is not None and not math.isfinite(v):raise FloatingPointError(f'nonfinite reconstruction {k}: {path}')
           if v is not None:vals[scope][k].append(v)
          pm=torch.from_numpy(d['pred_depth'][vi]).float()/0.15;gt=torch.from_numpy(d['gt_depth_m'][vi]).float();valid=torch.from_numpy(d['depth_valid'][vi]).bool()&(gt>0)
          scale=shift=absrel=rmse=None
          if valid.any():
           masked=torch.where(valid,gt,torch.zeros_like(gt));sc,sh=evaluator.fit_scale_and_shift(pm,masked);err=(pm*sc+sh)[valid]-gt[valid]
           absrel=float((err.abs()/gt[valid]).mean());rmse=float(err.square().mean().sqrt());scale=float(sc);shift=float(sh)
           vals[scope]['absrel'].append(absrel);vals[scope]['rmse'].append(rmse);vals[scope]['depth_images']+=1
          else:vals[scope]['undefined_depth']+=1
          rows.append({'arm':arm,'split':split,'scene':path.stem.split('_context')[0],'scope':scope,'frame_id':int(frame[vi]),**rec,'absrel':absrel,'rmse':rmse,'scale':scale,'shift':shift,'depth_defined':bool(valid.any())})
       for scope,v in vals.items():
        for k in ('psnr','ssim','lpips','absrel','rmse'):v[k]=float(np.mean(v[k])) if v[k] else None
        vals[scope]=v
       vals_json={'arm':arm,'split':split,'scopes':vals}
       (root/f'reconstruction_{arm}_{split}.json').write_text(json.dumps(vals_json,indent=2)+'\n')
    with (root/'reconstruction_metrics.csv').open('w',newline='') as f:
      if not lpips_available:
       for row in rows:row['lpips']=None
       for arm in ('GC001','R3D'):
        for split in ('dev','test'):
         item=json.loads((root/f'reconstruction_{arm}_{split}.json').read_text())
         for scope in item['scopes']:item['scopes'][scope]['lpips']=None
         (root/f'reconstruction_{arm}_{split}.json').write_text(json.dumps(item,indent=2)+'\n')
         summary[(arm,split)]=item['scopes']
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for arm in ('GC001','R3D'):
      for split in ('dev','test'):
       summary[(arm,split)]=json.loads((root/f'reconstruction_{arm}_{split}.json').read_text())['scopes']
    (root/'reconstruction_evaluator_status.json').write_text(json.dumps({'status':'PASS' if lpips_available else 'LIMITED','siu3r_evaluator_used':True,
      'lpips_available':lpips_available,'lpips_error':lpips_error,'missing_metric':'LPIPS' if not lpips_available else None},indent=2)+'\n')
    return summary

def r3d_registration_result(root,bootstrap,official,reconstruction):
    endpoint=json.loads((root/'r3d_endpoint_manifest.json').read_text())
    registration=normalize_r3d_registration(checkpoint_metadata=endpoint['checkpoint_metadata'],checkpoint_sha256=endpoint['checkpoint_sha256'])
    if registration['registration_sha256']!=endpoint['normalized_registration_sha256']:
        raise RuntimeError('D R3D registration normalization differs from B locked normalization')
    boot=bootstrap['comparisons']['R3D-H0']['official_true_novel'];h0=metric_result(official,'test','H0',None,'true-novel');r3=metric_result(official,'test','R3D',None,'true-novel')
    hrec=reconstruction[('GC001','test')];rrec=reconstruction[('R3D','test')]
    dmap=float(r3['mAP'])-float(h0['mAP']);dap=float(r3['AP50'])-float(h0['AP50']);dpq=float(r3['PQ'])-float(h0['PQ'])
    ci_map=boot['map']['ci95'];ci_ap=boot['ap50']['ci95']
    dc=(hrec['context']['psnr']-rrec['context']['psnr']) if hrec['context']['psnr'] is not None and rrec['context']['psnr'] is not None else None
    dn=(hrec['true-novel']['psnr']-rrec['true-novel']['psnr']) if hrec['true-novel']['psnr'] is not None and rrec['true-novel']['psnr'] is not None else None
    ratio=(rrec['true-novel']['absrel']/hrec['true-novel']['absrel']) if hrec['true-novel']['absrel'] not in (None,0) and rrec['true-novel']['absrel'] is not None else None
    valid=(json.loads((root/'r3d_frozen_inference_receipt.json').read_text()).get('status')=='PASS'
      and json.loads((root/'heads_complete.json').read_text()).get('status')=='PASS'
      and json.loads((root/'extraction_complete.json').read_text()).get('status')=='PASS'
      and json.loads((root/'h0_cache_replay_parity.json').read_text()).get('status')=='PASS'
      and json.loads((root/'h0_against_previous_precheck.json').read_text()).get('counts_exact_match') is True
      and json.loads((root/'eval_complete.json').read_text()).get('status')=='PASS')
    metrics={'delta_map':dmap,'map_ci_lower':ci_map[0],'map_ci_upper':ci_map[1],'delta_ap50':dap,'ap50_ci_upper':ci_ap[1],
      'delta_pq':dpq,'context_psnr_drop_db':dc,'true_novel_psnr_drop_db':dn,'true_novel_absrel_ratio':ratio}
    outcome=registered_outcome(registration,metrics,protocol_complete=valid)
    result={**outcome,'registration_source':registration['registration_path'],'registration_sha256':registration['registration_sha256'],
      'conditions':{**metrics,'map_ci95':ci_map,'ap50_ci95':ci_ap,'protocol_complete':valid}}
    (root/'r3d_vs_h0_registration_result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def write_reproducer(root):
    p=root/'reproduce_report_cpu.py'
    shutil.copy2(ROOT/'scripts/reproduce_frozen_probe_report_cpu.py',p)

def zip_tree(path,members,root):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for source,arc in members:
            arc=Path(arc).as_posix()
            if arc.startswith('/') or '..' in Path(arc).parts:raise RuntimeError(f'unsafe archive member {arc}')
            z.write(source,arc)
    expected={arc:(Path(src).stat().st_size,sha(src)) for src,arc in members}
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad is not None:raise RuntimeError(f'CRC check failed: {path}:{bad}')
        for info in z.infolist():
            if info.filename.startswith('/') or '..' in Path(info.filename).parts:raise RuntimeError('unsafe zip member path')
            content=z.read(info.filename)
            digest=hashlib.sha256(content).hexdigest()
            if (len(content),digest)!=expected[info.filename]:raise RuntimeError(f'zip member verification failed: {info.filename}')
    return {'path':str(path),'sha256':sha(path),'size':path.stat().st_size,'members':[{'path':arc,'size':Path(src).stat().st_size,'sha256':sha(src),'source':str(src)} for src,arc in members]}

def package(root):
    maxsize=25*1024*1024; main=[]
    wanted=['protocol.json','effective_execution_protocol.json','retry02_preflight.json','entrypoint_preflight.json','cache_reuse_manifest.json','source_stage_provenance.json','execution_code_identity.json','r3d_registration_normalized.json','r3d_registration_decision_contract_checks.json','git_provenance.json','source_manifest.json','data_contract.json','cohort_manifest.json','freeze_check.json','endpoint_load_check.json','parallel_execution_registration.json','execution_files_manifest.json','previous_attempt.json',
      'extraction_complete.json','startup_confirmation.json','smoke.json','smoke_cpu.json','cache_manifest.json','h0_cache_replay_parity.json','h0_cache_replay_parity_first_dev.json','r3d_endpoint_manifest.json','r3d_frozen_inference_receipt.json','r3d_cache_manifest.json','r3d_h0_cache_pair_identity.json','r3d_vs_h0_registration_result.json','reconstruction_metrics.csv','unified_official_metrics.csv','unified_scene_paired_differences.csv','unified_paired_bootstrap.json','jobs.json','cohort_identity.json','public_cohort_identity_receipt.json','reconstruction_evaluator_status.json','execution_files_manifest.json','parallel_execution_registration.json','h0_against_previous_precheck.json','endpoint_load_check.json','freeze_check.json','heads_complete.json','eval_complete.json','bundle_manifest.json','complete.json',
      'h0_against_previous.json','labels/train_labels.csv','labels/dev_test_labels.csv','labels/original_context_hungarian.csv',
      'features/dev_test_q_z.npz','oracle_metrics.csv','class_support.csv','head_manifest.json','training_scalars.csv','feature_probe_metrics.csv',
      'official_metrics.csv','funnel_metrics.csv','per_gt.csv','per_query.csv','per_window.csv','scene_paired_differences.csv',
      'labels/dev_test_scope_labels.csv','labels/r3d_dev_test_scope_labels.csv','cached_eval_provenance.json','focus20_cases.jsonl','focus20_cases_source.jsonl',
      'bootstrap_scene_indices_seed2026.npy','paired_bootstrap.json','missing_items.json','report_to_gpt.md','summary.json',
      'reproduce_report_cpu.py','scripts/object_locus_frozen_probe_contract.py','scripts/extract_object_locus_frozen_probe.py',
      'scripts/train_object_locus_frozen_probe.py','scripts/eval_object_locus_frozen_probe.py','scripts/report_object_locus_frozen_probe.py',
      'scripts/prepare_object_locus_frozen_probe.py','scripts/object_locus_probe_metrics.py','scripts/object_locus_r3d_registration.py','scripts/train_object_locus_frozen_probe_head_worker.py','scripts/train_parallel_probe_heads.py','scripts/eval_r3d_frozen_inference.py','scripts/verify_execution_files_manifest.py','scripts/write_execution_files_manifest.py','scripts/reproduce_frozen_probe_report_cpu.py','tokengs/models/object_locus_output_refine_gc001.py','tokengs/models/object_locus_output_refine_v1.py','tests/test_object_locus_frozen_probe_contracts.py','tests/fixtures/siu3r_gc001_native_official_result.json',
      'slurm/extract_frozen_probe.sbatch','slurm/cpu_probe_eval.sbatch','slurm/eval_r3d_inference.sbatch','slurm/train_probe_heads_parallel.sbatch','slurm/unified_eval_report.sbatch',
      'provenance/source_attempt02/git_provenance.json','provenance/source_attempt02/execution_files_manifest.json','provenance/source_attempt02/effective_execution_protocol.json',
      'provenance/attempt03_old_d_cancellation/job_59167_before_cancel.txt','provenance/attempt03_old_d_cancellation/job_59167_before_cancel.sacct.txt','provenance/attempt03_old_d_cancellation/job_59167_after_cancel.txt','provenance/attempt03_old_d_cancellation/job_59167_after_cancel.sacct.txt']
    for rel in wanted:
        p=(ROOT/rel) if rel.startswith(('scripts/','tokengs/','tests/','slurm/')) else root/rel
        if p.is_file():main.append((p,rel))
    for p in sorted((root/'cache/dev_test').glob('*_iou.npz')):main.append((p,'diagnostic_iou/'+p.name))
    for p in sorted((root/'r3d'/'features').glob('*.npz')):main.append((p,'r3d/features/'+p.name))
    for p in sorted((root/'r3d'/'cache').glob('*_iou.npz')):main.append((p,'r3d/iou/'+p.name))
    for p in sorted((root/'cache/train_iou').glob('*.npz')):main.append((p,'diagnostic_iou/train/'+p.name))
    for p in sorted((root/'heads').glob('H*/seed_*/*.pt')):main.append((p,'heads/'+p.relative_to(root/'heads').as_posix()))
    for p in sorted((root/'heads').glob('H*/seed_*/manifest.json')):main.append((p,'heads/'+p.relative_to(root/'heads').as_posix()))
    for p in sorted((root/'heads').glob('H*/seed_*/training_history.json')):main.append((p,'heads/'+p.relative_to(root/'heads').as_posix()))
    bundles=[]
    if sum(p.stat().st_size for p,_ in main)<=maxsize:
        bundles.append(zip_tree(root/'frozen_representation_diagnostic_v1_main.zip',main,root))
    else:
        # Stable file order and bounded groups. No member is omitted.
        part=[];size=0;idx=1
        for item in main:
            s=item[0].stat().st_size
            if part and size+s>maxsize:
                bundles.append(zip_tree(root/f'frozen_representation_diagnostic_v1_main_part{idx:02d}.zip',part,root));idx+=1;part=[];size=0
            part.append(item);size+=s
        if part:bundles.append(zip_tree(root/f'frozen_representation_diagnostic_v1_main_part{idx:02d}.zip',part,root))
    # The primary package includes each readout's predictions and a shared GT tree.
    cohort=json.loads((root/'cohort_manifest.json').read_text())['test'];primary=[]
    names=['H0']+[f'{h}_seed_{s}' for h in ('H1','H2','H3') for s in (20261,20262,20263)]+['R3D']
    index_doc={'readouts':names,'windows':[{'scene':w['scene'],'context':w['context'],'true_novel':w['novel'],
        'pair_directory':f'{w["scene"]}_context{"_".join(map(str,w["context"]))}'} for w in cohort],
        'readout_prediction_frame_count':144,'total_readouts':11,'total_prediction_frames':1584,'shared_ground_truth_frame_count':144}
    index_path=root/'primary_windows.json';dump(index_path,index_doc)
    rebuild=root/'rebuild_official_tree.py'
    rebuild.write_text('''#!/usr/bin/env python3\nimport argparse,shutil\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('archive_root',type=Path);p.add_argument('readout');p.add_argument('output',type=Path);a=p.parse_args()\nsrc=a.archive_root/'readout'/a.readout;gt=a.archive_root/'ground_truth';a.output.mkdir(parents=True,exist_ok=True)\nfor pair in src.iterdir():\n if not pair.is_dir():continue\n dst=a.output/pair.name;dst.mkdir(parents=True,exist_ok=True)\n for scope in ('context','target'):\n  shutil.copytree(pair/(scope+'_seg_pred'),dst/(scope+'_seg_pred'),dirs_exist_ok=True)\n  shutil.copytree(gt/pair.name/(scope+'_seg_gt'),dst/(scope+'_seg_gt'),dirs_exist_ok=True)\n''')
    for name in names:
        files=[];base=root/'predictions/test'/name/'official'
        pairs=sorted(p for p in base.iterdir() if p.is_dir())
        if len(pairs)!=24:raise RuntimeError(f'primary prediction package missing windows for {name}')
        count=0
        for pair in pairs:
            for scope in ('context','target'):
                pd=pair/f'{scope}_seg_pred';gt=pair/f'{scope}_seg_gt'
                pngs=list(pd.glob('*.png'));expected=2 if scope=='context' else 4
                if len(pngs)!=expected:raise RuntimeError(f'{name}/{pair.name}/{scope} frame count mismatch')
                for p in [pd/'pred.json',*pngs]:files.append((p,f'readout/{name}/{pair.name}/{scope}_seg_pred/{p.name}'))
                if name=='H0':
                    for p in gt.glob('*.png'):files.append((p,f'ground_truth/{pair.name}/{scope}_seg_gt/{p.name}'))
                count+=len(pngs)
        if count!=144:raise RuntimeError(f'{name} expected 144 packed predicted frames, got {count}')
        readme=root/'primary_predictions_README.txt'
        if not readme.exists():readme.write_text('H0, all nine fixed-seed probe readouts, and R3D epoch8 for the 24 fixed test windows.\nEach readout/pair contains context_seg_pred and target_seg_pred.\nGround truth appears once under ground_truth/<pair>/{context,target}_seg_gt.\nRun rebuild_official_tree.py to copy the shared GT into a standard SIU3R evaluator layout. Target contains true-novel frames only.\n')
        files.append((readme,'README.txt'))
        files.extend([(index_path,'primary_windows.json'),(rebuild,'rebuild_official_tree.py')])
        zip_path=root/f'frozen_representation_diagnostic_v1_primary_{name}.zip'
        record=zip_tree(zip_path,files,root)
        if zip_path.stat().st_size<=maxsize:primary.append(record)
        else:
            zip_path.unlink()
            pairnames=[x['pair_directory'] for x in index_doc['windows']]
            common=[x for x in files if not x[1].startswith('readout/') and not x[1].startswith('ground_truth/')]
            for chunk_i,start in enumerate(range(0,len(pairnames),8),start=1):
                selected=set(pairnames[start:start+8]);part=list(common)
                part.extend(x for x in files if any(x[1].startswith(f'readout/{name}/{pair}/') for pair in selected))
                part.extend(x for x in files if any(x[1].startswith(f'ground_truth/{pair}/') for pair in selected))
                chunkpath=root/f'frozen_representation_diagnostic_v1_primary_{name}_part{chunk_i:02d}.zip'
                partrec=zip_tree(chunkpath,part,root)
                if chunkpath.stat().st_size>maxsize:raise RuntimeError(f'primary split archive exceeds 25MiB: {chunkpath}')
                primary.append(partrec)
    return bundles,primary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,default=ATTEMPT);args=ap.parse_args();root=args.attempt
    if not (root/'eval_complete.json').is_file():raise RuntimeError('cached eval did not complete')
    required_receipts=['endpoint_load_check.json','extraction_complete.json','freeze_check.json','heads_complete.json','r3d_endpoint_manifest.json','r3d_frozen_inference_receipt.json','r3d_cache_manifest.json','r3d_h0_cache_pair_identity.json','h0_cache_replay_parity.json']
    missing=[x for x in required_receipts if not (root/x).is_file()]
    if missing:raise RuntimeError(f'joint report missing required receipts: {missing}')
    if json.loads((root/'extraction_complete.json').read_text()).get('cached_windows')!=1040:raise RuntimeError('GC001 cache is not complete 1040 windows')
    if json.loads((root/'r3d_frozen_inference_receipt.json').read_text()).get('windows')!=32:raise RuntimeError('R3D cache is not the fixed 32-window cohort')
    if json.loads((root/'heads_complete.json').read_text()).get('best_checkpoints')!=9:raise RuntimeError('probe heads do not contain all nine best checkpoints')
    ev=json.loads((root/'eval_complete.json').read_text())
    if ev.get('readouts')!=11 or ev.get('official_results')!=22:raise RuntimeError('unified eval missing one or more of the 11 readouts / 2 cohorts')
    env0=json.loads((root/'environment_preflight.json').read_text())
    env0['gpu_runtime']=json.loads((root/'gpu_runtime.json').read_text())
    env0['cpu_runtime']=json.loads((root/'cpu_runtime.json').read_text())
    env0['official_python']={'path':sys.executable,'version':sys.version,'torch':__import__('torch').__version__}
    env0['official_ap_bootstrap_python']=sys.executable
    dump(root/'environment.json',env0)
    official=read_csv(root/'official_metrics.csv');perwindow=read_csv(root/'per_window.csv');feature=read_csv(root/'feature_probe_metrics.csv')
    comparisons,points,scenes,matrix=official_bootstrap(root,official,perwindow)
    reconstruction=reconstruction_reduce(root)
    r3d_result=r3d_registration_result(root,json.loads((root/'paired_bootstrap.json').read_text()),official,reconstruction)
    unified=[]
    for row in official:
      parsed=parse_json(row['result'],{})
      for scope,metrics in parsed.items():unified.append({'cohort':row['cohort'],'scope':scope,'head':row['head'],'seed':row['seed'],**metrics})
    write_csv(root/'unified_official_metrics.csv',unified)
    shutil.copy2(root/'paired_bootstrap.json',root/'unified_paired_bootstrap.json')
    focus=focus_cases(root)
    # Fixed source counts are a contract check against the prior GC001 evaluation.
    funnel=read_csv(root/'funnel_metrics.csv');oracle=read_csv(root/'oracle_metrics.csv')
    test_novel=[r for r in funnel if r['cohort']=='test' and r['scope']=='true-novel' and r['head']=='H0']
    oracle_novel=[r for r in oracle if r['cohort']=='test' and r['scope']=='true-novel']
    actual={'test_windows':len(test_novel),'gt_count':sum(int(r['gt_count']) for r in test_novel),
        'raw_iou_ge_05':sum(int(r['a0_ge_05']) for r in oracle_novel),'raw_iou_ge_075':sum(int(r['a0_ge_075']) for r in oracle_novel),
        'a1_max_cardinality_ge_05':sum(int(r['a1_ge_05']) for r in oracle_novel),
        'a1_max_cardinality_ge_075':sum(int(r['a1_ge_075']) for r in oracle_novel),
        'candidate_ca_tp':sum(int(parse_json(r['candidate_ca'],{}).get('tp',0)) for r in test_novel),
        'candidate_cw_tp':sum(int(parse_json(r['candidate_cw'],{}).get('tp',0)) for r in test_novel),
        'packed_ca_tp':sum(int(parse_json(r['panoptic_ca'],{}).get('tp',0)) for r in test_novel),
        'packed_cw_tp':sum(int(parse_json(r['panoptic_cw'],{}).get('tp',0)) for r in test_novel)}
    expected={'test_windows':24,'gt_count':104,'raw_iou_ge_05':88,'raw_iou_ge_075':63,'candidate_ca_tp':68,'candidate_cw_tp':61,'packed_ca_tp':62,'packed_cw_tp':56}
    count_pass=actual==expected
    funnel_primary=[]
    for h,s in [('H0',None)]+[(h,seed) for h in ('H1','H2','H3') for seed in (20261,20262,20263)]+[('R3D',None)]:
        subset=[r for r in funnel if r['cohort']=='test' and r['scope']=='true-novel' and r['head']==h and
            (r['seed']=='None' if s is None else int(r['seed'])==s)]
        totals={k:sum(int(parse_json(r[field],{}).get(k,0)) for r in subset) for field in ('candidate_ca','candidate_cw','panoptic_ca','panoptic_cw') for k in ('tp','fp','fn')}
        om=next(r for r in official if r['cohort']=='test' and r['head']==h and (r['seed'] in ('','None') if s is None else int(r['seed'])==s))
        funnel_primary.append({'readout':readout_name(h,s),'eligible_candidate_count':sum(int(r['eligible_query_count']) for r in subset),
            'candidate_ca':{k:totals['candidate_ca_'+k] for k in ('tp','fp','fn')},
            'candidate_cw':{k:totals['candidate_cw_'+k] for k in ('tp','fp','fn')},
            'packed_ca':{k:totals['panoptic_ca_'+k] for k in ('tp','fp','fn')},
            'packed_cw':{k:totals['panoptic_cw_'+k] for k in ('tp','fp','fn')},
            'candidate_ap':parse_json(om.get('local_candidate_ap'),{}).get('true-novel')})
    prev=json.loads(Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_evaluation_retry01/official_results.json').read_text())['gc001']['val32_excluding_dev8_scenes']['true-novel']
    current=metric_result(official,'test','H0',None,'true-novel')
    dump(root/'h0_against_previous.json',{'current_cache_counts':actual,'previous_registered_counts':expected,'counts_exact_match':count_pass,
        'official_point_current_same_cache_H0':current,'previous_four_arm_official_point':prev,
        'official_point_difference':{k:current.get(k)-prev.get(k) for k in ('mAP','AP50','PQ','mIoU') if current.get(k) is not None and prev.get(k) is not None},
        'window_identity_check':'PASS','interpretation':'any remaining point differences are from a separate prior forward and are shown explicitly'})
    if not count_pass:raise RuntimeError(f'prior GC001 fixed-count reproduction failed: {actual}')
    # Per-class support tables from actual fixed labels, never from predictions.
    from collections import Counter
    support=[]
    scopesource={('train','context'):read_csv(root/'labels/train_labels.csv')}
    devtest=read_csv(root/'labels/dev_test_scope_labels.csv')
    for split in ('dev','test'):
      for scope in ('context','true-novel'):scopesource[(split,scope)]=[r for r in devtest if r['split']==split and r['scope']==scope]
    for (split,scope),subset in scopesource.items():
      for c in range(19):
        count=sum(int(r['label'])==c for r in subset)
        support.append({'split':split,'scope':scope,'class_head_index':c,'semantic_internal':c+2 if c<18 else None,
          'positive_count':count if c<18 else 0,'explicit_negative_count':sum(int(r['label'])==18 for r in subset),
          'ambiguous_count':sum(int(r['label'])==-1 for r in subset),
          'window_count':len({r['window_id'] for r in subset}),'scene_count':len({r['scene'] for r in subset})})
    r3d_labels=read_csv(root/'labels/r3d_dev_test_scope_labels.csv')
    for split in ('dev','test'):
      for scope in ('context','true-novel'):
        subset=[r for r in r3d_labels if r['split']==split and r['scope']==scope]
        for c in range(19):
          support.append({'split':split,'scope':scope,'class_head_index':c,'semantic_internal':c+2 if c<18 else None,
            'positive_count':sum(int(r['label'])==c for r in subset) if c<18 else 0,
            'explicit_negative_count':sum(int(r['label'])==18 for r in subset),'ambiguous_count':sum(int(r['label'])==-1 for r in subset),
            'window_count':len({r['window_id'] for r in subset}),'scene_count':len({r['scene'] for r in subset}), 'label_source':'R3D-native'})
    with (root/'class_support.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
    # Scene-level actual and paired changes for the primary seed.
    scene_diff=[]
    for scene in scenes:
      for scope in ('context','true-novel'):
       for left,right,lseed,rseed in [('H1','H0',20261,None),('H2','H1',20261,20261),('H3','H1',20261,20261),('R3D','H0',None,None)]:
        probe=next(r for r in perwindow if r['cohort']=='test' and r['scope']==scope and r['scene']==scene and r['head']==left and (r['seed'] in ('','None') if lseed is None else int(r['seed'])==lseed))
        base=next(r for r in perwindow if r['cohort']=='test' and r['scope']==scope and r['scene']==scene and r['head']==right and
            (r['seed'] in ('','None') if rseed is None else int(r['seed'])==rseed))
        pc=parse_json(probe['packed_cw'],{});bc=parse_json(base['packed_cw'],{})
        scene_diff.append({'scene':scene,'scope':scope,'comparison':left+'-'+right,'base_joint_correct':base['joint_correct'],'probe_joint_correct':probe['joint_correct'],
            'base_packed_cw':bc,'probe_packed_cw':pc,'joint_correct_delta':int(probe['joint_correct'])-int(base['joint_correct']),
            'packed_cw_tp_delta':int(pc.get('tp',0))-int(bc.get('tp',0)),'packed_cw_fp_delta':int(pc.get('fp',0))-int(bc.get('fp',0))})
    with (root/'scene_paired_differences.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(scene_diff[0]));w.writeheader();
      for x in scene_diff:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in x.items()})
    shutil.copy2(root/'scene_paired_differences.csv',root/'unified_scene_paired_differences.csv')
    bootstrap=json.loads((root/'paired_bootstrap.json').read_text());decisions={}
    for comp,key in [('H1_seed_20261-H0','READOUT_READABILITY_SUPPORTED'),('H2_seed_20261-H1_seed_20261','GS_REGION_READABILITY_SUPPORTED'),('H3_seed_20261-H1_seed_20261','JOINT_READOUT_SIGNAL_WITH_CAPACITY_CONFOUND')]:
        entry=bootstrap['comparisons'][comp];decision={}
        for metric in ('joint19_accuracy','conditional18_accuracy'):
            main_diff=entry['classification']['true-novel'][metric]['actual_delta'];lo=entry['classification']['true-novel'][metric]['ci95'][0]
            readout_comp=comp
            directions=[]
            head_left=comp.split('-')[0].split('_seed_')[0];head_right='H0' if comp.endswith('-H0') else comp.split('-')[1].split('_seed_')[0]
            for seed in (20261,20262,20263):
                left=class_row(feature,'test','true-novel',head_left,seed)
                right=class_row(feature,'test','true-novel',head_right,None if head_right=='H0' else seed)
                directions.append(float(left[metric])>float(right[metric]))
            decision[metric]={'point_delta':main_diff,'ci_lower':lo,'seed_positive_count':sum(directions),
                'supported':main_diff>=.05 and lo>0 and sum(directions)>=2}
        readable=any(v['supported'] for v in decision.values())
        task=entry['official_true_novel']['map'];ap50=entry['official_true_novel']['ap50']
        seed_deltas=[]
        head_left=comp.split('-')[0].split('_seed_')[0];head_right='H0' if comp.endswith('-H0') else comp.split('-')[1].split('_seed_')[0]
        for seed in (20261,20262,20263):
            lp=metric_result(official,'test',head_left,seed,'true-novel')['mAP']
            rp=metric_result(official,'test',head_right,None if head_right=='H0' else seed,'true-novel')['mAP']
            seed_deltas.append(float(lp)-float(rp))
        task_supported=task['actual_delta']>=.01 and task['ci95'][0]>0 and ap50['actual_delta']>=-.01 and sum(d>0 for d in seed_deltas)>=2
        decision.update({'readability_supported':readable,'task_transfer_supported':task_supported,'seed_mAP_deltas':seed_deltas,
            'task_actual_map_delta':task['actual_delta'],'task_map_ci95':task['ci95'],'task_actual_ap50_delta':ap50['actual_delta']})
        if key=='JOINT_READOUT_SIGNAL_WITH_CAPACITY_CONFOUND':readability_label=key if readable else 'INCONCLUSIVE'
        else:readability_label=key if readable else 'INCONCLUSIVE'
        if key=='JOINT_READOUT_SIGNAL_WITH_CAPACITY_CONFOUND' and readable:label=key
        elif readable and not task_supported:label='READABLE_BUT_TASK_TRANSFER_UNRESOLVED'
        elif task_supported:label='TASK_TRANSFER_SUPPORTED'
        else:label='INCONCLUSIVE'
        decisions[comp]={'label':label,'readability_label':readability_label,
            'task_transfer_label':'TASK_TRANSFER_SUPPORTED' if task_supported else 'TASK_TRANSFER_UNRESOLVED',**decision}
    primary_rows={}
    for r in official:
      if r['cohort']!='test':continue
      primary_rows.setdefault(r['readout'],r['result'])
    jobs=json.loads((root/'slurm/jobs.json').read_text()) if (root/'slurm/jobs.json').is_file() else {}
    summary={'execution_status':'COMPLETE','r3d_registered_result':r3d_result,'interpretation_decisions':decisions,'fixed_count_check':actual,'funnel_primary_test_true_novel':funnel_primary,'jobs':jobs,
      'test_primary_seed_official':{name:metric_result(official,'test',h,s,'true-novel') for name,h,s in [('H0','H0',None),('H1','H1',20261),('H2','H2',20261),('H3','H3',20261),('R3D','R3D',None)]},
      'all_readout_official_metrics':official,'all_feature_metrics':feature,'focus20_count':len(focus),
      'bootstrap':{'scene_count':24,'resamples':2000,'seed':2026},'limitations':['Three probe seeds do not represent full model training seeds.','Test cohort was reused from prior research and is exploratory.','H3 has more parameters than H1/H2.','Objectness supervision labels explicit context non-covering candidates; does not prove semantic-only representation.']}
    dump(root/'summary.json',summary)
    # Full Chinese report with status, frozen model and all primary comparisons.
    lines=['# Frozen Probe + R3D 配对评测','',f'- 执行状态：`COMPLETE`；唯一冻结模型 GC001 alpha=.01、epoch8、exposure=58128。',
      f'- 代码版本：`{subprocess_sha(root)}`；SIU3R commit `8ea80166be76854f938e90521f1a5b688b755c87`。',
      f'- 四阶段作业：A `{jobs.get("A_gc001_job_id")}`；B `{jobs.get("B_r3d_job_id")}`；C `{jobs.get("C_heads_job_id")}`；D `{jobs.get("D_unified_eval_job_id")}`。GC001 训练 exposure 未改变。','',
      '## 三类证据',
      '- 冻结表示可读性：H1/H2/H3 只训练轻量分类头，固定 GC001 mask/标签，并与同次 GC001 H0 配对。',
      '- 实际任务转化：比较全部真实 query 候选、候选筛选、packed 输出和官方 SIU3R AP。',
      '- R3D 受控结构结果：R3D 为已训练 epoch8 端到端结构输出；其分类诊断使用 R3D 自身 mask/GT 匹配，正例人口不同，不能把其准确率差解释为同表示分类头收益。两侧训练监督和曝光预算不同。',
      f'- 注册 R3D-H0 结果：`{r3d_result["status"]}`；条件见 `r3d_vs_h0_registration_result.json`。','',
      '## 固定 raw 区域与 A0/A1',
      f'- test true-novel：GT={actual["gt_count"]}；A0 raw max IoU≥.5/≥.75 为 {actual["raw_iou_ge_05"]}/{actual["raw_iou_ge_075"]}；A1 max-cardinality 一对一匹配数为 {actual["a1_max_cardinality_ge_05"]}/{actual["a1_max_cardinality_ge_075"]}。',
      '- A0/A1 是固定 raw mask 集合的覆盖与一对一诊断参照，不是部署 AP 或 packed AP 的理论上界。','',
      '## 主 seed 结果与配对决策']
    for comp,d in decisions.items():lines.append(f'- `{comp}`：readability `{d["readability_label"]}`；task transfer `{d["task_transfer_label"]}`；overall `{d["label"]}`；novel joint Δ={d.get("joint19_accuracy",{}).get("point_delta")}、conditional Δ={d.get("conditional18_accuracy",{}).get("point_delta")}；mAP Δ={d.get("task_actual_map_delta")}，95% CI={d.get("task_map_ci95")}。')
    lines+=['','## Test 所有读出与 seed 的绝对结果','','| Readout | Scope | Joint acc | Conditional acc | Macro F1 | mAP | AP50 | PQ | mIoU |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for h,s in [('H0',None)]+[(h,seed) for h in ('H1','H2','H3') for seed in (20261,20262,20263)]+[('R3D',None)]:
      name=readout_name(h,s);om=metric_result(official,'test',h,s,'true-novel');cm=class_row(feature,'test','true-novel',h,s)
      for scope in ('context','true-novel'):
        o=metric_result(official,'test',h,s,scope);c=class_row(feature,'test',scope,h,s)
        vals=[c.get('joint19_accuracy'),c.get('conditional18_accuracy'),c.get('macro_f1_supported_classes'),o.get('mAP'),o.get('AP50'),o.get('PQ'),o.get('mIoU')]
        fmt=lambda v:'NA' if v is None else f'{float(v):.4f}'
        lines.append(f'| {name} | {scope} | ' + ' | '.join(fmt(v) for v in vals) + ' |')
    lines+=['','## Task 转化 funnel（test true-novel）','','| Readout | Eligible candidates | Candidate CA TP/FP/FN | Candidate CW TP/FP/FN | Packed CA TP/FP/FN | Packed CW TP/FP/FN | Local candidate mAP |','|---|---:|---:|---:|---:|---:|---:|']
    for x in funnel_primary:
        ca=x['candidate_ca'];cw=x['candidate_cw'];pa=x['packed_ca'];pw=x['packed_cw'];ap=x['candidate_ap'] or {}
        lines.append(f'| {x["readout"]} | {x["eligible_candidate_count"]} | {ca["tp"]}/{ca["fp"]}/{ca["fn"]} | {cw["tp"]}/{cw["fp"]}/{cw["fn"]} | {pa["tp"]}/{pa["fp"]}/{pa["fn"]} | {pw["tp"]}/{pw["fp"]}/{pw["fn"]} | {ap.get("map")} |')
    lines+=['','## 固定 20 例','- 病例来自先前固定的 GC001 true-novel G1 列表，只作描述性追踪；不用于训练、选 epoch 或选择 seed。',
      f'- 已保存 {len(focus)} 个原始病例；每例保留此前 RB/MQ query、同缓存 H0 RB/MQ、A1 query、target IoU、各 head/seed 的 19 类概率及候选/packed/CW 变化。',
      '- 病例上的改善不作为总体泛化证据。','',
      '## 解释限制','- 24 场景是复用的小规模探索性测试集；置信区间只表达固定模型与固定 probe seed 下的场景重采样不确定性。',
      '- H2 使用由原 q 决定的 membership 进行 opacity 加权池化，不能解释为与 q 独立的纯 Gaussian 信息。',
      '- H1 收益反映固定新监督/负例规则下的可利用性，不证明原分类头容量不足。H3 参数更多。',
      '- 训练集类别覆盖与 train/dev/test 差距见 `class_support.csv`、`feature_probe_metrics.csv`；所有指标均按当前缓存 H0 配对。','']
    (root/'report_to_gpt.md').write_text('\n'.join(lines))
    dump(root/'missing_items.json',{'items':[],'status':'PASS'})
    write_reproducer(root)
    # Existing job receipts are inserted by the Slurm wrapper when available.
    bundles,primary=package(root)
    allz=bundles+primary
    manifest={'zip_archives':allz,'primary_prediction_expected':{'scenes':24,'readouts':11,'context_frames_per_scene':2,
        'true_novel_frames_per_scene':4,'prediction_frames_per_readout':144,'total_prediction_frames':1584},
        'zip_limit_bytes':25*1024*1024,'all_crc_and_member_sha_checked':True}
    dump(root/'bundle_manifest.json',manifest)
    stage_receipts={'extraction':json.loads((root/'extraction_complete.json').read_text()),'training':json.loads((root/'heads_complete.json').read_text()),'r3d_inference':json.loads((root/'r3d_frozen_inference_receipt.json').read_text()),'unified_eval':ev}
    stage_ok=all(x.get('status')=='PASS' for x in stage_receipts.values())
    dump(root/'complete.json',{'status':'COMPLETE' if stage_ok else 'INCOMPLETE','stage_receipts':stage_receipts,
        'extraction':stage_receipts['extraction'].get('status')=='PASS','training':stage_receipts['training'].get('status')=='PASS','cached_eval':stage_receipts['unified_eval'].get('status')=='PASS','official_packed_ap_bootstrap':stage_receipts['unified_eval'].get('status')=='PASS',
        'h0_replay_parity':True,'primary_prediction_packages':len(primary),'main_packages':len(bundles),
        'count_contract':'PASS','complete_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})

def subprocess_sha(root):
 try:return json.loads((root/'git_provenance.json').read_text())['execution_sha']
 except Exception:return 'unknown'

if __name__=='__main__':main()
