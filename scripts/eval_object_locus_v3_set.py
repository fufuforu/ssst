"""Local candidate/panoptic evaluation for Object-Locus V3-Set."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

def _targets(sem, ins):
    valid=(sem>=0)&(sem<=19)&((sem<2)|(ins>0))
    ids=torch.unique(ins[valid&(sem>=2)&(ins>0)],sorted=True)
    rows=[]
    for iid in ids:
        m=valid&(sem>=2)&(ins==iid)
        labels=sem[m]; counts=torch.bincount(labels,minlength=20)
        cls=int(torch.nonzero(counts==counts.max(),as_tuple=False)[0,0])
        rows.append((cls,int(iid),m))
    return valid,rows

def _candidate_stats(out,batch,view_ids):
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    v=list(view_ids); sem=batch['semantic_label_all'][0,v].long(); ins=batch['instance_label_all'][0,v].long()
    valid,gtrows=_targets(sem,ins); masks=out['region_mass'][0,v,:100].float(); alpha=out['alpha'][0,v,0]
    pclass=out['p_class'][0]; bestprob,cls0=pclass[:,:18].max(-1); eligible=(pclass.argmax(-1)!=18)&(bestprob>=.05)
    raw_all=(masks>=.5)&(alpha[:,None]>.05)&valid[:,None]
    raw=raw_all & eligible[None,:,None,None]
    predictions=[]
    for q in range(100):
        if bool(eligible[q]) and bool(raw[:,q].any()):
            mean_membership=float(masks[:,q][raw[:,q]].mean())
            predictions.append((q,int(cls0[q])+2,float(bestprob[q])*mean_membership,raw[:,q]))
    gt_masks=[x[2] for x in gtrows]; gtc=[x[0] for x in gtrows]
    iou=np.zeros((len(predictions),len(gtrows)),np.float64)
    for i,(_,_,_,pm) in enumerate(predictions):
        for j,gm in enumerate(gt_masks):
            inter=int((pm&gm).sum()); union=int((pm|gm).sum()); iou[i,j]=inter/union if union else 0.
    tp=fp=fn=0; ctp=cfp=cfn=0
    if iou.size:
        ri,ci=linear_sum_assignment(-iou)
        usedp=set(); usedg=set()
        for i,j in zip(ri,ci):
            if iou[i,j]>=.5:
                usedp.add(i);usedg.add(j);tp+=1
                if predictions[i][1]==gtc[j]:ctp+=1
        fp=len(predictions)-len(usedp);fn=len(gtrows)-len(usedg)
        # CW uses its own class-constrained one-to-one assignment.
        cw_cost=np.full_like(iou,1e6)
        for pi,(_,pcls,_,_) in enumerate(predictions):
            for gj,gcls in enumerate(gtc):
                if pcls==gcls and iou[pi,gj]>=.5:cw_cost[pi,gj]=-iou[pi,gj]
        if cw_cost.size:
            cri,ccj=linear_sum_assignment(cw_cost)
            ctp=sum(1 for pi,gj in zip(cri,ccj) if cw_cost[pi,gj]<0)
        cfp=len(predictions)-ctp;cfn=len(gtrows)-ctp
    else: fp=len(predictions);fn=len(gtrows);cfp=fp;cfn=fn
    raw_best=[]; per_gt=[]
    gt_query_rows=[]
    for j,(cls,gid,gm) in enumerate(gtrows):
        all_ious=[]
        for q in range(100):
            pm=raw_all[:,q]; inter=int((pm&gm).sum()); union=int((pm|gm).sum())
            all_ious.append(inter/union if union else 0.)
        rawq=int(np.argmax(all_ious)); best=float(all_ious[rawq]); raw_best.append(best)
        bi=int(iou[:,j].argmax()) if len(predictions) else -1
        per_gt.append({'class':cls,'instance_id':gid,'best_raw_iou':best,'best_query':rawq,
                       'best_query_class':predictions[bi][1] if bi>=0 else None,
                       'best_query_score':predictions[bi][2] if bi>=0 else None,
                       'best_eligible_candidate_iou':float(iou[bi,j]) if bi>=0 else 0.})
        gt_query_rows.append({'gt_class':cls,'gt_instance_id':gid,'raw_iou':best,
                              'candidate_iou':float(iou[bi,j]) if bi>=0 else 0.,
                              'panoptic_iou':None})
    # TorchMetrics consumes one scene/window as one image; views are stacked vertically.
    pm=torch.stack([x[3] for x in predictions]) if predictions else torch.zeros((0,*sem.shape),dtype=torch.bool,device=sem.device)
    gt_masks_tensor=torch.stack(gt_masks) if gt_masks else torch.zeros((0,*sem.shape),dtype=torch.bool,device=sem.device)
    pan_sem,pan_ins,_=assemble_panoptic(out); pan_sem=pan_sem[v];pan_ins=pan_ins[v]
    pgt=[]; pcls=[]
    for cls,gid,gm in gtrows:pgt.append(gm);pcls.append(cls)
    pqueries=[]
    for q in range(100): pqueries.append(pan_ins==(q+1))
    pious=np.zeros((100,len(gtrows)),float)
    for q in range(100):
        for j,gm in enumerate(gt_masks):
            pred=pqueries[q]&valid
            inter=int((pred&gm).sum());union=int((pred|gm).sum());pious[q,j]=inter/union if union else 0
    pca=pcw=0
    if pious.size:
        ri,ci=linear_sum_assignment(-pious)
        for q,j in zip(ri,ci):
            if pious[q,j]>=.5:
                pca+=1
        pan_cw_cost=np.full_like(pious,1e6)
        for q in range(100):
            for j,gcls in enumerate(gtc):
                if int(cls0[q])+2==gcls and pious[q,j]>=.5:pan_cw_cost[q,j]=-pious[q,j]
        if pan_cw_cost.size:
            pri,pcj=linear_sum_assignment(pan_cw_cost)
            pcw=sum(1 for q,j in zip(pri,pcj) if pan_cw_cost[q,j]<0)
    # Semantic confusion from the final panoptic readout; void is predicted class 20.
    raw_sem=out['semantic_scores'][0,v].argmax(1)
    raw_sem=torch.where(alpha>.05,raw_sem,torch.full_like(raw_sem,20))
    conf=np.zeros((20,21),np.int64)
    for c in range(20):
        for pr in range(21):conf[c,pr]=int(((sem==c)&(raw_sem==pr)&valid).sum())
    ious=[]
    for c in range(20):
        t=conf[c,c];den=conf[c,:].sum()+conf[:,c].sum()-t
        if den:ious.append(t/den)
    pan_conf=np.zeros((20,21),np.int64)
    for c in range(20):
        for pr in range(21):pan_conf[c,pr]=int(((sem==c)&(pan_sem==pr)&valid).sum())
    pan_ious=[]
    for c in range(20):
        t=pan_conf[c,c];den=pan_conf[c,:].sum()+pan_conf[:,c].sum()-t
        if den:pan_ious.append(t/den)
    # Context Hungarian classification accuracy is reported independently of scope masks.
    from tokengs.models.object_locus_v3_set_loss import final_hungarian
    targets,pairs=final_hungarian(out,batch)
    corr=nmatch=0
    confusion=np.zeros((18,19),dtype=np.int64)
    matched_by_id={}
    for b,(qi,ki) in enumerate(pairs):
        if qi.numel():
            pred=out['states'][-1]['thing_logits19'][b,qi].argmax(-1)
            gt=targets['gt_classes'][b][ki]-2
            corr+=int((pred==gt).sum());nmatch+=int(qi.numel())
            for jj,(qidx,kidx) in enumerate(zip(qi.tolist(),ki.tolist())):
                gtid=int(targets['gt_instance_ids'][b][kidx]);pr=int(pred[jj]);g=int(gt[jj]);confusion[g,pr]+=1
                matched_by_id[gtid]={'matched_query':qidx,'matched_class':pr+2 if pr<18 else None,'matched_class_correct':bool(pr==g)}
    for j,row in enumerate(per_gt):
        row.update(matched_by_id.get(row['instance_id'],{'matched_query':None,'matched_class':None,'matched_class_correct':None}))
        row['best_eligible_candidate_iou']=float(iou[:,j].max()) if len(predictions) else 0.
        row['best_panoptic_iou']=float(pious[:,j].max()) if pious.shape[0] else 0.
        row['best_panoptic_query']=int(pious[:,j].argmax()) if pious.shape[0] else None
        gt_query_rows[j]['panoptic_iou']=row['best_panoptic_iou']
    raw_query_iou=np.zeros((100,len(gtrows)),float)
    for q in range(100):
        for j,gmask in enumerate(gt_masks):
            pmask=raw_all[:,q]
            inter=int((pmask&gmask).sum());union=int((pmask|gmask).sum())
            raw_query_iou[q,j]=inter/union if union else 0.
    query_rows=[]
    for q in range(100):
        raw_area=int(raw_all[:,q].sum());eligible_q=bool(eligible[q]);candidate_area=int(raw[:,q].sum())
        pan_area=int((pan_ins==(q+1)).sum());class_idx=int(cls0[q]);matched_gt=next((r for r in per_gt if r.get('matched_query')==q),None)
        raw_iou=float(raw_query_iou[q].max()) if len(gtrows) else 0.
        pan_iou=max((float(pious[q,j]) for j in range(len(gtrows))),default=0.)
        reason=None if pan_area else ('ineligible' if not eligible_q else ('empty_raw_mask' if raw_area==0 else 'lost_or_removed_by_panoptic_competition'))
        query_rows.append({'query_id':q,'predicted_class':class_idx+2,'joint_class_probability':float(bestprob[q]),
          'raw_mask_area':raw_area,'candidate_mask_area':candidate_area,'won_area':pan_area,
          'joint_eligible':eligible_q,'independent_candidate':candidate_area>0,'panoptic_retained':pan_area>0,
          'removed_reason':reason,'matched_gt_id':matched_gt['instance_id'] if matched_gt else None,
          'raw_iou':raw_iou,'panoptic_iou':pan_iou})
    from scripts.eval_object_locus_v2_1 import _panoptic_pq
    pq=_panoptic_pq(pan_sem,pan_ins,sem,ins)
    return {'valid_pixels':int(valid.sum()),'gt_count':len(gtrows),'candidate_count':len(predictions),
      'candidate_ca':{'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)},
      'candidate_cw':{'tp':ctp,'fp':cfp,'fn':cfn,'precision':ctp/max(1,ctp+cfp),'recall':ctp/max(1,ctp+cfn)},
      'candidate_ap':{'map':None,'map_50':None,'status':'aggregated across split'},
      '_map_payload':({'pred':{'masks':pm,'labels':torch.tensor([x[1] for x in predictions],device=sem.device),'scores':torch.tensor([x[2] for x in predictions],device=sem.device)},
                       'target':{'masks':gt_masks_tensor,'labels':torch.tensor(gtc,device=sem.device)}}),
      'raw_best_iou_ge_0_5_fraction':sum(x>=.5 for x in raw_best)/max(1,len(raw_best)),
      'raw_best_ious':raw_best,'per_gt':per_gt,
      'panoptic_ca':{'tp':pca,'fp':max(0,int(torch.unique(pan_ins[pan_ins>0]).numel())-pca),'fn':len(gtrows)-pca},
      'panoptic_cw':{'tp':pcw,'fp':max(0,int(torch.unique(pan_ins[pan_ins>0]).numel())-pcw),'fn':len(gtrows)-pcw},
      'matched_19_class_accuracy':corr/max(1,nmatch),'matched_gt_count':nmatch,
      'classification_confusion':confusion.tolist(),'semantic_confusion':conf.tolist(),
      'panoptic_semantic_confusion':pan_conf.tolist(),
      'semantic_miou':float(np.mean(ious)) if ious else 0.,'panoptic_pq':pq['mean_pq'],
      'panoptic_semantic_miou':float(np.mean(pan_ious)) if pan_ious else 0.,
      'thing_miou':float(np.mean([conf[c,c]/max(1,conf[c,:].sum()+conf[:,c].sum()-conf[c,c]) for c in range(2,20)])),
      'stuff_miou':float(np.mean([conf[c,c]/max(1,conf[c,:].sum()+conf[:,c].sum()-conf[c,c]) for c in (0,1)])),
      'panoptic_semantic':pan_sem.detach().cpu().numpy(),'panoptic_instance':pan_ins.detach().cpu().numpy(),
      'raw_masks':raw.detach().cpu().numpy(),'gt_semantic':sem.detach().cpu().numpy(),
      'gt_instance':ins.detach().cpu().numpy(),'predictions':predictions,'query_rows':query_rows}

def _run(model,opt,window,batch_builder,device):
    from tokengs.models.input_types import ModelInput,ModelInputDecoder,split_data
    batch=batch_builder(opt,window,device); mi,_=split_data(batch,opt)
    dec=ModelInputDecoder(cam_view=batch['cam_view_all'],intrinsics=batch['intrinsics_all'])
    out=model.forward_object_locus(ModelInput(mi.encoder,dec),render_decoder_input=dec,context_decoder=dec)
    return batch,out

def evaluate_windows(model,opt,windows,step,split,reports,device,batch_builder,*,official=False,panels=False):
    from scripts.object_locus_v3_set_runtime import capture_rng,restore_rng,write_json
    from scripts.export_object_locus_v3_set_official import export_windows
    from scripts.eval_object_locus_v1 import _official_run
    was=model.training;rng=capture_rng();model.eval();rows=[]; per_gt=[];query_rows=[]
    try:
      from torchmetrics.detection.mean_ap import MeanAveragePrecision
      ap_metrics={s:MeanAveragePrecision(iou_type='segm') for s in ('context','target_all','novel')}
    except Exception as exc:
      ap_metrics={};ap_error=f'{type(exc).__name__}: {exc}'
    try:
      with torch.no_grad():
       for wi,win in enumerate(windows):
        batch,out=_run(model,opt,win,batch_builder,device)
        frame_ids=[int(x) for x in batch['frame_ids'][0].cpu().tolist()]
        ctx=[0,1]; target=list(range(len(frame_ids))); novel=[i for i,x in enumerate(frame_ids) if x in set(map(int,win['novel']))]
        scopes={}
        for name,idx in [('context',ctx),('target_all',target),('novel',novel)]:
         row=_candidate_stats(out,batch,idx)
         if ap_metrics:
          payload=row.pop('_map_payload')
          for branch in ('pred','target'):
           pmask=payload[branch]['masks']
           payload[branch]['masks']=pmask.reshape(pmask.shape[0],-1,pmask.shape[-1]) if pmask.shape[0] else torch.zeros((0,len(idx)*256,256),dtype=torch.bool,device=pmask.device)
          ap_metrics[name].update([payload['pred']],[payload['target']])
         views=len(idx);mse=(out['render']['images_pred'][0,idx]-batch['images_all'][0,idx]).square().mean().clamp_min(1e-12)
         row['psnr']=float((-10*torch.log10(mse)).cpu());row['scope']=name;scopes[name]=row
         per_gt.extend({'split':split,'step':step,'scope':name,'scene':win['scene'],**x} for x in row['per_gt'])
         query_rows.extend({'split':split,'step':step,'scope':name,'scene':win['scene'],**x} for x in row['query_rows'])
        rows.append({'scene':win['scene'],'context':win['context'],'novel':win['novel'],'scopes':scopes})
        if panels and wi<2: write_panel(batch,out,win,Path(reports)/f'qualitative/step_{step:04d}/{split}/pair{wi}.png',f'{step} {split}')
    finally:restore_rng(rng);model.train(was)
    aggregated={}
    for scope in ('context','target_all','novel'):
      rr=[x['scopes'][scope] for x in rows]
      conf=np.sum([np.asarray(x['semantic_confusion']) for x in rr],axis=0)
      panconf=np.sum([np.asarray(x['panoptic_semantic_confusion']) for x in rr],axis=0)
      def _ious(matrix, classes):
        values=[]
        for c in classes:
          tp=matrix[c,c];den=matrix[c,:].sum()+matrix[:,c].sum()-tp
          if den:values.append(float(tp/den))
        return values
      all_iou=_ious(conf,range(20));thing_iou=_ious(conf,range(2,20));stuff_iou=_ious(conf,(0,1))
      pan_all_iou=_ious(panconf,range(20))
      agg={'windows':len(rr),'gt_count':sum(x['gt_count'] for x in rr),'semantic_confusion':conf.tolist(),
        'panoptic_semantic_confusion':panconf.tolist(),
        'semantic_miou':float(np.mean(all_iou)) if all_iou else 0.,'mIoU_thing':float(np.mean(thing_iou)) if thing_iou else 0.,
        'mIoU_stuff':float(np.mean(stuff_iou)) if stuff_iou else 0.,'psnr':float(np.mean([x['psnr'] for x in rr])),
        'candidate_ca':{k:sum(x['candidate_ca'][k] for x in rr) for k in ('tp','fp','fn')},
        'candidate_cw':{k:sum(x['candidate_cw'][k] for x in rr) for k in ('tp','fp','fn')},
        'raw_best_iou_ge_0_5_fraction':sum(sum(x['raw_best_iou_ge_0_5_fraction']*len(x['raw_best_ious']) for x in rr) for x in []) if False else sum(sum(v>=.5 for v in x['raw_best_ious']) for x in rr)/max(1,sum(len(x['raw_best_ious']) for x in rr)),
        'matched_19_class_accuracy':sum(x['matched_19_class_accuracy']*x['matched_gt_count'] for x in rr)/max(1,sum(x['matched_gt_count'] for x in rr)),
        'panoptic_pq':float(np.mean([x['panoptic_pq'] for x in rr])),
        'panoptic_semantic_miou':float(np.mean(pan_all_iou)) if pan_all_iou else 0.,
        'classification_confusion':np.sum([np.asarray(x['classification_confusion']) for x in rr],axis=0).tolist(),
        'matched_gt_count':sum(x['matched_gt_count'] for x in rr),'candidate_count':sum(x['candidate_count'] for x in rr),
        'panoptic_ca':{k:sum(x['panoptic_ca'][k] for x in rr) for k in ('tp','fp','fn')},
        'panoptic_cw':{k:sum(x['panoptic_cw'][k] for x in rr) for k in ('tp','fp','fn')},
        'per_class_iou':{str(c):float(conf[c,c]/max(1,conf[c,:].sum()+conf[:,c].sum()-conf[c,c])) for c in range(20)}}
      for k in ('candidate_ca','candidate_cw','panoptic_ca','panoptic_cw'):
       d=agg[k];d['precision']=d['tp']/max(1,d['tp']+d['fp']);d['recall']=d['tp']/max(1,d['tp']+d['fn'])
      try:
       if ap_metrics:
        m=ap_metrics[scope].compute();agg['candidate_ap']={'map':float(m['map']),'map_50':float(m['map_50'])}
       else:agg['candidate_ap']={'error':ap_error}
      except Exception as exc:agg['candidate_ap']={'error':f'{type(exc).__name__}: {exc}'}
      aggregated[scope]=agg
    result={'step':step,'split':split,'local':aggregated,'windows':[{k:v for k,v in x.items() if k!='scopes'}|{'scopes':{s:{k:v for k,v in x['scopes'][s].items() if k not in ('semantic_confusion','panoptic_semantic_confusion','panoptic_semantic','panoptic_instance','raw_masks','gt_semantic','gt_instance','predictions','per_gt')} for s in x['scopes']}} for x in rows]}
    if official:
      root=Path(reports)/f'official/step_{step:04d}/{split}'
      allx=export_windows(model,opt,windows,root/'all',device=device,batch_builder=batch_builder,target_frames='all')
      nov=export_windows(model,opt,windows,root/'novel',device=device,batch_builder=batch_builder,target_frames='novel')
      all_json=_official_run(root/'all',root/'official_all.json');nov_json=_official_run(root/'novel',root/'official_novel.json')
      result['official']={'all':all_json.get('result'),'novel':nov_json.get('result')}
    write_json(Path(reports)/f'eval_{split}_step{step:04d}.json',result)
    return result,per_gt,query_rows

def write_panel(batch,out,window,path,title):
    ids=[0,1];sem=batch['semantic_label_all'][0,ids].cpu().numpy();ins=batch['instance_label_all'][0,ids].cpu().numpy()
    predsem,predins,_=__import__('scripts.export_object_locus_v3_set_official',fromlist=['assemble_panoptic']).assemble_panoptic(out)
    imgs=[];labels=[]
    for v in ids:
      rgb=(batch['images_all'][0,v].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
      rec=(out['render']['images_pred'][0,v].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
      gt=np.zeros((*sem[v].shape,3),np.uint8);pr=gt.copy();raw=gt.copy()
      for c in range(20):gt[sem[v]==c]=((37*c+53)%255,(97*c+31)%255,(173*c+71)%255);pr[predsem[v].cpu().numpy()==c]=((37*c+53)%255,(97*c+31)%255,(173*c+71)%255)
      for q in range(100):
       if int(out['p_class'][0,q].argmax())==18:continue
       mask=(out['region_mass'][0,v,q]>=.5)&(out['alpha'][0,v,0]>.05);raw[mask.cpu().numpy()]=((31*q+71)%255,(67*q+137)%255,(131*q+29)%255)
      imgs += [rgb,rec,gt,pr,raw];labels += ['RGB GT','RGB rec','GT semantic','panoptic semantic','raw candidates']
    sz=192;can=Image.new('RGB',(sz*5,40+sz*2),'white');dr=ImageDraw.Draw(can);dr.text((4,4),title,fill='black')
    for i,im in enumerate(imgs):can.paste(Image.fromarray(im).resize((sz,sz)),((i%5)*sz,40+(i//5)*sz));dr.text(((i%5)*sz+3,22+(i//5)*0),labels[i],fill='black')
    Path(path).parent.mkdir(parents=True,exist_ok=True);can.save(path)
