#!/usr/bin/env python3
"""One-pass, read-only step-3500 anchor/mask/depth audit for Object-Locus V1."""
from __future__ import annotations

import csv, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from scripts.object_locus_v1_runtime import build_batch, build_model
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.object_locus_v1_loss import build_visible_anchor_targets, final_hungarian
from tokengs.models.anchor_group_loss import project_points, THING, WALL, FLOOR, IGNORE, resolve_anchor_observations
from scripts.export_object_locus_v1_official import assemble_panoptic

OUT = Path('/space/mawb/ssst/group_plus/object_locus_v1_review_bundle')
REPORTS = Path('/space/mawb/ssst/group_plus/object_locus_v1')
RUN = Path('/space/mawb/ssst/workspace_group_plus/object_locus_v1')
CKPT = RUN / 'checkpoints/step_00003500/train_state.pt'
MIN_AREA, ALPHA_MIN, IOU_TP = 50, .05, .5

def json_write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def iou(a, b):
    inter = int((a & b).sum()); union = int((a | b).sum())
    return inter / union if union else 0.0

def gts_for(sem, ins):
    out = {}
    for v in range(sem.shape[0]):
        thing = (sem[v] >= 2) & (sem[v] <= 19) & (ins[v] > 0)
        for iid in torch.unique(ins[v][thing]).tolist():
            cls = int(torch.mode(sem[v][thing & (ins[v] == iid)]).values)
            out.setdefault((cls, int(iid)), torch.zeros_like(sem, dtype=torch.bool))[v] = thing & (ins[v] == iid)
    return out

def visible_iou(pred, gt, q, views):
    vis = [v for v in views if int(gt[v].sum()) >= MIN_AREA]
    if not vis: return None
    ys = torch.cat([gt[v].reshape(-1) for v in vis])
    ps = []
    for v in vis:
        m = pred[v, q]
        ps.append((m if int(m.sum()) >= MIN_AREA else torch.zeros_like(m)).reshape(-1))
    return iou(torch.cat(ps), ys)

def greedy_recall(pred, cls, score, is_thing, gts, class_aware):
    keys = [k for k, m in gts.items() if any(int(m[v].sum()) >= MIN_AREA for v in range(m.shape[0]))]
    used, tp, fp = set(), 0, 0
    matched_by_query={}; fp_queries=[]
    for q in torch.argsort(score, descending=True, stable=True).tolist():
        if not bool(is_thing[q]): continue
        cand=[]
        for j,key in enumerate(keys):
            if j in used or (class_aware and int(cls[q]) != key[0]): continue
            val=visible_iou(pred,gts[key],q,range(pred.shape[0]))
            if val is not None and val >= IOU_TP: cand.append((val,j))
        if cand:
            j=max(cand)[1]; used.add(j); tp+=1; matched_by_query[int(q)]=keys[j]
        elif any(int(pred[v,q].sum())>=MIN_AREA for v in range(pred.shape[0])):
            fp+=1; fp_queries.append(int(q))
    return {'tp':tp,'fp':fp,'fn':len(keys)-len(used),'gt':len(keys),'recall':tp/max(1,len(keys)),
            'matched_gt_keys':[list(keys[j]) for j in sorted(used)],
            'matched_by_query':{str(q):list(key) for q,key in matched_by_query.items()},
            'fp_queries':fp_queries,'eligible_gt_keys':[list(k) for k in keys]}

def image_palette(sem):
    arr=np.zeros((*sem.shape,3),np.uint8)
    for c in range(20): arr[sem==c]=((37*c+53)%255,(97*c+31)%255,(173*c+71)%255)
    arr[(sem<0)|(sem>19)]=(40,40,40)
    return arr

def sample_idx(length, maxn=12000):
    stride=max(1, length//maxn)
    return torch.arange(0,length,stride)

def depth_roundtrip(batch, view):
    # Cameras and anchor geometry use the provider's scene-scaled coordinate system.
    dep=batch['depth_gt_scene_all'][0,view,0]
    valid=batch['depth_gt_valid_all'][0,view,0] & (dep>0)
    H,W=dep.shape
    flat=torch.nonzero(valid.reshape(-1),as_tuple=False).flatten()
    flat=flat[sample_idx(flat.numel())]
    vv=(flat//W).float(); uu=(flat%W).float(); z=dep.reshape(-1)[flat]
    fx,fy,cx,cy=batch['intrinsics_all'][0,view]
    cam=torch.stack(((uu-cx)*z/fx,(vv-cy)*z/fy,z),-1)
    cv=batch['cam_view_all'][0,view].transpose(0,1)
    c2w=torch.linalg.inv(cv)
    world=cam @ c2w[:3,:3].T + c2w[:3,3]
    recovered=(world-c2w[:3,3]) @ c2w[:3,:3]
    zr=recovered[:,2]; ur=fx*recovered[:,0]/zr+cx; vr=fy*recovered[:,1]/zr+cy
    return {'sample_count':int(flat.numel()),'pixel_abs_error':{'mean':float(torch.sqrt((ur-uu)**2+(vr-vv)**2).mean()),'p95':float(torch.quantile(torch.sqrt((ur-uu)**2+(vr-vv)**2),.95)),'max':float(torch.sqrt((ur-uu)**2+(vr-vv)**2).max())},'depth_abs_error_scene_units':{'mean':float((zr-z).abs().mean()),'p95':float(torch.quantile((zr-z).abs(),.95)),'max':float((zr-z).abs().max())}}

def write_alignment_image(batch, layer_mu, window, path):
    # First two context frames: depth unprojection/reprojection points overlaid on RGB,
    # with the same image's depth and semantic maps alongside for registration inspection.
    tiles=[]; labels=[]
    for v in range(2):
        rgb=(batch['images_all'][0,v].cpu().numpy().transpose(1,2,0).clip(0,1)*255).astype(np.uint8)
        dep=batch['depth_gt_m_all'][0,v,0].cpu().numpy(); sem=batch['semantic_label_all'][0,v].cpu().numpy()
        valid=batch['depth_gt_valid_all'][0,v,0].cpu().numpy()
        dnorm=np.zeros_like(rgb); vv=dep[valid]
        if vv.size:
            lo,hi=np.quantile(vv,[.02,.98]); g=np.clip((dep-lo)/max(hi-lo,1e-6),0,1)*255
            dnorm[:]=g[...,None].astype(np.uint8); dnorm[~valid]=(10,10,10)
        semrgb=image_palette(sem)
        overlay=Image.fromarray(rgb); draw=ImageDraw.Draw(overlay)
        u,pv,z=project_points(layer_mu, batch['cam_view_all'][0,v], batch['intrinsics_all'][0,v])
        ok=torch.isfinite(u)&torch.isfinite(pv)&torch.isfinite(z)&(z>0)&(u>=0)&(u<rgb.shape[1])&(pv>=0)&(pv<rgb.shape[0])
        ai=torch.nonzero(ok).flatten()[::max(1,int(ok.sum())//250)]
        xx=u[ai].long().cpu().tolist(); yy=pv[ai].long().cpu().tolist()
        for x,y,a in zip(xx,yy,ai.tolist()):
            good=bool(valid[y,x]) and abs(float(z[a])-float(dep[y,x]*.15)) <= .10*float(dep[y,x]*.15)
            col='lime' if good else 'red'
            draw.ellipse((x-2,y-2,x+2,y+2),outline=col,width=1)
        tiles.extend([np.asarray(overlay),dnorm,semrgb]); labels.extend([f'frame {int(batch["frame_ids"][0,v])} RGB + projected anchors', 'GT depth (m)', 'GT semantic'])
    # Cross-view registration check: backproject source context depth, project into
    # the other context camera, and compare projected z with its GT depth.
    src=0; dst=1
    dep=batch['depth_gt_m_all'][0,src,0]; valid=batch['depth_gt_valid_all'][0,src,0]&(dep>0)
    H,W=dep.shape; flat=torch.nonzero(valid.reshape(-1),as_tuple=False).flatten()
    flat=flat[sample_idx(flat.numel(),6000)]
    vv=(flat//W).float(); uu=(flat%W).float(); z=dep.reshape(-1)[flat]*.15
    fx,fy,cx,cy=batch['intrinsics_all'][0,src]
    cam=torch.stack(((uu-cx)*z/fx,(vv-cy)*z/fy,z),-1)
    c2w0=torch.linalg.inv(batch['cam_view_all'][0,src].transpose(0,1))
    world=cam@c2w0[:3,:3].T+c2w0[:3,3]
    u1,v1,z1=project_points(world,batch['cam_view_all'][0,dst],batch['intrinsics_all'][0,dst])
    inside=torch.isfinite(u1)&torch.isfinite(v1)&torch.isfinite(z1)&(z1>0)&(u1>=0)&(u1<W)&(v1>=0)&(v1<H)
    ids=torch.nonzero(inside).flatten(); x1=u1[ids].long(); y1=v1[ids].long()
    dep1=batch['depth_gt_scene_all'][0,dst,0,y1,x1]
    dv1=batch['depth_gt_valid_all'][0,dst,0,y1,x1]&(dep1>0)
    ok=dv1&((z1[ids]-dep1).abs()<=.10*dep1)
    rgb1=(batch['images_all'][0,dst].cpu().numpy().transpose(1,2,0).clip(0,1)*255).astype(np.uint8)
    overlay=Image.fromarray(rgb1); draw=ImageDraw.Draw(overlay)
    for x,y,good in zip(x1.cpu().tolist(),y1.cpu().tolist(),ok.cpu().tolist()):
        draw.ellipse((x-1,y-1,x+1,y+1),outline=('lime' if good else 'red'),width=1)
    tiles.append(np.asarray(overlay)); labels.append(f'frame {int(batch["frame_ids"][0,src])} depth projected to frame {int(batch["frame_ids"][0,dst])} RGB; green=depth-consistent')
    dep1rgb=np.zeros_like(rgb1); target_depth=batch['depth_gt_m_all'][0,dst,0].cpu().numpy(); target_valid=batch['depth_gt_valid_all'][0,dst,0].cpu().numpy()
    dvv=target_depth[target_valid]
    if dvv.size:
        dlo,dhi=np.quantile(dvv,[.02,.98]); dg=np.clip((target_depth-dlo)/max(dhi-dlo,1e-6),0,1)*255
        dep1rgb[:]=dg[...,None].astype(np.uint8);dep1rgb[~target_valid]=(10,10,10)
    depth_overlay=Image.fromarray(dep1rgb);ddraw=ImageDraw.Draw(depth_overlay)
    for x,y,good in zip(x1.cpu().tolist(),y1.cpu().tolist(),ok.cpu().tolist()):
        ddraw.ellipse((x-1,y-1,x+1,y+1),outline=('lime' if good else 'red'),width=1)
    tiles.append(np.asarray(depth_overlay));labels.append(f'frame {int(batch["frame_ids"][0,dst])} GT depth + reprojected samples')
    sem_overlay=Image.fromarray(image_palette(batch['semantic_label_all'][0,dst].cpu().numpy()));sdraw=ImageDraw.Draw(sem_overlay)
    for x,y,good in zip(x1.cpu().tolist(),y1.cpu().tolist(),ok.cpu().tolist()):
        sdraw.ellipse((x-1,y-1,x+1,y+1),outline=('lime' if good else 'red'),width=1)
    tiles.append(np.asarray(sem_overlay));labels.append(f'frame {int(batch["frame_ids"][0,dst])} semantic GT + reprojected samples')
    h,w=tiles[0].shape[:2]; cols=4; rows=math.ceil(len(tiles)/cols); canvas=Image.new('RGB',(cols*w,rows*h+46*rows),'white'); d=ImageDraw.Draw(canvas)
    for i,(tile,label) in enumerate(zip(tiles,labels)):
        x=(i%cols)*w;y=(i//cols)*(h+46)+36;canvas.paste(Image.fromarray(tile),(x,y));d.text((x+5,(i//cols)*(h+46)+5),label,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    err=(z1[ids][dv1]-dep1[dv1]).abs()/dep1[dv1]
    pixerr=torch.sqrt((u1[ids][dv1]-x1[dv1].float())**2+(v1[ids][dv1]-y1[dv1].float())**2)
    return {'source_depth_samples':int(flat.numel()),'target_in_bounds':int(ids.numel()),
            'target_valid_depth':int(dv1.sum()),'target_depth_consistent_10pct':int(ok.sum()),
            'target_depth_consistency_fraction':float(ok.sum()/max(1,dv1.sum())),
            'relative_depth_error':{'median':float(err.median()) if err.numel() else None,'p90':float(torch.quantile(err,.9)) if err.numel() else None,'max':float(err.max()) if err.numel() else None},
            'projected_pixel_rounding_error':{'mean':float(pixerr.mean()) if pixerr.numel() else None,'p95':float(torch.quantile(pixerr,.95)) if pixerr.numel() else None}}

def stage_stats(state,batch,targets):
    mu=state['mu'][0].detach(); H,W=batch['semantic_label_all'].shape[-2:]
    T=mu.shape[0]; observations=[[[] for _ in range(2)] for _ in range(T)]
    per_view=[]; depth_rel=[]; stage_masks=[torch.zeros(T,dtype=torch.bool,device=mu.device) for _ in range(6)]
    thing_masks=[torch.zeros_like(stage_masks[0]) for _ in range(6)]
    cats={'zero_trusted':0,'one_trusted':0,'two_consistent':0,'two_conflict':0}
    pre_instance={}; depth_instance={}
    for v in range(2):
        u,p,z=project_points(mu,batch['cam_view_all'][0,v],batch['intrinsics_all'][0,v])
        finite=torch.isfinite(u)&torch.isfinite(p)&torch.isfinite(z)
        positive=finite&(z>0); inside=positive&(u>=0)&(u<W)&(p>=0)&(p<H)
        stage_masks[0][:]=True; stage_masks[1]|=positive; stage_masks[2]|=inside
        ids=torch.nonzero(inside).flatten(); px=u[ids].long(); py=p[ids].long()
        dv=batch['depth_gt_valid_all'][0,v,0,py,px]&(batch['depth_gt_scene_all'][0,v,0,py,px]>0)
        stage_masks[3][ids[dv]]=True
        gd=batch['depth_gt_scene_all'][0,v,0,py,px]; zz=z[ids]
        cons=dv&((zz-gd).abs()<=.10*gd)
        stage_masks[4][ids[cons]]=True
        if bool(dv.any()): depth_rel.append(((zz[dv]-gd[dv]).abs()/gd[dv]).detach())
        sem=batch['semantic_label_all'][0,v,py,px]; ins=batch['instance_label_all'][0,v,py,px]
        trusted_sem=((sem==0)|(sem==1)|((sem>=2)&(sem<=19)&(ins>0)))
        trusted=cons&trusted_sem
        candidate=(sem>=2)&(sem<=19)&(ins>0)
        for stage, gate in ((1,torch.ones_like(candidate)),(2,torch.ones_like(candidate)),
                            (3,dv),(4,cons),(5,trusted)):
            thing_masks[stage] |= torch.zeros_like(inside).scatter(0,ids,gate&candidate)
        for local,aid in enumerate(ids.tolist()):
            if bool(trusted[local]):
                s=int(sem[local]);iid=int(ins[local]); obs=(WALL,0,0) if s==0 else ((FLOOR,1,0) if s==1 else (THING,s,iid))
                observations[aid][v].append(obs)
            if bool(inside[aid]):
                s0=int(sem[local]); iid0=int(ins[local])
                if 2<=s0<=19 and iid0>0:
                    pre_instance.setdefault((s0,iid0),set()).add(aid)
                    if bool(cons[local]): depth_instance.setdefault((s0,iid0),set()).add(aid)
        per_view.append({'view':v,'finite_positive':int(positive.sum()),'inside':int(inside.sum()),'depth_valid':int(dv.sum()),'depth_consistent':int(cons.sum()),'trusted':int(trusted.sum())})
    kinds=[]
    for a in range(T):
        obs=[o for v in range(2) for o in observations[a][v]]
        if len(obs)==0: cats['zero_trusted']+=1
        elif len(obs)==1: cats['one_trusted']+=1
        elif len(obs)==2 and obs[0]==obs[1]: cats['two_consistent']+=1
        elif len(obs)==2: cats['two_conflict']+=1
        else: # repeated projections are not expected: each anchor has at most one per view
            cats['two_conflict']+=1
        k,_,_=resolve_anchor_observations(obs); kinds.append(k)
    kind=torch.tensor(kinds,device=mu.device)
    stage_masks[5]=(kind==THING)|(kind==WALL)|(kind==FLOOR)
    # Candidate thing counts at each stage use positive thing observations at that gate.
    # At final stage use the production consensus labels.
    thing_masks[5]=kind==THING
    thing_counts=[1024,int(thing_masks[1].sum()),int(thing_masks[2].sum()),int(thing_masks[3].sum()),int(thing_masks[4].sum()),int(thing_masks[5].sum())]
    all_counts=[1024,int(stage_masks[1].sum()),int(stage_masks[2].sum()),int(stage_masks[3].sum()),int(stage_masks[4].sum()),int(stage_masks[5].sum())]
    rel=torch.cat(depth_rel) if depth_rel else torch.empty(0,device=mu.device)
    support=targets['Y_anchor'][0].sum(-1)>0
    gt=[]
    for k,(cls,iid) in enumerate(zip(targets['gt_classes'][0].tolist(),targets['gt_instance_ids'][0].tolist())):
        area=[]
        for v in range(2):
            sem=batch['semantic_label_all'][0,v]; ins=batch['instance_label_all'][0,v]
            area.append(int(((sem==cls)&(ins==iid)).sum()))
        gt.append({'class':int(cls),'instance_id':int(iid),'context_pixel_area_by_view':area,
                   'projected_anchor_count_pre_depth':len(pre_instance.get((int(cls),int(iid)),set())),
                   'anchor_count_depth_filtered':len(depth_instance.get((int(cls),int(iid)),set())),
                   'anchor_count_consensus':int(targets['Y_anchor'][0,k].sum()),'has_support':bool(support[k])})
    depth_summary={'projected_valid_depth_observations':int(rel.numel()),'relative_error':None}
    if rel.numel(): depth_summary['relative_error']={'median':float(rel.median()),'p50':float(torch.quantile(rel,.5)),'p90':float(torch.quantile(rel,.9)),'p95':float(torch.quantile(rel,.95)),'p99':float(torch.quantile(rel,.99)),'max':float(rel.max()),'pass_fraction_10pct':float((rel<=.10).float().mean())}
    return {'all_anchor_stage_counts':dict(zip(['all_1024','projected_finite_zpositive','image_bounds','gt_depth_valid','depth_consistent_10pct','two_view_consensus_valid'],all_counts)),
            'thing_candidate_stage_counts':dict(zip(['all_1024','projected_finite_zpositive','image_bounds','gt_depth_valid','depth_consistent_10pct','two_view_consensus_thing'],thing_counts)),
            'consensus_observation_categories':cats,'per_view_filter_counts':per_view,'depth_error':depth_summary,'per_gt':gt,
            'anchor_valid_count':int(stage_masks[5].sum()),'thing_anchor_count':int((kind==THING).sum()),'wall_anchor_count':int((kind==WALL).sum()),'floor_anchor_count':int((kind==FLOOR).sum()),'ignore_anchor_count':int((kind==IGNORE).sum()),
            'gt_with_anchor_support':int(support.sum()),'gt_without_anchor_support':int((~support).sum()),
            '_depth_error_values':rel.detach().cpu()}

def mask_stats(out,batch,split,window):
    targets,pairs=final_hungarian(out,batch)
    sem=batch['semantic_label_all'][0,:2];ins=batch['instance_label_all'][0,:2]
    gts=gts_for(sem,ins); region=out['region_mass'][0,:2,:100]
    alpha=out['alpha'][0,:2,0]; raw=region>.5
    final=raw&(alpha[:,None]>.05)
    pred_sem,pred_ins,_=assemble_panoptic(out)
    p=out['p_class'][0]; cls=p[:,:18].argmax(-1)+2; score=p[:,:18].sum(-1); isthing=score>=.5
    iou_rows=[]; best_vals=[]; pergt=[]; cls_correct=cls_total=0; matched=[]
    qi,ki=pairs[0]
    for q,k in zip(qi.tolist(),ki.tolist()):
        true=int(targets['gt_classes'][0][k]); best_index=int(p[q].argmax()); pred_label='no-object' if best_index==18 else best_index+2
        correct=(best_index<18 and best_index+2==true); cls_correct+=int(correct);cls_total+=1
        matched.append({'query':q,'gt_index':k,'gt_class':true,'pred_class':pred_label,'class_correct':correct,'no_object_probability':float(p[q,18]),'true_class_probability':float(p[q,true-2]),'score':float(score[q])})
    for gt_key,gtmask in gts.items():
        vals=[]
        for q in range(100):
            value=visible_iou(raw,gtmask,q,range(2)); vals.append(float(value or 0.0))
            iou_rows.append({'split':split,'scene':window['scene'],'context':window['context'],'class':gt_key[0],'instance_id':gt_key[1],'query':q,'raw_mask_iou':float(value or 0.0),'gt_has_anchor_support':None})
        bestq=int(np.argmax(vals)); best=vals[bestq]; best_vals.append(best)
        best_index=int(p[bestq].argmax());best_pred='no-object' if best_index==18 else best_index+2
        pergt.append({'class':gt_key[0],'instance_id':gt_key[1],'best_query':bestq,'best_raw_iou':best,
                      'best_slot_class':best_pred,'best_slot_class_correct':best_index<18 and best_index+2==gt_key[0],
                      'best_slot_thing_probability':float(score[bestq]),'best_slot_no_object_probability':float(p[bestq,18]),
                      'anchor_support':int(targets['Y_anchor'][0, list(targets['gt_instance_ids'][0].tolist()).index(gt_key[1])].sum()) if gt_key[1] in targets['gt_instance_ids'][0].tolist() else 0})
    pre_ca=greedy_recall(raw,cls,score,isthing,gts,False);pre_cw=greedy_recall(raw,cls,score,isthing,gts,True)
    alpha_ca=greedy_recall(final,cls,score,isthing,gts,False);alpha_cw=greedy_recall(final,cls,score,isthing,gts,True)
    post_pred=(pred_ins[:2]>0)
    # Convert final instance maps into 100 query masks for registered evaluator mask recall.
    post=torch.stack([torch.stack([pred_ins[v]==q+1 for q in range(100)]) for v in range(2)])
    post_ca=greedy_recall(post,cls,score,isthing,gts,False);post_cw=greedy_recall(post,cls,score,isthing,gts,True)
    class_metrics={}
    for c in range(2,20):
      ca_pairs=pre_ca['matched_by_query'];cw_pairs=pre_cw['matched_by_query']
      ca_tp=sum(1 for key in ca_pairs.values() if int(key[0])==c)
      cw_tp=sum(1 for key in cw_pairs.values() if int(key[0])==c)
      ca_fp=sum(1 for q in pre_ca['fp_queries'] if int(cls[q])==c)
      cw_fp=sum(1 for q in pre_cw['fp_queries'] if int(cls[q])==c)
      ng=sum(1 for key in pre_ca['eligible_gt_keys'] if int(key[0])==c)
      if ng or ca_fp or cw_fp:
        class_metrics[str(c)]={'gt':ng,'class_agnostic':{'tp':ca_tp,'fp':ca_fp,'fn':ng-ca_tp},
                               'class_aware':{'tp':cw_tp,'fp':cw_fp,'fn':ng-cw_tp}}
    classes=targets['gt_classes'][0]; gtids=targets['gt_instance_ids'][0]
    support_count={int(i):int(targets['Y_anchor'][0,k].sum()) for k,i in enumerate(gtids.tolist())}
    for row in iou_rows: row['gt_has_anchor_support']=support_count.get(row['instance_id'],0)>0
    bins={'lt_0.1':0,'0.1_to_0.25':0,'0.25_to_0.5':0,'ge_0.5':0}
    for x in best_vals: bins['lt_0.1' if x<.1 else ('0.1_to_0.25' if x<.25 else ('0.25_to_0.5' if x<.5 else 'ge_0.5'))]+=1
    class_counts={str(c):int((cls==c).sum()) for c in range(2,20)}
    return {'targets':targets,'pairs':pairs,'iou_rows':iou_rows,'summary':{'split':split,'scene':window['scene'],'context':window['context'],'gt_count':len(gts),
      'best_raw_iou_bins':bins,'best_raw_iou_mean':float(np.mean(best_vals)) if best_vals else None,
      'final_hungarian_matched_class_accuracy':cls_correct/max(1,cls_total),'matched_count':cls_total,'matched_rows':matched,
      'best_mask_slot_class_accuracy':float(np.mean([x['best_slot_class_correct'] for x in pergt])) if pergt else None,
      'gt_rows':pergt,'gt_class_histogram':{str(c):sum(1 for k in gts if k[0]==c) for c in range(2,20)},'class_instance_metrics':class_metrics,
      'query_predicted_class_counts':class_counts,'query_no_object_probability':{'mean':float(p[:,18].mean()),'median':float(p[:,18].median()),'p90':float(p[:,18].quantile(.9))},
      'query_true_class_probability':{'mean':float(p[:,:18].max(-1).values.mean()),'median':float(p[:,:18].max(-1).values.median()),'p90':float(p[:,:18].max(-1).values.quantile(.9))},
      'query_no_object_argmax_fraction':float((p.argmax(-1)==18).float().mean()),'class_agnostic_recall_raw_slot':pre_ca,'class_aware_recall_raw_slot':pre_cw,
      'class_agnostic_recall_after_alpha_gate_before_winner':alpha_ca,'class_aware_recall_after_alpha_gate_before_winner':alpha_cw,
      'class_agnostic_recall_after_evaluator_postprocess':post_ca,'class_aware_recall_after_evaluator_postprocess':post_cw,
      'class_agnostic_success_class_aware_failure_gt_count':len(set(map(tuple,pre_ca['matched_gt_keys']))-set(map(tuple,pre_cw['matched_gt_keys']))),
      'gt_support_iou':{'supported_mean':float(np.mean([r['best_raw_iou'] for r in pergt if r['anchor_support']>0])) if any(r['anchor_support']>0 for r in pergt) else None,
                        'unsupported_mean':float(np.mean([r['best_raw_iou'] for r in pergt if r['anchor_support']==0])) if any(r['anchor_support']==0 for r in pergt) else None},
      'active_slots':int(isthing.sum()),'predicted_instance_pixels':int((pred_ins[:2]>0).sum()),'raw_slot_pixels':int(raw.sum()),'post_alpha_raw_pixels':int(final.sum())}}

def main():
    if not torch.cuda.is_available(): raise RuntimeError('review audit requires CUDA')
    OUT.mkdir(parents=True,exist_ok=True)
    opt=__import__('scripts.object_locus_v1_runtime',fromlist=['build_options']).build_options()
    model,opt,_=build_model('cuda',opt=opt)
    blob=torch.load(CKPT,map_location='cpu',weights_only=False)
    model.load_state_dict(blob['model'],strict=True); model.eval()
    windows={}
    md=json.loads((REPORTS/'monitor_train16.json').read_text())
    windows['train16']=[{'scene':w['scene'],'context':w['context'],'novel':w['novel']} for w in md['windows']]
    vd=json.loads((REPORTS/'monitor_32pairs.json').read_text())
    windows['val32']=[{'scene':w['scene'],'context':w['context'],'novel':w['novel']} for w in vd['pairs']]
    rows=[]; per_window=[]; all_iou=[]; class_summaries=[]; depth_align={}; depth_error_vectors={}
    for split,items in windows.items():
      for wi,w in enumerate(items):
        batch=build_batch(opt,w,'cuda')
        mi,_=split_data(batch,opt); dec=ModelInputDecoder(cam_view=batch['cam_view_all'],intrinsics=batch['intrinsics_all'])
        with torch.no_grad():
          out=model.forward_object_locus(ModelInput(mi.encoder,dec),render_decoder_input=dec,context_decoder=ModelInputDecoder(cam_view=batch['cam_view_all'][:,:2],intrinsics=batch['intrinsics_all'][:,:2]),step=3500)
        layer_rows={}
        for layer in (6,8,10,12):
          state=next(s for s in out['states'] if int(s['layer'])==layer)
          target=build_visible_anchor_targets(state['mu'],batch)
          stage=stage_stats(state,batch,target); layer_rows[str(layer)]=stage
          depth_error_vectors.setdefault((split,layer),[]).append(stage['_depth_error_values'])
          rows.append({'split':split,'window_index':wi,'scene':w['scene'],'context':w['context'],'layer':layer,
             'anchor_valid_count':stage['anchor_valid_count'],'anchor_thing_count':stage['thing_anchor_count'],
             'anchor_wall_count':stage['wall_anchor_count'],'anchor_floor_count':stage['floor_anchor_count'],
             'anchor_ignore_count':stage['ignore_anchor_count'],'gt_with_anchor_support':stage['gt_with_anchor_support'],
             'gt_without_anchor_support':stage['gt_without_anchor_support'],
             'all_anchor_stage_counts':stage['all_anchor_stage_counts'],
             'thing_candidate_stage_counts':stage['thing_candidate_stage_counts'],
             'consensus_observation_categories':stage['consensus_observation_categories'],
             'per_view_filter_counts':stage['per_view_filter_counts'],'depth_error':stage['depth_error']})
          for gt in stage['per_gt']:
            prow={'split':split,'window_index':wi,'scene':w['scene'],'layer':layer,**gt,
                  'anchor_valid_count':stage['anchor_valid_count'],'anchor_thing_count':stage['thing_anchor_count'],
                  'anchor_wall_count':stage['wall_anchor_count'],'anchor_floor_count':stage['floor_anchor_count'],
                  'anchor_ignore_count':stage['ignore_anchor_count'],'gt_with_anchor_support':stage['gt_with_anchor_support'],
                  'gt_without_anchor_support':stage['gt_without_anchor_support']}
            for prefix,key in (('allstage','all_anchor_stage_counts'),('thingstage','thing_candidate_stage_counts'),('consensus','consensus_observation_categories')):
              prow.update({f'{prefix}_{name}':value for name,value in stage[key].items()})
            for view_row in stage['per_view_filter_counts']:
              for name in ('finite_positive','inside','depth_valid','depth_consistent','trusted'):
                prow[f'view{view_row["view"]}_{name}']=view_row[name]
            if stage['depth_error']['relative_error']:
              prow.update({f'depth_rel_{name}':value for name,value in stage['depth_error']['relative_error'].items()})
            per_window.append(prow)
        m=mask_stats(out,batch,split,w); all_iou.extend(m['iou_rows']);class_summaries.append(m['summary'])
        if split=='train16' and wi==0 or split=='val32' and wi==0:
          depth_align[split]={'scene':w['scene'],'context':w['context'],'frame_ids':batch['frame_ids'][0].cpu().tolist(),
             'roundtrip_by_context_view':[depth_roundtrip(batch,v) for v in range(2)],
             'units_and_alignment':{'raw_depth':'SIU3R processed uint16 depth PNG / 1000 -> meters','camera_translation':'raw camera-to-world translations multiplied by provider scene_scale 0.15 after first-camera normalization','scene_depth':'metric depth meters multiplied by same fixed 0.15','RGB_depth_panoptic_frame':'all loaded from same requested frame_id in SIU3RProcessedScanNet.get_data','labels_transform':'same center crop and nearest resize as RGB spatial transform','intrinsics_transform':'same crop shift and resize scale applied in Provider._preprocess','cross_view_overlay':'qualitative/depth_alignment_'+split+'.png'} }
          s6=next(s for s in out['states'] if int(s['layer'])==6)
          depth_align[split]['cross_view_depth_registration']=write_alignment_image(batch,s6['mu'][0].detach(),w,OUT/'qualitative'/('depth_alignment_'+split+'.png'))
        if (wi%8)==0: print(f'[review-forward] {split} {wi+1}/{len(items)}',flush=True)
        del out,batch,m
    # Aggregate per-window fixed-stage distributions by split/layer.
    summary={'checkpoint':str(CKPT),'checkpoint_step':3500,'splits':{},'depth_alignment':depth_align,'mask_classification':class_summaries}
    for split in windows:
      summary['splits'][split]={}
      for layer in (6,8,10,12):
        subset=[r for r in rows if r['split']==split and r['layer']==layer]
        keys=['anchor_valid_count','anchor_thing_count','anchor_wall_count','anchor_floor_count','anchor_ignore_count','gt_with_anchor_support','gt_without_anchor_support']
        stagekeys=('all_anchor_stage_counts','thing_candidate_stage_counts','consensus_observation_categories')
        summary['splits'][split][str(layer)]={'windows':len(subset),'means':{k:float(np.mean([r[k] for r in subset])) for k in keys},'totals':{k:int(sum(r[k] for r in subset)) for k in keys},
          'gt_support_fraction':sum(r['gt_with_anchor_support'] for r in subset)/max(1,sum(r['gt_with_anchor_support']+r['gt_without_anchor_support'] for r in subset)),
          'stage_totals':{name:{key:int(sum(r[name][key] for r in subset)) for key in subset[0][name]} for name in stagekeys},
          'consensus_observation_totals':{key:int(sum(r['consensus_observation_categories'][key] for r in subset)) for key in subset[0]['consensus_observation_categories']},
          'depth_error_window_summaries':[r['depth_error'] for r in subset],
          'per_view_filter_totals':[{'view':v,'finite_positive':sum(r['per_view_filter_counts'][v]['finite_positive'] for r in subset),
             'inside':sum(r['per_view_filter_counts'][v]['inside'] for r in subset),
             'depth_valid':sum(r['per_view_filter_counts'][v]['depth_valid'] for r in subset),
             'depth_consistent':sum(r['per_view_filter_counts'][v]['depth_consistent'] for r in subset),
             'trusted':sum(r['per_view_filter_counts'][v]['trusted'] for r in subset)} for v in range(2)]}
        vals=torch.cat(depth_error_vectors[(split,layer)])
        summary['splits'][split][str(layer)]['depth_error_overall']={
          'projected_valid_depth_observations':int(vals.numel()),
          'relative_error_quantiles':{str(q):float(torch.quantile(vals,q)) for q in (.5,.9,.95,.99)},
          'relative_error_max':float(vals.max()) if vals.numel() else None,
          'depth_consistency_pass_fraction_10pct':float((vals<=.10).float().mean()) if vals.numel() else None}
    json_write(OUT/'anchor_supervision_summary.json',summary)
    with (OUT/'anchor_supervision_per_window.csv').open('w',newline='') as f:
      cols=list(dict.fromkeys(k for row in per_window for k in row));wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(per_window)
    with (OUT/'gt_slot_iou.csv').open('w',newline='') as f:
      cols=['split','scene','context','class','instance_id','query','raw_mask_iou','gt_has_anchor_support'];wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(all_iou)
    json_write(OUT/'classification_summary.json',{'checkpoint_step':3500,'by_window':class_summaries})
    print('[review-audit] complete',flush=True)

if __name__=='__main__': main()
