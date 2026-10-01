"""Task evaluation for V2 independent Gaussian memberships."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _metrics_helpers():
    from scripts.eval_object_locus_v1 import _legacy_evaluator
    return _legacy_evaluator()


def _panoptic_pq(pred_sem, pred_ins, sem, ins):
    per_class = {}
    for v in range(sem.shape[0]):
        classes = torch.unique(sem[v][(sem[v] >= 0) & (sem[v] <= 19)]).tolist()
        for cls in classes:
            gt_ids = torch.unique(ins[v][(sem[v] == cls) & (ins[v] > 0)]).tolist() if cls >= 2 else [0]
            pr_ids = torch.unique(pred_ins[v][(pred_sem[v] == cls) & (pred_ins[v] > 0)]).tolist() if cls >= 2 else [0]
            gt_masks = [(sem[v] == cls) & ((ins[v] == gid) if cls >= 2 else torch.ones_like(sem[v], dtype=torch.bool)) for gid in gt_ids]
            pr_masks = [(pred_sem[v] == cls) & ((pred_ins[v] == pid) if cls >= 2 else torch.ones_like(sem[v], dtype=torch.bool)) for pid in pr_ids]
            used = set(); tp = fp = fn = 0; sum_iou = 0.0
            for gm in gt_masks:
                best = (0.0, -1)
                for j, pm in enumerate(pr_masks):
                    if j in used: continue
                    inter = int((gm & pm).sum()); union = int((gm | pm).sum())
                    iou = inter / union if union else 0.0
                    if iou > best[0]: best = (iou, j)
                if best[0] > 0.5:
                    used.add(best[1]); tp += 1; sum_iou += best[0]
                else: fn += 1
            fp = len(pr_masks) - len(used)
            row = per_class.setdefault(int(cls), [0.0, 0, 0, 0])
            row[0] += sum_iou; row[1] += tp; row[2] += fp; row[3] += fn
    pq = {str(c): row[0] / max(1e-12, row[1] + .5 * row[2] + .5 * row[3])
          for c, row in per_class.items()}
    return {"per_class_pq": pq, "mean_pq": float(np.mean(list(pq.values()))) if pq else 0.0}


def _assemble(out):
    from scripts.export_object_locus_v2_1_official import assemble_panoptic
    return assemble_panoptic(out)


def _one_window(model, opt, window, device, batch_builder, scope):
    from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
    from tokengs.models.object_locus_v2_1_loss import final_hungarian
    legacy = _metrics_helpers()
    batch = batch_builder(opt, window, device)
    mi, _ = split_data(batch, opt)
    views = 2 if scope == "context" else int(batch["cam_view_all"].shape[1])
    dec = ModelInputDecoder(cam_view=batch["cam_view_all"][:, :views],
                            intrinsics=batch["intrinsics_all"][:, :views])
    out = model.forward_object_locus(ModelInput(mi.encoder, dec), render_decoder_input=dec,
                                     context_decoder=dec)
    pred_sem, pred_ins, raw_sem = _assemble(out)
    sem = batch["semantic_label_all"][0, :views].long()
    ins = batch["instance_label_all"][0, :views].long()
    raw = (out["region_mass"][0, :views, :100] >= .5) & (out["alpha"][0, :views, 0, None] > .05)
    fused = torch.stack([pred_ins == (q + 1) for q in range(100)], dim=1)
    gts = legacy._gt_masks(sem, ins)
    conf = np.zeros((20, 21), dtype=np.int64)
    valid = (sem >= 0) & (sem <= 19)
    for c in range(20):
        for p in range(21): conf[c, p] = int(((sem == c) & (pred_sem == p) & valid).sum())
    semread = out["semantic_scores"][0, :views].argmax(1)
    semread = torch.where(out["alpha"][0, :views, 0] > .05, semread,
                          torch.full_like(semread, 20))
    read_conf=np.zeros((20,21),np.int64)
    for c in range(20):
        for p in range(21): read_conf[c,p]=int(((sem==c)&(semread==p)&valid).sum())
    ious = []
    for c in range(20):
        tp=conf[c,c]; fp=conf[:,c].sum()-tp; fn=conf[c,:].sum()-tp
        if tp+fp+fn: ious.append(float(tp/(tp+fp+fn)))
    joint_prob = out["p_class"][0, :, :18]
    cls0 = joint_prob.argmax(-1); class_id = cls0 + 2
    class_conf = joint_prob.max(-1).values
    conditional = out["conditional_class_prob"][0]
    objectness = out["objectness_prob"][0]
    joint_eligible = (out["p_class"][0].argmax(-1) != 18) & (class_conf >= .05)
    raw_nonempty = raw.flatten(2).any(-1).any(0)
    raw_nonempty_eligible = int((joint_eligible & raw_nonempty).sum())
    # Scope score follows the export contract: class confidence times mean mask
    # membership over the query's final assigned pixels in this scope.
    scores=[]; isthing=[]
    for q in range(100):
        assigned = fused[:, q]
        score = class_conf[q] * out["region_mass"][0, :views, q][assigned].mean() if bool(assigned.any()) else class_conf[q] * 0
        scores.append(score)
        isthing.append(bool(joint_eligible[q] and assigned.any()))
    scores=torch.stack(scores); isthing=torch.tensor(isthing, device=device)
    ca = legacy._instance_metrics(fused, class_id, scores, isthing, gts, class_aware=False)
    aware = legacy._instance_metrics(fused, class_id, scores, isthing, gts, class_aware=True)
    raw_recall = legacy._raw_recall50(raw, gts)
    targets,pairs=final_hungarian(out,batch)
    support_by_id={int(i):bool(targets["Y_anchor"][0,k].sum()>0)
                   for k,i in enumerate(targets["gt_instance_ids"][0].tolist())}
    gt_rows=[]; raw_hits=0
    for (gcls,gid), gm in gts.items():
        best, bestq = 0.0, 0
        for q in range(100):
            iou=legacy._multiview_iou_for_gt(raw,q,gm)
            if iou is not None and iou>best: best,bestq=iou,q
        raw_hits += int(best >= .5)
        gt_rows.append({"class":gcls,"instance_id":gid,"best_raw_iou":best,"best_raw_query":bestq,
                        "has_anchor_support":support_by_id.get(int(gid),False)})
    correct=matched=conditional_correct=0
    matched_obj=[]; unmatched_obj=[]; class_confusion=np.zeros((18,18),dtype=np.int64)
    for b,(qi,ki) in enumerate(pairs):
        if qi.numel():
            gtclass=targets["gt_classes"][b][ki]-2
            pred_cond=out["conditional_class_prob"][b,qi].argmax(-1)
            pred_joint=out["states"][-1]["thing_logits19"][b,qi].argmax(-1)
            conditional_correct += int((pred_cond==gtclass).sum())
            correct += int((pred_joint==gtclass).sum())
            np.add.at(class_confusion,(gtclass.detach().cpu().numpy(),pred_cond.detach().cpu().numpy()),1)
            matched += int(qi.numel())
        mask=torch.zeros(100,dtype=torch.bool,device=device)
        mask[qi]=True
        matched_obj.extend(objectness[mask].detach().cpu().tolist())
        unmatched_obj.extend(objectness[~mask].detach().cpu().tolist())
    matched_by_id={}
    for b,(qi,ki) in enumerate(pairs):
        for q,k in zip(qi.tolist(),ki.tolist()):
            gid=int(targets["gt_instance_ids"][b][k])
            cond_idx=int(conditional[q].argmax())
            joint_idx=int(out["states"][-1]["thing_logits19"][b,q].argmax())
            matched_by_id[gid]={"matched_query":q,"matched_conditional_class":cond_idx+2,
              "matched_joint_class":joint_idx+2 if joint_idx<18 else None,
              "matched_conditional_probability":float(conditional[q,cond_idx]),
              "matched_objectness_probability":float(objectness[q]),
              "matched_joint_probability":float(out["p_class"][b,q,joint_idx])}
    for gt in gt_rows:gt.update(matched_by_id.get(int(gt["instance_id"]),{}))
    pixel_index = list(range(views))
    rgb_pred=out["render"]["images_pred"][0,:views]
    rgb_gt=batch["images_all"][0,:views]
    mse=(rgb_pred-rgb_gt).square().mean().clamp_min(1e-12)
    psnr=float((-10*torch.log10(mse)).cpu())
    ctx_count=len(window["context"])
    novel_indices=[i for i,f in enumerate(batch["frame_ids"][0].detach().cpu().tolist())
                   if i < views and int(f) in set(map(int,window.get("novel",[])))]
    novel_psnr=None
    if novel_indices:
        nmse=(out["render"]["images_pred"][0,novel_indices]-batch["images_all"][0,novel_indices]).square().mean().clamp_min(1e-12)
        novel_psnr=float((-10*torch.log10(nmse)).cpu())
    pq=_panoptic_pq(pred_sem, pred_ins, sem, ins)
    final_instance_count=int(torch.unique(pred_ins[pred_ins>0]).numel())
    return {"batch":batch,"out":out,"semantic":pred_sem,"instance":pred_ins,
        "raw_semantic":raw_sem,"scope":scope,"views":views,
        "scene":window["scene"],"context_frames":[int(x) for x in window["context"]],
        "novel_frames":[int(x) for x in window["novel"]],
        "semantic_confusion":conf.tolist(),"semantic_readout_confusion":read_conf.tolist(),"semantic_miou":float(np.mean(ious)) if ious else 0,
        "thing_miou":float(np.mean([conf[c,c]/max(1,conf[c].sum()+conf[:,c].sum()-conf[c,c]) for c in range(2,20)])),
        "stuff_miou":float(np.mean([conf[c,c]/max(1,conf[c].sum()+conf[:,c].sum()-conf[c,c]) for c in (0,1)])),
        "local_pq":pq,"class_agnostic":ca,"class_aware":aware,
        "raw_recall50":raw_recall,"raw_best_iou":gt_rows,"raw_mask_recall50":raw_hits/max(1,len(gt_rows)),
        "matched_class_accuracy":correct/max(1,matched),
        "joint_classification_accuracy":correct/max(1,matched),
        "conditional_classification_accuracy":conditional_correct/max(1,matched),
        "matched_gt_count":matched,"conditional_class_confusion":class_confusion.tolist(),
        "matched_objectness_prob_mean":float(np.mean(matched_obj)) if matched_obj else 0.0,
        "matched_objectness_count":len(matched_obj),
        "matched_objectness_recall_p50":float(np.mean(np.asarray(matched_obj)>=.5)) if matched_obj else 0.0,
        "unmatched_objectness_prob_mean":float(np.mean(unmatched_obj)) if unmatched_obj else 0.0,
        "unmatched_objectness_count":len(unmatched_obj),
        "unmatched_objectness_p50_fraction":float(np.mean(np.asarray(unmatched_obj)>=.5)) if unmatched_obj else 0.0,
        "filtering_waterfall":{"all_slots":100,"joint_class_eligible":int(joint_eligible.sum()),
          "nonempty_raw_mask":int(raw_nonempty.sum()),
          "nonempty_raw_mask_after_eligibility":raw_nonempty_eligible,
          "area_competition_retained":final_instance_count,"final_output_instances":final_instance_count},
        "psnr":psnr,"novel_psnr":novel_psnr,"active_queries":int(isthing.sum()),
        "within_parent_membership_std":float(out["gaussian_membership"][0].reshape(1024,64,102).std(1).mean())}


def _palette(sem):
    arr=np.zeros((*sem.shape,3),np.uint8)
    for c in range(20): arr[sem==c]=((37*c+53)%255,(97*c+31)%255,(173*c+71)%255)
    return arr


def write_panel(row, path, title):
    batch,out=row["batch"],row["out"]; vcount=row["views"]
    images=[]; labels=[]
    sem_gt_all=batch["semantic_label_all"][0,:vcount].cpu()
    ins_gt_all=batch["instance_label_all"][0,:vcount].cpu()
    gt_candidates=[]
    for cls in range(2,20):
        ids=torch.unique(ins_gt_all[(sem_gt_all==cls)&(ins_gt_all>0)]).tolist()
        for gid in ids:
            gm=(sem_gt_all==cls)&(ins_gt_all==int(gid))
            gt_candidates.append((int(gm.sum()),cls,int(gid),gm))
    gt_candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
    raw=out["region_mass"][0,:vcount,:100]>=.5
    alpha=out["alpha"][0,:vcount,0]>.05
    chosen=[]
    for area,cls,gid,gm in gt_candidates[:3]:
        best_iou,best_q=0.0,0
        for q in range(100):
            pm = raw[:, q] & alpha
            gt_on_device=gm.to(device=pm.device)
            inter=int((pm & gt_on_device).sum()); union=int((pm | gt_on_device).sum())
            iou=inter/union if union else 0.0
            if iou>best_iou:best_iou,best_q=iou,q
        chosen.append((area,cls,gid,gm,best_q,best_iou))
    raw_rgb=np.zeros((*sem_gt_all.shape[-2:],3),np.uint8)
    query_colors=[(245,80,70),(60,190,90),(70,120,245)]
    raw_views=[]
    for v in range(vcount):
        canvas=np.zeros((*sem_gt_all.shape[-2:],3),np.uint8)
        for ci,(_,_,_,_,q,_) in enumerate(chosen):canvas[raw[v,q].cpu().numpy()&alpha[v].cpu().numpy()]=query_colors[ci]
        raw_views.append(canvas)
    def instance_rgb(sem,ins):
        arr=_palette(np.where((sem>=0)&(sem<=19),sem,20))
        for iid in np.unique(ins):
            if iid<=0:continue
            color=((37*int(iid)+91)%255,(83*int(iid)+57)%255,(149*int(iid)+23)%255)
            arr[ins==iid]=color
        return arr
    for v in range(vcount):
        gt_rgb=(batch["images_all"][0,v].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
        pr_rgb=(out["render"]["images_pred"][0,v].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
        sem_gt=batch["semantic_label_all"][0,v].cpu().numpy()
        ins_gt=batch["instance_label_all"][0,v].cpu().numpy()
        gt_pan=instance_rgb(sem_gt,ins_gt)
        sem_pr=row["semantic"][v].cpu().numpy(); ins_pr=row["instance"][v].cpu().numpy()
        pr_pan=instance_rgb(sem_pr,ins_pr)
        images.extend([gt_rgb,pr_rgb,_palette(np.where((sem_gt>=0)&(sem_gt<=19),sem_gt,20)),_palette(sem_pr),gt_pan,pr_pan,raw_views[v]])
        labels.extend(["RGB GT","RGB reconstruction","semantic GT","semantic readout","instance GT","final panoptic","raw best masks"])
    sz=160; header=48; legend=128; canvas=Image.new("RGB",(sz*7,header+sz*vcount+legend),"white"); draw=ImageDraw.Draw(canvas)
    for i,l in enumerate(labels[:6]): draw.text((i*sz+2,2),l,fill="black")
    draw.text((6*sz+2,2),labels[6],fill="black")
    for i,img in enumerate(images): canvas.paste(Image.fromarray(img).resize((sz,sz)),((i%7)*sz,header+(i//7)*sz))
    pcat=out["conditional_class_prob"][0]; pobj=out["objectness_prob"][0]; pjoint=out["p_class"][0]
    text_y=header+sz*vcount+4
    draw.text((4,text_y),f"{title} | membership>=0.5; alpha>0.05 | raw masks are unfiltered",fill="black")
    for idx,(_,cls,gid,_,q,iou) in enumerate(chosen):
        pred=int(pcat[q].argmax())+2; confidence=float(pcat[q].max()); obj=float(pobj[q]); joint=float(pjoint[q,:18].max())
        text_y+=20
        draw.text((4,text_y),f"GT class={cls} id={gid} area={_} | raw best slot={q} IoU={iou:.3f} | conditional={pred} pcat={confidence:.3f} p_fg={obj:.3f} joint={joint:.3f}",fill=query_colors[idx])
    Path(path).parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def evaluate_windows(model,opt,windows,step,split,reports,device,batch_builder,*,official=False,panels=False):
    from scripts.object_locus_v2_1_runtime import capture_rng,restore_rng,write_json
    from scripts.export_object_locus_v2_1_official import export_windows
    was_training=model.training; rng=capture_rng(); model.eval()
    rows={"context":[],"target":[]}; artifacts=[]
    try:
        with torch.no_grad():
            for window_index, win in enumerate(windows):
                for scope in ("context","target"):
                    row=_one_window(model,opt,win,device,batch_builder,scope)
                    summary={k:v for k,v in row.items() if k not in ("batch","out","semantic","instance","raw_semantic")}
                    rows[scope].append(summary)
                    # Keep the registered fixed-window qualitative set compact:
                    # the first three windows in file order for each split.
                    if panels and window_index < 2:
                        write_panel(row,Path(reports)/f"qualitative/step_{step:04d}/{split}/{'_'.join(map(str,win['context']))}_{scope}.png",f"{step} {split} {scope}")
                    del row
    finally:
        restore_rng(rng); model.train(was_training)
    def agg(scope):
        rr=rows[scope]
        conf=np.sum([np.asarray(x["semantic_confusion"]) for x in rr],axis=0)
        iou=[]; thing_iou=[]; stuff_iou=[]; per_class={}
        for c in range(20):
            t=conf[c,c]; fp=conf[:,c].sum()-t; fn=conf[c,:].sum()-t
            per_class[str(c)]=float(t/(t+fp+fn)) if t+fp+fn else None
            if t+fp+fn:
                iou.append(t/(t+fp+fn))
                (stuff_iou if c<2 else thing_iou).append(t/(t+fp+fn))
        readconf=np.sum([np.asarray(x["semantic_readout_confusion"]) for x in rr],axis=0)
        readiou=[]
        for c in range(20):
            t=readconf[c,c]; fp=readconf[:,c].sum()-t; fn=readconf[c,:].sum()-t
            if t+fp+fn: readiou.append(t/(t+fp+fn))
        return {"windows":len(rr),"semantic_miou":float(np.mean(iou)) if iou else 0,
          "mIoU_all_nonempty":float(np.mean(iou)) if iou else 0,
          "mIoU_thing":float(np.mean(thing_iou)) if thing_iou else 0,
          "mIoU_stuff":float(np.mean(stuff_iou)) if stuff_iou else 0,
          "per_class_iou":per_class,
          "semantic_readout_miou":float(np.mean(readiou)) if readiou else 0,
          "psnr":float(np.mean([r["psnr"] for r in rr])),
          "local_pq":float(np.mean([r["local_pq"]["mean_pq"] for r in rr])),
          "class_agnostic_tp":sum(r["class_agnostic"]["tp"] for r in rr),
          "class_agnostic_fp":sum(r["class_agnostic"]["fp"] for r in rr),
          "class_agnostic_fn":sum(r["class_agnostic"]["fn"] for r in rr),
          "gt_count":sum(r["class_agnostic"]["n_gt"] for r in rr),
          "class_aware_tp":sum(r["class_aware"]["tp"] for r in rr),
          "class_aware_fp":sum(r["class_aware"]["fp"] for r in rr),
          "class_aware_fn":sum(r["class_aware"]["fn"] for r in rr),
          "raw_recall50":sum(r["raw_recall50"].get("tp",0) for r in rr)/max(1,sum(r["raw_recall50"]["n_gt"] for r in rr)),
          "raw_best_iou_ge_0_5_fraction":sum(x["best_raw_iou"]>=.5 for r in rr for x in r["raw_best_iou"])/max(1,sum(len(r["raw_best_iou"]) for r in rr)),
          "raw_gt_count":sum(len(r["raw_best_iou"]) for r in rr),
          "gt_with_anchor_support":sum(x["has_anchor_support"] for r in rr for x in r["raw_best_iou"]),
          "gt_support_fraction":sum(x["has_anchor_support"] for r in rr for x in r["raw_best_iou"])/max(1,sum(len(r["raw_best_iou"]) for r in rr)),
          "matched_class_accuracy":sum(r["matched_class_accuracy"]*r["matched_gt_count"] for r in rr)/max(1,sum(r["matched_gt_count"] for r in rr)),
          "conditional_classification_accuracy":sum(r["conditional_classification_accuracy"]*r["matched_gt_count"] for r in rr)/max(1,sum(r["matched_gt_count"] for r in rr)),
          "joint_classification_accuracy":sum(r["joint_classification_accuracy"]*r["matched_gt_count"] for r in rr)/max(1,sum(r["matched_gt_count"] for r in rr)),
          "matched_objectness_prob_mean":sum(r["matched_objectness_prob_mean"]*r["matched_objectness_count"] for r in rr)/max(1,sum(r["matched_objectness_count"] for r in rr)),
          "matched_objectness_count":sum(r["matched_objectness_count"] for r in rr),
          "matched_objectness_recall_p50":sum(r["matched_objectness_recall_p50"]*r["matched_objectness_count"] for r in rr)/max(1,sum(r["matched_objectness_count"] for r in rr)),
          "unmatched_objectness_prob_mean":sum(r["unmatched_objectness_prob_mean"]*r["unmatched_objectness_count"] for r in rr)/max(1,sum(r["unmatched_objectness_count"] for r in rr)),
          "unmatched_objectness_count":sum(r["unmatched_objectness_count"] for r in rr),
          "unmatched_objectness_p50_fraction":sum(r["unmatched_objectness_p50_fraction"]*r["unmatched_objectness_count"] for r in rr)/max(1,sum(r["unmatched_objectness_count"] for r in rr)),
          "within_parent_membership_std":float(np.mean([r["within_parent_membership_std"] for r in rr])),
          "filtering_waterfall":{k:float(np.mean([r["filtering_waterfall"][k] for r in rr])) for k in ("all_slots","joint_class_eligible","nonempty_raw_mask","nonempty_raw_mask_after_eligibility","area_competition_retained","final_output_instances")},
          "conditional_class_confusion":np.sum([np.asarray(r["conditional_class_confusion"]) for r in rr],axis=0).tolist(),
          "per_class_iou":{str(c):float(conf[c,c]/max(1,conf[c,:].sum()+conf[:,c].sum()-conf[c,c])) for c in range(20)}}
    result={"step":step,"split":split,"local":{"context":agg("context"),"target_all":agg("target")},
            "true_novel_psnr":float(np.mean([r["novel_psnr"] for r in rows["target"] if r["novel_psnr"] is not None])) if rows["target"] else None,
            "local_rows":rows}
    if official:
        root=Path(reports)/f"official/step_{step:04d}/{split}"
        allres=export_windows(model,opt,windows,root/"all",device=device,batch_builder=batch_builder,target_frames="all")
        novres=export_windows(model,opt,windows,root/"novel",device=device,batch_builder=batch_builder,target_frames="novel")
        from scripts.eval_object_locus_v1 import _official_run,_official_metric
        all_json=_official_run(root/"all",root/"official_all.json")
        nov_json=_official_run(root/"novel",root/"official_novel.json")
        result["official"]={"all":all_json.get("result"),"novel":nov_json.get("result"),
            "metric_paths":{"all":"result.*","novel":"result.*"}}
    write_json(Path(reports)/f"eval_{split}_step{step:04d}.json",result)
    return result
