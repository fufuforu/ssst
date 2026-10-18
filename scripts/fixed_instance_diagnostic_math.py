"""Shared class-agnostic 2D mask metrics and clustering compatibility exports."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from tokengs.instance_clustering import ClusterRecipe,RECIPE,cluster_view,label_masks


def mask_iou(predictions,targets,valid):
    """Void ignored for both; keep predictions with any area on valid canvas."""
    valid=np.asarray(valid,dtype=bool)
    p=np.asarray(predictions,dtype=bool);g=np.asarray(targets,dtype=bool)
    if p.ndim!=3 or g.ndim!=3 or p.shape[1:]!=valid.shape or g.shape[1:]!=valid.shape:
        raise ValueError('Mask canvas mismatch')
    pp=p[:,valid];gg=g[:,valid]
    pk=pp.any(1);gk=gg.any(1);pp=pp[pk];gg=gg[gk]
    # Chunked BLAS avoids broadcasting [P,G,H,W]. Float32 exactly counts 256^2 pixels.
    inter=pp.astype(np.float32)@gg.astype(np.float32).T
    union=pp.sum(1)[:,None]+gg.sum(1)[None]-inter
    iou=inter/np.maximum(union,1)
    return iou,pk,gk


def evaluate_masks(predictions,targets,valid):
    """All nonempty valid GT; no source-matching restriction or GT-based selection."""
    iou,pk,gk=mask_iou(predictions,targets,valid);npred,ngt=iou.shape
    best=iou.max(0) if npred else np.zeros(ngt)
    matched=np.zeros(ngt);tp=0
    if npred and ngt:
        r,c=linear_sum_assignment(-iou);matched[c]=iou[r,c]
        # Maximize number of >.5 matches first, sum IoU only breaks ties.
        reward=(iou>.5).astype(float)+iou/(min(npred,ngt)+1)
        r,c=linear_sum_assignment(-reward);tp=int((iou[r,c]>.5).sum())
    precision=tp/npred if npred else (1. if not ngt else 0.)
    recall=tp/ngt if ngt else None
    return dict(gt_count=ngt,pred_count=npred,tp50=tp,fp50=npred-tp,fn50=ngt-tp,
        precision50=precision,recall50=recall,f1_50=2*tp/(npred+ngt) if npred+ngt else None,
        best_gt_iou=float(best.mean()) if ngt else None,
        one_to_one_gt_iou=float(matched.mean()) if ngt else None,
        best_gt_iou_values=best.tolist(),one_to_one_gt_iou_values=matched.tolist(),
        retained_pred_indices=np.flatnonzero(pk).tolist(),retained_gt_indices=np.flatnonzero(gk).tolist())


def supported_best_iou(predictions,targets,valid,supported):
    """Secondary coverage-controlled upper bound; unmatched predictions not penalized."""
    iou,_,gk=mask_iou(predictions,targets,valid)
    support=np.asarray(supported,dtype=bool)[gk]
    best=iou.max(0) if len(iou) else np.zeros(iou.shape[1])
    return dict(count=int(support.sum()),best_gt_iou=float(best[support].mean()) if support.any() else None)
