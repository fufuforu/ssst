"""Prediction-only 8D clustering and shared, class-agnostic 2D mask metrics."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN


@dataclass(frozen=True)
class ClusterRecipe:
    min_cluster_size: int = 100
    min_samples: int = 10  # sklearn includes the point itself.
    metric: str = 'euclidean'
    cluster_selection_method: str = 'eom'
    cluster_selection_epsilon: float = 0.
    allow_single_cluster: bool = False
    alpha_threshold: float = 1e-6
    normalization_epsilon: float = 1e-6
    leaf_size: int = 40
    n_jobs: int = 4


RECIPE = ClusterRecipe()


def cluster_view(features, alpha, recipe=RECIPE):
    """Full-resolution HDBSCAN, no GT/XY, no subsampling or noise reassignment."""
    f=np.asarray(features,dtype=np.float32)
    a=np.asarray(alpha,dtype=np.float32)
    if f.ndim!=3 or f.shape[0]!=8 or a.shape!=f.shape[1:]:
        raise ValueError('Expected [8,H,W] features and [H,W] alpha')
    if not np.isfinite(f).all() or not np.isfinite(a).all():
        raise ValueError('Nonfinite features/alpha')
    norm=np.linalg.norm(f,axis=0)
    visible=(a>recipe.alpha_threshold)&(norm>recipe.normalization_epsilon)
    labels=np.full(a.shape,-1,dtype=np.int32)
    x=(f[:,visible]/norm[visible]).T.copy()
    if len(x)>=max(recipe.min_cluster_size,recipe.min_samples):
        fit=HDBSCAN(min_cluster_size=recipe.min_cluster_size,min_samples=recipe.min_samples,
            metric=recipe.metric,cluster_selection_method=recipe.cluster_selection_method,
            cluster_selection_epsilon=recipe.cluster_selection_epsilon,allow_single_cluster=recipe.allow_single_cluster,
            algorithm='kd_tree',leaf_size=recipe.leaf_size,n_jobs=recipe.n_jobs,copy=True)
        labels[visible]=fit.fit_predict(x)
    count=int(len(np.unique(labels[labels>=0])))
    return labels,dict(recipe=asdict(recipe),input_pixels=int(a.size),eligible_pixels=int(visible.sum()),
        clustered_pixels=int((labels>=0).sum()),noise_pixels=int((visible&(labels<0)).sum()),
        excluded_pixels=int((~visible).sum()),clusters=count,full_resolution=True,gt_used=False,
        post_render_l2=True,xy_used=False,noise_reassigned=False),visible


def label_masks(labels,background=-1):
    labels=np.asarray(labels)
    ids=np.unique(labels);ids=ids[ids!=background]
    return labels[None]==ids[:,None,None],ids


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
