"""Prediction-only full-resolution 8D HDBSCAN; shared by model and diagnostics."""
from dataclasses import asdict,dataclass
import numpy as np


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
    from sklearn.cluster import HDBSCAN
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


