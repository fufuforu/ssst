"""Read-only boundary diagnostics; no model or optimization policy changes."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


def boundary_regions(semantic, instance):
    """Scene/view grids [V,H,W], consistent thing IDs, exclude void/unlabeled things."""
    valid=(semantic>=0)&(semantic<20)&((semantic<2)|(instance>0))
    thing=valid&(semantic>=2)
    identity=torch.where(thing,instance,-semantic-1)
    dx=(identity[...,1:]!=identity[...,:-1])&valid[...,1:]&valid[...,:-1]&(thing[...,1:]|thing[...,:-1])
    dy=(identity[...,1:,:]!=identity[...,:-1,:])&valid[...,1:,:]&valid[...,:-1,:]&(thing[...,1:,:]|thing[...,:-1,:])
    edge=F.pad(dx,(0,1))|F.pad(dx,(1,0))|F.pad(dy,(0,0,0,1))|F.pad(dy,(0,0,1,0))
    band=F.max_pool2d(edge[:,None].float(),5,1,2)[:,0].bool()&valid
    # Interior is thing-only, separated from a five-pixel-wide boundary band.
    return dict(valid=valid,thing=thing,edge=edge&valid,band=band,interior=thing&~band)


def boundary_response(features):
    """[V,C,H,W] -> raw four-neighbor max and original 5x5 max response."""
    z=F.normalize(features.float(),dim=1,eps=1e-6)
    dx=(1-(z[...,1:]*z[...,:-1]).sum(1)).clamp_min(0)
    dy=(1-(z[...,1:,:]*z[...,:-1,:]).sum(1)).clamp_min(0)
    raw=torch.maximum(F.pad(dx,(0,1)),F.pad(dx,(1,0)))
    raw=torch.maximum(raw,F.pad(dy,(0,0,0,1)));raw=torch.maximum(raw,F.pad(dy,(0,0,1,0)))
    return raw,F.max_pool2d(raw[:,None],5,1,2)[:,0]


def distribution(x):
    x=x.detach().float().flatten().cpu()
    if not len(x):return dict(count=0)
    return dict(count=len(x),mean=float(x.mean()),q50=float(x.median()),q90=float(torch.quantile(x,.9)),
        q99=float(torch.quantile(x,.99)),maximum=float(x.max()))


def binary_metrics(positive,negative,threshold=.15,temperature=.05):
    """Population AUROC/AP; threshold effects and correspondence-shuffle control."""
    a=np.asarray(positive,dtype=np.float64);b=np.asarray(negative,dtype=np.float64)
    result=dict(positive_count=len(a),negative_count=len(b))
    if not len(a) or not len(b):return result
    x=np.concatenate((a,b));y=np.concatenate((np.ones(len(a)),np.zeros(len(b))))
    weight=lambda z:1/(1+np.exp(-np.clip((z-threshold)/temperature,-60,60)))
    rng=np.random.default_rng(20261018)
    result.update(auroc=float(roc_auc_score(y,x)),average_precision=float(average_precision_score(y,x)),
        prevalence=float(y.mean()),shuffle_auroc=float(roc_auc_score(y,x[rng.permutation(len(x))])),
        tpr=float((a>=threshold).mean()),fpr=float((b>=threshold).mean()),
        boundary_to_interior_weight_ratio=float(weight(a).mean()/max(weight(b).mean(),1e-12)))
    return result


def bounded_samples(x,limit=4096):
    """Evenly spaced deterministic sample; identical code for all variants."""
    x=x.detach().float().flatten().cpu()
    if len(x)>limit:x=x[torch.linspace(0,len(x)-1,limit).long()]
    return x.numpy()


def score_map(features,semantic,instance,coverage=None):
    regions=boundary_regions(semantic,instance);raw,response=boundary_response(features)
    coverage=regions['valid'] if coverage is None else regions['valid']&coverage
    pos=regions['band']&coverage;neg=regions['interior']&coverage
    a=response[pos];b=response[neg]
    result=dict(raw_at_gt_edge=distribution(raw[regions['edge']&coverage]),
        expanded_at_boundary=distribution(a),expanded_at_interior=distribution(b),
        coverage_fraction=float(coverage.sum()/regions['valid'].sum().clamp_min(1)),
        metrics=binary_metrics(a.cpu().numpy(),b.cpu().numpy()),
        positive_samples=bounded_samples(a).tolist(),negative_samples=bounded_samples(b).tolist())
    return result,response


def calibrate_on_train(rows):
    """Equal-sized per-scene samples; threshold maximizes train TPR-FPR only."""
    a=np.concatenate([np.asarray(r['positive_samples']) for r in rows if r['positive_samples']]) if any(r['positive_samples'] for r in rows) else np.array([])
    b=np.concatenate([np.asarray(r['negative_samples']) for r in rows if r['negative_samples']]) if any(r['negative_samples'] for r in rows) else np.array([])
    if not len(a) or not len(b):return None
    values=np.concatenate((a,b));candidates=np.unique(np.quantile(values,np.linspace(0,1,201)))
    scores=np.array([(a>=t).mean()-(b>=t).mean() for t in candidates]);threshold=float(candidates[scores.argmax()])
    temperature=max(float(np.quantile(values,.75)-np.quantile(values,.25))/4,1e-6)
    return dict(threshold=threshold,temperature=temperature,train_youden=float(scores.max()),
        fit_scope='four training probes only; diagnostic calibration, no weight update')


@torch.no_grad()
def dominant_contributor(renderer,gaussians,decoder,chunk=32):
    """Exact argmax alpha-compositing contribution via one-hot channel chunks.

    Every call keeps ALL Gaussian geometry/opacities; only diagnostic colors
    differ. Never render a Gaussian subset, which would alter transmittance.
    """
    n=gaussians.shape[1];best=indices=total=None
    for start in range(0,n,chunk):
        width=min(chunk,n-start)
        colors=gaussians.new_zeros(1,n,width)
        i=torch.arange(width,device=gaussians.device);colors[0,start+i,i]=1
        image=renderer.render_feature_channels(gaussians,colors,decoder.cam_view,decoder.intrinsics)['images_pred']
        value,index=image.max(2);index=index+start
        if best is None:best=value;indices=index;total=image.sum(2)
        else:
            replace=value>best;indices=torch.where(replace,index,indices);best=torch.maximum(best,value);total+=image.sum(2)
    return indices,best,total


def hard_feature_map(features,indices):
    return features[0][indices[0]].permute(0,3,1,2).contiguous()


@torch.no_grad()
def gaussian_gt_ownership(gaussians,decoder,renderer,targets,semantic,purity_threshold=.7):
    """Contribution-weighted diagnostic labels, including two stuff codes."""
    from tokengs.models.object_locus_panoptic_v1_lift import RendererTranspose
    masks=targets['gt_pixel_masks'][0].permute(1,0,2,3)[None].float()
    valid=targets['valid_pixels'][:,:,None].float()
    stuff=torch.stack(((semantic==0)&targets['valid_pixels'][0],(semantic==1)&targets['valid_pixels'][0]),1)[None].float()
    all_masks=torch.cat((masks,stuff),2)
    mass=RendererTranspose.apply(valid,gaussians,decoder.cam_view,decoder.intrinsics,renderer)
    label_mass=RendererTranspose.apply(all_masks,gaussians,decoder.cam_view,decoder.intrinsics,renderer)
    top,all_label=label_mass[0].max(-1);purity=top/mass[0,:,0].clamp_min(1e-6)
    all_label=all_label.masked_fill((mass[0,:,0]<=1e-6)|(purity<purity_threshold),-1)
    thing_label=all_label.masked_fill(all_label>=masks.shape[2],-1)
    return all_label,thing_label,purity,all_masks.shape[2]


def gaussian_pair_probe(features,xyz,labels,parent,seed=42):
    """GT-balanced same/different instance distances; same-parent pairs separated."""
    from scipy.spatial import cKDTree
    valid=torch.nonzero(labels>=0).flatten().cpu();f=F.normalize(features.detach().float().cpu(),dim=-1)
    lab=labels.cpu();par=parent.cpu();generator=torch.Generator().manual_seed(seed)
    result=dict(confident_gaussians=len(valid));pairs=[];targets=[]
    ids=lab[valid].unique()
    for identity in ids:
        a=valid[lab[valid]==identity];other=valid[lab[valid]!=identity]
        if len(a)>1:
            left=a[torch.randint(len(a),(256,),generator=generator)];right=a[torch.randint(len(a),(256,),generator=generator)]
            keep=left!=right;pairs.append(torch.stack((left[keep],right[keep]),1));targets.extend([0]*int(keep.sum()))
        if len(other):
            left=a[torch.randint(len(a),(256,),generator=generator)];right=other[torch.randint(len(other),(256,),generator=generator)]
            pairs.append(torch.stack((left,right),1));targets.extend([1]*len(left))
    if pairs:
        pair=torch.cat(pairs);distance=(1-(f[pair[:,0]]*f[pair[:,1]]).sum(-1)).clamp_min(0);target=torch.tensor(targets).bool()
        shuffled=valid[torch.randperm(len(valid),generator=generator)];lookup=torch.arange(len(f));lookup[valid]=shuffled
        shuffled_distance=(1-(f[lookup[pair[:,0]]]*f[lookup[pair[:,1]]]).sum(-1)).clamp_min(0)
        result.update(global_pairs=binary_metrics(distance[target].numpy(),distance[~target].numpy()),
            correspondence_shuffled_auroc=float(roc_auc_score(target.numpy(),shuffled_distance.numpy())) if target.any() and (~target).any() else None,
            same_gt_cross_parent=distribution(distance[(~target)&(par[pair[:,0]]!=par[pair[:,1]])]),
            same_gt_same_parent=distribution(distance[(~target)&(par[pair[:,0]]==par[pair[:,1]])]),
            different_gt=distribution(distance[target]))
    # Local physical neighbors, kept separate from the global separation probe.
    if len(valid)>1:
        points=xyz[valid.to(xyz.device)].detach().float().cpu().numpy();k=min(9,len(valid))
        distances,neighbors=cKDTree(points).query(points,k=k)
        a=valid[:,None].expand(-1,k-1).reshape(-1);b=valid[torch.as_tensor(neighbors[:,1:].copy()).reshape(-1)]
        keep=a<b;a=a[keep];b=b[keep]
        different=lab[a]!=lab[b];d=(1-(f[a]*f[b]).sum(-1)).clamp_min(0)
        result.update(local_pairs=binary_metrics(d[different].numpy(),d[~different].numpy()),
            local_same_distance=distribution(d[~different]),local_different_distance=distribution(d[different]),
            neighbor_distance=distribution(torch.as_tensor(distances[:,1:].copy())))
    return result
