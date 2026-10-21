"""Fixed, forward-only diagnostic definitions. No model dependencies."""
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter


def summary(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1);x=x[np.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
                **{f'p{p}':float(np.percentile(x,p)) if len(x) else None for p in (10,50,90)})


def attribute(contribution,mass,purity=.7):
    """Assign only visible, >=50% thing-covered, >=purity-pure GS."""
    c=np.asarray(contribution);mass=np.asarray(mass).reshape(-1)
    thing=c.sum(-1)
    if c.shape[1]:
        dominant=c.argmax(-1);pure=c.max(-1)/np.maximum(thing,1e-12)
    else:dominant=np.zeros(len(mass),dtype=int);pure=np.zeros(len(mass))
    visible=mass>=1.;covered=thing>=.5*mass
    valid=visible & covered & (pure>=purity) & (thing>0)
    return np.where(valid,dominant,-1),pure,visible,thing


def feature_metrics(shared,residual,final,membership,contribution,mass):
    a=len(shared);g=a*64
    assert residual.shape==(a,64,shared.shape[-1]) and final.shape==(g,shared.shape[-1])
    assert len(mass)==g and len(contribution)==g
    labels,purity,visible,thing=attribute(contribution,mass)
    labels=labels.reshape(a,64);mass=np.asarray(mass).reshape(a,64)
    ac=contribution.reshape(a,64,contribution.shape[-1]).sum(1);at=ac.sum(-1)
    significant=(ac>=1.) & (ac>=.1*at[:,None]);cross=significant.sum(-1)>=2
    identifiable=significant.sum(-1)>=1
    ratio=np.linalg.norm(residual,axis=-1)/np.maximum(np.linalg.norm(shared,axis=-1)[:,None],1e-12)
    p=np.clip(membership.reshape(a,64,-1),1e-7,1-1e-7)
    entropy=-(p*np.log(p)+(1-p)*np.log(1-p)).mean(-1)
    categorical=p/p.sum(-1,keepdims=True);cat_entropy=-(categorical*np.log(categorical)).sum(-1)/np.log(p.shape[-1])
    i,j=np.triu_indices(64,k=1)
    features={'child':shared[:,None]+residual,'final':final.reshape(a,64,-1),'residual':residual}
    result={};rows=[]
    for scope,sel in [('all',np.ones(a,dtype=bool)),('cross',cross)]:
        result[scope+'.anchors']=int(sel.sum())
        for name,x in [('residual_shared_norm_ratio',ratio),('shared_norm',np.broadcast_to(np.linalg.norm(shared,axis=-1)[:,None],(a,64))),('residual_norm',np.linalg.norm(residual,axis=-1)),('bernoulli_entropy',entropy),('normalized_categorical_entropy',cat_entropy),('membership_saturation',((p<.01)|(p>.99)).mean(-1))]:
            result[scope+'.'+name]=summary(x[sel])
        result[scope+'.visible_residual_shared_norm_ratio']=summary(ratio[sel[:,None]&visible.reshape(a,64)])
        result[scope+'.assigned_residual_shared_norm_ratio']=summary(ratio[sel[:,None]&(labels>=0)])
        for threshold in (.5,.7,.9):
            lab=attribute(contribution,mass.reshape(-1),threshold)[0].reshape(a,64)
            valid=(lab[:,i]>=0)&(lab[:,j]>=0)&sel[:,None]
            same=valid&(lab[:,i]==lab[:,j]);diff=valid&(lab[:,i]!=lab[:,j])
            masks={'same':same,'different':diff}
            for name,x in features.items():
                normalized=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
                cos=(normalized[:,i]*normalized[:,j]).sum(-1)
                for rel,m in masks.items():result[f'{scope}.purity{threshold}.{name}.cosine_{rel}']=summary(cos[m])
            # Distribution distances independent of query index permutation.
            pi,pj=categorical[:,i],categorical[:,j];mid=.5*(pi+pj)
            js=.5*((pi*np.log(pi/mid)).sum(-1)+(pj*np.log(pj/mid)).sum(-1))
            tv=.5*np.abs(pi-pj).sum(-1)
            bern_tv=np.abs(p[:,i]-p[:,j]).mean(-1)
            for name,x in [('membership_js',js),('membership_tv',tv),('membership_sigmoid_l1',bern_tv)]:
                for rel,m in masks.items():result[f'{scope}.purity{threshold}.{name}_{rel}']=summary(x[m])
    result.update(anchors=a,multi_instance_anchors=int(cross.sum()),multi_instance_fraction=float(cross.mean()),
                  gt_identifiable_anchors=int(identifiable.sum()),multi_instance_fraction_identifiable=float(cross.sum()/max(1,identifiable.sum())),
                  visible_gs=int(visible.sum()),assigned_gs=int((labels>=0).sum()),total_gs=g,gt_purity=summary(purity[visible]))
    for k in range(a):
        valid=(labels[k,i]>=0)&(labels[k,j]>=0);different=valid&(labels[k,i]!=labels[k,j]);same=valid&(labels[k,i]==labels[k,j])
        row=dict(anchor=k,cross_instance=bool(cross[k]),assigned_gs=int((labels[k]>=0).sum()),
                 significant_gt_instances=int(significant[k].sum()),residual_shared_norm_ratio=float(ratio[k].mean()))
        for name,x in features.items():
            z=x[k]/np.maximum(np.linalg.norm(x[k],axis=-1,keepdims=True),1e-12);cos=(z[i]*z[j]).sum(-1)
            for rel,m in [('same',same),('different',different)]:row[name+'_cosine_'+rel]=float(cos[m].mean()) if m.any() else None
        rows.append(row)
    return result,rows


def gt_boundary(semantic,instance):
    valid=(semantic>=0)&(semantic<20);thing=valid&(semantic>=2)&(instance>0)
    # Stuff has no instance ID, but thing/stuff interfaces are valid edges.
    key=np.where(thing,semantic.astype(np.int64)*1000000+instance,0)
    edge=np.zeros_like(valid)
    for axis in (0,1):
        lo=[slice(None)]*2;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        lo,hi=tuple(lo),tuple(hi)
        d=(key[lo]!=key[hi])&valid[lo]&valid[hi]&(thing[lo]|thing[hi])
        edge[lo]|=d;edge[hi]|=d
    return edge,valid,thing


def boundary_response(regions):
    """L2 query-space finite differences; no per-image renormalization."""
    if not len(regions):return np.zeros(regions.shape[-2:],np.float32)
    dx=np.zeros_like(regions);dy=np.zeros_like(regions)
    dx[:,:,:-1]=regions[:,:,1:]-regions[:,:,:-1]
    dy[:,:-1,:]=regions[:,1:,:]-regions[:,:-1,:]
    return np.sqrt((dx*dx+dy*dy).sum(0)/2).astype(np.float32)


def average_precision(score,target):
    if not target.any():return None
    order=np.argsort(-score,kind='stable');y=target[order];s=score[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[ends];prec=tp/(ends+1);rec=tp/y.sum()
    # Group ties, step-wise AP, same semantics as sklearn average_precision_score.
    return float(np.sum(np.diff(np.r_[0,rec])*prec))


def boundary_metrics(raw,semantic,instance):
    gt,valid,thing=gt_boundary(semantic,instance)
    soft=(1-np.exp(-gaussian_filter(raw,sigma=1)/.1)).astype(np.float32)
    pred=(soft>=.5)&valid
    out=dict(raw_mean=float(raw[valid].mean()) if valid.any() else None,
             gt_edge_pixels=int(gt.sum()),prediction_edge_pixels=int(pred.sum()))
    for r in (1,2,3):
        gb=maximum_filter(gt,size=2*r+1)&valid;pb=maximum_filter(pred,size=2*r+1)&valid
        precision=float((pred&gb).sum()/max(1,pred.sum()));recall=float((gt&pb).sum()/max(1,gt.sum()))
        out.update({f'precision_{r}px':precision,f'recall_{r}px':recall,f'f1_{r}px':2*precision*recall/max(1e-12,precision+recall)})
    band=maximum_filter(gt,size=5)&valid
    out['continuous_ap_2px']=average_precision(raw[valid],band[valid])
    interior=thing&~maximum_filter(gt,size=7)
    out['interior_pixels']=int(interior.sum())
    out['interior_soft_mean']=float(soft[interior].mean()) if interior.any() else None
    out['interior_threshold_fraction']=float(pred[interior].mean()) if interior.any() else None
    out['interior_raw_response_fraction']=float(raw[interior].sum()/max(1e-12,raw[valid].sum()))
    out['interior_raw_mean']=float(raw[interior].mean()) if interior.any() else None
    return out,soft,gt
