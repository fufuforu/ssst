"""Prediction-only, class-aware context Mask NMS. Does not remove Gaussians."""
import torch


@torch.no_grad()
def context_mask_nms(p_class,region_mass,alpha,iou_threshold=.7):
    """Rank by class probability * mean foreground mask probability.

    Compare concatenated two-context masks. The same keep mask is reused for
    target views; never inspect target labels or derive a new target identity.
    """
    if not 0<iou_threshold<=1:raise ValueError('Invalid NMS threshold')
    p=p_class[0];n=len(p);prob,cls=p[:,:18].max(-1)
    eligible=(p.argmax(-1)!=18)&(prob>=.05)
    masks=region_mass[0,:2,:n].permute(1,0,2,3)
    masks=(masks>=.5)&(alpha[0,:2,0][None]>.05)
    masks=masks.flatten(1);area=masks.sum(1);eligible &= area>0
    values=region_mass[0,:2,:n].permute(1,0,2,3).flatten(1)
    scores=prob*(values*masks).sum(1)/area.clamp_min(1)
    order=torch.argsort(scores,descending=True,stable=True)
    keep=torch.zeros(n,dtype=torch.bool,device=p.device);selected=[]
    for q in order.tolist():
        if not bool(eligible[q]):continue
        peers=[j for j in selected if int(cls[j])==int(cls[q])]
        if peers:
            inter=(masks[peers]&masks[q]).sum(1)
            iou=inter/(area[peers]+area[q]-inter).clamp_min(1)
            if bool((iou>iou_threshold).any()):continue
        keep[q]=True;selected.append(q)
    return keep,scores
