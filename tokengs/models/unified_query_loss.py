"""Dynamic superpoint-query instance loss; context GT is used only after generation."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets,_probability


@torch.no_grad()
def support_relation(prediction,batch,targets,purity,return_scores=False):
    cloud=prediction['cloud'];source=cloud['source_indices'];h,w=batch['semantic_label_all'].shape[-2:]
    labels=batch['instance_label_all'][0,:2].reshape(-1)[source]
    sem=batch['semantic_label_all'][0,:2].reshape(-1)[source]
    ids=prediction['superpoints']['source_superpoint_ids'];n=prediction['num_thing_queries']
    valid=(sem>=0)&(sem<=19)&((sem<2)|(labels>0))
    counts=torch.bincount(ids[valid],minlength=n).clamp_min(1)
    scores=torch.zeros(n,len(targets['gt_instance_ids'][0]),device=ids.device)
    for k,iid in enumerate(targets['gt_instance_ids'][0]):
        hit=valid&(sem>=2)&(labels==iid)
        hits=torch.bincount(ids[hit],minlength=n)
        scores[:,k]=hits.float()/counts
    allowed=(scores>0)&(scores>=purity)
    return (allowed,scores) if return_scores else allowed


def unified_instance_loss(prediction,batch,config):
    logits=prediction['thing_logits19'].float();n=logits.shape[1]
    if logits.shape[0]!=1:raise ValueError('Ragged unified loss expects one scene per rank')
    targets=build_context_instance_targets(batch);classes=targets['gt_classes'][0];k=len(classes)
    regions=_probability(prediction['region_mass'][0,:2].float());valid=targets['valid_pixels'][0]
    if not valid.any():raise ValueError('No valid context supervision')
    flat=regions.permute(1,0,2,3).reshape(n+2,-1)[:,valid.flatten()]
    gt=targets['gt_pixel_masks'][0].reshape(k,-1)[:,valid.flatten()].float() if k else flat.new_empty((0,flat.shape[1]))
    with torch.no_grad():
        allowed=support_relation(prediction,batch,targets,config.support_purity) if config.support_matching else torch.ones(n,k,dtype=torch.bool,device=logits.device)
        if k and config.matching_mode=='source_many_to_one':
            allowed,support=support_relation(prediction,batch,targets,config.support_purity,return_scores=True)
            qi=allowed.any(1).nonzero().flatten()
            ki=support.masked_fill(~allowed,-1)[qi].argmax(1)
        elif k:
            masks=flat[:n].clamp(1e-6,1-1e-6)
            bce=-torch.log1p(-masks).mean(-1,keepdim=True)-(masks.log()-torch.log1p(-masks))@gt.T/masks.shape[1]
            dice=1-(2*masks@gt.T+1)/(masks.sum(-1,keepdim=True)+gt.sum(-1)[None]+1)
            cost=2*-logits[0].softmax(-1)[:,classes-2]+5*bce+5*dice
            # Filter forbidden assignments after solving; no fabricated match
            # when a GT has no sufficiently pure supporting superpoint.
            row,col=linear_sum_assignment(cost.masked_fill(~allowed,1e6).cpu().numpy())
            qi=torch.as_tensor(row,device=logits.device);ki=torch.as_tensor(col,device=logits.device)
            keep=allowed[qi,ki];qi,ki=qi[keep],ki[keep]
        else:qi=ki=torch.empty(0,dtype=torch.long,device=logits.device)
    labels=torch.full((1,n),18,device=logits.device,dtype=torch.long);labels[0,qi]=classes[ki]-2
    weights=logits.new_ones(19);weights[18]=.1
    def instance_mean(values):
        # Each supported GT has equal weight regardless of its superpoint count.
        unique,inverse=torch.unique(ki,sorted=True,return_inverse=True)
        sums=values.new_zeros(len(unique)).index_add(0,inverse,values)
        return (sums/torch.bincount(inverse,minlength=len(unique)).to(values.dtype)).mean()
    if config.matching_mode=='source_many_to_one':
        per_query=F.cross_entropy(logits.transpose(1,2),labels,reduction='none')[0]
        negative=torch.ones(n,dtype=torch.bool,device=logits.device);negative[qi]=False
        groups=len(ki.unique());denominator=groups+(.1 if negative.any() else 0.)
        positive=instance_mean(per_query[qi])*groups if len(qi) else per_query.sum()*0
        background=.1*per_query[negative].mean() if negative.any() else per_query.sum()*0
        classification=(positive+background)/max(denominator,1e-6)
    else:classification=F.cross_entropy(logits.transpose(1,2),labels,weight=weights)
    zero=flat.sum()*0
    if len(qi):
        p,y=flat[qi],gt[ki]
        bce_values=F.binary_cross_entropy(p,y,reduction='none').mean(-1)
        dice_values=1-(2*(p*y).sum(-1)+1)/(p.sum(-1)+y.sum(-1)+1)
        bce=instance_mean(bce_values) if config.matching_mode=='source_many_to_one' else bce_values.mean()
        dice=instance_mean(dice_values) if config.matching_mode=='source_many_to_one' else dice_values.mean()
    else:bce=dice=zero
    stuff_bce=[];stuff_dice=[];sem=batch['semantic_label_all'][0,:2][valid]
    for cls in (0,1):
        p=flat[n+cls];y=(sem==cls).float()
        stuff_bce.append(F.binary_cross_entropy(p,y));stuff_dice.append(1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1))
    sb=torch.stack(stuff_bce).mean();sd=torch.stack(stuff_dice).mean()
    loss=.1*(2*classification+5*bce+5*dice+5*sb+5*sd)
    prediction['final_pairs']=[(qi,ki)];prediction['final_targets']=targets
    return loss,dict(loss_classification=classification.detach(),loss_thing_bce=bce.detach(),loss_thing_dice=dice.detach(),loss_stuff_bce=sb.detach(),loss_stuff_dice=sd.detach(),matched_gt_count=len(ki.unique()),matched_query_count=len(qi),gt_count=k,
        mean_positive_queries_per_matched_gt=len(qi)/max(1,len(ki.unique())),
        unsupported_GT_count=int((~allowed.any(0)).sum()) if k else 0,matched_classification_accuracy=float((logits[0,qi].argmax(-1)==classes[ki]-2).float().mean()) if len(qi) else 0.)
