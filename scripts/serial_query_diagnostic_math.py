"""Read-only, variable-query mask tracing and GT-assisted diagnostic metrics."""
import torch
import torch.nn.functional as F
from scripts.gfm_instance_diagnostic_math import iou_matrix
from scripts.export_object_locus_v3_set_official import assemble_panoptic
from tokengs.models.unified_query_nms import context_mask_nms


def distribution(x):
    x=x.detach().float().flatten()
    if not len(x):return dict(count=0,mean=None,min=None,p10=None,p50=None,p90=None,max=None)
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite diagnostic values')
    sample=x if len(x)<=65536 else x[torch.linspace(0,len(x)-1,65536,device=x.device).long()]
    q=torch.quantile(sample,torch.tensor([.1,.5,.9],device=x.device))
    return dict(count=len(x),quantile_sample_count=len(sample),mean=float(x.mean()),min=float(x.min()),p10=float(q[0]),p50=float(q[1]),p90=float(q[2]),max=float(x.max()))


def original_readout(core,q,generated):
    """Replay the trained original-Unified head at a captured token state."""
    semantic=core.semantic_adapter(q);g=generated['gaussians'];last=generated['states'][-1]
    centers,radii=last['mu'],last['radii'];n=q.shape[1];c=core.child_index.num_embeddings
    parent=semantic.repeat_interleave(c,1);index=core.child_index.weight.repeat(n,1)[None].expand(len(q),-1,-1)
    local=(g[...,:3]-centers.repeat_interleave(c,1))/radii.repeat_interleave(c,1)[...,None]
    children=core.child_feature(torch.cat((parent,index,local),-1))
    masks=core.query_mask(semantic);stuff=core.stuff_feature(semantic.mean(1)).reshape(len(q),2,core.config.dim)
    logits=children@torch.cat((masks,stuff),1).transpose(1,2)/core.config.dim**.5
    return dict(gaussian_mask_logits=logits,gaussian_membership=logits.sigmoid(),thing_logits19=core.classifier(semantic))


def mask_stages(prediction):
    """Trace the exact native thresholds/NMS/competition; verify area-filter replay."""
    mass=prediction['region_mass'][0,:2];alpha=prediction['alpha'][0,:2,0];pc=prediction['p_class'][0];n=len(pc)
    prob,cls=pc[:,:18].max(-1);class_ok=(pc.argmax(-1)!=18)&(prob>=.05)
    keep=prediction.get('instance_eligible')
    keep=keep if keep is not None else context_mask_nms(prediction['p_class'],prediction['region_mass'],prediction['alpha'])[0]
    if keep.ndim==2:keep=keep[0]
    if keep.shape!=(n,) or keep.dtype!=torch.bool:raise ValueError('Unexpected native NMS mask shape')
    raw=mass[:,:n]>=.5;visible=raw&(alpha[:,None]>.05)
    after_class=visible&class_ok[None,:,None,None];after_nms=after_class&keep[None,:,None,None]
    stuff=(mass[:,n:n+2]>=.5)&(alpha[:,None]>.05)
    candidates=torch.cat((after_nms,stuff),1)
    score=torch.cat((mass[:,:n]*prob[None,:,None,None],mass[:,n:n+2]),1)
    maximum,winner=torch.where(candidates,score,torch.full_like(score,-1.)).max(1)
    ids=torch.arange(n,device=mass.device)[None,:,None,None]
    won=(winner[:,None]==ids)&(maximum[:,None]>=0)
    raw_area=after_nms.sum((-2,-1));won_area=won.sum((-2,-1))
    retained=(raw_area>0)&(won_area.float()/raw_area.clamp_min(1)>=.5)
    replay=won&retained[:,:,None,None]
    _,ins,_=assemble_panoptic(prediction)
    final=ins[:2,None]==ids+1
    if not torch.equal(final,replay):raise ValueError('Diagnostic assembly differs from production area filtering')
    stages=dict(raw_no_alpha=raw,raw_alpha=visible,class_filtered=after_class,nms_filtered=after_nms,pixel_competition=won,final=final)
    return {k:v.permute(1,0,2,3) for k,v in stages.items()},dict(class_eligible=class_ok,nms_kept=keep,raw_area_per_view=raw_area.T,won_area_per_view=won_area.T,final_area_per_view=final.sum((-2,-1)).T,classes=cls+2)


def matrix_summary(matrix,allowed=None,include_gt=None):
    if not matrix.shape[1]:return dict(gt_count=0,best_iou_mean=None,recall_iou50=None,best_iou=[])
    values=matrix.masked_fill(~allowed,0) if allowed is not None else matrix
    best=values.max(0).values
    if include_gt is not None:best=best[include_gt]
    if not len(best):return dict(gt_count=0,best_iou_mean=None,recall_iou50=None,best_iou=[])
    return dict(gt_count=len(best),best_iou_mean=float(best.mean()),recall_iou50=float((best>=.5).float().mean()),best_iou=best.cpu().tolist())


def representation_stats(q,labels,seed=1729):
    x=q[0].float();valid=labels>=0;selected=x[valid];y=labels[valid]
    result=dict(query_norm=distribution(x.norm(dim=-1)),mean_channel_variance=float(x.var(0,unbiased=False).mean()))
    if len(selected)<2:return dict(result,same_gt_cosine=None,different_gt_cosine=None,cosine_separation=None,shuffle_separation_mean=None,shuffle_separation_std=None)
    normalized=F.normalize(selected,dim=-1);cos=normalized@normalized.T
    off=~torch.eye(len(y),device=y.device,dtype=torch.bool);same=(y[:,None]==y[None])&off;different=(y[:,None]!=y[None])&off
    def mean(mask):return float(cos[mask].mean()) if mask.any() else None
    a,b=mean(same),mean(different);shuffled=[]
    generator=torch.Generator().manual_seed(seed)
    for _ in range(16):
        yy=y[torch.randperm(len(y),generator=generator).to(y.device)]
        ss=(yy[:,None]==yy[None])&off;dd=(yy[:,None]!=yy[None])&off
        if ss.any() and dd.any():shuffled.append(float(cos[ss].mean()-cos[dd].mean()))
    return dict(result,same_gt_cosine=a,different_gt_cosine=b,cosine_separation=a-b if a is not None and b is not None else None,shuffle_separation_mean=sum(shuffled)/len(shuffled) if shuffled else None,shuffle_separation_std=float(torch.tensor(shuffled).std(unbiased=False)) if shuffled else None)


def analyze_native(prediction,targets,allowed,support,seed=1729,detailed=True):
    stages,trace=mask_stages(prediction);gt=targets['gt_pixel_masks'][0];valid=targets['valid_pixels'][0];classes=targets['gt_classes'][0]
    n=len(trace['classes']);k=len(classes)
    matrices={name:iou_matrix(mask,gt,valid) for name,mask in stages.items()}
    correct=(trace['classes'][:,None]==classes[None])&trace['class_eligible'][:,None] if k else allowed
    result=dict(gt_count=k,supported_gt=int(allowed.any(0).sum()),query_count=n,
        stage_metrics={name:dict(class_agnostic=matrix_summary(m),correct_class=matrix_summary(m,correct),source_supported_all_gt=matrix_summary(m,allowed),source_supported_only=matrix_summary(m,allowed,allowed.any(0))) for name,m in matrices.items()})
    qi=allowed.any(1).nonzero().flatten();ki=support.masked_fill(~allowed,-1)[qi].argmax(1) if k else qi
    labels=torch.full((n,),-1,device=valid.device,dtype=torch.long);labels[qi]=ki
    flat_all=prediction['region_mass'][0,:2,:n].permute(1,0,2,3).flatten(1)
    p=flat_all[:,valid.flatten()]
    y=gt.flatten(1)[:,valid.flatten()].float()
    positive=p[qi];truth=y[ki].bool() if len(qi) else positive.bool()
    soft_inter=(positive*truth).sum(1);soft_iou=soft_inter/(positive.sum(1)+truth.sum(1)-soft_inter).clamp_min(1e-6)
    gm=prediction['gaussian_membership'][0,:,:n];logits=prediction['gaussian_mask_logits'][0,:,:n]
    children=gm.shape[0]//n
    own=gm.reshape(n,children,n)[torch.arange(n,device=gm.device),:,torch.arange(n,device=gm.device)]
    other_mean=(gm.sum(0)-own.sum(1))/max(1,gm.shape[0]-children)
    visible=(prediction['alpha'][0,:2,0]>.05).flatten()
    raw_max=flat_all.max(1).values;visible_p=prediction['region_mass'][0,:2,:n].permute(1,0,2,3).flatten(1)[:,visible]
    max_visible=visible_p.max(1).values if visible.any() else raw_max.new_zeros(n)
    result.update(alpha=distribution(prediction['alpha']),alpha_visible_fraction=float(visible.float().mean()),gaussian_logits=distribution(logits),gaussian_probability=distribution(gm),
        query_probability_on_gt=distribution(positive[truth]),query_probability_off_gt=distribution(positive[~truth]),positive_query_soft_iou=distribution(soft_iou),
        class_eligible=int(trace['class_eligible'].sum()),nms_kept=int(trace['nms_kept'].sum()),final_instances=int((stages['final'].flatten(1).sum(1)>0).sum()),
        gaussian_own_parent_probability=distribution(own),gaussian_other_parent_mean_probability=distribution(other_mean),gaussian_parent_warning='Parent identity proxy only; no ground-truth labels are assigned to moved 3D Gaussians',gaussian_query_max=distribution(gm.max(0).values),rendered_query_max=distribution(raw_max),visible_query_max=distribution(max_visible),
        queries_gaussian_below_half=int((gm.max(0).values<.5).sum()),queries_no_rendered_half=int((raw_max<.5).sum()),queries_no_visible_half=int((max_visible<.5).sum()))
    if not detailed:return result,labels,stages,matrices
    result['threshold_diagnostic_only']={}
    for threshold in [.1,.2,.3,.4,.5,.6,.7]:
        raw=prediction['region_mass'][0,:2,:n].permute(1,0,2,3)>=threshold
        with_alpha=raw&(prediction['alpha'][0,:2,0][None]>.05)
        result['threshold_diagnostic_only'][str(threshold)]={name:matrix_summary(iou_matrix(mask,gt,valid),allowed,allowed.any(0)) for name,mask in [('no_alpha',raw),('alpha',with_alpha)]}
        positive_masks=with_alpha[qi].flatten(1)[:,valid.flatten()]
        result['threshold_diagnostic_only'][str(threshold)].update(positive_query_intersection_pixels=int((positive_masks&truth).sum()),positive_query_prediction_pixels=int(positive_masks.sum()),positive_query_gt_pixels=int(truth.sum()))
    generator=torch.Generator().manual_seed(seed);shuffle=[]
    for _ in range(16):
        order=torch.randperm(n,generator=generator).to(valid.device)
        summary=matrix_summary(matrices['raw_alpha'][order],allowed,allowed.any(0))
        if summary['best_iou_mean'] is not None:shuffle.append(summary['best_iou_mean'])
    result['shuffled_query_correspondence_best_iou_mean']=sum(shuffle)/len(shuffle) if shuffle else None
    result['queries']=[]
    for q in range(n):
        target=int(labels[q]);entry=dict(query=q,source_gt=target,predicted_class=int(trace['classes'][q]),no_object=bool(prediction['p_class'][0,q].argmax()==18),
            class_eligible=bool(trace['class_eligible'][q]),nms_kept=bool(trace['nms_kept'][q]),gaussian_max=float(gm[:,q].max()),rendered_max=float(raw_max[q]),visible_max=float(max_visible[q]),
            raw_pixels=int(stages['raw_alpha'][q].sum()),competition_pixels=int(stages['pixel_competition'][q].sum()),final_pixels=int(stages['final'][q].sum()))
        if target>=0:entry.update(gt_class=int(classes[target]),raw_iou=float(matrices['raw_alpha'][q,target]),final_iou=float(matrices['final'][q,target]))
        result['queries'].append(entry)
    return result,labels,stages,matrices
