"""Context-GT diagnostics on fixed predicted memberships; never create GT masks.

All decisions take exactly two context views. Returned decisions are applied
unchanged to all six rendered views. These are diagnostic oracles, not inference.
"""
import torch

ARMS = ('E0', 'E1', 'E2_class', 'E2_filter', 'E2_rank', 'E2_joint')


def associate(membership, alpha, semantic, instance):
    """Class-blind many-to-one association by >50% raw-mask purity.

    Mask area includes background/void, so tiny object intersections cannot
    claim a predominantly background query. Instance identity is (class, id).
    """
    if membership.ndim != 4 or membership.shape[0] != 2:
        raise ValueError('Oracle association requires exactly two context views')
    if semantic.shape != instance.shape or semantic.shape != membership.shape[:1] + membership.shape[-2:]:
        raise ValueError('Context GT shape differs')
    masks = (membership >= .5) & (alpha[:, None] > .05)
    qmask = masks.permute(1, 0, 2, 3).flatten(1)
    keys = torch.unique(torch.stack((semantic.flatten(), instance.flatten()), 1), dim=0)
    keys = keys[(keys[:, 0] >= 2) & (keys[:, 0] <= 19) & (keys[:, 1] > 0)]
    gt = torch.stack([(semantic == c) & (instance == i) for c, i in keys]) if len(keys) else masks.new_zeros((0, *semantic.shape))
    flat = gt.flatten(1)
    overlap = qmask.float() @ flat.float().T
    area = qmask.sum(1)
    assignment = torch.full((len(qmask),), -1, device=qmask.device, dtype=torch.long)
    purity = area.new_zeros(len(qmask), dtype=torch.float)
    if len(keys):
        inter, best = overlap.max(1)
        purity = inter / area.clamp_min(1)
        supported = (area > 0) & (purity > .5)
        assignment[supported] = best[supported]
    return dict(keys=keys, gt=gt, masks=qmask, assignment=assignment,
                purity=purity, area=area, overlap=overlap)


def nms_with_candidates(out, candidates, threshold=.7):
    """Original class-aware NMS with only its input eligibility replaced."""
    p = out['p_class'][0]
    n = len(p)
    probability, cls = p[:, :18].max(-1)
    masks = ((out['region_mass'][0, :2, :n] >= .5) & (out['alpha'][0, :2] > .05)).permute(1, 0, 2, 3).flatten(1)
    values = out['region_mass'][0, :2, :n].permute(1, 0, 2, 3).flatten(1)
    area = masks.sum(1)
    scores = probability * (values * masks).sum(1) / area.clamp_min(1)
    keep = torch.zeros_like(candidates)
    selected = []
    for q in torch.argsort(scores, descending=True, stable=True).tolist():
        if not bool(candidates[q]) or not int(area[q]):
            continue
        peers = [j for j in selected if int(cls[j]) == int(cls[q])]
        if peers:
            inter = (masks[peers] & masks[q]).sum(1)
            if bool((inter / (area[peers] + area[q] - inter).clamp_min(1) > threshold).any()):
                continue
        keep[q] = True
        selected.append(q)
    return keep


def corrected_classes(pclass, association):
    """Swap thing probabilities; preserve max thing score and no-object value."""
    result = pclass.clone()
    for q, g in enumerate(association['assignment'].tolist()):
        if g < 0:
            continue
        original = int(pclass[0, q, :18].argmax())
        corrected = int(association['keys'][g, 0]) - 2
        result[0, q, original] = pclass[0, q, corrected]
        result[0, q, corrected] = pclass[0, q, original]
        # Equal maxima otherwise retain the old lowest-index label. Break only
        # those ties by one ULP, keeping the exact maximum/no-object score.
        maximum = result[0, q, corrected]
        ties = (result[0, q, :18] == maximum)
        ties[corrected] = False
        result[0, q, :18][ties] = torch.nextafter(maximum, maximum.new_zeros(()))
    return result


def oracle_variants(out, context_semantic, context_instance, *, nms_threshold=.7):
    n = out['p_class'].shape[1]
    a = associate(out['region_mass'][0, :2, :n], out['alpha'][0, :2, 0], context_semantic, context_instance)
    p = out['p_class'][0]
    gates = (p.argmax(-1) != 18) & (p[:, :18].max(-1).values >= .05)
    keep = gates & out['instance_eligible']
    result = {'E0': (out, keep)}
    # E1: merge only baseline-kept queries. Unassigned queries remain unchanged.
    # Fixed representative: highest original context ranking; no GT class/score.
    merged = dict(out)
    mass = out['region_mass'].clone()
    merged_keep = keep.clone()
    raw_masks = a['masks']
    values = out['region_mass'][0, :2, :n].permute(1, 0, 2, 3).flatten(1)
    scores = p[:, :18].max(-1).values * (values * raw_masks).sum(1) / a['area'].clamp_min(1)
    groups = []
    coverage = []
    for g in range(len(a['keys'])):
        members = ((a['assignment'] == g) & keep).nonzero().flatten()
        all_members = (a['assignment'] == g).nonzero().flatten()
        gt = a['gt'][g].flatten()
        def stats(ids):
            if not len(ids):
                return dict(count=0, best_iou=0., union_iou=0., best_coverage=0., union_coverage=0.)
            masks = raw_masks[ids]
            inter = (masks & gt).sum(1)
            union = masks.any(0)
            return dict(count=len(ids), best_iou=float((inter / (masks.sum(1) + gt.sum() - inter).clamp_min(1)).max()),
                        union_iou=float((union & gt).sum() / (union | gt).sum().clamp_min(1)),
                        best_coverage=float(inter.max() / gt.sum().clamp_min(1)),
                        union_coverage=float((union & gt).sum() / gt.sum().clamp_min(1)))
        coverage.append(dict(gt_class=int(a['keys'][g, 0]), gt_id=int(a['keys'][g, 1]), raw=stats(all_members), baseline_kept=stats(members)))
        if len(members) < 2:
            continue
        representative = int(members[torch.argsort(scores[members], descending=True, stable=True)[0]])
        mass[0, :, representative] = out['region_mass'][0, :, members].amax(1)
        others = members[members != representative]
        mass[0, :, others] = 0
        merged_keep[others] = False
        groups.append(dict(representative=representative, queries=members.tolist(), gt_key=a['keys'][g].tolist()))
    merged['region_mass'] = mass
    result['E1'] = (merged, merged_keep)
    classified = dict(out, p_class=corrected_classes(out['p_class'], a))
    result['E2_class'] = (classified, keep)
    supported = a['assignment'] >= 0
    result['E2_filter'] = (out, nms_with_candidates(out, supported, nms_threshold))
    result['E2_rank'] = (out, keep)
    result['E2_joint'] = (classified, nms_with_candidates(classified, supported, nms_threshold))
    audit = dict(oracle_gt_scope='context_only', purity_threshold_strict=.5, membership_threshold=.5,
                 alpha_threshold=.05, queries=n, baseline_gate_count=int(gates.sum()), baseline_keep_count=int(keep.sum()),
                 supported_queries=int(supported.sum()), supported_rejected_no_object=int((supported & (p.argmax(-1) == 18)).sum()),
                 supported_rejected_confidence=int((supported & (p[:, :18].max(-1).values < .05)).sum()),
                 supported_rejected_nms=int((supported & gates & ~out['instance_eligible']).sum()),
                 wrong_class_supported=int((supported & (p[:, :18].argmax(-1) != torch.tensor([int(a['keys'][g, 0])-2 if g >= 0 else -1 for g in a['assignment'].tolist()], device=p.device))).sum()),
                 merge_groups=groups, instances=coverage)
    return result, audit


def context_ranking_scores(out, eligible, context_semantic, context_instance):
    """Class-blind max IoU of final context masks; never use target GT."""
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    context = dict(out)
    for key in ('region_mass', 'alpha', 'semantic_scores'):
        context[key] = out[key][:, :2]
    _, ins, _ = assemble_panoptic(context, eligible_override=eligible)
    n = out['p_class'].shape[1]
    # Inputs are membership-shaped binary final masks; retain the fixed identities.
    masks = torch.stack([(ins[:2] == q+1).float() for q in range(n)], 1)
    a = associate(masks, torch.ones_like(ins[:2]), context_semantic, context_instance)
    if not len(a['keys']):
        return [0.] * n
    iou = a['overlap'] / (a['area'][:, None] + a['gt'].flatten(1).sum(1)[None] - a['overlap']).clamp_min(1)
    return iou.max(1).values.tolist()
