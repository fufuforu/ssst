"""Read-only matching audit. Production support/cost/loss functions stay unchanged."""
from dataclasses import replace
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.models import unified_query_loss as native


@torch.no_grad()
def source_histogram(prediction, batch, targets):
    source = prediction['cloud']['source_indices']
    ids = prediction['superpoints']['source_superpoint_ids']
    n = prediction['num_thing_queries']
    sem = batch['semantic_label_all'][0, :2].flatten()[source]
    ins = batch['instance_label_all'][0, :2].flatten()[source]
    assert source.ndim == ids.ndim == 1 and len(source) == len(ids)
    assert ids.numel() and int(ids.min()) >= 0 and int(ids.max()) < n
    valid = (sem >= 0) & (sem <= 19) & ((sem < 2) | (ins > 0))
    counts = torch.bincount(ids[valid], minlength=n)
    hits = torch.stack([torch.bincount(ids[valid & (sem >= 2) & (ins == iid)], minlength=n)
                        for iid in targets['gt_instance_ids'][0]], dim=1) if len(targets['gt_instance_ids'][0]) else counts.new_empty((n, 0))
    purity = hits.float() / counts.clamp_min(1)[:, None]
    allowed = (hits > 0) & (purity >= .5)
    ref_allowed, ref_scores = native.support_relation(prediction, batch, targets, .5, return_scores=True)
    if not torch.equal(purity, ref_scores) or not torch.equal(allowed, ref_allowed):
        raise AssertionError('support_relation exact parity FAILED')
    return hits, counts, purity, allowed


@torch.no_grad()
def capture_native_matching(prediction, batch, config):
    """Capture the exact masked cost passed by production loss to SciPy.

    The temporary observer only wraps the solver, restores it in finally, and
    is used in a single-threaded audit process. No production file is edited.
    """
    captured = []
    original = native.linear_sum_assignment
    def observe(cost):
        captured.append(cost.copy())
        return original(cost)
    native.linear_sum_assignment = observe
    try:
        loss, metrics = native.unified_instance_loss(prediction, batch, replace(config, matching_mode='hungarian'))
    finally:
        native.linear_sum_assignment = original
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite native audit loss')
    qi, ki = prediction['final_pairs'][0]
    pair = list(zip(qi.tolist(), ki.tolist()))
    n = prediction['num_thing_queries']; k = metrics['gt_count']
    cost = captured[0] if captured else np.empty((n, k), dtype=np.float32)
    assert len(captured) == int(k > 0)
    # Offline solve vs the actual loss output, same mask and exact cost.
    allowed = native.support_relation(prediction, batch, prediction['final_targets'], config.support_purity).cpu().numpy()
    row, col = original(cost)
    offline = [(int(q), int(g)) for q, g in zip(row, col) if allowed[q, g]]
    if pair != offline:
        raise AssertionError('Hungarian native pair parity FAILED')
    # A second unobserved native call validates that observation did not change pairs.
    loss2, _ = native.unified_instance_loss(prediction, batch, replace(config, matching_mode='hungarian'))
    q2, g2 = prediction['final_pairs'][0]
    if pair != list(zip(q2.tolist(), g2.tolist())) or not torch.equal(loss, loss2):
        raise AssertionError('Native observer parity FAILED')
    return cost, pair


def oracle_and_top1(allowed, cost):
    n, k = allowed.shape
    max_pairs = maximum_bipartite_matching(csr_matrix(allowed), perm_type='column') if n and k else np.full(n, -1)
    top = []
    for g in range(k):
        candidates = np.flatnonzero(allowed[:, g])
        if len(candidates):
            q = int(candidates[np.argmin(cost[candidates, g])])
            top.append((q, g))
    # Faithful OneFormer SparseMatcher(topk=1): strict < second minimum.
    # Its actual topk(2) is undefined for fewer than two queries.
    official = None
    if n >= 2:
        masked = np.where(allowed, cost, np.float32(1e8))
        second = np.sort(masked, axis=0)[1:2]
        official = [tuple(map(int, x)) for x in np.argwhere(masked < second)]
    counts = np.bincount([q for q, _ in top], minlength=n)
    return dict(support_upper=int(allowed.any(0).sum()), maximum=int((max_pairs >= 0).sum()), top=top,
                official=official, conflict_queries=np.flatnonzero(counts > 1).tolist(),
                conflict_excess=int(np.maximum(counts - 1, 0).sum()), multi_gt_queries=int((allowed.sum(1) > 1).sum()))


@torch.no_grad()
def raw_iou(prediction, batch, targets):
    """Context-only, class-agnostic raw masks >=.5; official semantic valid pixels.

    Official export valid is 0<=semantic<=19, which retains thing/instance=0
    pixels. Targets are the same native positive-instance context GT. No NMS,
    minimum area, score threshold, class filtering, or packed-label competition.
    """
    n = prediction['num_thing_queries']; k = len(targets['gt_instance_ids'][0])
    sem = batch['semantic_label_all'][0, :2]
    valid = ((sem >= 0) & (sem <= 19)).flatten()
    p = prediction['region_mass'][0, :2, :n].permute(1, 0, 2, 3).flatten(1)[:, valid]
    y = targets['gt_pixel_masks'][0].flatten(1)[:, valid].float()
    masks = (p >= .5).float()
    inter = masks @ y.T
    iou = inter / (masks.sum(1)[:, None] + y.sum(1)[None, :] - inter).clamp_min(1)
    return iou.cpu().numpy(), int((valid & ~targets['valid_pixels'][0].flatten()).sum())


@torch.no_grad()
def audit_window(prediction, batch, config, identity, diagnostic_dir):
    targets = build_context_instance_targets(batch)
    try:
        hits, counts, scores, allowed = source_histogram(prediction, batch, targets)
        cost, pairs = capture_native_matching(prediction, batch, config)
    except Exception:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(identity=identity, source_indices=prediction['cloud']['source_indices'].cpu(),
                        source_superpoint_ids=prediction['superpoints']['source_superpoint_ids'].cpu(),
                        semantic_context=batch['semantic_label_all'][:, :2].cpu(),
                        instance_context=batch['instance_label_all'][:, :2].cpu(),
                        thing_logits19=prediction['thing_logits19'].cpu(),
                        region_mass=prediction['region_mass'][:, :2].cpu()),
                   diagnostic_dir / (identity['window_id'] + '_parity_failure.pt'))
        raise
    h, den, p, a = [x.cpu().numpy() for x in (hits, counts, scores, allowed)]
    if not np.isfinite(cost).all():
        raise FloatingPointError('Nonfinite matching cost')
    oracle = oracle_and_top1(a, cost)
    iou, extra_valid = raw_iou(prediction, batch, targets)
    n, k = a.shape; hp = {g: q for q, g in pairs}; tp = {g: q for q, g in oracle['top']}
    op = {g: q for q, g in oracle['official']} if oracle['official'] is not None else {}
    classes = targets['gt_classes'][0].tolist()
    predicted = prediction['thing_logits19'][0].argmax(-1).cpu().numpy()
    areas = targets['gt_pixel_masks'][0].flatten(1).sum(1).tolist()
    thing_hits = h.sum(1); query_purity = p.max(1) if k else np.zeros(n)
    rows = []; comparison = []; sweep = []
    for g, iid in enumerate(targets['gt_instance_ids'][0].tolist()):
        candidates = int(a[:, g].sum()); sampled = int(h[:, g].sum()); q = hp.get(g); t = tp.get(g)
        raw = iou[:, g] >= .5; supported_raw = raw & a[:, g]
        reason = 'covered' if candidates else 'zero_source_hits' if not sampled else 'below_purity'
        row = dict(**identity, gt_index=g, gt_instance_id=iid, semantic_class=classes[g], area_pixels=areas[g],
                   source_hits=sampled, sparse_source_lt3=0 < sampled < 3, sparse_source_lt5=0 < sampled < 5,
                   eligible_queries=candidates, max_purity=float(p[:, g].max()) if n else 0.,
                   max_query_hits=int(h[:, g].max()) if n else 0, uncovered_reason=reason,
                   hungarian_query=q, top1_query=t, official_strict_query=op.get(g),
                   hungarian_matched=q is not None, top1_matched=t is not None,
                   same_selected_query=q is not None and q == t,
                   raw_mask=bool(raw.any()), supported_raw_mask=bool(supported_raw.any()),
                   matched_mask_good=bool(q is not None and iou[q, g] >= .5),
                   supported_no_raw=bool(candidates and not raw.any()),
                   raw_not_matched=bool(raw.any() and q is None),
                   raw_not_selected=bool(raw.any() and (q is None or not raw[q])),
                   raw_all_noobject=bool(raw.any() and (predicted[raw] == 18).all()),
                   hungarian_noobject=bool(q is not None and predicted[q] == 18),
                   best_raw_iou=float(iou[:, g].max()) if n else 0,
                   best_raw_query=int(iou[:, g].argmax()) if n else None,
                   best_raw_noobject=bool(raw.any() and predicted[iou[:, g].argmax()] == 18),
                   hungarian_cost=float(cost[q, g]) if q is not None else None,
                   top1_cost=float(cost[t, g]) if t is not None else None)
        rows.append(row)
        comparison.append({key: row[key] for key in (*identity.keys(), 'gt_index', 'gt_instance_id', 'hungarian_query', 'top1_query', 'official_strict_query', 'same_selected_query', 'hungarian_cost', 'top1_cost')})
        for tau in (.3, .5, .7, .9):
            for mh in (1, 3, 5):
                eligible = (h[:, g] >= mh) & (p[:, g] >= tau)
                sweep.append(dict(**identity, gt_index=g, area_pixels=areas[g], purity=tau, min_hits=mh, eligible_queries=int(eligible.sum()), covered=bool(eligible.any())))
    eligible_counts = a.sum(0)
    scene = dict(**identity, queries=n, gt=k, source_observations=int(len(prediction['cloud']['source_indices'])),
                 valid_source_observations=int(den.sum()), zero_thing_queries=int((thing_hits == 0).sum()),
                 zero_valid_queries=int((den == 0).sum()), support_upper=oracle['support_upper'], maximum_matching=oracle['maximum'],
                 hungarian_matched=len(pairs), hungarian_unmatched=k-len(pairs),
                 eligible_unmatched=int(sum(g not in hp and a[:, g].any() for g in range(k))),
                 matched_queries=len({q for q, _ in pairs}), top1_matched=len(tp),
                 identical_pairs=set(pairs) == set(oracle['top']), same_query_gt=sum(r['same_selected_query'] for r in rows),
                 different_gt=sum(hp.get(g) != tp.get(g) for g in range(k)),
                 multi_gt_queries=oracle['multi_gt_queries'], top1_conflict_queries=len(oracle['conflict_queries']), top1_conflict_excess=oracle['conflict_excess'],
                 official_strict_defined=oracle['official'] is not None, official_strict_matches=len(op),
                 official_diff_top1=set(op.items()) != set(tp.items()) if oracle['official'] is not None else None,
                 hungarian_cost_sum=sum(float(cost[q, g]) for q, g in pairs), top1_cost_sum=sum(float(cost[q, g]) for q, g in oracle['top']),
                 common_gt_cost_delta=sum(float(cost[hp[g], g]-cost[tp[g], g]) for g in set(hp)&set(tp)),
                 eligible_mean=float(eligible_counts.mean()) if k else None,
                 eligible_median=float(np.median(eligible_counts)) if k else None,
                 eligible_p10=float(np.quantile(eligible_counts, .1)) if k else None,
                 eligible_p90=float(np.quantile(eligible_counts, .9)) if k else None,
                 raw_mask_gt=sum(r['raw_mask'] for r in rows), supported_raw_gt=sum(r['supported_raw_mask'] for r in rows),
                 matched_good_gt=sum(r['matched_mask_good'] for r in rows), raw_valid_extra_pixels=extra_valid,
                 support_parity=True, hungarian_parity=True)
    for tau in (.3, .5, .7, .9):
        scene[f'query_purity_ge_{tau}'] = int((query_purity >= tau).sum())
    # All compact candidate graphs are saved, even if no failure occurs.
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(diagnostic_dir / (identity['window_id'] + '.npz'), hits=h, valid_counts=den,
                        purity=p, allowed=a, masked_cost=cost, context_iou=iou,
                        gt_ids=np.asarray(targets['gt_instance_ids'][0].tolist()), gt_classes=np.asarray(classes),
                        hungarian=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                        top1=np.asarray(oracle['top'], dtype=np.int64).reshape(-1, 2),
                        predicted_class=predicted)
    return scene, rows, comparison, sweep
