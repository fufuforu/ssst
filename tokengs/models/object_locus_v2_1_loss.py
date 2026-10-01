"""Independent-mask supervision for Object-Locus V2.1."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tokengs.models.anchor_group_loss import IGNORE, THING, WALL, FLOOR
from tokengs.models.instance_state_loss import (
    CLASS_CE_WEIGHT, MASK_BCE_WEIGHT, MASK_DICE_WEIGHT,
    _flat_regions, _linspace_indices, identity_loss,
)
from tokengs.models.object_locus_v1_loss import build_visible_anchor_targets, object_locus_stuff_loss


@torch.no_grad()
def final_hungarian(prediction, batch, targets=None, match_points=4096):
    final = prediction["states"][-1]
    targets = targets or build_visible_anchor_targets(final["mu"], batch)
    sem, ins = batch["semantic_label_all"][:, :2].long(), batch["instance_label_all"][:, :2].long()
    region = _flat_regions(prediction["region_mass"][:, :, :100], 100)
    classes_logits = final["thing_logits19"]
    mask_logits = final["anchor_mask_logits"]
    pairs = []
    for b in range(sem.shape[0]):
        classes, masks = targets["gt_classes"][b], targets["gt_pixel_masks"][b]
        k = int(classes.numel())
        if k == 0:
            empty = torch.empty(0, device=sem.device, dtype=torch.long)
            pairs.append((empty, empty.clone())); continue
        valid_pixels = (sem[b] >= 0) & (sem[b] <= 19) & ((sem[b] < 2) | (ins[b] > 0))
        flat = torch.nonzero(valid_pixels.reshape(-1), as_tuple=False).flatten()
        if flat.numel() == 0:
            raise RuntimeError("valid context pixel set is empty for V2 Hungarian")
        flat = flat[_linspace_indices(int(flat.numel()), match_points).to(flat.device)]
        p = region[b, :, flat].clamp(1e-6, 1.0 - 1e-6)
        z = torch.logit(p)
        y = masks.flatten(1)[:, flat].float()
        class_prob = torch.softmax(classes_logits[b].float(), dim=-1)
        class_cost = -class_prob[:, classes - 2]
        pixel_bce = F.softplus(z).mean(1, keepdim=True) - z @ y.T / float(flat.numel())
        sig = torch.sigmoid(z)
        pixel_dice = 1.0 - (2 * sig @ y.T + 1) / (sig.sum(1, keepdim=True) + y.sum(1)[None] + 1)
        valid_anchor = targets["anchor_valid"][b]
        if bool(valid_anchor.any()):
            la = mask_logits[b, valid_anchor, :100].T.float()  # [Q,N], thing queries only
            ya = targets["Y_anchor"][b, :, valid_anchor].float()  # [K,N]
            support = targets["Y_anchor"][b].sum(-1) > 0
            anc_bce = F.softplus(la).mean(1, keepdim=True) - la @ ya.T / float(la.shape[1])
            pa = torch.sigmoid(la)
            anc_dice = 1 - (2 * pa @ ya.T + 1) / (pa.sum(1, keepdim=True) + ya.sum(1)[None] + 1)
            anc_bce = anc_bce * support[None]
            anc_dice = anc_dice * support[None]
        else:
            anc_bce = torch.zeros_like(class_cost); anc_dice = torch.zeros_like(class_cost)
        cost = -class_prob[:, classes - 2] + 5 * pixel_bce + 5 * pixel_dice + 2 * anc_bce + 2 * anc_dice
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
        pairs.append((torch.as_tensor(rows, device=sem.device, dtype=torch.long),
                      torch.as_tensor(cols, device=sem.device, dtype=torch.long)))
    return targets, pairs


def _category_objectness_losses(state, pairs, targets):
    cat_logits = state["category_logits18"]
    obj_logits = state["objectness_logits"]
    cat_rows, cat_targets = [], []
    positive_logits, negative_logits = [], []
    conditional_correct = joint_correct = matched_count = 0
    positive_prob, negative_prob = [], []
    confusion = torch.zeros((18, 18), dtype=torch.long, device=cat_logits.device)
    for b, (qi, ki) in enumerate(pairs):
        matched_mask = torch.zeros(100, dtype=torch.bool, device=cat_logits.device)
        if qi.numel():
            matched_mask[qi] = True
            y = targets["gt_classes"][b][ki] - 2
            cat_rows.append(cat_logits[b, qi])
            cat_targets.append(y)
            conditional_pred = cat_logits[b, qi].argmax(-1)
            joint_pred = state["thing_logits19"][b, qi].argmax(-1)
            conditional_correct += int((conditional_pred == y).sum())
            joint_correct += int((joint_pred == y).sum())
            matched_count += int(qi.numel())
            confusion.index_put_((y, conditional_pred),
                                 torch.ones_like(y, dtype=torch.long), accumulate=True)
        positive_logits.append(obj_logits[b, matched_mask])
        negative_logits.append(obj_logits[b, ~matched_mask])
    cat_zero = cat_logits.sum() * 0.0
    category_ce = (F.cross_entropy(torch.cat(cat_rows).float(), torch.cat(cat_targets).long())
                   if cat_rows else cat_zero)
    pos = torch.cat(positive_logits) if positive_logits else obj_logits.new_empty((0,))
    neg = torch.cat(negative_logits) if negative_logits else obj_logits.new_empty((0,))
    pos_bce = F.binary_cross_entropy_with_logits(pos.float(), torch.ones_like(pos).float()) if pos.numel() else obj_logits.sum() * 0.0
    neg_bce = F.binary_cross_entropy_with_logits(neg.float(), torch.zeros_like(neg).float()) if neg.numel() else obj_logits.sum() * 0.0
    if pos.numel() and neg.numel(): objectness_bce = 0.5 * pos_bce + 0.5 * neg_bce
    elif pos.numel(): objectness_bce = pos_bce
    elif neg.numel(): objectness_bce = neg_bce
    else: objectness_bce = obj_logits.sum() * 0.0
    with torch.no_grad():
        probs = torch.sigmoid(obj_logits)
        for b, (qi, _) in enumerate(pairs):
            mask = torch.zeros(100, dtype=torch.bool, device=probs.device)
            if qi.numel(): mask[qi] = True
            positive_prob.append(probs[b, mask])
            negative_prob.append(probs[b, ~mask])
        pp = torch.cat(positive_prob) if positive_prob else probs.new_empty((0,))
        np_ = torch.cat(negative_prob) if negative_prob else probs.new_empty((0,))
    return category_ce, objectness_bce, pos_bce, neg_bce, {
        "conditional_classification_accuracy": conditional_correct / matched_count if matched_count else 0.0,
        "joint_classification_accuracy": joint_correct / matched_count if matched_count else 0.0,
        "matched_classification_accuracy": joint_correct / matched_count if matched_count else 0.0,
        "matched_objectness_recall_p50": float((pp >= .5).float().mean()) if pp.numel() else 0.0,
        "matched_objectness_prob_mean": float(pp.mean()) if pp.numel() else 0.0,
        "unmatched_objectness_p50_fraction": float((np_ >= .5).float().mean()) if np_.numel() else 0.0,
        "unmatched_objectness_prob_mean": float(np_.mean()) if np_.numel() else 0.0,
        "classification_confusion": confusion.detach().cpu().tolist(),
        "matched_count": matched_count,
    }


def _anchor_terms(state, targets, b, qi, ki):
    logits = state["anchor_mask_logits"][b].float()
    probs = torch.sigmoid(logits)
    valid = targets["anchor_valid"][b]
    kinds = targets["anchor_kind"][b]
    zero = logits.sum() * 0.0
    if not bool(valid.any()):
        return zero, zero, zero, zero
    thing_bce, thing_dice = [], []
    for query, gt in zip(qi.tolist(), ki.tolist()):
        y_all = targets["Y_anchor"][b, gt]
        if not bool(y_all.sum() > 0):
            continue
        y = y_all[valid]
        z = logits[valid, query]
        p = probs[valid, query]
        thing_bce.append(F.binary_cross_entropy_with_logits(z, y, reduction="mean"))
        thing_dice.append(1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1))
    t_bce = torch.stack(thing_bce).mean() if thing_bce else zero
    t_dice = torch.stack(thing_dice).mean() if thing_dice else zero
    stuff_bce, stuff_dice = [], []
    for channel, kind in ((100, WALL), (101, FLOOR)):
        y = (kinds[valid] == kind).float()
        z, p = logits[valid, channel], probs[valid, channel]
        stuff_bce.append(F.binary_cross_entropy_with_logits(z, y, reduction="mean"))
        stuff_dice.append(1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1))
    s_bce = torch.stack(stuff_bce).mean()
    s_dice = torch.stack(stuff_dice).mean()
    return 0.5 * (t_bce + s_bce), 0.5 * (t_dice + s_dice), t_bce, s_bce


def _semantic_nll(prediction, batch):
    sem = batch["semantic_label_all"][:, :2].long()
    valid = (sem >= 0) & (sem <= 19)
    scores = prediction["semantic_scores"]
    if bool(valid.any()):
        selected = scores.permute(0, 1, 3, 4, 2)[valid]
        target = sem[valid]
        return -(selected.gather(1, target[:, None]).squeeze(1) + 1e-6).log().mean()
    return scores.sum() * 0.0


def object_locus_v2_1_losses(prediction, batch, opt=None):
    del opt
    targets, pairs = final_hungarian(prediction, batch)
    state = prediction["states"][-1]
    sem, ins = batch["semantic_label_all"][:, :2].long(), batch["instance_label_all"][:, :2].long()
    regions = _flat_regions(prediction["region_mass"][:, :, :100], 100)
    zero = prediction["gaussians"].sum() * 0.0
    bce_rows, dice_rows = [], []
    anchor_bce_rows, anchor_dice_rows = [], []
    for b, (qi, ki) in enumerate(pairs):
        abce, adice, _, _ = _anchor_terms(state, targets, b, qi, ki)
        anchor_bce_rows.append(abce); anchor_dice_rows.append(adice)
        valid = (sem[b] >= 0) & (sem[b] <= 19) & ((sem[b] < 2) | (ins[b] > 0))
        if qi.numel() and bool(valid.any()):
            vf = valid.reshape(-1)
            p = regions[b, qi][:, vf].float().clamp(0, 1)
            y = targets["gt_pixel_masks"][b][ki].reshape(ki.numel(), -1)[:, vf].float()
            bce_rows.append(F.binary_cross_entropy(p, y, reduction="none").mean())
            inter = (p * y).sum(-1)
            dice_rows.append((1 - (2 * inter + 1) / (p.sum(-1) + y.sum(-1) + 1)).mean())
        else:
            bce_rows.append(zero); dice_rows.append(zero)
    mean = lambda xs: torch.stack(xs).mean() if xs else zero
    pbce, pdice = mean(bce_rows), mean(dice_rows)
    category_ce, objectness_bce, objectness_pos_bce, objectness_neg_bce, cls_metrics = _category_objectness_losses(state,pairs,targets)
    thing = 2 * category_ce + 2 * objectness_bce + 5 * pbce + 5 * pdice
    stuff, stuff_metrics = object_locus_stuff_loss(prediction, batch)
    semantic = _semantic_nll(prediction, batch)
    identity, identity_metrics = identity_loss(prediction, batch, max_points=64)
    abce, adice = mean(anchor_bce_rows), mean(anchor_dice_rows)
    anchor = abce + adice
    final = .1 * thing + .1 * stuff + .1 * semantic + .01 * identity + .1 * anchor

    aux_rows, metrics = [], {}
    for layer in (6, 8, 10):
        aux_state = next(s for s in prediction["states"] if int(s.get("layer", -1)) == layer)
        aux_targets = build_visible_anchor_targets(aux_state["mu"], batch)
        layer_anchor = []
        for b, (qi, ki) in enumerate(pairs):
            if not torch.equal(aux_targets["gt_instance_ids"][b], targets["gt_instance_ids"][b]):
                raise RuntimeError("V2 auxiliary scene-global GT instance ordering mismatch")
            ac, ad, _, _ = _anchor_terms(aux_state, aux_targets, b, qi, ki)
            layer_anchor.append(ac + ad)
        lce, lobj, lpos, lneg, _ = _category_objectness_losses(aux_state,pairs,aux_targets)
        lanchor = mean(layer_anchor)
        aux = .2 * lce + .2 * lobj + .1 * lanchor
        aux_rows.append(aux)
        metrics[f"aux_ce_layer{layer}"] = lce.detach()
        metrics[f"aux_objectness_layer{layer}"] = lobj.detach()
        metrics[f"aux_objectness_positive_layer{layer}"] = lpos.detach()
        metrics[f"aux_objectness_negative_layer{layer}"] = lneg.detach()
        metrics[f"aux_anchor_mask_layer{layer}"] = lanchor.detach()
    aux_mean = torch.stack(aux_rows).mean()
    understanding = final + .25 * aux_mean
    support = targets["Y_anchor"].sum(-1) > 0
    kinds = targets["anchor_kind"]
    metrics.update({
        "loss_thing_2d": thing.detach(), "category_ce": category_ce.detach(),
        "objectness_bce": objectness_bce.detach(),
        "objectness_positive_bce": objectness_pos_bce.detach(),
        "objectness_negative_bce": objectness_neg_bce.detach(),
        "thing_ce": category_ce.detach(),
        "pixel_bce": pbce.detach(), "pixel_dice": pdice.detach(),
        "loss_stuff_2d": stuff.detach(), "loss_semantic": semantic.detach(),
        "loss_identity": identity.detach(), "loss_anchor_group": anchor.detach(),
        "anchor_mask_bce": abce.detach(), "anchor_mask_dice": adice.detach(),
        "anchor_valid_count": int(targets["anchor_valid"].sum()),
        "anchor_thing_count": int((kinds == THING).sum()),
        "anchor_wall_count": int((kinds == WALL).sum()),
        "anchor_floor_count": int((kinds == FLOOR).sum()),
        "anchor_ignore_count": int((kinds == IGNORE).sum()),
        "gt_with_anchor_support": int(support.sum()),
        "gt_without_anchor_support": int((~support).sum()),
        **cls_metrics,
        "loss_final_understanding": final.detach(), "loss_aux_mean": aux_mean.detach(),
        "loss_understanding": understanding.detach(),
        "matched_gt_count": cls_metrics["matched_count"],
    })
    metrics.update(stuff_metrics); metrics.update(identity_metrics)
    prediction["final_pairs"] = [(q.detach(), k.detach()) for q, k in pairs]
    prediction["final_targets"] = targets
    return understanding, metrics
