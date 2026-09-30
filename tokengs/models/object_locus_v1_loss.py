"""Visible-anchor GT, one final Hungarian solve, and Object-Locus V1 losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tokengs.models.anchor_group_loss import (
    FLOOR, IGNORE, THING, WALL, pairwise_anchor_bce_cost, project_points,
    resolve_anchor_observations,
)
from tokengs.models.instance_state_loss import (
    CLASS_CE_WEIGHT, MASK_BCE_WEIGHT, MASK_DICE_WEIGHT, NO_OBJECT_INDEX,
    UNMATCHED_CLASS_WEIGHT, _check_labels, _flat_regions, _linspace_indices,
    _matching_cost, identity_loss, semantic_loss, stuff_loss,
)


def build_visible_anchor_targets(mu, batch):
    """Project all anchors into the two context views and gate by GT depth."""
    if mu.ndim != 3 or mu.shape[1:] != (1024, 3):
        raise ValueError(f"mu must be [B,1024,3], got {tuple(mu.shape)}")
    semantic = batch["semantic_label_all"][:, :2].long()
    instance = batch["instance_label_all"][:, :2].long()
    _check_labels(semantic, instance, where="object_locus visible anchor GT")
    required = ("depth_gt_scene_all", "depth_gt_valid_all")
    if any(key not in batch for key in required):
        raise RuntimeError("Object-Locus V1 requires GT depth fields from its dedicated provider")
    depth = batch["depth_gt_scene_all"][:, :2]
    depth_valid = batch["depth_gt_valid_all"][:, :2].bool()
    if depth.ndim != 5 or depth.shape[:3] != (mu.shape[0], 2, 1) or depth.shape[-2:] != semantic.shape[-2:]:
        raise RuntimeError(f"bad aligned depth shape {tuple(depth.shape)}")
    if depth_valid.shape != depth.shape:
        raise RuntimeError("depth valid mask shape differs from depth GT")

    classes_all, ids_all, masks_all = [], [], []
    kind_all, anchor_id_all, anchor_class_all = [], [], []
    B, T = mu.shape[:2]
    H, W = semantic.shape[-2:]
    for b in range(B):
        sem = semantic[b]
        ins = instance[b]
        thing = (sem >= 2) & (sem <= 19) & (ins > 0)
        ids = torch.unique(ins[thing], sorted=True)
        classes, masks = [], []
        for iid_t in ids:
            iid = int(iid_t)
            class_ids = torch.unique(sem[thing & (ins == iid)])
            if class_ids.numel() != 1:
                raise RuntimeError(f"scene-global instance {iid} has semantic classes {class_ids.tolist()}")
            classes.append(int(class_ids.item()))
            masks.append(torch.stack([thing[v] & (ins[v] == iid) for v in range(2)]))
        if len(classes) > 100:
            raise RuntimeError(f"{len(classes)} GT instances exceed the fixed 100 thing states")
        classes_t = torch.tensor(classes, device=mu.device, dtype=torch.long)
        ids_t = ids.to(device=mu.device, dtype=torch.long)
        masks_t = (torch.stack(masks) if masks else
                   torch.zeros((0, 2, H, W), device=mu.device, dtype=torch.bool))
        observations = [[] for _ in range(T)]
        for view in range(2):
            u, v, z = project_points(mu[b].detach(), batch["cam_view_all"][b, view],
                                     batch["intrinsics_all"][b, view])
            finite = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z)
            inside = finite & (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            anchor_ids = torch.nonzero(inside, as_tuple=False).flatten()
            if anchor_ids.numel() == 0:
                continue
            # Coordinates are nonnegative after the inside check; long() is floor.
            px = u[anchor_ids].long()
            py = v[anchor_ids].long()
            trusted_depth = depth_valid[b, view, 0, py, px] & (depth[b, view, 0, py, px] > 0)
            gt_depth = depth[b, view, 0, py, px]
            consistent = trusted_depth & ((z[anchor_ids] - gt_depth).abs() <= 0.10 * gt_depth)
            selected = torch.nonzero(consistent, as_tuple=False).flatten()
            for local in selected.tolist():
                anchor = int(anchor_ids[local])
                sem_id = int(semantic[b, view, py[local], px[local]])
                iid = int(instance[b, view, py[local], px[local]])
                if sem_id == 0:
                    obs = (WALL, 0, 0)
                elif sem_id == 1:
                    obs = (FLOOR, 1, 0)
                elif 2 <= sem_id <= 19 and iid > 0:
                    obs = (THING, sem_id, iid)
                else:
                    continue
                observations[anchor].append(obs)
        kinds = torch.full((T,), IGNORE, device=mu.device, dtype=torch.long)
        anchor_ids = torch.zeros((T,), device=mu.device, dtype=torch.long)
        anchor_classes = torch.zeros((T,), device=mu.device, dtype=torch.long)
        for anchor, obs in enumerate(observations):
            kind, iid, cls = resolve_anchor_observations(obs)
            kinds[anchor] = kind
            anchor_ids[anchor] = iid
            anchor_classes[anchor] = cls
        classes_all.append(classes_t)
        ids_all.append(ids_t)
        masks_all.append(masks_t)
        kind_all.append(kinds)
        anchor_id_all.append(anchor_ids)
        anchor_class_all.append(anchor_classes)

    max_gt = max((int(x.numel()) for x in ids_all), default=0)
    Y = torch.zeros((B, max_gt, T), device=mu.device, dtype=torch.float32)
    for b in range(B):
        for k, iid_t in enumerate(ids_all[b]):
            Y[b, k] = ((kind_all[b] == THING) & (anchor_id_all[b] == iid_t)).float()
    kinds = torch.stack(kind_all)
    valid = (kinds == THING) | (kinds == WALL) | (kinds == FLOOR)
    return {
        "gt_classes": classes_all,
        "gt_instance_ids": ids_all,
        "gt_pixel_masks": masks_all,
        "Y_anchor": Y,
        "anchor_valid": valid,
        "anchor_kind": kinds,
        "anchor_instance_id": torch.stack(anchor_id_all),
        "anchor_semantic_class": torch.stack(anchor_class_all),
    }


def final_hungarian(prediction, batch, targets=None, match_points=4096):
    """One unified Hungarian solve using final-layer pixel and visible-anchor terms."""
    final = prediction["states"][-1]
    targets = targets or build_visible_anchor_targets(final["mu"], batch)
    sem = batch["semantic_label_all"][:, :2].long()
    ins = batch["instance_label_all"][:, :2].long()
    regions = _flat_regions(prediction["region_mass"][:, :, :100], 100)
    logits = final["thing_logits19"]
    ownership = final["anchor_assignment"][:, :, :100].transpose(1, 2)
    pairs = []
    for b in range(sem.shape[0]):
        classes = targets["gt_classes"][b]
        masks = targets["gt_pixel_masks"][b]
        k = int(classes.numel())
        valid_pixels = (sem[b] >= 0) & (sem[b] <= 19) & ((sem[b] < 2) | (ins[b] > 0))
        if k == 0:
            empty = torch.empty(0, device=sem.device, dtype=torch.long)
            pairs.append((empty, empty.clone()))
            continue
        flat = torch.nonzero(valid_pixels.reshape(-1), as_tuple=False).flatten()
        if flat.numel() == 0:
            raise RuntimeError("valid context pixel set is empty; refusing to fabricate targets")
        flat = flat[_linspace_indices(int(flat.numel()), match_points).to(flat.device)]
        z = torch.logit(regions[b, :, flat].clamp(1e-6, 1.0 - 1e-6))
        y = masks.flatten(1)[:, flat].float()
        pixel_cost = _matching_cost(logits[b], z, y, classes)
        anchor_valid = targets["anchor_valid"][b]
        num_valid = int(anchor_valid.sum())
        if num_valid:
            pa = ownership[b, :, anchor_valid].float()
            ya = targets["Y_anchor"][b, :, anchor_valid].float()
            bce, dice = pairwise_visible_anchor_cost(pa, ya,
                                                      targets["Y_anchor"][b].sum(-1) > 0)
        else:
            bce = dice = torch.zeros_like(pixel_cost)
        cost = pixel_cost + 2.0 * bce + 2.0 * dice
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
        pairs.append((torch.as_tensor(rows, device=sem.device, dtype=torch.long),
                      torch.as_tensor(cols, device=sem.device, dtype=torch.long)))
    return targets, pairs


def pairwise_visible_anchor_cost(pa, ya, support):
    """Matching BCE/Dice on trusted anchors, zeroing unsupported GT columns."""
    if pa.ndim != 2 or ya.ndim != 2 or pa.shape[1] != ya.shape[1]:
        raise ValueError("pa/ya must be [Q,N]/[K,N]")
    if support.shape != (ya.shape[0],):
        raise ValueError("support must have one bool per GT")
    if pa.shape[1] == 0:
        return pa.new_zeros((pa.shape[0], ya.shape[0])), pa.new_zeros((pa.shape[0], ya.shape[0]))
    bce = pairwise_anchor_bce_cost(pa, ya)
    dice = 1.0 - (2.0 * (pa @ ya.T) + 1.0) / (
        pa.sum(-1, keepdim=True) + ya.sum(-1)[None, :] + 1.0
    )
    keep = support.to(device=pa.device, dtype=pa.dtype)[None, :]
    return bce * keep, dice * keep


def _targets_for_pairs(targets, batch_index, qi, ki, logits, ownership):
    device = logits.device
    target = torch.full((100,), NO_OBJECT_INDEX, device=device, dtype=torch.long)
    if qi.numel():
        target[qi] = targets["gt_classes"][batch_index][ki] - 2
    class_weight = torch.ones(19, device=device, dtype=torch.float32)
    class_weight[18] = UNMATCHED_CLASS_WEIGHT
    ce = F.cross_entropy(logits.float(), target, weight=class_weight, reduction="mean")

    kinds = targets["anchor_kind"][batch_index]
    iid = targets["anchor_instance_id"][batch_index]
    anchor_target = torch.full_like(kinds, -1)
    anchor_target[kinds == WALL] = 100
    anchor_target[kinds == FLOOR] = 101
    id_to_gt = {int(value): index for index, value in enumerate(targets["gt_instance_ids"][batch_index].tolist())}
    gt_to_query = {int(gt): int(query) for query, gt in zip(qi.tolist(), ki.tolist())}
    for instance_id, gt_index in id_to_gt.items():
        mask = (kinds == THING) & (iid == instance_id)
        if bool(mask.any()):
            if gt_index not in gt_to_query:
                raise RuntimeError("visible thing anchor belongs to an unmatched GT instance")
            anchor_target[mask] = gt_to_query[gt_index]
    valid_anchor = targets["anchor_valid"][batch_index] & (anchor_target >= 0)
    if bool(valid_anchor.any()):
        rows = torch.nonzero(valid_anchor, as_tuple=False).flatten()
        anchor_ce = -torch.log(ownership[rows, anchor_target[rows]].clamp_min(1e-6)).mean()
    else:
        anchor_ce = ownership.sum() * 0.0
    dice_terms = []
    for query, gt in zip(qi.tolist(), ki.tolist()):
        y = targets["Y_anchor"][batch_index, gt]
        if float(y.sum()) == 0.0:
            continue
        p = ownership[targets["anchor_valid"][batch_index], query]
        yy = y[targets["anchor_valid"][batch_index]]
        dice_terms.append(1.0 - (2.0 * (p * yy).sum() + 1.0) / (p.sum() + yy.sum() + 1.0))
    anchor_dice = torch.stack(dice_terms).mean() if dice_terms else ownership.sum() * 0.0
    return ce, anchor_ce, anchor_dice, target


def loss_with_pairs(prediction, batch, targets, pairs):
    device = prediction["gaussians"].device
    sem = batch["semantic_label_all"][:, :2].long()
    ins = batch["instance_label_all"][:, :2].long()
    regions = _flat_regions(prediction["region_mass"][:, :, :100], 100)
    logits = prediction["states"][-1]["thing_logits19"]
    ownership = prediction["states"][-1]["anchor_assignment"]
    zero = prediction["gaussians"].sum() * 0.0
    ce_rows, bce_rows, dice_rows, anchor_ce_rows, anchor_dice_rows = [], [], [], [], []
    class_correct, class_total, class_margin = [], [], []
    for b, (qi, ki) in enumerate(pairs):
        ce, ace, adice, class_target = _targets_for_pairs(
            targets, b, qi, ki, logits[b], ownership[b]
        )
        ce_rows.append(ce); anchor_ce_rows.append(ace); anchor_dice_rows.append(adice)
        valid = (sem[b] >= 0) & (sem[b] <= 19) & ((sem[b] < 2) | (ins[b] > 0))
        if qi.numel() and bool(valid.any()):
            valid_flat = valid.reshape(-1)
            p = regions[b, qi][:, valid_flat].clamp(1e-6, 1.0 - 1e-6)
            z = torch.logit(p)
            y = targets["gt_pixel_masks"][b][ki].reshape(ki.numel(), -1)[:, valid_flat].float()
            bce_rows.append(F.binary_cross_entropy_with_logits(z, y, reduction="none").mean())
            prob = torch.sigmoid(z)
            intersection = (prob * y).sum(-1)
            dice_rows.append((1.0 - (2.0 * intersection + 1.0) /
                              (prob.sum(-1) + y.sum(-1) + 1.0)).mean())
            pred_cls = logits[b, qi].argmax(-1)
            true_cls = class_target[qi]
            class_correct.append((pred_cls == true_cls).float().sum())
            class_total.append(torch.tensor(float(qi.numel()), device=device))
            sorted_prob = torch.softmax(logits[b, qi].float(), -1)
            right = sorted_prob.gather(1, true_cls[:, None]).squeeze(1)
            masked = sorted_prob.clone()
            masked.scatter_(1, true_cls[:, None], -1.0)
            class_margin.append((right - masked.max(-1).values).mean())
        else:
            bce_rows.append(zero); dice_rows.append(zero)
    ce = torch.stack(ce_rows).mean() if ce_rows else zero
    pixel_bce = torch.stack(bce_rows).mean() if bce_rows else zero
    pixel_dice = torch.stack(dice_rows).mean() if dice_rows else zero
    thing_2d = CLASS_CE_WEIGHT * ce + MASK_BCE_WEIGHT * pixel_bce + MASK_DICE_WEIGHT * pixel_dice
    stuff, stuff_metrics = stuff_loss(prediction, batch)
    semantic, semantic_metrics = semantic_loss(prediction, batch)
    identity, identity_metrics = identity_loss(prediction, batch, max_points=64)
    anchor_ce = torch.stack(anchor_ce_rows).mean() if anchor_ce_rows else zero
    anchor_dice = torch.stack(anchor_dice_rows).mean() if anchor_dice_rows else zero
    anchor_group = anchor_ce + anchor_dice
    final = 0.1 * thing_2d + 0.1 * stuff + 0.1 * semantic + 0.01 * identity + 0.1 * anchor_group
    class_acc = (torch.stack(class_correct).sum() / torch.stack(class_total).sum()
                 if class_correct else zero)
    class_margin_mean = torch.stack(class_margin).mean() if class_margin else zero
    kinds = targets["anchor_kind"]
    support = targets["Y_anchor"].sum(-1) > 0
    metrics = {
        "loss_thing_2d": thing_2d.detach(), "thing_ce": ce.detach(),
        "thing_ce_weighted": (CLASS_CE_WEIGHT * ce).detach(),
        "pixel_bce": pixel_bce.detach(), "pixel_bce_weighted": (MASK_BCE_WEIGHT * pixel_bce).detach(),
        "pixel_dice": pixel_dice.detach(), "pixel_dice_weighted": (MASK_DICE_WEIGHT * pixel_dice).detach(),
        "loss_stuff_2d": stuff.detach(), "loss_stuff_weighted": (0.1 * stuff).detach(),
        "loss_semantic": semantic.detach(), "loss_semantic_weighted": (0.1 * semantic).detach(),
        "loss_identity": identity.detach(), "loss_identity_weighted": (0.01 * identity).detach(),
        "loss_anchor_group": anchor_group.detach(), "anchor_ce": anchor_ce.detach(),
        "anchor_dice": anchor_dice.detach(), "loss_anchor_group_weighted": (0.1 * anchor_group).detach(),
        "anchor_valid_count": int(targets["anchor_valid"].sum()),
        "anchor_thing_count": int((kinds == THING).sum()),
        "anchor_wall_count": int((kinds == WALL).sum()),
        "anchor_floor_count": int((kinds == FLOOR).sum()),
        "anchor_ignore_count": int((kinds == IGNORE).sum()),
        "gt_with_anchor_support": int(support.sum()),
        "gt_without_anchor_support": int((~support).sum()),
        "matched_classification_accuracy": class_acc.detach(),
        "matched_classification_margin": class_margin_mean.detach(),
        "loss_understanding": final.detach(),
    }
    metrics.update(stuff_metrics); metrics.update(semantic_metrics); metrics.update(identity_metrics)
    return final, metrics


def aux_with_pairs(prediction, batch, layer_targets, pairs):
    losses, metrics = [], {}
    for layer_index, targets in zip((6, 8, 10), layer_targets):
        state = next(s for s in prediction["states"] if int(s.get("layer", -1)) == layer_index)
        ce_terms, ace_terms, adice_terms = [], [], []
        for b, (qi, ki) in enumerate(pairs):
            # GT identity ordering is fixed by scene-global instance id at every layer.
            if not torch.equal(targets["gt_instance_ids"][b], layer_targets[-1]["gt_instance_ids"][b]):
                raise RuntimeError("auxiliary GT identity ordering differs across layers")
            ce, ace, adice, _ = _targets_for_pairs(
                targets, b, qi, ki, state["thing_logits19"][b], state["anchor_assignment"][b]
            )
            ce_terms.append(ce); ace_terms.append(ace); adice_terms.append(adice)
        zero = state["q"].sum() * 0.0
        ce = torch.stack(ce_terms).mean() if ce_terms else zero
        ace = torch.stack(ace_terms).mean() if ace_terms else zero
        adice = torch.stack(adice_terms).mean() if adice_terms else zero
        aux = 0.2 * ce + 0.1 * (ace + adice)
        losses.append(aux)
        metrics[f"aux_ce_layer{layer_index}"] = ce.detach()
        metrics[f"aux_anchor_ce_layer{layer_index}"] = ace.detach()
        metrics[f"aux_anchor_dice_layer{layer_index}"] = adice.detach()
        metrics[f"loss_aux_layer{layer_index}"] = aux.detach()
    mean = torch.stack(losses).mean()
    return mean, metrics


def object_locus_v1_losses(prediction, batch, opt=None):
    del opt
    targets, pairs = final_hungarian(prediction, batch)
    final, metrics = loss_with_pairs(prediction, batch, targets, pairs)
    layer_targets = [build_visible_anchor_targets(
        next(s for s in prediction["states"] if int(s.get("layer", -1)) == layer)["mu"], batch
    ) for layer in (6, 8, 10)]
    aux, aux_metrics = aux_with_pairs(prediction, batch, layer_targets, pairs)
    total = final + 0.25 * aux
    metrics.update(aux_metrics)
    metrics["loss_final_understanding"] = final.detach()
    metrics["loss_aux_mean"] = aux.detach()
    metrics["loss_aux_weighted"] = (0.25 * aux).detach()
    metrics["loss_understanding"] = total
    prediction["final_pairs"] = [(q.detach(), k.detach()) for q, k in pairs]
    prediction["final_targets"] = targets
    return total, metrics
