"""Set-prediction GT targets and losses for Object-Locus V3-Set."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

NUM_THING = 100
NO_OBJECT = 18


def build_context_instance_targets(batch):
    sem = batch["semantic_label_all"][:, :2].long()
    ins = batch["instance_label_all"][:, :2].long()
    valid = (sem >= 0) & (sem <= 19) & ((sem < 2) | (ins > 0))
    ids_rows, classes_rows, masks_rows = [], [], []
    for b in range(sem.shape[0]):
        ids = torch.unique(ins[b][valid[b] & (sem[b] >= 2) & (ins[b] > 0)], sorted=True)
        row_ids, row_classes, row_masks = [], [], []
        for iid in ids:
            mask = valid[b] & (sem[b] >= 2) & (ins[b] == iid)
            labels = sem[b][mask]
            if labels.numel() == 0:
                continue
            counts = torch.bincount(labels, minlength=20)
            cls = torch.nonzero(counts == counts.max(), as_tuple=False)[0, 0]
            row_ids.append(iid)
            row_classes.append(cls)
            row_masks.append(mask)
        ids_rows.append(torch.stack(row_ids).long() if row_ids else ins.new_empty((0,)))
        classes_rows.append(torch.stack(row_classes).long() if row_classes else ins.new_empty((0,)))
        masks_rows.append(torch.stack(row_masks) if row_masks else
                          torch.zeros((0, 2, *sem.shape[-2:]), dtype=torch.bool, device=sem.device))
    return {"gt_instance_ids": ids_rows, "gt_classes": classes_rows,
            "gt_pixel_masks": masks_rows, "valid_pixels": valid}


@torch.no_grad()
def final_hungarian(prediction, batch, targets=None):
    targets = build_context_instance_targets(batch) if targets is None else targets
    logits = prediction["states"][-1]["thing_logits19"].float()
    probs = torch.softmax(logits, dim=-1)
    masks = prediction["region_mass"][:, :2, :100].float()
    pairs = []
    for b in range(logits.shape[0]):
        classes, gt_masks = targets["gt_classes"][b], targets["gt_pixel_masks"][b]
        k = int(classes.numel())
        if not k:
            empty = torch.empty(0, dtype=torch.long, device=logits.device)
            pairs.append((empty, empty.clone()))
            continue
        valid = targets["valid_pixels"][b].reshape(-1)
        flat_masks = masks[b].permute(1, 0, 2, 3).reshape(100, -1)[:, valid].clamp(1e-6, 1 - 1e-6)
        gt = gt_masks.reshape(k, -1)[:, valid].float()
        if flat_masks.shape[1] == 0:
            raise RuntimeError("V3-Set matching received no valid context pixels")
        logp, log1mp = flat_masks.log(), torch.log1p(-flat_masks)
        bce = -log1mp.mean(-1, keepdim=True) - (logp - log1mp) @ gt.T / flat_masks.shape[-1]
        dice = 1 - (2 * flat_masks @ gt.T + 1) / (flat_masks.sum(-1, keepdim=True) + gt.sum(-1)[None] + 1)
        class_cost = -probs[b, :, classes - 2]
        cost = 2 * class_cost + 5 * bce + 5 * dice
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        pairs.append((torch.as_tensor(rows, device=logits.device, dtype=torch.long),
                      torch.as_tensor(cols, device=logits.device, dtype=torch.long)))
    return targets, pairs


def _probability(p):
    if not torch.isfinite(p).all():
        raise FloatingPointError("nonfinite rendered membership probability")
    if p.numel() and (float(p.detach().min()) < -1e-5 or float(p.detach().max()) > 1 + 1e-5):
        raise FloatingPointError("rendered membership lies outside probability domain")
    return p.clamp(0.0, 1.0)


def v3_set_losses(prediction, batch, opt=None):
    del opt
    targets, pairs = final_hungarian(prediction, batch)
    sem = batch["semantic_label_all"][:, :2].long()
    logits = prediction["states"][-1]["thing_logits19"].float()
    device = logits.device
    class_targets = torch.full((logits.shape[0], NUM_THING), NO_OBJECT, dtype=torch.long, device=device)
    matched = 0
    for b, (qi, ki) in enumerate(pairs):
        if qi.numel():
            class_targets[b, qi] = targets["gt_classes"][b][ki] - 2
            matched += int(qi.numel())
    class_weight = logits.new_ones(19)
    class_weight[NO_OBJECT] = 0.1
    classification = F.cross_entropy(logits.transpose(1, 2), class_targets,
                                     weight=class_weight, reduction="mean")
    regions = prediction["region_mass"][:, :2].float()
    valid = targets["valid_pixels"]
    zero = regions.sum() * 0.0
    thing_bce, thing_dice = [], []
    for b, (qi, ki) in enumerate(pairs):
        if not qi.numel():
            continue
        vf = valid[b].reshape(-1)
        p = _probability(regions[b].permute(1, 0, 2, 3).reshape(102, -1)[qi][:, vf])
        y = targets["gt_pixel_masks"][b][ki].reshape(ki.numel(), -1)[:, vf].float()
        thing_bce.append(F.binary_cross_entropy(p, y, reduction="none").mean(-1))
        intersection = (p * y).sum(-1)
        thing_dice.append(1 - (2 * intersection + 1) / (p.sum(-1) + y.sum(-1) + 1))
    pixel_bce = torch.cat(thing_bce).mean() if thing_bce else zero
    pixel_dice = torch.cat(thing_dice).mean() if thing_dice else zero
    stuff_bces, stuff_dices = [], []
    for b in range(regions.shape[0]):
        for channel, cls in ((100, 0), (101, 1)):
            vf = valid[b]
            p = _probability(regions[b, :, channel][vf])
            y = (sem[b][vf] == cls).float()
            if p.numel():
                stuff_bces.append(F.binary_cross_entropy(p, y))
                inter = (p * y).sum()
                stuff_dices.append(1 - (2 * inter + 1) / (p.sum() + y.sum() + 1))
            else:
                stuff_bces.append(zero)
                stuff_dices.append(zero)
    stuff_bce = torch.stack(stuff_bces).mean() if stuff_bces else zero
    stuff_dice = torch.stack(stuff_dices).mean() if stuff_dices else zero
    understanding = 0.1 * (2 * classification + 5 * pixel_bce + 5 * pixel_dice
                           + 5 * stuff_bce + 5 * stuff_dice)
    with torch.no_grad():
        correct = 0
        confusion = torch.zeros((18, 19), dtype=torch.long, device=device)
        for b, (q, k) in enumerate(pairs):
            if q.numel():
                gt = targets["gt_classes"][b][k] - 2
                pr = logits[b, q].argmax(-1)
                correct += int((pr == gt).sum())
                confusion.index_put_((gt, pr), torch.ones_like(gt), accumulate=True)
    metrics = {
        "loss_classification": classification.detach(), "loss_thing_bce": pixel_bce.detach(),
        "loss_thing_dice": pixel_dice.detach(), "loss_stuff_bce": stuff_bce.detach(),
        "loss_stuff_dice": stuff_dice.detach(), "loss_stuff": (5 * stuff_bce + 5 * stuff_dice).detach(),
        "loss_understanding": understanding.detach(),
        "matched_classification_accuracy": correct / max(1, matched),
        "matched_gt_count": matched,
        "gt_count": int(sum(x.numel() for x in targets["gt_classes"])),
        "classification_confusion": confusion.cpu().tolist(),
    }
    prediction["final_pairs"] = [(q.detach(), k.detach()) for q, k in pairs]
    prediction["final_targets"] = targets
    return understanding, metrics


object_locus_v3_set_losses = v3_set_losses
