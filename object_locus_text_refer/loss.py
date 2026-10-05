from __future__ import annotations

import torch
from torch.nn import functional as F


def refer_loss(scores, soft_mask, gt_mask, valid_mask, slot_target=None, visible=True, matched=True):
    """Fixed CE + probability BCE + Dice; visible unmatched targets skip only CE."""
    p = soft_mask[valid_mask].float().clamp(0, 1)
    y = gt_mask[valid_mask].float()
    zero = soft_mask.sum() * 0.0
    pixel_bce = F.binary_cross_entropy(p, y) if p.numel() else zero
    pixel_dice = 1 - (2 * (p*y).sum() + 1) / (p.sum() + y.sum() + 1) if p.numel() else zero
    if slot_target is not None and (not visible or matched):
        slot_ce = F.cross_entropy(scores, torch.as_tensor([slot_target], device=scores.device))
    else:
        slot_ce = zero
    loss = slot_ce + 5.0 * pixel_bce + 5.0 * pixel_dice
    return loss, {"slot_ce": slot_ce.detach(), "mask_bce": pixel_bce.detach(), "mask_dice": pixel_dice.detach()}


def resolve_slot_target(object_id, gt_instance_ids, matched_slots, visible):
    """Join by scene object ID; unmatched visible is None, truly invisible is null slot."""
    if not visible: return 100
    id_to_index = {int(value): i for i, value in enumerate(gt_instance_ids)}
    if int(object_id) not in id_to_index: return None  # missing annotation, never a null negative
    gt_index = id_to_index[int(object_id)]
    return matched_slots.get(gt_index)
