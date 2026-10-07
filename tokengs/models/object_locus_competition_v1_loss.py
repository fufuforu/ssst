"""Final-context pixel competition loss for Object-Locus Panoptic V1."""
from __future__ import annotations

import torch

from tokengs.models.object_locus_v3_set_loss import _probability


EPSILON = 1e-6
TEMPERATURE = 1.0
COMPETITION_LAMBDA = 2.0
EXTERNAL_WEIGHT = 0.1


def competition_log_probabilities(region_mass: torch.Tensor,
                                  p_class: torch.Tensor) -> torch.Tensor:
    """Return FP32 log competition probabilities over all 102 final queries."""
    if region_mass.ndim != 5 or region_mass.shape[1:3] != (2, 102):
        raise ValueError(f"expected region_mass [B,2,102,H,W], got {tuple(region_mass.shape)}")
    if p_class.shape != (region_mass.shape[0], 100, 19):
        raise ValueError(f"expected p_class [B,100,19], got {tuple(p_class.shape)}")
    regions = _probability(region_mass[:, :2].float())
    classes = p_class.float()
    if not torch.isfinite(classes).all():
        raise FloatingPointError("nonfinite final thing class probabilities")
    if classes.numel() and (float(classes.detach().min()) < -1e-5 or
                            float(classes.detach().max()) > 1 + 1e-5):
        raise FloatingPointError("final class probability outside probability domain")
    regions = regions.clamp(0.0, 1.0)
    a = classes[..., :18].max(dim=-1).values
    log_mask = regions.clamp_min(EPSILON).log()
    thing = log_mask[:, :, :100] + a.clamp_min(EPSILON).log()[:, None, :, None, None]
    stuff = log_mask[:, :, 100:102]
    return torch.cat((thing, stuff), dim=2).div(TEMPERATURE).log_softmax(dim=2)


def context_competition_loss(prediction: dict, batch: dict) -> torch.Tensor:
    """Per-window equal-region-balanced final-context competition loss.

    The original loss stores the one Hungarian matching in ``final_pairs`` and
    its targets in ``final_targets``. This function only consumes those values;
    it never rematches or sends gradients through the assignment.
    """
    logp = competition_log_probabilities(prediction["region_mass"], prediction["p_class"])
    pairs = prediction.get("final_pairs")
    targets = prediction.get("final_targets")
    if pairs is None or targets is None:
        raise RuntimeError("competition loss requires the original final Hungarian match")
    sem = batch["semantic_label_all"][:, :2].long()
    batch_losses = []
    for b in range(logp.shape[0]):
        regions = []
        valid = targets["valid_pixels"][b].bool()
        for qids, tids in [pairs[b]]:
            for q, target_i in zip(qids.tolist(), tids.tolist()):
                # A Hungarian target is a thing instance only when its original
                # valid GT mask contains at least one pixel.
                mask = targets["gt_pixel_masks"][b][target_i].bool() & valid
                if mask.any():
                    regions.append(-logp[b, :, int(q)][mask].mean())
        for channel, semantic_id in ((100, 0), (101, 1)):
            mask = valid & (sem[b] == semantic_id)
            if mask.any():
                regions.append(-logp[b, :, channel][mask].mean())
        batch_losses.append(torch.stack(regions).mean() if regions else logp[b].sum() * 0.0)
    return torch.stack(batch_losses).mean()


def competition_loss_reference(region_mass: torch.Tensor, p_class: torch.Tensor,
                               target_channels: torch.Tensor, region_masks: torch.Tensor,
                               valid: torch.Tensor) -> torch.Tensor:
    """Simple independent reference used by CPU contracts.

    target_channels is [B,R] with -1 for skipped regions, region_masks is
    [B,R,2,H,W], and each nonnegative region contributes one mean.
    """
    lp = competition_log_probabilities(region_mass, p_class)
    per_batch = []
    for b in range(lp.shape[0]):
        rows = []
        for r, channel in enumerate(target_channels[b].tolist()):
            if channel < 0:
                continue
            mask = region_masks[b, r].bool() & valid[b].bool()
            if mask.any():
                rows.append(-lp[b, :, channel][mask].mean())
        per_batch.append(torch.stack(rows).mean() if rows else lp[b].sum() * 0)
    return torch.stack(per_batch).mean()
