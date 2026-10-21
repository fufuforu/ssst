"""Detached C-feature readouts: quality ranking or object eligibility, separately."""
import torch
from torch import nn
import torch.nn.functional as F


class ScalarReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if features.shape[-2:] != (100, 256) or not torch.isfinite(features).all():
            raise ValueError('expected finite100x256 frozen classification features')
        return self.net(features.detach().float()).squeeze(-1)


def objectness_logits(original_logits, residual):
    return torch.logsumexp(original_logits.detach().float()[..., :18], -1) - original_logits.detach().float()[..., 18] + residual


def candidate_probabilities(original_logits, residual):
    # Shifting all18 thing logits preserves their conditional class ratios.
    shifted = torch.cat((original_logits.detach().float()[..., :18] + residual[..., None],
                         original_logits.detach().float()[..., 18:]), -1)
    return shifted.softmax(-1)


def candidate_eligibility(original_logits, residual):
    probabilities = candidate_probabilities(original_logits, residual)
    return (probabilities.argmax(-1) != 18) & (probabilities[..., :18].max(-1).values >= .05)


def diagnostic_loss(head, batch, arm):
    raw = head(batch['features'])
    if arm == 'quality':
        weights = batch['quality_valid'].float()
        per_query = F.smooth_l1_loss(raw.sigmoid(), batch['quality_target'].float(), reduction='none', beta=1.)
    elif arm == 'objectness':
        target = batch['object_target'].float()
        weights = torch.where(target > .5, torch.ones_like(target), torch.full_like(target, .1))
        per_query = F.binary_cross_entropy_with_logits(objectness_logits(batch['class_logits'], raw), target, reduction='none')
    else:
        raise ValueError('unknown separate readout arm')
    # Equal window weighting preserves globalbatch8 with1x8 or2x4 microbatches.
    return ((per_query * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()
