"""Finite CPU contracts for the preregistered final-context competition loss."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tokengs.models.object_locus_competition_v1_loss import (
    competition_log_probabilities, competition_loss_reference,
    context_competition_loss,
)


def run_cpu_contracts() -> dict:
    torch.manual_seed(17)
    # A small but non-symmetric tensor makes channel and region reductions
    # independently checkable without renderer or model dependencies.
    mass = (torch.rand(2, 2, 102, 3, 4, dtype=torch.float32) * 0.8 + 0.1).requires_grad_()
    logits = torch.randn(2, 100, 19, dtype=torch.float32, requires_grad=True)
    p = logits.softmax(-1)
    lp = competition_log_probabilities(mass, p)
    assert lp.shape == (2, 2, 102, 3, 4)
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp.exp().sum(2), torch.ones((2, 2, 3, 4)), atol=1e-6, rtol=1e-6)

    channels = torch.tensor([[7, 100, 101], [2, -1, -1]])
    masks = torch.zeros(2, 3, 2, 3, 4, dtype=torch.bool)
    masks[0, 0, :, :2, :] = True       # thing spans both context images
    masks[0, 1, :, 0, :2] = True       # wall region
    masks[0, 2, 1, 2, 2:] = True       # floor region only in image 2
    masks[1, 0, 0, 1:, 1:] = True
    valid = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    valid[0, 1, 0, 0] = False
    value = competition_loss_reference(mass, p, channels, masks, valid)

    # Independent scalar reference: form the exact 102 scores, normalize them,
    # average pixels within each region across both views, then average regions
    # and finally windows (including the empty-window zero contribution).
    ref_windows = []
    for b in range(2):
        ref_regions = []
        for r, channel in enumerate(channels[b].tolist()):
            if channel < 0:
                continue
            score = mass[b].detach().clamp_min(1e-6).log()
            thing_factor = p[b].detach()[:, :18].max(-1).values.clamp_min(1e-6).log()
            score = torch.cat((score[:, :100] + thing_factor[None, :, None, None], score[:, 100:]), 1)
            lprob = score.log_softmax(1)
            selected = masks[b, r] & valid[b]
            ref_regions.append(-lprob[:, channel][selected].mean())
        ref_windows.append(torch.stack(ref_regions).mean() if ref_regions else lp[b].sum().detach() * 0)
    expected = torch.stack(ref_windows).mean()
    assert torch.allclose(value.detach(), expected, atol=1e-6, rtol=1e-6)

    # Joint query/target permutation: reorder all 100 thing channels in the
    # prediction and update the mapped GT query while stuff slots stay fixed.
    perm = torch.randperm(100)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(100)
    perm_mass = torch.cat((mass.detach()[:, :, perm], mass.detach()[:, :, 100:]), 2)
    perm_p = p.detach()[:, perm]
    perm_channels = channels.clone()
    for b in range(2):
        is_thing = perm_channels[b] < 100
        perm_channels[b, is_thing] = inv[perm_channels[b, is_thing]]
    perm_value = competition_loss_reference(perm_mass, perm_p, perm_channels, masks, valid)
    assert torch.allclose(value.detach(), perm_value, atol=1e-6, rtol=1e-6)

    # Integration contract for the actual matched-target path: only matched
    # thing GT and valid wall/floor pixels are supervised; an unmatched thing
    # and void pixel are ignored. One Hungarian map is shared across views.
    region = torch.rand(1, 2, 102, 2, 3) * .7 + .1
    cls = torch.randn(1, 100, 19).softmax(-1)
    valid_i = torch.ones(1, 2, 2, 3, dtype=torch.bool)
    valid_i[0, 1, 1, 2] = False
    gt0 = torch.zeros(2, 2, 3, dtype=torch.bool)
    gt0[:, 0, :2] = True
    gt1 = torch.zeros_like(gt0); gt1[:, 1, 1:] = True
    sem_i = torch.full((1, 2, 2, 3), 2, dtype=torch.long)
    sem_i[0, :, 0, 0] = 0
    sem_i[0, :, 1, 0] = 1
    sem_i[0, 1, 1, 2] = 20  # invalid GT pixel
    pred_i = {"region_mass": region, "p_class": cls,
              "final_pairs": [(torch.tensor([3]), torch.tensor([0]))],
              "final_targets": {"valid_pixels": valid_i,
                  "gt_pixel_masks": [torch.stack((gt0, gt1))]}}
    batch_i = {"semantic_label_all": sem_i}
    integrated = context_competition_loss(pred_i, batch_i)
    expected_i = competition_loss_reference(
        region, cls, torch.tensor([[3, 100, 101]]),
        torch.stack((gt0, valid_i[0] & (sem_i[0] == 0),
                     valid_i[0] & (sem_i[0] == 1)), dim=0)[None],
        valid_i)
    assert torch.allclose(integrated, expected_i, atol=1e-6, rtol=1e-6)

    # Increasing a competition query's mask score increases the selected
    # target's loss; both region and class scores receive finite gradients.
    fixed_masks = torch.ones(1, 1, 2, 1, 1, dtype=torch.bool)
    one_valid = torch.ones(1, 2, 1, 1, dtype=torch.bool)
    one_mass = torch.full((1, 2, 102, 1, 1), 0.02)
    one_mass[:, :, 4] = 0.6
    one_mass[:, :, 5] = 0.1
    one_logits = torch.zeros(1, 100, 19)
    one_p = one_logits.softmax(-1)
    tgt = torch.tensor([[4]])
    base = competition_loss_reference(one_mass, one_p, tgt,
                                      fixed_masks, one_valid)
    raised = one_mass.clone(); raised[:, :, 5] = 0.55
    high = competition_loss_reference(raised, one_p, tgt,
                                      fixed_masks, one_valid)
    assert high > base
    grad_mass = one_mass.clone().requires_grad_()
    grad_logits = one_logits.clone().requires_grad_()
    loss = competition_loss_reference(grad_mass, grad_logits.softmax(-1), tgt,
                                      fixed_masks, one_valid)
    gm, gc = torch.autograd.grad(loss, (grad_mass, grad_logits))
    assert torch.isfinite(gm).all() and torch.isfinite(gc).all()
    assert gm[:, :, :100].abs().sum() > 0 and gc.abs().sum() > 0

    # All-empty window remains a graph-connected finite zero.
    empty = competition_loss_reference(mass.detach(), p.detach(),
        torch.full((2, 1), -1), torch.zeros(2, 1, 2, 3, 4, dtype=torch.bool), valid)
    assert float(empty) == 0.0 and torch.isfinite(empty)
    return {
        "status": "PASS",
        "device": "cpu",
        "temperature": 1.0,
        "epsilon": 1e-6,
        "channels_normalize_to_102": True,
        "independent_reference_atol_rtol": 1e-6,
        "query_permutation_invariant": True,
        "cross_view_region_mean_and_equal_region_weight": True,
        "matched_thing_and_valid_stuff_only_unmatched_and_void_ignored": True,
        "competition_score_increases_target_loss": True,
        "thing_mask_and_class_gradients_finite": True,
        "stuff_and_empty_region_contract": True,
    }


def main():
    result = run_cpu_contracts()
    path = Path("/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/cpu_contracts.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
