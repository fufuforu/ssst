"""Targeted gradient contracts for Object-Locus V1.1 probability/logit losses."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tokengs.models.anchor_group_loss import IGNORE, THING
from tokengs.models.object_locus_v1_loss import (
    _physical_probability, _probability_mask_losses, _targets_for_pairs,
    aux_with_pairs, object_locus_stuff_loss,
)


def _one_supported_gt():
    kinds = torch.full((1024,), IGNORE, dtype=torch.long)
    kinds[0] = THING
    ids = torch.zeros((1024,), dtype=torch.long)
    ids[0] = 17
    valid = kinds != IGNORE
    y_anchor = torch.zeros((1, 1, 1024))
    y_anchor[0, 0, 0] = 1
    return {
        "gt_classes": [torch.tensor([2])],
        "gt_instance_ids": [torch.tensor([17])],
        "anchor_kind": kinds[None],
        "anchor_instance_id": ids[None],
        "anchor_valid": valid[None],
        "Y_anchor": y_anchor,
    }


def _extreme_logits():
    logits = torch.zeros((1024, 103), dtype=torch.float32)
    logits[0, 4] = -30.0  # matched query target; 30 below the highest logit.
    logits.requires_grad_()
    return logits


def test_anchor_ce_uses_raw_logits_and_keeps_extreme_target_gradient_for_final_and_aux():
    targets = _one_supported_gt()
    qi, ki = torch.tensor([4]), torch.tensor([0])
    class_logits = torch.zeros((100, 19), requires_grad=True)

    # Final anchor CE is the direct cross entropy over raw 103-channel logits.
    final_logits = _extreme_logits()
    final_probs = final_logits.softmax(-1)
    _ce, final_ce, _dice, _ = _targets_for_pairs(
        targets, 0, qi, ki, class_logits, final_logits, final_probs
    )
    grad = torch.autograd.grad(final_ce, final_logits)[0]
    expected = F.cross_entropy(final_logits[0:1], torch.tensor([4]))
    assert torch.equal(final_ce, expected)
    assert torch.isfinite(final_ce)
    assert torch.isfinite(grad).all()
    assert float(grad[0, 4]) < 0.0
    assert float(grad[0, 4]) != 0.0

    # Auxiliary supervision uses the same raw-logit CE helper for L6/L8/L10.
    aux_logits = _extreme_logits()
    aux_states = []
    for layer in (6, 8, 10):
        layer_logits = aux_logits if layer == 6 else torch.zeros((1024, 103), requires_grad=True)
        aux_states.append({
            "layer": layer,
            "q": torch.zeros((1, 102, 256)),
            "thing_logits19": class_logits[None],
            "ownership_logits": layer_logits[None],
            "anchor_assignment": layer_logits.softmax(-1)[None],
        })
    prediction = {"states": aux_states}
    aux, metrics = aux_with_pairs(prediction, {}, [targets, targets, targets], [(qi, ki)])
    aux_grad = torch.autograd.grad(aux, aux_logits)[0]
    assert torch.isfinite(aux)
    assert float(aux_grad[0, 4]) < 0.0
    assert float(aux_grad[0, 4]) != 0.0


def test_probability_bce_has_finite_nonzero_interior_extreme_gradients_and_endpoint_values():
    p = torch.tensor([1e-8, 1e-7, 0.5, 1.0 - 1e-7], dtype=torch.float32,
                     requires_grad=True)
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    bce, dice, used = _probability_mask_losses(p, y, name="test pixel mask")
    (bce + dice).backward()
    assert torch.equal(used, p.detach())
    assert torch.isfinite(bce) and torch.isfinite(dice)
    assert torch.isfinite(p.grad).all()
    assert p.grad[0] < 0 and p.grad[0] != 0
    assert p.grad[3] > 0 and p.grad[3] != 0

    endpoint = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)
    endpoint_y = torch.tensor([1.0, 0.0])
    endpoint_bce, endpoint_dice, _ = _probability_mask_losses(
        endpoint, endpoint_y, name="endpoint pixel mask"
    )
    endpoint_grad = torch.autograd.grad(endpoint_bce + endpoint_dice, endpoint)[0]
    assert torch.isfinite(endpoint_bce) and torch.isfinite(endpoint_dice)
    assert torch.isfinite(endpoint_grad).all()


def test_mask_dice_uses_physical_probability_without_epsilon_cutoff():
    raw = torch.tensor([[1e-8, 1e-7, 0.5, 1.0 - 1e-7]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    bce, dice, p = _probability_mask_losses(raw, target, name="dice path")
    expected_dice = 1.0 - (2 * (p * target).sum(-1) + 1) / (
        p.sum(-1) + target.sum(-1) + 1
    )
    assert torch.equal(dice, expected_dice.mean())
    assert torch.equal(p, raw.clamp(0.0, 1.0))
    assert torch.isfinite(bce + dice)


def test_object_locus_stuff_loss_uses_probability_domain_bce_and_raw_probability_dice():
    region = torch.zeros((1, 2, 103, 1, 2), dtype=torch.float32)
    region[:, :, 100] = torch.tensor([[[[1e-8, 0.5]], [[1e-7, 1.0 - 1e-7]]]])
    region[:, :, 101] = torch.tensor([[[[0.5, 1e-8]], [[1.0 - 1e-7, 1e-7]]]])
    region.requires_grad_()
    prediction = {"region_mass": region, "gaussians": torch.ones((1, 1, 1), requires_grad=True)}
    sem = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.long)
    ins = torch.zeros_like(sem)
    batch = {"semantic_label_all": sem, "instance_label_all": ins}
    loss, metrics = object_locus_stuff_loss(prediction, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(region.grad).all()
    assert metrics["stuff_bce"] >= 0.0 and metrics["stuff_dice"] >= 0.0
    # Wall p=1e-8 against positive target has a live negative probability gradient.
    assert region.grad[0, 0, 100, 0, 0] < 0.0
    # Floor p=1-1e-7 against negative target has a live positive gradient.
    assert region.grad[0, 1, 101, 0, 0] > 0.0


def test_probability_guard_rejects_nonfinite_and_material_range_errors():
    with pytest.raises(FloatingPointError, match="nonfinite"):
        _physical_probability(torch.tensor([float("nan")]), name="test")
    with pytest.raises(FloatingPointError, match="outside probability range"):
        _physical_probability(torch.tensor([1.0001]), name="test")
    # Only float-boundary noise is clamped to the physical interval.
    assert torch.equal(_physical_probability(torch.tensor([-1e-6, 1 + 1e-6]), name="test"),
                       torch.tensor([0.0, 1.0]))
