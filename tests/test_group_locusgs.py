"""CPU-only unit tests for the group G0/G1 building blocks."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from tokengs.models.group_locusgs import (
    GroupQueryHead,
    GroupToTokenFeedback,
    group_instance_loss,
)
from tokengs.models.ssst_loss import OUTER_SEGMENTATION_WEIGHT, class_aware_context_loss


def make_opt(**overrides) -> SimpleNamespace:
    base = dict(
        enc_embed_dim=16,
        dec_patch_size=2,
        seed=0,
        num_object_queries=100,
        semantic_class_count=20,
        group_num_heads=4,
        group_mlp_ratio=2.0,
        anchor_num_freqs=2,
        anchor_extent=0.3,
        query_spatial_pe_std=0.02,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_group_head_slot_distribution_and_outputs() -> None:
    torch.manual_seed(0)
    opt = make_opt()
    head = GroupQueryHead(opt)
    tokens = torch.randn(2, 7, opt.enc_embed_dim)
    anchors = torch.randn(2, 7, 3) * 0.2
    radii = torch.full((2, 7), 0.15)
    slot_logits, objectness, class_logits, features = head.forward_full(tokens, anchors, radii)
    assert slot_logits.shape == (2, 7, opt.num_object_queries + 1)
    assert objectness.shape == (2, opt.num_object_queries)
    assert class_logits.shape == (2, opt.num_object_queries, 20)
    assert features.shape == (2, opt.num_object_queries, opt.enc_embed_dim)
    probability = torch.softmax(slot_logits, dim=-1)
    assert torch.allclose(probability.sum(-1), torch.ones(2, 7), atol=1e-6)
    # the background slot is a learned, token-independent parameter
    assert slot_logits[..., -1].std() < 1e-6


def test_feedback_is_exactly_zero_at_init_and_bounded_afterwards() -> None:
    torch.manual_seed(0)
    opt = make_opt()
    feedback = GroupToTokenFeedback(opt)
    tokens = torch.randn(1, 5, opt.enc_embed_dim)
    slot_prob = torch.softmax(torch.randn(1, 5, opt.num_object_queries + 1), dim=-1)
    query_features = torch.randn(1, opt.num_object_queries, opt.enc_embed_dim)
    updated = feedback(tokens, slot_prob, query_features)
    assert torch.equal(updated, tokens)
    with torch.no_grad():
        feedback.gate.fill_(0.5)
    updated = feedback(tokens, slot_prob, query_features)
    delta = (updated - tokens).norm(dim=-1)
    assert float(delta.min()) > 0.0
    # LayerNorm + tanh(g) <= 1 keeps the write-back bounded
    assert float(delta.max()) < math.sqrt(opt.enc_embed_dim) * (
        float(feedback.proj.weight.norm()) + float(feedback.proj.bias.norm()) + 1.0
    )


def test_group_instance_loss_matches_the_audited_composition() -> None:
    torch.manual_seed(0)
    queries, views, height, width = 100, 2, 6, 6
    class_logits = torch.randn(1, queries, 21)
    mask_logits = torch.randn(1, queries, views, height, width)
    gt_classes = [torch.tensor([5, 12])]
    gt_masks = [(torch.rand(2, views, height, width) > 0.6).float()]
    ours = group_instance_loss(class_logits, mask_logits, gt_classes, gt_masks)
    audited = class_aware_context_loss(
        class_logits, mask_logits, gt_classes, gt_masks
    )
    assert torch.allclose(
        ours["loss"], audited["loss"] / OUTER_SEGMENTATION_WEIGHT, atol=1e-5
    )
    assert ours["matched"] == [2]
    assert ours["targets"] == [2]


def test_group_instance_loss_handles_an_empty_target_scene() -> None:
    torch.manual_seed(0)
    class_logits = torch.randn(1, 100, 21)
    mask_logits = torch.randn(1, 100, 2, 5, 5)
    loss = group_instance_loss(
        class_logits, mask_logits, [torch.empty(0, dtype=torch.long)],
        [torch.empty(0, 2, 5, 5)],
    )
    assert torch.isfinite(loss["loss"])
    assert loss["matched"] == [0]


import math  # noqa: E402  (kept next to its use for readability)
