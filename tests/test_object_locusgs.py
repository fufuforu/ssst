"""CPU-only unit tests for the object-aware LocusGS building blocks."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from tokengs.models.object_locusgs import (
    GSAttributeHead,
    ObjectRelationTokenUpdate,
    instance_keys,
    ramp_weight,
    semantic_supervision_mask,
    thing_instance_mask,
)


def make_opt(**overrides) -> SimpleNamespace:
    base = dict(
        enc_embed_dim=8,
        dec_patch_size=2,
        seed=0,
        object_semantic_classes=4,
        object_instance_dim=3,
        object_relation_neighbours=3,
        object_relation_temperature=0.2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_semantic_and_thing_masks() -> None:
    semantic = torch.tensor([[[0, 1, 2], [19, 255, 3]]])
    instance = torch.tensor([[[0, 0, 7], [0, 5, 0]]])
    supervision = semantic_supervision_mask(semantic)
    assert bool(supervision[0, 0, 0])  # class 0 is a real class
    assert not bool(supervision[0, 1, 1])  # 255 is the only ignore
    thing = thing_instance_mask(semantic, instance)
    assert bool(thing[0, 0, 2])  # class 2 with instance 7
    assert not bool(thing[0, 0, 0])  # stuff class with instance 0
    assert not bool(thing[0, 0, 1])  # stuff class (floor)
    assert not bool(thing[0, 1, 0])  # thing class with instance id 0
    assert not bool(thing[0, 1, 1])  # void with a nonzero instance id
    assert not bool(thing[0, 1, 2])  # thing class with instance id 0


def test_instance_key_is_scene_stable() -> None:
    semantic = torch.tensor([2, 2, 19])
    instance = torch.tensor([7, 7, 999])
    keys = instance_keys(semantic, instance)
    assert keys.tolist() == [3007, 3007, 20999]
    assert int(keys.max()) < 21000  # (19 + 1) * 1000 + 999


def test_ramp_weight() -> None:
    assert ramp_weight(0, 1000) == 0.0
    assert ramp_weight(500, 1000) == 0.5
    assert ramp_weight(1000, 1000) == 1.0
    assert ramp_weight(5000, 1000) == 1.0
    assert ramp_weight(0, 0) == 1.0


def test_attribute_head_order_and_initialisation() -> None:
    opt = make_opt()
    head = GSAttributeHead(opt)
    tokens = torch.randn(2, 5, opt.enc_embed_dim)
    semantic, embedding = head(tokens)
    assert semantic.shape == (2, 5 * opt.dec_patch_size**2, opt.object_semantic_classes)
    assert embedding.shape == (2, 5 * opt.dec_patch_size**2, opt.object_instance_dim)
    reshaped = head.semantic(tokens).reshape(2, 5, head.patches, opt.object_semantic_classes)
    assert torch.equal(semantic, reshaped.reshape(2, -1, opt.object_semantic_classes))
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(2, 5 * 4), atol=1e-6)
    # the 64 (here 4) Gaussians of one token do not share one attribute vector
    assert float(semantic[0].reshape(5, head.patches, -1).std(dim=1).mean()) > 1e-6
    twin = GSAttributeHead(opt)
    assert torch.equal(head.semantic.weight, twin.semantic.weight)
    assert torch.equal(head.instance.weight, twin.instance.weight)


def test_relation_update_is_exactly_zero_at_initialisation() -> None:
    opt = make_opt()
    module = ObjectRelationTokenUpdate(opt)
    tokens = torch.randn(1, 6, opt.enc_embed_dim, requires_grad=True)
    mu = torch.randn(1, 6, 3)
    updated = module(tokens, mu)
    assert torch.equal(updated, tokens)
    assert float(module.gate.detach()) == 0.0


def test_relation_update_gradients_and_neighbourhood() -> None:
    torch.manual_seed(0)
    opt = make_opt(object_relation_neighbours=4)
    module = ObjectRelationTokenUpdate(opt)
    with torch.no_grad():
        module.gate.fill_(0.1)
    tokens = torch.randn(1, 6, opt.enc_embed_dim, requires_grad=True)
    mu = torch.randn(1, 6, 3)
    indices = module.neighbour_indices(mu)
    assert indices.shape == (1, 6, 4)
    assert torch.equal(indices, module.neighbour_indices(mu))  # deterministic
    distances = torch.cdist(mu[0], mu[0])
    for token in range(6):
        picked = indices[0, token]
        assert len(set(picked.tolist())) == picked.numel()
        worst = distances[token, picked].max()
        others = torch.ones(6, dtype=torch.bool)
        others[picked] = False
        if others.any():
            assert float(worst) <= float(distances[token, others].min()) + 1e-6
    loss = module(tokens, mu).sum()
    loss.backward()
    assert tokens.grad is not None and float(tokens.grad.abs().sum()) > 0
    for name in ("norm", "key", "value", "out"):
        parameter = getattr(module, name)
        if hasattr(parameter, "weight"):
            assert parameter.weight.grad is not None
