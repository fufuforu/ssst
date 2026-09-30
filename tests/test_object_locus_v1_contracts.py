"""Targeted CPU contracts for the registered Object-Locus V1 implementation."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from tokengs.models.object_locus_v1_controller import (
    ObjectLocusV1Controller, _stable_neighborhoods, _stable_seed_indices,
    scene_normalization,
)
from tokengs.models.object_locus_v1 import inherit_gaussian_ownership
from tokengs.models.object_locus_v1_loss import (
    build_visible_anchor_targets, object_locus_v1_losses,
    pairwise_visible_anchor_cost,
)
from tokengs.models.instance_state_loss import _flat_regions
from scripts.export_object_locus_v1_official import (
    assemble_panoptic, write_official_pair,
)


def _inputs(seed=9):
    g = torch.Generator().manual_seed(seed)
    mu = torch.rand((1, 1024, 3), generator=g) * 0.5
    tokens = torch.randn((1, 1024, 32), generator=g)
    radii = torch.full((1, 1024), 0.04)
    origin, ell = scene_normalization(mu)
    return mu, tokens, radii, origin, ell


def _controller_inputs(seed=9):
    ctrl = ObjectLocusV1Controller(token_dim=32)
    mu, tokens, radii, origin, ell = _inputs(seed)
    a = ctrl.encode_token(tokens, mu, radii, ell)
    q, c, s, seeds, neighbors, weights = ctrl.initialize_states(a, tokens, mu, origin, ell)
    return ctrl, mu, tokens, radii, origin, ell, a, q, c, s, seeds, neighbors, weights


def _visible_batch():
    sem = torch.zeros((1, 2, 4, 4), dtype=torch.long)
    ins = torch.zeros_like(sem)
    # A thing exists in pixel GT but has no anchor support in these tests.
    sem[:, :, 3, 3] = 2
    ins[:, :, 3, 3] = 7
    sem[:, 1, 1, 3] = 1  # conflict at the last anchor's projected pixel
    cams = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1)
    intr = torch.tensor([1.0, 1.0, 1.0, 1.0]).reshape(1, 1, 4).repeat(1, 4, 1)
    depth = torch.full((1, 4, 1, 4, 4), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[0, 1, 0, 1, 0] = False  # anchor 6: one trusted view remains
    valid[0, :, 0, 1, 2] = False  # anchor 3: no trusted depth
    return {"semantic_label_all": sem, "instance_label_all": ins,
            "cam_view_all": cams, "intrinsics_all": intr,
            "depth_gt_scene_all": depth, "depth_gt_valid_all": valid}


def test_c1_state_evidence_ownership_and_class_shapes_dtype_and_gaussian_inheritance():
    ctrl, mu, tokens, radii, _origin, ell, a, q, c, s, seeds, _neighbors, _weights = _controller_inputs()
    state = ctrl.forward_registered_layer(tokens, mu, radii, ell, q, c, s, anchor_embedding=a)
    assert a.shape == (1, 1024, 256)
    assert state["q"].shape == (1, 102, 256)
    assert state["c"].shape == state["s"].shape == (1, 100, 3)
    assert state["evidence_attention"].shape == (1, 8, 102, 1024)
    assert state["anchor_assignment"].shape == (1, 1024, 103)
    assert state["thing_class_logits"].shape == (1, 100, 21)
    for value in (a, state["q"], state["c"], state["s"], state["evidence_attention"],
                  state["anchor_assignment"], state["thing_class_logits"]):
        assert value.dtype == torch.float32
    ownership = torch.rand((1, 1024, 103))
    owned = inherit_gaussian_ownership(ownership)
    assert owned.shape == (1, 65536, 103)
    assert torch.equal(owned.reshape(1, 1024, 64, 103), ownership[:, :, None].expand(-1, -1, 64, -1))


def test_c2_seed_ties_repeat_exactly_and_choose_100_distinct_anchors():
    mu = torch.zeros((1024, 3))
    h = torch.zeros((1024, 32))
    origin = torch.zeros((1, 3))
    ell = torch.tensor(0.05)
    a = _stable_seed_indices(mu, h, origin, ell)
    b = _stable_seed_indices(mu, h, origin, ell)
    assert torch.equal(a, b)
    assert a.tolist() == list(range(100))
    assert torch.unique(a).numel() == 100
    n1, _ = _stable_neighborhoods(mu, ell, a)
    n2, _ = _stable_neighborhoods(mu, ell, b)
    assert torch.equal(n1, n2)
    assert torch.equal(n1[:, 0], a)
    # Full scene-conditioned initialization is repeatable for the same controller seed.
    torch.manual_seed(31415)
    ctrl1 = ObjectLocusV1Controller(token_dim=32)
    torch.manual_seed(31415)
    ctrl2 = ObjectLocusV1Controller(token_dim=32)
    mu_b, tokens, radii, origin, ell_b = _inputs()
    a1 = ctrl1.encode_token(tokens, mu_b, radii, ell_b)
    a2 = ctrl2.encode_token(tokens, mu_b, radii, ell_b)
    init1 = ctrl1.initialize_states(a1, tokens, mu_b, origin, ell_b)
    init2 = ctrl2.initialize_states(a2, tokens, mu_b, origin, ell_b)
    for left, right in zip(init1[:4], init2[:4]):
        assert torch.equal(left, right)


def test_c3_initialization_uses_sixteen_neighbors_but_evidence_keeps_all_1024():
    ctrl, mu, tokens, radii, _origin, ell, a, q, c, s, _seeds, neighbors, weights = _controller_inputs()
    assert neighbors.shape == (1, 100, 16)
    assert weights.shape == (1, 100, 16)
    assert torch.allclose(weights.sum(-1), torch.ones((1, 100)), atol=1e-5)
    result = ctrl.forward_registered_layer(tokens, mu, radii, ell, q, c, s, anchor_embedding=a)
    assert result["evidence_attention"].shape[-1] == 1024
    assert result["anchor_assignment"].shape[1] == 1024


def test_c4_evidence_and_ownership_probabilities_are_separately_normalized():
    ctrl, mu, tokens, radii, _origin, ell, a, q, c, s, *_ = _controller_inputs()
    result = ctrl.forward_registered_layer(tokens, mu, radii, ell, q, c, s, anchor_embedding=a)
    assert torch.allclose(result["evidence_attention"].sum(-1), torch.ones((1, 8, 102)), atol=1e-5)
    assert torch.allclose(result["anchor_assignment"].sum(-1), torch.ones((1, 1024)), atol=1e-5)


def test_c5_evidence_and_ownership_parameters_are_independent():
    ctrl, mu, tokens, radii, _origin, ell, a, q, c, s, *_ = _controller_inputs()
    evidence_before = ctrl.read_evidence(a, mu, q, c, s)[0].detach().clone()
    with torch.no_grad():
        ctrl.W_own_e.weight.add_(0.1)
    evidence_after = ctrl.read_evidence(a, mu, q, c, s)[0].detach()
    assert torch.equal(evidence_before, evidence_after)
    result = ctrl.forward_registered_layer(tokens, mu, radii, ell, q, c, s, anchor_embedding=a)
    ownership_before = result["anchor_assignment"].detach().clone()
    with torch.no_grad():
        ctrl.W_Q.weight.add_(0.1)
    ownership_after = ctrl.ownership(a, result["q"], mu, result["c"], result["s"])[0]
    # The ownership formula consumes only its own projections and geometric state.
    assert torch.equal(ownership_after, ownership_before)
    assert ownership_before.shape == (1, 1024, 103)
    ev_parameters = {id(p) for name, p in ctrl.named_parameters() if name.startswith(("ln_ev_", "W_Q", "W_K", "W_V", "W_O"))}
    own_parameters = {id(p) for name, p in ctrl.named_parameters() if name.startswith(("ln_own_", "W_own_", "W_void"))}
    assert not (ev_parameters & own_parameters)


def test_c6_geometry_is_finite_bounded_and_capped_per_layer():
    ctrl, mu, tokens, radii, _origin, ell, a, q, c, s, *_ = _controller_inputs()
    result = ctrl.forward_registered_layer(tokens, mu, radii, ell, q, c, s, anchor_embedding=a)
    assert torch.isfinite(result["c"]).all() and torch.isfinite(result["s"]).all()
    assert torch.all(result["s"] >= 0.05 * ell[:, None, None])
    assert torch.all(result["s"] <= 2.0 * ell[:, None, None])
    assert torch.linalg.vector_norm(result["c"] - c, dim=-1).max() <= 0.25 * ell.max() + 1e-6


def test_c7_geometry_update_does_not_read_ownership_probabilities():
    ctrl, mu, tokens, radii, _origin, ell, a, q, c, s, *_ = _controller_inputs()
    R, Rbar, z = ctrl.read_evidence(a, mu, q, c, s)
    qnew = ctrl.decode_queries(q, z)
    before = ctrl.update_geometry(Rbar, mu, qnew[:, :100], c, s, ell)
    with torch.no_grad():
        ctrl.W_own_e.weight.add_(0.7)
        ctrl.W_own_u.bias.add_(0.4)
        ctrl.W_void.bias.add_(2.0)
    after = ctrl.update_geometry(Rbar, mu, qnew[:, :100], c, s, ell)
    assert all(torch.equal(x, y) for x, y in zip(before, after))
    assert R.shape[-1] == 1024


def test_c8_detached_scene_normalization_and_live_head_gradients():
    ctrl, mu, tokens, radii, origin, ell, a, q, c, s, *_ = _controller_inputs()
    assert not origin.requires_grad and not ell.requires_grad
    mu = mu.clone().requires_grad_(True)
    a = ctrl.encode_token(tokens, mu, radii, ell)
    result = ctrl.forward_registered_layer(tokens, mu, radii, ell, q, c, s, anchor_embedding=a)
    gen = torch.Generator().manual_seed(144)
    loss = ((result["c"] * torch.randn(result["c"].shape, generator=gen)).sum()
            + (result["s"] * torch.randn(result["s"].shape, generator=gen)).sum()
            + (result["evidence_attention"] * torch.randn(result["evidence_attention"].shape, generator=gen)).sum()
            + (result["anchor_assignment"] * torch.randn(result["anchor_assignment"].shape, generator=gen)).sum()
            + (result["thing_logits19"] * torch.randn(result["thing_logits19"].shape, generator=gen)).sum())
    params = (ctrl.W_c.weight, ctrl.W_s.weight, ctrl.W_Q.weight, ctrl.W_V.weight,
              ctrl.W_own_e.weight, ctrl.W_own_u.weight, ctrl.thing_classifier.weight)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    for grad in grads:
        assert grad is not None and torch.isfinite(grad).all() and float(grad.norm()) > 0


def test_c9_visible_depth_consensus_conflicts_invalid_and_unmatched_gt_retention():
    batch = _visible_batch()
    mu = torch.zeros((1, 1024, 3))
    mu[:] = torch.tensor([0.0, 0.0, 2.0])
    mu[0, 1] = torch.tensor([0.0, 0.0, -1.0])
    mu[0, 2] = torch.tensor([20.0, 0.0, 2.0])
    mu[0, 3] = torch.tensor([2.0, 0.0, 2.0])  # u=2, invalid depth pixel
    mu[0, 4] = torch.tensor([0.0, 0.0, 2.3])
    mu[0, 5] = torch.tensor([0.0, 0.0, 1.7])
    mu[0, 6] = torch.tensor([-2.0, 0.0, 2.0])  # u=0, one trusted view
    mu[0, 7] = torch.tensor([4.0, 0.0, 2.0])   # u=3, wall/floor conflict
    t = build_visible_anchor_targets(mu, batch)
    kinds = t["anchor_kind"][0]
    assert bool(t["anchor_valid"][0, 0]) and kinds[0].item() == 1
    for index in (1, 2, 3, 4, 5, 7):
        assert not bool(t["anchor_valid"][0, index]) and kinds[index].item() == -1
    assert bool(t["anchor_valid"][0, 6]) and kinds[6].item() == 1
    assert t["gt_instance_ids"][0].tolist() == [7]
    assert t["gt_classes"][0].tolist() == [2]
    assert int(t["Y_anchor"][0, 0].sum()) == 0
    assert not bool(t["anchor_valid"][0, 7])  # conflict is ignore, never void supervision


def _mini_prediction_and_batch():
    H = W = 2
    batch = {"semantic_label_all": torch.full((1, 4, H, W), 2, dtype=torch.long),
             "instance_label_all": torch.full((1, 4, H, W), 5, dtype=torch.long),
             "cam_view_all": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1),
             "intrinsics_all": torch.tensor([1., 1., 0., 0.]).reshape(1, 1, 4).repeat(1, 4, 1),
             "depth_gt_scene_all": torch.full((1, 4, 1, H, W), 2.0),
             "depth_gt_valid_all": torch.ones((1, 4, 1, H, W), dtype=torch.bool),
             "images_all": torch.rand((1, 4, 3, H, W))}
    region_logits = torch.randn((1, 2, 103, H, W), requires_grad=True)
    region = torch.softmax(region_logits, dim=2)
    ownership_logits = torch.randn((1, 1024, 103), requires_grad=True)
    ownership = torch.softmax(ownership_logits, dim=-1)
    logits = torch.randn((1, 100, 19), requires_grad=True)
    states = []
    for layer in (6, 8, 10, 12):
        states.append({"layer": layer, "mu": torch.tensor([0., 0., 2.]).reshape(1, 1, 3).expand(1, 1024, 3),
                       "q": torch.randn((1, 102, 256), requires_grad=True),
                       "c": torch.randn((1, 100, 3), requires_grad=True),
                       "s": torch.ones((1, 100, 3), requires_grad=True),
                       "thing_logits19": logits if layer == 12 else torch.randn((1, 100, 19), requires_grad=True),
                       "anchor_assignment": ownership if layer == 12 else torch.softmax(torch.randn((1, 1024, 103), requires_grad=True), -1)})
    prediction = {"gaussians": torch.randn((1, 65536, 14), requires_grad=True),
                  "states": states, "region_mass": region,
                  "semantic_scores": torch.softmax(torch.randn((1, 2, 20, H, W), requires_grad=True), 2),
                  "identity_render": torch.randn((1, 2, 16, H, W), requires_grad=True),
                  "alpha": torch.full((1, 2, 1, H, W), .8, requires_grad=True)}
    prediction["p_class"] = torch.softmax(logits, -1)
    return prediction, batch


def test_c10_one_final_hungarian_aux_reuses_pairs_flat_order_and_zero_support_cost(monkeypatch):
    import tokengs.models.object_locus_v1_loss as loss_mod
    original = loss_mod.linear_sum_assignment
    calls = []
    def counted(cost):
        calls.append(cost.shape)
        return original(cost)
    monkeypatch.setattr(loss_mod, "linear_sum_assignment", counted)
    prediction, batch = _mini_prediction_and_batch()
    loss, metrics = object_locus_v1_losses(prediction, batch)
    assert len(calls) == 1
    assert loss.requires_grad and torch.isfinite(loss)
    assert len(prediction["final_pairs"]) == 1
    assert metrics["gt_with_anchor_support"] == 1
    encoded = torch.arange(2 * 2 * 2 * 2).reshape(1, 2, 2, 2, 2)
    flattened = _flat_regions(encoded, 2)
    assert flattened[0, 0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    pa = torch.full((3, 4), .25)
    ya = torch.zeros((2, 4))
    bce, dice = pairwise_visible_anchor_cost(pa, ya, torch.tensor([False, False]))
    assert torch.equal(bce, torch.zeros_like(bce)) and torch.equal(dice, torch.zeros_like(dice))


def test_c11_registered_loss_weights_aux_average_warmup_and_empty_anchor_graph():
    from scripts.object_locus_v1_runtime import understanding_weight
    assert [understanding_weight(x) for x in (0, 200, 201, 600, 999, 1000, 5000)] == [0., 0., .00125, .5, .99875, 1., 1.]
    prediction, batch = _mini_prediction_and_batch()
    loss, metrics = object_locus_v1_losses(prediction, batch)
    assert torch.isfinite(loss) and loss.requires_grad
    expected = metrics["loss_final_understanding"] + metrics["loss_aux_weighted"]
    assert torch.equal(loss.detach(), expected)
    ownership = torch.softmax(torch.randn((1024, 103), requires_grad=True), -1)
    # Empty valid-anchor CE/Dice are graph-connected zeros.
    zero = ownership.sum() * 0
    assert zero.requires_grad and zero.item() == 0
    grad = torch.autograd.grad(zero, ownership)[0]
    assert torch.equal(grad, torch.zeros_like(grad))


def test_c12_official_export_void_zero_class_offset_ids_and_frame_sets(tmp_path):
    V, H, W = 4, 3, 3
    sem_scores = torch.zeros((1, V, 20, H, W))
    sem_scores[:, :, 2] = .8
    alpha = torch.ones((1, V, 1, H, W))
    alpha[:, :, :, 0, 0] = 0
    region = torch.zeros((1, V, 103, H, W))
    region[:, :, 0, 1, 1] = .9
    p = torch.zeros((100, 19)); p[:, 18] = 1
    p[0, 0] = .9; p[0, 18] = .1
    out = {"semantic_scores": sem_scores, "alpha": alpha, "region_mass": region,
           "p_class": p[None], "render": {"images_pred": torch.rand((1, V, 3, H, W))}}
    pred_sem, pred_ins, _ = assemble_panoptic(out)
    assert pred_sem.shape == pred_ins.shape == (V, H, W)
    assert pred_sem[0, 0, 0].item() == 20 and pred_ins[0, 0, 0].item() == 0
    batch = {"frame_ids": torch.tensor([[10, 11, 20, 21]]),
             "semantic_label_all": torch.zeros((1, V, H, W), dtype=torch.long),
             "instance_label_all": torch.zeros((1, V, H, W), dtype=torch.long),
             "images_all": torch.rand((1, V, 3, H, W))}
    batch["semantic_label_all"][:, :, 1, 1] = 2
    batch["instance_label_all"][:, :, 1, 1] = 1
    window = {"scene": "scene_test", "context": [10, 11], "novel": [20, 21]}
    allrow = write_official_pair(out, batch, window, tmp_path / "all", target_frames="all")
    novelrow = write_official_pair(out, batch, window, tmp_path / "novel", target_frames="novel")
    pair = tmp_path / "all/scene_test_context10_11"
    target_files = sorted(x.name for x in (pair / "target_seg_pred").glob("*.png"))
    context_files = sorted(x.name for x in (pair / "context_seg_pred").glob("*.png"))
    assert len(target_files) == 4 and len(context_files) == 2
    novel_pair = tmp_path / "novel/scene_test_context10_11"
    assert len(list((novel_pair / "target_seg_pred").glob("*.png"))) == 2
    rows = json.loads((pair / "target_seg_pred/pred.json").read_text())
    assert any(row["id"] == 1 and row["label_id"] == 3 for row in rows)
    assert allrow["target_frames"] == [10, 11, 20, 21]
    assert novelrow["target_frames"] == [20, 21]
    void = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(
        pair / "target_seg_pred/scene_test_pred10.png"))
    packed = int(void[0, 0, 0]) + 256 * int(void[0, 0, 1]) + 65536 * int(void[0, 0, 2])
    assert packed // 1000 == 0


def test_c13_optimizer_exhaustive_four_groups_and_registered_lrs():
    from scripts.object_locus_v1_runtime import build_optimizer, set_optimizer_lr
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.object_locus = nn.Module()
            self.object_locus.weight = nn.Parameter(torch.ones(2, 2))
            self.object_locus.bias = nn.Parameter(torch.ones(2))
            self.object_locus.stuff_seed = nn.Parameter(torch.ones(2, 2))
            self.reconstruction = nn.Module()
            self.reconstruction.weight = nn.Parameter(torch.ones(3, 3))
            self.reconstruction.bias = nn.Parameter(torch.ones(3))
    model = Tiny()
    optimizer, audit = build_optimizer(model)
    assert [x["name"] for x in audit["groups"]] == ["object_locus_decay", "object_locus_nodecay", "reconstruction_decay", "reconstruction_nodecay"]
    assert audit["all_trainable_once"] and audit["missing"] == audit["duplicates"] == []
    set_optimizer_lr(optimizer, 200)
    lrs200 = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert lrs200["object_locus_decay"] == 1e-4 and lrs200["reconstruction_decay"] == 1e-5
    set_optimizer_lr(optimizer, 5000)
    lrs5k = {g["name"]: g["lr"] for g in optimizer.param_groups}
    assert lrs5k["object_locus_decay"] == pytest.approx(2e-6)
    assert lrs5k["reconstruction_decay"] == pytest.approx(2e-7)
    assert next(g for g in optimizer.param_groups if g["name"] == "object_locus_nodecay")["weight_decay"] == 0
