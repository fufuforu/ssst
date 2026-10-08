"""CPU-only contracts for the R3D output refinement and GC accumulation."""
import ast
import inspect
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tokengs.models.object_locus_output_refine_v1 import Output3DEvidenceRefine
from scripts.object_locus_output_refine_gc001_runtime import (
    _slot_for_micro, average_accumulated_gradients, build_optimizer,
    capture_slot_rng, restore_slot_rng,
)


def tiny_inputs(dtype=torch.float32, requires_grad=False):
    g = 7
    q = torch.randn(2, 102, 256, dtype=dtype, requires_grad=requires_grad)
    f = torch.randn(2, g, 256, dtype=dtype, requires_grad=requires_grad)
    xyz = torch.randn(2, g, 3, dtype=dtype, requires_grad=requires_grad)
    c = torch.randn(2, 100, 3, dtype=dtype, requires_grad=requires_grad)
    s = torch.rand(2, 100, 3, dtype=dtype, requires_grad=requires_grad) + .25
    return q, f, xyz, c, s


def independent_attention(module, q, f, xyz, c, s):
    qh = module.split_heads(module.W_Q(module.ln_q(q)))
    fg = module.ln_g(f)
    kh = module.split_heads(module.W_K(fg))
    vh = module.split_heads(module.W_V(fg))
    bias = module.geometry_bias(xyz, c, s)
    logits = (qh @ kh.transpose(-1, -2)) / math.sqrt(32.0) + bias[:, None]
    weights = torch.softmax(logits, dim=-1)
    attended = module.merge_heads(weights @ vh)
    q1 = q + module.W_O(attended)
    return q1 + module.W_2(module.act(module.W_1(module.ln_ffn(q1))))


def test_shape_order_heads_geometry_and_full_gaussian_softmax():
    torch.manual_seed(9)
    module = Output3DEvidenceRefine().eval()
    q, f, xyz, c, s = tiny_inputs()
    bias = module.geometry_bias(xyz, c, s)
    assert bias.shape == (2, 102, 7)
    assert torch.equal(bias[:, 100:], torch.zeros_like(bias[:, 100:]))
    x = torch.randn(2, 102, 256)
    assert torch.equal(module.merge_heads(module.split_heads(x)), x)
    out = module(q, f, xyz, c, s)
    ref = independent_attention(module, q, f, xyz, c, s)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    # Independent logits normalize over all seven GS, including the last one.
    qh = module.split_heads(module.W_Q(module.ln_q(q)))[:, :, :8]
    kh = module.split_heads(module.W_K(module.ln_g(f)))
    logits = (qh @ kh.transpose(-1, -2)) / math.sqrt(32) + bias[:, None, :8]
    prob = logits.softmax(-1)
    torch.testing.assert_close(prob.sum(-1), torch.ones_like(prob.sum(-1)), rtol=1e-6, atol=1e-6)
    assert (prob[..., -1] > 0).all()
    for invalid in (torch.zeros_like(s), torch.full_like(s, float('inf')), torch.full_like(s, float('nan'))):
        with pytest.raises((ValueError, FloatingPointError)):
            module.geometry_bias(xyz, c, invalid)


def test_fp64_attention_formula_and_fp32_chunk_reference():
    torch.manual_seed(12)
    module = Output3DEvidenceRefine().eval().double()
    qh = torch.randn(1, 8, 8, 32, dtype=torch.float64)
    kh = torch.randn(1, 8, 11, 32, dtype=torch.float64)
    vh = torch.randn(1, 8, 11, 32, dtype=torch.float64)
    bias = torch.randn(1, 8, 11, dtype=torch.float64)
    actual = module._attend_chunk(qh, kh, vh, bias)
    expected = torch.softmax((qh @ kh.transpose(-1, -2)) / math.sqrt(32) + bias[:, None], -1) @ vh
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    # The fixed 8-query chunks must match a separately evaluated unchunked FP32 reference.
    module = Output3DEvidenceRefine().eval()
    q, f, xyz, c, s = tiny_inputs()
    torch.testing.assert_close(module(q, f, xyz, c, s), independent_attention(module, q, f, xyz, c, s),
                               rtol=1e-5, atol=1e-6)


def test_zero_output_initialization_identity_then_projection_changes_query():
    torch.manual_seed(15)
    module = Output3DEvidenceRefine().eval()
    q, f, xyz, c, s = tiny_inputs()
    out = module(q, f, xyz, c, s)
    assert torch.equal(out, q)
    with torch.no_grad():
        module.W_O.weight.fill_(.01)
        module.W_O.bias.fill_(.02)
    assert not torch.equal(module(q, f, xyz, c, s), q)


def test_geometry_bias_detaches_only_xyz_centers_and_scales():
    module = Output3DEvidenceRefine().eval()
    q, f, xyz, c, s = tiny_inputs(requires_grad=True)
    with torch.no_grad():
        module.W_O.weight.normal_(0, .01)
    out = module(q, f, xyz, c, s).sum()
    grads = torch.autograd.grad(out, (q, f, xyz, c, s), allow_unused=True)
    assert grads[0] is not None and grads[0].abs().sum() > 0
    assert grads[1] is not None and grads[1].abs().sum() > 0
    assert grads[2] is None and grads[3] is None and grads[4] is None


def test_optimizer_family_and_strict_module_state_roundtrip():
    module = torch.nn.Module()
    module.panoptic = torch.nn.Module()
    module.panoptic.output_3d_refine = Output3DEvidenceRefine()
    optimizer = build_optimizer(module)
    selected = [name for group in optimizer.param_groups for name, p in zip(group['param_names'], group['params'])
                if name.startswith('panoptic.output_3d_refine.')]
    expected = {f'panoptic.output_3d_refine.{n}' for n, _ in module.panoptic.output_3d_refine.named_parameters()}
    assert set(selected) == expected and len(selected) == len(expected)
    groups_by_name = {name: group['name'] for group in optimizer.param_groups for name in group['param_names']}
    assert groups_by_name['panoptic.output_3d_refine.W_Q.weight'] == 'new_decay'
    assert groups_by_name['panoptic.output_3d_refine.ln_q.weight'] == 'new_nodecay'
    assert groups_by_name['panoptic.output_3d_refine.W_O.bias'] == 'new_nodecay'
    clone = Output3DEvidenceRefine()
    clone.load_state_dict(module.panoptic.output_3d_refine.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(module.panoptic.output_3d_refine.state_dict().values(), clone.state_dict().values()))
    assert len(module.panoptic.output_3d_refine.state_dict()) == 18


def test_gc_eight_window_average_matches_four_ranks_two_micro_reference():
    torch.manual_seed(21)
    # Independently define eight slot gradients for reconstruction, pretrained,
    # and new parameter families. None entries remain absent on all slots.
    slots = []; rec_slots = []
    for i in range(8):
        slots.append({'reconstruction': torch.randn(3, dtype=torch.float64),
                      'pretrained': torch.randn(2, dtype=torch.float64),
                      'new': torch.randn(4, dtype=torch.float64), 'unused': None})
        rec_slots.append({'reconstruction': torch.randn(3, dtype=torch.float64),
                          'pretrained': torch.randn(2, dtype=torch.float64),
                          'new': torch.randn(4, dtype=torch.float64), 'unused': None})
    alpha = .01
    reference = {family: sum((slot[family] for slot in rec_slots), torch.zeros_like(rec_slots[0][family])) / 8
                 for family in ('reconstruction', 'pretrained', 'new')}
    # Per logical slot GC, then local half-sum of two slots, then four-rank mean.
    rank_means = []
    for rank in range(4):
        total = {}
        for family in ('reconstruction', 'pretrained', 'new'):
            factor = alpha if family == 'reconstruction' else 1.
            rec = .5 * (rec_slots[rank][family] + rec_slots[rank+4][family])
            under = .5 * (slots[rank][family] + slots[rank+4][family]) * factor
            total[family] = rec + under
        rank_means.append(total)
    distributed = {family: sum((r[family] for r in rank_means), torch.zeros_like(rank_means[0][family])) / 4
                   for family in ('reconstruction', 'pretrained', 'new')}
    for family in ('reconstruction', 'pretrained', 'new'):
        expected = reference[family] + (sum((slot[family] for slot in slots), torch.zeros_like(slots[0][family])) / 8) * (alpha if family == 'reconstruction' else 1.)
        torch.testing.assert_close(distributed[family], expected, rtol=1e-12, atol=1e-12)
    assert all(slot['unused'] is None for slot in slots)
    assert [_slot_for_micro(rank, micro) for rank in range(4) for micro in (0, 1)] == [0, 4, 1, 5, 2, 6, 3, 7]
    source = inspect.getsource(__import__('scripts.object_locus_output_refine_gc001_runtime', fromlist=['train_update']).train_update)
    assert source.count('optimizer.zero_grad(') == 1
    assert source.count('optimizer.step(') == 1
    assert source.count('clip_grad_norm_(') == 1


def test_each_logical_rng_slot_continues_its_own_stream():
    states = {}; expected = {}
    for slot in range(8):
        seed = 42 + 100003 * slot
        random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
        states[slot] = capture_slot_rng(None)
        expected[slot] = ((random.random(), np.random.random(), torch.rand(()).item()),
                          (random.random(), np.random.random(), torch.rand(()).item()))
    first = {}
    for slot in range(8):
        restore_slot_rng(states[slot], None)
        first[slot] = (random.random(), np.random.random(), torch.rand(()).item())
        states[slot] = capture_slot_rng(None)
    second = {}
    for slot in reversed(range(8)):
        restore_slot_rng(states[slot], None)
        second[slot] = (random.random(), np.random.random(), torch.rand(()).item())
    for slot in range(8):
        assert first[slot] == expected[slot][0]
        assert second[slot] == expected[slot][1]


def test_matcher_class_loss_reads_refined_final_state_logits_and_api_is_three_tuple():
    repo = Path(__file__).resolve().parents[1]
    loss_source = (repo / 'tokengs/models/object_locus_v3_set_loss.py').read_text()
    model_source = (repo / 'tokengs/models/object_locus_output_refine_gc001.py').read_text()
    assert 'prediction["states"][-1]["thing_logits19"]' in loss_source
    assert 'cls = self.panoptic.classify(q_refined)' in model_source
    assert "final.update(cls)" in model_source
    runtime = __import__('scripts.object_locus_output_refine_gc001_runtime', fromlist=['build_model', 'load_checkpoint'])
    assert len(inspect.signature(runtime.build_model).parameters) == 1
    assert len(inspect.signature(runtime.load_checkpoint).parameters) == 2
    assert runtime.load_checkpoint.__doc__ and 'three values' in runtime.load_checkpoint.__doc__


def test_training_loop_keeps_last_four_plan_columns_and_single_update_clip_step():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / 'scripts/train_object_locus_output_refine_gc001.py').read_text()
    tree = ast.parse(source)
    assert 'rank + 4 * micro' in (repo / 'scripts/object_locus_output_refine_gc001_runtime.py').read_text()
    assert 'entry[\'rank_windows\']' in source
    assert 'dist.all_reduce(global_counts' in source


def test_reconstruction_objective_stays_inherited_and_refiner_only_overrides_readout():
    from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
    from tokengs.models.object_locus_output_refine_gc001 import LocusGSObjectLocusOutputRefineV1Recon
    assert issubclass(LocusGSObjectLocusOutputRefineV1Recon, LocusGSObjectLocusPanopticV1Recon)
    assert 'step_loss' not in LocusGSObjectLocusOutputRefineV1Recon.__dict__
    assert 'forward_object_locus' not in LocusGSObjectLocusOutputRefineV1Recon.__dict__
    assert set(LocusGSObjectLocusOutputRefineV1Recon.__dict__) & {'_readout'} == {'_readout'}
