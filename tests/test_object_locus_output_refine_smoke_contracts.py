import types

import pytest
import torch

from scripts.object_locus_output_refine_smoke_contracts import (
    DIAGNOSTIC_NAMES, EXTRA_R3D_TENSORS, STRICT_NAMES, TOLERANCE_NAMES,
    assert_tensor_tree_equal, checked_lift_replacement, checked_renderer_cache,
    cleanup_distributed_if_initialized, compare_independent_forward,
    patch_shared_readout_inputs,
)


def _full_classified_pair():
    keys = set(STRICT_NAMES) | set(TOLERANCE_NAMES) | set(DIAGNOSTIC_NAMES) | {
        'states.0.q', 'render.rgb', 'anchor_membership', 'anchor_assignment', 'anchor_mask_logits'
    }
    reference = {key: torch.tensor([1.0]) for key in keys}
    candidate = {key: value.clone() for key, value in reference.items()}
    for key in EXTRA_R3D_TENSORS:
        candidate[key] = torch.tensor([1.0])
    return reference, candidate


def test_classifier_covers_exact_tolerance_and_only_three_diagnostic_tensors():
    reference, candidate = _full_classified_pair()
    candidate['gaussian_mask_logits'] += 1e-3
    row = compare_independent_forward(reference, candidate)
    assert row['classification']['diagnostic_only_keys'] == sorted(DIAGNOSTIC_NAMES)
    assert row['classification']['tolerance_keys'] == sorted(TOLERANCE_NAMES)
    assert row['independent_full_forward_diagnostic']['diagnostic_failed_elements_including_alias_duplicate'] == 1
    reference['unclassified_public_key'] = torch.tensor([1.0])
    candidate['unclassified_public_key'] = torch.tensor([1.0])
    with pytest.raises(AssertionError, match='unclassified public tensor paths'):
        compare_independent_forward(reference, candidate)


def test_strict_identity_and_tolerance_failures_are_rejected():
    reference, candidate = _full_classified_pair()
    candidate['F_m'] += 1
    with pytest.raises(AssertionError, match='strict C/R identity mismatch'):
        compare_independent_forward(reference, candidate)
    reference, candidate = _full_classified_pair()
    candidate['lifting_gate'] += 1e-2
    with pytest.raises(AssertionError, match='locked C/R tolerance failed'):
        compare_independent_forward(reference, candidate)


def test_lift_consumers_check_every_input_before_returning_cached_evidence():
    grid, gaussians = torch.ones(2), torch.ones(3)
    cam, intr = torch.ones(4), torch.ones(5)
    gs = object(); cached = (torch.tensor([7.]), torch.tensor([8.]), torch.tensor([9.]))
    counts = {'shared_lift_consumers': 0}
    replacement = checked_lift_replacement(grid, gaussians, cam, intr, gs, cached, counts)
    camera = types.SimpleNamespace(cam_view=cam.clone(), intrinsics=intr.clone())
    assert replacement(grid.clone(), gaussians.clone(), camera, gs) is cached
    camera.cam_view[0] += 1
    with pytest.raises(AssertionError, match='cam_view'):
        replacement(grid, gaussians, camera, gs)
    assert counts['shared_lift_consumers'] == 1


def test_renderer_cache_rejects_changed_membership_before_reusing_output():
    class GS:
        pass
    gs = GS(); counts = {'real_renderer_calls': 0, 'shared_renderer_consumers': 0}
    real_calls = []
    def real(gaussians, membership, cam, intr):
        real_calls.append(1)
        return {'images_pred': membership + 2}
    wrapper, state = checked_renderer_cache(real, gs, counts)
    inputs = (torch.ones(2), torch.zeros(3), torch.ones(4), torch.ones(5))
    out = wrapper(*inputs)
    assert wrapper(inputs[0].clone(), inputs[1].clone(), inputs[2].clone(), inputs[3].clone()) is out
    assert len(real_calls) == 1 and counts['shared_renderer_consumers'] == 1
    changed = list(inputs); changed[1] = torch.ones(3)
    with pytest.raises(AssertionError, match='membership differs'):
        wrapper(*changed)
    assert state['output'] is out


def test_scoped_patches_restore_all_originals():
    old = types.SimpleNamespace(lift_features=lambda *args: 'old')
    new = types.SimpleNamespace(lift_features=lambda *args: 'new')
    class GS:
        def render_feature_channels(self, *args): return 'renderer'
    gs = GS(); old_lift, new_lift, renderer = old.lift_features, new.lift_features, gs.render_feature_channels
    counts = {'real_renderer_calls': 0, 'shared_renderer_consumers': 0}
    cached = (torch.ones(1), torch.ones(1), torch.ones(1))
    replacement = lambda *args: cached
    with patch_shared_readout_inputs(old, new, gs, replacement, renderer, counts):
        assert old.lift_features is replacement and new.lift_features is replacement
        assert gs.render_feature_channels is not renderer
    assert old.lift_features is old_lift and new.lift_features is new_lift
    assert gs.render_feature_channels.__func__ is renderer.__func__


def test_single_process_group_cleanup_does_not_barrier_or_destroy():
    class Dist:
        barriers = 0; destroys = 0
        def is_initialized(self): return False
        def barrier(self): self.barriers += 1
        def destroy_process_group(self): self.destroys += 1
    dist = Dist()
    cleanup_distributed_if_initialized(dist)
    assert dist.barriers == dist.destroys == 0
    dist.is_initialized = lambda: True
    cleanup_distributed_if_initialized(dist)
    assert dist.barriers == dist.destroys == 1


def test_exact_tree_rejects_dtype_and_value_changes():
    assert_tensor_tree_equal({'x': torch.ones(2)}, {'x': torch.ones(2)})
    with pytest.raises(AssertionError, match='exact identity'):
        assert_tensor_tree_equal({'x': torch.ones(2)}, {'x': torch.ones(2, dtype=torch.float64)})
