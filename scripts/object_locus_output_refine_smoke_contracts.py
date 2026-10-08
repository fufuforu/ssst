"""Small, explicit contracts used by the R3D GPU smoke validator."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch


EXTRA_R3D_TENSORS = {"q_base", "q_refined", "states.11.q_refined"}
STRICT_PREFIXES = ("states.", "render.", "anchor_")
STRICT_NAMES = {
    "F_m", "q_pre", "gaussians", "alpha", "pixel_void_mass",
    "child_feature_residual", "p_class", "thing_class_logits",
    "thing_logits19", "class_logits19", "conditional_class_prob",
    "objectness_prob",
}
TOLERANCE_NAMES = {
    "lifting_mass", "lifting_gate", "gaussian_feature", "membership_mass",
    "region_mass", "pixel_membership", "semantic_scores",
}
DIAGNOSTIC_NAMES = {"gaussian_mask_logits", "assignment", "gaussian_membership"}


def flattened_tensors(obj, prefix=""):
    out = {}
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flattened_tensors(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (tuple, list)):
        for index, value in enumerate(obj):
            out.update(flattened_tensors(value, f"{prefix}.{index}" if prefix else str(index)))
    return out


def classify_public_tensor(key):
    if key in EXTRA_R3D_TENSORS:
        return "r3d_extra"
    if key in DIAGNOSTIC_NAMES:
        return "independent_diagnostic"
    if key in TOLERANCE_NAMES:
        return "independent_tolerance"
    if key in STRICT_NAMES or key.startswith(STRICT_PREFIXES):
        return "independent_strict_identity"
    return None


def compare_independent_forward(reference, candidate):
    """Apply fixed key/dtype/finite gates and the three explicit comparison classes."""
    ref = flattened_tensors(reference)
    cand = flattened_tensors(candidate)
    ref_keys, cand_keys = set(ref), set(cand)
    missing = sorted(ref_keys - cand_keys)
    extra = sorted(cand_keys - ref_keys)
    if set(extra) != EXTRA_R3D_TENSORS:
        raise AssertionError(f"unexpected R3D-only tensor paths: {extra}")
    if missing:
        raise AssertionError(f"missing C/R public tensor paths: {missing}")
    unclassified = sorted(key for key in ref_keys | cand_keys if classify_public_tensor(key) is None)
    if unclassified:
        raise AssertionError(f"unclassified public tensor paths: {unclassified}")

    strict, tolerant, diagnostic = [], [], []
    for key in sorted(ref_keys):
        a, b = ref[key], cand[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"shape/dtype mismatch at {key}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError(f"nonfinite public tensor at {key}")
        category = classify_public_tensor(key)
        if category == "independent_strict_identity":
            if not torch.equal(a, b):
                raise AssertionError(f"strict C/R identity mismatch at {key}")
            strict.append(key)
        elif category == "independent_tolerance":
            passed = (a - b).abs() <= (1e-6 + 1e-5 * b.abs())
            result = {"key": key, "failed_elements": int((~passed).sum()),
                      "maximum_absolute_difference": float((a - b).abs().max()) if a.numel() else 0.0}
            tolerant.append(result)
            if result["failed_elements"]:
                raise AssertionError(f"locked C/R tolerance failed at {key}: {result}")
        elif category == "independent_diagnostic":
            passed = (a - b).abs() <= (1e-6 + 1e-5 * b.abs())
            diagnostic.append({"key": key, "failed_elements": int((~passed).sum()),
                               "maximum_absolute_difference": float((a - b).abs().max()) if a.numel() else 0.0})
    extras = {}
    for key in sorted(extra):
        value = cand[key]
        if not torch.isfinite(value).all():
            raise AssertionError(f"nonfinite R3D-only tensor at {key}")
        extras[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {
        "common_tensor_count": len(ref_keys), "extra_r3d_tensors": extras,
        "classification": {
            "strict_identity_keys": strict,
            "tolerance_keys": sorted(row["key"] for row in tolerant),
            "diagnostic_only_keys": sorted(row["key"] for row in diagnostic),
        },
        "strict_identity_count": len(strict), "tolerance_results": tolerant,
        "independent_full_forward_diagnostic": {
            "tolerance": "abs(reference-candidate) <= 1e-6 + 1e-5*abs(candidate)",
            "diagnostic_only": diagnostic,
            "diagnostic_failed_elements_including_alias_duplicate": sum(x["failed_elements"] for x in diagnostic),
            "assignment_gaussian_membership_alias_note": "assignment and gaussian_membership are the same membership output; failure counts include both aliases",
        },
    }


def assert_tensor_tree_equal(left, right, label="tensor tree"):
    a, b = flattened_tensors(left), flattened_tensors(right)
    if set(a) != set(b):
        raise AssertionError(f"{label} key mismatch: {sorted(set(a) ^ set(b))}")
    for key in sorted(a):
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype or not torch.equal(a[key], b[key]):
            raise AssertionError(f"{label} exact identity failed at {key}")


def assert_value_tree_equal(left, right, label="value tree"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise AssertionError(f"{label} tensor/scalar type mismatch")
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"{label} tensor identity failed")
    elif isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)) or set(left) != set(right):
            raise AssertionError(f"{label} dictionary keys differ")
        for key in left:
            assert_value_tree_equal(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"{label} sequence shape differs")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_value_tree_equal(a, b, f"{label}.{index}")
    elif left != right:
        raise AssertionError(f"{label} scalar identity failed: {left!r} != {right!r}")


def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


def _check_camera(actual, expected, label):
    if not torch.equal(actual.cam_view, expected[0]) or not torch.equal(actual.intrinsics, expected[1]):
        raise AssertionError(f"{label} camera input changed")


def checked_lift_replacement(expected_feature_grid, expected_gaussians, expected_cam_view,
                            expected_intrinsics, expected_gs, cached, counts):
    def replacement(feature_grid, gaussians, camera, gs):
        if gs is not expected_gs:
            raise AssertionError("lift_features received a different gs object")
        for actual, expected, label in (
            (feature_grid, expected_feature_grid, "feature_grid"),
            (gaussians, expected_gaussians, "gaussians"),
            (camera.cam_view, expected_cam_view, "cam_view"),
            (camera.intrinsics, expected_intrinsics, "intrinsics"),
        ):
            if not torch.equal(actual, expected):
                raise AssertionError(f"lift_features {label} differs from shared cached input")
        counts["shared_lift_consumers"] += 1
        return cached
    return replacement


def checked_renderer_cache(original, expected_gs, counts):
    state = {"inputs": None, "output": None}

    def wrapper(gaussians, membership, cam_view, intrinsics):
        if state["output"] is None:
            state["inputs"] = tuple(x.detach().clone() for x in (gaussians, membership, cam_view, intrinsics))
            state["output"] = original(gaussians, membership, cam_view, intrinsics)
            counts["real_renderer_calls"] += 1
            return state["output"]
        for actual, expected, label in zip((gaussians, membership, cam_view, intrinsics), state["inputs"],
                                           ("gaussians", "membership", "cam_view", "intrinsics")):
            if not torch.equal(actual, expected):
                raise AssertionError(f"shared renderer {label} differs; cached output cannot hide it")
        counts["shared_renderer_consumers"] += 1
        return state["output"]

    return wrapper, state


@contextmanager
def patch_shared_readout_inputs(old_lift_module, new_lift_module, gs, lift_replacement,
                                original_renderer, counts):
    """Scope lift substitutions and the real-old/cached-new renderer wrapper."""
    renderer_wrapper, renderer_state = checked_renderer_cache(original_renderer, gs, counts)
    with ExitStack() as stack:
        stack.enter_context(patch.object(old_lift_module, "lift_features", lift_replacement))
        stack.enter_context(patch.object(new_lift_module, "lift_features", lift_replacement))
        stack.enter_context(patch.object(gs, "render_feature_channels", renderer_wrapper))
        yield renderer_state


def cleanup_distributed_if_initialized(dist):
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
