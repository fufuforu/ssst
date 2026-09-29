#!/usr/bin/env python3
"""Validate the registered V1-GC gradient-routing and audit artifacts."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "group_plus/anchor_group_v1_gc"
EXPECTED = {
    "GC-C1_forward_parity",
    "GC-C2_raw_gradient_reference",
    "GC-C3_reconstruction_formula_step1000",
    "GC-C4_group_unscaled_step1000",
    "GC-C5_warmup_composition_step600",
    "GC-C6_zero_understanding_step200",
    "GC-C7_hook_cleanup_and_next_recon",
    "GC-C8_no_parameter_mutation",
    "GC-C9_optimizer_parity",
    "GC-C10_effective_ratio_0_01x_raw",
    "GC-C11_cosine_invariant_positive_scale",
    "GC-C12_phase_a_18_of_18_regression",
    "GC-C13_reconstruction_parity",
    "GC-C14_s0_s1_legacy_regression",
    "GC-C15_rtx3090_v1_vs_gc_one_step",
}


def _finite_tree(obj):
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_finite_tree(x) for x in obj.values())
    if isinstance(obj, list):
        return all(_finite_tree(x) for x in obj)
    return True


def _helper_source_contract():
    source = (ROOT / "scripts/anchor_group_v1_gc.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "backward_gradient_controlled")
    segment = ast.get_source_segment(source, node)
    checks = {
        "fixed_scale_constant": "SHARED_UNDERSTANDING_GRAD_SCALE = 0.01" in source,
        "recon_backward_retains_graph": "loss_recon.backward(retain_graph=True)" in segment,
        "group_excluded_from_hooks": 'name.startswith("anchor_group.")' in segment,
        "hooks_removed_in_finally": "finally:" in segment and "handle.remove()" in segment,
        "no_optimizer_step_in_helper": not any(isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "step"
            for n in ast.walk(node)),
        "scale_cli_override_absent": "--shared-scale" not in source and
                                     "--understanding-grad-scale" not in source,
    }
    return checks


def main():
    path = OUT / "gc_contracts.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing GC contract results: {path}")
    doc = json.loads(path.read_text())
    checks = doc.get("checks", {})
    if set(checks) != EXPECTED:
        raise RuntimeError(f"GC contract set mismatch missing={sorted(EXPECTED-set(checks))} extra={sorted(set(checks)-EXPECTED)}")
    if doc.get("total") != 15 or doc.get("passed") != 15:
        raise RuntimeError(f"expected 15/15 GC contracts, got {doc.get('passed')}/{doc.get('total')}")
    if not all(x.get("passed") for x in checks.values()):
        raise RuntimeError("one or more GC contract rows failed")
    helper = _helper_source_contract()
    if not all(helper.values()):
        raise RuntimeError(f"backward helper source contract failed: {helper}")
    scale = json.loads((OUT / "gradient_scale_audit_fresh16.json").read_text())
    if len(scale.get("windows", [])) != 16 or not scale.get("effective_ratio_and_cosine_contract_pass"):
        raise RuntimeError("fresh 16-window gradient scale audit failed")
    smoke = json.loads((OUT / "v1_vs_gc_one_step_smoke.json").read_text())
    if smoke.get("status") != "pass" or smoke.get("optimizer_step_count") != 1:
        raise RuntimeError("RTX3090 one-step A/B smoke is not passing")
    root = json.loads((OUT / "parity_noise_root_cause.json").read_text())
    c1 = checks["GC-C1_forward_parity"]
    if not c1.get("C1a_forward_tensor_parity", {}).get("passed"):
        raise RuntimeError("C1a strict model/batch/forward tensor parity failed")
    if not c1.get("C1b_empirical_loss_parity", {}).get("passed"):
        raise RuntimeError("C1b empirical repeat-noise loss parity failed")
    if not root.get("C1a_forward_tensor_parity", {}).get("passed") or not root.get("C1b_empirical_loss_parity", {}).get("passed"):
        raise RuntimeError("parity noise root-cause audit does not establish corrected C1 gates")
    if not smoke.get("forward_tensor_parity", {}).get("all_exact"):
        raise RuntimeError("C15 strict forward tensor parity failed")
    if not smoke.get("empirical_loss_parity", {}).get("all_components_pass"):
        raise RuntimeError("C15 corrected scalar loss gate failed")
    if not smoke.get("model_state_exact_before_forward", {}).get("model_state_equal"):
        raise RuntimeError("C15 fresh model state parity failed")
    if not smoke.get("batch_exact_and_shared", {}).get("tensor_manifest_after_equal"):
        raise RuntimeError("C15 batch identity/integrity failed")
    if smoke.get("query_init_gradient_max_abs_diff_before_clip", float("inf")) > 1e-6:
        raise RuntimeError("C15 query gradient parity failed")
    if not smoke.get("optimizer_groups_identical") or not smoke.get("learning_rates_identical"):
        raise RuntimeError("C15 optimizer or LR parity failed")
    if not smoke.get("hook_registered_removed_equal"):
        raise RuntimeError("C15 temporary hook cleanup failed")
    if not _finite_tree(doc) or not _finite_tree(scale) or not _finite_tree(smoke) or not _finite_tree(root):
        raise RuntimeError("nonfinite audit artifact value")
    print(json.dumps({"contracts": "15/15 PASS", "helper_source_contract": helper,
                      "fresh_gradient_windows": 16, "gpu_smoke": smoke["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
