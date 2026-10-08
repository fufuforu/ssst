"""Pure CPU contracts for the GC001 frozen representation diagnostic.

No model/runtime imports belong in this module.  The matcher is deliberately
named separately from the model's historical Hungarian training assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def diagnostic_max_cardinality(iou: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Match the most threshold-valid edges, then maximize their IoU sum.

    Rows are GTs (already ordered by ascending instance ID), columns are query
    IDs in their natural order. Invalid edges have exactly zero weight; dummy
    columns permit every GT to remain unmatched.
    """
    values = np.asarray(iou, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("iou must be a finite rank-2 matrix")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("IoU values must be in [0,1]")
    ngt, nq = values.shape
    if ngt == 0 or nq == 0:
        return {"matches": [], "unmatched_gt": list(range(ngt)), "objective": [0, 0.0]}
    k = min(ngt, nq) + 1
    valid = values >= float(threshold)
    weights = np.zeros((ngt, nq + ngt), dtype=np.float64)
    weights[:, :nq] = np.where(valid, k + values, 0.0)
    rows, cols = linear_sum_assignment(-weights)
    matches = [(int(g), int(q), float(values[g, q]))
               for g, q in zip(rows, cols) if q < nq and valid[g, q]]
    used = {g for g, _, _ in matches}
    matches.sort(key=lambda x: (x[0], x[1]))
    return {"matches": matches,
            "unmatched_gt": [g for g in range(ngt) if g not in used],
            "objective": [len(matches), float(sum(x[2] for x in matches))]}


def labels_from_context(iou_gt_query: np.ndarray, gt_classes: list[int], *,
                        threshold: float = 0.5, negative_below: float = 0.1,
                        query_count: int = 100) -> dict[str, Any]:
    """Create the fixed 19-way labels; class inputs are internal IDs 2..19."""
    iou = np.asarray(iou_gt_query, dtype=np.float64)
    if iou.shape != (len(gt_classes), query_count):
        raise ValueError("IoU shape does not match GT and fixed query count")
    if any(int(c) < 2 or int(c) > 19 for c in gt_classes):
        raise ValueError("GT semantic classes must use internal IDs 2..19")
    result = np.full(query_count, -1, dtype=np.int64)
    roles = np.full(query_count, "AMBIGUOUS", dtype=object)
    match = diagnostic_max_cardinality(iou, threshold)
    for gi, qi, _ in match["matches"]:
        result[qi] = int(gt_classes[gi]) - 2
        roles[qi] = "POSITIVE"
    matched = {q for _, q, _ in match["matches"]}
    maxima = iou.max(axis=0) if len(gt_classes) else np.zeros(query_count, np.float64)
    for qi in range(query_count):
        if qi not in matched and maxima[qi] < negative_below:
            result[qi] = 18
            roles[qi] = "NEGATIVE"
    return {"labels": result, "roles": roles, "matches": match,
            "max_iou": maxima}


def opacity_membership_pool(feature: np.ndarray, membership: np.ndarray,
                            opacity: np.ndarray, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Compute z from w=a*P; low-mass rows are zero, never query fallbacks."""
    f = np.asarray(feature, dtype=np.float32)
    p = np.asarray(membership, dtype=np.float32)
    a = np.asarray(opacity, dtype=np.float32)
    if f.ndim != 2 or p.ndim != 2 or f.shape[0] != p.shape[0]:
        raise ValueError("feature and membership must be [G,D] and [G,Q]")
    if a.shape != (f.shape[0],):
        raise ValueError("opacity must have shape [G]")
    if not (np.isfinite(f).all() and np.isfinite(p).all() and np.isfinite(a).all()):
        raise ValueError("pool inputs must be finite")
    w = a[:, None] * p
    mass = w.sum(axis=0, dtype=np.float32)
    z = (w.T @ f) / np.maximum(mass, eps)[:, None]
    z[mass < eps] = 0
    return z.astype(np.float32, copy=False), mass


def contract_self_check() -> dict[str, Any]:
    """Small deterministic CPU mathematical contract, called by preflight."""
    # Two GTs share the same best query. Maximum coverage must choose the
    # alternate valid edge, even though greedy IoU would leave one GT unmatched.
    m = diagnostic_max_cardinality(np.array([[.90, .51], [.89, .0]]), .5)
    assert m["objective"][0] == 2 and {(g, q) for g, q, _ in m["matches"]} == {(0, 1), (1, 0)}
    dup = labels_from_context(np.array([[.8, .7, .2, .0]]), [4], query_count=4)
    assert dup["labels"].tolist() == [2, -1, -1, 18]
    empty = diagnostic_max_cardinality(np.zeros((0, 3)), .5)
    assert empty["matches"] == [] and empty["unmatched_gt"] == []
    boundary = diagnostic_max_cardinality(np.array([[.5]]), .5)
    assert boundary["objective"][0] == 1
    # A zero-opacity/zero-membership query is exactly zero and cannot leak q.
    z, mass = opacity_membership_pool(np.array([[99., -4.]], np.float32),
                                      np.zeros((1, 2), np.float32),
                                      np.ones(1, np.float32))
    assert np.array_equal(z, np.zeros((2, 2), np.float32)) and np.array_equal(mass, [0, 0])
    # Same seed construction contract and parameter counts.
    import torch
    from torch import nn
    def h1(seed: int):
        torch.manual_seed(seed)
        ln, linear = nn.LayerNorm(256, eps=1e-5), nn.Linear(256, 19)
        nn.init.xavier_uniform_(linear.weight); nn.init.zeros_(linear.bias)
        return ln, linear
    a, b = h1(20261), h1(20261)
    assert sum(p.numel() for mod in a for p in mod.parameters()) == 5395
    assert all(torch.equal(x, y) for ma, mb in zip(a, b) for x, y in zip(ma.parameters(), mb.parameters()))
    lnq, lnz, head = nn.LayerNorm(256, eps=1e-5), nn.LayerNorm(256, eps=1e-5), nn.Linear(512, 19)
    assert sum(p.numel() for mod in (lnq, lnz, head) for p in mod.parameters()) == 10771
    return {"passed": True, "checks": ["max_cardinality", "duplicate_ambiguous", "empty_gt",
            "threshold_boundary", "zero_fallback_no_q_leak", "head_shapes_and_initialization"],
            "h1_h2_parameters": 5395, "h3_parameters": 10771}
