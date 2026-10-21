#!/usr/bin/env python3
"""Deterministic per-instance point sampling for the mask loss (design + read-only check).

This module contains **no training path**: it implements the candidate sampling
rule for the next single-variable round and, on real plan windows, quantifies
what it would give compared with the current `_sample_points` even-index grid.
Nothing here is wired into the loss, and no checkpoint is written.

Rule (per scene, per GT instance i, total budget kept at `point_count = 4096`):

* positives: `k_i = min(|P_i|, K_POS_MAX)` points, taken from the instance's valid
  pixels P_i in raster order, evenly spaced (deterministic) when |P_i| > k_i;
* negatives: the remaining `4096 - k_i` points, evenly spaced (deterministic) over
  the complement `[0, V*H*W) \\ P_i` in raster order;
* the resulting index vector `idx_i` is identical for **every candidate query**
  and every call (pure function of the GT mask), so the Hungarian cost and the
  post-match BCE/Dice keep the same tensor shapes and the same total point count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import packed_panoptic_to_labels  # noqa: E402
from tokengs.models.ssst_loss import _sample_points, build_context_segments  # noqa: E402
from scripts.audit_instance_sampling import build_plan_windows  # noqa: E402

DEFAULT_POINT_COUNT = 4096
DEFAULT_K_POS_MAX = 2048


def evenly_spaced(values: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic, raster-order even sub-sampling (no RNG)."""
    if count >= values.numel():
        return values
    positions = torch.linspace(0, values.numel() - 1, count).round().long()
    return values.index_select(0, positions)


def instance_point_indices(
    mask: torch.Tensor,
    *,
    point_count: int = DEFAULT_POINT_COUNT,
    k_pos_max: int = DEFAULT_K_POS_MAX,
) -> tuple[torch.Tensor, int]:
    """[point_count] index vector (positives first, then negatives) + k_pos."""
    flat = mask.reshape(-1) > 0.5
    positive = torch.nonzero(flat, as_tuple=False).squeeze(-1)
    k_pos = int(min(positive.numel(), k_pos_max))
    if positive.numel() == 0:
        # no positive support at all: fall back to the plain even grid
        return evenly_spaced(torch.arange(flat.numel()), point_count), 0
    chosen_pos = evenly_spaced(positive, k_pos)
    complement = torch.nonzero(~flat, as_tuple=False).squeeze(-1)
    k_neg = point_count - k_pos
    chosen_neg = evenly_spaced(complement, min(k_neg, complement.numel()))
    indices = torch.cat([chosen_pos, chosen_neg])
    if indices.numel() < point_count:  # degenerate: pad by repeating the last point
        pad = indices[-1:].repeat(point_count - indices.numel())
        indices = torch.cat([indices, pad])
    return indices[:point_count], k_pos


def sampled_support(
    masks: torch.Tensor,
    *,
    point_count: int = DEFAULT_POINT_COUNT,
    k_pos_max: int = DEFAULT_K_POS_MAX,
) -> list[dict]:
    """Compare the current even-index grid with the proposed rule for one scene."""
    flat = masks.reshape(masks.shape[0], -1)
    current = _sample_points(masks, point_count=point_count)
    rows = []
    for position in range(masks.shape[0]):
        instance = flat[position]
        indices, k_pos = instance_point_indices(instance, point_count=point_count,
                                                k_pos_max=k_pos_max)
        new_positive = int(instance.index_select(0, indices).sum().item())
        rows.append({
            "area": int(instance.sum().item()),
            "current_positive_sampled_points": int(current[position].sum().item()),
            "proposed_k_pos": k_pos,
            "proposed_positive_sampled_points": new_positive,
            "deterministic": bool(
                torch.equal(indices, instance_point_indices(
                    instance, point_count=point_count, k_pos_max=k_pos_max)[0])
            ),
            "index_vector_length": int(indices.numel()),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--windows", type=int, default=300)
    parser.add_argument("--points", type=int, default=DEFAULT_POINT_COUNT)
    parser.add_argument("--k-pos-max", type=int, default=DEFAULT_K_POS_MAX)
    parser.add_argument("--out", default="workspace_group_plus/instance_sampler_v2_check.json")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    selected = build_plan_windows(plan, args.windows, split)

    rows = []
    for entry, scene_root in selected:
        semantic, instance = [], []
        for frame in entry["context"]:
            sem, ins = packed_panoptic_to_labels(scene_root / "panoptic" / f"{frame}.png")
            semantic.append(sem)
            instance.append(ins)
        semantic = torch.stack(semantic)[None]
        instance = torch.stack(instance)[None]
        classes, masks = build_context_segments(semantic, instance, (0, 1), stuff_class_count=2)
        labels = classes[0]
        things = masks[0][labels >= 2]
        if things.numel() == 0:
            continue
        rows.extend(sampled_support(things, point_count=args.points, k_pos_max=args.k_pos_max))

    def bucket(value: int) -> str:
        if value == 0:
            return "0"
        if value <= 4:
            return "1-4"
        if value <= 19:
            return "5-19"
        return ">=20"

    current = {name: 0 for name in ("0", "1-4", "5-19", ">=20")}
    proposed = dict.fromkeys(current, 0)
    for row in rows:
        current[bucket(row["current_positive_sampled_points"])] += 1
        proposed[bucket(row["proposed_positive_sampled_points"])] += 1
    zero_rows = [r for r in rows if r["current_positive_sampled_points"] == 0]
    zero_areas = sorted(r["area"] for r in zero_rows)
    report = {
        "scope": "read-only data-level check of the proposed sampling rule; no training",
        "windows": len(selected),
        "instances": len(rows),
        "point_count": args.points,
        "k_pos_max": args.k_pos_max,
        "current_support_counts": current,
        "proposed_support_counts": proposed,
        "current_zero_positive": len(zero_rows),
        "zero_positive_area_median": (
            float(np.median(zero_areas)) if zero_areas else 0.0
        ),
        "zero_positive_area_max": int(zero_areas[-1]) if zero_areas else 0,
        "zero_positive_area_min": int(zero_areas[0]) if zero_areas else 0,
        "proposed_min_positive": int(min(
            (r["proposed_positive_sampled_points"] for r in rows), default=0
        )),
        "all_index_vectors_deterministic": all(r["deterministic"] for r in rows),
        "all_index_vectors_full_length": all(
            r["index_vector_length"] == args.points for r in rows
        ),
        "sample_rows": rows[:40],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sample_rows"}, indent=1))
    print(f"[sampler-v2] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
