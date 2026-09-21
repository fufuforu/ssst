# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen SIU3R protocol constants and shape guards for the SSST joint run.

The view protocol, class count and matching/loss weights are inherited from the
audited SIU3R-aligned criterion so that training-time supervision and the
official SIU3R evaluator describe the same task.
"""

from __future__ import annotations

import torch

TRAIN_CONTEXT_VIEWS = 2
TRAIN_NOVEL_VIEWS = 2
TRAIN_TARGET_RECORDS = TRAIN_CONTEXT_VIEWS + TRAIN_NOVEL_VIEWS
VAL_CONTEXT_VIEWS = 2
VAL_NOVEL_VIEWS = 4
VAL_TARGET_RECORDS = VAL_CONTEXT_VIEWS + VAL_NOVEL_VIEWS

SEMANTIC_CLASS_COUNT = 20
NO_OBJECT_CLASS = SEMANTIC_CLASS_COUNT
QUERY_COUNT = 100

# Hungarian matching costs and supervision weights (inherited unchanged).
MATCH_COST_CLASS = 1.0
MATCH_COST_MASK_BCE = 5.0
MATCH_COST_DICE = 5.0
LOSS_WEIGHT_CLASS_CE = 2.0
LOSS_WEIGHT_MASK_BCE = 5.0
LOSS_WEIGHT_DICE = 5.0
NO_OBJECT_CE_WEIGHT = 0.1
OUTER_SEGMENTATION_WEIGHT = 0.05
# Disabled for the first joint experiment: ScanNet stuff/void pixels share
# instance ID 0, so the "same instance" test cannot separate wall/floor/void and
# would smooth depth across real boundaries.  The term stays implemented for a
# later mutual-benefit ablation.
INSTANCE_DEPTH_SMOOTHNESS_WEIGHT = 0.0
POINT_SAMPLE_COUNT = 4096


def validate_view_protocol(*, context_views: int, target_records: int, phase: str) -> None:
    expected = {
        "train": (TRAIN_CONTEXT_VIEWS, TRAIN_TARGET_RECORDS),
        "validation": (VAL_CONTEXT_VIEWS, VAL_TARGET_RECORDS),
    }
    if phase not in expected:
        raise ValueError(f"unknown phase {phase!r}")
    want_context, want_target = expected[phase]
    if (int(context_views), int(target_records)) != (want_context, want_target):
        raise ValueError(
            f"{phase} requires {want_context} context views and {want_target} target records, "
            f"got {context_views}/{target_records}"
        )


def validate_variable_target_batch(batch: dict, *, phase: str) -> None:
    """Reject padding/copying and enforce the frozen train/validation view count."""
    if "images_input" not in batch or "images_output" not in batch:
        raise ValueError("batch must expose separated input/output RGB tensors")
    context = int(batch["images_input"].shape[1])
    # The provider's ``images_output`` holds only the novel records; the target
    # record count additionally includes the two context records that are also
    # rendered and supervised.
    novel = int(batch["images_output"].shape[1])
    validate_view_protocol(context_views=context, target_records=context + novel, phase=phase)


def validate_query_outputs(
    class_logits: torch.Tensor,
    assignment_logits: torch.Tensor,
    mask_logits: torch.Tensor | None = None,
) -> None:
    """Assert the unified-query tensor contract."""
    if class_logits.ndim != 3 or tuple(class_logits.shape[1:]) != (
        QUERY_COUNT,
        SEMANTIC_CLASS_COUNT + 1,
    ):
        raise ValueError(f"query class logits must be [B,{QUERY_COUNT},21], got {tuple(class_logits.shape)}")
    if assignment_logits.ndim != 3 or assignment_logits.shape[0] != class_logits.shape[0]:
        raise ValueError(f"assignment logits must be [B,{QUERY_COUNT},N], got {tuple(assignment_logits.shape)}")
    if assignment_logits.shape[1] != QUERY_COUNT:
        raise ValueError(
            f"assignment logits must have {QUERY_COUNT} queries, got {assignment_logits.shape[1]}"
        )
    if mask_logits is not None:
        if mask_logits.ndim != 5 or mask_logits.shape[:2] != (class_logits.shape[0], QUERY_COUNT):
            raise ValueError(f"query masks must be [B,{QUERY_COUNT},V,H,W], got {tuple(mask_logits.shape)}")
    for name, value in (("class", class_logits), ("assignment", assignment_logits)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"query {name} logits are non-finite")


def query_semantic_maps(class_logits: torch.Tensor, mask_prob: torch.Tensor):
    """Class-aware aggregation of the query bank into semantic probabilities.

    The softmax runs over all 21 logits first, so the no-object probability
    suppresses that query's semantic contribution instead of being renormalized
    away by a 20-class softmax.
    """
    full_prob = torch.softmax(class_logits.float(), dim=-1)
    semantic_prob_per_query = full_prob[..., :SEMANTIC_CLASS_COUNT]
    semantic_prob = torch.einsum("bqc,bqvhw->bcvhw", semantic_prob_per_query, mask_prob.float())
    return semantic_prob, semantic_prob_per_query


__all__ = [
    "INSTANCE_DEPTH_SMOOTHNESS_WEIGHT",
    "LOSS_WEIGHT_CLASS_CE",
    "LOSS_WEIGHT_DICE",
    "LOSS_WEIGHT_MASK_BCE",
    "MATCH_COST_CLASS",
    "MATCH_COST_DICE",
    "MATCH_COST_MASK_BCE",
    "NO_OBJECT_CE_WEIGHT",
    "NO_OBJECT_CLASS",
    "OUTER_SEGMENTATION_WEIGHT",
    "POINT_SAMPLE_COUNT",
    "QUERY_COUNT",
    "SEMANTIC_CLASS_COUNT",
    "TRAIN_CONTEXT_VIEWS",
    "TRAIN_NOVEL_VIEWS",
    "TRAIN_TARGET_RECORDS",
    "VAL_CONTEXT_VIEWS",
    "VAL_NOVEL_VIEWS",
    "VAL_TARGET_RECORDS",
    "query_semantic_maps",
    "validate_query_outputs",
    "validate_variable_target_batch",
    "validate_view_protocol",
]
