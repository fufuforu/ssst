#!/usr/bin/env python3
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

"""Offline query-grouping diagnostics for an SSST checkpoint.

Loads a checkpoint (no training, no evaluator) and averages the query-grouping
diagnostics over the fixed validation manifest:

* assignment temperature / entropy / no-object ratio (existing statistics)
* scene update magnitude of the queries vs their learnable seed
* pairwise cosine similarity between query features
* pairwise soft Dice between rendered query masks
* per-layer LayerScale magnitudes of the query decoder blocks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO_ROOT)

from scripts.evaluate_ssst_validation import load_model, load_options  # noqa: E402
from tokengs.data.siu3r_processed import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    SIU3RProcessedProvider,
)
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data  # noqa: E402
from tokengs.models.ssst_diagnostics import query_layer_scale_stats  # noqa: E402


QUERY_METRICS = (
    "assignment_temperature",
    "assignment_entropy",
    "no_object_ratio",
    "query_usage_mean",
    "active_query_count",
    "query_scene_update_norm_mean",
    "query_scene_update_norm_p95",
    "query_pairwise_cosine_mean",
    "query_pairwise_cosine_p95",
    "query_mask_pairwise_cosine_mean",
    "query_mask_pairwise_cosine_p95",
    "query_mask_pairwise_dice_mean",
    "query_mask_pairwise_dice_p95",
)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--val-root", default=str(Path(DEFAULT_DATA_ROOT) / "val"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA is not available; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    opt = load_options(checkpoint_dir, args.config)
    opt = opt.evolve(evaluating=True, use_input_supervision=False, num_views=6)
    model = load_model(checkpoint_dir, opt, device, log)
    provider = SIU3RProcessedProvider(
        opt,
        root=args.val_root,
        subset="all",
        training=False,
        val_pair_json=args.manifest,
        rank=0,
    )
    records = provider.dataset.val_pairs
    if args.limit is not None:
        records = records[: args.limit]
        provider.dataset.val_pairs = records

    totals = {name: 0.0 for name in QUERY_METRICS}
    count = 0
    for index in range(len(records)):
        sample = provider[index]
        batch = move_to_device(default_collate([sample]), device)
        model_input, _ = split_data(batch, opt)
        decoder = ModelInputDecoder(
            cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
        )
        with torch.no_grad():
            output = model.forward_joint(
                ModelInput(model_input.encoder, decoder),
                mask_decoder_input=decoder,
                render_decoder_input=decoder,
            )
        stats = output["query_stats"]
        for name in QUERY_METRICS:
            value = stats[name]
            totals[name] += float(value)
        count += 1

    averaged = {name: value / max(count, 1) for name, value in totals.items()}
    layer_scale = {
        name: float(value) for name, value in query_layer_scale_stats(model.object_queries.blocks).items()
    }
    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "manifest": str(Path(args.manifest).resolve()),
        "records": count,
        "query_metrics": averaged,
        "query_layer_scale": layer_scale,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"[grouping] {checkpoint_dir.name}: " + json.dumps({k: round(v, 4) for k, v in averaged.items()}))
    log("[grouping] layer scale: " + json.dumps({k: round(v, 6) for k, v in layer_scale.items()}))
    log(f"[grouping] written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
