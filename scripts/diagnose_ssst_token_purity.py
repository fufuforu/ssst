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

"""Offline token->GT-instance purity diagnostic for an SSST checkpoint.

For every spatial token the script renders that token's actual contribution to
the context views and measures how much of it lands on each ground-truth thing
instance.  It is meant to be run per checkpoint (not during training) and to be
comparable across variants (plain TokenGS, anchor-only, anchor + ray bias).
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
from tokengs.models.ssst_contracts import TRAIN_CONTEXT_VIEWS  # noqa: E402
from tokengs.models.ssst_diagnostics import token_purity_metrics  # noqa: E402


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
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--min-mass", type=float, default=1.0)
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

    per_record = []
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
        metrics = token_purity_metrics(
            model,
            output["gaussians"],
            decoder.select_batch(slice(None), slice(0, TRAIN_CONTEXT_VIEWS)),
            batch["semantic_label_all"][:, :TRAIN_CONTEXT_VIEWS],
            batch["instance_label_all"][:, :TRAIN_CONTEXT_VIEWS],
            chunk_size=args.chunk_size,
            min_mass=args.min_mass,
        )
        scene = records[index].get("scan", records[index].get("scene"))
        entry = {
            "record_index": index,
            "scene": scene,
            "context_ids": [int(x) for x in records[index]["context_ids"]],
            **{key: float(value) for key, value in metrics.items()},
        }
        per_record.append(entry)
        log(
            f"[purity] {scene} purity_mean {entry['token_gt_purity_mean']:.4f} "
            f"purity>0.8 {entry['token_gt_purity_gt_08_ratio']:.3f} "
            f"entropy {entry['token_gt_instance_entropy_mean']:.4f} "
            f"bg_ratio {entry['token_gt_background_mass_ratio']:.4f} "
            f"valid_tokens {entry['token_gt_valid_token_ratio']:.3f}"
        )

    keys = [key for key in per_record[0] if key.startswith("token_gt_")] if per_record else []
    summary = {
        key: sum(entry[key] for entry in per_record) / max(len(per_record), 1) for key in keys
    }
    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "manifest": str(Path(args.manifest).resolve()),
        "records": len(per_record),
        "min_mass": args.min_mass,
        "chunk_size": args.chunk_size,
        "summary": summary,
        "per_record": per_record,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"[purity] summary {json.dumps({k: round(v, 5) for k, v in summary.items()})}")
    log(f"[purity] written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
