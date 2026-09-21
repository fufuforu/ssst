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

"""Invoke the unmodified pinned SIU3R official evaluator on adapter output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SIU3R_REPO = "/space/mawb/SIU3R"
SIU3R_COMMIT = "8ea80166be76854f938e90521f1a5b688b755c87"
SIU3R_PYTHON = "/space/mawb/SIU3R/.venv_gpu_v4/bin/python"


def evaluate(eval_path: str | Path, *, device: str = "cuda") -> dict:
    sys.path.insert(0, SIU3R_REPO)
    try:
        from src.config import EvaluatorCfg
        from src.evaluator import Evaluator
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"the pinned SIU3R evaluator is missing a dependency ({error}); "
            f"run this script with {SIU3R_PYTHON} inside a GPU allocation"
        ) from error
    from src.utils.scannet_constant import (
        PANOPTIC_SEMANTIC2NAME,
        STUFF_CLASSES,
        THING_CLASSES,
    )

    cfg = EvaluatorCfg(
        dataset_name="scannet",
        eval_context_miou=True,
        eval_context_pq=True,
        eval_context_map=True,
        eval_target_miou=True,
        eval_target_pq=True,
        eval_target_map=True,
        eval_image_quality=True,
        eval_depth_quality=True,
        id2label=PANOPTIC_SEMANTIC2NAME,
        stuffs=STUFF_CLASSES,
        things=THING_CLASSES,
        device=device,
        eval_path=str(eval_path),
    )
    evaluator = Evaluator(cfg)
    evaluator.setup()
    return evaluator.evaluate(Path(eval_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate(args.eval_path, device=args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "official_evaluator_used": True,
                "siu3r_repo": SIU3R_REPO,
                "siu3r_commit": SIU3R_COMMIT,
                "eval_path": str(args.eval_path),
                "result": result,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
