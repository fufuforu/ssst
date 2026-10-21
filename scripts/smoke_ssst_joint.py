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

"""Single-batch smoke test for the SSST joint model.

Runs one real forward/backward (including the Gaussian renderer), checks the
spatial-grounding diagnostics for collapse or explosion, verifies a strict
checkpoint save/reload, and optionally exercises the validation adapter on one
record.  It never trains for more than the requested number of steps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO_ROOT)

from tokengs.data.siu3r_processed import (  # noqa: E402
    SIU3RProcessedProvider,
    validate_batch_frame_order,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import Options, config_defaults  # noqa: E402
import tyro  # noqa: E402


def scalar_metrics(metrics: dict) -> dict:
    return {
        key: float(value)
        for key, value in metrics.items()
        if torch.is_tensor(value) and value.ndim == 0
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/tmp/ssst_smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--eval-limit", type=int, default=1)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    report: dict = {"workspace": str(workspace), "steps": args.steps}

    opt: Options = config_defaults["train_siu3r_ssst"].evolve(
        workspace=str(workspace), num_workers=0
    )
    torch.manual_seed(opt.seed)
    device = torch.device(args.device)

    provider = SIU3RProcessedProvider(
        opt, root=None, subset=args.scene, training=True, rank=0
    )
    batch = default_collate([provider[0]])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    validate_batch_frame_order(batch, provider.last_pair, phase="train")
    report["batch"] = {
        "scene": provider.last_pair["scene_id"],
        "context_frame_ids": provider.last_pair["context_frame_ids"],
        "novel_frame_ids": provider.last_pair["novel_frame_ids"],
        "pair_iou": provider.last_pair["pair_iou"],
        "images_all": list(batch["images_all"].shape),
        "semantic_labels": list(batch["semantic_label_all"].shape),
    }

    model = model_registry[opt.model_type](opt).to(device)
    if args.init_checkpoint:
        report["init"] = model.init_from_checkpoint(args.init_checkpoint, log=print)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.95))

    for step in range(args.steps):
        model.set_step_context(step, "train")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output, metrics = model.joint_step(batch, step=step, phase="train")
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {scalar_metrics(metrics)}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        report[f"step_{step}"] = {
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "metrics": scalar_metrics(metrics),
        }
        print(f"[smoke] step {step} loss {float(loss.detach()):.4f} grad_norm {float(grad_norm):.3f}")

    # Re-run one forward so the reported spatial state reflects the update.
    model.eval()
    with torch.no_grad():
        output = model(batch, skip_loss=True)
    anchors = output["anchors"].detach().float()
    radii = output["radii"].detach().float()
    positions = output["gaussians"].detach().float()[..., :3].reshape(
        anchors.shape[0], anchors.shape[1], model.gaussians_per_token, 3
    )
    offsets = (positions - anchors.unsqueeze(2)).norm(dim=-1)
    report["spatial_after_step"] = {
        "anchor_mean": float(anchors.mean()),
        "anchor_abs_max": float(anchors.abs().max()),
        "radius_mean": float(radii.mean()),
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "offset_mean": float(offsets.mean()),
        "offset_max": float(offsets.max()),
        "offset_over_radius_max": float((offsets / radii.unsqueeze(-1)).max()),
        "finite_gaussians": bool(torch.isfinite(output["gaussians"]).all()),
        "assignment_entropy": float(
            -(
                output["query_assignment_prob"].detach().clamp_min(1e-8).log()
                * output["query_assignment_prob"].detach()
            ).sum(1).mean()
        ),
    }

    checkpoint_dir = workspace / "checkpoints" / f"step_{args.steps:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    with open(checkpoint_dir / "config.yaml", "w", encoding="utf-8") as handle:
        handle.write(tyro.extras.to_yaml(opt))
    (checkpoint_dir / "COMPLETE").write_text("smoke\n", encoding="utf-8")

    reloaded = model_registry[opt.model_type](opt).to(device)
    reloaded.lpips_loss = None
    result = reloaded.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=False),
        strict=True,
    )
    report["strict_reload"] = {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
    print("[smoke] strict checkpoint reload OK")

    if not args.skip_eval and args.val_manifest:
        eval_dir = workspace / "eval_smoke"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_ssst_validation.py"),
            "--checkpoint-dir", str(checkpoint_dir),
            "--manifest", args.val_manifest,
            "--output", str(eval_dir),
            "--limit", str(args.eval_limit),
            "--device", args.device,
        ]
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        report["eval_smoke"] = {"returncode": completed.returncode, "output": str(eval_dir)}
        report["eval_report"] = json.loads((eval_dir / "eval_report.json").read_text())

    (workspace / "smoke_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "init"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
