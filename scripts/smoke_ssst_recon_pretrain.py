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

"""Reconstruction-only (Experiment 1) smoke test.

Proves at runtime that the understanding branch is never touched:
`UnifiedObjectQueryHead.forward`, `render_query_masks`, `forward_joint`,
`build_context_segments`, `hungarian_match` and `class_aware_context_loss` are
replaced by guards that raise if they are called, and the optimizer / gradient
audit shows the query parameters are frozen, absent from the optimizer and have
no gradients.  Runs one real optimizer step, saves a checkpoint and then
initializes a full joint model from it (query branch must stay fresh).
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

import tyro  # noqa: E402

from tokengs.data.siu3r_processed import (  # noqa: E402
    SIU3RProcessedProvider,
    validate_batch_frame_order,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import (  # noqa: E402
    ModelInput,
    ModelInputDecoder,
    split_data,
)
from tokengs.models.siu3r_joint_ssst import SIU3RJointSSST  # noqa: E402
from tokengs.models.unified_object_queries import UnifiedObjectQueryHead  # noqa: E402
import tokengs.models.ssst_loss as ssst_loss  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def install_understanding_guards() -> dict:
    """Make any understanding-path call a hard failure; returns the originals."""

    def guard(name):
        def _guard(*args, **kwargs):
            raise AssertionError(f"reconstruction-only violated: {name} was called")

        return _guard

    originals = {
        "query_forward": UnifiedObjectQueryHead.forward,
        "render_query_masks": SIU3RJointSSST.render_query_masks,
        "forward_joint": SIU3RJointSSST.forward_joint,
        "build_context_segments": ssst_loss.build_context_segments,
        "hungarian_match": ssst_loss.hungarian_match,
        "class_aware_context_loss": ssst_loss.class_aware_context_loss,
    }
    UnifiedObjectQueryHead.forward = guard("UnifiedObjectQueryHead.forward")
    SIU3RJointSSST.render_query_masks = guard("SIU3RJointSSST.render_query_masks")
    SIU3RJointSSST.forward_joint = guard("SIU3RJointSSST.forward_joint")
    ssst_loss.build_context_segments = guard("build_context_segments")
    ssst_loss.hungarian_match = guard("hungarian_match")
    ssst_loss.class_aware_context_loss = guard("class_aware_context_loss")
    return originals


def restore_understanding_guards(originals: dict) -> None:
    UnifiedObjectQueryHead.forward = originals["query_forward"]
    SIU3RJointSSST.render_query_masks = originals["render_query_masks"]
    SIU3RJointSSST.forward_joint = originals["forward_joint"]
    ssst_loss.build_context_segments = originals["build_context_segments"]
    ssst_loss.hungarian_match = originals["hungarian_match"]
    ssst_loss.class_aware_context_loss = originals["class_aware_context_loss"]


def count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/tmp/ssst_recon_smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene", default="scene0048_01")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    opt = config_defaults["train_siu3r_ssst"].evolve(
        workspace=str(workspace), num_workers=0, reconstruction_only=True
    )
    torch.manual_seed(opt.seed)
    device = torch.device(args.device)
    report: dict = {"workspace": str(workspace), "reconstruction_only": True}

    provider = SIU3RProcessedProvider(opt, root=None, subset=args.scene, training=True, rank=0)
    batch = default_collate([provider[0]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    validate_batch_frame_order(batch, provider.last_pair, phase="train")
    report["batch"] = {
        "scene": provider.last_pair["scene_id"],
        "context_frame_ids": provider.last_pair["context_frame_ids"],
        "novel_frame_ids": provider.last_pair["novel_frame_ids"],
        "records": int(batch["images_all"].shape[1]),
    }

    model = model_registry[opt.model_type](opt).to(device)
    frozen = model.freeze_object_queries()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=opt.lr, betas=(0.9, 0.95)
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    query_parameters = dict(model.object_queries.named_parameters())
    report["parameters"] = {
        "total_model_params": count_parameters(model),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "object_query_params": sum(p.numel() for p in query_parameters.values()),
        "object_query_params_frozen": sum(
            1 for p in query_parameters.values() if not p.requires_grad
        ),
        "object_query_params_in_optimizer": sum(
            1 for p in query_parameters.values() if id(p) in optimizer_ids
        ),
        "frozen_names": len(frozen),
    }

    originals = install_understanding_guards()
    model.train()
    for step in range(args.steps):
        model.set_step_context(step, "train")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output, metrics = model.joint_step(batch, step=step, phase="train")
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite reconstruction-only loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        report[f"step_{step}"] = {
            "loss": float(loss.detach()),
            "loss_recon": float(metrics["loss_recon"]),
            "loss_spatial": float(metrics["loss_spatial"]),
            "psnr": float(metrics["psnr"]),
            "grad_norm": float(grad_norm),
            "metric_keys": sorted(metrics.keys()),
        }
        print(
            f"[recon-smoke] step {step} loss {float(loss.detach()):.4f} "
            f"(recon {float(metrics['loss_recon']):.4f} + spatial {float(metrics['loss_spatial']):.6f}) "
            f"psnr {float(metrics['psnr']):.3f} grad_norm {float(grad_norm):.3f}"
        )

    report["gradient_audit"] = {
        "object_query_grads_none": all(
            parameter.grad is None for parameter in query_parameters.values()
        ),
        "object_query_grad_values": {
            name: (None if parameter.grad is None else float(parameter.grad.abs().sum()))
            for name, parameter in query_parameters.items()
        },
        "has_understanding_metric": any("understanding" in key for key in metrics),
    }
    restore_understanding_guards(originals)

    checkpoint_dir = workspace / "checkpoints" / f"step_{args.steps:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    with open(checkpoint_dir / "config.yaml", "w", encoding="utf-8") as handle:
        handle.write(tyro.extras.to_yaml(opt))
    (checkpoint_dir / "COMPLETE").write_text("recon-smoke\n", encoding="utf-8")
    report["checkpoint"] = str(checkpoint_dir)

    # ---- checkpoint compatibility: recon checkpoint -> full joint model ----
    joint_opt = config_defaults["train_siu3r_ssst"].evolve(
        workspace=str(workspace), num_workers=0
    )
    torch.manual_seed(1234)
    joint = model_registry[joint_opt.model_type](joint_opt).to(device)
    query_before = {
        name: parameter.detach().clone()
        for name, parameter in joint.object_queries.named_parameters()
    }
    load_report = joint.init_from_reconstruction_checkpoint(
        str(checkpoint_dir), log=lambda message: None
    )
    query_after = dict(joint.object_queries.named_parameters())
    report["joint_initialization"] = {
        "loaded_keys": len(load_report["loaded"]),
        "partially_loaded_keys": len(load_report["partially_loaded"]),
        "skipped_keys": [entry["key"] for entry in load_report["skipped"]],
        "missing_keys": len(load_report["missing"]),
        "unexpected_keys": len(load_report["unexpected"]),
        "shape_mismatch": len(load_report["shape_mismatch"]),
        "query_branch_unchanged": all(
            torch.equal(query_before[name], query_after[name]) for name in query_before
        ),
        "spatial_backbone_loaded": torch.equal(
            joint.spatial_decoder.anchor_pre.detach(),
            model.spatial_decoder.anchor_pre.detach(),
        )
        and torch.equal(
            joint.enc_dec_backbone.encoder_norm.weight.detach(),
            model.enc_dec_backbone.encoder_norm.weight.detach(),
        ),
        # The Gaussian head must be loaded verbatim (all 14 raw channels): a
        # reconstruction-only checkpoint shares the local-offset semantics.
        "gaussian_head_full_load": torch.equal(
            joint.activation_head.deconv.weight.detach(),
            model.activation_head.deconv.weight.detach(),
        )
        and torch.equal(
            joint.activation_head.deconv.bias.detach(),
            model.activation_head.deconv.bias.detach(),
        ),
        "gaussian_head_partially_loaded": len(load_report["partially_loaded"]),
    }

    # ---- functional equality: same batch, same reconstruction path --------
    model.eval()
    joint.eval()
    functional_input, _ = split_data(batch, opt)
    functional_decoder = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    with torch.no_grad():
        source_out = model.forward_reconstruction_only(
            ModelInput(functional_input.encoder, functional_decoder),
            render_decoder_input=functional_decoder,
        )
        target_out = joint.forward_reconstruction_only(
            ModelInput(functional_input.encoder, functional_decoder),
            render_decoder_input=functional_decoder,
        )
    report["functional_reconstruction_equality"] = {
        "gaussians_bitwise_equal": bool(
            torch.equal(source_out["gaussians"], target_out["gaussians"])
        ),
        "gaussians_max_abs_diff": float(
            (source_out["gaussians"] - target_out["gaussians"]).abs().max()
        ),
        "rgb_max_abs_diff": float(
            (source_out["render"]["images_pred"] - target_out["render"]["images_pred"])
            .abs()
            .max()
        ),
        "depth_max_abs_diff": float(
            (source_out["render"]["depths_pred"] - target_out["render"]["depths_pred"])
            .abs()
            .max()
        ),
    }
    model.train()
    joint.eval()
    joint_input, _ = split_data(batch, joint_opt)
    joint_decoder = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    with torch.no_grad():
        joint_out = joint.forward_joint(
            ModelInput(joint_input.encoder, joint_decoder),
            mask_decoder_input=joint_decoder,
            render_decoder_input=joint_decoder,
        )
    report["joint_forward_after_init"] = {
        "gaussians": list(joint_out["gaussians"].shape),
        "query_class_logits": list(joint_out["query_class_logits"].shape),
        "query_mask_prob": list(joint_out["query_mask_prob"].shape),
        "all_finite": bool(
            torch.isfinite(joint_out["gaussians"]).all()
            and torch.isfinite(joint_out["query_class_logits"]).all()
            and torch.isfinite(joint_out["query_mask_prob"]).all()
        ),
    }

    (workspace / "recon_smoke_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
