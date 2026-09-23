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

"""One-stage SIU3R joint training entry for spatially grounded shared tokens.

Single process on one GPU, or multi-process via `torchrun`.  The model is
complete from step 1 (no reconstruction warm-up checkpoint is required); the
understanding loss simply ramps in with a configurable curriculum.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, default_collate

import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.runtime_bootstrap import prepare_runtime

prepare_runtime(_REPO_ROOT)

from tokengs.data.siu3r_processed import (
    DEFAULT_DATA_ROOT,
    SIU3RProcessedProvider,
    validate_batch_frame_order,
)
from tokengs.models import model_registry
from tokengs.models.ssst_diagnostics import shared_gradient_diagnostic
from tokengs.options import Options, config_defaults


TRAIN_ROOT = str(Path(DEFAULT_DATA_ROOT) / "train")
VAL_ROOT = str(Path(DEFAULT_DATA_ROOT) / "val")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Output workspace directory.")
    parser.add_argument(
        "--preset",
        default="train_siu3r_ssst",
        help=(
            "Base Options preset from tokengs.options.config_defaults, e.g. "
            "train_siu3r_ssst (joint), train_siu3r_locusgs_recon, "
            "train_siu3r_plain_tokengs_canonical_recon."
        ),
    )
    parser.add_argument("--num-steps", type=int, required=True, help="Optimizer steps to run.")
    parser.add_argument("--train-root", default=TRAIN_ROOT)
    parser.add_argument("--val-root", default=VAL_ROOT)
    parser.add_argument("--val-manifest", default=None, help="Fixed validation manifest (unused unless --val-freq > 0).")
    parser.add_argument("--val-freq", type=int, default=0, help="0 disables in-loop validation.")
    parser.add_argument("--val-limit", type=int, default=1)
    parser.add_argument("--scene", default="all", help="Training subset (comma separated scene names or 'all').")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--amp", default=None, choices=["bf16", "fp16", "no"], help="Autocast dtype.")
    parser.add_argument("--ckpt-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 keeps the in-process context/novel pair audit active.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Optional warm start (only compatible keys are loaded).")
    parser.add_argument("--resume", default=None, help="Checkpoint directory or .pt file to resume weights and step from.")
    parser.add_argument("--save-optimizer", action="store_true", help="Also store optimizer state (large).")
    parser.add_argument("--understanding-warmup-steps", type=int, default=None)
    parser.add_argument("--understanding-start-weight", type=float, default=None)
    parser.add_argument("--understanding-final-weight", type=float, default=None)
    parser.add_argument("--spatial-compactness-weight", type=float, default=None)
    parser.add_argument("--spatial-radius-weight", type=float, default=None)
    parser.add_argument(
        "--gradient-diagnostic-freq",
        type=int,
        default=None,
        help="Steps between shared-task gradient diagnostics (0 disables).",
    )
    parser.add_argument("--num-object-queries", type=int, default=None)
    parser.add_argument(
        "--reconstruction-only",
        action="store_true",
        default=None,
        help=(
            "Experiment 1: train only the spatial reconstruction path. The unified "
            "object-query branch is frozen and never executed."
        ),
    )
    parser.add_argument("--anchor-center-z", type=float, default=None)
    parser.add_argument("--anchor-extent", type=float, default=None)
    parser.add_argument("--anchor-init-radius", type=float, default=None)
    parser.add_argument("--anchor-local-offset-bound", type=float, default=None)
    parser.add_argument(
        "--anchor-ray-sigma0",
        type=float,
        default=None,
        help="Width of the anchor-to-ray geometric bias (softplus/r-adaptive scale).",
    )
    parser.add_argument(
        "--anchor-ray-bias-clamp",
        type=float,
        default=None,
        help="Lower clamp of the geometric bias before the learnable per-layer gamma.",
    )
    parser.add_argument(
        "--anchor-ray-bias-init",
        type=float,
        default=None,
        help="Initial raw gamma; softplus(-6) ~ 2.5e-3 keeps the warm start intact.",
    )
    parser.add_argument("--allow-completed-workspace", action="store_true")
    return parser.parse_args(argv)


def build_options(args: argparse.Namespace) -> Options:
    if args.preset not in config_defaults:
        raise ValueError(
            f"unknown preset {args.preset!r}; available: "
            + ", ".join(sorted(k for k in config_defaults if k.startswith(("train_", "eval_"))))
        )
    opt = config_defaults[args.preset].evolve(workspace=args.workspace)
    overrides = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lr": args.lr,
        "gradient_clip": args.grad_clip,
        "mixed_precision": args.amp,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "init_checkpoint": args.init_checkpoint,
        "understanding_warmup_steps": args.understanding_warmup_steps,
        "understanding_start_weight": args.understanding_start_weight,
        "understanding_final_weight": args.understanding_final_weight,
        "spatial_compactness_weight": args.spatial_compactness_weight,
        "spatial_radius_weight": args.spatial_radius_weight,
        "gradient_diagnostic_freq": args.gradient_diagnostic_freq,
        "num_object_queries": args.num_object_queries,
        "reconstruction_only": args.reconstruction_only,
        "anchor_center_z": args.anchor_center_z,
        "anchor_extent": args.anchor_extent,
        "anchor_init_radius": args.anchor_init_radius,
        "anchor_local_offset_bound": args.anchor_local_offset_bound,
        "anchor_ray_sigma0": args.anchor_ray_sigma0,
        "anchor_ray_bias_clamp": args.anchor_ray_bias_clamp,
        "anchor_ray_bias_init": args.anchor_ray_bias_init,
    }
    return opt.evolve(**{k: v for k, v in overrides.items() if v is not None})


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def lr_at(step: int, args: argparse.Namespace, base_lr: float) -> float:
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return base_lr * float(step + 1) / float(args.warmup_steps)
    total = max(int(args.num_steps), 1)
    progress = min(1.0, max(0.0, (step - args.warmup_steps) / max(1, total - args.warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (args.lr_min_ratio + (1.0 - args.lr_min_ratio) * cosine)


def setup_distributed() -> tuple[int, int, int, torch.device, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, torch.device("cuda", local_rank), True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, device, False


def build_loader(provider, opt, args, training: bool, rank: int, world_size: int):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            provider, num_replicas=world_size, rank=rank, shuffle=training, drop_last=training
        )
    return DataLoader(
        provider,
        batch_size=opt.batch_size,
        shuffle=bool(training and sampler is None),
        sampler=sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=bool(training),
        collate_fn=default_collate,
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    workspace: Path,
    opt: Options,
    args: argparse.Namespace,
    extra: dict,
) -> Path:
    checkpoint_dir = workspace / "checkpoints" / f"step_{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    with open(checkpoint_dir / "config.yaml", "w", encoding="utf-8") as handle:
        handle.write(tyro.extras.to_yaml(opt))
    metadata = {
        "global_optimizer_step": step,
        "world_size": args.world_size,
        "batch_size_per_rank": opt.batch_size,
        "gradient_accumulation_steps": opt.gradient_accumulation_steps,
        "architecture": getattr(model, "architecture_name", type(model).__name__),
        **extra,
    }
    with open(checkpoint_dir / "global_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    if args.save_optimizer:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint_dir


def resolve_checkpoint_file(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        for name in ("model.pt", "model.safetensors", "state.pt"):
            if (candidate / name).is_file():
                return candidate / name
        raise FileNotFoundError(f"no model file in checkpoint directory {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint not found: {candidate}")
    return candidate


def load_resume(model, optimizer, args, device, log) -> int:
    if not args.resume:
        return 0
    checkpoint_file = resolve_checkpoint_file(args.resume)
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {key: value for key, value in state.items() if "lpips_loss" not in key}
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state and "lpips_loss" not in key]
    unexpected = [key for key in state if key not in model_state]
    mismatched = [
        key for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    ]
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        log(
            f"[resume] non-strict load: missing={result.missing_keys[:8]} "
            f"unexpected={result.unexpected_keys[:8]}"
        )
    log(f"[resume] missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}")
    step = 0
    metadata_path = Path(args.resume) / "global_metadata.json" if Path(args.resume).is_dir() else None
    if metadata_path and metadata_path.is_file():
        step = int(json.loads(metadata_path.read_text(encoding="utf-8")).get("global_optimizer_step", 0))
    optimizer_path = Path(args.resume) / "optimizer.pt" if Path(args.resume).is_dir() else None
    if optimizer_path and optimizer_path.is_file():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
        log(f"[resume] restored optimizer state from {optimizer_path}")
    return step


def format_metrics(metrics: dict, step: int, lr: float, elapsed: float) -> str:
    fields = {
        "loss": metrics.get("loss"),
        "recon": metrics.get("loss_recon"),
        "und": metrics.get("loss_understanding"),
        "lam_u": metrics.get("lambda_understanding"),
        "spatial": metrics.get("loss_spatial"),
        "psnr": metrics.get("psnr"),
        "r_mean": metrics.get("spatial/radius_mean"),
        "r_max": metrics.get("spatial/radius_max"),
        "off_p95": metrics.get("spatial/local_offset_norm_p95"),
        "a_upd": metrics.get("spatial/anchor_update_norm_mean"),
        "ent": metrics.get("spatial/assignment_entropy"),
        "qact": metrics.get("spatial/active_query_count"),
        "noobj": metrics.get("spatial/no_object_ratio"),
        # Corrected LocusGS reconstruction diagnostics (detached, per log interval).
        "l6_rgb": metrics.get("loss_rgb_layer6"),
        "l6_ssim": metrics.get("loss_ssim_layer6"),
        "l6_gvis": metrics.get("loss_gaussian_visibility_layer6"),
        "l6_avis": metrics.get("loss_anchor_visibility_layer6"),
        "l12_rgb": metrics.get("loss_rgb_layer12"),
        "l12_ssim": metrics.get("loss_ssim_layer12"),
        "l12_gvis": metrics.get("loss_gaussian_visibility_layer12"),
        "l12_avis": metrics.get("loss_anchor_visibility_layer12"),
        "mu_min": metrics.get("anchor_min"),
        "mu_max": metrics.get("anchor_max"),
        "mu_std": metrics.get("anchor_std"),
        "r_p50": metrics.get("radius_p50"),
        "r_p95": metrics.get("radius_p95"),
        "d_mean": metrics.get("delta_norm_mean"),
        "d_p95": metrics.get("delta_norm_p95"),
        "d_max": metrics.get("delta_norm_max"),
        "gs_xyz_min": metrics.get("gaussian_xyz_min"),
        "gs_xyz_max": metrics.get("gaussian_xyz_max"),
        "gs_z_p01": metrics.get("gaussian_z_p01"),
        "gs_z_p50": metrics.get("gaussian_z_p50"),
        "gs_z_p99": metrics.get("gaussian_z_p99"),
        "alpha": metrics.get("alpha_mean"),
        "alpha_nz": metrics.get("alpha_nonzero_fraction"),
        "depth_nz": metrics.get("depth_nonzero_fraction"),
        "gamma": metrics.get("gamma_mean"),
        "bclip": metrics.get("ray_bias_clamped_fraction"),
    }
    parts = [f"step {step}", f"lr {lr:.2e}", f"dt {elapsed:.2f}s"]
    for name, value in fields.items():
        if value is None:
            continue
        parts.append(f"{name} {float(value):.4f}")
    return "[train] " + " ".join(parts)


def run_validation(raw_model, provider, opt, args, device, log, step):
    raw_model.eval()
    loader = DataLoader(provider, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.val_limit > 0 and index >= args.val_limit:
                break
            batch = move_to_device(batch, device)
            _, metrics = raw_model.joint_step(batch, step=step, phase="validation")
            for key, value in metrics.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
    raw_model.train()
    if count == 0:
        return {}
    averaged = {key: value / count for key, value in totals.items()}
    log(f"[val] step {step} " + " ".join(f"{k} {v:.4f}" for k, v in sorted(averaged.items())))
    return averaged


def accumulate_microbatches(
    epoch_batches,
    *,
    num_steps: int,
    grad_accum: int,
    start_step: int = 0,
):
    """Yield micro-batches with an accumulation window that spans epochs.

    ``epoch_batches(epoch)`` returns the iterable of micro-batches for one
    epoch.  The caller performs one optimizer step exactly every
    ``grad_accum`` micro-batches, so an epoch whose batch count is not divisible
    by ``grad_accum`` neither loses its partial gradient nor borrows the next
    window: the window simply continues into the next epoch.

    Yields:
        ``(step, epoch, batch, do_optimizer_step)`` where ``step`` is the number
        of completed optimizer steps (1-based for the first completed step).
    """
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    step = int(start_step)
    micro_step = 0
    epoch = 0
    while step < num_steps:
        epoch_batch_count = 0
        for batch in epoch_batches(epoch):
            epoch_batch_count += 1
            micro_step += 1
            do_step = micro_step % grad_accum == 0
            if do_step:
                step += 1
            yield step, epoch, batch, do_step
            if step >= num_steps:
                return
        if epoch_batch_count == 0:
            raise RuntimeError(f"no training batches in epoch {epoch}")
        epoch += 1


def record_keys_for_ranks(gathered: list[dict]) -> list[tuple[str, tuple[int, ...]]]:
    """Identity of the record each rank consumed, for the DDP audit.

    The key is ``(scene_name, frame_ids)``: different ScanNet scenes can share
    the same frame numbering, so frame ids alone would raise false alarms.
    """
    return [(str(item["scene_name"]), tuple(int(x) for x in item["frame_ids"])) for item in gathered]


def assert_distinct_records(gathered: list[dict]) -> list[tuple[str, tuple[int, ...]]]:
    """Fail only if two ranks really consumed the same scene+frames record."""
    keys = record_keys_for_ranks(gathered)
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"two ranks received the identical record (scene, frame ids): {keys}")
    return keys


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rank, local_rank, world_size, device, distributed = setup_distributed()
    args.world_size = world_size
    is_main = rank == 0

    def log(message: str) -> None:
        if is_main:
            print(message, flush=True)

    opt = build_options(args)
    seed_all(opt.seed)
    workspace = Path(args.workspace)
    if is_main:
        workspace.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    if (workspace / "status" / "COMPLETE").is_file() and not args.allow_completed_workspace and not args.resume:
        raise RuntimeError(
            f"{workspace}/status/COMPLETE exists; remove it or pass --allow-completed-workspace"
        )

    train_provider = SIU3RProcessedProvider(
        opt,
        root=args.train_root,
        subset=args.scene,
        training=True,
        rank=rank,
    )
    train_loader = build_loader(train_provider, opt, args, training=True, rank=rank, world_size=world_size)
    log(
        f"[setup] scenes={len(train_provider)} batch_size={opt.batch_size} "
        f"grad_accum={opt.gradient_accumulation_steps} world_size={world_size} device={device}"
    )

    val_provider = None
    if args.val_freq > 0:
        if not args.val_manifest:
            raise ValueError("--val-freq requires --val-manifest")
        val_provider = SIU3RProcessedProvider(
            opt,
            root=args.val_root,
            subset="all",
            training=False,
            val_pair_json=args.val_manifest,
            rank=rank,
        )
        log(f"[setup] validation records={len(val_provider)}")

    model = model_registry[opt.model_type](opt).to(device)
    if opt.model_type == "siu3r_locusgs_recon":
        anchor_decoder = getattr(model, "anchor_decoder", None)
        if anchor_decoder is None:
            raise RuntimeError("LocusGS preset selected but the model has no anchor decoder")
        pe_mode = str(getattr(anchor_decoder, "pe_mode", "?"))
        log(
            f"[setup] LocusGS pe_mode={pe_mode} (persistent PE is forbidden for the "
            f"formal run) | gamma_raw_init={float(opt.locusgs_gamma_raw_init)} "
            f"gamma_init={float(torch.nn.functional.softplus(torch.tensor(float(opt.locusgs_gamma_raw_init)))):.6f} "
            f"r_init={float(opt.locusgs_radius_init)} sigma0={float(opt.locusgs_sigma0)} "
            f"clamp={float(opt.locusgs_bias_clamp)}"
        )
        log(
            f"[setup] LocusGS impl={str(getattr(anchor_decoder, 'impl', '?'))} "
            f"refine_hidden={int(getattr(anchor_decoder, 'refine_hidden', 0))} | "
            f"per_layer_pe={'yes' if getattr(anchor_decoder, 'pe_mlps', None) is not None else 'no'} "
            f"| locusgs_specific_params="
            f"{sum(p.numel() for n, p in model.named_parameters() if n.startswith(('anchor_decoder.',)) and not n.startswith('anchor_decoder.decoder_blocks.')):,}"
        )
        log(
            f"[setup] anchor mu init range=[{float(anchor_decoder.mu.min()):.4f}, "
            f"{float(anchor_decoder.mu.max()):.4f}] "
            f"radii init mean={float(anchor_decoder.activated_radius(anchor_decoder.rho).mean()):.4f}"
        )
        if pe_mode != "injected":
            raise RuntimeError(
                f"formal LocusGS runs require locusgs_pe_mode='injected', got {pe_mode!r}"
            )
    if opt.init_checkpoint:
        log(f"[setup] warm start from {opt.init_checkpoint}")
        model.init_from_checkpoint(opt.init_checkpoint, log=log)
    if opt.reconstruction_only:
        frozen = model.freeze_object_queries()
        log(
            f"[setup] reconstruction-only: froze {len(frozen)} object-query parameters "
            "(no query forward, no mask rendering, no Hungarian, no understanding loss)"
        )
    # TokenGS-style parameter grouping: 1-D parameters (norms, biases, LayerScale
    # gammas) and any tensor flagged `_no_weight_decay` (the whole activation head)
    # are excluded from weight decay; only the remaining >=2-D weights decay.
    decay_params, nodecay_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() == 1 or getattr(parameter, "_no_weight_decay", False):
            nodecay_params.append(parameter)
        else:
            decay_params.append(parameter)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(args.weight_decay)})
    if nodecay_params:
        groups.append({"params": nodecay_params, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=opt.lr, betas=(0.9, 0.95))
    log(
        f"[setup] optimizer groups: decay(wd={args.weight_decay})={sum(p.numel() for p in decay_params):,} params "
        f"| nodecay(wd=0)={sum(p.numel() for p in nodecay_params):,} params"
    )
    step = load_resume(model, optimizer, args, device, log)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    model.train()

    log(f"[setup] model={type(model).__name__} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if is_main:
        with open(workspace / "config.yaml", "w", encoding="utf-8") as handle:
            handle.write(tyro.extras.to_yaml(opt))

    use_amp = opt.mixed_precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if opt.mixed_precision == "bf16" else torch.float16
    # Single source of truth for the low-frequency shared-gradient diagnostic:
    # the CLI flag is optional and the option default (200) applies when it is
    # omitted, so the loop must never compare the raw argparse value.
    if opt.gradient_diagnostic_freq is None:
        raise ValueError("gradient_diagnostic_freq must be set (Options default is 200)")
    gradient_diagnostic_freq = int(opt.gradient_diagnostic_freq)
    if opt.reconstruction_only and gradient_diagnostic_freq > 0:
        log("[setup] reconstruction-only: shared-gradient diagnostic disabled (no understanding loss)")
        gradient_diagnostic_freq = 0
    history: list[dict] = []
    accumulator = max(1, int(opt.gradient_accumulation_steps))
    start_time = time.time()
    raw_model = model.module if distributed else model

    def epoch_batches(epoch: int):
        train_provider.set_rng_epoch(epoch)
        train_provider.pair_rng.seed(int(opt.seed) + 7919 * epoch + 100003 * rank)
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        return train_loader

    # A clean accumulation state before the first window.  The window then runs
    # continuously across epoch boundaries: an epoch whose batch count is not a
    # multiple of grad_accum hands its partial gradients to the next epoch
    # instead of losing them at the epoch boundary.
    optimizer.zero_grad(set_to_none=True)
    for completed_step, epoch, batch, do_step in accumulate_microbatches(
        epoch_batches,
        num_steps=args.num_steps,
        grad_accum=accumulator,
        start_step=step,
    ):
        del epoch
        batch = move_to_device(batch, device)
        if opt.num_workers == 0:
            validate_batch_frame_order(batch, train_provider.last_pair, phase="train")
        # Everything inside a window shares the same curriculum / LR step: the
        # number of optimizer steps completed before this window.
        step_for_schedule = completed_step - 1 if do_step else completed_step
        raw_model.set_step_context(step_for_schedule, "train")
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(batch)
        metrics = output["metrics"]
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite joint loss at step {completed_step}: "
                + " ".join(
                    f"{k}={float(v):.4f}"
                    for k, v in metrics.items()
                    if torch.is_tensor(v) and v.ndim == 0
                )
            )
        (loss / accumulator).backward()
        if not do_step:
            continue

        lr = lr_at(step_for_schedule, args, opt.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step = completed_step

        if step % args.log_freq == 0 or step == 1:
            elapsed = time.time() - start_time
            start_time = time.time()
            log(format_metrics(metrics, step, lr, elapsed) + f" grad_norm {float(grad_norm):.3f}")
            if is_main:
                history.append(
                    {
                        "step": step,
                        **{
                            key: float(value)
                            for key, value in metrics.items()
                            if torch.is_tensor(value) and value.ndim == 0
                        },
                        "grad_norm": float(grad_norm),
                        "lr": lr,
                    }
                )
                per_layer = metrics.get("spatial/anchor_update_norm_per_layer")
                if torch.is_tensor(per_layer):
                    history[-1]["anchor_update_norm_per_layer"] = [
                        float(x) for x in per_layer.detach().cpu()
                    ]
        if (
            gradient_diagnostic_freq > 0
            and step % gradient_diagnostic_freq == 0
            and is_main
        ):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                diagnostic = shared_gradient_diagnostic(
                    raw_model, batch, step=step_for_schedule, phase="train"
                )
            reachability = diagnostic.pop("grad_spatial_grounding")
            diagnostic["grad_spatial_grounding_reached"] = {
                "anchor_pre": bool(
                    reachability.get("spatial_decoder.anchor_pre", {}).get(
                        "received_understanding_grad", False
                    )
                ),
                "radius_pre": bool(
                    reachability.get("spatial_decoder.radius_pre", {}).get(
                        "received_understanding_grad", False
                    )
                ),
                "refine_heads": bool(
                    reachability.get("spatial_decoder.refine_heads.0.weight", {}).get(
                        "received_understanding_grad", False
                    )
                ),
            }
            log(
                "[gradient] "
                f"step {step} recon_norm {diagnostic['grad_recon_norm']:.4g} "
                f"und_raw {diagnostic['grad_understanding_raw_norm']:.4g} "
                f"und_eff {diagnostic['grad_understanding_effective_norm']:.4g} "
                f"(lambda_u {diagnostic['lambda_understanding']:.4g}, "
                f"ratio {diagnostic['grad_understanding_to_recon_ratio']:.4g}) "
                f"cosine {diagnostic['grad_recon_understanding_cosine']:.4f} "
                f"reached {diagnostic['grad_spatial_grounding_reached']}"
            )
            if history:
                history[-1]["gradient_diagnostic"] = diagnostic
        if val_provider is not None and args.val_freq > 0 and step % args.val_freq == 0:
            validation = run_validation(raw_model, val_provider, opt, args, device, log, step)
            if is_main and history:
                history[-1]["validation"] = validation
        if step % args.ckpt_freq == 0 or step == args.num_steps:
            if is_main:
                save_checkpoint(
                    raw_model,
                    optimizer,
                    step,
                    workspace,
                    opt,
                    args,
                    extra={"loss": float(loss.detach())},
                )
                log(f"[checkpoint] saved step {step} to {workspace}/checkpoints/step_{step:08d}")
            if distributed:
                dist.barrier()
        if distributed and step == 1:
            # Verify that the sampled records actually differ per rank.  Compare
            # (scene, frame ids): different ScanNet scenes can legitimately share
            # the same frame numbering, so frame ids alone would false-positive.
            scene_names = batch["scene_name"]
            if isinstance(scene_names, (list, tuple)):
                scene_name = scene_names[0]
            else:
                scene_name = str(scene_names)
            local = {
                "rank": rank,
                "scene_name": str(scene_name),
                "frame_ids": batch["frame_ids"][0].detach().cpu().tolist(),
            }
            gathered: list = [None] * world_size
            dist.all_gather_object(gathered, local)
            if is_main:
                order = assert_distinct_records(gathered)
                log(f"[ddp] per-rank (scene, frame ids) at step 1: {order}")

    if is_main:
        with open(workspace / "training_log.json", "w", encoding="utf-8") as handle:
            json.dump({"history": history, "args": vars(args), "options": opt.__dict__}, handle, indent=2, default=str)
        status_dir = workspace / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / "COMPLETE").write_text(f"completed {step} steps\n", encoding="utf-8")
        print(f"[done] finished {step} steps in {workspace}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
