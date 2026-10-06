#!/usr/bin/env python3
"""Fixed 12,000-update formal head-only run with atomic recovery checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import default_collate

from object_locus_text_refer.adapter import FULL1201_SHA256, load_full1201_frozen
from object_locus_text_refer.data import choose_context_candidate
from object_locus_text_refer.head import ObjectLocusTextReferHead, build_head_optimizer
from object_locus_text_refer.text_encoder import EXPECTED_REVISION, load_frozen_clip_text
from object_locus_text_refer.training_runtime import (
    append_metric, atomic_torch_save, build_checkpoint, capture_rng_state,
    progress_payload, restore_checkpoint, trim_metrics_after_update, write_json_atomic,
)
from scripts import object_locus_v3_set_runtime as runtime
from scripts.train_object_locus_text_refer import (
    GLOBAL_SEED, HEAD_SEED, draw_visible_context, initialize_global_seed, run_train_update,
)

TOTAL_UPDATES = 12000
SAVE_UPDATES = (0, 1000, 3000, 6000, 9000, 12000)
METRIC_INTERVAL = 20
DEFAULT_VISUAL = "/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt"
DEFAULT_DATA = "/space/mawb/SIU3R/data/scannet"
DEFAULT_OUTPUT = "/space/mawb/ssst/group_plus/object_locus_text_refer_v1_train"
DEFAULT_CHECKPOINT_DIR = "/space/mawb/ssst/workspace_group_plus/object_locus_text_refer_v1_train"
CLIP_CACHE = "/space/mawb/ssst/group_plus/object_locus_text_refer_v1/hf_cache"
CLIP_PROVENANCE = "/space/mawb/ssst/group_plus/object_locus_text_refer_v1/text_encoder_provenance.json"


def code_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def provider_rngs(provider):
    state = {"pair_rng": provider.pair_rng.getstate()}
    provider_rng = getattr(provider, "rng", None)
    if provider_rng is not None and hasattr(provider_rng, "bit_generator"):
        state["numpy_rng"] = provider_rng.bit_generator.state
    return state


def checkpoint_path(directory, update):
    return Path(directory) / f"head_update_{int(update):05d}.pt"


def save_update_checkpoint(update, directory, head, optimizer, sample_rng, provider, config,
                           visual_path, clip_provenance, commit, scene_names,
                           last_loss=None, elapsed_seconds=0.0):
    path = checkpoint_path(directory, update)
    state = build_checkpoint(head, optimizer, update,
        capture_rng_state(sample_rng, provider), config, visual_path, FULL1201_SHA256,
        clip_provenance, commit, scene_names, last_loss=last_loss,
        elapsed_seconds=elapsed_seconds)
    # Provider sampling RNG is explicit in addition to the global and local streams.
    state["provider_rng"] = provider_rngs(provider)
    atomic_torch_save(state, path)
    return path


def restore_provider_rng(provider, state):
    saved = state.get("provider_rng", {})
    if "pair_rng" in saved: provider.pair_rng.setstate(saved["pair_rng"])
    if "numpy_rng" in saved:
        rng = getattr(provider, "rng", None)
        if rng is None or not hasattr(rng, "bit_generator"):
            raise RuntimeError("resume checkpoint has provider NumPy RNG but provider does not expose it")
        rng.bit_generator.state = saved["numpy_rng"]


def write_progress(output, completed, last_loss, latest, started_at, status="RUNNING", failure=None,
                   elapsed_before=0.0):
    write_json_atomic(progress_payload(completed, TOTAL_UPDATES, last_loss, latest, started_at,
                                       current_elapsed=elapsed_before + time.monotonic() - started_at,
                                       status=status, failure=failure), Path(output) / "progress.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-checkpoint", default=DEFAULT_VISUAL)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", default=None, help="explicit prior formal checkpoint; omitted means fresh head")
    args = parser.parse_args()

    if not torch.cuda.is_available(): raise RuntimeError("formal head training requires CUDA")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("formal run requires exactly one visible RTX3090")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.resume is None:
        initialize_global_seed(GLOBAL_SEED)

    output = Path(args.output_dir); ckpt_dir = Path(args.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    progress_path = output / "progress.json"
    if args.resume is None and (any(ckpt_dir.glob("head_update_*.pt")) or metrics_path.exists()
            or progress_path.exists() or (output / "startup_confirmation.json").exists()
            or (output / "COMPLETE").exists()):
        raise FileExistsError("fresh run output/checkpoint directory is not empty; use --resume explicitly")
    if not Path(args.visual_checkpoint).is_file(): raise FileNotFoundError(args.visual_checkpoint)
    if not Path(CLIP_PROVENANCE).is_file(): raise FileNotFoundError(f"CLIP provenance missing: {CLIP_PROVENANCE}")

    model, opt, visual_exposure = load_full1201_frozen("cuda:0", args.visual_checkpoint)
    if visual_exposure != 50064: raise RuntimeError(f"unexpected visual exposure {visual_exposure}")
    tokenizer, encoder, clip_provenance = load_frozen_clip_text(
        cache_dir=CLIP_CACHE, provenance_path=CLIP_PROVENANCE, revision=EXPECTED_REVISION)
    encoder = encoder.cuda().eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("CLIP text encoder must remain fully frozen")
    head = ObjectLocusTextReferHead(HEAD_SEED).cuda().float()
    optimizer = build_head_optimizer(head)
    head_parameter_ids = {id(parameter) for parameter in head.parameters()}
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != head_parameter_ids:
        raise RuntimeError("formal optimizer must contain only text head parameters")

    root = Path(args.data_root)
    train_refs = json.loads((root / "train_refer_seg_data.json").read_text(encoding="utf-8"))
    provider = runtime.ObjectLocusV1Provider(opt, root=str(root / "train"), subset="all", training=True, rank=0)
    scene_to_index = {path.name: index for index, path in enumerate(provider.dataset.sample_list)}
    scene_names = sorted(set(train_refs) & set(scene_to_index))
    if not scene_names: raise RuntimeError("train annotations/provider scene intersection is empty")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("visual model must remain in eval mode and fully frozen")

    # The task-local sampler starts from global seed 42 after model construction.
    sample_rng = random.Random(GLOBAL_SEED) if args.resume is None else random.Random()
    config = {
        "total_updates": TOTAL_UPDATES, "save_updates": list(SAVE_UPDATES),
        "metric_interval": METRIC_INTERVAL, "batch_size": 1, "precision": "fp32",
        "global_seed": GLOBAL_SEED, "head_seed": HEAD_SEED,
        "optimizer": "AdamW", "lr": 1e-4, "weight_decay_matrix": 0.05,
        "weight_decay_bias_norm_1d": 0.0, "betas": [0.9, 0.95], "eps": 1e-8,
        "grad_clip": 1.0, "loss": {"slot_ce": 1.0, "pixel_bce": 5.0, "pixel_dice": 5.0},
        "visual_exposure": 50064, "clip_model": "openai/clip-vit-base-patch32",
        "clip_revision": EXPECTED_REVISION, "train_json": str(root / "train_refer_seg_data.json"),
        "scene_count": len(scene_names), "output_dir": str(output),
        "checkpoint_dir": str(ckpt_dir), "visual_checkpoint": str(args.visual_checkpoint),
        "visual_checkpoint_sha256": FULL1201_SHA256,
        "head_architecture": {
            "text_hidden": 256, "object_hidden": 256, "decoder_blocks": 2,
            "attention_heads": 8, "ffn_hidden": 1024, "thing_slots": 100,
            "null_slot": 100, "score_temperature": 0.07,
        },
        "text_length": 77, "text_hidden": 512,
        "sampling": {
            "max_provider_candidates_per_update": 128,
            "scene_selection": "uniform from sorted train annotation/provider intersection",
            "referent_selection": "context frame2object ∩ described ∩ valid visible thing IDs",
            "effective_valid_expression": "(sem >= 0) & (sem <= 19) & ((sem < 2) | (ins > 0))",
            "thing_expression": "sem >= 2", "null_negative_sampling": False,
        },
    }
    commit = code_commit()
    latest_checkpoint = None
    completed = 0
    last_loss = None
    elapsed_before = 0.0
    started_at = time.monotonic()

    if args.resume:
        resume_path = Path(args.resume)
        completed, restored = restore_checkpoint(resume_path, head, optimizer, sample_rng, provider,
            expected_config=config, expected_visual_sha256=FULL1201_SHA256,
            expected_clip_revision=EXPECTED_REVISION, device="cuda:0")
        restore_provider_rng(provider, restored)
        latest_checkpoint = str(resume_path)
        last_loss = restored.get("last_loss")
        elapsed_before = float(restored.get("elapsed_seconds", 0.0))
        trim_metrics_after_update(metrics_path, completed)
        # Resume RNG comes from checkpoint. Do not reseed it here.
        del restored
    else:
        latest_checkpoint = str(save_update_checkpoint(0, ckpt_dir, head, optimizer, sample_rng, provider,
            config, args.visual_checkpoint, clip_provenance, commit, scene_names))
        write_progress(output, 0, None, latest_checkpoint, started_at, elapsed_before=elapsed_before)

    try:
        for update in range(completed + 1, TOTAL_UPDATES + 1):
            active_update = update
            update_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            sample = draw_visible_context(provider, scene_to_index, train_refs, scene_names, sample_rng,
                                          attempts=128, device="cuda:0")
            row = run_train_update(model, tokenizer, encoder, head, optimizer, sample, update)
            skipped_contexts = list(sample.get("skipped_before_selection", []))
            del sample
            torch.cuda.synchronize()
            elapsed = time.monotonic() - update_started
            lr = float(optimizer.param_groups[0]["lr"])
            metric = {
                "update": int(update), "total_loss": float(row["loss"]),
                "slot_ce": float(row["parts"]["slot_ce"]),
                "pixel_bce": float(row["parts"]["mask_bce"]),
                "pixel_dice": float(row["parts"]["mask_dice"]),
                "lr": lr, "head_grad_norm_before_clip": float(row["gradient_norm_before_clip"]),
                "matched": bool(row["matched"]), "match_status": "matched" if row["matched"] else "visible_unmatched_skip_ce",
                "scene": row["scene"], "context_frame_ids": row["context_frame_ids"],
                "object_id": int(row["object_id"]), "text_index": int(row["text_index"]),
                "text": row["text"], "visual_beta_registered_states": row["visual_beta"][5::2],
                "elapsed_seconds": elapsed,
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "skipped_contexts": skipped_contexts,
            }
            if not all(torch.isfinite(torch.tensor(metric[key])) for key in ("total_loss", "slot_ce", "pixel_bce", "pixel_dice", "head_grad_norm_before_clip")):
                raise FloatingPointError(f"nonfinite scalar at update {update}")
            last_loss = metric["total_loss"]
            completed = update
            if update == 1 or update % METRIC_INTERVAL == 0 or update == TOTAL_UPDATES:
                append_metric(metric, metrics_path)
            if update in SAVE_UPDATES:
                accumulated_elapsed = elapsed_before + time.monotonic() - started_at
                latest_checkpoint = str(save_update_checkpoint(update, ckpt_dir, head, optimizer, sample_rng,
                    provider, config, args.visual_checkpoint, clip_provenance, commit, scene_names,
                    last_loss=last_loss, elapsed_seconds=accumulated_elapsed))
            write_progress(output, update, last_loss, latest_checkpoint, started_at,
                           elapsed_before=elapsed_before)
            if update == 1:
                write_json_atomic({
                    "status": "PASS", "job_id": os.environ.get("SLURM_JOB_ID"),
                    "node": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
                    "visual_checkpoint": str(args.visual_checkpoint),
                    "visual_checkpoint_sha256": FULL1201_SHA256, "visual_exposure": visual_exposure,
                    "clip_revision": clip_provenance["revision"],
                    "update_zero_checkpoint": str(checkpoint_path(ckpt_dir, 0)),
                    "completed_updates": 1, "first_loss": last_loss,
                    "first_head_grad_norm_before_clip": metric["head_grad_norm_before_clip"],
                    "first_step_finite": True, "metrics_path": str(metrics_path),
                    "progress_path": str(progress_path),
                }, output / "startup_confirmation.json")
            print(json.dumps({"update": update, "loss": last_loss,
                              "grad_norm": metric["head_grad_norm_before_clip"],
                              "matched": metric["matched"]}, flush=True))
        write_progress(output, TOTAL_UPDATES, last_loss, latest_checkpoint, started_at,
                       status="COMPLETE", elapsed_before=elapsed_before)
        marker = output / "COMPLETE"
        marker.write_text(f"completed_updates={TOTAL_UPDATES}\n", encoding="utf-8")
        with marker.open("rb") as stream: os.fsync(stream.fileno())
    except BaseException as error:
        failed_update = min(TOTAL_UPDATES, completed + 1) if "active_update" not in locals() else int(active_update)
        write_progress(output, completed, last_loss,
                       latest_checkpoint, started_at, status="FAILED",
                       failure={"update": failed_update, "exception": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc()}, elapsed_before=elapsed_before)
        raise


if __name__ == "__main__": main()
