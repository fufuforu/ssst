"""Atomic checkpoint, RNG recovery, and CPU-only progress/log helpers."""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def capture_rng_state(sample_rng, provider):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "sample_rng": sample_rng.getstate(),
        "provider_pair_rng": provider.pair_rng.getstate(),
    }
    provider_rng = getattr(provider, "rng", None)
    if provider_rng is not None and hasattr(provider_rng, "bit_generator"):
        state["provider_numpy_rng"] = provider_rng.bit_generator.state
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, sample_rng, provider):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    sample_rng.setstate(state["sample_rng"])
    provider.pair_rng.setstate(state["provider_pair_rng"])
    if "provider_numpy_rng" in state:
        provider_rng = getattr(provider, "rng", None)
        if provider_rng is None or not hasattr(provider_rng, "bit_generator"):
            raise RuntimeError("checkpoint contains provider NumPy RNG but provider does not expose it")
        provider_rng.bit_generator.state = state["provider_numpy_rng"]
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_checkpoint(head, optimizer, completed_updates, rng_state, config, visual_checkpoint,
                     visual_sha256, clip_provenance, code_commit, scene_names,
                     last_loss=None, elapsed_seconds=0.0):
    return {
        "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "completed_updates": int(completed_updates),
        "last_loss": None if last_loss is None else float(last_loss),
        "elapsed_seconds": float(elapsed_seconds),
        "rng_state": rng_state,
        "config": dict(config),
        "visual_checkpoint": str(visual_checkpoint),
        "visual_checkpoint_sha256": visual_sha256,
        "visual_exposure": 50064,
        "clip_revision": clip_provenance["revision"],
        "clip_provenance": clip_provenance,
        "code_commit": str(code_commit),
        "scene_names": list(scene_names),
    }


def restore_checkpoint(path, head, optimizer, sample_rng, provider, *, expected_config,
                       expected_visual_sha256, expected_clip_revision, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != expected_config:
        raise RuntimeError("resume checkpoint training config does not match this run")
    if checkpoint.get("visual_checkpoint_sha256") != expected_visual_sha256:
        raise RuntimeError("resume checkpoint visual SHA does not match")
    if checkpoint.get("clip_revision") != expected_clip_revision:
        raise RuntimeError("resume checkpoint CLIP revision does not match")
    head.load_state_dict(checkpoint["head"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value): state[key] = value.to(device)
    restore_rng_state(checkpoint["rng_state"], sample_rng, provider)
    return int(checkpoint["completed_updates"]), checkpoint


def write_json_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_metric(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def trim_metrics_after_update(path, completed_updates):
    """Drop uncheckpointed tail rows when explicitly resuming a prior checkpoint."""
    path = Path(path)
    if not path.exists(): return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        record = json.loads(line)
        if int(record["update"]) <= int(completed_updates): kept.append(record)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in kept: stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def progress_payload(completed_updates, total_updates, last_loss, latest_checkpoint,
                     started_at, current_elapsed=None, status="RUNNING", failure=None):
    elapsed = float(time.monotonic() - started_at if current_elapsed is None else current_elapsed)
    speed = elapsed / completed_updates if completed_updates else 0.0
    remaining = max(0, total_updates - completed_updates)
    eta = speed * remaining if completed_updates else None
    result = {
        "status": status,
        "completed_updates": int(completed_updates),
        "total_updates": int(total_updates),
        "recent_loss": None if last_loss is None else float(last_loss),
        "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint),
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": None if eta is None else float(eta),
        "estimate_basis": "elapsed training seconds / completed updates; excludes evaluation",
    }
    if failure is not None: result["failure"] = failure
    return result
