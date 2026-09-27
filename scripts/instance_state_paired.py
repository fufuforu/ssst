#!/usr/bin/env python3
"""Paired C/E training driver for LOCUSGS_INSTANCE_STATE_V1 (spec sections 9-12).

Only the loop / optimizer / checkpoint / schedule plumbing lives here; the
network, the losses and every metric definition stay in their own modules.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.run_instance_state_v1 import (  # noqa: E402
    ARM_C, ARM_E, PRESET_C, PRESET_E, PRETRAINED, SEED, build_options,
    build_optimizer, sha256_file, transfer_reconstruction_weights, write_json,
)

PAIRED_STEPS = 2000
WARMUP = 100
FULL_STEPS = 50000
FULL_WARMUP = 2000
EVAL_STEPS = (0, 200, 500, 1000, 2000)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_arm_model(arm: str, device):
    preset = PRESET_C if arm == ARM_C else PRESET_E
    opt = build_options(preset)
    model = model_registry[opt.model_type](opt)
    payload = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    transfer_reconstruction_weights(model, payload.get("model", payload), opt)
    return model.to(device), opt


def paired_batch(opt, window, device):
    provider = SIU3RProcessedProvider(opt, root="/space/mawb/SIU3R/data/scannet/train",
                                      subset=[window["scene"]], training=True, rank=0)
    want = np.array([*window["context"], *window["novel"]], dtype=np.int64)
    provider._get_indices_static = lambda idx: (want, [])      # noqa: SLF001
    batch = move(default_collate([provider[0]]), device)
    frames = [int(x) for x in batch["frame_ids"][0]]
    if frames != want.tolist():
        raise RuntimeError(f"frame order {frames} != plan {want.tolist()}")
    return batch


def run_paired(reports: Path, run_root: Path, device: str = "cuda", arm: str | None = None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads((reports / "plan_paired_2000.json").read_text(encoding="utf-8"))
    if sha256_file(reports / "plan_paired_2000.json") == "":
        raise SystemExit("plan hash unavailable")
    arms = [arm] if arm else [ARM_C, ARM_E]
    summaries = {}
    for current in arms:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model, opt = build_arm_model(current, device)
        optimizer, groups = build_optimizer(model, opt)
        write_json(reports / f"optimizer_groups_{current}.json", groups)
        model.train()
        out_dir = run_root / f"arm_{current}"
        out_dir.mkdir(parents=True, exist_ok=True)
        history = []
        started = time.time()
        for entry in plan["entries"]:
            step = int(entry["step"])
            batch = paired_batch(opt, entry, device)
            optimizer.zero_grad(set_to_none=True)
            _, metrics = model.step_loss(batch, step=step)
            metrics["loss"].backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, group["lr"] if False else
                                    (1e-4 if group["name"].startswith("instance_state")
                                     else 1e-5), WARMUP, PAIRED_STEPS)
            optimizer.step()
            if step % 100 == 0 or step == 1:
                history.append({"arm": current, "step": step,
                                "scene": entry["scene"], "context": entry["context"],
                                "novel": entry["novel"],
                                "loss": float(metrics["loss"]),
                                "loss_recon": float(metrics["loss_recon"]),
                                "loss_understanding": float(metrics["loss_understanding"]),
                                "loss_thing": float(metrics["loss_thing"]),
                                "loss_stuff": float(metrics["loss_stuff"]),
                                "loss_sem": float(metrics["loss_sem"]),
                                "loss_id": float(metrics["loss_id"]),
                                "rseg": float(metrics["rseg"]), "grad_norm": norm,
                                "n_gt_thing": metrics.get("n_gt_thing"),
                                "lr_backbone": 1e-5, "lr_state": 1e-4})
                print(f"[paired] {current} step {step} loss {float(metrics['loss']):.4f} "
                      f"recon {float(metrics['loss_recon']):.4f} "
                      f"und {float(metrics['loss_understanding']):.4f} grad {norm:.2f}",
                      flush=True)
            if step == PAIRED_STEPS:
                payload = {"model": model.state_dict(), "arm": current, "step": step,
                           "config": {"preset": PRESET_C if current == ARM_C else PRESET_E,
                                      "coupled": current == ARM_E},
                           "plan_sha256": sha256_file(reports / "plan_paired_2000.json")}
                torch.save(payload, out_dir / "endpoint_model.pt")
                (out_dir / "COMPLETE").write_text(f"endpoint {current} step {step}\n")
        (out_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(row) for row in history) + "\n", encoding="utf-8")
        summaries[current] = {"steps": len(plan["entries"]),
                              "wall_seconds": time.time() - started,
                              "final": history[-1] if history else None}
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(reports / "paired_run_summary.json", summaries)
    return 0


def run_full(reports: Path, run_root: Path, device: str = "cuda",
             until_step: int | None = None, resume: str | None = None,
             arm: str | None = None):
    """Full-data continuation of a passing arm (5000-step serial segments)."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    plan_path = reports / "plan_full_50000.json"
    if not plan_path.is_file():
        raise SystemExit(
            "plan_full_50000.json is missing: generate it with scripts/gen_object_plan.py "
            "--split <verified full_split copy> --preset <new preset> --steps 50000 "
            "--seed 42 --verify-batches 8 before the full phase")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    current = arm or ARM_E
    model, opt = build_arm_model(current, device)
    optimizer, groups = build_optimizer(model, opt)
    start_step = 1
    out_dir = run_root / f"full_{current}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        payload = torch.load(Path(resume) / "train_state.pt", map_location="cpu",
                             weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        start_step = int(payload["step"]) + 1
    stop = min(until_step or FULL_STEPS, FULL_STEPS)
    entries = plan["entries"] if isinstance(plan, dict) else plan
    model.train()
    for entry in entries:
        step = int(entry["step"])
        if step < start_step or step > stop:
            continue
        batch = paired_batch(opt, entry, device)
        optimizer.zero_grad(set_to_none=True)
        _, metrics = model.step_loss(batch, step=step)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in optimizer.param_groups:
            peak = 1e-4 if group["name"].startswith("instance_state") else 1e-5
            group["lr"] = lr_at(step, peak, FULL_WARMUP, FULL_STEPS)
        optimizer.step()
        if step % 500 == 0:
            print(f"[full] {current} step {step} loss {float(metrics['loss']):.4f}",
                  flush=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": stop, "arm": current, "plan_sha256": sha256_file(plan_path),
               "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}
    tmp = out_dir / ".inprogress"
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp / "train_state.pt")
    (tmp / "COMPLETE").write_text(f"full {current} step {stop}\n")
    final = out_dir / f"ckpt_step{stop}"
    if final.exists():
        import shutil
        shutil.rmtree(tmp)
    else:
        tmp.rename(final)
    return 0


__all__ = ["run_paired", "run_full", "lr_at"]
