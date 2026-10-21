#!/usr/bin/env python3
"""Shared production runtime for instance_state_v1: grouping, LR, one step, RNG, checkpoints.

No network and no loss code lives here; the smoke, the paired arms and the full
phase all drive training through ``train_one_step`` so the tested path and the
production path cannot drift apart.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

BACKBONE_PEAK_LR = 1e-5
STATE_PEAK_LR = 1e-4
GRAD_CLIP = 1.0


def peak_lr(name: str) -> float:
    return STATE_PEAK_LR if name.startswith("instance_state") else BACKBONE_PEAK_LR


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Registered schedule: linear warmup then cosine to 2 % of the peak."""
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_optimizer(model):
    """The single grouping implementation; the report is derived FROM the groups."""
    backbone_decay, backbone_nodecay, state_decay, state_nodecay = [], [], [], []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        is_state = name.startswith("instance_state.")
        # matrix weights decay; bias / norm / query_init / _no_weight_decay do not
        no_decay = (param.dim() == 1) or name.endswith("query_init") \
            or bool(getattr(param, "_no_weight_decay", False))
        if is_state:
            (state_decay if not no_decay else state_nodecay).append(param)
        else:
            (backbone_decay if not no_decay else backbone_nodecay).append(param)
    spec = [("backbone_decay", backbone_decay, 0.05),
            ("backbone_nodecay", backbone_nodecay, 0.0),
            ("instance_state_decay", state_decay, 0.05),
            ("instance_state_nodecay", state_nodecay, 0.0)]
    groups = [{"params": params, "weight_decay": wd, "name": name,
               "lr": peak_lr(name)} for name, params, wd in spec]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    # build the report from the ACTUAL param_groups, never from a second classifier
    name_of = {id(p): n for n, p in model.named_parameters()}
    rows, covered = [], set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            rows.append({"group": group["name"], "name": name_of.get(id(param), "?"),
                         "shape": list(param.shape), "numel": int(param.numel()),
                         "weight_decay": group["weight_decay"], "peak_lr": group["lr"]})
            covered.add(id(param))
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    report = {
        "rows": rows, "unique_params": len(covered),
        "trainable_params": len(trainable),
        "every_trainable_exactly_once": covered == trainable,
        "groups": [{"name": g["name"], "params": len(g["params"]),
                    "weight_decay": g["weight_decay"], "peak_lr": g["lr"],
                    "numel": int(sum(p.numel() for p in g["params"]))}
                   for g in optimizer.param_groups],
    }
    if not report["every_trainable_exactly_once"]:
        raise RuntimeError("some requires_grad parameters are not covered exactly once")
    return optimizer, report


def set_lrs(optimizer, step: int, total_steps: int, warmup: int) -> dict:
    """Set each group's LR from the registered schedule *before* the update."""
    actual = {}
    for group in optimizer.param_groups:
        lr = lr_at(step, peak_lr(group["name"]), warmup, total_steps)
        group["lr"] = lr
        actual[group["name"]] = lr
    return actual


def capture_rng() -> dict:
    return {"python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_one_step(model, optimizer, batch, step: int, arm: str,
                   total_steps: int, warmup: int) -> dict:
    """One production optimizer step.  ``arm`` is 'C' or 'E'."""
    model.train()
    model.understanding_step = int(step)
    lrs = set_lrs(optimizer, step, total_steps, warmup)
    optimizer.zero_grad(set_to_none=True)
    output, metrics = model.step_loss(batch, step=step, coupled=(arm == "E"))
    for key in ("loss", "loss_recon", "loss_understanding"):
        if not bool(torch.isfinite(metrics[key])):
            raise RuntimeError(f"non-finite {key} at step {step}: {float(metrics[key])}")
    metrics["loss"].backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                                error_if_nonfinite=True))
    optimizer.step()
    return {"step": step, "arm": arm, "lr": lrs, "grad_norm": norm,
            "loss": float(metrics["loss"]), "loss_recon": float(metrics["loss_recon"]),
            "loss_understanding": float(metrics["loss_understanding"]),
            "loss_thing": float(metrics["loss_thing"]), "loss_stuff": float(metrics["loss_stuff"]),
            "loss_sem": float(metrics["loss_sem"]), "loss_id": float(metrics["loss_id"]),
            "rseg": float(metrics["rseg"]),
            "n_gt_thing": metrics.get("n_gt_thing"),
            "beta": float(output["prediction"]["beta"])}


def checkpoint_payload(model, optimizer, step: int, arm: str, plan_sha: str,
                       total_steps: int, warmup: int, config: dict) -> dict:
    def cpu_state(state: dict) -> dict:
        out = dict(state)
        out["state"] = {
            k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                for kk, vv in v.items()}
            for k, v in state["state"].items()}
        return out

    return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": cpu_state(optimizer.state_dict()), "step": int(step), "arm": arm,
            "plan_sha256": plan_sha, "plan_position": int(step),
            "scheduler": {"total_steps": int(total_steps), "warmup": int(warmup),
                          "backbone_peak_lr": BACKBONE_PEAK_LR,
                          "state_peak_lr": STATE_PEAK_LR, "floor_factor": 0.02},
            "config": config, "rng": capture_rng()}


def save_checkpoint_atomic(payload: dict, target: Path) -> Path:
    """Write to a temporary directory, read it back, then rename into place."""
    import shutil
    tmp = target.parent / f".inprogress_{target.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(payload, tmp / "train_state.pt")
    again = torch.load(tmp / "train_state.pt", map_location="cpu", weights_only=False)
    if int(again["step"]) != int(payload["step"]):
        raise RuntimeError("checkpoint read-back mismatch")
    (tmp / "COMPLETE").write_text(f"{payload['arm']} step {payload['step']}\n",
                                  encoding="utf-8")
    if target.exists():
        return target
    tmp.rename(target)
    return target


def verify_state_dict(model, payload: dict) -> dict:
    """Compare a checkpoint payload against a model without loading it."""
    current = model.state_dict()
    return {k: float((current[k].detach().cpu().float()
                      - payload["model"][k].float()).abs().max()) for k in current}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["build_optimizer", "set_lrs", "lr_at", "train_one_step", "capture_rng",
           "restore_rng", "checkpoint_payload", "save_checkpoint_atomic",
           "verify_state_dict", "peak_lr", "read_json",
           "BACKBONE_PEAK_LR", "STATE_PEAK_LR", "GRAD_CLIP"]
