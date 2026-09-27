#!/usr/bin/env python3
"""Paired C/E training driver for LOCUSGS_INSTANCE_STATE_V1 (closure round).

Only orchestration lives here: gating, plan validation, batch caching, the
production `train_one_step`, in-loop evaluation, atomic endpoints and the full
phase.  Network, losses and metrics stay in their own modules.
"""

from __future__ import annotations

import json
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
from scripts.instance_state_runtime import (  # noqa: E402
    build_optimizer, checkpoint_payload, save_checkpoint_atomic, train_one_step,
)
from scripts.run_instance_state_v1 import (  # noqa: E402
    ARM_C, ARM_E, PRESET_C, PRESET_E, PRETRAINED, SEED, build_options, sha256_file,
    transfer_reconstruction_weights, write_json,
)

PAIRED_STEPS = 2000
PAIRED_WARMUP = 100
FULL_STEPS = 50000
FULL_WARMUP = 2000
PAIRED_EVAL_STEPS = (0, 200, 500, 1000, 2000)
VAL_EVAL_STEPS = (0, 1000, 2000)
REGISTERED_PLAN_SHA = "32c40a72e45843a9"      # prefix of the locked paired plan


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def require_green(reports: Path, *, phase: str) -> dict:
    """Training may only start from a fully green smoke, contract and plan."""
    contract = json.loads((reports / "loss_contract.json").read_text(encoding="utf-8"))
    smoke_path = reports / "smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {}
    if not contract.get("ok"):
        raise SystemExit(f"loss contract not green: {contract.get('failed')}")
    if not smoke.get("ok"):
        raise SystemExit(f"smoke not green: {smoke.get('failed')}")
    digest = sha256_file(reports / "plan_paired_2000.json")
    if not digest.startswith(REGISTERED_PLAN_SHA):
        raise SystemExit(f"paired plan SHA {digest[:16]} != registered {REGISTERED_PLAN_SHA}")
    return {"smoke_steps": len(smoke.get("checks", [])), "plan_sha256": digest}


def validate_plan(plan: dict, windows: list[dict]) -> None:
    entries = plan["entries"]
    if len(entries) != PAIRED_STEPS:
        raise SystemExit(f"plan has {len(entries)} entries, expected {PAIRED_STEPS}")
    for index, entry in enumerate(entries):
        if int(entry["step"]) != index + 1:
            raise SystemExit(f"plan step {entry['step']} != {index + 1}")
        window = windows[index % 4]
        if entry["scene"] != window["scene"] or list(entry["context"]) != list(window["context"]) \
                or list(entry["novel"]) != list(window["novel"]):
            raise SystemExit(f"plan entry {index} does not match locked window {index % 4}")


def cache_batches(opt, windows, device):
    """Fix the four training windows once; the loop only moves them to the device."""
    cached = []
    for window in windows:
        provider = SIU3RProcessedProvider(opt, root="/space/mawb/SIU3R/data/scannet/train",
                                          subset=[window["scene"]], training=True, rank=0)
        want = [*window["context"], *window["novel"]]
        provider.pin_pair(scene_id=window["scene"], context_frame_ids=window["context"],
                          novel_frame_ids=window["novel"],
                          pair_iou=float(window.get("pair_iou") or float("nan")))
        batch = default_collate([provider[0]])
        frames = [int(x) for x in batch["frame_ids"][0]]
        if frames != want:
            raise SystemExit(f"{window['scene']}: frames {frames} != locked {want}")
        cached.append(batch)
    return cached


def build_arm_model(arm: str, device):
    preset = PRESET_C if arm == ARM_C else PRESET_E
    opt = build_options(preset)
    model = model_registry[opt.model_type](opt)
    payload = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    transfer_reconstruction_weights(model, payload.get("model", payload), opt)
    return model.to(device), opt


def run_paired(reports: Path, run_root: Path, device: str = "cuda", arm: str | None = None):
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise SystemExit("GPU phase requires CUDA; refusing to fall back to CPU")
    device = torch.device(device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    gate = require_green(reports, phase="paired")
    plan = json.loads((reports / "plan_paired_2000.json").read_text(encoding="utf-8"))
    pilot = json.loads((reports / "pilot_windows.json").read_text(encoding="utf-8"))
    monitor = json.loads((reports / "monitor_8pairs.json").read_text(encoding="utf-8"))
    windows = pilot["windows"]
    validate_plan(plan, windows)
    from scripts.eval_instance_state_v1 import evaluate_windows, aggregate
    arms = [arm] if arm else [ARM_C, ARM_E]
    results = {}
    for current in arms:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model, opt = build_arm_model(current, device)
        optimizer, groups = build_optimizer(model)
        write_json(reports / f"optimizer_groups_{current}.json", groups)
        cached = cache_batches(opt, windows, device)
        plan_entries = plan["entries"]
        out_dir = run_root / f"arm_{current}"
        out_dir.mkdir(parents=True, exist_ok=True)
        history_path = reports / f"history_{current}.jsonl"
        started = time.time()
        curves = {}
        for index, entry in enumerate(plan_entries):
            step = int(entry["step"])
            batch = move(cached[index % 4], device)
            record = train_one_step(model, optimizer, batch, step, current,
                                    PAIRED_STEPS, PAIRED_WARMUP)
            record["scene"] = entry["scene"]
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            if step % 100 == 0:
                print(f"[paired] {current} step {step} loss {record['loss']:.4f} "
                      f"recon {record['loss_recon']:.4f} und {record['loss_understanding']:.4f} "
                      f"lr_state {record['lr']['instance_state_decay']:.2e}", flush=True)
            if step in PAIRED_EVAL_STEPS and step > 0:
                rows = []
                for scope in ("context", "target"):
                    res = evaluate_windows(model, opt, windows, step, scope,
                                           reports / f"eval_paired_{current}",
                                           arm=current, device=str(device),
                                           batch_builder=_batch_from_cache(cached, scope))
                    rows.append(aggregate(res["windows"]))
                curves[step] = {"context": rows[0], "target": rows[1]}
                print(f"[paired] {current} EVAL {step} ctx mIoU "
                      f"{rows[0]['semantic_miou']:.3f} recall {rows[0]['recall50_class_aware']:.3f} "
                      f"rawR {rows[0]['raw_recall50']:.3f} pqTP {rows[0]['n_thing_tp_panoptic']} "
                      f"| novel PSNR {rows[1]['psnr']:.2f}", flush=True)
            if step in VAL_EVAL_STEPS and step > 0:
                res = evaluate_windows(model, opt, monitor["pairs"], step, "target",
                                       reports / f"eval_val8_{current}", arm=current,
                                       device=str(device), batch_builder=_monitor_builder(monitor))
                curves[f"val8_{step}"] = aggregate(res["windows"])
                print(f"[paired] {current} VAL8 {step} mIoU "
                      f"{curves[f'val8_{step}']['semantic_miou']:.3f} recall "
                      f"{curves[f'val8_{step}']['recall50_class_aware']:.3f}", flush=True)
        payload = checkpoint_payload(model, optimizer, PAIRED_STEPS, current,
                                     gate["plan_sha256"], PAIRED_STEPS, PAIRED_WARMUP,
                                     {"preset": PRESET_C if current == ARM_C else PRESET_E,
                                      "coupled": current == ARM_E})
        save_checkpoint_atomic(payload, out_dir / "endpoint")
        torch.save(payload["model"], out_dir / "endpoint_model.pt")
        results[current] = {"wall_seconds": time.time() - started,
                            "curves": curves,
                            "endpoint": str(out_dir / "endpoint_model.pt")}
        write_json(reports / f"curves_{current}.json", curves)
        del model, optimizer, cached
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(reports / "paired_run_summary.json", results)
    return 0


def _batch_from_cache(cached, scope):
    def builder(opt, window, device):
        index = next(i for i, w in enumerate(cached)
                     if w["frame_ids"][0][0].item() == window["context"][0])
        return move(cached[index], device)
    del scope
    return builder


def _monitor_builder(monitor):
    cache = {}

    def builder(opt, window, device):
        key = (window["scene"], tuple(window["context"]))
        if key not in cache:
            provider = SIU3RProcessedProvider(
                opt, root="/space/mawb/SIU3R/data/scannet/val", subset=[window["scene"]],
                training=False, val_pair_json="/space/mawb/SIU3R/data/scannet/val_pair.json",
                rank=0)
            index = next(i for i, r in enumerate(provider.dataset.val_pairs)
                         if r["scan"] == window["scene"]
                         and [int(x) for x in r["context_ids"]] == list(window["context"]))
            cache[key] = default_collate([provider[index]])
        return move(cache[key], device)
    del monitor
    return builder


def run_full(reports: Path, run_root: Path, device: str = "cuda",
             until_step: int | None = None, resume: str | None = None,
             arm: str | None = None):
    """Full-data continuation of the gate-selected arm (5000-step segments)."""
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise SystemExit("GPU phase requires CUDA")
    gate_path = reports / "gate.json"
    if not gate_path.is_file():
        raise SystemExit("gate.json is missing: the full phase needs a passing gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    selected = gate.get("selected_arm")
    if selected not in (ARM_C, ARM_E):
        raise SystemExit(f"gate.selected_arm is {selected!r}; refusing to pick a default")
    if arm and arm != selected:
        raise SystemExit(f"--arm {arm} contradicts gate.selected_arm {selected}")
    current = selected
    if until_step and until_step > 5000 and not resume:
        raise SystemExit("until_step > 5000 requires --resume of the previous segment")
    plan_path = reports / "plan_full_50000.json"
    if not plan_path.is_file():
        raise SystemExit("plan_full_50000.json missing (generate with scripts/gen_object_plan.py)")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = plan["entries"] if isinstance(plan, dict) else plan
    device = torch.device(device)
    model, opt = build_arm_model(current, device)
    optimizer, groups = build_optimizer(model)
    start = 1
    if resume:
        payload = torch.load(Path(resume) / "train_state.pt", map_location="cpu",
                             weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        from scripts.instance_state_runtime import restore_rng
        restore_rng(payload["rng"])
        start = int(payload["step"]) + 1
    stop = min(until_step or FULL_STEPS, FULL_STEPS)
    cached = {}
    for entry in entries:
        step = int(entry["step"])
        if step < start or step > stop:
            continue
        key = (entry["scene"], tuple(entry["context"]), tuple(entry["novel"]))
        if key not in cached:
            provider = SIU3RProcessedProvider(
                opt, root="/space/mawb/SIU3R/data/scannet/train", subset=[entry["scene"]],
                training=True, rank=0)
            provider.pin_pair(scene_id=entry["scene"], context_frame_ids=entry["context"],
                              novel_frame_ids=entry["novel"])
            cached = {(key): default_collate([provider[0]])}
        record = train_one_step(model, optimizer, move(cached[key], device), step, current,
                                FULL_STEPS, FULL_WARMUP)
        if step % 500 == 0:
            print(f"[full] {current} step {step} loss {record['loss']:.4f}", flush=True)
    payload = checkpoint_payload(model, optimizer, stop, current,
                                 sha256_file(plan_path), FULL_STEPS, FULL_WARMUP,
                                 {"preset": PRESET_C if current == ARM_C else PRESET_E})
    save_checkpoint_atomic(payload, run_root / f"full_{current}" / f"ckpt_step{stop}")
    return 0


__all__ = ["run_paired", "run_full"]
