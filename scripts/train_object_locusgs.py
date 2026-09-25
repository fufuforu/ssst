#!/usr/bin/env python3
"""Strictly paired A/B training for the object-aware LocusGS development run.

* one shared, pre-registered 6000-step batch plan (`scripts/gen_object_plan.py`)
  -> both arms read the *same* scene / context / novel window at every step;
* one shared, model-only source checkpoint (step 2000 of the 32/8 LocusGS run)
  -> both arms build *fresh* AdamW optimizers and a fresh experiment-local
  schedule (steps 0..6000, 200-step linear warm-up, cosine to 2 % of peak);
* one difference: `--arm b` enables the predicted-instance-relation token update
  between decoder layers 10 and 11; `--arm a` keeps the module instantiated
  with identical initial parameters but never applies it and never optimizes it.

Existing LocusGS parameters use lr 1e-5, the new attribute/relation parameters
use lr 1e-4; the decay/no-decay split follows the original rule (any >1-d
tensor without `_no_weight_decay` decays).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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
from scripts.object_locusgs_eval import (  # noqa: E402
    build_val_entries,
    evaluate_entry,
    summarise,
)

NEW_PARAM_PREFIXES = ("attributes.", "relation.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state(state: dict, keys=None) -> str:
    digest = hashlib.sha256()
    names = sorted(state.keys()) if keys is None else sorted(keys)
    for name in names:
        tensor = state[name].detach().cpu().contiguous().to(torch.float32)
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(move(x, device) for x in v)
    return v


def is_new_parameter(name: str) -> bool:
    return name.startswith(NEW_PARAM_PREFIXES)


def lr_at(local_step: int, peak: float, warmup: int, total: int) -> float:
    """Experiment-local schedule: linear warm-up, then cosine to 2 % of peak."""
    if warmup > 0 and local_step < warmup:
        return peak * float(local_step + 1) / float(warmup)
    progress = min(1.0, max(0.0, (local_step - warmup) / max(1, total - warmup)))
    return peak * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _cpu_state_dict(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        if key == "state":
            out[key] = {
                k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                for k, v in value.items()
            }
        elif torch.is_tensor(value):
            out[key] = value.detach().cpu()
        else:
            out[key] = value
    return out


def ckpt_steps(out_dir: Path) -> list[int]:
    steps = []
    for path in out_dir.glob("ckpt_step*"):
        if path.is_dir() and (path / "COMPLETE").is_file():
            try:
                steps.append(int(path.name.replace("ckpt_step", "")))
            except ValueError:
                continue
    return sorted(steps)


def save_checkpoint(out_dir, step, model, optimizer, meta, state, *, keep_steps):
    final = out_dir / f"ckpt_step{step}"
    if final.is_dir() and (final / "COMPLETE").is_file():
        return final
    tmp = out_dir / f".inprogress_step{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": step}, tmp / "model.pt")
    torch.save(
        {
            "step": step,
            "optimizer": _cpu_state_dict(optimizer.state_dict()),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "meta": meta,
            **state,
        },
        tmp / "train_state.pt",
    )
    (tmp / "COMPLETE").write_text(f"step {step} arm {meta.get('arm')}\n", encoding="utf-8")
    os.rename(tmp, final)
    for old in ckpt_steps(out_dir):
        if old not in set(keep_steps) and old != step:
            shutil.rmtree(out_dir / f"ckpt_step{old}")
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_object_locusgs_ab")
    parser.add_argument("--source-ckpt", default="workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step2000")
    parser.add_argument("--source-sha256",
                        default="f0e791b8bb9d49deeba160f0a5fcf75594e4a0638d79ed79b9021ccd6da31c6d")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--lr-existing", type=float, default=1e-5)
    parser.add_argument("--lr-new", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[0, 2000, 6000])
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--ramp-override", type=int, default=None,
                        help="smoke only: force object_loss_ramp_steps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke only: stop early (no long run is started implicitly)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--deterministic", action="store_true",
                        help="enable torch/CUDA deterministic kernels (warn_only for gsplat)")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(
        f"[ab] deterministic_algorithms={torch.are_deterministic_algorithms_enabled()} "
        f"warn_only={torch.is_deterministic_algorithms_warn_only_enabled()} "
        f"cudnn_deterministic={torch.backends.cudnn.deterministic} "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
        flush=True,
    )
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = sha256_file(plan_path)
    if list(plan["train_scenes"]) != list(split["train_scenes"]):
        raise SystemExit("plan train scenes differ from the split")
    if list(plan["val_scenes"]) != list(split["val_scenes"]):
        raise SystemExit("plan validation scenes differ from the split")
    if int(plan["steps"]) != int(args.steps):
        raise SystemExit(f"plan has {plan['steps']} steps, --steps is {args.steps}")
    entries = plan["entries"]
    if len(entries) != args.steps:
        raise SystemExit("plan entry count does not match --steps")
    validation = set(split["val_scenes"])
    leak = sorted({entry["scene"] for entry in entries} & validation)
    if leak:
        raise SystemExit(f"validation scenes leaked into the plan: {leak}")

    source_dir = Path(args.source_ckpt)
    source_hash = sha256_file(source_dir / "model.pt")
    if args.source_sha256 and source_hash != args.source_sha256:
        raise SystemExit(f"source checkpoint hash mismatch: {source_hash}")
    source = torch.load(source_dir / "model.pt", map_location="cpu", weights_only=False)
    if not (source_dir / "COMPLETE").is_file():
        raise SystemExit(f"source checkpoint {source_dir} is not marked COMPLETE")
    if "optimizer" in source or "train_state" in source:
        raise SystemExit("source checkpoint is not model-only; refusing to pretend to restore it")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    opt = config_defaults[args.preset].evolve(
        seed=int(args.seed),
        object_arm=args.arm,
        batch_size=1,
        num_workers=0,
        num_input_views=2,
        num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        workspace=str(out_dir),
        experiment_name=f"siu3r_object_locusgs_{args.arm}_v1",
    )
    if args.ramp_override is not None:
        opt = opt.evolve(object_loss_ramp_steps=int(args.ramp_override))
    model = model_registry[opt.model_type](opt).to(device)
    model.freeze_object_queries()
    model.train()

    missing, unexpected = model.load_state_dict(source["model"], strict=False)
    if unexpected:
        raise SystemExit(f"source checkpoint has unexpected keys: {unexpected[:5]}")
    if sorted(missing) != sorted(k for k in model.state_dict() if is_new_parameter(k)):
        raise SystemExit(f"unexpectedly missing keys: {sorted(missing)}")

    state = model.state_dict()
    existing_keys = [k for k in state if not is_new_parameter(k)]
    new_keys = [k for k in state if is_new_parameter(k)]
    weight_hashes = {
        "existing": sha256_state(state, existing_keys),
        "new": sha256_state(state, new_keys),
        "source_ckpt_sha256": source_hash,
        "source_step": int(source["step"]),
    }

    decay, nodecay, new_decay, new_nodecay = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = (decay, nodecay) if not is_new_parameter(name) else (new_decay, new_nodecay)
        if parameter.dim() != 1 and not getattr(parameter, "_no_weight_decay", False):
            group[0].append(parameter)
        else:
            group[1].append(parameter)
    if args.arm == "a":  # the disabled module is never optimized
        for collection in (decay, nodecay, new_decay, new_nodecay):
            collection[:] = [
                p for p in collection
                if not any(p is q for q in model.relation.parameters())
            ]
    groups = [
        {"params": decay, "lr": args.lr_existing, "peak_lr": args.lr_existing,
         "weight_decay": args.weight_decay, "name": "existing_decay"},
        {"params": nodecay, "lr": args.lr_existing, "peak_lr": args.lr_existing,
         "weight_decay": 0.0, "name": "existing_nodecay"},
        {"params": new_decay, "lr": args.lr_new, "peak_lr": args.lr_new,
         "weight_decay": args.weight_decay, "name": "new_decay"},
        {"params": new_nodecay, "lr": args.lr_new, "peak_lr": args.lr_new,
         "weight_decay": 0.0, "name": "new_nodecay"},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, lr=args.lr_existing, betas=(0.9, 0.95))

    print(f"[ab] arm={args.arm} model={opt.model_type} out={out_dir}", flush=True)
    print(f"[ab] source {source_dir} sha256={source_hash} step={source['step']}", flush=True)
    print(f"[ab] plan {plan_path} sha256={plan_sha} steps={len(entries)}", flush=True)
    print(f"[ab] initial weight hash existing={weight_hashes['existing'][:16]} "
          f"new={weight_hashes['new'][:16]}", flush=True)
    for group in groups:
        print(f"[ab] group {group['name']}: {sum(p.numel() for p in group['params']):,} params "
              f"lr {group['peak_lr']:.1e} wd {group['weight_decay']}", flush=True)
    print(f"[ab] relation update enabled={model.use_relation_update} "
          f"layer={opt.object_relation_layer}", flush=True)

    provider = SIU3RProcessedProvider(
        opt, root=split["train_root"], subset=split["train_scenes"], training=True, rank=0
    )
    scene_index = {path.name: idx for idx, path in enumerate(provider.dataset.sample_list)}
    val_entries = [] if args.no_eval else build_val_entries(opt, split, device)
    for entry in val_entries:
        print(f"[ab] val {entry['scene']} ctx={entry['context']} novel={entry['novel']} "
              f"scale={entry['scale']:.4f}", flush=True)

    manifest = {
        "experiment": "object_aware_locusgs_ab_v1",
        "arm": args.arm,
        "scope": "32 train / 8 unseen development split; NOT an SIU3R official metric",
        "note": "model uses GT camera poses to build rays; SIU3R is an unposed setting",
        "source_checkpoint": {"path": str(source_dir), "sha256": source_hash,
                              "step": int(source["step"]), "kind": "model-only"},
        "fresh_optimizer": True,
        "optimizer_groups": [
            {"name": group["name"], "params": sum(p.numel() for p in group["params"]),
             "peak_lr": group["peak_lr"], "weight_decay": group["weight_decay"]}
            for group in groups
        ],
        "schedule": {"warmup_steps": args.warmup, "total_steps": args.steps,
                     "final_lr_fraction": 0.02, "betas": [0.9, 0.95],
                     "grad_clip": args.grad_clip},
        "loss": {"sem_weight": opt.object_sem_loss_weight,
                 "inst_weight": opt.object_inst_loss_weight,
                 "ramp_steps": opt.object_loss_ramp_steps,
                 "min_alpha": opt.object_min_alpha,
                 "instance_pixels_per_view": opt.object_instance_pixels,
                 "instance_budget": opt.object_instance_budget},
        "relation_module": {"enabled": bool(model.use_relation_update),
                            "layer": int(opt.object_relation_layer),
                            "neighbours": int(opt.object_relation_neighbours),
                            "temperature": float(opt.object_relation_temperature)},
        "plan": {"path": str(plan_path), "sha256": plan_sha, "steps": len(entries)},
        "split": {"path": args.split, "sha256": sha256_file(Path(args.split)),
                  "train_scenes": len(split["train_scenes"]), "val_scenes": len(split["val_scenes"])},
        "val_windows": {entry["scene"]: {"context": entry["context"], "novel": entry["novel"]}
                        for entry in val_entries},
        "initial_weight_hashes": weight_hashes,
        "git_commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "config": {key: value for key, value in vars(opt).items()
                   if isinstance(value, (int, float, str, bool, tuple, type(None)))},
        "args": vars(args),
    }
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    start_step = 1
    history: list[dict] = []
    checkpoint_meta = {
        "arm": args.arm,
        "plan_sha256": plan_sha,
        "source_ckpt_sha256": source_hash,
        "source_step": int(source["step"]),
        "fresh_optimizer": True,
    }
    if args.resume:
        payload = torch.load(Path(args.resume) / "train_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(
            torch.load(Path(args.resume) / "model.pt", map_location="cpu", weights_only=False)["model"],
            strict=True,
        )
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        start_step = int(payload["step"]) + 1
        print(f"[ab] resumed at step {payload['step']} -> {start_step}", flush=True)

    total_steps = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"
    low_alpha_streak = 0
    collapse_evals = 0
    started = time.time()

    def run_eval(step: int) -> dict:
        model.eval()
        rows = [evaluate_entry(model, val_entry, opt) for val_entry in val_entries]
        model.train()
        summary = summarise(rows)
        print(f"[ab] VAL step {step}: ctx {summary['ctx_psnr']:.2f} novel {summary['novel_psnr']:.2f} "
              f"(grey {summary['novel_grey']:.2f}) ssim {summary['ctx_ssim']:.3f}/"
              f"{summary['novel_ssim']:.3f} sem-mIoU {summary['sem_miou']:.3f} "
              f"AP50 {summary['novel_ap50']:.3f} TP/FP/FN {summary['novel_tp']}/"
              f"{summary['novel_fp']}/{summary['novel_fn']}", flush=True)
        serialisable = [
            {k: v for k, v in row.items() if not k.startswith("_")} for row in rows
        ]
        with (out_dir / "val_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "summary": summary, "rows": serialisable}) + "\n")
        (out_dir / "history.json").write_text(
            json.dumps({"manifest": manifest, "train": history}, indent=1), encoding="utf-8"
        )
        return summary

    if 0 in args.save_steps and start_step == 1:
        model.eval()
        if not args.no_eval:
            run_eval(0)
        else:
            model.train()
        save_checkpoint(out_dir, 0, model, optimizer, checkpoint_meta, {"plan_step": 0}, keep_steps=args.save_steps)
        model.train()
        print("[ab] checkpoint ckpt_step0 (initialisation, fresh optimizer) written", flush=True)

    for step in range(start_step, total_steps + 1):
        entry = entries[step - 1]
        provider.pin_pair(
            scene_id=entry["scene"],
            context_frame_ids=entry["context"],
            novel_frame_ids=entry["novel"],
            pair_iou=entry["pair_iou"],
        )
        try:
            batch = move(default_collate([provider[scene_index[entry["scene"]]]]), device)
        except Exception as error:  # noqa: BLE001 - the plan is pre-registered
            raise SystemExit(
                f"step {step}: pre-registered window {entry} failed to decode ({error}); "
                f"the plan must not be edited mid-run"
            ) from error
        frames = [int(x) for x in batch["frame_ids"][0].tolist()]
        if frames != entry["context"] + entry["novel"]:
            raise SystemExit(f"step {step}: batch frames {frames} != plan {entry}")
        if "semantic_label_all" not in batch or "instance_label_all" not in batch:
            raise SystemExit("provider did not return semantic/instance labels")

        optimizer.zero_grad(set_to_none=True)
        _, metrics = model.step_loss(batch, step=step - 1, phase="train")
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            print("[ab] non-finite metric dump: " + json.dumps({
                key: (float(value) if torch.is_tensor(value) and value.numel() == 1 else str(value))
                for key, value in metrics.items()
            }), flush=True)
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta, {"plan_step": step},
                            keep_steps=[step])
            raise SystemExit(f"non-finite loss at step {step}; stopped with evidence")
        loss.backward()

        grad_by_group = {group["name"]: 0.0 for group in groups}
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad_by_group[group["name"]] += float(parameter.grad.detach().float().pow(2).sum())
        grad_by_group = {key: math.sqrt(value) for key, value in grad_by_group.items()}
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(grad_norm):
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta, {"plan_step": step},
                            keep_steps=[step])
            raise SystemExit(f"non-finite gradient at step {step}; stopped with evidence")
        if args.arm == "a":
            strays = [name for name, p in model.relation.named_parameters() if p.grad is not None]
            if strays:
                raise SystemExit(f"arm A optimized the disabled module: {strays}")

        local_step = step - 1
        for group in optimizer.param_groups:
            group["lr"] = lr_at(local_step, float(group["peak_lr"]), args.warmup, args.steps)
        snapshot = None
        if step == 1 or step % args.log_every == 0 or step in args.save_steps:
            snapshot = {
                group["name"]: [p.detach().clone() for p in group["params"]] for group in groups
            }
        optimizer.step()

        alpha_mean = float(metrics["alpha_mean"])
        low_alpha_streak = low_alpha_streak + 1 if alpha_mean < 0.005 else 0
        if low_alpha_streak >= 200:
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                            {"plan_step": step, "reason": "alpha_collapse"}, keep_steps=[step])
            raise SystemExit(f"alpha collapsed for {low_alpha_streak} steps at step {step}; stopped")

        record = {
            "step": step,
            "scene": entry["scene"],
            "context": entry["context"],
            "novel": entry["novel"],
            "loss": float(loss),
            "recon_loss": float(metrics["recon_loss"]),
            "loss_sem": float(metrics["loss_sem"]),
            "loss_inst": float(metrics["loss_inst"]),
            "ramp": float(metrics["ramp"]),
            "psnr": float(metrics["psnr"]),
            "grad_norm": grad_norm,
            "grad_by_group": grad_by_group,
            "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "alpha_mean": alpha_mean,
            "alpha_nonzero_fraction": float(metrics["alpha_nonzero_fraction"]),
            "radius_mean": float(metrics["radius_mean"]),
            "radius_min": float(metrics["radius_min"]),
            "radius_max": float(metrics["radius_max"]),
            "anchor_max": float(metrics["anchor_max"]),
            "sem_coverage": float(metrics["sem_coverage"]),
            "sem_supervised_pixels": float(metrics["sem_supervised_pixels"]),
            "instances_present": float(metrics["instances_present"]),
            "instances_used": float(metrics["instances_used"]),
            "instance_pixels": float(metrics["instance_pixels"]),
            "instance_push": float(metrics["instance_push"]),
            "gate_abs_tanh": float(metrics["gate_abs_tanh"]),
            "token_update_norm": (
                float(metrics["token_update_norm"]) if "token_update_norm" in metrics else None
            ),
            "attribute_alpha_gap": float(metrics["attribute_alpha_gap"]),
        }
        if snapshot is not None:
            record["update_by_group"] = {}
            for group in groups:
                total = 0.0
                for before, parameter in zip(snapshot[group["name"]], group["params"]):
                    total += float((parameter.detach().float() - before.float()).pow(2).sum())
                record["update_by_group"][group["name"]] = math.sqrt(total)
            del snapshot
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if step % args.log_every == 0 or step == 1 or step == total_steps:
            elapsed = (time.time() - started) / max(1, step - start_step + 1)
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
            print(
                f"[ab] step {step:>5} arm={args.arm} scene={entry['scene']} "
                f"loss {record['loss']:.4f} (recon {record['recon_loss']:.4f} "
                f"sem {record['loss_sem']:.3f} inst {record['loss_inst']:.3f} ramp {record['ramp']:.2f}) "
                f"psnr {record['psnr']:.2f} grad {grad_norm:.2f} "
                f"| {elapsed:.2f}s/step peak {peak:.1f}G "
                f"eta {(total_steps - step) * elapsed / 3600:.1f}h",
                flush=True,
            )
            if record.get("update_by_group"):
                print(f"[ab]   updates {record['update_by_group']} "
                      f"lr {record['lr']} alpha {record['alpha_mean']:.3f} "
                      f"r {record['radius_min']:.4f}-{record['radius_max']:.4f} "
                      f"anchor {record['anchor_max']:.3f} gate {record['gate_abs_tanh']:.2e}", flush=True)

        if not args.no_eval and (step % args.eval_every == 0 or step in args.save_steps):
            summary = run_eval(step)
            collapse = summary["novel_psnr"] < summary["novel_grey"] + 0.5
            collapse_evals = collapse_evals + 1 if collapse else 0
            if collapse_evals >= 2:
                save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                                {"plan_step": step, "reason": "reconstruction_collapse"},
                                keep_steps=[step])
                raise SystemExit(
                    f"novel PSNR stayed at the grey-image level for {collapse_evals} evaluations "
                    f"at step {step}; stopped with evidence"
                )

        if step in args.save_steps:
            model.eval()
            save_checkpoint(
                out_dir, step, model, optimizer, checkpoint_meta,
                {"plan_sha256": plan_sha, "plan_step": step, "arm": args.arm},
                keep_steps=[int(x) for x in args.save_steps],
            )
            model.train()
            print(f"[ab] checkpoint ckpt_step{step} written", flush=True)

    (out_dir / "history.json").write_text(
        json.dumps({"manifest": manifest, "train": history}, indent=1), encoding="utf-8"
    )
    final_hash = sha256_state(model.state_dict())
    (out_dir / "final_state_hash.txt").write_text(final_hash + "\n", encoding="utf-8")
    print(f"[ab] done: {len(history)} steps in {time.time() - started:.0f}s "
          f"final_state_sha256={final_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
