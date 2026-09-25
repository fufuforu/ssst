#!/usr/bin/env python3
"""From-scratch, strictly paired G0/G1 training for the group-feedback experiment.

* both arms start from **random initialization** (no reconstruction checkpoint,
  no query head, no external weights) with the same seed, so the shared LocusGS
  parameters *and* the group parameters are bit-identical before the first
  update (verified by hash and written to `init_report.json`);
* both arms read the same pre-registered 6000-step batch plan, so step *k* sees
  the same scene/context/novel window in both arms;
* the only difference is the gated group -> token write-back between decoder
  layers 10 and 11 (G1), whose gate starts at exactly 0;
* `L = L_recon + lambda(step) * [0.05 L_instance + 0.05 L_semantic]`,
  `lambda = min(1, step/2000)`, peak lr 1e-4, 2000-step warm-up, cosine to 2 %
  over 6000 steps, AdamW with the repository's decay/no-decay split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from scripts.group_eval_v2 import (  # noqa: E402
    build_val_entries,
    evaluate_entry_v2,
    summarise_v2,
)
from scripts.train_object_locusgs import (  # noqa: E402
    lr_at,
    move,
    save_checkpoint,
    sha256_file,
    sha256_state,
)

SHARED_PARAM_PREFIXES = ("activation_head.", "anchor_decoder.", "enc_dec_backbone.",
                         "patch_embed.", "patch_plucker_embed.", "gs_tokens")


def parameter_blocks(model) -> dict[str, list[str]]:
    """Split the state dict into the blocks the report has to compare."""
    state = model.state_dict()
    groups = {
        "reconstruction": [k for k in state if k.startswith(SHARED_PARAM_PREFIXES)],
        "attributes": [k for k in state if k.startswith("attributes.")],
        "groups": [k for k in state if k.startswith("groups.")],
        "feedback": [k for k in state if k.startswith("feedback.")],
    }
    claimed = {k for values in groups.values() for k in values}
    groups["other"] = [k for k in state if k not in claimed]
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("g0", "g1", "g0plus"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[0, 2000, 4000, 6000])
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--init-from", default=None,
                        help="step-0 checkpoint of the reference run; the initialisation is "
                             "verified block-by-block against it (G0+ must start from the "
                             "same random state as G0)")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = sha256_file(plan_path)
    if list(plan["train_scenes"]) != list(split["train_scenes"]):
        raise SystemExit("plan train scenes differ from the split")
    if int(plan["steps"]) != int(args.steps) or len(plan["entries"]) != args.steps:
        raise SystemExit("plan length does not match --steps")
    leaks = sorted({e["scene"] for e in plan["entries"]} & set(split["val_scenes"]))
    if leaks:
        raise SystemExit(f"validation scenes leaked into the plan: {leaks}")

    preset = config_defaults[args.preset]
    if getattr(preset, "init_checkpoint", None):
        raise SystemExit(f"preset {args.preset} declares init_checkpoint; refusing to warm start")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    group_arm = "g1" if args.arm == "g1" else "g0"
    background_supervision = args.arm == "g0plus"
    opt = preset.evolve(
        seed=int(args.seed),
        group_arm=group_arm,
        group_bg_supervision=background_supervision,
        group_bg_loss_weight=1.0,
        lr=args.lr,
        pct_start_steps=args.warmup,
        batch_size=1,
        num_workers=0,
        num_input_views=2,
        num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        workspace=str(out_dir),
        experiment_name=f"siu3r_group_locusgs_{args.arm}_v1",
        init_checkpoint=None,
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.freeze_object_queries()
    model.train()

    blocks = parameter_blocks(model)
    state = model.state_dict()
    named_parameters = dict(model.named_parameters())
    block_parameters = {
        name: [named_parameters[key] for key in keys if key in named_parameters]
        for name, keys in blocks.items()
    }
    block_hashes = {name: sha256_state(state, keys) for name, keys in blocks.items()}
    initialisation_source = "fresh random (seed %d)" % int(args.seed)
    if args.init_from:
        source_dir = Path(args.init_from)
        loaded = torch.load(source_dir / "model.pt", map_location="cpu", weights_only=False)
        reference_state = loaded["model"]
        model.load_state_dict(reference_state, strict=True)
        state = model.state_dict()
        block_hashes = {name: sha256_state(state, keys) for name, keys in blocks.items()}
        reference_hashes = {
            name: sha256_state(reference_state, keys) for name, keys in blocks.items()
        }
        if reference_hashes != block_hashes:
            raise SystemExit("initialisation does not match the reference step-0 checkpoint")
        payload = torch.load(source_dir / "train_state.pt", map_location="cpu",
                             weights_only=False)
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        initialisation_source = (
            f"loaded from {source_dir} (model+optimizer+RNG provenance verified; "
            f"reference hashes match bitwise)"
        )
    init_report = {
        "arm": args.arm,
        "group_arm": group_arm,
        "background_supervision": background_supervision,
        "background_loss_weight": float(opt.group_bg_loss_weight),
        "initialisation": initialisation_source,
        "initialisation_from": args.init_from,
        "seed": int(args.seed),
        "parameter_counts": {
            name: int(sum(state[k].numel() for k in keys)) for name, keys in blocks.items()
        },
        "block_hashes": block_hashes,
        "feedback_enabled": bool(model.use_feedback),
        "feedback_gate_value": float(model.feedback.gate.detach()),
        "plan_sha256": plan_sha,
        "split_sha256": sha256_file(Path(args.split)),
        "preset": args.preset,
        "steps": args.steps,
        "optimizer": {"peak_lr": args.lr, "warmup": args.warmup,
                      "schedule": "linear warmup then cosine to 2% of peak",
                      "betas": [0.9, 0.95], "weight_decay": args.weight_decay,
                      "grad_clip": args.grad_clip},
        "loss": {
            "total": "L_recon + lambda(step) * [0.05 L_instance + 0.05 L_semantic]",
            "lambda": f"min(1, step/{opt.group_loss_ramp_steps})",
            "instance_outer_weight": opt.group_inst_loss_weight,
            "semantic_outer_weight": opt.group_sem_loss_weight,
            "instance_inner_weights": {"class_ce": 2.0, "mask_bce": 5.0, "mask_dice": 5.0,
                                       "no_object_ce": 0.1},
            "recon": "canonical LocusGS layers {6,12}, weights {1/3,2/3}",
        },
    }
    (out_dir / "init_report.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")
    print(f"[g] arm={args.arm} model={opt.model_type} out={out_dir}", flush=True)
    print(f"[g] random init hashes: reconstruction {block_hashes['reconstruction'][:16]} "
          f"attributes {block_hashes['attributes'][:16]} groups {block_hashes['groups'][:16]} "
          f"feedback {block_hashes['feedback'][:16]}", flush=True)
    print(f"[g] feedback enabled={model.use_feedback} gate={float(model.feedback.gate):.3e} "
          f"layer={opt.group_feedback_layer} | group queries={model.num_groups} + 1 background",
          flush=True)
    print(f"[g] plan {plan_path} sha256={plan_sha} steps={len(plan['entries'])}", flush=True)
    print(f"[g] loss: {init_report['loss']['total']} with "
          f"lambda=min(1, step/{opt.group_loss_ramp_steps})", flush=True)

    decay, nodecay = [], []
    excluded = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not model.use_feedback and name.startswith("feedback."):
            excluded.append(name)
            continue
        if parameter.dim() != 1 and not getattr(parameter, "_no_weight_decay", False):
            decay.append(parameter)
        else:
            nodecay.append(parameter)
    groups = [
        {"params": decay, "lr": args.lr, "peak_lr": args.lr,
         "weight_decay": args.weight_decay, "name": "decay"},
        {"params": nodecay, "lr": args.lr, "peak_lr": args.lr,
         "weight_decay": 0.0, "name": "nodecay"},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    init_report["optimizer_groups"] = [
        {"name": group["name"], "params": sum(p.numel() for p in group["params"]),
         "peak_lr": group["peak_lr"], "weight_decay": group["weight_decay"]}
        for group in groups
    ]
    init_report["feedback_parameters_excluded_from_optimizer"] = excluded
    for group in groups:
        print(f"[g] group {group['name']}: {sum(p.numel() for p in group['params']):,} params "
              f"lr {group['peak_lr']:.1e} wd {group['weight_decay']}", flush=True)
    (out_dir / "init_report.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")

    provider = SIU3RProcessedProvider(
        opt, root=split["train_root"], subset=split["train_scenes"], training=True, rank=0
    )
    scene_index = {path.name: idx for idx, path in enumerate(provider.dataset.sample_list)}
    val_entries = [] if args.no_eval else build_val_entries(opt, split, device)

    manifest = {
        "experiment": "group_locusgs_g0g1_v1",
        "arm": args.arm,
        "scope": "32 train / 8 unseen development split; NOT an SIU3R official metric",
        "note": "model uses GT camera poses to build rays; SIU3R is an unposed setting",
        "initialisation": initialisation_source,
        "single_variable_g0plus": (
            "background-slot pixel supervision on the two context views; the model "
            "structure, queries, assignment softmax, Hungarian matching, instance "
            "loss, semantic loss, reconstruction loss, optimizer and schedule are "
            "unchanged from G0"
        ) if background_supervision else None,
        "single_variable": "group -> reconstruction-token feedback between layers 10 and 11",
        "init_report": init_report,
        "val_windows": {entry["scene"]: {"context": entry["context"], "novel": entry["novel"]}
                        for entry in val_entries},
        "git_commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "args": vars(args),
    }
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history: list[dict] = []
    checkpoint_meta = {"arm": args.arm, "plan_sha256": plan_sha, "from_scratch": True,
                       "init_hashes": block_hashes}
    total_steps = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"
    val_path = out_dir / "val_history.jsonl"
    started = time.time()
    low_alpha_streak = 0
    collapse_evals = 0

    def run_eval(step: int) -> dict:
        model.eval()
        rows = [evaluate_entry_v2(model, entry, opt) for entry in val_entries]
        model.train()
        summary = summarise_v2(rows)
        print(f"[g] VAL step {step}: ctx {summary['ctx_psnr']:.2f} novel {summary['novel_psnr']:.2f} "
              f"ssim {summary['ctx_ssim']:.3f}/{summary['novel_ssim']:.3f} "
              f"mIoU {summary['sem_miou']:.3f} | AP50 ctx {summary['ctx_ap50']:.3f} "
              f"novel {summary['novel_ap50']:.3f} TP/FP/FN {summary['novel_tp']}/"
              f"{summary['novel_fp']}/{summary['novel_fn']} | P(thing)>=0.5 pass "
              f"{summary['novel_gate_counts']['score']} bg-mass stuff/thing "
              f"{summary['background_stuff_mass_mean']:.3f}/"
              f"{summary['background_thing_mass_mean']:.3f} "
              f"maskerr {summary['mask_alpha_max_error']:.2e}",
              flush=True)
        serialisable = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
        with val_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "summary": summary, "rows": serialisable}) + "\n")
        (out_dir / "history.json").write_text(
            json.dumps({"manifest": manifest, "train": history}, indent=1), encoding="utf-8"
        )
        return summary

    if 0 in args.save_steps:
        model.eval()
        if not args.no_eval:
            run_eval(0)
        model.train()
        save_checkpoint(out_dir, 0, model, optimizer, checkpoint_meta, {"plan_step": 0},
                        keep_steps=list(args.save_steps))
        print("[g] checkpoint ckpt_step0 (random initialisation) written", flush=True)

    for step in range(1, total_steps + 1):
        entry = plan["entries"][step - 1]
        provider.pin_pair(
            scene_id=entry["scene"],
            context_frame_ids=entry["context"],
            novel_frame_ids=entry["novel"],
            pair_iou=entry["pair_iou"],
        )
        try:
            batch = move(default_collate([provider[scene_index[entry["scene"]]]]), device)
        except Exception as error:  # noqa: BLE001
            raise SystemExit(
                f"step {step}: pre-registered window {entry} failed to decode ({error}); "
                f"the plan must not be edited mid-run"
            ) from error
        frames = [int(x) for x in batch["frame_ids"][0].tolist()]
        if frames != entry["context"] + entry["novel"]:
            raise SystemExit(f"step {step}: batch frames {frames} != plan {entry}")

        optimizer.zero_grad(set_to_none=True)
        # 1-based step: lambda(1) = 1/2000 > 0, i.e. the understanding loss is
        # active (non-zero) from the very first update instead of being delayed.
        _, metrics = model.step_loss(batch, step=step, phase="train")
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            print("[g] non-finite metric dump: " + json.dumps({
                key: (float(value) if torch.is_tensor(value) and value.numel() == 1 else str(value))
                for key, value in metrics.items()}), flush=True)
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta, {"plan_step": step},
                            keep_steps=[step])
            raise SystemExit(f"non-finite loss at step {step}; stopped with evidence")
        loss.backward()
        grad_by_group = {group["name"]: 0.0 for group in groups}
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    grad_by_group[group["name"]] += float(
                        parameter.grad.detach().float().pow(2).sum()
                    )
        grad_by_group = {key: math.sqrt(value) for key, value in grad_by_group.items()}
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(grad_norm):
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta, {"plan_step": step},
                            keep_steps=[step])
            raise SystemExit(f"non-finite gradient at step {step}; stopped with evidence")
        if args.arm == "g0" and any(
            p.grad is not None for p in model.feedback.parameters()
        ):
            raise SystemExit("G0 optimized the disabled feedback branch")

        local_step = step - 1
        for group in optimizer.param_groups:
            group["lr"] = lr_at(local_step, float(group["peak_lr"]), args.warmup, args.steps)
        snapshot = None
        block_snapshot = None
        if step == 1 or step % args.log_every == 0 or step in args.save_steps:
            snapshot = {group["name"]: [p.detach().clone() for p in group["params"]]
                        for group in groups}
            block_snapshot = {
                name: [p.detach().clone() for p in parameters]
                for name, parameters in block_parameters.items()
            }
        optimizer.step()

        alpha_mean = float(metrics["alpha_mean"])
        low_alpha_streak = low_alpha_streak + 1 if alpha_mean < 0.005 else 0
        if low_alpha_streak >= 200:
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                            {"plan_step": step, "reason": "alpha_collapse"}, keep_steps=[step])
            raise SystemExit(f"alpha collapsed for {low_alpha_streak} steps at step {step}")

        record = {
            "step": step,
            "scene": entry["scene"],
            "context": entry["context"],
            "novel": entry["novel"],
            "loss": float(loss),
            "recon_loss": float(metrics["recon_loss"]),
            "loss_inst": float(metrics["loss_inst"]),
            "loss_sem": float(metrics["loss_sem"]),
            "loss_inst_ce": float(metrics["loss_inst_ce"]),
            "loss_inst_bce": float(metrics["loss_inst_bce"]),
            "loss_inst_dice": float(metrics["loss_inst_dice"]),
            "loss_inst_total": float(metrics["loss_inst_total"]),
            "loss_bg": float(metrics.get("loss_bg", 0.0)),
            "loss_bg_stuff": float(metrics.get("loss_bg_stuff", 0.0)),
            "loss_bg_thing": float(metrics.get("loss_bg_thing", 0.0)),
            "bg_pixels_stuff": float(metrics.get("bg_pixels_stuff", 0.0)),
            "bg_pixels_thing": float(metrics.get("bg_pixels_thing", 0.0)),
            "bg_prob_stuff_mean": float(metrics.get("bg_prob_stuff_mean", 0.0)),
            "bg_prob_thing_mean": float(metrics.get("bg_prob_thing_mean", 0.0)),
            "lambda": float(metrics["ramp"]),
            "instance_weight": float(metrics["instance_weight"]),
            "semantic_weight": float(metrics["semantic_weight"]),
            "psnr": float(metrics["psnr"]),
            "grad_norm": grad_norm,
            "grad_by_group": grad_by_group,
            "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "alpha_mean": alpha_mean,
            "mask_alpha_max_error": float(metrics["mask_alpha_max_error"]),
            "objectness_mean": float(metrics["objectness_mean"]),
            "objectness_max": float(metrics["objectness_max"]),
            "background_mass_mean": float(metrics["background_mass_mean"]),
            "thing_targets": float(metrics["thing_targets"]),
            "matched_groups": float(metrics["matched_groups"]),
            "slot_entropy": float(metrics.get("slot_entropy", float("nan"))),
            "slot_max_prob_mean": float(metrics.get("slot_max_prob_mean", float("nan"))),
            "active_group_share": float(metrics.get("active_group_share", float("nan"))),
            "sem_coverage": float(metrics["sem_coverage"]),
            "semantic_pixel_alpha_gap": float(metrics["semantic_pixel_alpha_gap"]),
            "radius_min": float(metrics["radius_min"]),
            "radius_max": float(metrics["radius_max"]),
            "anchor_max": float(metrics["anchor_max"]),
            "feedback_gate_abs_tanh": float(metrics["feedback_gate_abs_tanh"]),
            "group_update_norm": (
                float(metrics["group_update_norm"]) if "group_update_norm" in metrics else None
            ),
        }
        if snapshot is not None:
            record["update_by_group"] = {
                group["name"]: math.sqrt(sum(
                    float((parameter.detach().float() - before.float()).pow(2).sum())
                    for before, parameter in zip(snapshot[group["name"]], group["params"])
                ))
                for group in groups
            }
            record["update_by_block"] = {
                name: math.sqrt(sum(
                    float((parameter.detach().float() - before.float()).pow(2).sum())
                    for before, parameter in zip(block_snapshot[name], parameters)
                ))
                for name, parameters in block_parameters.items() if parameters
            }
            del snapshot
            del block_snapshot
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if step % args.log_every == 0 or step == 1 or step == total_steps:
            elapsed = (time.time() - started) / max(1, step)
            peak = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else float("nan"))
            print(f"[g] step {step:>5} arm={args.arm} scene={entry['scene']} "
                  f"loss {record['loss']:.4f} (recon {record['recon_loss']:.4f} "
                  f"inst {record['loss_inst']:.3f} sem {record['loss_sem']:.3f} "
                  f"lambda {record['lambda']:.5f} w_inst {record['instance_weight']:.3e} "
                  f"w_sem {record['semantic_weight']:.3e}) "
                  f"psnr {record['psnr']:.2f} grad {grad_norm:.2f} obj {record['objectness_max']:.3f} "
                  f"maskerr {record['mask_alpha_max_error']:.1e} gate "
                  f"{record['feedback_gate_abs_tanh']:.2e} | {elapsed:.2f}s/step peak {peak:.1f}G "
                  f"eta {(total_steps - step) * elapsed / 3600:.1f}h", flush=True)

        if not args.no_eval and (step % args.eval_every == 0 or step == total_steps):
            summary = run_eval(step)
            collapse = summary["novel_psnr"] < summary["novel_grey"] + 0.5
            # A random-initialised model legitimately renders the grey background,
            # so step 0 is not counted as a collapse; two consecutive grey-level
            # evaluations after training has started are.
            collapse_evals = (
                collapse_evals + 1 if (collapse and step >= 500) else 0
            )
            if collapse_evals >= 2:
                save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                                {"plan_step": step, "reason": "reconstruction_collapse"},
                                keep_steps=[step])
                raise SystemExit(
                    f"novel PSNR stayed at the grey-image level for {collapse_evals} evaluations "
                    f"at step {step}; stopped with evidence"
                )

        if step in args.save_steps:
            save_checkpoint(out_dir, step, model, optimizer, checkpoint_meta,
                            {"plan_step": step}, keep_steps=list(args.save_steps))
            print(f"[g] checkpoint ckpt_step{step} written", flush=True)

    (out_dir / "history.json").write_text(
        json.dumps({"manifest": manifest, "train": history}, indent=1), encoding="utf-8"
    )
    final_hash = sha256_state(model.state_dict())
    (out_dir / "final_state_hash.txt").write_text(final_hash + "\n", encoding="utf-8")
    print(f"[g] done: {len(history)} steps in {time.time() - started:.0f}s "
          f"final_state_sha256={final_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
