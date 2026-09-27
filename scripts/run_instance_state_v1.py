#!/usr/bin/env python3
"""Driver for LOCUSGS_INSTANCE_STATE_V1 (spec: docs/instance_state_v1_codex_prompt.md).

Phases
  prepare : CPU - spec.json, interface_audit.md, init_report.json,
            optimizer_groups.json, pilot_windows.json, monitor_8pairs.json,
            plan_paired_2000.json, storage_budget.json, provenance.json
  smoke   : GPU - delegate to scripts/smoke_instance_state_v1.py
  paired  : GPU - arm C (coupled=False) then arm E (coupled=True), 2000 steps each
  full    : GPU - selected arm on all 1201 official train scenes, 5000-step segments

Read-only with respect to every historical artefact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.data.siu3r_processed import SIU3RProcessedProvider  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import Options, config_defaults  # noqa: E402

PRETRAINED = REPO / "workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt"
PRETRAINED_SHA = "5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f"
PRETRAINED_STEP = 47500
BASE_PRESET = "train_siu3r_locusgs_recon_bounded_delta_frozen_radius"
PRESET_C = "train_siu3r_instance_state_locusgs"
PRESET_E = "train_siu3r_instance_state_locusgs_coupled"
SEED = 42
ARM_C = "C"
ARM_E = "E"
PAIRED_STEPS = 2000
FULL_SEGMENT = 5000

SPEC = {
    "name": "instance_state_v1",
    "version": 1,
    "baseline_commit": "b3acb34946a60f760a7daabbb1959c8d6be2c696",
    "model_type": "siu3r_instance_state_locusgs",
    "architecture_name": "LOCUSGS_INSTANCE_STATE_V1",
    "dims": {"B": 1, "T": 1024, "P": 64, "C": 1024, "D": 256, "d_id": 16,
             "num_thing": 100, "num_stuff": 2, "void_channels": 1, "num_states": 103},
    "state_layers": (6, 8, 10, 12),
    "self_attn_bias_consumers": {"7": 6, "8": 6, "9": 8, "10": 8, "11": 10, "12": 10},
    "assign": {"temperature": 0.1, "thing_scale": 10.0, "tightness": 0.1,
               "tightness_clamp": 25.0, "norm": "F.normalize(dim=-1,eps=1e-6)",
               "layernorm_eps": 1e-5},
    "beta_ramp_steps": 200, "rseg_ramp_steps": 200,
    "beta_scales": {"h": 0.1, "mu": 0.1, "r": 0.05, "gs_residual": 0.1},
    "gs_decode_radius": 0.15,
    "fps": "deterministic, first = closest to mean(mu), then max min-distance, ties -> lowest index",
    "ell": "stopgrad(clamp(rms radius, min=0.05)) from mu6 only",
    "loss": {"w_thing": 0.1, "w_stuff": 0.1, "w_sem": 0.1, "w_id": 0.01,
             "match_cost": {"class": 1.0, "mask_bce": 5.0, "dice": 5.0},
             "mask_bce": 5.0, "mask_dice": 5.0, "class_ce": 2.0,
             "no_object_ce": 0.1, "match_points": 4096, "id_points": 64,
             "id_margin": 0.2, "id_alpha_min": 0.05},
    "optimizer": {"betas": (0.9, 0.95), "backbone_peak_lr": 1e-5,
                  "instance_state_peak_lr": 1e-4,
                  "matrix_weight_decay": 0.05, "no_decay_on": "bias, norm, query_init, _no_weight_decay",
                  "grad_clip": 1.0,
                  "warmup": 100,
                  "lr_schedule": "peak*step/100 for step 1..100 then peak*(0.02+0.98*0.5*(1+cos(pi*(step-100)/(total-100))))"},
    "paired": {"arms": [ARM_C, ARM_E], "steps": PAIRED_STEPS,
               "cycles": "w0,w1,w2,w3 x 500"},
    "full": {"train_scenes": 1201, "val_scenes": 312, "steps": 50000,
             "warmup": 2000, "segment": FULL_SEGMENT,
             "stop_rule": "step5000: 8val class-relevant TP all zero and semantic mIoU<0.05"},
    "init": {"checkpoint": str(PRETRAINED), "sha256": PRETRAINED_SHA,
             "step": PRETRAINED_STEP, "seed": SEED, "new_module_seed": 31415},
    "eval": {"gt_free_thing": "p_thing=sum(Pq[:18])>=0.5, class=argmax+2, mask=Mthing>0.5 & alpha>0.05, area>=50",
             "semantic": "argmax over 20 classes where alpha>0.05",
             "panoptic": "argmax score*Mthing per pixel; stuff for uncovered wall/floor; else void"},
    "local_structure_pass": {
        "context_semantic_miou": 0.35, "context_semantic_miou_gain": 0.10,
        "context_raw_recall50": 0.50, "context_class_recall50": 0.30,
        "context_class_recall50_gain": 0.20, "windows_with_tp": "3/4",
        "novel_class_recall50": 0.10, "novel_psnr_drop_db": 1.0,
        "local_panoptic_pq": ">0 and >=1 correct-class thing TP in 3/4 windows"},
    "instance_state_supported": {"e_minus_c_novel_class_recall50": 0.05,
                                 "semantic_miou_not_lower_by": 0.02,
                                 "psnr_not_lower_by_db": 0.5,
                                 "requires": "state intervention changes geometry"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def build_options(preset: str, *, num_views: int = 4) -> Options:
    return config_defaults[preset].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=SEED,
        num_input_views=2, num_views=num_views)


def transfer_reconstruction_weights(model, source_state: dict, opt) -> dict:
    """Load the pretrained LocusGSRecon strictly, then the new model key by key."""
    base = model_registry["siu3r_locusgs_recon"](opt)
    base.load_state_dict({k: v for k, v in source_state.items() if "lpips" not in k},
                         strict=True)
    report = {"source_keys": len(source_state), "matched": 0, "new_keys": 0,
              "mismatched": [], "missing_in_source": [], "shape_mismatch": [],
              "alias_conflicts": []}
    target = model.state_dict()
    source = base.state_dict()
    seen_values: dict[int, str] = {}
    for key, value in target.items():
        if key.startswith("instance_state."):
            report["new_keys"] += 1
            continue
        if key not in source:
            report["missing_in_source"].append(key)
            continue
        if tuple(source[key].shape) != tuple(value.shape):
            report["shape_mismatch"].append(key)
            continue
        if source[key].data_ptr() in seen_values and source[key].data_ptr() != 0:
            report["alias_conflicts"].append([key, seen_values[source[key].data_ptr()]])
        seen_values[source[key].data_ptr()] = key
        with torch.no_grad():
            value.copy_(source[key])
        report["matched"] += 1
    if report["missing_in_source"] or report["shape_mismatch"]:
        raise RuntimeError(f"initialisation transfer failed: {report}")
    report["strict_after_transfer"] = True
    del base
    return report


def optimize_groups(model) -> dict:
    """Deprecated shim: the single grouping implementation lives in the runtime module."""
    from scripts.instance_state_runtime import build_optimizer
    _, report = build_optimizer(model)
    return report


def _legacy_optimize_groups(model) -> dict:
    """Parameter groups with object-identity dedup (shared/aliased tensors once)."""
    backbone_decay, backbone_nodecay, state_decay, state_nodecay = [], [], [], []
    seen: set[int] = set()
    rows = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        row = {"name": name, "shape": list(param.shape), "numel": int(param.numel()),
               "requires_grad": True}
        if id(param) in seen:
            row["group"] = "duplicate(alias)"
            rows.append(row)
            continue
        seen.add(id(param))
        is_state = name.startswith("instance_state.")
        no_decay = (param.dim() == 1) or bool(getattr(param, "_no_weight_decay", False))
        row["group"] = ("instance_state" if is_state else "backbone") + \
            ("_nodecay" if no_decay else "_decay")
        rows.append(row)
        (state_decay if is_state else backbone_decay).append(param) if not no_decay \
            else (state_nodecay if is_state else backbone_nodecay).append(param)
    unique = [r for r in rows if r["group"] != "duplicate(alias)"]
    return {
        "rows": rows,
        "unique_param_count": len(unique),
        "total_named": len(rows),
        "groups": {
            "backbone_decay": sum(1 for r in unique if r["group"] == "backbone_decay"),
            "backbone_nodecay": sum(1 for r in unique if r["group"] == "backbone_nodecay"),
            "instance_state_decay": sum(1 for r in unique if r["group"] == "instance_state_decay"),
            "instance_state_nodecay": sum(1 for r in unique if r["group"] == "instance_state_nodecay"),
        },
        "n_params": {"backbone_decay": len(backbone_decay), "backbone_nodecay": len(backbone_nodecay),
                     "instance_state_decay": len(state_decay),
                     "instance_state_nodecay": len(state_nodecay)},
        "numel": {
            "backbone_decay": sum(p.numel() for p in backbone_decay),
            "backbone_nodecay": sum(p.numel() for p in backbone_nodecay),
            "instance_state_decay": sum(p.numel() for p in state_decay),
            "instance_state_nodecay": sum(p.numel() for p in state_nodecay),
        },
    }


def build_optimizer(model, opt=None, *, report: bool = False):
    """Delegate to the single runtime implementation (see scripts/instance_state_runtime)."""
    from scripts.instance_state_runtime import build_optimizer as _build
    del opt, report
    return _build(model)


def sample_scene_window(opt, train_root: Path, scene: str, seed: int, tries: int,
                        min_instances: int = 2, min_area: int = 100):
    """First officially-legal 2+2 pair whose context frames carry enough GT thing."""
    provider = SIU3RProcessedProvider(opt, root=str(train_root), subset=[scene],
                                      training=True, rank=0)
    provider.pair_rng.seed(seed)
    for attempt in range(tries):
        try:
            sample = provider[0]
        except Exception:                                   # noqa: BLE001
            continue
        pair = dict(provider.last_pair)
        sem = sample["semantic_label_all"][:2]
        ins = sample["instance_label_all"][:2]
        counts = {}
        for view in range(2):
            for value in torch.unique(ins[view][sem[view] >= 2]).tolist():
                if int(value) <= 0:
                    continue
                area = int(((ins[view] == int(value)) & (sem[view] >= 2)).sum())
                counts[int(value)] = counts.get(int(value), 0) + area
        good = [k for k, a in counts.items() if a >= min_area]
        if len(good) >= min_instances:
            pair.update({"attempt": attempt, "thing_instances": len(good),
                         "thing_area_total": int(sum(counts.values())),
                         "gt_instance_ids": sorted(good)})
            return pair
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "smoke", "paired", "full"), required=True)
    parser.add_argument("--reports", default="group_plus/instance_state_v1")
    parser.add_argument("--run-root", default="workspace_group_plus/instance_state_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--until-step", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--arm", choices=("C", "E"), default=None)
    args = parser.parse_args()
    reports = REPO / args.reports
    run_root = REPO / args.run_root
    reports.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    if args.phase == "prepare":
        return phase_prepare(reports, run_root)
    print(f"[isv1] phase {args.phase} must run on a SLURM GPU node; "
          f"see scripts/submit_instance_state_v1.sh", flush=True)
    if args.phase == "smoke":
        from scripts.smoke_instance_state_v1 import run_all
        return run_all(reports=reports, run_root=run_root, device=args.device)
    from scripts.instance_state_paired import run_paired, run_full
    if args.phase == "paired":
        return run_paired(reports=reports, run_root=run_root, device=args.device,
                          arm=args.arm)
    return run_full(reports=reports, run_root=run_root, device=args.device,
                    until_step=args.until_step, resume=args.resume, arm=args.arm)


def phase_prepare(reports: Path, run_root: Path) -> int:
    head = git("rev-parse", "HEAD")
    status = git("status", "--short")
    split_path = REPO / "group_plus/implementation_audit_v1/full_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_scenes = list(split["train_scenes"])
    val_scenes = list(split["val_scenes"])
    overlap = sorted(set(train_scenes) & set(val_scenes))
    train_root = Path(split["train_root"])
    val_root = Path(split["val_root"])
    print(f"[prepare] HEAD {head} | train {len(train_scenes)} val {len(val_scenes)} "
          f"overlap {len(overlap)}", flush=True)
    if overlap:
        raise SystemExit(f"split overlap: {overlap[:5]}")
    if head != SPEC["baseline_commit"]:
        print(f"[prepare] WARNING HEAD {head} != registered baseline "
              f"{SPEC['baseline_commit']}", flush=True)

    # ---- 1. spec.json -------------------------------------------------- #
    write_json(reports / "spec.json", SPEC)

    # ---- 2. pretrained provenance -------------------------------------- #
    pretrained_digest = sha256_file(PRETRAINED)
    cfg_path = PRETRAINED.parent / "config.yaml"
    rec_path = PRETRAINED.parent / "training_record.json"
    payload = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    init_report = {
        "checkpoint": str(PRETRAINED), "sha256": pretrained_digest,
        "sha256_expected": PRETRAINED_SHA, "sha256_ok": pretrained_digest == PRETRAINED_SHA,
        "step_in_file": int(payload.get("step", -1)), "step_expected": PRETRAINED_STEP,
        "config_yaml_present": cfg_path.is_file(),
        "training_record_present": rec_path.is_file(),
        "source_docs": ["docs/locusgs_full_valpair_recon_eval.md",
                        "docs/full_train_cleanup_manifest.json"],
    }
    if not init_report["sha256_ok"]:
        raise SystemExit(f"pretrained sha256 mismatch: {pretrained_digest}")
    init_report["training_record"] = (json.loads(rec_path.read_text(encoding="utf-8"))
                                      if rec_path.is_file() else None)

    # ---- 3. model construction + strict transfer ----------------------- #
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    opt_c = build_options(PRESET_C).evolve(evaluating=False, use_input_supervision=False)
    model = model_registry[opt_c.model_type](opt_c)
    source_state = payload.get("model", payload)
    init_report["transfer"] = transfer_reconstruction_weights(model, source_state, opt_c)
    init_report["state_dict_keys_total"] = len(model.state_dict())
    init_report["instance_state_keys"] = len(
        [k for k in model.state_dict() if k.startswith("instance_state.")])
    init_report["param_count"] = int(sum(p.numel() for p in model.parameters()))
    init_report["new_param_count"] = int(sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("instance_state.")))
    radius = model.activation_head.last_decode_radius if hasattr(
        model.activation_head, "last_decode_radius") else None
    init_report["decode_radius_mode"] = {
        "locusgs_freeze_decode_radius": bool(opt_c.locusgs_freeze_decode_radius),
        "locusgs_radius_init": float(opt_c.locusgs_radius_init),
        "locusgs_bound_delta": bool(opt_c.locusgs_bound_delta),
        "last_decode_radius": None if radius is None else [float(radius.min()),
                                                           float(radius.max())],
    }
    init_report["config"] = {
        "model_type": opt_c.model_type, "enc_embed_dim": int(opt_c.enc_embed_dim),
        "num_gs_tokens": int(opt_c.num_gs_tokens), "dec_depth": int(opt_c.dec_depth),
        "dec_patch_size": int(opt_c.dec_patch_size), "img_size": list(opt_c.img_size),
        "supervised_layers": list(opt_c.locusgs_supervised_layers),
        "instance_state_layers": list(opt_c.instance_state_layers),
        "instance_state_coupled": bool(opt_c.instance_state_coupled),
    }
    # C and E must share the identical new-module initialisation
    opt_e = build_options(PRESET_E).evolve(evaluating=False, use_input_supervision=False)
    model_e = model_registry[opt_e.model_type](opt_e)
    same = all(torch.equal(a, b) for a, b in zip(
        [p for n, p in model.named_parameters() if n.startswith("instance_state.")],
        [p for n, p in model_e.named_parameters() if n.startswith("instance_state.")]))
    init_report["C_E_identical_new_init"] = bool(same)
    del model_e
    write_json(reports / "init_report.json", init_report)
    print(f"[prepare] init: matched {init_report['transfer']['matched']} keys, "
          f"new {init_report['instance_state_keys']} instance_state tensors "
          f"({init_report['new_param_count']:,} params), C/E init equal {same}", flush=True)

    # ---- 4. optimizer groups ------------------------------------------- #
    groups = optimize_groups(model)
    write_json(reports / "optimizer_groups.json", groups)
    print(f"[prepare] optimizer groups {groups['groups']} "
          f"numel {groups['numel']}", flush=True)
    del model

    # ---- 5. pilot windows ---------------------------------------------- #
    opt = build_options(BASE_PRESET)
    sentinel = {"scene": "scene0009_02", "context": [209, 253], "novel": [215, 247],
                "pair_iou": 0.34302523732185364,
                "sentinel_instances": [{"instance": 18032, "class": 17},
                                       {"instance": 20030, "class": 19}]}
    windows = [dict(sentinel)]
    skipped = []
    for scene in train_scenes:
        if scene == sentinel["scene"]:
            continue
        found = sample_scene_window(opt, train_root, scene, seed=SEED, tries=100)
        if found is None:
            skipped.append(scene)
            continue
        windows.append({
            "scene": scene, "context": found["context_frame_ids"],
            "novel": found["novel_frame_ids"], "pair_iou": found["pair_iou"],
            "attempt": found["attempt"], "thing_instances": found["thing_instances"],
            "thing_area_total": found["thing_area_total"],
            "gt_instance_ids": found["gt_instance_ids"]})
        if len(windows) == 4:
            break
    if len(windows) != 4:
        raise SystemExit(f"could not build 4 pilot windows: {windows}")
    write_json(reports / "pilot_windows.json",
               {"windows": windows, "skipped_scenes": skipped,
                "selection": "official provider pairs, seed42, first window with >=2 things "
                             "each >=100px GT area in the context frames"})
    # ---- 5b. re-verify the selected windows with the corrected GT rule ------ #
    recheck = []
    for window in windows:
        provider = SIU3RProcessedProvider(opt, root=str(train_root),
                                          subset=[window["scene"]], training=True, rank=0)
        want = np.array([*window["context"], *window["novel"]], dtype=np.int64)
        provider._get_indices_static = lambda idx: (want, [])      # noqa: SLF001
        sample = provider[0]
        sem = sample["semantic_label_all"][:2]
        ins = sample["instance_label_all"][:2]
        areas: dict[int, int] = {}
        for view in range(2):
            thing = (sem[view] >= 2) & (sem[view] <= 19) & (ins[view] > 0)
            for value in torch.unique(ins[view][thing]).tolist():
                area = int((thing & (ins[view] == int(value))).sum())
                areas[int(value)] = areas.get(int(value), 0) + area
        good = sorted(k for k, a in areas.items() if a >= 100)
        recheck.append({"scene": window["scene"],
                        "context": window["context"], "novel": window["novel"],
                        "thing_instances_ge100px": good, "n_qualified": len(good),
                        "semantic_range": [int(sem.min()), int(sem.max())],
                        "qualified": len(good) >= 2})
    write_json(reports / "pilot_recheck.json", {"windows": recheck})
    failed = [r["scene"] for r in recheck if not r["qualified"]]
    if failed:
        raise SystemExit(f"DATA_PROTOCOL_BLOCKED: windows failing the corrected GT rule: {failed}")
    print(f"[prepare] pilot recheck passed for all {len(recheck)} windows", flush=True)

    # ---- 6. monitor pairs ---------------------------------------------- #
    val_pairs = json.loads((Path(split["val_root"]).parent / "val_pair.json").read_text(
        encoding="utf-8"))
    keyed = sorted(val_pairs, key=lambda r: (r["scan"], list(r["context_ids"])))
    seen_scenes, monitor = set(), []
    for record in keyed:
        if record["scan"] in seen_scenes:
            continue
        seen_scenes.add(record["scan"])
        context = [int(x) for x in record["context_ids"]]
        novel = [int(x) for x in record["target_ids"] if int(x) not in set(context)]
        monitor.append({"scene": record["scan"], "context": context,
                        "target": [int(x) for x in record["target_ids"]],
                        "novel": novel, "pair_iou": record.get("iou")})
        if len(monitor) == 8:
            break
    write_json(reports / "monitor_8pairs.json",
               {"pairs": monitor, "source": "official val_pair.json, first pair per scene, "
                                            "first 8 distinct scenes, no GT/prediction filtering"})

    # ---- 7. paired plan ------------------------------------------------ #
    plan = []
    for cycle in range(500):
        for index, window in enumerate(windows):
            plan.append({"step": cycle * 4 + index + 1, "cycle": cycle, "window": index,
                         "scene": window["scene"], "context": window["context"],
                         "novel": window["novel"]})
    write_json(reports / "plan_paired_2000.json",
               {"steps": len(plan), "arms": [ARM_C, ARM_E], "entries": plan})
    digests = {}
    for name in ("pilot_windows.json", "plan_paired_2000.json", "monitor_8pairs.json"):
        digests[name] = sha256_file(reports / name)
        print(f"[prepare] {name} sha256 {digests[name][:16]}", flush=True)
    pilot_digest = digests["pilot_windows.json"]
    plan_digest = digests["plan_paired_2000.json"]
    monitor_digest = digests["monitor_8pairs.json"]

    # ---- 8. storage budget --------------------------------------------- #
    stat = os.statvfs(REPO)
    free_bytes = stat.f_bavail * stat.f_frsize
    ckpt_bytes = 880165419 * 3                      # model + 2 Adam moments + margin
    budget = {
        "actual_free_bytes": free_bytes, "actual_free_gib": free_bytes / 2**30,
        "one_checkpoint_bytes_estimate": ckpt_bytes,
        "paired_peak": {"arms": 2, "model_only_endpoints": 2 * 880165419},
        "full_peak": {"latest_resumable": ckpt_bytes, "next_in_progress": ckpt_bytes,
                      "paired_endpoints": 2 * 880165419},
        "safety_margin_bytes": 2**30,
        "full_allowed": None,
    }
    budget["full_required_bytes"] = (budget["full_peak"]["latest_resumable"]
                                     + budget["full_peak"]["next_in_progress"]
                                     + budget["full_peak"]["paired_endpoints"]
                                     + budget["safety_margin_bytes"])
    budget["full_allowed"] = bool(free_bytes >= budget["full_required_bytes"])
    write_json(reports / "storage_budget.json", budget)
    print(f"[prepare] free {budget['actual_free_gib']:.2f} GiB | full requires "
          f"{budget['full_required_bytes']/2**30:.2f} GiB -> {budget['full_allowed']}",
          flush=True)

    # ---- 9. provenance -------------------------------------------------- #
    provenance = {
        "commit": head, "git_status": status.splitlines(),
        "split": str(split_path), "split_sha256": sha256_file(split_path),
        "train_scenes": len(train_scenes), "val_scenes": len(val_scenes),
        "pretrained": str(PRETRAINED), "pretrained_sha256": pretrained_digest,
        "pilot_windows_sha256": pilot_digest,
        "plan_paired_2000_sha256": plan_digest,
        "monitor_8pairs_sha256": monitor_digest,
        "siu3r_commit": "8ea80166be76854f938e90521f1a5b688b755c87",
        "python": sys.version.split()[0], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "device_capability": (torch.cuda.get_device_capability(0)
                              if torch.cuda.is_available() else None),
        "amp": False, "tf32": False, "batch_size": 1, "num_workers": 0,
        "seed": SEED, "new_module_seed": 31415,
    }
    write_json(reports / "provenance.json", provenance)
    print("[prepare] wrote spec.json, init_report.json, optimizer_groups.json, "
          "pilot_windows.json, monitor_8pairs.json, plan_paired_2000.json, "
          "storage_budget.json, provenance.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
