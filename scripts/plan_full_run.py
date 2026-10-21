#!/usr/bin/env python3
"""Prepare (never launch) the full 50000-step group recipe run.

On-disk facts this script verifies instead of assuming:

* the official train tree is ``SIU3R/data/scannet/train``; **1201** scene folders
  carry a ``panoptic`` directory (the rest are unextracted archives), which is
  where the brief's "1201" comes from;
* the official val tree is ``SIU3R/data/scannet/val`` (312 scenes) and is
  disjoint from the train set;
* 5 of the old 32/8 development scenes lie in the official train tree ant 3 in
  the official val tree, so after the full run the old 55/110 numbers are no
  longer unseen-validation results.

Writes the split file, the 50000-step scene plan (via ``gen_object_plan.py``),
the monitoring list, ``gate.json``, and a fail-fast sbatch.  Running the sbatch
requires ``gate.json`` to be passed, so it cannot be started by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/space/mawb/ssst")
DATA = Path("/space/mawb/SIU3R/data/scannet")
OUT = REPO / "group_plus/implementation_audit_v1"
OLD_DEV_SCENES = ("scene0059_00", "scene0072_02", "scene0132_01", "scene0472_01",
                  "scene0559_01", "scene0568_02", "scene0615_00", "scene0695_00")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_scenes() -> tuple[list[str], list[str]]:
    train = sorted(p.name for p in (DATA / "train").iterdir()
                   if p.is_dir() and (p / "panoptic").is_dir())
    val = sorted(p.name for p in (DATA / "val").iterdir() if p.is_dir())
    return train, val


def monitoring_list(val_pair: list[dict], count: int = 8) -> list[dict]:
    """First pair of each of the first `count` distinct scenes, scene/context sorted."""
    ordered = sorted(val_pair, key=lambda r: (r["scan"], tuple(r["context_ids"])))
    picked, seen = [], set()
    for record in ordered:
        if record["scan"] in seen:
            continue
        seen.add(record["scan"])
        picked.append({"scene": record["scan"],
                       "context": list(record["context_ids"]),
                       "target": list(record["target_ids"])})
        if len(picked) >= count:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peak-lr", type=float, default=2e-5)
    parser.add_argument("--cosine-floor", type=float, default=4e-7)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--generate-plan", action="store_true",
                        help="actually run gen_object_plan.py (CPU, ~minutes)")
    parser.add_argument("--candidate", default=None,
                        help="which arm passed the gates (pure4 | stop_grad | BLOCKED)")
    args = parser.parse_args()

    train, val = official_scenes()
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise SystemExit(f"official train/val trees overlap: {overlap[:5]}")
    old_in_train = sorted(set(OLD_DEV_SCENES) & set(train))
    old_in_val = sorted(set(OLD_DEV_SCENES) & set(val))
    val_pair = json.loads((DATA / "val_pair.json").read_text(encoding="utf-8"))
    val_pair_scenes = sorted({r["scan"] for r in val_pair})

    split_path = OUT / "full_split.json"
    split = {
        "source": "official SIU3R processed ScanNet tree (scenes carrying panoptic labels)",
        "train_root": str(DATA / "train"),
        "val_root": str(DATA / "val"),
        "train_scenes": train,
        "val_scenes": val,
        "n_train": len(train),
        "n_val": len(val),
        "disjoint": not overlap,
        "note": "training scenes = folders under data/scannet/train with a panoptic/ "
                "directory; validation scenes = the official val tree",
    }
    split_path.write_text(json.dumps(split, indent=1), encoding="utf-8")

    mon = monitoring_list(val_pair)
    (OUT / "full_monitor_windows.json").write_text(json.dumps({
        "source": "official val_pair.json, sorted by (scene, context), first pair of the "
                  "first 8 distinct scenes",
        "windows": mon,
    }, indent=1), encoding="utf-8")

    plan_path = OUT / f"plan_full_{args.steps}.json"
    plan_sha = sha256_file(plan_path) if plan_path.is_file() else None
    if args.generate_plan and not plan_path.is_file():
        cmd = [sys.executable, "scripts/gen_object_plan.py",
               "--split", str(split_path), "--steps", str(args.steps),
               "--seed", str(args.seed), "--out", str(plan_path),
               "--verify-batches", "4"]
        print("[plan] running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
        plan_sha = sha256_file(plan_path)

    candidate = args.candidate
    if candidate is None:
        candidate = "PENDING_GATE"
    blocked = candidate in (None, "PENDING_GATE", "BLOCKED")
    gate = {
        "entry_conditions": {
            "implementation_audit_passed": "see runtime_diagnostics.json and "
                                           "implementation_audit.md",
            "official_format_smoke_passed": "see export_smoke.json",
            "no_unresolved_training_behaviour_bug": "see implementation_audit.md "
                                                    "'unresolved' section",
            "candidate_arm_reaches_all_five_dev_gates_at_step6000":
                "see the A/B comparison table; five gates = novel PSNR >= 18.957, "
                "routing_v1 IoU mean >= 0.2311, IoU>=0.5 >= 6/55, GT-free TP >= 6, "
                "AP50 >= 0.1675",
        },
        "candidate": candidate,
        "blocked": blocked,
        "gate_rule": "pure4 is selected only if it passes all five; otherwise the "
                     "stop-gradient arm is selected only if it passes all five; "
                     "otherwise BLOCKED",
    }
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")

    config = {
        "run_name": "recipe_full_50000",
        "steps": args.steps,
        "seed": args.seed,
        "peak_lr": args.peak_lr,
        "cosine_floor": args.cosine_floor,
        "warmup": args.warmup,
        "optimizer": "AdamW betas (0.9, 0.95), weight_decay 0.05 for 2-D weights, "
                     "0.0 for 1-D/bias, grad clip 1.0",
        "schedule": "linear warmup to peak, then cosine to the floor",
        "recipe": {
            "instance_outer_weight": 0.05,
            "instance_ramp_steps": 1500,
            "assign_coef": 0.2,
            "assign_every": 1,
            "assign_stop_shared_grad": "per the selected candidate",
            "semantic_outer_weight": 0.05,
            "semantic_ramp_steps": 2000,
            "head_mode": "per the selected candidate",
        },
        "data": {"train_scenes": len(train), "val_scenes": len(val),
                 "context_views": 2, "novel_views": 2, "dtype": "fp32"},
        "init": "arm_g0/ckpt_step0 + deep decoder seed 1743 (same cold start as 32/8)",
        "why_low_lr": "the historical full-data reconstruction run collapsed at lr 1e-4 and "
                      "recovered under a 2e-5 cap; this is a new full-data stability recipe, "
                      "not a single-variable change, and the low-LR cold start is not yet "
                      "demonstrated stable",
        "monitoring": {"windows": len(mon), "every_steps": 2500,
                       "final_report_step": args.steps,
                       "keep_checkpoints": "latest 2 plus 25000/50000"},
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "contamination_note": {
            "old_dev_scenes_in_official_train": old_in_train,
            "old_dev_scenes_in_official_val": old_in_val,
            "consequence": "after the full run the old 55/110 development numbers are "
                           "no longer unseen-validation results",
        },
        "official_val_pair": {"records": len(val_pair), "scenes": len(val_pair_scenes)},
        "estimated_cost": {
            "reference_6000_steps": "58m06s (recipe_v1, 1x3090)",
            "linear_50000_estimate_hours": round(58.06 / 60 * 50000 / 6000, 2),
            "planning_range_hours": "8-12 (I/O and the extra probe forwards can exceed "
                                    "the linear estimate; this is a plan, not a measurement)",
            "scene_sampling_density": {
                "full": round(args.steps / len(train), 2),
                "dev": round(6000 / 32, 2),
                "steps_needed_to_match_dev_density": int(6000 / 32 * len(train)),
            },
        },
    }
    (OUT / "full_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    sbatch = (OUT / "submit_full.sh")
    sbatch.write_text(f"""#!/bin/bash
#SBATCH --job-name=recipe-full
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=3dimage-13
#SBATCH --output=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/full-%j.log
#SBATCH --error=/space/mawb/ssst/workspace_group_plus/implementation_audit_v1/full-%j.err
#
# FULL 50000-step run.  Refuses to start unless gate.json says the candidate
# passed all five development gates.  NOT started by this round.
set -euo pipefail
export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH
cd /space/mawb/ssst
python - <<'PY'
import json, sys
gate = json.load(open("group_plus/implementation_audit_v1/gate.json"))
if gate.get("blocked") or gate.get("candidate") in (None, "PENDING_GATE", "BLOCKED"):
    sys.exit("gate.json is BLOCKED or pending: refusing to start the full run")
print("gate ok:", gate["candidate"])
PY
python -u scripts/train_group_locusgs.py \\
  --arm g0 --recipe \\
  --recipe-head-mode {candidate if candidate in ("pure4",) else "legacy_prefix"} \\
  --out-dir workspace_group_plus/implementation_audit_v1/recipe_full_50000/run \\
  --plan group_plus/implementation_audit_v1/plan_full_{args.steps}.json \\
  --split group_plus/implementation_audit_v1/full_split.json \\
  --init-from workspace_group_locusgs/arm_g0/ckpt_step0 \\
  --instance-outer-weight 0.05 --assign-coef 0.2 --assign-every 1 \\
  --lr {args.peak_lr} --warmup {args.warmup} \\
  --steps {args.steps} --save-steps 0 25000 {args.steps} \\
  --eval-every 2500
""", encoding="utf-8")
    sbatch.chmod(0o755)

    print(json.dumps({
        "train_scenes": len(train), "val_scenes": len(val),
        "disjoint": not overlap, "plan_sha256": plan_sha,
        "monitor_windows": len(mon), "gate_blocked": blocked,
        "old_dev_in_official_train": old_in_train,
        "old_dev_in_official_val": old_in_val,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
