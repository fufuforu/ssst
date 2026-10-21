#!/usr/bin/env python3
"""CPU-only: complete the v1/v2 step curves from the on-disk run logs.

The Git copy of ``group_plus/recipe_v1/val_curves.jsonl`` only reaches step
2500; the authoritative full logs live under ``workspace_group_plus``.  This
script records the run/job/checkpoint/plan identity and the requested steps
(0/1000/3000/6000), writing ``missing`` where a field genuinely does not exist.
Read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUNS = {
    "recipe_v1": {"job": 55512, "eval_job": 55591,
                  "run": "workspace_group_plus/recipe_v1/run",
                  "instance_outer_weight": 0.1},
    "recipe_v2": {"job": 55609, "eval_job": 55610,
                  "run": "workspace_group_plus/recipe_v2/run",
                  "instance_outer_weight": 0.05},
}
STEPS = (0, 1000, 3000, 6000)
TRAIN_FIELDS = ("loss", "recon_loss", "loss_inst", "loss_sem", "assign_ce",
                "assign_thing_tokens", "assign_rest_tokens",
                "assign_argmax_agreement", "seg_ramp", "instance_weight",
                "semantic_weight", "grad_norm", "alpha_mean",
                "mask_alpha_max_error", "fixed_void_max_abs", "lr", "scene",
                "context", "novel")


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/implementation_audit_v1/curves.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    args = parser.parse_args()

    plan_sha = sha256_file(Path(args.plan))
    payload = {
        "plan_sha256": plan_sha,
        "preregistered_plan_sha256":
            "a2a65c1382da0307a68fb6d27e5c1345aa78b46331acb2927e88b47a3c3d08bb",
        "note": "values are read from the on-disk runs (authoritative); the Git copy of "
                "recipe_v1/val_curves.jsonl stops at step 2500 and is not used",
        "runs": {},
    }
    for name, meta in RUNS.items():
        run = Path(meta["run"])
        train = {}
        log_path = run / "train_log.jsonl"
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("step") in STEPS:
                    train[row["step"]] = {k: row.get(k, "missing") for k in TRAIN_FIELDS}
        val = {}
        val_path = run / "val_history.jsonl"
        if val_path.is_file():
            for line in val_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("step") in STEPS:
                    summary = row.get("summary", {})
                    val[row["step"]] = {
                        k: summary.get(k, "missing") for k in
                        ("novel_psnr", "ctx_psnr", "novel_ssim", "ctx_ssim", "sem_miou",
                         "novel_ap50", "context_ap50", "novel_tp", "novel_fp", "novel_fn",
                         "novel_best_over_groups_iou", "mask_alpha_max_error",
                         "background_mass_fraction", "group_usage_active",
                         "slot_entropy_mean", "p_thing_mean")
                    }
        checkpoint = run / "ckpt_step6000"
        payload["runs"][name] = {
            "job": meta["job"], "eval_job": meta["eval_job"],
            "run_dir": str(run),
            "instance_outer_weight": meta["instance_outer_weight"],
            "plan_sha256": plan_sha,
            "checkpoint_model_sha256": sha256_file(checkpoint / "model.pt"),
            "checkpoint_dir_exists": checkpoint.is_dir(),
            "train_log_lines": sum(1 for _ in log_path.open()) if log_path.is_file() else 0,
            "train": {str(s): train.get(s, "missing") for s in STEPS},
            "val": {str(s): val.get(s, "missing") for s in STEPS},
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({name: {"train_log_lines": v["train_log_lines"],
                             "steps_present": [s for s in map(str, STEPS)
                                               if v["train"].get(s) != "missing"]}
                      for name, v in payload["runs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
