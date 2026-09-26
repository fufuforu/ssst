#!/usr/bin/env python3
"""Field-by-field diff of the *effective* training config: recipe_v1 vs recipe_v2.

Both option objects are built with exactly the same `evolve(...)` call as
`scripts/train_group_locusgs.py`; the only intended difference is the main
instance/group outer weight (0.1 -> 0.05).  Output paths and experiment
identifiers are excluded as metadata.  The script asserts that the remaining
diff contains exactly that one field, so the "single variable" claim is checked
rather than asserted by hand.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tokengs.options import config_defaults

META = {"workspace", "experiment_name"}
PRESET = "train_siu3r_group_locusgs_ab"


def build(seg_weight: float, out_dir: str, exp: str):
    preset = config_defaults[PRESET]
    return preset.evolve(
        seed=42, group_arm="g0", group_bg_supervision=False, group_bg_loss_weight=1.0,
        group_recipe=True, group_recipe_seg_weight=float(seg_weight),
        group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        lr=1e-4, pct_start_steps=2000, batch_size=1, num_workers=0,
        num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        workspace=out_dir, experiment_name=exp, init_checkpoint=None,
    )


def main() -> int:
    v1 = build(0.1, "workspace_group_plus/recipe_v1/run", "siu3r_group_locusgs_g0_v1")
    v2 = build(0.05, "workspace_group_plus/recipe_v2/run", "siu3r_group_locusgs_g0_v2")
    a, b = dataclasses.asdict(v1), dataclasses.asdict(v2)
    keys = sorted(k for k in a if k not in META)
    diff = {k: {"recipe_v1": a[k], "recipe_v2": b[k]} for k in keys if a[k] != b[k]}
    payload = {
        "role": "recipe_v2 single arm: recipe_v1 with the main instance/group outer "
                "weight 0.1 -> 0.05; every other field identical (single variable)",
        "preset": PRESET,
        "excluded_metadata": sorted(META),
        "n_fields_compared": len(keys),
        "differing_fields": diff,
        "single_variable_ok": list(diff) == ["group_recipe_seg_weight"],
        "recipe_v1": {
            "out_dir": "workspace_group_plus/recipe_v1/run",
            "checkpoint_step6000": "workspace_group_plus/recipe_v1/run/ckpt_step6000",
            "instance_outer_weight": 0.1, "assign_coef": 0.2, "assign_every": 1,
            "commit": "de37ef0", "train_job": 55512, "eval_job": 55591,
        },
        "recipe_v2": {
            "out_dir": "workspace_group_plus/recipe_v2/run",
            "instance_outer_weight": 0.05, "assign_coef": 0.2, "assign_every": 1,
            "cold_start": "workspace_group_locusgs/arm_g0/ckpt_step0",
            "new_module_seed": 1743,
        },
        "unchanged_explicitly": {
            "seg_ramp": "min(1, step/1500) (linear, steps 1..1500)",
            "assign_term": "seg_ramp * 0.2 * CE_balanced (NOT scaled by the instance weight)",
            "semantic": "0.05, min(1, step/2000)",
            "deep_group_decoder": "4 layers, width 1024, 8 heads, MLP 2048, dropout 0",
            "slot_softmax": "101 columns, 101st logit fixed 0",
            "hungarian_and_mask_sampling": "unchanged (BCE/Dice on the 4096-point sample)",
            "gt_free_thresholds": "P(thing)>=0.5, thing class, mask>0.5, area>=50 px",
            "optimizer": "AdamW, lr 1e-4, warmup 2000, cosine to 2% over 6000, clip 1.0",
            "seed": 42, "plan_sha256": "a2a65c1382da0307a68fb6d27e5c1345aa78b46331acb2927e88b47a3c3d08bb",
        },
    }
    out = Path("group_plus/recipe_v2/config_diff.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({"differing_fields": diff,
                      "single_variable_ok": payload["single_variable_ok"],
                      "n_fields_compared": len(keys)}, indent=1))
    if not payload["single_variable_ok"]:
        raise SystemExit("config diff is not the single intended variable; refusing to train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
