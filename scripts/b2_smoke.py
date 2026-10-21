#!/usr/bin/env python3
"""B2 fail-fast smoke: only the token-assignment CE's shared gradient changes.

Builds two models from the *same* step-0 state (the matched control and the
stop-gradient arm), then asserts the pre-registered conditions:

* parameters bitwise identical (the only training variable is the flag);
* main and auxiliary slot logits identical at the same parameter state
  (atol 1e-6 / rtol 1e-6) - the auxiliary term is only a *remeasured* version of
  the same function;
* the auxiliary gradient into the shared token/anchor/encoder is None or exactly
  zero for the stop-gradient arm, and non-zero for the control;
* the main gradient into the shared representation is non-zero for both;
* the auxiliary gradient into the group head is finite and non-zero;
* the total forward loss is identical between the two arms.

Read-only: no optimizer.step, no checkpoint write.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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
from scripts.group_eval_v2 import forward_group  # noqa: E402
from scripts.train_group_locusgs import SHARED_PARAM_PREFIXES  # noqa: E402
from scripts.train_object_locusgs import move  # noqa: E402


def build(opt, state, device, *, stop):
    arm_opt = opt.evolve(group_recipe_assign_stop_shared_grad=bool(stop))
    model = model_registry[arm_opt.model_type](arm_opt).to(device)
    model.load_state_dict(state, strict=True)
    model.train()
    return arm_opt, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-init", required=True)
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--head-mode", default="pure4")
    parser.add_argument("--instance-outer-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    opt = config_defaults[args.preset].evolve(
        seed=args.seed, group_arm="g0", group_bg_supervision=False, group_recipe=True,
        group_recipe_head_mode=str(args.head_mode),
        group_recipe_seg_weight=float(args.instance_outer_weight),
        group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    state = torch.load(Path(args.control_init) / "model.pt", map_location="cpu",
                       weights_only=False)["model"]
    control_opt, control = build(opt, state, device, stop=False)
    test_opt, test = build(opt, state, device, stop=True)

    report: dict = {"checks": {}}
    delta = max(float((test.state_dict()[k].float()
                       - control.state_dict()[k].float()).abs().max())
                for k in state)
    report["step0_param_max_delta"] = delta
    report["checks"]["step0_params_identical"] = delta == 0.0

    provider = SIU3RProcessedProvider(opt, root=split["train_root"],
                                      subset=split["train_scenes"], training=True, rank=0)
    index = {p.name: i for i, p in enumerate(provider.dataset.sample_list)}
    item = plan["entries"][0]
    provider.pin_pair(scene_id=item["scene"], context_frame_ids=item["context"],
                      novel_frame_ids=item["novel"], pair_iou=item["pair_iou"])
    batch = move(default_collate([provider[index[item["scene"]]]]), device)

    for tag, model in (("control", control), ("stop", test)):
        model.zero_grad(set_to_none=True)
        main, aux, stats = model.recipe_probe_losses(batch, step=500)
        named = dict(model.named_parameters())
        head = [n for n in named if n.startswith("groups.") and named[n].requires_grad]
        shared = [n for n in named if n.startswith(SHARED_PARAM_PREFIXES)
                  and named[n].requires_grad]
        g_main = torch.autograd.grad(main, [named[n] for n in head + shared],
                                     retain_graph=True, allow_unused=True)
        g_aux = torch.autograd.grad(aux, [named[n] for n in head + shared],
                                    retain_graph=True, allow_unused=True)
        table = {}
        for name, gm, ga in zip(head + shared, g_main, g_aux):
            table[name] = {
                "is_shared": name.startswith(SHARED_PARAM_PREFIXES),
                "main": None if gm is None else float(gm.norm()),
                "aux": None if ga is None else float(ga.norm()),
            }
        report[tag] = {
            "main_norm_head": math.sqrt(sum(v["main"] ** 2 for v in table.values()
                                            if not v["is_shared"] and v["main"])),
            "aux_norm_head": math.sqrt(sum(v["aux"] ** 2 for v in table.values()
                                           if not v["is_shared"] and v["aux"])),
            "main_norm_shared": math.sqrt(sum(v["main"] ** 2 for v in table.values()
                                              if v["is_shared"] and v["main"])),
            "aux_norm_shared": math.sqrt(sum(v["aux"] ** 2 for v in table.values()
                                             if v["is_shared"] and v["aux"])),
            "aux_shared_all_none_or_zero": all(
                v["aux"] is None or v["aux"] == 0.0
                for v in table.values() if v["is_shared"]),
            "aux_any_nonfinite": any(
                v["aux"] is not None and not math.isfinite(v["aux"])
                for v in table.values()),
            "forward_aux": float(aux.detach()), "forward_main": float(main.detach()),
        }
    # slot-logit equality at the same parameter state
    with torch.no_grad():
        f_control = forward_group(control, batch, control_opt)
        f_test = forward_group(test, batch, test_opt)
    logits_control = f_control["group"]["slot_logits"]
    logits_test = f_test["group"]["slot_logits"]
    report["slot_logits_max_abs_diff"] = float(
        (logits_control - logits_test).abs().max())
    report["checks"]["slot_logits_agree_1e-6"] = bool(torch.allclose(
        logits_control, logits_test, atol=1e-6, rtol=1e-6))

    c, t = report["control"], report["stop"]
    report["checks"]["aux_shared_none_or_zero_for_stop"] = t["aux_shared_all_none_or_zero"]
    report["checks"]["aux_shared_nonzero_for_control"] = c["aux_norm_shared"] > 0
    report["checks"]["main_shared_nonzero_for_both"] = (
        c["main_norm_shared"] > 0 and t["main_norm_shared"] > 0)
    report["checks"]["aux_head_nonzero_for_stop"] = t["aux_norm_head"] > 0
    report["checks"]["aux_grads_finite"] = not (t["aux_any_nonfinite"] or c["aux_any_nonfinite"])
    report["checks"]["forward_aux_identical"] = abs(c["forward_aux"] - t["forward_aux"]) <= 1e-6
    report["checks"]["forward_main_identical"] = abs(c["forward_main"] - t["forward_main"]) <= 1e-6

    with torch.no_grad():
        _, m_control = control.step_loss(batch, step=500, phase="train")
        _, m_test = test.step_loss(batch, step=500, phase="train")
    report["total_loss_control"] = float(m_control["loss"])
    report["total_loss_stop"] = float(m_test["loss"])
    report["checks"]["total_loss_identical"] = (
        abs(report["total_loss_control"] - report["total_loss_stop"]) <= 1e-6)

    report["all_checks_passed"] = all(
        v for v in report["checks"].values() if isinstance(v, bool))
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("checks", "all_checks_passed", "slot_logits_max_abs_diff",
                               "total_loss_control", "total_loss_stop")}, indent=1))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
