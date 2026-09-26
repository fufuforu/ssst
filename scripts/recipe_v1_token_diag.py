#!/usr/bin/env python3
"""Read-only token-level diagnostic for the recipe-v1 checkpoint.

Reports, separately for thing-bearing and rest-only tokens: the auxiliary
assignment CE, the argmax agreement, the token counts; plus effective group
usage, void mass fraction, alpha and reconstruction PSNR on the fixed windows.
Uses the model's own target/CE code; no training, no optimizer step, no writes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.eval_group_locusgs import build_train_entries  # noqa: E402
from scripts.group_eval_v2 import build_val_entries, forward_group  # noqa: E402


def token_stats(model, batch, alpha):
    """thing/rest split CE + agreement from the model's own target code."""
    group = model.layer10_group
    context_views = tuple(range(int(model.opt.num_input_views)))
    instance = model.group_loss_terms(
        batch, model.last_gaussians, model.last_decoder_input, context_views
    )
    contributions = model.token_instance_contributions(
        model.last_gaussians, batch, alpha, context_views
    )
    target, kept, _ = model.assignment_target(
        contributions, instance["segment_keys"], instance["matched_rows"]
    )
    log_prob = torch.log_softmax(group["slot_logits"][0].float(), dim=-1)
    ce = -(target * log_prob).sum(-1)
    agree = (log_prob.argmax(-1) == target.argmax(-1)).float()
    thing = kept & (target[:, : model.num_groups].sum(-1) > 0)
    rest = kept & (~thing)
    out = {
        "thing_tokens": int(thing.sum()), "rest_tokens": int(rest.sum()),
        "dropped_tokens": int((~kept).sum()),
        "thing_ce": float(ce[thing].mean()) if bool(thing.any()) else None,
        "rest_ce": float(ce[rest].mean()) if bool(rest.any()) else None,
        "thing_agreement": float(agree[thing].mean()) if bool(thing.any()) else None,
        "rest_agreement": float(agree[rest].mean()) if bool(rest.any()) else None,
        "overall_agreement": float(agree[kept].mean()) if bool(kept.any()) else None,
        "void_probability_mean": float(group["slot_prob"][0, :, -1].mean()),
        "effective_groups": int(
            (group["slot_prob"][0, :, : model.num_groups].mean(0) > 1.0 / model.num_groups).sum()
        ),
    }
    return out


def run_window(model, opt, entry, device, want_mask=False):
    batch = entry["batch"]
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
    model.last_gaussians = forward["output"]["gaussians"]
    from tokengs.models.input_types import ModelInputDecoder

    model.last_decoder_input = ModelInputDecoder(
        cam_view=batch["cam_view_all"], intrinsics=batch["intrinsics_all"]
    )
    stats = token_stats(model, batch, forward["masks"]["alpha"])
    stats["alpha_mean"] = float(forward["masks"]["alpha"].mean())
    stats["scene"] = entry["scene"]
    if want_mask:
        from scripts.audit_group_routing import fragmentation

        semantic = batch["semantic_label_all"][0].long().cpu().numpy()
        instance = batch["instance_label_all"][0].long().cpu().numpy()
        ious = []
        for view in (2, 3):
            sem, ins = semantic[view], instance[view]
            packed = (sem + 1) * 1000 + ins
            visible = (sem >= 2) & (sem < 20) & (ins > 0)
            mass = forward["masks"]["group_mass"][0, view].float().cpu().numpy()
            keys = sorted(set(int(k) for k in np.unique(packed[visible])))
            for key in keys:
                truth = visible & (packed == key)
                others = [visible & (packed == other) for other in keys if other != key]
                ious.append(fragmentation(mass, truth, others)["iou1"])
        stats["best_over_groups_iou_mean"] = float(np.mean(ious)) if ious else None
        stats["n_novel_records"] = len(ious)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--out", default="group_plus/recipe_v1/token_diag.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    opt = config_defaults["train_siu3r_group_locusgs_ab"].evolve(
        seed=42, group_arm="g0", group_recipe=True,
        group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    results = {"training_windows": [], "validation_windows": []}
    for entry in build_train_entries(opt, split, plan, device, args.windows):
        stats = run_window(model, opt, entry, device, want_mask=True)
        results["training_windows"].append(stats)
        print(f"[tok] train {stats['scene']}: thing {stats['thing_tokens']} "
              f"rest {stats['rest_tokens']} | CE {stats['thing_ce']}/{stats['rest_ce']} "
              f"| agree {stats['thing_agreement']}/{stats['rest_agreement']} "
              f"| groups {stats['effective_groups']} | mask IoU {stats.get('best_over_groups_iou_mean')}",
              flush=True)
    for entry in build_val_entries(opt, split, device):
        stats = run_window(model, opt, entry, device, want_mask=True)
        results["validation_windows"].append(stats)
        print(f"[tok] val {stats['scene']}: thing {stats['thing_tokens']} "
              f"rest {stats['rest_tokens']} | CE {stats['thing_ce']}/{stats['rest_ce']} "
              f"| agree {stats['thing_agreement']}/{stats['rest_agreement']} "
              f"| groups {stats['effective_groups']} | mask IoU {stats.get('best_over_groups_iou_mean')}",
              flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[tok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
