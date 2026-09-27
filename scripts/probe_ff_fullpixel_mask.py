#!/usr/bin/env python3
"""structure_probe_v2 §4.A: feed-forward probe with a full-pixel mask loss.

Only used when the token-assignment oracle passes.  Everything matches the
completed Probe-I run (same fixed sample, same frozen G0+ step6000 start, only
``groups.*`` trainable, AdamW lr 1e-4, weight_decay 0, clip 1.0, 1200 steps,
identical Hungarian matching, identical 21-way CE with the no-object weight,
identical background loss and identical 2/5/5 loss weights) **except one thing**:
the post-match mask BCE and soft Dice are computed over *all* pixels of the two
context views instead of the fixed 4096-point sample.

The script also asserts that the original sampled loss still reproduces the
step-0 value recorded by the completed Probe-I run, so any change is attributable
to the full-pixel term alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models.group_locusgs import (  # noqa: E402
    NO_OBJECT_CLASS,
    SIU3R_SEMANTIC_CLASS_COUNT,
    NO_OBJECT_CE_WEIGHT,
    _INSTANCE_WEIGHT_CLASS_CE,
    _INSTANCE_WEIGHT_MASK_BCE,
    _INSTANCE_WEIGHT_MASK_DICE,
    build_context_segments,
)
from tokengs.models.input_types import ModelInputDecoder  # noqa: E402
from tokengs.models.ssst_loss import hungarian_match  # noqa: E402
from scripts.group_eval_v2 import forward_group, group_predictions_v2, group_score_table  # noqa: E402
from scripts.probe_group_capacity import load_g0plus, sample_batch  # noqa: E402

G0PLUS = "workspace_group_plus/arm_g0plus/ckpt_step6000"
REQUIRED_SAMPLE_SHA = "98a4d35d97eea33169d2fb4f7346ed7c128c0385a385e223ab36732690bc83df"
PROBE_I_STEP0_LOSS = 4.444489002227783


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_pixel_terms(model, batch, gaussians, decoder_input, context_views):
    """`group_loss_terms` with all-pixel BCE/Dice; everything else unchanged."""
    group = model.layer10_group
    mask_decoder = decoder_input.select_batch(slice(None), slice(0, len(context_views)))
    rendered = model.render_group_masks(gaussians, group["slot_prob"], mask_decoder)
    group_mass = rendered["group_mass"].permute(0, 2, 1, 3, 4)
    mask_logits = torch.logit(group_mass.clamp(1e-5, 1.0 - 1e-5))
    class_logits = torch.cat([group["class_logits"], group["objectness"].unsqueeze(-1)],
                             dim=-1)
    gt_classes, gt_masks = build_context_segments(
        batch["semantic_label_all"], batch["instance_label_all"], tuple(context_views))
    things = [(labels[labels >= 2], masks[labels >= 2])
              for labels, masks in zip(gt_classes, gt_masks)]
    labels, masks = things[0]
    rows, cols = hungarian_match(class_logits[0], mask_logits[0], labels, masks)
    target = class_logits.new_full((class_logits.shape[1],),
                                   NO_OBJECT_CLASS, dtype=torch.long)
    if rows.numel():
        target[rows] = labels[cols]
    empty_weight = class_logits.new_ones(SIU3R_SEMANTIC_CLASS_COUNT + 1)
    empty_weight[NO_OBJECT_CLASS] = NO_OBJECT_CE_WEIGHT
    ce = F.cross_entropy(class_logits[0], target, weight=empty_weight)
    zero = class_logits.sum() * 0.0
    bce_terms, dice_terms = [], []
    for row, col in zip(rows.tolist(), cols.tolist()):
        # ALL pixels of the two context views - the single deliberate change
        logits = mask_logits[0, row]
        truth = masks[col].float()
        bce_terms.append(F.binary_cross_entropy_with_logits(logits, truth))
        probability = torch.sigmoid(logits)
        intersection = (probability * truth).sum()
        dice_terms.append(1.0 - (2.0 * intersection + 1.0) / (
            probability.sum() + truth.sum() + 1.0))
    bce = torch.stack(bce_terms).mean() if bce_terms else zero
    dice = torch.stack(dice_terms).mean() if dice_terms else zero
    loss = (_INSTANCE_WEIGHT_CLASS_CE * ce + _INSTANCE_WEIGHT_MASK_BCE * bce
            + _INSTANCE_WEIGHT_MASK_DICE * dice)
    background = model._background_loss_of(batch, rendered, context_views)
    if background is not None:
        background_loss, _ = background
        loss = loss + model.bg_weight * background_loss
    return {"loss_inst_total": loss, "instance": loss, "matched": int(rows.numel()),
            "n_gt": int(labels.numel())}


def sentinel_report(model, opt, batch, sample):
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
    sem = batch["semantic_label_all"][0].long().cpu().numpy()
    ins = batch["instance_label_all"][0].long().cpu().numpy()
    mass = forward["masks"]["group_mass"][0].float().cpu()
    table = group_score_table(forward)
    classes = table["class_argmax20"].long().cpu().numpy()
    p_thing = table["p_thing"].float().cpu().numpy()
    out = {"per_sentinel": {}, "reader": {}}
    best_queries = []
    for sentinel in sample["sentinels"]:
        key, cls = int(sentinel["packed_key"]), int(sentinel["internal_class"])
        best_iou, best_q = 0.0, None
        for view in (0, 1):
            packed = (sem[view] + 1) * 1000 + ins[view]
            truth = ((sem[view] >= 2) & (sem[view] < 20) & (ins[view] > 0)
                     & (packed == key))
            if not truth.any():
                continue
            for query in range(mass.shape[1]):
                hard = mass[view, query].numpy() > 0.5
                if int(hard.sum()) < 50:
                    continue
                union = int((hard | truth).sum())
                if not union:
                    continue
                iou = int((hard & truth).sum()) / union
                if iou > best_iou:
                    best_iou, best_q = iou, int(query)
        best_queries.append(best_q)
        out["per_sentinel"][str(key)] = {
            "gt_class": cls, "best_raw_query": best_q, "best_raw_iou": best_iou,
            "best_query_class": None if best_q is None else int(classes[best_q]),
            "best_query_class_correct": (best_q is not None
                                         and int(classes[best_q]) == cls),
            "best_query_p_thing": None if best_q is None else float(p_thing[best_q])}
    out["distinct_queries"] = len(set(q for q in best_queries if q is not None)) == 2
    for scope, views in (("context", (0, 1)), ("novel", (2, 3))):
        tp = fp = fn = hits = 0
        for view in views:
            predictions, _ = group_predictions_v2(forward, view)
            targets, gt_class = [], []
            packed = (sem[view] + 1) * 1000 + ins[view]
            visible = (sem[view] >= 2) & (sem[view] < 20) & (ins[view] > 0)
            for key in sorted(set(int(k) for k in np.unique(packed[visible]))):
                truth = visible & (packed == key)
                targets.append(truth)
                gt_class.append(int(sem[view][truth][0]))
            used = set()
            for prediction in sorted(predictions, key=lambda p: -p["score"]):
                best, best_i = 0.0, None
                for i, truth in enumerate(targets):
                    if i in used:
                        continue
                    union = int((prediction["mask"] | truth).sum())
                    if not union:
                        continue
                    iou = int((prediction["mask"] & truth).sum()) / union
                    if iou > best:
                        best, best_i = iou, i
                if best_i is not None and best >= 0.5:
                    used.add(best_i)
                    tp += 1
                    if int(prediction["class"]) == gt_class[best_i]:
                        hits += 1
                else:
                    fp += 1
            fn += len(targets) - len(used)
        out["reader"][scope] = {"tp": tp, "category_aware_hits": hits, "fp": fp, "fn": fn}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/structure_probe_v2/"
                                        "probe_ff_fullpixel.json")
    parser.add_argument("--sample", default="group_plus/structure_probe_v1/sample.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-steps", type=int, nargs="*", default=[0, 100, 300, 600, 1200])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    if sample["sha256"] != REQUIRED_SAMPLE_SHA:
        raise SystemExit("sample SHA mismatch")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    opt, model = load_g0plus(args.preset, args.seed, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if n.startswith("groups.")]
    for _, parameter in trainable:
        parameter.requires_grad_(True)
    model.eval()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    batch = sample_batch(opt, split, sample, device)
    decoder_input = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                      intrinsics=batch["intrinsics_all"])
    checkpoint_sha_before = sha256_file(Path(G0PLUS) / "model.pt")

    report: dict = {"scope": "single-sample feed-forward probe; success here proves local "
                             "fittability only, not official generalisation",
                    "sample_sha256": sample["sha256"],
                    "config": {"lr": args.lr, "weight_decay": 0, "steps": args.steps,
                               "clip": 1.0, "trainable": "groups.* only",
                               "change_vs_probe_i": "mask BCE/Dice over all pixels of the "
                                                    "two context views instead of the 4096 "
                                                    "point sample"}}
    # 1. the original sampled loss must still reproduce the Probe-I step-0 value
    model.eval()
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
        original = model.group_loss_terms(batch, forward["output"]["gaussians"],
                                          decoder_input, (0, 1))
    repro = float(original["loss_inst_total"])
    report["original_sampled_loss_step0"] = repro
    report["probe_i_recorded_step0"] = PROBE_I_STEP0_LOSS
    report["original_loss_reproduced"] = abs(repro - PROBE_I_STEP0_LOSS) <= 1e-6
    if not report["original_loss_reproduced"]:
        raise SystemExit(f"the sampled loss ({repro}) no longer reproduces the completed "
                         f"Probe-I step-0 value ({PROBE_I_STEP0_LOSS})")
    # 2. the new full-pixel term must be finite and reach groups.*
    optimizer.zero_grad(set_to_none=True)
    terms = full_pixel_terms(model, batch, forward["output"]["gaussians"], decoder_input, (0, 1))
    if not torch.isfinite(terms["loss_inst_total"]):
        raise SystemExit("full-pixel loss is not finite at step 0")
    terms["loss_inst_total"].backward()
    grads = {n: (None if p.grad is None else float(p.grad.norm())) for n, p in trainable}
    report["full_pixel_loss_step0"] = float(terms["loss_inst_total"])
    report["full_pixel_grad_norm_total"] = math.sqrt(sum(g * g for g in grads.values()))
    if not all(g is not None and math.isfinite(g) for g in grads.values()):
        raise SystemExit("full-pixel loss does not produce finite gradients for groups.*")
    optimizer.zero_grad(set_to_none=True)

    steps = 10 if args.smoke else args.steps
    curve = []
    for step in range(steps + 1):
        if step in args.log_steps or step == steps:
            model.eval()
            with torch.no_grad():
                forward = forward_group(model, batch, opt)
                terms = full_pixel_terms(model, batch, forward["output"]["gaussians"],
                                         decoder_input, (0, 1))
            metrics = sentinel_report(model, opt, batch, sample)
            row = {"step": step, "loss_inst_total": float(terms["loss_inst_total"]),
                   **metrics}
            curve.append(row)
            print(f"[ff] step {step} loss {row['loss_inst_total']:.4f} "
                  f"ctx {row['reader']['context']} novel {row['reader']['novel']} "
                  f"sentinels { {k: round(v['best_raw_iou'],3) for k,v in row['per_sentinel'].items()} }",
                  flush=True)
        if step == steps or args.smoke:
            break
        optimizer.zero_grad(set_to_none=True)
        forward = forward_group(model, batch, opt)
        terms = full_pixel_terms(model, batch, forward["output"]["gaussians"],
                                 decoder_input, (0, 1))
        if not torch.isfinite(terms["loss_inst_total"]):
            raise SystemExit(f"non-finite loss at step {step}")
        terms["loss_inst_total"].backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for _, p in trainable):
            raise SystemExit(f"non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        optimizer.step()

    last = curve[-1]
    sent = last["per_sentinel"]
    ok = (all(v["best_raw_iou"] >= 0.5 and v["best_query_class_correct"]
              for v in sent.values())
          and last["distinct_queries"]
          and last["reader"]["context"]["tp"] >= 2
          and last["reader"]["novel"]["tp"] >= 1)
    report.update({"curve": curve, "final": last, "feed_forward_pair_pass": bool(ok),
                   "checkpoint": {"sha256_before": checkpoint_sha_before,
                                  "sha256_after": sha256_file(Path(G0PLUS) / "model.pt")},
                   "smoke": bool(args.smoke), "elapsed_seconds": time.time() - started})
    report["checkpoint"]["unchanged"] = (report["checkpoint"]["sha256_before"]
                                         == report["checkpoint"]["sha256_after"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if not args.smoke:
        delta = {n: (p.detach().cpu() - torch.load(Path(G0PLUS) / "model.pt",
                                                   map_location="cpu",
                                                   weights_only=False)["model"][n])
                 for n, p in trainable}
        torch.save({"delta": delta, "steps": steps, "config": report["config"]},
                   out.parent / "probe_ff_fullpixel_delta.pt")
    print(json.dumps({"feed_forward_pair_pass": report["feed_forward_pair_pass"],
                      "distinct_queries": last.get("distinct_queries"),
                      "reader": last["reader"], "checkpoint_unchanged":
                      report["checkpoint"]["unchanged"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
