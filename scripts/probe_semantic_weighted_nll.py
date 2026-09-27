#!/usr/bin/env python3
"""structure_probe_v2 §5: class-weighted semantic NLL contrast on the fixed sample.

Run only because the corrected Probe-S read-out did not reach the local semantic
gate.  Single variable versus Probe-S: the per-pixel NLL is weighted by
``1/sqrt(freq)`` of the fixed sample's four views' valid GT class frequencies
(normalised to mean 1 over the classes that appear, clipped to [0.25, 4]; absent
classes do not take part in the average).  Everything else is unchanged: only
``attributes.semantic.*`` is trained, AdamW lr 1e-3, weight_decay 0, 800 steps.

Judged with the corrected valid-confusion IoU convention of §2 (prediction and
GT both restricted to ``GT in 0..19 and alpha > 0.05``; a class with no GT in a
view still accrues FP).
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

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from scripts.structure_probe_v2_corrected import (  # noqa: E402
    iou_from_binary,
    valid_confusion,
)
from scripts.group_eval_v2 import forward_group  # noqa: E402
from scripts.probe_group_capacity import load_g0plus, sample_batch  # noqa: E402

G0PLUS = "workspace_group_plus/arm_g0plus/ckpt_step6000"
SEMANTIC_CLASSES = 20
IGNORE = 255
REQUIRED_SAMPLE_SHA = "98a4d35d97eea33169d2fb4f7346ed7c128c0385a385e223ab36732690bc83df"


def class_weights(semantic: torch.Tensor, alpha: torch.Tensor, views=4):
    """1/sqrt(freq) weights over the sample's valid GT, normalised and clipped."""
    freq = torch.zeros(SEMANTIC_CLASSES)
    for view in range(views):
        valid = ((semantic[0, view] != IGNORE) & (semantic[0, view] >= 0)
                 & (semantic[0, view] < SEMANTIC_CLASSES)
                 & (alpha[0, view, 0] > 0.05))
        if not bool(valid.any()):
            continue
        counts = torch.bincount(semantic[0, view][valid], minlength=SEMANTIC_CLASSES)
        freq += counts.float().cpu()
    present = freq > 0
    raw = torch.zeros_like(freq)
    raw[present] = 1.0 / freq[present].sqrt()
    weights = torch.ones_like(freq)
    weights[present] = torch.clamp(raw[present] / raw[present].mean(), 0.25, 4.0)
    return weights, freq, present


def weighted_nll(prob, alpha, semantic, weights):
    """mean over valid pixels of w[gt] * (-log p[gt]); views averaged like the repo."""
    target = semantic.long().clamp(0, SEMANTIC_CLASSES - 1).unsqueeze(2)
    gathered = prob.gather(2, target).squeeze(2).clamp_min(1e-12)
    losses, supervised = [], 0
    for view in range(prob.shape[1]):
        valid = ((semantic[:, view] != IGNORE) & (semantic[:, view] >= 0)
                 & (semantic[:, view] < SEMANTIC_CLASSES)
                 & (alpha[:, view, 0] > 0.05))
        supervised += int(valid.sum())
        if not bool(valid.any()):
            continue
        w = weights.to(gathered.device)[semantic[:, view][valid]]
        losses.append((-gathered[:, view].log()[valid] * w).mean())
    if not losses:
        return prob.sum() * 0.0, {"supervised_pixels": 0}
    return torch.stack(losses).mean(), {"supervised_pixels": supervised}


def corrected_scores(prob, semantic, alpha):
    pred = prob.argmax(2)[0].long().cpu().numpy()
    sem = semantic.long().cpu().numpy()[0]
    a = alpha[0, :, 0].float().cpu().numpy()
    out = {}
    for scope, views in (("context", (0, 1)), ("novel", (2, 3))):
        conf = np.zeros((SEMANTIC_CLASSES, SEMANTIC_CLASSES), dtype=np.int64)
        for view in views:
            part, _ = valid_confusion(pred[view], sem[view], a[view])
            conf += part
        ious = []
        for cls in range(SEMANTIC_CLASSES):
            tp = int(conf[cls, cls])
            fn = int(conf[cls, :].sum() - tp)
            fp = int(conf[:, cls].sum() - tp)
            ious.append(iou_from_binary(tp, fp, fn))
        present = [v for v in ious if not math.isnan(v)]
        out[scope] = {"per_class_iou": ious,
                      "miou_present": float(np.mean(present)) if present else None}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/structure_probe_v2/"
                                        "probe_semantic_weighted.json")
    parser.add_argument("--sample", default="group_plus/structure_probe_v1/sample.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-steps", type=int, nargs="*", default=[0, 100, 200, 400, 800])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    if sample["sha256"] != REQUIRED_SAMPLE_SHA:
        raise SystemExit("sample SHA mismatch")
    sentinel_classes = [int(s["internal_class"]) for s in sample["sentinels"]]
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    opt, model = load_g0plus(args.preset, args.seed, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters()
                 if n.startswith("attributes.semantic.")]
    for _, parameter in trainable:
        parameter.requires_grad_(True)
    model.eval()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    batch = sample_batch(opt, split, sample, device)
    semantic = batch["semantic_label_all"]
    with torch.no_grad():
        forward0 = forward_group(model, batch, opt)
    weights, freq, present = class_weights(semantic, forward0["masks"]["alpha"])
    report: dict = {"scope": "single-variable semantic diagnostic; new experiment, does "
                             "not rewrite the previous round's conclusion",
                    "sample_sha256": sample["sha256"],
                    "config": {"lr": args.lr, "weight_decay": 0, "steps": args.steps,
                               "trainable": "attributes.semantic.*",
                               "change_vs_probe_S": "NLL weighted by 1/sqrt(class frequency)",
                               "weights": weights.tolist(),
                               "class_frequencies": freq.tolist(),
                               "present_classes": present.nonzero().flatten().tolist(),
                               "weight_clip": [0.25, 4.0],
                               "weight_mean_over_present": float(weights[present].mean())}}
    frozen = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name.startswith("attributes.semantic."):
            continue
        frozen.update(name.encode())
        frozen.update(parameter.detach().float().cpu().numpy().tobytes())

    steps = 10 if args.smoke else args.steps
    curve = []
    for step in range(steps + 1):
        if step in args.log_steps or step == steps:
            model.eval()
            with torch.no_grad():
                forward = forward_group(model, batch, opt)
                loss, stats = weighted_nll(forward["semantic_prob"],
                                           forward["masks"]["alpha"], semantic, weights)
            scores = corrected_scores(forward["semantic_prob"], semantic,
                                      forward["masks"]["alpha"])
            row = {"step": step, "weighted_nll": float(loss),
                   "supervised_pixels": stats["supervised_pixels"],
                   "context_sentinel_iou": {str(c): scores["context"]["per_class_iou"][c]
                                            for c in sentinel_classes},
                   "novel_sentinel_iou": {str(c): scores["novel"]["per_class_iou"][c]
                                          for c in sentinel_classes},
                   "context_miou_present": scores["context"]["miou_present"]}
            curve.append(row)
            print(f"[wS] step {step} nll {row['weighted_nll']:.4f} "
                  f"ctx {row['context_sentinel_iou']} novel {row['novel_sentinel_iou']}",
                  flush=True)
        if step == steps or args.smoke:
            break
        optimizer.zero_grad(set_to_none=True)
        forward = forward_group(model, batch, opt)
        loss, _ = weighted_nll(forward["semantic_prob"], forward["masks"]["alpha"],
                               semantic, weights)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for _, p in trainable):
            raise SystemExit(f"non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        optimizer.step()

    frozen_after = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name.startswith("attributes.semantic."):
            continue
        frozen_after.update(name.encode())
        frozen_after.update(parameter.detach().float().cpu().numpy().tobytes())
    last = curve[-1]
    local_pass = (all(v >= 0.50 for v in last["context_sentinel_iou"].values())
                  and all(v >= 0.25 for v in last["novel_sentinel_iou"].values()))
    report.update({"curve": curve, "final": last,
                   "semantic_local_pass_weighted": bool(local_pass),
                   "frozen_params_unchanged": frozen.hexdigest() == frozen_after.hexdigest(),
                   "smoke": bool(args.smoke),
                   "elapsed_seconds": time.time() - started})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"semantic_local_pass_weighted": report["semantic_local_pass_weighted"],
                      "final": last, "frozen_params_unchanged":
                      report["frozen_params_unchanged"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
