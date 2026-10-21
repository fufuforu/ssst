#!/usr/bin/env python3
"""structure_probe_v2 §2: correct the previous round's diagnostic read-outs.

Inference only — the frozen G0+ step6000 checkpoint is combined with the saved
head deltas (``probe_S_delta.pt`` / ``probe_I_delta.pt``) and the two probes are
re-scored under the convention this round pre-registers:

* semantic: a 20x20 confusion accumulated over the view set, where
  ``valid = (GT in 0..19) and alpha > 0.05`` and **both** the prediction and the
  GT are counted only inside ``valid``; a class with no GT in some view still
  accrues FP there.  IoU is computed once per class from the summed confusion.
* instance: the frozen reader (P(thing) >= 0.5, argmax is a thing class, raw mass
  > 0.5, area >= 50), candidates sorted by score, one-to-one greedy matching.
  A **category-aware** TP additionally requires the predicted class to equal the
  GT class; the previous round only counted the category-agnostic TP.  Both
  sentinels are reported individually - a scene-level TP count is never used as
  a substitute for "both sentinels were hit".

Nothing here updates the model or rewrites any historical JSON.
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
    forward_group,
    group_predictions_v2,
    group_score_table,
)
from scripts.train_object_locusgs import move  # noqa: E402

G0PLUS = "workspace_group_plus/arm_g0plus/ckpt_step6000"
SEMANTIC_CLASSES = 20
IGNORE = 255
REQUIRED_SAMPLE_SHA = "98a4d35d97eea33169d2fb4f7346ed7c128c0385a385e223ab36732690bc83df"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_g0plus(preset: str, seed: int, device):
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm="g0", group_bg_supervision=True, group_recipe=False,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(G0PLUS) / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True)
    model.eval()
    return opt, model


def apply_delta(model, delta_path: Path, label: str) -> dict:
    payload = torch.load(delta_path, map_location="cpu", weights_only=False)
    delta, names = payload["delta"], payload.get("names", list(payload["delta"]))
    named = dict(model.named_parameters())
    info = {"label": label, "path": str(delta_path), "sha256": sha256_file(delta_path),
            "n_tensors": len(delta), "steps": payload.get("steps"),
            "names_match_model": all(n in named for n in names), "tensors": {}}
    for name, value in delta.items():
        if name not in named:
            raise SystemExit(f"{label}: delta tensor {name} is not a model parameter")
        if tuple(named[name].shape) != tuple(value.shape):
            raise SystemExit(f"{label}: shape mismatch for {name}: "
                             f"{tuple(value.shape)} vs {tuple(named[name].shape)}")
        with torch.no_grad():
            named[name].add_(value.to(device=named[name].device, dtype=named[name].dtype))
        info["tensors"][name] = {"shape": list(value.shape),
                                 "absmax": float(value.abs().max())}
    return info


def batch_for_sample(opt, split_path: Path, sample: dict, device):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    provider = SIU3RProcessedProvider(opt, root=split["train_root"],
                                      subset=split["train_scenes"], training=True, rank=0)
    index = {p.name: i for i, p in enumerate(provider.dataset.sample_list)}
    provider.pin_pair(scene_id=sample["scene"], context_frame_ids=sample["context"],
                      novel_frame_ids=sample["novel"],
                      pair_iou=sample.get("pair_iou", 0.0))
    batch = move(default_collate([provider[index[sample["scene"]]]]), device)
    frames = [int(x) for x in batch["frame_ids"][0]]
    if frames != list(sample["frames"]):
        raise SystemExit(f"sample frames {frames} != recorded {sample['frames']}")
    return batch


def valid_confusion(pred: np.ndarray, sem: np.ndarray, alpha: np.ndarray):
    """Corrected convention: prediction and GT both restricted to the valid set."""
    valid = (sem != IGNORE) & (sem >= 0) & (sem < SEMANTIC_CLASSES) & (alpha > 0.05)
    conf = np.zeros((SEMANTIC_CLASSES, SEMANTIC_CLASSES), dtype=np.int64)
    if valid.any():
        conf += np.bincount(sem[valid] * SEMANTIC_CLASSES + pred[valid],
                            minlength=SEMANTIC_CLASSES ** 2).reshape(
            SEMANTIC_CLASSES, SEMANTIC_CLASSES)
    return conf, valid


def iou_from_binary(tp: int, fp: int, fn: int) -> float:
    union = tp + fp + fn
    return float(tp / union) if union else float("nan")


def semantic_scores(model, opt, batch, sentinels, context=(0, 1), novel=(2, 3)):
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
    sem = batch["semantic_label_all"][0].long().cpu().numpy()
    alpha = forward["masks"]["alpha"][0, :, 0].float().cpu().numpy()
    pred = forward["semantic_prob"].argmax(2)[0].long().cpu().numpy()
    out = {"sentinel_classes": [int(s["internal_class"]) for s in sentinels]}
    for scope, views in (("context", context), ("novel", novel)):
        conf = np.zeros((SEMANTIC_CLASSES, SEMANTIC_CLASSES), dtype=np.int64)
        legacy = {}
        for view in views:
            part, valid = valid_confusion(pred[view], sem[view], alpha[view])
            conf += part
            for sentinel in sentinels:
                cls = int(sentinel["internal_class"])
                mask = valid & (sem[view] == cls)
                p_any = (pred[view] == cls)
                legacy.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
                legacy[cls]["tp"] += int((mask & p_any).sum())
                legacy[cls]["fp"] += int((~mask & p_any).sum())
                legacy[cls]["fn"] += int((mask & ~p_any).sum())
        ious, tp, fp, fn, gt_px = [], {}, {}, {}, {}
        for cls in range(SEMANTIC_CLASSES):
            t = int(conf[cls, cls])
            f_n = int(conf[cls, :].sum() - t)
            f_p = int(conf[:, cls].sum() - t)
            tp[cls], fp[cls], fn[cls] = t, f_p, f_n
            gt_px[cls] = int(conf[cls, :].sum())
            ious.append(iou_from_binary(t, f_p, f_n))
        sentinel_rows = {}
        for sentinel in sentinels:
            cls = int(sentinel["internal_class"])
            sentinel_rows[str(sentinel["packed_key"])] = {
                "internal_class": cls,
                "corrected_iou": ious[cls], "tp": tp[cls], "fp": fp[cls], "fn": fn[cls],
                "gt_pixels": gt_px[cls],
                "pred_pixels_in_valid": tp[cls] + fp[cls],
                "legacy_iou": iou_from_binary(legacy[cls]["tp"], legacy[cls]["fp"],
                                              legacy[cls]["fn"]),
                "legacy_tp_fp_fn": legacy[cls],
            }
        present = [v for v in ious if not math.isnan(v)]
        out[scope] = {
            "confusion": conf.tolist(), "per_class_iou": ious,
            "miou_present_classes": float(np.mean(present)) if present else None,
            "sentinels": sentinel_rows,
            "n_valid_pixels": int(sum(int(conf[c, :].sum()) for c in range(SEMANTIC_CLASSES))),
        }
    del forward
    torch.cuda.empty_cache()
    return out


def instance_scores(model, opt, batch, sentinels, context=(0, 1), novel=(2, 3)):
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
    sem = batch["semantic_label_all"][0].long().cpu().numpy()
    ins = batch["instance_label_all"][0].long().cpu().numpy()
    table = group_score_table(forward)
    classes = table["class_argmax20"].long().cpu().numpy()
    out = {"sentinels": {}}
    for scope, views in (("context", context), ("novel", novel)):
        tp_cat_agnostic = tp_cat_aware = fp = fn = 0
        sentinel_hits = {str(s["packed_key"]): {"category_agnostic": False,
                                               "category_aware": False,
                                               "best_iou": 0.0, "best_query": None,
                                               "best_query_class": None}
                         for s in sentinels}
        for view in views:
            packed = (sem[view] + 1) * 1000 + ins[view]
            visible = (sem[view] >= 2) & (sem[view] < 20) & (ins[view] > 0)
            keys = sorted(set(int(k) for k in np.unique(packed[visible])))
            targets = {k: (visible & (packed == k)) for k in keys}
            gt_class = {k: int(sem[view][targets[k]][0]) for k in keys}
            predictions, _ = group_predictions_v2(forward, view)
            predictions = sorted(predictions, key=lambda p: -p["score"])
            used = set()
            for prediction in predictions:
                best, best_key = 0.0, None
                for key, truth in targets.items():
                    if key in used:
                        continue
                    union = int((prediction["mask"] | truth).sum())
                    if not union:
                        continue
                    iou = int((prediction["mask"] & truth).sum()) / union
                    if iou > best:
                        best, best_key = iou, key
                for sentinel in sentinels:
                    key = int(sentinel["packed_key"])
                    if key not in targets:
                        continue
                    union = int((prediction["mask"] | targets[key]).sum())
                    if not union:
                        continue
                    iou = int((prediction["mask"] & targets[key]).sum()) / union
                    row = sentinel_hits[str(key)]
                    if iou > row["best_iou"]:
                        row.update({"best_iou": iou, "best_query": int(prediction["group"]),
                                    "best_query_class": int(prediction["class"])})
                    if iou >= 0.5:
                        row["category_agnostic"] = True
                        if int(prediction["class"]) == int(sentinel["internal_class"]):
                            row["category_aware"] = True
                if best_key is not None and best >= 0.5:
                    used.add(best_key)
                    tp_cat_agnostic += 1
                    if int(prediction["class"]) == gt_class[best_key]:
                        tp_cat_aware += 1
                else:
                    fp += 1
            fn += len(targets) - len(used)
        out[scope] = {"category_agnostic_tp": tp_cat_agnostic,
                      "category_aware_tp": tp_cat_aware, "fp": fp, "fn": fn,
                      "sentinels": sentinel_hits}
    del forward
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/structure_probe_v2/corrected_metrics.json")
    parser.add_argument("--sample", default="group_plus/structure_probe_v1/sample.json")
    parser.add_argument("--s-delta", default="group_plus/structure_probe_v1/probe_S_delta.pt")
    parser.add_argument("--i-delta", default="group_plus/structure_probe_v1/probe_I_delta.pt")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    checks = {
        "sample_sha_matches_required": sample["sha256"] == REQUIRED_SAMPLE_SHA,
        "sample_scene": sample["scene"] == "scene0009_02",
        "sample_context": sample["context"] == [209, 253],
        "sample_novel": sample["novel"] == [215, 247],
        "sample_frames": sample["frames"] == [209, 253, 215, 247],
        "sentinel_keys": [s["packed_key"] for s in sample["sentinels"]] == [18032, 20030],
        "sentinel_classes": [s["internal_class"] for s in sample["sentinels"]] == [17, 19],
    }
    if not all(checks.values()):
        raise SystemExit(f"sample assertions failed: {checks}")

    checkpoint_sha_before = sha256_file(Path(G0PLUS) / "model.pt")
    baseline_sha = checkpoint_sha_before

    # ---- Probe-S: step 0 (baseline) and step 800 (baseline + delta) ---- #
    opt, model = load_g0plus(args.preset, args.seed, device)
    batch = batch_for_sample(opt, Path(args.split), sample, device)
    s_step0 = semantic_scores(model, opt, batch, sample["sentinels"])
    s_delta_info = apply_delta(model, Path(args.s_delta), "probe_S")
    s_delta_info["baseline_checkpoint_sha256"] = baseline_sha
    s_step800 = semantic_scores(model, opt, batch, sample["sentinels"])
    del model
    torch.cuda.empty_cache()

    # ---- Probe-I: step 0 (baseline) and step 1200 (baseline + delta) ---- #
    opt, model = load_g0plus(args.preset, args.seed, device)
    batch = batch_for_sample(opt, Path(args.split), sample, device)
    i_step0 = instance_scores(model, opt, batch, sample["sentinels"])
    i_delta_info = apply_delta(model, Path(args.i_delta), "probe_I")
    i_delta_info["baseline_checkpoint_sha256"] = baseline_sha
    i_step1200 = instance_scores(model, opt, batch, sample["sentinels"])
    del model
    torch.cuda.empty_cache()

    checkpoint_sha_after = sha256_file(Path(G0PLUS) / "model.pt")
    payload = {
        "scope": "inference-only correction of the previous round's read-outs; the "
                 "historical JSON files are untouched",
        "sample_checks": checks,
        "checkpoint": {"path": G0PLUS, "sha256_before": checkpoint_sha_before,
                       "sha256_after": checkpoint_sha_after,
                       "unchanged": checkpoint_sha_before == checkpoint_sha_after},
        "probe_S": {"step0": s_step0, "step800": s_step800, "delta": s_delta_info},
        "probe_I": {"step0": i_step0, "step1200": i_step1200, "delta": i_delta_info},
        "conventions": {
            "semantic_corrected": "20x20 confusion over the view set; valid = GT in 0..19 "
                                  "and alpha>0.05; prediction and GT both restricted to "
                                  "valid; a class with no GT in a view still accrues FP",
            "semantic_legacy": "previous round: GT restricted to valid, prediction counted "
                               "over the whole view",
            "instance": "frozen reader (P(thing)>=0.5, argmax thing, raw mass>0.5, area>=50), "
                        "score-descending greedy one-to-one; category-aware TP also requires "
                        "predicted class == GT class",
        },
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(json.dumps({
        "S": {"step0_ctx": {k: round(v["corrected_iou"], 4)
                            for k, v in s_step0["context"]["sentinels"].items()},
              "step800_ctx": {k: round(v["corrected_iou"], 4)
                              for k, v in s_step800["context"]["sentinels"].items()},
              "step0_novel": {k: round(v["corrected_iou"], 4)
                              for k, v in s_step0["novel"]["sentinels"].items()},
              "step800_novel": {k: round(v["corrected_iou"], 4)
                                for k, v in s_step800["novel"]["sentinels"].items()}},
        "I": {"step0_context": {k: i_step0["context"][k] for k in
                                ("category_agnostic_tp", "category_aware_tp", "fp", "fn")},
              "step0_novel": {k: i_step0["novel"][k] for k in
                              ("category_agnostic_tp", "category_aware_tp", "fp", "fn")},
              "step1200_context": {k: i_step1200["context"][k] for k in
                                   ("category_agnostic_tp", "category_aware_tp", "fp", "fn")},
              "step1200_novel": {k: i_step1200["novel"][k] for k in
                                 ("category_agnostic_tp", "category_aware_tp", "fp", "fn")},
              "step1200_sentinels": i_step1200["context"]["sentinels"]},
        "checkpoint_unchanged": payload["checkpoint"]["unchanged"],
    }, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
