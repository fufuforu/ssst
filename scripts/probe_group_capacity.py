#!/usr/bin/env python3
"""Isolated tiny-sample capacity probes for the group branch (structure_probe_v1).

Two *new* experiments, each on one fixed, pre-registered sample, both starting
from two independent loads of the same read-only G0+ step6000 checkpoint:

* **Probe-S (semantic head).**  Freeze everything except ``attributes.semantic.*``
  and optimise the existing per-pixel semantic NLL; geometry and alpha stay
  frozen.  AdamW lr=1e-3, wd=0, clip 1.0, at most 800 steps.
* **Probe-I (group head).**  Freeze everything except ``groups.*`` and optimise
  ``group_loss_terms(...)["loss_inst_total"]`` (G0+ includes its background
  supervision).  AdamW lr=1e-4, wd=0, clip 1.0, at most 1200 steps.

Each probe runs a 1-step smoke first (labels present, finite loss, gradients and
optimizer membership, only the target parameters move, frozen outputs
unchanged, checkpoint SHA unchanged) and refuses to continue if it fails.
Only the head *delta* relative to the frozen baseline is saved.

This is a local-capacity probe: success shows the representation can fit one
sample, not that it generalises, and it is not an official metric.
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

from tokengs.data.siu3r_processed import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    SIU3RProcessedProvider,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.input_types import ModelInputDecoder  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    forward_group,
    group_predictions_v2,
)
from scripts.plan_full_run import official_scenes  # noqa: E402
from scripts.train_object_locusgs import move  # noqa: E402

G0PLUS = "workspace_group_plus/arm_g0plus/ckpt_step6000"
SEMANTIC_CLASSES = 20
IGNORE = 255
CTX_VALID_MIN = 500
NOVEL_VALID_MIN = 300
COVERAGE_MIN = 0.70


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"sha256": sha256_file(path), "mtime": path.stat().st_mtime}


def build_opt(preset: str, seed: int):
    return config_defaults[preset].evolve(
        seed=seed, group_arm="g0", group_bg_supervision=True, group_recipe=False,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )


def load_g0plus(preset: str, seed: int, device):
    opt = build_opt(preset, seed)
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(G0PLUS) / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True)
    model.eval()
    return opt, model


def sentinel_stats(forward, batch, key, context_views, novel_views):
    sem = batch["semantic_label_all"][0].long().cpu().numpy()
    ins = batch["instance_label_all"][0].long().cpu().numpy()
    alpha = forward["masks"]["alpha"][0].float().cpu().numpy()[:, 0]
    out = {"context": {"pixels": 0, "covered": 0},
           "novel": {int(v): {"pixels": 0, "covered": 0} for v in novel_views}}
    for view in context_views:
        packed = (sem[view] + 1) * 1000 + ins[view]
        mask = ((sem[view] >= 2) & (sem[view] < 20) & (ins[view] > 0)
                & (packed == int(key)))
        out["context"]["pixels"] += int(mask.sum())
        out["context"]["covered"] += int((mask & (alpha[view] > 0.05)).sum())
    for view in novel_views:
        packed = (sem[view] + 1) * 1000 + ins[view]
        mask = ((sem[view] >= 2) & (sem[view] < 20) & (ins[view] > 0)
                & (packed == int(key)))
        out["novel"][int(view)]["pixels"] += int(mask.sum())
        out["novel"][int(view)]["covered"] += int((mask & (alpha[view] > 0.05)).sum())
    ctx = out["context"]
    ctx["coverage"] = ctx["covered"] / ctx["pixels"] if ctx["pixels"] else 0.0
    for view in novel_views:
        row = out["novel"][int(view)]
        row["coverage"] = row["covered"] / row["pixels"] if row["pixels"] else 0.0
    return out


def select_sample(plan, split, opt, device, max_scan: int):
    train_scenes, _ = official_scenes()
    train_scenes = set(train_scenes)
    provider = SIU3RProcessedProvider(opt, root=split["train_root"],
                                      subset=split["train_scenes"], training=True, rank=0)
    index = {p.name: i for i, p in enumerate(provider.dataset.sample_list)}
    eliminated = {"scene_not_official_train": 0, "fewer_than_two_thing_classes": 0,
                  "no_qualifying_sentinel_pair": 0}
    for position, item in enumerate(plan["entries"][:max_scan]):
        if item["scene"] not in train_scenes:
            eliminated["scene_not_official_train"] += 1
            continue
        provider.pin_pair(scene_id=item["scene"], context_frame_ids=item["context"],
                          novel_frame_ids=item["novel"], pair_iou=item["pair_iou"])
        batch = move(default_collate([provider[index[item["scene"]]]]), device)
        sem = batch["semantic_label_all"][0].long().cpu().numpy()
        ins = batch["instance_label_all"][0].long().cpu().numpy()
        # candidate instances per thing class, area summed over the context views
        per_class = {}
        for view in (0, 1):
            mask_valid = ((sem[view] >= 2) & (sem[view] < 20) & (ins[view] > 0))
            packed = (sem[view] + 1) * 1000 + ins[view]
            for key in np.unique(packed[mask_valid]):
                cls = int(sem[view][packed == key][0])
                entry = per_class.setdefault(cls, {})
                entry[int(key)] = entry.get(int(key), 0) + int(
                    (mask_valid & (packed == key)).sum())
        classes = sorted(per_class)
        if len(classes) < 2:
            eliminated["fewer_than_two_thing_classes"] += 1
            continue
        with torch.no_grad():
            forward = forward_group(model_forward_holder[0], batch, opt)
        best_pair, best_areas = None, -1
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1:]:
                k1 = max(per_class[c1], key=per_class[c1].get)
                k2 = max(per_class[c2], key=per_class[c2].get)
                s1 = sentinel_stats(forward, batch, k1, (0, 1), (2, 3))
                s2 = sentinel_stats(forward, batch, k2, (0, 1), (2, 3))
                total = s1["context"]["pixels"] + s2["context"]["pixels"]
                if total > best_areas:
                    best_pair, best_areas = (c1, k1, s1, c2, k2, s2), total
        if best_pair is None:
            eliminated["no_qualifying_sentinel_pair"] += 1
            continue
        c1, k1, s1, c2, k2, s2 = best_pair

        def qualifies(stats):
            if stats["context"]["pixels"] < CTX_VALID_MIN:
                return False
            if stats["context"]["coverage"] < COVERAGE_MIN:
                return False
            return any(row["pixels"] >= NOVEL_VALID_MIN and row["coverage"] >= COVERAGE_MIN
                       for row in stats["novel"].values())

        if qualifies(s1) and qualifies(s2):
            sample = {
                "plan_entry_index": position, "scene": item["scene"],
                "context": [int(x) for x in item["context"]],
                "novel": [int(x) for x in item["novel"]],
                "frames": [int(x) for x in batch["frame_ids"][0]],
                "pair_iou": float(item["pair_iou"]),
                "sentinels": [
                    {"packed_key": int(k1), "internal_class": int(c1),
                     "class_name_index": int(c1) + 1, "stats": s1},
                    {"packed_key": int(k2), "internal_class": int(c2),
                     "class_name_index": int(c2) + 1, "stats": s2}],
                "selection_rule": {
                    "context_valid_pixels_min": CTX_VALID_MIN,
                    "novel_valid_pixels_min_in_at_least_one_view": NOVEL_VALID_MIN,
                    "alpha_coverage_min": COVERAGE_MIN,
                    "per_class_instance": "largest context-merged area of that class"},
                "eliminated": eliminated,
            }
            sample["sha256"] = hashlib.sha256(
                json.dumps(sample, sort_keys=True).encode()).hexdigest()
            return sample, eliminated
        eliminated["no_qualifying_sentinel_pair"] += 1
        del forward
        torch.cuda.empty_cache()
    return None, eliminated


model_forward_holder = [None]


def sentinel_metrics_semantic(model, batch, sample, context_views, novel_views, prob):
    sem = batch["semantic_label_all"][0].long().cpu().numpy()
    pred = prob.argmax(2)[0].long().cpu().numpy()
    alpha = model_alpha = None
    out = {}
    sentinel_classes = [int(s["internal_class"]) for s in sample["sentinels"]]
    for scope, views in (("context", context_views), ("novel", novel_views)):
        per_class = {}
        for cls in range(SEMANTIC_CLASSES):
            inter = union = 0
            for view in views:
                valid = ((sem[view] != IGNORE) & (sem[view] >= 0)
                         & (sem[view] < SEMANTIC_CLASSES))
                mask = valid & (sem[view] == cls)
                if not mask.any():
                    continue
                p = (pred[view] == cls)
                inter += int((mask & p).sum())
                union += int((mask | p).sum())
            per_class[cls] = float(inter / union) if union else float("nan")
        present = [v for c, v in per_class.items() if not math.isnan(v)]
        out[scope] = {
            "sentinel_iou": {str(c): per_class[c] for c in sentinel_classes},
            "other_present_class_iou_mean": float(np.mean(
                [v for c, v in per_class.items()
                 if c not in sentinel_classes and not math.isnan(v)])) if present else None,
            "miou_present_classes": float(np.mean(present)) if present else None,
        }
    return out


def probe_s(opt, model, batch, sample, device, args, out_path):
    from tokengs.models.input_types import split_data, ModelInput

    report: dict = {"probe": "S", "config": {"lr": args.s_lr, "steps": args.s_steps,
                                             "optimizer": "AdamW wd=0",
                                             "clip": 1.0, "frozen": "all but "
                                             "attributes.semantic.*"}}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith("attributes.semantic."):
            parameter.requires_grad_(True)
            trainable.append((name, parameter))
    if not trainable:
        raise SystemExit("Probe-S found no attributes.semantic.* parameters")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.s_lr,
                                  weight_decay=0.0)
    names = [n for n, _ in trainable]
    report["optimizer_parameter_names"] = names
    report["optimizer_exactly_semantic_head"] = all(
        n.startswith("attributes.semantic.") for n in names)

    decoder_input = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                      intrinsics=batch["intrinsics_all"])
    baseline = {n: p.detach().clone() for n, p in trainable}
    frozen_hashes = frozen_state_hashes(model, trainable)
    with torch.no_grad():
        forward0 = forward_group(model, batch, opt)
    ref = {"rgb": forward0["output"]["render"]["images_pred"].clone(),
           "alpha": forward0["masks"]["alpha"].clone(),
           "gaussians": forward0["output"]["gaussians"].clone()}
    if args.smoke:
        report["smoke"] = smoke_step_s(model, opt, batch, trainable, optimizer, decoder_input)
        Path(out_path).write_text(json.dumps(report, indent=1, default=str))
        return report

    curve = []
    final = None
    for step in range(args.s_steps + 1):
        if step in args.s_eval_steps or step == args.s_steps:
            model.eval()
            with torch.no_grad():
                forward = forward_group(model, batch, opt)
                loss, stats, _ = model.semantic_loss_terms(
                    batch, forward["output"]["states"][-1]["tokens"],
                    forward["output"]["gaussians"], decoder_input,
                    forward["masks"]["alpha"])
            metrics = sentinel_metrics_semantic(
                model, batch, sample, (0, 1), (2, 3), forward["semantic_prob"])
            row = {"step": step, "semantic_loss": float(loss),
                   "supervised_pixels": stats["supervised_pixels"],
                   "coverage": float(stats["coverage"]), **metrics}
            curve.append(row)
            print(f"[S] step {step} loss {float(loss):.4f} "
                  f"ctx {metrics['context']['sentinel_iou']} "
                  f"novel {metrics['novel']['sentinel_iou']}", flush=True)
            final = row
            if step and meets_s_gate(curve):
                print("[S] gate met early", flush=True)
                break
        if step == args.s_steps:
            break
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        forward = forward_group(model, batch, opt)
        loss, stats, _ = model.semantic_loss_terms(
            batch, forward["output"]["states"][-1]["tokens"],
            forward["output"]["gaussians"], decoder_input, forward["masks"]["alpha"])
        if not torch.isfinite(loss):
            report["failure"] = f"non-finite loss at step {step + 1}"
            break
        loss.backward()
        all_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                         for _, p in trainable)
        if not all_finite:
            report["failure"] = f"non-finite gradient at step {step + 1}"
            break
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        optimizer.step()
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
    report["curve"] = curve
    report["final"] = final
    report["frozen_output_drift"] = {
        "rgb_max_abs_diff": float((ref["rgb"] - forward["output"]["render"]["images_pred"]
                                   ).abs().max()),
        "alpha_max_abs_diff": float((ref["alpha"] - forward["masks"]["alpha"]).abs().max()),
        "gaussians_max_abs_diff": float((ref["gaussians"]
                                         - forward["output"]["gaussians"]).abs().max()),
    }
    report["frozen_non_target_params_unchanged"] = frozen_state_hashes(
        model, trainable) == frozen_hashes
    report["updated_params"] = {n: float((p.detach() - baseline[n]).abs().max())
                                for n, p in trainable}
    save_delta(out_path.parent / "probe_S_delta.pt", names, baseline,
               [p for _, p in trainable], report)
    Path(out_path).write_text(json.dumps(report, indent=1, default=str))
    return report


def smoke_step_s(model, opt, batch, trainable, optimizer, decoder_input):
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    forward = forward_group(model, batch, opt)
    loss, stats, _ = model.semantic_loss_terms(
        batch, forward["output"]["states"][-1]["tokens"],
        forward["output"]["gaussians"], decoder_input, forward["masks"]["alpha"])
    if not torch.isfinite(loss):
        raise SystemExit("Probe-S smoke: non-finite loss")
    if stats["supervised_pixels"] <= 0:
        raise SystemExit("Probe-S smoke: no supervised pixels")
    loss.backward()
    grads = {n: (None if p.grad is None else float(p.grad.norm()))
             for n, p in trainable}
    if any(g is None or not math.isfinite(g) for g in grads.values()):
        raise SystemExit(f"Probe-S smoke: bad gradients {grads}")
    before = {n: p.detach().clone() for n, p in trainable}
    optimizer.step()
    updates = {n: float((p.detach() - before[n]).abs().max()) for n, p in trainable}
    if not any(v > 0 for v in updates.values()):
        raise SystemExit("Probe-S smoke: no parameter moved")
    return {"loss": float(loss), "supervised_pixels": stats["supervised_pixels"],
            "grad_norms": grads, "updates": updates, "passed": True}


def probe_i(opt, model, batch, sample, device, args, out_path):
    report: dict = {"probe": "I", "config": {"lr": args.i_lr, "steps": args.i_steps,
                                             "loss": "group_loss_terms.loss_inst_total",
                                             "optimizer": "AdamW wd=0", "clip": 1.0,
                                             "frozen": "all but groups.*"}}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if n.startswith("groups.")]
    for _, parameter in trainable:
        parameter.requires_grad_(True)
    if not trainable:
        raise SystemExit("Probe-I found no groups.* parameters")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.i_lr,
                                  weight_decay=0.0)
    names = [n for n, _ in trainable]
    report["optimizer_parameter_names"] = names
    report["optimizer_exactly_groups"] = all(n.startswith("groups.") for n in names)
    report["n_trainable"] = len(trainable)

    decoder_input = ModelInputDecoder(cam_view=batch["cam_view_all"],
                                      intrinsics=batch["intrinsics_all"])
    baseline = {n: p.detach().clone() for n, p in trainable}
    frozen_hashes = frozen_state_hashes(model, trainable)
    with torch.no_grad():
        forward0 = forward_group(model, batch, opt)
    ref = {"rgb": forward0["output"]["render"]["images_pred"].clone(),
           "alpha": forward0["masks"]["alpha"].clone(),
           "gaussians": forward0["output"]["gaussians"].clone(),
           "attributes": {n: p.detach().clone() for n, p in model.named_parameters()
                          if n.startswith("attributes.")}}
    if args.smoke:
        report["smoke"] = smoke_step_i(model, opt, batch, trainable, optimizer,
                                       decoder_input)
        Path(out_path).write_text(json.dumps(report, indent=1, default=str))
        return report

    curve = []
    final = None
    for step in range(args.i_steps + 1):
        if step in args.i_eval_steps or step == args.i_steps:
            model.eval()
            with torch.no_grad():
                forward = forward_group(model, batch, opt)
                loss = model.group_loss_terms(batch, forward["output"]["gaussians"],
                                              decoder_input, (0, 1))["loss_inst_total"]
                metrics = sentinel_metrics_instance(model, forward, batch, sample)
            row = {"step": step, "loss_inst_total": float(loss), **metrics}
            curve.append(row)
            print(f"[I] step {step} loss {float(loss):.4f} "
                  f"ctx reader TP {metrics['reader']['context']['tp']} "
                  f"novel TP {metrics['reader']['novel']['tp']}", flush=True)
            final = row
            if step and meets_i_gate(curve):
                print("[I] gate met early", flush=True)
                break
        if step == args.i_steps:
            break
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        forward = forward_group(model, batch, opt)
        loss = model.group_loss_terms(batch, forward["output"]["gaussians"],
                                      decoder_input, (0, 1))["loss_inst_total"]
        if not torch.isfinite(loss):
            report["failure"] = f"non-finite loss at step {step + 1}"
            break
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for _, p in trainable):
            report["failure"] = f"non-finite gradient at step {step + 1}"
            break
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        optimizer.step()
    with torch.no_grad():
        forward = forward_group(model, batch, opt)
    report["curve"] = curve
    report["final"] = final
    report["frozen_output_drift"] = {
        "rgb_max_abs_diff": float((ref["rgb"] - forward["output"]["render"]["images_pred"]
                                   ).abs().max()),
        "alpha_max_abs_diff": float((ref["alpha"] - forward["masks"]["alpha"]).abs().max()),
        "gaussians_max_abs_diff": float((ref["gaussians"]
                                         - forward["output"]["gaussians"]).abs().max()),
        "attributes_max_abs_diff": max(
            float((ref["attributes"][n] - p.detach()).abs().max())
            for n, p in model.named_parameters() if n.startswith("attributes.")),
    }
    report["frozen_non_target_params_unchanged"] = frozen_state_hashes(
        model, trainable) == frozen_hashes
    report["updated_params"] = {n: float((p.detach() - baseline[n]).abs().max())
                                for n, p in trainable}
    save_delta(out_path.parent / "probe_I_delta.pt", names, baseline,
               [p for _, p in trainable], report)
    Path(out_path).write_text(json.dumps(report, indent=1, default=str))
    return report


def sentinel_metrics_instance(model, forward, batch, sample):
    sem = batch["semantic_label_all"][0].long().cpu().numpy()
    ins = batch["instance_label_all"][0].long().cpu().numpy()
    mass = forward["masks"]["group_mass"][0].float().cpu()
    from scripts.group_eval_v2 import group_score_table
    table = group_score_table(forward)
    classes = table["class_argmax20"].long().cpu().numpy()
    p_thing = table["p_thing"].float().cpu().numpy()
    out = {"per_sentinel": {}, "reader": {}}
    best_queries = []
    for sentinel in sample["sentinels"]:
        key = int(sentinel["packed_key"])
        cls = int(sentinel["internal_class"])
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
    out["distinct_queries"] = (len(set(q for q in best_queries if q is not None)) == 2)
    for scope, views in (("context", (0, 1)), ("novel", (2, 3))):
        tp = fp = fn = 0
        for view in views:
            predictions, _ = group_predictions_v2(forward, view)
            targets = []
            packed = (sem[view] + 1) * 1000 + ins[view]
            visible = (sem[view] >= 2) & (sem[view] < 20) & (ins[view] > 0)
            for key in sorted(set(int(k) for k in np.unique(packed[visible]))):
                targets.append(visible & (packed == key))
            used = set()
            for prediction in predictions:
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
                else:
                    fp += 1
            fn += len(targets) - len(used)
        out["reader"][scope] = {"tp": tp, "fp": fp, "fn": fn}
    return out


def smoke_step_i(model, opt, batch, trainable, optimizer, decoder_input):
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    forward = forward_group(model, batch, opt)
    loss = model.group_loss_terms(batch, forward["output"]["gaussians"],
                                  decoder_input, (0, 1))["loss_inst_total"]
    if not torch.isfinite(loss):
        raise SystemExit("Probe-I smoke: non-finite loss")
    loss.backward()
    grads = {n: (None if p.grad is None else float(p.grad.norm()))
             for n, p in trainable}
    if any(g is None or not math.isfinite(g) for g in grads.values()):
        raise SystemExit("Probe-I smoke: bad gradients")
    before = {n: p.detach().clone() for n, p in trainable}
    optimizer.step()
    updates = {n: float((p.detach() - before[n]).abs().max()) for n, p in trainable}
    if not any(v > 0 for v in updates.values()):
        raise SystemExit("Probe-I smoke: no parameter moved")
    return {"loss_inst_total": float(loss), "grad_norms": grads, "updates": updates,
            "passed": True}


def frozen_state_hashes(model, trainable):
    names = {n for n, _ in trainable}
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name in names:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_delta(path, names, baseline, current, report):
    delta = {n: (p.detach().cpu() - baseline[n].cpu()) for n, p in zip(names, current)}
    torch.save({"delta": delta, "steps": report.get("final", {}).get("step"),
                "config": report.get("config"), "names": names}, path)
    report["delta_path"] = str(path)
    report["delta_bytes"] = path.stat().st_size


def meets_s_gate(curve):
    last = curve[-1]
    ctx = last["context"]["sentinel_iou"]
    novel = last["novel"]["sentinel_iou"]
    return all(v >= 0.50 for v in ctx.values()) and all(v >= 0.25 for v in novel.values())


def meets_i_gate(curve):
    last = curve[-1]
    sent = last["per_sentinel"]
    ctx_ok = all(v["best_raw_iou"] >= 0.50 and v["best_query_class_correct"]
                 for v in sent.values())
    return (ctx_ok and last["distinct_queries"]
            and last["reader"]["context"]["tp"] >= 2
            and last["reader"]["novel"]["tp"] >= 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="group_plus/structure_probe_v1")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe", choices=("both", "S", "I"), default="both")
    parser.add_argument("--s-lr", type=float, default=1e-3)
    parser.add_argument("--s-steps", type=int, default=800)
    parser.add_argument("--s-eval-steps", type=int, nargs="*", default=[0, 100, 200, 400, 800])
    parser.add_argument("--i-lr", type=float, default=1e-4)
    parser.add_argument("--i-steps", type=int, default=1200)
    parser.add_argument("--i-eval-steps", type=int, nargs="*",
                        default=[0, 100, 300, 600, 1200])
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    before = identity(Path(G0PLUS) / "model.pt")

    sample_path = out_dir / "sample.json"
    if sample_path.is_file():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        print("[probe] reusing the fixed sample", flush=True)
    else:
        opt, model = load_g0plus(args.preset, args.seed, device)
        model_forward_holder[0] = model
        sample, eliminated = select_sample(plan, split, opt, device, args.max_scan)
        del model
        torch.cuda.empty_cache()
        if sample is None:
            raise SystemExit(f"no qualifying sample in the first {args.max_scan} plan "
                             f"entries; eliminations={eliminated}")
        sample["selection_eliminations"] = eliminated
        sample_path.write_text(json.dumps(sample, indent=1), encoding="utf-8")
    print(f"[probe] sample {sample['scene']} {sample['frames']} "
          f"entry {sample['plan_entry_index']}", flush=True)

    results = {}
    if args.probe in ("both", "S"):
        opt, model = load_g0plus(args.preset, args.seed, device)
        batch = sample_batch(opt, split, sample, device)
        results["S"] = probe_s(opt, model, batch, sample, device, args,
                               out_dir / ("probe_S_smoke.json" if args.smoke
                                          else "probe_S.json"))
        del model
        torch.cuda.empty_cache()
    if args.probe in ("both", "I"):
        opt, model = load_g0plus(args.preset, args.seed, device)
        batch = sample_batch(opt, split, sample, device)
        results["I"] = probe_i(opt, model, batch, sample, device, args,
                               out_dir / ("probe_I_smoke.json" if args.smoke
                                          else "probe_I.json"))
        del model
        torch.cuda.empty_cache()

    after = identity(Path(G0PLUS) / "model.pt")
    payload = {"sample": sample, "checkpoint_before": before, "checkpoint_after": after,
               "checkpoint_unchanged": after == before,
               "elapsed_seconds": time.time() - started, "smoke": bool(args.smoke)}
    if args.smoke:
        (out_dir / "smoke.json").write_text(json.dumps(
            {**payload, "probe_smoke": {k: v.get("smoke") for k, v in results.items()}},
            indent=1, default=str), encoding="utf-8")
        print("[probe] smoke wrote", flush=True)
        return 0

    s_pass = gate_s(results.get("S"))
    i_pass = gate_i(results.get("I"))
    gate = {
        "capacity_pass": bool(s_pass["passed"] and i_pass["passed"]),
        "probe_S": s_pass, "probe_I": i_pass,
        "criteria": {
            "S": "both sentinel classes: context IoU >= 0.50 and novel IoU >= 0.25",
            "I": "both sentinels: distinct query with raw mask IoU >= 0.50 and correct "
                 "query class; formal reader TP >= 2 in context and >= 1 in novel",
        },
        "scope": "single-sample local capacity; success does not imply generalisation "
                 "and is not an official metric",
        "checkpoint_unchanged": payload["checkpoint_unchanged"],
    }
    (out_dir / "structure_gate.json").write_text(json.dumps(gate, indent=1),
                                                 encoding="utf-8")
    (out_dir / "probe_runs.json").write_text(json.dumps(
        {"sample": sample, "checkpoint": before, "results": results},
        indent=1, default=str), encoding="utf-8")
    print(json.dumps(gate, indent=1, default=str))
    return 0


def gate_s(result):
    if not result or "final" not in result:
        return {"passed": False, "reason": "probe S did not complete",
                "evidence": result.get("failure") if result else None}
    final = result["final"]
    ctx = final["context"]["sentinel_iou"]
    novel = final["novel"]["sentinel_iou"]
    ok = all(v >= 0.50 for v in ctx.values()) and all(v >= 0.25 for v in novel.values())
    blocker = None
    if not ok:
        blocker = ("context IoU below 0.50" if not all(v >= 0.50 for v in ctx.values())
                   else "novel IoU below 0.25")
    return {"passed": bool(ok), "final_step": final["step"],
            "context_sentinel_iou": ctx, "novel_sentinel_iou": novel,
            "semantic_loss": final["semantic_loss"], "blocker": blocker}


def gate_i(result):
    if not result or "final" not in result:
        return {"passed": False, "reason": "probe I did not complete",
                "evidence": result.get("failure") if result else None}
    final = result["final"]
    sent = final["per_sentinel"]
    mask_ok = all(v["best_raw_iou"] >= 0.50 for v in sent.values())
    class_ok = all(v["best_query_class_correct"] for v in sent.values())
    reader_ctx = final["reader"]["context"]
    reader_novel = final["reader"]["novel"]
    ok = (mask_ok and class_ok and final["distinct_queries"]
          and reader_ctx["tp"] >= 2 and reader_novel["tp"] >= 1)
    blocker = None
    if not ok:
        if not mask_ok:
            blocker = "raw mask IoU below 0.50 for a sentinel"
        elif not class_ok:
            blocker = "best query class wrong for a sentinel"
        elif not final["distinct_queries"]:
            blocker = "both sentinels collapse to the same query"
        elif reader_ctx["tp"] < 2:
            blocker = "formal reader misses a sentinel in context"
        else:
            blocker = "formal reader misses both sentinels in novel"
    return {"passed": bool(ok), "final_step": final["step"],
            "per_sentinel": sent, "distinct_queries": final["distinct_queries"],
            "reader_context": reader_ctx, "reader_novel": reader_novel,
            "loss_inst_total": final["loss_inst_total"], "blocker": blocker}


def sample_batch(opt, split, sample, device):
    provider = SIU3RProcessedProvider(opt, root=split["train_root"],
                                      subset=split["train_scenes"], training=True, rank=0)
    index = {p.name: i for i, p in enumerate(provider.dataset.sample_list)}
    provider.pin_pair(scene_id=sample["scene"],
                      context_frame_ids=sample["context"],
                      novel_frame_ids=sample["novel"],
                      pair_iou=sample.get("pair_iou", 0.0))
    batch = move(default_collate([provider[index[sample["scene"]]]]), device)
    frames = [int(x) for x in batch["frame_ids"][0]]
    if frames != list(sample["frames"]):
        raise SystemExit(f"sample frames {frames} != recorded {sample['frames']}")
    return batch


if __name__ == "__main__":
    raise SystemExit(main())
