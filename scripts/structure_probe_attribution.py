#!/usr/bin/env python3
"""Read-only attribution for the group branch (structure_probe_v1, phase 1).

Two checkpoints with real official results are compared on three fixed window
groups, using only frozen forward passes (no optimizer step, no GT in the
forward):

* ``train7``  -- the 7 training windows resolved by ``build_train_entries(..., 8)``
  (same set as ``audit_implementation_v1``; 2 context + 2 novel);
* ``dev8``    -- the original 8 development windows (2 context + 2 novel);
* ``val32``   -- the first pair of the first 32 distinct scenes of the official
  ``val_pair.json``, sorted by ``(scene, context)`` (2 context + 4 novel).

Outputs (all small JSON):

* A semantic: 20x20 confusion, per-class GT/predicted pixel counts and IoU under
  two conventions (GT-valid & alpha>0.05, and the official export convention
  where alpha<=0.05 predicts void); per GT thing class the fraction of its pixels
  with alpha<=0.05 and <=0.5; context/novel split.  The full-1860 per-class IoU is
  quoted from ``group_plus/implementation_audit_v1/B1_full/*_official_semantic.json``
  and is never mixed with the 32-pair estimate.
* B instance: for every GT thing, the best raw query (mass>0.5, area>=50),
  its predicted class correctness, P(thing), whether that query passes the frozen
  reader, and the reader's own best IoU, binned into four mutually exclusive
  buckets.  GT-assisted best-query numbers are diagnostics only.
* C assembly: per-class IoU of the independent semantic map vs the panoptic map,
  and a recount of the actual void pixels of the panoptic product.

The three groups are reported separately and never averaged together.
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
    record_scene,
)
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.eval_group_locusgs import build_train_entries  # noqa: E402
from scripts.group_eval_v2 import (  # noqa: E402
    build_val_entries,
    forward_group,
    group_predictions_v2,
    group_score_table,
)
from scripts.group_official_export import (  # noqa: E402
    panoptic_prediction,
    semantic_prediction,
)
from scripts.train_object_locusgs import move  # noqa: E402

SEMANTIC_CLASSES = 20
STUFF = (0, 1)
THING = tuple(range(2, 20))
IGNORE = 255

ARMS = {
    "g0plus": {"directory": "workspace_group_plus/arm_g0plus/ckpt_step6000",
               "recipe": False, "g0plus": True, "seg_weight": 0.05},
    "recipe_v1": {"directory": "workspace_group_plus/recipe_v1/run/ckpt_step6000",
                  "recipe": True, "g0plus": False, "seg_weight": 0.1},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"sha256": sha256_file(path), "mtime": path.stat().st_mtime}


def load_arm(name: str, preset: str, seed: int, device):
    meta = ARMS[name]
    opt = config_defaults[preset].evolve(
        seed=seed, group_arm="g0", group_bg_supervision=meta["g0plus"],
        group_recipe=meta["recipe"], group_recipe_head_mode="legacy_prefix",
        group_recipe_seg_weight=meta["seg_weight"],
        group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        batch_size=1, num_workers=0, num_input_views=2, num_views=4,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(Path(meta["directory"]) / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True,
    )
    model.eval()
    return opt, model


def official_val_windows(count: int) -> tuple[list[dict], dict]:
    manifest = Path(DEFAULT_DATA_ROOT) / "val_pair.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))
    ordered = sorted(records, key=lambda r: (r["scan"], tuple(r["context_ids"])))
    picked, seen = [], set()
    for record in ordered:
        if record["scan"] in seen:
            continue
        seen.add(record["scan"])
        picked.append(record)
        if len(picked) >= count:
            break
    if len(picked) < count:
        raise SystemExit(f"official val_pair only has {len(picked)} distinct scenes, "
                         f"need {count}")
    listing = [{"scene": r["scan"], "context": list(r["context_ids"]),
                "target": list(r["target_ids"])} for r in picked]
    digest = hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()
    return picked, {"count": len(listing), "listing_sha256": digest,
                    "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                    "windows": listing}


def build_val32_batches(opt, picked, device):
    provider = SIU3RProcessedProvider(
        opt.evolve(num_views=6), root=str(Path(DEFAULT_DATA_ROOT) / "val"),
        subset="all", training=False,
        val_pair_json=str(Path(DEFAULT_DATA_ROOT) / "val_pair.json"), rank=0,
    )
    provider.dataset.val_pairs = picked
    entries = []
    for index, record in enumerate(picked):
        batch = move(default_collate([provider[index]]), device)
        frames = [int(x) for x in batch["frame_ids"][0]]
        if frames[:2] != list(record["context_ids"]) or set(frames) != set(record["target_ids"]):
            raise SystemExit(f"val32 record {index} frames {frames} do not match the "
                             f"manifest {record}")
        entries.append({"scene": record_scene(record), "batch": batch,
                        "context": list(record["context_ids"]),
                        "novel": frames[2:], "kind": "val32"})
    return entries


def iou_from_confusion(confusion: np.ndarray) -> tuple[list[float], list[int], list[int]]:
    """confusion[gt_class, predicted_class] with a trailing column for predictions
    that are void (official 0).  Pixels predicted void are never a true or false
    positive for a class, but they *are* false negatives for their GT class."""
    ious, gt_pixels, pred_pixels = [], [], []
    for cls in range(SEMANTIC_CLASSES):
        tp = confusion[cls, cls]
        fn = confusion[cls, :].sum() - tp
        fp = confusion[:, :SEMANTIC_CLASSES][:, cls].sum() - tp
        union = tp + fp + fn
        ious.append(float(tp / union) if union else float("nan"))
        gt_pixels.append(int(confusion[cls, :].sum()))
        pred_pixels.append(int(confusion[:, cls].sum()))
    return ious, gt_pixels, pred_pixels


def semantic_block(forward, batch, num_views, views):
    semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
    alpha = forward["masks"]["alpha"][0].float().cpu().numpy()[:, 0]
    prob = forward["semantic_prob"]
    pred_all = prob.argmax(2)[0].long().cpu().numpy() if prob.dim() == 5 else \
        prob.argmax(1)[0].long().cpu().numpy()
    conf_a = np.zeros((SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1), dtype=np.int64)
    conf_b = np.zeros((SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1), dtype=np.int64)
    thing_alpha = {c: {"pixels": 0, "le_005": 0, "le_05": 0} for c in THING}
    # the same statistic split by context vs novel views, as required
    thing_alpha_split = {"context": {c: {"pixels": 0, "le_005": 0, "le_05": 0}
                                     for c in THING},
                         "novel": {c: {"pixels": 0, "le_005": 0, "le_05": 0}
                                   for c in THING}}
    per_view = {}
    for view in views:
        sem, pred, a = semantic_gt[view], pred_all[view], alpha[view]
        valid = (sem != IGNORE)
        covered = a > 0.05
        sel_a = valid & covered
        conf_a += np.bincount((sem[sel_a] * (SEMANTIC_CLASSES + 1) + pred[sel_a]),
                              minlength=SEMANTIC_CLASSES * (SEMANTIC_CLASSES + 1)).reshape(
            SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1)
        # official export convention: alpha<=0.05 is written as void (column 20)
        pred_official = np.where(covered, pred, SEMANTIC_CLASSES)
        sel_b = valid
        conf_b += np.bincount((sem[sel_b] * (SEMANTIC_CLASSES + 1) + pred_official[sel_b]),
                              minlength=SEMANTIC_CLASSES * (SEMANTIC_CLASSES + 1)).reshape(
            SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1)
        scope = "context" if int(view) < 2 else "novel"
        for cls in THING:
            m = valid & (sem == cls)
            thing_alpha[cls]["pixels"] += int(m.sum())
            thing_alpha[cls]["le_005"] += int((m & (a <= 0.05)).sum())
            thing_alpha[cls]["le_05"] += int((m & (a <= 0.5)).sum())
            row = thing_alpha_split[scope][cls]
            row["pixels"] += int(m.sum())
            row["le_005"] += int((m & (a <= 0.05)).sum())
            row["le_05"] += int((m & (a <= 0.5)).sum())
        per_view[int(view)] = {
            "n_valid_gt": int(valid.sum()),
            "n_covered": int(covered.sum()),
            "n_valid_and_covered": int(sel_a.sum()),
        }
    ious_a, gt_a, pred_a = iou_from_confusion(conf_a)
    ious_b, gt_b, pred_b = iou_from_confusion(conf_b)
    present = [v for v in ious_a if not math.isnan(v)]
    return {
        "views": [int(v) for v in views],
        "convention_gt_valid_and_alpha_gt_005": {
            "confusion": conf_a.tolist(), "iou": ious_a, "gt_pixels": gt_a,
            "pred_pixels": pred_a,
            "miou_present_classes": float(np.mean(present)) if present else None,
        },
        "convention_official_export_void_on_low_alpha": {
            "confusion": conf_b.tolist(), "iou": ious_b, "gt_pixels": gt_b,
            "pred_pixels": pred_b,
        },
        "thing_alpha": {str(c): thing_alpha[c] for c in THING
                        if thing_alpha[c]["pixels"] > 0},
        "thing_alpha_by_scope": {
            scope: {str(c): row for c, row in table.items() if row["pixels"] > 0}
            for scope, table in thing_alpha_split.items()},
        "per_view": per_view,
    }


def instance_block(forward, batch, views):
    semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
    instance_gt = batch["instance_label_all"][0].long().cpu().numpy()
    scores = group_score_table(forward)
    classes = scores["class_argmax20"].long().cpu().numpy()
    p_thing = scores["p_thing"].float().cpu().numpy()
    mass = forward["masks"]["group_mass"][0].float().cpu()
    buckets = {"no_mask_reaches_0.5": 0, "mask_ok_class_wrong": 0,
               "mask_and_class_ok_reader_missing": 0, "reader_true_positive": 0}
    per_instance = []
    for view in views:
        sem, ins = semantic_gt[view], instance_gt[view]
        packed = (sem + 1) * 1000 + ins
        visible = (sem >= 2) & (sem < 20) & (ins > 0)
        reader_predictions, _ = group_predictions_v2(forward, view)
        for key in sorted(set(int(k) for k in np.unique(packed[visible]))):
            truth = visible & (packed == key)
            gt_class = int(sem[truth][0])
            best_iou, best_q = 0.0, None
            for query in range(mass.shape[1]):
                hard = mass[view, query].numpy() > 0.5
                area = int(hard.sum())
                if area < 50:
                    continue
                union = int((hard | truth).sum())
                if not union:
                    continue
                iou = int((hard & truth).sum()) / union
                if iou > best_iou:
                    best_iou, best_q = iou, query
            reader_best, reader_query = 0.0, None
            for prediction in reader_predictions:
                union = int((prediction["mask"] | truth).sum())
                if not union:
                    continue
                iou = int((prediction["mask"] & truth).sum()) / union
                if iou > reader_best:
                    reader_best, reader_query = iou, int(prediction["group"])
            class_ok = best_q is not None and int(classes[best_q]) == gt_class
            reader_has = best_q is not None and any(
                int(p["group"]) == int(best_q) for p in reader_predictions)
            if reader_best >= 0.5:
                bucket = "reader_true_positive"
            elif best_iou >= 0.5 and class_ok:
                bucket = "mask_and_class_ok_reader_missing"
            elif best_iou >= 0.5:
                bucket = "mask_ok_class_wrong"
            else:
                bucket = "no_mask_reaches_0.5"
            buckets[bucket] += 1
            per_instance.append({
                "scene": None, "view": int(view), "key": key, "gt_class": gt_class,
                "gt_area": int(truth.sum()), "best_query": best_q,
                "best_raw_iou": best_iou,
                "best_query_class": int(classes[best_q]) if best_q is not None else None,
                "best_query_class_correct": bool(class_ok),
                "best_query_p_thing": float(p_thing[best_q]) if best_q is not None else None,
                "best_query_passes_reader": bool(reader_has),
                "reader_best_iou": reader_best, "reader_query": reader_query,
                "bucket": bucket})
    return {"buckets": buckets, "n_gt_instances": len(per_instance),
            "per_instance": per_instance}


def assembly_block(forward, batch, num_views, views):
    semantic_gt = batch["semantic_label_all"][0].long().cpu().numpy()
    # the exporter helpers expect alpha as [B, V, 1, H, W]
    alpha_t = forward["masks"]["alpha"]
    sem_maps = semantic_prediction(forward, forward["semantic_prob"], alpha_t, num_views)
    pan_maps, _, stats = panoptic_prediction(forward, forward["semantic_prob"], alpha_t,
                                             num_views)
    conf_sem = np.zeros((SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1), dtype=np.int64)
    conf_pan = np.zeros((SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1), dtype=np.int64)
    void = {"total_void_pixels": 0, "void_with_alpha_le_005": 0,
            "void_with_alpha_gt_005_unassigned_thing_argmax": 0,
            "counter_void_pixels_claimed": 0}
    for view in views:
        sem = semantic_gt[view]
        valid = sem != IGNORE
        alpha = alpha_t[0, view, 0].float().cpu().numpy()
        # official 1-based -> internal class; official 0 (void) -> the void column
        pred_sem = np.where(sem_maps[view][0] > 0, sem_maps[view][0] - 1,
                            SEMANTIC_CLASSES)
        pred_pan = np.where(pan_maps[view][0] > 0, pan_maps[view][0] - 1,
                            SEMANTIC_CLASSES)
        conf_sem += np.bincount((sem[valid] * (SEMANTIC_CLASSES + 1) + pred_sem[valid]),
                                minlength=SEMANTIC_CLASSES * (SEMANTIC_CLASSES + 1)
                                ).reshape(SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1)
        conf_pan += np.bincount((sem[valid] * (SEMANTIC_CLASSES + 1) + pred_pan[valid]),
                                minlength=SEMANTIC_CLASSES * (SEMANTIC_CLASSES + 1)
                                ).reshape(SEMANTIC_CLASSES, SEMANTIC_CLASSES + 1)
        is_void = pred_pan == SEMANTIC_CLASSES
        head_argmax = forward["semantic_prob"][0, view].argmax(0).long().cpu().numpy()
        unassigned = (pan_maps[view][1] == 0) & is_void
        void["total_void_pixels"] += int(is_void.sum())
        void["void_with_alpha_le_005"] += int((is_void & (alpha <= 0.05)).sum())
        void["void_with_alpha_gt_005_unassigned_thing_argmax"] += int(
            (is_void & (alpha > 0.05) & unassigned
             & np.isin(head_argmax, THING)).sum())
        void["counter_void_pixels_claimed"] += int(stats[view]["void_pixels"])
    ious_sem, _, _ = iou_from_confusion(conf_sem)
    ious_pan, _, _ = iou_from_confusion(conf_pan)
    return {"per_class_iou_semantic_only": ious_sem,
            "per_class_iou_panoptic": ious_pan,
            "void_recount": void,
            "panoptic_view_stats": stats,
            "conventions": {"semantic_only": "independent semantic head argmax, "
                                              "alpha<=0.05 -> void 0",
                            "panoptic": "thing masks from the frozen reader, stuff "
                                        "from the independent head, else void"}}


def analyse_arm(name, preset, seed, device, plan, split, val32_entries, groups, views):
    opt, model = load_arm(name, preset, seed, device)
    out = {"windows": {}, "semantic": {}, "instance": {}, "assembly": {}}
    for group_name, entries in groups.items():
        semantic, instance, assembly, window_rows = [], [], [], []
        for entry in entries:
            batch = entry["batch"]
            with torch.no_grad():
                forward = forward_group(model, batch, opt)
            view_list = views[group_name]
            semantic.append(semantic_block(forward, batch,
                                           int(batch["images_all"].shape[1]), view_list))
            instance.append(instance_block(forward, batch, view_list))
            assembly.append(assembly_block(forward, batch,
                                           int(batch["images_all"].shape[1]), view_list))
            window_rows.append({"scene": entry["scene"],
                                "context": list(entry["context"]),
                                "novel": list(entry["novel"]),
                                "frames": [int(x) for x in batch["frame_ids"][0]]})
            del forward
            torch.cuda.empty_cache()
        out["windows"][group_name] = window_rows
        out["semantic"][group_name] = semantic
        out["instance"][group_name] = instance
        out["assembly"][group_name] = assembly
    del model
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="group_plus/structure_probe_v1/attribution.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val32-count", type=int, default=32)
    parser.add_argument("--smoke", action="store_true",
                        help="1 window per development group and 4 official val pairs")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    before = {name: identity(Path(meta["directory"]) / "model.pt")
              for name, meta in ARMS.items()}

    picked, val32_meta = official_val_windows(4 if args.smoke else args.val32_count)
    opt0, _ = load_arm("g0plus", args.preset, args.seed, device)
    train_entries = build_train_entries(opt0, split, plan, device, 8)
    del _
    torch.cuda.empty_cache()
    opt1, model1 = load_arm("g0plus", args.preset, args.seed, device)
    dev_entries = build_val_entries(opt1, split, device)
    del model1
    torch.cuda.empty_cache()
    val32_entries = build_val32_batches(opt1, picked, device)
    del opt1
    torch.cuda.empty_cache()

    if args.smoke:
        train_entries, dev_entries = train_entries[:1], dev_entries[:1]
    groups = {"train7": train_entries, "dev8": dev_entries, "val32": val32_entries}
    views = {"train7": (0, 1, 2, 3), "dev8": (0, 1, 2, 3),
             "val32": tuple(range(6))}
    payload = {
        "scope": "read-only attribution; development windows and a 32-pair official "
                 "val subset are reported separately and never mixed",
        "protocol_note": "this model uses GT camera poses to build rays; SIU3R is "
                         "unposed, so official numbers are not same-input comparisons",
        "plan_sha256": sha256_file(Path(args.plan)),
        "split_sha256": sha256_file(Path(args.split)),
        "val32": val32_meta,
        "window_counts": {k: len(v) for k, v in groups.items()},
        "arms": {},
        "arm_config": {name: {k: v for k, v in meta.items() if k != "directory"}
                       | {"directory": meta["directory"]}
                       for name, meta in ARMS.items()},
        "started_at": started,
    }
    for arm in ARMS:
        print(f"[attr] arm {arm}", flush=True)
        payload["arms"][arm] = analyse_arm(arm, args.preset, args.seed, device, plan,
                                           split, val32_entries, groups, views)
        payload["arms"][arm]["checkpoint_before"] = before[arm]
    after = {name: identity(Path(meta["directory"]) / "model.pt")
             for name, meta in ARMS.items()}
    payload["checkpoint_after"] = after
    payload["checkpoints_unchanged"] = after == before
    payload["elapsed_seconds"] = time.time() - started
    payload["official_semantic_per_class_source"] = {
        arm: f"group_plus/implementation_audit_v1/B1_full/{arm}_official_semantic.json"
        for arm in ARMS}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"[attr] wrote {out} in {payload['elapsed_seconds']:.0f}s "
          f"unchanged={payload['checkpoints_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
