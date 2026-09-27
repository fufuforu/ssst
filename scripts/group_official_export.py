#!/usr/bin/env python3
"""Export the group model's predictions into the pinned SIU3R evaluator contract.

Two fixed products are written per pair, exactly as pre-registered:

* ``semantic_pred``  -- the independent semantic head (``semantic_prob.argmax``,
  internal 0..19 -> 1..20, ``alpha <= 0.05`` -> void 0, instance id 0 everywhere);
  evaluated with the semantic-only switch (mIoU only, no PQ/mAP).
* ``panoptic_pred``  -- thing masks from the frozen GT-free group reader
  (``P(thing) >= 0.5``, 20-way argmax in the 18 thing classes, raw group mass
  > 0.5, area >= 50 px), one colour per query, instance id = query index + 1 and
  fixed across the pair's views, class = the query's predicted class, score =
  ``P(thing)``; remaining pixels become their stuff class (from the independent
  semantic head) when ``alpha > 0.05``, otherwise void.

Ground truth is written only into the separate ``*_gt`` trees.  The model is
never given the target frames' labels, and no target-frame information enters
the prediction assembly.  Read-only with respect to the checkpoint.

Frame naming follows the existing exporter: ``{scene}_pred{frame}.png`` /
``{scene}_gt{frame}.png`` (the pinned evaluator pairs them by
``name.replace("pred", "gt")``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
from scripts.group_eval_v2 import forward_group, group_predictions_v2  # noqa: E402
from scripts.evaluate_ssst_validation import (  # noqa: E402
    gt_maps,
    move_to_device,
    save_depth,
    save_rgb,
    save_segment,
    write_json,
)

VAL_ROOT = str(Path(DEFAULT_DATA_ROOT) / "val")
DEPTH_UNIT_SCALE = 1.0 / 0.15
MASK_THRESHOLD = 0.5
MIN_PRED_PIXELS = 50
STUFF_CLASSES = (0, 1)          # internal ids: wall, floor
THING_MIN, THING_MAX = 2, 19    # internal ids
THING_CLASSES = tuple(range(THING_MIN, THING_MAX + 1))
VOID_ALPHA = 0.05


def checkpoint_identity(directory: Path) -> dict:
    model_file = directory / "model.pt"
    return {"sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            "mtime": model_file.stat().st_mtime}


def load_group_model(directory: Path, args, device):
    opt = config_defaults[args.preset].evolve(
        seed=args.seed, group_arm="g0", group_recipe=bool(args.recipe),
        group_bg_supervision=not bool(args.recipe),
        group_recipe_head_mode=str(args.head_mode),
        group_recipe_seg_weight=float(args.instance_outer_weight),
        group_recipe_assign_coef=0.2, group_recipe_assign_every=1,
        group_recipe_assign_stop_shared_grad=bool(args.assign_stop_shared_grad),
        evaluating=True, use_input_supervision=False, num_views=6,
        batch_size=1, num_workers=0, num_input_views=2,
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
    )
    model = model_registry[opt.model_type](opt).to(device)
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location="cpu",
                   weights_only=False)["model"], strict=True,
    )
    model.eval()
    return opt, model


def semantic_prediction(forward, semantic_prob, alpha, num_views):
    """Independent semantic head -> packed (1-based) semantic, instance 0."""
    out = []
    for view in range(num_views):
        # forward['semantic_prob'] is [B, V, C, H, W]
        prob = semantic_prob[0, view]
        labels = prob.argmax(0).long().cpu().numpy()          # internal 0..19
        covered = alpha[0, view, 0].float().cpu().numpy() > VOID_ALPHA
        semantic = np.where(covered, labels + 1, 0).astype(np.int64)
        out.append((semantic, np.zeros_like(semantic)))
    del forward
    return out


def panoptic_prediction(forward, semantic_prob, alpha, num_views):
    """GT-free thing masks + stuff fill -> packed (1-based) semantic and instance ids."""
    from scripts.group_eval_v2 import group_score_table

    group_mass = forward["masks"]["group_mass"][0].float()
    scores = group_score_table(forward)
    results, per_view_stats = [], []
    for view in range(num_views):
        mass = group_mass[view]
        predictions, counts = group_predictions_v2(forward, view)
        height, width = mass.shape[-2:]
        semantic = np.zeros((height, width), dtype=np.int64)
        instance = np.zeros((height, width), dtype=np.int64)
        best = torch.zeros((height, width), dtype=torch.float32)
        chosen = torch.full((height, width), -1, dtype=torch.long)
        overlaps = 0
        # group_predictions_v2 returns predictions in ascending query order, so
        # "highest score*mass wins, ties keep the smaller query id" is exactly
        # "first writer wins on ties".
        for prediction in predictions:
            query = int(prediction["group"])
            mask = torch.from_numpy(prediction["mask"])
            score_mass = mass[query].float().cpu() * float(prediction["score"])
            overlaps += int((mask & (chosen >= 0)).sum())
            take = mask & ((chosen < 0) | (score_mass > best))
            take_np = take.numpy()
            semantic[take_np] = int(prediction["class"]) + 1
            instance[take_np] = query + 1
            best[take] = score_mass[take]
            chosen[take] = query
        labels = semantic_prob[0, view].argmax(0).long().cpu().numpy()
        covered = alpha[0, view, 0].float().cpu().numpy() > VOID_ALPHA
        unassigned = (chosen < 0).numpy()
        stuff = unassigned & covered & np.isin(labels, STUFF_CLASSES)
        semantic[stuff] = labels[stuff] + 1
        instance[stuff] = 0
        per_view_stats.append({"view": view, "candidates": len(predictions),
                               "gate_counts": counts,
                               "overlap_pixels": int(overlaps),
                               "thing_pixels": int((chosen >= 0).sum()),
                               "stuff_pixels": int(stuff.sum()),
                               # Report-only fix: the real void pixels of the panoptic
                               # product are (a) uncovered pixels and (b) covered pixels
                               # that no query claimed whose independent-semantic argmax
                               # is a thing class.  The previous expression reduced to
                               # `~covered` and silently dropped (b).  Prediction arrays
                               # and therefore every PNG and every official metric are
                               # unchanged (see structure_probe_v1/void_counter_fix.json).
                               "void_pixels": int(
                                   (~covered).sum()
                                   + (covered & unassigned
                                      & np.isin(labels, THING_CLASSES)).sum())})
        results.append((semantic, instance))
    # pred.json: one entry per query id (the pinned evaluator looks up
    # `info["id"] == instance_id` and averages `score` over matches).
    entries = [{
        "id": int(query) + 1,
        "label_id": int(scores["class_argmax20"][query].item()) + 1,
        "score": float(scores["p_thing"][query].item()),
    } for query in range(int(scores["class_argmax20"].shape[0]))]
    return results, entries, per_view_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="/space/mawb/SIU3R/data/scannet/val_pair.json")
    parser.add_argument("--val-root", default=VAL_ROOT)
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--head-mode", choices=("legacy_prefix", "pure4"),
                        default="legacy_prefix")
    parser.add_argument("--recipe", action="store_true", default=True)
    parser.add_argument("--no-recipe", dest="recipe", action="store_false",
                        help="the arm has no recipe head (e.g. G0+: plain legacy group head)")
    parser.add_argument("--instance-outer-weight", type=float, default=0.1)
    parser.add_argument("--assign-stop-shared-grad", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="write only the first pair and print the checks")
    parser.add_argument("--panoptic-dir", default="official_predictions_panoptic")
    parser.add_argument("--semantic-dir", default="official_predictions_semantic")
    parser.add_argument("--products", default="both",
                        choices=("both", "panoptic", "semantic"))
    parser.add_argument("--semantic-with-images", action="store_true",
                        help="also write rgb/depth for the semantic product (default: "
                             "segmentation only, since its official run disables "
                             "image/depth quality)")
    parser.add_argument("--hardlink-shared", action="store_true",
                        help="hardlink the shared rgb/depth trees instead of re-rendering")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    identity_before = checkpoint_identity(checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    opt, model = load_group_model(checkpoint, args, device)
    provider = SIU3RProcessedProvider(opt, root=args.val_root, subset="all",
                                      training=False, val_pair_json=args.manifest, rank=0)
    records = list(provider.dataset.val_pairs)
    if args.limit is not None:
        records = records[: args.limit]
        provider.dataset.val_pairs = records

    panoptic_root = output / args.panoptic_dir
    semantic_root = output / args.semantic_dir
    for root in (panoptic_root, semantic_root):
        root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for index in range(len(records)):
        sample = provider[index]
        batch = move_to_device(default_collate([sample]), device)
        with torch.no_grad():
            forward = forward_group(model, batch, opt)
        batch_frames = [int(x) for x in batch["frame_ids"][0]]
        context = [int(x) for x in records[index]["context_ids"]]
        target = [int(x) for x in records[index]["target_ids"]]
        scene = record_scene(records[index])
        if len(context) != 2 or len(target) != 6 or len(batch_frames) != 6:
            raise RuntimeError(f"record {index}: expected 2 context + 6 target frames, got "
                               f"context={context} target={target} batch={batch_frames}")
        if batch_frames[:2] != context:
            raise RuntimeError(f"record {index}: batch frames {batch_frames[:2]} != context "
                               f"{context}")
        if set(batch_frames) != set(target):
            raise RuntimeError(f"record {index}: batch frames {batch_frames} != target_ids "
                               f"{sorted(target)}")
        novel = batch_frames[2:]
        if sorted(novel) != sorted(set(target) - set(context)):
            raise RuntimeError(f"record {index}: novel {novel} is not target minus context")

        alpha = forward["masks"]["alpha"]
        render = forward["output"]["render"]
        semantic_prob = forward["semantic_prob"]
        semantic_maps = semantic_prediction(forward, semantic_prob, alpha, 6)
        panoptic, pred_info, view_stats = panoptic_prediction(
            forward, semantic_prob, alpha, 6)

        products = []
        if args.products in ("both", "panoptic"):
            products.append((panoptic_root, panoptic, True))
        if args.products in ("both", "semantic"):
            products.append((semantic_root, semantic_maps, False))
        for root, maps, is_panoptic in products:
            scene_dir = root / (f"{scene}_context" + "_".join(str(x) for x in context))
            with_images = is_panoptic or args.semantic_with_images
            subdirs = ["context_seg_pred", "context_seg_gt", "target_seg_pred",
                       "target_seg_gt"]
            if with_images:
                subdirs += ["rgb", "rgb_gt", "depth", "depth_gt"]
            for sub in subdirs:
                (scene_dir / sub).mkdir(parents=True, exist_ok=True)
            if is_panoptic:
                write_json(scene_dir / "context_seg_pred" / "pred.json", pred_info)
                write_json(scene_dir / "target_seg_pred" / "pred.json", pred_info)
            for view, frame_id in enumerate(batch_frames):
                if with_images:
                    save_rgb(scene_dir / "rgb" / f"{scene}_{frame_id}.png",
                             render["images_pred"][0, view])
                    save_rgb(scene_dir / "rgb_gt" / f"{scene}_{frame_id}.png",
                             batch["images_all"][0, view])
                    save_depth(scene_dir / "depth" / f"{scene}_{frame_id}.png",
                               render["depths_pred"][0, view] * DEPTH_UNIT_SCALE)
                    depth_gt = np.asarray(
                        Image.open(Path(args.val_root) / scene / "depth" / f"{frame_id}.png")
                    ).astype(np.float32) / 1000.0
                    save_depth(scene_dir / "depth_gt" / f"{scene}_{frame_id}.png",
                               torch.from_numpy(depth_gt).unsqueeze(0))
                sem_gt, ins_gt = gt_maps(batch["semantic_label_all"][0, view],
                                         batch["instance_label_all"][0, view])
                semantic_map, instance_map = maps[view]
                save_segment(scene_dir / "target_seg_pred" / f"{scene}_pred{frame_id}.png",
                             torch.from_numpy(semantic_map), torch.from_numpy(instance_map))
                save_segment(scene_dir / "target_seg_gt" / f"{scene}_gt{frame_id}.png",
                             sem_gt, ins_gt)
                if view < 2:
                    save_segment(
                        scene_dir / "context_seg_pred" / f"{scene}_pred{frame_id}.png",
                        torch.from_numpy(semantic_map), torch.from_numpy(instance_map))
                    save_segment(
                        scene_dir / "context_seg_gt" / f"{scene}_gt{frame_id}.png",
                        sem_gt, ins_gt)
        report_rows.append({
            "record_index": index, "scene": scene,
            "context_ids": context, "target_ids": target,
            "batch_frame_ids": batch_frames, "novel_ids": novel,
            "panoptic_views": view_stats,
            "alpha_conservation": float(
                (forward["masks"]["group_mass"][0].sum(1)  # sum over the 100 query channels
                 + forward["masks"]["background_mass"][0, :, 0]
                 - alpha[0, :, 0]).abs().max()),
        })
        if args.smoke:
            print(json.dumps(report_rows[-1], indent=1)[:2000])
            break
        if index % 25 == 0:
            print(f"[export] {index}/{len(records)} {scene}", flush=True)
        del forward
        torch.cuda.empty_cache()

    identity_after = checkpoint_identity(checkpoint)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": identity_before["sha256"],
        "checkpoint_sha256_after": identity_after["sha256"],
        "checkpoint_unchanged": identity_after == identity_before,
        "manifest": args.manifest,
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "records": len(report_rows),
        "policy": {
            "semantic": "independent semantic head argmax; internal 0..19 -> 1..20; "
                        "alpha<=0.05 -> void 0; instance id 0",
            "panoptic": "P(thing)>=0.5, 20-way argmax in 2..19, raw group mass>0.5, "
                        "area>=50px; instance id = query+1 fixed across the pair; class = "
                        "query class; score = P(thing)",
            "stuff": "remaining pixels take their independent-semantic stuff class when "
                     "alpha>0.05; otherwise void (0,0)",
            "gt_free": True,
            "depth": f"rendered depth * {DEPTH_UNIT_SCALE} (fixed 1/0.15, never GT-fitted)",
        },
        "rows": report_rows,
        "prediction_dirs": {"panoptic": str(panoptic_root), "semantic": str(semantic_root)},
    }
    write_json(output / "export_report.json", report)
    print(f"[export] wrote {len(report_rows)} records; unchanged="
          f"{report['checkpoint_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
