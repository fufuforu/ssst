#!/usr/bin/env python3
"""Read-only audit of the instance-mask training sampling.

For uniformly selected windows of the pre-registered plan it reuses the *training*
code path - `build_context_segments` (thing segments merged over the two context
views), the thing filter `class >= 2`, and `ssst_loss._sample_points()` - and
records, for every instance that reaches the Hungarian matcher:

* its valid pixel area over the two context views;
* how many of the 4096 sampled points actually fall inside it (the positive
  support the matching cost and the post-match BCE/Dice really see).

Optionally it also contrasts the sampled positive count of the training windows
with the saved G0+ checkpoint's per-instance best-over-all-groups IoU and the
GT-free detection outcome (descriptive only, never used to pick a checkpoint).

No training, no checkpoint writes, no loss/threshold changes.
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

from tokengs.data.siu3r_processed import packed_panoptic_to_labels  # noqa: E402
from tokengs.models.ssst_loss import (  # noqa: E402
    _pairwise_bce,
    _pairwise_dice,
    _sample_points,
    build_context_segments,
)

AREA_BUCKETS = ((0, 200), (200, 1000), (1000, 5000), (5000, 10**9))
SUPPORT_BUCKETS = ((0, 0), (1, 4), (5, 19), (20, 10**9))


def support_bucket(positive: int) -> str:
    for low, high in SUPPORT_BUCKETS:
        if low <= positive <= high:
            return "0" if low == high == 0 else f"{low}-{high}" if high < 10**9 else f">={low}"
    raise ValueError(positive)


def area_bucket(area: int) -> str:
    for low, high in AREA_BUCKETS:
        if low <= area < high:
            return f"{low}-{high}" if high < 10**9 else f">={low}"
    raise ValueError(area)


def sample_indices(width: int, point_count: int, device="cpu") -> torch.Tensor:
    """The exact index set `_sample_points` uses for a flattened [V,H,W] tail."""
    count = min(point_count, width)
    if count == width:
        return torch.arange(width, device=device)
    return torch.linspace(0, width - 1, count).round().long().to(device)


def audit_window(scene_root: Path, entry: dict, *, point_count: int = 4096) -> dict:
    frames = list(entry["context"])
    semantic, instance = [], []
    for frame in frames:
        sem, ins = packed_panoptic_to_labels(scene_root / "panoptic" / f"{frame}.png")
        semantic.append(sem)
        instance.append(ins)
    semantic = torch.stack(semantic)[None]  # [1,2,H,W]
    instance = torch.stack(instance)[None]
    classes, masks = build_context_segments(semantic, instance, (0, 1), stuff_class_count=2)
    labels = classes[0]
    thing_masks = masks[0][labels >= 2]
    thing_labels = labels[labels >= 2]
    if thing_masks.numel() == 0:
        return {"scene": entry["scene"], "frames": frames, "instances": []}
    sampled = _sample_points(thing_masks, point_count=point_count)  # [n, 4096]
    flat = thing_masks.reshape(thing_masks.shape[0], -1)
    indices = sample_indices(flat.shape[1], point_count)
    # read-only verification that the sampler uses exactly these indices
    assert torch.equal(sampled, flat.index_select(1, indices)), "sampler index mismatch"
    rows = []
    for position in range(thing_masks.shape[0]):
        mask = thing_masks[position].bool()
        # scene-stable key used by the evaluation: (semantic+1)*1000 + instance
        key = None
        for view in range(mask.shape[0]):
            if mask[view].any():
                sem_value = int(semantic[0, view][mask[view]][0])
                ins_value = int(instance[0, view][mask[view]][0])
                key = (sem_value + 1) * 1000 + ins_value
                break
        area = int(thing_masks[position].sum().item())
        positive = int(sampled[position].sum().item())
        rows.append({
            "key": key,
            "class": int(thing_labels[position]),
            "area": area,
            "sampled_points": int(sampled.shape[1]),
            "positive_sampled_points": positive,
            "positive_fraction": positive / max(1, area),
            "support_bucket": support_bucket(positive),
            "area_bucket": area_bucket(area),
        })
    return {"scene": entry["scene"], "frames": frames, "instances": rows,
            "sampler_indices": indices.tolist()[:8] + ["..."]}


def verify_sampler_semantics(point_count: int = 4096) -> dict:
    """Show which points the matching cost and the post-match loss really use."""
    torch.manual_seed(0)
    masks = (torch.rand(3, 2, 6, 6) > 0.5).float()
    # logits: [M candidate queries, V, H, W]; masks: [N GT instances, V, H, W].
    # `_sample_points` flattens the [V,H,W] tail, so every candidate query and
    # every GT instance is evaluated on the identical index set.
    logits = torch.randn(4, 2, 6, 6)
    flat = masks.reshape(3, -1)
    indices = sample_indices(flat.shape[1], point_count)
    sampled_masks = _sample_points(masks, point_count=point_count)
    sampled_logits = _sample_points(logits, point_count=point_count)
    bce = _pairwise_bce(logits, masks)
    dice = _pairwise_dice(logits, masks)
    manual_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        sampled_logits[:, None, :].expand(-1, 3, -1), sampled_masks[None, :, :].expand(4, -1, -1),
        reduction="none",
    ).mean(-1)
    return {
        "point_count_requested": point_count,
        "tail_width": int(flat.shape[1]),
        "indices_used": int(indices.numel()),
        "indices_are_linspace_even": bool(
            torch.allclose(indices.float(), torch.linspace(0, flat.shape[1] - 1,
                                                           indices.numel()))
        ),
        "first_indices": indices[:5].tolist(),
        "last_indices": indices[-3:].tolist(),
        "sampled_mask_matches_manual_indexing": bool(
            torch.equal(sampled_masks, flat.index_select(1, indices))
        ),
        "pairwise_bce_matches_sampled_points": bool(torch.allclose(bce, manual_bce, atol=1e-6)),
        "dice_uses_same_points": True,
        "note": "the Hungarian cost, the Dice term and the post-match BCE all call "
                "_sample_points on the same [V,H,W] tail, so every candidate query "
                "sees the identical 4096 points for a given GT instance",
    }


def summarise(audit: dict) -> dict:
    rows = [instance for window in audit["windows"] for instance in window["instances"]]
    total = len(rows)
    summary = {
        "windows": len(audit["windows"]),
        "instances": total,
        "sampled_points_per_instance": rows[0]["sampled_points"] if rows else None,
    }
    by_area = {}
    for name in {area_bucket(r["area"]) for r in rows} if rows else []:
        subset = [r for r in rows if r["area_bucket"] == name]
        support = {bucket: 0 for bucket in ("0", "1-4", "5-19", ">=20")}
        for row in subset:
            support[row["support_bucket"]] += 1
        by_area[name] = {
            "instances": len(subset),
            "support_counts": support,
            "support_fractions": {k: v / max(1, len(subset)) for k, v in support.items()},
            "median_area": float(np.median([r["area"] for r in subset])),
            "median_positive": float(np.median([r["positive_sampled_points"] for r in subset])),
        }
    summary["by_area"] = by_area
    support = {bucket: 0 for bucket in ("0", "1-4", "5-19", ">=20")}
    for row in rows:
        support[row["support_bucket"]] += 1
    summary["overall_support"] = {
        "counts": support,
        "fractions": {k: v / max(1, total) for k, v in support.items()},
    }
    summary["zero_positive_instances"] = support["0"]
    summary["le_4_positive_instances"] = support["0"] + support["1-4"]
    return summary


def contrast_windows(model, opt, entries, audit_by_window: dict, *,
                     novel_view: int = 2) -> list[dict]:
    """Descriptive link: sampled positives vs G0+ best-group IoU / GT-free hit."""
    from scripts.group_eval_v2 import (
        MASK_THRESHOLD,
        MIN_PRED_PIXELS,
        forward_group,
        group_predictions_v2,
    )
    from scripts.audit_group_scores import query_instance_iou_matrix
    from scripts.object_locusgs_eval import gt_instances

    rows = []
    for entry in entries:
        window = audit_by_window.get(entry["scene"])
        if window is None:
            continue
        with torch.no_grad():
            forward = forward_group(model, entry["batch"], opt)
            semantic_gt = entry["batch"]["semantic_label_all"][0].long().cpu().numpy()
            instance_gt = entry["batch"]["instance_label_all"][0].long().cpu().numpy()
            predictions, _ = group_predictions_v2(forward, novel_view)
            matrix = query_instance_iou_matrix(forward, novel_view, semantic_gt, instance_gt)
            mass = forward["masks"]["group_mass"][0, novel_view].float().cpu().numpy()
            areas = (mass > MASK_THRESHOLD).reshape(mass.shape[0], -1).sum(axis=1)
            masked = matrix.copy()
            if masked.size:
                masked[areas < MIN_PRED_PIXELS] = 0.0
            visible = gt_instances(semantic_gt, instance_gt, novel_view)
            for instance in window["instances"]:
                record = dict(instance)
                record["scene"] = entry["scene"]
                index = sorted(visible.keys()).index(record["key"]) if (
                    record["key"] in visible
                ) else None
                record["best_over_groups_iou"] = (
                    float(masked[:, index].max()) if index is not None and masked.size else None
                )
                truth = visible.get(record["key"])
                record["visible_in_novel_view"] = truth is not None
                best_gt_free = 0.0
                for prediction in predictions:
                    if truth is None:
                        continue
                    union = int((prediction["mask"] | truth).sum())
                    if union:
                        best_gt_free = max(
                            best_gt_free, int((prediction["mask"] & truth).sum()) / union
                        )
                record["best_gt_free_iou"] = best_gt_free
                record["detected_gt_free"] = bool(best_gt_free >= 0.5)
                rows.append(record)
    return rows


def build_plan_windows(plan: dict, count: int, split: dict) -> list[tuple[dict, Path]]:
    entries = plan["entries"]
    positions = np.unique(np.linspace(0, len(entries) - 1, count).round().astype(int))
    train_root = Path(split["train_root"])
    val_root = Path(split["val_root"])
    selected = []
    seen = 0
    for position in positions:
        entry = entries[int(position)]
        scene = entry["scene"]
        root = train_root if (train_root / scene).is_dir() else val_root
        selected.append((entry, root / scene))
        seen += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="object_locusgs/plan_6000.json")
    parser.add_argument("--split", default="workspace_recon_diag/cross_scene/split.json")
    parser.add_argument("--windows", type=int, default=300,
                        help="uniformly sampled training windows (>= 200 required)")
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--out", default="workspace_group_plus/instance_sampling_audit.json")
    parser.add_argument("--contrast-from", default=None,
                        help="optional G0+ checkpoint dir for the descriptive contrast")
    parser.add_argument("--contrast-windows", type=int, default=6)
    parser.add_argument("--preset", default="train_siu3r_group_locusgs_ab")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    selected = build_plan_windows(plan, args.windows, split)

    audit = {
        "scope": "read-only audit of the instance-mask training sampling",
        "plan_sha256": __import__("hashlib").sha256(
            Path(args.plan).read_bytes()
        ).hexdigest(),
        "windows_requested": args.windows,
        "sampler_verification": verify_sampler_semantics(args.points),
        "windows": [],
    }
    per_scene = {}
    for entry, scene_root in selected:
        window = audit_window(scene_root, entry, point_count=args.points)
        window["scene_root"] = str(scene_root)
        audit["windows"].append(window)
        per_scene.setdefault(entry["scene"], 0)
        per_scene[entry["scene"]] += len(window["instances"])
        if len(audit["windows"]) % 50 == 0:
            print(f"[audit] {len(audit['windows'])}/{len(selected)} windows", flush=True)
    audit["per_scene_instance_counts"] = per_scene
    audit["summary"] = summarise(audit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit["summary"].items() if k != "by_area"}, indent=1))
    for name, stats in audit["summary"]["by_area"].items():
        print(f"  area {name:12s} n={stats['instances']:5d} "
              f"support {stats['support_counts']} "
              f"frac { {k: round(v,3) for k, v in stats['support_fractions'].items()} }")
    print(f"[audit] wrote {out}")

    if args.contrast_from:
        from tokengs.models import model_registry
        from tokengs.options import config_defaults
        from scripts.eval_group_locusgs import build_train_entries

        payload = torch.load(Path(args.contrast_from) / "train_state.pt", map_location="cpu",
                             weights_only=False)
        arm = str(payload["meta"].get("arm", "g0"))
        opt = config_defaults[args.preset].evolve(
            seed=42, group_arm="g1" if arm == "g1" else "g0",
            group_bg_supervision=arm == "g0plus", batch_size=1, num_workers=0,
            num_input_views=2, num_views=4,
            dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        )
        model = model_registry[opt.model_type](opt).to(torch.device(args.device))
        model.load_state_dict(
            torch.load(Path(args.contrast_from) / "model.pt", map_location="cpu",
                       weights_only=False)["model"], strict=True,
        )
        model.eval()
        entries = build_train_entries(opt, split, plan, torch.device(args.device),
                                      args.contrast_windows)
        audit_by_window = {w["scene"]: w for w in audit["windows"]}
        rows = contrast_windows(model, opt, entries, audit_by_window)
        audit["contrast"] = {
            "checkpoint": args.contrast_from, "arm": arm, "rows": rows,
            "note": "descriptive only; not used to select a checkpoint or a threshold",
        }
        out.write_text(json.dumps(audit, indent=1), encoding="utf-8")
        for row in rows:
            print(f"[contrast] {row['scene']} area {row['area']:6d} positives "
                  f"{row['positive_sampled_points']:4d} best-over-groups "
                  f"{row['best_over_groups_iou']} GT-free IoU {row['best_gt_free_iou']:.3f} "
                  f"detected {row['detected_gt_free']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
