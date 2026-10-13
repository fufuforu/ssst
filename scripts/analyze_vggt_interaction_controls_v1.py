"""Audit and plot saved interaction controls; never run a model or rescore images.

Uses native SIU3R saved per-frame reconstruction scores for a paired scene
bootstrap. Global segmentation metrics are copied, never averaged per scene.
Run with the tokengs Python environment (NumPy/Matplotlib), on CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from format_object_locus_posefree_siu3r_table import FIELDS, HEADERS, PRECISION, table_values

ARMS = ("bidirectional", "forward_only", "gated_feedback", "independent_feedback")
STEPS = (0, 100, 300, 500)
LAYERS = ("L6", "L8", "L10", "L12")
LABELS = ("Original", "No writeback", "Gated writeback", "Independent writeback")
CLASSES = ("wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
           "door", "window", "bookshelf", "picture", "counter", "desk",
           "curtain", "refrigerator", "shower curtain", "toilet", "sink",
           "bathtub", "otherfurniture")


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation = args.report_root / "evaluation"
    reduced = read(evaluation / "analysis/results.json")
    assert reduced["heldout_gt_identical"] and reduced["initial_rgb_identical"]
    assert reduced["optimizer_updates"] == 0
    startup = read(args.report_root / "startup_validation.json")
    assert startup["common_initial_parameters_identical"] and not startup["pending"]
    rows, tables, scenes, native, audit = {}, {}, {}, {}, {}
    reference_sampling = None
    reference_records = None
    for arm in ARMS:
        root = args.run_root / arm
        complete = read(root / "COMPLETE.json")
        assert complete["status"] == "INTERACTION_TRAINING_COMPLETED"
        assert (complete["completed_updates"], complete["completed_exposures"]) == (500, 4000)
        assert complete["frozen_parameters_unchanged"]
        assert complete["source_checkpoint_sha256"] == reduced["source_checkpoint_sha256"]
        assert complete["execution_git_sha"] == reduced["training_sha"]
        cfg = complete["config"]
        assert cfg["global_batch"] == 8 and cfg["geometry_quality_policy"] == "monitor_v1"
        assert cfg["adapter_understanding_scale"] == cfg["gs_understanding_scale"] == 0.01
        assert cfg["vggt_trainable"] is False and cfg["state_layers"] == [6, 8, 10, 12]
        ranks = [[json.loads(line) for line in (root / f"training_rank{i}.jsonl").read_text().splitlines()]
                 for i in range(4)]
        assert all(len(rank) == 500 for rank in ranks)
        for rank in ranks:
            assert [x["completed_updates"] for x in rank] == list(range(1, 501))
            assert [x["completed_exposures"] for x in rank] == list(range(8, 4001, 8))
            assert all(x["frozen_parameters_unchanged"] for x in rank)
            assert all(np.isfinite(x[key]) for x in rank for key in
                       ("loss_recon", "loss_understanding", "preclip_norm", "lr_multiplier"))
            assert all(x["actual_feedback_beta"] == (0 if arm == "forward_only" else x["beta"])
                       for x in rank)
        sampling = [[(x["window_slots"], x["source_indices"]) for x in rank] for rank in ranks]
        if reference_sampling is None:
            reference_sampling = sampling
        assert sampling == reference_sampling
        # Each complete 64-update epoch exposes all 512 cohort windows once.
        for start in (0, 64, 128, 192, 256, 320, 384):
            slots = [slot for rank in ranks for row in rank[start:start+64] for slot in row["window_slots"]]
            assert len(slots) == 512 and set(slots) == set(range(512))
        rows[arm] = ranks
        losses = {}
        for key in ("loss_recon", "loss_understanding", "preclip_norm"):
            losses[key] = {"epoch1_mean": float(np.mean([x[key] for rank in ranks for x in rank[:64]])),
                           "epoch7_mean": float(np.mean([x[key] for rank in ranks for x in rank[384:448]]))}
        interactions = {}
        for layer in LAYERS:
            observations = [x["interactions"].get(layer, {}) for rank in ranks for x in rank[400:]]
            interactions[layer] = {"last100_mean_delta_ratio": float(np.mean([x.get("delta_ratio_mean", 0) for x in observations]))}
            if arm == "gated_feedback":
                interactions[layer].update(last100_mean_gate=float(np.mean([x["gate_mean"] for x in observations])),
                                           last100_lowest_observed_gate=float(min(x["gate_min"] for x in observations)))
        audit[arm] = {"complete": complete, "losses": losses, "interactions": interactions,
                      "preclip_over1_fraction": float(np.mean([x["preclip_norm"] > 1 for rank in ranks for x in rank])),
                      "max_peak_allocated_gib": max(x["peak_allocated"] for rank in ranks for x in rank) / 2**30,
                      "evaluation_checkpoint_receipts": {}}
        for step in STEPS:
            condition = f"{arm}/{step}"
            ev = evaluation / arm / f"update{step:04d}"
            metrics = read(ev / "metrics.json")
            exported = read(ev / "export_complete.json")
            scored = read(ev / "score_complete.json")
            saved = read(root / f"checkpoint_update_{step:04d}.json")
            assert saved["status"] == "CHECKPOINT_VERIFIED" and saved["step"] == step
            assert metrics["checkpoint_sha256"] == exported["checkpoint_sha256"] == saved["sha256"]
            assert metrics["optimizer_updates"] == exported["optimizer_updates"] == scored["optimizer_updates"] == 0
            assert metrics["windows"] == metrics["unique_scenes"] == 64
            assert metrics["official_commit"] == "8ea80166be76854f938e90521f1a5b688b755c87"
            tables[condition] = table_values(metrics)
            assert tables[condition] == reduced["conditions"][condition]
            records = sorted(exported["records"], key=lambda x: x["name"])
            identifiers = [(x["name"], x["source_index"], x["frame_ids"], x["input_rgb_sha256"]) for x in records]
            if reference_records is None:
                reference_records = identifiers
            assert identifiers == reference_records and len(set(x[0] for x in identifiers)) == 64
            means = []
            for record in records:
                scene = ev / "all" / record["name"]
                render = read(scene / "render_scores.json")
                depth = read(scene / "depth_scores.json")
                assert len(render) == len(depth) == 6
                assert sorted(x["item"] for x in render) == sorted(x["item"] for x in depth)
                means.append([np.mean([x[k] for x in depth]) for k in ("absrel", "rmse")] +
                             [np.mean([x[k] for x in render]) for k in ("psnr", "ssim", "lpips")])
            scenes[condition] = np.asarray(means)
            assert np.isfinite(scenes[condition]).all()
            assert np.allclose(scenes[condition].mean(axis=0), tables[condition][:5], rtol=0, atol=1e-12)
            native[condition] = read(ev / "all/results.json")
            assert native[condition] == metrics["official_native_result"]
            audit[arm]["evaluation_checkpoint_receipts"][str(step)] = saved

    # Paired scenes, not frames; descriptive dataset uncertainty at one training seed.
    indices = np.random.default_rng(42).integers(0, 64, size=(10000, 64))
    bootstrap = {}
    for arm in ARMS:
        for reference in ("bidirectional/500", f"{arm}/0"):
            condition = f"{arm}/500"
            delta = scenes[condition] - scenes[reference]
            draws = delta[indices].mean(axis=1)
            bounds = np.quantile(draws, [0.025, 0.975], axis=0)
            bootstrap[f"{condition} minus {reference}"] = {
                k: {"mean_delta": float(delta[:, i].mean()), "scene_bootstrap_95_percentile_interval": bounds[:, i].tolist(),
                    "positive_scene_count": int((delta[:, i] > 0).sum())}
                for i, k in enumerate(FIELDS[:5])}
    write(args.output / "audit.json", {"status": "SAVED_ARTIFACTS_VERIFIED", "training": audit,
          "sampling_identical": True, "complete_epoch_sampling_verified": True,
          "all16_evaluation_sha_chains_verified": True,
          "scene_scores_reproduce_official_aggregates": True,
          "initial_rgb_identical": reduced["initial_rgb_identical"], "heldout_gt_identical": reduced["heldout_gt_identical"],
          "interaction_observation_scope": "Each rank/update logs the final microbatch's token summary; no full-token distribution claimed."})
    write(args.output / "bootstrap.json", {"seed": 42, "samples": 10000, "paired_scenes": 64,
          "scope": "Only reconstruction metrics; no bootstrap of global AP/PQ or training-seed uncertainty.", "comparisons": bootstrap})
    write(args.output / "scene_scores.json", {"scene_names": [x[0] for x in reference_records],
          "columns": list(FIELDS[:5]), "conditions": {k: v.tolist() for k, v in scenes.items()}})
    with (args.output / "siu3r_table1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", *FIELDS])
        for condition, values in tables.items():
            writer.writerow([condition, *values])
    final = ["bidirectional/0"] + [f"{arm}/500" for arm in ARMS]
    markdown = ["| Condition | " + " | ".join(HEADERS) + " |", "|---|" + "---:|" * 13]
    for condition in final:
        markdown.append("| " + condition + " | " + " | ".join("" if v is None else f"{v:.{p}f}"
                        for v, p in zip(tables[condition], PRECISION)) + " |")
    (args.output / "final_table.md").write_text("\n".join(markdown) + "\n")
    # Class IDs in mAP are zero-based (semantic ID minus one).
    with (args.output / "per_class_deltas.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "view", "metric", "class", "baseline", "variant", "delta"])
        for arm in ARMS[1:]:
            base, variant = native["bidirectional/500"], native[f"{arm}/500"]
            for view in ("context", "target"):
                for suffix in ("ious_per_class", "pqs_per_class"):
                    b, v = np.asarray(base[f"{view}_{suffix}"]).ravel(), np.asarray(variant[f"{view}_{suffix}"]).ravel()
                    for name, x, y in zip(CLASSES, b, v):
                        writer.writerow([arm, view, suffix, name, x, y, y-x])
                b, v = base[f"{view}_map"], variant[f"{view}_map"]
                assert b["classes"] == v["classes"]
                for class_id, x, y in zip(b["classes"], b["map_per_class"], v["map_per_class"]):
                    writer.writerow([arm, view, "AP", CLASSES[class_id], x, y, y-x])

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for ax, col in zip(axes.flat, [i for i in range(13) if i not in (8, 12)]):
        for arm, label in zip(ARMS, LABELS):
            ax.plot(STEPS, [tables[f"{arm}/{step}"][col] for step in STEPS], "o-", label=label)
        ax.set_title(FIELDS[col]); ax.set_xlabel("New optimizer updates"); ax.grid(alpha=.3)
        ax.axvline(125, color="gray", alpha=.4, linestyle=":")
        ax.axvline(200, color="gray", alpha=.4, linestyle="--")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    fig.suptitle("64-scene diagnostic; dotted: feedback beta reaches full; dashed: LR warmup ends")
    fig.savefig(args.output / "heldout_curves.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, ax in zip(("loss_recon", "loss_understanding", "preclip_norm"), axes.flat):
        for arm, label in zip(ARMS, LABELS):
            values = np.asarray([[x[key] for x in rank] for rank in rows[arm]]).mean(axis=0)
            ax.plot(np.arange(25, 501), np.convolve(values, np.ones(25)/25, mode="valid"), label=label)
        ax.set_title(key + " (25-update moving mean)"); ax.grid(alpha=.3); ax.set_xlabel("New optimizer updates")
    ax = axes.flat[-1]
    for layer in LAYERS:
        values = np.asarray([[x["interactions"][layer]["gate_mean"] for x in rank]
                             for rank in rows["gated_feedback"]]).mean(axis=0)
        ax.plot(np.arange(1, 501), values, label=layer)
    ax.set_title("Learned gate mean (last microbatch/rank)"); ax.set_ylim(.89, 1.005); ax.grid(alpha=.3); ax.legend()
    axes.flat[0].legend(fontsize=8)
    fig.savefig(args.output / "training_and_gates.png", dpi=160); plt.close(fig)
    print(json.dumps({"output": str(args.output), "audit": "PASS", "bootstrap": bootstrap,
                      "training_summary": {k: {"losses": v["losses"], "interactions": v["interactions"]}
                                           for k, v in audit.items()}}, indent=2))


if __name__ == "__main__":
    main()
