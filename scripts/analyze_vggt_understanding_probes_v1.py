"""Audit saved full-pool probes and compare native SIU3R scores, without inference.

Paired scene bootstrap is limited to reconstruction. Global AP/PQ and semantic
IoU are copied from the official evaluator, with no per-scene approximation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_vggt_interaction_controls_v1 import CLASSES, read, write
from format_object_locus_posefree_siu3r_table import FIELDS, HEADERS, PRECISION, table_values

ARMS = ("joint_full", "u_read", "u_no_anchor_read", "u_fusion")
STEPS = (0, 100, 300, 500)
LABELS = ("Full-pool joint", "Fixed reconstruction + anchor read",
          "Fixed reconstruction, no anchor feature residual", "Fixed reconstruction + geometry fusion")


def gradient_summary(ranks, selected_steps):
    output = {}
    for prefix in ("anchor_decoder.", "vggt_memory_adapter.", "understanding.", "panoptic."):
        observations = [d[prefix] for rank in ranks for row in rank
                        if row["completed_updates"] in selected_steps
                        for d in row.get("microbatch_gradient_diagnostics", []) if prefix in d]
        if not observations:
            continue
        valid = [x for x in observations if x["cosine"] is not None]
        scale = .01 if prefix in ("anchor_decoder.", "vggt_memory_adapter.") else 1.
        ratios = [scale*x["weighted_understanding_norm"]/x["reconstruction_norm"]
                  for x in valid if x["reconstruction_norm"] > 0]
        output[prefix] = {
            "local_microbatch_observations": len(observations),
            "nonzero_both_observations": len(valid),
            "mean_local_cosine": float(np.mean([x["cosine"] for x in valid])) if valid else None,
            "negative_local_cosine_fraction": float(np.mean([x["cosine"] < 0 for x in valid])) if valid else None,
            "median_scaled_U_over_R_norm": float(np.median(ratios)) if ratios else None,
            "p90_scaled_U_over_R_norm": float(np.quantile(ratios, .9)) if ratios else None,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-report-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation = args.report_root / "evaluation"
    reduced = read(evaluation / "analysis/results.json")
    assert reduced["heldout_gt_identical"] and reduced["initial_rgb_identical"]
    assert reduced["independent_understanding_rgb_unchanged"] and reduced["optimizer_updates"] == 0
    startup = read(args.report_root / "startup_validation_after_migration.json")
    assert startup["common_initial_parameters_identical"] and not startup["pending"]
    assert {x["arm"] for x in startup["rows"]} == set(ARMS)
    cohort = read(args.cohort)
    cohort_sha = hashlib.sha256(args.cohort.read_bytes()).hexdigest()
    expected_stream = cohort["train_source_indices"]
    assert len(expected_stream) == len(set(expected_stream)) == 4000
    assert np.array_equal(expected_stream, np.random.default_rng(42).permutation(8337)[:4000])
    rows, tables, scenes, native, audit = {}, {}, {}, {}, {}
    reference_sampling, reference_records = None, None
    for arm in ARMS:
        root = args.run_root / arm
        complete = read(root / "COMPLETE.json")
        cfg = complete["config"]
        assert complete["status"] == "INTERACTION_TRAINING_COMPLETED"
        assert (complete["completed_updates"], complete["completed_exposures"]) == (500, 4000)
        assert complete["frozen_parameters_unchanged"]
        assert complete["source_checkpoint_sha256"] == reduced["source_checkpoint_sha256"]
        assert complete["execution_git_sha"] == reduced["training_shas"][arm]
        assert cfg["cohort_sha256"] == cohort_sha
        assert cfg["global_batch"] == 8 and cfg["geometry_quality_policy"] == "monitor_v1"
        assert cfg["adapter_understanding_scale"] == cfg["gs_understanding_scale"] == .01
        assert cfg["vggt_trainable"] is False and cfg["state_layers"] == [6, 8, 10, 12]
        assert cfg["reconstruction_update"] == (arm == "joint_full")
        assert cfg["feedback_enabled"] == (arm == "joint_full")
        ranks = [[json.loads(line) for line in (root / f"training_rank{i}.jsonl").read_text().splitlines()]
                 for i in range(4)]
        assert all(len(rank) == 500 for rank in ranks)
        for rank in ranks:
            assert [x["completed_updates"] for x in rank] == list(range(1, 501))
            assert [x["completed_exposures"] for x in rank] == list(range(8, 4001, 8))
            assert all(x["frozen_parameters_unchanged"] for x in rank)
            assert all(np.isfinite(x[k]) for x in rank for k in
                       ("loss_recon", "loss_understanding", "preclip_norm", "lr_multiplier"))
            assert all(x["actual_feedback_beta"] == (x["beta"] if arm == "joint_full" else 0.) for x in rank)
        sampling = [[(x["window_slots"], x["source_indices"]) for x in rank] for rank in ranks]
        if reference_sampling is None:
            reference_sampling = sampling
        assert sampling == reference_sampling
        ordered = [idx for step in range(500) for rank in ranks for idx in rank[step]["source_indices"]]
        assert ordered == expected_stream
        rows[arm] = ranks
        audit[arm] = {
            "complete": complete,
            "losses": {k: {"first100_mean": float(np.mean([x[k] for rank in ranks for x in rank[:100]])),
                            "last100_mean": float(np.mean([x[k] for rank in ranks for x in rank[400:]]))}
                       for k in ("loss_recon", "loss_understanding", "preclip_norm")},
            "preclip_over1_fraction": float(np.mean([x["preclip_norm"] > 1 for rank in ranks for x in rank])),
            "max_peak_allocated_gib": max(x["peak_allocated"] for rank in ranks for x in rank)/2**30,
            "gradients_first20": gradient_summary(ranks, set(range(1, 21))),
            "gradients_late_probes": gradient_summary(ranks, {100, 300, 500}),
            "evaluation_checkpoint_receipts": {},
        }
        for step in STEPS:
            saved = read(root / f"checkpoint_update_{step:04d}.json")
            assert saved["status"] == "CHECKPOINT_VERIFIED" and saved["step"] == step
            audit[arm]["evaluation_checkpoint_receipts"][str(step)] = saved

    # Inspect the original native score files for all17 conditions, and the
    # preserved old data-only reference. No deleted old weights are required.
    conditions = [(f"{arm}/{step}", evaluation/arm/f"update{step:04d}") for arm in ARMS for step in STEPS]
    conditions += [("joint_full/initial_feedback_full", evaluation/"joint_full/initial_feedback_full"),
                   ("old128_bidirectional/500", args.old_report_root/"evaluation/bidirectional/update0500")]
    old_reduced = read(args.old_report_root/"evaluation/analysis/results.json")
    assert old_reduced["source_checkpoint_sha256"] == reduced["source_checkpoint_sha256"]
    assert old_reduced["conditions"]["bidirectional/0"] == reduced["conditions"]["joint_full/0"]
    for condition, ev in conditions:
        metrics = read(ev/"metrics.json")
        exported = read(ev/"export_complete.json")
        scored = read(ev/"score_complete.json")
        assert metrics["checkpoint_sha256"] == exported["checkpoint_sha256"]
        assert metrics["optimizer_updates"] == exported["optimizer_updates"] == scored["optimizer_updates"] == 0
        assert metrics["windows"] == metrics["unique_scenes"] == 64
        assert metrics["official_commit"] == "8ea80166be76854f938e90521f1a5b688b755c87"
        tables[condition] = table_values(metrics)
        if condition.startswith("old128"):
            assert tables[condition] == old_reduced["conditions"]["bidirectional/500"]
        else:
            assert tables[condition] == reduced["conditions"][condition]
            arm, step = condition.split("/")
            saved_step = 0 if step == "initial_feedback_full" else int(step)
            assert metrics["checkpoint_sha256"] == audit[arm]["evaluation_checkpoint_receipts"][str(saved_step)]["sha256"]
            if step == "initial_feedback_full":
                assert metrics["step"] == 0 and exported["feedback_exposure"] == 1000
                assert "untrained exact boundary" in metrics["intervention"]
        records = sorted(exported["records"], key=lambda x: x["name"])
        identifiers = [(x["name"], x["source_index"], x["frame_ids"], x["input_rgb_sha256"]) for x in records]
        if reference_records is None:
            reference_records = identifiers
        assert identifiers == reference_records and len(set(x[0] for x in identifiers)) == 64
        means = []
        for record in records:
            scene = ev/"all"/record["name"]
            render, depth = read(scene/"render_scores.json"), read(scene/"depth_scores.json")
            assert len(render) == len(depth) == 6
            assert sorted(x["item"] for x in render) == sorted(x["item"] for x in depth)
            means.append([np.mean([x[k] for x in depth]) for k in ("absrel", "rmse")]+
                         [np.mean([x[k] for x in render]) for k in ("psnr", "ssim", "lpips")])
        scenes[condition] = np.asarray(means)
        assert np.isfinite(scenes[condition]).all()
        assert np.allclose(scenes[condition].mean(axis=0), tables[condition][:5], rtol=0, atol=1e-12)
        native[condition] = read(ev/"all/results.json")
        assert native[condition] == metrics["official_native_result"]
    for arm in ARMS[1:]:
        for step in STEPS:
            assert tables[f"{arm}/{step}"][:5] == tables["joint_full/0"][:5]
    # Same step and same window on the two4090 independent arms.
    assert all(x["loss_recon"] == y["loss_recon"] for xr, yr in zip(rows["u_read"], rows["u_no_anchor_read"])
               for x, y in zip(xr, yr))
    comparisons = [
        ("joint_full/500", "old128_bidirectional/500"),
        ("joint_full/initial_feedback_full", "joint_full/0"),
        ("joint_full/500", "joint_full/initial_feedback_full"),
        ("joint_full/500", "joint_full/0"),
        ("u_read/500", "u_read/0"),
        ("u_read/500", "joint_full/500"),
        ("u_read/500", "u_no_anchor_read/500"),
        ("u_fusion/500", "u_read/500"),
        ("u_fusion/500", "u_fusion/0"),
    ]
    indices = np.random.default_rng(42).integers(0, 64, size=(10000, 64))
    bootstrap, deltas = {}, {}
    for condition, reference in comparisons:
        key = f"{condition} minus {reference}"
        delta = scenes[condition]-scenes[reference]
        bounds = np.quantile(delta[indices].mean(axis=1), [.025, .975], axis=0)
        bootstrap[key] = {k: {"mean_delta": float(delta[:, i].mean()),
                             "scene_bootstrap_95_percentile_interval": bounds[:, i].tolist(),
                             "positive_scene_count": int((delta[:, i] > 0).sum())}
                          for i, k in enumerate(FIELDS[:5])}
        deltas[key] = {k: None if x is None else y-x for k, x, y in zip(FIELDS, tables[reference], tables[condition])}
    write(args.output/"audit.json", {
        "status": "SAVED_ARTIFACTS_VERIFIED", "training": audit,
        "all_four_sampling_identical": True, "4000_unique_windows_verified": True,
        "all17_evaluation_sha_chains_verified": True,
        "old_reference_same_source_and_same_heldout_verified": True,
        "scene_scores_reproduce_official_aggregates": True,
        "independent_understanding_rgb_unchanged": True,
        "scope": "Receipt chains, persisted native score files and training logs; no new inference or optimizer calls.",
        "gradient_scope": "Local rank/microbatch norms before global reduction; U norms already include native loss warmup and accumulation, then multiplied by GC coefficient here. Anchor decoder prefix excludes shared blocks registered under enc_dec_backbone.decoder_blocks; cosines are not global gradient cosines.",
        "loss_scope": "First/last100 are different random windows, descriptive only. Logged raw native U loss excludes the external exposure warmup; gradient U norms include it.",
    })
    write(args.output/"deltas.json", deltas)
    write(args.output/"bootstrap.json", {"seed": 42, "samples": 10000, "paired_scenes": 64,
        "scope": "Reconstruction only; dataset uncertainty, not training seed uncertainty.", "comparisons": bootstrap})
    write(args.output/"scene_scores.json", {"scene_names": [x[0] for x in reference_records],
        "columns": list(FIELDS[:5]), "conditions": {k: v.tolist() for k, v in scenes.items()}})
    with (args.output/"siu3r_table1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["condition", *FIELDS])
        writer.writerows([[condition, *values] for condition, values in tables.items()])
    final = ["joint_full/0", "old128_bidirectional/500"]+[f"{arm}/500" for arm in ARMS]
    markdown = ["| Condition | "+" | ".join(HEADERS)+" |", "|---|"+"---:|"*13]
    for condition in final:
        markdown.append("| "+condition+" | "+" | ".join("" if v is None else f"{v:.{p}f}"
                        for v, p in zip(tables[condition], PRECISION))+" |")
    (args.output/"final_table.md").write_text("\n".join(markdown)+"\n")
    with (args.output/"per_class_deltas.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["comparison", "view", "metric", "class", "baseline", "variant", "delta"])
        for condition, reference in comparisons:
            for view in ("context", "target"):
                base, variant = native[reference], native[condition]
                for suffix in ("ious_per_class", "pqs_per_class"):
                    b, v = np.asarray(base[f"{view}_{suffix}"]).ravel(), np.asarray(variant[f"{view}_{suffix}"]).ravel()
                    assert len(b) == len(v) == 20
                    writer.writerows([[f"{condition} minus {reference}", view, suffix, name, x, y, y-x]
                                      for name, x, y in zip(CLASSES, b, v)])
                b, v = base[f"{view}_map"], variant[f"{view}_map"]
                assert b["classes"] == v["classes"]
                writer.writerows([[f"{condition} minus {reference}", view, "AP", CLASSES[c], x, y, y-x]
                                  for c, x, y in zip(b["classes"], b["map_per_class"], v["map_per_class"])])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for ax, col in zip(axes.flat, [i for i in range(13) if i not in (8, 12)]):
        for arm, label in zip(ARMS, LABELS):
            ax.plot(STEPS, [tables[f"{arm}/{step}"][col] for step in STEPS], "o-", label=label)
        ax.axhline(tables["joint_full/initial_feedback_full"][col], color="gray", ls=":", label="Untrained, full feedback")
        ax.set_title(FIELDS[col]); ax.set_xlabel("New optimizer calls"); ax.grid(alpha=.3)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=8)
    fig.suptitle("64 held-out scenes; full pool sampled without replacement; fixed reconstruction in three probes")
    fig.savefig(args.output/"heldout_curves.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, ax in zip(("loss_recon", "loss_understanding", "preclip_norm"), axes.flat):
        for arm, label in zip(ARMS, LABELS):
            values = np.asarray([[x[key] for x in rank] for rank in rows[arm]]).mean(axis=0)
            ax.plot(np.arange(25, 501), np.convolve(values, np.ones(25)/25, mode="valid"), label=label)
        ax.set_title(key+" (25-call moving mean)"); ax.grid(alpha=.3)
    ax = axes.flat[-1]
    for layer in ("L6", "L8", "L10", "L12"):
        values = np.asarray([[x["interactions"][layer]["delta_ratio_mean"] for x in rank] for rank in rows["joint_full"]]).mean(axis=0)
        ax.plot(np.arange(1, 501), values, label=layer)
    ax.set_title("Joint feedback delta/input norm (last microbatch)"); ax.legend(); ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=7)
    fig.savefig(args.output/"training_and_feedback.png", dpi=160); plt.close(fig)
    print(json.dumps({"output": str(args.output), "audit": "PASS", "deltas": deltas,
                      "joint_gradient_probes": audit["joint_full"]["gradients_late_probes"]}, indent=2))


if __name__ == "__main__":
    main()
