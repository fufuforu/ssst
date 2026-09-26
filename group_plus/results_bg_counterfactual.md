# Read-only counterfactual: folding G0+'s background mass back into the 100 groups

**Read-only.** No training, no optimizer step, no checkpoint/loss/threshold change,
no new long run; only a diagnostic script, a CSV/JSON and three small PNGs.  This
is a **post-hoc probe of the current checkpoint's mass flow**, not a deployable
inference method and not evidence about the dynamics of retraining with a
different background parameterisation.

## Reused code and checkpoints

* models/rendering: `LocusGSGroupRecon.render_group_masks`,
  `GaussianRenderer.render_feature_channels` (the *same* compositor as RGB)
* matching/diagnostics: `audit_mask_failure.{hard_metrics, soft_dice,
  best_over_groups_iou}`, `ssst_loss.build_context_segments`
* read-out: `group_eval_v2.{forward_group, group_score_table,
  group_predictions_v2, build_val_entries}`, `object_locusgs_eval.{gt_instances,
  instance_metrics}`
* checkpoints: main `workspace_group_plus/arm_g0plus/ckpt_step6000`
  (`a63dd485cc2d1b3f…`), reference
  `workspace_group_locusgs/arm_g0/ckpt_step6000` (`dbe5ea8311b466e6…`); both
  SHA256 **and mtime unchanged** after the run (checked in-script and re-checked
  afterwards).

## Counterfactual rule and its checks

`A_cf[t, q] = A[t, q] / sum_{j<100} A[t, j]` for the 100 groups and
`A_cf[t, bg] = 0`, computed directly as `softmax(slot_logits[t, :100])` (stable
when the background probability is near 1) and only then padded with a zero
background column.

* equivalence of the two definitions on real tokens:
  `max |softmax(100 logits) − conditional normalisation| = 3.0e-7` (tolerance
  1e-4), **0 anomalous tokens out of 9216** (anomaly = group mass ≤ 1e-3);
* `sum(100 counterfactual group masks) = alpha` to ≤ 1.5e-6 while
  `sum(100 original groups) + background = alpha` also holds to ≤ 1.5e-6;
* the counterfactual render returns exactly the same alpha (max Δ = 0.0) and a
  repeat forward reproduces RGB/depth bit-identically (Δ = 0.0), confirming the
  Gaussians, cameras, query logits and RGB/depth path are untouched;
* GT-free read-out, GT, valid pixels, frames, class scores and all thresholds are
  identical between the two variants; GT is used only for the separately labelled
  best-over-groups diagnostic.

## The 9 fixed training instances (novel views, paired)

Background mass inside the GT thing region goes `0.11–0.38 → 0.000` by
construction, but the mask does not become instance-shaped:

| instance (bucket) | view | GT px | bg mass orig→cf | best IoU orig→cf | pred/GT orig→cf | detected orig→cf |
|---|---|---|---|---|---|---|
| scene0009_00 (mid) | 2 | 323 | 0.303 → 0 | 0.024 → 0.019 | 13.7 → 19.6 | no → no |
| scene0009_00 (mid) | 3 | 1504 | 0.228 → 0 | 0.035 → 0.101 | 3.7 → 5.2 | no → no |
| scene0012_01 (mid) | 2 | 7755 | 0.170 → 0 | **0.633 → 0.736** | 0.71 → 0.95 | yes → yes |
| scene0012_01 (mid) | 3 | 1497 | 0.112 → 0 | **0.523 → 0.450** | 1.86 → 2.22 | **yes → no** |
| scene0012_01 (mid) | 2 | 4193 | 0.126 → 0 | 0.020 → 0.033 | 1.83 → 2.40 | no → no |
| scene0012_01 (mid) | 3 | 4838 | 0.216 → 0 | 0.009 → 0.027 | 1.57 → 3.23 | no → no |
| scene0001_01 (mid) | 2 | 5775 | 0.166 → 0 | 0.152 → 0.171 | 3.07 → 0.67 | no → no |
| scene0001_01 (mid) | 3 | 6389 | 0.165 → 0 | 0.153 → 0.157 | 2.82 → 0.88 | no → no |
| scene0009_00 (large) | 2 | 12236 | 0.284 → 0 | 0.356 → 0.293 | 0.74 → 2.40 | no → no |
| scene0009_00 (large) | 3 | 12316 | 0.377 → 0 | 0.174 → 0.242 | 0.45 → 2.04 | no → no |
| scene0012_01 (large) | 2 | 20535 | 0.137 → 0 | 0.323 → 0.367 | 0.37 → 0.49 | no → no |
| scene0012_01 (large) | 3 | 27870 | 0.138 → 0 | 0.315 → 0.338 | 0.31 → 0.34 | no → no |
| scene0001_01 (large) | 2 | 12691 | 0.189 → 0 | 0.137 → 0.262 | 1.40 → 0.48 | no → no |
| scene0001_01 (large) | 3 | 11064 | 0.197 → 0 | 0.144 → 0.247 | 1.63 → 0.57 | no → no |
| scene0007_00 (large) | 3 | 9352 | 0.013 → 0 | 0.051 → 0.051 | 0.60 → 1.03 | no → no |
| scene0007_00 (large) | 3 | 10749 | 0.008 → 0 | 0.010 → 0.009 | 0.52 → 0.90 | no → no |

(novel views with a non-empty GT for that instance; two further large instances
are not visible in the novel views of their window.)

Aggregate over these 9 instances (novel views): best IoU crossed 0.5 **up: 0**,
**down: 1**; GT-free TP gained 0, lost 1; FP 84 → 99; mean predicted/GT area
2.20 → 2.71 (the folded mass mostly **enlarges** the masks that already cover the
region).  Background mass on GT stuff over the same windows: 0.437 → 0.000.

## The 8 fixed unseen validation windows (novel views, unified GT-free rule)

| variant | TP | FP | FN | mean AP50 | small / medium / large recall@0.5 |
|---|---|---|---|---|---|
| G0 step6000 | 12 | 251 | 98 | 0.0398 | 0/11, 2/24, 4/20 |
| **G0+ step6000 (original)** | 6 | 140 | 104 | 0.0984 | 0/11, 0/24, 3/20 |
| **G0+ counterfactual** | **12** | 164 | **98** | **0.1365** | 0/11, 0/24, **7/20** |

Per scene (TP / FP / FN / AP50, novel views): scene0059_00 0/23/37/0.000 →
0/25/37/0.000; scene0072_02 0/11/8/0.000 → **3/13/5/0.188**; scene0132_01
0/16/19/0.000 → 0/19/19/0.000; scene0472_01 3/14/4/0.625 → **4/19/3/0.708**;
scene0559_01 3/16/16/0.163 → 2/17/17/0.113; scene0568_02 0/24/4/0.000 →
0/28/4/0.000; scene0615_00 0/18/12/0.000 → **3/18/9/0.083**; scene0695_00
0/18/4/0.000 → 0/25/4/0.000.

## Verdict (strictly case-based)

* The counterfactual **does recover detections**: on the unseen windows TP 6 → 12
  (exactly back to the G0 level), FN 104 → 98, mean AP50 0.098 → 0.137 (+39 %
  relative) and large-instance recall 3/20 → 7/20.  So background diversion is a
  real, directly measurable *suppressant* of borderline masks in this checkpoint.
* But it is **not a mask fix**: FP rise 140 → 164, mean predicted/GT area rises
  2.20 → 2.71 on the fixed failures, **no** fixed mid/large failure instance
  crosses IoU 0.5 upward (one crosses downward), and one previously detected
  instance is lost.  The folded mass mostly inflates whatever group already
  covers the region instead of separating the instance.
* This is therefore the "recall rises but FP / cross-object mixing worsens, best
  mask nearly unchanged for the hard cases" branch: **the per-token background
  logit cannot be listed as a proven priority remedy** for the mid/large mask
  failures.  The remaining evidence still points at the **token→group assignment /
  query mask formation** as the binding constraint (masks stay at IoU ≤ 0.44 for
  the hard instances even with the full mass budget available).
* Caveat, as required: this post-hoc counterfactual only tests how the *existing*
  checkpoint redistributes mass; it cannot rule out that retraining with a
  different background parameterisation would change the representation itself.

## Artifacts

| file | content |
|---|---|
| `group_plus/bg_counterfactual/summary.json` | full per-instance and per-scene records, checks, checkpoint hashes |
| `group_plus/bg_counterfactual/per_instance.csv` | one row per (instance, view) with original vs counterfactual metrics |
| `group_plus/bg_counterfactual/diversion_failure_scene0009_00_key20017.png` | high diversion (0.28-0.38) + failing original mask |
| `group_plus/bg_counterfactual/existing_true_positive_scene0012_01_key8017.png` | instance that was already a TP (0.633 → 0.736) |
| `group_plus/bg_counterfactual/stuff_region_scene0059_00.png` | GT semantic, original background mass, counterfactual background mass (0) |

Script: `scripts/audit_bg_counterfactual.py`; job: read-only GPU run (no queue
submission), total artifact size 0.5 MB.
