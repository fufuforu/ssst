# Read-only audit: instance-mask training sampling (G0/G0+)

**Read-only.** No training, no checkpoint writes, no loss/threshold changes, no
cleanup of the parallel full-data reconstruction work. Everything below reuses the
training code path (`build_context_segments`, the `class >= 2` thing filter and
`ssst_loss._sample_points`).

Artifacts: `scripts/audit_instance_sampling.py`,
`scripts/instance_sampler_v2.py`,
`group_plus/instance_sampling_audit.json`,
`group_plus/instance_sampler_v2_check.json`.

## 1. Which points are actually used

`verify_sampler_semantics()` (asserted, not estimated):

* `_sample_points(values, 4096)` flattens the `[V,H,W]` tail of each row and takes
  `linspace(0, W_tot-1, 4096).round()` — an **even ~32-px stride over the flattened
  two-view buffer** (W_tot = 2·256·256 = 131 072), deterministically, no RNG;
* the Hungarian cost (`_pairwise_bce`, `_pairwise_dice`), the Dice term and the
  **post-match** BCE/Dice in `group_instance_loss` all call
  `_sample_points(mask_logits[index, rows])` / `_sample_points(masks[cols])` on the
  same tail, so **every candidate query sees exactly the same 4096 points for a
  given GT instance** (verified by reproducing `_pairwise_bce` with a manual
  gather: equal to 1e-6);
* the per-instance positive count below is therefore what the matching cost *and*
  the post-match loss really see — not an area estimate.

## 2. Statistics over 300 uniformly selected trained windows

Windows: 300 of the 6000 plan entries (every 20th), all with the two context
frames the trainer actually used. 1073 thing instances reached the matcher
(12 windows contained no thing instance at all). All 32 training scenes are
represented, 13–102 matched instances per scene.

Positive sampled points per instance (of 4096):

| instance group (per-view area) | instances | 0 | 1–4 | 5–19 | ≥20 |
|---|---|---|---|---|---|
| small, < 1310 px | 333 (31.0 %) | **90 (27.0 %)** | 15 (4.5 %) | 57 (17.1 %) | 171 (51.4 %) |
| medium, < 6553 px | 389 (36.3 %) | 0 | 3 (0.8 %) | 1 (0.3 %) | 385 (99.0 %) |
| large, ≥ 6553 px | 351 (32.7 %) | 0 | 0 | 0 | 351 (100 %) |
| **all** | **1073** | **90 (8.4 %)** | 18 (1.7 %) | 58 (5.4 %) | 907 (84.5 %) |

With the raw 2-view areas: < 200 px → 70 % zero; 200–1000 px → 25.7 % zero;
1000–5000 px → 2.2 % zero; ≥ 5000 px → 0 %.

Consequence of a zero-positive instance: the BCE target is all-negative and the
Dice term `1 − (2·p·y + 1)/(Σp + Σy + 1)` is minimised by making the mask empty,
so such an instance is **actively trained to disappear** — and the small bucket of
the fixed GT-free evaluation has 0 TP in both G0 (0/11) and G0+ (0/11).

## 3. Descriptive contrast with the saved G0+ step6000 (training windows)

Per-instance, on the 7 training windows that were evaluated (novel view 2;
`best-over-groups` = best IoU over all 100 group masks at `mask > 0.5`,
`area ≥ 50`, no score gate):

| scene | GT area | positive sampled pts | best-over-groups IoU | GT-free detected |
|---|---|---|---|---|
| scene0000_02 | 89 | **0** | not visible | no |
| scene0000_02 | 4 777 | 155 | not visible | no |
| scene0000_02 | 19 926 | 629 | not visible | no |
| scene0000_02 | 38 282 | 1 172 | not visible | no |
| scene0016_01 | 22 993 | 711 | not visible | no |
| scene0016_01 | 1 295 | 35 | not visible | no |
| scene0007_00 | 53 050 | 1 679 | not visible | no |
| scene0007_00 | 735 | 21 | not visible | no |
| scene0012_01 | 29 813 | 958 | 0.633 | **yes** |
| scene0012_01 | 9 258 | 190 | 0.323 | no |
| scene0012_01 | 2 825 | 77 | 0.004 | no |
| scene0009_00 | 8 703 | 293 | 0.356 | no |
| scene0010_01 | 96 | 8 | not visible | no |
| scene0010_01 | 30 210 | 979 | not visible | no |
| scene0001_01 | 3 216 | 135 | 0.214 | no |

(“not visible” = the instance has no valid GT pixel in that novel view, so the
read-out cannot be scored there; it is not a failure of the model. The contrast is
descriptive and was **not** used to choose a checkpoint or a threshold.)

Reading: the starved tiny instances (0 and 8 positive points) are never detected,
but **large instances with 155–1679 positive points also fail** (IoU 0.000–0.36,
one detected) — so the sampling is not the first-order cause of the aggregate
failure, while it is a plausible cause of the small-instance zero recall.

## 4. Decision per the stated rule

* *“If small/medium instances largely fall at 0 or very few positive points”* —
  true for **small** instances (27 % zero, 31.5 % ≤ 4) and false for medium/large
  (0 % zero, ≈99 % ≥ 20).
* *“If the current sampling already gives the vast majority of instances enough
  positives, do not start that training”* — true globally (84.5 % ≥ 20; 100 % of
  large) and the contrast above shows well-sampled instances still fail.

**Therefore this round launches no training** (also per the explicit instruction
to keep the disk for the incoming full-data SIU3R evaluation and to start no new
long run). The audit does not support the sampling change as the *first-order*
fix; its measured target is narrow and well defined: 90/1073 = 8.4 % of matched
instances (27 % of the small bucket) are trained as empty, which is exactly the
bucket with 0 TP.

## 5. Concrete sampling rule for the (not yet run) single-variable round

For every GT instance i, with the total budget kept at 4096 points per
(query, instance) pair:

1. `P_i` = its valid pixels over the two context views, raster order;
   `k_i = min(|P_i|, 2048)`;
2. positives: all of `P_i` if `|P_i| ≤ k_i`, otherwise `k_i` points evenly spaced
   over the sorted `P_i` (deterministic, no RNG);
3. negatives: the remaining `4096 − k_i` points, evenly spaced over
   `[0, V·H·W) \ P_i`;
4. the resulting index vector is a pure function of the GT mask, so the Hungarian
   cost, the mask BCE and the Dice term stay shape-compatible and every candidate
   query is scored on the identical points for that instance.

Data-level smoke of the rule over the same 300 windows
(`instance_sampler_v2_check.json`, CPU only, no model): 0 instances with zero
positive points (was 90), 6 instances with 1–4 and 5 with 5–19, 1062 with ≥20 —
i.e. every instance now keeps all the positive support it actually has; every
index vector has length exactly 4096 and is bit-reproducible across calls.
Its expected effect is confined to the small bucket (recall there is currently
0/11); it cannot repair the medium/large failures, which already have ≥20
positives and still miss.
