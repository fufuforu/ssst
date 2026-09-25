# Shared InstanceQueryHead: failure localisation (step 4500)

Read-only.  Frozen LocusGS `cross_scene/lgs_lr1e4/ckpt_step6000`; head from
`instance_query/shared/instance_query_head_step4500.pt`.  Same valid-pixel,
thing-instance, visibility and threshold conventions as the shared run.
Script: `scripts/localize_shared_head_failure.py`.

## 1. Job and checkpoint

Job **55102 was stopped** (authorised: step-4500 validation regressed, additions
were FP, training windows also unlearned).  It was at **step 4850** when
cancelled; the latest saved checkpoint was **step 4500** and is verified complete:
`head` (20 tensors, 9.57 M params) + `optimizer` (1 group) + `step` + torch/numpy
RNG state + `frozen_checkpoint` reference + thresholds.  A copy is kept as
`shared/instance_query_head_step4500.pt`; `shared.out` is preserved.

## 2. The decisive finding: the gates are not empty, the ranking is wrong

Counting over **all 100 queries** in the valid region (no objectness pre-filter),
per view:

| window | obj >= 0.5 | +mask > 0.5 | +area >= 50 px | max mask (all queries) | max hard area (all queries) |
|---|---|---|---|---|---|
| scene0472_01 v2 | 5 | 3 | 3 | 1.00 | 24 388 |
| scene0559_01 v2 | 4 | 2 | 2 | 1.00 | 25 806 |
| scene0568_02 v2 | 4 | 2 | 2 | 1.00 | 25 755 |
| scene0615_00 v2 | 5 | 3 | 3 | 1.00 | 16 203 |
| scene0695_00 v2 | 4 | 1 | 1 | 1.00 | 23 475 |
| train scene0000_00 v2 | 5 | 2 | 2 | 1.00 | 34 064 |
| train scene0001_00 v2 | 4 | 2 | 1 | 1.00 | 6 007 |

So **the earlier statement "mask max ~1 with hard area 0" was wrong** - it was
measured *after* the objectness gate had emptied the candidate set.  Over all
queries the masks have hard areas of 6k-45k px.  The earlier "pred 0 / TP 0"
numbers at steps <= 3000 were real for those steps, but the "no mask area"
explanation must be withdrawn: the masks are large, they are simply **not
matched to any single GT instance** (areas are 1-3x the instance area, i.e. one
mask spans several objects), and the queries that do match well are not the ones
with high objectness.

## 3. Objectness is mis-ranked (the primary breakage for GT-free inference)

Per window, the Hungarian-matched positive query's objectness vs the rest:

| window | positive objectness (mean / max) | negative mean / max / p99 | #negatives >= 0.5 | positive share of the objectness loss |
|---|---|---|---|---|
| scene0472_01 | 0.277 / 0.975 | 0.055 / 0.986 / 0.979 | 4 | 0.42 |
| scene0559_01 | 0.463 / 0.982 | 0.042 / 0.982 / 0.977 | 2 | 0.50 |
| scene0568_02 | **0.003** / 0.003 | 0.060 / 0.964 / 0.964 | 4 | 0.28 |
| scene0615_00 | 0.287 / 0.992 | 0.057 / 0.981 / 0.976 | 4 | 0.45 |
| scene0695_00 | **0.013** / 0.013 | 0.062 / 0.975 / 0.975 | 4 | 0.21 |
| train scene0000_00 | 0.603 / 0.988 | 0.042 / 0.982 / 0.982 | 3 | - |
| train scene0000_01 | 0.610 / 0.972 | 0.040 / 0.957 / 0.957 | 2 | - |
| train scene0000_02 | 0.738 / 0.979 | 0.051 / 0.962 / 0.962 | 3 | - |

The best-matching query often has **near-zero objectness** (scene0568_02: the
query with hard IoU 0.72 on the only instance scores 0.003) while 2-4
non-matching queries score >= 0.5.  So a GT-free rule that selects by objectness
cannot reach the good masks at all - the objectness ordering, not the mask
quality, is what breaks the pipeline.  Also notable: the same few queries are
matched repeatedly across scenes (query 25 in 7/12 windows, 28 in 6, 0/63/83 in
5), which is the signature of a matching attractor rather than per-scene
instances.

## 4. Mask quality (diagnostic only, GT-matched, no objectness gate)

Best-query hard IoU per novel view: scene0568_02 0.72, scene0559_01 0.61,
scene0472_01 0.37, scene0695_00 0.24, scene0615_00 0.22, train scene0000_00
n_gt 4 - best per-instance IoU in the range 0.1-0.4 with areas 1-3x GT.  So the
masks are *partly* aligned (0.6-0.7 on the easy single-instance scenes) but
systematically over-large, and the per-instance oracle from the token grouping
reaches 0.46-0.68 (previous round) - i.e. the representation supports instances,
the head does not exploit it.

## 5. Audit of the shared script (read-only; confirmed items only)

* Token/Gaussian contribution maps, valid-pixel definition, thing-instance
  filter, Hungarian cost, BCE/Dice and the GT-free rule are the same code as the
  verified single-scene script (`token_maps`, `dice_loss` are imported).
* Frozen LocusGS / head-only updates: `requires_grad_(False)` on the frozen model
  and `AdamW(head.parameters())`, verified by 0/450 frozen params with gradients
  in the single-scene runs.
* Query input: the head consumes the frozen decoder's final-layer tokens, which
  depend only on the two context frames; novel GT is used only in the loss and
  offline metrics.
* **Confirmed evaluation bug (fixed)**: the oracle in `train_instance_query_shared.py`
  built **one union mask** from all assigned tokens and scored every instance
  against it, which is why the log shows a constant `oracle IoU 0.185/0.191`
  (~|instance|/|union|).  The evaluation now builds a **separate mask per
  instance**; this changes the reported oracle baseline only, not training.
* **Confirmed diagnostic bug (fixed)**: `frac_top2_ge_5pct`/`10pct` in
  `token_unit_granularity.py` read the *largest* share instead of the second
  largest; renamed to `frac_second_ge_*` and corrected.  The K-granularity IoU
  numbers were unaffected; the mixing numbers were meaningless and are withdrawn.

## 6. Not done in this round

* the corrected 8-scene K table and the corrected second-share mixing fractions
  (the fix is committed but the 8-scene re-run was not executed);
* the minimal shared-path single-window fit test (300-500 steps).
  Both are one short read-only/diagnostic run away with the committed scripts.

## 7. Answers

* **A.** Yes, two confirmed evaluation/diagnostic bugs, neither affecting
  training: the union-mask oracle (inflated the shared run's oracle reference by
  ~2.5x - the true per-instance context oracle is 0.46-0.68) and the
  second-share mixing index (invalidated the earlier mixing fractions).
* **B.** Primarily **objectness**; masks are secondly at fault.  The high-IoU
  masks exist (up to 0.72) but carry near-zero objectness, and the high-objectness
  masks are 1-3x too large and match nothing.  Same pattern on training and unseen
  windows (positive objectness 0.36-0.74 vs negatives up to 0.98).
* **C.** Not tested this round (shared-path single-window fit) - open.
* **D.** The corrected mixing fractions are not available (the fix is committed;
  the re-run was not executed).  The uncorrected numbers are withdrawn.
* **E.** Evidence points to the **objectness/assignment supervision in the head**
  first (attractor queries, anti-correlated objectness), then to making the
  64 Gaussians of a token carry distinct support; letting instance supervision
  into the LocusGS tokens/anchors is premature until the head can rank its own
  masks.
