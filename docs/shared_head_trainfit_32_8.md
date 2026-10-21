# Step-4500 shared head: 32 train + 8 val GT-free fit check (read-only)

Objects: frozen `cross_scene/lgs_lr1e4/ckpt_step6000`; head
`instance_query/shared/instance_query_head.pt`, verified **step = 4500**, same
frozen reference, same thresholds (`objectness 0.5`, `mask 0.5`, `area 50 px`),
seed 42; model in `eval()` + fp32, all frozen params `requires_grad_(False)`.
No training, no optimizer step, no threshold/split/checkpoint change.

## 0. Window provenance (as requested)

* **A - provably sampled during training: not recoverable.** The shared run was
  cancelled before `history.json` was written, and that file is where the sampled
  `(context, novel)` windows were dumped; the log records 98 logged steps over 31
  distinct scene names but **no frame IDs**.  I therefore do not claim any window
  was "seen in training", and the A-vs-B question is undecidable this round.
* **B - training scenes, fixed recorded windows**: all 32 training scenes, one
  window each chosen deterministically with pair seed `2000 + index`
  (frame IDs written into `per_instance.csv`).
* **C - the 8 unseen validation windows** (pair seed `1042 + index`), as before.

The single-window `fit_shared` head and its results are **not** mixed into this
evaluation.

## 1. GT-free results (IoU >= 0.5 = TP, no GT used to pick a query)

| group | novel instance-records | detected (IoU>=0.5) | mean IoU over all 100 queries | mean IoU after all gates |
|---|---|---|---|---|
| B_train (32 scenes) | 203 | 12 (**5.9 %**) | 0.210 | 0.152 |
| C_val (8 scenes) | 55 | **0 (0.0 %)** | 0.194 | 0.117 |

AP50 keeps the existing definition (per-view, greedy score-ordered matching at
IoU >= 0.5, all-point interpolation) and is summarised as a mean over
views/scenes; it is a development-split class-agnostic diagnostic, **not** SIU3R
official AP/mAP.

## 2. Failure attribution (GT-aided diagnostics, separate from the numbers above)

For every visible GT instance: the best IoU over **all 100 queries** (no gates),
the best after the objectness gate, and the best after all gates.

| group | no query mask reaches IoU 0.5 (mask quality) | good mask exists but objectness excluded it | good mask excluded by the area gate | detected |
|---|---|---|---|---|
| B_train novel (203) | **185** | 6 | **0** | 12 |
| C_val novel (55) | **51** | 4 | 0 | 0 |
| all records, context + novel (511) | **471** | 18 | 0 | 22 |

So the dominant failure is **mask generation**: for 91 % of the training-scene
instance-records (and 93 % of validation ones) *no* query produces a mask that
overlaps the instance at IoU >= 0.5 at all - the objectness gate and the area
gate are not what removes them.  Where a good mask does exist, the objectness
ranking loses it in 6 / 4 cases; the **area gate never** excludes a good mask.

This partly revises the earlier step-4500 statement ("primarily objectness"):
with the full 32 + 8 attribution the largest category is mask quality, with
objectness second and the area gate irrelevant.

## 3. Answers

1. **Has the step-4500 shared head learned GT-free segmentation on the 32
   training scenes?**  No - only 5.9 % of novel training instance-records are
   detected, and 91 % have no query mask above IoU 0.5.
2. **A vs other training windows?**  Undecidable: the sampled windows were not
   recorded (run cancelled before `history.json`; the log has no frame IDs).
3. **Train vs validation gap?**  Small: detected 5.9 % vs 0.0 %, mean IoU over
   all queries 0.210 vs 0.194.  The head is nearly as poor on the scenes it
   trains on as on unseen ones, i.e. this is **underfitting**, not a
   generalisation gap.
4. **Mask, objectness or gate?**  Mask generation (471 of 511 records have no
   IoU >= 0.5 candidate), then objectness ranking (18), and the area gate never
   (0).
5. **Next step?**  Fix the shared head training before considering joint
   training with reconstruction supervision: the mask branch is the bottleneck,
   and the objectness branch matters only for the 18 records where a usable mask
   already exists.  (Suggestion only; nothing was modified or trained.)

## 4. Artifacts and command

```
python scripts/eval_shared_head_trainfit.py --split .../cross_scene/split.json \
  --checkpoint .../lgs_lr1e4/ckpt_step6000 \
  --head .../instance_query/shared/instance_query_head.pt \
  --out .../instance_query/trainfit
```

`workspace_recon_diag/instance_query/trainfit/per_instance.csv` (one row per
window x view x visible GT instance: group, scene, frame, kind, gt_area,
iou_all_queries, q_all, area/objectness of that query, iou_after_objectness,
iou_after_all_gates, attribution) and `summary.json`.

Not delivered: the representative RGB | GT | GT-free prediction | error figures
(the script prints the numbers only), and the A-window recovery.
