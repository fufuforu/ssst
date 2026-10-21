# IV: shared-path fixed-window fit test; V: corrected granularity table

## IV. Shared training path on one fixed window (read-only diagnostic)

Command (same head forward / Hungarian / loss / optimizer code path as the
shared run; only the scene sampling is pinned):

```
python scripts/train_instance_query_shared.py --split .../cross_scene/split.json \
  --checkpoint .../lgs_lr1e4/ckpt_step6000 --fixed-scene scene0012_02 \
  --steps 400 --eval-every 100 --out .../instance_query/fit_shared
```

The fixed window is `ctx=[2043, 2075]`, `novel=[2045, 2055]`, contribution maps
computed once in-process and reused, frozen LocusGS, fresh head.

| quantity | result |
|---|---|
| loss (step 1 -> 50 -> 400) | 3.235 -> 0.783 -> **0.728** (bce 0.485->0.057, dice 0.983->0.307, obj 0.597->0.000) |
| Hungarian matching over 400 steps | **stable for all 400 steps**: q12 <-> GT 1, q55 <-> GT 0 |
| positive objectness | 0.48/0.41 at step 1 -> **1.000/1.000** from step ~37 onward |
| negative objectness | mean 0.447 at step 1 -> 0.000; **0 negatives >= 0.5 from step ~45** |
| GT-free on the 8 unseen windows | TP 0-1, FP 19-26, AP50 0.000-0.006 |
| corrected per-instance context oracle (same windows) | **0.566-0.578** |

**So the shared path *can* fit a single fixed window**: the masks drive the loss
down, the matching is stable, and the objectness supervision works exactly as
designed (positives -> 1, negatives -> 0, no negatives above the threshold).
What fails is **transfer**: on the unseen windows the same queries still fire
(~20-27 predictions) with masks that match nothing, and the corrected
per-instance oracle (0.57) shows the representation supports instances that the
head does not recover.  This is "single-window learnable, multi-scene shared
failure", not a single-scene implementation problem - the same conclusion as the
objectness/mask localisation, now with the training path exonerated.

## V. Corrected granularity table (8 scenes)

Command:

```
python scripts/token_unit_granularity.py --split .../cross_scene/split.json \
  --checkpoint .../lgs_lr1e4/ckpt_step6000 --out .../token_unit/units8
```

Novel-frame IoU over the 8 unseen scenes (same valid pixels, thing filter,
visibility and thresholds for every K):

| variant | mean IoU | min | max |
|---|---|---|---|
| A_context K=1 | 0.518 | 0.355 | 0.757 |
| A_context K=8 | 0.526 | 0.359 | 0.771 |
| A_context K=32 | 0.539 | 0.368 | 0.777 |
| A_context K=64 | **0.544** | 0.370 | 0.782 |
| B_fourview K=1 / 64 | 0.523 / 0.550 | | |

Per scene (context-assigned K=1/8/32/64): 0059_00 0.488/0.494/0.500/0.501,
0072_02 0.483/0.484/0.504/0.514, 0132_01 0.472/0.481/0.486/0.491,
0472_01 0.680/0.697/0.715/0.719, 0559_01 0.461/0.471/0.494/0.499,
0568_02 0.757/0.771/0.777/0.782, 0615_00 0.446/0.453/0.464/0.476,
0695_00 0.355/0.359/0.368/0.370.  Recall@0.5 moves by <=0.006 anywhere.

**Cross-instance mixing** (corrected: the *second* largest instance share of a
token's contribution mass, contribution-weighted, over the four views; tokens
with >= 1.0 total unit of contribution):

| scene | tokens w/ enough contribution | 2nd instance >= 5% | >= 10% |
|---|---|---|---|
| scene0059_00 | 744 | 0.366 | 0.273 |
| scene0072_02 | 562 | 0.000 | 0.000 |
| scene0132_01 | 942 | 0.285 | 0.224 |
| scene0472_01 | 499 | 0.112 | 0.094 |
| scene0559_01 | 762 | 0.217 | 0.175 |
| scene0615_00 | 607 | 0.152 | 0.122 |
| scene0568_02, scene0695_00 | 621, 277 | n/a (single-instance scenes) | n/a |
| **pooled over the 6 multi-instance scenes** | **4116** | **0.207** | **0.162** |

So roughly **21 % of contributing tokens carry >= 5 % of their contribution mass
on a second instance** (16 % at >= 10 %) - a minority, concentrated in the
scenes with the most boundary structure (0059_00 37 %, 0132_01 29 %) and absent
in others (0072_02 0 %).  The K gain (+0.026 mean IoU) is therefore **not** mainly
explained by those mixed tokens; it is spread evenly (all 8 scenes improve,
recall flat), consistent with finer partitioning of already-covered regions.

Note the single-instance scenes degenerate the second-share index (with one GT
instance there is no second column), so they are excluded from the pooled
fraction; the script now returns 0 for them instead of the largest share.

## Combined answer

* The **training path is exonerated** (fixed window: stable matching, objectness
  1.0/0.0, loss 3.24 -> 0.73).
* The failure is in **transfer of the objectness ranking and mask scale** to new
  scenes, plus a smaller representation limitation (a fifth of tokens genuinely
  mix two instances; the rest are largely single-instance).
* Single next factor to try (not executed): make the objectness head
  scene-adaptive rather than scene-invariant - e.g. normalise/calibrate the
  objectness logits per scene (a per-scene softmax or a learned bias from the
  scene tokens) so the same query does not fire on every scene.  This touches
  only the objectness branch, leaves the mask path and the frozen LocusGS
  untouched.
