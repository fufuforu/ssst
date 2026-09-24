# Instance query phase 1: corrected protocol and `scene0012_02` result

Frozen fp32 LocusGS step6000; only the head trains.  Same scene, four frames,
thing-instance filter, valid pixels and threshold (0.5) for every method.

## 1. Protocol fixes made this round

* **per-view IoU accumulator fixed** - the reporting loop used to share (and
  accumulate into) the logging list, printing nonsense like "v0 2062.3"; the
  figure/report path now uses a fresh accumulator.
* **one valid-pixel definition everywhere**:
  `valid = (semantic != 0) & (semantic != ignore_index)` and it is applied to
  **BCE, Dice, the Hungarian cost and the final IoU** (previously only BCE/Dice
  used it; the IoU counted unannotated pixels).
* **empty targets can no longer win the matching**: a pair whose GT has fewer
  than 50 valid pixels costs `1e3` instead of the degenerate `1 - (0+1)/(0+1) = 0`,
  and targets need `>= min_instance_pixels` valid pixels overall plus support in
  at least one context frame.
* **coverage reported separately from IoU**: the previous "coverage" row was an
  IoU dominated by the predicted area (56k px, i.e. whole-image alpha), which is
  why it read 0.15.  Coverage is now `|GT ∩ pred| / |GT|` (recall).

## 2. Per-instance result (novel views, mean over the two novel frames)

| instance | GT px | query IoU / coverage | context-oracle IoU / coverage | all-token coverage | tokens assigned | objectness |
|---|---|---|---|---|---|---|
| 8027 | 7573 | **0.874 / 0.892** | 0.486 / 1.000 | 1.000 | 214 | 20.29 |
| 8030 | 167 | **0.335 / 0.379** | 0.058 / 1.000 | 1.000 | 106 | 23.37 |
| 8028 | **0** in novel views | n/a | n/a | n/a | 4 | 11.18 |

## 3. The IoU = 0 instance is an evaluation artefact, not a model failure

Instance **8028 is not visible in the two novel frames** (0 valid GT pixels
there); it only has valid support in the context frames, so my "at least one
context frame" rule kept it as a target and the novel-view IoU is undefined
rather than zero.  The correct handling is to evaluate an instance only in views
where it has valid pixels and to list the others as "absent in this view".  Its
4 assigned tokens are irrelevant for those views.

The two instances that *are* visible in the novel views are both fully covered by
token support (all-token coverage **1.000**), so the remaining weakness is not
"no token renders there".

## 4. Query vs the same-protocol oracle

On this scene the learned query **beats** the context->novel oracle on IoU for
both visible instances (0.874 vs 0.486 on the large one, 0.335 vs 0.058 on the
167-px one), while the oracle has higher *coverage* (1.000) because the oracle
reassigns whole token groups (over-segmented, hence the low IoU).  So on this
single sample the head is not limited by the token representation - it already
outperforms the token-grouping baseline on mask IoU.  (This is a one-scene
functional check; the earlier 8-scene average oracle 0.567 is a different
protocol and is not used to interpret it.)

## 4b. Per-view visibility rule (implemented)

An instance with **0 valid GT pixels in a view** now yields IoU `N/A` and is
excluded from the mean; a prediction inside that view's valid region is counted
separately as a false positive.  Recomputed means over **visible instances only**
(`scene0012_02`):

| method | context IoU / recall (n) | novel IoU / recall (n) |
|---|---|---|
| query | 0.518 / 0.532 (5) | **0.605 / 0.635 (4)** |
| context oracle | 0.235 / 0.976 (5) | 0.272 / 1.000 (4) |
| all-token coverage | 0.101 / 0.976 (5) | 0.085 / 1.000 (4) |

N/A-instance false positives: **none** - where instance 8028 is invisible the
query predicts nothing, so it contributes neither IoU nor FP.  The per-instance x
view table (e.g. 8027 v0 0.889/0.905, v1 0.859/0.883, v2 0.877/0.898,
v3 0.871/0.886; 8030 v2 only 0.141) is consistent with the aggregate.

## 4c. Objectness audit

| group | logits | prob | BCE | targeted |
|---|---|---|---|---|
| matched (q 12, 55, 88) | 52.15 / 33.67 / 43.28 | 1.000 | 0.0 | 1 |
| unmatched (97 queries) | mean **-41.23**, 0 with logit > 0 | 0.000 | 0.0 | 0 |

So **there are no high-objectness unmatched queries in this run** - the earlier
phrase "obj loss ~0 while unmatched queries keep high objectness" was not
measured and is wrong.  The loss reads 0 because the BCE is saturated for *both*
groups (matched logits +33..+52, unmatched -41); it is not a weighting bug.

Minimal backprop check (hand-set logits `[20, 20, 20]`, target `[1, 0, 0]`):
loss 13.33, gradient w.r.t. logits `[0.0, 0.333, 0.333]` = `sigmoid(lg) - t`,
gradient w.r.t. the objectness bias 0.667 - i.e. **an unmatched high-logit query
does receive a non-zero no-object penalty** (the penalty is diluted by the mean
over 100 slots, so the per-query movement is 1/100 of the batch gradient).

## 4d. GT-free inference (no GT used to select queries)

Instances are chosen purely by objectness (>= 0.5 or >= 0.9), predicted masks
thresholded at 0.5, matched to GT by IoU >= 0.5:

| threshold | selected | TP | FP | FN | mean IoU of TP (novel) |
|---|---|---|---|---|---|
| 0.5 | 3 | 3 | **0** | 1 | 0.767 |
| 0.9 | 3 | 3 | 0 | 1 | 0.767 |

No false positives, one miss (the 167-px instance whose prediction falls below the
50-px prediction floor on one novel view).  These are the quantities a future
official mAP/PQ needs.

## 5. Training curve

600 steps from the existing head checkpoint, lr 3e-4 unchanged: loss
0.6540-0.6542 with `matched 3/3` from ~step 300 onward, IoU ctx 0.432 /
novel 0.403, all losses flat to 4 decimals.  The curve has clearly plateaued;
the remaining gap is on the 167-px instance and on the (undefined) invisible one.

## 5b. Item 4 (second training scene) - not yet run

The additional fixed-sample overfit on a *second* training scene with several
visible thing instances including a small one is not done; it is the next step
before any 32/8 shared-head run.  The single-factor candidates below are
therefore not yet justified by a second sample.

## 6. One changed factor (next, not run)

Since the oracle is *below* the query and the implementation is now correct, the
next step is not a matching/threshold fix.  The single factor to try is the
**assignment entropy / no-object weight** (the objection branch saturates: obj
loss is 0.0000 while unmatched queries keep objectness high), i.e. anneal the
token->slot softmax or rebalance the no-object term - one change only, then
re-check the same three instances before considering the 32/8 shared run.

## 7. Artefacts

`workspace_recon_diag/instance_query/phase1c/`: `instance_query_head.pt` (head
only), `rows.json`, `table.npy` (per-instance x view: GT area, pred area, IoU,
coverage, matched query, objectness), `scene0012_02_{best,worst}_instance*_novel.png`
(GT | all-token coverage | oracle | query prediction | error map).
No frozen-backbone copy is stored.
