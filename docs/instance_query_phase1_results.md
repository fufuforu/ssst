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

## 5. Training curve

600 steps from the existing head checkpoint, lr 3e-4 unchanged: loss
0.6540-0.6542 with `matched 3/3` from ~step 300 onward, IoU ctx 0.432 /
novel 0.403, all losses flat to 4 decimals.  The curve has clearly plateaued;
the remaining gap is on the 167-px instance and on the (undefined) invisible one.

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
