# Does the fit_shared head work under the GT-free rule on its own window?

Read-only.  Objects: frozen `cross_scene/lgs_lr1e4/ckpt_step6000`, head
`instance_query/fit_shared/instance_query_head.pt` (**step 400, verified**), the
exact training window `scene0012_02` frames **ctx [2043, 2075], novel [2045,
2055]**, pinned by overriding the provider's index selection (asserted, nothing
re-sampled).  Only the two context frames feed the model; GT is used for scoring
only.  Same contribution maps, GT-free rule and per-view visibility as training.

## Per-view results (objectness >= 0.5 -> mask > 0.5 -> area >= 50 px)

| view | frame | GT visible | after obj | after mask | final preds | TP | FP | FN | AP50 | mean IoU of TP |
|---|---|---|---|---|---|---|---|---|---|---|
| context v0 | 2043 | 2 | 2 | 1 | 1 | 1 | **0** | 1 | 0.500 | 0.843 |
| context v1 | 2075 | 2 | 2 | 2 | 2 | 2 | **0** | 0 | **1.000** | 0.819 |
| novel v2 | 2045 | 2 | 2 | 2 | 1 | 1 | **0** | 1 | 0.500 | 0.837 |
| novel v3 | 2055 | 2 | 2 | 2 | 2 | 2 | **0** | 0 | **1.000** | 0.675 |

Zero false positives in all four views.  Objectness >= 0.5 passes exactly 2
queries per view - the same two the Hungarian matched during training (q55, q12).

## Per visible GT instance

| view | instance | GT area | GT-free best IoU | query | objectness | pred area |
|---|---|---|---|---|---|---|
| v0 | 8027 | 6 943 | **0.843** | q55 | 1.000 | 5 918 |
| v0 | 8030 | 51 | 0.000 | - | - | 0 (no overlapping mask) |
| v1 | 8027 | 9 991 | **0.796** | q55 | 1.000 | 8 434 |
| v1 | 8030 | 10 703 | **0.842** | q12 | 1.000 | 9 649 |
| v2 | 8027 | 7 573 | **0.837** | q55 | 1.000 | 6 422 |
| v2 | 8030 | 167 | 0.000 | - | - | 0 |
| v3 | 8027 | 9 021 | **0.817** | q55 | 1.000 | 7 477 |
| v3 | 8030 | 2 100 | **0.532** | q12 | 1.000 | 1 535 |

## Diagnostic: best mask over ALL 100 queries (GT-matched, no gate)

| view | instance | best any-query IoU | query | objectness | area | passes obj / mask / area |
|---|---|---|---|---|---|---|
| v0 | 8027 | 0.843 | q55 | 1.000 | 5 918 | True / True / True |
| v0 | 8030 | **0.000** | - | - | 0 | - |
| v1 | 8027 / 8030 | 0.796 / 0.842 | q55 / q12 | 1.000 | 8 434 / 9 649 | all True |
| v2 | 8027 | 0.837 | q55 | 1.000 | 6 422 | True / True / True |
| v2 | 8030 | 0.141 | q12 | 1.000 | **44** | True / True / **False** |
| v3 | 8027 / 8030 | 0.817 / 0.532 | q55 / q12 | 1.000 | 7 477 / 1 535 | all True |

The best-over-all-queries IoU equals the GT-free best in every row: **no better
mask is being discarded by objectness or by the mask gate**.  The only failures
are on the small instance 8030:

* v2 (167 px): the best mask reaches IoU 0.141 with a predicted area of **44 px**,
  i.e. it is rejected by the **area >= 50 px gate** (the first, and only, failing
  gate for this case);
* v0 (51 px): no query produces any mask overlapping the instance at all (best
  any-query IoU 0.000).

## Consistency check with training

head step = 400 (asserted); frozen checkpoint matches the training reference;
frames asserted equal to [2043, 2075, 2045, 2055]; thing instances kept with
>= 200 valid pixels over the four frames are **[8027, 8030]** - identical to the
training run.  These numbers are **not** comparable to the earlier single-scene
overfit figures, which used different frames.

## Decision

**Yes** - on the fixed window it was trained on, this head detects and separates
the visible instances under the GT-free rule: 2/2 instances in the two views
where both are large enough (AP50 1.000, zero FP) and the large instance in all
four views with IoU 0.80-0.84.  **The fixed-window functional check passes**, so
the next thing to study is multi-scene sharing, not more single-window work.

The single failure mode is the small instance when it is tiny in a view: there
the first failing point is the **area gate** (v2: best mask 44 px < 50 px), or no
overlapping mask exists (v0).  Nothing here indicates an objectness-ranking
problem on this window, and no mask-quality problem on the large instance.

## Artifacts

`scripts/eval_fit_shared_window.py`,
`workspace_recon_diag/instance_query/fitwindow_eval.json`,
`workspace_recon_diag/instance_query/fitwindow_eval/scene0012_02_fitwindow_novel_v2.png`
(RGB | GT instances | GT-free predictions | error; red = missed, green = correct,
blue = over unannotated).
