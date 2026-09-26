# Round report — implementation audit, condition A (pure4), B1 official alignment, B2 stop-gradient

**Scope.** 32/8 development split, plus the official 1860-pair val manifest for
B1.  All development numbers below are **development** results, not SIU3R
official mAP/PQ, and this model uses **GT camera poses** to build rays while
SIU3R is an **unposed** setting — the input conditions are not equal.

Static audit baseline `f16d83386fcc1f495f9721083cb73e47fb3abd22`; the worktree was
clean at that commit and no uncommitted work existed.  Findings are in
`implementation_audit.md`; runtime evidence in `runtime_diagnostics.json`.

## Status summary

| item | status |
|---|---|
| static audit of the six requested items | **complete** (all six confirmed, 3 metrics-neutral, 2 reporting-layer, 1 real specification bug) |
| corrected re-evaluation of G0+ / v1 / v2 | **complete — reproduces every published number exactly** |
| condition A (`recipe_v2_pure4`, 6000 steps) | **complete — failed all five gates** |
| B1 official export + pinned evaluator | export for `recipe_v1` complete (1860 pairs); G0+ export + both official evaluations **in progress** |
| condition B2 (stop-gradient) | smoke **passed**; 6000-step run **in progress** |
| full-run preparation | **complete** (`full_config.json`, `full_split.json`, `plan_full_50000.json`, `gate.json`, `submit_full.sh` — all fail-fast, not launched) |

## 1. What the audit found (§A of `implementation_audit.md`)

Six static items were checked against the baseline source with file:line
evidence.  Only one changes the trained function class:

* **A2 (real bug):** `forward_full` ran the historical
  `cross_attn → query_norm → mlp` prefix **first** and then stacked the four-layer
  deep decoder on its output, while the brief (and `recipe_v1/config_diff.json`)
  said the deep decoder *replaces* the prefix.  Recipe_v1/v2 are therefore
  "deep-on-top-of-legacy", which is what condition A corrects.
* **A1 `logit_scale`**, **A3 self-comparing plan check**, **A5 union-mask figure**
  cannot change a metric; **A4 (PSNR read from a history file)** and **A6 (stale
  manifest loss/single-variable strings)** were reporting-layer defects.
* All six are fixed: `group_recipe_head_mode` (`legacy_prefix` default /
  `pure4`), a real plan/split provenance check against the pre-registered
  constants and the reference run, PSNR/SSIM recomputed from the checkpoint
  under evaluation, per-query colours with GT-assisted panels labelled, and
  manifest strings generated from effective options.

**These defects do not invalidate the published numbers**, and §D proves it.

## 2. Corrected re-evaluation reproduces the published values exactly

Job 55636 on the same 8 windows, novel views 2/3, aligned by
`(scene, view, frame_id, key)`:

| arm | Δ IoU1 mean | Δ AP50 | Δ novel PSNR | Δ SSIM | counts | records |
|---|---|---|---|---|---|---|
| G0+ | 0.0 | 0.0 | 0.0 | 0.0 | identical | 55 |
| recipe_v1 | 0.0 | 0.0 | 0.0 | 0.0 | identical | 55 |
| recipe_v2 | 0.0 | 0.0 | 0.0 | 0.0 | identical | 55 |

## 3. Runtime audit (§B) — zero failures

Read-only, 0 failures, `checkpoints_unchanged = true`, plan/split SHAs equal to
the pre-registered constants, 7 training + 8 validation windows per arm, and
the recipe contribution numerators identical to the routing-v2 statistics
(`per_key_max_abs_diff = 0.0`).  Key numbers:

* thing/rest/dropped token split and stratified entropy/CE/KL/agreement are in
  `implementation_audit.md` §B1 and `runtime_diagnostics.json`.
* The tokens that carry pixel mass are much less void-dominated than the token
  mean (weighted void probability 0.501 / 0.203 / 0.339 for G0+ / v1 / v2).
* Void accounting: mean `A_void` and the rendered void α-fraction must be quoted
  as a pair per arm (G0+ 0.431 / ≈0.52; v1 0.703 / ≈0.18–0.21; v2 0.767 /
  ≈0.31–0.35); the weighted identity holds to renderer float noise.
* Gradient routing: the auxiliary CE *does* reach the shared
  token/anchor/encoder (norm 0.73 vs main 4.57) — the B2 precondition.
* Permuting the GT labels leaves RGB/alpha/Gaussians/predictions **bit-identical**.

## 4. Condition A: specification-correct head (`pure4`)

Cold start from the same `arm_g0/ckpt_step0` with deep seed 1743, same plan /
seeds / optimizer / schedule / weights; the only effective difference from
recipe_v2 is `group_recipe_head_mode`.  Smoke (`A_smoke.json`) passed: parameters
bitwise identical to the v2 step-0 control, RGB/depth/alpha/GS identical,
historical prefix called **0** times in `pure4` (and 1× in the control), deep
decoder 4 layers, all 74 deep parameters in the optimizer exactly once,
main+aux gradients reach the head.

| arm (novel-only 55) | IoU1 mean | IoU≥0.5 | TP/FP/FN | AP50 | novel PSNR | SSIM |
|---|---|---|---|---|---|---|
| G0+ | 0.1811 | 3 | 3/71/52 | 0.1375 | 19.157 | 0.6598 |
| recipe_v1 | 0.2656 | 8 | 8/91/47 | 0.0938 | 17.359 | 0.6268 |
| recipe_v2 | 0.1927 | 3 | 3/59/52 | 0.0750 | 18.105 | 0.6388 |
| **recipe_v2_pure4** | **0.1886** | **1** | **1/68/54** | **0.0156** | **17.741** | 0.6314 |

**All five gates fail** (`final_table.json`).  The specification-correct head is
**not** better than the historical stacked one: it is slightly worse on every
metric.  Following the brief, this is recorded as a failed arm ("修复正确不保证
指标上涨"), and it does **not** justify re-attributing the recipe_v1 vs G0+
difference to the prefix bug.

Probe-window stability (`probe_mapping.jsonl`, 11 GT instances on the four
pre-registered probe windows; this is an *interval* stability rate, not an
adjacent-step rate):

| interval | same query for the same GT key |
|---|---|
| steps 0/1/2 | 11/11 |
| 1499/1500/1501 | 11/11 |
| 2999/3000/3001 | 9/11 |
| 5999/6000 | 11/11 |
| 0 vs 6000 | **0/11** |

The query bank has full permutation freedom: the assignment is stable over short
intervals but completely re-permuted over the run.  No matching change was made.

## 5. B2: stop-gradient on the auxiliary CE

Preconditions were met (A failed, audit clean, aux reaches the shared
representation), so the single-variable experiment was executed with control =
`recipe_v2_pure4`.  The smoke (`B2_smoke.json`) **passes all pre-registered
fail-fast checks**: parameters identical to the control, main/aux slot logits
identical at the same state (`max abs diff = 0.0`, atol/rtol 1e-6), **aux → shared
gradients None or exactly zero**, main → shared non-zero, aux → head finite and
non-zero, total forward loss identical (0.7686130404472351 both arms).

An earlier B2 smoke failed on `aux_shared_none_or_zero_for_stop`: the flag was
implemented in `step_loss` but not in `recipe_probe_losses`, so the *probe* path
ignored it.  The flag is now honoured in both paths and the smoke passes; the
run was not started before that fix.

| arm (novel-only 55) | IoU1 mean | IoU≥0.5 | TP/FP/FN | AP50 | novel PSNR | SSIM |
|---|---|---|---|---|---|---|
| G0+ | 0.1811 | 3 | 3/71/52 | 0.1375 | 19.157 | 0.6598 |
| control = `recipe_v2_pure4` | 0.1886 | 1 | 1/68/54 | 0.0156 | 17.741 | 0.6314 |
| **`recipe_v2_pure4_stopgrad`** | **0.1486** | **0** | **0/42/55** | **0.0000** | **18.985** | 0.6570 |

Gates: `all_five = false`.  The stop-gradient arm is the **only** arm in this
project to pass the novel-PSNR gate (18.985 ≥ 18.957) — stopping the auxiliary
CE's gradient into the shared token/anchor/encoder does recover reconstruction —
but instance detection collapses (0 TP, 0 records ≥0.5 IoU, AP50 0).  Following
§六.5 ("只恢复PSNR仍判失败"), the arm is recorded as a failure; the trade-off it
exposes (reconstruction recovered, instance output lost) is the clearest
single-variable evidence in this round that the two objectives compete through
the shared representation.

Because no candidate passed all five development gates, `gate.json` is
**BLOCKED** and the 50000-step full run is not started.

## 6. B1: official 1860-pair alignment

Interface verified read-only against the pinned implementation
(`official_interface.json`): SIU3R HEAD `8ea80166be76854f938e90521f1a5b688b755c87`
**matches** the required commit, evaluator/config/constant SHAs recorded, 1860
pairs / 312 scenes / `target_ids` = 6 frames including the 2 context frames.

The synthetic round-trip smoke (`export_smoke.json`) passes with the
**unmodified** `SIU3R.src.evaluator.Evaluator.process_segmentation`: PNG packing
and `pred.json` ids round-trip exactly, the reader reproduces our semantics and
instances exactly, it reads `label_id` and **`score`** from `pred.json` (so the
official mAP does consume our P(thing)), GT stuff classes are dropped from the
mAP GT, a GT-copy prediction is complete, and dropping one predicted thing while
keeping its GT produces an FN.

The real-model smoke on the first manifest pair likewise passed for both arms
(finite official mIoU; `B1_smoke_ok.json`).

**Exports:** both arms 1860/1860 pairs with `checkpoint unchanged = true`
(`recipe_v1` sha `39c53ce8…`, G0+ sha `a63dd485…`, before = after).

**Official 1860-pair results** (pinned SIU3R evaluator; `target` covers all six
frames the model saw, i.e. it **includes** the two context frames; `context`
covers the two context frames only):

| metric | recipe_v1 | G0+ | SIU3R full (paper, different protocol) |
|---|---|---|---|
| PSNR / SSIM / LPIPS | 15.42 / 0.589 / 0.663 | 17.42 / 0.619 / 0.655 | 25.96 / 0.8220 / — |
| depth absrel / rmse | 0.213 / 0.465 | 0.204 / 0.451 | — |
| context mIoU / target mIoU (panoptic product) | 0.0318 / 0.0320 | 0.0484 / 0.0483 | 0.5922 / 0.5920 |
| semantic-only context / target mIoU | 0.0350 / 0.0353 | 0.0408 / 0.0409 | — |
| context / target PQ | 0.00615 / 0.00610 | 0.02859 / 0.02848 | 0.6612 / 0.6495 |
| context / target mAP (segm) | 5.0e-5 / 6.5e-5 | 3.5e-5 / 3.2e-5 | 0.2817 / 0.2714 |
| context / target AP50 | 2.6e-4 / 2.3e-4 | 1.6e-4 / 1.6e-4 | — |
| context / target AP75 | 1.3e-6 / 1.8e-6 | 9.3e-6 / 2.6e-6 | — |

The official evaluator **does** read `score` from `pred.json`, so these mAP/AP
numbers do use our P(thing) ranking — but note that they are ~3–4 orders of
magnitude below the reference; on the official protocol neither arm is close to
SIU3R, and G0+ is ahead of recipe_v1 on every official segmentation and image
metric.  Development values were **not** a reliable predictor of that ordering.

`B1_official_summary.json` holds the machine-readable version; no value is
invented for metrics the evaluator did not return.

**Fixed assembly policy** (GT-free; ground truth only in the separate `*_gt`
trees): independent semantic head argmax → 1..20 with void 0; thing instances
from the frozen reader (`P(thing) ≥ 0.5`, thing class, group mass > 0.5, area ≥
50 px) with instance id = query+1 fixed across the pair and score = P(thing);
remaining covered pixels take their stuff class, otherwise void.

**Contamination warning (on-disk fact):** 5 of the 8 old development scenes are
inside the official train tree and 3 are official val scenes, so the old 55/110
numbers are not exchangeable with official results.

## 7. Full-run preparation (not launched)

`full_split.json` records the real trees: **1201** train scenes (folders with a
`panoptic` directory) and **312** val scenes, disjoint.  `gate.json` is
**BLOCKED** (no candidate passed the five development gates), and
`submit_full.sh` refuses to start unless `gate.json` reports a passing
candidate.  `full_config.json` fixes peak lr 2e-5, warmup 2000, cosine to 4e-7,
50000 steps, the 2500-step monitoring list (8 official val pairs) and the
low-LR-stability rationale (explicitly *not* a single-variable claim).

## 8. Reference numbers and policy calibration

`reference_numbers.json` records the SIU3R Table 1 values with the protocol
description.  The −0.2 dB PSNR guard used in the recipe rounds is a
**self-imposed engineering threshold**; the reference method's mask-guided
geometry refinement makes understanding and reconstruction complementary
(PSNR +0.45 for U→R), so an intra-instance depth-continuity term (λ≈0.05) is
listed as a strategic option once a recipe passes the gates.  No "close to
SIU3R" claim is made from development numbers.

## 9. Artifacts

Light (Git): this directory (`implementation_audit.md`, `report.md`,
`reference_numbers.json`, `official_interface.json`, `runtime_diagnostics.json`,
`curves.json`, `reeval_comparison.json`, `export_smoke.json`, `A_smoke.json`,
`B2_smoke.json`, `final_table.json`, `full_*.json`, `gate.json`, `plan_full_50000.json`,
`eval_*/…`) and the new scripts.

Heavy (not in Git): `workspace_group_plus/implementation_audit_v1/` — the
A/B2 checkpoints, `B1_full/<arm>/official_predictions_*`, run logs.
