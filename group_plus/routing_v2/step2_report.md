# Read-only soft token→instance diagnostic on frozen G0+ (B1)

**What this is.** For each of the 8 fixed unseen scenes, one temporary
`Z[T,K]` (K = context thing instances + 1 rest slot) was fitted with Adam for a
fixed 500 steps on the two **context** frames' GT, with G0+ step6000 entirely
frozen: no model parameter was updated, no checkpoint was written, and novel GT
was used only for scoring.  The result is a **context-GT-fitted diagnostic soft
assignment oracle** — it is *not* feed-forward inference, *not* a GT-free
instance segmentation result and *not* an official SIU3R mAP/PQ.

Artifacts: `group_plus/routing_v2/{manifest.json, step1_reproduction.json,
per_instance.csv, per_scene.json, purity_diagnostic.csv, summary.json,
smoke.json, 4 PNGs}` (~1.1 MB).  Script: `scripts/audit_soft_token_oracle.py`
(plus `scripts/audit_group_routing.py`, extended with an optional
`return_masks=True` that does not change its default behaviour).

## 0. Reproduction of Step 1 (required before any Z optimisation)

Reusing `audit_group_routing.token_oracle` and the same windows, the hard
majority-vote oracle was recomputed and aligned row by row against
`group_plus/routing_v1/oracle.csv` on `(scene, view, packed instance id)`:

| check | required | measured |
|---|---|---|
| rows | 55 | 55 |
| max per-row IoU gap | ≤ 1e-5 | **0.0** |
| oracle IoU ≥ 0.5 | 22/55 | **22/55** |
| size strata | 2/20 and 20/35 | **2/20 and 20/35** |

Phase 0 reproduced exactly, so the Z optimisation was allowed to start.

## 1. Setup (verbatim from the protocol)

`Z ∈ R^{1024×K}`, `A = softmax(Z, dim=-1)`, per-token softmax, one scene at a
time, columns = the context thing instances sorted by packed instance id plus a
rest column; initialisation `Z = 4.0` on the token's phase-0 majority column (the
rest column for tokens with no valid thing vote); Adam, lr 0.05, weight decay 0,
500 steps, no scheduler/warmup/clipping/early stopping; loss
`L_instance = 5·mean_i BCE_i + 5·mean_i Dice_i` (per-instance mean, then mean over
instances) `+ 1.0·L_rest` with the class-balanced rest term
`0.5·stuff BCE(p_rest,1) + 0.5·thing BCE(p_rest,0)` on pixels with the original
RGB alpha > 0.5 and valid annotation; `p_rest = rest_mass / clamp(alpha.detach(),
0.5)`.  Rendering uses the model's own `render_feature_channels` with the same
Gaussians/cameras as RGB; novel frames are rendered only for read-out.

Model parameters had `requires_grad=False` throughout, never appeared in the
optimizer and had no gradients (asserted in the smoke and in every scene run);
the checkpoint's SHA256 and mtime are identical before and after
(`checkpoint_unchanged: true`).

**Smoke (10 steps, scene0059_00, then discarded):** all 11 checks passed — A
shape/fp32/finite/row-sum ≤1e-6; `Σ K masks = alpha` ≤ 2e-6; RGB/alpha/GS/camera
unchanged; no model gradients; Z gradient finite and non-zero; loss 9.2226 →
7.9734 (≥1e-5 drop).

## 2. Branch decision (pre-registered; novel-only 55 records, step 500)

**B1 — capacity sufficient: soft oracle IoU ≥ 0.5 in 36/55 ≥ 28/55.**

| metric (novel-only 55) | value | bar |
|---|---|---|
| soft assignment novel IoU ≥ 0.5 | **36/55** | B1: ≥ 28 |
| merged context IoU ≥ 0.5 (same 55 denominator) | 40/55 | B2 would need ≥ 44 |
| Step-1 hard oracle (same rows) | 22/55 | reference |
| soft IoU p25 / p50 / p75 / p90 / max | 0.446 / 0.576 / 0.713 / 0.796 / 0.916 | — |
| context IoU p25 / p50 / p75 / p90 / max | 0.489 / 0.650 / 0.783 / 0.871 / 0.934 | — |

Bucket breakdown (novel-only, same rows):

| bucket | n | Step-1 hard pass | soft novel pass | soft context pass |
|---|---|---|---|---|
| small (<3000 px) | 20 | 2 | 3 | 7 |
| large (≥3000 px) | 35 | 20 | 33 | 33 |

Report-only diagnostics (never used for the vote):

| diagnostic | value |
|---|---|
| (a) purity-filtered hard oracle (token keeps its majority vote only when single-instance contribution / all valid annotated context contribution ≥ 0.50) | **31/55** (vs 22/55 hard), mean rest fraction 0.462 |
| (b) soft records with IoU<0.5 that cross 0.5 when only the area ≥ 50 gate is removed | **0** |

Both runs of the identical protocol (jobs 55409 and 55410) produced the same
decision and the same counts (36/55 and 40/55); run 2 is the delivered one because
run 1's per-step curves were not serialised.

## 3. Per-scene fit (step 0 → 500; context metrics only)

| scene | instances (K−1) | loss | context hard IoU | rest fraction |
|---|---|---|---|---|
| scene0059_00 | 11 | 9.223 → 4.881 | 0.402 → 0.524 | 0.121 → 0.261 |
| scene0072_02 | 2 | 4.469 → 1.771 | 0.374 → 0.688 | 0.286 → 0.536 |
| scene0132_01 | 5 | 5.794 → 3.138 | 0.543 → 0.611 | 0.034 → 0.119 |
| scene0472_01 | 4 | 5.463 → 3.015 | 0.400 → 0.553 | 0.292 → 0.502 |
| scene0559_01 | 5 | 7.337 → 3.856 | 0.423 → 0.614 | 0.054 → 0.241 |
| scene0568_02 | 1 | 3.172 → 1.224 | 0.498 → 0.752 | 0.111 → 0.325 |
| scene0615_00 | 4 | 6.082 → 3.798 | 0.292 → 0.450 | 0.079 → 0.320 |
| scene0695_00 | 1 | 4.103 → 3.235 | 0.189 → 0.365 | 0.622 → 0.799 |

Curves at steps 0/50/100/200/300/400/500 (loss, context hard IoU, assignment
entropy, rest fraction) are in `per_scene.json`; no novel value was ever used to
choose a step.  The window-level context GT area distribution is in
`per_instance.csv` (`context_gt_areas`) — e.g. the small bucket does contain
instances with only a few hundred context pixels, so "context_visible = 55/55"
does not mean "context pixels are large".

## 4. Two measurement caveats (required)

1. **Not comparable with training curves.** This diagnostic optimises a
   full-pixel BCE + Dice objective on the context frames, whereas the trained
   model's mask loss uses the fixed 4096-point sampling; the losses, and hence the
   curves above, must not be compared with training curves.
2. **B1 is an existence statement.** It says that, starting from the phase-0
   majority-vote solution, a ≥28/55 soft assignment is reachable within the fixed
   500 steps / Adam 0.05 / this loss and read-out.  It is not an
   optimizer-independent upper bound of the frozen representation, and it says
   nothing about feed-forward learnability.

Also withdrawn, as instructed: the previous round's candidate "compare the GT
world-space radius with the 0.15 decode displacement radius" is **retracted** —
0.15 is the frozen decode displacement radius, not the Gaussian footprint, so no
such comparison was performed or used.

## 5. Conclusion under the pre-registered branch

**B1 — the frozen token/GS footprint is sufficient under a context-GT-fitted soft
assignment for the majority of instances in this protocol; the bottleneck moves
to feed-forward assignment learning.**  The comparison between the hard
majority-vote oracle (22/55) and the fitted soft oracle (36/55) is exactly the
"assignment can be learned, the read-out as-is cannot express it" case; the
report-only purity variant (31/55) shows part of the gain is available even
without optimisation once mixed tokens are dropped.

Consequences: the next round should re-review group-head options A/B (this round
implements and trains nothing); the small bucket remains weak in absolute terms
(soft 3/20 novel, context 7/20), so any re-review must report the small/large
split, not only the aggregate.

## 6. Representative figures (fixed selection rules)

| file | rule |
|---|---|
| `small_soft_success_scene0059_00_id5019.png` | small instance whose soft fit turns the Step-1 failure into a success |
| `large_still_failing_scene0559_01_id8013.png` | large instance still failing (lowest soft IoU among large failures) |
| `context_ok_novel_fail_scene0059_00_id20034.png` | context IoU ≥ 0.5 but novel fails |
| `both_fail_scene0059_00_id5016.png` | soft and hard oracle both fail |

Each shows: context RGB, context GT, context soft mask, novel RGB, novel GT,
hard oracle mask, soft oracle mask, error map, with scene/frame/id and the IoUs.

## 7. Run record, limits and what was not done

* Jobs: `55409` (run 1) and `55410` (run 2, delivered), both `COMPLETED`,
  1:56 / 1:59 wall clock, single 3090 on `3dimage-11` (`--exclude=3dimage-13`),
  MaxRSS 6.5 GB / 5.0 GB.  The CUDA peak-allocator value was **not
  instrumented** in these runs; the diagnostic holds only `Z[1024,K]` plus K-channel
  renders, and the heaviest allocation is the 1024-channel per-token contribution
  render (~0.27 GB per view).
* No Z, A, dense per-token map or new checkpoint was saved; nothing under
  `tokengs/models/`, no loss/threshold/evaluation script changed, no training job
  was created, and the parallel SIU3R files/jobs were untouched.
* Limits: the oracle's columns come from context GT (so instances absent from the
  context frames have no column and score 0); the fit is per scene with 500 fixed
  steps; the 55-record novel denominator is the only voting basis (the
  110-record all-view numbers are deliberately not computed here).
