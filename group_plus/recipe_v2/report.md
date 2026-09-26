# Recipe v2 (single variable vs recipe_v1): main instance/group outer weight 0.1 → 0.05

**Scope.** 32 train / 8 unseen **development** split only; these are not SIU3R
official mAP/PQ, and the model uses **GT camera poses** for its rays while SIU3R
is an **unposed** setting.  This is **one** paired training run: it tests only
whether lowering the main instance/group weight (with the deep group decoder,
the fixed void and the assignment CE held fixed) restores reconstruction while
keeping the mask gain.  It does not licence claims about those other three
changes.

## 0. Result in one line

`acceptance.overall_passed = false` → **pre-registered branch 3: "lowering the
main group weight to 0.05 did not by itself restore reconstruction."**  Novel
PSNR recovered **+0.75 dB** (17.36 → 18.10) but is still **−1.05 dB** below the
gate, and the mask/TP/AP50 gains of recipe_v1 largely disappeared (IoU mean
0.2656 → 0.1927, IoU≥0.5 8/55 → 3/55, GT-free TP 8 → 3).

## 1. Pre-training read-only checks (this round)

### 1a. Erratum for the recipe_v1 report (counts of per-record change)

The recipe_v1 report said *"16 records improved, 39 unchanged, 0 decreased"*.
That was wrong and is corrected in `group_plus/recipe_v1/report.md`.  The
numbers had been read off the `delta` column of `eval_per_instance.csv`, which is
the **routing_v1 fragmentation Δ (top-3-group union minus best single group)** —
non-negative by construction — **not** the recipe-minus-baseline IoU.

Recomputed directly from `iou1` vs `baseline_iou1` (join against
`routing_v1/fragmentation.csv` verified row-by-row, 0 mismatches, 0 nulls):

| direction (recipe_v1 − G0+ best-over-groups IoU) | records |
|---|---|
| higher | **37** |
| lower | **16** |
| unchanged | **2** (both exactly 0.0) |

> Note on the instruction.  The requested wording ("16 升 / 2 降 / 37 平") does
> not reproduce either: it carries the same column confusion (16 is the count of
> records with a non-zero *fragmentation* Δ).  The verified counts are **37 up /
> 16 down / 2 unchanged**, and I record those rather than a number the data does
> not support.  What the instruction described correctly is the **gate** view: 7
> records crossed <0.5 → ≥0.5 and **2 fell** ≥0.5 → <0.5, so 3 − 2 + 7 = **8**
> ≥0.5 records at step 6000.

The single instance-level regression case is `scene0472_01` key **4004** in both
novel views (v2: 0.564→0.364, v3: 0.540→0.345).  With recipe_v2 that same
instance recovers to 0.375 / 0.421 — still below 0.5, i.e. it is not restored by
the weight change either.

### 1b. TP-ranking diagnostic of the GT-free read-out (read-only)

Same 8 unseen windows, novel views 2/3, frozen reader (P(thing) ≥ 0.5, thing
class, mask > 0.5, area ≥ 50 px), candidates ordered by P(thing) descending, the
**existing** greedy AP50 match (IoU ≥ 0.5, one GT per candidate) deciding TP.
The greedy loop is a faithful copy of `object_locusgs_eval.instance_metrics` and
is asserted to give the identical TP/FP/FN on every view (else the job fails).
Checkpoint SHA256/mtime unchanged (`checkpoints_unchanged: true`).

| arm | candidates | TP | FP | FN | TP median rank | TP at rank ≤1 / ≤3 / ≤5 | TP score median |
|---|---|---|---|---|---|---|---|
| G0+ step6000 | 74 | 3 | 71 | 52 | **1.0** | 3 / 3 / 3 | 0.926 |
| recipe_v1 step6000 | 99 | 8 | 91 | 47 | **3.0** | 2 / 6 / 8 | 0.890 |
| recipe_v2 step6000 | 62 | 3 | 59 | 52 | **2.0** | 1 / 3 / 3 | 0.912 |

Reading: G0+'s three TPs are all rank 1.  recipe_v1's eight TPs sit lower
(median rank 3; only two at rank 1), and its candidate set grew 74 → 99, so at
any score threshold that keeps its TPs it also keeps many more FPs → AP50 falls
even though TP rises.  recipe_v2 pulls the candidate set back to 62 (fewest FPs
of the three) but also loses the extra TPs.  **The AP50 differences track FP
density inside the ranked list, not a collapse of TP scores** (TP scores stay
0.89–0.93 in every arm).

## 2. Single training variable

`group_plus/recipe_v2/config_diff.json` compares the *effective* training config
field by field (158 fields, output paths / experiment name excluded):

```json
{"differing_fields": {"group_recipe_seg_weight": {"recipe_v1": 0.1, "recipe_v2": 0.05}},
 "single_variable_ok": true}
```

Everything else is identical and is *checked*, not asserted by hand: the deep
group decoder (4 layers, width 1024, 8 heads, MLP 2048, dropout 0), the 101-way
softmax with the 101st logit fixed at 0, the assignment CE (targets from the
verified routing-v2 per-token contributions, `assign_coef = 0.2`,
`assign_every = 1`, **not** scaled by the instance weight), Hungarian matching,
BCE/Dice on the 4096-point sample, semantic loss (0.05, ramp 1..2000),
reconstruction loss, fp32, AdamW grouping, lr 1e-4, warmup 2000, cosine to 2 %,
clip 1.0, seed 42, new-module seed 1743, `plan_6000.json`
(sha256 `a2a65c13…`), `arm_g0/ckpt_step0`.  The instance ramp is still
`min(1, step/1500)`.  The gradient-ratio and timing selections were **not**
redone (they stay `0.2` and `1`).

## 3. Step-0 equality and smoke (`smoke.json`, job 55604/55605 — all checks pass)

Cold start from the same `workspace_group_locusgs/arm_g0/ckpt_step0`; the new
decoder is initialised from seed 1743 and the checkpoint RNG is restored.

| check | result |
|---|---|
| all trainable parameters vs the recipe_v1 step-0 checkpoint | **bitwise equal**, 0 differing tensors, max Δ 0.0 |
| required forward outputs on the first batch (RGB, depth, alpha, Gaussian tensor, 101-way slot logits/probs, tokens, μ, radii, ρ, anchor/radius updates) | **exactly 0.0** difference |
| rasterizer `means2d_pred` / scalar recon loss | v2-vs-ref 7.8e-3 / 6.0e-8, i.e. **below the reference's own run-to-run noise** (1.78 / 4.5e-7) |
| total-loss gap at step 0 | −3.87897e-4 (predicted from the instance weight) vs −3.87907e-4 (actual); residual 1e-11 |
| ≤20-update smoke | losses and grads finite; loss 0.218587 → 0.218097; shared params **and** new decoder both moved; α conservation 1.19e-6; assignment target row-sum ok; fixed void exactly 0 |

## 4. Training (job 55609, 6000 steps, 3363 s, 1×3090)

Same plan, seeds, optimizer and schedule as recipe_v1; per-step scene and the
four frame IDs are asserted against `plan_6000.json`.  `w_inst` at step 1 is
`3.333e-05` = 0.05 × (1/1500), exactly half of recipe_v1's `6.667e-05`,
confirming the single change took effect.

| step | v2 novel PSNR | v1 novel PSNR | v2 mIoU | v2 AP50 | v2 TP/FP/FN |
|---|---|---|---|---|---|
| 500 | 12.47 | 11.78 | 0.062 | 0.0000 | 0/10/55 |
| 1000 | 16.62 | 15.09 | 0.145 | 0.0000 | 0/15/55 |
| 2000 | 16.60 | 16.43 | 0.146 | 0.0000 | 0/28/55 |
| 3000 | 17.32 | 16.01 | 0.158 | 0.0250 | 2/36/53 |
| 4000 | 17.25 | 17.16 | 0.181 | 0.0875 | 4/39/51 |
| 5000 | 17.77 | 17.21 | 0.174 | 0.0437 | 3/53/52 |
| 6000 | **18.10** | **17.36** | 0.155 | 0.0750 | 3/59/52 |

(These are this arm's own in-training read-out; the acceptance numbers below use
the routing_v1 per-record implementation.)  No collapse guard fired; α
conservation ≤ 2e-6 at every evaluation; `fixed_void_max_abs = 0`.

## 5. Final evaluation (job 55610) — same 8 windows, novel views 2/3, 55 records

Alignment asserted: exactly 55 records, 55 unique `(scene, view, key)`, v1 and v2
key sets identical, and the G0+ baseline column identical between the two arms
(`summary_three_way.json` → `alignment`).  Main metric is the routing_v1
per-record fragmentation IoU1 only; `evaluate_entry_v2`'s other best-over-groups
value is not used.

| metric (novel-only 55) | G0+ | recipe_v1 | **recipe_v2** | v2 − v1 | v2 − G0+ |
|---|---|---|---|---|---|
| best-over-groups IoU mean | 0.1811 | 0.2656 | **0.1927** | −0.0729 | +0.0116 |
| best-over-groups IoU ≥ 0.5 | 3/55 | 8/55 | **3/55** | −5 | 0 |
| GT-free TP | 3 | 8 | **3** | −5 | 0 |
| GT-free FP | 71 | 91 | **59** | −32 | −12 |
| GT-free FN | 52 | 47 | **52** | +5 | 0 |
| GT-free AP50 | 0.1375 | 0.0938 | **0.0750** | −0.0188 | −0.0625 |
| novel PSNR (dB) | 19.157 | 17.359 | **18.105** | **+0.745** | −1.053 |
| novel SSIM | 0.6598 | 0.6268 | **0.6388** | +0.0120 | −0.0210 |
| TP median rank (novel) | 1.0 | 3.0 | 2.0 | — | — |

### Pre-registered acceptance (fixed G0+ baseline thresholds)

| condition | required | recipe_v2 | pass |
|---|---|---|---|
| novel best-over-groups IoU mean | ≥ 0.2311 (+0.05) | 0.1927 (+0.0116) | ❌ |
| novel IoU ≥ 0.5 records | ≥ 6 (−0.5 count ≥ +3) | 3 (Δ 0) | ❌ |
| novel GT-free TP | ≥ 6 (Δ ≥ +3) | 3 (Δ 0) | ❌ |
| novel GT-free AP50 | ≥ 0.1675 (+0.03) | 0.0750 (−0.0625) | ❌ |
| novel PSNR | ≥ 18.957 dB (−0.20) | 18.105 (−1.053) | ❌ |

`acceptance.overall_passed = false`.

### Buckets and per scene

| bucket | n | G0+ mean / ≥0.5 | v1 mean / ≥0.5 | v2 mean / ≥0.5 | v2 GT-free hits |
|---|---|---|---|---|---|
| small < 3000 px | 20 | 0.0918 / 0 | 0.1363 / 1 | 0.0572 / 0 | 0/20 |
| large ≥ 3000 px | 35 | 0.2321 / 3 | 0.3395 / 7 | 0.2701 / 3 | 3/35 |

| scene | n | G0+ | v1 | v2 | v2 − v1 | pass G0+/v1/v2 |
|---|---|---|---|---|---|---|
| scene0059_00 | 19 | 0.1038 | 0.1764 | 0.1069 | −0.0695 | 0/1/0 |
| scene0072_02 | 4 | 0.2520 | 0.3518 | 0.4246 | +0.0729 | 0/0/0 |
| scene0132_01 | 10 | 0.1189 | 0.2849 | 0.1500 | −0.1350 | 0/0/0 |
| scene0472_01 | 2 | 0.5519 | 0.3546 | 0.3979 | +0.0432 | 2/0/0 |
| scene0559_01 | 10 | 0.2328 | 0.3668 | 0.2419 | −0.1249 | 1/6/2 |
| scene0568_02 | 2 | 0.2513 | 0.2331 | 0.2914 | +0.0583 | 0/0/0 |
| scene0615_00 | 6 | 0.2509 | 0.2503 | 0.1482 | −0.1022 | 0/0/0 |
| scene0695_00 | 2 | 0.1763 | 0.3279 | 0.3413 | +0.0134 | 0/1/1 |

Per record (v2 − v1): **31 down, 24 up, 0 unchanged**; the loss is concentrated
in scene0559_01 (6 → 2 passes), scene0132_01 and scene0615_00, while
scene0072_02 and scene0472_01 improve.  The behaviour is scene-dependent, not a
single outlier.

### Context and all-four-view (separate tables — never mixed with the 55)

| scope | G0+ TP/FP/FN, AP50 | v1 TP/FP/FN, AP50 | v2 TP/FP/FN, AP50 |
|---|---|---|---|
| context (v0,1) | — | 7/85/48, 0.1005 | 3/56/52, 0.0453 |
| novel (v2,3) | 3/71/52, 0.1375 | 8/91/47, 0.0938 | 3/59/52, 0.0750 |
| all four views (110) | 6/140/104, 0.0984 | 15/176/95, 0.0971 | 6/115/104, 0.0602 |

## 6. thing / rest token diagnostics (`token_diag.json`, read-only)

CE and argmax agreement are the model's own assignment target, split by token
type so the rest-dominated overall number cannot hide the thing tokens.

| arm / windows | thing CE | rest CE | thing argmax agr | rest argmax agr |
|---|---|---|---|---|
| v1 training windows | 2.284 | 0.370 | 0.410 | 0.959 |
| **v2 training windows** | **1.556** | 0.280 | **0.512** | 0.981 |
| v1 unseen windows | 2.075 | 0.237 | 0.396 | 0.994 |
| **v2 unseen windows** | **1.943** | 0.220 | **0.363** | 0.995 |

Window-mean versions: v2 train thing CE 1.563 / agreement 0.518; v2 unseen thing
CE 1.752 / agreement 0.431 (v1: 2.055/0.427 and 1.878/0.454).

Health read-outs on the unseen windows at step 6000: effective groups 7.00, void
(101st slot) probability mass 0.731 over tokens, α conservation 1.22e-6,
`fixed_void_max_abs = 0`.

## 7. Failure attribution (pre-registered order)

1. **Reconstruction PSNR gate — FAILED (first断层).**  18.105 dB vs the 18.957 dB
   requirement (still 0.85 dB short of the gate and 1.05 dB short of G0+).  The
   weight change *did* help: v2 is ahead of v1 at every 500-step evaluation
   point except step 3500 (e.g. +0.69 dB at 500, +1.53 dB at 1000, +0.09 dB at
   2000, +1.31 dB at 3000, +0.75 dB at 6000).  But both curves only climb toward
   G0+ from below and do not close the gap.  → **branch 3**.
2. **Training-window assignment learning (branch-3 sub-decision).**  v2's
   training-window thing CE is **1.556 ≤ 2.28** and its thing argmax agreement is
   **0.512 ≥ 0.41**, so the pre-registered sub-condition is **satisfied**.
3. **Cross-view transfer — supplementary evidence (not a separate branch).**
   Training-window assignment improved (CE 2.28→1.56, agreement 0.41→0.51) but
   the **unseen** windows did not (agreement 0.40→0.36, CE 2.08→1.94) and the
   unseen masks got worse.  So part of the loss is **迁移不足**, not only
   "did not learn".
4. **Assignment vs rendered mask — supplementary evidence.**  The arm that
   assigns better on training windows produces *worse* unseen masks, so a better
   assignment objective is not by itself producing better masks here.

Mechanism caveat, stated exactly as required: the deep group decoder does not
write into the RGB/GS forward, but the group loss does back-propagate into the
shared token/encoder, and the fixed void also changes that gradient.  Because
this round changed only the outer weight, it cannot rule the deep decoder or the
void out as the source of the residual PSNR gap.

### Next single variable (pre-registered by branch 3, **not** run this round)

Because the training-window thing CE (1.556) is ≤ 2.28 **and** the thing argmax
agreement (0.512) is ≥ 0.41, the pre-registered recommendation is:

> **Truncate the assignment auxiliary CE's gradient at the shared
> token/anchor/encoder boundary** — keep it driving the group head only.
> Implementation must be a **separate auxiliary forward** that feeds the group
> head from *detached* shared features (detaching the final logits would not do
> this).

Falsifiable expectation: with the same 6000-step plan, the same seeds and the
same 0.05 instance weight, novel PSNR rises to ≥ 18.96 dB **and** the training
windows' thing agreement is preserved (≥0.51) with unseen thing agreement no
longer below its step-500 level; if PSNR still falls short, the shared-gradient
path of the assignment term is not the binding constraint on reconstruction.

## 8. Artifacts

Light results (Git): `group_plus/recipe_v2/{manifest.json, config_diff.json,
smoke.json, eval_summary.json, eval_per_instance.csv, eval_per_scene.json,
eval_three_way.csv, summary_three_way.json, tp_rank_diag.json, token_diag.json,
report.md, recipe_scene*.png, submit_*.sh}`; scripts
`scripts/train_group_locusgs.py` (`--instance-outer-weight`, `--smoke-only`,
`--reference-init`), `scripts/recipe_v2_config_diff.py`,
`scripts/recipe_v2_three_way.py`, `scripts/recipe_tp_rank_diag.py`,
`scripts/eval_recipe_v1.py` (`--history`).

Recoverable checkpoint + logs (not in Git):
`workspace_group_plus/recipe_v2/run/` (`ckpt_step0/3000/6000`, `train_log.jsonl`,
`val_history.jsonl`, `init_report.json`), `workspace_group_plus/logs/recipe2_*.log`.

Representative figures (`RGB | GT instance | GT-free prediction | error`; the
evaluation entry picks the worst / median / best recipe_v2 record):

* **best** `recipe_scene0559_01_f242_id6002.png` — GT area 17 059 px, best-over-
  groups IoU G0+ 0.553 → v1 0.612 → **v2 0.673** (a case the weight change
  actually improved);
* **median** `recipe_scene0615_00_f554_id8001.png` — GT area 1 682 px,
  0.147 → 0.327 → **0.177** (a recipe_v1 gain that recipe_v2 gives back);
* **worst** `recipe_scene0059_00_f511_id8032.png` — GT area 1 933 px,
  0.189 → 0.317 → **0.000** (the clearest instance-level regression).

Protected checkpoints re-hashed after all jobs: `arm_g0/ckpt_step6000`,
`arm_g0plus/ckpt_step6000`, `recipe_v1/run/ckpt_step6000` and
`arm_g0/ckpt_step0` — SHA256 **and** mtime identical to the pre-run snapshot
(`PROTECTED_CHECKPOINTS_IDENTICAL`).  recipe_v2 `ckpt_step6000/model.pt`:
sha256 `373cac99ab1aa95ab7118b85ea4d7ebc56befb618928cc8c74d8804054caddcb`,
mtime 1790425961 (`checkpoint_unchanged: true` across its own evaluation).

32/8 numbers are development results only — not SIU3R official mAP/PQ.  The model
uses GT camera poses to build rays; SIU3R is an unposed setting.
