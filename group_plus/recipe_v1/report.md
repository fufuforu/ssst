# Recipe v1 (single arm, 4 changes together) — final step6000 result

**Scope.** 32 train / 8 unseen **development** split only.  These are not SIU3R
official mAP/PQ; the model uses **GT camera poses** to build rays while SIU3R is
an **unposed** input setting.  The four modifications were applied **together**,
so the outcome is the effect of the combination and is **not attributed to any
single item**.

**Verdict.** `acceptance.overall_passed = false`.  Two of the three
pre-registered groups pass (mask quality, GT-free TP); the GT-free AP50
condition and the reconstruction PSNR gate both fail.  In the pre-registered
order the **first断层 is the reconstruction PSNR gate** (novel PSNR −1.80 dB,
below the −0.20 dB allowance), so the single next-variable proposal targets
reconstruction stability of this four-change recipe.

## 0. Job / checkpoint verification (read-only)

| item | value |
|---|---|
| job 55512 | `COMPLETED` (00:58:06, 1×3090), final `[g] done: 6000 steps` |
| `ckpt_step6000/COMPLETE` | `step 6000 arm g0` |
| `train_state.pt` step | `6000`, plan_sha256 `a2a65c1382da0307…` |
| recipe config load | `eval_recipe_v1.py` → `load_state_dict(..., strict=True)` |
| eval job | 55591 `COMPLETED` (00:00:45) → `group_plus/recipe_v1/eval_summary.json` |

Plan SHA256 `a2a65c13…` matches `object_locusgs/plan_6000.json`; the 8 fixed
unseen windows are those of `split.json` used by `build_val_entries` (identical
to routing_v1).  Only **step 6000** is reported; step 3000 / any best-monitor
point is not used.

Checkpoint identity **before vs after evaluation** (unchanged, `IDENTICAL`):

| file | SHA256 | mtime |
|---|---|---|
| `ckpt_step6000/model.pt` | `39c53ce825b68a24292e84c838d836d02808526e4369bdb4483bbe934b7543ba` | 1790412379 |
| `ckpt_step6000/train_state.pt` | `a6dd2b217e95f0f2e200008f40d800838ef6fc1a1bf7eabe6811c363a70a73db` | 1790412384 |

## 1. What was implemented (default-off; old checkpoints unchanged)

`config_diff.json` holds the field-by-field diff.  In short:

1. **Deep group decoder** (`GroupQueryHead.deep`, `group_recipe=True`): 4 layers,
   width 1024, 8 heads, MLP 2048/GELU, dropout 0, order LN→query
   self-attention→residual, LN→token cross-attention→residual, LN→MLP→residual,
   final LN; 50,403,328 new parameters; the old single-layer path stays
   registered and is not called in the recipe forward (recipe off ⇒ no new
   parameters and old checkpoints load `strict=True`).
2. **Segmentation weight/ramp**: instance/group outer weight 0.1 with its own
   `min(1, step/1500)` ramp; the semantic branch keeps 0.05 and
   `min(1, step/2000)`.
3. **Fixed void**: the 101st slot logit is exactly 0 (softmax over 101); the
   learnable shared `background_bias` and the G0+ background pixel supervision
   are off; `fixed_void_max_abs = 0.0` every step.
4. **Token-assignment CE**: targets from the verified routing-v2 per-token
   contribution statistics on the two context frames only (thing → matched query
   column via the current Hungarian pairing, stuff → column 101,
   255/invalid/unmatched ignored), class-balanced
   `0.5·thing-tokens + 0.5·rest-only-tokens`, applied as
   `seg_ramp × assign_coef × CE_balanced`.

## 2. Preflight (`preflight.json`, job 55474; fixed before training)

| item | value |
|---|---|
| gradient-ratio probe windows | scene0000_02 0.0220, scene0012_01 0.0469, scene0009_02 0.0280, scene0006_01 0.0109 |
| median ratio | **0.0250** < 0.05 → **assign_coef = 0.2** |
| timing smoke (median with/without target) | **1.268** ≤ 1.5 → **assign_every = 1** (per-step targets) |
| training smoke | passed: finite losses/grads; loss 0.218975→0.218403 within 20 updates (both ramps frozen at the step-1 value); old params got gradients and were updated; new decoder updated; alpha conservation ≤2e-6; target row-sum error ≤1e-6; `fixed_void_max_abs = 0` |

## 3. Training (job 55512, cold start from `arm_g0/ckpt_step0`)

fp32, seed 42, AdamW lr 1e-4, warmup 2000, cosine to 2 % over 6000, grad clip
1.0, new parameters in the existing wd / no-decay groups.  Run 1 (job 55487) was
stopped early and is **not** reported (its per-step logging lacked the
assignment CE/argmax fields — a harness defect — so it was restarted from the
same step-0 state with logging fixed).

Fixed 8-window curves (novel views 2/3, this recipe's own in-training read-out):

| step | novel PSNR | ctx PSNR | novel mIoU | novel AP50 | TP/FP/FN |
|---|---|---|---|---|---|
| 0 | 10.70 | 10.81 | 0.013 | 0.0000 | 0/0/55 |
| 1500 | 15.47 | 15.28 | 0.138 | 0.1378 | 6/61/49 |
| 3000 | 16.01 | 16.28 | 0.144 | 0.0319 | 4/67/51 |
| 4500 | 17.43 | 17.54 | 0.170 | 0.0063 | 2/66/53 |
| 6000 | 17.36 | 17.75 | 0.174 | 0.0938 | 8/91/47 |

## 4. Final paired comparison (main metric: routing_v1 per-record IoU1)

Main metric uses **only** `audit_group_routing.fragmentation` (`IoU1`) on the
novel views 2/3 of the 8 unseen scenes, aligned by `(scene, view, key)` with
G0+ `fragmentation.csv`.  The aligned table has **exactly 55 unique
`(scene, view, key)` records with no duplicates or missing keys**, and all 55
carry a non-null `baseline_iou1` (`eval_per_instance.csv`, 55 data rows).
`evaluate_entry_v2`'s other
best-over-groups value (which reads ≈0.29) is **not** mixed in.  G0+ baseline was
reproduced first: novel-only 55 → IoU1 mean **0.1811**, ≥0.5 **3/55**; GT-free
TP/FP/FN **3/71/52**, AP50 **0.1375**; novel PSNR 19.157 dB.  All-four-view 110
records 6/140/104 / AP50 0.0984 is a **separate cross-check only**.

| metric (novel-only 55) | G0+ step6000 | recipe step6000 | Δ | requirement | pass |
|---|---|---|---|---|---|
| best-over-groups IoU mean | 0.1811 | **0.2656** | +0.0845 | ≥ +0.05 (≥0.2311) | ✅ |
| best-over-groups IoU ≥ 0.5 | 3/55 | **8/55** | +5 | ≥ +3 (≥6) | ✅ |
| GT-free TP | 3 | **8** | +5 | ≥ +3 (≥6) | ✅ |
| GT-free FP | 71 | **91** | +20 | listed | — |
| GT-free FN | 52 | **47** | −5 | — | — |
| GT-free AP50 | 0.1375 | **0.0938** | −0.0438 | ≥ 0.1675 (+0.03) | ❌ |
| novel PSNR (dB) | 19.157 | **17.359** | −1.798 | ≥ 18.957 (−0.20) | ❌ |

`acceptance.overall_passed = false` (mask_iou_mean ✅, mask_pass_count ✅,
gt_free_tp ✅, gt_free_ap50 ❌, gate_novel_psnr ❌).

### Area buckets (novel GT < 3000 px = 20 records; ≥ 3000 px = 35 records)

| bucket | n | G0+ mean IoU1 / ≥0.5 | recipe mean IoU1 / ≥0.5 | recipe GT-free hits |
|---|---|---|---|---|
| small < 3000 px | 20 | — / 0 | 0.1363 / **1** | 1/20 |
| large ≥ 3000 px | 35 | — / 3 | 0.3395 / **7** | 7/35 |

(G0+ small/large pass split from `baseline.json`: small 0/20, large 3/35.)

### Per-scene paired change (novel-only)

| scene | n | G0+ mean IoU1 | recipe mean IoU1 | Δ | pass G0+→recipe |
|---|---|---|---|---|---|
| scene0059_00 | 19 | 0.1038 | 0.1764 | +0.0726 | 0 → 1 |
| scene0072_02 | 4 | 0.2520 | 0.3518 | +0.0998 | 0 → 0 |
| scene0132_01 | 10 | 0.1189 | 0.2849 | +0.1660 | 0 → 0 |
| scene0472_01 | 2 | 0.5519 | 0.3546 | −0.1973 | 2 → 0 |
| scene0559_01 | 10 | 0.2328 | 0.3668 | +0.1340 | 1 → 6 |
| scene0568_02 | 2 | 0.2513 | 0.2331 | −0.0182 | 0 → 0 |
| scene0615_00 | 6 | 0.2509 | 0.2503 | −0.0006 | 0 → 0 |
| scene0695_00 | 2 | 0.1763 | 0.3279 | +0.1516 | 0 → 1 |

Per-record: **37 records improved, 16 decreased, 2 unchanged** in best-over-groups
IoU; 7 crossed to ≥0.5 (scene0059_00 v3/20034, scene0559_01 v2·3/6003, v2·3/8013,
v3/6002, scene0695_00 v2/20017), while **2 records lost** ≥0.5.  Both losses are
the same GT instance, `scene0472_01` key `4004`, in the two novel views
(0.564→0.364 at v2, 0.540→0.345 at v3) — the single instance-level regression
case.  The count identity holds: 3 − 2 + 7 = 8 ≥0.5 records at step 6000.  The
gain is **not** a single-scene artefact, but it is **not** uniform either.

> **Erratum (added in the recipe_v2 round).**  An earlier version of this report
> said "16 improved, 39 unchanged, 0 decreased".  That was wrong: it was read off
> the `delta` column of `eval_per_instance.csv`, which is the **routing_v1
> fragmentation Δ (union of the top-3 groups minus the best single group)** — a
> non-negative quantity by construction — **not** the recipe-minus-baseline IoU.
> The per-record recipe-vs-G0+ comparison, recomputed directly from `iou1` vs
> `baseline_iou1` (join verified row-by-row against
> `routing_v1/fragmentation.csv`, 0 mismatches, 0 nulls), is **37 up / 16 down /
> 2 unchanged**.  The CSV and the evaluation definition are unchanged.

### Context and all-4-view (never mixed with the 55-record figures)

| scope | recipe TP/FP/FN | recipe AP50 |
|---|---|---|
| context (v0,1) | 7/85/48 | 0.1005 |
| novel (v2,3) | 8/91/47 | 0.0938 |
| all four views (110) | 15/176/95 | 0.0971 |

## 5. Token diagnostics — thing-bearing vs rest-only (separate)

Read-only forward on the 7 verified training windows and the 8 unseen windows
(`token_diag.json`).  CE/agreement are the model's own assignment target, split
by token type so the rest-dominated overall number cannot hide the thing tokens:

| split | thing CE | rest CE | thing argmax agr | rest argmax agr | overall agr |
|---|---|---|---|---|---|
| training windows | **2.28** | 0.37 | **0.41** | 0.96 | 0.72 |
| unseen windows | **2.08** | 0.24 | **0.40** | 0.99 | 0.60 |

Token counts (training: 1168 thing / 1510 rest; unseen: 2304 thing / 1144 rest)
show the overall agreement in the log (0.29–0.96 across steps) swings with the
thing/rest *mix* of each batch, so it cannot be read as the thing-token
agreement.  **The thing-bearing tokens are only ≈0.40 argmax-agreement while the
rest-only tokens reach ≈0.96** — i.e. the auxiliary CE is learned well for rest
and only partially for thing.

Health read-outs on the unseen windows at step 6000: effective groups (mean per
view) **6.75**, void (101st slot) probability mass mean over tokens **0.768**,
but rendered **void mass fraction 0.154** of α (the many high-void tokens
contribute little pixel mass), α conservation error **1.36e-6** (≤2e-6), mean α
≈0.9997.

Relation between assignment and mask quality: across the 8 unseen windows
corr(thing-token agreement, best-over-groups IoU) ≈ **0.42** — positive but
weak; the highest-agreement windows are not uniformly the highest-IoU ones, i.e.
high thing agreement does **not** translate one-to-one into better masks.

## 6. Failed-acceptance diagnosis (pre-registered order)

1. **Reconstruction PSNR gate — FAILED (first断层).**  Novel PSNR is −1.80 dB vs
   G0+, well past the −0.20 dB allowance, and the gap appears early and never
   closes (it is ≈1.1–1.5 dB behind G0+ from step 1000 and stays there).  Under
   the fixed protocol this stops the mask improvement from being declared a
   usable baseline, so the PSNR gate is the load-bearing failure.
2. **Training-window assignment only partially learned for thing tokens.**
   Thing CE ≈2.1–2.3 with argmax agreement ≈0.40 on the training windows, vs
   rest ≈0.24–0.37 CE / ≈0.96 agreement.  (Effective average weight is the
   full per-step scheme — `assign_every = 1` — so this is not a sparse-weight
   artefact.)
3. **Transfer.**  Thing-token agreement is essentially the same on the training
   (0.41) and unseen (0.40) windows, so the weak thing-token learning is a
   general property of the representation, not a train/test gap.
4. **Assignment vs rendered mask.**  Overall best-over-groups IoU improved
   (+0.0845) and TP rose (3→8), but AP50 fell because FP grew 71→91 — masks
   improved on some records while producing more spurious detections, and the
   reconstruction regression co-occurs.  Because the reconstruction gate fails
   first, this is reported as unresolved rather than as a mask-formation
   verdict.

**Single next-variable proposal (not implemented this round).**  Target
**reconstruction stability of the four-change recipe**: run a single-variable
pair in which the **only** difference is the segmentation outer weight ×
its ramp (0.1 × 1..1500) versus G0+'s original segmentation weighting, holding
the deep decoder, fixed void and assignment CE fixed.  Mechanical falsifiable
criterion: recipe + reconstruction-preserving weighting must reach **novel
PSNR ≥ baseline − 0.20 dB (≥18.96 dB)** on the same 8 windows at step 6000 while
keeping best-over-groups IoU mean ≥ baseline + 0.05; if PSNR still falls below
18.96 dB the reconstruction interference is not explained by the segmentation
weight and the next variable becomes the deep-decoder capacity/interaction.

## 7. Artifacts

Light results (Git): `group_plus/recipe_v1/{baseline.json, config_diff.json,
manifest.json, preflight.json, val_curves.jsonl, eval_summary.json,
eval_per_instance.csv, eval_per_scene.json, token_diag.json, recipe_scene*.png,
report.md}`, scripts `scripts/eval_recipe_v1.py`,
`scripts/recipe_v1_token_diag.py`, submit `group_plus/recipe_v1/submit_eval.sh`.
Recoverable checkpoint + run logs (not in Git): `workspace_group_plus/recipe_v1/run/`
(step0/3000/6000 + rolling), `workspace_group_plus/logs/recipe_*.log`.

Representative figures (`RGB | GT instance | GT-free prediction | error`):

* success: `recipe_scene0059_00_f516_id20034.png` (IoU1 0.057 → 0.683, G0+→recipe);
* success: `recipe_scene0072_02_f752_id20031.png`;
* failure: `recipe_scene0059_00_f511_id5016.png`.

## 8. Evaluation-harness changes (this round, `exp:` commit)

Only evaluator code was touched; no model/loss/threshold/data change and no
optimizer step.  In `scripts/eval_recipe_v1.py`:

1. removed a non-existent key `n_pred` from the per-scene GT-free dict (crash);
2. fixed the conservation dim `group_mass[0,:2].sum(0)` → `.sum(1)` (crash);
3. **additive reporting** of GT-free hits split by the pre-registered novel area
   buckets (<3000 / ≥3000), obtained by calling the *existing* GT-free matcher
   (`object_locusgs_eval.instance_metrics`) with those two cuts only — the main
   metric, thresholds and read-out are unchanged, and the rerun reproduced the
   same 55-record numbers.

Note: 32/8 numbers are development results only (not SIU3R official mAP/PQ);
the model uses GT camera poses for rays while SIU3R is unposed.
