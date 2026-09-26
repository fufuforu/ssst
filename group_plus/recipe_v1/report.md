# Recipe v1 (single arm): four modifications applied together — interim status

**Scope.** 32 train / 8 unseen development split only; this is not an SIU3R
official metric (no official mAP/PQ), and the model uses GT camera poses for its
rays while SIU3R is an unposed setting.  All four modifications were applied
together, so any outcome is the effect of the *combination* and must not be
attributed to one item.

## 1. Fixed baseline (before training, `baseline.json`)

Extracted with the routing_v1 per-record implementation (`fragmentation.csv`,
`IoU1`, aligned by `(scene, view, key)`), novel views 2/3 only:

| metric (novel-only 55) | G0+ step6000 |
|---|---|
| best-over-groups IoU mean | 0.1811 |
| IoU ≥ 0.5 records | 3/55 (small 0/20, large 3/35) |
| GT-free TP/FP/FN | 3/71/52 |
| GT-free AP50 | 0.1375 |
| novel PSNR / SSIM | 19.16 dB / 0.6598 |

Cross-check (never mixed with the 55-record figures): all four views = 110
records, TP/FP/FN 6/140/104, AP50 0.0984 — reproduces the published G0+ numbers.

Pre-registered acceptance (all three): IoU mean ≥ 0.2311 **and** ≥ 6/55 records;
TP ≥ 6 **and** AP50 ≥ 0.1675 (FP listed); novel PSNR ≥ 18.96 dB.

## 2. What was implemented (default-off; old checkpoints unchanged)

`config_diff.json` holds the field-by-field diff.  In short:

1. **Deep group decoder** (`GroupQueryHead.deep`, `group_recipe=True`): 4 layers,
   width 1024, 8 heads, MLP 2048/GELU, dropout 0, order LN→query self-attention→
   residual, LN→token cross-attention→residual, LN→MLP→residual, final LN;
   50,403,328 new parameters; the old single-layer path stays registered and is
   not called in the recipe forward (recipe off ⇒ no new parameters and old
   checkpoints load with `strict=True`).
2. **Segmentation weight/ramp**: instance/group outer weight 0.1 with its own
   `min(1, step/1500)` ramp; the semantic branch keeps 0.05 and
   `min(1, step/2000)`.
3. **Fixed void**: the 101st slot logit is exactly 0 (softmax over 101), the
   learnable shared `background_bias` is unused (and the G0+ background pixel
   supervision is off); measured `fixed_void_max_abs = 0.0` every step.
4. **Token-assignment CE**: targets from the verified routing-v2 per-token
   contribution statistics on the two context frames only (thing → the matched
   query column via the current Hungarian pairing, stuff → column 101,
   255/invalid/unmatched ignored), class-balanced (0.5·thing tokens +
   0.5·rest-only tokens), `seg_ramp × assign_coef × CE_balanced`.

## 3. Preflight (`preflight.json`, job 55474; mechanically fixed before training)

| item | value |
|---|---|
| gradient-ratio probe windows | scene0000_02 0.0220, scene0012_01 0.0469, scene0009_02 0.0280, scene0006_01 0.0109 |
| median ratio | **0.0250** < 0.05 → **assign_coef = 0.2** |
| timing smoke (median with/without target) | **1.268** ≤ 1.5 → **assign_every = 1** (per-step targets) |
| training smoke | all checks passed: finite losses/grads; loss 0.218975 → 0.218403 within 20 updates (weights frozen at the step-1 ramp); old params received gradients and were updated; new decoder updated; alpha conservation ≤2e-6; assignment target row-sum error ≤1e-6; `fixed_void_max_abs = 0` |

Smoke-measurement note: the literal criterion is evaluated with both ramps frozen
at the step-1 value, because at steps 1..20 the pre-registered ramps are still
rising and a step-0 (ramp = 0) reference is unreachable by construction; both
numbers are recorded in the preflight manifest.

## 4. Training (job 55512, cold start from `arm_g0/ckpt_step0`)

Same plan (`a2a65c13…`), fp32, seed 42, AdamW lr 1e-4, warmup 2000, cosine to 2 %
over 6000, grad clip 1.0, new parameters in the existing wd/no-decay groups.
Run 1 (job 55487) was stopped early and is *not* reported as a result: its
per-step logging surface lacked the assignment CE/argmax fields (harness defect),
so the run was restarted from the same step-0 state with the logging fixed.

Interim curves (run 2, still training at the time of writing):

| step | novel PSNR | mIoU | AP50 | TP/FP/FN | assignment CE | argmax agreement |
|---|---|---|---|---|---|---|
| 0 | 10.70 | 0.013 | 0.0000 | 0/0/55 | 4.706 (step 1) | 0.000 |
| 500 | 11.78 | 0.069 | 0.1250 | 3/20/52 | 0.926 | 0.767 |
| 1000 | 15.09 | 0.095 | 0.0000 | 0/48/55 | 1.507 | 0.438 |
| 1500 | 15.47 | 0.138 | 0.1378 | 6/61/49 | 1.125 | 0.533 |
| 2000 | 16.43 | 0.145 | 0.0525 | 7/76/48 | 0.620 | 0.939 |
| 2500 | 16.05 | 0.141 | 0.0375 | 4/54/51 | 0.458 | 0.952 |

The assignment CE falls from 4.706 to ≈0.46 and argmax agreement rises to ≈0.95
on the training window, i.e. the auxiliary term is being learned.  Novel PSNR is
≈1.1–1.5 dB behind G0+ at the same steps, so the PSNR gate (≥18.96 dB) is at
risk and the final verdict must wait for step 6000.

## 5. Pending (must not be pre-empted)

* finish 6000 steps (checkpoints at step0/3000/6000, atomic, plus rolling);
* final evaluation with the routing_v1 per-record implementation:
  `python scripts/eval_recipe_v1.py --checkpoint workspace_group_plus/recipe_v1/run/ckpt_step6000`;
* fill the acceptance table (IoU mean, IoU ≥ 0.5 count, GT-free TP/FP/FN/AP50, and
  the novel-PSNR gate), buckets (small <3000 px / large ≥3000 px), context and
  all-4-view tables (labelled, never mixed with the 55-record figures), figures
  and the failure diagnosis in the pre-registered order (PSNR gate → training
  assignment CE/agreement → context-fits-but-novel-fails → assignment improves
  but masks do not).

Logs and checkpoints: `workspace_group_plus/recipe_v1/run/` (not in Git).
