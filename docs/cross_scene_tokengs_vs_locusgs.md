# Cross-scene 2+2 ScanNet: plain TokenGS vs the validated LocusGS recipe

One shared model per recipe, **not** per-scene training.  32 training scenes and
8 completely disjoint validation scenes, fixed in
`workspace_recon_diag/cross_scene/split.json` (validation = the manifest's
held-out split; training = the first 32 manifest train scenes present in the
processed tree).  processed ScanNet, 2 context + 2 novel per step, fp32, seed 42,
6000 steps, one 2+2 window per scene per step, evaluation every 500 steps on the
same fixed validation windows.

## 1. What the two recipes are (a comparison, not a single-variable ablation)

Recipe diff (`workspace_recon_diag/cross_scene/recipe_diff.json`):

| field | plain TokenGS | LocusGS (validated recipe) |
|---|---|---|
| `model_type` | `siu3r_plain_tokengs_canonical_recon` | `siu3r_locusgs_recon` |
| parameters | 218,871,040 | 220,002,620 |
| lr | 1e-4 | **4e-4** |
| warmup | 1000 | **2000** |
| `gaussian_z_offset` | 1.0 | 0.0 |
| `locusgs_bound_delta` | - | True (`delta = tanh`) |
| `locusgs_freeze_decode_radius` | - | True (`r = 0.15`) |

Behavioural differences not visible in the Options diff, all in model/loss code:

* **architecture** — learnable per-token anchors `mu` and refinements, anchor PE
  injected into self-attention, anchor-to-ray geometric bias with learnable
  `gamma`, per-layer residual anchor/radius refinement, anchor-centred decoding
  `centre = mu + r*tanh(delta)` with `r` fixed at 0.15;
* **supervision** — decoder layers {6, 12} with weights {1/3, 2/3} instead of the
  final layer only, so the reported loss is a weighted multi-layer sum;
* **loss** — plus the anchor-visibility term (weight 0.1).

Both recipes were validated individually on the fixed sample in the previous
rounds; this experiment compares their behaviour on unseen scenes.  It is not a
single-variable structural ablation.

## 2. Frozen-r LocusGS on processed data works

Smoke run before the main experiment: forward runs, no NaN, training loss
0.218 -> 0.182 over 50 steps, gradient norms 2-6, and the **decoding radius is
exactly 0.150000-0.150000 at every logged step** in the full run as well.

## 3. Held-out validation curve (mean over the 8 unseen scenes; grey ctx 10.81 / novel 10.70)

| step | TokenGS ctx/novel | TokenGS SSIM | TokenGS gain | TokenGS loc | LGS ctx/novel | LGS SSIM | LGS gain | LGS loc |
|---|---|---|---|---|---|---|---|---|
| 500 | 13.58 / 13.69 | 0.546 | +2.77 / +2.99 | 0.13 | **15.28 / 15.33** | 0.572 | +4.47 / +4.63 | **0.02** |
| 1000 | 14.67 / 14.84 | 0.563 | +3.86 / +4.14 | 0.13 | **17.75 / 17.83** | 0.633 | +6.94 / +7.14 | **0.05** |
| 1500 | 15.14 / 15.61 | 0.581 | +4.33 / +4.91 | 0.22 | **17.72 / 17.79** | 0.639 | +6.91 / +7.09 | **0.07** |
| 2000 | 16.34 / 16.61 | 0.603 | +5.53 / +5.91 | 0.17 | **17.84 / 17.87** | 0.631 | +7.03 / +7.17 | **0.12** |
| 2500 | 16.50 / 16.71 | 0.615 | +5.69 / +6.01 | 0.22 | **10.81 / 10.70** | 0.479 | +0.00 / +0.00 | 0.30 |
| 4000 | 16.97 / 17.37 | 0.626 | +6.16 / +6.68 | 0.18 | 10.81 / 10.70 | 0.479 | +0.00 | 0.29 |
| 6000 | **17.86 / 18.04** | **0.637** | **+7.05 / +7.34** | 0.18 | 10.81 / 10.70 | 0.479 | +0.00 | 0.29 |

`loc` = p50 of (Gaussian -> its own token's Gaussian centroid) divided by that
scene's ground-truth-depth visible-surface radius (model-independent, shared).

Training loss (every 500 steps): TokenGS 0.071/0.106/0.049/0.095/0.071/0.114/
0.069/0.068/0.071/0.103/0.065/0.047; LocusGS 0.039/0.075/0.035/0.087/**0.123**/
**0.201**/0.134/0.118/0.113/0.166/0.093/0.145.

Per-scene validation at the last healthy LocusGS step (2000): LocusGS is ahead on
6 of 8 scenes (e.g. scene0559_01 22.02 vs 17.84 ctx, scene0568_02 17.48 vs
15.53), behind on scene0695_00 (17.97 vs 19.48) and even on scene0472_01.

## 4. The LocusGS recipe collapses at the LR peak

Between the step-2000 and step-2500 evaluations the LocusGS model dies:

* rendered `alpha > 0.5` coverage goes 0.969 -> **0.000** and stays there;
* the render becomes exactly the grey background (per-scene PSNR equals the
  grey baseline to +0.00 for all 8 scenes);
* gradient norm goes to **0.00** and the training loss rises and stays at
  0.09-0.20.

Probe of the two checkpoints (`scripts/probe_collapse.py`):

| quantity | step 2000 (healthy) | step 6000 (collapsed) |
|---|---|---|
| rendered alpha mean/max | 0.904 / 0.9999 | **0.000 / 0.000** |
| Gaussian opacity mean / p50 | 0.063 / 0.027 | 0.053 / **0.000** |
| Gaussian scale mean / p50 | 1.87e-2 / 9.94e-3 | 1.92e-2 / **3.1e-10** |
| Gaussian centre z mean | 4.34 | **202.7** |
| decode radius | 0.150000-0.150000 | 0.150000-0.150000 |

So the anchors/Gaussians drift ~200 scene units in front of the cameras (cameras
at z ~ 0.00-0.014) while their scales collapse to ~1e-10: nothing rasterises, the
alpha and the gradient both go to zero and the model is permanently dead.
The decoding radius is constant at 0.15 in both states, so the frozen radius is
not what moved - the **unbounded anchor positions** are.

## 5. Answers to the three questions

1. **Can each model reconstruct unseen scenes?**  Yes.  Plain TokenGS improves
   monotonically to 17.86 / 18.04 dB (+7.05 / +7.34 over grey, SSIM 0.637/0.640)
   and is still improving at step 6000.  The frozen-r LocusGS reconstructs
   *better* while it is alive (17.84 / 17.87 at step 2000, ahead of TokenGS at
   the same step on 6/8 scenes) and then collapses.  Neither reaching ~30 dB at
   6000 steps is not treated as an architecture failure: TokenGS's curve is
   still rising and the LocusGS failure is an explicit training blow-up.
2. **Is LocusGS still more local?**  Yes, while healthy: 0.02-0.12 scene scales
   versus 0.13-0.22 for TokenGS, on both the full Gaussian set and the
   render-contributing subset (they agree to within 0.01-0.07 in every row).
3. **Did localisation cost validation quality?**  It did not cost quality before
   step 2000 - LocusGS was ahead.  It cost **training stability**: the validated
   LocusGS recipe, whose only difference from the fixed-sample winner is
   unchanged, blows up at the learning-rate peak over 32 scenes.  That is a
   stability cost, not a quality trade-off.

## 6. Next step (proposed, not run)

The same single variable that the fixed-sample round deliberately left alone is
now the one implicated: the **unconstrained anchor positions**.  Keep everything
else (bounded delta, frozen radius, loss, lr, supervision) and bound the anchors
to the normalised scene box (e.g. clamp/tanh on `mu` after each refinement), or
freeze the anchor refinement heads.  One change at a time, evaluated on the same
32/8 split.

## 7. Reproduce

```
python scripts/train_cross_scene.py --split .../split.json \
  --preset train_siu3r_plain_tokengs_canonical_recon \
  --steps 6000 --eval-every 500 --log-every 100 --amp fp32 \
  --save-steps 2000 4000 6000 --out-dir .../cross_scene/tokengs
# same with --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius
python scripts/probe_collapse.py --preset <preset> --checkpoints <ckpt...>
```
