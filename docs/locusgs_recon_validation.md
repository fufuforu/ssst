# LocusGS pure-reconstruction validation (fixed sample, fp32, from scratch)

Same ScanNet scene (`scene0048_01`), same four frames `[654, 664, 655, 659]`,
same 2-context + 2-novel raw data path, same fp32 training, seed 42, 4000 steps,
from scratch.  No queries, semantics or instances.  Code:
`scripts/compare_recon_recipes.py`, `scripts/analyze_gaussian_locality.py`,
`scripts/train_plain_recon_ab.py`.

## 1. What actually differs (not a single-variable ablation)

Options-level diff between `train_siu3r_plain_tokengs_canonical_recon` and
`train_siu3r_locusgs_recon` (`workspace_recon_diag/locusgs_recon/recipe_diff.json`):

| field | plain TokenGS | LocusGS |
|---|---|---|
| `model_type` | `siu3r_plain_tokengs_canonical_recon` | `siu3r_locusgs_recon` |
| lr | 1e-4 | **4e-4** |
| warmup (`pct_start_steps`) | 1000 | **2000** |
| `gaussian_z_offset` | 1.0 | **0.0** |
| parameters | 218,871,040 | 220,002,620 (+1.13 M) |

Behavioural differences that the Options diff does **not** show, all in the model
and loss code:

* **architecture** — learnable per-token anchor centres `mu` (1024x3) and raw
  radii `rho`; sinusoidal anchor PE injected into the self-attention input
  (`pe_mode=injected`); anchor-to-ray geometric bias on the cross-attention
  logits with a learnable per-layer `gamma` (softplus, sigma0=0.1, clamp
  [-20,0]); per-layer residual anchor/radius refinement heads (zero-init);
  anchor-centred Gaussian decoding `centre = mu + r*delta` with `delta`
  **unbounded**;
* **supervision** — multi-layer at decoder layers {6, 12} with weights {1/3, 2/3}
  instead of final-layer only, so the reported `loss` is a weighted multi-layer
  sum and is **not** numerically comparable with the plain model's single-layer
  loss;
* **loss** — adds the anchor visibility term (weight 0.1).

Because four groups of things change at once, the two results below are a
*comparison*, not an ablation.

## 2. Reconstruction, side by side (context / novel, grey ctx 9.03 / novel 9.17)

| model | ctx PSNR | novel PSNR | ctx SSIM | novel SSIM | alpha>0.5 |
|---|---|---|---|---|---|
| plain TokenGS, raw | **24.84** | **26.17** | 0.863 | 0.893 | 0.926 |
| plain TokenGS, processed | 24.64 | 25.95 | 0.844 | 0.869 | 0.922 |
| LocusGS, raw | 23.24 | 25.44 | 0.859 | 0.889 | 0.948 |
| LocusGS, processed | 23.51 | 25.70 | 0.841 | 0.872 | 0.931 |

Both reconstruct; LocusGS is 1.6 dB (ctx) / 0.7 dB (novel) behind on raw and
0.5-0.7 dB behind on processed.  **LocusGS does not collapse under fp32** — the
earlier LocusGS failures were the same bf16 problem documented for pure TokenGS.
Renders: `workspace_recon_diag/{locusgs_scratch_fp32,plain_ab_scratch_fp32/raw_ckpt}/*/images/`.

## 3. Locality (the second question)

Metric: for every token, the distance from each Gaussian it generates to that
token's centre, normalised by a **shared** visible-surface RMS radius taken from
a fixed reference reconstruction (0.2988 scene units) so the normaliser is a
scene property, not a property of the model being measured.  For LocusGS the
token centre is the *refined* anchor actually fed to the head.

| model | spread mean | spread / visible | p90 / visible | alpha>0.5 |
|---|---|---|---|---|
| plain TokenGS, raw | 0.441 | **1.48** | 2.52 | 0.926 |
| LocusGS, raw | 1.022 | 3.42 | 5.20 | 0.948 |
| LocusGS + bounded delta, raw | 0.939 | 3.14 | 4.25 | 0.937 |

LocusGS's anchors and radii drift far from their initialisation:

| quantity | LocusGS raw | LocusGS + bounded delta |
|---|---|---|
| anchor refinement ‖mu_final − mu_init‖ (mean)/max | 1.797 / 2.359 | 1.965 / 2.443 |
| final anchor z mean | 2.00 | 2.18 |
| radius (init) | 0.1499 | 0.1503 |
| **radius (final)** | **1.151** | **1.054** |
| ‖delta‖ mean / p99 | 0.881 / 3.244 | 0.888 / 1.456 |
| ‖r*delta‖ mean | 1.022 | 0.939 |

So: **"it reconstructs" holds, "it is actually more local" does not.**  In fact
plain TokenGS is the most local of the variants tested (1.48 visible radii vs
3.14-3.42), because its per-token Gaussians are a tight patch around the token's
own centroid while LocusGS's are a much wider cloud around a drifting anchor.

## 4. Single-variable attempts at locality

Two minimal, model-shape-preserving changes were tested on the *successful*
plain-TokenGS recipe (raw, fp32, from scratch, 4000 steps):

| variant | ctx PSNR | novel PSNR | spread | spread / visible |
|---|---|---|---|---|
| plain TokenGS (baseline) | 24.84 | 26.17 | 0.441 | 1.48 |
| + spread penalty, weight 0.05 | 24.03 | 24.58 | 0.0037 | 0.01 |
| + spread penalty, weight 0.2 | 24.41 | 25.35 | 0.0020 | 0.01 |
| LocusGS with `delta = tanh(delta_hat)` | 22.65 | 24.88 | 0.939 | 3.14 |

* The **unbounded spread penalty drives the spread to ~0** (p50 ~1e-6): the 64
  Gaussians of a token collapse onto a single point.  That is a degenerate
  solution, not a local neighbourhood — the penalty needs a *target* radius
  (e.g. a hinge on `|spread - r0|`), not a minimum at zero.
* **Bounding `delta` alone is not enough**: `r` is a free learnable parameter and
  it grows 7x during training (0.15 -> 1.05), so `‖r*delta‖` stays ~3 visible
  radii.  The bound has to be joint (bound `r*delta` together with `r`, or
  normalise the offset term by a shared radius), otherwise the constraint just
  moves into the radius.

## 5. Conclusion and next step

* LocusGS **does** reconstruct on the fixed sample under fp32 (22.6-23.5 dB ctx,
  24.9-25.7 dB novel) and does not collapse — so the earlier conclusion that
  "LocusGS fails" was a bf16 artefact, exactly like the pure-TokenGS one.
* It is **not more local** than plain TokenGS; anchor refinement moves the
  anchors ~1.8-2.0 scene units (~6 visible radii) from their initialisation and
  the learnable radius grows ~7x.
* Because the locality check fails, the cross-scene TokenGS-LocusGS comparison
  is **not** run (it is only meaningful once the fixed sample passes both
  checks).
* Next single variable (one change, not several): keep everything else fixed and
  make the offset constraint **joint** — decode `centre = mu + r * tanh(delta)`
  with `r` either frozen at `locusgs_radius_init` or itself bounded — and/or
  replace the zero-target spread penalty with a hinge at a target radius.  Do
  not touch attention, positional encoding, the number of supervised layers or
  the Gaussian head in the same step.

## 6. Reproduce

```
python scripts/train_plain_recon_ab.py --preset train_siu3r_locusgs_recon \
  --source raw --out-dir .../locusgs_scratch_fp32/raw --pair-seed 42 --seed 42 \
  --amp fp32 --total-steps 4000 --mid-step 2000 --save-steps final
python scripts/analyze_gaussian_locality.py --model .../ckpt_step4000 \
  --preset train_siu3r_locusgs_recon --source raw \
  --ref-model .../plain_ab_scratch_fp32/raw_ckpt/ckpt_step4000 \
  --ref-preset train_siu3r_plain_tokengs_canonical_recon --out .../locality.json
```
