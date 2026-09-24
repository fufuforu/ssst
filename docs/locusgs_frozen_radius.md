# LocusGS with a frozen decoding radius - fixed-sample reconstruction + locality

Same raw ScanNet scene (`scene0048_01`), same four frames `[654, 664, 655, 659]`,
same 2+2 protocol, fp32, seed 42, 4000 steps, from scratch, same lr/schedule and
same loss.  Relative to the previous `delta = tanh(delta_hat)` variant the **only**
change is that the radius used by the Gaussian decoder is fixed at
`locusgs_radius_init = 0.15`.  No spread penalty, no anchor-refinement change.

Implementation: `tokengs/options.py` gains `locusgs_freeze_decode_radius` and the
preset `train_siu3r_locusgs_recon_bounded_delta_frozen_radius`;
`LocusGSGaussianHead.forward` substitutes a constant radius for the decoding term
only (the learned radii still feed the anchor-to-ray bias) and records it as
`last_decode_radius`.

## 1. The radius really is constant during training

Logged at every evaluation step (values from the run log):

```
decode radius mean/min/max 0.150000/0.150000/0.150000 | learned radius mean 0.188
```

so the decoder uses exactly 0.15 at every step; the *learned* radii still move
(0.148-0.188, i.e. 0.98-1.25x init) but only affect the geometric bias.  For
contrast the two reference LocusGS runs decode with the learned radii:
unbounded 1.014 (6.8x init), bounded-delta 0.990 (6.6x init).

## 2. Reconstruction (context / novel; grey baseline 9.03 / 9.17)

| model | ctx PSNR | novel PSNR | ctx SSIM | novel SSIM | alpha>0.5 |
|---|---|---|---|---|---|
| plain TokenGS | 25.66 | 26.83 | 0.871 | 0.898 | 0.961 |
| LocusGS, unbounded delta | 24.07 | 26.03 | 0.870 | 0.895 | 0.965 |
| LocusGS, delta = tanh | 23.45 | 25.53 | 0.860 | 0.889 | 0.954 |
| **LocusGS, tanh + r = 0.15** | **31.65** | **33.21** | **0.945** | **0.958** | 0.985 |

Freezing the decoding radius does not trade reconstruction away - it *improves*
it by 6.0 dB (context) and 6.4 dB (novel) over the plain TokenGS baseline and by
~8 dB over the unbounded LocusGS.  Renders are visibly sharper
(`workspace_recon_diag/lgs_frozen_r/raw/images/raw_step4000.png`).

## 3. Locality, one measurement standard

`scripts/locality_unified.py`, same fixed batch and one shared scene scale
(visible-surface RMS radius of a fixed reference reconstruction = **0.3174**
scene units; the earlier round used inconsistent normalisers, see the note in
section 5).  p50 values, with the normalised value in brackets:

| metric | TokenGS | LGS unbounded | LGS tanh | **LGS tanh + r=0.15** |
|---|---|---|---|---|
| A: GS -> own token's GS centroid | 0.345 (1.09) | 0.633 (1.99) | 0.652 (2.05) | **0.076 (0.24)** |
| A p90 | 1.298 (4.09) | 1.038 (3.27) | 0.948 (2.99) | **0.117 (0.37)** |
| B: GS -> refined anchor | - | 0.742 (2.34) | 0.749 (2.36) | **0.081 (0.26)** |
| C: GS centroid -> refined anchor | - | 0.448 (1.41) | 0.362 (1.14) | **0.027 (0.09)** |
| D: anchor update vs init | - | 1.685 (5.31) | 1.907 (6.01) | **0.519 (1.64)** |
| D: final anchor z mean | - | 1.883 | 2.118 | **0.747** |

Collapse check (E) and contributing Gaussians (F):

| check | TokenGS | LGS unbounded | LGS tanh | **LGS tanh + r=0.15** |
|---|---|---|---|---|
| per-token span p50 | 0.913 | 0.849 | 0.681 | **0.084** |
| fraction of tokens collapsed (<1e-3) | 0.000 | 0.000 | 0.000 | **0.000** |
| fraction of GS inside frame & opacity>0.05 | 0.992 | 0.925 | 0.911 | 0.959 |
| A (contributing subset) p50 normalized | 1.07 | 1.97 | 2.07 | **0.24** |

Readings:

* **No collapse.** The 64 Gaussians per token never degenerate to a point in any
  variant (0% of tokens under 1e-3); with the frozen radius the span p50 is 0.084,
  i.e. a compact but genuinely extended neighbourhood (0.26 of the scene scale).
* **Locality is not bought by hiding Gaussians off-screen.** 96% of Gaussians are
  inside the frame with opacity > 0.05, and the contributing subset has exactly
  the same locality as the full set (0.24 vs 0.24 normalized).
* Anchors stop drifting as well: the anchor update drops from 5.3-6.0 scene
  scales to 1.6, and the anchor z mean from ~2.0 to 0.75.

## 4. Verdict

For the first time in this line **both** criteria hold at once: the model
reconstructs (31.65 / 33.21 dB, above the plain TokenGS baseline) **and** is
genuinely local (0.24 scene scales, 4.5x tighter than plain TokenGS).  The plain
TokenGS model is *not* the most local model any more once the decoding radius is
constrained - the earlier statement that "TokenGS is more local than LocusGS"
held only for the unbounded-radius LocusGS.

Because the fixed sample now passes both checks, the cross-scene
TokenGS-LocusGS comparison becomes meaningful; it is **not** run in this round.
Proposal (same train/val scene split, same recipe for both models, one seed):
4-6 fixed ScanNet scenes x 2+2 x fp32 x 4000 steps, reporting context/novel
PSNR/SSIM and metric A per scene, to test whether the locality constraint hurts
unseen-scene reconstruction.

## 5. Limitations of the earlier locality numbers

* The previous rounds' summaries are **not** directly comparable: metric A
  ("Gaussian to its token's GS centroid") was never recorded for LocusGS, and the
  LocusGS values reported there used the *initial* anchors while the head
  actually receives the *refined* ones.  Everything above is re-measured from
  checkpoints under one definition.
* The shared scene scale was wrong in the first unified pass (0.0499) because the
  reference model was built without loading its checkpoint; the correct value
  from the loaded reference is 0.3174, and all numbers above use it.

## 6. Reproduce

```
python scripts/train_plain_recon_ab.py \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius \
  --source raw --out-dir .../lgs_frozen_r/raw --pair-seed 42 --seed 42 --amp fp32 \
  --total-steps 4000 --mid-step 2000 --image-every 200 --save-steps final
python scripts/locality_unified.py --model .../lgs_frozen_r/raw/ckpt_step4000 \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius \
  --ref-model .../unified_refs/tokengs_raw/ckpt_step4000 --out .../locusgs_frozen_r.json
python scripts/probe_scene_scale.py --models <checkpoints>
```
