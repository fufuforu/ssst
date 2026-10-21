# LocusGS training-stability test: peak lr 4e-4 -> 1e-4

Same 32 train / 8 unseen validation scene split, same processed ScanNet data,
2+2, fp32, warmup 2000, 6000 steps, same model, same `delta = tanh`, same frozen
decoding radius 0.15, same loss and sampling.  The **only** change is the peak
learning rate: 4e-4 -> **1e-4**.  Anchor computation and coordinates are
untouched (no clamp).

Both runs start from the same initial state: step 1 is `scene0000_02`,
loss 0.2175, grad norm 4.70 in both; only the step-1 lr differs
(5.00e-08 vs 2.00e-07, exactly the 4x ratio).

## 1. It clears the collapse window

Held-out validation, mean over the 8 unseen scenes (grey ctx 10.81 / novel 10.70).
`loc` = p50 of Gaussian -> own token's Gaussian centroid, over the shared,
GT-depth-defined scene scale.

| step | TokenGS ctx/novel | LGS 4e-4 ctx/novel | LGS 1e-4 ctx/novel | LGS 1e-4 SSIM | LGS 1e-4 loc |
|---|---|---|---|---|---|
| 500 | 13.58 / 13.69 | 15.28 / 15.33 | 15.56 / 15.56 | 0.593 | 0.02 |
| 1000 | 14.67 / 14.84 | 17.75 / 17.83 | 17.36 / 17.58 | 0.627 | 0.03 |
| 1500 | 15.14 / 15.61 | 17.72 / 17.79 | 18.35 / 18.15 | 0.644 | 0.04 |
| 2000 | 16.34 / 16.61 | 17.84 / 17.87 | 19.13 / 18.91 | 0.651 | 0.03 |
| **2500** | 16.50 / 16.71 | **10.81 / 10.70 (dead)** | **19.83 / 19.66** | 0.668 | 0.03 |
| 3000 | 16.75 / 16.94 | 10.81 / 10.70 | 20.04 / 19.71 | 0.667 | 0.03 |
| 4000 | 16.97 / 17.37 | 10.81 / 10.70 | 21.91 / 20.79 | 0.712 | 0.02 |
| 5000 | 17.84 / 17.90 | 10.81 / 10.70 | 22.66 / 21.44 | 0.723 | 0.01 |
| 6000 | 17.86 / 18.04 | 10.81 / 10.70 | **23.23 / 21.75** | **0.733 / 0.715** | **0.01 / 0.02** |

The 4e-4 run died between step 2000 and 2500 and never recovered; the 1e-4 run
passes straight through that window and keeps improving to the end (23.23 /
21.75 dB, i.e. +12.4 / +11.1 over grey, and +5.4 / +3.7 dB over the plain
TokenGS baseline trained for the same 6000 steps).  Validation images:
`workspace_recon_diag/cross_scene/lgs_lr1e4/images/`.

## 2. Dense window 1800-3000 (every 50 steps)

Recorded per step in `lgs_lr1e4/history.json -> "dense"`: lr, loss and its
R/G/B/SSIM/visibility components, per-group gradient norms and per-group
parameter-update magnitudes, per-layer anchor positions and anchor-update
quantiles (12 decoder layers), Gaussian centre depth / scale / opacity for **all**
Gaussians and for the **render-contributing** subset, the rendered alpha
coverage, and the decode radius.  Selected rows:

| step | lr | alpha>0.5 | contrib frac | L12 mu_z | \|mu\|max | upd p99 | all z p50 | contrib z p50 | scale p50 | upd anchor_mu | upd head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1800 | 9.0e-5 | 0.969 | 0.89 | 0.96 | 1.26 | 2.4e-2 | 0.97 | 0.96 | 1.6e-2 | 9.7e-4 | 1.9e-2 |
| 2000 | 1.0e-4 | 0.840 | 0.99 | 1.11 | 1.42 | 3.9e-2 | 1.11 | 1.11 | 1.8e-2 | 1.1e-3 | 2.6e-2 |
| 2500 | 9.6e-5 | 1.000 | 0.65 | 0.99 | 1.42 | 4.6e-2 | 0.96 | 0.95 | 1.5e-2 | 9.4e-4 | 2.2e-2 |
| 3000 | 8.6e-5 | 0.984 | 0.56 | 1.06 | 1.73 | 8.1e-2 | 1.03 | 1.01 | 1.4e-2 | 9.7e-4 | 2.0e-2 |

The decoding radius is 0.150000-0.150000 in every dense record.  Across the
whole window the anchors stay at z ~ 0.96-1.31 with |mu|_max <= 1.8, per-layer
anchor updates have p99 <= 0.09 per step, the alpha coverage never drops below
0.84, the contributing-Gaussian fraction stays 0.41-0.99, and the
contributing subset's depth matches the full set (e.g. 0.96 vs 0.97 at step
1800, 1.03 vs 1.06 at step 3000) - so the model is not "localising" by pushing
Gaussians out of frame.

For contrast, the 4e-4 run one evaluation later had final anchors at z ~ 202.7
with scales collapsed to ~1e-10 and alpha 0.0.

## 3. Answers

1. **Does lowering the peak lr clear the step 2000-2500 collapse?**  Yes.  With
   everything else identical the 1e-4 run trains continuously through the
   window to step 6000, with anchor positions, update magnitudes, alpha coverage
   and the contributing fraction all staying in a normal range.  Since it did
   not collapse, no anchor constraint was needed and none was added.
2. **Are reconstruction quality and locality both retained on unseen scenes?**
   Yes, and both are better than before.  At step 6000 the held-out mean is
   23.23 / 21.75 dB with SSIM 0.733 / 0.715 (+12.4 / +11.1 over grey), versus
   17.86 / 18.04 dB and SSIM 0.637 for the plain TokenGS baseline - so the
   earlier "localisation costs something" reading is now reversed on this
   split.  Locality stays at 0.01-0.02 scene scales on both the full Gaussian
   set and the render-contributing subset, i.e. ~10x tighter than TokenGS
   (0.18) and tighter than the 4e-4 run while it was alive (0.02-0.12).

## 4. Reproduce

```
python scripts/train_cross_scene.py --split .../split.json \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius --lr 1e-4 \
  --steps 6000 --eval-every 500 --log-every 100 \
  --dense-start 1800 --dense-end 3000 --dense-every 50 --amp fp32 \
  --save-steps 2000 2500 3000 6000 --out-dir .../cross_scene/lgs_lr1e4
```
