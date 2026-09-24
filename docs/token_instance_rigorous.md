# Reworked token<->instance analysis (compositing-weighted, step-6000 checkpoints)

Read-only; no query, no instance loss, no model change.  Scripts:
`scripts/check_instance_identity_and_depth.py`,
`scripts/token_instance_compositing.py`.

## 1. Instance identity across frames - confirmed

4x4 frame pairs per scene (96 pairs over the 8 validation scenes).  For 4000
sampled GT-depth-valid annotated pixels per pair, back-project with the GT depth
and the camera of frame A, project into frame B, and compare the packed instance
key:

* mean agreement **0.984**, minimum 0.915;
* mean undecidable fraction 0.284 (the point projects outside frame B, or lands
  on a pixel with invalid depth or no annotation).

So the processed panoptic instance keys do refer to the same physical object
across frames of a scene; the earlier instance-key choice was sound.

## 2. Depth units / coordinates / validity - and why the previous gate was wrong

GT depth is uint16 millimetres -> metres; the models work in the normalised
scene space, so the comparison is `z_model` vs `0.15 * z_GT`.

| scene | GT-valid frac | annotated frac | both | GT depth p50 (m) | rendered/GT depth ratio | median abs diff |
|---|---|---|---|---|---|---|
| scene0059_00 | 0.912 | 0.678 | 0.634 | 3.12 | 1.075 | 0.107 |
| scene0072_02 | 0.908 | 0.698 | 0.689 | 2.03 | 1.699 | 0.184 |
| scene0132_01 | 0.972 | 0.998 | 0.971 | 1.46 | 2.001 | 0.230 |
| scene0472_01 | 0.990 | 0.856 | 0.851 | 2.00 | 1.633 | 0.183 |
| scene0559_01 | 0.994 | 1.000 | 0.994 | 2.12 | 1.323 | 0.100 |
| scene0568_02 | 0.970 | 0.885 | 0.874 | 1.97 | 1.590 | 0.178 |
| scene0615_00 | 0.944 | 0.877 | 0.832 | 1.61 | 1.881 | 0.212 |
| scene0695_00 | 0.957 | 0.647 | 0.637 | 1.03 | 3.388 | 0.331 |

The ratio is scene-dependent (1.08 - 3.39), so this is not a unit error but a
per-scene reconstruction-scale error: the models place the scene 1.3-3.4x
farther than the GT-normalised geometry (a global scale ambiguity that leaves
the rendered image nearly unchanged because the cameras sit within 0.014 of the
origin, but that makes any absolute GT-depth gate unusable).  Consequences:

* a pixel-level gate of the form "rendered depth == GT depth" passes only 0.3 %
  of Gaussian-view pairs for TokenGS and 28.6 % for LocusGS - different subsets
  of different sizes, which is why the previous round's 5.7 % vs 62.8 %
  cross-instance numbers were not comparable;
* the meaningful check is the **candidate Gaussian's own depth vs the pixel's GT
  depth**, reported below as its own subset with its own denominators.

## 3. Assignment from the actual compositing contribution (verified)

Each candidate Gaussian's EWA-projected 2D covariance, its opacity and
front-to-back ordering give the compositing weight
`c_i = w_i * prod_{j<i}(1 - w_j)`.  Verification against the renderer's own
alpha at the sampled pixels:

| model | pixels | alpha MAE | within 0.1 | within 0.2 |
|---|---|---|---|---|
| LocusGS step6000 | 12800 | **0.0050** | 99.5 % | 100 % |
| TokenGS step6000 | 12800 | **0.0058** | 100 % | 100 % |

So the approximation reproduces the rendered alpha and the weights are usable
(approximation conditions: EWA covariance with a 0.3 px dilation, no
view-dependent colour, per-pixel candidate gathering by projected footprint).

## 4. Token -> instance on the same annotated-pixel protocol

400 pixels per view, annotated and rendered-alpha > 0.5; 12800 pixels per model,
all covered.  Token purity is computed **per scene** (a token is a different unit
in each scene) and then aggregated.

| metric | TokenGS | LocusGS |
|---|---|---|
| token coverage (mean over scenes) | 1.000 | 0.938 |
| purity p10 / p50 / p90 | 0.380 / **0.640** / 0.988 | 0.689 / **1.000** / 1.000 |
| cross-instance tokens (purity < 0.5) | **31.4 %** | **1.0 %** |
| cross-instance (purity < 0.9) | 75.1 % | 22.8 % |
| instances touched per token (p50) | 2 | 1 |

Own-depth subset (Gaussian's own depth within 0.05 scene units of `0.15*z_GT`),
denominators per scene:

| metric | TokenGS | LocusGS |
|---|---|---|
| depth-ok pixels per scene | 525 - 1599 | 0 - 1096 |
| scene-token pairs with mass | 93 | 1146 |
| token coverage (mean) | **0.011** | **0.160** |
| purity p50 | 0.770 | 1.000 |
| cross-instance < 0.5 | 14.0 % | 0.3 % |
| tokens per instance (p50, instances with >= 50 hits) | 1.0 | 29.0 |

The depth-ok coverage differs by an order of magnitude between the models
(1.1 % vs 16.0 % of tokens), so those two rows must not be compared to each
other; each is a statement about its own geometry-consistent subset.

## 5. Do LocusGS's 64 Gaussians give distinct support?

| metric | TokenGS | LocusGS |
|---|---|---|
| effective Gaussians per token (weight > 0.1 at a sampled pixel) | 44.0 | **3.57** |
| fraction of within-token footprint pairs that overlap | 0.670 | **0.980** |
| mean centre spacing / pair radius | 0.931 | **0.243** |
| near-coincident fraction (from the previous round) | 1.3 % | 45.6 % |

**No.**  For the LocusGS recipe only ~3.6 of its 64 Gaussians ever contribute
with weight > 0.1 at a pixel, and 98 % of the within-token footprint pairs
overlap (mean centre spacing is a quarter of the pair radius).  The 64 Gaussians
behave as one thick blob, not as 64 distinct local samples.

## 6. Answers

1. **Is LocusGS's token really cross-instance mixed, or was that an artifact?**
   It was largely an artifact of the previous projection-based assignment.  With
   occlusion-aware compositing weights, LocusGS tokens are strongly
   single-instance: purity p50 **1.00**, only **1.0 %** with purity < 0.5 (and
   0.3 % on the own-depth subset).  In the same protocol TokenGS is the more
   mixed model (purity 0.64, 31.4 % below 0.5).  The previous "63 % cross-instance"
   number came from associating every Gaussian whose centre merely landed on an
   annotated pixel, with no occlusion or contribution weighting.
2. **Do the 64 Gaussians provide distinct effective spatial support?**  No:
   ~3.6 effective contributors per token and 98 % footprint overlap - the
   "local group" is effectively a single blob.  This is the concrete, and
   still open, weakness of the frozen-radius LocusGS recipe.
3. **What should the query stage address?**  Given (1), the earlier priority is
   reversed: token->instance mixing is *not* the LocusGS problem, so the next
   step is to make the 64 Gaussians of a token carry distinct support (e.g. a
   spread/coverage term or an explicit within-token partition) rather than to
   merge tokens across instances.  For TokenGS, which does show 31 % cross-
   instance tokens with 44 effective Gaussians per token, mixing remains a real
   issue.

## 7. Reproduce

```
python scripts/check_instance_identity_and_depth.py --split .../split.json \
  --model .../lgs_lr1e4/ckpt_step6000 \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius --out .../identity_locusgs
python scripts/token_instance_compositing.py --model .../ckpt_step6000 \
  --preset <preset> --split .../split.json --max-scenes 8 --pixels-per-view 400 \
  --out .../cx_locusgs
```
