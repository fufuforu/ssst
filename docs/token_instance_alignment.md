# Read-only token<->instance alignment (TokenGS vs LocusGS, step 6000)

The two saved checkpoints (`cross_scene/tokengs/ckpt_step6000`,
`cross_scene/lgs_lr1e4/ckpt_step6000`) analysed on the same 8 unseen validation
scenes.  No training, no model change; GT depth / instance / semantic maps are
used for analysis only.  Script: `scripts/analyze_token_instance.py`.

## 1. Frame alignment (checked first)

For every validation window's four frames, against the processed tree:

* `max |batch RGB - color/<f>.jpg| = 0.0` (the provider's tensor is bit-identical
  to the released JPEG) in all 8 scenes x 4 frames;
* `panoptic/<f>.png` decoded as `packed = R + 256G + 65536B`, `semantic =
  packed // 1000` equals `semantic/<f>.png` exactly; all annotation maps are
  256x256 like colour and depth;
* re-projecting the Gaussians with the batch intrinsics and the
  `cam_view_all`-derived C2W reproduces the renderer's own `means2d` with
  `|du| = |dw| = 0.00` (p50 and p99), so camera, K and pixels agree.

Label keys used: `packed` (`sem*1000+inst`) so ids never collide across scenes.

## 2. Contribution / visibility gates (documented approximation)

The renderer returns only `means2d`/`alpha`/`depth`, not per-Gaussian rasteriser
weights.  Approximation used: a Gaussian's weight at its own projected centre is
proportional to its opacity, so the per-Gaussian weight is `opacity`, and a
Gaussian is associated to the GT label at that centre pixel only when the
following hold:

1. the projected centre is inside the frame and in front of the camera;
2. the model's rendered alpha at that pixel is > 0.5 (the model covers it);
3. the pixel is annotated (semantic not void/0 and instance non-zero);
4. *loose gate*: nothing more.  *strict gate*: additionally the model's rendered
   depth agrees with the GT depth (|z_render - 0.15 z_gt| <= 0.25 scene units)
   and the Gaussian is not clearly behind the rendered surface.

Pass rates over all Gaussian-view pairs (8 scenes x 4 views x 65536):

| stage | TokenGS | LocusGS |
|---|---|---|
| inside frame | 0.977 | 0.870 |
| + rendered alpha > 0.5 | 0.976 | 0.861 |
| + not behind the surface | 0.036 | **0.504** |
| + rendered depth matches GT depth | 0.003 | **0.286** |
| + annotated pixel (loose gate) | 0.758 | 0.683 |

LocusGS's geometry agrees with the GT depth far more often than TokenGS's (28.6%
vs 0.3% of pairs), which is an independent sign that it is the better-aligned
reconstruction on unseen scenes.

## 3. Token -> instance statistics

Strict gate (geometry-consistent pixels only):

| metric | TokenGS | LocusGS |
|---|---|---|
| tokens with any contribution | 1020 / 1024 (99.6 %) | 947 / 1024 (92.5 %) |
| instance purity p10 / p50 / p90 | 0.50 / **0.69** / 0.81 | 0.22 / **0.41** / 0.92 |
| cross-instance token ratio (purity < 0.5) | **5.7 %** | **62.8 %** |
| cross-instance token ratio (purity < 0.9) | 95.5 % | 89.2 % |
| instances touched per token (p50 / p90) | 2 / 3 | 3 / 5 |
| instances with >= 200 Gaussian hits | 3 | 45 |
| tokens per instance (p50 / mean) | **155 / 346** | **1.0 / 7.8** |
| instances with no pure token | 0 / 3 | **17 / 45** |

Loose gate (annotated pixels only, for sensitivity): both models give low purity
(p50 0.17 TokenGS, 0.20 LocusGS), every token contributes and each touches ~5
instances - this gate is dominated by one large instance that soaks up most
Gaussian hits, so it is reported only to show that the numbers are gate-sensitive
and that the strict (geometry-consistent) gate is the meaningful one.

## 4. Is the small LocusGS distance just near-coincident Gaussians?

| metric (mean over scenes) | TokenGS | LocusGS |
|---|---|---|
| nearest-neighbour distance / own scale, p50 (all GS) | 1.307 | **0.279** |
| ... restricted to render-contributing GS | 1.378 | 0.279 |
| token span / own scale, p50 (all GS) | 3.981 | **1.021** |
| ... contributing GS | 4.128 | 1.010 |
| near-coincident fraction (nn < 0.25 scale) | 1.3 % | **45.6 %** |
| Gaussian scale p50 | 0.0283 | 0.0081 |

**Yes** - for the LocusGS recipe the small local distances are largely an artefact
of near-coincidence: 46 % of its Gaussians have a neighbour within 0.25 of their
own scale, and a token's 64 Gaussians span only ~1x their own scale, i.e. the
"local group" is essentially a single blob (its Gaussians are also 3.5x smaller).
TokenGS's tokens instead span ~4x their scale with only 1.3 % near-coincident
Gaussians - a genuinely extended local neighbourhood.  Restricting to
render-contributing Gaussians changes nothing for either model, so this is not an
off-screen effect.

## 5. Answers

1. **Do the Gaussians form a useful local group?**  For TokenGS yes (span ~4x
   scale, 1.3 % near-coincident).  For the frozen-r LocusGS the group is local
   but nearly degenerate (span ~1x scale, 46 % near-coincident), so "local" here
   mostly means "the 64 Gaussians sit on top of each other".
2. **Do the local groups fall mainly on a single instance?**  Under the
   geometry-consistent gate, TokenGS tokens mostly do (purity p50 0.69, only
   5.7 % below 0.5) while LocusGS tokens mostly do not (p50 0.41, 62.8 % below
   0.5).  Caveat: the strict gate keeps different subsets for the two models
   (28.6 % vs 0.3 % of pairs), so this is a statement about each model's own
   geometry-consistent pixels, not a like-for-like pixel comparison.
3. **What should the query stage solve first?**  The two models fail differently.
   TokenGS has high-purity tokens but massive redundancy - 155 tokens per
   instance - so its problem is **merging tokens of the same instance**.
   LocusGS has almost one token per instance but 63 % of its tokens cross
   instances and 17 of 45 annotated instances have no pure token at all, so for
   the recipe being validated the priority is **reducing cross-instance mixing
   and instance under-coverage first**; merging would be premature.

## 6. Reproduce

```
python scripts/analyze_token_instance.py \
  --model .../cross_scene/lgs_lr1e4/ckpt_step6000 \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius \
  --split .../cross_scene/split.json --max-scenes 8 --image-scenes 3 \
  --out .../token_instance/locusgs2
```
