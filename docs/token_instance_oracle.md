# Token -> instance oracle from compositing contributions (read-only)

Both step-6000 checkpoints, the same 8 unseen validation scenes, the same frames
and colours.  No query training and no change to the Gaussians.  Script:
`scripts/token_instance_oracle.py`.

## 1. Method

Per view, each Gaussian's EWA-projected 2D covariance, opacity and front-to-back
ordering give its compositing weight `c_i(p) = w_i(p) * prod_{j<i}(1 - w_j)`.
Contributions of the Gaussians of one token are summed into a per-token pixel
map; summing over all tokens must reproduce the renderer's alpha.  Verification
over **all** pixels (not only alpha > 0.5):

* alpha reconstruction MAE: LocusGS 0.0052 (max 0.0137), TokenGS 0.0043 (max
  0.0155) across all scenes and views.

Token assignment uses **only the two context frames'** GT instance masks: a
token takes the instance with the largest share of its context-frame mass, or
background if that share is below 0.30.  The threshold is fixed on context, and
the assignment is one per token (consistent across both context frames by
construction; the per-frame argmax agrees for 62.8 % of LocusGS and 80.9 % of
TokenGS tokens).  The grouped token contributions form the instance masks
(threshold 0.5) which are then evaluated on the two **novel** frames over the
complete annotated region.  A `four-view oracle` (assignment from all four
frames' GT) is reported separately as an optimistic upper bound; novel GT is
never used in the context->novel assignment.

## 2. Results (86 instances with >= 200 GT pixels, both models)

| metric | TokenGS | LocusGS |
|---|---|---|
| novel IoU mean / median | 0.247 / 0.177 | **0.567 / 0.598** |
| novel IoU p10 / p90 | 0.000 / 0.634 | 0.276 / 0.849 |
| four-view oracle IoU (upper bound) | 0.258 | 0.577 |
| coverage mean (1 - miss) | 0.506 | **0.877** |
| tokens per instance p50 / mean | 20 / 189 | 122 / 172.7 |
| instances receiving **zero** tokens | **22 / 86** | 0 / 86 |

By instance size (GT pixels):

| size | TokenGS IoU / coverage | LocusGS IoU / coverage |
|---|---|---|
| small < 500 px (n=5) | 0.000 / 0.000 | 0.231 / 0.404 |
| medium 500-3000 (n=22) | 0.059 / 0.327 | 0.371 / 0.812 |
| large >= 3000 (n=59) | 0.338 / 0.616 | 0.669 / 0.942 |

Boundary pixels are rendered (alpha > 0.5 at 99.9 % / 100 %), so the errors are
assignment errors, not missing geometry; the per-frame error maps show the
mis-assigned band along object borders (yellow) and unannotated predictions
(blue).

## 3. Answer: is the existing local token set enough for a query stage?

* **LocusGS: close to sufficient.**  The context-only oracle already reaches
  mean novel IoU 0.567 (median 0.598) with 87.7 % coverage, and the four-view
  upper bound is only 0.010 higher - so almost nothing is lost by assigning from
  the context frames alone.  Errors concentrate on **small and medium
  instances** (IoU 0.23 / 0.37) while large instances reach 0.67; every instance
  receives tokens (median 122).
* **TokenGS: not sufficient.**  Mean IoU 0.247, coverage 0.506, and 22 of 86
  instances receive no token at all; the four-view oracle barely improves it
  (0.258), so the ceiling is not caused by the context-only assignment.

## 4. Where the remaining LocusGS gap comes from

The upper bound being only 0.010 above the context->novel result rules out
"not enough context information" as the limiting factor.  The residual error is
on small/medium instances and along boundaries, and the previous rounds showed
the two candidate causes:

* **instance coverage / cross-instance contribution** - 62.8 % of tokens are
  assigned below the 0.30 share threshold (background) and ~3.6 of 64 Gaussians
  per token carry all the weight, so a token's support is thin;
* **within-token GS redundancy** - 98 % of the within-token footprint pairs
  overlap, so a token cannot represent more than one small region;
* the **depth-scale** mismatch (GT-vs-model ratio 1.08-3.39 across scenes)
  prevents the model's own depth from being used as an extra grouping cue.

Since the four-view ceiling is flat, the next step is not "better context
information" but a token representation that can carry distinct support for
small instances (spread/coverage), and only afterwards query-based grouping.

## 5. Figures

`workspace_recon_diag/token_instance/oracle_{locusgs,tokengs}/<scene>_oracle_overview.png`
- for all 8 scenes and both models, one overview each: rows = the four frames
(two context, two novel), columns = RGB | GT instances | token contribution
coverage | context->novel oracle mask | error map (red = missed, yellow =
mis-assigned, blue = predicted where GT is unannotated) | four-view oracle
(labelled as the optimistic upper bound).  GT and oracle share one colour per
instance.  Frame ids and context/novel tags are burned in.

Note on a reported field: `gt_pixels_covered_by_tokens` in the JSON is
normalised by the whole image rather than by the instance size and is therefore
not used; the `coverage` column is the correctly normalised quantity.

## 6. Reproduce

```
python scripts/token_instance_oracle.py --model .../lgs_lr1e4/ckpt_step6000 \
  --preset train_siu3r_locusgs_recon_bounded_delta_frozen_radius \
  --split .../cross_scene/split.json --max-scenes 8 \
  --out .../token_instance/oracle_locusgs
```
