# Instance query head, phase 1: implementation + single-sample overfit

Frozen fp32 LocusGS step6000 (`cross_scene/lgs_lr1e4/ckpt_step6000`); encoder,
decoder, anchors and Gaussian head all frozen.  Only the new head trains.
Code: `tokengs/models/instance_query_head.py`,
`scripts/train_instance_query_overfit.py`.

## 1. What the head does

* Input: the frozen decoder's final-layer **scene tokens** `[T=1024, C]`, which
  depend only on the two context frames (the decoder never sees novel images).
* 100 learned queries cross-attend to those tokens; the head then produces, for
  every token, logits over **100 query slots + 1 background slot**
  (`sim(token, query)` plus a learned background bias) and a per-query
  objectness logit.  A 20-class semantic head is instantiated but unused.
* Masks: `mask_q(p) = sum_t A[t,q] M_t(p)` with `M_t` the verified per-token
  compositing contribution map, so
  **`sum_q mask_q + mask_bg = alpha`** — measured max error **4.2e-7**
  (the earlier direction, query->token, did not have this property).

## 2. Frozen-model verification (smoke)

| check | result |
|---|---|
| repeat-forward self difference (renderer determinism) | rgb 0.0, depth 0.0 |
| rgb before vs after attaching the head | **0.0** |
| depth before vs after attaching the head | **0.0** |
| reconstruction PSNR before -> after | 23.2320 -> **23.2320** |
| parameters receiving gradients | head 18/20, **frozen 0/450** |

So attaching the head leaves the reconstruction bit-identical (the renderer is
deterministic here, so no tolerance was needed), and no frozen weight is touched.
The splatted alpha reproduces the renderer alpha with MAE 0.003-0.006.

## 3. Single-sample overfit (`scene0012_02`, 3 thing instances, 300 steps)

| step | loss | bce | dice | matched | mean IoU (all 4 views) |
|---|---|---|---|---|---|
| 1 | 2.933 | 1.460 | 0.983 | 3/3 | 0.000 |
| 50 | 0.765 | 0.061 | 0.322 | 3/3 | 0.292 |
| 150 | 0.692 | 0.046 | 0.300 | 3/3 | 0.397 |
| 300 | **0.658** | 0.038 | 0.291 | 3/3 | **0.407** |

Per-instance on the novel view: best matched query reaches **IoU 0.846**, the
worst **0.000** — the head learns to group but the overfit plateaus around 0.41
mean IoU and leaves one instance unclaimed.

Outputs saved: `instance_query_head.pt` (head only, 38 MB; the 220M frozen
backbone is referenced, not copied), `rows.json`, and figures
`scene0012_02_query_overview.png` (4 rows = 4 frames, columns RGB | GT | query
mask | error map | context oracle) plus `scene0012_02_instance_{best,worst}_q*.png`
zooms.

## 4. First weak link (why the overfit plateaus)

1. **Token support is redundant**: ~3.6 of 64 Gaussians per token carry all the
   weight and 98 % of within-token footprints overlap, so `M_t` is a blob; a
   query mask is a convex combination of such blobs bounded by alpha and cannot
   carve thin structures.
2. **Single-sample supervision**: one sample with 3 instances and 100 queries
   gives a very weak matching signal (many queries never match, and the
   no-object term drives them down); a Hungarian step over 3 targets cannot
   shape 100 queries.
3. **No annealing** of the assignment softmax or the matching cost; a hard
   assignment schedule is the obvious next knob.

None of these is the frozen backbone: the reconstruction path is verified
unchanged, so this is a head/representation issue.

## 5. Not done in this round

32/8 shared-head training, official-protocol export (one packed PNG per frame),
semantic labels, and any AP/IoU comparison against the context oracle at scale.
Per the protocol review the 32/8 numbers would be development numbers only.

## 6. Reproduce

```
python scripts/train_instance_query_overfit.py \
  --checkpoint workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step6000 \
  --scene scene0012_02 --steps 300 --out workspace_recon_diag/instance_query/overfit
```
