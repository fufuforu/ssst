# Object-aware LocusGS: strictly paired A/B development experiment

**Scope.** This is a *development* validation on the frozen 32 training / 8
unseen-scene split of the processed ScanNet tree
(`workspace_recon_diag/cross_scene/split.json`). It is **not** an SIU3R official
evaluation: the official 1860-pair `2 context + 4 novel` test set is never
touched (a single official pairing is used for a shape / "novel GT never enters
the model" forward smoke only). No official mAP/PQ is claimed.

**Pose caveat.** The model consumes GT camera poses to build rays
(`rays_os` / `rays_ds` and the Plücker patch embedding); SIU3R is an *unposed*
setting, so these numbers are not comparable with the SIU3R leaderboard.

## Question

Starting from the same early reconstruction state, is the understanding head
enough (A), or does adding a predicted-instance-relation token update *before*
Gaussian generation improve instance grouping on unseen scenes (B) while
keeping reconstruction?

| | arm A | arm B |
|---|---|---|
| GS semantic logits + instance embedding, shared LocusGS trainable | yes | yes |
| predicted-instance-relation token update before layer 11 | module present, **never applied, never optimized** | applied once, after decoder layer 10 |

The only structural variable between the arms is the relation token update.

## Shared starting point (verified)

| item | value |
|---|---|
| checkpoint | `workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step2000` (model-only, `COMPLETE`) |
| sha256(model.pt) | `f0e791b8bb9d49deeba160f0a5fcf75594e4a0638d79ed79b9021ccd6da31c6d` |
| preset | `train_siu3r_locusgs_recon_bounded_delta_frozen_radius` |
| params / keys | 220,002,620 / 450 |
| source git commit | `3bedbe640a88131d2de153f7451d5a4a9fd382b1` |

Both arms build **fresh AdamW optimizers** and a fresh experiment-local
schedule; the source checkpoint's optimizer/RNG state does not exist and is not
pretended to be restored.

## Recipe (identical for A and B)

* 14-d Gaussian geometry/RGB layout unchanged; `tanh(delta)` bounded offsets;
  Gaussian decode radius frozen at 0.15; fp32; encoder input = 2 context
  images; supervision = 2 context + 2 novel records.
* Layer-6 and layer-12 reconstruction objective and weights unchanged
  (`L_rec = MSE + 0.2*(1-SSIM)/2 + 1.0*L_vis(Gaussians) + 0.1*L_vis(anchors)`,
  weights `{1/3, 2/3}`).
* New GS attribute heads (identical init in both arms): `Linear(1024, 64*20)`
  semantic logits and `Linear(1024, 64*16)` instance embedding, reshaped with
  the Gaussian head's token -> 64-GS order, embedding L2-normalised per GS.
* Understanding loss `ramp(t) * (0.05 * L_sem + 0.10 * L_inst)`,
  `ramp(t) = min(1, t/1000)`, `t` = experiment-local optimizer step.
* Optimizer: AdamW, betas (0.9, 0.95), grad clip 1.0; existing LocusGS
  parameters lr 1e-5, new parameters lr 1e-4; decay/no-decay split follows the
  original rule (>1-d tensors without `_no_weight_decay` decay; new weight
  matrices therefore decay, biases/norms do not). 200-step linear warm-up, then
  cosine to 2 % of the group peak over 6000 steps.
* Batch plan: `object_locusgs/plan_6000.json`, pre-registered before either arm
  starts; step *k* of arm A and arm B read the same scene/context/novel window.
  Recorded `pair_iou` and frame ids are part of the artifact.

### Relation token update (arm B only)

```
h_i  = LayerNorm(token_i)                                     # layer-10 tokens
z_i  = L2Normalize(Linear(C,32)(h_i))
a_ij = softmax_j((z_i . z_j)/0.2)     over the 16 nearest anchors (self included,
                                      by layer-10 refined anchor coordinates)
m_i  = sum_j a_ij Linear(C,C)(h_j)
token'_i = token_i + tanh(g) * Linear(C,C)(m_i),   g initialised to 0
```

Neighbour *indices* are discrete, detached and computed under `no_grad`; every
token/anchor path that enters the update stays differentiable. `tanh(0) = 0`
makes the step-0 forward bit-identical to arm A. There is no GT instance
assignment, no local-unit split, no extra query and no extra geometric loss.

## Losses

* **Semantic**: per-GS `softmax` over 20 logits, composited with the RGB
  renderer's own weights (the same Gaussians, cameras, opacity, scale,
  rotation); the composited mass is normalised by the rendered alpha to a pixel
  class distribution. Per-view NLL over `GT semantic in [0,19] & alpha>0.05`,
  averaged over the four records. Class 0 (wall) is supervised; 255 is ignore.
  Valid-pixel coverage is logged with the loss.
* **Instance**: valid *thing* pixels only (`semantic in [2,19]`,
  `instance_id>0`, `alpha>0.05`); per (instance, view) up to 64 uniformly sampled
  pixels under a fixed RNG; at most 16 instances per step sampled with a fixed
  per-step RNG; identities are the scene-stable key
  `(semantic_id+1)*1000 + instance_id`. `pull` = within-instance
  `mean(1 - e.c_k)`, `push` = `mean(max(0, c_k.c_l - 0.2)^2)` (0 for <2
  instances), instances equally weighted.
* No token-purity, anchor-attraction or spread penalty is added.

## Frozen GT-free instance reader

Fixed before training and identical for both arms; it never uses GT:

1. context candidates: `alpha >= 0.5`, predicted semantic class a thing,
   class probability `>= 0.5`, at most 8192 pixels in fixed view-major raster
   order;
2. traverse by `alpha x class probability` descending, create a prototype when
   the best embedding cosine to the existing prototypes is `< 0.8`, cap 100;
3. two rounds of nearest-cosine assignment (`>= 0.8`) + unit-mean prototype
   update, deleting prototypes whose context assignment is `< 50 px`;
4. prototype class = majority vote of its assigned context pixels;
5. novel pixels (`alpha >= 0.5`) are assigned to the nearest prototype of the
   same predicted class with cosine `>= 0.8`, otherwise "unpredicted"; novel
   masks `< 50 px` are dropped;
6. instance score = mean predicted class probability x mean cosine.

This read-out is a shared instrument; its quality is not attributed to the
learned heads.

## Reporting

Per scene and per evaluated step: context/novel PSNR and SSIM (plus grey-image
baselines), semantic mIoU over valid GT, class-agnostic instance AP50, TP/FP/FN,
small/medium/large GT-instance buckets, plus diagnostics that are labelled as
such: GT-assisted same/different-instance embedding cosine, and token/GS
instance contribution purity (analytic 3-sigma kernel mass, no occlusion
ordering, never a gradient path). Outputs: `val_history.jsonl` curves,
`history.json`, `train_log.jsonl` (per-step scene/context/novel ids, losses,
per-group grad and update norms, learning rates) and `images/` tiles.

## Files

| path | role |
|---|---|
| `tokengs/models/object_locusgs.py` | attribute head, relation module, object model |
| `tokengs/models/locusgs_recon.py` | additive optional decoder hook (default off) |
| `tokengs/data/siu3r_processed.py` | additive `pin_pair` for pre-registered windows |
| `scripts/gen_object_plan.py` | writes the shared 6000-step plan |
| `scripts/train_object_locusgs.py` | arm A / arm B training entry |
| `scripts/object_locusgs_eval.py` | metrics, frozen reader, diagnostics |
| `scripts/eval_object_locusgs.py` | evaluation CLI + A/B side-by-side report |
| `scripts/smoke_object_locusgs.py` | pre-run smoke suite (8 checks) |

Checkpoints and logs stay in `workspace_object_locusgs/` (not in Git); the
manifest, plan, smoke report and evaluation report are the tracked artifacts.
