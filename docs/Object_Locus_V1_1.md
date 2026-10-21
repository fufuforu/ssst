# Object-Locus V1.1

## Version and scope

V1.1 reuses the registered `siu3r_object_locus_v1` model type and file layout. Its manifest/report version is `Object-Locus V1.1`, and its architecture label is `LOCUSGS_OBJECT_LOCUS_V1_1`. The V1 implementation remains available in the preceding Git commit.

The only model/loss changes from V1 are:

1. Remove c/s spatial bias from ownership logits.
2. Compute final and auxiliary anchor CE from raw ownership logits.
3. Compute matched pixel-mask BCE and Dice directly from rendered probability mass.
4. Use a dedicated probability-domain wall/floor loss with the same target masks and reductions as the prior Object-Locus stuff loss.

The diagnostic-only nonfinite capture path is retained. This version does not claim to explain or fix the V1 step-4090 failure.

## Fixed model behavior

The model keeps 1024 anchors, 100 thing queries, two stuff queries, and a non-query void channel. State layers remain 6, 8, 10 and 12. Evidence reading, query update, c/s update, Gaussian generation, parent-anchor ownership inheritance, 16-dimensional identity features, semantic readout, and classifier dimensions are unchanged.

For anchor embedding `a` and updated query state `q_new`:

```python
e = normalize(W_own_e(LN_own_a(a)), dim=-1, eps=1e-6)
u = normalize(W_own_u(LN_own_q(q_new)), dim=-1, eps=1e-6)
feature_logits = einsum("btd,bqd->btq", e, u) / 0.1
ownership_logits = cat([feature_logits, W_void(a)], dim=-1)
A = softmax(ownership_logits, dim=-1)
```

`ownership_logits` and `A` have shape `[B,1024,103]`. Channels 0–99 are things, 100/101 are wall/floor, and 102 is void. Ownership contains no c/s spatial term. c/s remain in evidence reading and the existing geometry update; all registered layers, including L12, still update c/s. Gaussian children inherit their parent's anchor ownership exactly; there is no Gaussian-level reassignment.

## Fixed loss behavior

### Anchor CE

Matched thing anchors target their matched query; wall/floor target channels 100/101; ignored anchors are excluded. On valid anchors:

```python
anchor_ce = F.cross_entropy(
    ownership_logits[valid].float(), anchor_target[valid], reduction="mean"
)
```

If there are no valid anchors, the differentiable zero is `ownership_logits.sum() * 0.0`. Anchor Dice is unchanged and still consumes ownership probabilities. One final L12 Hungarian solve is reused for L6/L8/L10 auxiliary terms.

### Pixel masks

For matched thing queries and valid context pixels, raw `region_mass` is checked for finiteness and for gross range violations beyond `[-1e-5, 1+1e-5]`; only physical floating-point boundary noise is clamped to `[0,1]`.

```python
pixel_bce = F.binary_cross_entropy(p, target.float(), reduction="none").mean()
pixel_dice = mean(1 - (2 * sum(p * target) + 1)
                    / (sum(p) + sum(target) + 1))
```

There is no logit conversion, interior probability cutoff, custom gradient, or label smoothing. PyTorch's standard probability-domain BCE is used. This avoids the explicit V1 probability cutoff; it does not promise that every extreme renderer/softmax state is unsaturated.

### Stuff

`object_locus_stuff_loss` is specific to this model. It retains the prior two-context-view target/valid-pixel rules and averages wall/floor BCE terms and Dice terms separately. BCE consumes `[0,1]` probability mass directly; Dice uses that same probability. The return remains `5 * stuff_bce + 5 * stuff_dice`. The legacy `instance_state_loss.py::stuff_loss` is unchanged.

All other losses and weights remain fixed:

```text
thing_2d = 2*class_CE + 5*pixel_BCE + 5*pixel_Dice
stuff_2d = 5*stuff_BCE + 5*stuff_Dice
anchor_group = anchor_CE + anchor_Dice
final_understanding = 0.1*thing_2d + 0.1*stuff_2d
                   + 0.1*semantic_NLL + 0.01*identity
                   + 0.1*anchor_group
aux_l = 0.2*class_CE_l + 0.1*(anchor_CE_l + anchor_Dice_l)
understanding = final_understanding + 0.25*mean(aux_6, aux_8, aux_10)
```

No-object class weight remains 0.1. Hungarian cost, target construction, semantic loss, identity loss, thresholds and class mapping are unchanged.

## Training recipe

Fresh initialization uses the locked pretrained reconstruction checkpoint (`step=47500`, SHA256 `5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`) with strict reconstruction transfer and a newly initialized object branch (`seed=31415`; global seed 42). It does not resume an Object-Locus V1 checkpoint.

The fixed manifest and plan are `train128_windows1024.json` (SHA256 `1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483`) and `plan_C_frozen_5000.json` (SHA256 `ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323`). “Frozen” is only the historical plan filename; all reconstruction, decoder and geometry parameters remain jointly trainable.

The recipe is batch size one, FP32, no AMP, 5000 optimizer steps, AdamW (`betas=(0.9,0.95)`, `eps=1e-8`, `amsgrad=False`, `foreach=False`, `fused=False`), four exhaustive object/reconstruction decay/no-decay groups, peak LR `1e-4` / `1e-5`, weight decay `0.05` / `0`, and global grad clip `1.0`. Gradient control remains:

```text
shared reconstruction: g_reconstruction + 0.01*w*g_understanding
object branch:          w*g_understanding
```

The warm-up is zero through step 200, linear to one by step 1000, then fixed at one. LR warms linearly through 200 and follows the registered cosine schedule to 2% peak at 5000. No beta schedule or object-to-anchor injection is used; beta is zero.

Evaluation nodes are 0, 1000, 2000, 3500 and 5000. Metrics are reported for train16, val8 and val32; local context/target-all, official all/novel, and float context/novel PSNR remain distinct. V1.1 does not alter the official evaluator.

## Validation and interpretation

CPU contracts and targeted gradient tests verify the ownership formula, c/s independence of ownership, raw-logit anchor CE gradients for final/aux layers, probability-domain pixel BCE/Dice/stuff behavior, guard behavior, and optimizer coverage. The RTX 3090 smoke uses the locked plan step-1000 batch, executes one optimizer step, and runs the fixed val32 single-pair evaluator/export interface. Smoke completion is an execution check, not evidence that instance/panoptic tasks succeeded.

Recorded pre-training validation: the existing 13 Object-Locus CPU contracts and 5 V1.1 gradient tests pass (`18/18`). The single smoke job completed on `3dimage-13`, NVIDIA GeForce RTX 3090 (23.68 GiB), using the locked step-1000 batch; strict pretrained transfer matched 450 reconstruction tensors with no missing or shape-mismatch keys. The optimizer step and local plus official export/evaluator interface completed, with peak allocated/reserved memory of 5.03/5.38 GiB. The single-window evaluator reported zero local instance recall and undefined official AP for its empty prediction set; this is not an understanding-task success claim. The smoke audit also records the module paths, source hashes, and execution worktree HEAD.

Task success is assessed from complete object masks, classification, semantic IoU, PQ, official mAP/AP50, and reconstruction PSNR. A completed run with weak masks/AP must be reported as a completed run whose understanding task has not reached an effective level. V1 has no step-5000 result; only a same-protocol step-3500 comparison is valid.
