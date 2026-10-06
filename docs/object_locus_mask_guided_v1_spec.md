# Object-Locus Mask-Guided Feedback v1

This paired experiment changes only the object-to-anchor feedback route. The
scientific base is `7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3` on branch
`object-locus-mask-guided-v1`, in `/space/mawb/ssst_object_locus_mask_guided_v1`.

## Arms

**C / control** keeps the original registered layers and route:

`normalize(f_anchor) @ normalize(mask_embedder(q_new)).T / 0.1 + geometry_bias(mu,c_new,s_new)`,
with a fixed zero void logit and softmax over the 103 object/stuff/void channels.

**M / mask-guided** swaps each existing L6/L8/L10/L12 layer's Python class in
place. It adds no parameters and preserves every state-dict key, shape, dtype,
and value. The route uses the raw mask-head dot products:

`z = f_anchor @ mask_embedder(q_new).T`,
`A_feedback = softmax(concat(z, zeros([B,1024,1])), dim=-1)`.

Thus `z` is `[B,1024,102]`, `A_feedback` is `[B,1024,103]`, and normalization
is over channels. `sigmoid(z)` remains the predicted anchor membership. Evidence
attention remains normalized over 1024 anchors. Both arms use the same
`A_feedback[..., :102] @ layer_norm(q_new)` message and unchanged capped token
injection. Geometry, child masks, classifier, losses, rendering, and injection
constants remain the baseline implementation.

## Fresh paired recipe

Both arms are built from the original panoptic runtime and fresh initialization:
global seed 42, object seed 31415; reconstruction step 47500, MASt3R encoder,
and COCO panoptic epoch 60. Required load counts are 450 reconstruction, 292
MASt3R encoder, 187 adapter, and 326 mask decoder tensors; the other 725
MASt3R tensors are excluded. No trained Full1201 or frozen-encoder weights are
loaded. All visual and object parameters remain trainable; `W_inject` starts
at zero.

The fixed V2.1 manifest SHA is
`ebed1a133d64ed38ef7afce17aaaf27bbe65c6b0edea950c229b5d1e4ff77bf0`. Training
uses the eight registered scenes, their 56 `train_all56` windows, and the
original two-context-view inputs. Each epoch uses `default_rng(42+epoch)` and
seven consecutive global batches of eight, with no padding. Each arm runs 64
epochs, 448 global updates and 3584 window exposures; each window appears 64
times. Evaluation lists are `train_all56`, `same_scene_holdout8`, `dev8`, and
`val32` and are never used for training except `train_all56`.

Optimizer groups, AdamW settings, independent reconstruction/understanding
gradients, GC alpha 0.01, explicit gradient averaging, FP32, and global clip
1.0 follow the panoptic runtime. Peak LRs are `1e-6`, `1e-5`, and `1e-4` for
reconstruction, pretrained understanding, and new object parameters. LR warmup
is 25 updates (200 window exposures), then cosine decay through update 448.
Understanding loss weight and injection beta use current forward exposure
`8 * completed_updates`.

## Gates and outputs

The CPU contract covers state identity, route shape/formula/axes, and fixed
sampling. Single-GPU smoke uses two temporary updates per arm; eight-GPU smoke
uses forty. Smoke checkpoints are never resumed into formal training. Formal
checkpoints are complete at epochs 0/8/16/32/64 (updates 0/56/112/224/448) and
record model, optimizer, per-rank RNG, data plan, update/exposure counts, and
code SHA. Logs include the first update, every ten updates and each epoch
endpoint; `progress.json` records loss, exposure, checkpoint and ETA. Every
100 updates also logs route and injection summaries, for observation only.

Formal evaluation is deferred until requested. It compares the epoch-64
endpoints with the pinned official evaluator. Context reads official `all` /
`context`; target-all reads `all` / `target`; true-novel reads `novel` /
`target`. Forward evaluation uses endpoint exposure 3584 so token injection
uses its trained endpoint scale. Candidate and packed-panoptic metrics stay separate. The primary
endpoint comparison is same-scene holdout true-novel official packed AP50 with
2000 paired scene resamples, seed 2026, and global AP recomputed from sampled
packed predictions (never a mean of per-scene AP values).

Conclusion B requires the AP50 difference 95% CI lower bound above zero,
train-all context official AP50 no more than 0.02 below C, and context/true-
novel PSNR no more than 0.5 dB below C on every split. Otherwise report C and
the measured deltas without tuning. If all four M injection weights stay zero,
report “未形成有效干预”; do not infer that mask-guided coupling has no value.
Cross-scene dev8/val32 results are independent reports, and all conclusions
remain in the ground-truth-pose setting.

No Sinkhorn/OT, extra attention, added losses, geometry contraction, render to
query refinement, or follow-on scale-up is part of this experiment.
