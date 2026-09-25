# G0/G1 implementation map (against the code actually on the server)

Every row below names the module that already implements the required function
and the *minimal* interface that has to be added.  Nothing that exists is
reimplemented: no second panoptic decoder, no second Gaussian-contribution
implementation and no second evaluation protocol.

| required function | existing implementation (reused) | minimal new interface |
|---|---|---|
| 12-layer LocusGS reconstruction, `delta = tanh(delta_hat)`, frozen decode radius 0.15, layers {6,12} with weights {1/3,2/3} | `tokengs/models/locusgs_recon.py` (`LocusGSAnchorDecoder`, `LocusGSGaussianHead`, `tokengs/options.py` `locusgs_bound_delta` / `locusgs_freeze_decode_radius`), `tokengs/models/canonical_recon_models.py::LocusGSRecon._decode/_layer_objective` | one extra **optional** decoder hook `set_group_feedback_hook(hook, layer)`; the committed A/B hook `set_token_update_hook` keeps its `(tokens, mu)` signature, so the old experiment stays byte-identical |
| group head: 100 learnable group queries + 1 background slot, token→101-slot softmax, per-group objectness + 20-class semantics | `tokengs/models/instance_query_head.py::InstanceQueryHead` (queries, `token_norm/token_proj`, cross-attention, `mlp`, `objectness`, `background_bias`, 20-class `semantic` head) | additive `forward_full(tokens, anchors=None, radii=None)` returning `(slot_logits[B,T,101], objectness_logits[B,Q], group_class_logits[B,Q,20], query_features)` plus an anchor/radius projection built from the existing `spatial_grounded_tokens.build_anchor_encoding`; the existing `forward` is untouched |
| differentiable per-group pixel masks from the RGB Gaussians; every Gaussian inherits its token's assignment | `tokengs/models/siu3r_joint_ssst.py::SIU3RJointSSST.render_query_masks` (broadcast token→GS then `GaussianRenderer.render_feature_channels`) and `tokengs/rendering/gs.py::render_feature_channels` | none beyond parameterising the head count (100 + background); the new model calls the same two primitives |
| alpha conservation `Σ group masks + background mask == rendered alpha` | same as above (softmax over 101 slots makes `Σ_t-w` exactly the alpha compositor output) | numeric check in the smoke + per-step metric |
| scene-level Hungarian matching (Dice + BCE + class CE, no-object weight) | `tokengs/models/ssst_loss.py::hungarian_match`, `_pairwise_bce`, `_pairwise_dice`, `_sample_points`, `ssst_contracts.py` weights | a thin `group_instance_loss` that applies the audited `LOSS_WEIGHT_CLASS_CE/MASK_BCE/DICE` + `NO_OBJECT_CE_WEIGHT` composition **without** the audited outer 0.05, because this round's total applies the single scalar `0.05 * lambda(step)` |
| cross-view consistent instance identity, context-supported thing targets, empty-target protection | `tokengs/models/ssst_loss.py::build_context_segments` (one segment per thing instance, merged over the 2 context views; `stuff_class_count=2` = wall/floor) | filter the returned segments to `class >= 2` (things), as this round matches thing instances only |
| per-GS 20-class semantic prediction + pixel semantic NLL on valid pixels | `tokengs/models/object_locusgs.py::GSAttributeHead` (semantic half) and `LocusGSObjectRecon.semantic_loss` | factor those two bodies into module-level `render_semantic_probability()` / `semantic_pixel_nll()` (the existing method delegates, so the committed A/B numbers stay reproducible) and add a `use_instance_head=False` flag so this round allocates no unused 16-d embedding head |
| ScanNet panoptic conventions: 20 classes, class 0 valid, 255 ignore, stuff = 0/1, thing = 2..19, packed ids `semantic*1000 + instance` | `tokengs/data/siu3r_processed.py::packed_panoptic_to_labels`, `object_locusgs.{semantic_supervision_mask, thing_instance_mask, instance_keys}`, `ssst_contracts.SEMANTIC_CLASS_COUNT` | none (asserted in the smoke) |
| context-only model input, novel-GT leakage test | `SIU3RProcessedProvider` / `split_data` (encoder reads the first 2 records) | reuse the existing `scripts/smoke_official_pair.py` leakage pattern for the new model |
| 32/8 split, processed ScanNet, 2 context + 2 novel, fp32, fixed validation windows | `workspace_recon_diag/cross_scene/split.json`, `tokengs/data/siu3r_processed.py::pin_pair`, `scripts/object_locusgs_eval.py::build_val_entries` | none (reused as-is); the pre-registered 6000-step plan `object_locusgs/plan_6000.json` is reused for both arms |
| from-scratch joint training loop, atomic resumable checkpoints, plan-driven batch assertions, guards | `scripts/train_object_locusgs.py` (plan replay, `pin_pair`, per-group grad/update norms, atomic `save_checkpoint`, alpha/grey/non-finite guards) | a sibling trainer that seeds the global RNG identically per arm, builds a **fresh** model, and verifies the shared-weight hash equality of G0/G1 |
| GT-free inference rule (objectness >= 0.5, mask > 0.5, area >= 50 px) and class-agnostic AP50 | `scripts/train_instance_query_shared.py::ap50`, `scripts/eval_fit_shared_window.py` (the frozen gate order) | a group-mask reader in the new eval module that consumes `objectness`, `group_class_logits` and the rendered group masks |
| reconstruction / semantic / instance metrics, per-instance IoU and recall, size buckets, purity, embedding diagnostics, locality | `scripts/object_locusgs_eval.py` (`reconstruction_row`, `semantic_confusion`, `miou_from_confusion`, `gt_instances`, `instance_metrics`, `embedding_similarity_diagnostic`, `purity_diagnostic`), `scripts/train_cross_scene.py::locality` | one optional `contributing_mask` argument on the purity helper (to separate all GS from the GS that actually render) |
| per-token/per-GS contribution maps for diagnostics | `scripts/train_instance_query_overfit.py::token_maps` (analytic, `no_grad`, explicitly **not** a training path) | none; used only inside the diagnostics section |

## The single structural variable

```
G0: tokens(layer10) -> group head -> {slot logits, objectness, class}   (no write-back)
G1: tokens(layer10) -> group head -> {slot logits, objectness, class}
        + token' = token + tanh(g) * Projection(LayerNorm(Σ_q A[i,q] * q_feat_q))
          with g initialised to exactly 0, applied between layers 10 and 11
```

Both arms read the group at layer 10 with the same dimensions, the same
token→slot direction and identical losses; G1 only adds the gated write-back
(its parameters are the only extra ones, reported in the manifest).
