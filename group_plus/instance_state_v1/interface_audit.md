# interface_audit - instance_state_v1

Baseline check, 2026-09-27.

| item | value |
|---|---|
| repository | `/space/mawb/ssst` |
| HEAD at prepare | `bb37a4be71d2f45bde7cbc25039a93582c36fa91` |
| registered baseline in the spec | `b3acb34946a60f760a7daabbb1959c8d6be2c696` |
| commits in between | one cleanup/audit commit (`bb37a4b`), **no diff** in any dependency file |
| dependency files checked | `tokengs/models/canonical_recon_models.py`, `tokengs/models/locusgs_recon.py`, `tokengs/models/enc_dec.py`, `tokengs/models/attention.py`, `tokengs/rendering/gs.py`, `tokengs/models/ssst_loss.py`, `tokengs/options.py` |
| verdict | HEAD advanced but nothing mathematical or protocol-relevant changed; **fail-fast not triggered** |
| AGENTS.md | none found under `/space/mawb` (searched to depth 3) |
| GPU | partition `3090`, nodes `3dimage-11/13` idle, `3dimage-12` mixed; used `3dimage-13` |
| disk | quota `1.5 TiB`, free at prepare `76.39 GiB` |

## Reused interfaces (verified, not re-implemented)

* `LocusGSRecon._decode` / `_layer_objective` / `forward_reconstruction_only` /
  `_full_supervision` - `tokengs/models/canonical_recon_models.py`
* `LocusGSAnchorDecoder.forward` (the exact per-layer order that `forward_stateful`
  copies) and `LocusGSGaussianHead` - `tokengs/models/locusgs_recon.py`
* `DecoderBlock.SelfAttnBlock` + `Attention` (`fused_attn=True`, `rope=None`,
  `flex_attn_score_mod=None` in the resolved config) - `tokengs/models/enc_dec.py`,
  `tokengs/models/attention.py`; the new self-attention helper re-uses
  `block.norm`, `attn.qkv/q_norm/k_norm/proj/proj_drop/attn_drop` and
  `block.gs_self_attn_scale`, and raises if the verified default SDPA path is
  not available.
* `GaussianRenderer.render_feature_channels` (`deferred_bp=False`) -
  `tokengs/rendering/gs.py`
* `build_context_segments` - `tokengs/models/ssst_loss.py`
* `SIU3RProcessedProvider` (+ `pin_pair`), `model_registry`, `config_defaults`
* format helpers in `scripts/evaluate_ssst_validation.py`,
  `scripts/group_official_export.py`, `scripts/invoke_siu3r_official_evaluator.py`

## New files (all new symbols are additive)

```
tokengs/models/instance_state_locusgs.py     LocusGSInstanceStateRecon, InstanceStateController,
                                             InstanceStateDecoder, deterministic_fps
tokengs/models/instance_state_loss.py        instance_state_losses (+ thing/stuff/semantic/identity)
scripts/run_instance_state_v1.py             --phase {prepare,smoke,paired,full}
scripts/smoke_instance_state_v1.py           checks A-J
```

`tokengs/models/__init__.py` and `tokengs/options.py` only gained the new registry
entry (`siu3r_instance_state_locusgs`) and the two derived preset arms
(`train_siu3r_instance_state_locusgs`, `..._coupled`).  No shared attention code,
no historical group model and no historical training script was modified.

## Pretrained initialisation source

`workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt`

* step 47500 (the best *healthy* monitor weight of the completed full-data run;
  the step-50000 file had already been pruned by the earlier retention policy)
* sha256 `5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`
  (verified equal at prepare time)
* provenance: `docs/locusgs_full_valpair_recon_eval.md`,
  `docs/full_train_cleanup_manifest.json`

Resolution order used by the driver: build `LocusGSRecon` from the *same* options
object and load it with `strict=True`, then copy every non-`instance_state.*` key
into the new model with shape/alias checks.  The adapter's
`config_defaults["eval_siu3r_ssst"]` fallback (`model_type="siu3r_joint_ssst"`)
is never used, because `prepare`, `smoke` and the eval path always pass an
explicit preset/config.
