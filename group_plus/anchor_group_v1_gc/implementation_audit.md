# Anchor-Group V1-GC Implementation Audit

Experiment identity: `ANCHOR_GROUP_V1_GC_ALPHA001`; architecture remains `LOCUSGS_ANCHOR_GROUP_V1` / model type `siu3r_anchor_group_locusgs`.

## Gradient routing

Forward and loss math use the unchanged V1 model/loss implementation. The logged total remains `L_recon + understanding_weight * L_understanding`; only backward routing changes. Reconstruction loss is unscaled. During understanding backward only, reconstruction-parameter gradient contributions receive fixed factor `0.01`; `anchor_group.*` receives the full understanding contribution. Warm-up remains the canonical V1 helper, the four-group AdamW is reused, and global clip remains 1.0 after both backward contributions.

Step 200 has weight 0, skips the understanding backward and matches V1 reconstruction-only gradients. Step 600 has weight 0.5 and applies `gR + 0.005*gU` to reconstruction, `0.5*gU` to group. Step 1000 applies `gR + 0.01*gU` to reconstruction and `1.0*gU` to group.

Hooks are registered only on unique, trainable, non-`anchor_group.*` parameters around the understanding backward and are removed in `finally`. The helper contains no optimizer step. The future train driver is registered in this script but was not invoked.

## Scalar parity noise audit

The C1 gate uses exact equality for model state, reused batch, Gaussian/RGB/A_post/region_mass/semantic/class outputs, and empirical same-arm repeat envelopes for scalar losses. GPU reduction/render repeat noise is measured independently per scalar; no fixed numeric tolerance replaces that envelope. Root cause: `render_level_numerical_nondeterminism`; first non-exact component: `loss_gaussian_visibility_layer6`.

## Contracts

GC contracts: **15/15 PASS** after the fresh paired RTX 3090 smoke.

| Contract | Result |
|---|---|
| GC-C1_forward_parity | PASS |
| GC-C2_raw_gradient_reference | PASS |
| GC-C3_reconstruction_formula_step1000 | PASS |
| GC-C4_group_unscaled_step1000 | PASS |
| GC-C5_warmup_composition_step600 | PASS |
| GC-C6_zero_understanding_step200 | PASS |
| GC-C7_hook_cleanup_and_next_recon | PASS |
| GC-C8_no_parameter_mutation | PASS |
| GC-C9_optimizer_parity | PASS |
| GC-C10_effective_ratio_0_01x_raw | PASS |
| GC-C11_cosine_invariant_positive_scale | PASS |
| GC-C12_phase_a_18_of_18_regression | PASS |
| GC-C13_reconstruction_parity | PASS |
| GC-C14_s0_s1_legacy_regression | PASS |
| GC-C15_rtx3090_v1_vs_gc_one_step | PASS |

## Same-forward gradient formula contracts

C3 used one FP32 step-1000 forward graph. The production reconstruction backward was captured before understanding backward; the same temporary reconstruction hooks captured each raw understanding gradient before returning `0.01 * grad`, while Group capture hooks returned gradients unchanged. Reconstruction: `450` trainable tensors, `gR nonzero=448`, `gU nonzero=450`, both nonzero `448`, only gR `0`, only gU `2`; `max_abs_diff=0`, `relative_l2_diff=0`. Group: `59` tensors, `37` nonzero, max diff `0`, relative L2 diff `0`, `query_init` diff `0`. Temporary hooks registered/removed `509/509`. No parameter value changed before optimizer step.

Step 600 uses captured understanding gradients that already include the `0.5` loss coefficient; reconstruction and Group formula max/relative differences were both zero. Step 200 skipped understanding backward, had zero temporary hooks, and Group gradients were None/zero.

## Fresh step0 gradient scale

| Category | raw median ratio | effective median ratio | combined/recon norm median | combined vs recon cosine median | frac cosine < 0.9 | frac < 0.5 | frac < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_reconstruction | 34.6928 | 0.346928 | 1.06956 | 0.945999 | 0.1250 | 0.0000 | 0.0000 |
| encoder | 205.791 | 2.05791 | 2.22583 | 0.424630 | 0.9375 | 0.5625 | 0.0000 |
| decoder | 591.154 | 5.91154 | 5.99351 | 0.169622 | 1.0000 | 0.9375 | 0.0000 |
| anchor_geometry | 7.66546 | 0.0766546 | 1.00374 | 0.997250 | 0.0000 | 0.0000 | 0.0000 |
| activation_head | 1.49667 | 0.0149667 | 1.00002 | 0.999898 | 0.0000 | 0.0000 | 0.0000 |
| other_reconstruction | 66.7741 | 0.667741 | 1.19154 | 0.832065 | 0.5625 | 0.1875 | 0.0000 |

Maximum relative effective/raw ratio error: `2.77e-08`; maximum raw/effective cosine difference: `1.07e-08`. Positive scaling controls magnitude, not direction conflict.

## RTX 3090 one-step A/B smoke

Status: **pass**; step 1000, one optimizer step per independent fresh model. Both 509-tensor initial states and the reused batch matched exactly; Gaussian/RGB/A_post/region_mass/semantic/class forward tensors were exact. V1/GC `loss_recon` was `0.0207563415 / 0.0208124444`, diff `5.61029e-5`, envelope `7.68676e-4`; understanding and anchor-group loss diffs were zero. Preclip global/reconstruction/Group norms were `63.6115/59.9950/21.1429` for V1 and `21.2793/2.40556/21.1429` for GC. Decoder norms `58.5575/0.592880`, anchor-mu gradient norm `12.2013/2.32867`, activation-head norms `0.188350/0.0868897`. Query gradient max diff: `5.96e-7`. Clip coefficients: `0.0157204/0.0469940`. Query deltas `9.34377e-5/9.34377e-5`; decoder deltas `9.29832e-6/9.29832e-6`; anchor-mu `9.56655e-6/9.53674e-6`; activation head `9.34303e-6/9.34303e-6`. GC peak allocated/reserved: `4.977/5.451 GiB.


No formal V1-GC 5000-step training was started. No query-starvation fix was introduced. No model, loss, Hungarian definition, optimizer, or LR was changed.
