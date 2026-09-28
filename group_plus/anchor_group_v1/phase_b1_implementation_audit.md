# Anchor-Group V1 Phase-B1 Implementation Audit

## Gates

- Strict pretrained transfer: **PASS**. Checkpoint SHA256 `5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`, step 47500; 450 reconstruction tensors copied, 59 `anchor_group.*` tensors retained at seed 31415. Missing non-group keys, unexpected source keys, and shape mismatches are empty.
- Reconstruction trainability: **PASS**. Total/trainable parameters 222,787,916; reconstruction trainable 220,002,620; Anchor-Group trainable 2,785,296; frozen 0. Canonical baseline also has no frozen parameters. No freeze helper is used.
- Optimizer: **PASS**. AdamW betas `(0.9, 0.95)`, four exhaustive groups, 509 unique trainable tensors covered exactly once; duplicate, missing, and multi-group lists are empty.

| Parameter group | Tensors | Numel | Peak LR | Weight decay |
| --- | ---: | ---: | ---: | ---: |
| anchor_group_decay | 18 | 2,743,496 | 1e-4 | 0.05 |
| anchor_group_nodecay | 41 | 41,800 | 1e-4 | 0 |
| reconstruction_decay | 115 | 218,773,504 | 1e-5 | 0.05 |
| reconstruction_nodecay | 335 | 1,229,116 | 1e-5 | 0 |

Differential LR ratio is 10.0. At steps 200 / 1000 / 5000, group and reconstruction LRs are respectively `1e-4 / 1e-5`, `9.34352447854375e-5 / 9.34352447854375e-6`, and `2e-6 / 2e-7`.

## Warm-up and gradient audits

The single canonical understanding-weight helper passes all exact points: steps 0, 1, 199, 200 = 0; 201 = 0.00125; 600 = 0.5; 999 = 0.99875; 1000 and 5000 = 1. LR multiplier checks pass at steps 0, 1, 100, 200, 201, 1000, 2500, and 5000. Warm-up/optimizer CPU helper contract: 2/2 PASS.

Understanding-only backward (locked train manifest window 0, override 1.0) passed for:

- `anchor_group.query_init`: norm 12.12576.
- `enc_dec_backbone.decoder_blocks.11.gs_self_attn.norm.weight`: norm 0.002725193.
- `anchor_decoder.mu`: norm 2.741025; `anchor_decoder.rho`: 0.03979911.
- `anchor_decoder.refine_mu.0.weight`: norm 1.145462; `anchor_decoder.refine_rho.0.weight`: 0.03865088.
- `activation_head.deconv.weight`: norm 0.3133686.

All are finite and nonzero. Thus understanding gradients reach query state, late decoder, anchor geometry, and reconstruction feature paths.

Reconstruction-only backward passed: late decoder norm 0.0002259451, `anchor_decoder.mu` norm 0.1347099, and `activation_head.deconv.weight` norm 0.08181583. `anchor_group.query_init.grad` is None; it receives no reconstruction-only gradient.

## Evaluator, GPU optimizer smoke, and regression

One-window context evaluator interface smoke: **PASS**, finite semantic/instance metrics and PSNR (23.97909); `n_gt_instances=8`.

The only optimizer step executed was the required fully joint smoke, from fresh pretrained + seed-31415 initialization, on **NVIDIA GeForce RTX 3090 (24 GB class, 23.6843 GiB reported)** at synthetic global step 1000 and understanding weight 1.0. Forward, backward, and optimizer step were finite. Loss total/reconstruction/understanding/anchor-group: `3.7652690 / 0.02075634 / 3.7445128 / 5.1167779`. LRs: `9.34352448e-5 / 9.34352448e-6`. `query_init` max delta `9.34377e-5`; updated reconstruction parameter `enc_dec_backbone.decoder_blocks.11.gs_self_attn.norm.weight` max delta `9.29832e-6`. Peak allocated/reserved memory: `4.38779 / 4.54492 GiB`.

Phase-A remains **18/18 PASS**. The C3 source contract recognizes `fps_index=None` solely as an evaluator compatibility alias and continues to verify there is no FPS/local8 selection operation. Fresh RTX 3090 reconstruction parity: Gaussian max diff 0, RGB max/mean diff 0/0, PSNR diff 0. Baseline and Anchor-Group renderer repeat RGB diffs are both 0.

Legacy S0/S1 GPU forward and loss regression: **PASS**. S0 uses `LOCUSGS_INSTANCE_STATE_V1`, `instance_state_local3d=False`, single-anchor init. S1 uses `LOCUSGS_INSTANCE_STATE_V1`, `instance_state_local3d=True`, `k=8`, local8 init. Both render, Gaussian tensors, region mass, semantic scores, and loss are finite. Legacy evaluator contracts 8/8, legacy loss contracts 16/16, and S1 helper contracts 12/12 pass.

Monitor files were copied byte-identically; manifest and plan hashes were checked against the locked sources. The real-batch target audit and GPU parity numbers are in their Phase-A JSON artifacts.

**Phase-B1 gates: PASS.** Future `--phase train` driver is implemented but was not invoked. **Formal 5000-step training NOT started.**
