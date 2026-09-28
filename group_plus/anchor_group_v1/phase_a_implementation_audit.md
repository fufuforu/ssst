# Anchor-Group V1 Phase-A implementation audit

Scope: §0–§30, §38–§40 and §41 reconstruction parity only. Phase-B sections and training were not started.

1. **1024 anchors:** all decoder anchors are encoded by `encode_token` and retained. CPU gradient contract measured nonzero finite gradient on all 1024 rows.
2. **FPS/local8:** the new grouping decoder has no FPS/local8 path or dependency; its source independence check passes.
3. **Queries:** 100 thing plus 2 stuff queries (102 total), initialized only from `query_init`.
4. **Void:** one per-anchor `token_void(a)` channel; it is not a query.
5. **Assignment:** `A_pre` and `A_post` are `[B,1024,103]`; simplex maximum error was `3.58e-7`.
6. **Gaussian ownership:** each child repeats its parent's `A_post` by expand/reshape. Exact inheritance contract passes with `torch.equal`.
7. **GT affiliation:** detached final `mu` is projected to the two context views with the established `inverse(cam_view.T)` camera convention and integer `u.long()/v.long()` indexing. Thing IDs are ascending and each must map to one semantic class.
8. **Conflicts:** different thing IDs/classes, thing/stuff, wall/floor, invalid pixels, and no observations resolve to IGNORE. IGNORE is excluded from ownership CE.
9. **Unified Hungarian:** one `linear_sum_assignment` call combines class, deterministic 4096-point pixel BCE/Dice, and valid-anchor BCE/Dice terms with weights `1,5,5,2,2`. Zero-support GT has zero anchor matching terms.
10. **Shared pairs:** class, pixel and anchor matched losses consume the same returned Hungarian pairs; CPU C15 calls the production matcher and passes.
11. **Anchor losses:** ownership CE is mean negative log probability on confident anchors. Anchor Dice uses matched thing GT with positive support and `(2 sum(p*y)+1)/(sum(p)+sum(y)+1)`. `L_anchor_group = CE + Dice`; understanding weights follow the specification.
12. **Reconstruction parity:** same pretrained checkpoint and locked window 0 on an NVIDIA RTX 4090 produced Gaussian max diff **0.0**, RGB max/mean diff **0.0/0.0**, baseline and Anchor-Group PSNR **22.1214867/22.1214867 dB**, and PSNR absolute diff **0.0**. Baseline and Anchor-Group repeat RGB max diffs were both **0.0**; gsplat forward passed.
13. **Legacy S0/S1:** evaluator contracts passed 8/8; legacy loss contracts passed 16/16; S1 local3d helper checks passed 8/8. On the same real window and RTX 4090, S0 and S1 forwards and losses were finite and passed. S0 used the single-anchor initialization; S1 retained local8 with k=8.
14. **CPU contracts:** Anchor-Group contracts passed **18/18**, including C1–C18. The shared-init comparison covered 59 parameter tensors with zero mismatches; initial void logit max abs was 0.0. Pairwise anchor BCE matched brute-force BCEWithLogits to max abs `3.58e-7`; extreme logits remained finite.

## Real batch target audit

Locked manifest window 0 (`scene0000_00`, context frames 3551 and 3596): 1024 anchors; 211 thing, 236 wall, 431 floor, 146 IGNORE. There were 5 GT thing instances: 4 with anchor support and 1 without. Per-GT counts: `[3,157,44,0,7]`; min 0, median 7, p90 157, max 157. No data was filtered.

## Gate status

All Phase-A contracts and GPU verification gates pass. The Phase-A.1 fixes only align shared controller initialization and stabilize the anchor matching BCE implementation; architecture and loss weights are unchanged. **Phase-A complete.** Phase-B was not started; formal training was not started.
