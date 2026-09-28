# S1 local-3D initialization: frozen-C 5k report

## A. Provenance
- baseline `29d0c80`; S1 harness fix `ae7183f`; this results commit.
- manifest `1f37d08c...` (identical to S0, asserted at start-up); plan = S0 `plan_C_frozen_5000` (sha matches).
- endpoint `workspace_group_plus/instance_state_v2_s1_local3d/arm_C/endpoint`.

## B. Completion / SLURM audit
`slurm_final_audit.json`: job 56187 step COMPLETED (0:0); log `[s1] step 5000 ... EVAL 5000 ... finished 5000 steps in 2603s`; outer bash stale -> cancelled.

## C. S0/S1 experiment identity
`s0_s1_experiment_identity.json`: manifest sha identical, monitors byte-identical, same plan/seed/optimizer/LR/loss/evaluator/frozen backbone, C-only beta=0.  Only structural variable = q_thing_init.

## D. Initialization contract
12/12 green (`s1_initialization_contract_corrected.json`): true pre-update fps_index/c_init/s_init/q_stuff_init/anchor_mu_init all torch.equal; only q_thing_init differs (2.44); param count 222,787,916 both; projection parity 0.00e+00.

## E. Coverage (occurrence-weighted, 3965 occurrences / 1398 unique scene-instances)
overall FPS 0.860 / local8 0.918; small(346) FPS 0.176 / local8 0.379; medium(1420) 0.812/0.923; large(2199) 0.998/1.000.  Macro FPS 0.839 / local8 0.909.

## F. Training curves
S1 eval at 0/200/500/1000/2000/3500/5000 on train16/val8/val32 x context/target (`curves_*.json`).

## G. Primary comparison S0@5k vs S1@5k
| metric | S0 | S1 | delta |
|---|---|---|---|
| train16 ctx thing mIoU | 0.0936 | 0.0933 | -0.0004 |
| val8 ctx thing mIoU | 0.0486 | 0.0543 | +0.0057 |
| val32 ctx thing mIoU | 0.0435 | 0.0358 | -0.0077 |
| val32 target thing mIoU | 0.0480 | 0.0376 | -0.0104 |
| val32 ctx ca-R50 | 0.0184 | **0.0368** | **+0.0184** |
| val32 target ca-R50 | 0.0305 | **0.0427** | **+0.0122** |
| val32 ctx TP/FP/FN | 3/98/160 | **6/84/157** | TP +3, FP -14, FN -3 |
| val32 target TP/FP/FN | 5/97/159 | **7/83/157** | TP +2, FP -14, FN -2 |
| active thing queries | 10.16 | 9.78 | -0.38 |
| PSNR (ctx/tgt) | 25.633 / 24.401 | 25.633 / 24.401 | 0.000 |

## H. State mechanism
`mechanism_comparison_5k.json` + `init_q_diversity_corrected.json`: pre-update q cosine S0 0.509 -> S1 0.706 (local averaging smooths); post-first-update 0.557 -> 0.719; final-state diagnostics compared in the JSON.

## I. Reconstruction / frozen integrity
`frozen_integrity_step5000.json`: 450 tensors, changed=[], bitwise_unchanged=true, identical to the pretrained reference; PSNR identical to 4 decimals at every eval step.

## J. Qualitative
**Not executed**: S1 panels and the S0|S1 side-by-side were not generated in this session.

## K. SIU3R official-metric definitions on locked val32
**Not executed.**

## L. Verdict
**positive (primary endpoint), with caveats.**  ca-R50 doubles on val32 context (0.0184 -> 0.0368) and rises on target (0.0305 -> 0.0427); TP 3->6 (ctx) and 5->7 (tgt); FP 98->84 / 97->83 and FN 160->157 / 159->157 both decrease.  Thing mIoU is flat on train16 and slightly lower on val32.  Absolute levels remain very low (6 TP of 163 GT).

## M. Next single scientific action
Small-instance proposal coverage is the next bottleneck to test (FPS 0.176 / local8 0.379 for 50-499 px), before any further change to the pooling.
