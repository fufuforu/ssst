# Frozen-C generalization: 5k -> 15k continuation

**The 5k->15k extension uses a controlled LR restart (k=global-5000: linear from the checkpoint's
lr_5000=2.000e-06 to 1e-4 over 200 steps, then cosine peak 1e-4 floor 2e-6 to k=10000).  It therefore
tests optimization sufficiency / capacity, NOT a single uninterrupted 15k cosine trajectory.**

## A. Provenance

- commits: `58572cf` (5k pilot), `0058787` (export shape fix), `2dee857` (15k driver + verified plan)
- manifest sha256: `1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483`
- plan5k sha256: `ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323`
- plan15k sha256: `665858b4cc9b6d64ce9164e77c1d14ebeb05989f752fc4124cfcfb203d615569` (prefix 5000 entry-identical, n_mismatch=0)
- endpoint: `workspace_group_plus/instance_state_v1_generalization/arm_C_frozen/endpoint_step15000/`

## B. SLURM completion audit

`slurm_15k_final_audit.json`: job 56136 step `56136.0 COMPLETED` (exit 0:0); the log reached
`[gen] step 15000`, ran EVAL 15000 and printed `continuation finished in 4077s`; no python child and
no GPU compute process remained; the endpoint carried its COMPLETE marker.  The outer bash was a
stale srun wrapper and was cancelled.  `squeue -u mawb` is empty.

## C. Frozen integrity

| checkpoint | compared tensors | changed | bitwise |
|---|---|---|---|
| step0 | 450 | [] | True |
| step1000 | 450 | [] | True |
| step5000 | 450 | [] | True |
| step10000 | 450 | [] | True |
| step15000 | 450 | [] | True |

Chain check against the pretrained reconstruction checkpoint: `frozen_bitwise_equal_pretrained = True` and against step5000: `True`.  So pretrained -> step5000 -> step10000 -> step15000 is bitwise unchanged for all 450 non-`instance_state.*` tensors.

## D. Optimization

- resumed global step 5000 (endpoint step5000), continued 5001 -> 15000 = 10000 updates
- optimizer groups after resume: ['instance_state_decay', 'instance_state_nodecay'] (no backbone group)
- optimizer state entries: 53; RNG keys: ['cuda', 'numpy', 'python', 'torch']
- LR trajectory (logged): 5500 9.98e-05, 6000 9.84e-05, 10000 ~3.7e-05 ... 14000 4.50e-06, 15000 2.00e-06
- total updates 15000 over 1024 distinct windows = ~14.65 passes

## E. Semantic curves (all 11 evaluation steps)

| step | tr16ctx ovr | tr16ctx thing | v8ctx ovr | v8ctx thing | v32ctx ovr | v32ctx thing | v32tgt ovr | v32tgt thing | v32tgt stuff |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.015 | 0.017 | 0.004 | 0.004 | 0.004 | 0.005 | 0.005 | 0.005 | 0.000 |
| 200 | 0.043 | 0.010 | 0.066 | 0.025 | 0.044 | 0.014 | 0.044 | 0.013 | 0.273 |
| 500 | 0.058 | 0.018 | 0.079 | 0.024 | 0.044 | 0.009 | 0.044 | 0.009 | 0.310 |
| 1000 | 0.054 | 0.006 | 0.098 | 0.028 | 0.062 | 0.014 | 0.063 | 0.014 | 0.431 |
| 2000 | 0.119 | 0.072 | 0.107 | 0.031 | 0.064 | 0.016 | 0.067 | 0.017 | 0.461 |
| 3500 | 0.145 | 0.097 | 0.143 | 0.049 | 0.094 | 0.042 | 0.100 | 0.047 | 0.522 |
| 5000 | 0.144 | 0.094 | 0.142 | 0.049 | 0.096 | 0.044 | 0.101 | 0.048 | 0.528 |
| 7500 | 0.137 | 0.083 | 0.146 | 0.058 | 0.083 | 0.027 | 0.088 | 0.031 | 0.538 |
| 10000 | 0.170 | 0.121 | 0.160 | 0.081 | 0.090 | 0.038 | 0.092 | 0.038 | 0.523 |
| 12500 | 0.215 | 0.160 | 0.127 | 0.042 | 0.095 | 0.042 | 0.102 | 0.047 | 0.538 |
| 15000 | 0.257 | 0.206 | 0.154 | 0.081 | 0.105 | 0.051 | 0.109 | 0.054 | 0.547 |

## F. Instance curves (val32 context)

| step | GT | active q | class-aware TP/FP/FN | cw R50 | class-agnostic TP/FP/FN | ca R50 | raw R50 | local PQ |
|---|---|---|---|---|---|---|---|---|
| 5000 | 163 | 10 | 1/100/162 | 0.006 | 3/98/160 | 0.018 | 0.018 | 0.021 |
| 7500 | 163 | 11 | 4/74/159 | 0.025 | 9/69/154 | 0.055 | 0.055 | 0.043 |
| 10000 | 163 | 11 | 2/113/161 | 0.012 | 8/107/155 | 0.049 | 0.049 | 0.043 |
| 12500 | 163 | 9 | 1/109/162 | 0.006 | 7/103/156 | 0.043 | 0.043 | 0.040 |
| 15000 | 163 | 9 | 0/110/163 | 0.000 | 7/103/156 | 0.043 | 0.043 | 0.054 |

### val32 target

| step | GT | active q | class-aware TP/FP/FN | cw R50 | class-agnostic TP/FP/FN | ca R50 | raw R50 | local PQ |
|---|---|---|---|---|---|---|---|---|
| 5000 | 164 | 10 | 2/100/162 | 0.012 | 5/97/159 | 0.030 | 0.030 | 0.030 |
| 7500 | 164 | 11 | 4/74/160 | 0.024 | 9/69/155 | 0.055 | 0.055 | 0.044 |
| 10000 | 164 | 11 | 2/114/162 | 0.012 | 7/109/157 | 0.043 | 0.043 | 0.042 |
| 12500 | 164 | 9 | 1/110/163 | 0.006 | 6/105/158 | 0.037 | 0.037 | 0.041 |
| 15000 | 164 | 9 | 0/110/164 | 0.000 | 8/102/156 | 0.049 | 0.049 | 0.056 |

## G. State diagnostics (5k / 10k / 15k)

### step 5000
- val8: entropy 0.9496, active thing states 100, low-mass 0, mass mean 8.3065 median 0.0194 p90 0.3637 max 517.6172
- val8 q cosine offdiag mean 0.7412 p90 0.9743 max 0.9993; no-object mean 0.8775 max 0.9891
- train16: entropy 1.1091, active 100, low-mass 0, mass max 292.3685; q cosine max 0.9980

### step 10000
- val8: entropy 1.3404, active thing states 100, low-mass 0, mass mean 6.9560 median 0.1085 p90 3.4786 max 144.0298
- val8 q cosine offdiag mean 0.7054 p90 0.9846 max 0.9995; no-object mean 0.8540 max 0.9936
- train16: entropy 0.9912, active 100, low-mass 0, mass max 258.8715; q cosine max 0.9995

### step 15000
- val8: entropy 1.4017, active thing states 100, low-mass 0, mass mean 7.3353 median 0.1565 p90 1.2341 max 212.5272
- val8 q cosine offdiag mean 0.7465 p90 0.9835 max 0.9995; no-object mean 0.8677 max 0.9955
- train16: entropy 0.7691, active 100, low-mass 0, mass max 252.0938; q cosine max 0.9995

## H. Reconstruction (frozen => must not move)

| step | val8 ctx PSNR | val32 ctx PSNR | val32 target PSNR |
|---|---|---|---|
| 0 | 25.174 | 25.633 | 24.401 |
| 5000 | 25.174 | 25.633 | 24.401 |
| 10000 | 25.174 | 25.633 | 24.401 |
| 15000 | 25.174 | 25.633 | 24.401 |

Identical to three decimals at every evaluated step; combined with the bitwise frozen check this
means the reconstruction path is untouched by the state controller.

## I. Qualitative

`group_plus/instance_state_v1_generalization/qualitative/` contains 60 PNGs covering
train_idx 0/45/90/127 and val_idx 0/5/10/15/20/25/30/31 at steps 0/1000/5000/10000/15000
(RGB | GT semantic | pred semantic | GT instance | pred instance | reconstruction, context and novel rows).

## J. 5k -> 15k incremental analysis (is 5k merely undertrained?)

- train16 context **thing mIoU**: 5k 0.094 -> 10k 0.121 -> 15k 0.206 (delta 5k->15k **+0.112**)
- val32 context **thing mIoU**: 5k 0.044 -> 15k 0.051 (delta **+0.008**); val8 thing 0.049 -> 0.081 (**+0.033**)
- val32 context **class-agnostic recall50**: 5k 0.018 -> 7500 0.055 -> 15k 0.043 (delta **+0.025**)
- val32 context TP/FP/FN: 5k 3/98/160 -> 15k 7/103/156 (of 163 GT instances)
- train semantic is still climbing at 15k while val32 semantic gain is modest and instance recall plateaued
  after step 7500 - semantic and instance are clearly **out of sync**.

## K. Final Case

**Case C** - semantic evidence can be read by the current state system, but object-level
grouping / state formation is the bottleneck.

Supporting measurements: val32 context thing mIoU 0.044 -> 0.051 and val8 context thing mIoU
0.049 -> 0.081 between 5k and 15k (semantic improves), while val32 context class-agnostic recall50
only reaches 0.043 and plateaus after 7500, with **TP 7-9 of 163 GT instances and FP 69-113** - the
system fires few, badly-placed object groups rather than recovering the objects it can already
classify.  train16 thing mIoU keeps rising to 0.206, so this is not a capacity wall on the fitting
side; it is the cross-scene object-level state that fails.

## L. Next step (one action only)

Test **local-3D-evidence initialisation**: replace the layer-6 single-anchor FPS seed of each thing
state by a seed formed from local 3D neighbourhood evidence pooling, keeping everything else fixed.
Not implemented in this round.

