# Object-Locus GC Sweep V1

This fixed three-arm experiment changes only the reconstruction-parameter gradient coefficient for the understanding loss:

| Arm | alpha |
|---|---:|
| `gc001` | 0.01 |
| `gc010` | 0.1 |
| `gc100` | 1.0 |

Each arm starts from the same full epoch-6 `LocusGSObjectLocusPanopticV1Recon` checkpoint, loaded strictly with all parameters and buffers. AdamW is newly initialized. Every model parameter remains trainable. No model, loss, feedback, matching, class, or renderer changes are part of this experiment.

For reconstruction parameters the combined gradient is `g_recon + alpha * g_understanding`. For understanding and object parameters it is `g_recon + g_understanding`. Rank gradients are averaged across eight ranks before one global norm clip at 1.0 and one optimizer update.

The source manifest and source training plan come from the saved 128-scene panoptic run. The sweep creates one shared plan using eight permutations `default_rng(42 + epoch).permutation(1008)`, with eight consecutive windows assigned rank 0 through rank 7. Each of 1008 windows is exposed once per epoch for eight epochs: 1008 updates and 8064 new exposures per arm. The source exposure is 50064; the final model exposure is 58128.

The local understanding-loss weight is `min(8*u/200, 1)`, where `u` is the zero-based local optimizer update. The model forward step is `50064 + 8*u`, preserving the source model's feedback beta. LR uses the fixed 25-update linear warm-up and 983-update cosine tail; group peaks are reconstruction `1e-6`, pretrained understanding `1e-5`, and new object modules `1e-4`.

Checkpoint nodes are epochs 0, 2, 4, and 8 (updates 0, 252, 504, and 1008). No training-time evaluation is performed. The epoch-8 model is the registered endpoint; epoch 4 is an intermediate observation only.

The evaluation entry point is deliberately not invoked by training. After explicit user notice that all arms have ended, evaluate the five fixed splits and the context, target-all, and true-novel scopes. Map them to official all/context, all/target, and novel/target exports. The two val32 true-novel alpha comparisons use 2000 paired scene bootstrap resamples with seed 2026 and identical resampled scene indices. No result automatically selects a model or starts more training.

The registered engineering criteria compare each higher alpha against `gc001`: the val32 true-novel mAP difference 95% interval lower bound is above zero; val32 true-novel AP50 and PQ each fall by no more than 0.01; context and true-novel PSNR fall by no more than 0.5 dB on all five splits; and val32 true-novel AbsRel increases by no more than 5%. Bootstrap intervals are exploratory and are not a multiplicity-adjusted significance claim.
