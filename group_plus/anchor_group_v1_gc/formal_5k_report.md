# Anchor-Group V1-GC formal 5k report

## A. Completion and provenance

Training completed 5000/5000 steps on `NVIDIA GeForce RTX 3090`; FP32. Fresh initialization used pretrained step 47500 and a seed-31415 Anchor-Group module. Manifest SHA `1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483`; plan SHA `ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323`; pretrained SHA `5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`.

## B. Recipe

Architecture, forward, losses, optimizer, LR, warm-up, and clipping match Anchor-Group V1. The sole scientific difference is understanding-to-reconstruction gradient scale `1.0 → 0.01`; Anchor-Group parameters retain full understanding gradients.

## C. Registered seven-point V1-GC curves

| Step | ctx/target thing mIoU | ctx/target ca-R50 | ctx/target class-aware R50 | ctx TP/FP/FN | target TP/FP/FN | ctx/target PSNR | anchor acc | thing-anchor correct | supported-GT recall50 | active queries | assignment entropy | mass median/max/max:median | q cosine mean | no-object mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 0/0/163 | 0/0/164 | 25.633269/24.401157 | 0.148183 | 0.271550 | 0.106383 | 100.000 | 4.352878 | 8.54681/31.7598/3.79505 | 0.818061 | 0.022323 |
| 200 | 0.002233/0.002161 | 0.000000/0.000000 | 0.000000/0.000000 | 0/0/163 | 0/0/164 | 25.272311/24.118074 | 0.143178 | 0.260755 | 0.085106 | 100.000 | 4.347607 | 8.61166/31.064/3.70124 | 0.816121 | 0.024259 |
| 500 | 0.003239/0.002758 | 0.000000/0.000000 | 0.000000/0.000000 | 0/3/163 | 0/4/164 | 25.344698/24.220729 | 0.398314 | 0.112817 | 0.007143 | 8.781 | 1.260549 | 0.018409/190.158/13337.9 | 0.856468 | 0.888716 |
| 1000 | 0.012405/0.012247 | 0.000000/0.000000 | 0.000000/0.000000 | 0/33/163 | 0/33/164 | 25.276380/24.090342 | 0.470593 | 0.292662 | 0.112676 | 9.469 | 1.120162 | 0.0334019/240.511/8554.92 | 0.847913 | 0.879966 |
| 2000 | 0.021173/0.023239 | 0.036810/0.048780 | 0.006135/0.006098 | 6/71/157 | 8/69/156 | 25.251828/24.193879 | 0.510975 | 0.356243 | 0.145833 | 7.719 | 0.816384 | 0.0103484/284.348/41494.7 | 0.852125 | 0.916381 |
| 3500 | 0.026853/0.028477 | 0.061350/0.060976 | 0.012270/0.012195 | 10/77/153 | 10/77/154 | 25.610690/24.338872 | 0.536406 | 0.440886 | 0.243056 | 7.938 | 0.923229 | 0.0168791/311.748/33738.3 | 0.850037 | 0.912595 |
| 5000 | 0.037604/0.040334 | 0.061350/0.048780 | 0.024540/0.024390 | 10/79/153 | 8/82/156 | 25.717188/24.397832 | 0.542765 | 0.415947 | 0.220690 | 8.531 | 0.868965 | 0.0264087/299.069/28081.7 | 0.853376 | 0.903388 |

## D. Same-step V1 vs V1-GC

| Step | Model | ctx/target thing mIoU | ctx/target ca-R50 | ctx/target class-aware R50 | ctx/target PSNR | ctx/target anchor accuracy | ctx/target supported-GT recall50 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | V1 | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 25.633269/24.401157 | 0.148183/0.148183 | 0.106383/0.106383 |
| 0 | V1-GC | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 25.633269/24.401157 | 0.148183/0.148183 | 0.106383/0.106383 |
| 200 | V1 | 0.002147/0.002144 | 0.000000/0.000000 | 0.000000/0.000000 | 25.179514/24.044546 | 0.135950/0.135950 | 0.100000/0.100000 |
| 200 | V1-GC | 0.002233/0.002161 | 0.000000/0.000000 | 0.000000/0.000000 | 25.272311/24.118074 | 0.143178/0.143178 | 0.085106/0.085106 |
| 500 | V1 | 0.005199/0.004899 | 0.000000/0.000000 | 0.000000/0.000000 | 22.800855/22.306535 | 0.390857/0.390857 | 0.020690/0.020690 |
| 500 | V1-GC | 0.003239/0.002758 | 0.000000/0.000000 | 0.000000/0.000000 | 25.344698/24.220729 | 0.398314/0.398314 | 0.007143/0.007143 |
| 1000 | V1 | 0.022243/0.022140 | 0.012270/0.012195 | 0.006135/0.006098 | 20.945787/20.771711 | 0.444637/0.444637 | 0.157534/0.157534 |
| 1000 | V1-GC | 0.012405/0.012247 | 0.000000/0.000000 | 0.000000/0.000000 | 25.276380/24.090342 | 0.470593/0.470593 | 0.112676/0.112676 |
| 2000 | V1 | 0.013503/0.014801 | 0.036810/0.036585 | 0.006135/0.006098 | 20.424346/20.332696 | 0.456620/0.456620 | 0.205882/0.205882 |
| 2000 | V1-GC | 0.021173/0.023239 | 0.036810/0.048780 | 0.006135/0.006098 | 25.251828/24.193879 | 0.510975/0.510975 | 0.145833/0.145833 |
| 3500 | V1 | 0.037527/0.039006 | 0.092025/0.097561 | 0.006135/0.006098 | 19.885681/19.882585 | 0.553924/0.553924 | 0.298507/0.298507 |
| 3500 | V1-GC | 0.026853/0.028477 | 0.061350/0.060976 | 0.012270/0.012195 | 25.610690/24.338872 | 0.536406/0.536406 | 0.243056/0.243056 |
| 5000 | V1 | 0.060500/0.062734 | 0.085890/0.091463 | 0.018405/0.024390 | 19.866162/19.863766 | 0.553812/0.553812 | 0.285714/0.285714 |
| 5000 | V1-GC | 0.037604/0.040334 | 0.061350/0.048780 | 0.024540/0.024390 | 25.717188/24.397832 | 0.542765/0.542765 | 0.220690/0.220690 |

## E. S1 / V1 / V1-GC endpoints

| Model | Scope | thing mIoU | ca-R50 | class-aware R50 | PSNR | active queries |
|---|---|---:|---:|---:|---:|---:|
| S1@5k | val32_context | 0.035801 | 0.036810 | 0.012270 | 25.633269 | 9.78125 |
| S1@5k | val32_target | 0.037642 | 0.042683 | 0.012195 | 24.401157 | 9.78125 |
| V1@5k | val32_context | 0.060500 | 0.085890 | 0.018405 | 19.866162 | 8.4375 |
| V1@5k | val32_target | 0.062734 | 0.091463 | 0.024390 | 19.863766 | 8.4375 |
| V1-GC@5k | val32_context | 0.037604 | 0.061350 | 0.024540 | 25.717188 | 8.53125 |
| V1-GC@5k | val32_target | 0.040334 | 0.048780 | 0.024390 | 24.397832 | 8.53125 |

## F. Direct anchor grouping

The seven-point table reports anchor ownership accuracy, confident thing-anchor correct fraction, and supported-GT recall50 using the registered evaluator/Hungarian path.

## G. Reconstruction preservation and H. Parameter drift

Val32 context PSNR changed by 0.083919 dB (step 0 to 5000); target changed by -0.003325 dB. V1 context/target changes were -5.767107/-4.537390 dB.

| Category | V1 relative L2 drift | V1-GC relative L2 drift | V1 max abs | V1-GC max abs |
|---|---:|---:|---:|---:|
| decoder | 0.0227567 | 0.0158272 | 0.00467001 | 0.00442863 |
| activation_head | 0.0848306 | 0.0260237 | 0.0134609 | 0.00546032 |
| anchor_geometry | 0.00612121 | 0.00344233 | 0.00319658 | 0.00247952 |
| other_reconstruction | 0.0238957 | 0.0129757 | 0.00381149 | 0.00347259 |

## I. Query utilization / starvation diagnostic

| Scope | Model | matched / never | top5 / top10 share | match Gini | effective query count | ownership Gini | active queries |
|---|---|---:|---:|---:|---:|---:|---:|
| train1024 | V1 | 56/100 ; 44/100 | 0.7612/0.9407 | 0.9181 | 10.488 | 0.9650 | 7.438 |
| train1024 | V1-GC | 58/100 ; 42/100 | 0.7777/0.9441 | 0.9227 | 9.998 | 0.9643 | 7.453 |
| val32 | V1 | 22/100 ; 78/100 | 0.6548/0.8690 | 0.9013 | 12.992 | 0.9615 | 8.438 |
| val32 | V1-GC | 21/100 ; 79/100 | 0.6488/0.8810 | 0.9052 | 12.570 | 0.9645 | 8.531 |

## J. Training clip diagnostics

| Run | median pre-clip norm | p10 | p90 | max | median clip coefficient | fraction clipped |
|---|---:|---:|---:|---:|---:|---:|
| V1 | not available for V1 formal run | not available | not available | not available | not available | not available |
| V1-GC | 5.88349 | 3.61443 | 9.80509 | 18.9492 | 0.169975 | 0.9800 |

## K. Train16 to val32 generalization at step 5000

| Metric | train16 context | val32 context | val32 minus train16 |
|---|---:|---:|---:|
| mIoU_thing | 0.125658 | 0.037604 | -0.088054 |
| class_agnostic_recall50 | 0.121212 | 0.061350 | -0.059862 |
| class_aware_recall50 | 0.075758 | 0.024540 | -0.051218 |

## L. Final interpretation from measured results

- Reconstruction protection: yes by val32 context PSNR; GC change 0.0839 dB vs V1 -5.7671 dB (target GC -0.0033 dB vs V1 -4.5374 dB). Absolute context drop reduction: 5.6832 dB.
- Understanding at val32 context: thing mIoU GC/V1 0.037604/0.060500 (GC below V1); ca-R50 0.061350/0.085890 (GC below V1).
- Versus frozen S1@5k, GC val32 context/target thing mIoU is 0.037604/0.040334 vs 0.035801/0.037642; ca-R50 is 0.061350/0.048780 vs 0.036810/0.042683. GC is above S1 on both reported metrics/scopes, but below V1 on thing mIoU and ca-R50.
- Direct anchor grouping at GC endpoint: ownership accuracy 0.542765, thing-anchor correct fraction 0.415947, supported-GT recall50 0.220690.
- Direct grouping change from step0: ownership accuracy 0.148183 → 0.542765; thing-anchor correct fraction 0.271550 → 0.415947; supported-GT recall50 0.106383 → 0.220690. This shows continued learning in the anchor domain.
- Active query output mean on train1024: GC 7.453; V1 7.438. On val32: GC 8.531; V1 8.438. Match starvation indicators: 58/100 matched and 42/100 never matched; top5 match share 0.7777, match Gini 0.9227, effective count 9.998.
- Relative reconstruction drift was reduced in categories: decoder, activation_head, anchor_geometry, other_reconstruction; decoder/activation-head drift reductions are shown in section H.
- Query match concentration remains severe (42/100 train queries never matched, 77.77% of matches in the top five), so slot starvation remains the clearest measured bottleneck; this run made no query-starvation intervention.

No query-starvation fix was introduced. No alpha sweep was run. No next experiment was started.
