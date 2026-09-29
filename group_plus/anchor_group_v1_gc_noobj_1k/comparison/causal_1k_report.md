# Unmatched No-Object CE Causal Ablation — Paired 1k

## Completion and provenance

- Architecture LOCUSGS_ANCHOR_GROUP_V1; parent recipe ANCHOR_GROUP_V1_GC_ALPHA001.
- Control unmatched no-object CE scale: 1.0.
- Ablation unmatched no-object CE scale: 0.0.
- Shared understanding-to-reconstruction gradient scale: 0.01.
- Pretrained SHA: 5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f; manifest SHA: 1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483.
- Parent 5000-step plan SHA: ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323; only its first 1000 entries were used.
- Both arms start fresh from the same pretrained reconstruction and seed-31415 Anchor-Group initialization. No V1-GC endpoint or smoke checkpoint was used.

## Registered task and direct grouping curves

| Step | Arm | thing mIoU ctx/tgt | ca-R50 ctx/tgt | class-aware R50 ctx/tgt | ctx TP/FP/FN | target TP/FP/FN | PSNR ctx/tgt | anchor ownership acc | thing-anchor correct | supported-GT recall50 | active queries |
|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | control | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 0/0/163 | 0/0/164 | 25.63327/24.40116 | 0.14818 | 0.27155 | 0.10638 | 100.00 |
| 0 | ablation | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 0/0/163 | 0/0/164 | 25.63327/24.40116 | 0.14818 | 0.27155 | 0.10638 | 100.00 |
| 200 | control | 0.001416/0.001557 | 0.000000/0.000000 | 0.000000/0.000000 | 0/0/163 | 0/0/164 | 25.19149/24.08661 | 0.15250 | 0.27352 | 0.09790 | 100.00 |
| 200 | ablation | 0.001985/0.001924 | 0.000000/0.000000 | 0.000000/0.000000 | 0/0/163 | 0/0/164 | 25.23065/24.09086 | 0.14059 | 0.25445 | 0.10563 | 100.00 |
| 500 | control | 0.006600/0.007236 | 0.000000/0.000000 | 0.000000/0.000000 | 0/12/163 | 0/12/164 | 25.29082/24.20546 | 0.35617 | 0.09381 | 0.02098 | 6.72 |
| 500 | ablation | 0.011804/0.015368 | 0.000000/0.000000 | 0.000000/0.000000 | 0/20/163 | 0/21/164 | 25.00190/23.97436 | 0.38836 | 0.14642 | 0.03597 | 100.00 |
| 1000 | control | 0.020826/0.020040 | 0.018405/0.018293 | 0.018405/0.018293 | 3/33/160 | 3/33/161 | 25.25812/24.07187 | 0.45662 | 0.29479 | 0.15493 | 10.88 |
| 1000 | ablation | 0.012846/0.011220 | 0.000000/0.000000 | 0.000000/0.000000 | 0/25/163 | 0/26/164 | 25.35531/24.17857 | 0.46774 | 0.28111 | 0.08966 | 100.00 |

## Online training positive exposure

| Step | Arm | matches | unique matched | never matched | Top5 | Top10 | Gini | effective queries | first-positive coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | control | 805 | 79 | 21 | 0.28696 | 0.46335 | 0.64891 | 46.237 | 79/100 |
| 200 | ablation | 805 | 82 | 18 | 0.29068 | 0.45963 | 0.64381 | 46.987 | 82/100 |
| 500 | control | 1959 | 92 | 8 | 0.42420 | 0.59214 | 0.70662 | 36.342 | 92/100 |
| 500 | ablation | 1959 | 95 | 5 | 0.42471 | 0.58703 | 0.69619 | 37.338 | 95/100 |
| 1000 | control | 3966 | 96 | 4 | 0.53253 | 0.71407 | 0.78571 | 25.428 | 96/100 |
| 1000 | ablation | 3966 | 98 | 2 | 0.56808 | 0.71457 | 0.76821 | 25.874 | 98/100 |

## Manifest-wide train1024 utilization

| Step | Arm | GT matches | unique | never | Top5 | Top10 | Gini | effective queries | ownership Gini |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | control | 4062 | 95 | 5 | 0.27400 | 0.45716 | 0.63055 | 49.082 | 0.20614 |
| 0 | ablation | 4062 | 95 | 5 | 0.27400 | 0.45716 | 0.63055 | 49.082 | 0.20614 |
| 200 | control | 4062 | 95 | 5 | 0.29321 | 0.45815 | 0.62005 | 49.824 | 0.21637 |
| 200 | ablation | 4062 | 94 | 6 | 0.29788 | 0.45815 | 0.62909 | 48.887 | 0.21247 |
| 500 | control | 4062 | 66 | 34 | 0.73240 | 0.91260 | 0.90531 | 11.731 | 0.94768 |
| 500 | ablation | 4062 | 84 | 16 | 0.71787 | 0.88676 | 0.88340 | 13.499 | 0.94624 |
| 1000 | control | 4062 | 52 | 48 | 0.76588 | 0.90473 | 0.91375 | 10.975 | 0.96470 |
| 1000 | ablation | 4062 | 52 | 48 | 0.71221 | 0.89217 | 0.91027 | 11.749 | 0.96375 |

## Step1000 no-object and maximum thing probability

| Arm | never-matched P(no-object) median | never-matched max thing probability median |
|---|---:|---:|
| control | 0.978626 | 0.008159 |
| ablation | 0.000172 | 0.413236 |

## Reconstruction drift at step1000

| Arm | category | relative L2 drift | max absolute drift |
|---|---|---:|---:|
| control | other_reconstruction | 0.00615098 | 0.00147226 |
| control | decoder | 0.00839991 | 0.00194194 |
| control | activation_head | 0.00939472 | 0.00176312 |
| control | anchor_geometry | 0.00222968 | 0.00122057 |
| ablation | other_reconstruction | 0.00643255 | 0.00178346 |
| ablation | decoder | 0.00844685 | 0.00193078 |
| ablation | activation_head | 0.00932065 | 0.00176123 |
| ablation | anchor_geometry | 0.00226481 | 0.000991134 |

## Direct grouping mechanism diagnostics

| Step | Arm | assignment entropy | thing mass median | mass max | max/median | query cosine mean/p90 | no-object mean |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | control | 4.35288 | 8.54681 | 31.75985 | 3.80 | 0.81806/0.96625 | 0.02232 |
| 0 | ablation | 4.35288 | 8.54681 | 31.75985 | 3.80 | 0.81806/0.96625 | 0.02232 |
| 200 | control | 4.33235 | 8.36774 | 32.68349 | 4.01 | 0.81572/0.96569 | 0.02369 |
| 200 | ablation | 4.34204 | 8.56875 | 32.01416 | 3.85 | 0.80999/0.96509 | 0.02384 |
| 500 | control | 1.44054 | 0.07823 | 178.67391 | 2967.67 | 0.88004/0.99839 | 0.91062 |
| 500 | ablation | 1.35320 | 0.07809 | 175.33347 | 3883.48 | 0.84856/0.99046 | 0.00229 |
| 1000 | control | 1.19221 | 0.01950 | 265.59129 | 18558.97 | 0.85345/0.99977 | 0.87818 |
| 1000 | ablation | 1.17800 | 0.02644 | 251.40471 | 13401.81 | 0.86072/0.99592 | 0.00015 |

## Control sanity against committed V1-GC

Renderer-level outputs can have intrinsic GPU numerical noise; this table reports the measured values without a bit-exact claim.

| Step | Recipe | thing mIoU ctx/tgt | ca-R50 ctx/tgt | class-aware R50 ctx/tgt | PSNR ctx/tgt |
|---:|---|---:|---:|---:|---:|
| 0 | V1-GC | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 25.63327/24.40116 |
| 0 | Control | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 25.63327/24.40116 |
| 0 | Ablation | 0.001328/0.001460 | 0.000000/0.000000 | 0.000000/0.000000 | 25.63327/24.40116 |
| 200 | V1-GC | 0.002233/0.002161 | 0.000000/0.000000 | 0.000000/0.000000 | 25.27231/24.11807 |
| 200 | Control | 0.001416/0.001557 | 0.000000/0.000000 | 0.000000/0.000000 | 25.19149/24.08661 |
| 200 | Ablation | 0.001985/0.001924 | 0.000000/0.000000 | 0.000000/0.000000 | 25.23065/24.09086 |
| 500 | V1-GC | 0.003239/0.002758 | 0.000000/0.000000 | 0.000000/0.000000 | 25.34470/24.22073 |
| 500 | Control | 0.006600/0.007236 | 0.000000/0.000000 | 0.000000/0.000000 | 25.29082/24.20546 |
| 500 | Ablation | 0.011804/0.015368 | 0.000000/0.000000 | 0.000000/0.000000 | 25.00190/23.97436 |
| 1000 | V1-GC | 0.012405/0.012247 | 0.000000/0.000000 | 0.000000/0.000000 | 25.27638/24.09034 |
| 1000 | Control | 0.020826/0.020040 | 0.018405/0.018293 | 0.018405/0.018293 | 25.25812/24.07187 |
| 1000 | Ablation | 0.012846/0.011220 | 0.000000/0.000000 | 0.000000/0.000000 | 25.35531/24.17857 |

## Causal interpretation
- Online step1000 exposure: unique matched queries control/ablation 96/98; effective count 25.428/25.874; Top10 share 0.71407/0.71457.
- Full train1024 checkpoint replay: unique 52/52, never 48/48, effective count 10.975/11.749, Top10 share 0.90473/0.89217.
- Val32 context direct grouping: ownership accuracy 0.45662/0.46774, thing-anchor fraction 0.29479/0.28111, supported-GT recall50 0.15493/0.08966.

Classification: Outcome C. manifest-wide slot utilization did not show the combined improvement required for Outcome A; the observed 1k data do not support unmatched no-object CE as the dominant cause of representation-level dead slots.

No arbitrary success threshold was introduced; raw values and deltas are reported for review. No corrective model or follow-on experiment was implemented.
