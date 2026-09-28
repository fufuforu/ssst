# Anchor-Group V1 Endpoint Failure Diagnostics

Read-only inference/Hungarian replay and backward-only gradient measurement. No optimizer or model update was performed.

## Query utilization / starvation

| metric | train1024 | val32 |
|---|---:|---:|
| total GT matches | 4062 | 168 |
| unique matched queries | 56 | 22 |
| never matched queries | 44 | 78 |
| top1 share | 0.176022 | 0.142857 |
| top5 share | 0.761201 | 0.654762 |
| top10 share | 0.940670 | 0.869048 |
| top20 share | 0.983259 | 0.988095 |
| match Gini | 0.918050 | 0.901310 |
| normalized entropy | 0.510353 | 0.556841 |
| effective query count | 10.488314 | 12.992158 |
| ownership Gini | 0.965024 | 0.961480 |
| active output queries (mean) | 7.438 | 8.438 |

### Top 20 matched queries

| scope | query | matches | unique scenes | scene match fraction |
|---|---:|---:|---:|---:|
| train1024 | 13 | 715 | 126 | 0.9844 |
| train1024 | 66 | 685 | 124 | 0.9688 |
| train1024 | 52 | 625 | 121 | 0.9453 |
| train1024 | 12 | 609 | 124 | 0.9688 |
| train1024 | 35 | 458 | 117 | 0.9141 |
| train1024 | 34 | 228 | 84 | 0.6562 |
| train1024 | 60 | 212 | 77 | 0.6016 |
| train1024 | 84 | 140 | 65 | 0.5078 |
| train1024 | 4 | 80 | 42 | 0.3281 |
| train1024 | 68 | 69 | 45 | 0.3516 |
| train1024 | 22 | 53 | 31 | 0.2422 |
| train1024 | 49 | 29 | 21 | 0.1641 |
| train1024 | 27 | 21 | 15 | 0.1172 |
| train1024 | 46 | 17 | 15 | 0.1172 |
| train1024 | 40 | 13 | 12 | 0.0938 |
| train1024 | 33 | 11 | 10 | 0.0781 |
| train1024 | 48 | 11 | 11 | 0.0859 |
| train1024 | 88 | 8 | 8 | 0.0625 |
| train1024 | 36 | 5 | 3 | 0.0234 |
| train1024 | 56 | 5 | 5 | 0.0391 |
| val32 | 52 | 24 | 24 | 0.7500 |
| val32 | 66 | 24 | 24 | 0.7500 |
| val32 | 13 | 23 | 23 | 0.7188 |
| val32 | 12 | 20 | 20 | 0.6250 |
| val32 | 35 | 19 | 19 | 0.5938 |
| val32 | 60 | 11 | 11 | 0.3438 |
| val32 | 34 | 9 | 9 | 0.2812 |
| val32 | 4 | 6 | 6 | 0.1875 |
| val32 | 84 | 6 | 6 | 0.1875 |
| val32 | 49 | 4 | 4 | 0.1250 |
| val32 | 68 | 4 | 4 | 0.1250 |
| val32 | 22 | 3 | 3 | 0.0938 |
| val32 | 48 | 3 | 3 | 0.0938 |
| val32 | 27 | 2 | 2 | 0.0625 |
| val32 | 33 | 2 | 2 | 0.0625 |
| val32 | 46 | 2 | 2 | 0.0625 |
| val32 | 32 | 1 | 1 | 0.0312 |
| val32 | 38 | 1 | 1 | 0.0312 |
| val32 | 41 | 1 | 1 | 0.0312 |
| val32 | 45 | 1 | 1 | 0.0312 |

train1024 Spearman: match count vs no-object probability rho=-0.6509925867320872, p=2.261161753873948e-13; match count vs ownership mass rho=0.17336427731331502, p=0.08453940882942898.

val32 Spearman: match count vs no-object probability rho=-0.5573184608488335, p=1.7176083609140447e-09; match count vs ownership mass rho=0.45052461880414096, p=2.558779614019075e-06.

## Gradient conflict

Cosine < 0 means gradient conflict on the measured batch/category; cosine > 0 means locally aligned. Gradient scale is the measured `||g_under|| / ||g_recon||`, not the loss scalar ratio.

| model | category | median cos | frac cos < 0 | median ||gU||/||gR|| | p90 ratio |
|---|---|---:|---:|---:|---:|
| fresh_step0 | all_reconstruction | 0.003001 | 0.4375 | 34.6928 | 50.4199 |
| fresh_step0 | encoder | -0.001963 | 0.5000 | 205.791 | 316.835 |
| fresh_step0 | decoder | 0.005893 | 0.2500 | 591.154 | 994.752 |
| fresh_step0 | anchor_geometry | 0.030257 | 0.3750 | 7.66547 | 9.08413 |
| fresh_step0 | activation_head | -0.031901 | 0.5000 | 1.49667 | 3.45254 |
| fresh_step0 | other_reconstruction | -0.063862 | 0.6250 | 66.7741 | 140.493 |
| endpoint_step5000 | all_reconstruction | 0.025989 | 0.4375 | 21.9282 | 59.6645 |
| endpoint_step5000 | encoder | 0.016051 | 0.4375 | 46.0266 | 124.645 |
| endpoint_step5000 | decoder | 0.001505 | 0.4375 | 93.8748 | 241.94 |
| endpoint_step5000 | anchor_geometry | 0.025615 | 0.5000 | 18.1792 | 45.9593 |
| endpoint_step5000 | activation_head | 0.090186 | 0.3125 | 8.76371 | 15.4718 |
| endpoint_step5000 | other_reconstruction | 0.046423 | 0.4375 | 32.3625 | 118.582 |

### Fresh versus endpoint

| category | step0 cosine median | endpoint cosine median | step0 norm ratio median | endpoint norm ratio median |
|---|---:|---:|---:|---:|
| all_reconstruction | 0.003001 | 0.025989 | 34.6928 | 21.9282 |
| encoder | -0.001963 | 0.016051 | 205.791 | 46.0266 |
| decoder | 0.005893 | 0.001505 | 591.154 | 93.8748 |
| anchor_geometry | 0.030257 | 0.025615 | 7.66547 | 18.1792 |
| activation_head | -0.031901 | 0.090186 | 1.49667 | 8.76371 |
| other_reconstruction | -0.063862 | 0.046423 | 66.7741 | 32.3625 |

### Activation head and focused shared paths

| model | module | median cosine | fraction cosine < 0 | median norm ratio | p90 norm ratio |
|---|---|---:|---:|---:|---:|
| fresh_step0 | decoder_block_11 | -0.012660 | 0.5625 | 352.606 | 599.683 |
| fresh_step0 | anchor_decoder.mu | -0.018611 | 0.6250 | 25.4355 | 54.9313 |
| fresh_step0 | anchor_decoder.rho | -0.027330 | 0.6875 | 34.0134 | 71.076 |
| fresh_step0 | activation_head | -0.031901 | 0.5000 | 1.49667 | 3.45254 |
| endpoint_step5000 | decoder_block_11 | -0.020148 | 0.5625 | 80.806 | 219.094 |
| endpoint_step5000 | anchor_decoder.mu | -0.023282 | 0.5625 | 61.3072 | 128.246 |
| endpoint_step5000 | anchor_decoder.rho | -0.005699 | 0.5625 | 44.4795 | 97.9894 |
| endpoint_step5000 | activation_head | 0.090186 | 0.3125 | 8.76371 | 15.4718 |

## Factual interpretation

The measurements above describe these locked windows and this endpoint only. Positive-match concentration and never/rarely matched slots are evidence consistent with query starvation/slot collapse when concentrated; they are not a theoretical proof.

Gradient comparisons are local per-window measurements in eval mode with autograd enabled. Negative cosine denotes conflict on that measured batch/category; positive cosine denotes local alignment. No structural or optimization recommendation is made here.

### Measured answers

- Train positive supervision reached 56/100 queries; 44/100 were never matched. Val32 reached 22/100; 78/100 were never matched.
- Train top-5/top-10 took 76.1%/94.1% of matches; effective matched-query count was 10.49. Val32 values were 65.5%/86.9% and 12.99.
- Ownership mass was also concentrated (Gini 0.9650 train, 0.9615 val32). Match count versus no-object probability was Spearman rho -0.6510 (p=2.26e-13) train and -0.5573 (p=1.72e-09) val32. Match count versus ownership mass was rho 0.1734 (p=0.0845) train and 0.4505 (p=2.56e-06) val32.
- Median understanding/reconstruction gradient norm ratio for all shared reconstruction parameters was 34.7 at step0 and 21.9 at step5000. The measured understanding gradient norm was larger than the reconstruction gradient norm on these aggregate windows.
- Decoder had the largest category median scale ratio (591 fresh, 93.9 endpoint). Activation-head ratio changed from 1.5 to 8.76; its negative-cosine fraction changed from 50.0% to 31.2%.
- Direction conflict was already present on fresh windows: negative-cosine fractions were 43.8% overall, 25.0% decoder, 50.0% activation head, and 37.5% anchor geometry. Endpoint fractions were 43.8%, 43.8%, 31.2%, and 50.0%, respectively.
