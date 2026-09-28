# Anchor-Group V1 Formal 5k Result

## Completion and provenance

- Completed: **5000/5000 registered training steps**.
- Endpoint audit: **PASS**; architecture `LOCUSGS_ANCHOR_GROUP_V1`, joint=true, beta=0.
- Fresh initialization: pretrained reconstruction checkpoint SHA `5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`, step 47500; Anchor-Group seed 31415; global seed 42. The Phase-B1 smoke model/optimizer state was not loaded.
- Manifest SHA `1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483`; plan SHA `ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323`; 128 scenes / 1024 windows, 5000 exact plan entries.
- Hardware/runtime: NVIDIA GeForce RTX 3090; CUDA visible device `0`; PyTorch `2.7.0+cu126`; CUDA `12.6`。
- Recipe: FP32, batch 1, AdamW betas (0.9, 0.95), LR peaks 1e-4/1e-5, differential ratio 10, WD 0.05 for decay groups, grad clip 1.0, all model parameters trainable.
- Evaluations ran at steps 0, 200, 500, 1000, 2000, 3500, 5000 on locked train16/val8/val32, context and target scopes.

## Registered val32 curves

| Step | val32 ctx thing mIoU | val32 target thing mIoU | ctx ca-R50 | target ca-R50 | ctx TP/FP/FN | target TP/FP/FN | ctx/target PSNR | anchor acc | thing-anchor correct | GT recall50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.001328 | 0.001460 | 0.000000 | 0.000000 | 0/0/163 | 0/0/164 | 25.633269 / 24.401157 | 0.148183 | 0.271550 | 0.106383 |
| 200 | 0.002147 | 0.002144 | 0.000000 | 0.000000 | 0/0/163 | 0/0/164 | 25.179514 / 24.044546 | 0.135950 | 0.247904 | 0.100000 |
| 500 | 0.005199 | 0.004899 | 0.000000 | 0.000000 | 0/5/163 | 0/5/164 | 22.800855 / 22.306535 | 0.390857 | 0.150206 | 0.020690 |
| 1000 | 0.022243 | 0.022140 | 0.012270 | 0.012195 | 2/21/161 | 2/21/162 | 20.945787 / 20.771711 | 0.444637 | 0.357197 | 0.157534 |
| 2000 | 0.013503 | 0.014801 | 0.036810 | 0.036585 | 6/93/157 | 6/93/158 | 20.424346 / 20.332696 | 0.456620 | 0.392465 | 0.205882 |
| 3500 | 0.037527 | 0.039006 | 0.092025 | 0.097561 | 15/97/148 | 16/96/148 | 19.885681 / 19.882585 | 0.553924 | 0.480024 | 0.298507 |
| 5000 | 0.060500 | 0.062734 | 0.085890 | 0.091463 | 14/97/149 | 15/96/149 | 19.866162 / 19.863766 | 0.553812 | 0.449063 | 0.285714 |


Step 200 has understanding weight 0 and is a reconstruction-only adaptation checkpoint; its grouping readout is untrained. Step 500 uses weight 0.375 and remains in the ramp. Step 1000 is the first full-joint checkpoint; steps 2000–5000 are the primary structural comparison interval.

## S0/S1/Anchor-Group endpoint comparison

| Model | Scope | Thing mIoU | all-nonempty mIoU | stuff mIoU | ca-R50 | TP/FP/FN | class-aware R50 | active queries | PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0@5k | val32_context | 0.043511 | 0.096301 | 0.518618 | 0.018405 | 3/98/160 | 0.006135 | 10.156250 | 25.633269 |
| S0@5k | val32_target | 0.048011 | 0.101384 | 0.528371 | 0.030488 | 5/97/159 | 0.012195 | 10.156250 | 24.401157 |
| S1@5k | val32_context | 0.035801 | 0.088935 | 0.514007 | 0.036810 | 6/84/157 | 0.012270 | 9.781250 | 25.633269 |
| S1@5k | val32_target | 0.037642 | 0.091562 | 0.522920 | 0.042683 | 7/83/157 | 0.012195 | 9.781250 | 24.401157 |
| Anchor-Group@5k | val32_context | 0.060500 | 0.115005 | 0.551045 | 0.085890 | 14/97/149 | 0.018405 | 8.437500 | 19.866162 |
| Anchor-Group@5k | val32_target | 0.062734 | 0.117652 | 0.556996 | 0.091463 | 15/96/149 | 0.024390 | 8.437500 | 19.863766 |


## Direct anchor-group diagnostics at step 5000

On val32 context: anchor ownership accuracy `0.553812`, thing-anchor correct fraction `0.449063`, and supported-GT recall50 `0.285714` (38/133 supported GTs). Counts: valid/thing/wall/floor/ignore anchors = 24976/15156/5670/4150/7792; GTs with/without support = 133/35.

Final-layer mechanism diagnostics: assignment entropy `0.857837`; thing ownership mass mean/median/p10/p90/max/max-over-median `6.088042/0.031890/0.029512/0.114208/339.321349/22243.392153`; query cosine off-diagonal mean/p90/max `0.846948/0.999641/0.999904`; no-object probability mean/max `0.897169/0.982418`; active queries mean `8.437500`.

## Train16 vs val32 gap at step 5000

| Metric | train16 context | val32 context | val32 minus train16 |
|---|---:|---:|---:|
| mIoU_thing | 0.220523 | 0.060500 | -0.160023 |
| class_agnostic_recall50 | 0.212121 | 0.085890 | -0.126232 |
| class_aware_recall50 | 0.121212 | 0.018405 | -0.102807 |

## Reconstruction tradeoff

Val32 context PSNR: step0 `25.633269`, Anchor-Group step5000 `19.866162`, frozen S1@5k `25.633269`. Val32 target PSNR at step0/step5000: `24.401157` / `19.863766`. Relative to pretrained step0, the context PSNR change is `-5.767107` dB.

The trained model parameter drift from the pretrained reconstruction checkpoint is saved in `formal_reconstruction_drift_audit.json`; category L2 / relative L2 / max absolute deltas:

| Category | L2 delta | Relative L2 delta | Max abs delta |
|---|---:|---:|---:|
| other_reconstruction | 3.481038 | 0.023896 | 0.003811 |
| decoder | 6.613448 | 0.022757 | 0.004670 |
| activation_head | 1.697310 | 0.084831 | 0.013461 |
| anchor_geometry | 0.394271 | 0.006121 | 0.003197 |


## Interpretation

- Direct anchor grouping learned from initialization: val32-context ownership accuracy rose from 0.1482 at step0 to 0.5538 at step5000; thing-anchor correct fraction rose from 0.2716 to 0.4491; supported-GT recall50 rose from 0.1064 to 0.2857. This is clear learning, but remains incomplete: fewer than one third of supported GTs pass the 50% grouping criterion, and the metrics peak at step3500 before a small step5000 decline.
- Relative to frozen S1@5k, Anchor-Group@5k improves val32 context/target ca-R50 from 0.03681/0.04268 to 0.08589/0.09146, and thing mIoU from 0.03580/0.03764 to 0.06050/0.06273. This is a substantial numerical improvement in instance recall and thing semantics, with increased false positives (context 84 to 97; target 83 to 96).
- Thus joint training is better than frozen S1 on the registered val32 thing/instance metrics, while causing a large reconstruction tradeoff: PSNR falls by 5.77 dB on context and 4.54 dB on target relative to step0/frozen S1. Reconstruction degradation is clear and substantial.
- The largest remaining bottlenecks are cross-scene generalization (train16-to-val32 gaps: thing mIoU -0.1600, ca-R50 -0.1262, class-aware R50 -0.1028), incomplete anchor grouping, and concentrated query ownership (thing mass max/median 22,243; mean no-object probability 0.897; 8.44 active queries). On val32, instance false positives also remain high at 97 context / 96 target.
- Direct grouping is assessed by ownership accuracy, thing-anchor correct fraction, and GT recall50 above; these measure anchor-domain grouping using the registered unified Hungarian assignment.
- Val32 instance change versus frozen S1: context ca-R50 `0.049080`; target ca-R50 `0.048780`. Thing mIoU changes: context `0.024699`, target `0.025092`.
- Cross-scene generalization is summarized by the train16/val32 gap table. A train improvement without a val32 improvement indicates a remaining cross-scene generalization bottleneck.
- If anchor-domain grouping scores are strong while 2D ca-R50 remains low, the failure lies downstream in Gaussian rendering, query classification, 2D region readout, or cross-view projection; no architecture change is made here.
- Reconstruction change is quantified by PSNR and parameter drift above; the experiment did not alter the registered recipe in response to intermediate metrics.

All per-window standard evaluator rows, per-scope curve JSONs, direct grouping diagnostic curves, endpoint audit, and parameter drift audit are retained under `group_plus/anchor_group_v1/`. Formal training ended at step 5000. No next experiment was started.
