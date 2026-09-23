# Old TokenGS ScanNet checkpoint: positive control + warm-start A/B

## 1. Positive control (read-only)

**Checkpoint.** `scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors`
is **not** a partial backbone.  It holds 395 tensors / 218,871,040 parameters:

| group | tensors | params |
|---|---|---|
| `enc_dec_backbone` (encoder + all 12 decoder blocks) | 384 | 216,308,096 |
| `gs_tokens` | 1 | 1,048,576 |
| `activation_head.deconv` (Gaussian head) | 2 | 918,400 |
| `patch_plucker_embed` | 4 | 396,288 |
| `patch_embed` | 4 | 199,680 |

It loads into this repo's `PlainTokenGSCanonicalRecon` **exactly** — matched
395/395 tensors, 218,871,040/218,871,040 params = 100.00 %, missing/unexpected/
mismatched all 0 — for both `num_input_views=2` and `num_input_views=8`.

**Re-evaluation.** `scripts/eval_old_tokengs_scannet_8x7.py` re-renders the
manifest first validation window per scene with the original 8-context + 7-novel
protocol on **raw .sens** frames:

| scene | ctx PSNR | target PSNR | target grey | target SSIM | alpha |
|---|---|---|---|---|---|
| scene0072_02 | 17.12 | 16.60 | 10.68 | 0.577 | 0.949 |
| scene0615_00 | 17.11 | 16.81 | 10.06 | 0.662 | 0.955 |
| scene0568_02 | 25.91 | 25.68 | 9.94 | 0.757 | 0.990 |
| scene0059_00 | 19.26 | 19.17 | 11.32 | 0.654 | 0.971 |
| mean | 19.85 | 19.57 | | 0.663 | |

The renders carry **correct structure** (sofa, table, cabinet and backpack in the
right places, blurry but geometrically right) — see
`workspace_recon_diag/old_tokengs_positive_control/images/*_target_gt_vs_pred.png`.
This reproduces the reported ~19-20.5 dB.  The historical 20.12 number was
flagged in the tokengs handoff as coming from a broken eval passthrough; the
number reproduced here is independent of that path.

## 2. Warm-start A/B (one seed per arm)

Both arms load the same checkpoint strictly (100 % match reported above), use the
same four frames `[654, 664, 655, 659]` and the same plan (4000 steps, warmup
1000, cosine to 2 %, lr 1e-4, bf16, canonical loss).

| run | step 1 (init) | best | step 200 | final | grey |
|---|---|---|---|---|---|
| 2+2 processed, lr 1e-4 | 22.03 | 22.03 | 4.71 | 4.55 | 9.03 |
| 2+2 raw, lr 1e-4 | 22.48 | 22.48 | 4.17 | 5.27 | 9.03 |
| 2+2 raw, lr 1e-5 | 22.49 | 22.49 | 16.03 | 4.96 | 9.03 |
| 8+7 raw, lr 1e-4 | 19.26 | 19.26 | 4.67 | 4.21 | 11.19 |

**Both arms start with correct structure and near-identical quality** (22.03
processed vs 22.48 raw), and the 8+7 step-1 value reproduces the independent
positive-control number for `scene0059_00` exactly (ctx 19.26 / target 19.17).
So the batch, K, camera and ray handling of *both* data paths is correct and
usable at inference.

**Training then destroys it in every configuration.**  The canonical loss rises
from 0.027 at step 1 to ~0.42 while PSNR falls to ~4-5 dB and alpha saturates to
~1.0: the optimizer diverges away from the converged solution instead of
converging (compare `raw8x7/images/raw_step100.png` with `raw_step4000.png`).

## 3. Localisation of the 8+7 to 2+2 difference

Requested localisation: is the difference the input views or the training
targets?

* 8-context + 7-novel **also collapses** under the same plan (step 200 ctx 4.67),
  so it is not the 2+2 supervision.
* Lowering lr to 1e-5 **delays** the collapse (step 200 ctx 16.03 instead of
  4.17) but does not prevent it, so it is not only the step size.
* Both data sources behave identically, so it is not the data source.

Conclusion: the blocker is the **optimisation step itself** on this objective —
the canonical MSE + 0.2*(1-SSIM)/2 + 1.0*L_vis objective with
`gaussian_z_offset = 1.0` fine-tuning a *converged* model on a fixed sample at
this learning-rate scale.  The successful 8+7 result is a forward-only
evaluation of a model trained over 1425 scenes; a fixed-sample optimisation of
the same objective destroys it.

## 4. Next single-variable to probe (optimisation, not data)

Keep the 2+2 protocol and both data sources frozen.  Vary one at a time:
`gaussian_z_offset` (1.0 vs 0.0), the Gaussian-visibility weight (1.0 vs 0.0),
freezing the Gaussian head / GS tokens, and a much smaller effective step
(e.g. lr 1e-6 with no warmup).  The question is which term pushes the converged
model off its solution.
