# From-scratch fixed-sample validation: fp32 vs bf16

Re-run of the previously-failed 2-context + 2-novel fixed-sample experiment
(`scene0048_01`, frames `[654, 664, 655, 659]`, seed 42, `PlainTokenGSCanonicalRecon`,
canonical loss, AdamW decay/no-decay grouping, lr 1e-4, warmup 1000, cosine to
2 %, 4000 steps, same evaluation code).  **The only change is that the training
forward and loss run without bf16 autocast** (`--amp fp32`).  No ScanNet
checkpoint was loaded; the model is the same random initial state as before.

## 1. The two runs are identical except for precision

Init, batch and hyper-parameters verified equal: the same
`plain_ab_shared_init.pt` (md5 `3ca43d8c5d8b06681a78466df1b21c8f`), the same
provider/batch, the same plan (only `--amp` differs).  Step-1 metrics of the
old bf16 run and the new fp32 run agree to 5 decimals:

| arm | ctx PSNR | novel PSNR | loss | loss_rgb | loss_ssim | alpha |
|---|---|---|---|---|---|---|
| processed bf16 / fp32 | 9.03384 / 9.03384 | 9.18231 / 9.18231 | 0.16083 / 0.16083 | 0.12282 / 0.12282 | 0.19004 / 0.19004 | 0.02667 / 0.02667 |
| raw bf16 / fp32 | 9.03252 / 9.03252 | 9.16969 / 9.16969 | 0.15971 / 0.15971 | 0.12301 / 0.12301 | 0.18347 / 0.18347 | 0.02531 / 0.02531 |

## 2. Dtypes actually used (`scripts/probe_dtypes.py`)

Instrumented training step, same model and batch:

| tensor | fp32 mode | bf16 autocast mode |
|---|---|---|
| model parameters | float32 | float32 |
| batch `images_all` / `input` (rgb+Pluecker) | float32 | float32 |
| Gaussian head output | float32 | float32 |
| renderer inputs (gaussians, cam_view, intrinsics) | float32 | float32 |
| renderer outputs (`images_pred`, `alphas_pred`, `means2d_pred`, `depths_pred`) | float32 | float32 |
| loss total / `loss_rgb` / `loss_ssim` / visibility | float32 | float32 |
| parameter gradients | float32 | float32 |
| **internal matmul** | **float32** | **bfloat16** |

So every *interface* tensor - parameters, the Gaussian head's output, the
gsplat renderer's inputs and outputs, and all loss terms - is fp32 in both
modes; both repos' renderers additionally cast the Gaussian parameters to
`.float()` before `rasterization`, so the rasteriser is fp32 either way.
Autocast changes **only the internal matmul/conv precision** of the backbone,
head and SSIM convolutions, and that is enough to change the gradient (see
`pretrained_finetune_degradation.md`: bf16 grad norm 9.61 vs fp32 0.995,
cosine 0.12).

## 3. Trajectories

| step | bf16 processed | fp32 processed | bf16 raw | fp32 raw |
|---|---|---|---|---|
| 1 | 9.03 | 9.03 | 9.03 | 9.03 |
| 250 | 9.04 | **14.80** | 9.04 | **15.41** |
| 500 | 9.21 | 18.68 | 9.26 | 18.94 |
| 1000 | 8.64 | 19.53 | 7.84 | 19.36 |
| 2000 | 6.34 | 21.71 | 5.19 | 21.28 |
| 4000 | 5.20 | **24.64** | 5.69 | **25.79** |

(context PSNR; grey baseline 9.03.)

Final state, all views:

| run | ctx PSNR | novel PSNR | ctx SSIM | novel SSIM | alpha | loss |
|---|---|---|---|---|---|---|
| bf16 processed | 5.20 | 5.33 | 0.417 | 0.428 | 0.999 | 0.3563 |
| **fp32 processed** | **24.64** | **25.95** | 0.844 | 0.869 | 0.922 | 0.0173 |
| bf16 raw | 5.69 | 5.84 | 0.462 | 0.476 | 0.999 | 0.3191 |
| **fp32 raw** | **25.79** | **26.77** | 0.866 | 0.891 | 0.894 | 0.0145 |

bf16 never leaves the grey baseline in either arm and ends ~3.5 dB *below* it
with alpha saturated to 1.0.  fp32 escapes at ~step 250 and improves
monotonically; **novel** views track context views (raw 26.77 vs 25.79), the
alpha coverage stays < 1 (0.89-0.92), and the renders keep the scene structure
in both context and novel views
(`workspace_recon_diag/plain_ab_scratch_fp32/*/images/*_step4000.png`).

Combined with the warm-start round, the picture is consistent:

| | bf16 | fp32 |
|---|---|---|
| from scratch | fails (5-6 dB) | works (24.6 / 25.8 dB) |
| pretrained start | collapses to 4.5-5.3 dB | best (36.2 / 39.0 dB) |

Precision is the variable that separates success from collapse in both
settings; the data source (raw vs processed) only shifts the result by ~1 dB.

## 4. Proposed small multi-scene validation

Since fp32 from scratch succeeds on the fixed sample, expand carefully:

1. 4-6 fixed ScanNet scenes x the same 2+2 protocol, fp32, 4000 steps, both
   arms, one seed; report context and novel PSNR/SSIM, grey baselines and alpha.
2. Same scenes with the 8+7 protocol to check the protocol effect separately.
3. Only then consider more seeds; keep the bf16 path out of the loop until the
   autocast boundary is fixed (e.g. force fp32 for the SSIM conv and the
   backbone matmuls, or use bf16 only where it is safe).

## 5. Reproduce

```
python scripts/train_plain_recon_ab.py --source raw \
  --out-dir .../plain_ab_scratch_fp32/raw \
  --init-state workspace_recon_diag/plain_ab_shared_init.pt \
  --pair-seed 42 --seed 42 --amp fp32 \
  --total-steps 4000 --mid-step 2000 --warmup-steps 1000 --lr 1e-4 \
  --log-every 50 --image-every 200 --save-steps final
python scripts/probe_dtypes.py --out .../dtypes.json
```
