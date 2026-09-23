# Why the pretrained TokenGS fine-tune collapses on a fixed sample

Setup: the strictly-loaded `scannet_recon_finetune_base_8k/tokengs_backbone_step_008000`
checkpoint, one fixed raw 8+7 batch (`scene0059_00`, ctx `[0,20,...,140]`,
novel `[10,30,...,130]`), `PlainTokenGSCanonicalRecon`, canonical loss, 50
optimizer steps.  Every measurement is a fresh fp32 forward on the current
parameters, in both `eval()` and `train()` mode, on the same batch/cameras/GT.
Scripts: `scripts/diagnose_pretrained_finetune.py`,
`scripts/probe_amp_gradient.py`.  Raw numbers:
`workspace_recon_diag/finetune_diag/diagnostic_summary.json`.

## 1. Degradation timeline (state transitions, lr 1e-4, bf16)

| state | ctx PSNR | target PSNR | loss | alpha | param disp |
|---|---|---|---|---|---|
| after load | 19.26 | 19.17 | 0.04664 | 0.971 | 0 |
| after load (repeat) | 19.26 | 19.17 | 0.04664 | 0.971 | 0 |
| after `model.train()` | 19.26 | 19.17 | 0.04664 | 0.971 | 0 |
| after 1st training forward | 19.26 | 19.17 | 0.04664 | 0.971 | 0 |
| after 1st backward | 19.26 | 19.17 | 0.04664 | 0.971 | 0 |
| after step 1 | 19.26 | 19.17 | 0.04664 | 0.971 | 0 |
| after step 2 | 19.26 | 19.17 | 0.04662 | 0.971 | 1.2e-3 |
| after step 5 | 19.26 | 19.17 | 0.04659 | 0.971 | 8.3e-3 |
| after step 10 | 19.24 | 19.16 | 0.04664 | 0.970 | 3.3e-2 |
| after step 20 | 19.15 | 19.07 | 0.04699 | 0.968 | 1.3e-1 |
| after step 50 | **17.84** | 17.77 | 0.05377 | 0.952 | 6.4e-1 |

* Switching to `train()` mode and running forward/backward changes **nothing**
  (bit-identical to the loaded state), and an `--lr 0` run is bit-identical for
  all 50 steps: there is no train/eval-mode branch and no forward side effect.
  The metrics are recomputed on the current parameters (the load/repeat rows
  agree to 5 decimals, so the fixed-batch eval noise here is < 1e-5).
* The **first step is a no-op** (lr = 0 on step 1 of this schedule); the first
  effective update is step 2.  **First measurable degradation: between step 5
  and step 10** (loss leaves its value); it is unmistakable by step 20 and the
  loss then rises monotonically (0.0466 -> 0.0538 by step 50).
* Gradients are non-trivial throughout: pre-clip total norm 9.6 at step 1,
  rising to 20.3 at step 50.

## 2. Single-variable controls (50 steps each)

| run | ctx@50 | dctx | loss@50 | param disp@50 |
|---|---|---|---|---|
| lr 0 (adamw) | 19.26 | 0.00 | 0.04662 | 0 |
| adamw 1e-7 | 19.26 | 0.00 | 0.04662 | 8.5e-4 |
| adamw 1e-6 | 19.26 | 0.00 | 0.04660 | 8.6e-3 |
| adamw 1e-5 | 19.19 | -0.07 | 0.04697 | 8.2e-2 |
| **adamw 1e-4 (bf16)** | **17.84** | **-1.42** | 0.05377 | 6.4e-1 |
| adamw 1e-4, wd 1e-6 | 17.59 | -1.67 | 0.05510 | 6.9e-1 |
| adamw 1e-4, no clip | 17.64 | -1.62 | 0.05476 | 7.5e-1 |
| sgd 1e-3 | 19.26 | 0.00 | 0.04657 | 2.3e-2 |
| sgd 1e-2 | 19.17 | -0.09 | 0.04700 | 2.3e-1 |
| sgd 3e-2 | 17.29 | -1.97 | 0.05836 | 6.9e-1 |
| **adamw 1e-4 (fp32)** | **20.47** | **+1.21** | 0.03615 | 8.0e-1 |
| adamw 1e-6 (fp32) | 19.32 | +0.06 | 0.04622 | 1.4e-2 |

Ruled out:

* **weight decay** - matching the old recipe (1e-6 instead of 0.05) does not
  help; **grad clipping** is not the cause (disabling it changes nothing).
* **Adam's normalised update** - SGD degrades at the same cumulative
  displacement as AdamW, so it is not Adam-specific.
* **displacement alone** - fp32 at lr 1e-4 moves *further* (disp 0.80) than
  bf16 (0.64) and *improves*.  The difference is the **direction**.

## 3. Root cause: the bf16-autocast gradient is not the true gradient

`scripts/probe_amp_gradient.py` computes the gradient of the same loss at the
same parameters twice, once in fp32 and once under `torch.autocast(dtype=bfloat16)`:

| quantity | fp32 | bf16 |
|---|---|---|
| loss | 0.04662 | 0.04932 |
| gradient L2 norm | **0.995** | **9.610** |

cosine similarity(fp32 grad, bf16 grad) = **0.124**, relative L2 difference 9.58.

The autocast gradient is ~10x larger and almost orthogonal to the true one: it
is dominated by bf16 quantisation noise in the differentiable rasteriser / SSIM
path, not by the loss gradient.  The optimizer therefore takes large,
misdirected steps, the training loss rises from the first effective update, and
the converged model leaves its (shallow) basin.  This also explains the
pre-clip gradient norm of 9.6 seen in every bf16 run - it is the noise norm, not
the signal norm (fp32 norm is 0.995).

## 4. Key differences from the old successful ScanNet training path

| factor | this A/B | old successful path |
|---|---|---|
| precision | `torch.autocast` bf16 for the forward/loss | accelerate `mixed_precision="bf16"` |
| reconstruction LR | 1e-4 | **1e-5** (`gsi_v2_reconstruction_lr`) |
| weight decay | 0.05 | **1e-6** |
| schedule | warmup 1000 + cosine | OneCycle (`pct_start_steps`) |
| data | **one fixed sample** | 1425 scenes / 5680 windows |

The old path never combined bf16 noise with a 1e-4 step on a *single* sample at
a shallow optimum; with 1425 scenes and lr 1e-5 the same noise is averaged and
the step is 10x smaller, so the model keeps descending.

## 5. Fix and validation

Fix: **run the fine-tune forward/loss in fp32** (`--amp fp32`;
`train_plain_recon_ab.py` now exposes `--amp {bf16,fp32}`).  From the same
checkpoint, same fixed sample, same plan (4000 steps, warmup 1000, cosine to
2 %, lr 1e-4):

| run | step 1 | step 100 | step 1000 | step 4000 | SSIM@4000 |
|---|---|---|---|---|---|
| 2+2 processed | 22.03 | 26.63 | 26.08 | **36.17** | 0.973 |
| 2+2 raw | 22.48 | 29.88 | 27.72 | **39.01** | 0.987 |
| 8+7 raw | 19.26 | 21.9 | 24.0 | (running) | |

PSNR never falls below its initial value and the renders keep the scene
structure (see `workspace_recon_diag/plain_ab_warm_fp32/*/images/*_step4000.png`).

## 6. Reproduce

```
srun -p 3090 -N1 -n1 --gpus-per-task=1 -c 8 --time=00:30:00 bash -lc \
  "cd /space/mawb/ssst && /space/mawb/anaconda3/envs/tokengs/bin/python \
   scripts/diagnose_pretrained_finetune.py --lr 1e-4 --amp bf16 --out .../diag_bf16.json"

# controls: --lr 0 | --lr 1e-6 | --optimizer sgd --lr 1e-2 | --no-clip | --no-amp
# gradient probe:
python scripts/probe_amp_gradient.py --out .../amp_gradient.json
# fixed fix:
python scripts/train_plain_recon_ab.py --source raw --init-safetensors <ckpt> \
  --amp fp32 --total-steps 4000 --mid-step 2000 --warmup-steps 1000 --lr 1e-4 ...
```
