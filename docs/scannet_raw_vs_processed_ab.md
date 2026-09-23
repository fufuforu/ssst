# Pure TokenGS reconstruction: raw ScanNet `.sens` vs SIU3R-processed ScanNet

Controlled fixed-sample A/B on `scene0048_01`, the **same four frames**
(context `[654, 664]`, novel `[655, 659]`, same order), the same
`PlainTokenGSCanonicalRecon` model, the same initial weights, loss, optimizer
grouping, bf16 autocast, seed and 4000-step learning-rate plan.

## 1. Headline

* **Completed:** a correct raw ScanNet data path now exists in this repo and is
  bit-verifiable against the shared provider pipeline.
* **Completed:** the final-batch comparison. The naive "stretch 1296x968 ->
  256x256" raw adaptation (commit `8c2a665`) is superseded; the corrected raw
  path changes the batch by RGB (22.9 dB PSNR vs processed), one intrinsic
  parameter (focal, +2.7 %), and the rays/Pluecker (<= 9e-3). Relative C2W and
  ray origins are identical. The data source **is** a real batch variable.
* **Failed (both arms):** under the 2-context + 2-novel, from-scratch canonical
  reconstruction protocol, **neither** the processed nor the raw path
  reconstructs. The peak PSNR over the grey-image baseline is +0.16 dB
  (processed) / +0.28 dB (raw) on average and every run ends ~2 dB *below*
  grey; the renders are flat foreground-coloured blobs, not geometry.
* **Insufficient evidence:** a single processed/raw pair looks different, but
  the training is **not** run-to-run reproducible (identical seed + identical
  init + identical data still gives different trajectories). The between-arm
  difference is not clearly larger than that noise, so this experiment does
  **not** show that "the raw data path reconstructs while the processed path
  cannot". The dominant failure is the protocol/objective, not the data source.

## 2. What the two paths are

| stage | original TokenGS raw path (B) | SIU3R processed path (A) |
|---|---|---|
| RGB source | raw `.sens` JPEG stream, 1296x968 | `<scene>/color/<f>.jpg`, already 256x256 |
| intrinsics | raw colour `K` (fx=fy=1170.188, cx=647.75, cy=483.75) | `intrinsic.txt` (fx=fy=318.013, cx=cy=127.932) |
| pose | raw OpenCV C2W from `.sens` | `extrinsic/<f>.txt` |
| crop/resize | `ImageTransform` centre-crop-to-fill 1296x968 -> 968x968 -> 256x256 | identity (256->256, no crop) |
| K update | `Provider._preprocess` (shift/scale of the same transform) | identity |
| rays/Pluecker | `ray_condition` on the updated K and relative C2W | same code, different K |

The raw reader is the **original** `ScanNetSensReader`
(`/space/mawb/tokengs/tokengs/data/static/scannet.py`), imported by file path in
`tokengs/data/scannet_raw_recon.py` so it cannot shadow this repo's `tokengs`
package. Nothing hand-scales K and nothing stretches the image. The adapter
exposes no semantics, instances, depth or mask:

* the provider substitutes an **all-ones** foreground mask and an **all-ones**
  depth map. The mask only multiplies the GT image by 1 (compositing on white
  is a no-op for a full mask), `lambda_mask == 0`, and depth is unused because
  `camera_scale_method == 'constant'`. Neither placeholder reaches the encoder
  input or the reconstruction loss.

## 3. Final batch comparison (`audit_raw_vs_processed_batch_v2.py`)

Both groups are pushed through the real `Provider`, so these are the exact
tensors a training step consumes.

| field | max abs diff | mean abs diff | PSNR |
|---|---|---|---|
| supervised RGB | 7.12e-1 | 3.31e-2 | 22.92 dB |
| intrinsics | 8.54e0 | 4.27e0 | — |
| relative C2W (`cam_view`) | 7.17e-7 | 1.62e-7 | 120 dB |
| `rays_o` | 1.43e-7 | 9.23e-8 | 120 dB |
| `rays_d` | 9.06e-3 | 3.88e-3 | 46.8 dB |
| Pluecker | 9.06e-3 | 1.95e-3 | 49.8 dB |

pixel K — processed `[318.013, 318.013, 127.932, 127.932]`,
raw `[309.471, 309.471, 127.934, 127.934]`: the principal point is identical to
2e-3, the focal differs by 2.7 % (FOV 43.85 deg vs 44.94 deg). Per-frame RGB
difference is 22.5 / 22.5 / 22.7 / 24.1 dB dB for frames 654/664/655/659.

Self-consistency (both groups): rays and Pluecker re-derived from that group's
own pixel K and relative C2W reproduce the emitted rays to <= 1.2e-7, and the
raw batch K exactly reproduces the 1296x968 -> 968x968 -> 256x256 transform
(`shift=(-164, 0)`, `scale=0.264463`). So RGB, K, pose and rays agree inside
each group; the earlier `8c2a665` K error (~87 px) came only from the stretch.

## 4. Paired training

Frozen across arms: `PlainTokenGSCanonicalRecon`, the shared initial weights,
a freshly built AdamW (decay wd=0.05 / no-decay wd=0, betas 0.9/0.95), bf16
autocast, grad-clip 1.0, the canonical objective (MSE + 0.2*(1-SSIM)/2 +
1.0*L_vis), 2 context + 2 novel, 4000 optimizer steps, warmup 1000, cosine to
2 %, seed 42, `scene_scale` 0.15, `gaussian_z_offset` 1.0.

### 4.1 Single pair (`workspace_recon_diag/plain_ab/preset`)

The same config was run twice and is **not** reproducible:

| arm | repeat | best ctx PSNR (gain) | final ctx PSNR (gain) |
|---|---|---|---|
| processed | #1 (log `preset.processed.slurm.out`) | 9.24 (+0.21) | 7.12 (-1.91) |
| processed | #2 (`preset/processed/*_rows.json`) | 9.21 (+0.18) | 5.20 (-3.84) |
| raw | #1 (log `preset.raw.slurm.out`) | 10.45 (+1.42) | 5.32 (-3.71) |
| raw | #2 (`preset/raw/*_rows.json`) | 9.26 (+0.23) | 5.69 (-3.34) |

Repeat #1 suggested "raw escapes, processed is stuck"; repeat #2 removes that
gap. The checkpoints in `preset/` are repeat #2. The grey baseline is 9.03 dB
for every arm (per-arm GT).

### 4.2 Replication (`workspace_recon_diag/plain_ab/repl`)

5 init seeds x 2 arms + a same-seed (42) repeat per arm, pair sampler pinned
(`--pair-seed 42`) so only the initial weights vary.

| statistic | processed (n=6) | raw (n=6) | paired raw-processed |
|---|---|---|---|
| best ctx PSNR gain | +0.159 +- 0.051 | +0.281 +- 0.266 | +0.122 (values +0.063, -0.140, +0.705, -0.058, +0.053, +0.109) |
| final ctx PSNR gain | -2.364 +- 0.306 | -2.008 +- 0.771 | +0.356 |
| best ctx SSIM | 0.6175 +- 0.0013 | 0.6312 +- 0.0065 | +0.0137 (all six positive) |

Same-seed repeat spread (pure run-to-run non-determinism, GPU rasteriser
atomics): best ctx differs by 0.058 dB (raw) / 0.145 dB (processed); final ctx
differs by 1.144 dB (raw) / 0.200 dB (processed).

Reading: no arm reconstructs (best gain <= 0.25 dB except one raw seed at
+0.87, all finals ~-2 dB, alpha saturating to ~1.0, renders are flat blobs).
The raw arm's peak is marginally and inconsistently higher than processed: the
SSIM difference (+0.014) is consistent across all six pairs, the PSNR
difference (+0.12 dB) is not distinguishable from the same-seed noise.

### 4.3 Localisation pilot (`workspace_recon_diag/plain_ab/swap_rgb`, n=1)

Pilot only: keep the arm's K/C2W/rays, swap in the other source's RGB.

| arm | camera/K | RGB | best ctx (gain) | final ctx (gain) |
|---|---|---|---|---|
| processed | processed | raw | 9.25 (+0.22) | 7.24 (-1.79) |
| raw | raw | processed | 9.24 (+0.21) | 7.11 (-1.92) |

The mixed arms land near each other and (in this single run) end higher than
the pure arms, but n=1 and the non-determinism above make this uninterpretable;
it is a candidate for the next round, not a result.

## 5. Fallback: the original tokengs ScanNet reconstruction recipe

Since both arms failed, the previously-successful ScanNet reconstruction run in
`/space/mawb/tokengs` was checked:

* **views**: 8 context + 7 novel (`context_views: 8`, `target_views: 7`,
  stride 20), not 2 + 2;
* **manifest**: `/space/mawb/tokengs/data/scannet_prompt/scannet_prompt_full_wide_8x7.json`
  (5680 train samples over 1425 scenes);
* **weights**: warm-started from a ScanNet reconstruction backbone
  `/space/mawb/tokengs/workspace/scannet_recon_finetune_base_8k/tokengs_backbone_step_008000.safetensors`
  (itself warm-started from the released DL3DV latent checkpoint
  `checkpoints/dl3dv_latent_6v_ssim.safetensors`);
* **result**: ~19-20.5 dB mean PSNR on held-out ScanNet views.

So the original recipe reaches ~20 dB with 8+7 views and a pretrained backbone;
this A/B's 2+2 from-scratch canonical objective never leaves the grey baseline.

## 6. Next experiments

1. Replicate the RGB/camera swap (`--swap-rgb`) over >= 5 init seeds before
   reading anything into it.
2. Make the run reproducible or quantify it: pin the rasteriser, or report
   every arm as a distribution (>= 5 seeds) rather than a single number.
3. Separate protocol from data: repeat the A/B with the old recipe's 8+7 views
   and/or a warm start from the ScanNet backbone, holding the data source as
   the only variable.

## 7. Artifacts

Code: `tokengs/data/scannet_raw_recon.py`, `scripts/audit_raw_vs_processed_batch_v2.py`,
`scripts/train_plain_recon_ab.py`, `scripts/summarize_plain_recon_ab.py`,
`scripts/analyze_plain_recon_repl.py`, `scripts/gen_plain_recon_init.py`.

Data (untracked, under `workspace_recon_diag/`):

* `raw_vs_processed_v2/` — batch comparison json + side-by-side tiles;
* `plain_ab/preset/{processed,raw}/` — primary pair: `*_rows.json`, `images/`
  (every 100 steps), `ckpt_step2000` and `ckpt_step4000` (model + optimizer);
* `plain_ab/repl/seed{42,42b,43,44,45,46}/{processed,raw}/` — replication
  (`*_rows.json`, images every 200 steps, final model only);
* `plain_ab/swap_rgb/` — localisation pilot (n=1);
* `plain_ab/repl_init/init_seed{42..46}.pt` — shared per-seed initial weights.

Run commands (SLURM, one GPU per job, `tokengs` env):

```
sbatch -p 3090 -N1 -n1 --gpus-per-task=1 -c 8 --time=02:00:00 --wrap \
  "cd /space/mawb/ssst && PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH \
   python scripts/train_plain_recon_ab.py --source processed \
     --out-dir workspace_recon_diag/plain_ab/preset/processed \
     --init-state workspace_recon_diag/plain_ab_shared_init.pt \
     --total-steps 4000 --mid-step 2000 --warmup-steps 1000 --lr 1e-4 \
     --log-every 50 --save-optimizer"
```

(the raw arm is identical with `--source raw`; add `--swap-rgb` and
`--pair-seed 42` for the localisation/replication arms).
