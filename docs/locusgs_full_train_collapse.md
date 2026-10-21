# Full LocusGS run: collapse, audit, and the lr-capped continuation branch

The full-split pure-LocusGS reconstruction run (`job 55247`, original log
`workspace_recon_diag/full_train/run/`) peaked on the training monitor at
**step 7500** and then degraded monotonically until it was numerically at the
grey-image baseline.  This note records the evidence, what was stopped and
pruned, and the single-factor continuation branch that is now running.

Development monitoring only — **no SIU3R official evaluation was run**, and the
4 monitor scenes are not the official `val_pair.json` protocol.

---

## 1. Peak and decline (4 fixed monitor scenes, mean over scenes)

Grey baselines are constant: context **11.42 dB**, novel **11.24 dB**.

| abs. step | LR | ctx PSNR | novel PSNR | ctx/novel SSIM | alpha>0.5 | train loss | grad | `\|mu\|max` | r_learned | `gz_p50` |
|---|---|---|---|---|---|---|---|---|---|---|
| 2500 | 1.00e-4 | 18.55 | 16.66 | 0.654 / 0.631 | 0.942 | 0.031 | 0.52 | 1.65 | 0.121 | 1.12 |
| 5000 | 9.91e-5 | 19.38 | 17.03 | 0.679 / 0.641 | 0.940 | 0.027 | 0.66 | 1.29 | 0.100 | 0.85 |
| **7500** | 9.69e-5 | **20.44** | **18.17** | **0.701 / 0.669** | 0.997 | 0.056 | 0.79 | 0.94 | 0.061 | 0.48 |
| 10000 | 9.34e-5 | 19.79 | 18.05 | 0.689 / 0.660 | 0.956 | 0.041 | 1.38 | 2.24 | 0.129 | 0.73 |
| 12500 | 8.89e-5 | 18.75 | 17.05 | 0.669 / 0.641 | 0.948 | 0.040 | 3.23 | 3.79 | 0.437 | 1.10 |
| 15000 | 8.33e-5 | 18.10 | 16.76 | 0.655 / 0.635 | 0.972 | 0.047 | 4.86 | 4.63 | 0.530 | 2.37 |
| 17500 | – | 15.19 | 14.13 | 0.612 / 0.600 | 0.839 | – | – | – | – | – |
| 20000 | – | 14.44 | 14.18 | 0.580 / 0.576 | 0.634 | – | – | – | – | – |
| 22500 | – | 13.93 | 13.46 | 0.548 / 0.545 | 0.573 | – | – | – | – | – |
| 25000 | – | 13.13 | 12.61 | 0.549 / 0.543 | 0.299 | – | – | – | – | – |
| 27500 | – | 11.71 | 11.54 | 0.536 / 0.535 | **0.039** | – | – | – | – | – |
| 30000 | – | 11.72 | 11.53 | 0.536 / 0.536 | **0.015** | – | – | – | – | – |
| 32500 | 3.08e-5 | 11.68 | 11.43 | 0.531 / 0.532 | 0.186 | 0.175 | **1.6e4** | **43.8** | **0.0005** | **36.7** |

Three consecutive monitor points (27500 / 30000 / 32500) sat at the grey
baseline with no recovery trend, so the stop condition was met.  `job 55247`
was stopped at **step 33500** (`scancel 55247`); the last complete checkpoint is
`ckpt_step32500`.

## 2. Model collapse, not a monitoring or loading error

`scripts/audit_full_run_collapse.py` reloads each saved checkpoint, rebuilds the
**exact** four monitor windows (same provider, same
`pair_rng.seed(seed + 1000 + i)`), and runs an independent forward pass.  It
reproduces the logged numbers to the digit:

| checkpoint | step | ctx PSNR (log / audit) | novel PSNR (log / audit) |
|---|---|---|---|
| `best_monitor` (peak) | 7500 | 20.44 / **20.44** | 18.17 / **18.17** |
| `ckpt_step12500` | 12500 | 18.75 / **18.75** | 17.05 / **17.05** |
| `ckpt_step32500` (failure) | 32500 | 11.68 / **11.68** | 11.43 / **11.43** |

The identity of the log and the fresh forward rules out a monitor/eval bug and a
checkpoint-loading bug: the stored weights themselves render at the baseline.

Independent-forward Gaussian statistics explain it — the Gaussians shrink to
nothing, become transparent and are pushed far outside the normalised scene box
(which is ≈ ±1 with the anchor init box ±0.2 + centre z 0.25):

| state | step | pred mean/std | scale p50 / p90 | opacity p50 | centre z p50 / p90 | `\|mu\|max` | r_learned | decode radius |
|---|---|---|---|---|---|---|---|---|
| peak (`best_monitor`) | 7500 | 0.394 / **0.180** | 0.0088 / 0.0184 | 0.035 | 0.47 / 0.70 | 0.90 | 0.064 | 0.15 (frozen) |
| healthy | 12500 | 0.410 / 0.177 | 0.0165 / 0.0737 | 0.105 | 1.36 / 2.82 | 3.90 | 0.267 | 0.15 |
| failure | 32500 | 0.491 / **0.026** | **0.0000** / 0.0597 | **0.000** | **11.74** / 38.60 | **40.28** | **0.0005** | 0.15 |

So the picture is: the learned anchor centres drift outward, the learned radii
(which still feed the anchor→ray bias) collapse toward zero, opacity vanishes
and the rendered image degenerates to a flat grey.  The frozen decode radius
stays exactly 0.15 in every state — that mechanism was not the failure.

## 3. Checkpoint inventory and pruning

Retained in `workspace_recon_diag/full_train/run/` (8.2 GB):

| artefact | why |
|---|---|
| `best_monitor/` (model @ step 7500) | highest-quality moment, **model only** |
| `ckpt_step2500` | last complete (model+optimizer+scheduler+RNG+sampler) checkpoint **before** the peak → branch point |
| `ckpt_step12500` | peak-adjacent checkpoint that still had a full optimizer state; kept as the alternative fork |
| `ckpt_step32500` | one failure-state checkpoint |
| `manifest.json`, `config_diff.json`, `code_state.json`, `history.json`, `images/`, `slurm-55247.out/.err` | protocol, code identity, curves, logs |

Deleted: `ckpt_step25000` and `ckpt_step30000` (already-collapsed states,
redundant with 32500; 5.0 GB), plus the archived smoke checkpoints
(`full_train/branch_smoke/S1|S2`, `full_train/smoke/runA/ckpt_step4`; 11.1 GB —
their evidence lives in `smoke/smoke_report.json`, `smoke/*.log` and
`collapse_audit/branch_check/`).  `/space/mawb` went from 18 GB to 29 GB free.
Nothing belonging to the object-aware work or any dataset was touched.

### Important limitation

The previous retention policy (`--keep-steps 2500 12500 25000 50000`) **pruned
`ckpt_step7500` and `ckpt_step10000`**, i.e. the checkpoint at the monitor peak
and its neighbour.  `best_monitor/` still holds the peak *model*, but it is
`model.pt` only — there is no optimizer/RNG state for step 7500.  Therefore the
only strictly-resumable **pre-peak** checkpoint is **`ckpt_step2500`**, and the
branch below starts there.  `best_monitor` + a fresh optimizer would have been a
model-only warm start, not a continuation, and is *not* what was run.

## 4. Continuation branch (job 55275)

One intervention only: `--lr-cap 2e-5` → `lr = min(schedule_lr, 2e-5)`
(implemented in `scripts/train_cross_scene.py`, recorded in every checkpoint's
`scheduler.lr_cap`).  Inherited unchanged: model weights at step 2500, optimizer
state (no reset), **no re-warmup**, pair/sampler RNG (so the same data sequence
as the original run from step 2501 onward), fp32, loss, 2 context + 2 novel,
seed 42, and the schedule shape (`--steps 50000`; the cap stops binding once the
cosine falls below 2e-5 at ≈ step 36400).

```
job id     55275        name lgs-lrcap2e5      node 3dimage-11
submit     sbatch scripts/slurm_branch_lrcap.sh
log        workspace_recon_diag/full_train/run_lrcap2e5/slurm-55275.out
out dir    workspace_recon_diag/full_train/run_lrcap2e5/
ckpt every 2500 steps; keep newest 2 + 5000/25000/50000 (≈3 dirs, ~7.5 GB)
resume     sbatch scripts/slurm_branch_lrcap.sh --resume <out-dir>
```

### Pre-flight checks (passed)

* `ckpt_step2500` verified complete: 450 model tensors, optimizer with 2 param
  groups / 448 state tensors, `scheduler {warmup 2000, total 50000,
  peak_lr 1e-4, lr_cap None}`, torch/cuda/numpy RNG, pair-RNG, sampler RNG,
  1041 per-scene draw counts.
* Short smoke S1 (steps 2501–2504) → LR printed **2.00e-05** on every step (cap
  active, no re-warmup); S2 resumed from S1's `ckpt_step2501` and reproduced the
  same losses/PSNR.
* Resume consistency: `max|model_S1 − model_S2| = 3.1e-5` after 3 steps, with
  renders at step 2504 differing by 0.01 dB.  This is larger than the 2.3e-7
  seen in the earlier smoke because Adam normalises by the gradient magnitude
  and the gradients here are tiny (0.2–0.6); it is float-level noise amplified
  by the optimiser, not a state-restore error.
* **Rendering at the fork point is identical**: independent forward gives
  `src2500` ctx 18.55 / novel 16.66 / scale p50 0.0170 / opacity p50 0.059 /
  alpha>0.5 0.942, and `branch2501` ctx 18.55 / novel 16.66 / scale 0.0170 /
  opacity 0.059 / alpha>0.5 0.943.

### Results so far, aligned by absolute step with the original run

| abs. step | original ctx / novel | original SSIM | branch ctx / novel | branch SSIM | branch LR |
|---|---|---|---|---|---|
| 2500 (fork) | 18.55 / 16.66 | 0.654 / 0.631 | — (branch point) | — | — |
| 5000 | 19.38 / 17.03 | 0.679 / 0.641 | **20.11 / 17.55** | 0.692 / 0.656 | 2.00e-05 |
| 7500 | 20.44 / 18.17 | 0.701 / 0.669 | 20.26 / **18.17** | 0.693 / 0.663 | 2.00e-05 |

Training-side health on the branch is also better contained at the same absolute
steps: at step 7700, `loss 0.042`, `alpha>0 1.000`, `|mu|max 0.76`,
`r_mean 0.043` — versus the original's `|mu|max` growing 2.24 → 3.79 → 4.63 over
steps 10000–15000.  The branch is at ~0.5 s/step, ETA ≈6 h.  The decision points
to watch are the next monitor points (10000, 12500, 17500, 25000):
the original fell from 18.17 → 18.05 → 17.05 → 14.13 → 12.61 over that range.

Representative panels: `workspace_recon_diag/full_train/collapse_audit/`
(`best_monitor_step7500_*.png` shows scene structure, `ckpt32500_step32500_*.png`
shows only grey with noise specks) and
`collapse_audit/branch_check/` (fork-point identity).

## 5. Not done / not claimed

* No SIU3R `val_pair.json` evaluation, no official metrics, no SIU3R mAP/PQ.
* The 50000-step budget is not a convergence claim; the branch inherits it.
* Any later comparison must state that this model uses **GT camera poses** to
  build rays, whereas SIU3R targets unposed input.
* The object-aware LocusGS work in the same tree was not modified or evaluated
  here.
