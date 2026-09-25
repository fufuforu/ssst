# Full processed-ScanNet pure-LocusGS reconstruction training

Goal: train one **pure LocusGS** reconstruction model on the whole official
SIU3R processed-ScanNet **train** split, using the recipe that was stable on the
32/8 development split, with a resumable checkpoint policy and a low-cost
training monitor.  No instance/query head, no semantics, no depth or instance
input.  **No SIU3R official evaluation was run in this round.**

Status: **submitted and running** (job `55247`).  The 50000-step budget is a
*training budget*, not a claim of convergence.  Validation numbers below are
**training monitoring only** and must not be quoted as SIU3R results.

---

## 1. Implementation

No model, loss or data code was forked.  The existing entry point
`scripts/train_cross_scene.py` was **extended** (defaults preserve the 32/8
recipe exactly) with:

| addition | purpose |
|---|---|
| `--train-scenes all` | build the training list from every `scene*` directory of the official train root |
| `--require-disjoint-val-root` | abort if any training scene also exists in the official val tree |
| `--monitor-scenes`, `--manifest-out` | fixed low-cost monitor list; persist the resolved lists + SHA1 hashes |
| `--resume`, `--allow-schedule-change` | resume from model **+ optimizer + scheduler + step + RNG + pair-sampler + per-scene draw counts** |
| `--ckpt-every`, `--keep-last`, `--keep-steps` | periodic resumable checkpoints with pruning to a bounded set |
| atomic write | checkpoint is built in `.inprogress_step*` and `os.rename`d into `ckpt_step*`; `COMPLETE` marker written last |
| `best_monitor/` | model-only snapshot whenever the monitor mean novel PSNR improves (for the later evaluation round) |
| per-step log | `decode_r`, `psnr`, `alpha`, `alpha>0`, `depth>0`, `r_mean`, `anchor|mu|max`, `gaussian_z_p50`, `delta_p95`, `s/step`, peak GPU mem, ETA; the run aborts on a non-finite loss |

New audit/driver code: `scripts/smoke_locusgs_full.py`, launcher
`scripts/slurm_full_locusgs.sh` (the running job used an identical copy at
`workspace_recon_diag/full_train/submit_full.sh`).

## 2. Config diff vs the stable 32/8 run

`workspace_recon_diag/full_train/run/config_diff.json`.  Model and loss are
**identical** — same preset `train_siu3r_locusgs_recon_bounded_delta_frozen_radius`
(δ = tanh(δ̂), decode radius frozen at `locusgs_radius_init = 0.15`, anchor
decoder, multi-layer canonical supervision, anchor-visibility weight 0.1).
Unchanged: `lr 1e-4`, `warmup 2000`, `amp fp32`, `seed 42`, `2 context + 2 novel`,
optimizer grouping (decay 218.8M / no-decay 1.2M, AdamW β=(0.9,0.95), wd 0.05),
`scene_scale 0.15`, camera/ray handling.

Changed: `steps 6000 → 50000`; `eval-every 500 → 2500`; dense diagnostics window
`1800–3000/50 → 200–2600/200`; train split `32 → 1201` scenes; monitor set
`8 dev-split → 4 official-val` scenes; checkpoint policy added.  **Nothing else.**

LR curve: linear warmup for 2000 steps, then cosine to 2 % of the 1e-4 peak over
the remaining 48000 steps (the same closed-form schedule as the 32/8 run, only
stretched).  Recorded in every checkpoint as `scheduler`.

## 3. Data

* Official train root `/space/mawb/SIU3R/data/scannet/train`: **1201** scene
  directories, list SHA1 **`0e811cf542bb4f1a`**.
* Official val root `.../val`: **312** scene dirs (incl. `val_pair.json`, 1860
  records / 312 scenes, which is the later evaluation manifest).
* Train ∩ val = **0** (asserted at runtime by `--require-disjoint-val-root`).
* Monitor scenes (SHA1 `7180fdbec1504eb7`): `scene0011_00, scene0246_00,
  scene0458_01, scene0621_00`, taken from the **official val tree**, so they can
  never enter training.  One fixed window each, frame IDs recorded in the log.
* Note for the development history: 5 of the 8 old "development val" scenes
  lived in the official **train** tree; they are legitimately part of this
  training set and are simply not used as a monitor here.

Sampling is uniform over the 1201 training scenes with a fresh official pair per
draw, and the per-scene draw count is checkpointed; by step 2500 the run had
already touched **1041 / 1201** distinct scenes.

## 4. Pre-flight smoke (all passed)

`scripts/smoke_locusgs_full.py` audits two real runs of the training entry:
A = 8 steps fresh, B = resumed from A's `ckpt_step4` and continued to step 8.
Report: `workspace_recon_diag/full_train/smoke/smoke_report.json`.

| check | result |
|---|---|
| distinct training scenes / windows | 8 distinct scenes, 8 distinct windows in 8 steps |
| frame bookkeeping | 8 records, all 2 context + 2 novel, all four IDs distinct, order `[c0,c1,n0,n1]` |
| decode radius | exactly `0.150000–0.150000` on every step |
| finiteness | loss / grad / PSNR finite; script aborts on non-finite loss |
| alpha / depth | `alpha>0` fraction 1.000, `depth>0` fraction 1.000 |
| anchor centres | `|mu|max = 0.45` at init, no drift |
| checkpoint contents | `model.pt`, `train_state.pt` (optimizer 2 param groups / 448 tensors, scheduler, torch/cuda/numpy RNG, pair-RNG, sampler RNG, scene counts), `COMPLETE` |
| save → restore → continue | `max\|model_A − model_B\| = 2.3e-7` after both paths ran steps 5–8 (float32/gsplat noise; equivalent to bit-identical) |

Recorded reproducibility settings: `seed 42`, fp32, `CUBLAS` default
(deterministic algorithms **off**), gsplat rasterisation, `num_workers=0`.
The pair sampler is a `random.Random` whose state is checkpointed; both the
torch and numpy RNG states are restored on resume.

## 5. Measured cost and the formal job

Measured on a 3090 with a 60-step timed run and confirmed by the production log:
**0.52–0.74 s/step**, peak GPU memory **6.1 GB**.  Extrapolated 50000-step
wall-clock ≈ **7.5–9 h** including startup, 20 monitor evaluations, 20
checkpoint writes and the dense diagnostic window — comfortably inside one
24 h SLURM allocation, so no requeue machinery is needed (the run is resumable
anyway).

```
job id      55247   (squeue -j 55247)
name        lgs-full-50k        partition 3090        node 3dimage-11
submit cmd  sbatch workspace_recon_diag/full_train/submit_full.sh
log         workspace_recon_diag/full_train/run/slurm-55247.out  (.err empty)
out dir     workspace_recon_diag/full_train/run
            manifest.json, config_diff.json, history.json (updated at every
            checkpoint/eval), images/, ckpt_step*/ , best_monitor/
ckpt policy every 2500 steps, keep the 2 newest + steps 2500/12500/25000/50000
            (~2.5 GB each; bounded at ≈16 GB).  best_monitor/ ≈0.9 GB, model only.
```

Watch / manage:

```
squeue -j 55247
tail -f  workspace_recon_diag/full_train/run/slurm-55247.out
grep -a "^\[xs\] VAL" workspace_recon_diag/full_train/run/slurm-55247.out
```

Resume after an interruption (same schedule — pass the same `--steps`):

```
sbatch workspace_recon_diag/full_train/submit_full.sh   # add: --resume <out-dir>
```

Extend the budget later (a *new* recipe; the cosine schedule is recomputed):

```
... --steps 100000 --resume workspace_recon_diag/full_train/run --allow-schedule-change
```

## 6. Early health of the running job

| step | loss | PSNR | alpha | alpha>0 | `|mu|max` | decode r | s/step |
|---|---|---|---|---|---|---|---|
| 1 | 0.248 | 10.9 | 1.000 | 1.000 | 0.45 | 0.150000 | 8.5 |
| 200 | 0.201 | 9.1 | 0.994 | 0.998 | 0.75 | 0.150000 | 0.62 |
| 300 | 0.219 | 8.3 | 0.230 | 0.261 | 1.62 | 0.150000 | 0.59 |
| 1000 | 0.110 | 11.9 | 0.997 | 1.000 | 1.11 | 0.150000 | 0.53 |
| 2000 | 0.046 | 18.1 | 0.990 | 1.000 | 1.65 | 0.150000 | 0.57 |
| 2400 | 0.050 | 16.6 | 0.932 | 0.998 | 1.76 | 0.150000 | 0.57 |

Step 300 is a **single-batch transient** (one hard pair loses alpha) and it
recovers immediately; alpha stays 0.93–1.00 from step 400 onward with
`depth>0` tracking it.  The decode radius is exactly 0.15 on every step, and no
NaN/Inf occurred.

Anchor drift was compared against the *stable* 32/8 run at the same steps:

| step | full run `\|mu\|max` / `mu_z` | 32/8 `lgs_lr1e4` `\|mu\|max` / `mu_z` |
|---|---|---|
| 1800 | 1.41 / — | 1.26 / 0.96 |
| 2000 | 1.65 / — | 1.42 / 1.11 |
| 2200 | 1.81 / 1.37 | 1.57 / 1.15 |
| 2600 | 1.79 / 1.23 | 1.76 / 1.13 |

So the anchor movement seen here is the **same behaviour as the previously
stable 1e-4 run**, not a new drift/α-zero collapse (that pathology appeared with
peak lr 4e-4, which is not used).

First monitor evaluation (step 2500, 4 official-val scenes, fixed windows):
context PSNR **18.55** vs grey baseline 11.42, novel **16.66** vs grey 11.24,
SSIM 0.654 / 0.631.  This is a **training monitor**, not the SIU3R protocol.

## 7. What is deliberately NOT done / not claimed

* No `val_pair.json` evaluation, no PSNR/SSIM/LPIPS/depth table, no SIU3R
  official mAP/PQ, no per-scene official results.
* The 50000-step budget is not presented as convergence; the run may be
  extended from its full optimizer/scheduler state if the monitor curve is
  still improving.
* Any later comparison must state that **this model consumes GT camera poses to
  build rays, while SIU3R targets unposed input** — not a like-for-like
  condition even with the same scenes, frames and evaluator.
* The 32/8 development results, the InstanceQueryHead experiments and the
  SIU3R official code were not modified.
