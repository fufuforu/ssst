# Official SIU3R val_pair reconstruction evaluation (full-data pure LocusGS)

Read-only evaluation of the completed full-data pure-reconstruction LocusGS
checkpoint against the **official** `/space/mawb/SIU3R/data/scannet/val_pair.json`
(1860 records / 312 scenes) using the pinned, unmodified SIU3R evaluator through
the repository's existing adapter path.  No training, no fine-tuning, no
gradient step, no model change, and **no G0/G1/G0+ instance evaluation**.

## 1. Protocol, code and checkpoint identity

| item | value |
|---|---|
| official manifest | `/space/mawb/SIU3R/data/scannet/val_pair.json` - 1860 records, 312 scenes |
| official processed val tree | `/space/mawb/SIU3R/data/scannet/val` |
| SIU3R repo / commit | `/space/mawb/SIU3R` @ `8ea80166be76854f938e90521f1a5b688b755c87` ("update readme") |
| SIU3R python env | `/space/mawb/SIU3R/.venv_gpu_v4/bin/python` (torch 2.4.1+cu118) |
| adapter (export) | `scripts/evaluate_ssst_validation.py --reconstruction-only` |
| evaluator entry | `scripts/invoke_siu3r_official_evaluator.py --recon-only` -> `SIU3R/src/evaluator.py::Evaluator.evaluate` |
| prediction dir | `workspace_recon_diag/full_train/eval_official_valpair/predictions/official_predictions` |
| official result | `.../eval_official_valpair/official_evaluator_result.json` (+ the evaluator's own `results.json`) |
| model config | `.../eval_official_valpair/config_locusgs_best47500.yaml` (generated from the training preset; `tyro` round-trip verified) |

Two minimal, protocol-preserving fixes were needed in the adapter (no metric
code was copied or rewritten):

1. **frame-id naming** - outputs are now named from the model's *actual* batch
   `frame_ids`, with assertions that the six ids equal the manifest
   `target_ids`, that the first two are `context_ids`, and that the remaining
   four are `target_ids \\ context_ids`.
2. **reconstruction-only evaluator switch** - the `--run-official` path now
   calls `evaluate(..., segmentation=False)`, and the full run used the
   standalone `--recon-only` entry point, so the evaluator computes image +
   depth quality only and never looks for semantic/instance PNGs.

Checkpoint selection (fixed by the training monitor **before** looking at any
val_pair result):

| run | COMPLETE checkpoints | `best_monitor` step | monitor novel / context PSNR |
|---|---|---|---|
| `full_train/run` (job 55247, stopped at 33500 after collapsing to grey) | ckpt_step2500, 12500, 32500 | **7500** | 18.17 / 20.44 |
| `full_train/run_lrcap2e5` (job 55275, lr<=2e-5 branch, finished 50000) | ckpt_step5000, 25000, 47500, 50000 | **47500** | **20.88 / 24.55** |

The original run did fall to the grey baseline (novel 11.43 dB at step 32500);
the lr-capped continuation recovered and was still healthy at step 50000
(20.86).  The main checkpoint is therefore the **`run_lrcap2e5/best_monitor`
weight at step 47500**:

```
path      /space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt
step      47500          (model only - no optimizer state in best_monitor)
sha256    5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f
loaded    strict=True, 450 tensors, model_type siu3r_locusgs_recon
```

`best_monitor/` has no `config.yaml`; passing `--config` explicitly is essential
because the adapter's fallback is `config_defaults["eval_siu3r_ssst"]`, whose
`model_type` is `siu3r_joint_ssst` (verified: that fallback was hit once during
development and produced `Error(s) in loading state_dict for SIU3RJointSSST`).
The supplied config pins `model_type=siu3r_locusgs_recon`,
`locusgs_bound_delta=True` (delta = tanh(delta_hat)),
`locusgs_freeze_decode_radius=True` with `locusgs_radius_init=0.15`,
`locusgs_supervised_layers=(6,12)`, `num_gs_tokens=1024`, fp32, 2 context
inputs, 6 rendered views, `img_size=(256,256)`.  The frozen decode radius was
checked in the model: `last_decode_radius` min = max = **0.150000** on every
evaluated record, and the model was **not** loaded through the eval preset.

## 2. The three interface gates (all passed)

**Gate 1 - frame ids.**  The manifest lists `target_ids` with the two context
frames usually *not* in the first positions: for record 0,
`context_ids=[1727,1744]`, `target_ids=[1727,1729,1732,1738,1739,1744]`, while
the model actually receives `[1727,1744,1729,1732,1738,1739]`.  The old export
loop named outputs with `enumerate(target_ids)`, so **every record mislabelled
views 1-5** (the second context prediction/GT would have been written as 1729
instead of 1744).  After the fix:

* all 1860 records satisfy the id/order assertions
  (`gates.batch_frame_ids_match_manifest = true`);
* `rgb_gt/scene0011_00_1744.png` is **byte-identical** to
  `val/scene0011_00/color/1744.jpg`, and `..._1729.png` to `color/1729.jpg`;
* `depth_gt/scene0011_00_1744.png` is byte-identical to
  `val/scene0011_00/depth/1744.png`.

**Gate 2 - pure-reconstruction evaluator.**  The evaluator ran with
`eval_context_miou = eval_target_miou = eval_context_pq = eval_target_pq =
eval_context_map = eval_target_map = False` and returned only
`psnr, ssim, lpips, absrel, rmse`.  **Semantic mIoU / PQ / mAP are N/A for this
run** (no semantic or instance map exists); they are not zero.

**Gate 3 - depth units.**  The provider builds the model frame by scaling the
camera-to-world translation by the fixed `scene_scale = 0.15` (intrinsics
untouched), so gsplat's expected depth (`render_mode="RGB+ED"`) is in
**metres x 0.15**.  The adapter now multiplies the rendered depth by the fixed
`1/0.15 = 6.6667` before writing millimetre PNGs - a constant derived from the
training configuration, **never** from GT and never fitted per scene.  The
evaluator then re-applies its own internal per-image scale+shift fit
(`Evaluator.fit_scale_and_shift`), which is SIU3R's own depth protocol.

Empirical check over all 1860 records: predicted depth range **0.000-5.004 m**
(non-zero fraction 0.9996), GT depth range **0.269-9.987 m** (valid-pixel
fraction 0.931), and the median of `GT/pred` per frame averages **0.968** - i.e.
the fixed 1/0.15 conversion lands on the metric scale (residual <5% is model
error, not a unit error).  No GT-based alignment was used anywhere in the
export.

Smoke before the full run (`.../eval_official_valpair/smoke2`, 2 records,
including record 0 where the second context is last in `target_ids`): six
correctly named RGB/depth predictions + GT per record, all finite, and the
official evaluator returned psnr 21.53 / ssim 0.725 / lpips 0.426 /
absrel 0.217 / rmse 0.527.

## 3. Official result (all 1860 records / 312 scenes)

```
psnr    24.811
ssim     0.7795
lpips    0.3826
absrel   0.1770
rmse     0.3967
```

These are the numbers the pinned SIU3R evaluator returned; no metric was
re-implemented.  Per-scene distribution (312 scenes): PSNR min 19.07, p25 23.51,
median **24.72**, p75 26.26, max 29.19; per-scene AbsRel median 0.1685, max
0.3295.  Per-record and per-scene tables:
`per_record_metrics.csv`, `per_scene_metrics.json`.

Diagnostic split of the evaluator's **own per-item scores** (clearly *not* the
official aggregate, and not a replacement for it):

| views | items | PSNR | SSIM | LPIPS | AbsRel | RMSE |
|---|---|---|---|---|---|---|
| official aggregate (all 6 per record) | 11160 | 24.811 | 0.7795 | 0.3826 | 0.1770 | 0.3967 |
| context only (2 per record) | 3706 | 26.032 | 0.8022 | 0.3656 | 0.1765 | 0.3958 |
| novel only (4 per record) | 7454 | 24.203 | 0.7682 | 0.3910 | 0.1773 | 0.3971 |

Coverage sanity: predicted RGB/depth are finite for every record and predicted
depth is non-zero on 99.96 % of pixels, so these averages are not computed over
grey images or all-zero depth.

## 4. Mandatory caveats

* **Checkpoint selection overlaps the evaluation data.**  `best_monitor` was
  chosen on the 4 monitor scenes that live in the official **val** tree -
  `scene0011_00, scene0246_00, scene0458_01, scene0621_00` - and all four
  **do appear in `val_pair.json`**, contributing **24 of 1860 records (1.3 %)**.
  The number above is therefore a validation score on a set that partially
  informed checkpoint selection, not a clean held-out test score.
* **Input condition.**  This LocusGS consumes **GT camera poses** to build
  rays; SIU3R's official method targets **unposed input**.  Even with the same
  data pairing, the same 2-context/6-frame protocol and the same evaluator, the
  two are **not** like-for-like.
* Only the 2 context frames are model input; the 4 novel RGB/depth are never
  fed forward (the adapter contract asserts `gt_to_forward = False`).
* The evaluator's depth metric is affine-invariant per image
  (`fit_scale_and_shift`), so AbsRel/RMSE cannot be read as absolute metric
  depth accuracy.

## 5. Commands and artefacts

```bash
# export (GPU, tokengs env) - 1860 records, ~55 min
python -u scripts/evaluate_ssst_validation.py \
  --checkpoint-dir workspace_recon_diag/full_train/run_lrcap2e5/best_monitor \
  --config workspace_recon_diag/full_train/eval_official_valpair/config_locusgs_best47500.yaml \
  --manifest /space/mawb/SIU3R/data/scannet/val_pair.json \
  --val-root /space/mawb/SIU3R/data/scannet/val \
  --output workspace_recon_diag/full_train/eval_official_valpair/predictions \
  --reconstruction-only

# official metrics (GPU, SIU3R venv) - ~1.5 h
cd /space/mawb/SIU3R && .venv_gpu_v4/bin/python -u \
  /space/mawb/ssst/scripts/invoke_siu3r_official_evaluator.py \
  --eval-path /space/mawb/ssst/workspace_recon_diag/full_train/eval_official_valpair/predictions/official_predictions \
  --output   /space/mawb/ssst/workspace_recon_diag/full_train/eval_official_valpair/official_evaluator_result.json \
  --recon-only

# summary + figures
python scripts/summarize_siu3r_valpair_eval.py --eval-root <eval dir> --out <eval dir>/summary.json
python scripts/make_valpair_eval_figures.py --eval-root <eval dir> --out <eval dir>/figures
```

Artefacts under `workspace_recon_diag/full_train/eval_official_valpair/`:
`official_evaluator_result.json`, `predictions/{eval_report.json, official_predictions/, raw_predictions/}`,
`summary.json`, `per_record_metrics.csv`, `per_scene_metrics.json`,
`figures/` (6 panels: best/median/worst scene, one context + one novel view,
`GT rgb | pred rgb | GT depth | pred depth`), `smoke2/` (gate evidence).
Total 2.4 GB; `/space/mawb` has 11 GB free.

## 6. Not done

* No semantic/instance/PQ/mAP evaluation (nothing to evaluate for a pure
  reconstruction model) and no G0/G1/G0+ metrics.
* No further training, no checkpoint changes, no SIU3R code changes (the pinned
  evaluator was used exactly as released).
