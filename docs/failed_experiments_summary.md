# Failed / closed experiment lines (do not re-run)

Kept so the same negative results are not re-derived.  Checkpoints, optimizer
states and duplicate render strips for the lines below were deleted on
2026-09-23 to reclaim space; metrics/logs/configs were kept.

## 1. Plain TokenGS canonical reconstruction on SIU3R-processed ScanNet

**Conclusion: fails.**  Fixed-sample (scene0048_01) and full runs both stay at
the grey-image baseline.  Peak context PSNR gain over the per-arm grey baseline
is <= +0.25 dB, every run ends ~2 dB *below* grey, alpha saturates to ~1.0 and
the renders are flat foreground-coloured blobs rather than structure.

Evidence kept: `workspace_recon_diag/plain_tokengs_baseline/*_rows.json`,
`*_summary.json`, `scale_audit.json`, `images/`;
`workspace_recon_diag/plain_canonical_eval/plain_canonical_final.json` (+ split
variants); `workspace/siu3r_plain_tokengs_canonical_recon_v1/eval_fixed16_*`,
`train.log`, `config.yaml`.

Deleted: `plain_tokengs_baseline/ckpt_preset_step{250,500,750,800}`,
`ckpt_paper_like_step{250,500,750,800}`, `base/ckpt_*`, `ss06/ckpt_*`,
`z025/ckpt_*`; `workspace/siu3r_plain_tokengs_canonical_recon_v1/checkpoints`.

## 2. Raw ScanNet vs SIU3R-processed ScanNet, 2 context + 2 novel, from scratch

**Conclusion: neither data path reconstructs.**  Both arms fail exactly as in
line 1; the single-pair difference that initially looked like "raw escapes,
processed is stuck" is not reproducible (identical seed + identical shared init
+ identical data still diverges), and the replicated between-arm difference
(best ctx PSNR +0.12 dB, best ctx SSIM +0.014) is at or below the run-to-run
noise.  So the failure cannot be attributed to the processed data, to the view
count, or to the loss individually.

Evidence kept: `workspace_recon_diag/raw_vs_processed_v2/` (batch comparison +
tiles); `workspace_recon_diag/plain_ab/preset/{processed,raw}/*_rows.json`,
`summary.json`, `config_shared.yaml`, `images/` (peak + final per arm);
`plain_ab/repl/seed*/{processed,raw}/*_rows.json` + `repl/repl_summary.json`;
`plain_ab/swap_rgb/summary.json` (n=1 pilot, inconclusive);
`docs/scannet_raw_vs_processed_ab.md`.

Deleted: `plain_ab/preset/*/ckpt_step{2000,4000}` (model+optimizer),
`plain_ab/repl/*/*/ckpt_step4000/model.pt`, `plain_ab/swap_rgb/*/ckpt_*`,
`plain_ab/repl_init/` (regenerable with `scripts/gen_plain_recon_init.py`), and
the per-100-step render strips except the peak and final step per arm.

## 3. LocusGS-faithful ScanNet reconstruction line

**Conclusion: closed — geometry collapse / no advantage over plain TokenGS.**
The corrected paper-recipe run collapsed (gradient spike step 380, r>1 step
450, empty render step 500, |mu|>10 step 1200, grad=0 step 1400) and was
archived in `workspace_recon_diag/failed_locusgs_archive/`.  The inferred-v2 and
gamma-calibrated diagnostics and the locusgs smoke runs were purely
single-variable diagnostics of that line.

Evidence kept: `workspace_recon_diag/{locusgs_*, plain_tokengs_diag}`,
`workspace/siu3r_locusgs_inferred_v2_diag1000/eval_fixed16_*`, `train.log`,
`training_log.json` (+ the gamma-calibrated variant).

Deleted: `workspace/siu3r_locusgs_inferred_v2_diag1000/checkpoints`,
`workspace/siu3r_locusgs_inferred_v2_gamma_calibrated_diag1000/checkpoints`,
`workspace_locusgs_smoke/*/checkpoints`.

## 4. Original TokenGS golden-path probes (A0-A5, M8, CUR)

**Conclusion: all failed** — kept (tiny, ~450 KB) as the historical baseline
for the same negative result.  `workspace_recon_diag/original_tokengs_golden/`.

## Not touched (purpose not clearly closed)

`workspace/siu3r_ssst_joint_final_v1`, `siu3r_ssst_joint_temperature_fix_v2`,
`siu3r_ssst_spatial_recon_pretrain_v1` checkpoints — the SSST joint / spatial
main line, not part of the closed reconstruction A/B.  Eval directories and
`raw_vs_processed/` (superseded but tiny, see its `SUPERSEDED.md`) were kept.
