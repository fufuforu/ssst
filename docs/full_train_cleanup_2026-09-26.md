# Full-data pure-reconstruction cleanup (2026-09-26, task A)

Read-only inventory first, then an explicit, itemised deletion; nothing else in
the repository was touched.  Details and the full kept/deleted inventory with
sizes live in [full_train_cleanup_manifest.json](full_train_cleanup_manifest.json).

## Status before cleanup

* Queue: **empty** - both full-data jobs (`55247` crashed run, `55275`
  lr-capped branch) had finished, so no running job could be reading these files.
* `workspace_recon_diag/full_train/` = 22 GB: `run` 8.2 GB, `run_lrcap2e5` 11 GB,
  `eval_official_valpair` 2.4 GB, plus `collapse_audit`, `smoke`, `timing`,
  `branch_smoke` (all < 6 MB).
* No symlinks and no hard links anywhere under `full_train`; every `model.pt` is
  an independent file (link count 1, distinct inode).

## Main evaluation model preserved

`run_lrcap2e5/best_monitor` (step 47500) was already an independent file (own
inode, links = 1, **not** a link to `ckpt_step47500`), and its weights were made
self-contained before anything was deleted:

* `config.yaml` - the Options/preset snapshot that was previously only in
  `eval_official_valpair/config_locusgs_best47500.yaml`;
* `training_record.json` - preset name, entry point, lr / lr-cap, steps, warmup,
  amp, seed, clip, train/monitor scene lists, the key architecture switches and
  the branch point.

Post-cleanup verification with `scripts/verify_full_train_checkpoint.py` (fresh
model from the recorded preset, `strict=True` load, one real 2+2 window):

| check | result |
|---|---|
| `model.pt` SHA256 | `5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f` (unchanged) |
| strict load | PASS (no missing / unexpected keys; 220,002,620 parameters) |
| decode radius | 0.150000006 min = max (frozen radius 0.15) |
| alpha mean on scene0059_00 ctx [510,524] | 0.984 |
| config snapshot next to the weights | present (`config.yaml`, `training_record.json`) |

## Deleted (8 directories, 19.36 GB, all explicit paths)

| path | size | reason |
|---|---|---|
| `run/best_monitor` | 0.88 GB | best monitor of the crashed full-data run |
| `run/ckpt_step2500` | 2.64 GB | crashed-run step checkpoint (documented branch point) |
| `run/ckpt_step12500` | 2.64 GB | crashed-run step checkpoint |
| `run/ckpt_step32500` | 2.64 GB | crashed-run step checkpoint |
| `run_lrcap2e5/ckpt_step5000` | 2.64 GB | branch step checkpoint, superseded by best_monitor |
| `run_lrcap2e5/ckpt_step25000` | 2.64 GB | branch step checkpoint |
| `run_lrcap2e5/ckpt_step47500` | 2.64 GB | branch step checkpoint (weights identical to best_monitor) |
| `run_lrcap2e5/ckpt_step50000` | 2.64 GB | branch end-of-schedule checkpoint |

`run/ckpt_step2500` was the documented resume point of the lr-capped branch; its
role is recorded in `run_lrcap2e5/manifest.json`, `submit_branch_lrcap.sh` and the
new `training_record.json`, but the checkpoint itself was removed as instructed.
The crashed run's collapse evidence remains as `collapse_audit/` (JSON + 6 PNGs,
5.5 MB) - the deleted checkpoints were the inputs, not the record.

Nothing was deleted from `smoke/`, `branch_smoke/`, `timing/`: they contain only
logs, manifests and curves (no checkpoints), so there was no smoke checkpoint to
remove.

## Kept

* `run_lrcap2e5/`: `best_monitor/` (self-contained), `history.json`, `manifest.json`,
  `images/`, Slurm logs.
* `run/`: `history.json`, `manifest.json`, `config_diff.json`, `code_state.json`,
  `images/`, Slurm logs.
* `eval_official_valpair/`: `official_evaluator_result.json` (PSNR 24.81,
  SSIM 0.7795, LPIPS 0.3826, abs-rel 0.1770, SIU3R commit `8ea80166…`),
  `per_scene_metrics.json`, `per_record_metrics.csv`, `summary.json`,
  `config_locusgs_best47500.yaml`, `full_run.log`, the 6 representative figures
  and `predictions/`.
* `collapse_audit/`, `smoke/`, `timing/`, `branch_smoke/`, the submit scripts and
  `timing.log`.

## Official predictions (default keep)

`eval_official_valpair/predictions/official_predictions` (2.44 GB) and
`predictions/raw_predictions` (4 KB) are **kept**: they are the re-check material
for the official evaluator.  No deletion candidate was found - `raw_predictions`
is tiny and `official_predictions` is the only full copy (the evaluator smoke
under `smoke2/` is a separate 2.8 MB run, also kept).

## Space

Free space on `/space/mawb`: **10.87 GB -> 30.24 GB** (19.36 GB freed).
