# U128 epoch8 official evaluation

User fixed U128 epoch8 for this evaluation because epoch6 was not retained under
the registered epoch4/endpoint checkpoint policy. No checkpoint selection occurs.

- Training SHA: `574889048498d33df2542a81f4768f3758b72e11`.
- Strict model load from the U128 endpoint; no optimizer or scheduler load.
- Restore image memory size 128 and `understanding_step=66752` explicitly.
- Full official ScanNet `val_pair.json`: 1860 fixed pairs, 312 scenes, no training
  eligibility/thing-count/area filters and no silent sample skipping.
- Two context RGB images, original crop/provider, FP32, TF32 disabled, GT camera
  poses. Encoder/lifting read context; rendering uses requested cameras.
- Original packed exporter and pinned SIU3R evaluator commit
  `8ea80166be76854f938e90521f1a5b688b755c87` unchanged.
- Scope mapping: context → all/context; target-all → all/target;
  true novel → novel/target. Never use novel/context as true novel.
- Reuse the completed Full1201 metric create/update/state-merge functions
  verbatim. Canonical pair order and global AP from sufficient states;
  never average scene AP. Also report the subset excluding dev8 scenes.
- PSNR/SSIM/LPIPS use the same Full1201 implementations and per-window reduction.
  Semantic/candidate local diagnostics and qualitative images are out of scope.
- All model predictions run `eval()` and `no_grad()`; backward and optimizer
  updates are zero. Evaluation processes are independent, without distributed
  initialization or gradient synchronization.
- Real-window official evaluator/aggregation smoke precedes full prediction.
  Pipeline: smoke → 8 prediction shards → completeness/export verification →
  8 official state shards → global reduction/CSV/JSON/comparison/report.
- Consistent, completed pair receipts are reusable; prediction records retain
  checkpoint, protocol, code SHA and export file hashes.
- Comparison reuses existing Full1201 epoch6 results on the same list/scopes.
  U128 epoch8 (8344 updates) and C32 epoch6 (6258 updates) are different training
  nodes; the difference cannot be attributed solely to memory resolution.
- Do not retrain, resume, change model/loss/thresholds/export/evaluator, or add
  probes. No automatic deletion of prediction evidence or checkpoints.

Outputs live in
`/space/mawb/ssst/group_plus/object_locus_image_memory_u128_full8gpu_v1/evaluation_epoch08/`:
`protocol.json`, `full_validation_metrics.csv/json`,
`comparison_full1201_epoch6.csv/json`, `official_aggregated.json`, `report.md`,
`evaluation_complete.json`, and Slurm logs. Scripts operate only on this run.
