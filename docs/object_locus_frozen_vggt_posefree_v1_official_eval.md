# Final epoch 8 official SIU3R evaluation

Training job 59740 completed successfully: 8 epochs, 8344 updates, 66752 new exposures. Training code SHA: e80c99380a4eb04456dc7fb69c7382ff11af1051. Final checkpoint SHA256: 40e93e3e9d1157f1d6af97e51414bb098a116f6f2444b190993dbc0a33a47824.

Use the existing `eval_object_locus_frozen_vggt_posefree_v1.py` inference entry with `--official-png-export`. The model, VGGT identity, context-only generation, camera calibration, monitor_v1 policy and unchanged panoptic export thresholds are preserved. Disjoint scene shards export native RGB uint8 and millimetre-depth uint16 PNGs using existing save helpers. Completion records permit engineering restarts without repeating completed windows. The frozen training model is never updated.

The cohort is the pinned SIU3R val_pair.json: 1860 windows / 312 scenes, SHA256 59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b. Deliver one row with the 13 metric columns of SIU3R Table 1; leave both text mIoU columns empty. Keep detailed scope calculations only in the raw evidence. Context-only scene generation is followed by independent RGB-based target camera calibration; rendering specified target views requires target cameras.

`score_object_locus_posefree_official.py` runs the unmodified SIU3R Evaluator at commit 8ea80166be76854f938e90521f1a5b688b755c87 in SIU3R/.venv_gpu_v4. Image/depth scores come directly from Evaluator.evaluate over the exported PNGs, including per-image positive-GT scale-and-shift alignment. Context/novel reconstruction scopes filter its native per-image outputs. Global mIoU/PQ sufficient states are summed and COCO AP is recomputed globally in canonical pair order. The exact merge helpers come from existing ssst commit f49b492bf3030d434a39f7cb7321bc149561022e; no scene/shard AP averaging is used. A two-real-window contract compares each shard's merge with the native evaluator at 1e-7 before aggregation.

mIoU_s is semantic segmentation. mIoU_t is **text-referred segmentation**, not target-view semantic IoU. This visual checkpoint has no trained text branch, so mIoU_t is NOT_TRAINED / unmeasured. Adding or training a text branch would change the locked model and is outside this evaluation.

CPU contract checks, syntax checks, and real GPU export/official-scoring validation precede the full run. `--limit` results are explicitly marked PARTIAL_ENGINEERING_CHECK; the full reducer requires exact coverage of all 1860 unique windows and 312 scenes. Metrics, per-class official states, checkpoint provenance, geometry warnings, progress/remaining-time estimates and a readable summary are retained under the new evaluation evidence directory, leaving training and historical failures intact.

## Verified official metric device acceleration

The initial full job 60299 completed all 1860 exports and all native RGB/depth scores, then ran segmentation on CPU. Measured CPU throughput projected about 46 additional minutes. Job 60302 verified that moving the **unchanged** official processing and metric objects to CUDA reproduces mIoU/PQ/mAP/AP50/AP75 for both real all/novel exports at 1e-7; warmed novel processing took 0.058 seconds per pair. The continuation uses `EVAL_SCORE_ONLY=1 EVAL_GPU_SEGMENTATION=1` to retain all existing exports and reconstruction scores, repeats device contracts on each shard, and computes remaining segmentation on allocated GPUs. Global AP reduction still uses the unchanged official state helpers on CPU. Historical CPU progress and logs remain retained; training is untouched. Inference SHA and scoring SHA are separately recorded in the final report.

## Completed full-cohort results

Final scoring job 60304 completed with exit 0:0 on 2026-10-10 at 14:42:34 Asia/Shanghai. All 1860 unique windows / 312 scenes were evaluated. Inference SHA: 3e114b8b5d5f3fb7efd05ca68aaf7277f8dd9d32; scoring SHA: f4a2254adae3a0082cef449fd10959d57cdc8d5f.

| AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUₛ↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUₜ↑ | 新视图 mIoUₛ↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUₜ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.18276 | 0.3969 | 18.55 | 0.6491 | 0.6177 | 0.5553 | 0.2378 | 0.6184 |  | 0.5518 | 0.2240 | 0.6029 |  |

按 SIU3R Table 1 的顺序输出 13 个指标列；前 5 列为重建，随后各 4 列为输入视图与新视图场景理解。两处 mIoUₜ 留空：文本分支未训练。

取值严格对齐固定版本官方 Evaluator 的返回字段：重建用其原生图像/深度汇总，两组场景理解分别用 context_* 与 target_*。官方 val_pair.json 每条记录的 target_ids 含 2 张输入帧和 4 张额外帧；官方代码未剔除输入帧。因此表中“新视图”列沿用官方 target 集合，未替换为仅额外 4 帧的另一种汇总。

Target-all includes both context frames. Image counts: context 3720, target-all 11160, true-novel 7440. Native per-image RGB/depth means, global official mIoU/PQ/COCO AP, complete coverage, finite results, and all seven CUDA/CPU contracts passed final checks. Geometry quality: 1808 OK / 52 WARNING; all fits numerically valid and all windows retained.

Evidence directory: `/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2_monitor/evaluation/final_epoch08_official`. `metrics.json` retains full precision and official per-class results; `summary.md`, `coverage_and_geometry.json`, `final_validation.json`, and `COMPLETE.json` record outcome and provenance. `../show_progress.py` is a read-only status/result viewer. Historical training and evaluation logs remain unchanged.


## Requested Table 1 presentation

The final deliverable is `siu3r_table1.csv` (exactly 13 columns, full precision, two empty cells), `siu3r_table1.md`, and `siu3r_table1.json`. The formatter `scripts/format_object_locus_posefree_siu3r_table.py --root <evidence-directory>` only reads the completed metrics and writes presentation artifacts. No inference, scoring, or training is repeated. The original three-scope summary is preserved under `report_versions/summary_three_scopes_original.md`; raw `metrics.json` and evaluation identity remain unchanged.

The Table 1 “Novel Views” columns map to the pinned official evaluator's `target_miou`, `target_map.map`, and `target_pq` on the original six-frame target set. `src/data/components/scannet_dataset.py` loads `target_ids` directly; `src/visualizer.py` exports all configured targets; `src/evaluator.py` aggregates all exported target images and segmentation masks without excluding input frames. The separately filtered four-frame output stays diagnostic and is not mixed into this official table row. Both context and target masks use the existing model's Gaussian readout, consistent with the official validation pipeline's rendered mask path.

## Completed epoch 4 comparison (2026-10-10)

Job 60306 completed with exit 0:0 at 15:53:25 Asia/Shanghai (elapsed 00:34:42). Epoch 4 checkpoint: 4172 updates / 33376 new exposures; SHA256 ef5ae41cb850e05a8e47b91863662c98828f7fbe4bac7b26fab3a77099b48761. Inference/scoring snapshot SHA f54fe66121c1e1ae7c1ac68f07ffd8ea47417911. Epoch 8 training and results were retained; no new optimizer updates were submitted. Model, training math, data and camera conditions did not change.

| Checkpoint | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUₛ↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUₜ↑ | 新视图 mIoUₛ↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUₜ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch4 | 0.18202 | 0.3937 | 18.45 | 0.6486 | 0.6145 | 0.5596 | 0.2408 | 0.6166 |  | 0.5566 | 0.2274 | 0.5989 |  |
| epoch8 | 0.18276 | 0.3969 | 18.55 | 0.6491 | 0.6177 | 0.5553 | 0.2378 | 0.6184 |  | 0.5518 | 0.2240 | 0.6029 |  |

The 13 metric columns follow the same official mapping described above; both mIoU_t cells remain empty. The CSV has exactly 13 metric columns, with data rows in epoch 4/8 order.

Both checkpoints cover identical 1860 windows / 312 scenes / 11160 target image observations. Epoch 4 passes finite-value, exact-coverage, clock/hash and all seven real CUDA/CPU metric contracts. Geometry is 1808 OK / 52 WARNING; all windows remain included under monitor_v1. Frozen VGGT prediction/calibration diagnostics match epoch 8 for all windows. Epoch 8 raw metrics SHA256 remains 1d6b6be9f8604994a989337e68eddfb1ca4076a2c1361343e84bc60f87393af7.

From epoch 4 to 8: PSNR +0.096956dB, SSIM +0.000520, LPIPS +0.003234 (worse), AbsRel +0.000740 (worse), RMSE +0.003180 (worse). Context mIoU/mAP decrease 0.4278/0.3004 percentage points while PQ increases 0.1815 points; official target mIoU/mAP decrease 0.4727/0.3382 points while PQ increases 0.4062 points. This weak/mixed change provides little support for assuming additional epochs alone will recover prior reconstruction quality; it does not isolate the causal effect of freezing or cameras.

Evidence: `.../evaluation/epoch04_official` contains COMPLETE/validation/raw metrics; `.../evaluation/epoch04_vs_epoch08` contains the exact-column CSV, paired comparison and full precision JSON. The paired PSNR mean over 11160 matching observations agrees with the difference of official aggregates to 1e-10. No column-wise best checkpoint selection was made.

Read-only progress/results: `scripts/show_object_locus_posefree_official_progress.py --epoch 4` or `--epoch 8`. `scripts/finalize_object_locus_posefree_official_eval.py` validates completed evidence without repeating inference/scoring. `scripts/compare_object_locus_posefree_epochs.py` matches frame IDs and provenance before generating the comparison.

See [the six-paper VGGT / ScanNet research review](object_locus_vggt_scannet_literature_20261010.md) for prior reconstruction adaptation, frozen teacher versus trainable student, camera-input differences, and the limits of applying paper recipes to this locked experiment.
