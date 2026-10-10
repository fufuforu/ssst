# Final epoch 8 official SIU3R evaluation

Training job 59740 completed successfully: 8 epochs, 8344 updates, 66752 new exposures. Training code SHA: e80c99380a4eb04456dc7fb69c7382ff11af1051. Final checkpoint SHA256: 40e93e3e9d1157f1d6af97e51414bb098a116f6f2444b190993dbc0a33a47824.

Use the existing `eval_object_locus_frozen_vggt_posefree_v1.py` inference entry with `--official-png-export`. The model, VGGT identity, context-only generation, camera calibration, monitor_v1 policy and unchanged panoptic export thresholds are preserved. Disjoint scene shards export native RGB uint8 and millimetre-depth uint16 PNGs using existing save helpers. Completion records permit engineering restarts without repeating completed windows. The frozen training model is never updated.

The cohort is the pinned SIU3R val_pair.json: 1860 windows / 312 scenes, SHA256 59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b. Report context (2 frames), target-all (6 frames, including context) and true-novel (4 frames). Context-only scene generation is followed by independent RGB-based target camera calibration; rendering specified target views requires target cameras.

`score_object_locus_posefree_official.py` runs the unmodified SIU3R Evaluator at commit 8ea80166be76854f938e90521f1a5b688b755c87 in SIU3R/.venv_gpu_v4. Image/depth scores come directly from Evaluator.evaluate over the exported PNGs, including per-image positive-GT scale-and-shift alignment. Context/novel reconstruction scopes filter its native per-image outputs. Global mIoU/PQ sufficient states are summed and COCO AP is recomputed globally in canonical pair order. The exact merge helpers come from existing ssst commit f49b492bf3030d434a39f7cb7321bc149561022e; no scene/shard AP averaging is used. A two-real-window contract compares each shard's merge with the native evaluator at 1e-7 before aggregation.

mIoU_s is semantic segmentation. mIoU_t is **text-referred segmentation**, not target-view semantic IoU. This visual checkpoint has no trained text branch, so mIoU_t is NOT_TRAINED / unmeasured. Adding or training a text branch would change the locked model and is outside this evaluation.

CPU contract checks, syntax checks, and real GPU export/official-scoring validation precede the full run. `--limit` results are explicitly marked PARTIAL_ENGINEERING_CHECK; the full reducer requires exact coverage of all 1860 unique windows and 312 scenes. Metrics, per-class official states, checkpoint provenance, geometry warnings, progress/remaining-time estimates and a readable summary are retained under the new evaluation evidence directory, leaving training and historical failures intact.

## Verified official metric device acceleration

The initial full job 60299 completed all 1860 exports and all native RGB/depth scores, then ran segmentation on CPU. Measured CPU throughput projected about 46 additional minutes. Job 60302 verified that moving the **unchanged** official processing and metric objects to CUDA reproduces mIoU/PQ/mAP/AP50/AP75 for both real all/novel exports at 1e-7; warmed novel processing took 0.058 seconds per pair. The continuation uses `EVAL_SCORE_ONLY=1 EVAL_GPU_SEGMENTATION=1` to retain all existing exports and reconstruction scores, repeats device contracts on each shard, and computes remaining segmentation on allocated GPUs. Global AP reduction still uses the unchanged official state helpers on CPU. Historical CPU progress and logs remain retained; training is untouched. Inference SHA and scoring SHA are separately recorded in the final report.

## Completed full-cohort results

Final scoring job 60304 completed with exit 0:0 on 2026-10-10 at 14:42:34 Asia/Shanghai. All 1860 unique windows / 312 scenes were evaluated. Inference SHA: 3e114b8b5d5f3fb7efd05ca68aaf7277f8dd9d32; scoring SHA: f4a2254adae3a0082cef449fd10959d57cdc8d5f.

| Metric | Context (2) | Target-all (6) | True-novel (4) |
|---|---:|---:|---:|
| AbsRel ↓ | 0.183012 | 0.182756 | 0.182628 |
| RMSE ↓ (m) | 0.397442 | 0.396913 | 0.396649 |
| PSNR ↑ (dB) | 18.667394 | 18.550626 | 18.492242 |
| SSIM ↑ | 0.652257 | 0.649123 | 0.647556 |
| LPIPS ↓ | 0.615719 | 0.617734 | 0.618741 |
| mIoU_s ↑ | 0.555298 | 0.551823 | 0.550072 |
| mAP ↑ | 0.237761 | 0.224000 | 0.227495 |
| PQ ↑ | 0.618400 | 0.602949 | 0.605482 |
| mIoU_t ↑ | NOT_TRAINED | NOT_TRAINED | NOT_TRAINED |

Target-all includes both context frames. Image counts: context 3720, target-all 11160, true-novel 7440. Native per-image RGB/depth means, global official mIoU/PQ/COCO AP, complete coverage, finite results, and all seven CUDA/CPU contracts passed final checks. Geometry quality: 1808 OK / 52 WARNING; all fits numerically valid and all windows retained.

Evidence directory: `/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2_monitor/evaluation/final_epoch08_official`. `metrics.json` retains full precision and official per-class results; `summary.md`, `coverage_and_geometry.json`, `final_validation.json`, and `COMPLETE.json` record outcome and provenance. `../show_progress.py` is a read-only status/result viewer. Historical training and evaluation logs remain unchanged.
