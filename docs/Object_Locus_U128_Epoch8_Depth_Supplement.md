# U128 epoch8 depth-only supplement

Supplement the completed 1860-pair, 312-scene official evaluation with AbsRel
and RMSE. Reuse exactly the registered validation list, checkpoint identity,
image memory size 128, exposure 66752, crop, provider and GT poses.

Do not modify model/loss/provider/renderer or rerun completed segmentation/RGB
metrics. Predictions use model.eval()/torch.no_grad(); no optimizer or backward.

Export rendered expected depth using the existing fixed scene-scale inverse
1/0.15, and the provider's original nearest-cropped metric GT. Use existing
save_depth to round/clamp uint16 millimetre PNG. Invalid GT stays zero.

Invoke the unchanged pinned SIU3R Evaluator with segmentation/image quality
disabled and depth quality enabled. Its original fit_scale_and_shift aligns
each frame on GT>0, then computes AbsRel and RMSE in metres. Average official
per-frame scores over all requested frames, matching its official aggregate.
Retain context/target-all/true-novel splits of those same scores and the existing
cohort excluding dev8 scenes; no second depth metric implementation.

Paper depth columns use the official target-all aggregate, including context.
NVS columns remain true novel. mIoU_t remains missing. No old depth values copied.
Undefined metrics stay UNDEFINED, not zero. Data errors fail with an evidence
record; no sample is silently omitted.

First one real-window smoke, then eight independent prediction/depth-score
shards and automatic reduction/report/table-row generation. Slurm can use free
cards across partitions 3090 and 4090; A6000 is excluded by partition choice.
No distributed initialization or synchronization between evaluation processes.

Outputs are isolated under the existing report directory's
evaluation_epoch08/depth_supplement. Original completion receipts and metrics
remain intact. Write depth_metrics.csv/json, per_frame_depth.csv/json,
full_validation_metrics_with_depth.csv/json, report_with_depth.md,
paper_ours_row.tex and depth_complete.json with hashes.
