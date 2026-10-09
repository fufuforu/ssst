# Calibration v2 design note

Protocol: `shared_context_depth_sim3_v2`.

The calibration-only aggregator receives the exact two generation contexts plus the existing target RGB views in one full-window call. Its camera head predicts all cameras; its depth head decodes only the two context views from those same globally aggregated tokens. The context depth outputs may carry target-view information through VGGT attention. They are not passed to the generation model.

Fixed 37x37 correspondences per view use pixel centers at `(7+14*r+0.5, 7+14*c+0.5)`. Independent axial depth and K518 backproject into raw source-world X; saved context-only `predicted_points` are target Y in the generation scene. Validity requires finite positive depth and official confidence in both passes and finite X/Y. Confidence is ranked by empirical midrank, transformed with `0.05 + 0.95*sqrt(pct_A*pct_B)`, then normalized to total weight 0.5 for each view.

The FP64 weighted Umeyama fit centers and RMS-normalizes each point cloud, checks second/first covariance eigenvalue ratio >=1e-6 (planar accepted), performs one initial fit plus exactly five Huber refits, and restores the transform to original coordinates. It maps cameras directly into the generation frame. It never reapplies first-camera normalization or `a_scale`.

Predeclared per-window engineering acceptance: >=32 valid pairs per view, positive-Z ratio >=0.95, 256-pixel reprojection median <=4 and p90 <=12, finite camera and Sim(3), SO(3) tolerance 1e-8, positive scale. Failures stop the stage with recorded evidence; no GT fallback, skipped windows, altered threshold, or prior calibration fallback.

Expected artifacts: `window4253_old_alignment.json`, `window4253_context_sim3_v2.json`, and its compact points NPZ; then single/eight GPU smoke reports, formal plan/run manifests, and 20-update startup confirmation. No real GPU result is claimed until the ordered job has produced it.
