# Calibration v2 design note

Protocol: `shared_context_depth_sim3_v2`.

The calibration-only aggregator receives the exact two generation contexts plus the existing target RGB views in one full-window call. Its camera head predicts all cameras; its depth head decodes only the two context views from those same globally aggregated tokens. The context depth outputs may carry target-view information through VGGT attention. They are not passed to the generation model.

Fixed 37x37 correspondences per view use pixel centers at `(7+14*r+0.5, 7+14*c+0.5)`. Independent axial depth and K518 backproject into raw source-world X; saved context-only `predicted_points` are target Y in the generation scene. Validity requires finite positive depth and official confidence in both passes and finite X/Y. Confidence is ranked by empirical midrank, transformed with `0.05 + 0.95*sqrt(pct_A*pct_B)`, then normalized to total weight 0.5 for each view.

The FP64 weighted Umeyama fit centers and RMS-normalizes each point cloud, checks second/first covariance eigenvalue ratio >=1e-6 (planar accepted), performs one initial fit plus exactly five Huber refits, and restores the transform to original coordinates. It maps cameras directly into the generation frame. It never reapplies first-camera normalization or `a_scale`.

Current policy: `geometry_quality_policy=monitor_v1`. Hard validity requires >=32 valid pairs per view, nondegenerate positive RMS and covariance second/first ratio >=1e-6, finite invertible cameras/K and fit, SO(3) tolerance 1e-8, positive scale, and finite loss/gradients/updated parameters. Quality reference values (positive-Z ratio >=0.95, 256-pixel reprojection median <=4 and p90 <=12) generate per-view warnings only. A valid fit returns status=PASS, fit_status=VALID and quality_status=OK/WARNING. PASS means the computation contract passed; it does not establish teacher accuracy. Warnings do not alter cameras, losses, weights or exposure, and do not stop train/smoke/eval.

Each rank writes scalar geometry_monitor_rankN.jsonl, and a summary every 20 formal updates. Only hard failures and the first two distinct warning windows retain full point/camera evidence. Historical strict-policy reports below are retained as failures under that historical policy. Engineering repairs may be retried with a newly pushed fixed snapshot; fresh smoke state never enters formal training.

Observed execution (Slurm 59658, code SHA
`d4107c881b0c5ce4e0bb187f620d0607e9e41454`): fixed window 4253 passed with
v2 scale `0.257346004`, reprojection median `[1.332, 1.379]` and p90
`[2.894, 5.975]` pixels. The old method's signed scale on that window was
`1.151038197`; the earlier failure was not reproduced on that diagnostic
window. The single-card real smoke passed its two updates. The eight-card
smoke stopped at update 1, rank 5, scene `scene0563_00`, context `[145,197]`:
view medians were `[4.837, 4.602]` pixels and view 0 p90 was `21.183`, above
the historical strict acceptance limits. Full error diagnostics and point evidence are in
the external run evidence directory `.../calibration_v2/attempts/59658/`.
Status is `GEOMETRY_BLOCKED`; formal training did not start and no geometry
threshold or fallback was changed.
