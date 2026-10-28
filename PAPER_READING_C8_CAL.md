# C8_Cal paper reading entry

This branch is the completed **Baseline epoch8 + frozen Token/GS quality head + calibration** version. It is **not** QPanchor1, QuerySplat reconstruction, 4096-token finetuning, or the current B/C/D experiments.

The original executable code is unchanged from commit `02392745114b538fe1c30901c5add8e6e18745e4`. Only paper reading documents and small completed-result/configuration references are added. Cluster-local paths in historical provenance documents describe the original execution environment; use the repository-relative links below to read code in a browser. Weights, datasets and rendered predictions are not part of this publication.

## Start here

- [Chinese reading prompt](PAPER_PROMPT_ZH.md)
- [Completed result analysis](docs/paper_c8_cal/frozen_baseline8_gs_quality_completed_review_20261025.md)
- [Model reference,32 GS features,formula,metric names](docs/paper_c8_cal/C8_CAL_MODEL_REFERENCE_20261025.json)
- [Original training and continuation](docs/paper_c8_cal/codex_handoff_baseline_epoch8_completed_20261023.md)
- [All11 metrics and five variants](docs/paper_c8_cal/formal_11metrics_comparison.csv)
- [Source code identity](docs/paper_c8_cal/BASELINE8_SOURCE_CODE_IDENTITY.json): all10 completed Baseline8 model/recipe/export files match byte-for-byte.

## Code reading map

|Read|Purpose|
|---|---|
|[export_unified_query_checkpoint.py](scripts/export_unified_query_checkpoint.py)|Checkpoint metadata to actual model constructor; do not pick a class from its name alone.|
|[superpoint_structural_variants.py](tokengs/models/superpoint_structural_variants.py)|Model registry; target architecture OMEGA_V4INIT_NO_ANCHOR_CA_ISOLATED_V4_RECIPE_V1.|
|[v4init_isolated_noca.py](tokengs/models/v4init_isolated_noca.py)|Actual Baseline8 core/forward,detached understanding reads,no learned object-anchor CA.|
|[olda4_initialization_ablation.py](tokengs/models/olda4_initialization_ablation.py)|Inherited initialization and geometric object layer.|
|[superpoint_dual_core.py](tokengs/models/superpoint_dual_core.py)|Inherited reconstruction/understanding components.|
|[unified_query_v1.py](tokengs/models/unified_query_v1.py)|Input extraction,generation,rendering and target camera calibration; follow imports.|
|[unified_query_loss.py](tokengs/models/unified_query_loss.py)|Main model losses; distinguish them from head regression.|
|[v4init_isolated_recipe.py](scripts/v4init_isolated_recipe.py),[v4init_extend8_policy.py](scripts/v4init_extend8_policy.py)|Original recipe and epoch4-to8 continuation.|
|[cache_gs_quality.py](scripts/cache_gs_quality.py)|Single frozen forward,Token/GS descriptors,label construction,candidate identities.|
|[gs_quality_core.py](scripts/gs_quality_core.py)|The32 GT-free GS descriptors and small head definitions.|
|[fit_gs_quality.py](scripts/fit_gs_quality.py)|Train/dev split,scaler,regression,best-epoch selection.|
|[frozen_iou_head_core.py](scripts/frozen_iou_head_core.py)|IoU targets and ranking helper.|
|[export_gs_quality.py](scripts/export_gs_quality.py)|Same-forward Native/Rank/Cal variants;quality score and pixel competition.|
|[reciprocity_panoptic.py](scripts/reciprocity_panoptic.py),[export_object_locus_v3_set_official.py](scripts/export_object_locus_v3_set_official.py)|Output assembly and thresholds.|
|[score_gs8_quality.py](scripts/score_gs8_quality.py)|Formal metrics and comparison.|

## Facts that must survive paper drafting

- Main model: epoch8,8344updates,SHA bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5.
- Quality stage: main model frozen; C8 has256 Token +32 GS features,74753 parameters. Head SHA9de8c6f6096ca3fb6fed9c60b44112ba0d9f7029f298bd344f8084ba81b0d130.
-1860 windows /312 scenes. Target-all6 includes Context2; it is not pure novel-only.
- Native8 and head variants share RGB/depth renderings. Head/calibration does not improve reconstruction metrics.
- C8_Cal prioritizes local mIoU/mAP; PQ is lower than Native8. Single seed,unequal A/C head capacity,missing epoch8 shuffled-GS control.
- Depth uses the existing per-image scale/shift evaluation. Do not call its error absolute metric depth accuracy.
- Separate target camera/geometric alignment from quality-based pixel competition calibration. Check use of target images and GT before claiming pose-free or context-only inference.
- No unverified novelty,causal,real-time or cross-paper superiority claims.

## GitHub source provenance

The GitHub source mirror is [acd3618f63489138e947eb20e93106b1d58caa83](https://github.com/fufuforu/ssst/tree/acd3618f63489138e947eb20e93106b1d58caa83). Its entire Git tree is `d15f5fcdabce49061b369ac93b2f20a43de4cc49`, exactly equal to original execution commit `02392745114b538fe1c30901c5add8e6e18745e4`. Commit identities differ because the upload uses the GitHub API with the existing remote Baseline8 commit as parent; executable file bytes are identical. The original execution SHA is provenance and is not a separately published GitHub commit.
