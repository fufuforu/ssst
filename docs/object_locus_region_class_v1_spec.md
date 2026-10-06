# Object-Locus Region-Conditioned Classification V1

## Registered question and arms

Test whether pre-trained image features pooled over the predicted instance mask's rendered context-view region improve the conversion of existing masks into candidate instances and final panoptic output.

- C is the original Object-Locus Panoptic V1 model and retains cosine plus geometry feedback.
- R differs only at final L12 classification: its 100 thing queries receive a residual built from pooled image features. Stuff queries and all mask, Gaussian, geometry, feedback, renderer, and loss paths remain unchanged.

No endpoint weights from an earlier experiment are an initialization source. Both arms use fresh reconstruction step 47500, MASt3R encoder, adapter, and COCO panoptic pretraining.

## R readout

The existing encoder output `fm` is interpolated to `[B,2,256,256,256]`. The existing final Gaussian mask prediction is `sigmoid(features @ mask_embedder(q).T)`, shaped `[B,65536,102]`. That predicted membership is alpha-composited with `render_feature_channels` into the two actual context camera views. The first 100 rendered thing channels are the pooling weights `[B,2,100,256,256]`; they are not binarized, GT-masked, or alpha-normalized.

FP32 joint two-view pooling uses `d=sum(weights)` and `n=einsum(weights,feature_grid)`. `z=n/clamp_min(d,1e-6)` and rows with `d<=1e-6` are zero. No detach is used, preserving gradients to mask membership and understanding features.

The sole R parameter is `panoptic.region_class_proj: Linear(256,256,bias=False)`, initialized to zero under forked RNG seed 31417. For classification only, `z` is functionally normalized with LayerNorm epsilon `1e-5`, projected, and added to thing rows of `final['q']`. The 2 stuff rows are copied unchanged. Original `final['q']` and all earlier state remain untouched. Existing 19-channel class head and semantic readout consume the resulting class probabilities.

The R addition is 65,536 parameters. C has 572,432,103 parameters; R has 572,497,639. Projection belongs to `new_decay`, peak LR `1e-4`, weight decay `0.05`.

## Locked training

The exact manifest and plan are loaded from the existing mask-guided Control artifacts and compared item-by-item. Training uses 8 scenes, 56 windows, 2 context images per window, 64 epochs, 2 RTX3090 ranks, per-rank micro-batch 1, and 4 accumulation steps per optimizer update. Each saved global plan row contains 8 windows; rank 0 processes positions 0/2/4/6 and rank 1 processes 1/3/5/7, in order. Every micro-batch uses the same update exposure and loss warm-up. Its gradients are scaled by 1/4, accumulated locally, averaged across the two ranks, then globally clipped once and stepped once. Training remains 448 updates and 3,584 window exposures per arm; each window has exactly 64 exposures. Total C+R training is 896 optimizer updates and 7,168 exposures.

The source recipe is the original panoptic runtime: AdamW `(0.9,0.95)`, epsilon `1e-8`, FP32, TF32 off, global clip 1.0; reconstruction/understanding/new peak LR `1e-6/1e-5/1e-4`; 200 exposure warm-up (25 global updates), then cosine to 0.1; understanding weight `min(exposure/200,1)`; injection beta `0.1*min(exposure/1000,1)`; original independent reconstruction/understanding loss gradients and explicit two-rank gradient mean. The logged loss is averaged over all 8 windows. No additional loss is used.

Checkpoints are saved at epochs 0/8/16/32/64 (updates 0/56/112/224/448) with complete model and optimizer state, RNG, manifest/plan, code revision and endpoint metadata.

## Deferred evaluation

Evaluation is not run by training. After the user confirms both arms are complete, compare fixed epoch64 outputs on `train_all56`, `same_scene_holdout8`, `dev8`, and `val32`, with context, target-all, and true-novel scopes. Official scopes are context=`all/context`, target-all=`all/target`, true-novel=`novel/target`; local candidate and official packed-panoptic results remain distinct. Report official mIoU/PQ/mAP/AP50, candidate mAP/AP50 and CA/CW precision/recall, raw mask coverage, context Hungarian classification, PSNR/SSIM/LPIPS, per-GT query correspondences, scene/window/frame IDs, and matched qualitative pairs.

Primary endpoint is val32 true-novel official packed mAP, with 2,000 paired scene bootstrap draws, seed 2026, recomputing global AP after resampling scenes while keeping each scene's windows together. The preregistered gates and camera-pose limits are defined in the user task specification. No training-time evaluation is enabled.
