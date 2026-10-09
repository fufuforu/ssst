# Object-Locus Frozen VGGT Pose-Free v1

## Review status and fixed identity

This branch implements the VGGT-assisted pose-free migration path on top of
`object-locus-gc-sweep-v1` at `9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0`.
The independent branch is `object-locus-frozen-vggt-posefree-v1`.

The official VGGT source repository is
`https://github.com/facebookresearch/vggt`, source commit
`a288dd0f14786c93483e45524328726ab7b1b4ce` (remote `main` SHA observed during
implementation and checked out locally with `git rev-parse`). The model
identifier is `facebook/VGGT-1B`. The wrapper loads
only official aggregator, camera head, and depth head weights; it constructs the
official model with point and track heads disabled, explicitly records their
source keys, shape-checks participating keys, then calls `load_state_dict` with
`strict=True`.

The official Hugging Face metadata resolved the full model revision to
`860abec7937da0a4c03c41d3c269c366e82abdf9`. The single `model.safetensors`
file (5,026,367,224 bytes; SHA256
`f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`) and
`config.json` (62 bytes; SHA256
`a73e929a168b900546a84fe88cd70dfaaf2f8e39cf77355b12984eaa686f3855`) are in
the task-owned local cache. CPU strict key/shape loading passed for 1341 source
keys across aggregator, camera head and depth head. The real checkpoint's 456
unused keys are explicitly listed: 62 `point_head` keys and 394 `track_head`
keys. Missing, unexpected and shape-mismatch counts were zero. Parameters are
frozen; the aggregator is BF16 and both heads are FP32. No forward was run.
The task artifact manifest records paths, hashes, source/HF identities and the
full exclusion list. The loader only accepts that manifest, its fixed local
cache and the same revision; the future launcher disables network access.

## Pipeline

`LocusGSObjectLocusFrozenVGGT.generate(context_rgb, runtime_config=None)` accepts
only context RGB `[B,2,3,256,256]` in `[0,1]` and a runtime configuration. It
resizes the same crop to `518x518` with bilinear interpolation and
`align_corners=False`; it does not request another crop or claim new image detail.
The original understanding path continues to receive the 256 crop and performs
its existing resize to 512.

The official aggregator is called once for both views. Its true list positions
`[4,11,17,23]` are selected; `None` values are errors. With `patch_start_idx`
removed, each layer is `[B,2,1369,2048]` (`37x37` patches). Camera/register tokens
remain available to official heads, but do not enter the spatial memory.

The trainable memory adapter computes

```text
F_mix = sum_l softmax(a)[l] * LN_l(H_l_patch)
memory = LN1024(Linear2048_to_1024(F_mix))
```

There are four independent `LayerNorm(2048, eps=1e-5)` modules and four zero
initial logits. `memory` is `[B,2738,1024]` in view-major, then raster order. A
`Linear(1024,2048)` produces K/V, each `[B,16,2738,64]`; K passes through the
existing head-dimension definition, `LayerNorm(64)`. All new linear layers use
Xavier uniform and zero bias. New adapter initialization is isolated in
`fork_rng(seed=31415)`.

The old RGB/Plucker patch embeddings, reconstruction encoder, and old K/V
projection/norm are replaced or removed from this model instance and optimizer.
The existing decoder block objects remain shared with `anchor_decoder`; the
1024 anchors, 64 children, 12 decoder layers, L6/L8/L10/L12 object updates,
`c/s` states, feedback, Gaussian head, MASt3R understanding, panoptic/object
modules, lifting, membership, packed readout, and V3-Set losses are retained.
The legacy model and its default paths are unchanged.

## Frozen and trainable boundaries

All VGGT parameters have `requires_grad=False`; `train()` keeps VGGT in eval mode.
The aggregator is BF16. Camera/depth heads, pose decode, memory adapter, decoder,
Object-Locus, renderer, and losses are FP32. VGGT and independent calibration
passes use `torch.no_grad()` rather than inference mode. The feature layers are
cast to FP32 before entering trainable modules. Camera/depth/scale outputs are
detached; the LocusGS decoder geometry remains trainable.

## Coordinate and camera contract

VGGT pose decoding returns OpenCV world-to-camera extrinsics and pixel intrinsics.
The first predicted context camera defines the generation frame. The two context
depth maps contribute all finite positive pixels to a shared median; the scene
scale is `a_scale=0.25/median_depth`. The same first-camera transform and scale
are applied to camera translations, points, and depth. Invalid depth or camera
values raise errors. There is no GT fallback.

The edge-origin pixel resize transform is recorded as
`A=diag(256/518,256/518,1)` and `K256=A@K518`. Provider-compatible rays sample at
`(column+0.5,row+0.5)`. Renderer inputs use transposed homogeneous world-to-camera
`cam_view` and `[fx,fy,cx,cy]` pixel intrinsics. Memory keys align one-to-one with
the 2738 Plucker rays made from dense `518x518` rays and the existing average
pooling method with `patch_size=14`. Patch rays consume pixel-space `K518`
directly; the renderer and context lifting use pixel-space `K256=A@K518`.
This distinction is enforced in the actual `generate()` call. CPU references
compare directions and moments, verify corresponding continuous pixel
coordinates across resolutions, and show that substituting `K256` for `K518`
fails the reference.

## Training camera calibration and evaluation boundary

Training makes a separate frozen VGGT pass over the existing two-context plus
supervision-view RGB crops. It exposes only calibration cameras and the two
shared-context depth/confidence maps to the calibration wrapper; generated
Gaussians, generation features, and queries never consume this pass. It uses the
fixed shared pixel correspondences and weighted FP64 point Sim(3) specified in
the v2 section below. The resulting transform maps the independent cameras
directly into the context-only normalized generation scene. It does not apply
first-camera normalization or `a_scale` again. The two context loss cameras are
replaced by exact context-only predictions; novel cameras use the independently
aligned calibration pass. Geometry gates fail explicitly without GT fallback.

The loss wrapper shallow-copies the batch and substitutes predicted/aligned
camera matrices, pixel intrinsics, and rays. It leaves provider storage, RGB,
semantic labels, and instance masks untouched.

The official evaluator follows the same boundary: `generate` receives contexts;
an independent camera/depth calibration pass fits shared context points and
calibrates the fixed full-validation views;
`render_generated_at` then re-renders reconstruction and feature channels at
those cameras. It reuses the exact generated Gaussian membership and `p_class`;
it does not regenerate queries, lift understanding features, or call `_readout`
again. The existing exporter writes true-novel target IDs without adding the
first two context IDs to the novel set, and the pinned official SIU3R evaluator
continues to own candidate filtering, packing, void handling, and metrics.
The full-validation pair file is pinned to SHA256
`59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b`: 1860
windows across 312 unique scenes.

“场景生成只使用两张context；监督/目标相机使用独立图像标定。指定新视角渲染仍需要目标相机。”

## Initialization and fixed eight-card training plan

The epoch-06 source checkpoint is
`/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt`.
Its actual SHA256 was CPU-verified as
`68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a`; metadata
reports epoch 6, 6258 updates, and 50064 exposures. The new training clock starts
at zero. Old optimizer, scheduler, and RNG state are not restored. Migration
uses a retained-prefix whitelist, per-key shape checks, explicit copies, and
reports loaded, removed, new, missing, shape-mismatch, and unknown keys. The
retained families are GS tokens, decoder blocks, anchor geometry/PE/refinement,
Gaussian head, all understanding parameters/buffers, and panoptic/object
parameters/buffers. Removed keys are the legacy image/Plucker embedding and old
reconstruction encoder/K/V families. Only the adapter/KV is newly initialized
inside the Object-Locus model; VGGT parameters come from the separately
identified official artifact.

CPU construction with a controlled VGGT stub matched all 1377 retained
checkpoint keys and shapes, explicitly excluded 68 removed reconstruction keys,
and found no missing, mismatched, or unclassified checkpoint keys. The full
key-by-key mapping is included in
`object_locus_frozen_vggt_posefree_weight_mapping.json`.

The existing full1201 manifest is read in place: 1191 training scenes, 8337 fixed
windows, two contexts per window. No resampling is performed. The reviewed resource
adaptation uses `3dimage-13` with eight RTX3090 workers, per-rank microbatch 1,
accumulation 1 and global batch 8. Each update consumes eight consecutive entries
from `default_rng(42+epoch)`; rank `r` consumes the entry at offset `r`. Each epoch
pads to 8344 exposures / 1043 updates; totals remain 8344 updates and 66752 new
exposures. Understanding weight is
`min(exposure/200,1)` and beta is `0.1*min(exposure/1000,1)`. Gradient composition
is `g_rec + 0.01*g_under` for reconstruction parameters and
`g_rec + g_under` for understanding/object/memory parameters. Peak LRs are
`1e-5` reconstruction, `1e-5` understanding, `1e-4` object and new memory; AdamW
inherits the GC baseline decay exclusions (reconstruction remains no-decay;
understanding and object/memory use the baseline matrix/bias/norm exclusions),
warm-up 200 updates, cosine to 10% of
peak, and global clip 1. TF32 is disabled by the planned launcher. No new
Chamfer, opacity-floor, or depth loss is added.

## Validation limits

CPU contracts cover actual layer indexing/shapes, view/raster ordering, adapter
and K/V shapes, projection/backprojection, first-camera normalization/scale,
518-ray to 2738-patch alignment, orientation-aware Sim(3) recovery and degenerate
errors, explicit migration/optimizer exclusions, exposure/LR/sampler/GC math,
and a controlled VGGT stub. Stub results are not real VGGT validation. CPU
construction of the actual Object-Locus model matched and copied all 1377
retained checkpoint entries against the epoch-06 source; 68 removed entries were
excluded with no unknown, missing, or mismatched keys. The migration source
remains epoch 6 / 6258 updates / 50064 exposures; optimizer, scheduler, and RNG
are fresh. Resume checkpoints exclude frozen VGGT tensors, strictly restore all
other keys and shapes, retain optimizer and per-rank RNG state, and record source
and new exposure clocks separately. A CPU miniature checkpoint test verifies
parameters, optimizer state, clock, and the next RNG sample.

The real-weight GPU smoke entry points run first on one allocated RTX3090, then
eight RTX3090s. Both use the actual factory, local official artifact, fixed epoch-0
windows, two optimizer updates, generation/render checks, gradient-path probes,
and isolated outputs. Eight-card smoke records every rank. The eight-rank training runtime syncs
batch, forward, autograd, finite-gradient, and optimizer errors across ranks;
it stores a rolling atomic latest checkpoint and epoch-4/8 endpoints. It records
detached losses, exposure, beta, group LRs, preclip norm, and GPU memory. The
launcher does not submit jobs.

The real official VGGT weight file was CPU-loaded with strict key and shape
checks. The adapted CPU suite covers 20 contracts, including patch/view order with spatially
varying features, actual `generate()` K wiring, orientation-constrained Sim(3)
with a nontrivial first camera, degenerate and valid small baselines, the
eight-rank gradient average (one sample per rank) against an eight-sample reference, and small-model
checkpoint/RNG restoration. It does not establish real GPU operator
compatibility.

The prior design review did not run GPU smoke or training. This resource-adaptation
revision prepares a single Slurm job that runs single-card smoke, eight-card smoke,
then fresh training (or strict compatible eight-card resume when a valid latest
checkpoint already exists). The job writes startup confirmation after 20 synchronized
updates / 160 exposures. Formal evaluation remains a separate, unlaunched step.

## Approved launch and later evaluation commands

The task launcher sets the pinned source/cache paths and disables HF network
access. The approved launcher targets `3dimage-13`; it submits one 8-GPU job with
32 CPUs, 128 GiB host memory and a 48-hour walltime:

```bash
scripts/submit_object_locus_frozen_vggt_posefree_v1.sh
PYTHONPATH=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets/vggt_source_checkout:/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1 HF_HUB_CACHE=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets/hf_cache/hub HF_HUB_OFFLINE=1 python scripts/eval_object_locus_frozen_vggt_posefree_v1.py --checkpoint /space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1/checkpoint_epoch_08.pt --manifest /space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json --cohort full_validation --output-root /space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1_eval_epoch08 --vggt-revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --artifact-manifest /space/mawb/ssst_object_locus_frozen_vggt_posefree_v1/vggt_artifact_manifest.json --device cuda
```

## Shared-context depth point calibration v2 (2026-10-09)

The reviewed calibration protocol is now `shared_context_depth_sim3_v2`. The
independent full-window VGGT pass runs its aggregator once over context plus
supervision RGB, decodes all cameras, and applies the official depth head to the
first two view tokens from that same aggregated window. Because global/frame
attention may mix information from all input views, the independent context
features and resulting depth can depend on target RGB; these tensors remain
calibration-only and never enter `generate` or its Gaussian/query/membership
state.

At fixed rows/columns `7 + 14*i`, pixel centers in 518 coordinates, the
independent context axial depth and K518 produce raw-world points X. The
context-only pass's `predicted_points` provide target points Y directly in the
first-context normalized scene. Confidence scores from both passes are finite,
positive validity signals and are converted to within-view empirical midranks;
they are not interpreted as probabilities. The fixed base confidence weight is
`0.05 + 0.95*sqrt(pct_A*pct_B)` and each shared view contributes total weight
0.5. A weighted FP64 Umeyama Sim(3) is fit after weighted centering and RMS
normalization, followed by exactly one initial fit and five fixed Huber IRLS
refits. Normalized source and target covariance must have second/first
 eigenvalue ratio at least `1e-6`; planar sets are allowed, collinear sets fail.
The transform maps independent raw world points and cameras directly into the
context-only scene, so there is no second first-camera inverse or median-depth
scale application.

Every window must have at least 32 valid sampled correspondences in each view,
positive Z ratio at least 95%, reprojection median at most 4 pixels and p90 at
most 12 pixels at 256 resolution. The fit checks finite camera/transform values,
SO(3) at FP64 tolerance `1e-8`, and positive scale. Failure is a geometry block;
there is no GT fallback, window skipping, threshold relaxation, or old-method
fallback. Reports include per-view point counts, 3D residuals normalized by the
context scene median depth, reprojection residuals, positive-Z counts, confidence
weight sums and pre-override shared camera differences. Fixed window 4253 saves
old signed-baseline diagnostics and compact point/camera evidence; each training
window only adds compact scalars unless calibration fails.

The ordered 3dimage-13 job first ran fixed window 4253, then the fresh one-card
and eight-card real smokes. Window 4253 and the one-card smoke passed. The
eight-card smoke hit the fixed v2 geometry gate on zero-based update 1, rank 5,
scene `scene0563_00`: view 0 reprojection median/p90 were 4.837/21.183 pixels;
view 1 median was 4.602 pixels. The run stopped with `GEOMETRY_BLOCKED` before
formal training. The old method's signed scale on window 4253 was +1.151038197,
so the prior training error was not reproduced by that diagnostic window. The
full per-window evidence is retained outside Git under
`/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2/attempts/59658/`.
No thresholds were changed, no windows skipped, and no GT fallback applied.
