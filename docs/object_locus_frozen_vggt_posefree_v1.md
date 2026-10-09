# Object-Locus Frozen VGGT Pose-Free v1

## Review status and fixed identity

This branch implements the VGGT-assisted pose-free migration path on top of
`object-locus-gc-sweep-v1` at `9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0`.
The independent branch is `object-locus-frozen-vggt-posefree-v1`.

The official VGGT source repository is
`https://github.com/facebookresearch/vggt`, source commit
`a288dd0f14786c93483e45524328726ab7b1b4ce` (remote `main` SHA observed during
implementation). The model identifier is `facebook/VGGT-1B`. The wrapper loads
only official aggregator, camera head, and depth head weights; it constructs the
official model with point and track heads disabled, explicitly records their
source keys, shape-checks participating keys, then calls `load_state_dict` with
`strict=True`.

**VGGT artifact identity is incomplete for this review.** This host could not
reach Hugging Face because its configured proxy refused the connection. No HF
revision or actual model-file SHA256 was therefore observed, and no VGGT weights
were downloaded. Runtime construction requires a pinned `VGGT_HF_REVISION`, a
local HF cache (`local_files_only=True`), and an imported source Git checkout at
the exact source commit. `FrozenVGGT.from_pretrained` records
resolved snapshot revision and SHA256 for each model file when the official
artifact is available. This source commit does not substitute for model-file
identity.

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
pooling method with `patch_size=14`.

## Training camera calibration and evaluation boundary

Training makes a separate frozen VGGT pass over the existing two-context plus
supervision-view RGB crops. It exposes only calibration cameras to the loss
wrapper; generated Gaussians, features, depth, and query never consume this pass.
The shared context camera orientations determine the SO(3) Procrustes rotation;
the shared context baseline determines a positive least-squares scale; shared
centers determine translation. The target cameras are transformed into the
context-only first-camera frame and receive `a_scale`. The two context loss
cameras are replaced by exact context-only predictions. Novel cameras use the
independently aligned calibration pass. Degenerate baselines, nonpositive scales,
and nonfinite cameras raise errors.

The loss wrapper shallow-copies the batch and substitutes predicted/aligned
camera matrices, pixel intrinsics, and rays. It leaves provider storage, RGB,
semantic labels, and instance masks untouched.

The evaluation helper follows the same boundary: `generate` receives contexts;
an independent camera-only pass calibrates the requested context/target images;
`render_generated_at` then accepts that target camera for rendering.

**Scene generation uses only two context views; supervision/target cameras use
independent image calibration.** This does not mean that a requested novel view
can be rendered without its target camera.

## Initialization and training plan (not executed)

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

The existing full1201 manifest was read in place: 1191 actual training scenes,
8337 fixed windows, two contexts per window. No resampling is performed. Planned
phase one is 8 epochs, global batch 8, four RTX4090 workers, rank batch 1,
accumulation 2. Each epoch uses `default_rng(42+epoch)` permutation, padded by
repeating its prefix to 8344 windows / 1043 optimizer updates. Totals are 8344
updates and 66752 exposures. Understanding weight is
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
excluded with no unknown, missing, or mismatched keys. No real VGGT checkpoint,
HF-unused-head key inventory, or GPU forward was available for verification;
the runtime enumerates and records exact point/track source keys when the
pinned official artifact is loaded.

No GPU smoke, training, evaluation, or Slurm submission is part of this review.
