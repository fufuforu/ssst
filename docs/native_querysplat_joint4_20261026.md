# Native QuerySplat with Baseline4 understanding on full ScanNet

This experiment implements the user's two authorized next steps: score the
completed 4096-token Baseline4 model against the existing 1024-token result,
and train native QuerySplat reconstruction with Baseline4 understanding on
all 8337 windows / 1191 scenes for four epochs (4172 global-batch8 updates).

The native files under third_party/native_querysplat are byte-identical to
upstream commit 3465a1d2c789d8ebe71f4f9c0bae3f2c2fd726ae. native_scannet/core
retains the previously validated CachedEncoder and activation-checkpoint
wrapper, but supplies raw frozen features online; a second frozen backbone
is not instantiated in the reconstruction branch. Native width2048,
1024 queries, 64 Gaussians/query, 12 geometry and 6 appearance blocks, SH1,
camera/register/eye tokens, live layer mixing, RGB ImageNet normalization,
separate RGB/ray patch normalization and geometry-to-appearance coupling
are unchanged. Hooks only retain detached L6/8/10/12 outputs for understanding.
Original full/native random parameters are initialized with seed42, never
restored from the 8-window fitting checkpoint or the failed D variant.

The understanding initializer, semantic memory, controller and GFM projection
retain the original Baseline4 keys/shapes; 227 retained tensors are copied
exactly from the original ZERO checkpoint, not the trained Baseline4. Dynamic
256D object queries, ESAM SAM partitions, geometric No-Anchor-CA routing,
FM attention, child feature lifting and source-many-to-one losses remain.
The sole new learned interface is LN2048 + Linear2048->1024 on detached states.
No reconstruction geometry, tokens or mask rasterization receive LU gradients.
No understanding state feeds native Gaussian generation. Read-only intermediate
native geometry heads provide anchor centers/radii; reconstruction supervision
uses the final native output, without the previous synthetic intermediate loss.

Native coordinates retain input predicted depth median1. Public/understanding
coordinates retain baseline median.25. GS positions/scales and camera
translations are divided by4, near/far are divided by4, SH/opacity/rotation
and pixel intrinsics stay fixed. RGB is rendered by the original native SH
renderer after the inverse conversion; returned depth is in public .25 units.
Understanding feature rasterization uses the same converted geometry/clips.
CPU projection invariance is tested; GPU RGB/alpha/depth/gradient equivalence
is required by the dependency gate. The gate also compares native manual
selected-layer inference with the actual online frozen aggregator outputs.

Training uses the successful independent native loss: L1+.2SSIM+visibility,
early bidirectional squared Chamfer1 and opacity-floor.1 regularizers ending
at279/4172 updates, and LPIPS ramp0->.05 from1/3 to1/2 of the budget. This is
the same relative-budget adaptation used in the 512-step learning validation,
not claimed to be unpublished author training code. Baseline shared-input
Sim3 calibration pins Context cameras exactly; only Context RGB/features
produce the scene. Novel RGB supplies the original training supervision and
independent target-camera prediction, not generated GS or object queries.

Both branches use AdamW1e-4, betas .9/.95, eps1e-8, original matrix decay .05
and zero decay for bias/norm; native warmup is5%=209 updates, understanding
preserves the baseline100-update LR warmup. Cosine floor.1, understanding
loss ramps to1 across1043 updates. Clip norm1 is applied separately to the
two branches so LU magnitude cannot attenuate native updates through joint
clipping. ZeRO shards Adam states across8 GPUs; native decoder training uses
BF16 autocast/checkpointing with FP32 weights, understanding remains FP32,
and evaluation uses FP32. EMA.9995 is maintained on CPU and saved separately;
formal comparison uses the final live epoch4 weights without test selection.

The eight-GPU two-step engineering gate does not supply training weights.
Main training starts fresh and automatically exports/scores the same fixed
1860 windows /312 scenes as Baseline4. Both experiments report the existing
11 Native metrics with exact input and GT SHA equality, plus pure Novel4
reconstruction metrics. The existing 1024 export supplies reference PNGs;
no reference retraining or quality/language head is introduced. Source-data,
frozen assets, code SHA and run configurations are sealed before submission.

CPU tests establish unchanged native forward/all parameter gradients at a
microarchitecture, detached semantic reads/no write, no Novel RGB/target-camera
leak into native generation, coordinate projection invariance, expanded native
schedule, exact upstream source bytes, and independent-clipping update
invariance. Existing baseline/ABCD tests also pass. CPU preflight constructs
the full native model on meta and performs real CPU zero-tensor copies; it
does not claim a native-width GPU pass. Runtime receipts are authoritative.
