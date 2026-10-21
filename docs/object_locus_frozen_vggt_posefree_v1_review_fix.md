# Frozen VGGT pose-free v1 review fixes

Base reviewed: `bd3329a110803e3750aaeef573650237e536f9e2`.

| Original issue | Fix | Evidence |
|---|---|---|
| `generate()` passed 256-resolution intrinsics to 518-resolution patch rays. | The patch-ray API names and consumes `K518`; `generate()` passes `vggt_result.intrinsics518`. Renderer and context lifting retain `K256=A@K518`. | CPU reference compares actual generation wiring, direction and moment pooling, view/raster order, continuous corresponding pixels, and confirms the old `K256` wiring fails. |
| Sim(3) compared a squared-distance denominator with a length threshold and could substitute a normalized camera inverse. | Independent and context baselines are checked as lengths against the same `1e-6`; the required raw first-camera inverse is explicit. R/s/t, source cameras and aligned cameras are checked for finite values. | CPU tests recover a known orientation-constrained transform with a nonidentity first camera, reject zero baselines, and accept a valid `2e-4` baseline whose square is below `1e-6`. |
| Pose-free evaluation rendered only RGB and could fall through inherited GT-camera forward aliases. | Target re-rendering returns RGB, fixed-membership alpha/region/pixel membership and semantic scores using stored `gaussian_membership` and `p_class`. Legacy ModelInput forward aliases raise. The CLI exports fixed full-validation true-novel views through the existing packed exporter and official evaluator. | Controlled renderer verifies one render, identity-preserved membership/class tensors, no regenerate/readout; CLI/export contract verifies fixed IDs and target files. No evaluation was run. |
| Training held two full-model gradient accumulators and lacked synchronized stage errors. | Each microbatch computes `g_rec` and weighted `g_under`, GC-combines them into one accumulator, frees graph outputs, then rank-averages, clips, and steps. Batch/forward/autograd/local-finite/optimizer failures synchronize across ranks. | Current CPU reference matches eight ranks × one sample against direct eight-sample gradients for all parameter families and `None`/frozen parameters. GPU execution is delegated to the ordered 3dimage-13 smoke job. |
| Runtime could not exactly restore the new training run. | Atomic rolling latest plus epoch-4/8 endpoints store non-VGGT state, optimizer, clocks, sampler position, manifest/code/source/artifact identity, and per-rank RNG. Resume strictly checks keys/shapes and restores the new exposure clock without re-warm-up. | CPU small-model checkpoint test verifies model and optimizer state, update/exposure clock, and next RNG value. |
| There was no real-weight smoke entry or verified official artifact gate. | Separate single-/eight-card smoke modes use the actual factory, fixed windows and verified local cache. Formal loader no longer has an unpinned default model constructor; injected models require explicit test-only permission. | CPU CLI/help and stub contracts only; real official weights are separately audited in `vggt_artifact_manifest.json`; GPU smoke remains not run. |

## Run boundary

This change is implementation and asset preparation. It does not perform GPU
forward/backward, training, formal evaluation, or Slurm submission. See
`object_locus_frozen_vggt_posefree_v1.md` for the fixed design and deferred
single-card, eight-card, training, resume, and evaluation commands.


## 8xRTX3090 launch adaptation

| Previous constraint | Adaptation | CPU evidence / execution evidence |
|---|---|---|
| Runtime and launch were fixed to 4xRTX4090, accumulation 2, node 14. | Runtime, plan, smoke CLI, and one ordered Slurm job now target eight RTX3090s on node 13, microbatch 1, accumulation 1, global batch 8. Each rank takes its matching entry from each consecutive padded block of eight. | CPU contracts compare eight rank shards and the averaged GC gradients with direct references; `3dimage-13` was observed idle with `gpu:3090:8`. |
| Smoke reporting retained only rank 0 and did not encode the new exposure clock. | The single smoke records 2 exposures; eight-rank smoke records all ranks and 16 exposures; formal run writes startup confirmation only after eight ranks synchronize update 20 (160 exposures). | CPU tests check config and smoke clocks; real GPU results and startup confirmation are written by the authorized ordered Slurm run. |
| Resume required the exact prior Git SHA. | Resume permits pure code/logging/interface revisions while requiring matching manifest, eight-rank recipe, artifact identity, strict non-VGGT keys/shapes and sampler boundary. | Resume configuration and eight RNG slots are covered by CPU contracts; formal resume is selected only when an eight-rank latest checkpoint exists. |
| Full-validation count was reported as scenes and reconstruction quality metrics were omitted. | Manifest now records 1860 windows / 312 unique scenes. Pose-free evaluator saves RGB/depth reconstruction caches and reduces context, target-all and true-novel PSNR/SSIM/LPIPS and SIU3R per-image scale-and-shift AbsRel/RMSE. | CPU cache contract checks view IDs and RGB/depth shapes. Formal evaluation remains unlaunched. |

## Shared-context depth calibration v2

| Original issue | Fix | Evidence |
|---|---|---|
| The old orientation-plus-baseline fit produced a scale error on fixed window 4253; its actual signed value had never been captured. | Added a non-throwing old-formula diagnostic with raw cameras, baselines, lengths, SO(3), numerator/denominator, signed scale, angle, center-formula comparison, finite flags and nonfinite element indices. The 4253 runtime writes this before attempting v2. | CPU diagnostic verifies signed scale/center formula agreement and records a NaN camera index without aborting. The real 4253 values await the node-13 run. |
| A camera baseline alone can fail to align two independent VGGT passes. | Replaced training calibration with fixed shared-pixel depth correspondences, per-view confidence midrank weights, FP64 normalized weighted Umeyama and exactly five Huber IRLS refits. Cameras map directly to context-only generation coordinates. | CPU fixtures recover a known transform, accept planar data, reject collinear/missing data, and robustly handle fixed outliers. Actual threshold compliance awaits fixed-window real validation. |
| A target calibration call could alter generated prediction state or fresh training could bypass required real validation. | Calibration asserts exact context RGB reuse, uses a separate aggregator pass, does not expose those features to generate, and CPU checks generated tensors remain unchanged. Fresh training requires successful 4253, single-card and eight-card reports as explicit inputs. | 24 CPU contracts pass; GPU stage reports are still pending. |

The fixed v2 thresholds are documented in
`object_locus_frozen_vggt_posefree_v1_calibration_v2.md`; they are treated as
stop conditions, never tuned from observed results.
