# Frozen VGGT pose-free v1 review fixes

Base reviewed: `bd3329a110803e3750aaeef573650237e536f9e2`.

| Original issue | Fix | Evidence |
|---|---|---|
| `generate()` passed 256-resolution intrinsics to 518-resolution patch rays. | The patch-ray API names and consumes `K518`; `generate()` passes `vggt_result.intrinsics518`. Renderer and context lifting retain `K256=A@K518`. | CPU reference compares actual generation wiring, direction and moment pooling, view/raster order, continuous corresponding pixels, and confirms the old `K256` wiring fails. |
| Sim(3) compared a squared-distance denominator with a length threshold and could substitute a normalized camera inverse. | Independent and context baselines are checked as lengths against the same `1e-6`; the required raw first-camera inverse is explicit. R/s/t, source cameras and aligned cameras are checked for finite values. | CPU tests recover a known orientation-constrained transform with a nonidentity first camera, reject zero baselines, and accept a valid `2e-4` baseline whose square is below `1e-6`. |
| Pose-free evaluation rendered only RGB and could fall through inherited GT-camera forward aliases. | Target re-rendering returns RGB, fixed-membership alpha/region/pixel membership and semantic scores using stored `gaussian_membership` and `p_class`. Legacy ModelInput forward aliases raise. The CLI exports fixed full-validation true-novel views through the existing packed exporter and official evaluator. | Controlled renderer verifies one render, identity-preserved membership/class tensors, no regenerate/readout; CLI/export contract verifies fixed IDs and target files. No evaluation was run. |
| Training held two full-model gradient accumulators and lacked synchronized stage errors. | Each microbatch computes `g_rec` and weighted `g_under`, GC-combines them into one accumulator, frees graph outputs, then rank-averages, clips, and steps. Batch/forward/autograd/local-finite/optimizer failures synchronize across ranks. | CPU reference matches four ranks × two microbatches against direct eight-sample gradients for all parameter families and `None`/frozen parameters. GPU execution remains not run. |
| Runtime could not exactly restore the new training run. | Atomic rolling latest plus epoch-4/8 endpoints store non-VGGT state, optimizer, clocks, sampler position, manifest/code/source/artifact identity, and per-rank RNG. Resume strictly checks keys/shapes and restores the new exposure clock without re-warm-up. | CPU small-model checkpoint test verifies model and optimizer state, update/exposure clock, and next RNG value. |
| There was no real-weight smoke entry or verified official artifact gate. | Separate future single-/four-card smoke modes use actual factory, fixed windows and verified local cache. Formal loader no longer has an unpinned default model constructor; injected models require explicit test-only permission. | CPU CLI/help and stub contracts only; real official weights are separately audited in `vggt_artifact_manifest.json`; GPU smoke remains not run. |

## Run boundary

This change is implementation and asset preparation. It does not perform GPU
forward/backward, training, formal evaluation, or Slurm submission. See
`object_locus_frozen_vggt_posefree_v1.md` for the fixed design and deferred
single-card, four-card, training, resume, and evaluation commands.
