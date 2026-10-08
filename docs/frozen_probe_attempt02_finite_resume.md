# Frozen Probe + R3D: attempt02 finite resume

This execution record implements the supplied finite-resume instructions and inherits the locked science and delivery protocol from `object_locus_frozen_probe_parallel_r3d_eval_codex_prompt.md`.

- New output root: `/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02`.
- Preserve attempt00 and attempt01. Do not reuse attempt01 partial training/cache as a complete run.
- Work from branch `object-locus-frozen-representation-diagnostic-v1`, initially verified at `a951ec86daf6d9752ca97d4832a0ab461cdd46d6`; commit/push all recovery fixes before the new A/B/C/D chain and record the final full SHA.
- Before submission, execute `compare_replay` on attempt01's actual first dev window inputs and validate the actual R3D registration, run manifest, completion/progress/roundtrip receipts, and epoch8 checkpoint metadata/hash.
- Keep one common CPU-only R3D registration parser for inference job B and report job D. Preserve the registered thresholds and failure clauses exactly; missing required reconstruction metrics produce an incomplete evaluation status.
- Run on `3dimage-11` / `3090`: A 1 GPU, B 1 GPU, C 3 independent head workers afterok A, D CPU afterok A:B:C. No requeue or automatic retry after a failed stage.
- The probe remains H0 plus H1/H2/H3 for seeds 20261, 20262, 20263. R3D remains read-only inference from its registered epoch8 checkpoint.
- Each Slurm process receives the same explicit `TASK_ATTEMPT_ROOT`; compute nodes verify the submitted Git SHA and immutable per-file manifest without checkout/fetch/pull/reset.

The user supplied this recovery instruction in the active task. The prior full parallel/R3D prompt and scientific protocol remain unchanged and are hashed in `effective_execution_protocol.json`.

## Recovery interface findings

- attempt01 `scene0011_00_context68_87` was reconstructed from the saved `cohort_identity.json` dev[0] entry. The real `compare_replay` call passed with 1 window, 2 scopes, 600 query-views, and candidate audit SHA256 `a12412bed631af803921fe1b45d12f941a88e564e0c4ff7ad5312df25bd182d9`. Its existing PNG/GT/prediction checks were not loosened.
- The R3D registration stores the plan hash at `fixed_training_plan_sha256`, logical slots under `training.logical_global_slots`, and physical/accumulation values in `run_manifest.json`. Its existing `success` object is compatible with the report's six threshold accesses. All original values and failure clauses were checked exactly.
- R3D outcome calculation now consumes a shared normalized registration. An incomplete protocol returns `INVALID`; missing required evaluation metrics return `INCOMPLETE` with no algorithm conclusion. The registered four algorithm branches were exercised using synthetic metric records.
- GC001 and R3D checkpoint files, the shared source checkpoint, and training plan were SHA256 rechecked against their locked identities. No checkpoint/model forward was used for the CPU replay preflight.
