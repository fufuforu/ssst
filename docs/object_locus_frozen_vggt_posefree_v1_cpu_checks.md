# CPU checks for Frozen VGGT Pose-Free v1

Run in `/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1` with the existing
`tokengs` Python environment. `CUDA_VISIBLE_DEVICES` was empty. No CUDA kernel
or GPU forward/backward ran. CPU forwards were limited to controlled stubs in
the contracts; the real VGGT artifact was only loaded and key-audited. No
training, formal evaluation, or Slurm submission ran.

## Contracts

Command:

```bash
CUDA_VISIBLE_DEVICES= TORCHINDUCTOR_COMPILE_THREADS=1 /space/mawb/anaconda3/envs/tokengs/bin/python -u scripts/smoke_object_locus_frozen_vggt_posefree_v1.py --cpu-contracts
```

Result: `Ran 18 tests ... OK` (exit 0).

The contracts cover actual `generate()` use of 518 intrinsics for 14-pixel
patch rays, independent ray/moment references, 518-to-256 continuous pixel
projection, the wrong-`K256` failure case, view/raster ordering with spatially
varying features, context-only API/legacy-camera rejection, target rendering
with fixed membership and classes, known orientation-constrained Sim(3), zero
and valid small baselines, optimizer exclusions, four-rank/two-microbatch GC
against an eight-sample gradient reference, and small checkpoint/optimizer/RNG
restore.

## Real official VGGT artifact (CPU load only)

Command:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets/vggt_source_checkout:/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1 HF_HUB_CACHE=/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets/hf_cache/hub /space/mawb/anaconda3/envs/tokengs/bin/python -u scripts/audit_frozen_vggt_artifact_cpu.py --artifact-manifest vggt_artifact_manifest.json --output /tmp/object_locus_final_vggt_audit.json
```

Result: `PASS_CPU_STRICT_KEYS_NO_FORWARD`; source commit
`a288dd0f14786c93483e45524328726ab7b1b4ce`; HF revision
`860abec7937da0a4c03c41d3c269c366e82abdf9`; loaded 1341/1341 keys with strict
key/shape checks; excluded 456 keys (point 62, track 394); missing 0, unexpected
0, shape mismatch 0. Aggregator BF16; camera/depth heads FP32; all parameters
frozen and eval. The real checkpoint exclusion names and source-file hashes are
in `vggt_artifact_manifest.json`. No model forward ran.

## Entry points and syntax

Passed with `CUDA_VISIBLE_DEVICES=`:

- `python scripts/eval_object_locus_frozen_vggt_posefree_v1.py --help`
- `python scripts/train_object_locus_frozen_vggt_posefree_v1.py --help`
- `python scripts/smoke_object_locus_frozen_vggt_posefree_v1.py --help`
- `bash -n scripts/run_object_locus_frozen_vggt_posefree_v1.sh`
- Python `py_compile` for the changed pose-free model, geometry, runtime,
  evaluator, smoke/train/audit entries, and contract test module.

The epoch-06 migration source checkpoint SHA and the 1377-loaded/68-excluded
strict CPU migration report are retained in
`docs/object_locus_frozen_vggt_posefree_weight_mapping.json` and
`review_manifest.json`; the source checkpoint itself is not part of this repo.
