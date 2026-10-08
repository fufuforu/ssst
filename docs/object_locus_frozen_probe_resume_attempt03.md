# Frozen Probe + R3D attempt03 cache-resume execution record

This recovery reuses the completed immutable A/B data from attempt02. GC001
extraction and R3D epoch8 inference remain attributed to execution SHA
`6489441d1814af23cafab9a9170ba77c4b1e5da3`; probe-head training and unified
evaluation use the new committed execution SHA recorded in attempt03's
`execution_code_identity.json`. No A/B job is resubmitted and no source-model
checkpoint is loaded or forwarded by the resume preparation or entrypoint
preflight.

The C parent, each H1/H2/H3 worker, D evaluator, report, and execution-manifest
verifier start as Python modules from the locked repository root. Each worker
command is constructed by `scripts.train_parallel_probe_heads.worker_command`
and that same function is used by the formal subprocess launch. Workers retain
their Slurm-assigned one-device visibility and select `cuda:0` within that
allocation. The report is launched with the locked SIU3R Python environment.

The resume preparation checks the original A/B receipts and cache manifests,
copies small metadata and labels byte-for-byte, and creates per-file symbolic
links for immutable feature, IoU, reconstruction, and GPU-export data. It
never links the labels directory as a whole. Attempt03 contains fresh head,
label-output, prediction, report, and Slurm paths. The original attempt02
execution records are retained under `provenance/source_attempt02/`.

The fixed scientific protocol, model endpoints, seeds, loss, budgets, official
evaluation, reconstruction protections, and paired bootstrap remain inherited
from the parallel R3D evaluation protocol and the attempt02 finite-resume
protocol. This document records only the module-entrypoint repair and reuse of
validated A/B outputs.
