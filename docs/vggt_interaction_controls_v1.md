# Four short decoder interaction controls

Authorized on 2026-10-13. These diagnostics run alongside full adapter0.01
rerun63552 and its official evaluation63553. They do not alter those jobs.

All arms restore the exact original reconstruction-adaptation epoch2 boundary
`a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c`
and adapted VGGT asset
`c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace`.
They retain the original LocusGS-derived decoder weights, pretrained MASt3R
understanding, task heads, losses, calibration, global batch8, native GPU
AdamW, GC and precision. VGGT (including AA) is frozen; understanding remains
trainable. Adapter and GS receive `g_rec+0.01*g_under`; understanding/panoptic
receive `g_rec+g_under`. The original exposure warmup starts at zero, with
fresh optimizer and exact four-rank adaptation-boundary RNG. Learning rates
follow the original200-update warmup and4172-update joint cosine, truncated
at500 updates. This avoids conflating decoder design with a changed schedule.

1. `bidirectional`: original mask-space dot-product feedback, unchanged.
2. `forward_only`: GS→query reads and full understanding loss retained;
   query→GS token injection exactly zero throughout training and evaluation.
   The understanding gradient to GS/adapter remains at0.01 through query reads
   and segmentation; this is a feedback ablation, not an understanding-loss ablation.
3. `gated_feedback`: original routing, then a learned sigmoid gate per GS token
   after the original bounded injection. Gate sees normalized token/message
   features; initialized0.9. It can learn selection as well as attenuation.
   Any gain alone does not distinguish learned selection from attenuation;
   a fixed-gate comparison would be a subsequent experiment.
4. `independent_feedback`: dedicated eight-head Q/K/V/O attention with GS
   anchor embeddings as queries and updated object queries as keys/values.
   Preserve geometric bias and an explicit zero key/value null choice. Routing
   no longer uses the segmentation mask embedder. Original output projection,
   tanh, norm cap and beta remain. Added parameter counts are recorded; this
   comparison tests routing design together with its added capacity.

All four keep interaction layers6/8/10/12. Every-layer experiments are deferred.
Existing production constructors instantiate no new modules and retain the
baseline calculation. Optional modules are initialized under isolated seed314159
after strict source restoration. All source parameters remain identical.

Frozen cohort: `docs/vggt_interaction_cohort_v1.json`, seed42 without replacement,
128 training scenes with four windows each (512 unique windows), and64 held-out
validation scenes with one official six-view window each. No score filtering.
Each arm uses the same shuffled512-window stream,500 updates/4000 exposures.
Checkpoints0/20/100/300/500:20 onward include optimizer, four-rank RNG and exact
sampling ownership. Progress files contain ETA and per-rank geometry/interaction
statistics. `monitor_v1` warnings do not stop training; hard numerical errors do.

Validation: CPU graph/sampler contracts, isolated real one-card and four-card
two-update preflights for all arms, mandatory native window6923 in four-card
preflight, and one real official13-column score per arm/preflight. Formal jobs
use the same pushed immutable commit and reset smoke optimizer/clocks.
Use four RTX4090 per arm, preferentially two jobs on node14 and two on node17,
240GiB host memory/job. Official evaluations request four cards, eligible on
14/17/18, with afterok dependencies. Each evaluates0/100/300/500 on all64 scenes;
AP/PQ are computed globally per condition by pinned unmodified SIU3R evaluator.
All results use one13-column row; both text mIoUt columns are empty.
These are diagnostic scores, not full312-scene validation. Automatic reduction
verifies identical GT and identical initial RGB across all arms (beta0), then
produces the combined table. No metric improvement is presumed before scores.

Evidence: `/space/mawb/ssst/group_plus/vggt_interaction_controls_v1/attempts/attempt01`.
Checkpoints: `/space/mawb/ssst/workspace_group_plus/vggt_interaction_controls_v1/attempt01`.
After startup confirmation of20 formal updates/160 new exposures per arm,
stop interactive polling and leave training/evaluation running.

## Directory synchronization repair

Initial commit `a7eec2f4ca91f96c6152d3ad21f339f0df36314b`: all four
single-GPU numerical and official-score preflights succeeded in63561.
Four-GPU63562 passed the bidirectional arm (including6923 and one geometry
warning), then failed before forward-only initialization because rank0 created
the new output directory before rank1 checked whether it already existed.
No formal update was submitted. Attempt01 and all logs remain intact.
The repair makes rank0 solely responsible for checking/creating the path and
broadcasts success/error through the native synchronized-error helper. No model,
data, RNG, precision, loss or optimizer computation changes. Reuse63561's
explicit original SHA single-GPU evidence; rerun all four distributed arms
under a new pushed immutable snapshot in attempt02. The launcher records this
engineering-only compatibility and never relabels63561 as a new-SHA run.

Node14 became fully occupied during preflight. Formal jobs therefore wait for
any four RTX4090 on the already authorized14/17/18 pool rather than binding
two arms to busy14. All numerical model/runtime/shell/cohort files stay
byte-identical to four-GPU preflight commit
`d4a1638dbb43d28581a0bd4c006db4cda08c8b8e`; the launcher checks the git diff
and accepts only launcher/docs changes. Actual proof SHAs remain explicit.
Pin the exact existing `gsplat_cuda_partial_1406/gsplat_cuda.so` binary loaded
by the accepted node17 preflights across all hosts; record its hash to avoid
host glibc selecting a different cached binary. This changes scheduling and
asset selection determinism, not renderer operations or training mathematics.
