# Source audit, 2026-10-20

Read base b7a775f19e232ea2395c8611a47c81b1365fb86e and the actual sem256 execution
directory `/space/mawb/ssst_sem_res256_code_20261019`. Existing run receipts and
handoff were read; historical RUNNING statements are not completion evidence.
No existing job was restarted, cancelled, polled, or otherwise changed.

## Final checkpoint identity

- `/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu/checkpoint_epoch4.pt`
- SHA256 `18aee68f2bc8cdb1d154c742697ccc54657f38ad02c6151bf993405a455f4560`
- Internal progress: epoch4, update_in_epoch0, full_updates4172. All optimizer
  state steps are 4172. TRAINING_COMPLETE.json and COMPLETE.json agree.
- Internal code_sha b7a775f19e232ea2395c8611a47c81b1365fb86e; architecture
  OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1. SAM ViT-B + VGGT Omega frozen; semantic
  memory256, dim256, heads8, depth12, point_grid48, max_points4096,
  max_superpoints512, point_voxel0.01, superpoint_voxel0.08, neighbors12,
  point_layers2, uncovered_clusters8, checkpoint_blocks=true,
  source_many_to_one, support_matching=true, support_purity0.5, NMS0.7.
- Four epochs / 8337 training windows / 1191 scenes; global batch8; LR1e-4,
  understanding_weight0.05, ramp1043. Not a zero-step checkpoint.
- Full metadata is checked and emitted by the audit entry point. The zero-step
  initial SHA in metadata is provenance only and never used as model weights.
- `hungarian_config.json` in attempt02 matches the A4 matching settings except
  matching_mode=hungarian. No attempt02 weight is loaded or auto-selected.

**Uncommitted execution delta:** b7a775f still contains SemanticMemory128; actual
A4 directory has its three spatial dimensions changed to256. The exact actual
class is saved in snapshot/semantic_memory.py, with original file/class hashes in
snapshot/identity.json. Other present source changes are environment-gated later
training additions (RG_EXTRA/RG_SUPERVISED) and unused imports, not used by
`generate()`. The audit refuses these environment variables. No original model
file is changed. The audit-local constructor binds the captured sem256 class,
restores all model tensors with original restore_state, validates actual
semantic-memory output (1,2,256,256,256), then freezes parameters/eval mode.

## Pixel / point / query / GT indexing

`build_batch()` reuses the native ObjectLocusV1Provider and deterministic seed
42+source_index. `pin_pair` fixes context then novel frame order; runtime checks
batch.frame_ids against the manifest. Labels use the RGB crop and nearest resize,
retain integer instance IDs, and are output in the same view order. The first two
views are context; only these two enter source support/GT and context generation.
Heldout full RGB float32 SHA is rechecked against the existing cohort record.

`lift_context_cloud()` uses predicted Omega points/depth/confidence, not GT
geometry. It retains finite points/confidence/depth with positive predicted depth;
samples adaptive max-confidence grid points, additionally one valid point per SAM
region. Uncovered sampled points get deterministic XYZRGB KMeans8. SAM regions
are view-specific (no assumed cross-view merge). Within a view, smaller masks
overwrite larger ones. Per-region points can be fused by labelled voxelization;
regions are never fused across boundaries. Region count overflow raises, rather
than truncating queries.

| Tensor | Exact meaning / ordering |
|---|---|
| cloud.source_indices | One index per retained source observation into flattened **two-view** predicted geometry/RGB/GT grid; index=view*H*W+y*W+x. Pre-voxel observations, not fused-point indices or Gaussian indices. |
| cloud.source_to_point | Maps each retained observation to its fused 3D point. |
| superpoints.point_superpoint_ids | Sorted-unique normalized region/query ID for each fused point. |
| superpoints.source_superpoint_ids | point_superpoint_ids[source_to_point], one ID per source observation, in exactly source_indices order. |
| batch.instance_label_all | Integer scene-local instance IDs, same RGB crop/resize/view order; nonpositive Thing IDs are invalid for native support and targets. IDs shared across views are treated as one GT by the native builder. |
| batch.semantic_label_all | Internal classes0..19; 0/1 Stuff, 2..19 Thing; out-of-range invalid. |
| targets.gt_instance_ids[0] | Sorted unique positive IDs among valid context Thing pixels; GT column order. |
| targets.gt_classes[0] | Majority semantic class for that ID across both context masks; ties choose the smallest class index. |
| targets.gt_pixel_masks[0] | K,2,H,W; native valid & sem>=2 & instance_id==ID; no minimum area. |
| prediction.region_mass | 1,views,Nquery+2,H,W rendered Gaussian membership / alpha, clamped by native _probability for loss. First N channels follow Thing queries; last2 are Stuff0/1. Not logits, normalized binary masks, or ground-truth probabilities. |
| prediction.thing_logits19 | 1,Nquery,19: classes0..17 map to internal Thing2..19; index18 is No-object. Same query order as superpoint IDs, region channels and final pairs. |

DualSuperpointInitialization requires identical source_superpoint_ids for its
reconstruction and understanding branches. Decoder updates query contents, not
this sequence. Core Thing count excludes the two appended Stuff queries.
Runtime checks dimensions, source bounds, source-to-point query alignment and
context RGB equality; contract tests check cross-view indices and semantic ties.

## Exact native support denominator

At each source observation:

```
valid = (0 <= sem <= 19) & ((sem < 2) | (instance_id > 0))
N_i = bincount(source_superpoint_ids[valid])
H_ik = bincount(ids[valid & (sem >= 2) & (instance_id == gt_id_k)])
P_ik = H_ik.float() / max(N_i, 1)
allowed = (P_ik > 0) & (P_ik >= purity)
```

Stuff is included even with instance ID0 or negative. Thing instance ID0 is
excluded. Do not replace this denominator with Thing-only observations, total
sampled observations, GT area, unique fused points, or Gaussian count. Repeated
observations of an object in different views count separately. At purity0.5 a
query can allow two different GT when both have exactly half of its valid
observations and there is no valid Stuff contribution. At purity>0.5 it cannot
allow two disjoint GT under these semantics.

The independent histogram uses exactly these masks and float32 divisions; native
allowed and scores must be torch.equal, no allclose. min_hits3/5 only apply in
offline sensitivity. Zero Thing observations are reported separately.

## Native Cost, oracle, and OneFormer reference

The audit observes the actual SciPy input inside unified_instance_loss and restores
the solver in finally. Model/decoder/head/query/loss source files are untouched.
Production Hungarian config is obtained by dataclasses.replace for an offline
loss call only. Cost is native class2+BCE5+Dice5 on native loss-valid context
pixels, mask clamp1e-6..1-1e-6, forbidden1e6, LSA then allowed filtering. Pair
parity is exact against offline SciPy and a second native call. No backward.
Graph maximum cardinality is independently computed with SciPy's sparse maximum
bipartite matching. A finite1e6 penalty is not a universal proof of cardinality;
a CPU contract explicitly constructs a counterexample.

OneFormer reference: https://github.com/filaPro/oneformer3d/blob/main/oneformer3d/instance_criterion.py
`SparseMatcher(topk=1)` computes the second smallest Cost and selects strictly
smaller entries. Exact minimum ties can produce no pair; topk2 is undefined for
Nquery<2. The requested per-GT argmin and faithful official strict selection are
both recorded using the **same native SSST Cost and allowed edges**. Reused
queries remain explicit conflicts. No one-to-one claim follows from Top-1 alone.

## Mask protocol boundary

Native source/Cost validity requires positive IDs for Thing pixels. Official
write_official_pair's export valid mask only checks semantic0..19, preserving
unknown-ID Thing pixels; raw IoU uses that semantic-valid pixel mask and records
the extra-pixel count. Targets still use native positive-ID context GT. Raw
membership >=0.5 and IoU>=0.5 are evaluated across concatenated two context views,
without classification/NMS/score/min-area/competition filtering. This is neither
official AP nor final packed-mask recall; it must not replace official scores.
