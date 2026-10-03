# Pinned official LocusGS on processed ScanNet: one fresh reconstruction run

SSST baseline: `b00b94fdc45af6d2f78c55f05671aaa75906204f`.
Upstream: https://github.com/leo-frank/LocusGS at
`9da24a896c4787d0bd90882fd2c9f001b3e153a5`.
The vendored `locusgs/`, LICENSE and README.md are byte-identical to that commit;
`third_party/locusgs_official/SOURCE_MANIFEST.json` records all 37 upstream file
SHA256 hashes. No nested Git repository, data or pretrained state is included.

## Model and adapter

`siu3r_official_locusgs_recon` owns an actual upstream `LocusGS` instance.
Model fields use upstream defaults except the explicitly registered overrides:
`cross_attn_variant=geometric_positional`, `use_anchor_radius=True`,
`anchor_radius_refinement=True`, `anchor_radius_affects_bias=True`.
Resolution/patch: 256x256/8; encoder/decoder depth: 3/12; dimension: 1024;
heads: 16; MLP ratio: 4; 1024 GS tokens x64 children =65536 Gaussians.
Dynamic tokens=0, time_embedding=False, full K/V (`geo_sparse_sampling_k=0`).
Self-attention is learned positional; dense sparse masks and handcraft=False.
Supervised layers=(6,12), radius init/min=1.0/0.001, local center variant=
`radius_scaled_offset`, official z offset=1.0, scale cap=0.075, token std=0.01.
Grey background, near/far=0.025/125; posed inputs; no token initialization from
existing weights. All remaining official fields and all adapter/provider fields
are serialized in full in `fixed_config.json` and every checkpoint.

The adapter converts every input field explicitly, preserving the existing
Plücker channel order and camera layout. Only two context RGB/ray/camera views
enter `forward_encoder`; `forward_decoder(return_intermediate_gaussians=True)`
returns L6/L12 Gaussians and raw anchors. Output channels remain upstream
XYZ, opacity, scale3, quaternion4, RGB3: `[B,65536,14]`. No scientific upstream
function is patched, no extra child tanh is added, and radius is not frozen.
The parameterless upstream renderer remains unused; all actual rendering uses
the existing TokenGS renderer for training, smoke and both evaluation arms.

## Data, objective and optimizer

Full processed ScanNet train/val roots are `/space/mawb/SIU3R/data/scannet/`:
1201 train scenes and312 val scenes, disjoint; sorted scene lists and hashes
are stored with the report. Full val_pair has1860 records, 2 contexts +4 novel.
Training order is `[c0,c1,n0,n1]`; all four views have RGB supervision.
Scene scale=0.15; existing first-camera normalization, crop/resize and K rules
are unchanged. GT poses are used; depth/panoptic GT are never model inputs.
FP32, batch=1, workers=0, no AMP/DDP/accumulation/reflection. Scene sampler is
`numpy.random.default_rng(42)`, pair sampler `random.Random(42)`; each draw
creates a new official pair. The old20-attempt train-only scene retry protocol
is retained and every successful exposure and failed attempt is logged.

Each layer has `MSE +0.2*(1-SSIM)/2 +Gvis +0.1*Avis`, with weights1/3 and2/3
for L6/L12. SSIM reuses canonical_recon.ssim_loss. Both visibility terms use
canonical analytic projection from actual Gaussian XYZ or official raw anchors.
Camera depth<=0.025 gets penalty1; otherwise projected coordinates are
`(2u/W-1,2v/H-1)`, with outside-frame ReLUs, minimum across views, clamp1 and
mean across points. Renderer means2d is never read by the loss. This is an
explicit ScanNet outer-loss adaptation, not an entirely native upstream recipe.
No depth, LPIPS training, semantic, instance, opacity, compactness or GC loss.

Fresh seed42 initializes every official parameter. No historical, MASt3R,
DL3DV or SIU3R weights are loaded. AdamW betas=(0.9,0.95), eps=1e-8; WD0.05
except1D parameters and upstream `_no_weight_decay` markers, which have WD0.
Every parameter remains trainable, with no duplicate or omitted optimizer entry.
Global gradient clip=1.0. Budget is exactly50000 optimizer updates.
For0-based t<2000, base LR=`1e-4*(t+1)/2000`; thereafter it is
`1e-4*(0.02+0.98*0.5*(1+cos(pi*(t-2000)/48000)))`.
Updates1..2500 use base LR; updates2501..50000 use min(base LR,2e-5).
The2500→2501 boundary does not reset optimizer or warmup. Initial LR=5e-8;
update2000=1e-4; update2501=2e-5; update50000≈2.000000105e-6.

## Contracts, smoke and launch

CPU contracts verify upstream hashes/configuration, field conversions, context
restriction, Gaussian channel layout, near-plane/camera conventions, layer
weights, LR boundaries and optimizer coverage. Disposable smoke runs on an
RTX3090 24GB on `3dimage-11`, as explicitly specified in task sections6/7.
The summary table's node13 mention is superseded by those execution clauses.

Smoke uses the first provider pair from scene0000_00 with independent seed42,
plus full val_pair record0 (scene0011_00, contexts1727/1744). It checks actual
forward/loss/backward/clip/optimizer behavior, render/depth/alpha finiteness,
nonzero token/anchor/radius/refinement/activation gradients, and wrapper versus
same-weight direct upstream calls against the same-arm repetition envelope.
Official parameters that are unused by the fixed branch remain trainable:
`tau_sample` is only used with sparse sampling; cross-attention anchor/plucker
and q/k positional LayerNorms are only called by learned-positional cross
attention. The smoke report lists every such parameter and exact reason.
Both records are exported and processed by the pinned official recon-only
SIU3R evaluator. No local-PSNR-only substitute is accepted. All smoke state is
discarded; the formal run reseeds and saves step0 plus its parameter hash.

Commit/integration uses an isolated worktree; original dirty files are not
modified, stashed or cleaned. Final tested SHA must be pushed and remotely
verified before formal launch. `training_authorization.json` records those
checks. `submit_official_locusgs_recon.sh` submits partition3090, node11,
1 node/task/GPU, 16 CPUs, 64G and24h. CPU/smoke are independent of storage
reserve; formal launch requires60GiB free. Only one fresh job registration
is allowed. No active experiment is preempted or terminated.

Checkpoints are atomic and COMPLETE-marked every2500 steps; permanently retain
0/2500/12500/25000/47500/50000 and the newest two recovery points. Full model,
optimizer, schedule, source SHA, configuration, global and provider/sampler/pair
RNG, scene counts, retry and window exposure state are saved. Interruption-only
`--resume` requires identical SHA/config/schedule and restores every state;
interrupted log tails are retained separately before exact replay. A STOP
marker on nonfinite/implementation failure forbids automatic rollback or tuning.
Four fixed val monitor windows use seeds42+1000+i at step0/every2500, reporting
context/true-novel PSNR/SSIM without checkpoint selection or training gates.

## Fixed historical comparison, evaluations and output

The protected historical best_monitor/model.pt SHA256 is
`5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`.
Legacy model type=`siu3r_locusgs_recon`, strict450 tensors, bounded delta,
frozen decode radius0.15, supervised layers6/12,1024x64 children. Its saved
explicit config is preferred; the bounded/frozen baseline preset supplies
fallback fields. The generic SSST eval preset is prohibited. Historical eval
has zero optimizer updates. A missing/mismatched weight blocks comparison
rather than substituting another checkpoint.

Both arms use the same unchanged `evaluate_ssst_validation` loader/export
helpers and renderer, with the new wrapper loader added externally. RGB/depth
files use actual context-first batch frame IDs. Depth is multiplied by1/0.15
and written as uint16 millimetres with the existing rounding/clipping rule.
All, context-only and true-novel-only scopes use identical per-window/frame
keys for both arms; duplicate scene+context-window+frame outputs overwrite in
manifest order, preserving historical naming/dedup semantics. View indices
record the actual denominator; target-all is never called novel-only.

The unchanged `invoke_siu3r_official_evaluator --recon-only` entry invokes
SIU3R Evaluator at commit`8ea80166be76854f938e90521f1a5b688b755c87`, using
`/space/mawb/SIU3R/.venv_gpu_v4/bin/python`. A separate pinned checkout is used
if the main evaluator checkout changes. Its depth affine fit remains unchanged.
Metrics are PSNR/SSIM/LPIPS/AbsRel/RMSE; understanding metrics are N/A.

Primary endpoint is fixed new50000 versus historical monitor-best47500 on full
val_pair. Supplemental new47500/new50000 and historical47500 use the locked
V3 val32 frame keys, with source/hash provenance. Submit creates a parallel
legacy eval job, an after-training new eval job and an after-both packaging job.
No additional training arm, extra experiment or result-based selection occurs.

Reports: `/space/mawb/ssst/group_plus/official_source_scannet_v1`.
Run: `/space/mawb/ssst/workspace_recon_diag/official_source_scannet_v1/run`.
The comparison includes three scopes and deltas,312-scene PSNR mean/median/
p10/p90 and per-scene CSV. Qualitative panels use records0/1/2 with one context
and one novel each, GT/old/new RGB and depth. Idempotent packages are<=28MiB
per ZIP and exclude weights, datasets and complete PNG trees.

Novel PSNR≥+0.3dB with nondecreasing SSIM/nonincreasing LPIPS and context
PSNR drop≤0.3dB supports “historical reconstruction clearly improved”. Both
context/novel |PSNR delta|≤0.3dB is “similar”, with mixed metric directions
reported explicitly. Otherwise report each improvement/degradation.
These are interpretation thresholds, not continuation gates.

Only one new run exists; there is no seed-variance estimate. Model structures
have multiple differences; historical47500 was monitor-selected while new50000
is a fixed endpoint with2500 extra updates. Analytic visibility reflects a
historical implementation correction. GT poses are not unposed SIU3R. No claim
of a strict causal experiment, universal upstream superiority, or reproduction
of DL3DV paper training is made; no understanding/object experiment is added.
