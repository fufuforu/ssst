# Baseline4 structure, 4096 reconstruction tokens, fresh full4

User authorized: train the 4096-token main model for four epochs. No quality
head, appearance residual, QuerySplat dual decoder, new experiment arm, or
continuation from trained B weights is included. Submit only one eight-rank
two-update engineering gate followed by independent fresh eight-rank full4.
Formal export/scoring and quality-MLP fitting are not submitted in this phase.

The model inherits the already validated B architecture, with a distinct
scratch architecture identity. It retains 1024 feature width, 64 Gaussian
children, RGB/DC appearance, original geometric refinement, No Anchor-CA and
detached reconstruction reads for understanding. The understanding query
initializer and count policy are unchanged. The 48-grid candidate policy is
the validated B policy: supplement finite pixels by descending confidence
only when needed to provide at least 4096 unique candidates, then FPS4096.
This necessary count-related pool supplementation is disclosed when comparing
against the historical Baseline4 that did not require 4096 candidates.

Zero source is the exact original Baseline4 untrained checkpoint SHA
bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397.
This archived source has the older joint architecture's 664 tensors; the
same 32 Anchor-CA tensors removed by Baseline4 are removed here. Of the 632
remaining tensors, 631 are copied exactly. The 1024x1024 reconstruction token
table becomes 4096x1024: original rows exact, extra3072 repeat original rows
plus deterministic1e-4 noise seed20261025. All optimizer states start empty.
VGGT/SAM pretrained assets match the original identity; task weights have
zero updates. All original trainable task names are trained, unlike the B
continuation that froze understanding/shared parameters.

Training: 8337 windows/1191 scenes, original2Context+2Novel256x256, global8
with eight one-sample ranks, 1043 updates/epoch, 4172 total. L6/L12 weights
1/3,2/3. AdamW betas.9/.95 eps1e-8, original no-decay flags, main1e-4 and
geometry1e-5; warmup100, original cosine factor floor.1; understanding warmup
1043, global norm clip1. Existing differentiable-bias math-SDPA engineering
guard and pinned gsplat partial_1406 renderer are retained.

CPU preflight compares the actual full-sized task tensors against a separately
initialized original Baseline4, checks trainable names, optimizer groups,
empty Adam, and rejects trained/dirty sources. CPU wrappers for frozen assets
are explicitly synthetic; they do not establish a real GPU forward pass.
The queued GPU gate checks actual assets, batch/262144GS/4096 unique sources,
both gradient isolation directions, positive gradients in all parameter
families/new rows, finite loss, parameter updates and eight-rank equality.
Both smoke and formal training reread the same zero checkpoint and restore
the original eight-rank RNG; formal training never reads smoke weights.

Checkpoint storage: zero plus rolling latest, every100 updates and epoch
boundaries; no retained per-epoch copies. Training writes rank JSONL and real
TensorBoard events (fixed scalar set averaged across8 ranks). TensorBoard
service6006 includes reconstruction4096_scratch4_live. There is no job-status
or JSONL polling process. After one bounded submission-status check, stop
and wait for user notification under /space/mawb/AGENTS.md.
