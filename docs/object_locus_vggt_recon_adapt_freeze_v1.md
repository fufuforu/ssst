# VGGT reconstruction adaptation 2 epochs → frozen joint training 4 epochs

On 2026-10-10 the user authorized the preceding literature recommendation and a shorter training budget, following the full official epoch4/epoch8 comparison. This is a new experiment, separate from the completed permanently frozen baseline. It implements reconstruction adaptation inspired by InsTok3D/Uni3R/C3G/AnySplat; it is not a reproduction of their architectures or supervision.

Branch/worktree: `object-locus-vggt-recon-adapt-freeze-v1`, `/space/mawb/ssst_object_locus_vggt_recon_adapt_freeze_v1`.
Run: `/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1`.
Evidence: `/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1`.

## Fixed budget and boundaries

- Reconstruction adaptation: **2 epochs / 2086 updates / 16688 exposures**. Train all 24 official frame and 24 global Transformer blocks, the existing Gaussian reconstruction modules and memory adapter. LR peaks: VGGT AA 1e-6, reconstruction 1e-5, adapter 1e-4. DINO patch encoder, special tokens and camera/depth heads remain fixed. Understanding/object modules are frozen and unused, and feedback is disabled. Reuse the original Gaussian decoder and canonical multi-layer reconstruction objective, without introducing a second renderer/head or new loss.
- Frozen joint training: **4 epochs / 4172 updates / 33376 exposures**. Retain the adapted VGGT, reconstruction and memory weights. Freeze all VGGT parameters, restore the original trainable understanding/object modules and feedback, and use the original V3-Set losses and GC rule. Start a fresh joint optimizer, understanding/beta exposure warmups and a 4-epoch cosine schedule.
- Total: **6258 updates / 50064 new exposures**. Epoch permutations are continuous `default_rng(42+epoch)` for epochs 0..5, with the same prefix padding. Every original window remains exposed; global batch8, four RTX3090 GPUs on node13, microbatch1 and accumulation2 (authorized on 2026-10-11).

Initialization is the same original full1201 epoch06 source (SHA256 `68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a`) and verified official VGGT revision `860abec7937da0a4c03c41d3c269c366e82abdf9`. The completed frozen experiment's optimizer/clock is not resumed into this different scientific recipe. Original retained modules and adapter seed31415 are unchanged; no model is entirely initialized from scratch.

Both phases use context-only generation and the same independent RGB-based target-camera calibration, FP64 shared-context Sim(3), and `monitor_v1`. No GT intrinsics/poses enter generation. The camera/depth heads do not receive new supervised losses or gradients; their inputs change with the adapted aggregator. This can adapt reconstruction features but does **not** establish that camera/metric-scale errors have been corrected.

## Precision, optimizer and artifacts

Stage1 AA parameters and AdamW masters/moments are FP32, forward attention uses BF16 autocast and official activation checkpointing, and the downstream decoder/head/loss stays FP32. The DINO patch encoder remains frozen. Task-owned CPU master parameters and native `torch.optim.AdamW(..., foreach=False)` implement the same update formula with betas(.9,.95), eps1e-8; gradients are globally averaged and clipped to1 before the CPU update. Optimizer placement is a memory measure, not BF16 optimizer arithmetic. Unused understanding/object modules reside on CPU during stage1.

Stage2 uses the original frozen BF16 aggregator / FP32 camera-depth precision and unchanged downstream precision. The complete adapted official VGGT state is saved as `adapted_vggt.pt` with the base revision, adaptation clocks and SHA256. Every frozen-stage checkpoint explicitly references this artifact; the official eval loader verifies the SHA and strict-loads the adapted weights before rendering. It cannot silently evaluate original VGGT weights.

Checkpoints at 20 updates and every epoch include optimizer, sampler position, all-rank RNG and exact per-window exposure counts. Stage1 includes tuned VGGT tensors; stage2 references the immutable adapted artifact. Engineering resume restores the active phase and clock; resuming the stage boundary can reconstruct the exact derived artifact from the stage1 checkpoint. Historical runs and failures remain separate.

## Execution and validation

Use the existing runtime/actual train entry, with `--staged-vggt-adapt --staged-world-size 4`; default legacy training remains unchanged. Real single-card smoke performs two isolated updates per phase, checks nonzero VGGT updates and joint understanding/adapter gradients, saves/reloads the tuned VGGT artifact, then proves VGGT remains unchanged during frozen updates. Fresh four-card smoke repeats both phases with two microbatches per rank, checks every rank and visits fixed window6923. Neither smoke optimizer nor exposure clock enters formal training.

`scripts/submit_object_locus_vggt_staged.sh single_smoke` prepares a clean, pushed immutable snapshot and submits the single-card check. Formal submission uses `TASK_SINGLE_REPORT=<passed-report> bash scripts/submit_object_locus_vggt_staged.sh train`. An optional `TASK_SINGLE_JOB=<job-id>` adds `afterok` while that check is still available in the controller; omit it for an already completed proof whose old job ID has expired. The formal job first performs fresh four-card smoke, then fresh stage1, then stage2 automatically. Resource occupation causes queueing, without changing batch or node.

The startup receipt is written separately for each phase after 20 synchronized formal updates /160 phase exposures, including finite losses/gradients/updated parameters, LR/weight/beta, geometry warnings, rank clocks, probe change and GPU peaks. A read-only viewer is `scripts/show_object_locus_vggt_staged_progress.py`; it shows Slurm state, progress and current-phase ETA when enough updates are logged.

The final scientific comparison remains SIU3R Table1's 13 metric columns, both text mIoU cells empty, with all warning windows retained. The exporter/scorer accept the completed adaptation2+joint4 checkpoint and its derived VGGT identity; this setup does not assert improved metrics before evaluation.

## Verified initial execution and queue history

Single GPU job **60339** completed exit0:0 in 00:05:45 at 2026-10-10 16:36:39 Asia/Shanghai, using immutable SHA `9c58270620c50165b8dfdd347c8ccfae3dbcd5b5`. Both adaptation and frozen joint stages performed two isolated updates. AA qkv probe change norm after adaptation was 7.636223e-6, and was exactly0 in both frozen updates. Real artifact save/hash/strict reload and nonzero understanding/adapter gradients passed. Peaks: 14,394,218,496 bytes adaptation, 18,654,650,368 bytes joint. These are engineering checks, with **zero formal updates** and no validation-quality claim.

Initial formal job **60341** used the same validated SHA and awaited eight-card resources. It was subsequently replaced before any formal update, as recorded below. The dependent official evaluation uses `checkpoint_frozen_joint_epoch_04.pt`, the saved adapted VGGT artifact, and the unchanged pinned SIU3R evaluator. Results will be under the evidence root's `evaluation/final_adapt2_joint4_official`.

The later source-only engineering fix restores saved rank RNG even when resuming exactly across the adaptation→joint boundary with a fresh optimizer. Its regression test proves that the next RNG sample is preserved and the previous-phase optimizer is not restored. This does not change the fresh-run calculation. CPU evidence covers 28 inherited contracts and five additional optimizer/gradient/artifact/phase/RNG contracts.

## Authorized five-node queue expansion, 2026-10-10

The user requested waiting on nodes **11,13,14,17,18** instead of node13 alone. Nodes11/13 have eight RTX3090 GPUs; nodes14/17 have eight RTX4090 GPUs and node18 has ten RTX4090 GPUs. Formal training still requests **one node / exactly eight GPUs**; official evaluation still requests one node / seven GPUs. Submit to `--partition=3090,4090 --exclude=3dimage-12 --nodes=1`. The union of these partitions minus node12 is exactly the five-node pool. A multi-node `--nodelist` would require all listed nodes and is therefore not used.

The staged runtime and batch host checks accept only those five hosts and their matching GPU models. Configuration metadata lists the pool; actual host/GPU names are recorded in smoke, run and startup manifests. All data, model weights, losses, precision, optimizer mathematics, batch size, epoch budget, sampler and `monitor_v1` remain unchanged. The completed node13 single-card proof is retained, with every scientific configuration field checked exactly; only the authorized node/GPU metadata is normalized for this check. Fresh two-phase eight-card smoke must pass on the allocated hardware before any formal update.

At replacement preparation, jobs60341/60352 were still PENDING with zero formal updates. Their stored batch scripts pin node13, so they must be replaced with a newly pushed immutable launcher snapshot, preserving the old submission records and keeping one training→`afterok` evaluation chain. This is a queue amendment before training starts. It does not restart an active training run or submit a second executing copy. Code on the original frozen baseline branch remains unchanged. After confirming the new queue constraints and dependency, stop resource polling as requested.

## Authorized four-card execution on node13, 2026-10-11

The user requested four GPUs on node13 after the eight-card job60458 continued waiting overnight. Job60459 was its unstarted dependent evaluator. The new execution is **one node13 / four RTX3090 GPUs / microbatch1 / accumulation2 / global batch8**. The earlier five-node/eight-card policy above is historical. Total epochs2+4, updates6258, exposures50064, all losses, learning rates, warmups, clipping, optimizer mathematics and `monitor_v1` are unchanged.

Each original eight-window optimizer update is partitioned into four consecutive two-window rank sequences. Each microbatch loss is divided by2 before its reconstruction/understanding gradient is computed. The original GC combination is applied per microbatch, gradients are accumulated and averaged over four ranks, then clipped once and passed to one optimizer step. This gives the same eight-sample mean gradient mathematically. Every epoch retains exactly the original permutation and prefix padding. Rank RNG follows the existing seed42+rank rule on the four physical ranks; this changes stochastic streams and floating-point summation order, so bitwise reproduction of an eight-rank stochastic trajectory is not claimed.

Fresh **four-card two-phase smoke** runs first with two updates /16 exposures per phase. It validates both accumulated microbatches, all ranks, mandatory window6923, nonzero adapted VGGT updates, frozen VGGT invariance, understanding/adapter gradients and real adapted-asset save/hash/reload. Smoke state is discarded before formal training. Geometry logs retain all microbatch windows; startup receipts include both microbatch calibration summaries on each rank. Checkpoints store four rank RNG/count streams and accumulation2 metadata; the evaluator strictly validates this supported layout.

The official evaluator is submitted with four GPU workers to avoid requiring seven free cards after four-card training. It retains the previously authorized five-node evaluation pool, the pinned native SIU3R code, the full1860-window/312-scene cohort, global segmentation-state reduction and the same13 metric columns with both text mIoU cells blank. Only parallelism and the dependency job ID change.

Before replacing the pending queue, validate full sampler coverage and the unchanged scientific configuration, and compare real four-process Gloo accumulation/GC/global clipping/AdamW updates against an eight-sample mean-loss reference for both reconstruction-only and joint gradients. The original frozen baseline and all prior job records remain intact. After the first20 formal updates /160 exposures are confirmed, stop polling and let training and its dependent evaluation continue.

### Confirmed four-card startup

Training **61835** started on node13 at **2026-10-11 09:24:05 Asia/Shanghai**, using immutable execution SHA `39bf57fa9f898e6604261aced9a8190500390adf`. Official evaluation **61836** depends on `afterok:61835`, with four workers in the authorized five-node evaluation pool. Pending jobs60458/60459 were held and cancelled before replacement; no formal training was restarted or duplicated. The implementation is pushed, and all35 CPU contracts passed, including actual four-process numerical equivalence.

Real four-card smoke passed both phases, two isolated updates /16 exposures per phase, with all16 rank/update rows checked. Maximum allocated GPU memory was20,847,478,272 bytes. The shared AA probe changed by6.695824e-6 after adaptation and exactly0 after each frozen update. Mandatory window6923, understanding/adapter gradients and adapted-asset save/hash/strict reload passed. Smoke formal updates are0.

At **09:34:27 Asia/Shanghai**, formal reconstruction adaptation reached **20 updates /160 new exposures**, with all four rank clocks matching. Shared probe change was4.425419e-4; losses, gradients, LR warmup and parameter checks passed. Geometry counts were156 OK /4 WARNING, with all160 windows retained under `monitor_v1`. Startup proof is `four_card_checks/startup_validation.json` in the evidence root; the exact runtime receipt is `startup_confirmation_reconstruction_adaptation.json` in the run directory. Resource/progress polling stopped after this confirmation, and training continues toward6258 updates /50064 exposures, followed automatically by the full official13-column evaluation. The phase ETA sampled at update20 was26,063 seconds (~7.2 hours remaining for adaptation only); it does not estimate the later four joint epochs.
