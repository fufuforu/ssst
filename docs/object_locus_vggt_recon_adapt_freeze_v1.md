# VGGT reconstruction adaptation 2 epochs → frozen joint training 4 epochs

On 2026-10-10 the user authorized the preceding literature recommendation and a shorter training budget, following the full official epoch4/epoch8 comparison. This is a new experiment, separate from the completed permanently frozen baseline. It implements reconstruction adaptation inspired by InsTok3D/Uni3R/C3G/AnySplat; it is not a reproduction of their architectures or supervision.

Branch/worktree: `object-locus-vggt-recon-adapt-freeze-v1`, `/space/mawb/ssst_object_locus_vggt_recon_adapt_freeze_v1`.
Run: `/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1`.
Evidence: `/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1`.

## Fixed budget and boundaries

- Reconstruction adaptation: **2 epochs / 2086 updates / 16688 exposures**. Train all 24 official frame and 24 global Transformer blocks, the existing Gaussian reconstruction modules and memory adapter. LR peaks: VGGT AA 1e-6, reconstruction 1e-5, adapter 1e-4. DINO patch encoder, special tokens and camera/depth heads remain fixed. Understanding/object modules are frozen and unused, and feedback is disabled. Reuse the original Gaussian decoder and canonical multi-layer reconstruction objective, without introducing a second renderer/head or new loss.
- Frozen joint training: **4 epochs / 4172 updates / 33376 exposures**. Retain the adapted VGGT, reconstruction and memory weights. Freeze all VGGT parameters, restore the original trainable understanding/object modules and feedback, and use the original V3-Set losses and GC rule. Start a fresh joint optimizer, understanding/beta exposure warmups and a 4-epoch cosine schedule.
- Total: **6258 updates / 50064 new exposures**. Epoch permutations are continuous `default_rng(42+epoch)` for epochs 0..5, with the same prefix padding. Every original window remains exposed; global batch 8, 8×3090, microbatch1 and accumulation1.

Initialization is the same original full1201 epoch06 source (SHA256 `68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a`) and verified official VGGT revision `860abec7937da0a4c03c41d3c269c366e82abdf9`. The completed frozen experiment's optimizer/clock is not resumed into this different scientific recipe. Original retained modules and adapter seed31415 are unchanged; no model is entirely initialized from scratch.

Both phases use context-only generation and the same independent RGB-based target-camera calibration, FP64 shared-context Sim(3), and `monitor_v1`. No GT intrinsics/poses enter generation. The camera/depth heads do not receive new supervised losses or gradients; their inputs change with the adapted aggregator. This can adapt reconstruction features but does **not** establish that camera/metric-scale errors have been corrected.

## Precision, optimizer and artifacts

Stage1 AA parameters and AdamW masters/moments are FP32, forward attention uses BF16 autocast and official activation checkpointing, and the downstream decoder/head/loss stays FP32. The DINO patch encoder remains frozen. Task-owned CPU master parameters and native `torch.optim.AdamW(..., foreach=False)` implement the same update formula with betas(.9,.95), eps1e-8; gradients are globally averaged and clipped to1 before the CPU update. Optimizer placement is a memory measure, not BF16 optimizer arithmetic. Unused understanding/object modules reside on CPU during stage1.

Stage2 uses the original frozen BF16 aggregator / FP32 camera-depth precision and unchanged downstream precision. The complete adapted official VGGT state is saved as `adapted_vggt.pt` with the base revision, adaptation clocks and SHA256. Every frozen-stage checkpoint explicitly references this artifact; the official eval loader verifies the SHA and strict-loads the adapted weights before rendering. It cannot silently evaluate original VGGT weights.

Checkpoints at 20 updates and every epoch include optimizer, sampler position, all-rank RNG and exact per-window exposure counts. Stage1 includes tuned VGGT tensors; stage2 references the immutable adapted artifact. Engineering resume restores the active phase and clock; resuming the stage boundary can reconstruct the exact derived artifact from the stage1 checkpoint. Historical runs and failures remain separate.

## Execution and validation

Use the existing runtime/actual train entry, with `--staged-vggt-adapt`; default legacy training remains unchanged. Real single-card smoke performs two isolated updates per phase, checks nonzero VGGT updates and joint understanding/adapter gradients, saves/reloads the tuned VGGT artifact, then proves VGGT remains unchanged during frozen updates. Fresh eight-card smoke repeats both phases, checks every rank and visits fixed window6923. Neither smoke optimizer nor exposure clock enters formal training.

`scripts/submit_object_locus_vggt_staged.sh single_smoke` prepares a clean, pushed immutable snapshot and submits the single-card check. Formal submission uses `TASK_SINGLE_REPORT=<passed-report> TASK_SINGLE_JOB=<job-id> bash scripts/submit_object_locus_vggt_staged.sh train`; it can queue with `afterok` while the single check runs. The formal job first performs fresh eight-card smoke, then fresh stage1, then stage2 automatically. Resource occupation causes queueing, without changing batch or node.

The startup receipt is written separately for each phase after 20 synchronized formal updates /160 phase exposures, including finite losses/gradients/updated parameters, LR/weight/beta, geometry warnings, rank clocks, probe change and GPU peaks. A read-only viewer is `scripts/show_object_locus_vggt_staged_progress.py`; it shows Slurm state, progress and current-phase ETA when enough updates are logged.

The final scientific comparison remains SIU3R Table1's 13 metric columns, both text mIoU cells empty, with all warning windows retained. The exporter/scorer accept the completed adaptation2+joint4 checkpoint and its derived VGGT identity; this setup does not assert improved metrics before evaluation.
