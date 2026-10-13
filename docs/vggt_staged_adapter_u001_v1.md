# Adapter understanding gradient 0.01: full rerun of adaptation 2 + frozen joint 4

The user authorized this rerun on 2026-10-13 to compare against the original
full-validation PSNR 19.37 run. Only the understanding gradient received by the
VGGT memory adapter changes: `g_rec + g_under` becomes `g_rec + 0.01*g_under`.
The understanding and panoptic modules remain trainable with their original
full understanding loss. The GS gradient rule, loss weights and warmup,
feedback beta, learning rates, precision, data, cameras and architecture retain
the original recipe. VGGT, including AA, is frozen in the joint phase.
The original LocusGS-derived GS decoder weights are retained and fine-tuned.

The reconstruction-only adaptation stage has no understanding loss, so it is
reused exactly rather than trained twice. The boundary is the completed original
four-card adaptation epoch 2 checkpoint, not a joint epoch 6 or A/B checkpoint:

- Checkpoint: `/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_v1/checkpoint_reconstruction_adaptation_epoch_02.pt`
- SHA256: `a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c`
- Original execution: `39bf57fa9f898e6604261aced9a8190500390adf`
- Adapted VGGT SHA256: `c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace`
- Reused history: 2086 updates and 16688 exposures; exact original four-rank RNG
  and sampling ownership are restored.
- New joint optimizer and joint warmup start at zero. No reconstruction
  optimizer moments or joint-trained weights are carried forward.

The joint phase repeats all four epochs from this boundary on node 13, four
RTX 3090 cards, global batch 8 (microbatch 1, accumulation 2). There are 4172
new optimizer updates / 33376 new exposures; cumulative history at completion
is 6258 updates / 50064 exposures. Sampler epochs 2 through 5 and the original
joint-phase schedules are retained. Geometry policy is `monitor_v1`.

Before formal training, the same committed, pushed, immutable code runs two
isolated updates per phase on one card, then on four cards. CPU contracts and
the real GPU checks verify original configuration compatibility, boundary
identity, fresh optimizer/RNG restoration, trainable understanding/adapter
gradient paths, frozen VGGT and sampling of mandatory window 6923. Smoke clocks
do not count as formal updates. Training checkpoints retain every epoch.

`scripts/submit_vggt_staged_adapter_u001_v1.py` submits preflight and then the
formal job. The formal submission immediately schedules a four-card full
official SIU3R evaluation with `afterok`, eligible on nodes 11, 13, 14, 17, 18.
Evaluation uses the same model restore and official metric code, 1860 windows
/ 312 scenes / 11160 rendered images, with global mAP/PQ state merging. It
produces one Table 1 row with 13 columns; the two text mIoUt columns are blank.
This evaluation has zero optimization steps.

Evidence: `/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1`.
Checkpoints: `/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1`.
Read progress and ETA with:

```bash
/usr/bin/python3 /space/mawb/ssst_vggt_overfit_diagnostics_v1/scripts/show_object_locus_vggt_staged_progress.py --adapter-understanding-scale 0.01
```

After verifying 20 formal joint updates and 160 new exposures, stop polling and
leave training and its dependent evaluation running. Diagnose further changes
separately after this authorized rerun has started; do not mix AA unfreezing,
decoder reinitialization or loss edits into this comparison.

The first local CPU attempt exposed a NumPy tuple-indexing error in the newly
added boundary ownership check. It was repaired before any GPU submission;
both attempts are preserved in `local_validation`. Historical experiment and
failure records remain intact. Retaining four joint checkpoints, latest,
smoke assets and full evaluation fits the measured 67 GiB free quota; the
formal launcher requires at least 45 GiB free before starting.
