# 2026-10-13: user-authorized old experiment checkpoint cleanup

Before starting the next decoder-interaction experiments, the user requested
cleanup of obsolete experimental files. This supersedes earlier instructions
to retain every old checkpoint. Scientific results and historical failure
records remain preserved; no new experiment was submitted in this cleanup.

The cleanup completed at 2026-10-13 10:25:14 Asia/Shanghai. Exactly 32
checkpoint files were deleted, totalling 193077818880 bytes (179.82 GiB).
Measured quota availability rose from 52.47 GiB to 232.29 GiB (`df -h` rounds
this to 53G → 233G). The observed increase was 179.82 GiB.

| Experiment group | Files removed | GiB | Retained models |
|---|---:|---:|---|
| Original frozen VGGT | 1 | 5.99 | Epoch4 and epoch8 |
| Original adaptation2 + frozen joint4 | 4 | 29.60 | Exact adaptation2 boundary, joint2, joint4, adapted VGGT |
| Completed batch16 A/B | 4 | 40.48 | Both epoch2 endpoints and B's final adapted VGGT |
| Cancelled batch8 A/B | 2 | 20.24 | Original common source; cancellation/startup/failure records |
| Four-window geometry/objective fits | 9 | 30.30 | Each arm's update0 and update300 |
| Adapter coefficients 0.01 / 0 | 6 | 20.20 | Both arms' update0 and update300 |
| Shared AA/decoder short controls | 6 | 32.99 | Each arm's update0 and update100 |

Removed files are intermediate training checkpoints, rolling checkpoints
superseded by retained completed endpoints, and the cancelled batch8 weights
that the user had replaced with a fresh batch16 run. The completed evaluations,
including all measured intermediate curves, were checked before deletion.
Deleted checkpoint metadata was archived before any unlink. The intermediate
weights themselves are no longer available for direct inference or resuming
those intermediate states; reproducing them requires rerunning their preserved
code and configuration. Existing result documents describe their historical
execution, not the present retention of every checkpoint.

The cleanup used an explicit 32-file allowlist under
`/space/mawb/ssst/workspace_group_plus`, without recursive directory deletion.
It checked completed evaluation receipts, completed/cancelled training records,
checkpoint clocks and retained endpoint existence. Each candidate's inode,
size, modification time and link count was rechecked before deletion. A
per-file deletion journal records actual successful unlinks.

43 protected files passed the before/after fingerprint comparison (inode,
size, modification time; SHA256 also checked for small files). These include
the legacy LocusGS source checkpoints, exact reconstruction adaptation2 source,
adapted VGGT assets, original and A/B final models, all diagnostic initial/final
models, manifest, result receipts and the current immutable execution scripts.
Results, PNG exports, plots, Slurm logs, failure evidence, immutable snapshots,
datasets, pretrained assets, CUDA extensions and environments were retained.

Current training63552 remained RUNNING on node13; four-card official eval63553
remained PENDING on its training dependency before and after deletion. The
active adapter-u001 run/evidence roots were excluded in full. The future
evaluation checkpoint and its source assets remain available as scheduled.
There was no continuing training-progress polling or training resubmission.

Audit and exact deletion records:

- `/space/mawb/ssst/group_plus/storage_cleanup_20261013/audited_manifest.json`
- `/space/mawb/ssst/group_plus/storage_cleanup_20261013/deletions.jsonl`
- `/space/mawb/ssst/group_plus/storage_cleanup_20261013/deletion_receipt.json`

Implementation: `scripts/cleanup_vggt_experiment_artifacts_20261013.py`.
The default invocation performed only the audited preparation; `--apply`
executed that unchanged allowlist. Both actual paths completed successfully,
along with Python syntax, CLI help and git whitespace checks. A completed
receipt prevents an accidental second execution.
