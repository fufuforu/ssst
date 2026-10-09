# Frozen Probe + R3D attempt11 focus JSON recovery

## Failure evidence

- The report-only job 59497 used execution SHA `56b286569914b30ac390648ad94e22f68a670fe2` on 3dimage-11 and failed after the paired bootstrap and reconstruction reductions had completed.
- Its traceback ended in `scripts/report_object_locus_frozen_probe.py::focus_cases` while encoding `focus20_cases.jsonl`: `TypeError: Object of type int64 is not JSON serializable`.
- The failed output and logs are retained at attempt10, including `slurm/report-59497.err`, `slurm/report_failure_59497.json`, and the empty `focus20_cases.jsonl`.

## Repair

The `focus_cases` writer now uses a strict JSON fallback that converts NumPy scalar values to their corresponding Python scalar via `.item()`. It raises for unsupported types instead of stringifying them. This handles NumPy query/window IDs produced by matching without changing focus selection, query order, scientific values, or report criteria.

## Verification and recovery

- Added a focused regression test for `np.int64`/`np.float32` serialization and rejection of unknown object types.
- The regression test passed; Python compilation, the existing 17-check report preflight over attempt06, Bash syntax validation, and `git diff --check` are required before submission.
- New report-only output root: `attempt11`. It is prepared with the existing `prepare_report_resume` workflow from the immutable attempt06 completed evaluation. Attempt10 remains unchanged as failure evidence.
- The new execution SHA is recorded in the attempt11 provenance and Slurm receipts after commit; it is intentionally not embedded here to avoid self-reference.
- No model, cache extraction, training, original-model forward, official evaluation, or GPU job is part of this recovery.
