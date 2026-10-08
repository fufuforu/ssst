# Frozen Probe D classification alignment recovery

Attempt06 repairs the train classifier aggregation failure observed in attempt05. Cached H0 probabilities and each trained probe's batch probabilities are checked against matching label dimensions, then flattened in C order so window order precedes query order and class remains the final axis. The strict `[N,19]` guard in `classification_summary` remains unchanged.

The formal evaluator and the CPU preflight call the same `flatten_aligned_batch` and `summarize_train_readout` functions. The preflight evaluates all 1008 cached train windows for H0 and the nine fixed best heads, then writes a separate first-dev-window, 11-readout CPU export and checks H0/R3D against their saved GPU exports. It does not load the original model checkpoints, perform original-model forward passes, train heads, compute official AP, or run bootstrap. The one-window check does not replace full 32-window parity in formal D.

Attempt06 reuses the verified A/B cache and completed C heads from attempt05's provenance chain. It does not create new A/B/C receipts or alter source files.
