# Object-Locus V3-Set Expanded

This registered run continues the completed V3-Set epoch-64 checkpoint (`global_optimizer_step=3584`) using the same model, loss, renderer, visibility, inference and optimizer state. It adds 32 complete passes over the fixed 1008-window / 128-scene expanded pool (32256 updates), with 200-update warmup for the already-fixed LR peaks (object `1e-4`, reconstruction `1e-6`) and GC alpha `0.01`. The understanding weight remains `1.0`.

The only experiment changes are training coverage and update count. There is no architecture, loss, sampling-weight, threshold, evaluator, AMP or optimizer-moment change. Data order is `default_rng(42 + epoch_index).permutation(1008)`. The final global update is 35840. Six registered checkpoints/evaluation nodes are epochs 0, 2, 4, 8, 16 and 32. The completed V3-Set original training windows receive 32 additional exposures (96 cumulative); other expanded windows receive 32.

The execution driver strictly restores the model, AdamW state and RNG from the epoch-64 source checkpoint. It verifies the locked source hashes and the frame/split contracts before allowing updates. Every epoch has a resumable checkpoint; evaluation preserves the training RNG. Low metrics do not stop the registered plan. A numerical or execution failure stops with the captured failure state.

This is continuation from a successful small-window checkpoint, not a fresh initialization or an equal-exposure paired comparison. It does not establish full unposed SIU3R benchmark performance.
