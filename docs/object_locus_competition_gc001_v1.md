# Object-Locus competition GC001 v1

This implementation adds one registered arm, `comp_gc001`, to the existing Object-Locus GC sweep. Its sole scientific change from `gc001` is the fixed final-context pixel competition supervision term: internal coefficient 2.0 and external coefficient 0.1. GC alpha remains 0.01. The original Panoptic V1 forward, parameterization, readout, and official export rules remain unchanged.

## Registered supervision

The method-only subclass reuses the original `step_loss`, Hungarian `final_pairs`, and `final_targets`. It computes FP32 log competition probabilities over all 102 final region channels, with epsilon `1e-6` and temperature 1.0. Thing channels receive `log(max(p_class[:18]))`; stuff channels 100 and 101 do not. Matched thing GT masks and valid wall/floor pixels form regions; each region averages over both context views and its pixels, regions are averaged equally within a window, then windows are averaged. Unmatched instances, void pixels, and empty regions do not add targets.

The new competition term is added to the existing understanding loss as `L_under_new = L_under_old + 0.1 * 2 * L_comp`. The existing understanding warm-up and GC-specific parameter gradient composition are reused without a new schedule. No parameter, module, or buffer is added.

## Source, plan, and execution

- Base: `9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0`; branch: `object-locus-competition-gc001-v1`.
- Initialization: strict full-state load from Full1201 epoch 6 (`68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a`). AdamW is fresh.
- Training uses the byte-identical GC sweep plan and source manifest: 128 scenes, 1008 windows, 8 ranks, 8 epochs, 1008 optimizer updates, 8064 new window exposures. Checkpoints are epoch 0/2/4/8.
- A two-update, eight-rank smoke uses the first two registered window identities at schedule positions u=25 and u=26, is isolated under the new report root, and is discarded before fresh formal initialization.
- The formal training job is submitted with `afterok` on the verified GC1.0 job. The four-arm endpoint evaluation job depends on successful completion of the new training job.

## Entrypoints and reports

CPU contracts: `python -m scripts.check_object_locus_competition_gc001`

Submit the dependent training and evaluation jobs:

```bash
scripts/submit_object_locus_competition_gc001.sh <verified-gc100-job-id>
```

The training runner executes the eight-card smoke before the fresh formal run. The evaluator performs one forward per registered window and reuses it for local statistics, official all/novel exports, and float reconstruction caches. A separate SIU3R environment process computes reconstruction metrics and paired scene bootstrap intervals. It does not run full-1860 validation or select a checkpoint.

## Evaluation-only recovery (retry01)

The original unified evaluation attempt failed before inference because its competition-model construction branch expected three return values while the registered constructor returns `(model, options)`. All four endpoint checkpoints have identical registered state structure, so the retry loads every endpoint through the registered Panoptic V1 constructor and applies `load_state_dict(..., strict=True)`. The competition checkpoint's training provenance remains its original training SHA; the retry records evaluation code SHA separately.

The retry writes only to `four_arm_evaluation_retry01/`; the failed `four_arm_evaluation/` attempt and all training provenance remain untouched. A single-GPU job first runs a one-window interface smoke for all four endpoints under `four_arm_eval_interface_smoke_retry01/`, then runs full inference/export and pinned SIU3R reduction/bootstrap in the same allocation. Submit only the evaluation job with:

```bash
scripts/submit_object_locus_competition_gc_eval_only.sh
```
