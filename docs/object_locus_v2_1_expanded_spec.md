# Object-Locus V2.1 Expanded Training

This is a registered continuation recipe from the completed V2.1 Stage S epoch-16 checkpoint. The model, controller, loss, evaluator, data provider, and renderer are unchanged. The sole recipe change is the reconstruction peak LR (`1e-5` to `1e-6`); the object peak LR remains `1e-4`, GC alpha remains `0.01`, and the source model/optimizer/RNG are continued without reinitialization.

## Fixed source and plan

- Source checkpoint: `workspace_group_plus/object_locus_v2_1/stage_s/checkpoint_S_epoch_16.pt`
- Source checkpoint SHA256: `8461937d92965ad10bf56b401a12d9012862b2eeb5330113a15ac9998419450d`
- Source code SHA: `d9a5cef3263dc7b16ad79784560b3c30d2045b09`
- Training windows: the registered `expanded_train_windows` list, 1008 windows over 128 scenes.
- Updates: 16 complete epochs, 16128 updates; global step 1792 to 17920.
- Per-epoch ordering: `np.random.default_rng(10042 + epoch_index).permutation(1008)`.
- Understanding weight stays 1.0. No warm-up is restarted.
- Object LR is `1e-4 * m(t)` and reconstruction LR is `1e-6 * m(t)`, where `m(t)=0.1+0.9*(1+cos(pi*(t-1)/16127))/2`.

The source Stage S expansion gate remains recorded as failed. This separately authorized expanded run does not alter that result and is not described as a same-recipe continuation: it has a registered lower reconstruction LR.

## Implementation validation

- Existing V2.1 CPU contracts and gradient tests plus expanded continuation tests: 30/30 pass.
- RTX3090 smoke job `57262`, node `3dimage-13`: source model and 540 AdamW state entries restored exactly (all internal steps 1792); the isolated first update used object/reconstruction LR `1e-4 / 1e-6` and understanding weight 1.
- Smoke losses and required gradients finite; category, objectness, shared fusion, child residual, and reconstruction paths received finite nonzero gradients. Peak allocated/reserved memory: 7.146 / 7.738 GiB.
- Fixed val32 single-pair local evaluation and official export completed. The smoke ZIP/CSV/JSON schema check passed.
- Source Stage S evaluation artifacts use a four-digit minimum step suffix (`step0000`, `step1792`); the continuation resolves those exact filenames.

## Evaluation and recovery

Epochs 0, 2, 4, 8, and 16 are registered evaluations. Epoch 0 reuses the matching Stage S endpoint results for old splits and evaluates only the newly registered expanded training probe. Every epoch writes a recoverable checkpoint with the next epoch/position cursor, optimizer, RNG, and asset hashes. Evaluation restores RNG and training mode.

No task metric is an early-stop or schedule gate. Numerical/runtime failures stop the run and retain failure evidence. Checkpoints and data are not included in the review bundle.
