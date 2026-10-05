# Object-Locus Text Refer v1 implementation and smoke

## Provenance

- Code baseline and checkpoint `git_sha`: `b2624d57ea9ad5a73fd6a8375bafbe4e4c263c9c`.
- Full1201 checkpoint: `/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt`; SHA256 `68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a`; internal epoch 6, 6258 updates, science SHA `7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3`, 100 thing queries, 2 input views, FP32. Strict load path is implemented in the new adapter.
- SIU3R reference audited at commit `8ea80166be76854f938e90521f1a5b688b755c87`, `src/data/components/scanrefer_dataset.py`.
- Actual `val_refer_pair.json`: 564 records of `{scene_name, context_views_id, context_objects, texts}`; `context_objects` is one scene-local raw object ID and `texts` is the raw description string. The dataset uses `objects[str(object_id)]`, and instance masks compare `packed_panoptic % 1000` with that raw ID. `panoptic_label_id` is category metadata only.
- CLIP model revision and local file hashes: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, in `text_encoder_provenance.json`.

## Implementation

- Added isolated text refer data, frozen text encoder, two-block text-to-object head, soft/hard 3D membership, original-renderer adapter, null/matching-aware loss, evaluation and standalone head training entry.
- Head outputs 101 probabilities (100 thing + null); text tokens are `[B,77,512]`, object states `[B,100,256]`, Gaussian membership `[B,65536]`.
- Loss is slot CE + `5 ×` probability BCE + `5 ×` smoothed Dice. Only the new head is in the optimizer; visual model and CLIP are frozen/eval/no-grad.
- Validation expands one raw description per current official pair in source order. This differs from official randomized-one-description sampling and is not paper `mIoU_t`. Current pair file has no aligned novel target-view list.

## Verification

- CPU contracts: 7 passed; Python compile and `git diff --check` passed. Head construction and initialization preserve the process RNG state.
- CLIP local/offline forward: token IDs and attention `[1,77]`; hidden `[1,77,512]` FP32; frozen and eval.
- Final real Slurm GPU job `58418` completed on `3dimage-11`, one RTX3090 allocated through Slurm (`CUDA_VISIBLE_DEVICES` isolated). Official first pair selected with zero earlier skips: `scene0011_00`, object `3`, frames `1239/1268`, original text preserved.
- Two temporary-head updates completed. Losses `14.81383`, `9.56815`; finite nonzero head gradient norm sums `3.58441`, `3.58117`; head changed; visual model and text encoder state digests unchanged and neither received gradients.
- Context hard-slot output shape `[1,2,256,256]`; per-view IoU `0.98069`, `0.84483`, mean `0.91276` (smoke only, random head; not a task performance claim). An additional same-sample explicit camera render returned `[1,1,256,256]` and is not an official novel benchmark.
- Peak CUDA allocated `3,694,198,784` bytes; reserved `3,990,880,256` bytes (measured from before frozen-model forward through completion).
- No formal training or full evaluation ran. Training entry requires explicit `--max-updates`.

## Delivery

Ready for task-branch commit and push. Main is not merged. Existing dirty files in `/space/mawb/ssst` were not edited; only the specifically requested new artifact directory was written there.
