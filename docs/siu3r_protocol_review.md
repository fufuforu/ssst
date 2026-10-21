# SIU3R official protocol (read-only review) + query-head interface

Reviewed from `/space/mawb/SIU3R` (README, `configs/main.yaml`, `src/config.py`,
`src/evaluator.py`, `src/data/components/scannet_dataset.py`,
`src/utils/scannet_constant.py`).  Nothing in that repository was modified.

## 1. Official test entry and pairing

* Entry: `python src/run.py experiment=siu3r_test mode=test ckpt_path=<ckpt>`
  (README "Evaluation").  `experiment=siu3r_test` is only the run/logger name;
  the behaviour is selected by `mode=test`.
* `mode in {val,test}` forces `num_extra_target_views = 4`
  (`src/config.py:180-181`), so the official view layout is
  **2 context + 2 extra target = ... `target_ids` has 6 entries** in the json
  (2 context + 4 targets).
* `data/scannet/val_pair.json` is a **list of 1860 records**
  `{"scan": str, "context_ids": [2 ints], "target_ids": [6 ints], "iou": float}`.
  Example: `scene0011_00`, context `[1727, 1744]`,
  target `[1727, 1729, 1732, 1738, 1739, 1744]` (the target list already
  contains the two context ids - the same convention the ssst adapter uses).

## 2. Inputs, images, cameras

| item | official SIU3R |
|---|---|
| dataset | processed ScanNet (`data/scannet/{train,val}`) |
| image size | `image_width = image_height = 256` |
| segmentation task | `seg_task: panoptic` |
| context views | 2 |
| target views | 4 (test) |
| camera convention | pixelSplat: intrinsics **normalised** (row 0 / width, row 1 / height); extrinsics **OpenCV camera-to-world** (+X right, +Y down, +Z into the screen) |
| camera input to the model | **unposed** - SIU3R predicts the cameras; poses are not given as input |

## 3. Classes, ignore rule, metrics

* Classes: `PANOPTIC_SEMANTIC2NAME` = **20 ScanNet classes**, 1-based
  (1 wall, 2 floor, 3 cabinet, 4 bed, 5 chair, 6 sofa, 7 table, 8 door,
  9 window, 10 bookshelf, ... 20 otherfurniture); `THING_CLASSES` = 18
  (indices 2-19), `STUFF_CLASSES` = 2 (indices 0,1).
* Ignore: the dataset is constructed with `ignore_index=255`
  (`src/data/components/scannet_dataset.py:70`).
* Metrics (`EvaluatorCfg`, all default True): context mIoU, context PQ,
  context mAP, target mIoU, target PQ, target mAP, image quality
  (PSNR/SSIM/LPIPS), depth quality.

## 4. Prediction format

Per pair, the evaluator reads a scene directory with
`context_seg_pred/` and `target_seg_pred/` (plus matching `*_seg_gt/`):

* one **PNG per frame** whose instance id is packed 24-bit
  `R + 256*G + 65536*B` (the same packing as the GT panoptic maps),
  read in `Evaluator.process_segmentation`;
* an optional `pred.json` next to the PNGs; when present it supplies the
  per-instance **labels**; when absent the labels are taken from the predicted
  semantic map at the instance mask (`pred_semantics[mask][0] - 1`, i.e. 1-based
  GT ids converted to 0-based);
* instance masks + labels (+ scores when available) are fed to
  `torchmetrics.MeanAveragePrecision` and `PanopticQuality` with the
  `things`/`stuffs` lists above.

## 5. Consequences for this project (must be stated in any comparison)

* The official table is produced by **one fixed pairing (`val_pair.json`, 1860
  records) and one evaluator**; our 32-train / 8-validation split is a
  development split, and its context-oracle IoU, class-agnostic AP and
  reconstruction PSNR are **not** comparable numbers.
* **Input difference**: SIU3R is an **unposed** method - it does not receive
  camera poses.  TokenGS/LocusGS here consume GT camera poses and use them to
  build rays.  Even with identical scenes, frames and evaluator, that is a
  different input regime and must be flagged in the final table.
* Class-agnostic masks alone cannot be scored by the official evaluator: PQ/mAP
  need labels.  Our phase-1 output must therefore keep a semantic head slot.

## 6. Planned interface for the class-agnostic query head (not yet implemented)

```
class InstanceQueryHead(nn.Module):
    """Frozen backbone -> 100 class-agnostic queries -> per-token assignment."""
    queries: nn.Parameter            # [100, C]
    token_proj: Linear              # scene tokens -> C
    cross_attn: nn.MultiheadAttention(C)
    assign: Linear(C, T)            # query -> token logits (T = 1024 tokens)
    objectness: Linear(C, 1)        # query is a valid object
    # reserved for phase 2:
    semantic: Linear(C, 20)         # ScanNet class logits (1-based ids)
```

* **Inputs**: encoder tokens of the 2 context frames only
  (`model.forward_encoder` on the context batch) - no GT, no novel images.
* **Mask rendering (corrected direction)**: the assignment is **token -> slot**.
  For every token the head produces a distribution over the 100 query slots plus
  one background slot, `A[t, q] = softmax_q(logits)[t, q]` with
  `sum_q A[t, q] = 1`, and
  `mask_q(p) = sum_t A[t, q] * M_t(p)`, `mask_bg(p) = sum_t A[t, bg] * M_t(p)`,
  where `M_t` is the verified per-token compositing contribution map
  (`scripts/token_instance_compositing.py` / `token_instance_oracle.py`).
  The correctness check is therefore
  **`sum_q mask_q(p) + mask_bg(p) = alpha(p)`** (verified to 4e-7 in
  `scripts/train_instance_query_overfit.py`).
* **Official export**: what the SIU3R evaluator consumes is **one packed
  prediction PNG per frame** (24-bit `R + 256G + 65536B` instance ids, per
  section 4).  The per-query masks are only the intermediate representation that
  is reduced to that single packed map per frame.
* **Training loss**: scene-level Hungarian matching between queries and GT
  instances using the context+novel masks, with Dice + BCE on the matched pairs
  plus a no-object term; **unannotated pixels (semantic void / instance 0) are
  excluded from the BCE negative set**, never counted as background.
* **Evaluation hooks**: per-query instance mask (packed PNG, 24-bit), objectness
  score, and a reserved semantic logit slot, so the same outputs can be fed to
  the official evaluator once labels exist.

## 7. Status

* `tokengs/models/instance_query_head.py` + `scripts/train_instance_query_overfit.py`
  are implemented and runnable (see `docs/instance_query_phase1.md`).
* The **32/8 shared-head training** and the **official-protocol export** are not
  done.  The 32/8 split is the *next-stage development set*; the final comparison
  must instead use `val_pair.json`, the complete semantic/instance output and the
  official evaluator, and must state that this model consumes **GT camera poses**
  while SIU3R is an **unposed** method.
