# Object-Locus Text Refer v1

This package reads official SIU3R `train_refer_seg_data.json`, `val_refer_seg_data.json`, and `val_refer_pair.json` and predicts a scene-local existing thing slot from a raw description. It does not condition or modify reconstruction geometry.

## Fixed behavior

- Frozen visual model produces detached `q [B,100,256]` and `P [B,65536,100]`.
- Frozen `CLIPTextModel(openai/clip-vit-base-patch32)` yields FP32 `T [B,77,512]`; its tokenizer ids and mask are kept.
- Head projects text through `Linear(512,256)+LayerNorm`, applies two independent text-query/object-key-value attention blocks, reads the actual EOT token, and predicts 100 thing scores plus null. `softmax` spans all 101 scores.
- Soft Gaussian membership is `sum_j pi_j * P_gj`; inference selects one slot once and reuses its raw membership for all views. Null gives zero membership.
- Context training objective is `CE + 5 * probability BCE + 5 * smoothed Dice`. Visible unmatched objects omit slot CE; missing annotations are not null negatives; truly invisible targets map to slot 100.
- Validation expands every expression in official pair order. This differs from official dataset random-one-description sampling and must not be called official mIoU_t. Official target-view identifiers are not present in the current pair file; novel evaluation accepts explicit cameras but is not an aligned official benchmark.

## Interfaces

Training starts from each real two-view provider context, then intersects its normalized `frame2object` IDs with described objects and currently visible valid thing IDs. Valid pixels use `(sem >= 0) & (sem <= 19) & ((sem < 2) | (ins > 0))`; thing instances use `sem >= 2`. The validation-only `SIU3RReferDataset` keeps the official pair order and raw descriptions. IDs stay scene-local, and `packed_panoptic % 1000` is the instance identity.

`scripts/train_object_locus_text_refer.py` is a frozen-visual/head-only training entry and requires explicit `--max-updates`. It validates and strictly loads Full1201 epoch 6 and always runs the visual model at checkpoint `completed_exposures=50064` (all four injection betas are 0.1). Defaults are batch 1, LR `1e-4`, matrix WD `0.05`, bias/norm WD `0`, AdamW betas `(0.9,0.95)`, eps `1e-8`, FP32, grad clip `1.0`. `scripts/eval_object_locus_text_refer.py` now loads the visual model, strict head checkpoint, fixed CLIP text model and official context pairs, writes `context_refer_records.json` and `context_refer_metrics.json`, and preserves failed expressions as zero-IoU records. Its IoU threshold is probability `> 0.5`; its expression aggregate is context-only, not paper `mIoU_t`.

The CPU contract command is:

```bash
PYTHONPATH=. python -m unittest tests.test_object_locus_text_refer_contracts -v
```

The independent entry points use a new adapter around the registered runtime. A future head run remains opt-in through explicit `--max-updates`; this repair smoke is limited to two temporary updates and one official validation expression.

Example for a future intentional head run (not executed in this branch):

```bash
python scripts/train_object_locus_text_refer.py \
  --max-updates 1000 \
  --checkpoint /space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt \
  --data-root /space/mawb/SIU3R/data/scannet \
  --output /space/mawb/ssst/group_plus/object_locus_text_refer_v1/head_1000.pt
```
