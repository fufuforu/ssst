# Short experiment: one shared InstanceQueryHead over 4 fixed windows

Question: can a single head fit **several windows that really do participate in
training**?  This is a localisation experiment under the 32/8 development
protocol, not a generalisation result and not SIU3R official mAP/PQ.

## Setup (traceable)

Manifest written before training (`instance_query/fixed4/manifest.json`), all
four scenes verified to be in the 32-scene training split, and every batch's
frame IDs **asserted** against the manifest at load time:

| scene | context | novel | provenance |
|---|---|---|---|
| scene0012_02 | [2043, 2075] | [2045, 2055] | recorded in `docs/fit_shared_window_eval.md`; single-window check passed |
| scene0010_01 | [510, 531] | [512, 522] | `pair_rng.seed(1042)`, frames re-derived then asserted |
| scene0000_00 | [3673, 3698] | [3682, 3689] | `pair_rng.seed(2000)` as recorded by the localisation script |
| scene0005_00 | [510, 529] | [512, 522] | seed 1042, recorded before training |

One head, random init, seed 42; frozen fp32 LocusGS
`cross_scene/lgs_lr1e4/ckpt_step6000`; head structure, Hungarian cost,
BCE/Dice/objectness weights, AdamW (lr 3e-4, wd 0, clip 1.0) and the inference
thresholds (objectness 0.5, mask 0.5, area 50 px) unchanged.  The only variable
is the sampling range: 4 fixed windows.  Contribution maps were computed once
per window in-process (16 windows, 263 s) and reused for every step and every
evaluation; nothing dense was cached.

Smoke (passed): 4 distinct scenes sampled and asserted, α identity max error
4.8e-7, checkpoint save/restore OK, and after the full run the frozen LocusGS
max |delta| = **0.0** with 18/20 head parameters receiving gradients.
Per-window update counts over 1600 steps: scene0012_02 397, scene0005_00 407,
scene0000_00 402, scene0010_01 394 (balanced rotation).

## Curves (novel views; IoU >= 0.5 = TP; AP50 = per-view greedy score-ordered)

| step | train4 detected/N | TP | FP | FN | AP50 | mean GT-free IoU | best-any-query IoU | same-scene extra AP50 / IoU | val8 AP50 / IoU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0/24 | 0 | 0 | 24 | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 200 | 6/24 | 6 | 2 | 18 | 0.219 | 0.196 | 0.318 | 0.000 / 0.017 | 0.000 / 0.033 |
| 400 | 5/24 | 5 | 11 | 19 | 0.219 | 0.212 | 0.433 | 0.000 / 0.096 | 0.032 / 0.138 |
| 600 | 5/24 | 5 | 2 | 19 | 0.250 | 0.150 | 0.494 | 0.083 / 0.089 | 0.087 / 0.057 |
| 800 | 9/24 | 9 | 6 | 15 | 0.312 | 0.319 | 0.475 | 0.000 / 0.055 | 0.138 / 0.086 |
| 1000 | 10/24 | 10 | 5 | 14 | 0.375 | 0.385 | 0.530 | 0.000 / 0.070 | 0.150 / 0.088 |
| 1200 | 12/24 | 12 | 11 | 12 | 0.510 | 0.432 | 0.551 | 0.108 / 0.089 | 0.006 / 0.056 |
| 1400 | 15/24 | 15 | 5 | 9 | 0.562 | 0.492 | 0.589 | 0.000 / 0.071 | 0.000 / 0.032 |
| **1600** | **19/24** | **19** | **0** | **5** | **0.812** | **0.608** | 0.633 | 0.000 / 0.059 | 0.131 / 0.045 |

Training loss components (every 50 steps) fall from ~2.1 to 0.69-0.98 with dice
0.16-0.40 and the objectness term 0.02-0.09.

## Answers

1. **Can one head fit the 4 exact windows simultaneously?**  **Yes, and it was
   still improving at step 1600**: 19 of 24 novel visible-instance records
   detected, AP50 **0.812**, **zero false positives**, mean GT-free IoU 0.608, up
   monotonically over the last three evaluations (0.375 -> 0.510 -> 0.562 ->
   0.812).  No single window is stuck; the 5 remaining misses are few-instance
   cases (the best-any-query IoU is 0.63, so usable masks exist for the rest).
2. **Transfer to another window of the same scenes?**  **No.**  The per-scene
   extra windows stay at AP50 0.000-0.108 and mean GT-free IoU 0.06-0.09 while
   their best-any-query IoU is 0.21 - i.e. masks are available but the head does
   not use them.
3. **Any improvement on the 8 unseen scenes?**  **No meaningful one** (AP50
   0.000-0.150 across steps, 3 of 55 records detected at the end, mean GT-free
   IoU 0.03-0.14).
4. **Next step?**  Fix the **shared training / mask supervision** before any
   joint training with reconstruction supervision: one head memorises 4 windows
   (0.81 AP50) yet does not transfer even to other windows of the *same* scenes,
   and the frozen representation demonstrably supports instances (best-any-query
   IoU 0.55-0.63 on the trained windows, 0.21 elsewhere).  Recommendation only -
   nothing was changed.

## Artifacts and command

```
python scripts/train_instance_query_fixed4.py --split .../cross_scene/split.json \
  --checkpoint .../lgs_lr1e4/ckpt_step6000 --manifest .../fixed4/manifest.json \
  --steps 1600 --eval-every 200 --out .../instance_query/fixed4/run
```

`instance_query/fixed4/manifest.json`, `fixed4/run/history.json` (manifest, loss
curve, per-step three-group results with per-window and per-instance detail),
`fixed4/run/instance_query_head.pt` (head + optimizer + step + RNG + frozen
reference; no 220M copy).  Representative figure panels were not rendered this
round (the script reports numbers only).
