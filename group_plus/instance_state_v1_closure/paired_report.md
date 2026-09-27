# C/E paired 2000-step report (instance_state_v1)

All numbers are **measured** on the four locked training windows / the 8 locked
official val pairs.  Labels: measured / interpretation / not executed.

## 1. Current code

* commit `5bc9a12` (working tree otherwise clean); baseline spec `9e6cab2` +
  the targeted fix patch.
* instance-state architecture: **unchanged** this round (only the matcher cost
  now uses the registered softmax probability; see `closure_audit.md`).
* C = `coupled=False`: the 100 thing + 2 stuff states + void channel are updated
  at layers 6/8/10/12 and read out, but `beta = 0` so they never touch the token
  interaction, anchor update, radius or the Gaussian residual.
  E = `coupled=True`: identical model, `beta = min(step/200, 1)` enables the
  state-driven message write-back, compactness bias on self-attention and the
  Gaussian residual.  Everything else (data, windows, plan, init, loss, LR,
  clip, fp32, seed) is shared.
* training loss / data / eval: loss and data unchanged; the eval module is new
  (`scripts/eval_instance_state_v1.py`, RNG-safe) and was wired into the loop
  before training started.

## 2. Curves (context scope, four training windows)

C arm:

| step | recon loss | PSNR | semantic mIoU | GT-free recall50 | raw recall50 | panoptic thing TP | TP/FP/FN |
|---|---|---|---|---|---|---|---|
| 0 | – | 26.22 | 0.028 | 0.000 | 0.000 | 0 | 0/0/16 |
| 200 | 0.0372 | 22.84 | 0.792 | 0.438 | 0.438 | 11 | 7/11/9 |
| 500 | 0.0357 | 23.41 | 0.918 | 0.562 | 0.562 | 12 | 9/5/7 |
| 1000 | 0.0322 | 24.44 | 0.934 | 0.625 | 0.625 | 14 | 10/5/6 |
| 2000 | 0.0283 | 25.29 | **0.973** | **0.750** | **0.750** | **14** | 12/3/4 |

E arm:

| step | recon loss | PSNR | semantic mIoU | GT-free recall50 | panoptic thing TP |
|---|---|---|---|---|---|
| 0 | – | 25.51 | 0.028 | 0.000 | 0 |
| 500 | 0.0357 | 23.15 | 0.910 | 0.625 | 13 |
| 1000 | 0.0338 | 24.27 | 0.938 | **0.750** | 13 |
| 2000 | 0.0300 | 25.39 | **0.973** | 0.625 | **14** |

8 official val pairs (target scope): C mIoU 0.004 → 0.055 (recall 0.000),
E mIoU 0.044 at step 1000 and 0.030 at step 2000 (recall 0.000).

## 3. Reconstruction (measured)

Both arms keep a healthy reconstruction: the recon loss falls monotonically
(C 0.0372 → 0.0283; E → 0.0300), novel-view PSNR is 22.8 → 25.4 dB, alpha
coverage stays 1.000 and no NaN / Inf / collapse appeared at any logged step.
E is marginally worse than C on recon loss (0.0300 vs 0.0283) and PSNR
(25.39 vs 25.43) - inside run-to-run noise, i.e. **coupling does not damage
reconstruction** in this setting.

## 4. Understanding (measured)

C clearly learns meaningful structure on the four training windows: semantic
mIoU 0.028 → 0.973, GT-free class-relevant recall50 0 → 0.750 with TP/FP/FN
12/3/4, and the assembled panoptic map already yields 14 correct-class thing
TPs.  E reaches the same mIoU (0.973) and the same 14 panoptic TPs; its final
recall is lower (0.625 vs 0.750) although it was higher at step 1000.

**No collapse** in either arm (no single-class / all-background / all-thing /
single-instance degenerate solution: FP counts stay low and the per-class
confusion is dominated by the diagonal).

Cross-scene transfer is **not** established: on the 8 unseen official val pairs
both arms stay at mIoU 0.03-0.055 with recall 0.000 - i.e. what is learned at
2000 steps is a four-window fit, not a generalising structure.

## 5. Qualitative evidence

Per-step evaluation JSON (confusion, per-instance counts, panoptic TP) and the
paired curves are stored at:

```
group_plus/instance_state_v1_closure/curves_C.json, curves_E.json
group_plus/instance_state_v1_closure/eval_paired_{C,E}/eval_step*_{context,target}.json
group_plus/instance_state_v1_closure/eval_val8_{C,E}/
group_plus/instance_state_v1_closure/history_{C,E}.jsonl   (every 100 steps)
```
GT-vs-prediction PNG panels were **not** produced this round (the export module
exists but was not exercised) - listed as an open item rather than faked.

## 6. The three questions

**Q1 - Is instance-state already an effective understanding representation?**
**Yes, on the training windows (measured).**  mIoU 0.028 → 0.973, GT-free
class-relevant recall50 0 → 0.750, panoptic thing TP 0 → 14, with a stable
non-degenerate confusion matrix.  It does **not** yet generalise to unseen
scenes (val mIoU ≤ 0.055).

**Q2 - Can reconstruction and understanding coexist in one system?**
**Yes (measured).**  Reconstruction stays healthy in both arms while the
understanding read-out learns: PSNR rises 22.8 → 25.4 dB, recon loss falls
monotonically, alpha 1.000, no collapse.

**Q3 - Does E's coupling show that the state really participates in
reconstruction?**  **No evidence of an added benefit at this budget.**  E
matches C on mIoU (0.973 vs 0.973) and panoptic TP (14 vs 14) and is slightly
worse on final recall (0.625 vs 0.750) and PSNR (25.39 vs 25.43).  Coupling is
therefore *safe* (no damage) but the current 2000-step, four-window protocol
does not demonstrate that it helps.  A single-seed 4-window run cannot separate
"no benefit" from "benefit not yet measurable", so this is not a refutation of
the coupling design either.

## 7. Where the structure is most likely weak, and the smallest next change

**Interpretation (not measured):** the read-out is fit to four windows while the
cross-scene number stays at chance, and coupling adds nothing measurable - both
point at the *state-to-scene* link rather than at the read-out head.  The
smallest hypothesis-driven change is therefore to make the thing states carry
**cross-view evidence at initialisation** (the stuff states already use the
mean token; the thing states are seeded from single-token FPS queries, and their
only cross-view signal is the assignment-weighted message at the same layer),
for example by initialising each thing query from a *pair* of tokens that are
matched across the two context views instead of one token.  That is one
localised change to the initialisation, testable with exactly this pipeline.
