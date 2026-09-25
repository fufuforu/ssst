# G0+ (background-slot pixel supervision): single-variable round

**Scope.** 32 train / 8 unseen development split only; AP50/mIoU here are
development numbers, **not** SIU3R official mAP/PQ. The model consumes GT camera
poses to build its rays; SIU3R is an unposed setting. G1 was not trained, the
query decoder was not replaced, and the token→101-slot structure is unchanged.

## 1. The single training change

`L_inst_new = L_inst_original + 1.0 · L_bg`, everything else identical to G0
(same model, queries, assignment softmax, Hungarian matching, instance loss,
semantic loss, reconstruction loss, optimizer, LR plan, data plan).

`L_bg` (two context views only, on pixels with rendered `alpha > 0.5`):

| pixel class | target for `p_bg = bg_mass / clamp(alpha.detach(), 0.5)` |
|---|---|
| semantic 0/1 (stuff) | 1 |
| semantic 2..19 with a valid instance id (thing) | 0 |
| semantic 255, unannotated, uncovered | excluded |

Both classes present → each ½; one class → that class; neither → 0. No new
parameters, no clearing/entropy/class-weight/temperature/area terms.

**Provenance.** G0+ starts from G0's own `ckpt_step0`; all parameter blocks hash
identically to a fresh seed-42 G0 (`reconstruction 64c4d09c…`, `attributes
c92ef647…`, `groups c877aa84…`, `feedback 8617fa23…`) and the first forward before
any update is bit-identical (RGB/depth/alpha/Gaussians/slot logits/class logits all
Δ = 0.0). With the new term disabled the loss equals G0's exactly (gap 0.0), and
with it enabled `L_total = L_G0 + λ(step)·0.05·1.0·L_bg` holds to 0.0 — the only
difference is the added term (smoke report, all checks PASS). Target pixels were
re-derived independently: 61 400 stuff / 1 285 thing / 68 387 ignored, no 255 or
uncovered pixel in either class, thing pixels without an instance id excluded.
Gradient direction on real data: stuff-only pixels push the background assignment
**up** (bias grad −0.99), thing-only pixels push it **down** (+0.011), and the
gradient reaches the group parameters.

## 2. Unified scoring convention (applied to G0 and G0+ alike)

21-way convention: classes 0/1 stuff, 2–19 thing, 20 no-object.

* `P(foreground) = 1 − softmax(21)[20]` — **diagnostic only**
* `P(thing) = softmax(21)[2:20].sum()` — **the gate**: `P(thing) ≥ 0.5`, predicted
  20-class label must be a thing class, then `mask > 0.5`, `area ≥ 50 px`

The legacy `sigmoid(z20)` reader stays in the repository untouched
(`group_locusgs/`), and the previous G0 report keeps its legacy numbers; all
numbers below use the unified rule. In these four checkpoints the class head puts
≈0 mass on the stuff classes and every group's argmax is a thing class
(`thing_class_fraction = 1.0`), so `P(foreground)` and `P(thing)` coincide
numerically and the thing-class gate never fires — reported separately anyway.

## 3. Results (unified rule, identical windows/steps)

Unseen windows (2 novel views, 55 GT instances):

| run | ctx / novel PSNR | novel SSIM | mIoU | AP50 | TP/FP/FN | bg mass stuff/thing | active groups | best-over-groups IoU |
|---|---|---|---|---|---|---|---|---|
| G0 step3000 | 17.82 / 17.56 | 0.625 | 0.161 | 0.0724 | 4/91/51 | 0.000 / 0.000 | 9.9 | 0.258 |
| G0 step6000 | 19.42 / 19.29 | 0.654 | 0.173 | 0.0451 | 6/129/49 | 0.000 / 0.000 | 9.8 | 0.316 |
| **G0+ step3000** | 18.27 / 17.95 | 0.635 | 0.163 | 0.0042 | 1/56/54 | 0.325 / 0.220 | 6.0 | 0.199 |
| **G0+ step6000** | 19.62 / 19.16 | 0.660 | 0.172 | **0.1375** | 3/71/52 | **0.549 / 0.250** | 6.6 | 0.242 |
| grey baseline | 10.81 / 10.70 | — | — | — | — | — | — | — |

Buckets at step 6000: G0 small 0/11, medium 2/24, large 4/20 → G0+ small 0/11,
medium 0/24, large 3/20.

Training windows actually drawn by the shared plan (7 distinct scenes; **not** a
statement about all 32 training scenes):

| run | novel PSNR | mIoU | AP50 | TP/FP/FN | best-over-groups |
|---|---|---|---|---|---|
| G0 step6000 | 16.98 | 0.228 | 0.0000 | 0/108/30 | 0.109 |
| G0+ step6000 | 16.99 | 0.200 | 0.0054 | 2/62/28 | 0.079 |

Per-scene (unseen, step 6000, TP/FP/FN over the two novel views):

| scene | G0 | G0+ |
|---|---|---|
| scene0059_00 | 2/18/17 AP 0.053 | 0/12/19 AP 0.000 |
| scene0072_02 | 2/6/2 AP 0.167 | 0/5/4 AP 0.000 |
| scene0132_01 | 1/16/9 AP 0.017 | 0/8/10 AP 0.000 |
| scene0472_01 | 0/16/2 AP 0.000 | **2/7/0 AP 1.000** |
| scene0559_01 | 0/21/10 AP 0.000 | 1/9/9 AP 0.100 |
| scene0568_02 | 0/16/2 AP 0.000 | 0/12/2 AP 0.000 |
| scene0615_00 | 0/18/6 AP 0.000 | 0/9/6 AP 0.000 |
| scene0695_00 | 1/18/1 AP 0.125 | 0/9/2 AP 0.000 |

## 4. Background slot, gates and mask capability

| metric (unseen, mean) | G0 step6000 | G0+ step6000 |
|---|---|---|
| bg mass on GT stuff pixels | 0.000 | **0.549** |
| bg mass on GT thing pixels | 0.000 | 0.250 |
| bg probability (÷α .detach) stuff / thing | 0.000 / 0.000 | 0.550 / 0.250 |
| background share of rendered mass | 0.0 % | 37.8 % |
| slot entropy / max prob | 0.65 / 0.745 | 0.90 / 0.649 |
| effective groups (share > 1/100) | 9.8 | 6.6 |
| novel gate counts score→class→mask→area | 228→228→136→135 | 180→180→75→74 |
| best-over-groups IoU / recall@0.5 (diagnostic) | 0.316 | 0.242 |

The best-over-groups number is GT-assisted and uses **no score gate**, but it
still applies the fixed `mask > 0.5` / `area ≥ 50` thresholds: it describes what
the current masks can reach under this read-out, **not** a theoretical upper
bound of the architecture or the token representation.

## 5. Answers

1. **Did `L_bg` bring stuff back to the background slot?** Yes, directionally:
   background mass on GT stuff went 0.000 → 0.549 (and the background share of
   all rendered mass 0 % → 37.8 %), with stuff ≫ thing (0.549 vs 0.250 at step
   6000). It is not a clean stuff-only sink: the class-balanced BCE with a single
   *shared* background logit settles near `mean_stuff + mean_thing ≈ 1`, which is
   exactly what is measured (0.55 + 0.25).
2. **Did it improve the mask shape?** No. The GT-assisted best-over-groups IoU
   fell 0.316 → 0.242 on unseen windows and 0.109 → 0.079 on training windows
   (better in 3 of 8 scenes, worse in 4, equal in 1), and the mean best IoU per
   GT stays ≈0.24, i.e. the group masks are still not instance-shaped.
3. **Did it reduce FP / increase TP?** FP fell 129 → 71 (−45 %) and the aggregate
   novel AP50 rose 0.045 → 0.138 (3.0×), but TP fell 6 → 3 and recall stayed
   ≈5 % (3/55); the AP50 gain is precision-driven and concentrated in one scene
   (`scene0472_01`: 0 TP/16 FP → 2 TP/7 FP, AP 1.0), while four scenes that had a
   weak detection in G0 lost it. The number of predictions passing the three
   gates halved (135 → 74).
4. **Did it hurt reconstruction?** No measurable cost: context PSNR +0.20 dB,
   novel PSNR −0.13 dB, novel SSIM +0.006, mIoU −0.0015, mask/alpha conservation
   ~1.3e-6 throughout, alpha and locality healthy and no guard ever fired.
5. **Verdict.** The missing background supervision was a real contributor to the
   false-positive flood (removing mass from spurious group masks halves the FP
   count and triples AP50), but it does **not** fix mask quality: the
   GT-assisted mask ceiling does not improve, recall is unchanged, and the
   background slot also absorbs 25 % of the thing-pixel alpha. The bottleneck
   remains the group masks / effective group count, not the score reading and not
   the gate chain.

### Next single variable (proposal only — nothing extra was run)

Replace the **single shared `background_bias` scalar** with a **per-token
background logit** (`Linear(C→1)` on the same token features that feed the 100
group similarities), keeping the 101-way softmax, the queries, the Hungarian
matching, every loss, the optimizer and the schedule unchanged, and using G0+ as
the new baseline. Evidence for this over alternatives: the measured background
masses (0.549 stuff / 0.250 thing) sit exactly at the class-balanced equilibrium
that a shared logit forces, while the effective group count fell to 6.6 and the
mask ceiling stayed ≈0.24 — the assignment has no per-token degree of freedom for
"this local unit is background". (Replacing the assignment with
`UnifiedObjectQueryHead` is *not* proposed here: it also changes the assignment
normalisation/temperature and the background-slot convention, so it needs its own
designed control rather than being called a single variable.)

## 6. Artifacts

| file | content |
|---|---|
| `group_plus/smoke_report.json` | G0+ smoke (init identity, loss identity, targets, gradient direction) |
| `group_plus/g0_rescore_v2.json` | read-only re-score of the existing G0 under `P(thing) ≥ 0.5` |
| `group_plus/eval_step3000_6000.json` | paired G0 / G0+ evaluation, unseen + training windows, per scene |
| `group_plus/manifest_g0plus.json`, `init_report_g0plus.json` | config, provenance, block hashes |
| `group_plus/val_curves_g0plus.jsonl`, `train_log_g0plus.txt` | every-500-step curves and step log |
| `group_plus/tile_*.png` | GT \| RGB render \| GT thing/stuff \| background mass \| GT-free instance mask \| error |
| `workspace_group_plus/arm_g0plus/ckpt_step{0,3000,6000}` | resumable checkpoints (not in Git) |

Jobs: G0+ training `55312` (COMPLETED, 3179 s), paired evaluation `55331`
(COMPLETED). Re-run: `sbatch group_locusgs/submit_train_plus.sh`;
`python scripts/eval_group_plus.py --run g0=… --run g0plus=…`.
