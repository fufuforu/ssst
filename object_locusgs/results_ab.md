# Object-aware LocusGS: A/B development result (32 train / 8 unseen)

**Not an SIU3R official evaluation.** No official mAP/PQ is computed; the
official 1860-pair test set is untouched. The model consumes GT camera poses to
build rays, whereas SIU3R is an *unposed* setting, so these numbers are not
comparable with the SIU3R leaderboard.

## 1. Pairing evidence

| item | value |
|---|---|
| source checkpoint | `workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step2000` (model-only) |
| sha256(model.pt) | `f0e791b8bb9d49deeba160f0a5fcf75594e4a0638d79ed79b9021ccd6da31c6d` |
| preset | `train_siu3r_locusgs_recon_bounded_delta_frozen_radius` |
| source step | 2000 |
| git commit at smoke time | `3bedbe64` (experiment code: `05f6b34`, `e68c7fe`, `de09d45`) |
| shared batch plan | `object_locusgs/plan_6000.json`, sha256 `a2a65c13…a3c3d08bb` |
| plan coverage | 6000 steps, 32/32 training scenes drawn (157–213 windows each), 0 validation scenes |
| initial weight hash, arm A / arm B | `7c2400384441002f…` (existing) and `c23d810b1405c0f9…` (new), **identical in both arms** |
| fixed validation windows | the original 2+2 windows of `lgs_lr1e4` (scene0059_00 … scene0695_00) |

Fresh AdamW per arm (no optimizer/RNG state exists in the source checkpoint and
none was pretended to be restored). Existing LocusGS parameters lr 1e-5, new
parameters lr 1e-4, 200-step linear warm-up then cosine to 2 % of peak, 6000
experiment-local steps, grad clip 1.0, betas (0.9, 0.95). Step 1 of both arms
reads the same window (`scene0000_02`, context `[5238, 5255]`, novel
`[5239, 5250]`) and produces the same loss `0.0556` (recon `0.0556`).

Arm A never applied the relation module (`token_update_norm = None`, gate
gradient absent), arm B drove the gate to `tanh(g) ≈ 0.022` by step 6000
(token update norm ≈ 4.7).

## 2. Headline result (mean over the 8 unseen scenes, step 6000)

| metric | arm A | arm B | B − A |
|---|---|---|---|
| context PSNR | 21.027 | 21.081 | **+0.055** |
| novel PSNR | 20.385 | 20.455 | **+0.070** |
| context SSIM | 0.6914 | 0.6920 | +0.0006 |
| novel SSIM | 0.6831 | 0.6843 | +0.0012 |
| grey-image baselines (ctx / novel) | 10.808 / 10.698 | 10.808 / 10.698 | — |
| semantic mIoU (valid GT, α>0.05) | 0.1666 | 0.1496 | **−0.0170** |
| instance AP50 (class-agnostic, novel) | 0.0000 | 0.0354 | **+0.0354** |
| TP / FP / FN | 0 / 2 / 55 | 2 / 8 / 53 | |
| GT buckets (small / medium / large) | 0/11, 0/24, 0/20 | 0/11, **2**/24, 0/20 | |

Per-scene novel-PSNR deltas (B − A): +0.27, +0.28, −0.14, +0.31, −0.13, −0.36,
+0.35, −0.02 dB — a scatter around zero, not a systematic reconstruction change.

Curves (every 1000 steps, mean over the 8 scenes):

| step | A ctx/novel PSNR | A mIoU | A AP50 | B ctx/novel PSNR | B mIoU | B AP50 |
|---|---|---|---|---|---|---|
| 0 | 19.13 / 18.91 | 0.009 | 0.000 | 19.13 / 18.91 | 0.009 | 0.000 |
| 1000 | 20.49 / 20.11 | 0.145 | 0.000 | 20.47 / 20.08 | 0.163 | 0.000 |
| 2000 | 20.72 / 20.19 | 0.170 | 0.000 | 20.72 / 20.23 | 0.168 | 0.000 |
| 3000 | 20.80 / 20.34 | 0.163 | 0.000 | 20.74 / 20.24 | 0.152 | 0.031 |
| 4000 | 20.98 / 20.35 | 0.163 | 0.000 | 21.03 / 20.33 | 0.160 | 0.004 |
| 5000 | 21.00 / 20.37 | 0.168 | 0.000 | 21.06 / 20.50 | 0.144 | 0.008 |
| 6000 | 21.03 / 20.39 | 0.167 | 0.000 | 21.08 / 20.46 | 0.150 | 0.035 |

## 3. Answers to the four questions

**1. Does B improve semantics/instances on unseen scenes?**
*Semantics: no.* B's mIoU is 0.017 lower (0.150 vs 0.167), and it is lower in 4 of
8 scenes (sometimes by −0.09). *Instances: only nominally.* Through the frozen
GT-free reader B produces 2 true positives (AP50 0.035) where A produces none,
but the absolute level is at the edge of detectability: 2 of 55 GT instances are
recovered and B also emits more false positives (8 vs 2). This is a weak,
not-yet-reliable direction, not a demonstrated capability.

**2. Does B change token/anchor/GS instance organisation, or only logits?**
It changes the *embedding* organisation measurably, and the geometric
organisation only marginally:

| diagnostic (GT-assisted, 8-scene mean) | A | B |
|---|---|---|
| embedding cosine, same instance | 0.7388 | **0.7512** |
| embedding cosine, different instance | 0.3369 | **0.2806** |
| same − different gap | 0.4019 | **0.4706** |
| token instance-contribution purity | 0.4326 | 0.4390 |
| GS instance-contribution purity | 0.4444 | 0.4555 |
| GS mass landing on stuff/void | 0.5745 | **0.5517** |

So B separates instances clearly better in the shared embedding space, and moves
a little more GS mass onto thing instances, while the per-token/per-GS purity
improves only slightly (≈ +0.006 / +0.011). The relation update is therefore
doing something structural, but the geometric re-organisation is much weaker
than the embedding-level change.

**3. Does B cost reconstruction?**
No measurable cost: +0.070 dB novel PSNR and +0.0012 novel SSIM, with per-scene
deltas scattered symmetrically around zero. Both arms stay far above the
grey-image baseline (+9.7 dB novel), keep the frozen 0.15 decode radius, keep
α close to 1, and neither hit the α-collapse / grey-collapse / non-finite guards.

**4. Where does the failure sit: learned representation, GT-free read-out, or
reconstruction stability?**
Not reconstruction stability (both arms are healthy). The dominant limit is the
**learned representation's confidence**, with the fixed reader acting as a hard
gate:

* the reader only considers context pixels with `α ≥ 0.5`, predicted class a
  thing and predicted class probability `≥ 0.5`; on those pixels the class
  probability p90 is only 0.344 (A) / 0.383 (B) and just 1.6 % / 5.1 % of them
  clear the 0.5 gate;
* consequently A produced on average 0.125 prototypes per scene and essentially
  no instance output, so its AP50 is structurally 0 rather than a clustering
  failure, while B produced 0.625 prototypes per scene and the small amount of
  detection that follows;
* mIoU 0.15–0.17 shows the semantic head itself is still weak: the understanding
  loss is ramped in over 1000 steps with weight 0.05 against a reconstruction
  objective that dominates, on 32 training scenes only.

Both readings of the read-out failure are reported rather than resolved: the
reader's 0.5 probability gate is a fixed pre-registered rule that was *not*
retuned (retuning it per arm would break the comparison), so the low AP50 must
not be read as "the heads cannot cluster".

## 4. Smoke / guard evidence (`object_locusgs/smoke_report.json`)

* step-0 A/B/source parity: max abs difference **0.0** for RGB, depth, alpha,
  14-d Gaussians, anchors and radii (the relation residual is exactly 0);
* attribute shapes `[1,65536,20]` / `[1,65536,16]`, Gaussian-head token order
  reproduced to 0.0, the 64 GS of a token do not share one attribute vector,
  per-GS embeddings unit to 1.8e-7;
* semantic conventions: class 0 supervised (99 024 px in the smoke batch), 255
  ignored, stuff never counted as a thing, instance keys `(sem+1)*1000+id`
  shared across the context views, max instance id 59 (<1000);
* attribute-render alpha == RGB alpha to **0.0**, `Σ_classes p = 1` to 2.4e-7 on
  covered pixels, all losses finite;
* `0.05·L_sem + 0.10·L_inst` alone back-propagates into the attribute heads,
  `anchor_decoder.mu/rho`, the layer-10/11/12 decoder blocks and refinement
  heads, `activation_head.deconv` and `gs_tokens`; arm A leaves the relation
  module untouched, arm B with gate ≠ 0 also reaches `relation.{norm,key,value,out}`;
* frozen decode radius exactly 0.15 over the short run, no NaN/Inf, no alpha or
  anchor jump;
* checkpoint save/restore: model and AdamW state restored identically, torch /
  CUDA / numpy RNG plus the experiment-local step are stored;
* CUDA determinism: `torch.use_deterministic_algorithms(True)` enables
  `torch.utils.deterministic.fill_uninitialized_memory`, and gsplat's partially
  written `means2d` buffer then becomes NaN, which poisons the canonical
  Gaussian-visibility term (loss = NaN). The deterministic setting is
  therefore unusable here. Residual randomness was measured instead: repeating
  the *identical* step gives loss gaps of 1.8e-6 … 1.1e-4 and grad-norm gaps up
  to 6.5e-3; two independent 3-step runs share identical frame ids and identical
  initial weights but end with different weight hashes. Nondeterminism comes
  from the rasterizer (atomics), not from the data plan or the optimiser.

## 5. Official-pair shape / leakage smoke

`object_locusgs/official_pair_smoke.json`: one official record
(`scene0011_00`, context `[1727, 1744]`, 4 novel views) forward pass produces
`[1,6,20,256,256]` semantic and `[1,6,16,256,256]` instance maps with the encoder
reading exactly 2 views. Replacing the novel records' RGB, semantic and instance
GT with garbage leaves every output bit-identical (max abs Δ = 0.0), i.e. novel
GT never enters the model. This is the interface reserved for a future official
evaluation; it is not a metric.

## 6. Artifacts

| file | content |
|---|---|
| `object_locusgs/plan_6000.json` | the pre-registered paired batch plan |
| `object_locusgs/smoke_report.json` | all eight pre-run checks |
| `object_locusgs/eval_report_step6000.json` | per-scene/per-arm dev metrics + diagnostics |
| `object_locusgs/official_pair_smoke.json` | official-pair shape/leakage smoke |
| `object_locusgs/val_curves_arm_{a,b}.jsonl` | every-1000-step curves, per scene |
| `object_locusgs/manifest_arm_{a,b}.json` | config, hashes, optimizer groups, val windows |
| `workspace_object_locusgs/arm_{a,b}/ckpt_step{0,2000,6000}/` | resumable checkpoints (not in Git) |
| `workspace_object_locusgs/images/` | RGB \| GT \| prediction \| error tiles |

Reproduce:

```bash
sbatch object_locusgs/submit_arm.sh a     # 6000 steps, ~52 min on one 3090
sbatch object_locusgs/submit_arm.sh b
sbatch object_locusgs/submit_eval.sh 6000
python -m pytest tests/test_object_locusgs.py
```
