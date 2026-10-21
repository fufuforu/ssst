# From-scratch object-aware LocusGS G0/G1: pair status and results

**Not an SIU3R official evaluation.** 32 training / 8 unseen development split
only; no official `val_pair.json` metric is computed, no official evaluator is
run, and no mAP/PQ is claimed. The model consumes GT camera poses to build its
rays, whereas SIU3R is an unposed setting.

## 1. Status

| item | value |
|---|---|
| jobs | `55287` (g0), `55288` (g1) — both `COMPLETED 0:0`, 6000 steps each |
| wall time | 2978 s (~50 min) per arm, both run concurrently on one 3090 |
| evaluation job | `55290` (`grp-lgs-eval`), final paired development report |
| logs | `workspace_group_locusgs/logs/train-55287.out`, `train-55288.out`, `eval-55290.out` |
| checkpoints kept per arm | `ckpt_step0` (random init), `ckpt_step3000`, `ckpt_step6000` |
| follow-up command | `sbatch group_locusgs/submit_eval.sh 6000` (rerun) / `--run ...=ckpt_step3000` for the mid checkpoint |

## 2. Strict pairing evidence

* **From scratch.** The preset declares `init_checkpoint=None` (asserted at
  startup) and the trainer never loads weights; only the seed is set before
  construction.
* **Identical initialisation.** With seed 42 and the same construction order the
  parameter blocks hash identically in both arms:

| block | parameters | sha256 prefix (g0 = g1) |
|---|---|---|
| reconstruction (12-layer LocusGS) | 220,002,620 | `64c4d09c5b15d2ce` |
| per-GS semantic head | 1,312,000 | `c92ef647efa9f029` |
| group head (100 + background) | 9,603,094 | `c877aa84413cf69c` |
| feedback module | 1,051,649 | `8617fa23ddb91fc4` |

  G1 therefore *optimises* exactly 1,051,649 parameters more than G0
  (230,917,714 → 231,969,363); the feedback module is instantiated in both arms
  with identical values but G0 never applies it and excludes it from AdamW.
* **Identical step-0 forward.** Same batch, before any update: max abs difference
  **0.0** for RGB, depth, alpha, Gaussian centres, anchors, radii, and the
  layer-10 slot logits; `tanh(gate) = 0` exactly; decode radius exactly 0.15.
* **Identical data.** Both arms replay `object_locusgs/plan_6000.json`
  (sha256 `a2a65c13…`) and assert the frame ids at every step; the plan contains
  only the 32 training scenes.
* **Identical recipe.** `L = L_recon + lambda(step)·[0.05 L_instance + 0.05 L_semantic]`,
  `lambda = min(1, step/2000)` (**1-based**, so `lambda(1) = 5e-4 ≠ 0`: the
  understanding loss is non-zero from the first update), AdamW at peak lr 1e-4
  for every parameter, 2000-step linear warm-up, cosine to 2 %, grad clip 1.0,
  fp32, `delta = tanh(delta_hat)`, frozen decode radius 0.15, 2 context + 2
  novel, layers {6,12} with weights {1/3,2/3}.

Smoke suite (`group_locusgs/smoke_report.json`, all PASS): initialisation
identity, step-0 reconstruction parity, label conventions (class 0 valid, 255
ignore, stuff 0/1 vs thing 2..19, keys `(sem+1)*1000+id`), **mask/alpha
conservation 1.2e-6**, `lambda(1)=5e-4`, finite losses, gradient routing
(group head, semantic head, shared decoder, and for G1 the feedback branch; the
gate gradient at `g=0` is non-zero, so it is not a dead end), context-only input
(tampering with novel RGB/semantic/instance GT leaves every output at Δ = 0.0),
and a full checkpoint round-trip.

## 3. Results on the 8 unseen development windows

| step | g0 ctx/novel PSNR | g0 mIoU | g0 AP50 | g1 ctx/novel PSNR | g1 mIoU | g1 AP50 |
|---|---|---|---|---|---|---|
| 0 (random) | 10.81 / 10.70 | 0.013 | 0.000 | 10.81 / 10.70 | 0.013 | 0.000 |
| 500 | 12.72 / 12.70 | 0.062 | 0.000 | 12.72 / 12.71 | 0.062 | 0.000 |
| 1000 | 16.72 / 16.83 | 0.141 | 0.062 | 16.92 / 16.94 | 0.136 | 0.000 |
| 2000 | 17.75 / 17.57 | 0.155 | 0.010 | 17.41 / 17.33 | 0.135 | 0.000 |
| 3000 | 17.82 / 17.56 | 0.161 | 0.012 | 17.92 / 18.00 | 0.163 | 0.042 |
| 4000 | 18.43 / 18.28 | 0.151 | 0.000 | 18.22 / 18.10 | 0.148 | 0.017 |
| 5000 | 19.25 / 19.18 | 0.170 | 0.000 | 18.76 / 18.56 | 0.168 | 0.006 |
| 6000 | **19.42 / 19.29** | **0.173** | 0.000 | **18.95 / 18.66** | 0.166 | **0.003** |
| grey baseline | 10.81 / 10.70 | — | — | 10.81 / 10.70 | — | — |

Step 6000 detail (GT-free rule: objectness ≥ 0.5, mask > 0.5, area ≥ 50 px,
class = argmax of the group semantic head):

| metric | g0 | g1 | G1 − G0 |
|---|---|---|---|
| novel PSNR / SSIM | 19.29 / 0.654 | 18.66 / 0.649 | **−0.63 dB / −0.005** |
| context PSNR | 19.42 | 18.95 | −0.47 dB |
| semantic mIoU (20 classes) | 0.173 | 0.166 | −0.007 |
| novel AP50 | 0.0000 | 0.0025 | +0.0025 |
| TP / FP / FN | 0 / 14 / 55 | 1 / 30 / 54 | +1 TP, +16 FP |
| per-instance recall@50 | 0.000 | 0.013 | +0.013 |
| GT buckets (small/medium/large) | 0/11, 0/24, 0/20 | 0/11, 0/24, 1/20 | |
| diagnostic: best IoU over **all** 100 groups (GT-assisted) | 0.316 (recall@50 0.151) | 0.250 (0.013) | −0.066 |

Per-scene novel PSNR deltas (G1 − G0): −0.22, −1.72, −0.47, +0.13, −1.82,
−0.39, −0.23, −0.35 dB — G1 is worse in 7 of 8 scenes. G1's single true positive
comes from one scene (`scene0559_01`, AP50 0.041); every other scene is 0.000
for both arms, so the instance difference is one detection, not a systematic
effect.

## 4. Training-window control (same GT-free rule, 7 distinct training windows)

| metric | g0 train | g0 unseen | g1 train | g1 unseen |
|---|---|---|---|---|
| ctx / novel PSNR | 16.97 / 16.98 | 19.42 / 19.29 | 16.56 / 16.64 | 18.95 / 18.66 |
| semantic mIoU | 0.228 | 0.173 | 0.226 | 0.166 |
| novel AP50 | 0.0000 | 0.0000 | 0.0107 | 0.0025 |
| TP / FP / FN | 0 / 6 / 30 | 0 / 14 / 55 | 2 / 40 / 28 | 1 / 30 / 54 |
| recall@50 | 0.000 | 0.000 | 0.032 | 0.013 |

(PSNR is not comparable across different windows — the training windows are
simply harder scenes; the semantic and instance numbers are the comparable part.)

## 5. Diagnosis

* **Reconstruction is healthy in both arms**: +8.6 dB over their own grey
  baseline, alpha ≈ 1, mask/alpha conservation 1.3–1.7e-6 throughout training,
  locality collapse fraction 0, no alpha/grey/non-finite guard ever fired.
* **The understanding objective is being optimised**: instance loss 11.52 → 6.0
  (g1) / 6.6 (g0), semantic NLL 3.00 → 1.56 / 1.63. This is not an optimisation
  failure.
* **The group→token assignment collapses**: slot entropy falls 4.49 → 0.57–0.70
  nats (uniform over 101 slots would be 4.62), `active_group_share` 0.38 → 0.09–0.12,
  the background slot receives ≈ 0 mass, and only ~9–12 of the 100 groups are
  used per scene. Token↔group purity (diagnostic) is 0.72/0.75, i.e. tokens do
  follow their dominant group, but there are far too few effective groups.
* **Objectness is uninformative**: 87.5 % (g0) / 92.3 % (g1) of the 100 groups
  exceed 0.5 with p90 = 1.0, so the GT-free gate admits 14–30 predictions per
  scene that are dominated by false positives.
* **The masks themselves are not instance-shaped**: even the GT-assisted
  "best over all 100 groups, ignoring every threshold" reaches only IoU 0.25–0.32
  with recall@50 0.013–0.151. So the loss is not merely gated badly — the group
  masks are wrong.
* **It is not (primarily) a generalisation gap**: on training windows the GT-free
  instance result is no better (recall 0.000 / 0.032) and the GT-assisted
  ceiling on unseen scenes is low; only the pixel semantics fits the training
  windows better (mIoU 0.228 vs 0.173).
* **Diagnostics recorded**: GS instance purity 0.438 (g0) / 0.453 (g1) over all
  GS and 0.422 / 0.444 over the subset that actually renders; token locality
  p50/scale 0.090 / 0.086 (all) and 0.106 / 0.101 (contributing).

## 6. Answers

1. **Can G0/G1 reconstruct and produce GT-free instance predictions from
   scratch on unseen scenes?** Yes for reconstruction (both ≈ +8.6 dB over grey
   with healthy coverage); yes for *emitting* predictions under the frozen rule,
   but the instance result is effectively empty: G0 0 TP / recall 0.000,
   G1 1 TP of 55 GT / recall 0.013, AP50 0.0025, mIoU 0.166–0.173.
2. **Does the group→token feedback help?** No. Instance and semantic differences
   are within noise (ΔAP50 +0.0025 from a single scene, ΔmIoU −0.007) while
   reconstruction degrades (novel PSNR −0.63 dB, worse in 7/8 scenes). The
   single structural variable made the model slightly worse, not better.
3. **Where is the failure?** First in the **mask/group quality**: the
   assignment collapses onto ~10 of 100 groups with an unused background slot
   and the masks are not instance-shaped even without any threshold (best-any-query
   IoU 0.25–0.32); second in **objectness**, which saturates so the GT-free gate
   cannot separate objects from background. Reconstruction/coverage is healthy
   and cross-scene generalisation is not the first-order cause (training windows
   fail under the same rule).
4. **Enough for the full-data / official SIU3R protocol?** No — recall ≈ 0 and
   mIoU ≈ 0.17 on unseen scenes is far from a usable panoptic baseline, so no
   official comparison should be claimed or started from this pair. The single
   most evidence-supported next variable is the **token→slot assignment
   mechanism itself**: replace the current shallow cosine-similarity assignment
   with the repository's audited query decoder (`UnifiedObjectQueryHead`: query
   self-attention blocks over spatially encoded tokens), keeping the data, the
   losses, the masks, G0's read-out structure and the schedule fixed. The
   evidence for this over the alternatives is that the losses do decrease, the
   reconstruction is healthy, and the *only* measured quantity that is clearly
   broken is the effective number of groups / mask quality (entropy 0.57–0.70,
   ~10 active groups, background mass ≈ 0, best-any-query IoU 0.25–0.32).

## 7. Artifacts

> **Addendum (read-only score audit, job 55295).** The GT-free score used in
> sections 3–6 — `sigmoid(raw_noobject_logit)` — is sign-inverted: the 21st class
> logit is the *no-object* logit, which the CE pushes **up** for unmatched queries
> and **down** for matched ones (minimal gradient check: +0.0105 → decrease for
> the matched query, −0.0174 → increase for the unmatched one). Re-scoring the
> same checkpoints with the CE-consistent `P(thing) = Σ_{c<20} softmax(21)[c]`
> and the same 0.5/0.5/50 thresholds raises unseen AP50 to 0.072 (g0@3000),
> 0.045 (g0@6000), 0.083 (g1@3000) and 0.013 (g1@6000) — i.e. the legacy numbers
> above understate the models — but 49–54 of 55 unseen GT instances are still
> missed and the GT-assisted best-over-groups ceiling is only 0.21–0.32 IoU, so
> the mask/group quality rather than the score gate remains the first-order
> limitation. Full tables: `group_locusgs/SCORE_AUDIT.md` and
> `group_locusgs/audit_scores.json`. The numbers in sections 3–6 are the legacy
> convention and are kept unchanged for traceability.

| file | content |
|---|---|
| `group_locusgs/IMPLEMENTATION_MAP.md` | module-by-module mapping of the required functions to the existing code |
| `group_locusgs/plan_6000.json` | the pre-registered paired batch plan (identical to `object_locusgs/plan_6000.json`) |
| `group_locusgs/smoke_report.json` | the eight pre-run checks |
| `group_locusgs/eval_report_step6000.json` | per-scene/per-arm development metrics + diagnostics + training-window control |
| `group_locusgs/init_report_g0.json`, `init_report_g1.json` | initialisation provenance, block hashes and counts |
| `group_locusgs/manifest_g0.json`, `manifest_g1.json` | config, plan/split hashes, optimizer groups, windows |
| `group_locusgs/val_curves_g0.jsonl`, `val_curves_g1.jsonl` | every-500-step curves, per scene |
| `group_locusgs/train_log_g0.txt`, `train_log_g1.txt` | step log (loss terms, weights, grad/update norms, gates) |
| `group_locusgs/tile_*_rgb.png`, `tile_*_masks.png` | GT \| prediction \| error and GT/predicted semantic + instance tiles |
| `workspace_group_locusgs/arm_{g0,g1}/ckpt_step{0,3000,6000}` | resumable checkpoints (not in Git) |
