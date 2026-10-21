# Same-scene multi-window control for one shared InstanceQueryHead

**Question.** On top of the fixed4 experiment, if ONE shared
`InstanceQueryHead` sees **two** windows of each scene instead of one, does it
transfer to a **third, never-sampled** window of the same scene?

This is a localisation experiment under the 32/8 **development** split.  Every
AP50 / IoU below is class-agnostic and development-only; **none of it is SIU3R
official mAP/PQ**.  All runs use GT camera poses (SIU3R is unposed).

---

## 1. Headline result

The intervention could **not** be isolated, because the fixed4 harness turned
out to be **non-reproducible**: the same configuration (`W0` only, 4 windows,
1600 steps, seed 42, identical head init) produced

| run of the *same* 1-window configuration | deterministic | W0 novel detected/24 | W0 novel AP50 |
|---|---|---|---|
| fixed4 (committed `4717107`) | no | **19** | **0.812** |
| `control_w0only` | no | 3 | 0.125 |
| `det_control_w0` | yes | 5 | 0.156 |
| `det_control_w0_repro` | yes | 5 | 0.156 |

Three of four samples of the recipe that the fixed4 document reported as
"**one head CAN fit 4 windows, AP50 0.812, still improving**" land at AP50
0.125–0.156.  **That claim must be withdrawn**: it was one favourable draw from
a chaotic process, not a reproducible property of the head.

Root cause (measured, not inferred): the per-token contribution maps in
`scripts/train_instance_query_overfit.py::token_maps` accumulate with
`Tensor.index_add_` / `+=`, which on CUDA use **non-deterministic atomics**.
`scripts/probe_multiwin_repro.py` calls `token_maps` twice on the *same* window
in the *same* process:

```
view 0: maps hash 76536530f9207489 vs 37816752960ce14e | max|dmap| 2.980e-07
view 1: maps hash 5a4ee8eae3c8306c vs 9c630df40dc77512 | max|dmap| 4.172e-07
...  (tokens and alpha are bit-identical)
```

With `torch.use_deterministic_algorithms(True)` (+`CUBLAS_WORKSPACE_CONFIG`) the
maps become bit-identical, both in-process and across processes
(`max|dmap| = 0.0`, identical hashes and identical head-init hash
`bf421f780a3c135d`).  Two full deterministic runs of the 1-window recipe then
produce **bit-identical** evaluation curves.

So a ~3e-7 perturbation in the contribution maps is amplified — through the
discrete Hungarian matching and a weakly-supervised objectness term — into a
±0.7 AP50 swing.  **Nothing about window count, representation or loss can be
attributed while that is in play.**

---

## 2. Manifest (written before training)

`workspace_recon_diag/instance_query/multiwin/manifest.json`.  `W0` is copied
verbatim from the fixed4 manifest; `W1` (second training window) and `H`
(held-out, never sampled) are drawn with the provider's own sampler from
recorded seeds, rejecting any draw whose four frames overlap an earlier window.
All 12 windows are frame-disjoint (overlap 0):

| scene | W0 ctx / novel | W1 ctx / novel (seed) | H ctx / novel (seed) |
|---|---|---|---|
| scene0012_02 | [2043,2075] / [2045,2055] | [3358,3444] / [3390,3418] (4000) | [1909,1942] / [1935,1939] (5000) |
| scene0010_01 | [510,531] / [512,522] | [850,881] / [855,864] (4001) | [1167,1186] / [1171,1181] (5001) |
| scene0000_00 | [3673,3698] / [3682,3689] | [5266,5330] / [5296,5323] (4002) | [5225,5243] / [5226,5227] (5002) |
| scene0005_00 | [510,529] / [512,522] | [494,511] / [495,499] (4003) | [326,361] / [345,351] (5003) |

Caveat: the sampled `H` for scene0000_00 has its novel frames (5226, 5227)
adjacent to a context frame — a low-baseline window by construction (it
nevertheless scored 0.000, see §4).

The single intervention is the sampling range: `W0` (4 windows) → `W0`+`W1`
(8 windows), one shared head, fresh init, seed 42, frozen fp32 LocusGS
`cross_scene/lgs_lr1e4/ckpt_step6000`.  Head structure, `token_maps`, the
Hungarian cost, BCE/Dice/objectness weights, AdamW (lr 3e-4, wd 0, clip 1.0)
and the inference thresholds (objectness 0.5, mask 0.5, area 50 px) are the
shared ones, unchanged.  H's GT is evaluation-only and never enters the sampler
(asserted per batch).

Smoke (passed): every batch's frame IDs asserted against the manifest, 8
training windows load, `H` is not in the sampler, α identity
`Σ_q mask_q + mask_bg = α` max error **4.2e-7**, frozen LocusGS gradient count
**0**, frozen max |Δ| after a step **0.0**, head params with grad **18/20**,
head+optimizer checkpoint save/restore OK.

---

## 3. Deterministic A/B (the actual comparison)

Window updates are the point of comparison, not total steps: at ~400 updates
per window the fixed4 recipe is directly comparable to the 2-window arm.

| | 1 window/scene (`det_control_w0`) | 2 windows/scene (`det_multiwin`) |
|---|---|---|
| steps | 1600 | 3200 |
| updates per window | 394–407 | 388–419 |
| head init hash | `bf421f780a3c135d` | `bf421f780a3c135d` |
| frozen LocusGS max |Δ| | 0.0 | 0.0 |

Novel views of the 4 `W0` windows (24 visible-instance records), IoU ≥ 0.5 = TP,
AP50 = unweighted mean of per-view greedy score-ordered AP50:

| step | 1-win W0 det/24 · AP50 | 2-win W0 det/24 · AP50 |
|---|---|---|
| 1 | 0 · 0.000 | 0 · 0.000 |
| 200 / 400 | 0 · 0.000 | 0 · 0.062 |
| 600 / 800 | 2 · 0.062 | 4 · 0.188 |
| 1000 / 1200 | 2 · 0.062 | 0 · 0.000 |
| 1400 / 1600 | 7 · **0.266** | 0 · 0.000 |
| 2000 / 2400 | – | 0 · 0.000 |
| 2800 / 3200 | – | 2 · 0.062 |
| **final** | 5 · **0.156** | 2 · **0.062** |

Four groups at the final step of the deterministic 2-window run
(`det_multiwin`, step 3200):

| group | windows | novel inst. records | detected | AP50 | TP/FP/FN | gate obj≥.5 → obj&mask → pred | mean best-any-query IoU (GT-aided) |
|---|---|---|---|---|---|---|---|
| **W0** (trained) | 4 | 24 | 2 | 0.062 | 2/16/22 | 72 → 18 → 18 | 0.335 |
| **W1** (trained) | 4 | 22 | 4 | 0.135 | 4/8/18 | 30 → 13 → 12 | 0.321 |
| **H** (held out, never sampled) | 4 | 27 | 2 | 0.104 | 2/12/25 | 20 → 15 → 14 | 0.299 |
| **val8** (8 unseen scenes) | 8 | 53 | 2 | 0.062 | 2/25/53 | – | – |

Per-window detail at step 3200 (novel views only):

| group | scene | ctx / novel | inst | det | AP50 | TP/FP/FN | best-any-query |
|---|---|---|---|---|---|---|---|
| W0 | scene0012_02 | [2043,2075] / [2045,2055] | 4 | 0 | 0.000 | 0/8/4 | 0.268 |
| W0 | scene0010_01 | [510,531] / [512,522] | 8 | 0 | 0.000 | 0/2/8 | 0.363 |
| W0 | scene0000_00 | [3673,3698] / [3682,3689] | 8 | 2 | 0.250 | 2/6/6 | 0.211 |
| W0 | scene0005_00 | [510,529] / [512,522] | 4 | 0 | 0.000 | 0/0/4 | 0.595 |
| W1 | scene0012_02 | [3358,3444] / [3390,3418] | 3 | 0 | 0.000 | 0/6/3 | 0.061 |
| W1 | scene0010_01 | [850,881] / [855,864] | 4 | 0 | 0.000 | 0/2/4 | 0.314 |
| W1 | scene0000_00 | [5266,5330] / [5296,5323] | 7 | 2 | 0.292 | 2/0/5 | 0.431 |
| W1 | scene0005_00 | [494,511] / [495,499] | 8 | 2 | 0.250 | 2/0/6 | 0.326 |
| H | scene0012_02 | [1909,1942] / [1935,1939] | 10 | 0 | 0.000 | 0/6/10 | 0.242 |
| H | scene0010_01 | [1167,1186] / [1171,1181] | 5 | 2 | 0.417 | 2/0/3 | 0.316 |
| H | scene0000_00 | [5225,5243] / [5226,5227] | 8 | 0 | 0.000 | 0/2/8 | 0.278 |
| H | scene0005_00 | [326,361] / [345,351] | 4 | 0 | 0.000 | 0/4/4 | 0.464 |

No window is learned: the best single window reaches 2 detected instances.
Training loss (BCE/Dice/objectness, every 50 steps) stays in the 1.3–2.5 range
and does not fall the way the fixed4 log did (0.69–0.98 at step 1600).

**Note on determinism and the non-deterministic runs.** Before the fix I also
ran the 2-window arm twice (`run`, `run2`); both ended at AP50 0.000 on all four
groups and, in the second half, their objectness gate collapsed so hard that
*zero* queries passed `objectness ≥ 0.5`.  The deterministic run collapses
differently (objectness over-fires: 72/100 on W0) but the mask quality is poor.
Either way the *final* state of the shared head is not a working segmenter.

---

## 4. Attributing the failure (GT-aided, diagnosis only — not a GT-free metric)

`attribution` counts each *missed* visible instance by the reason it was missed
(`best_any_query_iou` comes from all 100 queries, GT-aided, and is reported
separately from the GT-free numbers above):

| group | no good mask | good mask blocked by objectness | good mask blocked by area | detected |
|---|---|---|---|---|
| W0 | 17 | 5 | 0 | 2 |
| W1 | 15 | 3 | 0 | 4 |
| H | 22 | 3 | 0 | 2 |

The **dominant** failure is *no usable mask anywhere among the 100 queries*
(54 of 67 misses); a minority (11) has a usable mask that the objectness gate
discards, and **none** is discarded by the area gate.  This is the opposite of
the fixed4 story, where `best-any-query` IoU on the trained windows was 0.63.

Concrete example (W0, scene0012_02, frame 2045, instance 8027, gt_area 7573):
`best_any_query_iou` 0.719 with objectness 0.477 — a good mask rejected by the
objectness gate by 0.023.  In the same run, mis-ranked queries with huge masks
(area 10983 px on a 256² image) produce 16 false positives.

---

## 5. Answers

1. **Can one head fit W0 *and* W1 simultaneously?**  **No.**  With ~400 updates
   per window, the deterministic 2-window run reaches W0 2/24 (AP50 0.062) and
   W1 4/22 (AP50 0.135); the 1-window control on the same budget reaches W0
   5/24 (AP50 0.156).  Neither is learned, and the difference between the arms
   is far inside the run-to-run spread measured in §1.  **The 1-window arm is
   not a working baseline**, so "adding a second window" cannot be evaluated as
   an increment.
2. **Does H improve over fixed4's "same-scene extra windows"?**  **Not
   demonstrable.**  H ends at AP50 0.104 (2/27); fixed4's extra windows were
   0.000–0.108.  But (a) H uses *different* frame IDs than fixed4's extra
   windows, so this is not a paired comparison, and (b) fixed4's own reference
   curve is not reproducible (§1).  The comparison carries no information.
   The failure mode is *no good mask* (22/27) rather than *good mask, wrong
   objectness* (3/27).
3. **If W0/W1 succeed but H fails → widen coverage?**  Not the situation here:
   W0/W1 do not succeed either.  Since the same failure appears on trained and
   held-out windows alike, the evidence does **not** support "2 windows/scene is
   insufficient"; it also does **not** support blaming the LocusGS
   representation or the loss — the harness itself is not yet a controlled
   instrument.
4. **If H improves → scale to 32 scenes?**  No improvement was observed, so no
   scaling recommendation.

### Recommended next step (one factor, not executed here)

Make the training path **reproducible first**, then re-establish the baseline
distribution before touching windows or the representation:

1. Adopt `--deterministic` (already implemented; `torch.use_deterministic_algorithms(True)`
   + `CUBLAS_WORKSPACE_CONFIG`) as mandatory for every A/B in this project.
2. Re-run the *unchanged* 1-window (fixed4) arm **≥3 times** and report the mean
   and spread of AP50/updates — only then is a "the head learned the windows"
   claim meaningful.  This is not multi-seed hyper-parameter search; it is
   characterising a stochastic trainer whose variance is currently larger than
   any effect being measured.
3. Only after (2) should the question "wider per-scene window coverage vs joint
   LocusGS representation training" be decided, and then one factor at a time.

No head/LocusGS structure, loss, threshold or split was changed in this round.

---

## 6. Artefacts and commands

```
# intervention: one head, 8 training windows (W0+W1), H never sampled
srun -p 3090 -w 3dimage-13 -N1 -n1 --gpus-per-task=1 -c 16 --time=12:00:00 bash -lc '
  export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH; cd /space/mawb/ssst;
  python -u scripts/train_instance_query_multiwin.py \
    --split workspace_recon_diag/cross_scene/split.json \
    --checkpoint workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step6000 \
    --fixed4-manifest workspace_recon_diag/instance_query/fixed4/manifest.json \
    --manifest workspace_recon_diag/instance_query/multiwin/manifest.json \
    --out workspace_recon_diag/instance_query/multiwin/det_multiwin \
    --steps 3200 --eval-every 400 --deterministic'

# harness control: identical recipe with ONE window per scene
... --out .../multiwin/det_control_w0 --steps 1600 --eval-every 200 \
    --train-windows W0 --val8-mode every --deterministic

# non-reproducibility probe (read-only)
... python -u scripts/probe_multiwin_repro.py
```

* `scripts/train_instance_query_multiwin.py` — manifest, sampler, training,
  4-group evaluation, figures, deterministic mode.
* `scripts/probe_multiwin_repro.py` — bit-reproducibility probe for
  `token_maps`.
* `workspace_recon_diag/instance_query/multiwin/manifest.json`, `summary.json`.
* `.../multiwin/run_nonDet.log`, `run2_nonDet.log` — the two non-deterministic
  2-window runs (checkpoints deleted, logs kept as the evidence for the
  objectness collapse described in §3).
* `.../multiwin/control_w0only/history.json` — the non-deterministic 1-window
  control (checkpoints and figures deleted).
* `.../multiwin/det_multiwin/{history.json, ckpt_step1600.pt, ckpt_step3200.pt, figures/}`
* `.../multiwin/det_control_w0/{history.json, ckpt_step1600.pt, figures/}`
* `.../multiwin/det_control_w0_repro/history.json` (bit-identical repeat).
* Figures are `RGB | GT instance | GT-free prediction | error` for the two novel
  views of a chosen window, labelled with scene / tag / frame / kind / step.

`scripts/train_instance_query_fixed4.py` was refactored *without behaviour
change* so this script can import the same code paths: `build_window` gained an
opt-in `with_rgb=` argument, `eval_window` additionally records the gate counts
`n_obj_q`/`n_obj_mask_q`, and the Hungarian + loss block was extracted into
`hungarian_match()` / `query_losses()` with identical arithmetic.
