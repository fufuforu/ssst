# Read-only audit: the no-object logit and the inference score (G0/G1)

Scope: **read-only**. No training, no checkpoint writes, no threshold changes
(`0.5 / 0.5 / 50` untouched), no decoder swap. The legacy read-out results in
`eval_report_step6000.json` and `results_g0g1.md` are kept unchanged for
traceability; everything below is labelled as the *corrected diagnostic*.

Artifacts: `scripts/audit_group_scores.py`, `group_locusgs/submit_audit.sh`,
`group_locusgs/audit_scores.json` (job `55295`).

## 1. What the code says (read-only)

* `group_loss_terms()` builds the 21-way class tensor as
  `cat([group_class_logits (20), objectness_logit], dim=-1)`
  (`tokengs/models/group_locusgs.py`), and `NO_OBJECT_CLASS = 20`
  (`ssst_contracts.py`). **The 21st logit *is* the `objectness` head's raw
  output** — there is no separate objectness score.
* `group_instance_loss()` sets `target = 20` (no-object) for every query and then
  overwrites it with the GT class **only for the Hungarian-matched rows**; the CE
  weight of class 20 is `NO_OBJECT_CE_WEIGHT = 0.1`.
  With `dCE/dz_20 = p_20 − 1{y = 20}` this means:
  * matched query (`y = GT class`): gradient **positive** →
    gradient descent **decreases** `z_20`;
  * unmatched query (`y = 20`): gradient **negative** →
    gradient descent **increases** `z_20`.
* `group_view_predictions()` scores a query with
  `sigmoid(forward["group"]["objectness"])` and keeps it when the score is
  `>= 0.5` — i.e. it keeps **high** `z_20`, which is exactly the direction
  training gives to **unmatched / no-object** queries.

**Conclusion:** `sigmoid(z_20)` is `P(no-object)`, not objectness, so the legacy
GT-free gate is sign-inverted. The CE-consistent object score is
`P(thing) = Σ_{c<20} softmax(21 logits)[c] = 1 − softmax(21 logits)[20]`
(the same 21-way softmax the CE and the Hungarian class cost use).

## 2. Minimal gradient check (one matched + one unmatched query)

`minimal_gradient_check()` — 100 queries, one GT instance (class 5), the real
`group_instance_loss` and the real Hungarian matcher, CPU, seed 0; plus three
plain SGD steps with `lr = 1.0` on the same objective:

| query | ∂L/∂z₂₀ | training wants | z₂₀ before → after 3 SGD steps |
|---|---|---|---|
| matched (row 94) | **+0.0105** | **decrease** | +0.1673 → +0.1362 |
| unmatched (row 0) | **−0.0174** | **increase** | +0.0599 → +0.1121 |

(unmatched CE target = 20 = no-object.)

## 3. What the trained checkpoints actually contain

Score distributions on the novel views, averaged over the 8 unseen windows (and
the 7 recorded training windows); "matched" = GT-assisted, i.e. the query reaches
IoU ≥ 0.5 against some visible GT instance with `mask > 0.5` and `area ≥ 50`
(no score used in this split):

| run | window | matched queries / view | legacy `sigmoid(z20)` matched vs unmatched | corrected `P(thing)` matched vs unmatched | raw `z20` matched vs unmatched |
|---|---|---|---|---|---|
| g0 step3000 | unseen | 0.25 | 0.594 vs 0.943 | 0.819 vs 0.146 | +0.41 vs **+7.02** |
| g0 step3000 | training | 0.00 | — vs 0.938 | — vs 0.139 | — vs +7.14 |
| g0 step6000 | unseen | 0.38 | 0.210 vs 0.893 | 0.806 vs 0.121 | **−1.37** vs **+8.67** |
| g0 step6000 | training | 0.00 | — vs 0.879 | — vs 0.129 | — vs +7.99 |
| g1 step3000 | unseen | 0.25 | 0.502 vs 0.938 | 0.828 vs 0.138 | +0.01 vs +8.00 |
| g1 step3000 | training | 0.14 | 0.537 vs 0.938 | 0.887 vs 0.133 | +0.15 vs +8.85 |
| g1 step6000 | unseen | 0.06 | 0.635 vs 0.919 | 0.931 vs 0.123 | +0.55 vs +8.59 |
| g1 step6000 | training | 0.14 | 0.560 vs 0.923 | 0.739 vs 0.120 | +0.24 vs +8.54 |

The trained logits reproduce the gradient check exactly: matched queries end up
at `z20 ≈ −1.4 … +0.6` (low) and unmatched queries at `z20 ≈ +7 … +8.9` (high), so
the legacy score ranks the true detectors *below* the no-object queries. The
CE-consistent score reverses the ordering (0.74–0.93 vs 0.12–0.15).

## 4. Corrected re-evaluation (same checkpoints, same windows, same thresholds)

Novel views, 55 GT instances on the unseen windows and 30 on the training
windows; `pass` = queries passing the unchanged `score ≥ 0.5 → mask > 0.5 →
area ≥ 50` chain:

| run | windows | legacy pass / TP / FP / FN / AP50 | corrected pass / TP / FP / FN / AP50 | best-over-groups per GT (no score gate) |
|---|---|---|---|---|
| g0 step3000 | unseen | 71 / 3 / 68 / 52 / 0.0118 | **95 / 4 / 91 / 51 / 0.0724** | 0.258 |
| g0 step6000 | unseen | 14 / 0 / 14 / 55 / 0.0000 | **135 / 6 / 129 / 49 / 0.0451** | **0.316** |
| g1 step3000 | unseen | 32 / 2 / 30 / 53 / 0.0417 | **56 / 4 / 52 / 51 / 0.0833** | 0.212 |
| g1 step6000 | unseen | 31 / 1 / 30 / 54 / 0.0025 | **113 / 1 / 112 / 54 / 0.0125** | 0.250 |
| g0 step3000 | training | 57 / 0 / 57 / 30 / 0.0000 | 82 / 0 / 82 / 30 / 0.0000 | 0.080 |
| g0 step6000 | training | 6 / 0 / 6 / 30 / 0.0000 | 108 / 0 / 108 / 30 / 0.0000 | 0.109 |
| g1 step3000 | training | 27 / 2 / 25 / 28 / 0.0321 | 52 / 2 / 50 / 28 / 0.0161 | 0.066 |
| g1 step6000 | training | 42 / 2 / 40 / 28 / 0.0107 | 94 / 2 / 92 / 28 / 0.0054 | 0.132 |

Per-scene corrected results for the step-6000 checkpoints are in
`audit_scores.json` (per view: `n_pass`, `tp/fp/fn`, `ap50`, score
distributions, ceilings). Examples: `g0_step6000` goes from
`0/0/0/19` (legacy) to `20/2/18/17` with AP50 0.053 on `scene0059_00`, and from
`0/0/0/2` to `19/1/18/1` (AP50 0.125) on `scene0695_00`; `g1_step3000` goes from
`8/0/8/4` to `8/2/6/2` (AP50 0.500) on `scene0072_02`. The ceiling column is
per GT instance, over all 100 groups, with no score gate at all.

## 5. Verdict: score reading or masks?

* **The score reading was a real bug and is not negligible.** Fixing only the
  score definition (nothing else) raises AP50 in every one of the four
  checkpoints (0.000 → 0.045 and 0.012 → 0.072 on unseen g0, 0.042 → 0.083 on
  g1 step3000) and adds true positives (0 → 6 for g0 step6000), because the
  legacy gate was systematically selecting the no-object queries.
* **But the masks are the dominant limitation.** With the corrected score, 49–54
  of 55 unseen GT instances are still missed and precision stays ≈ 1:20
  (129–135 predictions for 6 TP); on the training windows the corrected AP50 is
  0.000–0.016. Crucially, the GT-assisted ceiling that uses **no score gate**
  (best of the 100 group masks per GT instance, `mask > 0.5`, `area ≥ 50`) is
  only **0.21–0.32 IoU on unseen** and 0.07–0.13 on training windows, with
  recall@0.5 ≈ 10–15 % (and the same numbers without the area filter). A perfect
  score gate therefore cannot lift the instance result: the group masks
  themselves are not instance-shaped.
* **Practical consequence for the next step:** the score must be corrected before
  any further comparison (it changes AP50 by 1.5–6×), but the evidence still
  points at the group/mask mechanism — not the score gate — as the first-order
  variable. No decoder swap and no new long run was started in this audit.
