# Read-only localisation: why mid/large masks stay wrong (G0 vs G0+)

**Read-only.** No training, no optimizer step, no checkpoint write, no threshold
change, nothing deleted; the parallel full-data SIU3R evaluation was untouched.
Everything here reuses the existing group rendering, Hungarian matcher,
valid-pixel/instance definitions and read-out.

Artifacts: `scripts/audit_mask_failure.py`,
`group_plus/mask_failure/{manifest.json, per_instance.csv, summary.json, *.png}`.
`best-over-groups` below always means *the best of the 100 group masks this
checkpoint already produces*, under the fixed `mask > 0.5` / `area >= 50`
thresholds - not a theoretical upper bound.

## 1. Verified inputs and the fixed sample

Four checkpoints, all sharing plan `a2a65c13...` and 479 model keys: `g0_step3000`,
`g0_step6000`, `g0plus_step3000`, `g0plus_step6000` (per-checkpoint arms, steps and
`model.pt` hashes are in `manifest.json`).

The sample was written to `manifest.json` **before** any per-checkpoint analysis,
using only area and best-over-groups IoU from the G0 step6000 reference: the
lowest-IoU mid instances (`1310 <= mean per-view GT area < 6553`), the lowest-IoU
large instances (`>= 6553`) visible in at least one novel view, plus the single
highest-IoU instance as success reference.  Result: 4 mid failures, 4 large
failures, 1 large reference; all 9 come from windows whose
`(scene, context, novel)` tuple appears in the executed 6000-step training logs
of both arms (`proven_trained_window`), i.e. windows the models really trained on.
Predicted masks keep the existing meaning (per-token 101-way slot distribution
broadcast to the 64 Gaussians of the token, composited by the same renderer as
RGB); valid pixels are `semantic != 255`.

## 2. Per-instance findings (novel views, 64 view-instance checks)

| metric | G0 3000 | G0 6000 | G0+ 3000 | G0+ 6000 |
|---|---|---|---|---|
| matched-group hard IoU, mean | 0.134 | 0.167 | 0.057 | 0.174 |
| matched-group precision / recall | 0.38 / 0.19 | 0.44 / 0.27 | 0.25 / 0.17 | 0.39 / 0.26 |
| matched predicted area / GT area | 0.77 | 2.01 | 2.18 | 0.87 |
| best-over-groups IoU mean / max | - | 0.237 / 0.439 | - | 0.191 / 0.633 |
| best-over-groups IoU >= 0.5 | 0/16 | 0/16 | 0/16 | **2/16** |
| best group kept by the P(thing) reader | 15/16 | 16/16 | 14/16 | 16/16 |
| mean background mass inside the GT thing | 0.000 | 0.000 | **0.248** | **0.177** |
| group mass inside the GT thing (of alpha) | 0.93 | 0.93 | - | 0.716 |
| GT-free detections | 0 | 0 | 0 | 2 |

Conservation (`sum over 100 group masks + background = alpha`) stayed <= 1.7e-6 in
every view; alpha coverage inside the GT was 0.93 (G0) and 0.89 (G0+).

Failure classification (novel views, fixed GT-free rule):

| mode | count |
|---|---|
| no good mask among the existing 100 groups (best IoU < 0.5) | 62/64 |
| good mask blocked by the P(thing) / class / area gate | 0 (the best group was kept in 61 of 62 scored cases) |
| good mask assigned to another instance | 0 |
| detected | 2/64 (the G0+ step6000 case, best IoU 0.633) |

Training signal, recomputed read-only on the same fixed context batches for the
matched pair: every selected instance has **226-1521 positive sampled points**
(never 0), sampled BCE 0.10-1.04 and Dice 0.43-1.00 (far from saturated), and the
matched group's mask-output gradient norm is finite and non-zero in all 36
(instance x checkpoint) recomputations.  These pairs therefore have live,
positive-rich, correctly directed mask supervision.

Full-valid-pixel counterfactual matching (no sampling, no backward): the
diagnostic arg-min differs from the fixed-sampling Hungarian match in 24 of 72
(instance, checkpoint, view) pairs.  Since the best of the 100 existing masks
already fails IoU 0.5 in 62/64 cases, **no alternative matching can turn these
instances into detections** - the mask itself is the problem.

## 3. Answers

**A. What dominates mid/large failures?** The mask itself, in the "missing /
wrongly shaped" sense: no group mask reaches IoU 0.5 for 62/64 checks, the
matched group's precision/recall are ~0.4/0.27 with predicted area 0.8-2.2x the GT
area (both under- and over-coverage), the read-out gate is not what blocks them
(it keeps the best group in 61/62 scored cases), and the matching is not the
cause (the full-pixel counterfactual cannot produce a good mask either).  Group
mass inside the GT is high (0.93 of alpha in G0), so it is not "no mass reaches
the instance" - it is mass that is not shaped like the instance.

**B. What did G0+ change?** It moves group mass to the background slot **inside**
thing instances: background mass in the GT thing region 0.000 -> 0.248 (step3000)
/ 0.177 (step6000), and the group mass inside the GT drops 0.93 -> 0.716 of alpha.
Best-over-groups IoU got worse for most sampled instances (0.392 -> 0.152,
0.439 -> 0.137, 0.171 -> 0.051, 0.189 -> 0.010) and better for a few
(0.137 -> 0.633, 0.224 -> 0.356, 0.275 -> 0.323).  Background diversion is thus a
measured companion of G0+'s mask degradation for the majority of these instances
(4 of 7 visible worse, with 13-30 % of alpha moved to background), but it is not
established as the single cause: the same term produces the only new detection.

**C. step3000 -> 6000.** Matched-group identity changed for 8 of 9 instances and
the best-group identity also changed; matched-group IoU improved for some
(0.192 -> 0.392, 0.113 -> 0.439) and worsened for others (0.209 -> 0.113,
0.258 -> 0.224) while every mask stays below IoU 0.5 at both steps.  With masks
bad at both steps and group identity churning, the evidence does not separate
"training not finished" from "wrong supervision target" from "group assignment
degradation": **undetermined**.

**D. If only one training variable is allowed next.** These mid/large instances
have ample positive sampled points, live gradients and full alpha coverage inside
the GT, yet no group mask is instance-shaped - and the G0+ variant loses a
measured 0.18-0.25 of alpha to the background slot inside the same regions.  The
single most supported change is therefore the one already proposed and still
untested: a **per-token background logit** (replace the shared `background_bias`
scalar with `Linear(C->1)` on the same token features, keeping the 101-way
softmax, queries, matcher, losses, optimizer and schedule fixed).  Falsifiable
expectation: background mass inside GT thing regions falls toward the G0 level
(~0) while stuff stays background-dominated, group mass inside thing GT returns
to ~alpha, and best-over-groups IoU for the sampled mid/large instances rises
above the current <=0.44 band for at least half of them; if it does not, the
binding constraint is the assignment/mask capacity itself, which would then
justify a separately controlled query-decoder comparison.  Nothing was
implemented or trained in this round.

## 4. Figures (5 panels: GT RGB | GT instance | Hungarian group mask | best-over-groups mask | GT-free prediction)

| file | case |
|---|---|
| `group_plus/mask_failure/scene0009_00_key18009_mid.png` | mid failure (matched group 35 -> 14, best IoU 0.03 -> 0.02) |
| `group_plus/mask_failure/scene0012_01_key8017_mid.png` | mid instance that G0+ turns into the only detection (0.137 -> 0.633) |
| `group_plus/mask_failure/scene0007_00_key8024_large.png` | large failure, background mass 0.013-0.39 |
| `group_plus/mask_failure/scene0007_00_key8006_large.png` | large failure (predicted area 19.5x GT at G0 6000) |
| `group_plus/mask_failure/scene0001_01_key6002_large.png` | large reference case (best IoU 0.439 at G0 6000, still not detected) |

## 5. Limitations

- 9 instances from 7 proven training windows: enough to characterise the
  mechanism, not a population estimate.
- `best-over-groups` and the full-pixel counterfactual use GT for diagnostic
  matching/scoring only; they never feed a GT-free prediction.
- Gradient norms are reported only as "finite and non-zero"; their magnitude is
  not interpreted as a training-health measure.
- The counterfactual cost uses all valid annotated context pixels as denominator,
  the training cost the 4096 sampled positions (void included); both are recorded
  in the CSV.
- The mid/large split uses the evaluation bucket definition (per-view area); the
  sampling-audit 2-view area is also recorded per instance in the manifest.
