# Structure probe v1 — can the current representation fit semantics and instances?

**One-page answer**

* **Reconstruction being learnable says nothing about segmentation.**  The
  G0+/recipe_v1 checkpoints render well-covered images (alpha ≈ 1 almost
  everywhere: on the 8 development windows **0 %** of GT thing pixels fall at
  alpha ≤ 0.05) and their semantic head has a usable loss, yet almost none of
  their *instance* masks are object-shaped.  The two abilities are produced by
  different heads and are not coupled by the shared trunk.
* **Where the loss is, measured:** with the frozen reader conditions (raw mass
  > 0.5, area ≥ 50) the dominant failure is **mask formation**, not score, class
  or assembly.  Over 110 development GT instances, G0+ lands
  **104 in "no mask reaches IoU 0.5"**, **0** in "mask reaches 0.5 but the class
  is wrong", **0** in "mask+class fine but the reader dropped it" and 6 in
  "reader true positive" (recipe_v1: 95 / 0 / 0 / 15).  The same shape holds on
  the 7 training windows and on the 32-pair official val subset.
* **Single-sample fitting:** **no** — `capacity_pass = false`.  Probe-S optimised
  *only* the semantic linear head for 800 steps (loss 0.727 → 0.226) and still
  only reached context IoU 0.290 / 0.365 for the two sentinel classes (gate:
  ≥ 0.50), although both passed the novel threshold (0.373 / 0.520).  Probe-I
  optimised *only* `groups.*` for 1200 steps (loss 4.444 → 2.537) and ended with
  one sentinel at raw mask IoU 0.421 (needs 0.50) and the other at 0.535 but
  with the wrong query class and `P(thing) ≈ 1.9e-15`, so the formal reader
  found **0** of the GT things in context and novel.
* **Full run: not started, and it could not have been.**  The probe gate failed
  first (that is the fail-fast that stopped the chain), and independently the
  filesystem had **0.96 GB** free at the start of the round while one full-run
  checkpoint is ≈ 3 GB (1.13 GB model + ≈ 2 GB optimizer state); a minimum
  policy of 4 retained checkpoints needs ≈ 12 GB.  See §5.

**Provenance.**  HEAD at the start of this round =
`fb48c459da1119240cabc85d3a8792da8a277d3e` (the stated baseline), worktree
clean, `git status --porcelain` empty, no `AGENTS.md` anywhere under
`/space/mawb` or `/space`.  Both source checkpoints are unchanged
(`checkpoints_unchanged = true` in every job).

---

## 1. Read-only attribution (`attribution.json`)

Two checkpoints with real official results are compared on three fixed window
groups.  They are **never averaged together** and the 32-pair subset is never
presented as the official 1860-pair result.

| group | windows | views | GT thing instances |
|---|---|---|---|
| `train7` | 7 (the `build_train_entries(..., 8)` set) | 2 context + 2 novel | 63 |
| `dev8` | 8 (the original 32/8 development windows) | 2 context + 2 novel | 110 |
| `val32` | 32 (first pair of the first 32 distinct official val scenes, sorted by scene+context; listing SHA `801c2671…`) | 2 context + 6 target | 1351 |

### A. Semantic head

Mean per-window mIoU over the classes present under convention A
(GT valid **and** alpha > 0.05):

| arm | train7 | dev8 | val32 (32-pair estimate) |
|---|---|---|---|
| G0+ | 0.1997 | 0.1716 | 0.0933 |
| recipe_v1 | 0.2181 | 0.1743 | 0.1485 |

These are *estimates on a subset*, not the official number.  The authoritative
1860-pair values are quoted from
`group_plus/implementation_audit_v1/B1_full/*_official_semantic.json`:
semantic-only target mIoU **0.0409 (G0+) / 0.0353 (recipe_v1)**,
panoptic-product target mIoU **0.0483 / 0.0320**.

Convention B (the official export convention, where alpha ≤ 0.05 is written as
void) is also stored per window in `attribution.json`; the difference is small
for the development windows and larger where low-alpha pixels exist.

GT thing pixels falling on low alpha, split context / novel:

| arm | group | context ≤0.05 / ≤0.5 | novel ≤0.05 / ≤0.5 |
|---|---|---|---|
| G0+ | train7 | 4.27 % / 9.21 % | 4.16 % / 9.62 % |
| G0+ | dev8 | **0 % / 0 %** | **0 % / 0 %** |
| G0+ | val32 | 0 % / 0.26 % | 0 % / 0.16 % |
| v1 | train7 | 4.72 % / 8.36 % | 4.93 % / 8.81 % |
| v1 | dev8 | **0 % / 0 %** | **0 % / 0 %** |
| v1 | val32 | 0.01 % / 0.07 % | 0 % / 0 % |

**Reading:** the renderer covers essentially all annotated thing pixels on the
evaluation windows, so "the object is not rendered" cannot explain the missing
instances.  Only the 7 training windows show a few percent of uncovered thing
pixels.

### B. Instance buckets (frozen reader; GT-assisted best query is a diagnostic)

| arm | group | no mask ≥0.5 | mask ok, class wrong | mask+class ok, reader dropped | reader TP |
|---|---|---|---|---|---|
| G0+ | train7 (63) | 58 | 0 | 0 | 5 |
| G0+ | dev8 (110) | **104** | 0 | 0 | 6 |
| G0+ | val32 (1351) | **1314** | 0 | 0 | 37 |
| v1 | train7 (63) | 57 | 0 | 0 | 6 |
| v1 | dev8 (110) | **95** | 0 | 0 | 15 |
| v1 | val32 (1351) | **1296** | 0 | 0 | 55 |

Per-instance rows (best raw query, its IoU, its class, `P(thing)`, whether it
passes the reader, the reader's own best IoU and the bucket) are in
`attribution_instances.csv` (3048 rows).  **Measured:** the binding constraint is
the shape of the raw masks; the score threshold, the class head and the reader's
extra gates are not where instances are lost at this level.

### C. Assembly (independent semantic map vs panoptic product)

Mean per-window mIoU, semantic-only vs panoptic:

| arm | train7 | dev8 | val32 |
|---|---|---|---|
| G0+ | 0.1974 → 0.1909 | 0.1716 → 0.1431 | 0.0933 → 0.0983 |
| v1 | 0.2149 → 0.1108 | 0.1743 → 0.0856 | 0.1485 → 0.0689 |

The panoptic assembly roughly halves recipe_v1's semantic IoU (its thing masks
and the void fill overwrite each other) while G0+ is nearly unchanged.

**Void recount (the reported counter was wrong).**  `group_official_export.py`
computed `void_pixels` as `~covered | ~(unassigned | (chosen>=0))`, which reduces
to `~covered` and silently drops the second category.  Recounting from the
actual product:

| arm | group | true void px | of which alpha ≤0.05 | of which alpha >0.05, unassigned, head says thing |
|---|---|---|---|---|
| G0+ | train7 | 147 290 | 63 584 | 83 706 |
| G0+ | dev8 | 119 008 | 0 | 119 008 |
| G0+ | val32 | 764 592 | 17 367 | 747 225 |
| v1 | train7 | 81 772 | 69 111 | 12 661 |
| v1 | dev8 | 107 755 | 0 | 107 755 |
| v1 | val32 | 744 664 | 8 451 | 736 213 |

**Report-only fix, proven inert** (`void_counter_fix.json`): the same two pairs
were exported before and after the one-line change — **112/112 PNG byte-identical,
every `pred.json` byte-identical, counter 0 → 1046 (G0+) / 1301 (recipe_v1), and
the pinned SIU3R evaluator returns identical metric dicts for both arms**.  The
prediction arrays were never touched.

## 2. Tiny-sample capacity probes (`probe_S.json`, `probe_I.json`)

**Sample** (`sample.json`, selected from `object_locusgs/plan_6000.json` in order,
first qualifying entry, never relaxed): entry index **2**, scene
`scene0009_02`, frames `[209, 253, 215, 247]`, sentinels
`key 18032 (class 17, 4261 context px, coverage 1.00)` and
`key 20030 (class 19, 2181 context px, coverage 1.00)`, selection SHA recorded;
eliminations before it: 2 entries skipped because their scene is not in the
official train tree, 0 failures of the sentinel conditions.

Both probes loaded their own copy of the frozen G0+ step6000 in `eval()` mode
with gradients enabled only on the target head, ran a 1-step smoke first
(finite loss, finite gradients, exact optimizer membership, parameters moved,
checkpoint SHA unchanged), and saved only the head delta.

**Probe-S** (only `attributes.semantic.*`; AdamW lr 1e-3, wd 0, clip 1, ≤800):

| step | semantic loss | ctx IoU cls17 / cls19 | novel IoU cls17 / cls19 |
|---|---|---|---|
| 0 | 0.727 | 0.10 / 0.13 | 0.06 / 0.16 |
| 100 | 0.276 | 0.188 / 0.324 | 0.224 / 0.492 |
| 400 | 0.237 | 0.257 / 0.347 | 0.305 / 0.510 |
| 800 | **0.226** | **0.290 / 0.365** | **0.373 / 0.520** |

Loss dropped 3.2×, novel gate passed, **context gate failed** → blocker
"context IoU below 0.50".  The frozen RGB/alpha/Gaussians were bit-identical
(max diff 0.0) and every non-target parameter hash was unchanged.

**Probe-I** (only `groups.*`, 22 tensors; loss = `loss_inst_total`; AdamW lr 1e-4,
wd 0, clip 1, ≤1200):

| step | loss_inst_total | reader ctx TP/FP/FN | reader novel TP/FP/FN |
|---|---|---|---|
| 0 | 4.444 | 0/6/7 | 0/6/7 |
| 600 | 2.541 | 0/6/7 | 0/6/7 |
| 1200 | **2.537** | **0/6/7** | **0/6/7** |

Final sentinels: `18032` → best raw IoU **0.421**, class 17 correct,
`P(thing) = 1.0`; `20030` → best raw IoU **0.535** but on query 22 whose class is
**7** (the GT is class 19) and whose `P(thing) = 1.9e-15`.  The two sentinels do
occupy different queries.  The frozen RGB/alpha/Gaussians and **all**
`attributes.*` parameters were bit-identical (max diff 0.0).  Blocker: "raw mask
IoU below 0.50 for a sentinel".

## 3. Gate and the full-run decision

`structure_gate.json → capacity_pass = false`.  Per the round's rule the
full-data run is therefore **not started**; the fail-fast that stopped the chain
is the probe gate (semantic context IoU, then raw-mask IoU), and the disk
measurement in §5 independently rules the run out.

Next single-variable suggestion, following the measured order
(mask shape → raw-mask/query behaviour → reader):

1. **Mask-shape supervision on the group head** is the best-supported single
   change: 95–104 of 110 development instances never reach IoU 0.5 even though
   the best raw query exists, and Probe-I can lower the training loss while the
   masks stay non-object-shaped.  A falsifiable test: with everything else
   fixed, add an instance-mask term that is evaluated on the *raw* query masks
   (not the per-token slot composition) and require the 110-instance
   `no_mask_reaches_0.5` bucket to drop by ≥30 % at step 6000.
2. If that fails, the next lever is the query *expressiveness* for one object
   (the sentinel `20030` case shows a query whose mask is nearly right but whose
   objectness is ~0), i.e. the score/class head, not the reader.

## 4. Not done / not claimed

* No official mAP/PQ claim is made from these numbers; the 32 pairs are an
  estimate and the development 55/110 are a different protocol.
* `implementation_audit_v1/gate.json` remains **BLOCKED**; nothing here changes
  it, and the five-gate policy is untouched.
* The full 50 000-step group run was **not** started (see §5).
* Frozen-run reconstruction still uses GT camera poses; SIU3R is unposed, so no
  same-input comparison is made.

## 5. Disk measurement for the (blocked) full run

Measured at the start of this round: **982 MB – 1.1 GB free** on `/space`.
One G0+ full-run checkpoint is `model.pt` 1.13 GB + optimizer/RNG state ≈ 2.0 GB
≈ **3.1 GB**; the pre-agreed retention (latest two rolling + step 25000 +
step 50000) needs ≈ **12 GB**, and even a single final checkpoint needs 3.1 GB.
The run was therefore not submitted, and no existing checkpoint was deleted.

## 6. Deliverables

`group_plus/structure_probe_v1/`: `report.md`, `attribution.json`,
`attribution_instances.csv`, `attribution_smoke.json`, `sample.json`,
`probe_S.json`, `probe_I.json`, `probe_S_smoke.json`, `probe_I_smoke.json`,
`probe_runs.json`, `structure_gate.json`, `smoke.json`, `void_counter_fix.json`,
`provenance.json`, `logs/`, and the submit scripts.  Head deltas
(`probe_S_delta.pt` 5 MB, `probe_I_delta.pt` 38 MB) are workspace artifacts and
are not committed.
