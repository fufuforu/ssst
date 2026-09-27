# structure_probe_v2 — same-sample assignment ceiling, directed single-variable probes, disk cleanup

**One-page answers**

| question | answer |
|---|---|
| Can the **frozen token/Gaussian representation** form the two correct instances on the fixed sample? | **Not demonstrated.** With the per-scene GT-fitted token assignment (500 Adam steps, lr 0.05) the best raw IoU is **0.309 / 0.455** in context and **0.341 / 0.482** in novel — neither sentinel reaches 0.5 anywhere. `oracle_pair_pass = false`. Wording rule respected: this is "not shown reachable under this fixed initialisation and 500-step budget", **not** a proof of impossibility. |
| Is it a *granularity* limit of the shared token assignment? | **Not demonstrated either.** The per-Gaussian oracle (§4.B, same loss/GT, Z over 65536 Gaussians, 300 steps, lr 0.01) also fails: **0.349 / 0.447** context, **0.358 / 0.414** novel. Per the pre-registered rule, the conclusion is **"fixed geometry/rendering reachability not demonstrated"**, and one sample cannot declare the whole data structure invalid. |
| Then why does the feed-forward group head learn nothing? | Two gaps are now measured and separated: (i) **the semantics themselves do not fit locally** — even training only the semantic linear head on this sample leaves the two sentinel *classes* at context IoU 0.291 / 0.366 (and 0.397 / 0.472 with the class-weighted contrast), while their **novel** IoU passes (0.374 / 0.520); (ii) **the instance masks do not fit even with GT** at either granularity. The feed-forward head additionally has no GT: its corrected step-1200 reader output is 0 category-aware TP in context and novel. |
| Was the full 50 000-step run started? | **No.** All three entry conditions are unmet: `semantic_local_pass = false`, the §4.A feed-forward sentinel probe is *not applicable* (its precondition `oracle_pair_pass` is false), and free space is **7.61 GiB < 15 GiB**. See `structure_v2_gate.json`. |

Provenance: HEAD at the start of the round = `f6ecc9df6ba1c8b5bd7f4ee91b074697501f1e9b`
(the interface baseline this instruction was written against), worktree clean, no
`AGENTS.md` under `/space/mawb`.  Every protected artifact re-hashed afterwards
and unchanged.

---

## 1. Disk cleanup (`cleanup_manifest.json`)

Start: **0.96 GiB** free.  Removed **74** audited items, **6.65 GiB** actually
released (measured with `statvfs` deltas, not apparent `du` of hardlinks) →
**7.61 GiB** free.

| group | items | du | why eligible | where the results live |
|---|---|---|---|---|
| B1_full `official_predictions_{panoptic,semantic}` × 2 arms | 4 | 4.02 GiB | PNG export of finished jobs 55676/55711 | `B1_full/*_official_{panoptic,semantic}.json`, `B1_official_summary.json` |
| full-reconstruction official val-pair export | 4 | 2.28 GiB | PNG export of a finished job | `eval_official_valpair/summary.json` (PSNR 24.81 / SSIM 0.7795 …) |
| `workspace/*/eval_fixed16_step*` prediction trees | 64 | 0.39 GiB | PNG exports of finished diagnostics | each tree's `eval_report.json` / `official_evaluator_result.json` |
| `structure_probe_v1/void_fix_after`, `implementation_audit_v1/B1_smoke` | 2 | 5 MB | proof/smoke exports, results recorded | `void_counter_fix.json`, `B1_smoke_ok.json` |

Every deletion records path, `du` bytes, filesystem delta, the ended source task,
where the final result lives and how to rebuild it.

**Shortfall.** After exhausting the candidates that provably belong to the two
allowed classes ("temporary exports of finished tasks", "`*.inprogress`
intermediates" — none of the latter exist), the space is **7.61 GiB**, i.e.
**7.39 GiB short of the 15 GiB target**.  The remaining large items are experiment
**results** and cannot be deleted under the rules; they are listed in
`cleanup_manifest.json → summary.ineligible_candidates_sorted`:

| path | size | why kept |
|---|---|---|
| `workspace_group_plus` | 29.14 GiB | protected v1/v2/pure4/B2 checkpoints + this round's artifacts |
| `workspace_group_locusgs` | 14.71 GiB | G0/G1 arm checkpoints of finished runs (results) |
| `workspace` | 12.24 GiB | `siu3r_*` training workspaces incl. their `checkpoints/` |
| `workspace_object_locusgs` | 9.22 GiB | object-aware run outputs (results) |
| `workspace_recon_diag` | 6.67 GiB | full-reconstruction run tree (published checkpoints + metrics) |

## 2. Corrected read-outs (`corrected_metrics.json`)

Sample asserted first: SHA `98a4d35d…` equals the required value, scene
`scene0009_02`, frames `[209, 253, 215, 247]`, sentinels `18032/class 17` and
`20030/class 19`.  Both head deltas were applied to a strictly loaded G0+
step6000 and their names/shapes/steps verified; the checkpoint is unchanged
(`checkpoint_unchanged = true`).

**The previous round's step-0 discrepancy, traced.**  `probe_S.json` records
step-0 sentinel IoU **0.0 / 0.0** (both scopes) — that is the true unmodified-head
value under the script's own convention.  The sentence in
`structure_probe_v1/report.md` claiming step-0 context IoU 0.10/0.13 and novel
0.06/0.16 was **wrong and is not reproduced by any stored record**; the historical
JSONs are left untouched and this round's report supersedes that line.

Probe-S, corrected convention (prediction *and* GT restricted to
`GT∈0..19 ∧ alpha>0.05`, per-class FP counted even where a view has no GT of that
class; IoU computed once per class from the summed confusion):

| step | context 18032 / 20030 | novel 18032 / 20030 | legacy convention (context) |
|---|---|---|---|
| 0 | **0.000 / 0.000** | 0.000 / 0.000 | 0.000 / 0.000 |
| 800 | **0.291 / 0.366** | **0.374 / 0.520** | 0.290 / 0.365 |

Sentinel TP/FP/FN at step 800 (context, corrected): `18032` 2162 / 1005 / 4264,
`20030` 1082 / 776 / 1099.

Probe-I, corrected, with the frozen reader and score-descending one-to-one
matching (category-aware TP requires the predicted class to match too):

| step | scope | category-agnostic TP | category-aware TP | FP | FN |
|---|---|---|---|---|---|
| 0 | context | 0 | 0 | 4 | 7 |
| 0 | novel | 0 | 0 | 4 | 7 |
| 1200 | context | 0 | 0 | 6 | 7 |
| 1200 | novel | 0 | 0 | 6 | 7 |

Sentinel-level at step 1200 (reader outputs): `18032` best IoU 0.421 (query 65,
class 17 correct, below the IoU gate); `20030` best IoU 0.179 (query 62, class 19
correct).  Neither is hit, in either scope — the scene-level TP count is never
used as a substitute.  Note the v1 report's 0.535 figure for `20030` came from a
**GT-assisted raw-mask** query (query 22) that the reader never emits
(`P(thing) ≈ 1e-15`); under the reader-based definition used here it is 0.179.

## 3. Token-assignment oracle (`oracle_pair.json`) — GT-assisted, per-scene

All legal context thing instances (**K = 4**: keys 8025, 18032, 18035, 20030) plus
a rest column; Z `[1024, 5]` initialised from the phase-0 contribution majority
vote (4.0 / 0.0); Adam lr 0.05, wd 0, 500 steps, seed 42; only Z is optimised.

| step | loss | ctx 18032 / 20030 | novel 18032 / 20030 | conservation |
|---|---|---|---|---|
| 0 | 9.374 | 0.182 / 0.172 | 0.185 / 0.158 | 1.4e-06 |
| 100 | 5.755 | — | — | ≤2e-06 |
| 500 | **5.681** | **0.309 / 0.455** | **0.341 / 0.482** | 1.4e-06 |

All four instances at step 500 (context best IoU): 8025 0.007, 18032 0.309,
18035 0.491, 20030 0.455.  RGB / depth / Gaussians are **bit-identical** after the
fit (`rgb_max_abs_diff = depth = gaussians = 0.0`), Σ(instances)+rest equals alpha
to ≤1.7e-06, and the G0+ checkpoint SHA and mtime are unchanged.

**`oracle_pair_pass = false`.**  The loss drops 9.37 → 5.68 (a 39 % reduction), so
the optimisation is working; the masks simply plateau below the instance gate.
A determinism re-run with the same seed reproduced the step-500 numbers
**bit-identically** (`oracle_pair_rerun.json`), and its saved final A
(`oracle_final_A.pt`, 22 KB) is the documented initialisation for §4.B.

## 4. Branch taken and its own result

Because `oracle_pair_pass = false`, **§4.A was not run** (its precondition is the
oracle passing; the pre-registered rule says not to add more BCE/Dice blindly).
The §4.B per-Gaussian oracle was run instead (`per_gaussian_oracle.json`):
Z over N = 65536 Gaussians initialised from the token oracle's final A replicated
over each token's 64 Gaussians, Adam lr 0.01, wd 0, 300 steps, same loss, same
context GT, same evaluation.

| step | ctx 18032 / 20030 | novel 18032 / 20030 | conservation |
|---|---|---|---|
| 0 | 0.309 / 0.455 | 0.341 / 0.482 | 1.4e-06 |
| 150 | 0.304 / 0.431 | 0.334 / 0.426 | 1.8e-06 |
| 300 | **0.349 / 0.447** | **0.358 / 0.414** | 1.7e-06 |

`per_gaussian_pass = false`.  Since **both** granularities fail, the
pre-registered reading is **"fixed geometry/rendering reachability not
demonstrated"** — the pair cannot be declared impossible, and this sample alone
cannot invalidate the whole data structure.

## 5. Semantic branch (§5)

`semantic_local_pass = false` from the corrected Probe-S read-out, so the single
pre-registered contrast was run: the same `attributes.semantic.*`-only fit with
the per-pixel NLL weighted by `1/sqrt(class frequency)` of the sample's four valid
GT views (weights normalised to mean 1 over the appearing classes, clipped to
[0.25, 4]; class 7 gets 3.03, classes 0/2 hit the 0.25 floor, class 17 gets 0.59).

| step | weighted NLL | ctx 17 / 19 | novel 17 / 19 |
|---|---|---|---|
| 0 | 0.304 | 0.000 / 0.000 | 0.000 / 0.000 |
| 400 | 0.089 | 0.378 / 0.467 | 0.507 / 0.547 |
| 800 | **0.084** | **0.397 / 0.472** | **0.523 / 0.546** |

Still below the 0.50 context gate for both sentinels, though clearly better than
the unweighted head (0.291 / 0.366).  Novel passes for both.  All non-target
parameters are bit-identical (`frozen_params_unchanged = true`).  This is a new
single-variable diagnostic and does not rewrite the previous round's conclusion.

## 6. Gate and what was *not* done

`structure_v2_gate.json → local_structure_verified = false`.  Missing items, in
the instruction's own order:

1. `semantic_local_pass` — false (corrected Probe-S and the weighted contrast both
   above 0.25 novel but below 0.50 context).
2. feed-forward sentinel success — **not applicable**: §4.A is conditional on
   `oracle_pair_pass`, which is false; running it would violate the pre-registered
   branch rule.
3. free disk ≥ 15 GiB — false: **7.61 GiB** (7.39 GiB short).

Consequently the 50 000-step G0+ full run was **not** prepared or submitted, and
`implementation_audit_v1/gate.json` remains **BLOCKED** — the original five gates
are untouched and are not claimed as passed.

Status of the round's own questions, tagged:

* **measured** — token-oracle and per-Gaussian-oracle ceilings (0.309/0.455 and
  0.349/0.447 context); corrected Probe-S (0.291/0.366 context, 0.374/0.520
  novel); weighted-NLL contrast (0.397/0.472 context); reader category-aware TP
  = 0 at step 1200; 6.65 GiB freed.
* **inferred from the measurements** — the semantic classes' *novel* transfer is
  already better than their *context* fit, so the local semantic deficit is in
  fitting the context views rather than in cross-view transfer; and since GT
  assistance does not buy the instance masks at either granularity, the binding
  constraint sits before the query head.
* **not yet verified / not done** — the feed-forward §4.A probe; the full run; any
  official metric for the new experiments (none is claimed).

## 7. Deliverables

`group_plus/structure_probe_v2/`: `report.md`, `cleanup_manifest.json`,
`corrected_metrics.json`, `oracle_pair.json`, `oracle_pair_firstrun.json`,
`oracle_pair_rerun.json`, `oracle_final_A.pt`, `oracle_smoke.json`,
`per_gaussian_oracle.json`, `per_gaussian_oracle_smoke.json`,
`probe_semantic_weighted.json`, `probe_semantic_weighted_smoke.json`,
`structure_v2_gate.json`, `provenance.json`, `smoke.json`, the submit scripts and
`logs/`.  New scripts: `scripts/disk_cleanup_v2.py`,
`scripts/structure_probe_v2_corrected.py`, `scripts/probe_token_assignment_oracle.py`,
`scripts/probe_per_gaussian_oracle.py`, `scripts/probe_semantic_weighted_nll.py`,
`scripts/probe_ff_fullpixel_mask.py` (prepared, not run).
