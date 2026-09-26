# Implementation audit — group recipe branch

**Static audit baseline:** `f16d83386fcc1f495f9721083cb73e47fb3abd22`.
On-site state at the start of this round: `git rev-parse HEAD` = that commit,
`git status --short` **clean**, and `git diff <baseline> -- .` **empty** — so the
audited source is exactly the baseline and no uncommitted work was at risk.
No `AGENTS.md` exists anywhere under `/space/mawb` or `/space`.

Evidence is split into **static** (source reading, file:line) and **runtime**
(measured on GPU with the frozen checkpoints).  A static defect is *not* by
itself a reason to discard previously published numbers; the re-evaluation in
§D decides that.

---

## A. Static findings (six requested items)

### A1. `logit_scale` is a non-learnable float `C^-0.5`

* `tokengs/models/instance_query_head.py:46` — `self.logit_scale = float(dim) ** -0.5`
  (used at `:56`), inherited by `GroupQueryHead`
  (`tokengs/models/group_locusgs.py:153`, `GroupQueryHead` subclasses
  `InstanceQueryHead`).
* **Actual behaviour:** with `enc_embed_dim = 1024` this is exactly `0.03125`; it
  is a plain Python float kept in the module `__dict__`, so it is **not** an
  `nn.Parameter`, not in `state_dict`, and cannot be in any optimizer group.
* **Runtime confirmation:** `optimizer_membership.logit_scale_is_parameter =
  false` and `logit_scale_not_a_parameter = true` for every audited checkpoint.
* **Impact:** none on training or reporting; it was correct as designed.  Only
  worth recording because "the similarity scale is fixed" is a real constraint
  on how much a temperature change could ever have helped.

### A2. `forward_full` runs the historical cross-attention prefix **before** the deep decoder

* `tokengs/models/group_locusgs.py:135-154` (baseline): the function computed
  `attended, _ = self.cross_attn(queries, token_features, token_features)`;
  `query_features = self.query_norm(queries + attended)`;
  `query_features = query_features + self.mlp(query_features)`; **then**
  `query_features = self.deep(query_features, token_features)`.
* **Design basis:** the recipe brief said the four-layer group decoder
  *"替换现有单层路径产生的 query 特征"* ("replaces the query features produced by
  the existing single-layer path").  `group_plus/recipe_v1/config_diff.json`
  says the same (`replaces`).
* **Actual behaviour:** the legacy prefix was **not** replaced — it ran first and
  the deep decoder was **stacked on top of its output**.  So recipe_v1/v2
  forward = `deep(old_prefix(queries, tokens), tokens)`.
* **Impact on training:** real.  The published recipe arms trained a strictly
  deeper/wider function class than the brief specified, with the legacy
  cross-attention in the path.  This is a genuine specification bug, not a
  reporting-only defect — **condition A (§E) is therefore required** and is not
  skipped by the "report/protection-only" exemption.
* **Impact on the published numbers:** none for *validity* — the reported
  metrics are what that architecture actually produced — but the arm must be
  described as "recipe with the historical prefix stacked", never as
  "the four-layer decoder replacing the single layer".
* **Fixed in this round:** `Options.group_recipe_head_mode` (`legacy_prefix`
  default / `pure4`), `GroupQueryHead.forward_full(..., head_mode=...)`; the old
  behaviour is preserved exactly under `legacy_prefix`.

### A3. The smoke's `same_plan_sha256` compared the plan to itself

* `scripts/train_group_locusgs.py:206` (baseline):
  `report["checks"]["same_plan_sha256"] = (sha256_file(Path(args.plan)) == sha256_file(Path(args.plan)))`.
* **Actual behaviour:** a tautology — it could never fail.
* **Impact:** the step-0 smoke's plan-provenance check was vacuous; the *real*
  protection was the separate assertion inside the training loop
  (`frames != entry["context"] + entry["novel"]` per step), which did hold.
* **Fixed:** now compares the current plan SHA against the pre-registered
  constant (`a2a65c13…`) **and** the reference run's
  `train_state.meta.plan_sha256`, with the same for the split file
  (`acf9afac…`); the training entry also refuses to start on a mismatch.

### A4. The evaluation read PSNR from the last history line, not from the checkpoint

* `scripts/eval_recipe_v1.py:178-188` (baseline): `history = Path("workspace_group_plus/recipe_v1/run/val_history.jsonl")`
  and `psnr_novel = last["summary"]["novel_psnr"]`.
* **Actual behaviour:** two defects — the path was hard-coded to recipe_v1 (so a
  recipe_v2 evaluation would have quoted recipe_v1's PSNR), and the value was
  not bound to the evaluated checkpoint at all.
* **Impact on published numbers:** none *in effect* (each arm was passed its own
  path via `--history`, and the history's last line did belong to that run's
  step 6000), but the gate value was not independently verifiable.
* **Fixed:** PSNR/SSIM are recomputed from **this checkpoint's current forward**
  on the 8 windows via `object_locusgs_eval.reconstruction_row`; `--history` is
  optional and only records a cross-check (step, scene, frame order).

### A5. Figures compared the union of all predicted masks against one GT instance

* `scripts/eval_recipe_v1.py:270-280` (baseline): `mask |= prediction["mask"]`
  over every surviving GT-free prediction, then the error map was
  `truth & ~union` / `~truth & union`.
* **Actual behaviour:** the "error" panel was dominated by the other objects'
  predictions and could not be read as the quality of a single query's mask.
* **Impact:** visual reporting only; no metric used it.
* **Fixed:** each predicted instance is drawn in its own colour with its query
  index, the GT-assisted best group is drawn/labelled separately, and the union
  error map is explicitly labelled as a union.

### A6. Manifest loss / single-variable fields were stale

* `scripts/train_group_locusgs.py:584-585` (baseline): `"total": "L_recon + lambda(step) * [0.05 L_instance + 0.05 L_semantic]"`
  regardless of the recipe switches; `:719`: `"single_variable": "group -> reconstruction-token feedback between layers 10 and 11"`.
* **Actual behaviour:** every recipe arm's manifest claimed the G0/G1 feedback
  single variable and the plain 0.05/0.05 loss, neither of which was true for
  recipe_v1/v2.
* **Impact:** reporting only — but it is exactly the kind of stale field that
  makes a later reader mis-attribute a result.
* **Fixed:** `effective_loss_formula(opt)` and `single_variable_description(args, opt)`
  generate the strings from the run's effective options.  The recipe_v2 manifest
  is now `L_recon + r1500*0.05*L_inst + r1500*0.2*CE + r2000*0.05*L_sem`
  (recipe_v1 differs only in the `0.1` instance weight).

---

## B. Runtime audit (read-only; `runtime_diagnostics.json`)

Job 55678, GPU 3090, read-only, **0 failures**, `checkpoints_unchanged = true`.
Each checkpoint is loaded with its own effective config (`strict=True`); the
historical arms keep `legacy_prefix`.

**B0 Provenance.** `plan_sha256 = a2a65c13…` and `split_sha256 = acf9afac…`,
both equal to the pre-registered constants.  Checkpoint steps are all `6000`;
`recipe_v1/model.pt = 39c53ce8…`, `recipe_v2/model.pt = 373cac99…` — matching
the required hashes.  Windows: **7 training** (`build_train_entries(..., 8)`
resolves 7 distinct scenes in this plan) and **8 validation**, with full frame
IDs stored in `runtime_diagnostics.json → windows`.

**B1 Assignment distributions** (first training window `scene0000_02`, all 1024
tokens, `[B,T] = [1,1024]`, 64 GS/token, N = 65536 = T·64 ✓, fp32, two context
frames):

| quantity | G0+ | recipe_v1 | recipe_v2 |
|---|---|---|---|
| slot logits p01 / p50 / p99 | −20.73 / −15.33 / +0.007 | −12.09 / −9.75 / 0.0 | −12.29 / −10.21 / 0.0 |
| void probability (mean over tokens) | 0.431 | 0.703 | 0.767 |
| undefined conditional (Σp₁₀₀ ≤ 1e-12) | 0 | 0 | 0 |
| thing-bearing tokens | 134 | 30 | 130 |
| rest-only tokens | 675 | 465 | 574 |
| dropped (contribution ≤1e-6) tokens | 215 | 529 | 320 |

Stratified (mean per stratum; `H101` = entropy of the 101-way softmax, `KL` =
CE − H(target), `agr` = 101-way argmax agreement):

| stratum | G0+ H101 / CE / KL / agr | v1 | v2 |
|---|---|---|---|
| thing-bearing | 0.991 / 1.392 / 1.249 / 0.515 | 0.875 / 0.649 / 0.573 / **0.800** | 0.861 / 0.490 / 0.366 / **0.938** |
| rest-only | 0.842 / 1.320 / 1.320 / 0.504 | 1.092 / 0.471 / 0.471 / 0.955 | 0.838 / 0.304 / 0.304 / 0.981 |
| dropped | 0.942 / 0 / 0 / 0 | 0.951 / 0 / 0 / 0 | 0.733 / 0 / 0 / 0 |

Contribution-weighted (not a plain token mean): weighted `H101` 0.774 / 0.781 /
0.801 and weighted void probability 0.501 / 0.203 / 0.339 for G0+ / v1 / v2 —
i.e. the tokens that actually carry pixel mass are much less void-dominated
than the token mean suggests.  Old-style "effective columns" (per-query
mean prob > 0.01) = 7 / 7 / 5; target row-sum error 1.2e-7.

**B2 Contribution / target reconciliation.** With the same frozen Gaussians and
alpha, the recipe helper's per-thing packed-key numerators match the routing-v2
statistics **exactly** (`per_key_max_abs_diff = 0.0`, atol 1e-4 / rtol 1e-5) for
all three arms; alpha conservation ≤ 1.55e-6; target row-sum ≤ 1.2e-7; semantic
labels are legal (0..19 / 255 only, asserted).  The oracle identity
`annotated_total − recipe_total − missing` (missing = covered, valid thing
pixels with instance id ≤ 0) closes to ≤ 9.8e-4 across arms.  That residual is
**not** a structural gap: the audit's same-forward control gives the *identical*
value (1.22e-4 / 4.88e-4 / 9.77e-4 = 2^-13 / 2^-11 / 2^-10) and the routing
helper's own repeat gap is exactly 0.0, i.e. it is float32 rounding on sums of
order 1e4 (relative error ≈1e-7), which is the expected scale for fp32.

**B3 Hungarian determinism.** Two identical solves give identical rows/cols for
every arm; rows and cols are unique, the GT count ≤ 100, and all GT are covered.

**B4 void accounting.** Mean `A_void` over tokens vs the *rendered* void mass
fraction of α:

| arm | mean A_void | rendered void fraction (v0..v3) | weighted identity error |
|---|---|---|---|
| G0+ | 0.431 | 0.515 / 0.538 / 0.521 / 0.535 | ≤ 3.9e-3 |
| recipe_v1 | 0.703 | 0.182 / 0.211 / 0.182 / 0.203 | ≤ 1.0e-3 |
| recipe_v2 | 0.767 | 0.346 / 0.308 / 0.342 / 0.310 | ≤ 2.0e-3 |

The weighted identity `Σ_t A_void[t]·M[t] = rendered void mass` holds to the
renderer's float noise.  Kept-token contribution share is 1.0000 and dropped
share 0.0000 in every arm — the "dropped" tokens genuinely carry no mass.  Note
that the token-mean void probability and the rendered void fraction are
different quantities and must be quoted as a pair per arm (a v1 mean cannot be
paired with a v2 fraction).

**B5 Gradients / optimizer / data.** 74 `groups.deep.*` parameters, each present
exactly once in the optimizer; `logit_scale` is not a parameter.  Both the main
instance loss and the auxiliary CE reach the group head (head norms 0.286 / 0.554
for v1) and the shared token/anchor/encoder (shared norms 4.569 / 0.730); the
auxiliary CE does reach the shared representation (norm 0.730 ≠ 0), which is the
precondition for condition B2.  Assignment targets are detached.  Data pipeline:
`camera_normalization_method = first_cam`, RGB `uint8/255 → [0,1]`, labels decoded
by integer arithmetic from the packed PNG (no interpolation), batch order
context-first (`[c0,c1,n0,n1]`).  Permuting the GT semantic/instance labels leaves
RGB, alpha, Gaussians and the group/semantic predictions **bit-identical**
(max abs diff 0.0) — no label leakage into the forward.

---

## C. Nothing in A1–A6 justifies discarding the published numbers

Three of the six items (A1, A3, A5) cannot change a single metric; A4 and A6 are
reporting-layer defects; only A2 changes the trained function class.  §D checks
the published numbers by re-running them with the corrected harness.

---

## D. Corrected re-evaluation of G0+ / recipe_v1 / recipe_v2

Job 55636, same 8 windows, novel views 2/3, records joined by
`(scene, view, frame_id, key)`.  All three arms reproduce the published numbers
**exactly**:

| arm | Δ IoU1 mean | Δ AP50 | Δ novel PSNR | Δ novel SSIM | ≥0.5 identical | TP/FP/FN identical | records |
|---|---|---|---|---|---|---|---|
| G0+ | 0.0 | 0.0 | 0.0 | 0.0 | ✔ | ✔ | 55 |
| recipe_v1 | 0.0 | 0.0 | 0.0 | 0.0 | ✔ | ✔ | 55 |
| recipe_v2 | 0.0 | 0.0 | 0.0 | 0.0 | ✔ | ✔ | 55 |

`reeval_comparison.json → all_passed = true`, key sets identical to G0+ with no
duplicates and no missing records.  The harness corrections (A3–A6) therefore
changed **no** published number, and the PSNR gate values are now recomputed from
each checkpoint's own forward rather than read from a history file.

---

## E. Condition A and B2 status

| | control (recipe_v2) | A: pure4 | B2: pure4 + stop-gradient |
|---|---|---|---|
| head mode | legacy_prefix (deep stacked on the historical prefix) | **pure4** (prefix not called) | pure4 |
| assign CE shared gradient | yes (0.73) | yes | **none / 0** |
| IoU1 mean (55) | 0.1927 | 0.1886 | 0.1486 |
| IoU≥0.5 | 3 | 1 | 0 |
| GT-free TP | 3 | 1 | 0 |
| AP50 | 0.0750 | 0.0156 | 0.0000 |
| novel PSNR | 18.105 | 17.741 | **18.985** |
| five gates | 0/5 | 0/5 | 1/5 (PSNR only) |

Condition A shows the specification-correct head is **not** better than the
historical stacked one.  Condition B2 is the only arm that passes the PSNR gate
while losing all instance detections.

---

## F. Unresolved / not covered

* The audit does **not** prove the absence of other bugs; it covers the checks
  listed in the brief plus the ones named above.
* `forward_full` call counters and the capture hook are diagnostics added this
  round; they do not alter the forward math.
* The official evaluator is the pinned SIU3R implementation; only the
  `EvaluatorCfg` we construct is ours (semantic-only and image/depth switches).
  See `official_interface.json` for the file SHAs and the exact reader contract.
