# closure_audit - instance_state_v1, closure round (baseline 9e6cab2)

Status labels: **measured** / **static** / **not executed**.  Historical CUDA
evidence is only *referenced* (`repair/cuda_repro.log`), never re-claimed.

## 1. Math fixes in the training objective (section 2)

| item | fix | status |
|---|---|---|
| 2.1 matcher class cost | new `_matching_cost(logits19, z, y, gt_class)` uses `P = softmax(logits19)`; `thing_loss` calls it (no inline copy) | measured |
| 2.2 matcher invariance test | `L4b.matcher_softmax_invariance`: adding +100 to all 19 logits leaves the cost unchanged (maxΔ 0.00e+00) and the 2-query/1-GT construction still selects query 0 | measured PASS |
| 2.3 cross-view class check | `validate_thing_targets` now merges both context views before requiring one class per positive instance id; 255 excluded | measured PASS (`c.rejects_cross_class_instance` fires on a cross-view conflict) |
| 2.3b identity batch | `identity_loss` asserts B == 1 | static |
| 2.4 genuinely different novel labels | contract check `d.novel_labels_ignored` now rotates novel semantics (+8 mod 20) and offsets ids (+100) and compares the **real** `instance_state_losses` outputs across three settings, with `requires_grad` predictions | measured PASS (loss/parts/grads all equal) |
| 2.5 CE from the real loss | `L4c.thing_loss_matches_numpy_reference` calls the real `thing_loss` and compares against an independent NumPy implementation of the registered formulas | **FAIL - 10.859805 vs 20.266098** |
| 2.6 single grouping | `scripts/instance_state_runtime.build_optimizer` builds the four AdamW groups and derives the report **from the actual `param_groups`**; `run_instance_state_v1.build_optimizer` and the old `optimize_groups` now delegate to it | measured (`every_trainable_exactly_once`) |

**Not resolved:** the `L4c` reference harness still disagrees with the
implementation (10.86 vs 20.27).  The two *semantic* sub-checks of the matcher
(probability-invariance and question-of-record argmin) pass, so the disagreement
is currently attributable to my reference re-implementation, not to a proven
implementation defect - but I have **not** proven that, so the check stays
FAILING and `loss_contract.json` reports `ok=false`.  No threshold was relaxed.

## 2. Runtime and H (section 3)

* `scripts/instance_state_runtime.py` added: `build_optimizer`, `set_lrs`,
  `lr_at`, `train_one_step` (asserts finite losses, `clip_grad_norm_` with
  `error_if_nonfinite=True`, per-group LR set *before* the update, no default
  AdamW), `capture_rng`/`restore_rng` (python/numpy/torch/cuda),
  `checkpoint_payload`, `save_checkpoint_atomic` (write -> read back -> rename),
  `verify_state_dict`.  **static/measured** for the parts exercised by the smoke.
* The H rewrite (shared first four steps, reference 8 steps vs 4+4, comparing
  **step 8 against step 8**) is **not executed** in this session: the smoke was
  not re-run after the runtime module landed.

## 3. Smoke / eval / paired (sections 4-6)

* `scripts/eval_instance_state_v1.py` and `scripts/export_instance_state_official.py`
  are **not yet written**; `scripts/instance_state_paired.py` still carries the
  pre-closure shape (no in-loop evaluation, `sha256_file(...)==""` guard, no gate
  read, `arm or E` default).  All of section 5 and most of section 6 is therefore
  **not executed**.
* No GPU smoke, no C/E 2000-step arms, no official 8-pair subset, no 55/110 table,
  no new `gate.json`, no full phase.

## 4. Locked inputs (unchanged)

`pilot_windows.json` `e5c5878b21d57218…`, `plan_paired_2000.json`
`32c40a72e45843a9…`, `monitor_8pairs.json` `5dae7077779dd398…` - byte copies of
the repair round, no re-sampling, no new paired plan.  The 4 pilot windows still
pass the corrected GT re-screen (`repair/pilot_recheck.json`).

## 5. What is measured vs claimed

Measured this round: the matcher probability semantics and its invariance, the
cross-view class validation, the novel-label independence with real losses and
gradients, and the single grouping/report implementation.  Everything else in the
closure prompt is **not executed** and no partial result is presented as a
passing gate.
