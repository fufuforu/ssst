# instance_state_v1 - closure report (baseline 9e6cab2)

Status labels: measured / static / not executed.  Historical CUDA evidence is
referenced, not re-claimed.

## 1. First-page answers

1. **Did the three tasks learn?** **Not executed** - no C/E arms were run.
2. **Paired benefit from coupling?** **Not executed** (that is the C/E contrast).
3. **Previously failing windows now?** **Not executed** (no step0/step2000 evaluation).
4. **Entered full training?** **No.** The blocker is the unfinished execution
   chain of section 5/6 (eval + export scripts and the paired-driver upgrades),
   not an external resource.  Pretrained 47500 reconstruction checkpoint; new
   training steps executed = 0.
5. **Pretrained / new steps:** 47500 / 0.

## 2. Measured this round

* `_matching_cost` now uses the softmax probability of the 19-d logits for the
  class term (the raw-logit version violated the registered formula), and the
  CPU contract proves the cost is invariant to a +100 logit shift (maxΔ 0.00e+00)
  and still prefers the query with the larger GT logit.
* `validate_thing_targets` merges the two context views, so a positive instance id
  must carry one class across the pair; the contract's cross-view conflict case
  now fires (previously only per-view conflicts were caught).
* The novel-label contract check is real: novel semantics rotated (+8 mod 20) and
  ids offset (+100), three settings compared through the actual
  `instance_state_losses` with `requires_grad` predictions - loss, all four
  components and all input gradients are identical.
* `scripts/instance_state_runtime.py` provides the single grouping implementation
  (report derived from the real `param_groups`, `every_trainable_exactly_once`),
  the registered schedule, `train_one_step` with finite-loss assertions and
  non-finite-aware clipping, RNG capture/restore and an atomic
  write-read-rename checkpoint.
* `L4c.thing_loss_matches_numpy_reference` **fails** (10.859805 vs 20.266098).
  I could not reconcile my independent NumPy reference inside this session; the
  check is left failing and `loss_contract.json` reports `ok=false`.  No tolerance
  was relaxed and no passing claim is made.

## 3. Not executed

`scripts/eval_instance_state_v1.py` and `scripts/export_instance_state_official.py`
do not exist yet; `scripts/instance_state_paired.py` still has the pre-closure
shape (no in-loop evaluation, a vacuous SHA guard, no smoke/gate read, `arm or E`
default).  Therefore the GPU smoke (A-J, including the rewritten H and the official
reader check I), the four-window step0 evaluation, C2000, E2000, the official
8-pair subset, the 55/110 table, the new `gate.json` and the full phase were **not
run**.  No table is fabricated for any of them.

## 4. Exactly what is needed next (implementation, not external)

1. Reconcile `L4c` (or fix the reference) until the CPU contract is green.
2. Write `eval_instance_state_v1.py` (`evaluate_windows(model, opt, windows, step,
   scope, output_dir)`, per-class IoU/confusion, raw + GT-free instance counts,
   local panoptic PQ/TP, PSNR, sentinels, RNG-safe eval mode) and
   `export_instance_state_official.py` (official reader format, PNG + pred.json).
3. Upgrade `instance_state_paired.py`: real plan-SHA binding, four cached batches
   via `pin_pair`, `train_one_step` + in-loop evaluation at
   step 0/200/500/1000/2000 and 8-val at 0/1000/2000, atomic endpoints, smoke and
   gate gating, and `full` restricted to `gate.selected_arm` with `--resume`.
4. Then run: CPU contract -> GPU A-J (with the rewritten H and I) -> four-window
   step0 -> C2000 -> E2000 -> official subset -> gate -> full segments.

## 5. Protected state

Original 47500 checkpoint (sha256 5fcf71b9…), `implementation_audit_v1/gate.json`,
G0/G1/G0+ results, every earlier checkpoint and the concurrent sessions' files
were not modified.  `group_plus/instance_state_v1_repair/` is kept; this round's
artefacts live in `group_plus/instance_state_v1_closure/` and
`workspace_group_plus/instance_state_v1_closure/`.
