# instance_state_v1 - repair report (revision 1)

Spec: `docs/instance_state_v1_e0e6805_repair_codex_prompt.md`.
Status labels: measured / interpretation / not executed.

## 1. First-page answers

1. **Did the three tasks learn?** **Not executed.** The repaired smoke has not
   been run yet, so the C/E arms were not started and no semantic / instance /
   panoptic metric exists.
2. **Paired benefit from coupling?** **Not executed** - that is the C/E contrast.
3. **What happens to the previously failing windows now?** **Not executed.**
4. **Entered full training?** **No.** Blocker in section 4.
   Pretrained initialisation = the 47500-step reconstruction checkpoint; new
   training steps executed = 0.

## 2. Measured in this round

* **CUDA assert attributed** (`cuda_repro.log`): with `CUDA_LAUNCH_BLOCKING=1` the
  first failing frame is `thing_loss`, `instance_state_loss.py:120`,
  `y = tgt.reshape(...)[:, flat_idx]` - a 2-view GT tensor indexed by a flat index
  built from the 4-view valid mask. This matches the static finding exactly.
* **CPU loss contract 14/14 PASS** (`loss_contract.json`), including the
  logit-vs-log round trip (0.900000 vs 0.473684), the permute-before-flatten
  requirement and its `idx=((v*H)+y)*W+x` decoding, the weighted CE closed form
  (0.126928), context-swap invariance, novel-label independence, degenerate-label
  finiteness and the two fail-fast paths.
* **All L1-L8 and S1-S7 defects fixed** - see `repair_audit.md` for the
  item-by-item table.
* **`prepare` re-run** in the repair dirs: pretrained sha256 verified, 450/450
  reconstruction keys matched + 59 new tensors, C/E identical init, four real
  optimizer groups, the 4 pilot windows **re-screened with the corrected GT rule
  and all still qualified** (no `DATA_PROTOCOL_BLOCKED`), and the window / plan /
  monitor SHAs are byte-identical to e0e6805 (`e5c5878b…`, `32c40a72…`,
  `5dae7077…`) - the samples and the plan were not re-drawn.
* **Corrections**: `corrected_provenance.json` (the e0e6805 report field that
  carried the monitor SHA instead of the checkpoint SHA; the checkpoint itself was
  never modified - its sha256 is still `5fcf71b9…`), and
  `actual_storage_budget.json` with a *measured* serialised checkpoint size
  (891,358,595 bytes) plus the explicit caveat that its Adam state was empty.

## 3. Blockers and next actions

Remaining work: run the repaired GPU smoke (`sbatch scripts/submit_instance_state_v1.sh
smoke`), then the paired arms, then write `eval_instance_state_v1.py` and
`export_instance_state_official.py`.  Those two eval scripts are **not yet
written**, so the post-training evaluation and the official 8-pair subset cannot
run yet; that is an implementation gap, not an external blocker.

```bash
sbatch scripts/submit_instance_state_v1.sh smoke    # A-J, must be fully green
sbatch scripts/submit_instance_state_v1.sh paired   # C then E, 2000 steps each
```

## 4. Provenance of this round

* commit: recorded in the commit message of this repair (`exp:` prefix)
* report dirs: `group_plus/instance_state_v1_repair/`,
  `workspace_group_plus/instance_state_v1_repair/` (e0e6805 dirs untouched)
* protected: original 47500 checkpoint, `implementation_audit_v1/gate.json`,
  G0/G1/G0+ results, every earlier checkpoint and the concurrent sessions' files
