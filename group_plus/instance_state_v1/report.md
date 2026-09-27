# instance_state_v1 - execution report

Spec: `docs/instance_state_v1_codex_prompt.md` (registration v1, 2026-09-27).
Status labels: **measured** / **interpretation** / **not executed**.

## 1. Answers to the four first-page questions

1. **Did the three tasks learn?**  **Not executed.**  The gating smoke did not
   reach a passing state, so the paired 2000-step training (and therefore any
   semantic / instance / panoptic metric) was not run.  No claim is made.
2. **Is there a paired benefit from state coupling?**  **Not executed** - this
   requires the C/E arms, which the spec forbids starting before the smoke passes.
3. **What happens to the previously failing windows now?**  **Not executed.**
   `sentinels.json` and the 55/110 table were not produced.
4. **Did we enter full training?**  **No.**  Exact blocker: section 5.
   Pretrained initialisation is the 47500-step reconstruction checkpoint; new
   training steps executed so far = 0 (only smoke-internal temporary steps).

## 2. What was completed and measured

**Interface audit** (`interface_audit.md`) - measured: HEAD is
`bb37a4be71d2f45bde7cbc25039a93582c36fa91`, one cleanup commit past the registered
baseline, with **zero diff** in every dependency file, so the spec's fail-fast is
not triggered.

**`prepare` (CPU) - completed.**  Written artefacts:

| artefact | measured content |
|---|---|
| `spec.json` | every constant/公式 of the spec, version 1 |
| `init_report.json` | pretrained sha256 verified; **450/450** reconstruction keys matched, **59** new `instance_state.*` tensors (2,785,552 params); C and E arms share an identical new-module initialisation; `locusgs_freeze_decode_radius=True`, `radius_init=0.15`, `bound_delta=True`, layers (6,12) |
| `optimizer_groups.json` | 509 named parameters, each appearing exactly once (no aliases); backbone 115 decay / 335 no-decay (218,773,504 / 1,229,116 numel), instance_state 19 decay / 40 no-decay (2,769,864 / 15,688 numel) |
| `pilot_windows.json` | 4 windows: sentinel `scene0009_02` ctx[209,253] novel[215,247] + 3 auto-selected official-train windows with >=2 GT things each >=100 px |
| `monitor_8pairs.json` | first official `val_pair.json` pair of the first 8 distinct scenes, no GT/prediction filtering |
| `plan_paired_2000.json` | w0..w3 x 500 = 2000 steps, sha256 `32c40a72e45843a9...` |
| `storage_budget.json` | free **76.39 GiB**; full budget requires 7.56 GiB -> **allowed** |
| `provenance.json` | commit, git status, split hash, checkpoint hash, SIU3R commit, torch/python, GPU |

**`smoke` - partially completed.**  13 of the 24 planned checks ran and **all 13
passed** (recorded in `smoke.json`):

* `A.strict_transfer`, `A.C_E_identical_init`
* **`A.beta0_h_mu_rho_bitwise`: max |dh| = |dmu| = |drho| = 0.0** - with beta = 0
  the new model reproduces the pretrained LocusGS reconstruction path *bit-exactly*
* **`A.beta0_render_and_gs`: max |d rgb| = |d alpha| = |d gs| = 0.0**
* `B.sum103_eq_alpha` 1.37e-6, `B.S_plus_void_eq_1` 1.37e-6,
  `B.assignment_rows_sum1` 2.98e-7 (all <= 1e-5 / 1e-6 as specified)
* `B.all_finite`, `B.radius_positive`, `B.fps_100_distinct`,
  `B.decode_radius_frozen` (decode radius exactly 0.15)
* `C.slot_identity_varies` 1.39e-4, `C.perturbation_propagates` 3.6e-5 - the 64
  Gaussians of one token are **not** forced to share an identity/assignment, and a
  single-slot `de` bias perturbation propagates (unit-construction check only)

## 3. Exact failure (measured)

The smoke then aborted with a **CUDA device-side assert** inside the *first
backward of the understanding loss* (check D):

```
File tokengs/models/instance_state_locusgs.py / smoke_instance_state_v1.py line 160
RuntimeError: CUDA error: device-side assert triggered
```

Checks D (gradient reach), E (coupling intervention), F (leakage), G (degenerate
labels), H (resume) and J (timing) therefore **did not execute**, and the paired
2000-step arms were **not started** - per spec section 10 the smoke must pass in
full before the formal paired run.

Four defects were found and fixed during the smoke iterations (all in the new
code, none in shared modules): the thing-state query initialisation used
`X6[selected]` instead of `mu6[selected]`; the state bank was sized 103 instead of
102 (the void entry is a channel, not a query); two `einsum` index patterns in
`update_states`/`token_message`; the Gaussian-residual reshape required a
token-major `[B,T,P,3]` view; `S_void` needed an explicit `unsqueeze(2)` (it was
broadcasting to 5 dimensions and produced the 0.19 conservation error seen in the
first run, now 1.37e-6); and the matcher indexed `p_class` with the semantic class
instead of the zero-based column.

**Interpretation.**  The remaining assert is inside the understanding-loss
backward; the identity/conservation/forward checks all pass, so the defect is
localised to the loss path (most likely an index built from labels during the
matched-mask/Dice or identity-contrastive term).  The correct next diagnostic is
to re-run the smoke with `CUDA_LAUNCH_BLOCKING=1` (and/or the same loss on CPU) to
get the failing kernel/index, fix it, and only then start the paired arms.

## 4. What was NOT executed

`paired` (C 2000 + E 2000), `full`, `eval_instance_state_v1.py`,
`export_instance_state_official.py`, `submit_instance_state_v1.sh`, and therefore
`C/E history.jsonl`, `paired_table.*`, `per_instance.csv`, `sentinels.json`,
`table_55_and_110.json`, the official 8-pair subset evaluation, the new
`gate.json`, and the final 1860-pair official evaluation.  **No fabricated table
is provided for any of them.**

## 5. Blocker and requested decision

Blocker: one CUDA device-side assert in the understanding-loss backward.  Nothing
about the architecture or the data is implicated yet - the failure is in my new
loss implementation, which the spec says I must fix and re-run rather than route
around.  I did not reduce steps, change the GPU precision, substitute samples or
weaken a threshold to get past it.

Next actions when the session continues:

```bash
srun -p 3090 -w 3dimage-13 -N1 -n1 --gpus-per-task=1 -c 8 --time=02:00:00 bash -lc '
  export PATH=/space/mawb/anaconda3/envs/tokengs/bin:$PATH; cd /space/mawb/ssst
  CUDA_LAUNCH_BLOCKING=1 python -u scripts/smoke_instance_state_v1.py'
# then, once smoke.json reports ok=true:
python scripts/run_instance_state_v1.py --phase paired      # on a GPU node
```

## 6. Git and workspace state

New code and reports are committed with an `exp:` prefix; large weights stay in
`workspace_group_plus/instance_state_v1/` and are not added to git.  Historical
checkpoints, `implementation_audit_v1/gate.json`, the G0/G1/G0+ results and the
untracked files of concurrent sessions were not modified.
