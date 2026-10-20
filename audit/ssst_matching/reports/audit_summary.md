# Audit preparation and stage handoff — 2026-10-20

CPU contracts: 13/13 PASS. Real-data CPU preflight: eight fixed heldout windows,
eight scenes, 37 valid context GT; GT counts 7/1/6/6/3/5/5/4. Native RGB cohort SHA,
context/novel frame ordering, nearest-resized integer label shapes, and GT targets
passed. This is data verification, not a real-prediction matching smoke result.

Main checkpoint is the completed sem256/A4 Epoch4 / step4172 file:
`/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu/checkpoint_epoch4.pt`.
SHA256 `18aee68f2bc8cdb1d154c742697ccc54657f38ad02c6151bf993405a455f4560`.
Recorded source commit b7a775f19e232ea2395c8611a47c81b1365fb86e plus actual sem256
SemanticMemory snapshot. All optimizer step values are4172. Main matching config
source_many_to_one/support_matching=true/support_purity0.5. Attempt02 Hungarian
config is read only for comparison; none of its weights are used.

Source audit and implementation are in the independent worktree
`/space/mawb/ssst_matching_audit_20261020`, branch
`audit/ssst-superpoint-gt-20261020`. No tracked model/loss/forward/config is changed.
Runtime captures native loss Cost and pairs, preserving exact native validity.

Q1: GT/small-object source coverage requires real superpoint generation; pending
GPU smoke and formal statistics. The 37 count above is not a coverage estimate.
Q2: Source exclusion and maximum one-to-one coverage pending real candidate graphs.
Q3: Hungarian/Top-1 agreement pending same-weight real prediction Cost. Source code
shows exact0.5 cross-GT candidates can conflict, and official strict Top-1 minimum
ties differ from argmin. No conclusion that a new matcher is needed is justified.
Q4: Cannot yet identify the dominant empirical bottleneck. The final trained A4
weight is available, so raw-mask diagnostics are implemented in the same inference.

No aggregate CSV is fabricated before inference. GPU smoke produces the requested
seven files for its eight-window engineering scope if successful; formal32/128
must be separate outputs and must not be claimed from smoke. All generated
matrices/predictions remain local; only lightweight reports are intended for Git.

Execution completed:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.test_contracts
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 /space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.run --stage cpu-preflight --cohort smoke8 --output /space/mawb/ssst_matching_audit_runs_20261020/cpu_preflight_final
```

Next: submit isolated GPU smoke, one bounded scheduler check, stop polling as
required by /space/mawb/AGENTS.md. On user completion notification, inspect smoke
receipt and diagnostics; fix only explicit engineering errors if needed; commit
and push code **after** smoke passes; verify remote; then execute Heldout32 and
expand to the existing128 if resources permit. Formal mode refuses to start
without the passed smoke and a verified push receipt. No training is submitted.

Do not restart existing training, load Hungarian latest, or rerun completed A4.
Do not automatically submit formal jobs behind this smoke: the user requires
reviewed smoke and code push first. GPU runtime is not yet measured for this
script; 45min job limit is a resource cap, not an estimated runtime.

## Submitted Smoke and stop point

Smoke job **69228**, accepted by Slurm. The single bounded status check found
**PENDING (Priority)** on partition3090; requests one RTX3090 / 8 CPUs / 64G.
No node is allocated; scheduler's planned node is3dimage-12. No GPU forward or
real-prediction parity is yet verified. Time cap45min; no measured runtime estimate.

Run and snapshot: `/space/mawb/ssst_matching_audit_runs_20261020`.
Submission command is fully preserved in smoke_submission.json; state receipt is
smoke_scheduler_check.json. GPU smoke runs the command in run_smoke.sbatch:

```bash
/space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.run --stage smoke --cohort smoke8 --output /space/mawb/ssst_matching_audit_runs_20261020/smoke8
```

No formal statistics, code commit, or push has been claimed. Current branch base
is b7a775f19e232ea2395c8611a47c81b1365fb86e; audit additions remain uncommitted
until real smoke passes, per the requested execution order. No training job or
formal evaluation job was submitted. **已停止轮询，等你通知完成后再检查。**

## User-authorized move to node14

After user authorization to use server14, a one-time read found node14 had
2/8 RTX4090 unallocated and job69228 was still PENDING. Updated the **same** job:

```bash
scontrol update JobId=69228 Partition=4090 ReqNodeList=3dimage-14 Gres=gpu:4090:1
```

The single post-change scheduler check confirmed **RUNNING, 3dimage-14,
1×RTX4090, 8 CPUs, 64G**. No cancellation, duplicate submission, input/weight/code
change, or new training. This supersedes the earlier PENDING3090 state; original
submission receipt remains historical provenance. GPU forward and real matching
parity are still unverified. Stopped polling; wait for user completion notification.

## Smoke completed and reviewed

Job69228 COMPLETED/0:0 on node14 / one4090, elapsed110 seconds. All eight windows
passed exact support scores/allowed, exact native Hungarian pairs, unchanged
parameter versions and no gradients. Saved graph/IoU CSV consistency is reviewed
in SMOKE_REVIEWED.json. Full lightweight eight-window results are in smoke8/.

Smoke only: support/oracle/Hungarian each37/37, Hungarian and Top-1 identical8/8
windows and37/37 GT, no Top-1 reuse conflict, two queries with multiple allowed
GT. Raw recall31/37, supported-query raw29/37, Hungarian-selected good mask28/37.
Small support13/13 but mean eligible2.08 versus large16.83. This is insufficient
to claim formal population coverage or final instance recall. See raw protocol
limitations in smoke8/audit_summary.md.

All128 fixed heldout windows passed CPU GT/RGB preflight. Next publish the exact
Smoke-tested code and light reports, then formal32 and dependent formal128 on
node14 / one4090 each. No model, loss, input/weight, or matching-cost change.
