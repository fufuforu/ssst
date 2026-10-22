# Old A4 V4init / NoCA submitted 2026-10-22 09:23 CST

Execution commit: `5335430b6f0e40fccb042162ae47f4723da32669`.
Execution snapshot: `/space/mawb/ssst_olda4_v4init_noca_code_20261022_v1`.
Run root: `/space/mawb/ssst/group_plus/semantic_resolution_v1/oldA4_v4init_noca_ablation_20261022_v1`.

|Arm|1GPU 2-step smoke|8GPU full4 train|8GPU official1860 eval|
|---|---|---|---|
|V4init|70714|70715, afterok:70714|70716, afterok:70715|
|NoCA|70717|70718, afterok:70717|70719, afterok:70718|

All six jobs accepted and PENDING at the single bounded acceptance query. No nodes allocated at that check. Smoke reason None; training/eval reason Dependency. Eligible single-node pool: 12/13/14/17/18, partition3090 or4090. GPU smoke/first update/DDP synchronization have not yet been verified.

CPU: five tests passed in 16.148 seconds. Actual source-zero audits passed: V4init 655 retained /38 removed /9 added, NoCA 661 retained /32 removed /0 added; retained tensors bitwise equal. Each arm rejects seven invalid migration cases. Eight original A4 model/loss/policy source bodies byte-identical.

Frozen recipe: original 2+2 data, FM128, L6/8/10/12 [.1,.2,.3,.4], four epochs4172 updates, global8, uniform baseLR1e-4 warmup100 cosine floor1e-5, U .05/1043, joint gradients without adapters or isolation. V4init intentionally changes the full reconstruction initialization including fixed1024 anchors. NoCA changes only Anchor CA and replaces its R with geometric softmax, keeping FM CA and c/s updates.

Smoke failure blocks downstream jobs. Formal training always reloads original old-A4 zero weights and RNG; no smoke parameters are carried forward. Evaluation freezes the completed checkpoint via hardlink, then uses fixed1860 protocol and verifies inputs/GT against old A4.

Polling stopped after the one acceptance query. Wait for the user's completion/status instruction before inspecting GPU logs or results. Previous C/D jobs70691–70696 were not queried, cancelled, changed or resubmitted.

This publication worktree advances only documentation; the sealed execution HEAD remains unchanged. No checkpoints, datasets or large intermediate tensors are published.

## Later verified updates; earlier paragraphs retain submission-time history

- 2026-10-22 09:24 CST: user cancelled previous P1024iso C/D training70692/70695 and dependent evaluations70693/70696. All four confirmed CANCELLED; local USER_CANCELLED.json records it. This supersedes the earlier statement that they had not been cancelled at submission time. Old-A470714–70719 untouched.
- 2026-10-22 09:49 CST: on the user's duration question, one bounded scheduler check found70715 RUNNING on13 (8×3090),70718 RUNNING on12 (8×3090),70716/70719 PENDING(Dependency). Follow-up receipt/log parsing confirmed smoke gates passed and eight-rank first-update hashes matched and changed for both arms. Rank0 updates were approximately150/4172 and163/4172, recent100 step means6.036s and5.304s. These are historical observations, not live status.
- No queue/progress checks were made solely to write the new handoff. Do not resubmit from stale records.

Latest handoff: [codex_handoff_olda4_v4init_noca_20261022.md](../codex_handoff_olda4_v4init_noca_20261022.md).
