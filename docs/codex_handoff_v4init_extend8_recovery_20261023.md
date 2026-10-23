> **最新：JSON修复后已重提两条链**：71192→71193（Baseline）、71194→71195（U005）；现有71188→71189（2+4）不动。见[最新两组恢复交接](codex_handoff_v4init_jsonfix_retry_two_20261023.md)。已停止轮询，不重复提交。

> **最新检查（2026-10-23 08:47）**：71188正常运行至本次读到4455步；71184/71186因恢复JSON元数据的betas容器类型误判失败，71185/71187取消。修复已完成CPU验证但未重提。见[检查及修复交接](codex_handoff_v4init_recovery_status_jsonfix_20261023.md)。下文PENDING为提交时历史状态。

# 最新交接：三组V4init续训恢复及epoch8评测已提交（2026-10-23）

遵守/space/mawb/AGENTS.md。用户清理空间后明确授权“先提交续训作业吧”。已提交三个独立恢复训练及其原先授权的三个8epoch评测，**不要重复提交**。提交后唯一调度核对已完成，已停止轮询，等用户通知后检查。

## 已提交作业

北京时间2026-10-23 08:06:13提交。唯一接收核对时，三训练均PENDING(None)，三评测均PENDING(Dependency)，尚未分配节点。每项单节点8卡3090/4090、32CPU/192G；允许池12/13/14/17/18，使用两分区并排除11，不是同时申请五个节点。

|组|理解权重|视角|恢复累计步|剩余更新|新训练|afterok评测|
|---|---:|---|---:|---:|---:|---:|
|Baseline|1.0|2+2|5500|2844|71184|71185|
|U005|0.05|2+2|4172|4172|71186|71187|
|Views2ctx4|1.0|2+4|4172|4172|71188|71189|

新三训练无失败旧作业依赖，可各自调度；评测分别afterok:71184、71186、71188。没有新增summary或GPU smoke，没有延长到16epoch。原三组4epoch及指标不改、不重提；旧71031/71033/71035失败目录不覆盖，71032/71034/71036取消状态也不恢复。

估计：基线余下训练4–5小时，另两组6–8小时；各正式1860评测约1–1.5小时，排队另计。依据旧Baseline新增1378步约2h05、原4epoch六七小时及已有正式评测耗时。没有为估时读新任务日志。

**本次GPU源加载、前反向、首次新增更新和八卡同步尚未核实**；CPU审计通过或之后RUNNING都不能替代GPU验证。写本交接没有再次查询队列或读取新训练日志。

## 路径、来源与代码封存

- 新执行快照：/space/mawb/ssst_v4init_extend8_recovery_code_20261023_v1
- 冻结执行commit：9c9ffcca5580a1b05003cc31d112177dd5000354，不改HEAD/源码。
- 新运行根：/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_recovery_20261023_v1，各Baseline/U005/Views2ctx4下train/eval。
- publication沿用独立worktree：/space/mawb/ssst_v4init_extend8_publication_20261022_v1；它的HEAD与执行快照不同是刻意隔离。
- GitHub新分支：experiments/v4init-extend8-recovery-20261023。
- 实现规格：[v4init_extend8_recovery_20261023.md](v4init_extend8_recovery_20261023.md)。本次源审计/提交记录归档：v4init_recovery_records_20261023/。

Baseline冻结旧71031的checkpoint_latest.pt，SHA13232437521ff24eaf7b808f344ea20a0a8be9294423ff743d175ec0689fe70c，progress=5/285/5500；旧日志5501–5550未保存，按5500状态重算这50步，不回退到4epoch。下一rank0训练窗1041，与旧日志5501完全一致，下一主LR2.437852545768077e-5。

U005冻结旧71033初始化后的latest，SHA7bad9b6390c3d613bac13c167548878947539510702d8459c6487356aac7447f，progress=4/0/4172；模型与本组epoch4最终权重逐值相同，保留continuation metadata、完整Adam和RNG。

Views2ctx4由本组原epoch4最终文件开始，SHA3a5954de0ea26ff6d70dd1a31c1f2aa073be17b44ffb83af01951cade6268fc5，progress=4/0/4172，通过原after-success训练门禁冻结源。

原4epochSHA：Baseline694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc、U005a5cb53ab2aa05e0198d0c6ba0930ee244d1f1bd08cc4abdd5b766a0f2ac313cb；共同零步bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397。原4epoch源代码310d49d/ef98a6f保留，旧续训源代码ce8446d保留，不能误替换成新执行SHA。

## 提交前完成的验证与容量处理

真实CPU实体审计三组均632条模型状态、604份Adam、8rank RNG；模型及完整Adam恢复逐值一致，step分别5500/4172/4172。数据8337窗/1191场景、2context不变，真实下一batch分别4/4/6个监督视角；下一rank索引详见CPU_SOURCE_VALIDATION.json和各PREFLIGHT。

11项CPU测试通过，18.476秒测试正文、含进程启动32.549秒：部分续训和刚初始化续训的下一次Adam更新与连续参考完全一致，错误配置/clock/LR/NaN/RNG/Adam拒绝，LR及理解warmup不重启，八rank新恢复收据校验和final latest冻结/重复拒绝。shell语法、Python编译、diff检查、Slurm test-only均在实际提交前通过。模型、loss、原采样/LR/评分等10个关键文件与ce8446d逐字节相同。

恢复入口以原4epoch metadata和当前runtime重建完整continuation identity，仅允许执行code SHA改变；实际权重哈希需匹配恢复源。模型、Adam时钟、8rank RNG、采样位置保留，不重启100步LR升温或1043步理解warmup；Baseline在原曲线上继续下降。几何LR始终主LR0.1倍。

新运行启用--no-epoch-checkpoints，只保留起点、latest、原子保存临时文件和评测最终硬链接；不保留epoch5/6/7权重，各epoch probes和TensorBoard累计步数保留。32GiB新增容量预算包含三组checkpoint峰值和正式导出；预检时可用约41.52GiB，提交前再次满足>32GiB。其他并行任务未独占预留，不能保证外部写入不会耗尽空间。

## 用户通知完成后的核验

1. 查71184/71186/71188训练和71185/71187/71189评测的终态/失败收据及Slurm日志，沿实际进度继续，不因本文PENDING重提。
2. Baseline/U005读取RECOVERY_SOURCE.json与RECOVERY_LOADED_rank00–07，source SHA和progress须为5500/4172；Views2ctx4读取CONTINUATION_SOURCE.json与CONTINUATION_LOADED_rank00–07，progress须4172。三组FIRST_UPDATE_VERIFIED必须新八rank同步且参数变化。
3. 新各rank训练日志条数应2844/4172/4172，累计5501..8344或4173..8344；TRAINING_COMPLETE必须8/0/8344。检查真实模型/Adam有限并有更新，未消费Write0状态无需变化。
4. 本次没有epoch8单独文件，eval门禁从已完成的train/checkpoint_latest.pt冻结最终source_checkpoint.pt；不因缺少checkpoint_epoch8.pt判训练失败。
5. 评测EVALUATION_VERIFIED.json、FINAL_CHECKPOINT_FROZEN.json、evaluation/metrics.json和checkpoint_identity.json，正式1860/312、更新0，输入GT与旧A4匹配，SIU3R8ea80166be76854f938e90521f1a5b688b755c87，target-all6含context。各comparison/epoch4_vs_epoch8.csv输出本组续训收益。

训练启动后失败不自动重跑，8→16需要另行指令。本轮未取消或修改排队中的spatial71165/71166/71167及其目录/源文件。

已停止轮询，等用户通知完成后再检查。
