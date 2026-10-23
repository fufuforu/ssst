> **旧权重清理更新**：已按用户授权删除102个不影响当前三组的旧checkpoint路径，释放68.00GiB；本轮续训/评测和所需输入均保留，不重提。见[清理完成记录](codex_handoff_obsolete_checkpoint_cleanup_completed_20261023.md)。

# 最新交接：JSON修复后两条续训链已重提（2026-10-23）

遵守/space/mawb/AGENTS.md。用户明确授权“修复之后提交续训作业吧”。只恢复Baseline/U005两组，现有2+4训练71188和依赖评测71189不改、不取消、不重提。已完成提交后唯一有界状态核对，**已停止轮询，不重复提交**。

## 作业、状态与资源

北京时间2026-10-23 **08:58:25**提交四项任务。

|组|理解权重/视角|恢复累计步→终点|新训练|新afterok评测|唯一核对状态|
|---|---|---|---:|---:|---|
|Baseline|1.0，2+2|5500→8344|71192|71193|训练RUNNING，3dimage-12；评测PENDING(Dependency)|
|U005|0.05，2+2|4172→8344|71194|71195|训练PENDING(None)，节点待分配；评测PENDING(Dependency)|

每项单节点8卡3090/4090、32CPU/192G；Baseline核对时分配3dimage-12的8×3090。允许节点池12/13/14/17/18，双分区排除11，未同时申请五台。三组实验仍只有原三组，本轮没有额外smoke、summary或16epoch。

RUNNING只代表调度状态；新代码GPU源加载、前反向、首次新增更新及八卡同步尚未核实。写交接/推送没有再次查询队列或读取新训练日志。

基线剩2844更新，估计4–5小时；U005剩4172更新，约6–8小时，各正式1860评测约1–1.5小时，排队另计。依据既有相同规模吞吐及评测耗时，没有等待新任务首步估时。

## 执行快照与实际恢复源

- 新执行快照：/space/mawb/ssst_v4init_extend8_recovery_code_20261023_v2_jsonfix
- 冻结执行SHA：**a959b9168f4a3049688171079ac63e52d616273c**，不改HEAD或源码。
- 新运行根：/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_recovery_20261023_v2_jsonfix，只有Baseline/U005，每组train/eval。
- GitHub：experiments/v4init-extend8-recovery-20261023。独立publication：/space/mawb/ssst_v4init_extend8_publication_20261022_v1，后续文档commit不改变执行快照。
- 规格：[v4init_jsonfix_retry_two_20261023.md](v4init_jsonfix_retry_two_20261023.md)。提交与审计归档：v4init_jsonfix_retry_two_records_20261023/。

Baseline使用旧71031保留下来的latest，progress=5/285/5500、SHA13232437521ff24eaf7b808f344ea20a0a8be9294423ff743d175ec0689fe70c；U005使用旧71033初始化后的latest，progress=4/0/4172、SHA7bad9b6390c3d613bac13c167548878947539510702d8459c6487356aac7447f，来源执行代码均ce8446d。71184/71186在STARTED前失败，无新增checkpoint可恢复，不将它们当源。

模型、Adam全状态、8rank RNG及累计数据位置保留；Baseline重算未落盘5501–5550。仍沿原4172→8344的100步LR升温/余弦曲线，基线在5500处继续降，不重新启动LR或理解warmup。几何LR为主LR0.1倍。理解权重、隔离/NoCA、L6/L12、FM128、query及训练2+2不变。

## 本轮先完成的修复验证与容量准备

仅恢复JSON导致的optimizer_groups.betas列表→元组类型，数值及其他模型/资产/数据/配方比较保持严格，真实beta变化仍拒绝。13项CPU测试通过（10.541秒测试正文）。

真实预检从已落盘source_epoch4_meta.json读回，再由实际模型config和optimizer groups重建runtime metadata，与GPU入口同路径；两组各632条模型状态、604份Adam、8rank RNG，模型和完整Adam恢复逐值一致，step5500/4172。下一rank0窗口1041/746，实际batch均2context、4个监督视角。记录serialized_metadata_path_exercised和runtime_identity_rebuilt_from_actual_config_and_optimizer均true。trainer、launcher和eval gate与71188执行快照9c9ffcc逐字节一致；模型/数据/LR/评分十个关键文件仍与ce8446d一致。shell语法、Python编译、diff及Slurm test-only在提交前通过。CPU通过不等于GPU验证。

继续启用--no-epoch-checkpoints，只保留起点、latest/原子临时文件及final评测硬链接，保留epoch probes和TensorBoard。不保留epoch5/6/7单独权重。容量门限28GiB覆盖两组新增峰值及现有2+4后续输出余量，其他任务写入未独占预留。

沿用户之前授权的无用实验清理，确认70692/70695为用户取消、无训练完成/正式评测收据、相关配置中无外部引用后，删除C/D两个已取消分支各checkpoint_initial/checkpoint_latest共4份中间文件，释放5.706GiB，清理后可用约36.238GiB。保留所有日志、配置、共享旧A4零步、已完成实验最终权重、当前三组及spatial输入。删除路径/stat/先记录再unlink的日志及完成收据存新运行根CANCELLED_CD_CLEANUP_*；记录一起归档。原9c9ffcc/ce8446d快照和旧失败目录不变。

## 之后检查入口

1. 本轮71192/71194训练与71193/71195评测；2+4继续查原71188/71189。不要因旧PENDING或旧失败状态重提。
2. 新Baseline/U005 train/RECOVERY_SOURCE、RECOVERY_LOADED_rank00–07：SHA正确、progress5500/4172、604份Adam对应step、RNG已恢复，参数哈希八rank一致。FIRST_UPDATE_VERIFIED须是本轮新收据且同步并有更新。
3. 各rank新增日志应2844/4172条，累计5501–8344/4173–8344；TRAINING_COMPLETE必须8/0/8344。最终模型/Adam有限并更新，Write0未消费状态无需改变。
4. 只有final latest，缺checkpoint_epoch8不是失败。评测从已完成latest冻结；eval/FINAL_CHECKPOINT_FROZEN、EVALUATION_VERIFIED、evaluation/metrics及checkpoint_identity，1860窗/312场景、更新0、SIU3R8ea80166be76854f938e90521f1a5b688b755c87、target-all6含context、输入/GT与旧A4一致。comparison/epoch4_vs_epoch8.csv给各自增益。

本轮没有检查2+4最新训练进展，沿本次开头调度观察保留其原链；没有碰spatial71165/71166/71167。STARTED后失败不自动重跑，8→16需之后另行指令。

已停止轮询，等用户通知完成后再检查。
