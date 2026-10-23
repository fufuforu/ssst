# 最新交接：旧checkpoint清理完成，当前三组及依赖保留（2026-10-23）

用户明确授权：“我觉得不影响现在基线的ckpt都可以删”。本次执行之前已审查的旧checkpoint清单，未再次索取确认；没有提交、取消、修改任何作业或冻结代码快照。

## 完成结果

- 删除 **102个checkpoint路径**，对应 **84份独立文件**，按实际分配块统计释放 **68.00 GiB**。
- 删除后全盘可用约 **134.73 GiB**，df显示135G、92%使用率。全盘前后可用变化102.87GiB与本次删除文件分配块不完全一致，可能包括并发/文件系统变化；不将该差额归因于本次清理。
- 两实验族semantic_resolution_v1和unified_query_training_v1内，49个目录受影响；逐目录写CHECKPOINTS_RETIRED_20261023.json，明确权重为用户主动退役、保留日志/指标，禁止误当失败后自动重训。
- 逐项删除前验证path范围、设备/inode/size/mtime/nlink，先落盘审计日志再unlink。先完整校验全部候选，再删除；涉及多个硬链接的文件按剩余link数核对，防止误删范围外共享对象。
- 重新扫描当前V4init运行和活动spatial输入引用，保护按inode传播。37项输入/参考文件在删除前后设备/inode/size/mtime一致；102项删除路径均已确认不存在。没有新引用或异常硬链接需要跳过。

## 保留范围

当前三组及全部恢复链不动：Baseline71192→71193、U00571194→71195、Views2ctx4原71188→71189；所有这些运行目录、epoch4及真实恢复源、起点/latest文件、旧失败审计均保留。source_code ce8446d、9c9ffcc与a959b91快照不改。

保留V4init所需初始化零步（旧V4init目录内bf5b5325...源文件）、原始旧A4初始化源及其共享引用。旧候选中21个依赖/共享初始化路径仍保留，并非清理遗漏。

保留旧A4 evaluation/export_complete.json与gt_reference_manifest.jsonl等当前正式评测参考、各组正式epoch4评测和11项指标、所有配置/日志/TensorBoard/完成收据、逐窗RGB/深度/分割导出及数据集。此次仅删除checkpoint，不清理评测图片或数组；代码Git worktree均保留。

活动spatial_pair_20261023_v1及其输入保留。只在本次开始查询一次队列用于活动输入保护，当时71192/71188与spatial71165在RUNNING，71194及评测仍等待；没有为清理等待训练首步、指标、完成或再次读取进度。该调度观察不是GPU训练验证。

## 审计与后续接手

工作目录：/space/mawb/ssst/group_plus/obsolete_experiment_retirement_20261023_v1。

- CHECKPOINT_RETIREMENT_REVIEW_PLAN.json：原执行前候选，保留历史REVIEW_ONLY状态。
- REVALIDATED_DELETION_PLAN.json：本次用户授权、最新依赖保护及102条选中路径。
- DELETION_LOG.jsonl：逐项unlink前后审计。
- DELETION_COMPLETE.json：最终完成收据与空间统计。
- delete_approved_checkpoints.py：受明确范围约束的执行脚本；完成后主动拒绝重复执行。
- CURRENT_BASELINE_FAMILY.json：当前三组登记及本次COMPLETED状态。

文档归档：obsolete_retirement_review_20261023_records/。完整11项指标文件未修改，仍可读取current_three_epoch4_11metrics.csv/json。正式8epoch评测尚未在本轮核验，不虚报新最终指标。

今后旧训练的TRAINING_COMPLETE/评测收据仅代表历史完成，不保证对应退役权重仍存在；先看CHECKPOINTS_RETIRED_20261023.json，不因文件主动清理而重复训练。当前三组检查仍按最新JSON修复恢复交接进行。

已完成清理，停止状态轮询，等用户通知后再检查训练。
