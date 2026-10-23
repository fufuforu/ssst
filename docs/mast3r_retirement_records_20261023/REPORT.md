# 旧 MASt3R 实验权重清理完成（2026-10-23）

用户授权：“之前用mast3r那些版本，也可以删除”。此次沿用此前清理边界，仅清理确认属于旧 MASt3R 路线的训练权重和模型补丁。

## 完成结果

删除90个路径，对应89份独立文件，按实际文件分配块统计释放342.8934 GiB（368179017216字节）。42个目录写入CHECKPOINTS_RETIRED_MAST3R_20261023.json，标记主动退役并禁止自动重训。

完成收据记录清理前可用133.79 GiB，清理后483.08 GiB；随后df显示484G可用、使用率69%。文件系统可用空间增量约349.29 GiB，可能包含并发变化，不把增量与本次删除权重占用混淆。

## 选择与保护

仅在/space/mawb/ssst/workspace_group_plus与group_plus的旧实验中筛选。完整模型以ZIP内data.pkl的实际understanding.encoder.enc_blocks、understanding.mask2former或pretrained_understanding参数名为证；补丁以对应实验文档中明确的MASt3R来源为证。仅检查元数据，无模型反序列化、GPU加载或训练。不是根据文件名含VGGT就删除。

每条候选在删除前校验设备、inode、大小、mtime与硬链接数；共享inode须全部链接均在授权清单内才删除。u128的checkpoint_epoch_08.pt与latest_epoch_08.pt是同一文件的两条链接，已合并计数。所有候选先完整验证，再逐项写审计日志并unlink；90个删除路径均核实不存在。

保留当前三组V4init的完整目录、checkpoint、epoch4及中间恢复源、初始化零步、旧A4来源及共享依赖；spatial_pair实验及其输入也保留。39项当前输入/参考文件删除前后设备、inode、大小、mtime一致。当前score_dual_stream_v1.py仍读取旧MASt3R对照的siu3r_table1.csv，该CSV保持完整。

保留40项旧实验官方评测状态/特征缓存，并逐项核验未改变；32项未确认为MASt3R的模型不在本次清单。保留全部代码/Git worktree、冻结执行快照、配置、日志、TensorBoard、正式指标CSV/JSON、GT清单、导出图片/数组和数据集。全局MASt3R/SIU3R预训练文件及当前重建预训练源在清理范围外，已检查仍存在。

## 当前作业与接手

未提交、取消、修改任何作业。此次仅为保护输入做了一次队列核对：71192、71188为RUNNING，71194为PENDING；各自评测71193、71189、71195仍等待依赖。该观察不代表GPU首步或正式8epoch评测已经核实，不在本次持续轮询。

当前训练继续以codex_handoff_v4init_jsonfix_retry_two_20261023.md及其关联恢复记录为准；三个执行快照保持原样。之前的旧checkpoint清理记录仍见codex_handoff_obsolete_checkpoint_cleanup_completed_20261023.md，此次是在其基础上补充退役旧MASt3R权重。

## 审计

本地目录：/space/mawb/ssst/group_plus/mast3r_checkpoint_retirement_20261023_v1。

- INVENTORY.json：最终90条路径、参数证据、未选项、保留项与家族统计。
- RAW_INVENTORY_BEFORE_METRIC_STATE_EXCLUSION.json：初始候选历史记录，其中评测状态和缓存后来明确保留；不能直接用来删除。
- REVALIDATED_DELETION_PLAN.json：最终授权范围、保护文件与选中路径。
- DELETION_LOG.jsonl / DELETION_COMPLETE.json：逐项删除日志、完成收据与空间统计。
- PRESERVATION_VERIFICATION.json：额外40项状态/缓存与全局资产保留核验。
- inventory_mast3r.py / delete_mast3r_checkpoints.py：可审查的清单与删除实现；删除完成后脚本拒绝重复执行。

上述资料归档到docs/mast3r_retirement_records_20261023/。不要因旧权重被主动删除而重复提交实验。已停止轮询，等用户通知完成后再检查训练。
