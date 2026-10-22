# 旧实验中间权重清理完成（2026-10-23）

用户授权清理之前无用实验文件，指出很多旧版本未调整几何头LR。本轮执行了一批明确的旧中间checkpoint清理，不把旧版本的所有对照结果判作无效。

共删除 **21** 份中间权重，按实际分配块释放 **45.47 GiB**；文件系统可用空间从约444 MiB增至 **45.90 GiB**，df为46G可用、98%使用率。

|旧实验目录（相对semantic_resolution_v1）|删除份数|释放GiB|
|---|---:|---:|
|oldA4_v4init_noca_ablation_20261022_v1/NoCA/train|3|6.46|
|oldA4_v4init_noca_ablation_20261022_v1/V4init/train|3|6.61|
|point1024_fmquery_2ctx4novel_full4_20261021_v2_iofix_node13_8gpu|3|6.47|
|point1024_isoadapter_noca_full4_20261021_v1_8gpu|3|6.43|
|single16_L6L12|9|19.50|

四组旧4epoch实验仅删除epoch1/2/3。single16_L6L12删除epoch1/2/3/5/6/7/9/10/11，保留实际存在的epoch4/8/12及latest/initial。没有声称single16训练已完成或保留不存在的epoch16文件。

旧A4/V4init联合配方和旧P1024配方未采用本轮geometry LR分组，历史证据见codex_handoff_olda4_v4init_noca_20261022.md第228行；本轮只删除非里程碑中间权重，旧实验仍可用最终模型复现正式评测。

保留内容：当前三组V4init隔离运行的全部文件、Baseline续训5500步latest、三组4epoch最终权重、共同零步、旧A4及旧实验最终权重、代码快照、配置、日志、TensorBoard、评测指标、完成收据、逐窗评测导出。此次没有删除任何评测图片/数组。

清理前仅一次查询发现71165/71166/71167空间分支相关任务仍排队；仅为保护其输入检查71165启动路径及源码，确认引用当前Baseline epoch4。spatial_pair_20261023_v1整个目录和源文件均保留，无作业取消/重提/监控。旧71031/71033/71035失败状态不因释放空间而自动续训，8epoch评测仍无结果。

删除前扫描相关10个实验族中configuration/source_configuration引用（排除evaluation/probes/cache等大产物目录），共记录104个路径值；不声称做了全工作区所有文本的穷举引用证明。候选均未被这些配置引用，且nlink=1。明确目录白名单、文件名/epoch限制，并逐文件核验设备/inode/size/mtime/nlink；先落盘审计日志再unlink，完成后检查26项保留checkpoint路径身份未变。无通配整目录删除。

计划、逐项删除日志、完成收据及执行脚本：`/space/mawb/ssst/group_plus/storage_cleanup_20261023_v1/`。其中PLAN.json保留21个被删路径及原stat，DELETION_LOG.jsonl记录每次unlink，COMPLETE.json记录实际前后空间。被删中间权重没有备份；保留最终权重和既有训练/评测记录。

本轮没有提交或恢复训练，没有修改任何冻结执行快照。当前空间是即时值，其他并行工作可能继续消耗；不保证未来三组长任务的完整容量预算。
