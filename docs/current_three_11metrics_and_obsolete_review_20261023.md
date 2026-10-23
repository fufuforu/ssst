# 当前三组11项指标与旧实验清理范围审查（2026-10-23）

用户询问没有修改几何头LR的旧实验是否都是错误实验、能否全部清除，并要求当前三组完整11项指标。本次没有提交作业，也没有批量删除旧最终权重；仅生成指标导出、当前主线登记和可审查清理候选。

几何头未单独降LR是一种不同优化配方，不足以将全部旧结果定性为错误。若原目的要求复现早期V4的几何LR1e-5，采用统一1e-4就不符合该复现条件；但它们仍能作为历史对照。当前重建提升伴随初始化、隔离、监督、loss权重等多项改变，不能把所有效果差异单独归因于几何LR。旧记录可退出当前主线，指标/配置/日志值得保留。

## 当前三组正式结果（4epoch）

11项数值来自各自eval/evaluation/metrics.json的official_segmentation.all；完成收据PASSED，固定1860窗/312场景、epoch4/4172更新、评测更新0。Target为target-all6，包含context。理解六项百分数，其余原始数值。当前8epoch续训尚未产生正式8epoch结果；不将训练日志PSNR当验证指标。

|配置|PSNR ↑|SSIM ↑|LPIPS ↓|AbsRel ↓|RMSE ↓|ctx mIoU ↑|ctx PQ ↑|ctx mAP ↑|tgt mIoU ↑|tgt PQ ↑|tgt mAP ↑|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Baseline_U1_2ctx2|23.052|0.7685|0.3856|0.05273|0.17646|58.17|68.42|24.54|57.78|66.06|23.14|
|U005_2ctx2|23.197|0.7718|0.3794|0.05239|0.17452|57.26|67.75|23.70|56.98|65.59|22.23|
|U1_2ctx4|22.978|0.7640|0.3913|0.05187|0.17333|57.45|68.71|24.50|57.23|66.32|23.05|


完整精度CSV：[current_three_epoch4_11metrics.csv](obsolete_retirement_review_20261023_records/current_three_epoch4_11metrics.csv)。11个指标是5项重建/深度和6项理解；另有配置标识列。三组分别对应当前71192→71193、71194→71195、71188→71189，仅做一次队列观察用于确认活动输入保护；本次未持续检查训练日志。

## 为什么不能直接删除整棵旧目录

- 当前三组仍引用旧V4init初始化零步，比如oldA4_v4init_noca_ablation_20261022_v1/V4init/train/checkpoint_initial.pt；其bf5b5325...权重实际是迁移初始化来源，不能因所在目录属于旧实验就删除。
- 当前评测仍以old_A4_official1860_20261021_node12_8gpu/evaluation/export_complete.json及gt_reference_manifest.jsonl核对输入与GT。代码见v4init_extend8_pipeline.py的finish→signatures(REFERENCE)/gt(REFERENCE)。这些参考记录必须保留。
- 当前三组全部运行目录、恢复源，以及活动spatial_pair实验和其输入均排除；冻结代码快照及数据集不属于清理候选，旧快照还存在Git worktree关联，不能直接整目录删除。

## 已准备的checkpoint候选（未删除）

在semantic_resolution_v1和unified_query_training_v1两个实验族盘点旧checkpoint：候选 **102 个路径、84 个独立inode，实际已分配约68.00 GiB**。已排除所有当前V4init运行；扫描当前配置/metadata/收据及spatial输入的文件引用，按inode保护初始化和恢复依赖；另排除存在审查范围外硬链接的候选，保留原始旧A4初始化源。共21个旧checkpoint路径保留，当前引用/参考文件登记37条。

候选包含旧最终权重，意味着永久失去这些旧模型再次推理/恢复训练的能力；本次未执行这批不可恢复删除。这里的分类依据是旧实验退出当前主线，不声称所有候选都已经逐项证明几何LR错误。四种较近旧执行SHA（5335430b/046d08d5/21a85947/526572b9）的trainer已核实为只按weight decay分组、所有组共享args.learning_rate；其他早期模型有不同架构/目的，未将它们自动判为同一LR错误。

候选逐文件路径、大小、device/inode/nlink/mtime及保留理由见CHECKPOINT_RETIREMENT_REVIEW_PLAN.json。JSON状态REVIEW_ONLY_NOT_DELETED；没有更改或删除其中任何权重。此计划仅覆盖checkpoint，尚未对旧逐窗RGB/深度/分割导出做完整容量盘点，也没有删除这些原始输出。

建议的批准范围：按候选清单删除旧checkpoint，保留所有指标/配置/日志及列出的初始化/评测/活动实验依赖；raw评测输出另按依赖审查，不批准整目录无差别删除。这样能把当前三组作为后续主线而保留历史对照证据。

工作文件：/space/mawb/ssst/group_plus/obsolete_experiment_retirement_20261023_v1。
本次未重提、取消或修改任何训练/评测，已停止状态轮询。
