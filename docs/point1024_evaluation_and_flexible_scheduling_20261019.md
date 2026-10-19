# 最新交接：补交两项正式评测；67976替换为68051择空节点八卡

更新时间：北京时间 **2026-10-19 14:22:39**。用户授权：补交67952、67973对应节点评测；67976允许13/14/17/18任一空出八卡的服务器。

## 当前有效作业映射

| 任务 | 作业号 | 节点/卡数 | 依赖与一次核对 |
|---|---:|---|---|
| A点云1024 full4训练 | 67952 | 13 / 8×3090 | 本轮初次只读核对RUNNING；之后未轮询训练进度 |
| A点云1024官方128评测 | **68049** | **13 / 1×3090** | **afterok:67952，PENDING (Dependency)** |
| Concat共享Decoder full4训练 | 67973 | 17 / 8×4090 | 本轮初次只读核对RUNNING；之后未轮询训练进度 |
| Concat共享Decoder官方128评测 | **68050** | **17 / 1×4090** | **afterok:67973，PENDING (Dependency)** |
| 原18节点限定的串行训练 | 67976 | 尚未分配GPU | **启动前已取消，最终核对CANCELLED，不得恢复或重提** |
| 完整重建→Object几何→语义，择空节点训练 | **68051** | **13/14/17/18任一单节点8卡** | **PENDING (None)**，3090/4090两分区通用GPU请求 |

上述状态是本轮有界核对的历史记录，不能直接当作之后的实时状态。评测与新训练尚无GPU首步/正式指标证据。
新评测每项4 CPU/64G/12小时上限，新训练32 CPU/192G/24小时上限，均不是预计耗时。
没有单独汇总作业，评测导出、评分和表格校验在各自同一个评测作业内完成。

## 评测流程和兼容修复

两训练当时均在运行，且没有既有评测提交记录、完成收据或导出收据，因此各新增一个afterok评测，无重复评测。
68049固定13服务器、68050固定17服务器；训练成功后且资源允许时执行，失败依赖自动终止，不读取中途checkpoint充当正式结果。

1. 使用**各自原始固定训练快照**的export_unified_query_checkpoint.py加载该run最终checkpoint_latest.pt。
2. 严格完整full4/4172更新门禁，导出同一固定heldout128；保持原图像、深度、SAM、分类/Mask/NMS/面积筛选及后处理。
3. 固定未修改的官方SIU3R评测器commit `8ea80166be76854f938e90521f1a5b688b755c87`，正式评分11个指标，两处文本mIoU留空。
4. 保留原评分器输出、原生无效值拒绝规则和同表old_C比较，再执行check_complete校验完整128场景与11项有限指标。

提交前发现两个新run/configuration.json没有旧评分脚本要求的recipe字段。
为避免评测导出后KeyError，独立控制包内的score_dual_stream_v1.py只作**结果标签兼容**：有recipe时沿用，否则使用architecture。
原官方评测器、指标算法、训练配置、训练代码快照均未修改；native_table_values等指标辅助函数AST逐值一致的回归检查通过。
没有调用旧compare_gfm_native_runs的架构前缀白名单，不用它误拒绝新的OMEGA_POINT1024架构。

68049提交记录：attempt21/EVALUATION_SUBMISSION.json，日志evaluation-slurm-68049.out。
68050提交记录：attempt22/EVALUATION_SUBMISSION.json，日志evaluation-slurm-68050.out。
评测开始后写EVALUATION_STARTED.json，失败写EVALUATION_FAILED.json；成功产物为evaluation/export_complete.json、
evaluation/metrics.json、evaluation/siu3r_table1.csv及run/COMPLETE.json。

## 67976为何更换作业号

原Slurm批脚本固定节点18，并在脚本中硬性校验18节点；直接改NodeList会导致在其他节点启动失败。
本轮先完成独立可审计启动器、源代码/初始文件SHA、官方评测器、8项调度/重复提交/标签兼容检查和Slurm --test-only预检。
然后在实际取消前再次确认67976为PENDING、无STARTED/任何训练checkpoint或完成收据；hold成功后再次确认未启动，才取消并提交单个替代作业68051。
最终一次有界核对确认旧67976为CANCELLED。没有中断任何已开始的训练，没有同时保留两个同组训练副本。

68051请求 `--partition=3090,4090 --nodes=1 --gres=gpu:8 --exclude=3dimage-11,3dimage-12`。
当前两分区节点为11/12/13/14/17/18，因此允许池恰为13/14/17/18；启动器再次核对四节点白名单。
不把四节点写成required --nodelist以免误要求多个节点；一台机器内8卡，由Slurm按资源和调度策略选择。
13是3090，14/17/18是4090，通用gpu:8同时兼容；没有每节点单独提交竞争作业、后台扫描或轮询。

**模型代码、664张量零步权重、RNG、loss、完整4epoch/4172更新/batch8、LR1e-4、100/1043 warmup均不变。**
仍执行原固定模型快照 `/space/mawb/ssst_recon_then_object_full_code_20261019`，只使用独立控制包中的灵活节点启动器。
零步文件SHA `f3cb086bf7cd96293dc2e65a2942e117c7426db64bfa8633b8dd8526c7603833` 未变；没有新增初始化或重新开始已执行的训练步。
用户已在后续答复中明确确认“先完成12层重建，再做几何→语义更新”，68051保持该方案。

运行目录保留原名称：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt23_recon_then_object_full_node18_8gpu`。名称包含node18是历史标签，不代表实际分配节点。
**最新训练ID以该目录ACTIVE_TRAINING_JOB.json为准：68051；原submission.json/67976仅为历史记录。**
调度变更记录FLEX_SUBMISSION.json、FLEX_SUBMISSION_REQUESTED.json，实际启动后FLEX_ALLOCATION.json记录真实节点/GPU类型。
新日志slurm-68051.out；后续检查TRAINING_COMPLETE、FIRST_GRADIENTS_rank00–07、FIRST_UPDATE_VERIFIED、checkpoint和八rank4172连续日志。
保留原664参数布局；Write=0的未用/零梯度参数不能机械按“全部参数必须改变/全有Adam状态”判失败。

## 独立控制包和接手规则

控制包：`/space/mawb/ssst_point1024_eval_flexible_control_20261019`。
SOURCE_MANIFEST.json冻结9个文件SHA；PREPARED_PLAN.json、VALIDATION_EVIDENCE.json、DISPATCH_RECEIPT.json、SUBMISSION_ONCE_CHECK.json记录全部动作。
8项有针对性的CPU控制检查通过，Python/shell语法和git diff --check通过，Slurm通用八卡test-only通过，官方评测器版本/src未改。
未修改任何原固定训练快照、原训练configuration、已有训练checkpoint或原提交收据；workspace新增调度脚本未commit/push。

本轮提交后只做一次有界状态核对，无日志等待，无后台监控。仅保留一个有效串行训练任务68051。
用户通知完成/检查后，依照68049/68050/68051和实际产物继续检查；不要恢复67976，不因旧文档PENDING重提或重训。
若仍在运行或等依赖，报告一次后停止；正式评测如失败先只读定位，不能绕过完整128/11指标门禁或用中途模型替代。

一次核对原文：

```text
68049|PENDING||(Dependency)|3090|gres/gpu:3090:1
68051|PENDING||(None)|3090,4090|gres/gpu:8
68050|PENDING||(Dependency)|4090|gres/gpu:4090:1
old67976=CANCELLED
```

**已停止轮询，等你通知完成后再检查。**
