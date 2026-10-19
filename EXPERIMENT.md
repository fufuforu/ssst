# Concat共享Decoder/GFM

本分支对应训练作业 **67973**，架构入口 `--architecture concat_shared`。

1024个重建token + N个超点理解token + 2个stuff token拼接，共享12层1024维Decoder和一套GFM Adapter；仅前1024个token生成GS。

官方128评测作业 **68050**，依赖训练成功；具体节点和历史状态见提交记录。

| 训练作业 | 架构注册名 | 对应分支 |
|---|---|---|
| 67952 | `dual_point1024` | [vggt-point1024-dual-full4-20261019](https://github.com/fufuforu/ssst/tree/vggt-point1024-dual-full4-20261019) |
| 67973 | `concat_shared` | [vggt-point1024-concat-shared-full4-20261019](https://github.com/fufuforu/ssst/tree/vggt-point1024-concat-shared-full4-20261019) |
| 68051 | `recon_then_object` | [vggt-point1024-recon-then-object-full4-20261019](https://github.com/fufuforu/ssst/tree/vggt-point1024-recon-then-object-full4-20261019) |

## 源码和运行记录

- 选中模型：[模型源码](tokengs/models/concat_shared_decoder.py)。
- 实际执行快照：`/space/mawb/ssst_concat_shared_full_code_20261019`。
- 原始基础提交：`58e86fa3b17ea044edbca46906546417c825102e`，来源分支 `vggt-overfit-diagnostics-v1`。
- 本分支精确源码提交：`4c524dd3054bd55dbee1adc9e402b9c8a50a8509`；后续提交只补充文档/运行元数据，不改上述源码。
- [发布来源与逐文件SHA](repro/concat_shared/PUBLISH_PROVENANCE.json)、[原执行清单](SOURCE_MANIFEST.json)。
- [该组原始提交记录](docs/concat_shared_full_submission_20261019.md)、[最新评测和择空节点调度记录](docs/point1024_evaluation_and_flexible_scheduling_20261019.md)。

训练预算为A full4：1191场景/8337窗口，4epoch、4172更新、batch8、LR1e-4、LR warmup100、理解权重.05及1043步ramp。
1024个重建锚点各生成64个GS，总65536个；理解thing Query数量由超点决定，并不是1024个实例Query。

## 复现边界

此分支以实际训练快照为准，模型、公共core、注册和trainer等清单文件逐字节保留；不是把当前混合工作区直接提交。
公共注册入口依赖的历史evidence/超点细化类也保留原样，属于运行依赖，不是本分支新加的实验组；未混入它们的启动流水线或数据产物。
原始快照文档可能写于提交前，当前作业映射以本页及最新调度记录为准。

原集群脚本包含固定绝对路径、原提交SHA及防重复门禁，按原样归档。
根SOURCE_MANIFEST里的base_git_sha记录真实训练起点，不应改成发布提交SHA。
继续现有实验应使用原集群快照和run；不要从新GitHub checkout直接重提原作业或绕过恢复校验。

`repro/shared_control_20261019`归档两项官方评测及68051灵活调度实际使用的独立控制包。
评分辅助代码只兼容缺少recipe字段时用architecture作为结果标签；官方评测器和数值算法不变。
该目录是执行记录，不应再次运行其中的提交程序。

原CPU检查13项的结果保存在验证记录中。本次发布再次核验SHA、Python/shell语法和Git差异；没有新增GPU验证或查询集群进度。
提交/调度记录为2026-10-19当次历史状态，不代表当前实时状态。
模型权重、数据集和新增训练日志没有上传；相应位置及SHA见运行记录。
