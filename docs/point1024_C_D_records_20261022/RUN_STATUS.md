# C/D 消融：提交收据与后续交接

2026-10-22 09:03（北京时间）已提交。执行commit **526572b9657ade79865188d168db1028ee74e680**。

|组|Anchor CA|FM CA|单卡smoke|八卡四epoch训练|八卡官方1860评测|
|---|---|---|---|---|---|
|C|开|关|70691|70692，afterok:70691|70693，afterok:70692|
|D|关，保留纯几何路由|关|70694|70695，afterok:70694|70696，afterok:70695|

节点池12/13/14/17/18；smoke1卡、每步累积8个microbatch，恰好2个optimizer step；训练与评测各单节点8卡。
唯一一次有界状态核对时，六个作业全部PENDING。两个smoke原因为None（尚未分配），训练与评测为Dependency。调度接收不等于GPU前向/梯度/首步已验证。
**已停止轮询，未检查GPU首步或结果。不得根据这份历史PENDING记录重提作业。**

## 已完成准备

原配方、2+4训练清单、初始checkpoint SHA、8337窗/1191场景、全局8、4epoch/4172 updates、LR1e-4/100步warmup/cosine→1e-5、理解0.05/1043、全参数clip1、L6/L12均保留。
7项CPU回归通过，保留权重逐值一致；移除FM attention后Query残差LayerNorm与c/s更新保留；未做LR实验、几何LR改动或其它新实验。
CPU收据见 CPU_VALIDATION.json；完整实现说明见 ../point1024_C_D_nofmca_20261022.md。
参考已完成A/B，3090八卡训练约7–8小时、1860评测约1小时，另计排队与启动；本轮实际速度未验证。

## 用户通知完成后的检查

- 先看70691/70694的SMOKE_PASSED.json或JOB_FAILED.json。核对恰好2步、真实1024anchor/65536GS、2+4、梯度隔离、adapter/L12梯度、首步哈希和显存/耗时。
- 任一smoke失败则其afterok下游应被阻断；报告原因，不静默重试。
- 核对70692/70695的8rank日志、4172步、TRAINING_COMPLETE、最终checkpoint与4epoch身份；不要把smoke更新混入正式训练预算。
- 核对70693/70696的EVALUATION_VERIFIED.json、1860窗/312场景、输入RGB和GT与A/B一致、原官方target-all口径。
- 后完成的评测在自身comparison/尝试生成A/B/C/D表；若另一组完成收据当时尚未可见，完成检查时可只执行CPU汇总，不重跑导出/评测。
- 以C−A、D−B读去除FM CA的条件效果，以D−C、B−A读去除Anchor CA的条件效果；逐类PQ、mIoU/mAP及TP/FP/FN必须结合查看，单种子小幅PQ差异不当通用能力结论。

运行根目录：`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_nofmca_ablation_20261022_v1`。
执行快照：`/space/mawb/ssst_point1024_nofmca_ablation_code_20261022_v1`，保持上述HEAD不可变。
文档publication worktree：`/space/mawb/ssst_point1024_nofmca_publication_20261022_v1`。
GitHub分支：`experiments/point1024-no-fm-ca-CD-20261022`。
