# 两个 adapter 短对照完成后的分析

两个新对照的训练及自动评测都已成功完成。仅将 adapter 接收的理解梯度系数从 1 改为 0.01 或 0，就在同四个训练窗口上恢复了重建拟合，同时降低了理解损失探针。结果支持优先验证 adapter 的多任务梯度路由，而不是沿原配方继续增加 epoch。它尚不能证明完整验证集的深度、语义和实例指标改善。

## 完成与核验

| 阶段 | Slurm ID | 状态 | 用时 |
|---|---:|---|---|
| adapter 理解梯度系数 0.01 | 63225 | COMPLETED / 0:0 | 54:07 |
| adapter 理解梯度系数 0 | 63226 | COMPLETED / 0:0 | 59:35 |
| 两组共同评测 | 63227 | COMPLETED / 0:0 | 04:50 |

每组完成 300 次正式更新、4800 次重复曝光，每个固定窗口曝光 1200 次；不是 4800 个不同样本。训练使用已推送不可变代码 `fd92c4669e0c3939dad89169a22ce2612e4c20e8`，共同 epoch6 权重和 adapted VGGT 资产保持原身份。

既有评测回执核验了八个 checkpoint 的源身份、模型键和形状、冻结张量、优化器时钟、八 rank RNG 和曝光计数。八个条件共 32 次窗口评测、128 张图像；新组初始渲染 PNG 与原联合基线逐字节相同，全部 GT PNG 与基线逐字节相同。沿用固定 SIU3R 官方 Evaluator、PNG 图像和 VGG LPIPS；评测零参数更新。

本次进一步审计四个预测几何对照各八 rank 的全部 300 条训练记录：更新数连续，曝光数一致，梯度范数和学习率跨 rank 一致，损失/范数有限，冻结状态始终不变。新组每次更新都保留理解损失权重 1、反馈 beta=0.1、GS 理解梯度系数 0.01。训练 stderr 为空，评测正常完成。

## 同四训练窗口的结果

所有组共同初始 PSNR 为 20.131，理解损失探针约为 0.28206。每个条件平均相同 16 张图像，不拆分视图范围。

| 对照 | PSNR 100↑ | PSNR 200↑ | PSNR 300↑ | 最终 SSIM↑ | 最终 LPIPS↓ | 最终理解损失探针↓ |
|---|---:|---:|---:|---:|---:|---:|
| 原联合：adapter 系数 1 | 19.573 | 18.900 | 20.467 | 0.6769 | 0.5744 | 0.30081 |
| 仅重建预测几何基线 | 23.179 | 23.963 | 24.348 | 0.7650 | 0.4382 | 0.27845 |
| 联合：adapter 系数 0.01 | 22.842 | 23.666 | 24.519 | 0.7618 | 0.4478 | 0.24258 |
| 联合：adapter 系数 0 | 22.909 | 23.894 | 24.657 | 0.7670 | 0.4438 | 0.24569 |

系数 0.01/0 相对原联合最终 PSNR 分别增加 4.053/4.190 dB，SSIM 增加 0.0849/0.0901，LPIPS 减少 0.1266/0.1306。四个场景的这三项图像指标都改善，收益不是某一个场景独自拉动的。

| 场景 | 原联合 PSNR | 系数 0.01 PSNR | 系数 0 PSNR | 仅重建 PSNR |
|---|---:|---:|---:|---:|
| scene0286_01 | 19.651 | 23.861 | 23.926 | 23.686 |
| scene0057_01 | 20.425 | 23.322 | 23.598 | 23.775 |
| scene0060_01 | 22.540 | 25.442 | 25.622 | 25.261 |
| scene0493_00 | 19.252 | 25.452 | 25.483 | 24.669 |

与仅重建相比，两个新组平均 PSNR 高 0.172/0.309 dB，理解损失探针也低约 0.036/0.033，但平均 LPIPS 略差。不能宣称它们在所有测量项上都胜过仅重建。

## 理解目标与优化稳定性

系数 0.01 和 0 的理解损失探针相对共同初始值分别下降约 14.0% 和 12.9%；原联合反而上升。这说明本诊断中恢复图像重建并未伴随理解目标值恶化，支持联合优化失衡的解释。它不是理解任务准确率的证明：Hungarian 匹配、不同掩码项和输出置信度均会影响损失；mIoU、mAP、PQ 必须实际评测。

| 对照 | 裁剪前梯度范数中位数 | 最大范数 | 最大值更新 |
|---|---:|---:|---:|
| 原联合系数 1 | 6.774 | 13401.041 | 218 |
| 仅重建 | 16.712 | 39.036 | 299 |
| adapter 系数 0.01 | 14.895 | 36.604 | 284 |
| adapter 系数 0 | 15.045 | 36.081 | 177 |

原联合的巨大参数梯度尖峰未在两个新对照中出现。四组所有更新仍触发原 clip=1，因此不能把“触发裁剪”本身当成失败信号。调节 adapter 梯度会同时改变共同裁剪和 Adam 历史，观察到的收益属于这条干预路径的整体效果，不能只解释为某一步裸梯度夹角改善。

此前零更新分析定位到 adapter 理解梯度范数为重建的 5–23 倍，实例 Dice 是最大的单项梯度来源。新实验提供了实际训练支持，但没有单独消融 Dice，因此仍不能把所有损害归因于 Dice 公式或认定为工程错误。

## 如何选择下一步

1. 当前最优先方向是给 VGGT adapter 单独设置理解梯度路由。两组都保留完整理解损失；系数 0 只阻断理解梯度进入 adapter，GS 仍接收原 0.01 理解梯度，反馈仍开启。未来若理解模块可训练，其自身完整理解梯度也应保留。
2. 不能只按 PSNR 选择系数 0。它比 0.01 多 0.138 dB，SSIM/LPIPS 更好，但 0.01 的理解损失探针更低；四窗口、单种子不足以确定正式联合任务的胜者。两者都应作为候选，以独立留出数据上的深度、重建、语义和实例指标选择。
3. 下一阶段应从同一个共同 epoch6 起点，使用相同预算，让理解模块参与训练，比较原系数 1、0.01、0，并同时测量全部已实现官方指标。候选可以先做固定小预算、多场景的训练/留出验证；源模型同范围结果作为参照。评价优先考虑重建和理解共同改善，明确报告取舍，不用任意加权总分掩盖某一任务退步。
4. 正式最终结果保持单行 13 列官方格式，两列 mIoUₜ 留空。本次没有测量 AbsRel、RMSE、mIoU、mAP、PQ，因此不能补造 13 列或把训练窗口的 24.66 当成完整验证集成绩，更不能与论文 25.96 直接比较。
5. 正式 B 仍在 node13 训练，后续官方评测 62847 等待 62845 完成。本次一次性读取到 760/1044 次续训更新，未修改 B、未重复提交任何训练或评测。先结合其完整结果判断 AA 解冻价值，再确定候选验证的冻结配置；不能同时改 AA 和 adapter 系数后把收益归于其中一个。

本次只检查并分析已完成任务，保存结果和审计证据；没有提交下一阶段训练，也没有继续轮询 B。

## 可追溯产物

- [原生汇总报告](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/analysis/report.md)、[原始数值](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/analysis/results.json)、[学习曲线 CSV](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/analysis/learning_curves.csv)。
- [学习曲线 PNG](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/analysis/learning_curves.png)、[SVG](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/analysis/learning_curves.svg)。
- [新完成审计与精确差值](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/analysis/verified_completion_analysis.json)、[checkpoint 校验](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/checkpoint_validation.json)、[评测完成回执](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation/COMPLETE.json)。
- [先前梯度归因与提交时状态](vggt_adapter_gradient_findings_20261012.md)。该记录保留提交时状态，本文件记录完成后的结论。
