# 三组四窗口拟合诊断：2026-10-12 评测结果

主要发现是：在共同初始权重、相同预测几何、相同训练预算下，去掉理解损失但保留反馈，PSNR 从 20.131 提升到 24.348；联合训练最终为 20.467，相差 3.881 dB。四个场景方向一致。GT 几何组最终为 23.611，没有提供平均优势。下一步优先隔离联合优化中的理解梯度，而不是直接增加训练轮数、重置 decoder 或更换 encoder。

这是四个**训练窗口**上的短预算拟合诊断，不是完整验证集的 SIU3R 13 列结果。这里的 24.348 不能直接与论文的 25.96 或历史完整验证集结果比较。正式 A/B 的完整评测另行进行。

## 已完成的任务与检查

| 任务 | Slurm ID | 状态/退出码 | 用时 | 正式更新 / 重复曝光 |
|---|---:|---|---|---|
| pred_joint：预测几何＋联合目标 | 63134 | COMPLETED / 0:0 | 55:08 | 300 / 4800 |
| pred_recon：预测几何＋仅重建目标 | 63135 | COMPLETED / 0:0 | 47:31 | 300 / 4800 |
| gt_recon：GT 几何＋仅重建目标 | 63136 | COMPLETED / 0:0 | 43:20 | 300 / 4800 |
| 十二个 checkpoint 评测 | 63219 | COMPLETED / 0:0 | 04:20 | 0 / 0 |

三组都从原来适配 2 轮＋冻结联合训练 4 轮的共同 epoch6 权重出发。训练保持 VGGT（包括 AA）、理解模块和反馈控制器参数冻结；训练 VGGTMemoryAdapter、GS tokens 和 GS decoder，反馈 beta=0.1 始终开启。两个仅重建组去掉的是理解目标及其梯度，没有关闭反馈。理解模块被冻结时，理解损失仍能通过固定模块向可训练 adapter 和 GS 模块传播梯度。

四窗口按已提交的 cohort 顺序固定选择，不按分数挑选；每窗口两张 context、两张 novel。每次更新 global batch=16，每窗口重复四次，300 更新合计 4800 次重复曝光，即每窗口 1200 次，不能解释为 4800 个不同训练样本。三组共同使用 20 更新 warmup 后保持原 peak LR 的诊断调度，GS 为 1e-5，adapter 为 1e-4；它不同于完整训练的长 warmup/cosine 调度。

已核实十二个 checkpoint 的完整 SHA、严格模型键和形状、更新数/曝光数、八 rank 计数及优化器时钟。三个初始模型与共同源模型一致，全部冻结非 VGGT 张量与源权重逐张量相等，外置 adapted VGGT 资产 SHA 相同。预测几何两组初始输出 PNG 逐字节相同；所有条件的 GT 图像逐字节相同；当前 provider RGB 与训练时哈希一致。原生与缓存 GS/损失重放验证通过。评测没有构造优化器，没有更新参数。

## 同四窗口、十六张图像的结果

每个 checkpoint 平均同样的十六张图像，调用固定版本、未修改的 SIU3R Evaluator，PSNR/SSIM/LPIPS 使用正式 PNG 协议和 VGG LPIPS，不拆分视图范围。

| 对照 | 初始 PSNR ↑ | 100 更新 | 200 更新 | 300 更新 | 最终 SSIM ↑ | 最终 LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 预测几何＋联合 | 20.131 | 19.573 | 18.900 | 20.467 | 0.6769 | 0.5744 |
| 预测几何＋仅重建 | 20.131 | 23.179 | 23.963 | **24.348** | **0.7650** | **0.4382** |
| GT 几何＋仅重建 | 19.514 | 22.161 | 23.842 | 23.611 | 0.7432 | 0.4663 |

仅重建预测组相对共同初始值提高 4.217 dB，100→200→300 更新继续上升，但增益从 0.784 减少到 0.384 dB。联合组仅提高 0.336 dB，SSIM 和 LPIPS 比初始值变差；在 100 和 200 更新时重建质量也下降，说明这不是简单的稳定、缓慢收敛曲线。固定首张 novel 帧的图像对照可见仅重建组的边界和物体细节改善，但仍有模糊。

| 场景 | 联合最终 PSNR | 仅重建最终 PSNR | GT 最终 PSNR | 去掉理解目标的差值 | 换 GT 几何的差值 |
|---|---:|---:|---:|---:|---:|
| scene0286_01 | 19.651 | 23.686 | 20.564 | +4.035 | -3.122 |
| scene0057_01 | 20.425 | 23.775 | 24.183 | +3.350 | +0.409 |
| scene0060_01 | 22.540 | 25.261 | 24.979 | +2.721 | -0.282 |
| scene0493_00 | 19.252 | 24.669 | 24.718 | +5.417 | +0.049 |

GT 平均差值为 -0.737 dB，主要由 scene0286_01 拉低，不能写成“GT 对所有场景都更差”。GT 组初始低 0.616 dB；其短预算结果还受源模型对预测相机的补偿、固定 VGGT 特征空间与 GT 射线配合影响。这个诊断替换了相机、内参和一致的监督/渲染射线，未替换成一个完整的 GT 几何特征教师，也未引入 GT 深度损失。因此当前结果降低了优先修相机的收益预期，但不能排除所有几何与坐标适配问题。

## 目标和梯度记录提供的线索

| 对照 | 初始→最终重建目标 ↓ | 初始→最终理解损失探针 ↓ | 裁剪前范数中位数 | 最大范数 |
|---|---|---|---:|---:|
| 预测几何＋联合 | 0.03962 → 0.03932 | 0.28206 → 0.30081 | 6.774 | 13401.041 |
| 预测几何＋仅重建 | 0.03962 → 0.02406 | 0.28206 → 0.27845 | 16.712 | 39.036 |
| GT 几何＋仅重建 | 0.04460 → 0.02830 | 0.28977 → 0.30911 | 9.291 | 22.259 |

理解探针对三组都调用原来的理解 readout 和损失，仅用于诊断。预测仅重建组并未在这个探针上变差；联合组反而更差。它不能替代理解任务的 mAP、PQ 或 mIoU，也不能证明完整训练取消理解损失会改善所有任务。

八 rank 的裁剪前范数逐次一致。三组每次更新都触发原 clip=1，因而“出现裁剪”本身不足以解释组间差异。联合组在第 204/218/238 次更新分别出现 8020/13401/5870 的尖峰，其他两组最大值为 39.036 和 22.259。这是需要追查的优化波动线索；现有日志没有分组 g_rec/g_under 的范数和夹角，不能直接断言每次都发生梯度冲突，更不能把 Adam 的实际更新简单等同于裁剪系数。

现有 `gc_combine` 给 reconstruction 家族使用 `g_rec + 0.01*g_under`，但 VGGTMemoryAdapter 被归入 object_memory，使用 `g_rec + 1*g_under`；adapter 的 peak LR 同时为 GS 的十倍。这是当前代码明确的数学设置，尚未证实为工程错误。去掉理解目标也会改变 Adam 历史和共同梯度裁剪；此对照识别的是理解目标/梯度路径对短预算拟合的影响，不能只归因于某一模块或某一系数。

浮点 RGB 与正式 PNG 的条件平均 PSNR 最大差异仅 0.002061 dB，远小于组间 3.881 dB 差距。渲染 alpha 最终均值为 0.995/0.988/0.982。当前效果不是 PNG 量化制造的收益，也没有显示平均覆盖率崩塌。

## 下一步优先级

1. 先在同样的小预算条件中隔离 **adapter 接收的理解梯度**：保留反馈及 GS 家族现有 0.01 系数，分别将 adapter 的理解梯度系数从 1 改为 0.01、0，并记录各家族 g_rec/g_under 的范数、夹角和裁剪前后贡献。这样能判断主要问题是否发生在新的 VGGT 特征桥接路径，避免一次改动全部模块。此处只是后续建议，本次未提交新训练。
2. 如果找到稳定方向，再在有独立留出窗口的短训练中检查重建和理解质量，之后才投入完整训练。诊断中的理解网络被冻结；正式 A/B 的理解网络可以适配，因此这里不能直接证明正式联合训练也有同样的严重程度。
3. 现有 decoder 和冻结 VGGT 已能在四窗口拟合到 24.348，说明“换了特征空间后模型必然卡在 19”不成立，但不能证明其初始化最优或其泛化达到 24.348。decoder 从零初始化、AA 解冻以及单 encoder 对照仍有价值，依据本次证据优先级排在梯度路径隔离之后。

继续保留正在运行的正式 A/B：A=62898（node11），B=62845（node13）；后续完整评测=62899/62847，读取时仍在等待各自训练依赖。没有修改、重启或重复提交这些任务。完整结果仍采用用户要求的 SIU3R 13 列表格，并留空未实现的文本指标。

## 可追溯产物

- 训练代码 SHA：`bff606fc721f83a4040e4880bc082487de23b8f3`；评测代码 SHA：`f144e88b876f935e911aa0c05d06612364ac3052`，均为提交推送后的不可变快照。
- 官方 SIU3R SHA：`8ea80166be76854f938e90521f1a5b688b755c87`，使用已有 `.venv_gpu_v4` 和离线 VGG LPIPS 资产。
- 共同源 checkpoint SHA：`01d49b092c8dfc2e02f15b5216e5d6ed1cfd88fae2cf693ee69334e45a9410ba`。
- 原始训练目录：`/space/mawb/ssst/workspace_group_plus/vggt_overfit_diagnostic_v1/{pred_joint,pred_recon,gt_recon}`。
- 本次评测目录：[attempt01](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01)。旧失败与停止轮询记录保留在原位置。
- [完成回执](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/COMPLETE.json)、[checkpoint 校验](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/checkpoint_validation.json)。
- [完整数值 JSON](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/analysis/results.json)、[学习曲线 CSV](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/analysis/learning_curves.csv)、[逐场景 CSV](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/analysis/per_scene.csv)、[原训练梯度日志审计](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/analysis/training_gradient_log_audit.json)。
- [学习曲线](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/analysis/learning_curves.png)、[SVG 曲线](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/analysis/learning_curves.svg)、[固定帧图像对照](/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01/analysis/fixed_frame_comparison.png)。
