# 理解梯度诊断：已测量结论与新短对照

后续两个短对照和自动评测均已完成；完成后的数值、训练记录审计及联合任务判断见[两个 adapter 短对照结果](vggt_adapter_gradient_results_20261012.md)。下文保留本记录形成时的测量及提交状态。

零更新分析已经成功完成，发现了一个具体的梯度耦合问题：**VGGT adapter 接收的完整理解梯度主导了它的梯度方向，而主要来源是掩码监督，尤其是实例 Dice。** 在相同权重上将adapter理解梯度系数从1降到0.01，组合方向明显回到重建梯度附近。实际训练能否恢复重建质量，由已提交的两个短对照检验。

这是四个固定训练窗口上的梯度诊断，不是完整数据上的最终改进结论。分析没有optimizer，没有参数更新；理解模块保持冻结。完整训练允许理解模块适配，因此后续还需检查完整理解任务是否保留。

## 实测梯度

四GPU分别分析四窗口，参数梯度先准确平均，再按模块统计。联合组选择0/20/100/200/300更新；仅重建预测组选择100/200/300更新，共8个checkpoint条件、32次窗口分析。两个仅重建组中的理解梯度只在诊断时计算，不追溯为它们原训练中存在的目标。

| 联合 checkpoint 更新 | Adapter 理解/重建范数比 | cos(重建,理解) | cos(重建,原组合) | cos(重建,adapter系数0.01组合) |
|---|---:|---:|---:|---:|
| 0 | 5.024 | -0.1457 | 0.0538 | 0.9987 |
| 20 | 10.478 | -0.3444 | **-0.2564** | 0.9948 |
| 100 | 22.762 | -0.0289 | 0.0151 | 0.9748 |
| 200 | 16.121 | -0.4213 | **-0.3683** | 0.9879 |
| 300 | 12.007 | -0.0264 | 0.0568 | 0.9928 |

原组合为adapter的 `g_rec + g_under`。第20和200更新的原组合与重建方向相反，其余三个checkpoint接近正交。降为 `g_rec + 0.01*g_under` 后，五个checkpoint组合方向与重建梯度的余弦均超过0.97。这是局部裸梯度的结果，不等同于带Adam历史的实际参数更新或完整训练收益。

GS decoder 的原生理解梯度系数已是0.01，因此其有效理解/重建范数比为约0.12–0.24。不能把全模型未乘系数的U/R比直接解释为实际decoder更新完全被理解梯度主导。adapter保留object/memory原学习率1e-4，同时继承系数1；它作为重建特征桥梁受到的理解作用显著强于decoder。这是应独立测试的数学设置，尚未证明为工程实现错误。

第200更新的adapter加权组件梯度范数如下：

| 原生理解组件 | 梯度范数 |
|---|---:|
| 分类 | 0.00458 |
| thing BCE | 0.31104 |
| thing Dice | **2.50759** |
| stuff BCE | 0.46844 |
| stuff Dice | 1.09689 |

实例Dice在五个联合checkpoint都是adapter上最大的单个组件；分类项很小。组件之间可能相互抵消，范数不能相加为贡献百分比。理解readout对最终Gaussian输出的梯度主要涉及位置和尺度，RGB方向小得多，说明掩码监督能通过几何路径作用于重建；不能只看RGB颜色分支。

概率饱和也确实存在：scene0286_01在joint100的thing BCE像素梯度约2.0e6，对应概率约1e-12；joint200的stuff BCE在概率0处约3.0e6。但大像素梯度经过归一化、遮挡/零alpha及概率边界后，不一定变为同样大的参数梯度；这批checkpoint上，Dice仍然是adapter最大组件。因此不能仅靠像素值就认定原第218更新的1.34e4参数梯度尖峰来自BCE。本次没有精确重放该瞬时更新，也没有改BCE/Dice公式。

## 验证与已提交作业

- 原生/cache四窗口GS和两个损失误差均为0。初始组合范数6.068666205，与原训练首步6.068665981一致。
- 原五个加权理解项还原总损失；五项全局梯度之和相对总理解梯度的最大误差为3.72e-6。
- 64项CPU检查通过：17项诊断、28项原契约、19项分阶段/续训；新增5项检验adapter限定路由、原损失与梯度分解、八进程累积/裁剪/Adam参考及介入checkpoint契约。
- 两个系数都通过真实八卡两更新验证：理解损失非零、反馈开启、八rank梯度/参数一致、冻结参数和buffer不变、真实参数更新及checkpoint保存/严格恢复通过。每卡峰值约7.19GiB。smoke不计入正式训练。

| 阶段 | Slurm ID | 资源 | 执行情况/依赖 |
|---|---:|---|---|
| 零更新梯度分析 | 63220 | node17 / 四卡4090 | COMPLETED 0:0，14:54 |
| 两个系数真实更新验证 | 63221 | node17 / 八卡4090 | COMPLETED 0:0，05:55 |
| adapter理解系数0.01 | 63225 | node17 / 八卡4090 | 已开始；300更新/4800重复曝光 |
| adapter理解系数0 | 63226 | node17 / 八卡4090 | 等待63225完成；相同预算 |
| 两组共同评测 | 63227 | node17 / 四卡4090 | 等待63226完成 |

两个新对照均保留全部理解损失、反馈beta0.1和GS理解梯度系数0.01，只改变adapter接收的理解梯度系数。模型、源权重、固定四窗口、freeze、RNG策略、global16、原AdamW/clip、20更新warmup到原peak均与父拟合实验相同。这里系数0不等于删除理解损失：GS仍接收其0.01梯度；完整理解损失仍计算。

评测自动接续，复用固定SIU3R原生PNG协议，检查新初始PNG和全部GT PNG与旧基线逐字节一致，再对比旧joint/rec-only曲线。每条件16张图像，不拆分视图范围；输出重建指标和理解损失探针。完整mIoU/mAP/PQ保留情况必须由有独立留出窗口、理解模块可训练的后续验证确认。

读取首组20次正式更新/320曝光启动证明后，沿用此前方式停止轮询；第二组和评测继续按依赖执行，各组runtime独立自动生成启动证明。两个训练作业预计各约50–60分钟，合计约两小时加评测/排队。停止时的实际进度另存STOPPED_POLLING.json，不把排队组写成训练已确认。

## 正式 A/B 的独立状态

A已完成完整1860窗口/312场景的官方评测：Slurm62899成功，用时30:45，PSNR19.14；共同epoch6源为19.37。A的SSIM0.6660/LPIPS0.5778，对应源0.6744/0.5536，继续冻结训练两轮没有改善这三个重建指标。batch16续训与共同源训练的batch8设置不同，不能仅凭这行将下降归为某一个原因。B仍在node13训练，原后续四卡评测62847保留。

正式指标保持用户要求的单行13列格式，未实现的文本指标留空。这里的正式19.14与四窗口拟合24.35属于不同数据范围，不能混合比较。

## 产物及身份

执行代码已提交推送，所有新GPU作业使用同一不可变SHA：`fd92c4669e0c3939dad89169a22ce2612e4c20e8`。本结果文档的后续提交不会改动执行快照。原正式A/B快照、所有历史失败及原三组诊断文件保持在原位置。

- [完整梯度报告](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/probe/report.md)、[全局梯度JSON](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/probe/global_gradient_statistics.json)、[梯度曲线](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/probe/gradient_families.png)、[零更新回执](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/probe/COMPLETE.json)。
- [八卡验证汇总](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/gpu_validation_summary.json)、[训练/评测提交回执](/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/formal_submission.json)。
- [原正式A的13列结果](/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_batch16_v1/frozen/evaluation/final_continuation2_official/siu3r_table1.md)。
- 新训练目录：`/space/mawb/ssst/workspace_group_plus/vggt_adapter_gradient_diagnostic_v1/attempt01/{adapter_u001,adapter_u0}`；新共同评测目录：`/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1/attempts/attempt01/evaluation`。
