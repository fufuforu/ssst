# VGGT 在 ScanNet 重建与理解中的训练方式：2026-10-10 调研

用户授权本轮完成 epoch 4 官方评测并阅读相关工作。此文记录研究证据与建议；没有实施新的微调、冻结日程、损失或结构变更。现有 checkpoint 的评测保持 monitor_v1、原数据、原相机标定数学和原 13 列指标映射。

现有训练从 full1201 epoch 6 迁移 1377 个保留键，排除旧重建编码器/嵌入/KV 的 68 个键，新建 17 个适配层键。Gaussian decoder、理解与对象模块继承前 6 轮，再训练 8 轮；VGGT 始终冻结。不能将其描述为整个重建与理解模型从零只训练 8 轮，也不能将保留模块的训练积累等同于新特征路径已得到充分适配。

## 最相关的原始文献

| 工作与原始来源 | VGGT 的训练边界 | 与我们相关的证据及限制 |
|---|---|---|
| [InsTok3D / Scenes as Objects, Not Primitives，2606.29513v1](https://arxiv.org/pdf/2606.29513v1)，§4 Implementation details、§3.1、§4.3、附录 A.1 | 先以辅助 pixel-aligned Gaussian head 微调 VGGT 数轮，解决预训练与数据尺度不匹配；后续 tokenizer 使用冻结几何骨干。 | 确实支持“先适配再冻结”的方案。另有 RGB/pointmap 特征融合与点图初始化 anchors；将 DINO patch 14 改成 16，并输入 GT 内参。不是原始官方 VGGT 直接完全冻结。分割权重预热 1500 updates；2-view tokenizer 约 20h/4 A6000，前置适配少于 6h/4 H200，墙钟不可直接与我们的 3090 对齐。 |
| [Uni3R，2508.03643v4](https://arxiv.org/pdf/2508.03643v4)，§3.1–3.3、§4.1、Table 7、§6.4 | 重建网络由 VGGT 初始化并接受渲染训练；另有冻结 VGGT 教师提供 pointmap 几何正则。 | “冻结教师”不代表“学生骨干冻结”。point head 通过渲染监督适配度量尺度，另有内参嵌入。Table 7 全 Transformer 冻结变体 PSNR=5.49，完整模型=25.53；这是该结构内的消融，不能当成我们解冻后的增益预测。其 8-view 50 epochs、arbitrary-view 100 epochs 日程不属于我们的 2-view 8 epochs 同条件对照。 |
| [C3G，2512.04021v2](https://arxiv.org/pdf/2512.04021v2)，§4 Implementation details、§3.2 | Gaussian 阶段视觉编码器 LR=1e-6、decoder LR=1e-4，进行骨干微调；feature lifting 阶段只训练 value projections。 | 同样区分“先训练重建”与“之后冻结/限制更新读出”。论文主训练 450K steps，数据以 RE10K 为主；不能把其 ScanNet 评测当成相同 ScanNet 专项训练。官方 gaussian_head 配置 backbone_lr_multiplier=0.01，与上述 LR 边界相符；当前代码 batch preset 与论文 batch 不完全相同，未照搬。 |
| [IGGT，2510.22706v2](https://arxiv.org/pdf/2510.22706v2)，Appendix A Training details、§3 | VGGT 初始化，骨干 LR=1e-6，geometry/instance heads LR=1e-5；在 InsScene-15K 上联合微调。 | 支持 geometry 与 instance 表征共同适配；使用 pose/depth/pointmap 和实例监督，15K scenes、8 A800/2 天，既不是完全冻结，也不是与我们同监督条件。不是 Gaussian 新视图渲染模型，不能直接比较 SIU3R 表中的 PSNR/mAP。 |
| [AnySplat，2505.23716v2](https://arxiv.org/html/2505.23716v2)，§4.1、§4.4、Table 5 | 仅冻结 patch embedding，继续训练 VGGT 初始化的 Transformer 和头；使用冻结教师蒸馏。 | 是相机/深度/渲染共同适应的参照，主要是重建。冻结全部 Transformer=17.84 PSNR，完整模型=18.25，仅约 0.41dB；证明“冻结”对不同结构的影响差异很大，不支持断言冻结解释我们全部约 6dB 差距。其跳过大 loss 更新的规则不适用我们既定的完整曝光要求。 |

上述“先冻结后微调”措辞必须针对具体模块：视觉 patch tokenizer、跨视图 Transformer、几何教师、Gaussian decoder、对象 tokenizer 或 semantic readout。仅看到论文里的 frozen 字样不能判断整个 VGGT 永久冻结。

InsTok3D 的 sequential 消融是先训练 anchor decoder、再冻结它训练 group decoder，区别于“先适配 VGGT、然后冻结 VGGT、联合训练下游”。它的 sequential PSNR/AP=23.65/0.032，联合训练=25.11/0.193；不能用来否定前置 VGGT 适配，也不能据此主张冻结我们的 Gaussian decoder。

## 与当前实验的对应关系

1. **新增路径的适配预算。** full1201 初始化的重建 checkpoint 已到 step47500；新 VGGT 适配层只有本轮 8344 updates。相关工作普遍给予新的重建路径独立适配或允许骨干训练。这让前置重建适配成为有证据的候选，而非证明 8 epochs 一定不足。
2. **相机条件。** full1201 用 GT 位姿；本轮由冻结 VGGT 预测相机并作独立目标标定。InsTok3D 明示输入 GT 内参，Uni3R 也有内参嵌入。相机条件、尺度处理、训练特征同时变化，不能从最终指标单独识别 VGGT 或训练时长的因果影响。纯 Sim(3) 尺度拟合也不等于已消除各视角的相对位姿和内参误差。
3. **联合训练时序。** 本轮理解权重在 new_exposure=200 达到 1，global batch=8，即第 26 次更新起；beta 在 exposure=1000 达到 0.1，即第 126 次更新起。InsTok3D 是 1500 个 optimizer updates 的分割预热。我们的理解头已预训练，两种损失/注入不同，不能直接替换这些数值；但“新重建特征还在适配，理解约束已全部开启”值得以后单独验证。
4. **几何与外观输入。** InsTok3D 显式融合 RGB、pointmap 和几何特征；我们将 VGGT 多层特征经新 memory adapter 接给旧 decoder。这可能产生重建接口的适配困难；当前没有消融证明缺少某个分支是根因。
5. **资源与曝光。** H200/A6000/H100 的小时数、输入视图数、分辨率和 batch 都不同。跨论文不能以墙钟或 epoch 数直接对齐训练预算，也不能将不同数据、类别或 class-agnostic AP 直接填入 SIU3R 的类感知指标表。

rank 0 每 20 updates 的训练抽样（非验证、非全 rank 完整 epoch 平均）重建 loss 均值：epochs 1–8 分别为 0.08155、0.05322、0.04864、0.05018、0.04859、0.04556、0.04640、0.04810。训练抽样趋于平缓，但不能证明验证收敛，也不支持保证继续加轮数会追平。

## 当前建议与证据边界

先完成用户指定的 epoch4/epoch8 同窗口、同相机条件、同官方指标对比，报告全部 13 列，不拼接各 checkpoint 的最优指标。若 epoch8 的重建和理解明显改善，继续适配时间仍有直接支持；若改善很小、恶化或任务分化，单纯延长当前日程的依据变弱。只有两个 checkpoint 仍不能排除优化、泛化和几何教师造成的瓶颈。

下一轮研究更值得优先考虑**独立的 ScanNet 重建适配，再冻结 VGGT，训练原有下游模型**，而非直接解除本轮全部冻结约束并沿用旧 optimizer/scheduler。需要明确哪些 VGGT 模块更新、相机/深度头与更新后 aggregator 是否一致、训练期/评测期是否仍纯图像输入、原理解权重如何迁移，以及对几何和分割的回归检查。文献没有给出可直接适用本模型的学习率、轮数或增益保证。

不得把 GT 内参引入当前 generate、改成 GT target cameras、增加 pointmap/Chamfer/depth/normal loss、改 patch size 或 warm-up 来冒充现有锁定实验的延长训练。这些是另一个方案的变量，本文只评估其研究意义。

## 可追溯来源

原 PDF、文本抽取和读取的官方代码保存在任务独立目录：
`/space/mawb/ssst/group_plus/object_locus_frozen_vggt_posefree_v1/calibration_v2_monitor/research/vggt_scannet_20261010`。

官方代码固定在读取时的提交：

- [C3G](https://github.com/cvlab-kaist/C3G/tree/39242766ca5dffc45736113334d879a67d165228)：`config/training/gaussian_head.yaml`、`src/model/model_wrapper.py`、VGGT backbone。
- [Uni3R](https://github.com/HorizonRobotics/Uni3R/tree/4a5dd00b6737a95afd4c6b1760cad166cf17d3d4)：训练入口、README、模型与训练代码。
- [AnySplat](https://github.com/InternRobotics/AnySplat/tree/5f5e208a7dd57d52e43ea0d553a95eab526e8775)：`src/model/encoder/anysplat.py` 的教师/学生与 freeze_module 分支。

另一个 TMLR 几何/语义联合微调候选的 OpenReview PDF 遇到 HTTP403/浏览器挑战，未获得全文，未据搜索摘要写入结论。
