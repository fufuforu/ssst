# A 有界几何更新与 Unified 软几何读取：2026-10-18

本轮实现用户指定的两项独立改动，没有提交训练、推理或评测作业，也没有恢复或重复旧训练。原版 A、Unified、B 和串行模型的入口、权重结构与运行快照保留。

## A：`--architecture dual_bounded`

架构标识：`OMEGA_3D_SUPERPOINT_DUAL_BOUNDED_GEOMETRY_V1`。

原版更新为 `mu += head_mu(h)`、`rho += head_rho(h)`，半径为 `epsilon + softplus(rho)`。新版保持原头、参数名称和形状，仅替换每层几何更新的解释：

```text
p = softplus(rho)
r = epsilon + p
mu_next = mu + (0.1 / sqrt(3)) * r * tanh(head_mu(h))
p_next = p * exp(0.1 * tanh(head_rho(h)))
rho_next = inverse_softplus(p_next)
```

每层中心位移的欧氏长度不超过 `0.1*r`。可学习半径部分每层缩放范围为 `[exp(-0.1), exp(0.1)]`，即约 `[0.905,1.105]`；总半径也处于这个相对界内，且保留原 epsilon 下限。零输出保持原几何；数值逆变换允许浮点舍入。局部尺度决定位移幅度，不再直接加一个与半径无关的绝对偏移。尺度等变性受固定 epsilon 下限影响。

两组超点 Query、GFM 两个分支、1024/256 宽度、12 层及 L6/8/10/12 理解交互、Write=0、Gaussian 子偏移、渲染与 Mask 读出、损失和梯度策略均沿用 A。没有顺带约束 Child Gaussian 偏移，因此本改动不能保证所有 Gaussian 都留在父 Anchor 的局部支撑内。

## Unified：`--architecture unified_ray_bias`

架构标识：`OMEGA_3D_SUPERPOINT_UNIFIED_RAY_BIAS_V1`。

保留一个共享 Query 序列、原共享 Decoder、自注意力、原有有界几何更新与重建/理解头。仅在每层图像 Cross-Attention 的 logit 中加入：

```text
geo_ij = clamp(-0.5 * distance(mu_i, ray_j)^2 / (2*r_i)^2, -20, 0)
bias_ij = softplus(gate_raw[layer]) * geo_ij
```

每层新增一个可学习标量，12 层时总计新增 12 个参数；初始化 `gate_raw=-4.6`，系数约 0.01，初始最大惩罚约 0.2 logit。门控归入共享参数，训练日志新增 `ray_bias_gate_mean`。几何距离在 FP32 中计算，带 `1e-8` 数值带宽下限。偏置有限，没有硬裁剪可见区域或屏蔽远处信息。

射线仅由当前上下文 Omega 预测的相机生成，不用 GT 相机。将 256 图像坐标的内参缩放到 GFM 原生 32×32 网格，在特征单元中心生成射线，并严格按视图→行→列顺序与 K/V 对齐。距离沿用已有 Plücker 点到直线实现，不额外判断前后遮挡。

第 1 层使用超点初始位置/半径；第 n+1 层使用第 n 层更新后的几何。checkpoint 将该层几何作为输入，并绑定该层 block 和索引，避免反向重算错误引用最后一层状态。

## 使用与实验边界

统一训练脚本已接入上述两个入口；官方导出与已有诊断通过模型注册表自动识别新架构，官方评测协议没有改动。旧架构默认行为没有切换。

本轮尚未选择新训练的初始化、预算或资源。不将新架构伪装成原 epoch8 的无改动续训：现有严格架构检查仍拒绝跨架构继续 Adam/RNG；需要在后续明确实验方案后准备独立初始化收据和快照。当前实现支持新架构各自的新训练、同架构零步初始化及同代码同配置断点恢复；没有放宽旧 continuation 策略。

验证范围：新增 CPU 测试覆盖 A 的极端输入边界、零更新、尺度和梯度，Unified 的射线顺序、软偏置尺度/梯度、零门控原版等价、逐层几何使用及 checkpoint 前反向一致性；另运行原版结构回归。GPU/CUDA、多卡同步与真实资产训练尚未验证，不能据 CPU 通过宣称指标改善。

实际结果：43 项 CPU 测试全部通过（运行 50.558 秒，不含环境导入耗时）：`tests.test_geometry_controlled_queries`、`tests.test_superpoint_structural_variants`、`tests.test_unified_query_v1`、`tests.test_serial_query`、`tests.test_serial_residual_query`。使用 tokengs Python 环境、OMP/MKL 线程数为 1；控制节点旧 CUDA 驱动警告保留，测试未使用 CUDA。日志保存至上述 maintenance 目录的 `geometry_controls_regression.log`。

## 实验文件清理

近期 `unified_query_training_v1` 目录实际占用约 31GB（按 du 统计，存在硬链接，不能简单累加文件大小）。本轮仅清理 12 个旧 Unified/结构/串行代码快照中未被 Git 跟踪的 Python 字节码：1087 个 `.pyc`，逻辑大小 19,500,077 字节，约 18.6MiB。文件均为单硬链接，实际释放量仍受文件系统影响。

逐文件清理收据：`/space/mawb/ssst/group_plus/unified_query_training_v1/maintenance_20261018/python_cache_cleanup.json`。

保留全部模型检查点、Adam/RNG、正式结果、逐场景诊断、可视化、日志和源码快照。旧实验不能仅因分数低就判定无用；epoch5/7 等独有状态也无法从汇总表恢复。后续若需要显著释放空间，优先单独核对失败零步副本和可由固定权重重新生成的中间导出，再确认引用与硬链接关系；不直接删除历史脚本及唯一模型。
