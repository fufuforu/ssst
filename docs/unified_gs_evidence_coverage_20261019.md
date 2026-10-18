# GS evidence 位置梯度修正与实例覆盖先验

基于 HEAD `b6b949b` 及上一轮未提交 evidence 实现。本轮仍不训练、不提交作业、不创建 attempt。旧证据报告 `unified_gs_evidence_validation_20261019.json` 保留，新的结果另存 `unified_gs_evidence_coverage_validation_20261019.json`。

## P0：采样位置 detach

`sample_gs_evidence` 的几何读取仅把 `mu = gaussians[0, :, :3]` 改为 `.detach()`。投影、半像素归一化、深度门控、双视图平均和无效置零均不变。它切断 mask loss 经 grid_sample 采样坐标移动 GS 去追逐特征的新增 shortcut；GFM/projection 的特征学习路径保留。既有 local/renderer 的几何梯度不受此 detach 影响，覆盖项也通过原 alpha 渲染路径作用于几何。

采样函数增加可选 `return_zero_fraction=True`，返回 `(evidence, python_float)`；默认仍返回 tensor，以保持原有诊断调用兼容。core forward 只调用一次采样，并在该次调用中以 `(evidence.abs().sum(-1) == 0).float().mean()` 计算行全零比例；统计在 no_grad 内转 float，保存到 prediction。模型 step_loss 调用原父类后只把这个已有 float 写入 metrics，不再次采样或重新计算该比例，不依赖类别/检测结果。

`evidence_zero_fraction > 0.25` 可用于后续观察证据失效范围，本实现只记录指标，没有添加自动停止或其它策略。行全零比例和元素零比例不同，单元测试专门构造二者不同的数据验证计算口径。

## 为什么用覆盖先验

alpha 是几何覆盖度，不是实例边界指示量。相邻物体的边界两侧 alpha 都可能接近 1，`|∇alpha|` 主要响应物体与未覆盖背景的轮廓，无法普遍提供相邻实例边界。共面相邻物体也未必存在深度跳变。因此本变体使用 GT 实例内部的覆盖约束：要求实例内部像素被几何覆盖，在无纹理、光度损失约束较弱的位置补充监督；没有将它命名或解释为边界对齐，也不声称它能保证正确实例划分。

新入口 `unified_gs_evidence_coverage`，身份 `OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_GS_EVIDENCE_COVERAGE_V1`，继承 evidence 模型，不增加模块、参数或 buffer。`CoverageRecipe` 为冻结 dataclass：loss_weight=0.1、erode_px=1、views=2、max_inside_fraction_guard=1.0；完整记录在 checkpoint metadata 的 `coverage_recipe` 中。

只使用父类已产生的 `pred['alpha']` 与 `pred['final_targets']`。对有效 GT 实例掩码先取并集，再按题定负 max_pool3×3 方法腐蚀一像素；它不是逐实例单独腐蚀，相邻实例并集内的连接处不会凭空产生缝隙。全图 True 的边缘处理严格沿用给定 max_pool padding 语义，3×3 独立 True 块只留下中心。零实例 K=0、全 False、有效性过滤后为空均返回零覆盖项、`loss_coverage_applied=False`。

非空时计算内部 alpha 到 1 的 BCE，先 clamp 到 [1e-6,1-1e-6]。总损失：

`loss_recon + understanding_weight * loss_understanding + 0.1 * loss_coverage`

**覆盖项不乘 understanding_weight，也不随理解任务 warmup 缩放**，因为它是几何覆盖正则。原共享理解损失、系数和归一化完全不变。新增日志：loss_coverage、loss_coverage_weighted、loss_coverage_applied、alpha_inside_mean、coverage_inside_pixels，统计值全部脱离梯度；只有加入 loss 的 cov 保留 alpha 的梯度。不重新生成 GT、不重新渲染。

## 验证

44 项测试一次全部通过：原 30 项（12 evidence + 18 原 Unified）与新增 14 项。原测试中“step_loss 与父类是同一函数”的身份断言，因任务明确要求 override 注入指标，调整为 generate 仍直接继承的断言；其余回归检查保留，新增测试验证 step_loss 只调用一次父类并保持原 loss 不变。

新增检查覆盖：detach 前后逐值相等、旧位置梯度非零而新梯度 None、GFM 梯度有限非零、行级零比例与元素零比例区分、一次采样与 float 转发、全 True/全 False/K=0/valid 全 False、覆盖梯度不受 understanding_weight 影响、腐蚀及 valid 过滤、相同 seed 全 state_dict 与 RNG 相等、配方 metadata、注册与匹配约束拒绝、错误形状明确拒绝。

真实 CPU 同窗口 `scene0279_02`、context 31/57、seed42：

| 项目 | 结果 |
| --- | --- |
| evidence shape | `[1,3520,256]`，55 Query / 3520 GS |
| 均值 | `-0.0072211166843771935` |
| 总体方差 | `0.21637889742851257` |
| evidence_zero_fraction（行全零） | `0.10085227340459824`，约 10.0852% |
| 元素零占比 | `0.10085227340459824`（该窗口恰好相同） |
| 与旧记录统计 | 逐值完全相同，包括 RGB hash/GS 数量/shape |
| detach 前后完整张量 | torch.equal=True，SHA256 相同 |
| detach 后对真实 GS 位置的梯度 | None |
| detach 前位置梯度最大绝对值 | `3915.02001953125`，非零 |
| detach 后 GFM 梯度 | 有限且非零 |
| 两变体全部 state_dict | 2026 键，集合相同、所有张量逐值相同、不一致清单 `[]` |
| 两变体可训练部分 | 301 张量 / 14,681,028 参数，新增 0 |
| attempt02 初始化对照 | 300/301 共有键完全相同；唯一差异 `core.child_feature.0.weight` |

前后 evidence 完整张量 SHA256 均为 `a5d75e3b99f36ee97fd5690a8c57fdecfbc505559076e64d056e49b9a1ffba59`。CPU 真实资产验证约 384.4 秒，无训练更新。

完整配置对照的冻结主干复用及 RNG 重放方式见下文；全部可训练张量独立初始化后比较，不从一个变体向另一个复制可训练权重。单元测试还验证独立 seed42 构造两模型的所有张量/RNG 一致。

真实 CPU 前向范围：冻结 Omega/SAM、encode_context、core/GS、evidence/membership；没有执行真实 CUDA renderer 或实际覆盖变体训练。覆盖计算的损失与 alpha 梯度由同尺寸 CPU fixtures 验证。所有位置/深度/特征/GT 网格形状显式断言，没有静默 resize 或维度适配。

## 约束与复现

两个入口都保持 UnifiedQueryConfig 默认配置并锁定 source_many_to_one、support_matching=true、support_purity=0.5，不定义 default_config。只有新 evidence 文件承载模型行为变化；原 unified_query_core.py、unified_query_loss.py、canonical_recon_models.py、direct_instance_attributes.py、unified_superpoints.py 与基线逐字一致。训练文件只有架构 choices 增加，注册文件只有新类 import/映射；预算、优化器、seed、数据顺序和已有损失配置不变。没有改任何历史 attempt、快照或正式评测产物。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /space/mawb/anaconda3/envs/tokengs/bin/python -m pytest -q tests/test_unified_gs_evidence.py tests/test_unified_gs_evidence_coverage.py tests/test_unified_query_v1.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /space/mawb/anaconda3/envs/tokengs/bin/python scripts/validate_unified_gs_evidence.py --output /tmp/unified_gs_evidence_coverage_real.json
```

真实验证脚本会同时核对 attempt02 初始 checkpoint、旧记录统计、detach 前后完整 evidence 张量及 SHA256、真实坐标上的位置/GFM 梯度、两个完整配置变体的参数一致性。为避免重复分配大型冻结主干，完整配置对照复用真实冻结资产，重放首次构造冻结资产后的 RNG 状态，独立初始化两份可训练模块；单元测试另外以 seed42 独立构造两套 mock 模型比较全部 state_dict/RNG。
