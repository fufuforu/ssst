# attempt02 单因素 GS 图像证据：实现与只读验证

后续 P0 修正（采样位置 detach）、零占比指标与无参数覆盖变体见 `unified_gs_evidence_coverage_20261019.md`。本页数值与原验证 JSON 保留为 detach 前记录；新验证逐值核对前向数值不变。

基于 `b6b949b557e3247defd6519c058de700dc1c96ca`，新增入口 `unified_gs_evidence`，模型身份 `OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_GS_EVIDENCE_V1`。只扩展原 Unified 的 child_feature 输入：`cat(parent, index, local, evidence)`，维度从 275 增为 531，隐藏/输出维度仍为 256。新增参数仅第一层多出的 65,536 个权重。

## 采样路径

**采用显式投影 + grid_sample。** 已检查并在单元测试中复现首选 `lift_features` 的不兼容：它是 RendererTranspose 渲染转置算子，要求 `grad_outputs` 与 renderer 输出空间尺寸一致；原始 GFM 为 `[1,2,256,32,32]`，实际 renderer 输出为 256×256，直接调用发生 shape mismatch。签名和通道数兼容，但空间尺寸不兼容；没有通过上采样或改 renderer 绕过问题。

沿用原版已投影的 GFM memory，恢复二维栅格；每个 GS 的世界位置经过各视图 inverse(c2w)、K 投影，按 256 分辨率及半像素公式构造采样坐标，`align_corners=False`、`padding_mode='border'` 双线性采样原生 GFM 和 Omega depth。只平均图内、正深度且深度误差 ≤5% 的视图；两视图均无效时置零。z=0 只用安全分母防止除零，并始终判无效。无可学习 evidence 生成器、通道转换、隐藏 resize、额外损失或聚类。

GS、memory、投影/采样中间张量、evidence 和 child 输入组成项均有显式形状断言；evidence 有有限性断言。模型要求 dim=256、64 children，错误配置显式拒绝。

## 初始张量一致性

初始化在原 child_feature 构造位置先消耗与原版完全相同的初始化抽样，再用 `torch.random.fork_rng(devices=[])` 构造扩展层，并复制原第一层 bias 和第二层 weight/bias。仅 fork_rng 而不复制这三个共有张量不能满足逐值一致要求。随后构造 stuff_feature；不存在读入旧 checkpoint 权重来冒充初始化一致性的操作。

真实 seed42 模型（加载真实冻结 Omega/SAM）与 attempt02 `checkpoint_untrained_initial.pt` 的 `model` 字段比较：

- 共有键 **301**，逐值完全一致 **300**。
- 不一致键清单：**仅 `core.child_feature.0.weight`**。
- 第一层权重：旧 `[256,275]` → 新 `[256,531]`。
- 无新增键、无删除键；冻结主干不在原 checkpoint 的 model 字段，未计入共有键数。
- 基线 checkpoint SHA256：`9034bac7a2b251ef78fd43c708eed05edc22a95dadcf052b4a6ef89948acc7a5`，只读校验前后相同。

同种子 mock 模型另外检查了原版/新组所有共有张量及构造后的 CPU RNG 状态，并检查两个新组实例全部张量一致。parameter_groups 仅包装继承结果作返回前断言：新 child_feature 全部在 understanding，不在 shared，没有重新分组。Decoder、辅助读出及模型训练/渲染/推理流程继承原版。

## 验证结果

**通过：12 项新增 CPU 测试 + 18 项原版 CPU 回归，共 30 项唯一测试；另有 1 次真实资产 CPU 初始化/前向验证通过。** py_compile、git diff --check 通过。

真实冻结权重 + 真实数据 `scene0279_02`，context 帧 31/57，seed42 零步模型：

| 项目 | 实测值 |
| --- | --- |
| Query / GS 数量 | 55 / 3520 |
| GFM grid | `[1,2,256,32,32]` |
| evidence shape | `[1,3520,256]` |
| evidence 均值 | `-0.0072211166843771935` |
| evidence 总体方差 | `0.21637889742851257` |
| evidence 零占比 | `0.10085227340459824`（约 10.0852%） |
| 整条 GS evidence 全零比例 | `0.10085227340459824` |
| evidence 全部有限 | true |
| membership logits | `[1,3520,57]` |

真实验证总耗时约 384.3 秒（包含真实权重构造与核对）。实际送入 child_feature 的末 256 维与再次采样结果逐值一致。mock 原版/新组 forward 的 Gaussian、Query 类别 logits、最终 Query tokens 逐值一致，membership 形状一致。

测试准备阶段出现过两次 mock fixture 错误：先是几何 8×8 与 depth 256×256 不一致，再是 mock GFM 仅一视图。仅修正 mock 为完整两视图、同分辨率几何/depth/SAM，没有修改生产代码或放宽断言；最后 12 项新检查全部通过。18 项原版回归已在首次组合运行通过，没有重复计数。

CPU 真实前向范围是冻结 Omega + SAM → encode_context → 原始 core/GS 生成 → evidence/原型 membership logits。CPU 不支持本项目实际 CUDA rasterizer，因此没有声称真实完整 generate/render 或 CUDA 前向通过，也没有测量训练效果。

## 对照约束与交付范围

`UnifiedQueryConfig()` 与 attempt02 checkpoint config 完全一致：`source_many_to_one / support_matching=true / support_purity=0.5`，不定义 default_config；新 core 明确拒绝改用其它匹配配方。原训练脚本唯一修改为增加 architecture choice；预算、优化器、understanding_weight、warmup、seed、数据顺序原样保留。原 unified_query_core.py、unified_query_loss.py、unified_superpoints.py、direct_instance_attributes.py 均逐字未改。

没有训练更新、优化器创建或作业提交，没有创建 attempt 目录，没有修改 attempt02、attempt17、历史快照或历史评测产物。当前修改未提交/未推送，按本次要求交付工作区 diff 与本报告/JSON。

复现命令（仓库根目录）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /space/mawb/anaconda3/envs/tokengs/bin/python -m pytest -q tests/test_unified_gs_evidence.py tests/test_unified_query_v1.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /space/mawb/anaconda3/envs/tokengs/bin/python scripts/validate_unified_gs_evidence.py --output /tmp/unified_gs_evidence_real_validation.json
```

运行记录、真实 evidence 统计、源文件 SHA256 见 `unified_gs_evidence_validation_20261019.json`。数值方差采用总体方差（unbiased=False），零占比指全部 evidence 元素恰为零的比例，另报整条 GS evidence 全零比例。
