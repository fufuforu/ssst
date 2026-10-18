# 直接8维Gaussian实例属性：全局一对一GT监督

用户要求取消来源多对一，Query数多于GT时确保每个GT有负责Query。本轮实现与CPU验证；未提交训练或GPU诊断作业。

## 新主线入口

- CLI/注册名：`unified_direct_instance_attributes_one_to_one`。
- checkpoint架构身份：`OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_ONE_TO_ONE_V1`。
- 配置：`matching_mode=global_hungarian`，`support_matching=False`。
- `LocusGSOneToOneDirectInstanceAttributes`继承直接8维属性模型，生成、原型点积、渲染Pull/Push/Cross、stuff与推理沿用原路径。它不增加可训练参数。
- 新入口默认生成正确配置；若显式提供旧来源多对一/受限匹配配置，直接拒绝，不静默改写配置。训练入口使用模型对应的默认配置，避免通用默认仍落到旧来源匹配。

历史原Unified和`unified_direct_instance_attributes`入口及旧checkpoint身份保留供复现。后续这项改动的实验必须使用新入口，不能继续用旧67390提交脚本硬编码的架构名。新旧模型同名参数形状可相同，但不能因此隐式跨架构续训旧Adam/已训练终点；现有严格身份检查仍然生效。

## 训练对应关系

对每个场景，N为实际生成的Query数，K为`build_context_instance_targets`得到的全部有效thing GT：

1. 构造所有N个预测与所有K个GT的完整成本矩阵，不查询来源纯度、初始超点GT、no-object预测或NMS资格。
2. 代价沿用已有匈牙利分支：`2*(-P(class_GT)) + 5*BCE(mask, GT) + 5*DiceLoss(mask, GT)`，在两个context的全部有效监督像素上计算。
3. 全局匈牙利得到K对：每个有效GT恰好一个负责Query，每个正Query只负责一个GT。没有匹配后来源过滤、IoU阈值剔除或低类别概率剔除。
4. 其余N-K个Query监督为no-object；K个正Query分别接受完整GT Mask监督。

直接8维属性/Query Prototype的点积属于前向归属计算，匈牙利属于训练GT责任分配；两者承担不同功能。一对一仍会通过监督影响模型推理行为，不能理解为“训练匹配与推理学到的内容无关”。

保证的是**训练GT配对覆盖**，不意味着模型已预测出高质量Mask，也不保证推理没有漏检。分配给GT的Query仍可能预测no-object或被最终组装删除，这些需要后续实验评估。

## 完整覆盖与失败条件

- 每步检查`N>=K`、`N>=1`；不足时明确报错并显示实际N/K，不创建GT驱动的额外Query、不截断GT、不接受部分匹配。
- 对匹配成本检查有限性；非有限直接报错，不替换为人为大值继续训练。
- 检查solver返回恰好K个不同Query且GT索引完整覆盖`0..K-1`；异常返回直接报错。
- 无thing GT场景K=0：无正Query，所有Query为no-object，stuff与其余原监督正常计算。
- 新增`unmatched_GT_count`日志字段；新模式正常步必须为0，且`matched_query_count=matched_gt_count=gt_count=K`。K>0时正Query/匹配GT比为1。
- `unsupported_GT_count`在新模式不再表示几何来源覆盖；新模式不构造来源硬约束，不能据该字段为0推断初始超点覆盖完美。

已有67390完整日志的33376次窗口曝光中，N-K最小值为13，没有N<K。这验证的是已有运行观测，不替代新训练每步运行时检查。

## 损失归一化需明确记录

新模式使用已有匈牙利分支的逐Query加权分类交叉熵，no-object类权重仍为0.1：

```text
Lcls = (sum_positive CE(q, class_GT) + 0.1*sum_background CE(q, no_object))
       / (K + 0.1*(N-K))
```

K=0时该式成为背景Query的平均CE。正Mask BCE/Dice对K个匹配实例等权平均。

这与旧多对一分类“每GT平衡、全部背景Query先平均后整个组乘0.1”不同。因而背景总权重随背景Query数量变化；没有另调0.1或理解总权重。此版本改变了匹配约束/正负分配与对应分类归一化，后续收益不能严格只归因于移除来源纯度门槛一个因素。

## 代码与验证

- `tokengs/models/unified_query_loss.py`：完整有限成本匈牙利、覆盖断言、无来源/无后过滤路径，复用逐Query加权CE。
- `tokengs/models/unified_superpoints.py`：新匹配模式及禁止来源约束的配置验证。
- `tokengs/models/direct_instance_attributes.py`：新身份/默认配置及匹配配方元数据。
- `tokengs/models/superpoint_structural_variants.py`：新架构注册。
- `scripts/train_unified_query_v1.py`：CLI及按架构选择默认配置。
- `scripts/evaluate_direct_instance_attributes.py`：固定权重Prototype/HDBSCAN读出识别两种架构身份；官方导出已有注册表支持。

针对性验证覆盖：完全没有来源信息、全部Query预测no-object、N=K/N>K、多GT/空GT、Query置换、Mask驱动匹配、逐Query背景CE、N<K报错、非有限成本和solver不完整返回报错、新旧身份拒绝隐式混用、同种子初始参数/前向保持一致。

另用实际两rank CPU/Gloo进行两次DDP更新：一rank N=K、另一rank N>K，均使用真实一对一损失和渲染实例损失，检查每GT覆盖、全部参数有限梯度、新属性/原型/stuff头梯度及全参数SHA同步。CPU测试不替代真实CUDA及新方案质量验证。

本轮验证收据见`docs/direct_instance_one_to_one_validation_20261018.json`；最终通过项数由收据记录。没有自动短测/全量续训，也没有改写旧作业结果或执行快照。

最终验证：37项唯一CPU模型/匹配检查（28旧回归＋9新检查）与1项真实两rank Gloo通过，共38项，重复执行不重复计数。初次测试3项失败来自测试输入logit20的FP32饱和梯度断言；修正合成输入为logit4后全部9新检查通过，生产实现未为此放宽断言或改变损失。checkpoint身份检查补强并单项通过。Python编译和diff检查通过。
