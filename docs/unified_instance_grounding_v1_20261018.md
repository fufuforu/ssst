# Unified Query：可渲染实例特征辅助监督

实现用户确认的 InstanceSplat 启发版本，入口为 `--architecture unified_instance_grounding`，架构标识 `OMEGA_3D_SUPERPOINT_UNIFIED_INSTANCE_GROUNDING_V1`。

本轮完成代码与 CPU 验证，不提交训练、推理或正式评测。基于原 Unified，不叠加此前独立实现的软几何注意力偏置。

## Pipeline

```text
RGB → 冻结 Omega → GFM 投影与三维融合点特征
    + SAM 区域 → 三维邻域编码、超点池化 → Q0
Q0 → 原12层共享Decoder → Q
Q → 原重建读出 → Child Gaussians
Q → 原语义投影 + Child index + Child局部坐标 → Gaussian Child特征
    ├─ 原Mask描述子点积 → Membership → Mask渲染 → 原分割损失
    └─ 新增 LayerNorm + Linear → 每个Gaussian的8维实例特征
                                   → 同一Gaussian几何上的特征渲染
                                   → pull / push / cross 损失
```

原超点初始化、Decoder、几何更新、Gaussian数量、分类、Membership、NMS与Panoptic组装保持原行为。低维实例特征是独立辅助读出，不替换当前Mask，不引入聚类、额外Object Query或文本编码器。它从逐Child的特征生成，各Gaussian可保留不同的实例描述，不直接将同一个父Token的embedding复制给所有Child。

`UnifiedQueryCore`提供默认返回空字典的辅助读出扩展点，仅新core公开 `gaussian_features` 并生成实例特征，复用现有Child MLP输出，不重复计算MLP。旧core不额外保留Child特征输出。新core增加 `instance_head`，宽度256时新增4个参数张量、2568个参数。构造新头使用CPU RNG保护；相同seed与配置下原公共参数及后续RNG不变。新增头归入understanding参数组。

## 有符号特征与监督

使用现有 `render_feature_channels` 对8维特征作零背景alpha合成，位置、尺度、旋转、透明度和相机与原生成结果一致。特征允许负值，不经过sigmoid、概率裁剪或Membership alpha归一化；在渲染之后对每个像素做L2归一化，epsilon为1e-6。

GT仅在generation完成后用于loss。使用上下文两视图中有效thing像素及跨视图一致的scene-local实例ID，不使用novel视图GT、Query匹配、SAM纯度、类别置信度或alpha阈值来筛除辅助监督。stuff/ignore像素不参与。GT实例在一个视图不可见时，不构造该视图的空原型；只有两视图都可见的同一实例产生cross项。

每个可见实例的原型为其归一化像素特征的均值，原型本身不再次归一化。

```text
pull  = mean_instances mean_pixels relu(||f_pixel - prototype||₂ - 0.5)
push  = mean_distinct_instance_pairs relu(1.0 - ||prototype_k - prototype_l||₂)
cross = mean_shared_instances relu(||prototype_view0 - prototype_view1||₂ - 0.5)
L_grounding = 2*pull + push + 2*cross
```

单视图先按实例/实例对均衡，再按有对应项的视图平均；cross按共同可见实例平均。batch按场景均衡，不跨场景混用实例ID。空项返回与图连接的零损失；全stuff、单实例、缺失视图和全零渲染特征均有有限结果。

本地实验margin为pull0.5/push1.0/cross0.5，属于本版本设计选择，不声称是作者未发布代码的精确超参数。当前push对所有不同实例均匀加权，同类别的不同物体也必须分开；没有加入语言语义hard-negative加权。

## 权重与梯度

保留现有理解权重u及其warmup：

```text
L_total = L_reconstruction + u * (L_understanding_original + 0.01 * L_grounding)
```

例如u=0.05时，辅助项在总loss中的有效系数为0.0005。`2/1/2`内部比例与`0.01`辅助系数明确分开；不是将论文的0.01直接作为本训练总loss系数。配方写入checkpoint meta的 `instance_grounding_recipe`，零步恢复及正式导出检查配方一致性；原架构的配置序列化不变。

新增实例loss默认通过Child特征及可微特征渲染向共享Decoder和Gaussian几何回传，未detach几何。这是本版新增的受权重控制的优化通路。它不直接更新分类头和Query Mask描述子；纯重建loss也不更新新增实例头。原有共享参数/梯度裁剪策略不变，冻结Omega/SAM不接收梯度。

训练日志记录pull/push/cross、原始/加权辅助loss、有效系数、实例视图数、跨视图对数、监督像素数及GT物体区域的非零alpha覆盖比例。覆盖率仅用于观察，不是loss门槛。

## 与论文的边界及下一步

依据[InstanceSplat §3.2](https://arxiv.org/html/2608.07144v1#S3.S2)的可渲染实例原型思路设计。保留本项目超点Query生成Gaussian的结构，不复刻其像素Gaussian/DPT骨干、语言teacher、HDBSCAN推理或实例边界加权RGB损失。后者需要在本版实例特征有效后独立验证，并明确边界权重的detach路径；本轮不同时改变这些因素。

新入口可独立训练及同架构恢复，官方输出仍使用原Mask组装协议。训练后可通过 `gaussian_instance_features`、`rendered_instance_features` 和 `instance_feature_alpha` 观察实例空间与图像空间；额外视图渲染接口也会返回这些字段。

现有Unified checkpoint缺少4个新张量，严格restore会拒绝直接当成新架构的完整状态。本轮没有放宽跨架构续训，也没有迁移已有模型/Adam。若后续选择已有完整Unified作为起点，需要另行记录明确的参数迁移和优化器策略。

## 验证

新增CPU测试覆盖损失数值、实例ID重命名/视图重排/像素复制不变性、无GT/缺失视图、signed特征、GT范围、梯度通路、旧输出与构造RNG等价、参数组完整性及checkpoint配方检查；并运行原A/Unified/B/串行/几何控制回归。

最终验证结果：**54项CPU测试通过，运行52.458秒（不含环境导入）**；其中新增11项。测试模块为 `tests.test_unified_instance_grounding`、`tests.test_geometry_controlled_queries`、`tests.test_superpoint_structural_variants`、`tests.test_unified_query_v1`、`tests.test_serial_query`、`tests.test_serial_residual_query`，OMP/MKL线程数均为1。语法及 `git diff --check` 通过。

CPU渲染器测试是可微局部混合的测试替身；控制节点CUDA驱动过旧，真实gsplat CUDA前后向、多卡和真实资产训练尚未验证。尚无新训练结果，不能将实现完成等同于性能改善。
