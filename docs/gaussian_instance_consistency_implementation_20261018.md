# 渲染前跨父Token Gaussian实例一致性：实现

用户授权：按已讨论方案实现渲染前8维Gaussian实例约束。本轮实现及CPU/合成分布式测试，不提交真实数据训练、GPU诊断或完整评测，不修改既有快照与checkpoint。

独立入口：`--architecture unified_gaussian_consistency`，架构身份 `OMEGA_UNIFIED_GAUSSIAN_INSTANCE_CONSISTENCY_V1`。代码在 `tokengs/models/gaussian_instance_consistency.py`，通过原训练入口及MODELS工厂注册，使用现有原Unified实例一致性基线，不叠加回采/边界RGB/软几何注意力。

## 保留内容

不新增参数或模块。保留原超点Query初始化、共享Decoder、Gaussian重建、分类、Query–Gaussian相似度Mask、NMS/Panoptic及渲染后实例一致性。推理generate不读取GT，输出键和值与相同权重的基线一致；只在step_loss中计算新增项。原生参数键/顺序、初始化RNG相同；新配方写入checkpoint元数据，由现有初始化/恢复/导出检查防止混用。没有自动把旧训练终点迁移为新版本。

## GT归属与置信度

只使用两个上下文视图的GT（复用原final_targets）。以实际alpha-compositing贡献的转置汇总每个Gaussian对GT实例的贡献；有效像素贡献总和作为分母，包含有效stuff像素，避免只在thing之间归一化导致错误高纯度。

- thing归属纯度≥0.7，且两个视图累计有效贡献质量≥0.001，才进入直接约束。
- 质量单位是合成权重之和，不是概率；阈值0.001是本项目首版选择，需GPU覆盖率检查，不声称已验证最优。
- GT归属、纯度、贡献质量、相机以及用于分配的全部Gaussian几何均detach；空GT直接返回未归属，不调用这部分渲染。
- 不从原Query匹配/分类/no-object资格推断归属，不将低纯度/不可见/stuff强行拉入实例。

## 跨父Token正样本

现有8维实例向量单位化为z。先在每个“GT×父Token”内用归属纯度加权汇总向量并单位化，得到父Token原型；某个父Token的Gaussian只能查询同GT的**其他父Token原型**，各其他父Token等权，再归一化。原型保留特征梯度，双向更新各父Token的实例特征；归属权重无梯度。

正项为 `relu(1-cos(z, other_parent_prototype)-0.1)`，按Gaussian置信度加权→父Token等权→GT等权。不使用同父Token的Children自充正样本；只有一个可靠父Token的GT跳过正项。其他父Token原型相消、范数≤1e-6时跳过该父Token正项并记录，避免用零方向作目标。

## 局部异实例负样本

在可靠thing Gaussian中寻找最多8个**其他GT**的三维近邻，距离不超过 `0.1×输入点云RMS场景尺度`。先排除同GT候选再选近邻，避免同实例稠密Children填满k个位置；不补用远处异实例。搜索使用detach位置/尺度，按256个anchor分块，不建立N×N反向图。

负项为 `relu(0.5-(1-cos(z_g,z_h)))`。邻居按纯度加权，anchor按纯度加权，再父Token/GT等权。没有半径内异实例时跳过，不制造负样本。0.1尺度/8邻居/余弦margin0.5均是明确的首版配置，需后续验证覆盖和效果。

## 新增项的梯度路径

不能只detach渲染器：原Child特征含有 `(Gaussian xyz-anchor center)/anchor radius`，直接用原8维向量仍会更新显式几何。

因此额外调用**同一组**semantic_adapter、child_index、child_feature、instance_head，使用同一共享Query，但把Gaussian xyz、center、radius作为detach常量重新计算8维向量。其数值与原生8维读出一致，仅梯度路径不同。

- 新项更新实例头、Child特征、语义适配、共享Query Decoder/初始化器/GFM投影。
- 新项不对Gaussian属性张量、中心、半径、geometry heads、activation_head、reconstruction_adapter、预测相机、GT分配或KNN选择求导。classifier/query_mask也没有新增直接监督，仍由原损失更新。
- 原RGB、Mask和渲染后实例辅助项的几何梯度保留；不是冻结重建。共享Decoder被新项更新后，重建输出仍可能间接受影响，需实际对照。

## 总损失和监测

`L=Lrec+u*(Lunder_original+0.01*Lrendered_instance+0.01*(Ldirect_pull+Ldirect_push))`。

新增项沿用理解系数u的预热；u=0.05时直接约束有效系数为0.0005，与现有渲染后实例辅助项同量级。原有损失不重加、不替换。新增配方完整入metadata，不能把该系数称为已经调优。

记录：可靠Gaussian数/比例/纯度/质量、GT数、有效跨父Token正样本GT/父Token/Gaussian数、相消原型跳过数、有效局部负样本GT/anchor/边数、两项损失、余弦距离、负项违反margin比例、实际加权损失与系数。空监督保持有限且与8维读出有零梯度图连接，适配不同rank监督数量不同的DDP。

## 验证及下一步边界

测试覆盖：有效/相消原型、跨父Token排除、自身单父Token跳过、GT/父Token平衡、复制Children与重命名GT不变性、stuff分母/低纯度/低质量过滤、场景尺度不变性、近邻不补远样本、显式几何/置信度无梯度、新项理解/共享梯度、旧渲染几何梯度保留、原生推理/RNG/参数完全相同、metadata及checkpoint检查、完整step_loss系数/上下文监督。

真实本机两rank Gloo合成测试覆盖可变Query数、某rank没有直接监督、只有正项无负项、checkpoint Decoder重放，两次更新后所有可训练参数SHA同步。它不是实际数据GPU冒烟或训练结果。

最终45项检查全部通过：新增14项损失/模型检查及1项真实两rank Gloo，加原实例一致性、Unified及探针30项回归。Python AST及git diff检查通过。测试日志、代码SHA及文件哈希存于下述证据目录。

验证证据目录：`/space/mawb/ssst/group_plus/gaussian_instance_consistency_v1/implementation_20261018/`。后续若授权GPU smoke，应首先核对真实归属覆盖/正负样本数量、新项单独梯度路径、数值有限及多卡同步，再决定短训；不能用旧smoke收据代替新项GPU验证。本轮未提交作业。
