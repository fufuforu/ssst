# V4 从头直接联合训练：13 节点八卡

用户纠正为“不微调适配的那个，从头开始训练”。以此为准：只加载官方冻结 VGGT-Omega 与 SAM；所有可训练分支按构造器重新初始化，optimizer 无历史状态，无 source checkpoint。之前准备的适配2/V4初始权重续训方案未提交，其 CPU预检不能用于本次作业。

新组为 `attempt11_geometric_sam_direct_joint_node13_8gpu`，专用不可变代码 `/space/mawb/ssst_gfm_v4_direct_joint_code_20261016`。13节点8×3090、32CPU、192GiB、global batch8，每rank microbatch1、累积1。与正在运行的65665/65671互不替换，已停止的65661不恢复。

## 实际训练

使用 V4 几何感知 SAM 初始化；重建和理解全部可训练模块从新初始化开始，直接在1191场景/8337窗口训练4epoch，共4172更新。无适配2epoch、无短集预训练、无任何旧训练模型或AdamW继承。`--v4-direct-joint` 明确禁止 source/transfer/adaptation/freeze 类参数。首次 progress 即 full/0，short_updates=0、adaptation_updates=0；不重跑以前的短训练。

重建 LR1e-4，几何 LR1e-5，理解 LR1e-4，首100步全局LR warmup后cosine。理解loss beta首1043步从1/1043升1，其后3epoch维持1。理解loss warmup不是独立重建适配阶段；首步开始两个分支同时更新。与从已适配模型继续时用的1e-5重建LR不同，这里随机初始化的重建使用原基准LR学习。

沿用同一个总loss反向传播；不加入梯度隔离或新前向结构。Omega/SAM冻结，Write=0。L6/8/10/12、数据、Gaussian输出和官方评测协议不变。只复用RGB/冻结资产标识锁定的缓存，不缓存或继承可训练特征、query、匹配关系。

每50步保存完整可恢复checkpoint，每100步固定8训练/8验证监测。首步检查八rank新池化梯度，有限loss/梯度/参数检查和冻结骨干版本检查沿用原逻辑。不要求短期loss下降，不在此处虚构任何质量通过结论。

## 预检与后续

预检校验两个官方资产SHA、完整V4 meta模型与所有可训练分支可训练、空AdamW状态、数据1191/8337和128 heldout无泄漏。运行报告在attempt11/preflight.json。GPU首步是否通过，以FULL_STARTED及训练journal为准，不能仅从RUNNING判断。

训练成功后自动导出同一128场景并走官方11指标评分，两项text留空；评分非有限仍保留并报错，不能改协议。比较任务等待65666、65672及本组评测结束，对比V3冻结、V4适配2后冻结、V4从头联合，以及旧C参考。这些组的初始化/训练预算不同，不能把它当成仅freeze或仅初始化单变量对照。

作业号/执行SHA以attempt11/submission.json为准。提交后单次核对分配，停止轮询，等待用户通知。

实际预检通过：两个官方资产SHA匹配；196010160可训练参数、空optimizer、无source checkpoint；8337训练窗口/128 heldout身份通过；25项CPU检查及Python编译、shell语法、diff检查通过。
