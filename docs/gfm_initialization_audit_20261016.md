# Object Query初始化专项审计：固定权重、无参数更新

用户授权：检查SAM+点云初始化为何未按预期形成最终实例，先审计初始化，不立即改分类头或重训。当前65996/66039及其快照不改。本轮以原官方输出复现稳定的已完成V4整体理解权重0.05组为主；不把隔离组跨3090/4090复现差异混入初始化因果判断。

## 固定来源和样本

- 运行：/space/mawb/ssst/group_plus/gfm_initialization_audit_v1/attempt01。
- 独立快照：/space/mawb/ssst_gfm_initialization_audit_code_20261016。
- 四个已存在checkpoint：checkpoint_initial（0步）、epoch1（1043）、epoch2（2086）、epoch4（4172）。SHA256、大小/mtime、数据/模型配置已CPU逐项核对并锁定在preflight.json。没有保留更早逐步状态，不重放训练或伪造逐步历史。
- 每阶段同一128留出场景，GT/输入RGB逐场景一致。主要条件分析固定为上一轮已确认“存在IoU≥0.5 SAM显式支持”的152个GT，名单在新推理前从旧产物锁定；其余GT保留作对照，不仅汇报成功候选。
- checkpoint间重建、语义Adapter和Decoder也都在变化，趋势不是初始化模块单独因果证据。
- 同一16个固定训练窗口做梯度探针，来自原有short训练候选选择，非按本轮结果挑选；不对其做任何更新。

## 审计问题

1. SAM候选筛选/合并/截断后是否仍覆盖GT？沿同一候选，初始化→L6/8/10/12→Raw GS Mask→no-object/竞争/面积组装，在哪一步失去实例支持？
2. 初始语义投影、几何编码与learned向量的范数/方向有何差异？同时记录V4逐区域点权重是否饱和、shape投影/加权语义分量的范数与方向；SAM与fallback读取Anchor的尺度、3σ内Anchor数、注意力熵/有效Anchor数是否明显不同？
3. Saved checkpoint之间SAM/fallback匹配、no-object、梯度活动如何变化？区分初始偏置和训练后变化，不能把终点一次Hungarian匹配当作历史正样本数量。

候选身份使用实际合并后mask的SHA和压缩支持mask，不把query索引当固定实例身份。相同GT的最佳候选可能随合并变化；报告hash缺失/变化。固定终点候选hash另外跨checkpoint追踪，避免“换了候选却声称跟踪同一个”。2D投影和中间层分类均为探针，非官方训练头，不将其IoU直接与GS读出作严格因果差值。

## 终点固定权重2×2干预

| 变体 | SAM槽q embedding | 初始c/s读取先验 |
|---|---|---|
| native | 原始池化投影+几何编码 | 原始局部中心/尺度 |
| embedding_fallback | 当前同索引fallback向量经过原norm | 原局部c/s |
| geometry_global | 原始q不变（其中几何编码也不变） | 改为fallback式场景中心/场景尺度 |
| both_fallback | 同索引fallback embedding | 场景中心/场景尺度 |

仅在审计脚本forward hook修改initializer输出，不改模型源代码/权重/分组/提议选择/官方协议。非SAM槽不变。每次干预核对候选数及Gaussian重建逐位不变，说明干预范围只在理解Query。干预是分布外的固定权重定位，不等于相同策略训练后成绩，不计算冒充官方mAP/PQ。

同索引learned向量可能经常被SAM覆盖，不能一概叫“充分训练过的learned Query”。CPU另读取原Adam exp_avg_sq与权重范数逐行记录：epoch1/2/4均95行有非零二阶矩，前5行为零。非零支持曾有梯度活动；零值也可能来自衰减/浮点下溢，不能证明从未训练。该证据用于解释embedding_fallback对照，尤其不能把它失败直接归罪于所有learned基底设计。

## 梯度审计

model.eval，重建参数不记录梯度；理解值与原生前向一致。调用原v3_set_losses（同matcher/权重）暴露live classification、thing BCE+Dice、总understanding，用autograd.grad测q_pre逐query及initializer各参数梯度，同时记录最终no-object logit梯度与native匹配。原生模块参数.grad保持None，不创建优化器、不step。

这是每个saved checkpoint对固定单窗口样本的pre-clip梯度，非历史batch8归约后的训练梯度。报告当前全局理解系数0.05×beta，原始组件范数单独保留；不把局部范数说成真实Adam更新或方向冲突。Adam历史二阶矩仅作活动证据，两类证据不混淆。

## 执行和验证

18节点4×4090。先4场景终点冒烟（含所有干预和每rank一个训练梯度探针），通过后才运行完整4checkpoint×128场景、终点三个干预各128场景、4checkpoint×16个训练梯度探针与汇总。没有参数更新，也没有续训。

每rank每stage首场：两次无观测器原生重复→带观测器输出，采用已验证的有硬上限重复性收据；每stage结束参数版本不变/.grad为空。保留source、cohort与干预匹配核对。结果写summary.json/report.md及候选支持NPZ、逐场景/逐梯度记录。

CPU来源预检通过，13项测试通过：干预独立性/未干预槽不变、native恒等、观测器无值改动、候选hash身份、autograd.grad无累积/无更新、固定条件汇总及既有诊断/有界重复性回归。py_compile/bash -n/diff --check通过。GPU冒烟尚未宣称通过，以新作业产物为准。提交后单次核对调度状态即停止轮询，等待用户通知。

## 已提交，停止轮询

作业66092，18节点4×4090，已单次核对RUNNING。执行SHA245baf9，独立不可变快照；GPU冒烟/反向探针尚未核实，不能称全量开始或通过。先冒烟，成功才全量及汇总。用户通知后检查本作业，不重复提交，不改训练。
