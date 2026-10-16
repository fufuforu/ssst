# V3 初始化隔离训练与18节点四卡固定权重诊断

用户授权：提交V3初始化的隔离训练，可用13/14/17/18；之后用18四卡推理诊断。两条任务独立，不等待当前65996训练完成。

## 新训练：单独改变初始化

- 运行attempt15_semantic_sam_v3_isolated_8gpu，沿用同一0步V4初始文件SHA b9d7bd597cc2bd0d2439a1eae3771760f2be55241a2db1c4ef3879b08a75a03e。
- V3保留SAM对语义Adapter特征的面积加权池化；只移除V4几何池化12张量/29505参数。637张量全部从未训练起点按名称复制；无预训练理解权重、无已训练参数迁移。
- 196M参数中的重建186881600保持；理解9099055（V4为9128560）。源Adam为空，新V3优化器为空，按名称核对共有参数的分组和顺序，保留原8rank RNG。
- 架构除该池化模块外不改；SAM筛选/合并阈值不变，但特征变化可能改变实际跨视图合并。不是仅改一个常数的完全同容量模型，差异本来就是29505参数的模块。
- 相同1191场景/8337窗口/批8/4172更新/4epoch，short/adapt0，理解beta首1043步升1；重建/几何/理解LR1e-4/1e-5/1e-4，首100LR warmup+cosine、clip1。
- 完全隔离：重建仅重建损失，理解仅完整理解损失；显式FP32 SUM/world reducer，首GPU步检查八rank参数相同。全局clip1依然存在范数耦合。
- 单节点八卡，可用分区3090/4090，只允许13/14/17/18；运行时硬件不同要披露，不能宣称逐位相同轨迹。当前65996及其快照不改、不停。
- 成功后自动官方128评测；11项有限、文本两项空白；比较V3隔离/V4隔离/整体0.05及旧C。

## 固定权重诊断：18节点四卡

- 独立运行 /space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt01。
- 模型：已完成4172步V4隔离65831与V4整体权重0.05的65838，各128相同场景。使用epoch04完整权重，不依赖65996或新V3训练。
- 每模型四rank独立推理，不创建优化器、不进行backward；全部参数版本前后不变。原始RGB身份校验、模型/heldout SHA及官方资产身份校验。
- 冒烟4场景（每rank一场、两个模型）后才运行完整128。每rank首场对比带/不带观测器输出：Gaussians、Child Membership、分类概率、渲染Membership逐元素完全相同。
- 跟踪原SAM候选逐可见视图覆盖、实际同视图去重/跨视图合并/80槽截断后的SAM支持；重放initializer分组并校验q_pre/中心/尺度与原结果一致，防止自建候选逻辑偏离。learned fallback没有显式SAM支持，不能因其无支持直接判为漏检。
- 初始及L6/8/10/12 Object Query的2D语义特征投影（非官方头）→继承Anchor渲染→几何Child渲染→最终融合Child原始渲染Mask→类别/no-object筛选→像素竞争→最终area-filter组装。
- GT仅在原生预测之后用于匹配/统计；native Hungarian与best-mask oracle同时保留。分类正确/类别错/no-object/低分、差Mask、小物体和重复预测分组。小物体定义为两个context总有效像素<1024，不等于COCO AP-small。
- L6/8/10/12 Read/Anchor Membership/路由与最终Child统一到Anchor空间，16次无固定点query打乱负对照；Read允许全局上下文。Write=0，因此路由是潜在分配而非实际反馈，不能把路由不一致叫跨实例写坏。
- 原始官方PNG输出差异比例逐场景记录（3090/4090跨卡数值可能有差异），不覆盖既有评测或输出。诊断recall/IoU不是官方mAP/PQ，原始SAM逐视图oracle与跨视图槽位覆盖分母不同，不直接作差当损失。
- 每场景保存一张典型GT/初始SAM支持/Raw Query/筛选/Final图，并生成summary.json/report.md、分类组和ownership关联；不预先声称定位完成或指标改善。

## 执行

代码固定到独立快照 /space/mawb/ssst_gfm_v3_isolated_diagnostics_code_20261016；作业号以新attempt15/submission.json及诊断submission.json为准。CPU验证和真实源预检以实际收据为准；提交后只单次检查分配状态，不持续轮询GPU冒烟，待用户通知后再检查结果。

验证记录：真实0步源预检PASSED（637共享张量、移除12张量/29505参数）；19项CPU测试通过（含V3共享权重迁移、初始化观测器不改变结果、有效像素IoU、官方area-filter阶段、真实双rank Gloo及旧策略回归）。py_compile/bash -n/diff --check通过。完整GPU冒烟由诊断作业首阶段执行，尚未宣称通过。
