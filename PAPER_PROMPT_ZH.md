我准备根据已完成的C8_Cal版本撰写论文。请先阅读实际代码、配置和结果，梳理模型pipeline与相关工作，先交付有证据的论文准备材料，不直接编造完整论文。

仅阅读和分析，不修改仓库、不运行训练、不操作作业。若链接无法访问，请明确指出，不要把无法读取的文件当作已经阅读。

一、固定研究版本

研究对象：Baseline epoch8＋Token＋GS质量头＋校准，即C8_Cal。不要混入QPanchor1/2、QuerySplat改造、4096token微调或BCD多视角实验。

阅读入口：
https://github.com/fufuforu/ssst/blob/paper/c8-cal-baseline8-20261028/PAPER_READING_C8_CAL.md

正式指标与模型身份：
https://github.com/fufuforu/ssst/blob/paper/c8-cal-baseline8-20261028/docs/paper_c8_cal/C8_CAL_MODEL_REFERENCE_20261025.json
https://github.com/fufuforu/ssst/blob/paper/c8-cal-baseline8-20261028/docs/paper_c8_cal/frozen_baseline8_gs_quality_completed_review_20261025.md
https://github.com/fufuforu/ssst/blob/paper/c8-cal-baseline8-20261028/docs/paper_c8_cal/codex_handoff_baseline_epoch8_completed_20261023.md
https://github.com/fufuforu/ssst/blob/paper/c8-cal-baseline8-20261028/docs/paper_c8_cal/formal_11metrics_comparison.csv

GitHub固定代码快照（完整文件树与原执行快照逐字节一致，原执行提交为02392745114b538fe1c30901c5add8e6e18745e4）：
https://github.com/fufuforu/ssst/tree/acd3618f63489138e947eb20e93106b1d58caa83

确认11项指标顺序：AbsRel/RMSE/PSNR/SSIM/LPIPS/Context mIoU/Context mAP/Context PQ/Target mIoU/Target mAP/Target PQ。
对应值：0.05036/0.17094/23.6308/0.78454/0.35547/59.8095/28.4329/67.6198/59.5257/26.8534/65.606。

二、主模型与质量头阅读入口

主模型实际架构：OMEGA_V4INIT_NO_ANCHOR_CA_ISOLATED_V4_RECIPE_V1。请从checkpoint加载入口与模型注册表确认，再沿继承/调用追踪，不要只依据类名或某段注释判断。

https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/export_unified_query_checkpoint.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/tokengs/models/superpoint_structural_variants.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/tokengs/models/v4init_isolated_noca.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/tokengs/models/olda4_initialization_ablation.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/tokengs/models/superpoint_dual_core.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/tokengs/models/unified_query_v1.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/tokengs/models/unified_query_loss.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/v4init_isolated_recipe.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/v4init_extend8_policy.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/cache_gs_quality.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/gs_quality_core.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/fit_gs_quality.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/frozen_iou_head_core.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/export_gs_quality.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/reciprocity_panoptic.py
https://github.com/fufuforu/ssst/blob/acd3618f63489138e947eb20e93106b1d58caa83/scripts/export_object_locus_v3_set_official.py

请沿相关import继续阅读，实现中的数据流和梯度流优先于概念命名。上面固定提交不包含新增论文说明，说明与结果文件使用前面的paper分支链接。

三、梳理完整pipeline

按输入→几何/视觉骨干→场景表示→重建/理解→渲染→质量预测→输出组装解释：
-训练与推理各使用什么视角、相机、图像和GT，哪些量来自预测。
-冻结与训练模块、多视角特征组织、坐标和尺度变换。
-重建token、锚点、理解query/superpoint/object token的区别、数量、维度、初始化和逐层更新。
-Gaussian属性如何预测，重建监督在哪些层施加。
-理解分支如何读取重建表示，哪些位置detach，是否存在反馈，如何产生语义/实例/全景输出。
-主模型训练损失、匹配、权重、优化方式和4→8epoch续训策略。
-Target图像用于什么操作；相机/几何校准与质量分数校准必须分别解释。

关键步骤提供文件/函数/行号、张量形状和必要公式，并标明上游沿用、适配或本项目新增的证据。

四、单独解释质量头

核实Token256和GS32的来源、32个统计量计算方式与含义、监督IoU的定义、train/dev/formal划分、标准化统计、head结构、最佳epoch选择和推理应用。

核对(q/q.mean().clamp_min(1e-6)).clamp(0.5,2).pow(0.25)在像素竞争中的实际位置。
区分Native8、Token-only、Token＋GS、Rank-only、Calibrated，说明各消融固定哪些量、改变哪些量。

五、相关工作与贡献边界

根据代码实际依赖和设计检索原论文、补充材料及官方代码，覆盖：多视角几何基础模型、前馈3DGS重建、3D场景语义/实例/全景理解、query/token/superpoint与Gaussian的结合、mask IoU/质量预测及排序/像素竞争校准。
重点核实SIU3R及实际使用的其他上游。QuerySplat若与本版本没有直接实现关系，只能作为相关工作，不能写成本模型组成。
每篇重点文献提供准确名称、年份、可核实论文/官方代码链接、核心思想、与本方法联系和差异。未核实标记待核实，不编造引用或首次提出结论。

六、必须守住的事实

质量头阶段冻结主模型，不是重建主干和质量头端到端联合训练；五个质量头案例重建渲染相同，不能将重建收益归于质量头。Target-all6含Context2，不是pureNovel4。深度逐图scale+shift评测，不等于绝对尺度精度。C8_Cal有PQ代价，不能声称全部指标全面提升。单种子、A/C容量不同、缺少部分控制实验，不足以证明严格几何因果机制。外部结果未同协议重测不能直接宣称全面胜出。无GT、pose-free、context-only、实时、泛化等主张均需逐项核对证据。

七、输出顺序

1.版本与证据清单。
2.完整pipeline及Mermaid数据流/梯度流图。
3.模块表：输入/输出/维度/作用/代码位置/来源。
4.主模型训练、质量头训练、最终推理的独立流程。
5.相关工作对照表与创新边界。
6.可支持的贡献候选，以及目前不支持的表述。
7.Method与Related Work章节提纲、公式和符号。
8.尚缺的消融/实验与待核实问题，按重要程度排序。

分析使用中文，保留准确英文术语。先忠实还原实现再讨论论文叙事。历史说明中的服务器绝对路径只是来源记录，不要求访问本地文件；阅读以GitHub源码、公开的配置/结果摘要和可验证论文为依据。
