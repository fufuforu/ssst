# 新对话交接：Unified 渲染前 Gaussian 实例一致性

## 最新授权与已提交：A evidence / B evidence+coverage full4（2026-10-19）

用户授权两组完整数据八卡，分别使用12、13节点。**A训练67700 RUNNING，3dimage-12、8×3090、32CPU/192G；B训练67702 RUNNING，3dimage-13、8×3090、32CPU/192G。** A官方128评测67701 afterok:67700、B评测67703 afterok:67702，各同节点1×3090/4CPU/64G；CPU汇总67704 afterok:67701:67703、12节点4CPU/16G。这是唯一一次有界调度检查；RUNNING只代表调度状态，**未核实CUDA门禁/正式首步、未读取训练日志，已停止轮询，等用户通知完成再检查，不能重提作业。**

执行SHA `1982cd18c9ecd3cb7643cba7ff143b91738fd050`；独立快照 `/space/mawb/ssst_gs_evidence_ab_full_code_20261019`。运行根 `/space/mawb/ssst/group_plus/unified_query_training_v1`，A `attempt18_gs_evidence_full_node12_8gpu`，B `attempt19_gs_evidence_coverage_full_node13_8gpu`。两组1191场景/8337窗口、4epoch/4172更新、batch8、LR1e-4/100 warmup/原cosine、under .05/1043 warmup、模型seed42，原数据排序seed31415不变。匹配仍source_many_to_one/support true/purity .5，绝非上一轮新一对一入口。

A `unified_gs_evidence`，逐GS投影原GFM采样256维证据，位置detach，新增evidence_zero_fraction float；B `unified_gs_evidence_coverage`，同A参数/前向，多GT实例内部腐蚀1像素后alpha→1 BCE，0.1权重不乘understanding_weight；无8D/HDBSCAN或匹配/后处理改动。A对比既有attempt02（不重训），B对比A，同official128终点epoch4。两组301初始张量逐值相同；相对attempt02 300/301完全一致，唯一不同child_feature.0.weight。新的零步空Adam checkpoint只重放原baseline零步8rank RNG，不继承任何训练权重/门禁更新。

52项CPU检查通过（44模型回归+8门禁策略），之前真实CPU同窗口evidence统计逐值一致、位置梯度None/GFM梯度有效、两变体全state_dict一致。新prepare核对真实冻结资产、配置/数据SHA、135955个所需数据文件可访问、11真实样本及8probe RGB一致、evaluator固定源码。每训练作业内先16窗口真实CUDA/采样梯度门禁+2次8rankDDP更新，再丢弃，重新进程从准备零步checkpoint正式训练；失败写JOB_FAILED阻止正式训练。不能把这些计划执行的门禁写成已通过。

用户通知后检查两组SMOKE_PASSED、FIRST_UPDATE_VERIFIED、TRAINING_COMPLETE、FULL_TRAINING_AUDIT、evaluation、comparison/FULL_EXPERIMENT_COMPLETE；若失败先查JOB_FAILED/EVALUATION_FAILED及Slurm状态，不自动恢复/重提。正式审计包括初始参数/RNG一致、Adam4172、301权重变化、8rank连续日志/原顺序/学习率、zero fraction及>.25次数、B总损失权重。耗时估计并行训练5–8小时、评测20–45分钟，排队另计。详见 `docs/gs_evidence_ab_full_20261019.md`、`docs/gs_evidence_ab_full_validation_20261019.json`、`docs/gs_evidence_ab_full_submission_20261019.json`。


## 最新实现要求：直接8维属性改为全局一对一（本轮未提交训练）

用户明确要求取消多对一、Query数多于GT时确保不漏GT训练匹配。新增主线入口`unified_direct_instance_attributes_one_to_one`，身份`OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_ONE_TO_ONE_V1`，强制`matching_mode=global_hungarian`、`support_matching=False`；完整有限成本匈牙利，无来源约束、无匹配后过滤。N>=K时每GT恰好一个正Query，其余Query为no-object；N<K/成本非有限/覆盖不完整明确报错。保证的是有效GT训练配对覆盖，不保证推理召回。

模型头、直接8维属性、原型/聚类及渲染监督延用原路径；新身份/元数据区分旧实验，旧入口/独立快照保留。新匹配复用逐Query加权分类CE、no-object0.1，区别于旧多对一整组背景平均，其相对权重变化需在后续实验归因中说明。已有67390全部33376次训练窗口曝光N-K最小13、无不足，但未来仍每步检查。

37项唯一CPU检查（28旧回归＋9新检查）及1项真实两rank Gloo两次更新通过；全GT覆盖、有限梯度/全参数SHA同步、配置/身份拒绝及初始前向等价均验证。CUDA/收益尚未验证。详见`docs/direct_instance_one_to_one_20261018.md`及验证JSON。用户本轮授权实现，不是自动短测/全量或续训；旧提交脚本硬编码旧架构，不可直接重用启动新组。旧67390产物仍是来源多对一，不得用当前新增模式解读它们。后续若用户要求实验，需要准备新身份/配方对应的独立预检与GPU验证，不能把CPU通过当CUDA或质量收益已验证。

## 最新交接要求：供另一个agent独立审查

用户要求解释多对一/匈牙利与未匹配GT，并先整理文档，让另一个agent自行远程阅读代码/实验结果再作比较。独立审查入口：**`docs/independent_review_handoff_20261018.md`**，包含四种对应关系、三条训练匹配路径、完整实验表、服务器目录/代码入口、已确认与未证实解释、独立审查输出建议。未启动子agent、GPU或作业，未改模型。

必须澄清：当前原Unified full4/直接8维full4实际`matching_mode=source_many_to_one`，**没有调用匈牙利**；11.28%GT无正Query属于该路径。普通稠密有限成本匈牙利在N≥K且无匹配后过滤时会覆盖全部有效GT。Unified备用受限匈牙利会在solver后过滤禁止配对，才可能因来源硬约束留下未匹配GT。预测no-object/最终漏检也不能混为训练未匹配。

## 最新工作：跨版本共同失败链路只读审查（无新作业）

用户质疑持续更换单双分支/Mask方式仍不佳，要求反思是否忽略共同问题。本轮只读审查代码、67390最后200更新×8rank的1600窗口曝光及两组完整128官方PNG，未改模型/损失/官方评测，未GPU前向或评分、未训练/提交。结论和可复核证据见 `docs/shared_failure_audit_20261018.md` / `.json`。没有证明存在一个贯穿所有版本的底层bug，不能直接声称已定位唯一根因。

已确认当前Unified按初始超点纯度固定多对一完整Mask正监督，同GT有多个正Query，推理再竞争/面积筛掉且不重分配void；负Query先整体平均再乘0.1。1600曝光：平均57.13Query/25.24正Query，窗口内正Query/GT比均值7.77、GT加权6.81，负分类组权重占比均值3.28%；754/6685=11.28%GT无归属/分类正Query，8维渲染监督仍覆盖这些GT，不能称完全无监督。不是已证明归一化bug，也不是早期所有Hungarian版本都有此实现。

官方PQ库有预测void忽略未匹配GT及扣void IoU规则。新增PNG统计target新组void有效GT像素覆盖6.36%→7.61%，但忽略未匹配thing GT从62/507降至58/507，故不能把PQ增益直接归因忽略更多GT。原官方分数未改，上一轮结果报告已补充限定。训练工程审计通过不等于目标/输出一致；重复出现的Mask/分类/竞争/排序问题应作为后续验收，重建18–19dB对旧C22.6dB的差距单列，不继续只改Mask头。

下一建议是固定全量权重补齐同128分阶段诊断、监督覆盖与置信度校准，并单列固定几何可表达性。尚未授权/提交该GPU诊断或任何续训，不自动改匹配/权重/后处理或再开训练。

## 最新完成核查：67390/67391全量直接实例属性（覆盖下文RUNNING/PENDING历史）

用户通知完成后核实：**67390 COMPLETED/0:0，4h32m23s，13节点8×3090；67391 COMPLETED/0:0，23m55s，同节点1×3090**。4epoch/4172更新、8rank各4172连续有限日志、完整数据顺序、306张量全部有限且变化、306份Adam4172步、初始空Adam及首更新八rankSHA同步通过，已重新执行CPU训练审计。未重训/重提/重新GPU前向或评分。

正式128同cohort/帧/RGB，原始官方值与11数值CSV一致：PSNR18.08546→18.04698；context mAP18.92472%→16.70285%、PQ46.73369%→47.09803%；target mAP16.51922%→15.31136%、PQ45.01139%→45.42030%；语义mIoU context43.87631%→43.01703%、target43.75185%→43.16650%。重建接近、PQ小增、mAP/语义下降，当前组合方案没有整体超过原Unified full4，不能据小样本IoU或PQ局部增益判胜。

辅助4train+4heldout全GT每图IoU：本组heldout raw70.02%、NMS67.46%、最终原型54.79%、HDBSCAN49.45%；最终原型TP26/36、P86.67%、R72.22%，聚类TP18/36、P12%、R50%。比短测原型7.89%/2TP、聚类12.71%/0TP明显改善，不能再以短测否定泛化；但这些不是官方mAP/PQ，也不能替代正式等预算对照。当前固定读出优先原型。

8窗口×2视图×4读出的64组指标已从真实GT重算并完全一致，固定权重SHA/版本/无grad核对通过，基线初末文件SHA未变。正式导出checkpoint收据的集群`/datapool/mawb`路径已映射到本地`/space/mawb`核实同一文件。运行目录`COMPLETION_REVIEW.json`为PASSED，`review_figures/`有两张CPU缓存对比图。详见 `docs/direct_instance_attributes_full_results_20261018.md` 及同名JSON。当前建议先固定全量checkpoint定位候选/类别分数/NMS/组装阶段的AP差异，**未提交该诊断或任何续训，不自动延长8epoch**。主对照仍原Unified full4（无渲染实例监督），是组合方案比较，不是直接属性单因素消融。下文提交时RUNNING/PENDING/未核实均为历史，不能据此重提。

## 最新授权与已提交：直接实例属性完整训练集4epoch（覆盖下文“不自动全量”旧边界）

用户在67369完成核查后明确要求直接完整训练集训练并与已完成全量基线比较。**已提交67390训练/审计、67391正式128评测/辅助读出/比较**。唯一调度核对：67390 RUNNING，3dimage-13、8×3090、32CPU、192G；67391 PENDING(Dependency)，同节点1×3090、4CPU、64G，afterok:67390。执行SHA `3d06c39ad97657959052329c892bda023f07436b`；独立快照 `/space/mawb/ssst_direct_instance_full_code_20261018`；运行目录 `/space/mawb/ssst/group_plus/unified_query_training_v1/attempt17_direct_instance_attributes_full_node13_8gpu`。

1191训练场景/8337窗口、4epoch/4172更新、batch8、LR1e-4/100步warmup/原cosine、理解权重0.05/1043步ramp；保持67369直接属性/原型/渲染Pull-Push-Cross配方，不调损失或聚类、不增加67292约束。初始化只用67369零步空Adam checkpoint，4个完全相同的零步RNG扩至8rank，306张量全保留、与原全量基线292共同张量逐值相同；绝不继承短训终点。

主对照复用66432原Unified full4及其有效官方128结果（66433指标成功、66528补汇总），不重训基线；原full8为不同预算历史参考。原全量基线无渲染实例监督，因此是组合方案比较，不能声称单因素消融。每epoch探针/权重保留，终点评测固定epoch4；官方128使用原型原生输出，HDBSCAN只作相同8窗口类别无关辅助读出，非官方AP/PQ。

21项唯一CPU检查、实际资产/数据/零步预检和169614个所需数据文件可访问检查通过。既有模型/训练/推理代码与67369真实CUDA执行版本未变，复用其门禁，不重复smoke；8rank正式首步由训练入口检查，**本轮未核实首步、未读训练日志、已停止轮询**。预计训练5–8小时、评测20–45分钟，排队另计，非保证。等待用户通知后检查FULL_TRAINING_AUDIT、evaluation、readouts_final、comparison_to_baseline_full4、FULL_EXPERIMENT_COMPLETE及失败收据；不重提、不轮询、不自动追加8epoch。详见 `docs/direct_instance_attributes_full_20261018.md` 与提交JSON。

## 最新完成核查：67369直接实例属性短测（覆盖下文RUNNING历史）

用户通知完成后已检查：**67369 COMPLETED/0:0，43分28秒，3dimage-13、4×3090**。16真实窗口GPU门禁/四rank两步smoke通过并丢弃；正式零步空Adam且SHA与smoke零步相同，共同292张量与基线逐值相同；最终306张量有限并实际改变、Adam256、四rank各256连续日志。初末两checkpoint各8窗口双读出齐全；从缓存/原GT重算128组指标与汇总完全相同，初末各4worker参数SHA/版本不变、无grad。没有重训/重提。

统一全GT每图一对一IoU，基线→本组：train raw28.73%→40.19%、最终16.71%→32.74%、HDBSCAN20.49%→22.20%；heldout raw15.23%→14.35%、最终8.62%→7.89%、HDBSCAN15.00%→12.71%。本组heldout原型raw TP3/36、最终2/36、HDBSCAN0/36。PSNR train19.747→19.225、heldout10.021→10.562。旧来源匹配探针另报train40.72%、heldout8.17%，不能混用口径。

当前判断：结构训练有效、训练探针收益明显，heldout尚无整体实例收益；暂以原型归属为后续主候选，聚类保留对照，不直接全量，未提交下一实验。详见`docs/direct_instance_attributes_short_results_20261018.md`及同名JSON；运行目录`COMPLETION_REVIEW.json`和`review_figures/`。下文“RUNNING/未核实/等待完成”均为提交时历史，不能据此再次训练或提交。

## 最新授权：直接实例属性短测（2026-10-18）

**已提交67369，唯一调度核对：北京时间16:20:57，RUNNING，3dimage-13、4×3090、16CPU、128G。** 执行快照SHA `f9b5a5d2e68d0701a6b81ebb651d01db316142a0`。同一作业内依次专用GPU门禁、丢弃smoke两步、全新16场景256步训练、初末checkpoint各自prototype/HDBSCAN评测及自动汇总。CPU50项+真实两rank Gloo1项通过；292个共同零步张量逐值相同，旧头13张量删除/新头14张量新增；实际资产和20个窗口预检通过。真实GPU门禁/首步/结果尚未核实，提交后未读日志，已停止轮询。**等用户通知完成后再检查，不能重提或重训本组。** 收据见`docs/direct_instance_attributes_short_submission_20261018.json`；预计40–60分钟，真实耗时未测。

用户已在实现后明确授权“短测一下”。按原16场景/256步，准备一个新架构`unified_direct_instance_attributes`的专用真实GPU门禁→丢弃smoke两步→全新seed42/空Adam短训→初末checkpoint各自prototype/HDBSCAN双读出→自动基线比较。范围与停止轮询约定见`docs/direct_instance_attributes_short_20261018.md`。独立run为`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt16_direct_instance_attributes_short_node13_4gpu`，代码快照为`/space/mawb/ssst_direct_instance_short_code_20261018`。实际提交状态以该run的submission.json和后续提交记录为准，不能仅凭下文历史“未授权/未提交”重新询问或重复提交。

## 最新实现任务：直接Gaussian实例属性（2026-10-18）

用户授权实现“直接预测Gaussian实例属性，同时支持Query原型归属和HDBSCAN聚类推理”。新入口`unified_direct_instance_attributes`，身份`OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_V1`，实现说明见`docs/direct_instance_attributes_implementation_20261018.md`。最终Query直接为每个Child输出8维有符号属性，为每Query输出8维原型；单位化点积/独立sigmoid生成Membership，同一属性直接可微渲染8通道并沿用原Pull/Push/Cross。保留stuff/类别/重建/来源匹配/NMS，移除该新架构后置几何Child特征到8维的路径，不叠加67292直接Gaussian一致性项。

本轮只实现/CPU验证，**未提交GPU/训练作业**；不要把实现请求当已提交实验。旧checkpoint不兼容，不能隐式迁移；旧preflight/smoke含旧instance_head/305张量假设，不能直接拿来启动新架构。真实CUDA和实验收益尚未核实。用户之前已明确授权向指定fufuforu/ssst分支推送代码与实验记录，无需重复询问同一目的地。

验证完成：45项CPU检查及1项真实两rank Gloo测试通过，包含两次更新、不同Query数量、checkpointed Decoder、全部参数梯度/全参数SHA同步；训练/双读出评测CLI和语法检查通过。新评测脚本`scripts/evaluate_direct_instance_attributes.py`支持同checkpoint `--readout prototype|cluster|both`，区分prototype raw/NMS/assembled与类别无关HDBSCAN。详见实现文档和`docs/direct_instance_attributes_validation_20261018.json`。

## 最新完成核查：67349（覆盖下文RUNNING历史）

用户要求检查后确认67349 COMPLETED/0:0，6分58秒，3dimage-13、4×3090。两组16窗口/32图、8张配对图完整完成；source/before/after参数SHA一致、所有参数版本不变、无grad/optimizer、0更新，源checkpoint文件hash未变。真实缓存重算128组主要指标与汇总完全相同。详见`docs/fixed_instance_readout_results_20261018.md`及同名JSON；运行根目录`COMPLETION_REVIEW.json`。

新约束heldout同GT跨父8维距离0.5304→0.3183，但异GT也0.7154→0.4038；全局AUROC0.6997→0.6666、局部0.5990→0.5940。固定HDBSCAN未稳定改善：heldout一对一IoU基线15.00%、新组10.96%，低于raw Membership15.23%/16.52%；最终原生组装8.62%→6.30%。本轮统一口径为每图、全部GT、类别无关，不是旧来源匹配两视图IoU或官方mAP/PQ。

**须修正旧解释**：旧来源探针8.46%→5.50%本次精确重现，但不代表所有原始Mask全面退化。scene0030_01的GT16指定Query37/38最好IoU66.15%→2.50%，新组其他Query18仍80.99%，说明指定来源Query与实际Mask响应会脱节。不能只凭旧指标宣判原生Mask全部消失；也不能因新raw略升就忽略最终组装和8维局部分离问题。未重训/重提，未提交后续实验。

## 新增授权：固定权重机制与8维聚类（2026-10-18）

**已提交67349**，唯一调度核对为北京时间14:58:16，RUNNING，3dimage-13、4×3090、16CPU、128G。独立执行快照`/space/mawb/ssst_fixed_instance_readout_code_20261018`，SHA `8803dd4e3562c4727ebe6cfd1c5969c4e9aae2bc`。同一作业依次GPU机制/导出、CPU HDBSCAN/评测/汇总，无训练、无参数更新。12项CPU测试、32张合成图完整流程及真实模型/资产/RGB/GT预检通过；提交后未查看日志，真实GPU门禁和结果尚未核实。已停止轮询，等用户通知完成后再检查，不能再次提交本组。最新收据见`docs/fixed_instance_readout_submission_20261018.json`。

用户已明确授权67050/67292固定权重机制诊断，以及渲染8维→L2 Normalize→HDBSCAN→2D Mask与原Membership Mask对照。范围与口径见`docs/fixed_instance_readout_diagnostic_20261018.md`。仅复用两组256步终点，原4train+4heldout窗口；没有训练或新增训练组。该授权覆盖新GPU诊断提交，提交记录后续以专用运行目录`/space/mawb/ssst/group_plus/fixed_instance_readout_v1/attempt01_67050_67292/submission.json`为准，不要据下文“新诊断未授权”历史句子重复询问。

## 最新完成核查（2026-10-18，覆盖下文待检查状态）

用户通知67292完成后已检查：COMPLETED/0:0，36分00秒，3dimage-13，4×3090；GPU门禁、256步短训、自动汇总均完成。CPU重新核实305个共同零步参数逐值相同、空Adam/未继承smoke更新、最终Adam全为256步、参数有限且实际变化、四rank日志连续。未重训、未提交新任务、未修改执行快照。

新约束相对67050：train PSNR19.747→19.610、Mask IoU27.40%→27.09%；heldout PSNR10.021→9.994、Mask IoU8.46%→5.50%。heldout下降主要来自scene0030_01（11.03%→0.42%），另外三场景略增。新项真实活动，但局部负margin大量未满足；尚无固定口径的基线/新组8维表征对照，不能宣称表征已改善，也不能简单归因训练不足。

完整结果与限制见 `docs/gaussian_instance_consistency_short_results_20261018.md`。不建议直接全量；可建议固定权重机制诊断，未获新授权不得自动提交。下文“唯一待检查67292”“未核实GPU”等是提交时历史状态，不能据此再次提交或重复训练。

更新时间：2026-10-18。本文件优先于旧交接文档中的历史状态。当前用户只要求整理交接；**本轮没有查询67292的实时状态，也没有读取其训练日志或重新提交任何任务。**

## 1. 接手原则与当前任务

先读 `/space/mawb/AGENTS.md`。用户明确要求：**提交作业之后停止轮询，跑完由用户通知。** 提交后唯一一次调度核对已经做过，不能为了新对话接手再次持续检查；用户通知“已完成/检查一下”后，再检查收据、日志、checkpoint并分析。

当前唯一待检查实验是 **67292**：真实GPU验证通过后，自动进行16场景256步短训，结束自动汇总。不是只实现代码，也不是等待再授权短训；这两步均已授权并提交。不得重复提交、重训基线、恢复历史取消任务或自行增加全量训练/官方128评测。

研究问题：在实例专属8维空间直接改善“同GT跨父Token Gaussian一致性＋局部异实例分离”，是否能改善原生实例Mask，同时不明显损伤重建？这是对已有诊断的针对性验证，尚无本实验GPU结果结论。

## 2. 代码与执行身份

- 开发仓库：`/space/mawb/ssst_vggt_overfit_diagnostics_v1`
- GitHub：`https://github.com/fufuforu/ssst.git`
- 分支：`vggt-overfit-diagnostics-v1`
- 此交接前开发HEAD：`7319f69fba5d9ce3dc1f2878222293f732ae4f76`（67292提交记录）；交接文档提交后开发HEAD会再前进。
- **67292执行快照固定SHA**：`0882660021e17e51eae9471b5921d5a6db951b90`
- 独立只读执行worktree：`/space/mawb/ssst_gaussian_instance_consistency_short_code_20261018`
- 模型实现提交：`b006a4c8e15866e65933a3f7bdb3ce4174fb2098`
- Python：`/space/mawb/anaconda3/envs/tokengs/bin/python`

开发仓库与执行快照不同是正常的；不得改写正在/已经执行的快照。需要后续新快照时，用 `GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach ... SHA`，避免LFS下载鉴权失败。控制节点CUDA驱动较旧，CPU测试通过不代表实际CUDA通过，真实GPU操作用Slurm。

用户已明确授权向上述远端推送代码及本轮实验记录（包括内部路径、作业号、指标、清理说明）。历史推送使用 `git -c http.https://github.com/fufuforu/ssst.git.proxy=http://127.0.0.1:17899 push origin vggt-overfit-diagnostics-v1`。不要向其他远端导出。

## 3. 作业67292：只报告已知状态

运行根目录：

`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt15_gaussian_instance_consistency_short_node13_4gpu`

- 资源：13节点 `3dimage-13`，4×3090，16CPU，128G，时限4小时。
- 提交：北京时间2026-10-18 13:43:54。
- **唯一调度核对**：北京时间13:44:06，`RUNNING`，`gres/gpu:3090:4`。
- 上述RUNNING是历史调度状态，**没有核实GPU冒烟、首步、短训完成**；不能写成已验证通过。
- 先前估计40–70分钟，排队另计；增加了贡献转置/KNN，实际耗时未测，不能当实时ETA。

调度证据：运行根目录的 `submission.json`、`SCHEDULING_CHECK.json`；已归档到开发仓库 `docs/gaussian_instance_consistency_short_submission_20261018.json`。

同一作业顺序执行：

1. 原有有符号CUDA特征渲染oracle、原渲染实例loss对实例头/几何/共享特征的梯度检查。
2. 原16个真实训练窗口分配到四rank，检查新项正负覆盖、独立梯度、显式几何隔离、重算8维与原生读出逐值一致。允许单场景/单rank无监督，但合计必须有跨父正样本、局部异GT负样本及实例/理解/共享非零有限梯度；不临时放宽配方。
3. 两次四rank DDP优化更新、全参数SHA同步检查。
4. **丢弃全部smoke模型/Adam更新**；独立新torchrun从seed42/空Adam重新开始短训。
5. 16场景、256更新、global batch8（每卡1×累积2）；LR1e-4、20步LR预热和短cosine，理解权重0.05/20步ramp、clip1，每32步checkpoint。
6. 自动原探针汇总、实例头/初始化审计、新项活动审计及与基线67050比较。

任何门禁失败写 `JOB_FAILED.json` 并停止，不自动进入短训。不要求loss下降，不自动完整数据训练或官方128。真实资产/样本CPU预检通过，与67050共同305个零步参数逐值一致，无新增参数。此轮17项CPU检查通过（模型/新门禁16项＋真实两rank Gloo1项）；模型实现轮另有45项完整CPU回归通过，勿将两个计数相加宣称独立测试总数。

## 4. 本次模型到底改了什么

入口 `--architecture unified_gaussian_consistency`；架构身份 `OMEGA_UNIFIED_GAUSSIAN_INSTANCE_CONSISTENCY_V1`。

基于原实例一致性Unified：VGGT-Omega冻结特征/相机/点云 → ESAM式SAM区域辅助三维超点、3D特征编码与池化初始化唯一共享Query → 共享Decoder → Gaussian重建、类别、原Query–Gaussian相似度Mask；原渲染后实例8维辅助头仍保留。**没有叠加回采、边界RGB、软几何偏置，也没有新增参数或改推理输出/Mask/NMS/Panoptic。** 不要混淆早期错误的二维SAM池化初始化、双分支A和旧VGGT+MASt3R预训练理解分支；当前SAM用于三维超点分组，非二维mask直接池化query。

只新增训练期直接Gaussian约束：

- 两个context视图GT按实际alpha-compositing贡献转置到Gaussian；有效stuff也进入分母。
- thing纯度≥0.7、累计贡献质量≥0.001才可靠。GT归属/权重/几何/KNN选择全部detach；未归属/stuff/不可见不是错误标签。
- 8维实例向量单位化。同GT按父Token汇聚原型，每个Gaussian只拉近**其他父Token**的同GT原型；margin0.1。单父GT或相消原型跳过。Gaussian置信度加权→父Token等权→GT等权。
- 可靠thing中先排除同GT，找半径0.1×场景RMS内最多8个异GT邻居，负margin0.5；不拿远处负样本补齐。
- 新项重算原semantic_adapter/child_feature/instance_head，使用同一共享Query，但detach局部坐标里的Gaussian xyz、center、radius：数值与原8维相同，仅梯度路线改变。
- 新项更新实例/Child/语义及共享Decoder/GFM投影；不直接更新显式Gaussian、几何头、activation_head、reconstruction_adapter。**这不是冻结重建**，共享参数仍会间接改变重建；原RGB/Mask/渲染后实例项的几何梯度保留。

总loss：`Lrec + u*(Lunder_original + 0.01*Lrendered_instance + 0.01*(Ldirect_pull + Ldirect_push))`；u0.05时新增有效系数0.0005。配方完整存 `gaussian_consistency_recipe` 元数据。未将67050的256步终点作为新组初始化，而是同零步全新训练，以免改变对照。

## 5. 为什么做这一组：已有结果

**基线67050已完成，不能重跑。** 根目录 `.../unified_query_training_v1/attempt11_unified_instance_grounding_short_node13_4gpu`。4×3090、同16场景256步，约34分26秒。初始/终点均有4train+4heldout探针。

| 版本 | train PSNR | train原始Mask IoU | heldout PSNR | heldout原始Mask IoU |
|---|---:|---:|---:|---:|
| 基线67050 | 19.747 | 27.40% | 10.021 | 8.46% |
| 回采67128 | 19.678 | 28.61% | 10.076 | 7.45% |
| 边界RGB67129 | 19.780 | 27.64% | 9.898 | 9.53% |
| 组合67130 | 19.819 | 28.69% | 10.122 | 6.09% |

三变体工程/训练均通过，回采约91%可见、残差约6%，并非空通路；边界权重接近常数（边界/内部约1.0003），尚未验证有效双向增益。不能直接把组合当下一主线。以上只是4+4探针，原始IoU仅对初始化来源支持GT计分，**不是官方mAP/PQ，不是完整训练结果**；不能据此与旧完整数据PSNR约23比较。

后续固定权重边界定位 **67235已完成/0:0/3分39秒**：四模型32窗口、0参数更新，源权重/资产/同次渲染验证通过。

- heldout实例8维全局同异GT AUROC约0.660–0.748，shuffle约0.5；局部仅0.603–0.619。不是完全塌缩。
- 基线同GT同父余弦距离0.1067，同GT跨父0.5285，异GT0.7182；边界组0.1154/0.5477/0.6755，跨父一致性值得验证。
- 正常混合二维边界AUROC0.519–0.544；精确最大贡献Gaussian约0.494–0.529，没有恢复GT边界。不能简单归因“渲染混合平滑”。
- 原阈值0.15远偏离响应，但train-only校准到约4e-5–8e-5后heldout权重比也仅1.03–1.07；只改阈值不足。
- 全局特征统计仅3个多实例有效heldout场景；GT是贡献代理归属，GT one-hot上界覆盖不足；不能认定唯一根因。高纯度thing仅20%不意味着80%Gaussian归属错误。

因此先做实例专属8维跨父一致性＋局部异实例负样本；不要把所有几何/外观共享Token强压成同一向量。

## 6. 用户通知完成后，怎样检查

先核对67292调度最终状态/退出码/耗时，再读取运行根目录，不重训。建议先看：

- `JOB_FAILED.json`（如有）、`slurm-67292.out`：失败在哪个门禁/阶段，OOM/非有限/梯度/同步还是汇总脚本问题。
- `SMOKE_PASSED.json`：必须含新项 `direct_gaussian_validation` 及16窗口 `direct_checks`，不能把旧smoke当本组通过；两步四rankSHA一致、旧梯度保留、新项几何隔离、有真实正负样本。
- `FIRST_UPDATE_VERIFIED.json`、`STARTED.json`、`checkpoint_initial.pt`、`checkpoint_latest.pt`：256连续更新/四rank/Adam256、参数有限且实际变化；零步空Adam/初始化SHA与smoke零步及基线共同305张量相同，未继承smoke两步。
- `train_rank00.jsonl` 至 `train_rank03.jsonl`：每rank256行；新项有效系数、可靠GT/正父Token/负边覆盖、pull/push、正负距离、违反margin比例。不能只看loss下降，也不能无监督跳过就当学习正常。
- `SHORT_VALIDATION_SUMMARY.json`、`INSTANCE_SHORT_AUDIT.json`、`GAUSSIAN_SHORT_AUDIT.json`、`COMPLETE.json`。
- `PROBES_INITIAL.json`、`PROBES_FINAL.json`及已有探针缓存：与67050同场景/帧/输入、零步差异和终点增量；给量化证据和典型案例，同时看PSNR和原生Mask。新8维表征变好，不自动等于原Query–Gaussian Mask变好；若二者脱节，明确指出读出未获益。

训练成功但仅CPU汇总失败，应优先修复汇总、复用已完成产物，不重新训练。需要新的GPU诊断/扩展训练时先提出基于结果的建议，不能默认自动开展。用户需要明确结论：新约束是否活动、是否改善跨父/局部分离、是否转化为Mask收益、重建代价与下一步最值得验证方向；保留小样本限制。

## 7. 关键代码和文档索引

- 模型：`tokengs/models/gaussian_instance_consistency.py`
- 注册：`tokengs/models/superpoint_structural_variants.py`
- 训练：`scripts/train_unified_query_v1.py`
- 门禁：`scripts/smoke_unified_instance_grounding.py`、`scripts/gaussian_consistency_smoke_policy.py`、`scripts/validate_unified_instance_smoke.py`
- 预检/执行/提交：`scripts/preflight_unified_instance_short.py`、`scripts/run_unified_instance_short.sh`、`scripts/submit_gaussian_consistency_short.py`（硬编码attempt15，**不能再次运行来接手任务**）
- 汇总：`scripts/summarize_gaussian_consistency_short.py`、`scripts/summarize_unified_instance_short.py`、`scripts/summarize_unified_query_short.py`
- 测试：`tests/test_gaussian_instance_consistency.py`、`tests/test_gaussian_instance_consistency_ddp.py`、`tests/test_gaussian_consistency_smoke_policy.py`
- 实现细节：`docs/gaussian_instance_consistency_implementation_20261018.md`（其中“未授权GPU”是实现轮历史状态，已被本文件覆盖）
- 本组流程/收据：`docs/gaussian_instance_consistency_short_20261018.md`、`docs/gaussian_instance_consistency_short_submission_20261018.json`
- 前三组：`docs/unified_instance_coupling_short_results_20261018.md`
- 定位诊断：`docs/instance_boundary_localization_results_20261018.md`
- 历史总交接：`docs/codex_handoff_vggt_reading_controls_20261015.md`，顶部新记录优先，后面很多RUNNING/建议/未提交均属历史，不能据此恢复任务。

## 8. 给新对话的首条消息

> 请读取并接手 `/space/mawb/ssst_vggt_overfit_diagnostics_v1/docs/codex_handoff_gaussian_consistency_20261018.md`。沿当前67292进度继续，不重复训练或重提作业，遵守提交后停止轮询、等我通知完成再检查的约定。

若用户明确说67292已完成，则按第6节检查现有结果并汇报；如果只是接手，读取本文即可，不必为了“核实”重新轮询GPU作业。
