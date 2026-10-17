# VGGT 重建与理解实验交接：2026-10-15

> **最新授权两组结构短验证：A旧双分支交互/相似度读出＋双三维超点初始化，B共享Query＋直接三维Gaussian非线性Mask＋Gaussian语义。** 用户明确B采用三维Gaussian空间。A保留1024/256宽度和L6/8/10/12，S可变；B Query生成MLP参数对所有Gaussian几何/外观求Mask，另直接预测20类Gaussian语义，panoptic由实例/语义一致组装。各seed42新初始化/空Adam、同16训练场景、256更新/全局批8、weight0.05联合、当前多对一/NMS；18节点各4×4090并行。A1.938亿/B1468万，B新增0.1semanticCE，非容量匹配单变量。28 CPU＋两组2rank Gloo/两次更新SHA一致及资产/数据预检通过，native GPU未验证。新attempt03/04及独立ssst_superpoint_structural_pair_code_20261017；已有66432/33不查询不改、旧B不恢复。详见[结构实验](superpoint_structural_pair_20261017.md)，job以structural_pair_20261017/submission.json为准，提交后单次核对停止轮询。

> **最新完成分析：66353/66354/66355全部COMPLETED、正式128/11项有效。** 旧V4 learned-base＋SAM残差隔离4epoch4172步，训练4小时34分/评测11分；PSNR23.2936、AbsRel0.05458、target mAP18.2400%/PQ51.7436%。相对原V4隔离23.1940/18.9803%/53.6217%，重建保持、context mAP+1.50pp，但target mAP−0.74pp/PQ−1.88pp，理解无整体收益。门控0.100→0.098971，没有塌到0；没有执行本组SAM来源候选存活新审计，不能声称其淘汰问题已修复。详见[完成分析](gfm_sam_residual_completed_results_20261017.md)。归档为旧双流完成对照，本次未重跑、未提交新任务；Unified Query66432/66433仍沿前次提交记录等待用户通知，未额外查询或改动。

> **最新已提交，停止轮询：完整数据Unified Query训练66432／自动评测66433。** 2026-10-17 12:50:08，北京时间；单次核对训练RUNNING、11节点8×3090；评测PENDING(Dependency)、afterok:66432，同节点单卡，含固定128官方11项及旧C/V4已完成对照汇总。执行SHA09063a1／独立ssst_unified_esam_full_code_20261017，run为`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt02_esam_many_full_node11_8gpu`。1191场景8337窗口、4epoch4172更新；读取66414的0步/空Adam/8rank RNG，不是256步终点。新GPU首步/八rank同步尚未核实，旧短测成功不是新任务GPU收据。21项CPU/实际零步资产与数据预检通过。保留首末及epoch探针/epoch checkpoint；已停止轮询，用户通知再检查，不重提、不影响旧作业。详见[全量记录](unified_query_many_full4_20261017.md)。

> **最新授权：开始Unified Query完整数据4epoch。** 新attempt02_esam_many_full_node11_8gpu，11节点8×3090（18/14/17八卡已占用），1191场景/8337窗口、4172更新/全局批8。保持多对一完整Mask监督＋context NMS；官方Omega/SAM冻结，其他参数读取66414的**零步**权重/8rank RNG，空Adam，不继承256步拟合终点、不重复适配。LR1e-4/100步预热，理解weight0.05/1043步ramp，无隔离。独立ssst_unified_esam_full_code_20261017；初始/每epoch固定探针及epoch checkpoint，训练后依赖官方128/11项与已完成组比较。真实源/资产/数据/官方评分器预检及21项CPU测试通过，新的GPU首步尚未验证。详见[全量计划](unified_query_many_full4_20261017.md)，作业号以新run/submission.json为准；提交后单次核对停止轮询。

> **最新：66414已完成并核实，可训练性通过但泛化未通过。** COMPLETED/0:0、13分58秒、8rank各256更新；301可训练张量全部变化/Adam均256，首步8rank全参数SHA一致，无非有限值。4train探针PSNR10.935→19.721、理解loss2.078→0.676、支持GT最佳Mask IoU（GT等权）0.1345→0.2673；4heldout PSNR11.625→10.049、Mask IoU0.1151→0.0722。20训练支持GT对应172正Query，分类约100%，证明多对一真实生效，但复杂场景Mask和最终组装仍弱。16场景重复128遍的小样本拟合不是正式128/11指标结果。详见[完成分析](unified_query_many_short_results_20261017.md)；运行目录新增RESULT_AUDIT.json和probe_comparison.png。建议保持结构做完整数据训练，未提交新作业、不重跑66414、不恢复B、不改66353。下方RUNNING均为历史状态。

> **最新已提交并停止轮询：Unified Query多对一短验证66414。** 单次核对18节点8×4090 RUNNING；执行SHA6aa16a0，独立ssst_unified_esam_many_code_20261017。16固定训练场景/256更新/全局批8；前后4train＋4heldout探针及内置结束汇总，不是正式128评测，无全量自动续训。真实CPU权重/数据预检与21项检查通过；GPU首步、区域预算、CUDA反向及八rank SHA尚未验证。新run为`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu`，以submission.json和结束收据为准。保持66353/旧快照不动，B不恢复。等待用户通知再检查，不重复提交、不继续轮询。详见[运行记录](unified_query_many_short_20261017.md)。

> **2026-10-17最新授权：实现多对一完整实例监督，并提交Unified Query短训练验证。** 默认`source_many_to_one`：同一GT的多个合格超点Query均预测完整Mask，损失按GT平衡；context同类Mask NMS去重，身份复用于目标视图，不删除Gaussian。新attempt01_esam_many_short_node18_8gpu与独立快照ssst_unified_esam_many_code_20261017，18节点8×4090、固定16训练场景、256更新、全局批8、LR1e-4/20步warmup/短cosine、理解weight0.05/20步ramp；仅冻结Omega/SAM，其余全新，不加载旧checkpoint。训练前后各4train＋4heldout探针、首步全参数8rank SHA及梯度证据、结束自动汇总，不自动启动全量训练或声称官方128指标。20项CPU单测＋真实2rank Gloo测试、真实Omega/SAM权重/数据预检通过。GPU尚未验证；提交状态以新run/submission.json为准。旧66353不改、B不恢复。详见[短验证记录](unified_query_many_short_20261017.md)。

> **2026-10-17 最新指令：Unified Query默认采用ESAM式SAM超点生成。** 使用VGGT预测相机/深度对应的点云替代实测几何；SAM区域按面积降序写标签，小覆盖大；未覆盖点按ESAM MV的XYZRGB KMeans8补分组。采样保留每个有效区域，同区域内压缩点，禁止跨区域融合，保留像素→点→超点映射。3D邻域编码后池化为query，非二维SAM mask直接池化。现有SAM ViT-B与原版H不同，三维骨干仍为本项目MLP，未宣称完整ESAM复现。详见[实现说明](unified_superpoint_query_v1_implementation_20261017.md)。不提交训练，不恢复B，不修改既有作业；16项CPU测试、语法和diff检查通过；GPU真实SAM/渲染及预算尚未验证。

> **最新：用户授权的Unified Superpoint Query V1已实现，未提交训练。** 详见 [unified_superpoint_query_v1_implementation_20261017.md](unified_superpoint_query_v1_implementation_20261017.md)。独立注册siu3r_unified_query_v1：冻结Omega GFM→提升三维点/体素跨视图融合→3D邻域编码→超点几何池化→唯一共享Query序列/12层decoder→Gaussian与类别/跨Gaussian实例Mask头。超点首版为可替换的三维体素过分割，预留融合点上的外部IDs接口，不依赖SAM，不冒称ESAM超点生成器复现；默认最多512 Query/32768 Gaussian，容量不同于旧版。来源约束动态匹配仅在预测后读取GT；训练入口独立且understanding-weight必须显式提供。12项CPU测试、5例旧100-query组装逐位一致核对及编译检查通过，合成Omega/假渲染器不能替代真实GPU验证。尚无真实Omega+CUDA冒烟/训练结果，不自动提交。B已取消且不得恢复，66353及旧快照不动。

> **最新覆盖：用户明确停止B，66380/66381/66382均已CANCELLED，66353保持不动。** 原因是目标初始化应为三维点特征→超点特征→Query，而非几何辅助二维池化或当前SAM支持的更新Anchor插值。B日志/快照保留，200步完整checkpoint硬链接保留，收据attempt17/USER_CANCELLED_20261017.json。勿自动恢复B、重复提交或依据其未完成结果作性能结论；尚未授权新超点训练。详见 [gfm_updated_anchor_seed_experiment_b_20261017.md](gfm_updated_anchor_seed_experiment_b_20261017.md)。

> **并行结构实验B已提交66380，停止轮询。** 请求18节点8×4090，单次核对PENDING(Priority)，未宣称已分配或GPU通过；官方128评测66381 afterok，比较66382 afterany:66381:66354。执行SHA678a47c，新attempt17，详见 [gfm_updated_anchor_seed_experiment_b_20261017.md](gfm_updated_anchor_seed_experiment_b_20261017.md)。24项CPU检查和真实零步649张量预检通过。先重建→SAM支持汇聚detach终点Anchor→learned-base残差query→理解重放原四层，原多层重建梯度保留、完全隔离、4epoch4172步。66353及其快照不改，用户通知后检查新作业，勿重复提交或继续轮询。

> **2026-10-17新授权：并行结构实验B，更新后的Anchor初始化Object Query。** 见 [gfm_updated_anchor_seed_experiment_b_20261017.md](gfm_updated_anchor_seed_experiment_b_20261017.md)。新attempt17/独立ssst_gfm_anchor_seed_code_20261017，18八张4090；先重建12层，复用现有under投影对detach终点Anchor编码，SAM预测3D支持局部4近邻插值与原几何池化形成seed，保留learned base＋.1门控，随后理解重放保存的L6/8/10/12，不重跑重建。返回states保留原始每层重建张量/监督梯度；Write0/完全隔离/同零步649共享张量、空Adam、8rank RNG、4epoch4172、adapt/short0。24项CPU测试/真实源预检通过；未宣称GPU通过。66353保持原快照不动，比较等待本组评测和66354；作业号以attempt17/submission.json为准，提交后单次核对停止轮询。

> **SAM残差初始化隔离训练66353已提交，停止轮询。** 13节点8×3090单次确认RUNNING；官方128评测66354 afterok，比较汇总66355 afterany。新attempt16、执行快照SHA767cbe8，详见 [gfm_sam_residual_isolated_20261017.md](gfm_sam_residual_isolated_20261017.md)。14项CPU单测/真实649张量逐位预检通过；GPU首步与八rank同步尚未核实，待用户通知检查，不重提交、不改执行快照。

> **2026-10-17新授权：提交learned-base＋SAM残差初始化隔离训练。** 见 [gfm_sam_residual_isolated_20261017.md](gfm_sam_residual_isolated_20261017.md)。新attempt16，独立快照ssst_gfm_sam_residual_code_20261017；13八张3090，同零步649共享张量/空Adam/8rank RNG，新增1个gate参数。保留learned base，SAM内容/几何编码无仿射LN并校准到base RMS，以初始.1可学习门控加回；候选/c-s/交互/匹配/loss不改，完全隔离，4epoch4172更新、adapt/short0。14项CPU测试及真实源实体张量预检通过。训练成功自动官方128与V3/V4原隔离比较；作业号以attempt16/submission.json为准，提交后单次核对停止轮询。此前组已完成不重跑，GPU首步以新作业收据为准。

> **2026-10-17：65996/66039训练与全部官方评测、比较已完成；66092 GPU审计完成，末尾CPU汇总失败已仅CPU修复。** 详见 [gfm_completed_controls_and_initialization_audit_20261017.md](gfm_completed_controls_and_initialization_audit_20261017.md)。选择性组PSNR21.522/Target mAP21.203%/PQ58.437%，未保住隔离组23.194PSNR；V3隔离23.232/16.423%/52.683%，理解低于V4隔离。审计SAM eligible零步5619/5619→终点10/5623，GT匹配143→40；固定152合格SAM候选仍在，终点全部noobject，梯度非零且正匹配SAM梯度接近fallback。单独全局c/s未恢复，换embedding提高SAM资格但整体Final无收益。当前建议单变量learned-base＋门控SAM残差初始化，未实现/未提交。66092原Slurm FAILED和旧日志保留，恢复收据位于attempt01/CPU_SUMMARY_RECOVERY_20261017.json；GPU/训练均未重跑。下方等待状态均为历史，勿重复提交。

> **初始化专项审计66092已提交，停止轮询。** 18节点4×4090单次确认RUNNING，执行SHA245baf9，新gfm_initialization_audit_v1/attempt01；见 [gfm_initialization_audit_20261016.md](gfm_initialization_audit_20261016.md)。固定已完成0.05组0/1043/2086/4172 checkpoint，各128留出场景；固定先前152个合格SAM支持GT，按mask hash追踪候选而非query索引；记录投影/几何/形状分量、点权重、分层读取与noobject/匹配。终点native/替换embedding/全局c-s/两者联合的2×2固定权重干预，另4阶段各16训练窗口autograd.grad探针，无optimizer/参数更新。CPU来源SHA/配置校验和13项测试通过；Adam二阶矩另存历史活动证据（零值不证明从无梯度），固定权重干预不代表训练后成绩。先4场景含梯度冒烟才全量，GPU结果未核实。65996/66039不动；用户通知后检查，勿重复提交。

> **66047诊断已完成并审计，未提交新任务。** 详见 [gfm_instance_diagnostic_results_20261016.md](gfm_instance_diagnostic_results_20261016.md)。COMPLETED0:0/7:40，两模型各128场景492相同GT、更新0、参数版本不变、8份有界原生重复性收据通过。主要证据用0.05组：官方768帧767逐位一致另1帧差1像素；Raw Recall@.5 385→noobject资格297→竞争273→Final260，88个在资格阶段失去有效Mask；SAM槽5623中10eligible，官方420输出419fallback/1SAM，native匹配SAM40中39判noobject；小可见实例Raw36/112、Final13/112。不能将重复best-query的noobject计数直接当漏检，也不能说fallback没利用SAM间接信息。Ownership比shuffle更一致、Write0，无证据跨实例写坏。隔离组原官方packed跨运行复现有差异（平均1.61%、最坏帧50%，同inode权重），原Final262 vs诊断259，原因未定，仅辅助，不改原分数。新增CPU审计与两张典型图；建议Mask支持分类读回，等V3对照判断初始化因素，未实施。旧失败66042保留。

> **诊断66042失败，已仅推理恢复66047，停止轮询。** 详见 [gfm_instance_diagnostic_recovery_20261016.md](gfm_instance_diagnostic_recovery_20261016.md)。66042在4场景冒烟因逐位完全相同断言拒绝约1e-6的Membership差异而FAILED1:0，完整128尚未开始，不能报告诊断结论。新attempt02/独立恢复快照SHAcafad90，18四卡先测两次无观测器原生重复误差，再用有固定上限的经验波动界验证观测器，并核对最终packed分割；6项CPU测试通过，GPU证据待作业产出，不能提前称原生非确定性已证实。模型/训练/官方协议未变，66039及65996和旧快照不动。旧失败保留，无训练重跑。用户通知后检查66047，不重复提交。

> **V3初始化隔离对照66039＋18四卡诊断66042已提交，停止轮询。** 见 [gfm_v3_isolated_and_parallel_diagnostics_20261016.md](gfm_v3_isolated_and_parallel_diagnostics_20261016.md)。新attempt15，637共享零步张量/空Adam/8rank RNG，去掉V4几何池化12张量29505参数，直接4epoch4172完全隔离。训练可用13/14/17/18的单节点八卡；66040 afterok官方128评测，66041 afterany对照汇总。独立66042在18四卡，对已完成V4隔离/整体0.05各128场景做候选/分层Query/Anchor-Child/组装诊断＋shuffle负对照＋失败图，4场景无更新冒烟通过才全量；不依赖新训练。执行SHA63bf675。真实源预检和19项CPU测试通过；单次核对两新任务PENDING(Priority)，未宣称GPU通过。65996不改不停止，待用户通知后检查，勿重复提交。

> **选择性梯度对照已提交65996，停止轮询。** 13节点8×3090单次确认RUNNING；官方128评测65997 afterok:65996，六模型比较65998 afterany:65997。见 [gfm_v4_selective_routing_node13_20261016.md](gfm_v4_selective_routing_node13_20261016.md)。重建梯度=dLrec+0.05×beta×dLunder，理解梯度=beta×dLunder；同零步初始化/空Adam/8rank RNG/4epoch4172/LR/clip1，未改前向结构、不重复适配。新attempt14及不可变快照ssst_gfm_v4_routed005_code_20261016，执行SHA0b92add。真实源预检及13项CPU测试（含真实双rankGloo）通过；GPU首步尚未核实，不能把RUNNING当反向验证。等待用户通知后检查，不重提交。

> **65831/65838及全部评测、比较已完成，结果核实。** 详情见 [gfm_v4_isolated_vs_weight005_results_20261016.md](gfm_v4_isolated_vs_weight005_results_20261016.md)。两组相同0步起点、4epoch4172更新，官方128/11项有限。隔离组PSNR23.194/AbsRel.05463/target mAP18.98%/PQ53.62%；权重0.05组21.910/.12436/19.85%/58.79%；普通joint13.208/.22148/7.70%/46.13%。优化耦合与loss尺度都重要，架构具备正常学习能力；隔离重建最好、0.05理解更好，均未追平旧C实例mAP。未测梯度cosine，不能唯一归因方向冲突；globalclip1仍耦合。下一步仅建议分开理解自身梯度与流向重建的梯度强度，未实现或提交新作业。下方RUNNING/等待通知均为历史状态。

> **权重0.05普通联合已提交65838，停止轮询。** 18节点8×4090已单次确认RUNNING；官方128评测65839 afterok:65838；普通joint/隔离joint/权重0.05joint/冻结/旧C比较65840 afterany:65832:65839。新attempt13_geometric_sam_weight005_node18_8gpu，执行SHA06bed1b。相同0步未训练权重/空Adam/8rank RNG，under有效权重0.05×首1043步ramp，正常DDP总loss反向，重建不冻结。25项CPU检查及真实起点预检通过；GPU首步未宣称通过。65831首GPU更新全参数8rank一致性已验证，本组不修改或重启它。

> **新实验：18节点八张4090，V4普通joint理解loss权重0.05。** 见 [gfm_v4_weight005_joint_node18_20261016.md](gfm_v4_weight005_joint_node18_20261016.md)。同0步随机初始权重/空Adam/8rank RNG，直接4epoch；理解有效权重0.05×首epoch ramp，普通DDP联合反向，重建不冻结、不隔离。与普通joint65761和隔离65831比较。真实起点预检/25项CPU检查通过；硬件4090与13节点3090不同。65831资源核实时约150步、实际GPU首步全参数8rank SHA一致收据已确认，无失败，不改其快照。新attempt13及独立快照，提交后单次核对停止轮询。

> **梯度隔离V4从头joint4已提交65831，停止轮询。** 13节点8×3090已单次确认RUNNING；官方128评测65832 afterok:65831；对比普通joint/隔离joint/冻结/旧C的65833 afterany:65832。新attempt12_geometric_sam_isolated_node13_8gpu，执行SHAa54fcd3；65761的0步随机权重/空Adam/8rank RNG作为共同起点，未继承训练后模型。28项CPU检查（含真实双rank Gloo）及真实起点预检通过。GPU首步及八卡参数一致性未宣称通过，以ISOLATED_GRADIENT_SYNC_FIRST_UPDATE.json为准。数据、LR、warmup、4epoch、全局clip1与普通joint一致，仅梯度归属变化，前向不改。新增固定验证深度零值监测。不再轮询，用户通知后核对结果。

> **新授权：V4从头joint4梯度隔离对照。** 见 [gfm_v4_isolated_joint_node13_20261016.md](gfm_v4_isolated_joint_node13_20261016.md)。与65761保持初始权重/8rank RNG/数据/学习率/warmup/clip/4epoch一致，仅重建参数接收重建loss、理解参数接收理解loss；复用的是0步空Adam初始状态，不是训练后checkpoint。普通DDP不支持该autograd.grad用法，本模式显式FP32 SUM/world及全局unused语义，首步检查全参数跨rank SHA。两进程CPU Gloo验证已通过；GPU同步以ISOLATED_GRADIENT_SYNC_FIRST_UPDATE.json为准，不能仅称RUNNING通过。新attempt12_geometric_sam_isolated_node13_8gpu及独立快照；提交后单次核对停止轮询。

> **V4从头联合最终完成但质量未达标。** 65761/65762/65763全部COMPLETED；4172步、adapt/short0、官方128共11项有限。PSNR13.208、AbsRel.22148、RMSE.45588、target mAP7.70%/PQ46.13%，落后V4适配2后冻结（21.971/.06695/.19633/19.87%/55.45%）。详见 [gfm_v4_fresh_joint_results_20261016.md](gfm_v4_fresh_joint_results_20261016.md)。无新训练提交；建议单变量梯度隔离仍未实现。不要重跑已完成训练或将质量失败称程序崩溃；对照预算与初始化不同，不足以单独证明唯一原因。

> **V4从头直接联合已提交：65761，停止轮询。** 13节点8×3090已单次确认RUNNING；官方128评测65762 afterok:65761；四模型比较65763 afterany:65666:65672:65762。新attempt11_geometric_sam_direct_joint_node13_8gpu，执行SHA5a3ff5f。source_checkpoint=null，adapt/short更新0，196010160可训练参数新初始化/空Adam，只加载官方冻结Omega/SAM；直接联合4epoch4172步，重建/几何/理解LR1e-4/1e-5/1e-4，理解loss首1043步warmup，无梯度隔离。25项CPU检查及官方资产/数据预检通过，GPU首步未宣称通过。适配2后续训草案未提交，不得误用旧草案目录的preflight。65665/65671保留，65661已取消不恢复。

> **用户最新纠正：V4从头直接联合4epoch，不微调适配。** 见 [gfm_v4_fresh_direct_joint_node13_20261016.md](gfm_v4_fresh_direct_joint_node13_20261016.md)。仅官方冻结Omega/SAM，其余全部新初始化，Adam无历史，不继承任何旧checkpoint，adapt/short更新0；重建/几何/理解LR=1e-4/1e-5/1e-4，首epoch理解loss warmup。13节点八卡新attempt11_geometric_sam_direct_joint_node13_8gpu；适配2后续训草案尚未提交且已取消。保持65665/65671，65661不恢复。作业以新submission.json为准，提交后单次核对停止轮询。

> **2026-10-16 11:11 用户提出停止退化的降低LR+warmup对照，已发送取消。** 仅停止65661及失效依赖65662/65663/65667，保留V3冻结65665/65666、V4几何冻结65671/65672及二者比较65673。最新固定8验证PSNR初始21.482→2600步15.510，未把这组称作完成或程序崩溃。2650步完整checkpoint已硬链接保留为attempt08/checkpoint_stopped_warmup_control.pt，收据STOPPED_WARMUP_CONTROL.json。后续优先建议隔离理解loss到重建参数的梯度，同时允许重建由自身loss继续优化；尚未实现或提交该新策略。当前固定8 PSNR不能直接与论文/128场景官方PSNR比较。

> **V4 几何感知初始化八卡训练已提交，停止轮询。** 训练65671已单次确认RUNNING，18节点8×4090；官方评测65672 afterok:65671；与V3冻结组/旧C比较65673 afterany:65666:65672。执行SHA175418b，新attempt10，源适配2已完成、637参数张量/637 Adam状态按名保留，新增29505初始化参数，冻结重建理解4epoch。23项CPU检查及真实源重建指纹预检通过；GPU首步尚未宣称通过。未改65661/65665。额外GitHub推送首次被自动审批拒绝；用户随后明确授权向 https://github.com/fufuforu/ssst.git 推送，允许继续推送本轮代码。

> **新授权：18节点八张4090训练V4几何感知初始化，选择冻结重建理解4epoch。** 见 [gfm_geometric_frozen_node18_8gpu_20261016.md](gfm_geometric_frozen_node18_8gpu_20261016.md)。同一适配2起点/批8/理解LR及warmup与65665对照，仅改初始化；637共享tensor及637个Adam状态按名迁移，新增29505参数。真实源预检与重建指纹一致性通过，23项CPU检查通过。新attempt10_geometric_sam_frozen_node18_8gpu及独立代码快照；训练/评测/比较job以submission.json为准。不得重复适配或修改65661/65665，提交后单次核对并停止轮询。

> **2026-10-16 新结构代码：仅初始化改为 ESAM 风格几何感知池化，独立 V4。** 见 [gfm_geometric_sam_pool_v4_20261016.md](gfm_geometric_sam_pool_v4_20261016.md)。区域内 VGGT 相对几何预测权重来池化 F_sem，并加整体形状编码；候选规则、交互层、Write 和 Gaussian 输出保持原设定。没有提交训练，没有迁移旧 checkpoint，不改 65661/65665 的不可变快照；旧 checkpoint 不能直接当新版本 resume。验证结果以该文档记录为准。

> **冻结重建八卡对照已提交，停止轮询**：训练65665（11节点8×3090，单次核对RUNNING）；官方评测65666 afterok:65665；五模型比较65667 afterany:65662:65666。代码1ee41b0，attempt09_frozen_recon_node11_8gpu。与13节点65661同适配2起点/batch8/理解LR及首epoch损失warmup；仅冻结重建参数/优化器/模式，并验证固定场景Gaussian不变。14项CPU验证通过，源预检通过；GPU首步/冻结指纹尚未宣称通过。13节点原warmup不受影响。用户通知后检查两组及评测，不重提交、不重做适配。

> **新授权：11节点八卡增加冻结重建、只训练理解对照，13节点warmup65661保持运行。** 见[gfm_frozen_reconstruction_node11_8gpu_20261016.md](gfm_frozen_reconstruction_node11_8gpu_20261016.md)。同一适配2checkpoint与batch8，理解LR1e-4及首epoch loss warmup与65661一致，只冻结重建参数/模式/优化器更新，并检查Gaussian输出不变。新attempt09_frozen_recon_node11_8gpu，14项CPU检查及源预检通过，训练理解4epoch，接官方128评测与五模型比较。提交后单次核对停止轮询；不能重复适配、不取消65661、不将RUNNING当GPU冻结保证。

> **warmup八卡链已提交，停止轮询**：训练65661（13节点8×3090，单次核对RUNNING），官方评测65662 afterok:65661，四模型比较65663 afterany:65662。执行SHA32da5ef；新attempt08_joint_warmup_node13_8gpu，source为适配2完成checkpoint，sha256 2b07922b1b2c3e8f5dfa6c1591951e043bc418874f2cea379588003ddde6f3e3。重建/几何/理解LR=1e-5/1e-6/1e-4，理解beta首1043步线性升1，新的联合4epoch；global batch8不变，4→8rank RNG映射已记录。14项CPU检查和源预检通过；仅资源分配已确认，尚未宣称新GPU首步通过。用户通知后检查新链，勿重跑已完成适配或重复提交。

> **新授权已实现：用户选择warmup并要求13节点八卡。** 详见[gfm_joint_warmup_node13_8gpu_20261016.md](gfm_joint_warmup_node13_8gpu_20261016.md)。从现有适配2epoch完成checkpoint继承权重/AdamW，新增联合4epoch；重建LR1e-5、几何1e-6、理解1e-4，理解loss beta在1043步从接近0升1，原100步LR warmup保留。不重复适配、不用坏掉的joint4终点、不改结构/冻结/Write/官方协议。13节点8×3090，global batch仍8。新attempt08_joint_warmup_node13_8gpu，14项CPU检查及真实源预检通过。作业号以submission.json为准，提交后单次核对并停止轮询；不能把RUNNING当八卡反向证明。

> **2026-10-16最新：两组训练全部完成，结果未达标。** 65251直接joint4/4172与65252 adapt2/2086+joint4/4172均COMPLETED；128场景官方计算和导出均完成，65256/57因原生AbsRel/RMSE NaN在严格JSON汇总时失败，65258取消。已有原始results无需重算，已恢复9项有效指标＋2项显式无效NaN对比。报告[gfm_joint4_vs_adapt2_joint4_results_20261016.md](gfm_joint4_vs_adapt2_joint4_results_20261016.md)：官方PSNR直接12.826、分阶段12.496，旧C22.597；target mAP6.91%/5.79%，旧C27.38%。固定8验证适配2后PSNR21.486→联合1时13.271→联合4时12.704，首要证据指向联合阶段重建退化。直接4/768、分阶段2/768全零深度导致官方NaN；不能删帧/改协议伪造有限指标。比较表/曲线/失败图已保存。所有作业结束，无新训练/推理/评分提交；后续冻结重建只训练理解的控制仅为建议，文本分支亦未实施。

> **恢复链已提交，停止轮询**：65251直接组从全量250 checkpoint续训（11四3090，单次核对RUNNING）；65252 afterany:65236条件复用成功或仅恢复已知缓存错误，不中断健康65236；最终正式评测65256 afterok:65251、65257 afterok:65252，比较65258 afterok:65256:65257。旧65238及后续待执行65253/54/55均已替换；新launcher2cece59修复Slurm spool目录，训练/评测Python快照75f6ce1。源状态完整、无已完成更新丢失，但尚未宣称新GPU首步通过；用户通知后检查新链，勿重播250步或恢复65199。

> **65199已FAILED：SAM共享缓存发布FileExistsError，不是NaN/OOM；全量250步已完整checkpoint。** 修复与恢复见[gfm_cache_publication_failure_recovery_20261015.md](gfm_cache_publication_failure_recovery_20261015.md)。直接组新attempt07从251步继续，源optimizer/4rank RNG保留，不回到306短验证。65236当前健康、GPU冒烟及首个适配更新已通过，不能中断；为其设置afterany条件恢复/成功复用，再重接两组评测与对比依赖。旧65237/65239已因原parent失败取消，旧65238将替换，不能据旧submission重复提交。

> **已完成本轮提交，停止轮询**：新训练65236（17节点四4090，冒烟→adapt2→joint4），原组评测校验65237 afterok:65199，新组官方评测65238 afterok:65236，统一表65239 afterok:65237:65238。执行快照3608570；单次核对65236 RUNNING，其余Dependency，尚未宣称GPU冒烟/适配更新通过。收据在attempt06_adapt2_joint4_node17/submission.json。用户通知后再检查这条链，不重复训练/提交。

> **最新授权：保留65199直接联合4epoch，再独立提交2epoch重建适配＋4epoch联合，并设置训练成功后自动官方评测/统一13列11值比较。** 详见[gfm_semantic_pool_v3_staged_comparison_20261015.md](gfm_semantic_pool_v3_staged_comparison_20261015.md)。新组attempt06_adapt2_joint4_node17、17节点四张4090，共享短306起点，不使用direct组后续权重，不重复短训练。Omega/SAM仍冻结；适配时理解loss不反向且AdamW跳过理解参数，随后联合4epoch。先一场景GPU导出/官方评分冒烟，通过才训练。评测用afterok依赖；旧65199已有内置评分，依赖任务只校验并复用。提交后单次核对并停止轮询。

> **已提交全量续训65199**：attempt05_full_node11，执行SHA b403e7c，11节点四张3090；提交后单次调度核对PENDING/Priority，随后停止轮询，尚未宣称首个全量更新。用户通知后检查该作业，不要重复提交。下段“核对真实首个全量更新后停止”以此处单次核对后停止为准。

> **最新用户覆盖：短验证只需证明学习，不要求组合损失固定下降。** 按 [gfm_semantic_pool_v3_full_continuation_20261015.md](gfm_semantic_pool_v3_full_continuation_20261015.md) 使用显式learning_evidence_v2策略，从65175的306步权重/AdamW/4rank RNG原地时钟继续，直接进入全量4epoch；不重跑短验证、不修改模型/loss。新运行attempt05_full_node11，11节点四张3090。提交后核对真实首个全量更新并停止轮询。以下不放行建议已被用户该决定覆盖，原失败结果仍保留。

> **最新实测：65175已完成256次v3新结构短更新，累计时钟306；9/11门槛通过，重建/理解损失下降两项失败，保存checkpoint后主动停止，全量更新0。** 见 [gfm_semantic_pool_v3_short_results_20261015.md](gfm_semantic_pool_v3_short_results_20261015.md)。模型确实学习，但存在BCE上升、梯度尖峰及小样本分类泛化问题；不能称全量完成，也不能盲目恢复失败gate或重新从50步训练。当前检查未提交新作业。用户最新资源约束为仅11节点有四张卡，后续优先3dimage-11四张3090；提交后仍停止轮询。

> **最新覆盖：语义Adapter后池化 v3**。用户要求理解query由SAM池化F_sem而非原始GFM；代码/梯度与65174启动失败修复见 [gfm_semantic_pool_v3_implementation_20261015.md](gfm_semantic_pool_v3_implementation_20261015.md)。65174在迁移收据重复字段处失败，训练更新0，日志保留。新运行`/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt04_semantic_pool`仍从原50步共享权重/AdamW迁移，18节点4卡，短验证通过后全量4epoch。以该运行submission/STARTED/FIRST_UPDATE为准，不恢复已取消65167或失败65174。

> **最新目标已明确实现为GFM双Adapter v2**：用户确认Omega GFM分别供重建/语义Adapter，SAM先池化原始GFM初始化query，旧TokenGS encoder移除。见 [gfm_dual_stream_v2_implementation_20261015.md](gfm_dual_stream_v2_implementation_20261015.md)。65167的50步已安全保存，旧作业因目标结构变化停止；新运行 `/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt03_gfm_v2` 固定18节点4卡4090，迁移共享权重/AdamW，新增投影新初始化。新结构另做128/最多256更新的短验证，通过才全量4epoch。不要恢复旧2卡作业或重播50步。提交后停止轮询。

> **当前覆盖授权：用户同意双流短验证通过后直接进入完整数据训练。** 方案及转阶段条件见 [dual_stream_training_plan_20261015.md](dual_stream_training_plan_20261015.md)。冻结Omega/SAM、Write=0；短验证128/最多256更新，满足预设正常学习条件后接原1191场景/8337窗口全量4epoch。使用新独立训练运行，旧模型不重跑。作业与代码快照见 `/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/submission.json`；启动后停止轮询。下文“未授权正式训练”是此授权之前历史状态。

> **终点更新（2026-10-15）：四组300更新/2400曝光、128场景官方评测及64855汇总已全部成功完成。**
> 最新结果与补充审计见 [`vggt_reading_controls_results_20261015.md`](vggt_reading_controls_results_20261015.md)。
>
> **后续授权的固定权重Ownership诊断也已完成（64963 COMPLETED 0:0）：C/Baseline各128窗口、492个配对GT实例、新训练更新0。** 最新结论、失败恢复与典型图集见 [`vggt_ownership_results_20261015.md`](vggt_ownership_results_20261015.md)。不要重复提交推理或训练。本轮没有授权实施所建议的新结构；用户要求今后提交作业后停止轮询，等其通知完成再检查。
>
> **最新设计方向（用户后续已澄清）**：不直接复用QuerySplat，改为几何驱动的重建token＋实例/语义驱动的理解token，考虑VGGT点云初始化与SAM/其他模型候选初始化。新可编辑图和差异说明见 [`dual_stream_architecture_edit_notes_20261015.md`](dual_stream_architecture_edit_notes_20261015.md)。旧 [`querysplat_vggto_panoptic_design_20261015.md`](querysplat_vggto_panoptic_design_20261015.md)保留为历史方案。目前只是设计，未实施或提交新作业；SAM探针也尚未执行。不要自动下载/切换底座或启动长期训练。
> **最新覆盖授权**：用户已明确要求实现双流具体代码，并提供 Omega 路径、授权查找/下载 SAM。新增模型及CPU验证见 [`dual_stream_implementation_20261015.md`](dual_stream_implementation_20261015.md)，旧模型和已完成实验不重跑。本轮仅代码与真实权重无更新冒烟，未授权正式训练。GPU冒烟状态以 `/space/mawb/ssst/group_plus/dual_stream_v1_smoke/attempt01/` 为准；提交后停止轮询，等用户通知再检查。上方“仅设计、不可下载”是此前状态，已被此次用户授权覆盖。
> 下文08:11运行/排队状态保留为历史现场，不能据此重新提交或恢复任何训练。当前没有授权新的交互层对照、长期训练或干预叠加。

这份文档用于新对话直接接手当前工作。**先读当前 worktree、运行记录和 Slurm；沿实际最新进度继续，不回退、不重新设计、不重复提交训练。** 当前已授权的阶段是四组 decoder 读取短对照，等待全部完成后检查、比较、分析。没有授权自动开始新的长期训练或叠加新结构。

## 1. 给新对话的开场指令

可以直接把下面这段发给新对话：

> 请接手执行 `/space/mawb/ssst_vggt_overfit_diagnostics_v1/docs/codex_handoff_vggt_reading_controls_20261015.md`。先核实指定 worktree 的最新代码、运行记录和 Slurm 状态，沿当前进度继续，不重新设计实验、不回退或重复训练。本轮 geometry_quality_policy=monitor_v1，普通工程问题自行定位、修复、验证、提交推送和精确续跑；保留历史失败。当前四组均已有正式 20 更新/160 曝光的真实启动证明，不需要持续轮询等训练结束。若已全部完成，审计并按 SIU3R 官方 13 列比较分析，两个 mIoUt 空白，不另拆三套视图表；综合重建与理解指标判断下一步，不只看 PSNR。

## 2. 当前仓库与权限

- 仓库：`https://github.com/fufuforu/ssst`。
- **当前 worktree：`/space/mawb/ssst_vggt_overfit_diagnostics_v1`**。
- **当前分支：`vggt-overfit-diagnostics-v1`**。
- 写交接前 HEAD：`03a4f45`，已推送；本文档会再产生一个仅文档提交。接手时重新运行 `git status`、`git log` 和远端核验。
- 当前 GPU 执行的不可变快照：`649ef09aef473128849539baf7453654033ce281`。worktree 后续文档提交不改变正在运行的代码。
- 不要在运行中的 snapshot 上修改文件或 checkout/fetch。
- 用户已授权普通工程修复、受影响验证、提交推送、重试及恢复，不逐项等批准。保持模型、数据、loss、训练数学锁定，不能跳过窗口、隐藏 nonfinite、引入 GT/target 生成输入或悄悄改 batch/精度来绕过问题。
- 用户原始附件：`/space/mawb/ssst/docs/codex_vggt_posefree_autonomous_recovery_prompt.md`。其中 monitor_v1 和自主工程恢复仍有效；**旧起点 8502d978、旧 worktree、旧冻结范围和最初八轮训练不是当前实验配置，已被后续明确授权的方案更新。不要照旧附件重启最初实验。**
- `monitor_v1`：重投影 median/p90、正 Z 参考阈值超标只记质量告警，不因此停训；数值/资产/输入契约等硬错误仍要定位。旧 strict 失败不得改写成成功。
- 用户反复要求确认真实启动后停止轮询。当前四组都已有正式 20 更新/160 曝光与八 rank 同步证明，这个要求已完成；无需在新对话重新持续守候。

## 3. 最新现场状态（2026-10-15 08:11，Asia/Shanghai）

以下是写文档时真实读取的状态，不是未来保证；接手应再次只读核实。

证据 root：

`/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04`

| 分组 | 最新有效作业 | 核实状态 | 正式进度 |
|---|---|---|---|
| baseline 原版联合训练 | 64852 | RUNNING，3dimage-13 八张 3090 | 98/300 更新，784/2400 曝光；从 20 精确恢复 |
| object_local_read 对象局部读取 | 64717 训练/导出；64851 补评分 | 训练及 128 场景官方评测均完成；64851 COMPLETED 0:0，用时 2:01 | 300/300，2400/2400 |
| anchor_visual anchor 局部视觉证据 | 64853 | PENDING，Resources | 已有 20/300、160 曝光，续跑尚未启动 |
| reliable_depth 可靠深度约束 | 64854 | PENDING，Priority | 已有 20/300、160 曝光，续跑尚未启动 |
| 四组自动汇总 | 64855 | PENDING，Dependency | 等待上述三组续跑/评测成功 |

64852/64853/64854 申请同型 8×3090、单节点、32 CPU、384G、4h，partition=3090，exclude=3dimage-12，因此可用 11 或 13，不再锁死 11。三组之间无无关串行训练依赖；它们共同的补评分前置 64851 已成功。当前 user 授权优先用空闲八卡，不能把本轮 microbatch1/global8 和此前 A/B 的 global16 混为一谈。

基线记录的约 3.92 秒/更新对应剩余纯训练约 13.2 分钟；**这不包括初始化、排队、checkpoint 保存、导出和评分，不能当成总完成 ETA**。排队组不能给确定结束时间。

### 已完成局部读取组的真实 13 列（仅当前 128 场景短对照）

| AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUs↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUt↑ | 新视图 mIoUs↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUt↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.08657|0.2115|22.07|0.7725|0.3745|0.5515|0.2975|0.6321||0.5490|0.2625|0.6072||

这是独立留出 128 场景各一窗口、300 步后的成绩，尚没有同预算 baseline 终点，**暂不能判断局部读取有没有增益，更不能拿它与完整 312 场景成绩直接作差**。

## 4. 当前四组究竟在做什么

完整设计及历史执行记录：`docs/vggt_reading_controls_v1.md`。以下是接手必需的固定设置。

### 共同起点 C

命名需辨清：历史也有 A/B/C 等字母。**本轮 C 是“完整 adapter0.01/feedback0.05 模型 + 分类对象支撑读回补丁”**，不是旧 feedback factorial 中某个也叫 C 的短对照。

完整底座 A：先重建适配 2 epoch，再冻结 VGGT 联合理解/重建 4 epoch，adapter 理解梯度路由系数 0.01、feedback 峰值 0.05。

```text
/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/checkpoint_frozen_joint_epoch_04.pt
SHA256 ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0
```

C 分类对象支撑读回补丁（A 原核心冻结，仅分类 LN/head 及对象支撑读回训练 300 步得到）：

```text
/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/training/classification_patch_update0300.pt
SHA256 ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9
```

本轮四组都在同一 C 上进行真正的下游联合短训练。保留 C 读回原有 detach 行为，不暗中改分类梯度连接。

### 共同训练合同

- 每组 300 正式更新/2400 新窗口曝光；8×3090，microbatch1、accumulation1、global batch8。
- 使用完整 8337 窗口池中的固定 **2400 个不同训练窗口**，覆盖 1075 场景；不是在 128 场景反复训练。四组窗口顺序一致，update i/rank r 取 stream[i*8+r]。
- 训练池共 1191 场景。留出为固定 128 不同场景各一窗口，场景与训练无重叠。不是完整 312 场景/1860 窗口验证。
- **VGGT 整体和 MASt3R image encoder 都冻结/eval**；其余按 native 原 requires_grad + C 分类读回规则训练，所有 BatchNorm 保持 eval。不要误认为本轮只冻结 DINO。
- 原重建 loss + 完整 V3 set loss、matcher、辅助层 loss 均不变。理解总权重 1。
- GS 与 VGGT memory adapter 的理解梯度系数均为 0.01。这是模块梯度路由，**不是把全局理解 loss 乘 0.01**。
- feedback 峰值 0.05、双向结构及 6/8/10/12 交互保留。成熟源联合曝光时钟 35776，新训练计数另记 2400，不重启成熟任务的理解/feedback warmup。
- 短阶段起点 fresh native AdamW：原 decay 分组，beta .9/.95，eps1e-8，clip1；LR 原峰值 rec1e-5/理解1e-5/object_memory1e-4，warmup200，原 cosine4172 截断到300。
- 四组参数集合、容量、共同起点完全相同，新增可训练参数 **0**。
- 每组先保存 20 步 checkpoint，统一验证后 **恢复该 optimizer/RNG/采样计数** 到300；不是在续跑时重新 fresh，也不重复前20步。
- 三种干预强度前20次更新按 min(completed_new_updates/20,1) 开启，之后固定。零强度必须复现原 C。

### 三种独立干预及基线

1. **baseline**：保持原计算，只增加观测，不改输出。
2. **object_local_read**：只改变 object query→anchor CA 的对象读取偏置。L6 原样，L8/10/12 使用**上一交互层**预测 anchor soft mask（detach），增加 λ·log(0.1+0.9·mask/max)，λ峰值0.5；空/常量支撑及 stuff query 保持全局回退。原 anchor 内容、Q/K/V/O 和 feedback 方程不改，但后续 anchor 可经 query/feedback 间接受影响。没有 GT 或最终 mask 预知。
3. **anchor_visual**：保留原 CA 方程，把已有 Fm 在 anchor μ 投影到两个预测 context 相机后双线性采样，有效视图平均，无效为0。归一化残差在峰值不超过 anchor norm 的2.5%；visual λ峰值0.1。不加 encoder/投影参数。K/V 内容变了实际 attention 也可能变，不能称为固定 attention 权重的纯 V 实验。
4. **reliable_depth**：改变 **GS anchor→VGGT patch CA**，与第二组不是同一处 CA。保留点到无限射线的垂直距离偏置，额外加入 λ·reliability·clamp(-0.5·沿射线深度偏差²/radius²,-20,0)，λ峰值0.1。复用已有 predicted points/confidence，14×14池化与 patch rays 顺序对齐；每视图 q20/q80 归一化 confidence，退化常量为0。不是概率标定，不加载 point_head，不引入 GT depth 或新 loss。

强度预先固定，不在留出集调参。无增益只否定本次实现/强度下的收益，不能直接否定整个研究方向。

## 5. 当前结果怎样评测和判断

- 固定原 SIU3R 官方仓库 `/space/mawb/SIU3R`，commit `8ea80166be76854f938e90521f1a5b688b755c87`，不修改 evaluator 数学。
- 只用 SIU3R 表格 **13 个指标列**：5 个重建 + 输入4项 + 新视图4项；两个 mIoUt 空白。不要另做 context/target-all/true-novel 三套表。
- “新视图”沿用官方 target 集合（2输入帧+4额外帧），不能悄悄改成仅4额外帧。深度按官方 scale/shift 对齐。
- 每组训练300完成后自动八卡导出128窗口，再官方评分；mAP/PQ/mIoU 用官方状态全局合并，**不平均各 rank AP**。八rank分别验证真实 CPU/GPU 状态合并合同。
- 所有组相同 input/frame IDs/GT、相同训练流；summary 自动核验。记录实际每头 V norm、完整 O 输出 norm、每头 attention 熵，不能拿 mean(R)@原anchor 的代理宣称完整 CA 失效。
- 本轮 **不使用旧 Q**。结构变了输入分布也变；若后续加 Q，应各组等预算重新拟合，属于后续单独决定的工作。
- 综合重建、语义、实例、PQ及候选/匹配诊断判断，不能只看 PSNR 或数“多少项上涨”。不要把固定训练窗口拟合成绩当正式泛化结果。
- 若四组出现不同取舍，先报告作用位置和证据，再提出下一阶段；不未经讨论自动叠加干预或开4epoch。

最终自动产物在 attempt04：

```text
{arm}/training/progress.json
{arm}/training/checkpoint_latest.pt 和 checkpoint_latest.json
{arm}/training/startup_confirmation.json
{arm}/training/run_manifest.json 和 resume_manifest.json
{arm}/training/training_rank00.jsonl ... training_rank07.jsonl
{arm}/evaluation/metrics.json、siu3r_table1.csv、siu3r_table1.md、COMPLETE.json
{arm}/evaluation/matched_query_diagnostics.json
analysis/report.md、siu3r_comparison.csv、results.json
COMPLETE.json
```

## 6. 为什么走到这一步：已测量的主要结论

### 6.1 不能把指标差简单归因于训练不足

旧 adapter 系数1的适配2+冻结联合4：完整 PSNR19.37；再冻结联合2降到19.14；再解冻 AA 联合2为19.35，没有全面改善。后两组 global16/fresh optimizer 与源 global8 历史有差别，不能把变化全部归因训练时长。

AA 确实更新过，非冻结实现错误。低预算 AA 冻结/解冻对照也未得到清晰综合收益。新 GS 从零在100步预算明显更差，支持当前继续沿用旧 GS 初始化；**不能据此断言从第一阶段开始训练的新 decoder 长期上限更低**。

参考：`docs/vggt_b_and_joint_short_controls_results_20261013.md`。

### 6.2 adapter 理解梯度路由是已确认的重要改善点

固定窗口梯度诊断发现 adapter 理解梯度主导，主要来自 mask/Dice，部分时刻联合梯度与重建相反。系数1→0.01保留理解损失，改变 adapter 接收的理解梯度。

完整同预算配对后 PSNR 提升约 **3.02667dB** 到22.40。这支持此前19dB的大幅下降与路由设置密切相关；不能把全部 SIU3R 指标差距都归因于它。

固定四训练窗口24.5左右的拟合数不是完整验证成绩。相关报告：`docs/vggt_adapter_gradient_results_20261012.md`、`docs/vggt_adapter_full_and_feedback_factorial_results_20261013.md`。

### 6.3 feedback0.05 的完整收益很小，并非理解全面提升

完整数据适配2+冻结联合4，feedback0.1→0.05：PSNR22.40→22.42（约+0.0200dB），其他重建也略好；六个理解指标五降一升，不能称为全任务最优。短对照优势不等于长期终点同等收益。

当前保留0.05作为后续共同底座，不继续盲目搜 feedback，也不把理解偏弱主要归因于 feedback。

参考：`docs/vggt_feedback_b005_full_results_20261014.md`。

### 6.4 C 对象分类读回：小收益，未解决实例质量排序

在 A 的分类输入加入本 query 预测 mask×context可见性加权的对象支撑，原 mask/feedback query 不改。完整312场景六项理解有小收益，但未参与筛选的184场景 mAP 基本持平，增益有类别集中现象。

好 mask / no-object 冲突与实例质量排序不一致仍存在。验证时 GT Hungarian 匹配和部署时 no-object 过滤、最终官方预测/GT匹配是不同步骤；不能说预测query被删后训练matcher自动替它分配另一个query。

参考：`docs/vggt_classification_readback_full312_results_20261014.md`。

### 6.5 Q 只是外部质量读出，不参与主模型联合训练

Q 为 detached query 256→128→1 MLP，单独训练300步，预测最终同类 mask IoU；实例 score=s_native×sigmoid(Q)。保持分类/候选/no-object筛选/像素赢家/最终mask及RGB不变，只改变实例置信度排序。

固定 Q 完整312场景验证：输入 mAP26.2508→26.6855、新视图24.7191→25.1409（+0.4347/+0.4218个百分点）；新增184场景也为正。不是 object query 的简称，也不是用 Q 训练主模型4epoch。

### 6.6 表征不足还是读取不足：尚未定位单一主因

冻结诊断1024训练窗口/707场景、128独立留出：query 已有类别信息，交互后没有明显整体类别塌缩；q12相对q_pre条件类别准确率差的区间跨0，原分类头不逊于新线性probe。

现有对象 Gaussian 支撑、局部 Fm 对质量预测有互补；DINO类信息有小信号，但不足以宣称必须新增encoder。confidence分层与VGGT深度误差相关，但不是几何干预收益的因果证明。

用户提出 TokenSplat式渐进GFM融合、confidence几何偏置、DINO/point-map补充。当前优先分开检验读取局部性、已有局部视觉证据和可靠深度约束，正是正在跑的四组。多层大融合、point_head、AA解冻等不与本轮叠加。

参考：`docs/vggt_feature_reading_results_20261014.md`。其中“建议下一轮未提交”是历史时态；**本轮后来已获得授权且正在执行，不要再提交第二套**。

## 7. 已有完整验证参照：不要与128场景短对照混比

以下均为官方312场景/1860窗口。理解值为0–1，差值转成百分点时乘100。

| 模型 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入mIoUs↑ | 输入mAP↑ | 输入PQ↑ | 输入mIoUt↑ | 新视图mIoUs↑ | 新视图mAP↑ | 新视图PQ↑ | 新视图mIoUt↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A 完整adapter0.01/feedback0.05|0.08684|0.2237|22.42|0.7818|0.3636|0.5719|0.2614|0.6407||0.5676|0.2462|0.6231||
|C A+对象分类读回|0.08684|0.2237|22.42|0.7818|0.3636|0.5765|0.2625|0.6442||0.5720|0.2472|0.6261||
|C+固定Q|0.08684|0.2237|22.42|0.7818|0.3636|0.5765|0.2669|0.6442||0.5720|0.2514|0.6261||

真实完整表：`/space/mawb/ssst/group_plus/vggt_quality_full312_v1/attempts/attempt01/analysis/siu3r_comparison.md`。

## 8. 工程恢复历史：不能误判成还没训练或再次重训

- attempt01：观测 hook 访问临时移除的 reading_control；正式更新0，修复后保留旧记录。
- attempt02：native runtime bootstrap 导入顺序导致 fused_ssim_cuda 缺失；正式更新0。
- attempt03：把原生 CUDA atomic lift 的微小读出波动误要求逐位相等；正式更新0。后续引入原模型自身连续两次前向波动见证；上游 Gaussian/Fm/query/anchor 要求差0，logits/membership 的原生重复及零干预差均≤1e-5且finite。没有放宽训练数值合法性。
- attempt04 GPU预检64667：完整单卡/八卡、四组真实2步joint backward及官方真实窗口评分通过，用时2:25:10。
- 正式startup64668/64669/64670/64671：四组各20更新/160曝光通过；64672统一核验成功。证明文件 `ALL_STARTUPS_VERIFIED.json`。
- 共享存储紧张：只改checkpoint无损序列化，使用node-local临时torch archive→ZIP_DEFLATED level1→共享atomic替换，锁串行共享写；参数、optimizer、RNG、计数roundtrip严格相同，load不使用mmap。没有删优化器或改精度。
- 64717 object_local_read 实际已训练300并导出128窗口；失败发生在 **官方评分环境顶层误导入TokenGS，缺lpips**。不能写成训练失败或重新跑300步。
- 修复评分、reduce、summary不导入主模型，只在export延迟导入；649ef09是当前执行快照。64851只补已有导出评分并成功，官方8rank合并合同通过。
- 64716/64718在PENDING且RunTime0时取消替换为64852/64853；64719此前因64717失败依赖取消，恢复为64854。旧日志、失败及取消记录保留。
- 18项数学文件SHA/受保护函数AST/正式update-loop AST与实际预检快照410bd30一致，因此没有再重复2.4小时GPU预检。

关键回执：

```text
attempt04/submission.json
attempt04/schedule_scoring_recovery_20261015.json
attempt04/scoring_import_recovery_preflight_reuse.json
attempt04/serializer_only_preflight_reuse.json
attempt04/ALL_STARTUPS_VERIFIED.json
attempt04/slurm/
```

## 9. 环境与代码入口

训练/模型导出：`/space/mawb/anaconda3/envs/tokengs/bin/python`。

官方评分/状态合并：`/space/mawb/SIU3R/.venv_gpu_v4/bin/python`。

**官方环境没有完整TokenGS依赖及匹配的gsplat运行库，不要让score/reduce/summary导入模型。** 不要为了它修改共享官方环境；已经有正确的纯评分入口。native模型导入前需要原runtime兼容bootstrap。

资产：`/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets`，离线复用。现有launcher设置HF离线、VGGT源码PYTHONPATH、预编译gsplat、线程数和geometry evidence路径，优先复用。

主要入口（相对当前worktree）：

| 文件 | 作用 |
|---|---|
| `tokengs/models/object_locus_reading_controls_v1.py` | 三种干预与真实CA观测 |
| `scripts/train_vggt_reading_controls_v1.py` | smoke/train，--resume精确恢复 |
| `scripts/eval_vggt_reading_controls_v1.py` | export/score/reduce/summary |
| `scripts/run_vggt_reading_controls_v1.sh` | 固定快照Slurm入口；含score_existing |
| `scripts/submit_vggt_reading_controls_v1.py` | 最初完整链提交器，**已有本轮作业时不能直接重跑** |
| `scripts/show_vggt_reading_controls_progress_v1.py` | 标准库只读进度，含squeue |
| `scripts/vggt_reading_numerical_contract_v1.py` | 工程恢复时核验18项训练数学身份 |
| `scripts/vggt_lossless_checkpoint_v1.py` | checkpoint无损压缩/恢复 |
| `scripts/format_object_locus_posefree_siu3r_table.py` | 固定13列映射及空白文本指标 |
| `tests/test_vggt_reading_controls_v1.py` | 局部读取/几何采样/零强度等关键合同 |

临时恢复脚本 `/tmp/recover_vggt_reading_schedule_20261015.py` **已经执行，不可再次执行**；正式收据在共享attempt04，不依赖/tmp永久存在。

Slurm/网络命令在Codex受限环境通常需 `sandbox_permissions=require_escalated`。未升级直接执行squeue可能报socket权限错误并耗时；这不是集群故障。无需用长轮询等待。

Git代理现有旧127.0.0.1:7890不可用；已成功使用逐命令覆盖，不改全局配置：

```bash
git -c http.https://github.com/fufuforu/ssst.git.proxy=http://127.0.0.1:17899 push origin vggt-overfit-diagnostics-v1
```

## 10. 数据、源权重与清理边界

训练manifest：`/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json`，8337窗口/1191场景；实际key是`windows`。

固定流：当前repo `docs/vggt_understanding_probe_cohort_v1.json` 的 `train_source_indices` 前2400项。

固定留出：`/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json`；records包含name/source_index/frame_ids/input_rgb_sha256，没有scene字段。

完整官方验证：`/space/mawb/SIU3R/data/scannet/val_pair.json`，312场景/1860窗口。

**除A、C、VGGT资产外，下列legacy源即使看似“很旧”也不可删除**，当前构造恢复仍会用到：

```text
/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt
SHA256 68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a
```

Q补丁（未来复核需要，当前四组不加载）：

```text
/space/mawb/ssst/group_plus/vggt_readout_probe_v1/attempts/attempt03/training/head_checkpoint_update0300.pt
SHA256 61dd9a8fbd8915c1c8779e61c6e0a3997b2b3d15a2a9639e297b1c0946ffcee5
```

用户已明确授权清理无用旧实验。已执行26文件清理，释放 **78.21GiB**：旧feedback factorial三组15个checkpoint、已被epoch4替代的3个epoch1–3 checkpoint、8份已完成feature诊断features.pt缓存。清理当时可用91.34GiB，不代表现在实时可用量。

保留当前任务、必要source/资产、C/Q补丁、小型指标/分析/失败记录。**旧feedback短对照已没有可重跑权重，特征诊断已没有features.pt，不要承诺直接从这些缓存继续。** 小型系数/分析仍保留。

清理证据：`/space/mawb/ssst/group_plus/storage_cleanup_20261014_reading_controls/{manifest.json,deletions.jsonl,receipt.json}`。

## 11. 新对话应该如何继续

1. 读取本文和 `docs/vggt_reading_controls_v1.md`，核实git状态/当前执行SHA与Slurm；已有后续提交或作业则先查明，不回退。
2. 只读检查64852/64853/64854/64855及上述COMPLETE/进度/导出收据；识别“训练完成而评分失败”与“训练失败”的区别。
3. 尚在正常训练/排队时，汇报实际阶段即可，不再重复提交或长期轮询。四组20/160证据已齐。
4. 工程失败时先定位，保留失败日志，保持数学身份修复并做受影响验证，提交推送新不可变快照；尽量只补失败阶段。训练恢复使用实际latest checkpoint及其optimizer/RNG/采样计数。
5. 全部完成后，审计每组300更新/2400曝光、八rank采样/finite/冻结/恢复、128场景导出和官方合并合同、共同GT与起点、Q未启用、13列有限性及两空白。读取自动 `analysis/report.md/results.json`，再分析；自动汇总本身也可能有可修工程问题，不把它误报为训练失败。
6. 比较三干预相对同预算baseline，结合真实CA观测与对象匹配诊断解释因果责任及局限。需要时做合理的保存状态CPU分析，不自动启动新GPU实验。短对照结果不足以宣布长期上限；提出下一轮方案交用户决定。

可用只读命令（Slurm部分在支持集群访问的环境运行）：

```bash
cd /space/mawb/ssst_vggt_overfit_diagnostics_v1
git status --short
git log -5 --oneline
python3 scripts/show_vggt_reading_controls_progress_v1.py
squeue -j 64852,64853,64854,64855 -o '%.18i %.36j %.10T %.12M %.18R'
sacct -j 64851,64852,64853,64854,64855 --format=JobID,JobName%36,State,ExitCode,Elapsed,NodeList -P
```

文档所列历史报告中的“当前排队/下一步未实施”只代表该报告形成时刻；本交接第3节是本次最新观测，但接手时仍以实时证据为准。
