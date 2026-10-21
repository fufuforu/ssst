# 全量直接joint4与adapt2+joint4对照及自动评测

用户授权保留当前4epoch作业，并新增重建适配2epoch＋联合4epoch，可用11/13/14/17/18中的空闲四卡；要求检查评测后自动给出官方13列（11个实际数值）对比。

## 当前作业耗时与资源

65199确实进入全量训练；检查时rank0已完成15个全量更新，最近更新中位数16.94秒。总4172更新，以当前速率估计训练约20小时，加首次缓存、监测和导出/评分，暂估还需20–24小时。早期样本很少且冻结SAM缓存后续可复用，该估计不是精确承诺。

检查时11四卡已被65199使用，13八卡全部分配，14为ALLOC，18只余两卡；17有四张4090空闲、实际可用内存约248GiB。因此新增对照固定17节点四张4090、20CPU、128GiB内存、72小时上限。后续动态变化以Slurm为准，不持续轮询。

## 共同起点与阶段

两条分支共享65175短验证终点：源50步＋新结构256步，累计306步。新分支不是使用65199已经全量训练后的权重，也不重跑已完成短验证。源文件保留为attempt05的checkpoint_source_update0306.pt，新分支硬链接为checkpoint_shared_short0306.pt，SHA256为56d4b7f8f0d680fc0367a97fc10ae7e4d3ebe2f441a01d777861f674352b093f。模型、AdamW、4rank RNG全部保留。

- 直接组：共同短验证起点→联合全量4epoch，4172新更新。
- 分阶段组：共同短验证起点→纯重建全量2epoch（2086更新）→联合全量4epoch（4172更新），合计6258新更新。

全量1191场景、8337窗口、batch8，每epoch1043更新/8344曝光，补齐7窗口曝光。两阶段都使用完整训练manifest；适配与联合分别记录epoch计数和覆盖，不将适配epoch算作联合epoch。不重置共享参数或optimizer，联合阶段重新执行既定100步warmup/余弦调度。

适配只反向重建loss，理解参数（understanding/query_initializer/panoptic）的grad置None，确保AdamW动量和weight decay也不更新；每步校验所有这些参数版本不变。重建Adapter、anchor视觉投影、重建decoder/几何/GS头参与适配。随后理解损失权重恢复1、理解参数正常更新。**Omega/SAM仍冻结、Write=0，模型结构和损失公式不变；这不是额外解冻Omega的实验。**

这是共享短验证后的训练日程对照，不能称完全从零的2+4实验，也不是等算力对照。新组多两个完整重建epoch，目标是验证稳定适配流程；旧VGGT+MASt3R C作为成熟指标参考，不将其差异全部归因为日程。

## 测试与自动评测

CPU检查覆盖：直接/分阶段源metadata白名单、拒绝错误数据/world size/已全量更新来源、实际Task重建分支不产生理解梯度、AdamW在适配阶段保持理解权重及moments不变、13列严格11个有限数值及text两列空白、重复/冒烟队列不能冒充完整128评测。训练/导出/评分脚本编译及shell语法检查。

新训练作业先用共同306checkpoint对固定heldout队列第一个场景完成真实无更新导出及原版SIU3R全指标评分，包含RGB/GT哈希、6帧与分割格式核对；SMOKE_PASSED后才进入2+4训练。冒烟结果单独写protocol_smoke，状态SMOKE_EXPORTED，不作为正式128场景评测。未看到SMOKE_PASSED前不能宣称GPU端到端测试通过。

新分支TRAINING_COMPLETE后通过afterok依赖独立1GPU任务加载最终checkpoint、导出同一固定128场景并官方评分；检查joint4及adapt2都完成，冻结资产/数据哈希匹配，生成期间权重不变。官方评测commit固定8ea80166be76854f938e90521f1a5b688b755c87，src未修改，GT PNG严格匹配旧C。13列表格的两个text mIoU留空，按官方target集合（输入2帧＋额外4帧）汇总，不自行替换为纯4帧。

65199原脚本已经内置导出与评分，不改正在执行的不可变快照。给其增加afterok校验任务：已有COMPLETE时检查13列/11值、与原生metrics一致及官方版本，直接复用，不再推理或重算指标。

两组评测成功后，再由CPU依赖任务检查场景名/source_index/6帧/RGB哈希一致，并生成旧C、direct joint4、adapt2 joint4和staged−direct差值的一张表。结果目录：`/space/mawb/ssst/group_plus/dual_stream_training_v1/comparison_joint4_vs_adapt2_joint4_20261015`。

新运行目录attempt06_adapt2_joint4_node17；不可变代码快照`/space/mawb/ssst_gfm_semantic_pool_v3_staged_code_20261015`。作业号以运行目录submission.json为准。训练或冒烟失败时afterok评测不会启动，失败记录保留；按用户偏好，提交并单次核对调度后停止轮询。

## 已提交作业

执行SHA：36085703f1fe6b2426668aead700980a79680af1，已推送。CPU共12项测试通过（policy/cache/sampler 6＋staged/Task/AdamW/评测表格 6），编译、shell语法及diff检查通过。旧C真实参考CSV也通过13列/11值/空白text检查。

| 作业 | 功能 | 依赖 |
|---|---|---|
| 65199 | 原直接联合4epoch，11节点四卡 | 原作业保留 |
| 65236 | 17节点四4090，一场景冒烟→adapt2→joint4 | 冒烟在脚本内前置，失败则不训练 |
| 65237 | 原组完成后的评测结果校验，已有结果直接复用 | afterok:65199 |
| 65238 | 新组最终checkpoint固定128导出＋官方评分 | afterok:65236 |
| 65239 | 旧C/direct4/staged2+4统一13列11值与差值表 | afterok:65237:65238 |

提交后只核对一次：65236 RUNNING、17节点确实分配四4090；其余三个任务PENDING/Dependency，65238明确afterok:65236。此时尚无新作业GPU冒烟或首个适配更新证明，不得将RUNNING解释为测试通过。随后停止轮询。提交收据写attempt06/submission.json及comparison目录submission.json，原组dependent_evaluation_submission.json也已记录。不要重复提交。
