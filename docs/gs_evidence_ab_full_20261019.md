# A evidence / B evidence+coverage：授权全量八卡对照

用户授权 A、B 两组完整数据八卡训练，使用 12、13 节点。A 对照已完成 attempt02，B 对照 A；不重训旧组、不续训 attempt17、不增加其它配方。本文件的准备状态不是完成结果；提交与调度状态另见提交 JSON 和各运行目录。

| 组 | 架构 | 节点 | 独立运行目录 |
| --- | --- | --- | --- |
| A | unified_gs_evidence | 3dimage-12 / 8×3090 | attempt18_gs_evidence_full_node12_8gpu |
| B | unified_gs_evidence_coverage | 3dimage-13 / 8×3090 | attempt19_gs_evidence_coverage_full_node13_8gpu |

运行目录根为 `/space/mawb/ssst/group_plus/unified_query_training_v1`；执行快照 `/space/mawb/ssst_gs_evidence_ab_full_code_20261019`。A 只增加逐 GS GFM evidence，采样位置 detach。B 与 A state_dict/初始权重完全相同，只增加权重 0.1 的 GT 实例内部覆盖 BCE（不乘 understanding_weight）。匹配仍 source_many_to_one / support_matching=true / purity=0.5。没有新 8D/HDBSCAN 读出、匈牙利更改或后处理更改。

两组：1191 训练场景 / 8337 窗口，4 epoch / 4172 optimizer updates，每卡一个窗口、global batch8；AdamW LR1e-4、betas .9/.95、eps1e-8、矩阵 weight_decay .05/一维0、clip1；100 更新 LR warmup + 原 cosine；理解权重 .05 / 1043 更新 warmup。模型初始化 seed42；数据顺序沿旧 epoch_order 的 seed31415+epoch，不改成42。每 epoch 保存权重并在固定4train+4heldout探针检查，终点统一 epoch4，不挑最佳 checkpoint。

CPU 准备以真实冻结 Omega/SAM 构造 seed42 A 权重，独立初始化 B 可训练模块，核对 A/B 全部301张量完全相同；对 attempt02 300/301共有张量完全相同，仅 core.child_feature.0.weight 扩展输入维度。两个新零步 checkpoint 都是空 Adam、0 更新；只重放 attempt02 原零步八rank RNG，不继承任何训练后的权重或优化器。prepare 检查配置、源资产 SHA、数据 manifest/cohort SHA、所有需要的数据文件可访问、11个实际样本、8探针 RGB 与原组相同、正式 evaluator 源码固定未改。保存初始 SHA 与可审计收据。

每个训练作业内依次执行：

1. 八rank真实 CUDA evidence/位置无梯度/GFM 有梯度检查，覆盖16个固定窗口。
2. 原 renderer/完整损失的两步 DDP 工程检查，要求全部参数梯度有限、新 evidence 列梯度活动、八rank更新 SHA一致、冻结主干无梯度；B 覆盖项实际非零。没有损失下降门槛。
3. 丢弃门禁两步，重新启动正式训练进程，读入独立零步 checkpoint 和原始 RNG、创建新空 Adam。门禁更新不进入4172预算。
4. 正式训练结束自动审计全部301权重实际变化、Adam4172、八rank日志/数据顺序/LR/权重配方、初始 RNG/参数与准备值一致、首更新同步、evidence 零占比及 >.25 次数。B 另审计覆盖项权重与总损失公式。

CPU 检查通过不代表 CUDA/首步已经通过；上述 CUDA 阶段在作业内自检，失败会阻止正式训练并写 JOB_FAILED.json。提交后不等门禁或首步，不以 RUNNING 冒充验证通过。

依赖：A_train → A_eval128，B_train → B_eval128；两 eval128 都成功 → AB_comparison（CPU，无GPU）。每个正式评测在对应节点1×3090执行原官方导出与固定 SIU3R evaluator；比较程序核对同128场景、帧、RGB，输出旧组/A/B及两项差值的原始13列/11数值表，文本mIoU列仍空白，不修改历史评测。辅助 source-supported 探针不当作官方 AP/PQ。

每组新目录收据：preflight.json、configuration.json、checkpoint_untrained_initial.pt、submission.json；运行后 SMOKE_PASSED.json、FIRST_UPDATE_VERIFIED.json、TRAINING_COMPLETE.json、FULL_TRAINING_AUDIT.json、evaluation、FULL_EXPERIMENT_COMPLETE.json。训练预计5–8小时/组（两组并行，参考原4172更新8×3090约4.5小时并留门禁/额外采样余量）；各评测约20–45分钟，排队另计。仅单seed，不能据小差异宣称统计显著。

提交后只做一次有界调度核对，完成记录/推送后停止轮询，等用户通知完成再检查。不能因本页准备状态或历史 RUNNING 自动重复提交。

## 已提交收据

执行SHA `1982cd1`。唯一调度核对：A训练67700（12节点8卡）与B训练67702（13节点8卡）均RUNNING；评测67701/67703依赖各自训练，CPU汇总67704依赖两评测，均PENDING(Dependency)。CUDA门禁与正式首步尚未核实，未读训练日志；已停止轮询，等用户通知完成后检查。完整记录见 `gs_evidence_ab_full_submission_20261019.json`。
