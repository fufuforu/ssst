# A从epoch4继续训练至epoch8

用户授权使用13节点空卡继续A。只续训已完成A，不重跑0–4epoch、不换结构、不重新初始化或适配。源为`attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_epoch4.pt`，epoch4/4172更新，693张量和693份Adam状态，8rank RNG。独立run为`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt08_legacy_dual_superpoint_extend8_node13_8gpu`。

## 训练方案

- 13节点8×3090，全局batch8，每rank1窗口；1191场景/8337窗口，继续epoch索引4–7（人类编号第5–8轮）。追加4172更新、33376曝光（含每轮补齐7窗口），累计8344更新/8epoch。
- 源权重、Adam一二阶动量/step、各rank Python/NumPy/Torch/CUDA RNG全部恢复；不把学习率重启误做成Adam重置。不重播旧轮次；数据顺序接确定性`epoch_order(...,4)`。
- Omega/SAM继续冻结；两组3D超点初始化、A的1024/256宽度、Adapter、L6/8/10/12交互、相似度Mask头、多对一监督、NMS/官方组装全部保持。Write=0；正常联合训练、理解权重0.05，不重启1043步理解权重ramp。
- 源LR末尾约1e-5；新增100步从1e-5线性升到3e-5，之后cosine下降到3e-6。这是“额外训练＋明确LR重启”的对照，不是纯epoch数量单变量；先前4epoch已经执行的cosine不可能追溯改成8epoch曲线。
- 复用原训练入口，新增明确source checkpoint/SHA及严格延训metadata/Adam形状验证；原`--resume`继续要求同一run代码/config完全一致，不放松历史恢复规则。
- 第一新增更新（累计4173）执行各参数组梯度/冻结分支无梯度和八rank全参数SHA同步检查。提交时未核实新GPU首步，不把既有A训练收据当本轮新证据。
- 每50步保存；初始探针及epoch5/6/7/8固定4train/4heldout探针、每epoch checkpoint均保留。新run的checkpoint_initial含源epoch4权重和Adam，不是零步。

## 自动评测与比较

训练成功后，两个独立单卡任务分别评测实际epoch6/epoch8 checkpoint（均等到epoch8训练完成后执行）。保持原官方128场景、帧/RGB身份、固定SIU3R提交、11数值/2文本空列及全部阈值。epoch6导出必须显式指定`--evaluation-epoch 6`，验证epoch边界/6258累计更新和延训policy，不将未完成epoch当终点，也不改checkpoint的计划epoch8元数据。

两个评测的独立根为`official_epoch6`、`official_epoch8`，原生checkpoint_identity记录实际评测epoch与原meta。其TRAINING_COMPLETE收据复用真实父run epoch8完成收据，不能解读成epoch6 checkpoint已训练8epoch。评测配置与对比CSV标明evaluation_epoch，避免原评分脚本固定epoch4名称误导。

CPU afterok汇总核对三组128RGB/cohort一致、11有效值与原始metrics相符、源SHA和实际epoch正确后，输出旧C、A4/A6/A8及相对A4差值表。若训练失败，依赖评测不运行；不启动后台轮询。

## 验证与提交记录

提交前检查真实epoch4/Adam状态、冻结资产SHA、manifest/heldoutSHA与样本实际读取、A关键模型代码与源版本相同、官方评分器未改；针对性CPU测试覆盖源身份/数据/策略拒绝、完整保留Adam与下一更新数值、累计进度与新epoch采样、LR边界及显式epoch6导出拒绝错误进度，另回归现有探针/结构/训练策略。实际项数、执行SHA与检查收据见新run的`VALIDATION_EVIDENCE.json`、`preflight.json`。

执行独立干净快照`/space/mawb/ssst_superpoint_dual_extend_code_20261017`。作业号与一次调度状态见`submission.json`和`SUBMISSION_STATUS_ONCE.json`。提交后停止轮询，等待用户通知完成。按A先前13八卡4epoch耗时5h40m，预计本轮训练约5.5–6小时，另两组评测各约15–20分钟（可并行，排队另计）。

## 已提交，停止轮询

执行SHA **7965e85c0573fb2d3fe007195364fd18214dc19a**，独立worktree干净。真实源/冻结资产/数据预检通过；21项CPU测试通过，Python编译、shell语法与diff检查通过。新GPU首步尚未核实；运行中会要求8份CONTINUATION_LOADED精确源权重摘要/Adam4172收据及第一新增更新同步证明，不将调度RUNNING等同于这些检查通过。

- **66640**：A epoch4→8续训，13节点8×3090、32CPU/192G；唯一一次调度核对RUNNING。
- **66641**：epoch6官方128评测，13节点1×3090；afterok:66640，单次核对PENDING(Dependency)。
- **66642**：epoch8官方128评测，13节点1×3090；afterok:66640，单次核对PENDING(Dependency)。
- **66643**：A4/6/8及旧C CPU汇总，afterok:66641:66642，单次核对PENDING(Dependency)。

没有轮询GPU首步、训练进度或评测状态；已停止轮询，等待用户通知完成后检查。
