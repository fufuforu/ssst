# 原Unified单分支从epoch4续训至epoch8

> 2026-10-18已完成核实，66648–66651全部成功；Unified8 PSNR18.4225、target mAP15.5542%/PQ48.2612%。恢复权重/Adam/RNG、8344累计步和正式11指标检查通过；mAP相对4轮下降。完整比较见[完成分析](continuation_and_serial_residual_results_20261018.md)。下方RUNNING为提交时历史状态，不重跑。

用户授权先续训原Unified，不启动B续训或串行结构实验。独立运行：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt09_unified_query_extend8_node11_8gpu`。11节点8张3090空闲；与原Unified完整训练相同节点/卡型，也与A续训保持相同3090卡型、8rank和全局batch8。原A作业/目录/执行快照不改，不查询其训练进度。

## 起点与训练安排

- 源为`attempt02_esam_many_full_node11_8gpu/checkpoint_epoch4.pt`，原Unified完整4epoch/4172更新，不是B直接Mask或串行版，不是短测终点或零步源。
- 恢复301个可训练参数张量、301份Adam状态（一二阶动量与step4172）、8rank Python/NumPy/Torch/CUDA RNG。保持原多对一监督、共享12层Decoder与原Child–Query相似度Mask读出，不换初始化或结构。
- 1191场景/8337窗口，新增epoch索引4–7，4172次更新/33376曝光（含padding），累计8epoch/8344更新；不重播前4轮。checkpoint_initial记录本次真实epoch4起点，不伪称零步。
- Omega与SAM仍冻结，正常联合梯度、理解权重0.05不重新ramp，全局batch8、clip1，保持原配置。
- 使用与A续训完全相同的LR：新增第1步（累计4173）为1e-5，100步升至3e-5，此后cosine衰减至3e-6；保留Adam状态。因此比较的是增加训练＋同样的LR重启策略，不是追溯修改原4epoch的学习率历史。
- 保存初始与每epoch固定4train/4heldout探针、epoch5/6/7/8 checkpoint，每50步保存可恢复checkpoint。第一新增更新要求梯度有效/冻结无梯度、8rank全参数摘要一致；8份CONTINUATION_LOADED收据要求源权重摘要精确一致、Adam仍4172。提交时这些新GPU检查尚未核实。

## 代码与可比性

复用A续训入口/预检/评测/汇总，只增加明确的Unified policy与301张量身份，按architecture匹配。A policy仍只能接受A693张量，Unified policy只能接受原Unified301张量；B/串行没有授权policy，错误源/数据/参数/优化器/世界尺寸必须拒绝。正在运行A使用已提交的旧独立快照，因此不受本次通用入口扩展影响。

与原Unified执行SHA09063a1相比，当前UnifiedQueryCore和三维超点模块文件完全一致；模型封装仅有历史的等价构造器重构：`UnifiedQueryCore(...)`替换为默认值仍为该类的`self.core_type(...)`。预检逐字确认仅这两处改动，并确认实际core_type类身份。不是暗中换结构。

源模型与Adam有限性、301键/形状/dtype及参数组/动量布局、冻结Omega/SAM文件SHA、训练/128验证manifest SHA、实际读取样本、官方评分器提交及源码未变，全部在真实预检中检查。测试覆盖A/Unified policy互斥、拒绝B/串行、累计进度/LR、Adam保留及下一步数值、显式epoch6导出、两模型汇总身份和RGB不一致拒绝，以及原Unified前向/梯度与探针回归。

## 自动评测

训练成功后，在11节点两个单卡任务分别评测epoch6与epoch8实际checkpoint，两者都等epoch8训练完成后执行。复用原官方128场景/相同帧与RGB身份、固定SIU3R提交、原NMS与组装阈值，输出11项有效指标/两个文本空列。

评测使用独立`official_epoch6`和`official_epoch8`，checkpoint_identity保留计划epochs8与实际evaluated_epoch，不改写检查点元数据。子评测根中的TRAINING_COMPLETE是父run实际epoch8完成收据，不表示epoch6权重经历了8epoch。

依赖CPU汇总比较旧C与原Unified epoch4/6/8，核对同128输入身份、实际checkpoint epoch和源SHA，输出`comparison_epoch4_6_8/{comparison.csv,comparison.md,COMPLETE.json}`。与A续训的综合对照待两组完成后用户通知再分析，不额外轮询A。

独立干净代码快照：`/space/mawb/ssst_unified_query_extend_code_20261017`。具体SHA/测试项数/作业号/一次状态记录以run的preflight、VALIDATION_EVIDENCE、submission和SUBMISSION_STATUS_ONCE收据为准。原Unified此前同节点8卡4epoch耗时4h27m，本轮预计训练约4–4.5小时；评测各约15–20分钟可并行，排队另计。提交后停止轮询，等待用户通知。

## 已提交并停止轮询

执行SHA **4787f3696f8ef1121005d5edf17cc34811c500ff**，独立快照干净。真实源301张量/Adam4172/8rank RNG、冻结资产/数据及模型等价性预检通过，34项CPU测试、Python编译、shell语法与diff检查通过。新GPU加载/第一新增更新尚未核实，RUNNING仅代表调度状态。

- **66648**：原Unified epoch4→8，11节点8×3090、32CPU/192G；唯一一次状态核对RUNNING。
- **66649**：epoch6官方128，11节点1×3090；afterok:66648，PENDING(Dependency)。
- **66650**：epoch8官方128，11节点1×3090；afterok:66648，PENDING(Dependency)。
- **66651**：原Unified4/6/8及旧C CPU汇总；afterok:66649:66650，PENDING(Dependency)。

已停止轮询，不等待GPU首步、训练进展或结果。没有查询A进度、重启A或提交B/串行新实验；用户通知完成后再检查。
