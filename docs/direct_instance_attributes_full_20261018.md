# 直接Gaussian实例属性：完整训练集4epoch与已有基线比较

用户在67369短测核查后明确要求直接完整训练集训练，并复用已有完整训练基线。只新增一个训练组，不重训任何基线，不继承256步短训终点，不自动追加8epoch。

## 主对照与预算

- 基线：原Unified 66432完整4epoch，`attempt02_esam_many_full_node11_8gpu`；4172更新及正式128场景结果均已完成。其官方评测66433在指标成功后因旧汇总兼容问题退出，66528已补齐纯CPU汇总，不能据Slurm FAILED重跑。
- 新组：`unified_direct_instance_attributes`，直接Query→每Gaussian 8维属性、Query 8维原型，单位化点积归属；同一属性可微渲染，保留Pull/Push/Cross和原stuff/分类/匹配/NMS/组装。配方与已通过真实CUDA门禁及短训的67369相同，不增加67292约束、不调聚类参数。
- 1191训练场景、8337窗口；每epoch补齐至8344次曝光／1043更新，4epoch共4172更新／33376曝光。8×3090、每rank1窗口、global batch8；seed42、空AdamW、LR1e-4／100步warmup／原cosine，weight decay0.05（1D为0），clip1；理解权重0.05／1043步ramp，渲染实例项内部权重0.01、Pull/Push/Cross=2/1/2。冻结Omega/SAM。
- 每50步保存latest，每epoch保留checkpoint和同4train+4heldout探针；正式终点评分预先固定epoch4，不按探针挑最好epoch。
- 申请3dimage-13、8×3090、32CPU、192G、24小时；基线为11节点同8×3090。原基线4epoch耗时4h27，新方案有额外特征渲染，暂估约5–8小时，非保证；随后单卡评测约20–45分钟，排队另计。

主对照是相同4epoch预算的原Unified。它没有67050的渲染实例监督，因此比较包含直接属性/8维原型/渲染监督多个变化，不能解释成单因素消融。原Unified8epoch延训与旧C可作历史参考，不作为等预算主对照；不隐式把训练改成8epoch。

## 初始化、验证与固定快照

运行目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt17_direct_instance_attributes_full_node13_8gpu`。

独立执行快照：`/space/mawb/ssst_direct_instance_full_code_20261018`。

初始化仅使用67369的`checkpoint_initial.pt`（0更新、空Adam），严格验证四rank零步RNG完全相同后重复扩展至8rank。模型306个张量逐值保留，292个与原Unified完整训练零步共同张量逐值一致；绝不读取短训终点参数或Adam。训练严格加载同架构零步，正式初末checkpoint重新写入full4元数据。

复用67369已完成的16真实窗口CUDA/正负8通道渲染/独立梯度/四rank两步同步门禁与审计；检查所有既有模型/训练/推理实现自其执行SHA未变，仅新增全量调度与审计脚本。本轮不重复smoke训练，8rank正式首更新梯度与全参数SHA由训练入口检查；提交时不能宣称已通过8rank首步。

CPU预检：实际Omega/SAM资产hash、初始张量、数据清单和划分、全量边界窗口与全部8探针真实RGB，固定官方评测版本/源码，已有基线11指标/128记录及checkpoint身份。8337窗口和128验证场景沿用同清单，不重新抽样。全量结束自动审计4172条×8rank日志、每个rank完整数据顺序、Adam4172、有限参数实际变化、首步SHA同步及零步空Adam。

## 依赖与评测

1. 训练作业：4epoch训练→CPU完整性审计；失败不启动后续。
2. `afterok`依赖单卡评测作业：原生Prototype正式128导出→固定SIU3R官方11指标→有效性检查→固定8窗口Prototype raw/NMS/assembled与HDBSCAN辅助读出→与已有full4基线同RGB/帧/cohort比较。

官方128保留原context＋4额外帧口径、PNG深度量化和官方scale+shift，不能称纯novel或绝对尺度误差。表中13列有11数值，两个文本mIoU空列保持空白。HDBSCAN仍为全分辨率渲染8维→L2→固定sklearn HDBSCAN，类别无关/视图独立，辅助8窗口结果不是正式AP/PQ，不能与Prototype正式分数混比。

关键收据：`preflight.json`、`configuration.json`、`submission.json`、`SCHEDULING_CHECK.json`、`FIRST_UPDATE_VERIFIED.json`、`FULL_TRAINING_AUDIT.json`、`TRAINING_COMPLETE.json`、`evaluation/`、`readouts_final/`、`comparison_to_baseline_full4/`、`FULL_EXPERIMENT_COMPLETE.json`。训练失败看`JOB_FAILED.json`，评测失败看`EVALUATION_FAILED.json`。若仅评测失败，后续复用已保存checkpoint修复，不重训。

提交后只进行一次有界调度核对，记录作业号/资源/依赖后停止轮询；等待用户通知完成再检查。作业号、执行SHA、提交前验证与实际状态将在提交收据及交接文档记录。

## 提交与唯一调度核对

已提交训练/审计 **67390**，3dimage-13、8×3090、32CPU、192G；依赖评测/辅助读出/比较 **67391**，同节点1×3090、4CPU、64G，`afterok:67390`。唯一核对为UTC 2026-10-18T09:31:39.790358+00:00：67390 RUNNING，67391 PENDING(Dependency)。执行SHA `3d06c39ad97657959052329c892bda023f07436b`。

提交前21项唯一CPU检查通过（两次命令有2项重叠，不重复计数），包括全量日志/样本顺序审计与官方指标/比较流程合成端到端验证；shell/Python语法、git diff和实际CPU预检通过。全部169614个所需数据文件可访问，未发现缺失。复用67369真实CUDA门禁，未重复smoke。

**RUNNING仅表示调度运行状态；本轮未读取训练/评测日志，未核实8rank首步或GPU进展。已停止轮询，等用户通知完成后再检查。** 不重复提交或重训；作业收据见同目录`direct_instance_attributes_full_submission_20261018.json`。
