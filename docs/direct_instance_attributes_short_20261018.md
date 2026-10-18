# 直接Gaussian实例属性：专用GPU门禁与16场景短测

用户在2026-10-18授权“短测一下”。只提交一个新架构实验，复用已完成67050和67349的基线产物，不重训基线、不重复67292，不自动全量或官方128。

## 设计与可比性

架构`unified_direct_instance_attributes` / `OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_V1`。Query直接预测每Child的8维实例属性、每Query的8维Prototype，单位化点积/独立sigmoid计算全部Query归属；同一属性经原Renderer可微渲染，沿用Pull/Push/Cross。两个stuff通道、类别、重建、来源匹配及原生NMS/组装保留。本组不叠加67292直接Gaussian一致性项。

原16训练场景、相同数据顺序和4train+4heldout探针；seed42、全新模型、空Adam，256次更新、global batch8（4卡每卡1×累积2），LR1e-4/20步warmup/原短cosine，理解权重0.05/20步ramp，原实例项内部权重0.01，clip1，每32步checkpoint。不会用任一256步终点初始化。

共享/几何/RGB/分类的共同零步张量与67050逐值核对，新增和删除的头单独列出；不再要求旧305张量全部存在，不要求新归属Mask零步与旧读出相同。实际计数由`preflight.json`记录。检查实际Omega/SAM资产hash、原数据清单、16训练+4heldout真实窗口，8个探针的RGB与GT核对既有缓存。模型初始化、配方和数据身份写入收据。

独立运行目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt16_direct_instance_attributes_short_node13_4gpu`。

独立执行快照：`/space/mawb/ssst_direct_instance_short_code_20261018`。提交脚本限定上述目录和干净快照，拒绝重复提交。

## 同一作业内顺序依赖

1. **专用GPU门禁**：四rank检查原16个真实训练窗口。直接属性重算逐值一致、属性/实例特征是同一张量、全部Query单位化点积一致；属性预测对显式Gaussian/center/radius及几何头无直接梯度。原Renderer的8通道正负号/alpha oracle及同次特征重渲染通过。
2. 独立梯度：Membership项到属性/Prototype/stuff/共享投影，渲染Pull/Push/Cross到属性/几何/共享投影且不直接到Prototype。允许单窗口零监督，但16窗口合计每条必需通路有非零有限梯度，不放宽配方。
3. 四rank两次DDP更新，使用原真实训练批次和累积；每个新头有有限正梯度、全部可训练参数有有限梯度、各新头实际变化、Adam=2、逐步全参数SHA同步、冻结资产无grad。
4. 严格验证门禁收据，**丢弃全部smoke模型与Adam**，新torchrun从seed42/空Adam短训256步。门禁失败写`JOB_FAILED.json`，不启动训练、不修改配方重试。
5. 对保存的`checkpoint_initial.pt`和`checkpoint_latest.pt`分别固定权重评测：相同4train+4heldout、每窗口两图，prototype raw/NMS/assembled及HDBSCAN共用同一次前向。HDBSCAN使用67349同一固定全分辨率配方，无GT/XY/噪声填充；不按结果选阈值或重新训练。四rank同时工作，每rank4CPU。
6. 专用审计：确认smoke零步SHA=正式训练零步SHA、共同零步参数=基线、零步空Adam、最终Adam256/参数有限并改变、四rank各256连续日志、原首步SHA同步、新头/几何/共享分支更新，初末双读出参数不变。原生PSNR和来源匹配探针、全GT每图一对一IoU/P/R分别比较既有基线，不能混用两个IoU口径。

训练、评测、汇总均在同一作业内按上述顺序运行，不另起训练组；提交后停止轮询。

## 资源与检查方式

拟申请3dimage-13、4×3090、16CPU、128G、时限4小时。既有同预算短训约34–36分钟、固定8维诊断约7分钟；本组增加原16窗口梯度检查及初末双读出，预计约40–60分钟，排队另计，真实耗时以完成记录为准。

关键收据：`preflight.json`、`VALIDATION_EVIDENCE.json`、`submission.json`、`SCHEDULING_CHECK.json`、`SMOKE_PASSED.json`、`FIRST_UPDATE_VERIFIED.json`、`TRAINING_COMPLETE.json`、初末checkpoint、四rank日志、`readouts_initial/`与`readouts_final/`的`SUMMARY.json`/`FIXED_WEIGHTS_rankXX.json`/`COMPLETE.json`、根目录`SHORT_VALIDATION_SUMMARY.json`和`COMPLETE.json`。异常看`JOB_FAILED.json`的stage。

提交后只做一次调度接收/节点/卡数检查。RUNNING不代表真实CUDA门禁或首步已验证，不等待这些结果。用户通知完成后再读日志及结果；若训练完成而仅评测/汇总失败，复用已有checkpoint和缓存修复，不重新训练。

## 代码

新专用入口：`preflight_direct_instance_short.py`、`smoke_direct_instance_attributes.py`、`direct_instance_short_policy.py`、`run_direct_instance_short.sh`、`summarize_direct_instance_short.py`、`submit_direct_instance_short.py`。双读出用`evaluate_direct_instance_attributes.py`，已修复多rank输出拒绝检查的时序竞争，并对训练探针缓存核对RGB。旧smoke/旧305张量审计不用于新架构。

提交前软件验证：50项CPU检查（模型/原架构/聚类45、专用门禁3、端到端汇总/拒绝继承更新2），另1项真实两rank Gloo两步更新/全参数同步通过。Python编译、shell语法和git diff检查通过。GPU门禁尚未执行，不能把CPU通过写成CUDA已验证。
