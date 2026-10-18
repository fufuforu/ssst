# 渲染前 Gaussian 实例一致性：GPU门禁与短测

用户授权GPU验证后短测，完成后由用户通知检查；提交后只核对一次调度状态，不等待GPU结果。

唯一新增组 `unified_gaussian_consistency`，复用已完成67050为基线，不重训基线。运行目录 `/space/mawb/ssst/group_plus/unified_query_training_v1/attempt15_gaussian_instance_consistency_short_node13_4gpu`；独立执行快照 `/space/mawb/ssst_gaussian_instance_consistency_short_code_20261018`。作业号、执行SHA和唯一调度核对以运行目录 submission.json / SCHEDULING_CHECK.json 为准。

流程为同一作业内顺序门禁：真实CUDA有符号特征渲染及原渲染loss梯度 → 原16场景逐一检查新项有效正负样本、8维重算逐值一致、独立实例/理解/共享梯度和显式几何隔离 → 四rank两次DDP更新与参数SHA同步 → 丢弃smoke模型/Adam → 独立新进程从seed42空Adam短训256步 → 原探针汇总及新项活动/基线比较。任一门禁失败即停止并写JOB_FAILED，不自动修改配方或启动训练。

申请13节点4×3090、16CPU、128G、时限4小时。与67050共同305个零步参数逐值相同，实际Omega/SAM、数据清单/图像/训练窗口预检；16场景、global batch8（每卡1×累积2）、LR1e-4/20步warmup、理解权重0.05/20步ramp、原短cosine、seed42、每32步checkpoint。新增约束配方不改：纯度0.7、贡献质量0.001；同GT跨父Token正样本、局部异GT最多8邻居/0.1场景RMS半径；新增loss在理解项内权重0.01。原渲染实例loss保留，推理输出/官方组装不变。

门禁允许个别场景/单rank跳过无监督样本，但原16场景合计必须有真实跨父正样本、局部异GT负样本，且新项对实例头、child_feature、semantic_adapter、共享decoder/GFM投影产生非零有限独立梯度。新项对显式Gaussian/center/radius以及geometry/activation/reconstruction adapter参数不得有梯度。初始样本覆盖门禁不改变训练配方、也不要求loss下降。

短测自动保存4train+4heldout初末原生探针、每步直接loss/pull/push/正负覆盖与距离、`GAUSSIAN_SHORT_AUDIT.json`及与67050的PSNR/支持GT原始Mask IoU比较。不是官方mAP/PQ，不自动全量或官方128。旧基线约34分钟，上一轮变体约37分钟；本次增加贡献转置与局部搜索尚无GPU耗时实测，预计40–70分钟，排队另计。

提交时CPU验证通过不等于新GPU通过；GPU门禁/训练结果等用户通知后核查。

## 提交收据

已提交 **67292**，唯一一次调度核对为 **RUNNING，3dimage-13，4×3090**（2026-10-18 13:44北京时间）。执行SHA `0882660021e17e51eae9471b5921d5a6db951b90`，CPU新增门禁/模型测试16项及真实2rank Gloo1项通过；真实资产/样本预检通过，共同305零步张量逐值相同，无新增参数。GPU门禁和短训结果未核实，不读取日志、不继续轮询。预计40–70分钟，新增成本实测未知。用户通知后检查SMOKE_PASSED、JOB_FAILED、COMPLETE和GAUSSIAN_SHORT_AUDIT。
