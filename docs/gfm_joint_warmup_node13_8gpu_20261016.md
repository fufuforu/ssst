# 重建低学习率＋理解损失warmup：13节点八卡

用户同意从已完成适配2epoch的模型出发，降低重建学习率并warmup理解损失，要求直接使用13节点八卡。实查3dimage-13 IDLE，8×RTX3090，GPU/CPU/内存均未分配，实际可用内存约258GiB。

## 明确的新训练协议

新recipe：omega_gfm_semantic_pool_v3_adapt2_joint4_lr10_uwarm1_v1。

- 来源：attempt06_adapt2_joint4_node17/checkpoint_adaptation_complete.pt，执行SHA75f6ce182b4ee79e75485db541daf0ffc85f91a9。适配2086更新/2epoch已完成，joint_updates=0，short_updates=306。不是已退化的联合4epoch终点。
- SHA256：2b07922b1b2c3e8f5dfa6c1591951e043bc418874f2cea379588003ddde6f3e3。新运行硬链接保留为checkpoint_source_adaptation2.pt。不重跑适配或短验证。
- 完整保留模型权重与637项AdamW参数状态/moments；显式覆盖param groups的base_lr/lr，避免load_state_dict把新学习率改回旧值。
- 重建Adapter/重建decoder/GS头基础LR=1e-5；几何细化=1e-6；语义/Query/Mask与分类模块=1e-4。保留原100步全局LR warmup及4epoch余弦衰减。
- 训练理解损失权重beta=min(1,(joint_update+1)/1043)。首步约0.000959，首epoch结束达到1，后续3epoch为1。重建损失始终保留。两组分支均参与更新，不增加stop-gradient或永久冻结。
- 模型结构、Omega/SAM冻结、Write=0、loss公式、优化器其余超参、训练/评测队列不变；不增加文本分支。
- 8卡×每卡microbatch1×累积1=有效batch8。相比此前4卡×microbatch1×累积2不改变global batch和epoch更新数。数据sampler已验证4/8rank覆盖同一global batch次序。
- 4→8rank是显式拓扑迁移；rank0–3恢复原RNG，rank4–7固定seed=31415+80000+rank。不能宣称与4卡dropout轨迹位级相同。
- 新训练联合4epoch/4172更新，1191场景/8337窗口，epoch补齐7曝光。适配2086仅作为来源继承，不计入新作业训练更新。

源metadata白名单只允许本次recipe/LR/损失warmup/监测频率及world size/accumulation/code变化；模型资产、数据hash、其余配置和适配终点必须一致。普通resume继续严格检查新recipe/代码/8rank状态，beta按已保存full_updates推进，不重置ramp。

训练journal记录实际beta、加权理解loss、各组实际LR、梯度范数及参数更新证据。固定8训练/8验证monitor仍按beta=1计算诊断loss，以便跨阶段同口径比较；它不是当前beta下实际优化loss。新协议每100更新监测，50更新checkpoint，首个更新也保存。第一份monitor记录适配2后的起点，随后检查是否保住约21.49dB验证监测水平；不据验证监测提前选checkpoint或改变4epoch长度。

## 资源与作业

新运行：/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt08_joint_warmup_node13_8gpu

不可变代码快照：/space/mawb/ssst_gfm_joint_warmup_code_20261016

训练固定3dimage-13，8×3090，32CPU，192GiB内存，最长72小时。训练成功后释放8卡，afterok独立1GPU任务加载最终checkpoint作固定128场景导出及原版官方评分。Evaluator commit/GT PNG/输入RGB/6帧目标集合/13列表格均沿用原协议，两个text mIoU留空。

评测包装修复仅保存和标记非有限结果：若必须的11项有NaN，写metrics_invalid.json及siu3r_table1_invalid.csv并明确失败；不删帧、不补深度、不改官方计算。有效11项照常生成COMPLETE。原始all/results.json始终保留。依赖afterany的CPU对比任务可从已保存原生结果生成旧C/direct4/adapt2+joint4/新warmup四模型表，并明确无效字段；若评测未完成、缺少原生文件，则拒绝生成伪完整结果。

提交脚本及收据记录准确job IDs；提交后仅单次核对调度，随后停止轮询。RUNNING不等于已完成八卡GPU首步反向，需未来检查FULL_STARTED/JOINT_WARMUP_TRANSITION及新journal。

## 已完成的预检

14项CPU检查通过：beta边界/恢复时钟、实际Task的梯度权重且重建仍可训练、AdamW moments保留与LR覆盖、真实适配终点的显式metadata迁移、无效深度保留标记、既有batch8 sampler/cache及并发发布。真实源checkpoint metadata和SHA256已核对，预检写入preflight.json。新对比脚本已在已有两组原生结果上验证队列一致、13列及NaN保留。Python编译、shell语法、diff检查通过。
