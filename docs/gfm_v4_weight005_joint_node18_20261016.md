# V4 普通联合：理解损失权重0.05对照

用户希望利用18节点空卡再跑实验，选择理解损失降低20倍的对照，用于与普通joint（65761，已完成）及梯度隔离（65831，运行中）比较。新attempt13_geometric_sam_weight005_node18_8gpu，18节点8×4090、32CPU、192GiB，独立不可变代码 /space/mawb/ssst_gfm_v4_weight005_code_20261016。

总loss = L_recon + [0.05 * min(1,(joint_update+1)/1043)] * L_under。首epoch理解权重从0.05/1043升至0.05，之后保持0.05；重建/几何/理解LR仍1e-4/1e-5/1e-4，首100步LR warmup和4epoch cosine、global batch8、global clip1、4epoch4172更新不变。

这是普通DDP+总loss.backward，保留理解loss到重建参数的梯度，没有梯度隔离或重建冻结，没有新结构、参数或额外loss。0.05是预设的20倍缩放对照，不是由SIU3R权重直接推导的最优值，也不是对验证指标调参选择的结果。

## 可比性与预检

与65761/65831共用未训练checkpoint_initial.pt，0 short/0 adaptation/0 full、空AdamW、同649参数张量和8rank RNG。源SHA b9d7bd597cc2bd0d2439a1eae3771760f2be55241a2db1c4ef3879b08a75a03e。严格元数据检查只允许loss权重/config recipe和执行SHA变化；所有旧参数、optimizer group和rate保持相同。重新渲染initial monitor时保留训练RNG，不使用训练后的权重。

source0步真实预检已通过。25项CPU检查通过：包括损失值与解析梯度验证，确认理解梯度仍作用到重建参数，只缩放权重；拒绝已有训练曝光/optimizer历史/学习率变化；既有初始化/冻结/warmup/策略回归。编译、shell语法和diff检查通过。

硬件与普通joint/隔离组不同：本组18节点4090，对照13节点3090；训练精度和batch相同，但不能保证逐位相同轨迹。默认数学之外没有额外新分支。

## 监测与比较

journal记录真实有效理解权重及0.05缩放系数；monitor继续使用未缩放理解loss作为诊断值，不能把monitor总loss直接当作实际加权优化loss。每100步的固定8验证增加与隔离组相同的深度非有限/导出零值比例诊断，不修改官方深度评测。

训练后接同一128场景官方评测，13列11有效值、两项text留空。比较任务等待本组评测与65832结束，再比较普通joint、权重0.05 joint、梯度隔离joint、V4适配后冻结及旧C参考。若权重0.05恢复质量，支持损失/梯度平衡是可用修复方向；若仅隔离恢复，仍不能凭单个权重点断言纯方向冲突，需要进一步梯度诊断。

65831本次资源核实时已约150步，ISOLATED_GRADIENT_SYNC_FIRST_UPDATE.json确认8rank更新后全可训练参数SHA一致，重建/理解裁剪前梯度范数首步0.5297/0.00909，无失败记录。本组不修改或重启它。

实际job ID以attempt13/submission.json为准。提交后只核对一次分配并停止轮询，等待用户通知。

## 已提交并停止轮询

训练65838，单次确认RUNNING，18节点8×4090；官方128评测65839 afterok:65838；权重0.05/隔离/普通联合/冻结/旧C比较65840 afterany:65832:65839。执行SHA06bed1b8631531882080757fcc46c4721df4560d。这里只确认调度分配；GPU首步尚未宣称通过。未更改13节点65831。
