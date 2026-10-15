# 306步后按学习证据进入全量训练

用户在检查65175结果后明确指出短验证不必要求损失固定下降，只要模型在学习即可。这覆盖此前“损失门槛失败后不放行”的建议；不是修改既有实验结果。原attempt04及其GATE_FAILED、checkpoint、监测记录保持原样。

新策略 `learning_evidence_v2`：将重建损失下降10%、理解损失下降5%移为observations，仍记录真假；其余原有学习/非崩溃条件保留，并明确检查训练监测指标有限。原 `legacy_v1` 仍可复现原失败结论。不修改模型、loss数学、学习率、batch、冻结策略、数据和官方评测。

新运行 `/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt05_full_node11` 保留源306步checkpoint、源SHA256、AdamW及4个rank RNG，复制其初始/306步monitor。源execution SHA固定为 `f5a6134260c66a4012d7850e2e6acf5c08a478a0`。只通过显式 `--resume-source-sha` 接受代码/门槛版本变化；数据、架构、配置、world size等metadata必须完全一致。普通resume继续严格校验代码版本。

恢复后重新评估已保存306步monitor，通过即保存short_passed checkpoint并直接开始全量；不再次执行短训练更新。顺便修复完成短验证后resume空循环导致gate变量未定义的问题。原始证据保留，新门槛结果单独写入新运行。

全量1191场景、8337窗口、有效batch8、4epoch，总4172次全量更新。每epoch补齐7个窗口曝光。使用11节点四张3090，每卡microbatch1/累积2、20CPU、96GiB内存。启动脚本 `scripts/launch_gfm_semantic_pool_v3_full_node11.sbatch`；不可变代码快照 `/space/mawb/ssst_gfm_semantic_pool_v3_full_code_20261015`。完成后沿原协议固定128窗口导出并官方评分。作业号和真实启动证据以新运行submission、RESUME_POLICY_TRANSITION、SHORT_PASSED及FULL_STARTED为准。

后续监测继续保留BCE、分类、IoU、PSNR和梯度范数；损失上升仍是风险信号，但不单独阻止全量适配。本次不新增结构实验。按用户要求，提交后只核对一次调度状态即停止轮询，等用户通知完成；只有FULL_STARTED存在才宣称真实全量更新。

已提交 **65199**，执行SHA为b403e7c；提交后单次核对状态PENDING/Reason=Priority，指定11节点四张3090，尚未证明首个全量反向更新。submission.json保存在attempt05_full_node11。当前文档提交不改变作业不可变快照。

验证：6项policy/cache/sampler测试通过，包含新策略允许损失上升而阻止缺失更新/非有限值，以及续训只允许显式代码/门槛变化、拒绝数据/world size/LR变化。训练脚本编译、Slurm脚本语法及diff检查通过。
