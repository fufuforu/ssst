# 双流真实权重冒烟审计：65163

用户通知结束后检查，Slurm COMPLETED / 0:0，耗时3分44秒。产物目录 `/space/mawb/ssst/group_plus/dual_stream_v1_smoke/attempt01` 中 COMPLETE 状态 REAL_OMEGA_SAM_FORWARD_AND_CALIBRATION_PASSED，无 FAILED.json。代码提交4b433c2。

真实两张context前向输出1024 anchors、65536 Gaussians、100 thing queries＋2 stuff membership；输出有限性、Omega/SAM冻结、Write严格为0、独立全窗口标定不改变已生成Gaussian/保留context相机均通过。前向23.82秒，CUDA峰值分配7.99 GiB，仅代表本次单窗口无更新验证，不能充当训练吞吐或反向显存。共享context标定scale=1.013800，归一化场景坐标残差median=0.0005925；单个窗口不足以证明跨场景标定稳定，不能换算成米精度。

该样例58个SAM候选query、42个learned fallback、5个跨视图合并组。未做GT关联，不能称58个正确物体或5个正确匹配，也未测试旧C的85个小物体覆盖率。

总参数1,467,572,932，可训练230,023,275。Omega与SAM为预训练并冻结；重建RGB+ray编码器、anchor内容MLP、重建decoder/Gaussian head、语义adapter/局部RGB分支、query初始化投影/回退query、object decoder及实例/类别heads均新初始化。未迁移旧C或QuerySplat已训练GS decoder权重。语义adapter当前requires_grad=True，但冒烟没有反向/更新。固定权重feature lifting内的renderer adjoint求导不等于完整训练反向验证。

建议下一步先冻结Omega/SAM、Write关闭，做少量固定训练场景的完整loss/反向/优化短验证，同时抽取不重叠验证场景。检查重建与实例分项loss、实际梯度/参数变化、训练显存和非塌缩输出，确认小样本能学到重建及实例对应后即可进入完整数据训练，无需再重复旧128场景诊断或完整旧2+4 epoch流程。总loss下降本身不足以排除no-object塌缩、背景拟合或视图记忆；需结合matched thing mask、类别和独立视图RGB检查。验证集指标不要求短程超过旧成熟模型。

adapter应训练，而Omega暂不必解冻；新分支先适配稳定的几何/视觉坐标。后续如全量验证呈稳定平台再单独评估Omega后部block/LoRA小学习率适配。当前FrozenOmega包含requires_grad冻结及forward的no_grad，仅调整optimizer不能解冻，需显式修改梯度路径并保持几何预测/相机标定契约一致。SAM继续冻结。没有证据要求照搬旧版的Omega微调时长。

本次仅结果检查及建议，没有提交新作业或启动训练。
