> **最新状态：2026-10-22 已核查完成。** 70026/70042/70162/70183/70234/70235 均 COMPLETED、exit 0。两项训练各4172 updates，两个官方1860评测与128窗纯诊断均成功。实际报告见 [完成核查与结论](../anchor_child_boundary_results_20261021/REPORT.md)。下文 PENDING 是历史提交时快照，禁止据此重提作业。

# 提交记录与结果交接

执行 commit：`3ac6b4f1878168929dfb3b773710acc193eb4676`。
执行快照：`/space/mawb/ssst_anchor_child_boundary_diagnostics_code_20261021_v1`，保持不可变；本提交只在独立 publication worktree 增补文档。
运行目录：`/space/mawb/ssst/group_plus/anchor_child_boundary_diagnostics_20261021_v1`。

2026-10-21 12:22:38 UTC 的唯一一次有界调度核对：

|作业|内容|资源|状态|原因|
|---|---|---|---|---|
|70234|两个固定窗 × A4/A4+4 的纯前向 smoke|单节点 1 GPU|PENDING|Resources|
|70235|128 个配对场景诊断、六图、统计与报告|单节点 8 GPU|PENDING|Dependency afterok:70234|

可用节点池为12/13（3090）、14/17/18（4090），具体节点待分配。
没有提交训练。原 checkpoint、模型架构与其它作业未修改。
8 项 CPU 验证和实际 checkpoint 身份/SHA/续训核对通过，见 CPU_VALIDATION.json / PREFLIGHT.json。
GPU forward、数值等价、权重不变收据及实际指标均**尚未验证**；PENDING/RUNNING 不能视为通过。
已停止轮询，不根据本文状态重新提交。

## 用户通知完成后的步骤

只检查这两个作业、运行日志、smoke/SMOKE_PASSED.json 或 JOB_FAILED，以及 full/COMPLETE.json。
若 smoke 或 full 失败，先报告原因与明确的下一步；不静默重试，不跳窗，不调阈值。
完整成功时检查128窗覆盖、256个Context视角/checkpoint、所有rank固定权重哈希、反向调用与参数更新为0、配对GT/RGB/source_superpoint身份一致。
依据 paired_scene_statistics.csv 与 conclusions.json，分析跨实例 child/final cosine gap 与 membership JS gap、覆盖率、Bernoulli/categorical entropy及Boundary准确性/伪边界。
不要从已有1860指标推断本轮128窗结论，也不以单纯高cosine证明共享父特征压制。

将 `full/publication/` 的实际报告、统计表、收据、前4个固定窗口16幅六图复制到本 publication worktree 的 `docs/anchor_child_boundary_results_20261021/`，审核后推送本 GitHub 分支。
连续 raw/soft NPZ 与全部六图留在运行目录，不上传权重、完整贡献矩阵或大量中间张量。
publication worktree：`/space/mawb/ssst_anchor_child_boundary_publication_20261021_v1`。
GitHub 分支：`diagnostics/anchor-child-boundary-20261021`。
执行 worktree 的 HEAD 保持上述 commit，以免破坏 smoke/full 代码身份断言。
