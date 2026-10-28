# 固定权重Oracle诊断：71219失败核查及修复（2026-10-23）

本轮用户通知“已完成，请检查”后核验了终态和真实日志，未查询或修改原三条8epoch训练链。

|作业|终态|节点/资源|实际耗时|
|---|---|---|---|
|导出71219|FAILED，ExitCode=1:0|3dimage-18，6×4090、24CPU/192G|4分01秒|
|依赖评分汇总71220|CANCELLED，未启动|未分配|0|

6个rank全部在第一个窗口生成时失败。没有progress_rank收据、query_audit、export_complete、E0复现收据、COMPLETE或正式metrics；完成窗口=0。不能声称E0/E1/E2已经评测，也不能用原4epoch指标冒充新诊断结果。

根因是我新写的scripts/export_fixed_query_diagnostics.py外层使用torch.inference_mode()。模型内部V4InitIsolatedNoCACore调用lift_features，RendererTranspose.forward通过torch.enable_grad()和autograd.grad对临时dummy特征张量求渲染器伴随。这是推理算法的一部分，不是模型训练。inference_mode会阻断该局部求导，即使enable_grad也不能恢复，因此报“element 0 of tensors does not require grad and does not have a grad_fn”。不是6卡分片、源权重、显存、数据或NCCL同步故障。日志中的GLIBC预编译扩展警告已有兼容fallback加载成功，不是本轮根因。

最小修复：外层恢复为原正式导出器采用的torch.no_grad()，保留内部已有enable_grad，只对临时张量计算伴随，不调用optimizer或更新模型参数。模型、GT关联、Membership、评分、6卡分片及E0复现门禁不变。

独立修复源码：/space/mawb/ssst_fixed_oracle_diagnostics_code_20261023_v3_gradfix，分支experiments/fixed-oracle-autograd-fix-20261023。v1/v2已执行快照及失败目录均不修改源码或HEAD；新提交入口预先指向独立v3运行路径，防止覆盖旧失败目录。尚未准备/提交v3集群运行。

两项针对性CPU回归通过：使用实际RendererTranspose/lift_features代码及CPU线性renderer复现旧inference_mode失败；从实际新导出函数AST读取no_grad上下文，验证伴随、归一化evidence/gate/mass数学结果、参数version/内容不变、参数grad为空。未声称真实CUDA渲染或完整E0复现已通过。Python编译与git diff检查通过；既有9项Oracle测试所覆盖代码未改变，不重复运行。

失败运行根：/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v2_flexgpu。新审计文件：FAILURE_AUDIT_20261023.json、GRADFIX_CPU_TESTS.json、gradfix_cpu_tests.log；原export_FAILED.json和export-71219.out保留。

遵守“失败先诊断，不自动重启或覆盖旧目录”的既定约定，本次没有重提作业、没有训练、没有后台监控。后续若用户授权重新执行，在独立v3运行根完成必要源身份/调度预检，仍按6卡或实际可分配卡数导出。应先通过真实首窗E0复现门禁再扩展到完整1860窗，不能把本CPU回归当GPU成功。原三条8epoch链保持不变，不因本故障重训。
