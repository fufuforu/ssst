# A 双分支点云1024重建token：八卡full4提交

北京时间2026-10-19 10:52:52，已提交唯一训练作业 **67952**。

- 申请节点：**3dimage-13 / 8×3090 / 32 CPU / 192G**；时限24小时，不是预计耗时。
- 唯一一次调度核对：`67952|PENDING||(None)`。当次未见JOB_FAILED；GPU首步、反传、参数更新及八卡同步尚未核实。
- 无独立评测或汇总作业，无后台监控。用户本次授权训练，没有追加其他实验。
- 输出目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt21_A_dual_point1024_full_node13_8gpu`。
- 独立固定代码快照：`/space/mawb/ssst_A_point1024_full_code_20261019`；HEAD `58e86fa3b17ea044edbca46906546417c825102e`，另有11个文件的SHA清单 `SOURCE_MANIFEST.json`。
- 日志：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt21_A_dual_point1024_full_node13_8gpu/slurm-67952.out`。

## 架构与训练预算

基线是A full4 `attempt05_legacy_dual_superpoint_full_node13_8gpu`（PSNR18.5003、Target PQ62.7599%）。
新注册键 `dual_point1024`，架构 `OMEGA_POINT1024_RECON_SUPERPOINT_UNDERSTANDING_DUAL_V1`。
重建侧恢复原GFM双分支的点云网格/置信度筛选和确定性FPS1024锚点，
学习基础token+几何/RGB内容+来源对应GFM投影；每锚点64 GS，总65536 GS。
理解侧继续A的动态超点query，不将1024个锚点假定为1024个实例query；GS父关系使用 `gaussian_parent_anchor`。
宽度1024/256、12层、L6/L8/L10/L12交互、Write=0、正常联合梯度、损失、匹配和后处理保持A逻辑。

完整1191场景/8337窗口，4epoch、4172更新、全局batch8、8rank每rank一场景。
AdamW LR1e-4、100步warmup/cosine，理解权重.05及1043步warmup，weight decay.05（1D为0），clip1。
每50步保存checkpoint，每epoch保留权重；初始和epoch末保留原固定探针。
没有16窗口短测、额外适配训练或已训练A权重微调。

## 初始化及预检证据

A的真实零步来源：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt`，SHA256 `3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744`。
新模型664个张量，655个共享张量逐值与A零步相同；移除38个旧重建初始化张量，新增9个张量以seed42初始化。
模型不同，不能声称全部参数与A逐值相同。progress全0、空Adam；验证共同8rank零步RNG，经严格1→8扩展后与A完全相同。

新初始文件：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt21_A_dual_point1024_full_node13_8gpu/checkpoint_untrained_initial.pt`；SHA256 `0ea7d0a97743f06bf2e38595c602d6b2a6557262dbe2791db96060a1833a6e64`。
保存前后均通过 `validate_zero_initial`，全部664张量保存前后逐值相同且有限。
实际冻结资产SHA、训练/heldout清单SHA一致，33376次历史A训练曝光顺序全部可重建；实际构造首末训练窗口和heldout样本成功。
20项针对性CPU检查在同一对话前一阶段通过，模型/测试代码未再修改；本次执行了实际零步预检、快照SHA、启动前verify-only、Python编译、shell语法及diff检查。
CPU替代renderer不代表真实GPU渲染通过。本地主机旧CUDA驱动与glibc提示不作为训练GPU验证。

## 后续接手

先读本报告及run下的preflight、submission和SUBMISSION_ONCE_CHECK；旧PENDING仅为10:52:52历史核对。
**不要重复提交67952，不自动恢复或重训，不用缺少指标判断训练失败。**
用户通知完成/检查后，再一次性检查调度、日志、STARTED、FIRST_GRADIENTS_rank00–07、FIRST_UPDATE_VERIFIED、TRAINING_COMPLETE及checkpoint：
完整4172步/八rank连续日志、实际初始化一致、参数更新与Adam步数、固定探针、数据顺序及有限性。
若遇JOB_FAILED，先定位工程错误，保留原始日志和权重。
本次没有自动官方128评测，训练后仍需按后续授权安排。

现有67940及所有旧运行目录/快照未修改；没有取消、恢复或查询其他作业。本次没有commit或push。
新1024锚点配置尚无实际GPU吞吐，暂不提供可靠总耗时预测；申请24小时仅为调度时限。

**已停止轮询，等你通知完成后再检查。**
