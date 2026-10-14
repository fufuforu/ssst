# 完整adapter0.01/feedback0.05权重：理解失配复核

用户2026-10-14授权复核已完整训练权重的两种失配：原生Hungarian匹配query有较好mask却预测no-object，以及实例置信度与mask质量排序不一致。此次不训练、不改encoder/decoder/head/loss/GC/clip/feedback，不启动新融合对照。

之前诊断的A/B/C均为适配epoch2起点的500步联合训练权重，adapter理解梯度系数都是0.01。A原视觉训练/feedback0.1；B视觉冻结/feedback0.1；C原视觉训练/feedback0.05。此前44/321的no-object失配和0.236063→0.322882的GT排序诊断均来自C，不是未调整feedback的A。

本次仅加载完整4 epoch/4172步联合训练的adapter0.01、feedback0.05权重：
`/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/checkpoint_frozen_joint_epoch_04.pt`。
checkpoint SHA ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0；累计6258更新/50064曝光，其中复用适配2086/16688，新增联合4172/33376。使用原生官方_restore_checkpoint，严格恢复recipe、适配资产、实际feedback与曝光时钟；没有优化器。

1. 原始query诊断复用此前seed4242固定随机128场景、每场景一个窗口的同一cohort，不重新按错误选样。四卡3090与完整官方导出硬件匹配，先四rank各一个真实窗口预检，核对原生panoptic PNG、RGB provider哈希及匹配字段；通过后同一不可变快照执行128窗口前向。GT仅在context-only generate后用于原生matcher和离线统计，不进入generate/feedback。记录原始mask、资格筛选、竞争、面积过滤、native Hungarian、no-object/类别概率，检查参数版本未变。
2. 排序诊断直接复用完整官方312场景/1860窗口的四份saved states，不重复渲染或GPU评分。分别对输入/新视图计算原生mAP/AP50/AP75，再固定mask和类别，以同类别GT IoU替换分数。另筛出与此前C相同248场景/1478窗口，核对GT完全相同，用于同cohort比较。GT排序是不可部署的诊断，不是正式成绩或保证可达收益。

旧CLI默认A/B/C行为保留；扩展原诊断脚本增加严格完整checkpoint来源和单组汇总，不新增训练runtime。几何政策monitor_v1不变，告警保留。输出root：`/space/mawb/ssst/group_plus/vggt_full_b005_bottleneck_recheck_v1`；各次执行使用已push的固定SHA快照，旧证据不覆盖，提交前检查重复作业。

此次复核只判断长期训练后上述现象是否保留，不直接归因某模块，更不确定整个decoder上限。下一阶段对照待结果分析后决定；不在本次叠加梯度loss分项重放。

## 完成记录

64161已COMPLETED0:0，5分55秒，四卡3090。固定执行SHA2fe23430e6dc76562c075ad5b7f5e493d16acc2a，
4次真实预检和128次正式冻结前向通过，模型参数未变、零优化器调用。
两种失配仍在：完整权重的好mask匹配query有50/334判no-object；完整新视图GT排序诊断0.246245→0.333883。
同cohort旧C500对比、完整性与解释见[vggt_full_b005_bottleneck_recheck_results_20261014.md](vggt_full_b005_bottleneck_recheck_results_20261014.md)。
本次没有提交后续结构训练或loss分项重放，用户队列为空。
