# VGGT 适配重建2 epoch → 冻结联合4 epoch：完整官方评测

训练与评测已正常完成。北京时间 2026-10-12 03:09:30，训练作业61835（node13，四卡）完成，耗时17:45:27；03:41:34，评测作业61836（node11，四卡）完成，耗时00:32:04。两者均为 COMPLETED / exit0:0，评测无资源等待。

正式训练总计6258次更新、50064新增曝光：重建适配2086次/16688曝光，冻结联合4172次/33376曝光。最终 checkpoint 为 checkpoint_frozen_joint_epoch_04.pt（总epoch6）。严格加载并独立校验 adapted_vggt.pt，未回退到原始冻结权重。没有追加训练或评测作业。

下表沿用 SIU3R Table1 的13个指标列，每个模型只有一行；mIoUs、mAP、PQ使用0–1尺度。两处文本 mIoUt 未训练，留空。

| 方案 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUs↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUt↑ | 新视图 mIoUs↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUt↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 冻结 VGGT · epoch4 | 0.18202 | 0.3937 | 18.45 | 0.6486 | 0.6145 | 0.5596 | 0.2408 | 0.6166 |  | 0.5566 | 0.2274 | 0.5989 |  |
| 冻结 VGGT · epoch8 | 0.18276 | 0.3969 | 18.55 | 0.6491 | 0.6177 | 0.5553 | 0.2378 | 0.6184 |  | 0.5518 | 0.2240 | 0.6029 |  |
| 适配重建2 epoch → 冻结联合4 epoch | 0.16726 | 0.3725 | 19.37 | 0.6744 | 0.5536 | 0.5634 | 0.2503 | 0.6262 |  | 0.5586 | 0.2334 | 0.6071 |  |

重建5列及新视图理解字段严格使用固定官方 Evaluator 的原生汇总。官方 target_ids 包含2张输入帧和4张额外帧，因此表中“新视图”沿用该官方集合。

相较原冻结 epoch8，11个已测指标均改善：PSNR +0.81856 dB，SSIM +0.02532，LPIPS −0.06415，AbsRel −0.01550（相对下降8.48%），RMSE −0.02440。输入理解 mIoUs/mAP/PQ 分别提高0.81/1.25/0.78个百分点，新视图分别提高0.68/0.94/0.42个百分点。相较原冻结 epoch4，11个指标也均改善。

这支持当前“先适配重建，再冻结”的组合方案有效；重建改善更明显，理解改善较小。但本次新方案共6 epoch、原对照8 epoch，且四卡累积与原八卡的物理执行不同，因此不能将改善单独归因于 AA 微调，或断言单纯增加训练时间就能弥补旧版本的差距。这次不是某一模块的独立消融，也没有多随机种子统计。后续修改需围绕具体假设做对照。

完整性与协议校验 PASS：312场景/1860窗口/11160官方目标图像，无遗漏或重复；所有指标有限；四个分片真实 CUDA 官方一致性检查均 PASS；原生重建加权、全局 COCO AP 合并、13列取值和文本留空均核实。monitor_v1 保留全部31个告警窗口（1829个OK）；所有拟合均数值有效。此前原冻结版本为52个告警窗口，告警减少不等于相机几何已被完全修复。

执行与证据：

- 训练、导出与评测执行 SHA：`39bf57fa9f898e6604261aced9a8190500390adf`。
- 固定官方 SIU3R SHA：`8ea80166be76854f938e90521f1a5b688b755c87`，源代码无修改。
- 最终 checkpoint SHA256：`01d49b092c8dfc2e02f15b5216e5d6ed1cfd88fae2cf693ee69334e45a9410ba`。
- adapted VGGT SHA256：`c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace`。
- 当前报告与收尾脚本提交仅补充完整性验证和记录，不改变上述实际执行快照、模型、损失、指标或原始结果。

[完整13列 CSV](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1/evaluation/final_adapt2_joint4_official/siu3r_table1.csv) · [三版本对照 CSV](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1/evaluation/final_adapt2_joint4_official/comparison_siu3r_table1.csv) · [原始指标](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1/evaluation/final_adapt2_joint4_official/metrics.json) · [最终验证](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1/evaluation/final_adapt2_joint4_official/final_validation.json) · [覆盖及告警](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1/evaluation/final_adapt2_joint4_official/coverage_and_geometry.json) · [对照验证](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_v1/evaluation/final_adapt2_joint4_official/comparison_validation.json)
