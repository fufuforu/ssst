# Baseline8 Token+GS质量头正式完成交接（2026-10-25）

此文替代旧71685/71686 PENDING交接。两作业已完成，不恢复旧失败作业、不重做缓存/训练、不新增实验。完整分析见[frozen_baseline8_gs_quality_completed_review_20261025.md](frozen_baseline8_gs_quality_completed_review_20261025.md)。

## 当前完成状态

71685 COMPLETED 0:0，12服务器8×3090，35:30；71686 COMPLETED 0:0，11服务器5×3090，1:03:24。完整覆盖1860窗口/312场景，Target all6含Context。五案例结果、全部RAW_METRICS、metrics、RESULTS、COMPLETE均通过。

独立运行：/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline8_gs_quality_20261025_v4_export_recovery

独立代码：/space/mawb/ssst_frozen_baseline8_gs_quality_code_20261025_v4_export_recovery

SHA：02392745114b538fe1c30901c5add8e6e18745e4。源Baseline8 SHA bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5，epoch8/fullupdates8344。缓存和A8/C8训练复用已完成v3，恢复过程没有重复训练；主模型更新0。官方评测器commit8ea80166be76854f938e90521f1a5b688b755c87不变。

## 正式Target结果

|方案|mIoU|mAP|PQ|
|---|---:|---:|---:|
|Native8|58.35304|23.62591|66.72041|
|A8仅排序|58.35304|24.85313|66.72041|
|C8仅排序|58.35304|25.77186|66.72041|
|A8校准|59.11291|25.47526|65.79164|
|C8校准|59.52567|26.85339|65.60599|
|旧C4校准|59.14866|26.26957|64.97899|

C8校准Context为59.80950/28.43290/67.61981。

以用户mIoU/mAP优先目标，C8_Cal为当前本地主结果首选，C8_Rank为保持Native8 PQ的备选。C8Cal-A8Cal Target为+.412756 mIoU/+1.378128 mAP/-.185656 PQ，说明同源配对中GS头有收益；C8Rank-A8Rank Target mAP+.918733，固定mask指标严格相同。C8Cal-Native8 Target为+1.172627/+3.227477/-1.114424，校准PQ代价仍存在。

C8Cal-C4Cal Target三项+.377011/+.583819/+.626997，但改变了源Baseline和训练数据特征，不能将收益全归因于GS。单种子，无epoch8 shuffled-GS容量控制，A/C参数量66561/74753不完全相等。此为冻结主模型的质量头应用重建GS信息，不是联合训练或双向主干改进的证据。

## 重建与可比性

Native8重建测量一次，五案例共享：PSNR23.63080379/SSIM.784539519/LPIPS.355466981/AbsRel.050364286/RMSE.170943181。与历史Baseline8重建和mAP完全相同，mIoU/PQ最大差2.98e-6pp。没有有意义的renderer/hardware漂移，重建相对Baseline4的进步来自源Baseline8，不是质量头。

恢复导出日志无[integrity]重读事件、无Traceback，导出/评分前后源PT SHA均OK，故本次未复现旧71664的哈希失败；不声称已证实旧错误的具体原因。8份runtime显示12节点3090/同一partial_1406 renderer，8份原缓存冻结证据通过。

Rank GT与渲染链接采用计算节点/datapool/mawb路径，登录节点不直接解析该绝对路径。独立核查用等价/space/mawb路径读取，保持归档文件不改；计算节点评分成功、Native8历史对齐，属于路径可移植性限制。

## 其他路线与后续

G3早已完成：119 dev Context2旧校准固定mask下新同类IoU排序mAP下降.332245pp，未通过门槛，正式正常SKIPPED。与G1/G2合并，不扩展该版本同类IoU头。GS8此次正结果沿用原class-agnostic IoU目标，与该新头不同。

建议保留C8_Cal作正式主方案；后续若用户授权改进，围绕校准造成的重叠像素/PQ代价诊断，不再仅凭头MSE替代下游指标。当前没有新增作业，不进行后台监控。

COMPLETION_AUDIT_20261025.json记录独立指标、文件/代码哈希、覆盖、GT及Rank hardlink核对；formal_11metrics_comparison.csv提供原用户总表格式的11指标。

主方案检查点/GS特征/校准公式/正式指标引用：新run/C8_CAL_MODEL_REFERENCE_20261025.json。该文件只是可复现实验引用，未执行部署或新增训练。

最终独立核查PASSED：全部66960份GT文件哈希匹配，29760个仅排序PNG与Native硬链接一致，7440处计算节点GT链接作只读路径映射。五案例RAW指标重新展平、单位、配对差值及场景覆盖一致；没有重新执行GPU评测。导出/评分日志均无Traceback、无哈希重读事件，源PT在两个阶段前后SHA均OK。当前两项均已完成，无新提交作业。
