> **最新已授权并提交：Baseline8→16**。唯一新链71272→71273，八卡续训及afterok正式评测/4-8-16汇总；候选12/13/14/17/18，单次接收检查均PENDING、节点未分配。保留Adam/RNG，LR3e-6→3e-7，无rewarm；20项CPU及真实来源预检通过，GPU尚未核验，已停止轮询。见[16epoch续训交接](codex_handoff_baseline_extend16_submitted_20261023.md)。覆盖旧“8→16未授权”记录，勿重复提交。

# Baseline 8epoch 正式结果核查（2026-10-23）

训练71192与正式评测71193均COMPLETED/0:0，都在3dimage-12八卡执行；本次恢复训练耗时4:54:46（从5500到8344，非整个4→8训练耗时），导出/评分/比较耗时1:05:10。没有新作业、重复训练或重跑评测。

固定理解权重1.0、2context+2novel，与已完成同组4epoch对比。累计8epoch/update_in_epoch0/full_updates8344；原4epoch4172，本次恢复段新增2844，合计4→8新增4172。既定续训采用主LR约1e-5→3e-5→3e-6的追加曲线、几何LR为主组0.1倍；从5500恢复时延续时钟，没有重复升温，完整Adam和八rank RNG保留。不是把4epoch原曲线追溯拉长，因此结果是这套既定续训方案的收益。

全部11项主要指标向好，重建提升相对明显、理解小幅提升。单seed结果不能证明统计稳定性，也不能把全部收益唯一归因于增加epoch数。

|指标|4epoch|8epoch|8−4|
|---|---:|---:|---:|
|PSNR↑|23.051962|23.630804|+0.578841|
|SSIM↑|0.768544|0.784540|+0.015996|
|LPIPS↓|0.385600|0.355467|-0.030133|
|AbsRel↓|0.052728|0.050364|-0.002364|
|RMSE↓|0.176459|0.170943|-0.005516|
|Context mIoU↑|58.165844|58.681155|+0.515310|
|Context PQ↑|68.419362|68.756587|+0.337225|
|Context mAP↑|24.535201|25.121948|+0.586747|
|target-all6 mIoU↑|57.782599|58.353039|+0.570440|
|target-all6 PQ↑|66.064329|66.720411|+0.656082|
|target-all6 mAP↑|23.135531|23.625909|+0.490378|

理解指标为百分数，差值为百分点。target-all6包含两张Context，不是novel-only。深度使用既定SIU3R逐图scale+shift拟合，不代表绝对metric尺度误差。

|理解补充指标|4epoch|8epoch|差值（pp）|
|---|---:|---:|---:|
|context AP50|42.058048|42.182323|+0.124276|
|context AP75|23.649131|24.907835|+1.258704|
|context AR100|33.594367|34.578183|+0.983816|
|target AP50|39.790663|40.628427|+0.837764|
|target AP75|22.129527|23.058604|+0.929077|
|target AR100|31.912822|32.779479|+0.866657|

AR100为COCO IoU .50:.05:.95平均召回，不是Recall50。

评测1860窗/312场景、256²、SIU3R 8ea80166be76854f938e90521f1a5b688b755c87；官方评测src与训练执行快照a959b9168f4a3049688171079ac63e52d616273c均未修改。已有正式完成收据确认相同输入和GT、非smoke、评测参数更新0，全部指标有限。

最终checkpoint SHA256：bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5；恢复来源5500更新SHA13232437521ff24eaf7b808f344ea20a0a8be9294423ff743d175ec0689fe70c；4epoch参考最终权重SHA694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc。

既有评测自动生成comparison/epoch4_vs_epoch8.csv只含9项，本次完整指标表（11项主要指标及AP/AR补充）已保存至eval/comparison/epoch4_vs_epoch8_full_metrics.csv；原表和指标文件均保留。

本次只检查Baseline8。U005续训已取消，不恢复；Views2ctx4及两个IoU head实验未查询、未修改。两个IoU实验原先约定的Baseline4来源和4epoch结构对照不自动替换为Baseline8。8→16未授权，不自行提交。

独立完成核验已通过（COMPLETION_AUDIT_20261023.json）：最终checkpoint实际SHA与冻结收据一致且与train latest同inode；632模型张量及604份Adam状态全部有限，Adam step8344，八rank RNG完整；恢复以来实际改变604个模型张量。每个rank恰好2844条新增日志，连续5501–8344；恢复身份及首步八卡参数SHA一致；全部1860窗输入签名和GT manifest与4epoch逐窗相同。全量正式GT字节一致由已成功完成的导出门禁核验，本次未重读全部GT PNG，也未重新评分。

运行目录：/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_recovery_20261023_v2_jsonfix/Baseline
