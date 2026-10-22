# 旧A4 V4init / NoCA 完成检查（2026-10-22）

两组训练和正式1860评测全部 COMPLETED / ExitCode 0:0；未重复提交。

|模型|PSNR↑|AbsRel↓|RMSE↓|Context mIoU↑|Context PQ↑|Target mIoU↑|Target PQ↑|Target mAP↑|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|old_A4|18.20926|0.18695|0.41005|55.94314|65.35344|55.23170|63.14721|19.10869|
|V4init|18.13261|0.21330|0.46093|53.04885|61.75372|51.39058|57.14629|14.51325|
|NoCA|18.12626|0.18926|0.41635|55.55143|66.15839|55.13307|63.85902|18.94069|

mIoU/PQ/mAP以百分数展示；Target是all6（含context），不是novel4。

- V4init：训练70715，评测70716；3dimage-13，各8×3090。耗时6:50:14 / 1:04:29。
- NoCA：训练70718，评测70719；3dimage-12，各8×3090。耗时6:18:36 / 1:09:37。
- Smoke70714/70717均COMPLETED；两组首步8rank权重哈希一致且发生更新。
- 两组8rank日志均4172条更新（1..4172）；训练完成收据及最终checkpoint进度均epoch4/update_in_epoch0/full_updates4172。
- 两组EVALUATION_VERIFIED均PASSED；独立重比对1860窗输入签名及完整GT manifest，与旧A4相同；正式评测参数更新0，无非有限指标。
- 导出metadata的unique_scenes字段错误地记为1860窗；按records[].official_pair.scene去重实际为312场景，两组均如此。该命名问题不改变cohort或指标。

V4init相对旧A4：PSNR -0.07665 dB；AbsRel +0.02635（约14.10%）；RMSE +0.05088（约12.41%）；Context/Target PQ -3.59972/-6.00092点；Target mIoU -3.84112点，mAP -4.59545点。Target PQ仅curtain改善，其余19类下降。完整初始化方案在当前联合训练配方与预算下无收益；不能据此单独否定持久可学习基底，因Anchor/GS数量、内容编码和视觉投影同时变化。

NoCA相对旧A4：PSNR -0.08300 dB；AbsRel +0.00231；RMSE +0.00630；Context/Target PQ +0.80496/+0.71182点；Target mIoU -0.09863点、mAP -0.16800点。Context/Target各14/20类PQ提高；Target较大的正差为shower curtain +4.644、bookshelf +3.910、refrigerator +2.375点，较大的负差为window -0.940、table -0.841点。Things平均+0.78124点，stuff平均+0.08697点。

原官方评分只持久化PQ等最终结果，没有TP/FP/FN原始累计状态。本次未重新评分或生成预测，因此不能判断PQ改变由FP、FN或匹配质量中的哪项驱动；单训练seed也不能确认稳定增益。

当前建议保留旧A4作为基线。NoCA可视为理解读出路径的简化候选，暂不足以判定全面更优；当前结果不支持优先追加V4init+NoCA组合。没有新增实验、修改训练代码或SIU3R/src。

详细权重核验见RESULT_CHECKS_20261022.json；各组原收据在train/和eval/。
