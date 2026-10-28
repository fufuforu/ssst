# Baseline8冻结Token/GS质量头正式评测完成（2026-10-25）

按优先提升mIoU/mAP的目标，C8_Cal（Token+GS+校准）成为当前本地首选。正式1860窗口、312场景，Target统计全部6帧（含2帧Context）；不是119场景dev。单种子，正式GT不用于选epoch或调参。

71685导出在12服务器8卡COMPLETED/0:0，35:30；71686评分在11服务器5卡COMPLETED/0:0，1:03:24。恢复共1:38:54（不含排队），未重做缓存或训练。完整五案例/RESULTS/COMPLETE均生成。

|指标|Baseline8|Token8 排序|Token+GS8 排序|Token8 校准|Token+GS8 校准|旧Token+GS4 校准|
|---|---:|---:|---:|---:|---:|---:|
|AbsRel ↓|0.05036|0.05036|0.05036|0.05036|0.05036|0.05280|
|RMSE ↓|0.17094|0.17094|0.17094|0.17094|0.17094|0.17653|
|PSNR ↑|23.63080|23.63080|23.63080|23.63080|23.63080|23.05168|
|SSIM ↑|0.78454|0.78454|0.78454|0.78454|0.78454|0.76854|
|LPIPS ↓|0.35547|0.35547|0.35547|0.35547|0.35547|0.38556|
|Context mIoU ↑|58.68116|58.68116|58.68116|59.36503|59.80950|59.35329|
|Context mAP ↑|25.12195|26.40893|27.36793|26.94350|28.43290|27.75315|
|Context PQ ↑|68.75659|68.75659|68.75659|67.51985|67.61981|66.78950|
|Target mIoU ↑|58.35304|58.35304|58.35304|59.11291|59.52567|59.14866|
|Target mAP ↑|23.62591|24.85313|25.77186|25.47526|26.85339|26.26957|
|Target PQ ↑|66.72041|66.72041|66.72041|65.79164|65.60599|64.97899|

## 收益分解

- 同一Baseline8、同一冻结前向、相同IoU监督：C8_Cal相对A8_Cal的Target mIoU +0.412756pp、mAP +1.378128pp、PQ -0.185656pp；Context分别+0.444470、+1.489395、+0.099961pp。这是当前最直接的Token+GS对Token-only配对比较。
- 固定Native8掩码，只换score：C8_Rank相对A8_Rank，Target mAP +0.918733pp、Context mAP +0.959003pp；mIoU/PQ/mAR100严格相同。
- C8_Cal相对Native8：Target mIoU +1.172627pp、mAP +3.227477pp，PQ -1.114424pp。校准的mIoU/AP收益仍伴随PQ代价。
- C8_Cal相对C8_Rank：Target mIoU +1.172627pp、mAP +1.081526pp、PQ -1.114424pp。PQ优先时可保留C8_Rank为备选（58.35304/25.77186/66.72041）。
- 新C8_Cal相对旧C4_Cal：Target mIoU +0.377011pp、mAP +0.583819pp、PQ +0.626997pp，Context +0.456216/+0.679749/+0.830315pp；三项共同提升，但此比较改变了Baseline源检查点和头训练，不能全部归于GS。

## 重建与历史可比性

五个质量头案例共用同一主模型渲染，因此重建五指标相同。PSNR23.63080/SSIM.78454/LPIPS.35547/AbsRel.0503643/RMSE.1709432。相对epoch4的重建改善来自原Baseline继续训练到epoch8，不是额外MLP改善了渲染。

当前Native8与原历史Baseline8重建及mAP完全一致；mIoU/PQ最大绝对差仅2.98e-6pp，为数值汇总差。没有发现本次固定renderer/重导出造成可见指标漂移。

相对用户总表的SIU3R参考数值，C8_Cal的Target mIoU +0.325669pp、mAP -0.286614pp、PQ +0.655987pp；Context mIoU +0.589503pp、mAP +0.262900pp、PQ +1.499810pp。这些SIU3R数字不是本次本地重测，不宣称同设置下全面胜出；本模型PSNR/SSIM/LPIPS仍明显落后该参考。

## 训练和部署方式

主模型为已完成的Baseline epoch8，SOURCE_SHA bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5，epoch8/full_updates8344。本实验冻结主模型，只训练A8(66561参数)和C8(74753参数)小质量头，两头各6张量更新。原class-agnostic Context Native-mask IoU作MSE监督，GS32不接收GT输入，scaler只用1072 train场景统计；119个头dev场景选择best epoch15，不用312正式场景GT选参数。头各运行2700训练步；恢复阶段训练步为0。

C8_Cal用同一预测IoU替换JSON排序分数，并以(q/q.mean).clamp(.5,2)^.25调整像素竞争；原候选阈值/NMS/面积规则不变。此结果沿用原质量目标，与此前类别感知G1/G2/G3失败的头不同。正式C8头：运行目录heads/C/head_best.pt（只读指向原v3已完成头）；其SHA9de8c6f6096ca3fb6fed9c60b44112ba0d9f7029f298bd344f8084ba81b0d130。

## 限制与后续

单种子；A8和C8参数量略不同，未做epoch8 shuffled-GS容量对照，不能将全部差异解释成已经严格证明的几何因果机制。原IoU目标在当前配对实验中有实际增益；不因之前同类别IoU头的负结果推论质量监督整体无效。建议先保留C8_Cal作mIoU/mAP主方案，C8_Rank作PQ备选，再围绕校准引起的像素竞争/PQ下降做诊断。本次没有新增实验或作业。

完整机器结果为运行目录RESULTS.json；逐项核查及文件哈希收据保存为COMPLETION_AUDIT_20261025.json，CSV为formal_11metrics_comparison.csv。Rank分支部分只读链接使用计算节点/datapool/mawb绝对路径，在登录节点以等价/space/mawb映射做只读核查，不改归档输出；重新评测或迁移需保持对应路径。

最终独立核查PASSED：全部66960份GT文件哈希匹配，29760个仅排序PNG与Native硬链接一致，7440处计算节点GT链接作只读路径映射。五案例RAW指标重新展平、单位、配对差值及场景覆盖一致；没有重新执行GPU评测。导出/评分日志均无Traceback、无哈希重读事件，源PT在两个阶段前后SHA均OK。当前两项均已完成，无新提交作业。
