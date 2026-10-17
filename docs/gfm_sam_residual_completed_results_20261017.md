# 66353：SAM残差初始化隔离训练完成分析

结论：**重建保持，理解没有整体改善；不支持把learned-base＋SAM残差初始化作为有效的整体质量改进。** 它是旧V4双分支架构的初始化单变量对照，与当前Unified Query全量训练分别记录。本次仅核对已完成任务，无新训练、无重跑、未查询或改动66432/66433。

## 实验与完成证据

相同共同零步/空Adam/八rank RNG、同V4几何初始化及局部c/s、完全梯度隔离、4epoch4172更新；唯一新增learned-base＋归一化SAM内容/几何残差的门控，初值0.1。并非三维ESAM超点初始化或多对一匹配。

- 训练66353 COMPLETED/0:0，2026-10-17 08:50:22–13:24:30，耗时4小时34分08秒。
- 正式128评测66354 COMPLETED/0:0，耗时11分27秒。
- 比较66355 COMPLETED/0:0，耗时1秒。
- 八rank分别连续1..4172更新，loss/梯度/PSNR/门控值均有限；首更新全部可训练参数跨rank SHA一致。
- 完成收据确认adapt/short0，4172 full更新、33376曝光、8337唯一训练窗口；冻结权重未变，各训练参数族及门控有更新。
- 正式导出128唯一场景，11项指标均有限、两列文本mIoU空；native/CSV一致性检查通过，与V4/V3比较的128 RGB cohort一致。

## 正式指标：原V4隔离 vs SAM残差隔离

理解指标以百分数表示，差值为百分点；Target按既有官方context＋4个extra视图协议，不能称为纯novel-only。

| 指标 | 原V4隔离 | SAM残差66353 | 差值 |
|---|---:|---:|---:|
| AbsRel ↓ | 0.05463 | 0.05458 | -0.00005 |
| RMSE ↓ | 0.17001 | 0.17136 | +0.00135 |
| PSNR ↑ | 23.19 | 23.29 | +0.10 |
| SSIM ↑ | 0.76686 | 0.76915 | +0.00229 |
| LPIPS ↓ | 0.38911 | 0.38905 | -0.00006 |
| Context mIoU ↑ | 48.75% | 47.43% | -1.33 pp |
| Context mAP ↑ | 18.79% | 20.28% | +1.50 pp |
| Context PQ ↑ | 54.80% | 53.74% | -1.05 pp |
| Target mIoU ↑ | 49.26% | 47.31% | -1.95 pp |
| Target mAP ↑ | 18.98% | 18.24% | -0.74 pp |
| Target PQ ↑ | 53.62% | 51.74% | -1.88 pp |

重建PSNR增加0.10 dB，深度和LPIPS近似保持。Context mAP增加1.50个百分点，但context mIoU/PQ以及三个target理解指标均下降，target mAP减少0.74、PQ减少1.88个百分点。因此不是一次稳定的理解提升；尚无多seed显著性验证，不能把小差异解释为确定因果。

旧VGGT＋MASt3R C：PSNR22.5966、target mAP27.3756%、PQ60.9790%。本组PSNR23.2936较高，但target mAP18.2400%、PQ51.7436%仍明显落后旧C。不能用更高PSNR替代理解质量判断。

## 门控与初始化解释

门控八rank均从约0.100000→0.0989714，训练期间范围约0.0989565–0.1000151，没有塌到0。完成收据也记录sam_residual_gate参数族更新。因此结果不是残差完全关闭造成的。

但门控非零/梯度非零**不等于SAM候选获得最终GT匹配或保留为实例输出**。本组训练/官方评测未执行新的按SAM来源划分的候选存活固定权重审计，不得声称已修复此前SAM Query被淘汰的问题；也不能仅凭最终mAP推断SAM全被淘汰。已完成正式结果足以说明该初始化修改没有带来整体收益。

建议将其作为旧双流的完成负对照归档，保留当前Unified Query全量实验；新模型还同时改变超点表示、共享Query、实例分配和推理去重，未来比较不属于单变量因果消融。本次不增加旧架构训练或GPU诊断。

运行：`/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt16_learned_base_sam_residual_isolated_node13_8gpu`；完成复核收据COMPLETION_REVIEW_20261017.json。完整对比位于`/space/mawb/ssst/group_plus/dual_stream_training_v1/comparison_sam_residual_vs_v4_isolated_20261017/comparison.md`。训练执行SHA767cbe87。
