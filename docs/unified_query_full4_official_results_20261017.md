# 原Unified完整4epoch：官方128结果

用户通知评测完成后审计：66433运行12m55s后在compare_completed_controls因旧平铺收据读取触发KeyError退出，官方128导出、评分与COMPLETE_VALID_11_METRICS均在此前成功。66528纯CPU补汇总COMPLETED/0:0、耗时1秒，已完成控制比较，没有重复训练、导出或评分。不能把66433的Slurm FAILED解释为训练失败或正式指标缺失。

## 结果完整性与口径

导出128条唯一场景记录，EXPORTED；官方评分HELDOUT128_OFFICIALLY_SCORED，commit8ea80166be76854f938e90521f1a5b688b755c87，无非有限指标，11个数值/2个文本空列，CSV与官方native一致；COMPLETE检查通过。与V4两组核对相同128 RGB/帧/cohort。旧C基准仍为锁定参考结果，导出复用相同参考协议与GT。当前128固定子集，不是全部官方验证集。

目标视图采用既定官方context＋四额外帧口径；CSV字段保留历史novel名称，不应解读成单独纯novel四帧。深度采用既定PNG量化及官方逐图scale+shift拟合，不是绝对尺度几何误差。context-only生成，目标相机独立标定，context同类Mask NMS>0.7后复用实例身份，官方像素组装不改。

分割指标以下转为百分数，差值为百分点；AbsRel/RMSE/LPIPS越低越好，其余越高越好。

| 指标 | 旧C：VGGT+MASt3R | 原Unified full4 | Unified减旧C |
|---|---:|---:|---:|
| AbsRel ↓ | 0.08730 | 0.08884 | +0.00154 |
| RMSE ↓ | 0.21321 | 0.21764 | +0.00443 |
| PSNR ↑ | 22.5966 | 18.0855 | -4.5111 |
| SSIM ↑ | 0.78076 | 0.63480 | -0.14596 |
| LPIPS ↓ | 0.36805 | 0.60988 | +0.24183 |
| Context语义mIoU ↑ | 56.0237% | 43.8763% | -12.1474 |
| Context mAP ↑ | 30.7991% | 18.9247% | -11.8744 |
| Context PQ ↑ | 64.5517% | 46.7337% | -17.8180 |
| Target语义mIoU ↑ | 55.2647% | 43.7519% | -11.5128 |
| Target mAP ↑ | 27.3756% | 16.5192% | -10.8564 |
| Target PQ ↑ | 60.9790% | 45.0114% | -15.9676 |

其他已完成对照：V4隔离组PSNR23.1940/target mAP18.9803%/PQ53.6217%；V4理解权重0.05组PSNR21.9101/mAP19.8482%/PQ58.7933%。Unified的context mAP18.9247%略高于隔离组18.7879%，但多数质量指标更弱，不能将此例外夸成整体优势。

## 结论与下一步边界

1. 原Unified在当前从零步联合4epoch配方下能正常学习，但完整质量未达到旧C，也整体弱于近期双分支控制；不能把短测成功或loss下降当作达到基线的依据。
2. 退化不仅是类别/最终Mask组装：RGB重建PSNR下降4.51dB，SSIM与LPIPS同步恶化。理解同样明显落后，未形成重建/理解两者互补的收益。
3. 深度分数接近旧C，只说明对齐后的深度指标相近，不能排除尺度、相机、几何或Gaussian重建问题。
4. 目前不能据此唯一归因于统一Query、初始化、特征适配、模型容量或任务梯度：与旧C/控制组结构、参数量、源权重和训练配方都存在差异。
5. 继续完成已授权A66506/B66509，做相同完整数据预算和正式128评测。A检验恢复双分支图像适配/交互/读出；B检验共享主干下直接Mask/语义头方案。两者均包含多个变化，不是单因素因果证明。此次没有新开训练或改变正在跑的作业。

证据：run/evaluation/{export_complete.json,metrics.json,siu3r_table1.csv}，run/comparison_to_completed_controls/{comparison.csv,comparison_status.json}，run/COMPARISON_RECEIPT_FIX_COMPLETE.json。run根为`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt02_esam_many_full_node11_8gpu`。停止轮询，待用户通知A/B完成后再检查。
