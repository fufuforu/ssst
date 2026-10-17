# B直接Gaussian Mask完整4epoch：官方128结果

66509训练COMPLETED/0:0、耗时3h24m35s；66510评测及汇总COMPLETED/0:0、耗时12m13s，无EVALUATION_FAILED。官方EXPORTED的128条唯一场景、11有限指标/2文本空列、COMPLETE及CSV/native一致性检查通过。与原Unified/V4同128 RGB和帧cohort，官方commit8ea80166be76854f938e90521f1a5b688b755c87，评测无参数更新。不是全部验证集；target沿用官方context+四额外视图口径。深度仍是官方逐图scale+shift对齐，不能当绝对几何精度。

| 指标 | 旧VGGT+MASt3R C | 原Unified full4 | B直接Mask full4 |
|---|---:|---:|---:|
| AbsRel ↓ | 0.08730 | 0.08884 | 0.09016 |
| RMSE ↓ | 0.21321 | 0.21764 | 0.21762 |
| PSNR ↑ | 22.5966 | 18.0855 | 18.0839 |
| SSIM ↑ | 0.78076 | 0.63480 | 0.63399 |
| LPIPS ↓ | 0.36805 | 0.60988 | 0.61498 |
| Context语义mIoU ↑ | 56.0237% | 43.8763% | 42.6759% |
| Context mAP ↑ | 30.7991% | 18.9247% | 14.6674% |
| Context PQ ↑ | 64.5517% | 46.7337% | 46.3465% |
| Target语义mIoU ↑ | 55.2647% | 43.7519% | 43.1581% |
| Target mAP ↑ | 27.3756% | 16.5192% | 13.0896% |
| Target PQ ↑ | 60.9790% | 45.0114% | 45.6597% |

## 判断

B相对原Unified：PSNR -0.0015dB，SSIM -0.00081，LPIPS +0.00510；重建基本不变。Context mAP下降4.26个百分点，target mAP下降3.43个百分点；target PQ仅上升0.65个百分点，context PQ下降0.39个百分点。整体没有取得收益。相对旧C，PSNR仍低4.51dB、target mAP低14.29个百分点、PQ低15.32个百分点。

B固定4 heldout探针最佳正Query Mask IoU0.7005，高于原Unified0.6067，但不能将它当最终检测/分割质量。它仅评价有来源支持GT的最佳正Query，不惩罚所有错误候选、不评价完整置信度排序，也没有经过最终像素竞争。128场景官方mAP下降证明先前这个小样本诊断未预测到最终收益；尚不能唯一确定是候选、分类、评分排序还是组装问题。若以后对B追因，应检查全部候选的GT匹配、分数校准和组装前后召回，而不是继续只看最佳Mask。

原Unified/B共用共享Decoder和重建路径，两者正式PSNR都约18.08。这支持把共同重建配置列为优先排查对象，但并不证明共享Query本身就是根因；A的不同宽度、特征适配与交互仍需完整评测。B还新增直接语义CE，当前比较不是纯Mask头单变量消融。

当前不建议仅给B追加训练，先完成已授权A的正式对照。此次没有取消、重启或新开训练。A66506检查时仍RUNNING，2767/4172步（66.3%），按近期速度估计剩112分钟训练；66507等待训练，66511等待两组评测成功后自动生成A/B表。停止轮询，待用户通知后检查。

证据：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt06_unified_direct_masks_full_node18_8gpu/evaluation/{metrics.json,siu3r_table1.csv,export_complete.json}`及同run的`comparison_to_completed_controls/{comparison.csv,comparison_status.json}`。
