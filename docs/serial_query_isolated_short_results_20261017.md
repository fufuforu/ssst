# 串行Query短验证66603：训练成功，当前Mask学习弱

用户通知后核实：66603 COMPLETED/0:0，18节点4×4090，耗时25m23s。16场景256更新/2048曝光，非完整数据训练，也无官方128指标。初末探针、自动汇总和检查点齐全，无失败收据。此次不新开训练、不自动转全量。

## 更新完整性

4rank各256条连续日志、512场景曝光，所有浮点统计有限；首更新全参数SHA四rank一致。0步/空Adam起点；最终383/383可训练张量全部变化且有限、383个Adam状态均step256、4rank RNG齐全，冻结文件/数据身份不变。GPU allocated峰值7.47GiB，平均更新约5.48秒。新增语义memory/seed/Decoder参数变化范数1.542/0.553/22.669，非未使用模块。首步重建/理解梯度范数0.03194/0.001613，均有限非零。

额外CPU核对：串行初始化与原Unified短测初始化共有301张量，301/301逐位相同；新增语义模块独立初始化。不能归咎于原重建/Mask头初始权重换了一套。仍存在新增4层语义Decoder/语义memory、梯度隔离和分别clip等多个变化，非单因素消融。

证据：run/RESULT_AUDIT.json、SOURCE_AND_CANDIDATE_AUDIT.json、SHORT_VALIDATION_SUMMARY.json、FIRST_UPDATE_VERIFIED.json。

## 相同固定探针结果

仅4train/4heldout。最佳Mask IoU对有来源支持GT取正匹配Query中的最佳Mask，再按GT等权汇总，尚未经过类别筛选及panoptic组装，不能替代mAP/PQ。train支持GT20/24、heldout20/20。

| 方案 | Train PSNR | Train最佳Mask IoU | Heldout PSNR | Heldout最佳Mask IoU |
|---|---:|---:|---:|---:|
| 原Unified短测66414 | 19.721 | 0.2673 | 10.049 | 0.0722 |
| A短测66502 | 18.861 | 0.6732 | 11.265 | 0.3478 |
| B短测66503 | 19.577 | 0.5171 | 10.998 | 0.2859 |
| 串行隔离66603 | 19.892 | 0.1438 | 10.213 | 0.0517 |

串行初始→最终：train PSNR10.935→19.892、最佳Mask IoU0.1207→0.1438、正Query分类3.49%→91.28%；heldout PSNR11.625→10.213、IoU0.1120→0.0517、分类0→34.27%。理解loss train2.078→0.976、heldout2.087→1.752。重建和类别确实在学习，但Mask提升不足，heldout仍呈16场景过拟合。四train最终仅3个实例ID、四heldout4个；原Unified对应6/6。

与原Unified短测相比，重建train仅+0.171dB、heldout+0.163dB，不能称为解决重建问题；train Mask IoU下降0.1235，heldout下降0.0204。硬件/rank划分不同，但共用训练预算、数据和共同初始化；小样本不支持完整训练上限结论。

## 典型失败定位

Train scene0570_00：PSNR23.688、正匹配Query分类88.24%，最佳Mask IoU0，最终无实例。保存的预测显示47/47个Query通过类别/no-object/0.05分数筛选，但context_mask_nms最终保留0。由于该NMS对首个有效候选必定保留，结合实现可确定：这些类别候选都没有在两context视图中形成同时满足membership>=0.5且alpha>0.05的非空区域；不能误报为全部被类别/no-object淘汰，也不是纯NMS重复抑制把所有候选删掉。保存的紧凑预测没有完整region_mass/alpha，不能进一步断言是membership偏低还是alpha/可见性限制。

Train scene0279_02：类别筛选55→Mask NMS13→最终1实例，最佳IoU0.0817；scene0051_00：48→8→1，IoU0.2621。Heldout scene0030_01：54→4→0，IoU0，说明也存在后续像素竞争/面积过滤损失，但前置Mask质量已弱，不能只改最终组装。

## 当前判断

串行实现通过可训练性验证，新增模块更新，但当前配方没有在相同短预算下取得收益。不能因此宣布“串行思想不可行”；原Unified短测也弱、全量后曾改善，256步不能预测完整上限。

建议暂不直接转全量，先固定权重检查Mask logits→Gaussian membership→渲染region_mass/alpha→阈值前后召回，并对照进入理解Decoder前后Query的可分性，定位为何分类已学到却形成不了完整Mask。现有证据不支持盲目再加Decoder层或只改NMS。此项仅建议，本次未提交新的GPU诊断/结构实验，未修改A/B。

run根：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt07_serial_query_isolated_short_node18_4gpu`。停止轮询。
