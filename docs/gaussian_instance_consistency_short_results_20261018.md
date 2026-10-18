# 67292 Gaussian 实例一致性短训结果

用户通知完成后检查，2026-10-18。结论：工程与训练通过，新约束有真实监督和梯度，但当前配方未带来整体原生 Mask 收益，不建议直接完整数据训练。没有重训、没有新提交、没有修改执行快照。

## 完成与可比性

- Slurm：67292，COMPLETED，退出码0:0，耗时36分00秒，3dimage-13，4×3090、16CPU、128G。smoke、训练和汇总在同一作业顺序执行。
- 运行目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt15_gaussian_instance_consistency_short_node13_4gpu`。
- 固定执行SHA：`0882660021e17e51eae9471b5921d5a6db951b90`。
- 重新在CPU读取checkpoint核实：零步空Adam；与67050共同305个参数张量逐值一致；零步参数SHA与smoke零步相同，未继承smoke两次更新。最终305个参数张量均有限、均发生变化，305个Adam参数状态的step全部为256。保存四rank RNG状态。
- 四rank日志各256行，update连续1–256，数值有限。smoke两次更新的各rank参数SHA相同，正式短训首步四rank参数SHA相同；没有额外声称最终四rank参数SHA已逐一核实。
- 重新运行只读`validate_direct_smoke`验证：原16窗口覆盖、实际正负样本、独立新loss梯度、显式几何隔离、GT分配detach、8维重算与原生读出逐值一致通过；旧渲染项几何/共享/实例头梯度检查通过。
- 与基线的训练窗口、数据清单hash、heldout hash、Omega/SAM身份、world size和global batch在初始checkpoint元数据中相同；探针场景、context帧、支持GT数量一致。零步PSNR一致到约1e-6 dB；原始IoU有微小跨运行数值差异，聚合heldout约0.0022个百分点，不宣称全部输出逐位确定。

## 原生任务结果

原始Mask IoU按支持GT数量加权（各scope共20个支持GT），PSNR按四场景平均。仅4train+4heldout探针；IoU使用来源匹配支持的GT与阈值化原生Query–Gaussian Mask，不是官方mAP/PQ。

| 终点版本 | train PSNR | train Mask IoU | heldout PSNR | heldout Mask IoU |
|---|---:|---:|---:|---:|
| 基线67050 | 19.7474 | 27.3960% | 10.0212 | 8.4553% |
| 新约束67292 | 19.6097 | 27.0930% | 9.9940 | 5.5025% |
| 新−基线 | −0.1377 dB | −0.3031个百分点 | −0.0272 dB | −2.9528个百分点 |

逐场景终点原生Mask IoU：

| scope | scene | 支持GT | 基线 | 新约束 | 差值/百分点 |
|---|---|---:|---:|---:|---:|
| train | scene0279_02 | 10 | 8.0414% | 7.5701% | −0.4713 |
| train | scene0570_00 | 5 | 11.4880% | 11.8340% | +0.3460 |
| train | scene0051_00 | 3 | 82.2414% | 79.6191% | −2.6224 |
| train | scene0454_00 | 2 | 81.6710% | 84.0657% | +2.3947 |
| heldout | scene0011_00 | 7 | 9.8158% | 10.2646% | +0.4488 |
| heldout | scene0019_01 | 1 | 34.2374% | 35.6050% | +1.3676 |
| heldout | scene0030_00 | 6 | 0.0013% | 0.0156% | +0.0143 |
| heldout | scene0030_01 | 6 | 11.0252% | 0.4167% | −10.6085 |

heldout下降主要由scene0030_01贡献，另外三场景略有上升，不能称全面退化。scene0030_00两组均接近零。scene0030_01的PSNR反而增加约0.020 dB，不能仅以整体重建崩溃解释其Mask下降；当前证据也不能确认具体失效机制。最终组装实例数量train为7→7、heldout为6→5，数量不是质量指标。

## 新项活动与尚未解决的问题

- 1024条rank-update日志中，1021条有跨父正样本、856条有局部异GT负边、1020条新项loss大于零。这里的单位是rank-update，不能写成1024次全局优化更新；每行还汇总了两次累积微批。
- 末20步日志均值：可靠Gaussian比例约26.29%，pull距离0.2089（margin0.1），局部负距离0.1471（margin0.5），pull loss0.1243、push loss0.3028。日志负margin违反率聚合均值约0.8054；这是包括无负样本微批零值的平衡聚合，不是全部负边的原始比例。
- 在有负边的7个终点探针中，负margin违反率为90.05%–99.77%；唯一无负边的是heldout scene0019_01，应跳过其负分离判定。局部分离目标仍明显未满足。
- 新项有效系数20步ramp后为0.0005，末20步加权loss均值约0.0002135。非零梯度证明通路活动，不能据loss数值小就判定权重不足，也不能据此断言约束足够强；正式训练尚无各任务梯度相对量/冲突的量化结果。
- 末段pull日志均值有所回落，push仍明显非零；这不足以认定已经收敛或只需延长训练。可靠归属、正原型、几何近邻会随训练变化，初期许多窗口无负边，因此早晚距离/loss不能直接当作固定样本学习曲线。
- 当前`direct_positive_distance`是Gaussian对其他父Token原型的距离，与67235中同GT跨父Gaussian成对距离0.5285不是同一统计量，不能宣称0.5285→0.2089就是表征提升。
- 基线已有探针缓存仅含RGB、语义/组装实例、类别及eligible，不含Gaussian8维、几何、父Token和贡献归属；基线原探针JSON也没有direct指标。因此本轮没有固定口径的基线/新组表征对照，不能声称表征已改善或已证明塌缩。

## 决策与下一步建议

当前属于“工程门槛通过、任务收益未通过、目标表征改善尚未证实”。小样本/训练不足仍是可能因素，但没有证据支持把它们作为默认解释，更没有依据直接扩大到全量。

下一步最有信息量的是固定权重对照诊断：复用67050与67292终点，同场景/帧、同一诊断口径，比较8维跨父成对距离、局部同异GT区分度、有效覆盖，并重点检查scene0030_01的原生Mask响应与支持匹配。如果表征改善而Mask退化，再查任务读出传递；如果表征也未改善，再查损失权重、任务梯度竞争及目标/代理标签适配。需要的新GPU诊断或延长训练尚未提交，须另行授权；不能以旧67235统计直接代替本组对照。

证据：运行目录`SMOKE_PASSED.json`、`FIRST_UPDATE_VERIFIED.json`、初末checkpoint、四rank日志、`PROBES_INITIAL.json`/`PROBES_FINAL.json`、`SHORT_VALIDATION_SUMMARY.json`、`INSTANCE_SHORT_AUDIT.json`、`GAUSSIAN_SHORT_AUDIT.json`、`TRAINING_COMPLETE.json`、`COMPLETE.json`，以及67050对应既有产物。
