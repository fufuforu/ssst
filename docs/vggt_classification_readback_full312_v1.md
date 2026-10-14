# A/C完整312场景验证

2026-10-14用户授权将已完成的分类末端对照扩到完整312场景，完成后再分析下一步。本阶段只评测已有A/C权重，没有训练或改变模型/loss。沿用SIU3R固定val_pair.json（SHA59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b）、312场景/1860窗口/11160图像，官方commit8ea80166be76854f938e90521f1a5b688b755c87，以及统一13列表，两处mIoUt空置。

A是adapter0.01/feedback0.05完整适配2＋冻结联合4的终点：checkpoint SHA ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0。A的完整1860窗口PNG/官方状态/重建评分已存在，复用并核验原生全局合并重现，不再执行A模型。

C是上述A加300步对象专属分类读回的最终补丁，SHA ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9，训练执行SHA8595edab9f5f627f7301ac33bc838d9a8f111287。核心/mask冻结，原双向结构与feedback保持，只改变分类输入。同128已完成PNG和官方状态均复用，仅新增1732窗口推理/官方评分。每rank32已有窗口＋433新增窗口，共465；1860窗口与manifest逐条对齐，不丢失、重复或新增采样。

训练、模型、原export/diagnose的文件哈希与已完成数值执行快照比对一致，旧C四分片合并重现原128成绩。新的调度/范围/合并逻辑先通过CPU身份/覆盖检查，再四卡3090真实预检：每rank重新推理一个旧C窗口及一个新窗口，共8窗口。旧C类别/实例PNG、诊断和RGB/depth/GT重现；置信度有限、FP32误差不超过1e−7且同类别实例排序一致。新窗口RGB/depth/GT逐帧重现A。实际GPU官方评分/分片合并对未修改CPU官方评测器误差小于1e−7。预检不计新训练曝光。

预检与CPU参考完成后自动启动四卡正式C评测。每个新窗口继续检查冻结重建与GT身份，复用A已完成的逐帧重建评分。旧C128窗口原始产物先复制，保护历史证据；官方IoU/PQ/AP状态与1732新窗口状态统一全局合并，AP按canonical图像顺序合并，不平均分片或窗口成绩。最终重建五项对A误差必须小于1e−12。

自动CPU汇总保存一张A/C原生13列表、逐类IoU/PQ/AP变化及GT质量排序诊断；GT排序不能计入模型性能。另保存C完整范围原生Hungarian/no-object诊断以及按类窗口内GT计数（对象跨窗口重复）。A没有完整范围Hungarian诊断，本次不为该诊断额外跑1860窗口A；A/C原生匹配对比仍仅使用原同128证据，不能把C1860总数与A128总数直接比较。

本次验证的重点是C在同128的小收益能否保持，尤其窗帘/浴帘集中增益、其他类别的退化及mAP/PQ/mIoU取舍。保持单一C训练种子，本阶段不是多种子复现。评测成功后再据实际结果决定后续，不自动启动质量头/联合训练或新的架构对照。

证据root：`/space/mawb/ssst/group_plus/vggt_classification_readback_full312_v1`。
提交脚本创建不可变已推送代码快照，保存Slurm提交收据：CPU reference、四卡 preflight、四卡 evaluate、CPU comparison。GPU申请13号四张3090、192G；无需等待八卡。真实预检通过且正式四rank开始正常导出后停止主动轮询。

只读查看进度及阶段内剩余时间估计：

```bash
python3 /space/mawb/ssst_vggt_overfit_diagnostics_v1/scripts/show_vggt_classification_full312_progress_v1.py
```

最终产物为`attempts/attempt01/analysis/siu3r_comparison.csv`、`.md`及`results.json`；C原生`metrics.json`在`object_support`，A复用核验报告在`reference`。

## 首次真实预检的工程恢复

64350完成A/旧C原生状态核验（2:11）；64351预检在6:19失败，原因仅为旧窗口pred.json中少数置信度有一个FP32 ULP（最大5.960464477539063e−8），触发过严的JSON字节比较。四个已知旧窗口全部分割PNG、对象ID、类别、同类别实例分数排序一致，原生匹配诊断也重现；RGB/depth/GT检查通过。依据实际输出，将pred.json score检查改为有限且绝对误差不超过1e−7，仍严格核验所有PNG字节、其他JSON字段和类别内排序；模型、loss、参数及推理kernel保持。保留失败快照/日志及ENGINEERING_FAILURE_REVIEW.json。

依赖正式64352及汇总64353自动取消，零正式推理、零训练更新。修复代码经实际已生成四旧窗口CPU合同验证，独立attempt02重做八窗口真实预检并续接正式评测；复制复用成功64350参考，避免重复A评分。进度脚本自动选择最新提交attempt。最终表格在成功attempt的analysis，原attempt01不覆盖。
