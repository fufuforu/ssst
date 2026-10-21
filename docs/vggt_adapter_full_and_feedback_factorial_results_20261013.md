# 完整数据adapter重训与feedback四组对照：2026-10-13复核

完整数据训练和三组新短训练都完成。四组短评测与汇总正常完成；完整数据63553的四个分片导出、官方评分已成功，但最后汇总因只允许旧recipe名称失败。本轮只修复严格recipe白名单，零GPU的CPU作业64030（1分38秒）复用原评分，成功生成完整官方结果。原失败日志、提交和不可变077执行快照保留。没有重复训练、渲染或评分，没有提交下一阶段训练。

**结论：adapter理解梯度系数0.01已得到完整验证集支持，11个已实现指标全部改善。弱feedback从训练开始使用时，多数指标改善；冻结视觉主要保护理解；两项叠加未形成更好的统一方案。当前还不能宣称所有指标追上SIU3R，或选出了整个结构的上限。**

## 完整数据：adapter系数1→0.01

两组共用同一重建适配epoch2边界，重新执行冻结联合4 epoch（4172次新调用、33376新增窗口曝光）。复用的适配阶段为2086次调用、16688曝光，最终总计6258调用/50064曝光。4卡3090/global batch8；VGGT整体冻结；MASt3R/ViT adapter/pixel decoder保持原联合训练范围；GS、query decoder、任务头、loss和双向交互保持原设置，feedback上限0.1。

改变的是memory adapter对理解梯度的系数：gR+gU→gR+0.01gU；没有把全局理解loss权重乘0.01，也没有把forward特征缩小。GS梯度系数本来就是0.01。四rank边界RNG恢复、fresh联合优化器通过已有启动审计；两组209条已保存rank0日志的窗口采样、阶段时钟、学习率和warmup逐条一致。

正式验证：312场景、1860六帧窗口、11160图像；固定官方Evaluator commit8ea80166。完整13列如下，理解指标0–1，两列文本mIoUt为空。

| Experiment | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUₛ↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUₜ↑ | 新视图 mIoUₛ↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUₜ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原adapter理解梯度系数1 | 0.16726 | 0.3725 | 19.37 | 0.6744 | 0.5536 | 0.5634 | 0.2503 | 0.6262 |  | 0.5586 | 0.2334 | 0.6071 |  |
| 新adapter理解梯度系数0.01 | 0.08831 | 0.2265 | 22.40 | 0.7807 | 0.3652 | 0.5740 | 0.2619 | 0.6430 |  | 0.5684 | 0.2449 | 0.6241 |  |

PSNR提高3.02667dB，配对312场景cluster bootstrap的95%区间为[2.94374,3.10921]dB；SSIM提高0.10621，LPIPS下降0.18838。AbsRel下降47.20%，RMSE下降39.19%。输入/新视图mIoU提高1.06/0.98个百分点、mAP提高1.16/1.15个百分点、PQ提高1.68/1.69个百分点。这是明显改善，理解增幅相对重建增幅仍小。

因此此前19dB可以**很大程度归因于adapter共享表示受到过强理解梯度影响**；不能说所有剩余差距都由它解释。相对论文PSNR25.96仍低约3.56dB，理解指标也不是已解决。历史LocusGS的24.4536取额外新帧，其同官方全视图口径值为24.9172；不能把两种口径的值直接作差。旧LocusGS还有GT几何条件，与当前预测几何不构成单独encoder替换对照。

深度使用官方逐图scale/shift对齐；深度指标改善不能被解释成原始VGGT相机已变准。适配VGGT资产相同，几何校准均1860次有限有效fit，旧/新各1829 OK、31 WARNING；monitor_v1下所有告警窗口保留评分。

## 最新四组：完整池采样的500步诊断

共同起点是上述重建适配epoch2边界，不是先在新adapter完整联合epoch4上继续微调。所有组adapter/GS理解梯度系数0.01；VGGT全冻结；原anchor读取层6/8/10/12、点积feedback、loss、GC、AdamW/clip1、200步LR warmup及4172步cosine截断到500都相同。500次新调用/4000唯一窗口来自1191场景完整池，实际覆盖1177场景，四rank流逐条相同。以下同一64留出场景/64六帧窗口官方成绩，**不能与上面的312场景成绩直接作差**。

A为原联合训练；B只冻结MASt3R encoder/ViT adapter/pixel decoder；C只将feedback上限0.1降为0.05；D同时做B/C两项。B/D仍训练memory adapter、GS、query decoder和panoptic，保留重建及理解联合优化。A/B/C训练4090，D按用户要求迁到3090，全部正式评测4090；D的结构归因保留训练硬件差异。

| Experiment | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUₛ↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUₜ↑ | 新视图 mIoUₛ↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUₜ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 共同起点/0 | 0.10747 | 0.2655 | 21.82 | 0.7688 | 0.3945 | 0.6593 | 0.3615 | 0.6959 |  | 0.6546 | 0.3435 | 0.6781 |  |
| A 原视觉训练/β0.1 | 0.10622 | 0.2618 | 21.50 | 0.7604 | 0.3974 | 0.6261 | 0.3478 | 0.6900 |  | 0.6223 | 0.3264 | 0.6687 |  |
| B 冻结视觉/β0.1 | 0.10517 | 0.2577 | 21.47 | 0.7638 | 0.3960 | 0.6579 | 0.3708 | 0.7009 |  | 0.6539 | 0.3451 | 0.6792 |  |
| C 原视觉训练/β0.05 | 0.10300 | 0.2544 | 21.72 | 0.7690 | 0.3902 | 0.6615 | 0.3709 | 0.6933 |  | 0.6603 | 0.3479 | 0.6642 |  |
| D 冻结视觉/β0.05 | 0.10510 | 0.2608 | 21.63 | 0.7656 | 0.3912 | 0.6506 | 0.3581 | 0.6952 |  | 0.6455 | 0.3367 | 0.6694 |  |

- **B保护理解，基本不改善PSNR。** 相对A，输入/新视图mIoU提高3.19/3.16个百分点，mAP提高2.29/1.87，PQ提高1.09/1.05。PSNR差−0.02273dB，64场景配对95%区间[−0.12644,+0.08313]，没有支持清晰RGB增益。其理解指标主要恢复到或略超过共同起点，不能把相对下降的A的优势都称为超越起点。
- **C是当前重建与语义/mAP较好的候选，仍有PQ代价。** 相对A，5个重建指标全部改善；PSNR+0.22036dB，区间[+0.13633,+0.30206]，64场景中50个场景提高。输入/新视图mIoU+3.55/+3.80个百分点，mAP+2.30/+2.14；输入PQ+0.33、新视图PQ−0.45。它不能称为所有指标最优，也不能把500步结果当作完整收敛上限。
- **D没有显示两项收益相加。** 500步的mIoU/mAP均不及B/C；300步mIoU/PQ高于它自己的500步，说明理解训练轨迹有波动。D训练硬件不同，不能把负交互全部归因于冻结与小feedback组合。

固定权重beta扫描与重新训练是两个问题。先前J500固定权重临时将beta0.1改0.05会损害多数指标；本轮C从相同起点以0.05训练，优于以0.1训练的A。结果支持训练与推理写回幅度一致，而不是训练后直接关掉feedback。

## 新定位线索：梯度尖峰与统一裁剪

日志验证原联合A第441步preclip norm799.335，同一批数据B/C/D为2.562/4.317/3.905，损失本身均有限且相近。clip1使A该步全部梯度近似乘0.00125；这涉及梯度对Adam矩估计的输入，不等于Adam最终参数更新也按相同比例缩小。不能直接由梯度范数推断最终参数步长。

A最后100步范数均值11.91、最大799.34；B为3.85/90.41，C为3.73/29.49，D为2.62/6.59。四组约98%–99%步骤均触发clip，所以“触发次数”本身不能定位问题。弱feedback/视觉冻结降低了尾部尖峰，但尖峰发生步缺少完整分模块R/U诊断，目前不能判定来自MASt3R、query-to-anchor路径、mask投影还是几项共同作用。

较晚日志中各交互层写回特征norm比值几乎总贴上限：A/B约2.5%，C/D约1.25%。说明实际幅度缩小已生效，没有误将日志原名义beta当成实际系数。只有少量rank/microbatch梯度探针，不能把其中cosine称为全局梯度冲突；现有探针也不支持“adapter仍被理解梯度整体压倒”作为唯一解释。

## 下一步建议

保留完整验证已支持的adapter系数0.01和现有双向结构。C的较小写回是优先候选，B是理解/PQ保护的备选；不直接叠加成D，也不直接关闭写回。

先用A/B/C已有500步权重在未参与beta选择的更多留出场景复评，确认短集收益是否延续。再做小预算的联合路径诊断：记录共享decoder、MASt3R/视觉adapter、query/panoptic等模块裁剪前后的全局R/U范数及实际参数更新，定位第441步一类尖峰。若确认为某一路径主导，再对该路径的写回归一化、上限或梯度路由做单变量修改；不能直接删除理解loss，也不凭此次结果更换encoder。此处只是建议，本轮未新提交训练或GPU评测。

## 完整性、产物与复现

四组四rank各500条日志、完整冻结状态、相同4000窗口流、0/100/300/500共16套checkpoint→export→score→metrics SHA收据通过。逐帧重建文件重新聚合与官方五指标吻合；语义/AP/PQ直接取原官方结果，全局AP没有按scene/shard平均。完整数据两版1860窗口与frame IDs逐条相同，四分片官方GPU/CPU真实两窗口合并合同通过，所有已实现指标有限。

官方表格“新视图”沿用Evaluator target集合（原val_pair包含2输入帧和4额外帧），未另拆context/target-all/true-novel三套表；两处mIoUt表示文本分支未训练，不是0。bootstrap只针对重建，是验证数据的不确定性，不包括训练seed不确定性；64场景已用于beta选择，因此需更广留出确认。

- [完整数据13列CSV](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1/evaluation/final_adapt2_joint4_official/review_final_20261013/siu3r_table1.csv) · [逐项差值及配对审计](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1/evaluation/final_adapt2_joint4_official/review_final_20261013/audit_and_comparison.json)
- [CPU恢复收据](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1/evaluation/final_adapt2_joint4_official/recovery/recovery_receipt.json) · [最终完成](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1/evaluation/final_adapt2_joint4_official/COMPLETE.json) · [原始官方指标](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_v1/evaluation/final_adapt2_joint4_official/metrics.json)
- [四组13列CSV](/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01/evaluation/review_final_20261013/siu3r_table1.csv) · [完整训练/评分审计](/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01/evaluation/review_final_20261013/audit.json)
- [四组差值、bootstrap与逐类变化](/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01/evaluation/review_final_20261013/comparisons.json) · [11指标曲线](/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01/evaluation/review_final_20261013/metric_curves.png)
- 本次只新增可复现CPU分析脚本`analyze_vggt_adapter_full_v1.py`和`analyze_vggt_feedback_factorial_v1.py`，修复`score_object_locus_posefree_official.py`的recipe白名单；未修改训练数学。

训练与推理代码077bd9a4064dfc6b2e54ea6b2a2256ba5437b1f0；CPU恢复汇总代码f2d3ca8b02089720d9bd20ddc68e88fe1a6bf327。新完整checkpoint SHA adb080ad6888be42c34a4dd62f5559314addb7b1d8a475240dbae6f927e26ff3。63553仍为FAILED1:0；64030为COMPLETED0:0，明确区分历史失败与已恢复结果。三组新短训练63839/63841/63902、评测63840/63842/63903、汇总63845均COMPLETED0:0；20步状态审计63927也完成。当前用户Slurm队列为空。
