# 完整数据feedback0.05：2026-10-14结果复核

训练64041、官方评测64042均COMPLETED0:0，分别用时8小时49分23秒、31分27秒。训练在13号四张3090，于2026-10-14 06:04:59结束；评测在11号四张3090，06:36:26结束（Slurm本地时间）。本轮只核查已完成作业与CPU分析，没有提交新训练/GPU推理/评分，当前用户队列为空。

结论：adapter理解梯度系数0.01保留；feedback上限0.1降至0.05在完整训练后带来小幅重建改善，理解指标有升有降。不能将500步对照优势直接外推为完整训练的同等优势，也不能把0.05称为整体最佳。

## 同起点、同预算的正式13列

两组共用重建适配epoch2边界SHA a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c，fresh联合训练4 epoch，4172次新优化器调用/33376新增窗口曝光；加上复用适配阶段累计6258/50064。并非从旧联合epoch6继续追加4 epoch。

两组同13号四卡3090、global batch8、micro1/accum2、适配VGGT资产、四rank原RNG及采样/LR/理解warmup。VGGT全冻结；原MASt3R/ViT adapter/pixel decoder联合训练范围、GS/query decoder、6/8/10/12双向交互、任务头/loss/GC/clip1及monitor_v1几何契约保留。唯一研究改变为原capped feedback注入乘0.5，实际峰值0.05，保留1000新增曝光warmup；adapter/GS理解梯度系数均为0.01，不等于全局理解loss乘0.01。

官方Evaluator固定8ea80166be76854f938e90521f1a5b688b755c87；312场景、1860六帧窗口、11160图像。理解指标0–1，两处mIoUt为空。沿用官方输入/新视图列，不另拆三套口径；官方新视图集合含2输入帧和4额外帧。

| Experiment | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUₛ↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUₜ↑ | 新视图 mIoUₛ↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUₜ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adapter0.01 / feedback0.1 | 0.08831 | 0.2265 | 22.40 | 0.7807 | 0.3652 | 0.5740 | 0.2619 | 0.6430 |  | 0.5684 | 0.2449 | 0.6241 |  |
| Adapter0.01 / feedback0.05 | 0.08684 | 0.2237 | 22.42 | 0.7818 | 0.3636 | 0.5719 | 0.2614 | 0.6407 |  | 0.5676 | 0.2462 | 0.6231 |  |

PSNR差+0.02003464 dB，312场景配对cluster bootstrap（10000次、seed42、保持官方窗口/图像加权）的95%区间[+0.00767337,+0.03218580] dB。区间不含0，但实际增益很小，且只描述验证数据不确定性，不包括训练seed不确定性。

SSIM+0.00115051，LPIPS−0.00164313，AbsRel−0.00146657（−1.66%），RMSE−0.00285252（−1.26%）。五项重建区间均支持改善。深度指标使用官方scale/shift对齐，不能由此推断原始VGGT相机更准：两组复用同一冻结几何资产。

输入mIoUs/mAP/PQ分别−0.2042/−0.0539/−0.2275个百分点；新视图分别−0.0867/+0.1329/−0.1016个百分点。未对全局AP/PQ/IoU做场景平均或宣称统计显著性。五个理解指标略降、一个略升，没有统一提升。

此前adapter1→0.01完整配对PSNR增益+3.02667 dB；本次feedback收益远小于它。此前19 dB的大幅改善主要来自adapter梯度路由调整，不能继续把理解偏弱主要归因于feedback强度。之前短训练C在独立248场景的优势不能当成4172步正式终点的提升幅度。

## 完整性与证据

两组1860个manifest index、名称与六帧frame IDs逐条一致。四rank导出、评分、checkpoint metadata与CUDA合并合同通过；逐帧重建分数重聚合与官方五指标一致。

本轮用原生Evaluator在CPU合并两组各四份official_states.pt，全局输入/新视图mIoU、PQ、mAP及AP50/AP75复现至1e−12，未平均分片AP；两组context/target AP groundtruth mask、label等保存状态逐条相同。

两组rank0各209条日志（每20次调用及末次）的采样、时钟、LR和warmup逐条一致，loss/preclip norm有限；新实际feedback beta逐条为旧名义beta一半。这不是每步、所有rank的loss/梯度检查。四rank边界RNG及fresh优化器都有初始化收据；启动冻结探针通过之前startup审计。

两组几何质量各1829 OK、31 WARNING，全部1860 fit有效且全部保留评分。保留历史失败与旧报告；本轮未改训练、模型或评测数学。

执行/推理/评分SHA990a5d4fbd497496c5a73748469254540781e095；新checkpoint SHA ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0；旧checkpoint SHA adb080ad6888be42c34a4dd62f5559314addb7b1d8a475240dbae6f927e26ff3。

## 理解诊断的解释及下一步判断

原生Hungarian在全部100query上使用类别、BCE与Dice联合cost，不先删除no-object预测。匹配query的分类监督是GT对象类别，thing mask loss也只作用于匹配query。此前随机128场景C短训练权重诊断：492个GT匹配query里321个hard mask IoU>=0.5，其中44个仍argmax no-object。这是验证时执行原生matcher的观察，不是断言44个GT曾参加训练，或新完整权重仍有同样比例。

推理没有GT，不会因query被过滤而指定GT改由另一个query负责。最终mAP用保留下来的同类mask重新按分数/IoU阈值匹配GT；有其他合格预测可匹配，否则GT为漏检。no-object筛选发生在像素竞争前，其他合格query/stuff可争取这些像素；稍后的面积过滤则留下void、不重新分配给第二名。PQ有原生预测void覆盖忽略规则，不能把所有被过滤对象一概算成PQ的FN。

置信度排序损失指分数与实例质量不一致：差mask高分、好mask低分使AP的precision-recall曲线下降。当前分数是类别概率乘获胜像素平均membership，不直接预测mask IoU。之前独立248场景C短训练输出固定mask/label，仅将分数换为同类别GT IoU，mAP由0.236063升至0.322882；这是不可部署的GT依赖离线诊断，不是模型成绩或可达上限，也不是本次完整权重的结果。

优先方向仍是query对象存在性/类别、mask支撑与实例质量的一致性；稳定性任务另将理解loss拆为分类、thing BCE/Dice、stuff BCE/Dice，定位主要梯度来源。候选结构修改为query读回自身mask/几何支撑后再用原分类头，保留encoder、原双向结构和loss；其效果尚未验证，不重复已无增益的旧三尺度融合。

应先在完整权重确认上述失配是否仍存在，再决定短训练对照。暂不继续扩大feedback超参搜索，不自动叠加视觉冻结、AA解冻或新loss。本轮未启动后续实验，等待用户根据完整结果决定。

- [13列完整精度CSV](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/evaluation/final_adapt2_joint4_official/review_final_20261014/siu3r_table1.csv)
- [差值、区间及审计](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/evaluation/final_adapt2_joint4_official/review_final_20261014/audit_and_comparison.json)
- [原始官方指标](/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/evaluation/final_adapt2_joint4_official/metrics.json)
- [之前理解瓶颈诊断](vggt_understanding_bottleneck_findings_20261014.md)

CPU复现脚本scripts/analyze_vggt_feedback_full_v1.py：--old/--new指定两组official目录，--old-training/--new-training指定两组workspace训练root，--output为新空目录。--native-state-check使用现有官方Python环境，仅重用保存状态，不执行模型前向。
