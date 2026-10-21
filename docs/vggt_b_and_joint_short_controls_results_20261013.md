# B 分支与三组共享短对照：2026-10-13 完成检查

三组共享短对照已完成训练和全部11项官方指标。它们没有在16场景留出集上恢复此前四训练窗口的24.5 PSNR。AA解冻有部分理解指标收益，但无明确综合优势；GS从零初始化在同短预算内全面落后。B的完整评测也已补齐：解冻续训相对A有小幅作用，但相对共同epoch6只有两项语义mIoU略高、其余9项数值更差。当前没有证据支持按原配方继续延长训练。

## 完成状态与评测恢复

| 阶段 | Slurm ID | 状态 | 用时 |
|---|---:|---|---|
| B：AA解冻续训2轮 | 62845 | COMPLETED / 0:0 | 17:00:39 |
| B：旧自动评测 | 62847 | FAILED / 1:0 | 01:59 |
| 小对照：AA冻结、沿用GS | 63326 | COMPLETED / 0:0 | 54:39 |
| 小对照：AA解冻、沿用GS | 63327 | COMPLETED / 0:0 | 1:39:19 |
| 小对照：AA冻结、从零GS | 63328 | COMPLETED / 0:0 | 57:40 |
| 小对照共同评测 | 63329 | COMPLETED / 0:0 | 05:19 |
| B：四卡恢复评测 | 63544 | COMPLETED / 0:0 | 23:27 |

B训练完整完成1044次续训更新／16688次新增曝光，累计7302更新／66752曝光、总epoch8。旧评测在加载B派生VGGT资产的SHA校验处报`NameError: sha256`，没有产生完整指标。修复仅导入原有校验函数；CPU回归覆盖B正确资产恢复、错误SHA及错误来源拒绝，A的原端点校验仍通过。修复已提交推送`2c6ad01dc85324b63fb50f82ac11c4c5700aa30b`，新不可变快照在node17四张4090上评测相同最终checkpoint，不重训、不改变指标协议。旧失败日志与导出root保留，恢复使用新目录。

进一步逐张量核实B派生VGGT：864个AA张量确有更新，最大绝对变化0.0004120823；其余477个张量在原生dtype转换后与共同适配源严格相同。不是解冻未执行，也没有额外解冻DINO或预测头。

## 完整验证集的13列

三行均为同一完整1860窗口／312场景、未改变的SIU3R官方协议。下方16场景对照另表注明范围，不能替代完整结果。

| 对照 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUₛ↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUₜ↑ | 新视图 mIoUₛ↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUₜ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 共同起点：适配2＋冻结联合4 | 0.16726 | 0.3725 | 19.37 | 0.6744 | 0.5536 | 0.5634 | 0.2503 | 0.6262 |  | 0.5586 | 0.2334 | 0.6071 |  |
| A：再冻结联合2轮 | 0.17569 | 0.3873 | 19.14 | 0.6660 | 0.5778 | 0.5621 | 0.2443 | 0.6237 |  | 0.5579 | 0.2298 | 0.6068 |  |
| B：再解冻AA联合2轮 | 0.17546 | 0.3872 | 19.35 | 0.6701 | 0.5675 | 0.5647 | 0.2463 | 0.6231 |  | 0.5611 | 0.2311 | 0.6041 |  |

B相对A：PSNR高0.2075dB、SSIM高0.0040、LPIPS低0.01037；五个重建指标均略好。输入／新视图mIoUs和mAP略高，但两项PQ更低。AA并非完全没有作用，但它主要缓解了冻结续训的退步。

B相对共同epoch6：PSNR19.3487对19.3692，仅低0.0205dB、基本相当；AbsRel、RMSE、SSIM、LPIPS均更差。输入／新视图mIoUs仅高0.1266／0.2497个百分点，mAP和PQ均更差。没有出现重建与理解全面改善，不能用两项mIoU的轻微增长宣称B更优，也不能仅按9项数值下降断言AA毫无价值。A相对epoch6的11项均更差。

B与A的完整对照使用adapter理解梯度系数1；小对照使用0.01。因此原B不是“调整adapter路由之后，AA是否有作用”的替代控制。A／B同为八张3090、global batch16和fresh optimizer，B相对A隔离AA解冻；与epoch6比较同时改变了batch和优化器历史，不把差异全部归为训练时长。

完成审计核实四rank评分完成、四个原生两窗口合并契约PASS，1860窗口名称与312场景覆盖完整，checkpoint7302／66752和续训1044／16688时钟正确，评分零更新。训练代码e61424a，推理及评分代码2c6ad01，官方Evaluator固定8ea80166；Slurm63544 COMPLETED/0:0。

## 16场景留出集的13列

每组同epoch6起点、global batch16、100更新／1600曝光，16个训练窗口各重复100次；另16个场景用于评测，训练和留出场景不重叠。理解／panoptic参与训练，完整理解损失权重1、反馈beta0.1；三组adapter理解梯度系数均为0.01。GS/AA理解梯度系数0.01。只有AA冻结状态或GS初始化不同。

| 对照／更新数 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUₛ↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUₜ↑ | 新视图 mIoUₛ↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUₜ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AA冻结＋旧GS／0 | 0.15000 | 0.3334 | 19.53 | 0.6765 | 0.5613 | 0.4335 | 0.2590 | 0.4304 |  | 0.4220 | 0.2235 | 0.4047 |  |
| AA冻结＋旧GS／50 | 0.14576 | 0.3349 | 19.31 | 0.6802 | 0.5547 | 0.4023 | 0.2551 | 0.4451 |  | 0.3943 | 0.2244 | 0.4133 |  |
| AA冻结＋旧GS／100 | 0.14567 | 0.3337 | 19.28 | 0.6780 | 0.5488 | 0.3845 | 0.2250 | 0.4064 |  | 0.3727 | 0.2107 | 0.4034 |  |
| AA解冻＋旧GS／0 | 0.15000 | 0.3334 | 19.53 | 0.6765 | 0.5613 | 0.4335 | 0.2590 | 0.4304 |  | 0.4220 | 0.2235 | 0.4047 |  |
| AA解冻＋旧GS／50 | 0.14447 | 0.3311 | 19.32 | 0.6796 | 0.5558 | 0.4013 | 0.2382 | 0.4296 |  | 0.3927 | 0.2222 | 0.3866 |  |
| AA解冻＋旧GS／100 | 0.14551 | 0.3343 | 19.22 | 0.6819 | 0.5525 | 0.4041 | 0.2258 | 0.4129 |  | 0.3960 | 0.2131 | 0.3802 |  |
| AA冻结＋新GS／0 | 0.16317 | 0.3694 | 10.55 | 0.4982 | 0.7490 | 0.0916 | 0.0141 | 0.2083 |  | 0.0793 | 0.0087 | 0.1940 |  |
| AA冻结＋新GS／50 | 0.17249 | 0.3944 | 10.65 | 0.4999 | 0.7565 | 0.2437 | 0.0506 | 0.2120 |  | 0.2358 | 0.0482 | 0.1966 |  |
| AA冻结＋新GS／100 | 0.18148 | 0.4114 | 10.75 | 0.4983 | 0.7581 | 0.1138 | 0.0078 | 0.2917 |  | 0.1070 | 0.0062 | 0.2480 |  |

两个mIoUt栏留空；各条件使用未改动的固定SIU3R Evaluator，在16场景上计算全局mAP/PQ，不平均单窗口mAP/PQ。官方新视图组保留官方target集合定义，不另改仅额外4帧的范围。源模型在此小集合PSNR19.53，与完整验证集19.37来自不同范围，不能交叉作差。

## AA与初始化问题的答案

1. AA解冻相对冻结的100步终点：PSNR低0.05744dB，SSIM高0.00398，LPIPS高0.00363（更差），AbsRel略好、RMSE略差。输入mIoUs高1.96个百分点、新视图mIoUs高2.33个百分点；输入PQ略好，新视图PQ低2.31个百分点。7/11项数值改善、4项变差，不用指标计数代替任务价值判断。PSNR仅7/16场景提升；场景配对描述性bootstrap95%区间为[-0.1815,0.0779]dB，跨0，单种子／16场景不足以宣称优越性。
2. 两个沿用GS组相对共同起点：PSNR分别低0.251／0.309dB，LPIPS和SSIM略好，全部六个理解指标均下降。训练损失下降没有转化为这次留出集合的整体收益。三组里没有adapter系数1或0且理解可训练的同预算控制，不能把本次变化全部归因于系数0.01；原B仍是系数1且完整数据／不同预算。
3. 从零GS终点PSNR10.749，比旧GS冻结组低8.531dB，16/16场景PSNR更低，全部11项更差。自身PSNR仅从10.550升至10.749，AbsRel/RMSE/LPIPS变差。重置的是全部GS重建状态（含tokens/decoder/anchor/head），保留已适配VGGT、adapter及理解参数；反馈从源时钟起即beta0.1。它支持当前预算保留旧GS，不证明从第一阶段起先重建适配的新GS长训练上限更低。

## 训练曲线与渲染诊断

| 对照 | 重建损失：前10→末10更新 | 理解损失：前10→末10更新 | 最大裁剪前范数 |
|---|---:|---:|---:|
| AA冻结＋旧GS | 0.04170→0.03460 | 0.22288→0.13699 | 57.35 |
| AA解冻＋旧GS | 0.04162→0.03412 | 0.22287→0.14845 | 46.33 |
| AA冻结＋新GS | 0.61311→0.13084 | 1.21435→14.99377 | 298.08 |

这里每更新先平均八rank，随后对前／末10更新平均。沿用GS两组训练损失均下降，但留出性能没有相应提升，提示需要检查泛化和联合优化而非只看拟合。新GS的理解损失在后段明显升高，渲染仍接近低细节灰色；96张图像平均图内RGB标准差只有0.0391，旧GS约0.1898。不能把“重建loss下降”当成新GS已学会场景。

八rank逐条审计800条／组正式记录：更新1—100连续、每窗口100曝光、梯度范数跨rank一致、损失与范数有限、冻结参数检查始终通过。9个评测条件0/50/100共144窗口／864图像，全部零优化器更新；checkpoint回执SHA与评测身份一致，两个沿用GS组初始RGB逐字节相同、所有GT跨条件相同。所有历史工程失败与几何WARNING保留，遵守monitor_v1。

## 下一步建议（不自动提交新训练）

B与三组结果合起来支持：不继续原配置A/B长训；旧GS继续作为优先初始化，AA解冻保留候选而不宣布无效。完整epoch6保留为重建、mAP/PQ参照；B可作为语义mIoU的补充参照，绝不能把不同任务优点混成一行虚构模型结果。

下一个小预算试验应直接回答adapter梯度路由能否泛化：同一个起点、AA暂冻结、理解可训练，对比系数1／0.01／0；使用更广的真实训练窗口、相同曝光预算以及固定独立留出集。这样补上本组三组没有的路由控制，并减少16窗口重复100次造成的拟合偏差。先看到重建和理解的共同收益或明确可接受取舍，再决定正式从头按阶段适配；不根据训练窗口24.5直接长跑，也不因本次短预算失败全面否定AA或从零初始化。

## 证据

- [三组原始13列汇总](/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1/attempts/attempt01/evaluation/analysis/report.md)、[CSV](/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1/attempts/attempt01/evaluation/analysis/siu3r_table1_comparison.csv)。
- [完成审计与逐场景差值](/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1/attempts/attempt01/analysis_review_20261013/verified_analysis.json)、[逐场景CSV](/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1/attempts/attempt01/analysis_review_20261013/per_scene_reconstruction.csv)。
- [11指标学习曲线](/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1/attempts/attempt01/analysis_review_20261013/heldout_metrics.png)、[训练loss曲线](/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1/attempts/attempt01/analysis_review_20261013/training_losses.png)、[固定前两场景渲染](/space/mawb/ssst/group_plus/vggt_joint_short_controls_v1/attempts/attempt01/analysis_review_20261013/fixed_scene_renders.png)。
- [B完整13列对照](/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_batch16_v1/checks/evaluation_recovery_20261013/full_validation_13_columns.md)、[CSV](/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_batch16_v1/checks/evaluation_recovery_20261013/full_validation_13_columns.csv)、[完整审计与差值](/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_batch16_v1/checks/evaluation_recovery_20261013/completion_analysis.json)、[B原始指标](/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_batch16_v1/joint_aa/evaluation/final_continuation2_official_recovery_20261013/metrics.json)。
- [B恢复提交回执](/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_batch16_v1/checks/evaluation_recovery_20261013/submission.json)、[B最终VGGT逐张量验证](/space/mawb/ssst/group_plus/object_locus_vggt_joint_continuation_ab_batch16_v1/checks/evaluation_recovery_20261013/final_vggt_tensor_verification.json)。
- 原启动协议与证据保留于[vggt_joint_short_controls_v1.md](vggt_joint_short_controls_v1.md)，此前四训练窗口结论保留于[vggt_adapter_gradient_results_20261012.md](vggt_adapter_gradient_results_20261012.md)。

进度查看脚本现在读取launch_plan中的恢复评测目录，不再固定查旧失败root；实际运行验证能显示补评测并在完成后展示新表。全部目标评测已完成，停止轮询，无新训练提交。
