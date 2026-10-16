# 统一GFM双Adapter：直接joint4与adapt2+joint4结果

结论：两组训练完整完成，但最终模型未达到旧VGGT+MASt3R C的质量。重建适配确实有效，随后联合训练发生明显重建能力退化；目前优先解决联合优化，而不是加入文本分支或新交互结构。不能将“作业结束”当作指标已达标。

## 完成状态与可比性

- 65251 COMPLETED 0:0，恢复后耗时7:07:36；最终joint epoch4/full_updates4172、8337窗口全部覆盖，冻结Omega/SAM未变化。
- 65252 COMPLETED 0:0，恢复后耗时6:24:57；最终adapt epoch2/updates2086＋joint epoch4/updates4172，8337窗口全部覆盖。
- 两组各128场景、每场景2输入＋4额外帧的固定权重导出全部成功。场景名、source_index、6帧ID、RGB SHA256逐项一致；导出过程中GT PNG已逐文件验证匹配旧C。官方Evaluator commit仍为8ea80166be76854f938e90521f1a5b688b755c87，src无修改。
- 65256/65257的官方Evaluator均已算完128场景，并保存evaluation/all/results.json；上层score脚本因AbsRel/RMSE NaN不能序列化为严格JSON而FAILED。65258因afterok依赖取消。两组没有COMPLETE，不能宣称正式11项全部有效。

## 原生官方结果

以下直接读取原版Evaluator保存的结果，无需重跑模型或评分。mIoU/mAP/PQ按百分数展示；SSIM/LPIPS保持原量纲。官方target含输入2帧＋额外4帧，不替换成纯额外4帧。

| 指标 | 旧C | 直接joint4 | adapt2+joint4 |
|---|---:|---:|---:|
| AbsRel↓ | 0.08730 | NaN | NaN |
| RMSE↓ | 0.21321 | NaN | NaN |
| PSNR↑ | 22.597 | 12.826 | 12.496 |
| SSIM↑ | 0.78076 | 0.50598 | 0.50409 |
| LPIPS↓ | 0.36805 | 0.74654 | 0.73697 |
| context mIoU_s↑ | 56.02 | 37.66 | 28.93 |
| context mAP↑ | 30.80 | 7.88 | 6.45 |
| context PQ↑ | 64.55 | 54.51 | 50.93 |
| target mIoU_s↑ | 55.26 | 35.29 | 27.91 |
| target mAP↑ | 27.38 | 6.91 | 5.79 |
| target PQ↑ | 60.98 | 48.36 | 46.16 |

两列text mIoU保持空白。原13列表格已恢复为comparison_native_with_invalid_depth.csv/md，明确写NaN，与text空白区分；不是合格的“11个有限数值”表，不修改原有完成检查来放行它。

分阶段组只在LPIPS上略优于直接组（-0.00957），其余8项有效指标均更差；PSNR低0.330dB，target mAP低1.12个百分点。直接组距旧C仍差9.77dB、target mAP差20.46个百分点，不能据当前结果选择任一组作为已达标基础模型。

## 最重要的阶段证据：适配学会了重建，联合阶段没有保住

下表是同一固定8验证窗口监测，不是128场景官方分数；用于同模型跨阶段比较。

| 分阶段模型时点 | PSNR/dB | 匹配二值Mask IoU | 匹配类别准确率 |
|---|---:|---:|---:|
| 共用short306起点 | 12.214 | 0.377 | 0.229 |
| 适配epoch1 | 20.167 | 0.446 | 0.162 |
| 适配epoch2 | 21.486 | 0.446 | 0.166 |
| 联合epoch1 | 13.271 | 0.574 | 0.359 |
| 联合epoch2 | 12.492 | 0.571 | 0.593 |
| 联合epoch3 | 12.779 | 0.613 | 0.680 |
| 联合epoch4 | 12.704 | 0.572 | 0.712 |

适配2后至联合1，PSNR下降8.215dB；至联合4下降8.783dB，同时理解分类和匹配Mask学习有进展。这是联合优化改变已学重建行为的直接阶段证据。Omega冻结，因此不是冻结Omega权重被微调坏；退化发生在可训练重建/共享token/几何/GS输出等路径。

Write=0仅关闭显式前向写回，并不阻止理解损失经Object读Anchor、Child Gaussian Membership和渲染等路径回传重建参数。以上阶段证据尚不能定位具体梯度通路或证明某一loss分项是唯一原因，需要后续受控干预；不能直接宣称Ownership或SAM初始化导致退化。

分阶段组固定8窗口匹配IoU比直接组高（0.572对0.408），正式128场景mAP却更低；不能依据小队列、GT匹配后的Mask IoU判定最终实例质量已达标。分类、no-object筛选、像素竞争/组装以及几何覆盖仍需分开定位。

## 深度NaN与失败案例

两组每组768张深度图：直接组4帧全零，分阶段组2帧全零，且这些帧GT均有超过6万有效像素。它们恰好对应全部非有限深度分数。官方fit_scale_and_shift使用有效GT上的torch.linalg.lstsq，零预测导致设计矩阵秩亏，在本次官方CUDA运行中返回NaN，继而AbsRel/RMSE整体mean变NaN。

- 直接组：scene0427_00的562/563/577，scene0277_01的108。
- 分阶段组：scene0406_01的598，scene0427_00的577。
- 共同典型失败：scene0427_00、输入帧577。GT是椅子与房间，两个模型均输出纯灰RGB、全零深度。这不是只在未见视角发生的细节模糊。

不删帧、不以nanmean替代官方mean、不改求解器或给预测补epsilon以制造有限分数。原生results.json与每帧depth_scores完整保留，审计报告将无效值表示为null并保留invalid标记；CSV仍明确NaN。

## 下一步建议（未提交）

最值得验证的一组控制：从现有checkpoint_adaptation_complete.pt开始，冻结重建Adapter、重建decoder/几何与GS预测参数，只训练语义Adapter、Object Query交互、分类及Mask模块；使用stop-gradient阻断理解损失对重建路径的更新。先验证能否同时保住适配后的重建质量并学会理解。这比继续增加联合epoch或立即加文本分支更能回答当前问题，且不需要先改模型结构。当前未授权/未提交这组新训练。

## 产物

目录：/space/mawb/ssst/group_plus/dual_stream_training_v1/comparison_joint4_vs_adapt2_joint4_20261015

- comparison_native_with_invalid_depth.csv / .md：原13列对比，9有效＋2无效＋2text空白。
- completed_runs_audit_20261016.json：完成状态、相同队列、768帧逐帧深度失败证据。
- fixed8_training_drift.png：固定验证窗口跨阶段PSNR/Mask IoU曲线。
- failure_scene0427_frame577.png：同一输入帧GT、两模型RGB/深度典型失败。

恢复脚本scripts/report_gfm_completed_comparison_20261016.py基于既有官方原始结果和导出PNG生成，参数更新0、新推理0、新评分0。所有Slurm作业已结束，当前无排队/运行任务；不重复训练或重新提交会产生相同NaN的评测。
