"""Build the report draft; the reviewed TP/FP/FN appendix is editorial and must be preserved."""
from pathlib import Path
import csv,json
root=Path('/space/mawb/ssst_anchor_child_boundary_publication_20261021_v1/docs/anchor_child_boundary_results_20261021')
models=list(csv.DictReader((root/'official1860_comparison.csv').open()))
feature={r['metric']:r for r in csv.DictReader((root/'paired_scene_statistics.csv').open())}
lines=['# 2026-10-22：P1024iso / No-CA 正式评测及 A4→A8 纯诊断','',
'两项训练、两项官方1860评测、诊断 smoke 与完整128窗均正常完成（Slurm COMPLETED，exit 0）。本轮核查未提交新作业。',
'', '|作业|内容|节点 / GPU|耗时|实际结果|','|---|---|---|---|---|',
'|70026|P1024iso full4|13 / 8×3090|7:17:16|4 epoch / 4172 updates|',
'|70042|P1024iso 官方1860|13 / 8×3090|0:57:25|1860窗 / 312场景，评分成功|',
'|70162|P1024iso No-CA full4|12 / 8×3090|7:28:47|4 epoch / 4172 updates|',
'|70183|No-CA 官方1860|12 / 8×3090|1:20:41|1860窗 / 312场景，评分成功|',
'|70234|纯前向诊断 smoke|14 / 1×4090|0:04:57|2窗 × 两个旧checkpoint，通过|',
'|70235|完整128窗纯诊断|13 / 8×3090|0:11:27|128场景 × 两checkpoint，通过|',
'', '## 正式1860结果：理解退化已被确认','',
'以下均为相同官方1860窗、312场景；mIoU/PQ/mAP用百分数。target 是既有 target-all 六视角导出口径，包含 Context，不是仅四个 novel 的指标。',
'', '|模型|PSNR↑|AbsRel↓|Context mIoU|Context PQ|Context mAP|Target mIoU|Target PQ|Target mAP|',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in models:
 lines.append('|'+r['model']+'|'+ '|'.join(f'{float(r[k]):.3f}' for k in ('psnr','absrel','context_miou','context_pq','context_map','target_miou','target_pq','target_map'))+'|')
lines+=['',
'相对旧 A4，P1024iso 的 PSNR +0.293 dB，但 Context/Target PQ -3.334/-4.091 pp，mIoU -1.772/-1.878 pp，mAP -3.385/-3.704 pp。',
'No-CA 相对带 CA 版本 PSNR +0.184 dB，Context/Target PQ +1.426/+1.471 pp，mIoU +0.802/+0.828 pp；仍低于旧 A4。',
'深度没有同步改善：AbsRel 从旧 A4 0.18695 变为0.20690/0.21118，RMSE从0.41005变为0.45148/0.46009，不能笼统称全部重建指标改善。',
'原4窗probe的警告被正式评测支持，但最终判断使用这里的完整评测，不把probe当正式mIoU/PQ。',
'', '## 身份、完整性与解释边界','',
'- 两项训练8个rank各4172条连续更新记录、全部 loss/grad_norm 有限，首步权重发生更新且8rank哈希一致；最终训练收据 epoch4 / update_in_epoch0。没有声称另外测过最终8rank权重哈希。',
'- 正式评测权重与训练latest为同一硬链接，流式重算SHA分别为 `b8d65a659d4296ef9aeb8172cf3026c08a16b38bf1f88ae5010cbf89c0577909`、`afe21b135c46315586cbb8331c46c26edd9a9128f261fde2e93210bf6b6bee36`，与冻结收据一致。',
'- 三组官方1860输入RGB与GT清单一致，官方 evaluator 为 `8ea80166be76854f938e90521f1a5b688b755c87`。评测参数更新为0。',
'- export_complete.json 历史字段 unique_scenes 写成1860，实际是窗口数；真实场景数312，已按官方cohort/校验收据报告，不沿用该字段。',
'- 旧A4与P1024iso同时改变Anchor数量、seed/query初始化、2+2→2+4、FM分辨率、监督层和隔离adapter，故不能把下降单独归因于隔离。',
'- No-CA保留纯几何归一化权重更新c/s，不是移除几何更新；只在当前配方、单种子下支持这个方向，不能从约1.4pp PQ就裁决所有CA无效。',
'- 当前训练仍对全部可训练参数统一clip_grad_norm_(...,1.)。detach切断理解到重建的梯度路径，但共享clip系数使两支优化动态并非完全独立；不能把No-CA带来的0.184dB直接解释为CA伤害重建。下轮调LR/理解权重应同时记录各组裁剪前梯度与clip系数，本轮不改clip。',
'', '## A4→A4+4：共享父特征与子GS区分度','',
'固定旧A4 epoch4 / 旧A4+4 epoch8，128个相同且不同的验证场景，每checkpoint256个Context视角。不能与上面的1860绝对指标混用。',
'GT归属使用全部GS正常遮挡下的渲染贡献乘GT实例mask，不使用中心投影标签。',
'', '|统计（配对场景均值）|A4|A4+4|A8−A4 的95% CI|','|---|---:|---:|---|']
items=[('全部Anchor residual/shared范数比','feature.all.residual_shared_norm_ratio'),('跨实例Anchor residual/shared范数比','feature.cross.residual_shared_norm_ratio'),('跨实例Anchor比例','feature.multi_instance_fraction'),('跨实例Anchor下 child 异实例cosine','feature.cross.purity0.7.child.cosine_different'),('跨实例Anchor下 final 异实例cosine','feature.cross.purity0.7.final.cosine_different'),('跨实例Anchor下 final 同−异cosine gap','feature.cross.final.cosine_gap_same_minus_different'),('跨实例Anchor下 membership 异−同JS gap','feature.cross.membership_js_gap_different_minus_same'),('跨实例Anchor下 Bernoulli entropy','feature.cross.bernoulli_entropy')]
for label,key in items:
 r=feature[key];lines.append(f'|{label}|{float(r["A4"]):.6f}|{float(r["A8"]):.6f}|[{float(r["ci95_low"]):.6f}, {float(r["ci95_high"]):.6f}]|')
lines+=['',
'结论：共享项主导原始 child 特征，异实例子GS几乎同向；续训增大residual，但最终特征与membership的同/异实例分离没有一致显著改善。跨实例Anchor比例下降，可能体现几何/贡献覆盖改善，不能当成同Anchor内部区分能力增强。',
'融合FM之后异实例cosine约0.55，而不是raw child的0.99999，说明最终实例区分并非只依赖共享父项。不能仅靠raw cosine证明共享Anchor导致能力上限；本轮没有移除共享项的因果消融。',
'纯度0.5/0.7/0.9的跨实例membership JS变化CI均跨0，判断不依赖单一纯度阈值。',
'128窗全部完成；主跨实例成对比较有96个可用配对场景（其它场景无足够可见/高纯度同Anchor异实例对），全部Anchor同/异实例 gap 比较有99个。不能误写成每项统计均包含128个有效配对场景。',
'平均每窗3922个GS，其中可分配高纯度GT的GS约1025→1083，约占26.1%→27.6%；不强行标注其余GS，结论覆盖范围受可见性/归属阈值限制。',
'', '## 连续边界：改善，但不足以进入当前Boundary-Aware RGB Loss','',
'所有512个Context视角六图及512份raw/soft float32 NPZ本地保留；本目录只发布预定前4个窗口的16幅六图、16份连续响应NPZ。',
'', '|边界指标|A4|A4+4|','|---|---:|---:|',
'|BF1@2px|0.3227|0.4021|',
'|连续 raw AP@2px|0.3079|0.3408|',
'|物体内部raw响应能量占比|44.92%|44.06%|',
'|物体内部soft≥0.5激活率|5.92%|8.45%|',
'',
'BF1提升约7.93pp（95% CI +6.23到+9.76pp），但仍偏低，且约44%的有效raw响应落在离GT边界3px之外的物体内部。连续AP有125个有效配对场景，3个没有可用GT边界的窗口不定义AP；BF1统计128窗。',
'人工检查固定第一窗：A8更清楚地响应门/椅子等部分轮廓，但椅背与物体内部仍有大量纹理状响应。先改善或筛选边界提取/置信度，再考虑小规模损失试验；目前不推荐直接用这张soft map作为RGB loss权重。此结论针对本次固定提取方法，不等于所有边界方法均不可行。',
'', '![A4固定第一窗Context0六图](A4_000_context0_sixpanel.png)',
'', '![A4+4同窗Context0六图](A4plus4_000_context0_sixpanel.png)',
'', '## 零反向与数值等价证据','',
'诊断过程 model forward固定权重、无optimizer；backward/autograd.grad API被禁止。全部rank哈希前后一致，backward_calls=0、parameter_updates=0。',
'128窗所有case最大alpha守恒误差6.56e-7、one-hot特征渲染重组误差9.54e-7、转置内积相对误差1.37e-8，最终mask logits重组误差0。',
'源/权重/完整性收据见 POSTRUN_AUDIT.json / COMPLETE.json；原始执行commit保持 `3ac6b4f1878168929dfb3b773710acc193eb4676`，本报告提交不修改执行快照。',
'', '## 下一步建议（未提交新实验）','',
'优先把理解下降拆成分类、漏检/重复预测、mask空间质量与stuff问题；基于下方TP/FP/FN及逐类结果决定理解侧LR或损失内部比例。不要仅凭总loss下降就继续提高总理解权重，也不再优先加容量或直接加Boundary Loss。',
'先保留当前两组结果作为对照；若要裁决隔离本身，必须保持P1024、FM/query初始化、输入视图、监督层一致，只改变隔离设置。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('report written')
