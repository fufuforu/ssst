# SSST Superpoint–GT Matching Audit：正式完成

固定 A4 权重下，**来源覆盖不足与 Mask 质量是两个主要损失环节；一对一 Hungarian 没有额外损失 GT 覆盖。当前没有证据支持为了提高覆盖率而重写 Matcher。** Hungarian 和来源约束 Top-1 高度接近，但存在真实 Query 复用冲突，不能宣称完全等价。

## 执行与身份

| 项目 | 已核实事实 |
|---|---|
| Heldout32 | 作业 69237，COMPLETED / 0:0，2分38秒；32场景/32窗口/122个GT/2139个Query |
| Heldout128（主结论） | 作业 69238，COMPLETED / 0:0，6分34秒；128场景/128窗口/492个GT/7844个Query |
| 资源/依赖 | 3dimage-14，各1×RTX4090、8CPU、64G；128为afterok:69237；汇总在各统计作业内执行 |
| CPU / 真实 Smoke | 13项CPU合同通过；8真实窗口/37GT Smoke通过；全128真实RGB/GT数据预检通过 |
| 执行代码 | `9a949c52b86ac8796f1eb1f4d2965f34347752aa`；固定标签 `audit-ssst-matching-v1-20261020` |
| A4训练源码 | `b7a775f19e232ea2395c8611a47c81b1365fb86e` + 实际sem256 SemanticMemory原样快照；详情见source_audit.md |
| 训练Manifest SHA256 | `87e3cde298569c4e7735cf75ee596868eadf01e9a6a95438c6be51e42f55a523` |
| 固定Heldout Manifest SHA256 | `9a310d764358ebd7cdd3d1844a48685264d25d48cced5d4263ef4da68c239a44` |

Checkpoint：`/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu/checkpoint_epoch4.pt`。
SHA256：`18aee68f2bc8cdb1d154c742697ccc54657f38ad02c6151bf993405a455f4560`。
Epoch4 / Global Step4172；全部Adam状态步数4172。使用其原始`source_many_to_one / support_matching=true / support_purity=0.5`配置构造模型；仅离线调用原Loss的Hungarian分支获取真实Cost。B/C/D共享同一次输入与预测，不加载Hungarian训练权重。

数据由原生`build_batch`加载已处理ScanNet，固定Manifest路径和逐窗口frame/source_index/RGB SHA写入各audit_config.json。原Provider按照场景存在情况使用`/space/mawb/SIU3R/data/scannet/train`或`val`。Heldout32是固定128记录中的前32，**不能相加为160个独立场景**。两份报告的规模分界分别按事先规定的1/3、2/3分位数计算，仅在同一固定集合内部共享分界。

全部窗口的原生来源scores/allowed与Hungarian配对逐值Parity通过；反向调用0、参数更新0、参数版本未变化、梯度为空。独立CPU复核全部保存矩阵、最大匹配、Top-1、IoU汇总、GT表及48行阈值/规模表通过。日志无Traceback或非有限值错误，未生成失败收据。

## Q1：来源支持覆盖多少GT，尤其小物体？

主口径为`purity>=0.5, hits>=1`，严格复现原生有效像素定义。分母包含有效Stuff与正Instance ID的有效Thing；零有效观测分母clamp到1，不改Matcher。

| 规模 | GT | 有候选 | Support Recall | 无候选 | 合格Query均值 / 中位数 / P10 / P90 | 0 / 1 / 2–3 / ≥4个Query |
|---|---:|---:|---:|---:|---|---|
| 总体 | 492 | 427 | **86.79%** | 65 | 7.45 / 4 / 0 / 17 | 65 / 57 / 105 / 265 |
| Small | 164 | 106 | **64.63%** | 58 | 1.53 / 1 / 0 / 4 | 58 / 38 / 50 / 18 |
| Medium | 164 | 157 | **95.73%** | 7 | 5.04 / 4 / 1 / 10 | 7 / 17 / 46 / 94 |
| Large | 164 | 164 | **100.00%** | 0 | 15.79 / 11 / 4 / 32 | 0 / 2 / 9 / 153 |

面积是两张Context有效GT Mask的像素和；Small≤2382像素，Medium为(2382,11130.667]像素，Large>11130.667。没有额外GT最小面积过滤，原生有效GT包含极小可见片段；这里的小物体指Context可见面积，不能直接等同真实三维物体尺寸。65个未覆盖GT中58个为Small（89.23%）。

未覆盖原因：**16个无任何采样来源观测，49个有观测但没有Query达到纯度0.5**。49个中9个仅1–2次GT来源观测，16个仅1–4次；这是重叠稀疏标记，不能把这些数量再与16+49相加，也不能证明唯一因果。

7844个Query中3150个（40.16%）Thing来源为零，其中1507个连有效来源也没有；不能把这些都算成有效实例候选或都解释为纯Stuff。Query最大GT纯度≥0.3/0.5/0.7/0.9的占比分别为49.77%/46.56%/41.94%/37.06%。

### 同批输入的阈值与最少命中数敏感性

下表为GT Support Recall，全部分母492。其余规模分层和Query数分布详见`heldout128/audit_threshold_sweep.csv`。

| Purity | min_hits=1 | min_hits=3 | min_hits=5 |
|---|---:|---:|---:|
| 0.3 | 441 / 89.63% | 428 / 86.99% | 411 / 83.54% |
| 0.5（原定义） | 427 / 86.79% | 415 / 84.35% | 402 / 81.71% |
| 0.7 | 409 / 83.13% | 400 / 81.30% | 388 / 78.86% |
| 0.9 | 388 / 78.86% | 375 / 76.22% | 362 / 73.58% |

降到0.3只离线新增14个覆盖GT；0.5下要求至少5次命中会减少25个覆盖GT。不能从这些覆盖率直接推断训练质量。本次未修改训练阈值。

## Q2：来源约束与一对一最大覆盖分别损失多少？

| 数量 | GT | 占全部GT |
|---|---:|---:|
| 至少有一个合格来源Query（上界） | 427 | 86.79% |
| 真正最大二分匹配 | 427 | 86.79% |
| 当前来源约束Hungarian | 427 | 86.79% |
| 来源约束排除 | 65 | 13.21% |
| 有候选却未被Hungarian匹配 | **0** | 0% |
| 最大匹配数与实际匹配数差距 | **0** | 0% |

原`source_many_to_one`规则在同一保存来源矩阵上的离线核对也覆盖427个GT、3652个正Query，没有额外argmax纯度并列漏配；这里没有计算另一组模型预测或重训。

每个场景都达到最大匹配数，不只总体相等。分类2/BCE5/Dice5、禁止边1e6、SciPy求解及事后过滤均来自原`unified_instance_loss()`。实际允许边Cost范围−1.730011至45.142689；本次没有观察到有限禁止边惩罚导致覆盖损失。该结果不否定CPU合同中的一般性反例。

## Q3：Hungarian与OneFormer来源约束Top-1是否等价？

- GT覆盖：两者均427/492（86.79%）。
- 场景配对集合完全相同：**127/128（99.22%）**。
- 同时选中相同Query：**426/492（86.59%，全部有效GT为分母）**；若仅在427个支持GT内计算，则426/427（99.77%）。65个双方都未匹配的GT没有计入“选中相同Query”分子。
- 不同配对：1个GT。15个Query允许多个GT；Top-1存在1个Query被2个GT复用，保留冲突、未去重。Hungarian使用427个不同Query；Top-1仅426个不同Query。
- Hungarian总Cost26.763007，Top-1为26.464195，差0.298812；Top-1允许复用Query，二者约束不同，不能把更低Cost解释为更好的全局一对一解。
- 官方`SparseMatcher(topk=1)`严格小于第二小Cost的规则与任务argmin在本次真实输入上没有差异；精确并列的边界差异已由CPU合同覆盖，不推广为普遍等价。

唯一冲突为`scene0500_01_context699_710`，Query41的有效来源数4，其中GT5与GT36各2个，因此二者纯度均**恰好0.5**：

| GT ID | Top-1 Query / Cost / IoU | Hungarian Query / Cost / IoU | 另一候选 |
|---|---|---|---|
| 5 | 41 / 2.627327 / 0.291214 | 12 / 2.926139 / 0.270408 | Query12纯度0.875 |
| 36 | 41 / 3.500990 / 0.076965 | 41 / 3.500990 / 0.076965 | Query17纯度1，Cost3.923816 |

对GT5改用Query12的代价0.298812，小于对GT36改用Query17的代价0.422826；Hungarian正确保留全局一对一。此场景仍覆盖全部9个有候选GT。两个GT的候选Mask本来均未达IoU0.5，不能把本例Mask漏检归因于该配对变化。

**结论：没有必要为提高当前候选图覆盖率重新实现Matcher；直接把Top-1当作等价的一对一替代存在真实冲突风险。** 固定A4权重的离线结果不能用于推断另一种匹配策略训练后的收益。

## Q4：主要瓶颈在来源、一对一冲突，还是Mask？

| 诊断指标 | GT | Recall |
|---|---:|---:|
| 来源支持 / Hungarian匹配 | 427 | 86.79% |
| 任意Query的Raw Mask IoU≥0.5 | 369 | 75.00% |
| 合格来源Query的Raw Mask IoU≥0.5 | 353 | 71.75% |
| Hungarian实际选中Mask的IoU≥0.5 | 348 | 70.73% |

相对492个GT，可分成下列**不重叠**的四组：

1. 65个无合格来源Query。
2. 74个有合格来源Query，但所有合格来源Query的Mask都未达IoU0.5。
3. 5个存在合格来源且Mask达标的Query，但原分类/BCE/Dice综合Cost选中了IoU未达标的候选；原Cost并不直接优化二值IoU。
4. 348个Hungarian选中Mask达标。

65+74+5+348=492。一对一全局候选冲突额外覆盖损失为0，不能当作主要瓶颈；来源覆盖与Mask质量都值得优先诊断，后者在已有候选中仍损失79/427个GT。

补充：67个有候选GT连任意Query都没有好Raw Mask；9个无合格来源的GT却存在好Raw Mask；12个已有候选且任意Query有好Mask的GT，其选中Mask未达标（包括上述5个以及7个只有不合格来源Query形成好Mask）。全部21个“有好Raw却未选中”的GT为9个未匹配+12个已匹配但选中不佳。

一个GT的所有好Raw Query都判为No-object：`scene0591_02` GT39，来源最大纯度0.45、无合格候选，最佳Raw IoU0.5450；另一个已匹配Query判No-object：`scene0695_01` GT1，但其任意最佳IoU也仅0.4778。数量不足以把No-object视为本次主要损失环节；本报告没有评完整类别正确率或官方最终召回。

Mask二值化`>=0.5`，IoU阈值`>=0.5`，目标为两张Context有效正ID GT的联合Mask。有效像素按官方export的semantic0..19过滤；这批输入与Loss有效过滤额外像素差为0。Raw诊断不加NMS、面积16、分类/置信度过滤、packed-mask竞争，也不是AP或目标视图评测。**70.73%不能称为最终实例召回，86.79%不能称为Mask召回。**

## 可复现性与边界

Heldout32的覆盖111/122=90.98%，Hungarian达到同样最大匹配，Top-1无冲突且配对32/32相同；Raw99/122，支持Raw93/122，选中达标91/122。主结论采用全128，不把32/Smoke重复计算为新场景。

32个重叠窗口的来源图（hits/counts/purity/allowed/GT IDs/classes）和Hungarian配对跨两次运行均逐值一致。预测Cost不是跨运行逐值一致，最大绝对差0.001686096；未通过修改容差掩盖该差异。原生Parity是在每次同一预测下严格比较，全部通过；B/C/D方法比较也只使用同一次预测。本次跨运行Cost变化未改变匹配结果，不能宣称所有预测逐位可复现。

数据/权重/GT均真实可用；A/B/C/D以及32→128已完成。未额外使用Hungarian训练终点重复诊断，未扩展Train统计，未启动任何新训练或参数更新。建议下一步只先检查小物体来源混杂/稀疏和已有支持Query的Mask失败案例；这些是诊断建议，本次没有改模型、Loss或Matcher。

## 命令、产物与Git

```bash
# CPU合同；其后有八窗口真实数据预检、八窗口GPU Smoke，收据位于reports/
/space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.test_contracts
# 这两条统计命令实际分别在69237和69238内运行；环境和隧道见原run_formal.sbatch
/space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.run --stage formal --cohort heldout32 --smoke-receipt /space/mawb/ssst_matching_audit_runs_20261020/smoke8/SMOKE_PASSED.json --push-receipt /space/mawb/ssst_matching_audit_runs_20261020/formal_v1/push_verified.json --output /space/mawb/ssst_matching_audit_runs_20261020/formal_v1/heldout32
/space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.run --stage formal --cohort heldout128 --smoke-receipt /space/mawb/ssst_matching_audit_runs_20261020/smoke8/SMOKE_PASSED.json --push-receipt /space/mawb/ssst_matching_audit_runs_20261020/formal_v1/push_verified.json --output /space/mawb/ssst_matching_audit_runs_20261020/formal_v1/heldout128
# 从已保存真实矩阵独立核对并归档，不进行推理或训练
/space/mawb/anaconda3/envs/tokengs/bin/python -m audit.ssst_matching.review_results --archive
```

Git分支`audit/ssst-superpoint-gt-20261020`。执行代码固定为9a949c5，最终报告后续Commit独立记录，不移动固定执行标签。

每个`reports/heldout32`、`reports/heldout128`目录包含要求的七份输出以及AUDIT_COMPLETE和RESULTS_REVIEWED收据。归档CSV仅换行从CRLF规范成LF，原表内容不变；原文件SHA256在RESULTS_REVIEWED.json。原始日志/矩阵仍位于`/space/mawb/ssst_matching_audit_runs_20261020/formal_v1`；不上传权重、数据、预测、矩阵或训练日志。执行/依赖/资源身份见formal_execution.json；唯一冲突完整候选、纯度、Cost和IoU也存于RESULTS_REVIEWED.json，原矩阵本地保留。
