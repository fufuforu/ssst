# 独立审查交接：Gaussian实例分割多版本实验、匹配机制与共同瓶颈

本文件供另一个agent远程连接服务器后独立阅读实验结果与具体代码，再与已有分析比较。用户要求的是独立分析，不是重复训练。请先核对原始产物与实现，再判断下面哪些解释成立；不要将本文件的假设当作已证明根因。

## 0. 可直接交给另一个agent的任务

> 请在服务器读取本文件，独立检查旧C、原Unified、A双分支及直接8维Gaussian实例属性方案的实现和已完成实验。重点判断：训练匹配/监督覆盖、Query职责、Gaussian到实例的归属、分类与Mask质量、输出竞争/筛选，以及重建基础是否存在被多个实验继承的问题。请区分确定的实现错误、可能不合适的设计选择、评测定义限制和仍缺乏证据的假设。先阅读原始CSV/JSON和代码，再阅读前一个agent的结论；列出认同、反驳、证据不足及新增发现，并给出至多三个优先问题及最小验证方案。本轮先只读，不改模型、官方指标或现有快照，不重训、不重复提交已完成作业。

## 1. 工作区、执行环境与边界

- 开发仓库：`/space/mawb/ssst_vggt_overfit_diagnostics_v1`。
- 分支：`vggt-overfit-diagnostics-v1`；远端：`https://github.com/fufuforu/ssst.git`。
- 本文之前最近审查记录提交：`cac0415`；全量结果记录：`c1ae625`。开发HEAD还会随文档更新前进，具体实验必须看独立快照SHA。
- 新全量实际执行快照：`/space/mawb/ssst_direct_instance_full_code_20261018`，SHA **`3d06c39ad97657959052329c892bda023f07436b`**。
- 新短测快照：`/space/mawb/ssst_direct_instance_short_code_20261018`，SHA `f9b5a5d2e68d0701a6b81ebb651d01db316142a0`。
- 常用Python：`/space/mawb/anaconda3/envs/tokengs/bin/python`。正式SIU3R评分Python：`/space/mawb/SIU3R/.venv_gpu_v4/bin/python`。
- 控制节点CUDA驱动旧，CPU导入可能有驱动告警；不要直接把控制节点不能CUDA前向解释为训练环境失败。真实训练/评测在Slurm GPU节点完成。
- 部分集群收据记录`/datapool/mawb/...`，控制节点以`/space/mawb/...`访问同一工作区。核对文件身份时需要明确处理此前缀映射，不因本地没有`/datapool`挂载就判为错误checkpoint。
- SIU3R：`/space/mawb/SIU3R`，固定提交 `8ea80166be76854f938e90521f1a5b688b755c87`。

先读`/space/mawb/AGENTS.md`。所有下列作业已经完成，不因旧文档中的RUNNING/PENDING重提。用户明确约定提交后只做一次调度核对，停止轮询，由用户通知后再检查。新的GPU诊断/训练不是本文件自动授权的动作。

不要重置开发分支、覆盖不可变快照或已有完成收据；独立审查建议写新文件。现有证据足够先做代码与CPU产物审查。需要额外前向时应先说明具体缺失证据与受控方案。

## 2. 首先澄清四种不同的对应关系

### 2.1 Parent Query → Gaussian：属性生成布局

一个最终Query产生多个Child Gaussian属性。Parent Query索引记录这些Gaussian由谁生成，不等于GT实例ID；一个GT物体可能覆盖多个超点/Parent Query。

### 2.2 Gaussian → Query Prototype：模型内部归属读出

最新方案直接从最终Query预测每Child Gaussian的8维实例属性，同时每Query预测8维Prototype。Gaussian属性与全部Query Prototype单位化点积，经可学习scale/bias和**独立sigmoid**产生归属概率。

这里没有用GT，也不是匈牙利匹配；独立sigmoid没有强制每个Gaussian只属于一个Query。即使Gaussian由q0生成，其实例归属也可指向q1等其他Query。

同一份有符号8维Gaussian属性还经原Renderer渲染8通道特征，渲染后L2归一化用于Pull/Push/Cross及HDBSCAN。Gaussian几何/RGB渲染布局仍为原14维，8维实例属性作为sidecar，不是把原Renderer的几何ABI直接改成22维。

### 2.3 Query预测 → GT实例：训练责任分配

预测Query编号没有天然GT实例语义。要计算“q的类别和Mask应该对哪个GT”，训练仍然需要分配对应关系。这里才涉及`source_many_to_one`或匈牙利两种不同策略。

当前原Unified full4和直接8维full4的实际配置都是：

```text
matching_mode = source_many_to_one
support_matching = true
support_purity = 0.5
```

**最新实验实际没有调用匈牙利分支。** 换了8维属性/原型读出，没有改变继承的训练责任分配。

### 2.4 预测Mask/聚类 → GT：评测对应关系

正式mAP/PQ以及辅助一对一GT IoU各有评测匹配规则。这些匹配只在预测之后用于计分，不反向决定训练Query标签。HDBSCAN本身也不需要GT匹配，评测其输出时才与GT比较。

当前HDBSCAN在每个视图独立聚类、类别无关，未关联跨视图实例ID；仅在固定8窗口作逐图辅助比较，没有把它当作正式128的多视图实例AP结果。

## 3. “多对一”具体是什么，为什么还要匹配

这里“多”指多个预测Query，“一”指同一个GT实例，不是多个GT被塞进同一个Query。

例如一把椅子A被分成两个超点，桌子B对应另一个超点：

```text
Query q0（椅子局部超点） → GT椅子A的完整Mask和类别
Query q1（椅子另一超点） → GT椅子A的完整Mask和类别
Query q2（桌子超点）     → GT桌子B的完整Mask和类别
```

实现位于`tokengs/models/unified_query_loss.py`：

1. `support_relation`使用初始超点来源像素的GT实例标签，统计每Query对各GT的来源纯度。分母包含有效thing和stuff来源像素。
2. 纯度至少0.5才允许成为该GT的正Query。
3. 对每个有允许GT的Query，在允许项中选来源纯度最高的GT；多个Query可以选同一个GT。
4. 这些正Query分别监督为所选GT的**完整两视图Mask**，不是只监督自己的局部超点。
5. 无正匹配Query监督为no-object，thing BCE/Dice只对正Query计算。

因此，Prototype预测的“我像哪个实例”并不会自动给出“我应该负责哪个GT”；后者仍由上述训练规则指定。

这不是认定多对一必错。多对一加去重可以有效；需要核实的是当前几何种子职责、完整Mask正监督、最终实例合并/选择是否一致。推理目前主要靠同类NMS与逐像素竞争，未显式把同GT的多个Query合并成唯一Prototype。

## 4. 匈牙利为什么可能出现GT没有Query：必须区分三条代码路径

### A. 普通、无硬约束的匈牙利

`tokengs/models/object_locus_v3_set_loss.py:final_hungarian`构造完整预测-目标成本矩阵，以类别/BCE/Dice代价调用`linear_sum_assignment`。在**有效Query数量N≥有效GT数量K、成本有限、没有后续删除匹配**时，会返回K对，每个GT都有Query。

分类概率很低或Mask很差，只会让匹配质量差，不会因此自动跳过某个GT。训练已经分配给GT的Query，模型仍可能预测no-object；这是分类预测错误，不等于matcher没有分配。

若N<K，最多返回N对，才会因为数量不足使一些GT无配对。已经被GT有效性规则排除的对象，也不能算作“匈牙利漏掉了有效GT”。

### B. 有来源硬约束的匈牙利备用分支

`unified_query_loss.py`中的非`source_many_to_one`分支可以先按来源关系设置`allowed`，把禁止项成本改为1e6，再调用匈牙利，最后用`allowed[qi,ki]`删除禁止配对。

所以即便N≥K，若GT没有允许Query，或允许关系无法形成覆盖所有GT的一对一匹配，**匹配后过滤**仍会留下未匹配GT。这不是普通匈牙利在有限稠密矩阵上主动漏配。

例如q0只允许A，q1不允许任何GT，A/B各一个：solver仍可在有限1e6矩阵上给出两对，之后B的禁止配对被删除。另一个例子是A、B都只允许同一个q0；每个GT有候选，也不足以保证一对一完全覆盖。

### C. 最新实际使用的来源多对一分支

这条路径不调用匈牙利。GT若没有足够纯的来源Query，就没有正监督；有允许Query也可能在逐Query argmax时被别的GT占用。

上一轮提到的**11.28%未获得正Query，来自这条路径，不能称为“匈牙利漏匹配”。**

67390最后200次全局更新×8rank，共1600训练窗口曝光：GT曝光6685，最终匹配GT5931；754/6685=11.28%没有分类/归属正Query。其中735个完全没有允许来源Query，另19个有允许关系但最终未被选中；不能直接把19个都归为0.5纯度并列，需重放关系确认细节。

该统计是训练尾段窗口曝光，不是全训练集独立GT统计，也不是heldout召回。渲染Pull/Push/Cross仍监督这些GT的像素，因此不能称它们完全没有监督。

## 5. 已完成实验与主结果

以下正式结果均为固定128场景协议；target沿用context＋4额外帧，不是纯novel四帧。分割转为百分数。旧C与新架构在源预训练、适配、容量和配方上有差别，是性能参考，不是全部单因素对照。

| 方案 | PSNR | Target mAP | Target PQ | 状态 |
|---|---:|---:|---:|---|
| 旧C：VGGT＋MASt3R参考 | 22.5966 | 27.3756% | 60.9790% | 已完成 |
| 较早V4梯度隔离 full4 | 23.1940 | 18.9803% | 53.6217% | 已完成 |
| A双分支 full4 | 18.5003 | 22.7075% | 62.7599% | 已完成 |
| A双分支累计8epoch | 19.3692 | 24.8446% | 61.5404% | 已完成 |
| 原Unified full4（主基线） | 18.0855 | 16.5192% | 45.0114% | 66432训练完成 |
| 原Unified累计8epoch | 18.4225 | 15.5542% | 48.2612% | 66648训练完成 |
| B直接Gaussian Mask full4 | 18.0839 | 13.0896% | 45.6597% | 已完成 |
| 新直接8维属性/Prototype full4 | 18.0470 | 15.3114% | 45.4203% | 67390/67391完成 |

新方案相对原Unified full4：PSNR−0.0385 dB；context mAP18.9247%→16.7029%；target mAP−1.2079个百分点，PQ+0.4089个百分点，语义mIoU小幅下降，尚未整体胜出。

新方案训练：1191场景/8337窗口、4epoch/4172更新、batch8、seed42、LR1e-4/100步warmup/cosine，理解权重0.05/1043步ramp。与原Unified full4相同预算；292个共同初始张量逐值一致，但新头、渲染实例监督等是组合改动。原基线没有渲染实例监督，不能把差异仅归于8维属性。

本组总目标：`Lrec + u * (0.1*(2*Lcls + 5*Lthing_BCE + 5*Lthing_Dice + 5*Lstuff_BCE + 5*Lstuff_Dice) + 0.01*(2*Lpull + Lpush + 2*Lcross))`，u预热后为0.05。系数及数值大小不等于实际梯度贡献，不能单凭这些常数判定监督是否太弱。

67390：COMPLETED/0:0，4h32m23s，3dimage-13、8×3090。67391：COMPLETED/0:0，23m55s，同节点1×3090。正式训练/初始空Adam/306张量变化与有限/Adam4172/8rank日志与样本顺序/首步SHA同步均核查通过。独立CPU复核64组辅助缓存指标一致，不是没有更新或训练提前终止。

## 6. 服务器原始产物位置

统一训练父目录：`/space/mawb/ssst/group_plus/unified_query_training_v1`。

| 内容 | 父目录下的运行目录/位置 |
|---|---|
| 原Unified full4 | `attempt02_esam_many_full_node11_8gpu` |
| A双分支 full4 | `attempt05_legacy_dual_superpoint_full_node13_8gpu` |
| B直接Gaussian Mask full4 | `attempt06_unified_direct_masks_full_node18_8gpu` |
| A累计8epoch | `attempt08_legacy_dual_superpoint_extend8_node13_8gpu`，正式结果在`official_epoch6/evaluation`、`official_epoch8/evaluation` |
| 原Unified累计8epoch | `attempt09_unified_query_extend8_node11_8gpu`，正式结果同上子目录 |
| 原渲染实例监督短测67050 | `attempt11_unified_instance_grounding_short_node13_4gpu` |
| Gaussian一致性短测67292 | `attempt15_gaussian_instance_consistency_short_node13_4gpu` |
| 直接8维短测67369 | `attempt16_direct_instance_attributes_short_node13_4gpu` |
| **直接8维全量67390/67391** | **`attempt17_direct_instance_attributes_full_node13_8gpu`** |
| 已有跨版本合并表 | `completed_comparison_20261018/official_11_metrics.csv` |

其他参考：

- 旧C正式表：`/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv`。
- V4隔离：`/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt12_geometric_sam_isolated_node13_8gpu`；理解权重0.05组：同父目录`attempt13_geometric_sam_weight005_node18_8gpu`。
- 67050/67292固定权重机制与HDBSCAN：`/space/mawb/ssst/group_plus/fixed_instance_readout_v1/attempt01_67050_67292`，作业67349已完成。
- 较早双分支128场景分阶段诊断：`/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt02/full`。

新全量run重点文件：

```text
configuration.json / preflight.json / STARTED.json
checkpoint_initial.pt / checkpoint_latest.pt / checkpoint_epoch1..4.pt
train_rank00..07.jsonl
FIRST_UPDATE_VERIFIED.json / FULL_TRAINING_AUDIT.json
TRAINING_COMPLETE.json / FULL_EXPERIMENT_COMPLETE.json / COMPLETION_REVIEW.json
evaluation/siu3r_table1.csv / metrics.json / all/results.json
evaluation/export_complete.json / checkpoint_identity.json
evaluation/all/<scene_context...>/{context,target}_seg_pred/*.png
evaluation/all/<scene_context...>/{context,target}_seg_pred/pred.json
evaluation/all/<scene_context...>/{context,target}_seg_gt/*.png
comparison_to_baseline_full4/comparison.csv / comparison_status.json
readouts_final/SUMMARY.json / 00..07.{pt,json} / FIXED_WEIGHTS_rank00.json
PROBES_INITIAL.json / PROBES_EPOCH1..3.json / PROBES_FINAL.json
SHARED_FAILURE_SUPERVISION_AUDIT.json / SHARED_FAILURE_PNG_AUDIT.json
review_shared_failure_png.py / review_figures/
```

注意：完整128正式导出主要保存最终PNG、类别/分数、RGB/depth与GT；不能由这些文件恢复所有原始Query Mask/几何贡献。`readouts_final`只有固定4train+4heldout的8窗口，保存原型raw/NMS/最终及HDBSCAN等缓存。要做完整128原始候选到最终的分阶段复核，可能需要新的固定权重GPU前向；现有8窗口不能冒充128场景。

## 7. 代码阅读路线

| 问题 | 代码 |
|---|---|
| 实际训练配置、损失与DDP | `scripts/train_unified_query_v1.py` |
| 全量预检/审计/比较及依赖提交 | `scripts/direct_instance_full.py`；两个`run_direct_instance_full*.sh` |
| 模型注册/变体选择 | `tokengs/models/superpoint_structural_variants.py` |
| Unified骨架、上下文生成与相机标定 | `tokengs/models/unified_query_v1.py` |
| Query/Child几何、原始读出 | `tokengs/models/unified_query_core.py` |
| 超点来源/初始化/特征池化 | `tokengs/models/unified_superpoints.py` |
| **来源匹配、多对一、受限匈牙利与损失归一化** | **`tokengs/models/unified_query_loss.py`** |
| GT有效性、GT类别与旧稠密匈牙利 | `tokengs/models/object_locus_v3_set_loss.py` |
| **直接8维属性/原型/推理** | **`tokengs/models/direct_instance_attributes.py`** |
| 渲染Pull/Push/Cross及有效权重 | `tokengs/models/unified_instance_grounding.py` |
| 特征渲染 | `tokengs/rendering/gs.py` |
| 归属alpha归一化 | `tokengs/models/object_locus_v3_set.py:alpha_normalize_membership` |
| 同类NMS | `tokengs/models/unified_query_nms.py` |
| **像素竞争、面积过滤、导出分数/标签** | **`scripts/export_object_locus_v3_set_official.py`** |
| 正式导出与GT身份检查 | `scripts/export_unified_query_checkpoint.py`、`scripts/dual_stream_official_export.py` |
| 固定官方评分入口 | `scripts/invoke_siu3r_official_evaluator.py`、`scripts/score_dual_stream_v1.py` |
| 官方加载与AP/PQ | `/space/mawb/SIU3R/src/evaluator.py` |
| 聚类配方与辅助评测 | `tokengs/instance_clustering.py`、`scripts/evaluate_direct_instance_attributes.py` |
| 重建目标/多层监督 | `tokengs/models/canonical_recon_models.py`、`canonical_recon.py` |
| 冻结Omega/SAM与GFM适配 | `tokengs/models/dual_stream_frozen_backbones.py`、`gfm_dual_stream_components.py` |

## 8. 已确认事实与未证实解释

### 已确认的代码行为/观测

1. 当前多对一为来源硬匹配；并非预测质量驱动的匈牙利。训练GT责任不会因最终Mask更好而自动转移。
2. 多个正Query分别学习完整GT，推理还有竞争和删除；被面积删除的像素保持void、不二次分配。
3. 负Query分类损失先平均再整个组乘0.1。尾段1600曝光中该组系数占比均值3.28%，**不是实测梯度贡献占比**。
4. 固定heldout辅助全GT每图IoU：本组raw70.02%、NMS67.46%、最终原型54.79%、HDBSCAN49.45%；最终原型TP26/36、聚类18/36。不是正式AP/PQ。
5. 全量相对短测：原型最终7.89%→54.79%、聚类12.71%→49.45%；数据与训练覆盖改变了泛化水平。但正式等预算AP没有胜出。
6. 多个新架构重建18–19 dB，与旧C22.60有差距；较早V4达23.19，不能说所有双分支都失败或Omega/GFM必然不足。

### 需要特别核对的PQ定义

实际评分环境文件：

```text
/space/mawb/SIU3R/.venv_gpu_v4/lib/python3.10/site-packages/torchmetrics/functional/detection/_panoptic_quality_common.py
```

其中`_filter_false_negatives`忽略预测void覆盖超过一半的未匹配GT，`_calculate_iou`还扣除预测void覆盖GT的union部分。正式分数不改写，但不能把PQ当作固定全GT召回。

完整128 PNG复核：target有效GT上的预测void覆盖6.36%→7.61%；被忽略的未匹配thing GT **62/507→58/507**，新组反而更少。因此**不能声称这次PQ增益是忽略了更多GT所致**。它仍是必须连同AP、覆盖和分母一起解释的指标。

### 尚未证实，不应当作结论

- 多对一训练或背景归一化就是AP下降的唯一原因。
- 所有失败都来自NMS/组装；早期一些版本的raw本身也弱。
- PQ上涨全部是void伪增益。
- 直接8维属性容量必然不足，或者延长训练必然有效。
- 现有工程审计排除了所有实现错误；它主要排除空跑、错误初始化、数据身份不一致和部分梯度/同步故障。
- 高GT-oracle最佳Mask/来源探针IoU等于官方检测性能。
- 类别正确、Mask质量、模型置信度三者已经对齐。

## 9. 独立agent建议输出格式

建议新写`docs/independent_review_findings_<日期>.md`，保留本文件和原始产物。

1. 环境/分支/审查版本；实际看过的原始文件。
2. 先明确当前实际匹配分支，分别报告未允许GT、无最终正Query、预测no-object、推理漏检，不能混为一谈。
3. 对现有每个主要假设给出：认同/反驳/证据不足、具体代码位置或数据、是否跨版本共有。
4. 新发现的确定bug与设计疑点分别列出；不要仅以代码看起来不常见证明错误。
5. 优先级最高的至多三个问题；每个给最小受控验证、应观察的指标，以及能排除什么解释。
6. 对比原agent分析，明确新增证据或分歧。若需要新的固定权重前向，说明现有缓存为何不足；不默认开始训练。

现有分析入口：

- `docs/shared_failure_audit_20261018.md`及`.json`：最新共同链路审查、日志统计与PNG void统计。
- `docs/direct_instance_attributes_full_results_20261018.md`及`.json`：67390/67391正式全量结果。
- `docs/direct_instance_attributes_short_results_20261018.md`：67369短测。
- `docs/gfm_instance_diagnostic_results_20261016.md`：较早双分支128场景原始Mask→类别→竞争→面积过滤。
- `docs/continuation_and_serial_residual_results_20261018.md`：A/Unified续训及串行诊断。
- `docs/vggt_understanding_bottleneck_findings_20261014.md`：更早分类/质量排序与PQ限制。
- `docs/codex_handoff_gaussian_consistency_20261018.md`：全时间线；下方有历史状态，**以文件顶部最新完成记录为准**。

先看原始结果、再看这些分析，避免把同一份解释换个agent复述当作独立验证。
