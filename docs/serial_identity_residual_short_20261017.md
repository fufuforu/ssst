# 串行Query：原超点身份残差短测

> 2026-10-18已完成核实，66653成功26m01s；386/386张量更新/Adam256，源383张量/4rank RNG及样本流匹配，32窗口新诊断与复用基线完成。身份区分度和空Mask比例略改善，最终验证IoU0.0404→0.0386、最终Recall@0.5仍0，不直接转全量。保留旧4+新4重复告警，严格重放/权重不变通过。完整证据与图集见[完成分析](continuation_and_serial_residual_results_20261018.md)。下方RUNNING为历史，不重跑。

用户确认“那就试试吧”，授权在串行语义seed中加入原超点Token门控残差，做同预算短测。只新增残差组，原串行66603训练和66637诊断直接复用；不重训基线，不查询或修改A/原Unified续训。

## 唯一模型变化

原始超点Query为Q0，重建12层后为Qrec；原语义seed路径保留原LayerNorm+Linear。新增：

`seed_new = seed_old(stopgrad(Qrec)) + sigmoid(gate_logit) * W0(layer_norm(stopgrad(Q0)))`

W0为256→256线性投影，权重单位阵、偏置0；门控初始0.1。归一化Q0使用无可训练参数的functional LayerNorm。只新增gate_logit、投影权重/偏置3个张量（65,793参数）；383原张量逐值复用，共386张量/19,490,501参数。重建参数14,209,457不变，理解参数5,281,044，仍无共享梯度组。新增参数构造使用CPU fork_rng，不推进原有模型/前端RNG流。

两路输入均detach，新投影/门控归理解组；重建Gaussian、Child几何及理解渲染仍按原串行隔离。理解4层Decoder、独立语义memory、原Child–Query点积Mask头、类别/语义组装、多对一监督/NMS阈值均不变。不是增加第二组独立Query，也不是修改Gaussian Mask读出。

旧SerialQueryCore只将原seed表达式抽成可覆盖方法，原方法逐字保持原计算；测试gate置0时原生理解输出与原串行逐值一致，正常gate时Gaussian重建输出也保持一致。原Unified/A正在执行旧独立快照，不受本次通用入口改变影响。

## 起点、数据与训练

源：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt07_serial_query_isolated_short_node18_4gpu/checkpoint_initial.pt`。拒绝256步终点、非空Adam、错误架构/数据/资产/世界尺寸/训练窗或多余缺失键。

用源383张量覆盖原有模块，仅保留新模块的显式初值；源4rank Python/NumPy/Torch/CUDA RNG逐rank复用，Adam为空，不加载基线训练后动量。准备产物`checkpoint_prepared_zero.pt`记录新架构及相同数据/几何配置，源文件/准备文件SHA链见preflight。

固定原16场景/256更新/2048曝光、global batch8，18节点4×4090、每rank microbatch1累积2。LR1e-4/20步warmup/原短cosine，理解weight0.05/20步ramp，重建/理解分别clip1，冻结Omega/SAM。未要求loss强制下降，不自动进入完整数据训练。

原训练的checkpoint_initial保存于初始探针之后。本次从该状态恢复，再执行新初始探针，因此探针必须保持所有RNG：补全既有probe对NumPy状态的保存/恢复（Python/Torch已有保护），单测明确调用NumPy随机数后检查不变。GPU训练前4份SOURCE_RNG_VERIFIED收据强制比较完整Python/NumPy/Torch/CUDA状态；初始保存再次与源rank_rng对比。第一步全参数SHA跨4rank一致及分支/冻结梯度检查继续强制执行。

新独立run：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt10_serial_identity_residual_short_node18_4gpu`。不复用失败/旧run目录。保留初末固定4train/4heldout探针、每32步恢复检查点、门控训练日志及最终参数变化记录。

## 自动原生Mask诊断与可视化

同一个作业训练成功后，固定终点在原16训练+16验证窗口诊断（32窗口）。复用66637的原串行32份原生记录及8张案例PNG，不重新运行原串行推理；源终点SHA、每份旧记录SHA和同一scene/context/RGB SHA/GT ID/类别必须核对。

新组复用已有observe：原生Gaussian membership→渲染Mask→alpha→类别→NMS→像素竞争→最终面积过滤，各阶段全部GT/正确类别/来源支持最佳IoU及Recall@0.5、空Mask Query比例、实际渲染贡献、Token实例区分度与打乱负对照等。主比较用原生结果，不根据中间状态套最终训练头的分布外读出宣判层有害。

同次原生seed/memory严格重放D4和最终Mask/类别头，4份参数逐值不变/无梯度检查，诊断无优化器/无参数更新。保留完整前向重复告警及原上限，不把旧66637的4个告警抹去。只评估额外随机模块的固定权重结果，不改变官方阈值或评测协议；这不是官方128 mAP/PQ。

新旧图像复用同一GT选择，固定概率色标；图集同时展示旧/新两行。输出`diagnostics/{SUMMARY.json,REPORT.md,gallery.html}`及run COMPLETE，明确原生最佳IoU/实例数不等于官方AP/PQ或正确匹配数。

## 判断依据

优先看训练与验证原始Mask是否改善、空渲染Mask Query是否下降、最终GT辅助召回是否增加，结合分类正确IoU和阈值扫描precision/recall，避免仅分类提升或低阈值扩大Mask的假收益。若Token区分度改善而原生Mask未改善，下一步应检查Gaussian特征/Mask读出，而不是继续堆初始化或Decoder深度；如果仅训练改善，则仍需区分短数据泛化问题。完整数据训练需等本轮结果再决定，本次没有自动授权。

## 验证与提交

真实预检：383源张量逐值一致、386新模型键/形状、空Adam/4rank RNG、原模型仅seed方法抽取的等价代码核对、冻结资产/训练验证manifest SHA、66637原生记录完整及4个实际RGB样本SHA通过。

20项CPU测试通过：精确残差公式/初值、构造器RNG与原参数不变、gate0恢复原理解输出、重建输出不变、全理解梯度不进入重建、零步迁移拒绝错误状态、原串行梯度/clip、完整原生重放/权重不变、警告保留、汇总与GT不一致拒绝、NumPy/Python/Torch探针RNG保护。Python编译、shell语法及diff检查通过。新GPU首步、完整训练及新GPU诊断尚未核实。

执行快照`/space/mawb/ssst_serial_identity_residual_code_20261017`，作业号与唯一一次状态见run/submission.json和SUBMISSION_STATUS_ONCE.json。预计训练约25–30分钟，加诊断与汇总约2–5分钟；排队另计。提交后停止轮询，等待用户通知完成。

## 已提交并停止轮询

作业 **66653**，18节点4×4090、16CPU/128G，唯一一次调度核对 **RUNNING**。执行SHA **db948ab55fc3fb70d66fbe1e3ac2a3e0bfd64cf2**，独立快照干净。20项CPU回归、真实零步/资产/数据/旧原生记录SHA预检以及编译/语法/diff检查通过；新GPU源RNG检查、第一步同步、256步训练与新GPU固定权重诊断尚未核实。RUNNING不代表这些检查通过。

同作业内训练→短测汇总→32窗口原生诊断→与复用66637结果比较并生成图集，无额外训练依赖、无官方128或全量自动提交。已停止轮询，不等待首步或结果；A66640、原Unified66648未查询或修改。用户通知完成后检查此run的COMPLETE、SHORT_VALIDATION_SUMMARY及diagnostics/REPORT、SUMMARY、gallery和所有验证收据。
