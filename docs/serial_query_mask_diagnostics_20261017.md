# 串行Query固定权重Mask链路诊断

> 更新：66634在初始完整前向重复校验失败，未产出完整诊断。后续修复使用attempt02与独立retry快照，改为同一次原生前向的seed/memory重放验证，不再给理解层加hook；原完整前向差异原样保留为稳定性告警。下文初版hook/witness叙述是历史设计，以下修复节优先。原Unified仅作辅助参照，不是架构单变量对照。

用户在66603短测后授权诊断，不更新参数、不修改结构或正式评测协议。新独立run：`/space/mawb/ssst/group_plus/serial_query_diagnostics_v1/attempt01`；独立快照：`/space/mawb/ssst_serial_query_diagnostics_code_20261017`。18节点4×4090/16CPU/128G；一个作业内完成推理与CPU汇总，提交后一次状态核对即停止轮询。

## 比较对象与数据

- 串行隔离短测66603的256步checkpoint_latest，383张量；原Unified短测66414的256步checkpoint_latest，301张量。均固定终点、不加载完整训练终点、不执行训练入口。
- 真实checkpoint SHA、冻结Omega/SAM SHA、模型几何配置/数据身份逐项核对。训练rank数不同不影响本次同卡型推理；既有训练策略/容量不同，不能将对比全部归因于某一模块。
- 原固定16训练场景全部使用，加固定128 heldout中的前16场景；32个不同场景/模型、共64窗口。其中包含先前4train/4heldout探针和scene0570_00失败个例。
- 模型输入只含两context图像；GT只在generate之后用于统计。两模型逐窗口核对context输入RGB SHA、场景/帧、GT实例ID/类别一致。heldout数据加载继续验证原固定cohort RGB SHA。

## 诊断内容

1. **状态变化**：只读hook提取4层理解Decoder输出，结合超点初始、Q_rec、语义seed，计算Query范数/通道方差、同GT/不同GT余弦及差值；16次打乱GT身份对应作为负对照。不会把正常全局语义聚合的高相似度直接判为错误。
2. **Gaussian→渲染**：归属logits/概率分布、每Query最高Gaussian概率、own-parent Child概率与其他parent均值、渲染最大概率及可见区域最大概率、目标像素/非目标像素概率和soft IoU。Parent只作来源身份代理，不给移动后的Gaussian硬贴3D GT标签。
3. **实际渲染贡献**：复用既有RendererTranspose，测每Gaussian在两context中的alpha-compositing积分贡献，以及对各GT区域的贡献。其内部只对临时feature probe求导，模型参数/几何全部detach、无优化器/参数更新；最终验证所有模型参数grad为None。贡献加权概率与逐像素概率不同，报告不混淆这两种平均。
4. **完整组装链路**：原始0.5 Mask（不加alpha）→alpha>0.05→类别/no-object/分数→context原NMS0.7→像素竞争→面积保留过滤。复用生产assemble_panoptic，并逐窗口验证诊断面积过滤重放与生产输出一致。所有阶段分别统计全部GT oracle、正确类别oracle、来源支持（含无支持GT零分）、仅支持GT结果；不将best-query统计当AP/PQ。
5. **阈值探测**：仅在诊断中扫描0.1–0.7，比较有/无alpha门槛，记录正Query与其GT的IoU、预测像素precision/recall，防止将整幅激活的高召回误判为有效改善。native输出/官方阈值仍为原值，扫描结果不写回模型或评测。
6. **中间层辅助读出**：用最终训练好的原Unified分类/Child-Query相似度Mask头读Q_rec、seed和D1–D3，保持最终Gaussian几何不变；D4头重放必须与原生logits/membership/class输出一致。早期状态对最终头是分布外输入，只提供定位线索，不能当成经过独立训练的消融或据此直接判某层有害。
7. **对应关系负对照**：固定16次打乱Query与来源GT对应，不重训/不重渲染，比较来源支持IoU；同时保留全部Query oracle，避免只看被监督正Query而遗漏其他候选。

## 无扰动与不更新保证

全模型eval且requires_grad=False，只有渲染器转置内部的临时features允许求导；没有optimizer构造、loss.backward或参数更新。每模型每rank首窗口进行两次无hook原生前向＋一次观察前向，复用已有repeat_calibrated_witness的数值/packed输出边界检验只读观察无扰动，不假设CUDA求和逐位确定。

原始checkpoint文件SHA在推理前再次核对；各模型处理完本rank所有窗口后，逐张量与源checkpoint值精确比较，并检查全模型无参数grad。4rank/2模型共8份UNCHANGED收据。汇总检查64窗口完整、同RGB/GT、全部witness、D4重放后才写COMPLETE。

## 输出和验证

每窗口JSON含原生指标、各阶段GT IoU、逐Query漏失记录和负对照；SUMMARY.json/REPORT.md汇总，gallery.html展示原4train/4heldout的16张图。串行图选该场景支持GT中最差者，原Unified图复用同一GT；跨层保持同一Query展示概率，统一blue0/green0.5/red1色标，避免不同色标制造差异。图像包含RGB、GT、alpha、原生概率、最终实例及各中间层辅助概率。

新增测试覆盖S=3/105可变Query的生产组装一致性与逐级Mask集合、低概率/低alpha区分、无支持GT零分与支持子集统计、完全相同Token/打乱负对照；完整模拟模型观察包括原生重复witness、D4头重放、各层捕获、临时feature渲染转置、无参数更新/梯度、JSON有限值、图像与64窗口汇总产物。另回归串行模型梯度边界和完整初末探针。

不做官方AP/PQ评分，不启动训练，不改A/B作业。本次GPU诊断尚未执行；提交后预计约10–20分钟，实际受场景Query数及渲染影响，排队另计。作业号/最终测试项数/执行SHA以submission.json和VALIDATION_EVIDENCE.json为准。

## 已提交并停止轮询

诊断作业 **66634**，18节点4×4090、16CPU/128G，唯一一次调度核对 **RUNNING**。执行SHA **e4ae921**、独立快照干净。提交前11项CPU测试通过，包含完整模拟观察/现有渲染器转置/图像/64窗口汇总产物；真实源checkpoint与冻结资产SHA、32窗口选择及实际数据读取预检通过。GPU原生witness/不可变性/D4重放检查尚待该作业实际执行，RUNNING不等于这些验证通过。无训练或额外评测依赖；结束同作业自动汇总。提交与状态记录见run/submission.json、SUBMISSION_STATUS_ONCE.json。现停止轮询，用户通知完成后再分析报告和典型图。

## 66634失败审计与单次前向重放修复

66634 FAILED/1:0，运行1m35s；4rank均在第一窗口`Observer exceeds native repeat witness tolerance`退出，无COMPLETE/正式窗口JSON/汇总结果。源权重没有更新。原代码在异常前未落盘具体witness差异，不能宣称已确认是哪个CUDA算子、VGGT/SAM还是观察hook造成；只能确定旧跨完整前向校验没有通过，不能据此给串行模型下结论。失败目录attempt01及快照e4ae921全部保留。

新方法不提高或改动原repeat_calibrated_witness的floor/ceiling。两次完整原生前向与第三次前向的原始差异保存为WITNESS文件；超界或Query形状改变显式写入native_repeat_warning。它只衡量完整路径稳定性，不再冒充“相同输入状态下的observer一致性”。汇总保留全部差异，若有告警则标记DIAGNOSTICS_COMPLETE_WITH_REPEAT_WARNINGS，不伪造通过。

不再注册理解层forward hook。用透明的encode_context返回值捕获器原样返回同一个对象，仅保留当次semantic memory；原生generate照常执行。随后使用当次原生semantic_query_initial与该memory按同样顺序重放4层理解Decoder，严格核对D4与原生understanding_query，rtol1e-5/atol1e-6，并确认seed/memory未被改写。最终Mask/class头重放、生产组装重放、源权重逐张量不变及参数无grad检查继续强制通过。该校验避免重新运行VGGT/SAM/点聚合引入不同实现的输入状态，也消除层hook对执行路径的潜在影响；没有修改模型、原生阈值或源checkpoint。

新增回归模拟变化的前端：原完整前向witness失败必须如实保存，但同次输入的D4重放仍需严格成功；另故意破坏D4重放结果，必须拒绝。汇总测试验证带告警的结果不会标为无告警完成。CPU回归及真实checkpoint/冻结资产/32场景预检重新执行，原Unified仍仅辅助参照。

修复诊断独立run：`/space/mawb/ssst/group_plus/serial_query_diagnostics_v1/attempt02`；独立不可变快照：`/space/mawb/ssst_serial_query_diagnostics_retry01_code_20261017`。推理与自动汇总预算/输出内容仍为两模型各32窗口、18节点4×4090，无训练。原失败尝试没有已完成窗口可复用，本轮不重复任何成功训练。新作业号及验证证据以attempt02/submission.json和VALIDATION_EVIDENCE.json为准；提交后一次核对即停止轮询。

修复提交前13项CPU回归通过（含变化前端、破坏重放必须拒绝、告警汇总、既有Mask/可视化/不可变性），32窗口真实数据与冻结资产/源checkpoint校验再次通过。独立修复执行SHA **e7a610c**。

补跑作业 **66637**：18节点4×4090、16CPU/128G，唯一一次调度核对 **RUNNING**；新GPU同输入重放/完整推理检查尚未核实。无训练更新。原66634及attempt01保留，补跑收据见attempt02/submission.json及SUBMISSION_STATUS_ONCE.json。已停止轮询，待用户通知完成后再检查；本次尚没有可汇报的模型诊断结论。
