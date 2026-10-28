# 固定4epoch权重：实例整合、分类与筛选诊断

用户选定已核验的 Baseline（理解权重1.0、2context+2novel）4epoch最终权重；不等待或改动现有8epoch三链。没有训练、优化器更新、参数拟合、目标GT决策、额外实验组或8→16。代码从publication c7e255f独立建worktree；原冻结执行快照不改。

## 固定来源与评测

源：`/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1/eval/source_checkpoint.pt`，SHA256 `694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc`，progress=4/0/4172。这是已验证的Baseline，不声称8epoch最佳或超参数意义上的全局最佳。

固定官方1860窗、312场景、256²、两张Context生成，原目标相机标定，target-all6包含Context。SIU3R固定 `8ea80166be76854f938e90521f1a5b688b755c87`，不改评分器。一次forward共享全部诊断。每组重新导出并评分理解结果；RGB、深度不受Membership干预，链接已核验Baseline导出并引用原五项重建指标，禁止重复宣称为新测量。深度仍为官方scale+shift拟合口径。

E0重新评分的六项理解指标必须与既有Baseline相符（百分数绝对差<=1e-4），且所有分割PNG逐文件SHA必须一致；不一致先写失败收据并停止汇总，不静默更换Baseline。所有组输入RGB及GT逐窗与原Baseline核对。

## Context Oracle 对应关系

取筛选前预测Membership>=0.5、alpha>0.05的两Context原始Query mask，忽略预测类别，以GT `(semantic_class, instance_id)`为实例身份。Query只有在一个GT实例占其完整预测mask面积严格超过50%时才关联，void/background也计入分母；无关联保持unmatched。不用target GT，不替换预测mask，不用GT填充缺失部分。规则固定，不按结果调门限。

这些是受限Oracle诊断干预，并非数学意义的全局最优上限。两个Context都不可见的物体无法获得Oracle身份；错误几何、跨物体mask及纯背景mask不能靠类别修正恢复。Context-GT辅助结果与常规预测结果必须清楚区分。

## 三组实验及E2内部拆分

|编号|操作|保持固定|用途|
|---|---|---|---|
|E0|原始Baseline门限、原Context NMS和像素竞争|全部原流程|复现全部11指标，并记录AP50/AP75、AR|
|E1|只对Baseline原本保留、且关联同一GT实例的多个Query合并；逐像素max Membership应用于全部六视角|原最高Context排序分数Query作为代表，保留它的预测类别及thing probability；其他Query、stuff、几何、alpha不改|隔离实例碎片合并收益，不同时救回被过滤候选或修正类别|
|E2_class|交换关联Query的最大thing类别与GT类别概率|最大thing probability、no-object probability、原Baseline keep、全部Membership固定|类别错误造成的损失|
|E2_filter|GT关联作为候选资格，绕过no-object及0.05置信度门限；按原分数和预测类别重新执行相同Context NMS|预测类别、概率、Membership及后续像素竞争固定|候选筛选造成的损失；单独记录no-object、置信度、NMS拒绝计数，计数可能重叠|
|E2_rank|只将最终预测实例的pred.json score改为其最终Context mask对所有Context GT实例的最大类无关IoU；同一分数用于target|像素归属、类别、候选集固定，不借GT重新分配像素|置信度排序造成的损失|
|E2_joint|类别修正、候选修正后重新原NMS，再使用上述Context IoU排序|Membership与几何固定，不进行E1合并|三项共同干预后的诊断结果，不能把收益相加|

E2的四个输出是同一编号实验内的因素拆分。E1继续用原导出打分公式（代表Query的thing probability × 合并后被分配像素的平均Membership），不使用GT分数。逐像素max是渲染后Membership并集，不等于对Gaussian Membership取max后再渲染，更不证明3D几何引导有效。

## 输出与解释

`metrics_and_deltas.csv`：全部11项、Context/target AP50/AP75及官方COCO AR@1/10/100、各自相对E0差值；理解/AP/AR以百分数报告、差值为百分点。AR是IoU .50:.05:.95平均召回，有候选数量上限；不冒充单一阈值Recall50。

`query_audit/*.json`：每个GT实例的Query数、最佳单Query IoU/覆盖率、并集IoU/覆盖率，分别统计全部原始关联Query与Baseline保留Query。记录真实merge组和代表Query。`diagnostic_summary.json`统计“best IoU<.5而union IoU>=.5”的碎片可恢复实例数，及各类筛选损失计数。这些是Context候选覆盖诊断，不冒充正式最终实例Recall。

E1若提升、且union覆盖显著超过最佳单Query，支持“已有部分覆盖但实例整合不足”。若仅E2_class提升，优先考虑类别判别；仅E2_filter提升提示候选门控损失；仅E2_rank提升主要提示分数与质量错配。无提升不证明问题不存在：purity门限、已过滤碎片、mask污染、竞争及不可见实例均限制Oracle。单seed不证明稳定排序，target含Context不能当novel-only收益。

本轮尚未直接测试Gaussian几何的预测性作用，也尚未测试理解反向改善重建；因此标题中的“互惠”是后续研究方向。本轮E0–E2回答瓶颈定位，为随后不依赖GT的Gaussian邻接/连通性/可见性干预提供依据。

## 执行与交接约束

运行根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v1`。
源码：`/space/mawb/ssst_fixed_oracle_diagnostics_code_20261023_v1`。

仅一项8卡export与一项afterok评分汇总；前者共享一次生成/标定/渲染，后者无需新增训练。提交前完成CPU针对性测试、实际checkpoint/数据身份与源文件SHA检查、shell/Python检查及Slurm test-only；CPU检查不声称GPU验证。GPU导出包含逐窗finite/输入/GT/参数version核对。任何失败先诊断，不自动重提。

提交记录单独保存作业号和唯一有界状态核对。提交后遵守AGENTS：不等待首步、指标或完成，不启动监控；已停止轮询，等用户通知完成后再检查。原71192/71193、71194/71195、71188/71189不查询、不改动、不重提。
