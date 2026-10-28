# E0–E2固定权重诊断：本轮配对对照与历史漂移审计

用户在核查71223失败后明确指示“继续”，授权沿已说明的验证口径继续修复与重新提交。本轮仍仅Baseline 4epoch固定权重（理解权重1.0、训练2context+2novel），E0、E1及E2_class/filter/rank/joint，更新0。源SHA256仍为694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc。1860窗、312场景、256²、target-all6包含Context，Oracle只读取两个Context GT。

## 为什么改变历史预测检查

原71223已FAILED，71224已CANCELLED；52窗全部分支完成，54窗部分E0导出432图中仅1像素与历史不同。GT、候选ID/类别一致，置信度最大差7.75e-7。原历史参考也在同一18节点4090/相同gsplat后端运行；具体微差来源未证明，不能把历史逐像素零差异作为推理可比性的充分必要条件。此前的门禁是我新增的过严执行约束，不是用户要求改变模型来拟合历史预测。旧失败记录和部分输出全部保留。

## 新的严格检查

每窗生成和标定仍只执行一次。任何Oracle之前调用未经改动的原生assemble_panoptic，冻结其六视角packed像素、原生候选资格和原始预测张量的地址/version。Oracle生成后要求E0仍为同一原始输出、keep等于原生keep、原始Membership/类别/alpha/semantic/NMS资格张量未变。导出E0再逐像素核对这份同前向原生装配；不一致直接失败。整轮模型参数version核对、有限值、1860唯一窗/312场景门禁保留。

源权重SHA、manifest/heldout/cohort身份、逐窗RGB哈希保持严格；各分支GT文件集合及全部PNG字节仍与历史固定GT完全相同。目标GT只用于评分/输入一致性核对，不用于Oracle决策。任何同前向E0变化、Oracle篡改输入张量、候选资格变化、输入或GT变化均直接失败。

导出完整后生成PAIRED_BASELINE_VERIFIED.json（NATIVE_SAME_FORWARD_E0_VERIFIED），不生成历史BASELINE_REPRODUCTION_VERIFIED.json，也不声称历史逐像素完全复现。评分开始必须验证1860条唯一record均具有native_control_verified以及完整配对收据。每个分支通过原样SIU3R固定commit评分，差值基准严格使用本轮E0。

## 历史漂移完整记录

历史E0预测不再要求逐像素/分数零差异；每窗query_audit仍记录逐文件字节差、解码像素差、语义/实例像素差、总像素数、具体差异文件、候选ID增减、类别变化和最大置信度绝对差。所有差异如实保留，不根据52窗结果选容差，不清零、不篡改参考，也不把有差异结果称为bitwise复现。

评分时生成HISTORICAL_BASELINE_METRIC_AUDIT.json（本轮E0与历史各指标及真实差值）与HISTORICAL_BASELINE_AUDIT.json（全1860窗的像素/候选/指标漂移汇总）。五项重建指标仍引用同权重已核验Baseline，不重复计算，五项零差由复用产生而非测得复现。六项理解及AP50/AP75/COCO AR@1/10/100都重新评分。历史审计没有声明容差通过，后续报告必须单独展示历史漂移，不能隐去大差异；E1/E2收益始终相对同轮E0。

E1/E2具体干预不变：E1只合并Baseline原本保留且关联同实例Query，以预测Membership逐像素max并集，不用GT补mask；E2分别修正类别、候选资格、排序及共同作用。候选关联仍为Context mask阈值.5、alpha>.05、GT占完整预测mask面积严格>.5，保持同一映射用于target。不训练、不直接验证Gaussian几何引导或理解反哺重建。

## 执行与验证

新代码独立worktree：/space/mawb/ssst_fixed_oracle_diagnostics_code_20261023_v4_paired；新运行根：/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v4_paired。旧v1/v2/v3冻结代码及运行记录不覆盖。继续申请18单节点6卡4090、24CPU/192G导出；afterok的评分汇总申请1卡、8CPU/96G；各12h时限，不是耗时承诺。

针对新验证逻辑测试历史一像素差保留、同轮E0一像素篡改拒绝、GT变更/缺失拒绝、输入张量或keep变更拒绝、1860唯一且全部验证收据，以及模拟完整六分支评分验证收益确实相对本轮E0且历史漂移未隐去。原局部求导/Oracle测试一并运行，Slurm test-only和源身份复核在提交前完成。CPU/模拟测试不代替真实GPU或完整评分成功；提交后只核对一次调度接收状态并停止轮询，等用户通知完成后再检查。
