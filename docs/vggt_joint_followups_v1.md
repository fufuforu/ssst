# 主训练启动后的联合路径诊断与A/B/C扩大复评

用户授权：先启动adapter0.01/feedback0.05完整联合4 epoch，再开展这两个后续任务。
主训练64041及依赖四卡官方评测64042使用不可变990a5d4fbd497496c5a73748469254540781e095。
真实单卡64039通过；64041内四卡两调用/阶段smoke通过，包含6923，smoke不计正式时钟。
主训练20正式更新/160新增曝光已通过独立CPU审计，VGGT冻结、四rank RNG/采样、
fresh AdamW、原学习率/理解warmup通过；实际beta0.0076，峰值19.43GiB，停止轮询。
后续代码和作业不修改该主训练快照、输出或旧checkpoint。

## 1. A/B/C已有500步权重扩大复评

A原视觉训练/feedback0.1；B冻结MASt3R/ViT adapter/pixel decoder/feedback0.1；
C原视觉训练/feedback0.05。adapter与GS理解梯度系数均0.01，VGGT整体冻结。
全量官方312场景1860窗口，同时给出未参与选beta的248场景1478窗口。
后者排除所有64个选择场景，包括这些场景的其他窗口，不只是排除原64个窗口。
每窗口2输入+4额外帧，两组结果均按官方13列，仅两处文本mIoUt为空。
不拆context/target-all/true-novel表；不把500步权重称为完整4epoch收敛模型。

所有渲染/评分4090四卡。每组复用原64个同权重窗口的导出，复制到新证据目录，
避免官方写回评分JSON污染历史文件。其余窗口实际重新渲染，不重复训练。
固定SIU3R commit8ea80166be76854f938e90521f1a5b688b755c87，无原生评分源码修改。
官方逐图RGB/depth平均；全局语义/AP/PQ合并原充分状态，按canonical pair排序，
不以scene/shard AP平均冒充全局AP。确认248使用自己的全局metric状态。
实际GPU预检：三个模型各两窗口，重现同一旧64窗口的全部PNG/JSON，
另一窗口来自确认场景；完整/确认两套原生CPU/GPU评分及state合并合同通过才正式提交。
最终自动CPU对照核验A/B/C输入RGB和frame IDs逐条相同。

## 2. 联合梯度尖峰与clip影响

分别从A/B/C原update0300 checkpoint恢复原model、AdamW moments和四rank RNG，
按原4000-window流重放301至441，共141次调用/1128重放曝光每组，423调用三组。
这是短诊断重放，不是从头重训，也不续写原训练；保留GC、clip1、LR/AdamW数学。
对301/400/440/441观测完整分模块全局R/U向量，四rank平均后再计算范数/cosine，
验证R+GC系数U与实际安装梯度重合。CPU保存观测梯度，小块GPU传输/collective，
不保存全量features，不在GPU长期堆叠额外梯度族。
记录裁剪前后实际范数、统一裁剪系数、实际参数变化及其相对范数；
不能把梯度裁剪系数直接当作Adam参数更新系数。

原shared decoder注册在enc_dec_backbone下，因此独立统计该组，避免遗漏它。
视觉encoder、视觉adapter、pixel decoder、query decoder、panoptic controller、
memory adapter、GS anchor/activation及其余重建参数分别统计，互不重复计数。
观测回调默认关闭；主运行快照未包含回调。CPU数值测试验证开/关观测后的模型、
AdamW state、原日志和Torch RNG逐位一致。真实四卡每组重放301一步预检才提交长重放。
原CUDA运行可能存在数值差异，报告逐步loss/norm漂移，并核实441历史尖峰是否重现；
不能因为新重放也测出了某一范数，就假定解释了原799尖峰。

证据root：`/space/mawb/ssst/group_plus/vggt_joint_followups_v1`。
预检/正式/重试分开目录，旧失败日志保留。三组扩大评测与一个三组顺序重放均四卡，
在14/17/18的4090资源等待，自动对照汇总零GPU。正式输出结果后由用户通知再分析，
无需一直交互轮询或等待主训练4epoch结束。

## 已验证预检与正式提交

后续执行SHA d2dc0ced0bacd8e400b4d8ed1503f3bf8806804b，已推送并使用不可变快照。
13项CPU合同通过，观测开/关保持原模型更新、AdamW矩、RNG与原日志逐位一致。
四卡扩大评测预检64045 COMPLETED0:0，3分17秒；三个模型的原导出逐文件重现，
full与confirmation两套实际官方CPU/GPU合并合同通过。
四卡梯度预检64046 COMPLETED0:0，10分12秒；三组各重放真实301一步。
A/B/C原/重放总范数分别完全一致：2.04890108、1.29637349、4.60892439。
全局R/U重组误差norm分别1.33e-7、8.98e-8、2.61e-7，冻结检查通过。

| 正式任务 | 作业 | 资源 | 提交后已核实状态 |
|---|---:|---|---|
| A已有500步全量复评 |64047|4×4090|17号RUNNING，四rank均实际导出|
| B已有500步全量复评 |64048|4×4090|17号RUNNING，四rank均实际导出|
| C已有500步全量复评 |64049|4×4090|18号RUNNING，四rank均实际导出|
| ABC自动对照汇总 |64050|0GPU|等待afterok:64047:64048:64049|
| 三组顺序梯度重放与自动分析 |64051|4×4090|Resources等待；已有相同代码四卡预检|

三组复评十二rank都已正常处理100个以上真实窗口，不是仅提交未启动。
主训练64041早已确认20/160并停止轮询，64042仍为训练完成后的自动官方评测。
所有GPU任务只要求四卡；停止轮询，不等待本轮任务结束。

证据：root下preflight_validation.json、dataset_partition_validation.json（1191训练/312验证无重叠）、
formal_eval_startup_validation.json、launch_summary.json、POLLING_STOPPED.json、submissions/。
正式root：`attempts/d2dc0ced0bacd8e400b4d8ed1503f3bf8806804b/formal`。
扩大复评最终表格在expanded/analysis/full312_siu3r_table1.csv和confirmation248_siu3r_table1.csv；
梯度重放最终摘要在gradient/analysis.json和report.md。
结果尚未完成，当前只报告启动/预检事实，不宣称训练或指标改善。
