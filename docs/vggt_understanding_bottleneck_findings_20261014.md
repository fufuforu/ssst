# 理解瓶颈第一阶段与实际尖峰：2026-10-14

已完成独立248场景保存输出分析、20场景原始query/组装诊断，以及A/B/C实际尖峰捕获。初步候选是query的mask、对象存在性/类别与输出置信度之间的对齐，以及理解路径的大梯度；不立即重复旧三尺度融合，不修改主训练、clip、loss或encoder。

20场景样例只有31个GT实例-窗口条目，不能据此给完整数据归因。固定随机128场景的补充冻结前向与原生Hungarian观察也已完成，扩大确认支持query对象存在性/类别与mask质量对齐这一诊断方向；仍未证明具体模块的因果根因。

## 已完成任务

| 内容 | 作业 | 状态 | 用时 |
|---|---:|---|---|
|原始mask三组真实预检|64084|COMPLETED0:0|3分08秒|
|补充query身份/概率后的预检|64086|COMPLETED0:0|2分15秒|
|实际范数触发三组预检|64085|COMPLETED0:0|10分44秒|
|20场景×三组冻结前向|64087|COMPLETED0:0|8分07秒|
|原始mask自动汇总|64091|COMPLETED0:0|2秒|
|A实际尖峰捕获|64088|COMPLETED0:0|13分43秒|
|B实际尖峰捕获|64089|COMPLETED0:0|17分33秒|
|C实际尖峰捕获|64090|COMPLETED0:0|7分10秒|
|实际尖峰自动汇总|64092|COMPLETED0:0|3秒|
|128场景原生匹配预检|64094|COMPLETED0:0|1分45秒|
|128场景×三组冻结前向|64095|COMPLETED0:0|8分32秒|
|128场景自动汇总|64096|COMPLETED0:0|1秒|

GPU原始诊断和尖峰正式SHA：202e0b95fa24a619d34193686d58cf58ba9243fc。
原预检SHA：775adf59493929b3f58c01aa7be23cd978942467 / dbd9659d72bcb3bdc681b5ccc200293ec3ad1d8d。
自动汇总使用独立SHA dd3dd40fedbcec3471fa1c72cc1f9a2e08331d0d，未改正在执行的GPU快照。
128场景及其CPU汇总使用913380eb26b6a81e11f0e8ff375e9ba010ef5a22。

主训练64041/依赖官方评测64042仍按原配置执行。本轮不轮询其训练指标，不触动其990a5d4快照或输出。
A/B/C均为同一重建适配2epoch起点的已有500步权重：adapter理解梯度0.01、VGGT冻结。
A原视觉训练/β0.1，B冻结MASt3R、ViT adapter、pixel decoder/β0.1，C原视觉训练/β0.05。

## 1. 保存输出：单纯修正已保留mask的类别不足以填补差距

原生COCO RLE、类别、分数均直接来自confirmation248的保存state。
A/B/C两个视图全部GT逐条相同；target包含2输入+4额外帧，沿用官方口径。
1478窗口共有5950个GT实例-窗口条目；不同窗口会重复同一物体，不是5950个唯一物理实例。

| 对照 | 同类别best-mask recall@0.5 | 类别无关best-mask recall@0.5 | 同类别recall@0.75 | 预测数 | 一对一匹配数 | 诊断FP |
|---|---:|---:|---:|---:|---:|---:|
|A|49.48%|54.81%|30.10%|5868|2944|2924|
|B|49.06%|54.24%|29.80%|5806|2919|2887|
|C|49.19%|54.32%|29.80%|6057|2927|3130|

best-mask recall逐GT寻找最佳mask；一对一匹配按分数排序、同类IoU>=0.5。
这些是诊断，不是SIU3R mAP/PQ的替代指标。
C未匹配的3023条中，44.59%的最佳保留mask IoU<0.1，45.32%介于0.1～0.5，10.09%有较好mask但类别不符。
C比A增加189个预测，匹配反而少17个，诊断FP多206，precision下降约1.85个百分点。
C的大/小/极小对象同类recall为61.85%/19.67%/1.84%；尺寸按多帧连接mask面积占比>=1%、0.1～1%、<0.1%，不等于3D尺寸。

上述结论只针对最终已保留mask。它不排除好mask在早期资格筛选中丢失，也不能只凭最终漏检认定encoder没有提取到对象。
原生PQ会忽略预测void覆盖超过一半的一些GT，故本次漏检统一用原生AP GT分母，不以各组PQ的TP+FN比较GT总数。

## 2. 固定mask与类别：置信度排序仍有损失

| 对照 | 原生mAP | GT同类IoU重排序诊断mAP | 诊断差值 |
|---|---:|---:|---:|
|A|0.237424|0.327173|+8.975个百分点|
|B|0.235403|0.320517|+8.511个百分点|
|C|0.236063|0.322882|+8.682个百分点|

只在离线评分输入中，将预测分数换成对同类GT的最佳IoU；mask、类别、GT、原生评分实现均不变。
这是GT依赖的排序诊断，不是模型成绩、可部署方法或保证可达到的提升。
baseline mAP/AP50/AP75独立重算与原保存结果匹配至1e-12。
原输出分数是类别概率乘获胜像素的平均membership，并不直接预测实例mask IoU。
因此应检查query置信度与mask质量的一致性，不能只靠继续调β解释实例指标差距。

## 3. 原始query与实例组装：存在资格阶段的损失，尚不能普遍归因

20个故障/拟合分层窗口来自独立248场景，三组共60次冻结前向，零优化器调用。
全部输入RGB哈希和最终原生panoptic PNG逐帧重现；GT仅在context-only generate之后用于诊断。

| 对照 | GT条目 | 原始100query mask类别无关recall@0.5 | 资格筛选后 | 像素竞争后 | 面积筛选后 |
|---|---:|---:|---:|---:|---:|
|A|31|80.65%|51.61%|51.61%|51.61%|
|B|31|83.87%|58.06%|51.61%|51.61%|
|C|31|83.87%|54.84%|48.39%|48.39%|

C有26条能在原始mask中找到IoU>=0.5的匹配，资格后17条，竞争后15条，面积过滤没有进一步损失。
资格阶段丢失9条中，最佳原始query有4条仅被no-object判空，5条同时no-object和thing概率低于0.05。
输入Fm×mask_embed(q_pre)类别无关代理：A/B26/31、C27/31；它不是原生预训练分割结果或真实模型上限。

重要限制：原生set loss只对匹配query施加thing mask损失，未匹配query仍可能偶然产生好mask，且应当被no-object抑制。
因此“原始100query中有好mask”不等于“筛选规则一定错误”，更不能直接取消筛选。
现有大量FP也不支持盲目放宽阈值。需要原生Hungarian观察，确认好mask是否来自应当负责该GT的query。
本20样例只有31条GT且按故障选择，不代表完整验证集；当前不因此宣布对象存在性头是唯一根因。

## 4. 稳定性：三个实际捕获尖峰均由理解梯度主导

恢复各自300步模型、AdamW与四rank RNG，原采样/global batch8/GC/LR/clip1保持。
每步暂存原本已计算的局部R/U向量，原生全局安装梯度后以只读norm>=20触发完整分模块SUM/4观测。
达到首次实际尖峰后停止，最多141次调用。实际A58、B114、C7次，共179次/1432重放曝光。

| 对照 | 捕获步 | 实际总范数 | 全局重建R范数 | GC缩放后的理解U范数 | clip系数 |
|---|---:|---:|---:|---:|---:|
|A|358|476.899|2.502|476.886|0.002097|
|B|414|114.520|2.708|114.483|0.008732|
|C|307|232.118|2.427|232.088|0.004308|

A的范数平方份额约query decoder42.9%、MASt3R29.9%、pixel decoder16.0%；
B约query decoder73.8%、panoptic controller26.0%；
C约MASt3R36.4%、query decoder32.4%、pixel decoder21.7%。
这些是梯度范数贡献，不是网络因果重要性的评分。
B冻结视觉后仍有理解侧大梯度，故冻结视觉不能保证消除此类尖峰；弱feedback也未消除它们。

全局R/U重组、clip后范数、原采样/LR与冻结状态核验通过；保存了真实Adam参数变化。
全局clip会缩放同一步的R和U，但clip系数不能当作最终Adam参数更新倍率。
普通步adapter缩放后的U小，不代表尖峰步也小；当前证据支持继续细分理解损失的梯度来源。
尚未区分classification、thing BCE/Dice、stuff BCE/Dice，不能指定其中一个为根因，也没有修改任何项或clip。

这些是实际重放事件，不是历史第441步799的直接归因。首个实际尖峰也不构成三组尖峰频率/稳定性的排名。
保存实际批次和更新前RNG，但未写大量事件model/optimizer checkpoint；该RNG文件不能单独恢复事件模型。
CPU合同证明观察开启/关闭保持原生更新、AdamW、日志和RNG逐位相同；CUDA重放允许并披露数值漂移。
预检C301范数与旧记录差4.768e-7、R/U loss最大差7.45e-9。首次提交器过严的逐位相等判定拒绝了正式提交，未产生新作业；
修正为1e-6相对范数容差，数值执行代码不变，失败收据与原预检保留。

## 5. 已完成的128场景确认与下一步决定

64095已COMPLETED0:0：四卡4090、128个固定随机场景，每场景一个窗口，从独立248池用seed4242选择，不依赖预测或评分挑选。
输入GT条目492，覆盖远大于故障20样例。A/B/C同一128窗口冻结前向，零训练。
64096已COMPLETED0:0：afterok:64095，零GPU自动汇总。
预检64094三组各一个7GT窗口，旧PNG复现、原生软mask/class Hungarian字段通过。
补充记录匹配query、其hard-mask IoU、类别/no-object概率及阶段损失。
查询概率和GT匹配只用于生成后的分析，不进入generate、feedback、相机或训练。

扩大确认共有输入492、target507个GT实例-窗口条目。

| 对照 | target GT条目 | 原始类别无关recall@0.5 | 资格筛选后 | 像素竞争后 | 面积筛选后 |
|---|---:|---:|---:|---:|---:|
|A|507|66.47%|56.80%|55.82%|55.62%|
|B|507|67.65%|57.40%|55.42%|55.23%|
|C|507|66.67%|55.82%|54.83%|54.24%|

C原始338条有较好mask，资格后283条，竞争后278条，面积后275条。
资格阶段丢55条，其中最佳原始query14条仅no-object、41条no-object且thing概率<0.05。
这比故障小集的约29个百分点资格下降小，确认小集不能直接代表全量，但资格阶段的损失确实保留。
竞争阶段既有丢失也有恢复：C丢14、恢复9，净少5；面积净少3。不能用简单单调漏斗解释所有mask质量变化。

更关键的是原生context Hungarian：

| 对照 | GT及匹配query数 | 匹配query有IoU>=0.5 mask | 其中被预测no-object | 全部匹配query被预测no-object |
|---|---:|---:|---:|---:|
|A|492|323|37|121|
|B|492|328|40|117|
|C|492|321|44|110|

C的321个好mask匹配query中，44个（13.71%）仍被判为no-object；这些query是原生loss匹配选来负责GT的，不是未匹配query的偶然mask。
C全部492个匹配query中110个被判无对象（22.36%），原生GT类别argmax正确354个（71.95%）。
A/B同样存在该现象。它证明分类/对象存在性与mask质量之间有可测量失配，但不能区分训练不足、loss优化与query特征交互的因果作用。
C输入2D q_pre代理337/492（68.50%），不能据此宣布视觉encoder无问题或模型上限，只说明本采样下也存在可用的视觉mask支撑。

暂不重复U3：旧三尺度VGGT query前融合已试过且没有收益，不能当作新实验重提。
优先验证query的对象存在性/类别与实际mask支撑的一致性，并检查置信度排序。可考虑将query自己的mask/几何支撑特征读回query，再交给原分类头，保留encoder、原双向结构与loss；这是候选后续小预算对照，本次未实现或训练。
在增加新的融合之前先分解理解损失梯度，防止特征改动与优化异常相互混淆。不能仅凭这些诊断宣布整个decoder上限，也不能解释SIU3R全部13列差距。
这些诊断权重只有500步；主训练完整4epoch结果仍用于判断该问题在长期训练后是否保留。
稳定性后续可按现有五个理解损失项做梯度分解，保持原系数和训练数学；本次尚未启动此进一步实验。

## 证据

- [第一阶段CPU报告](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/CPU_PHASE_REPORT.md)
- [独立248实例诊断](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/cpu_review_20261013/results.json)
- [固定mask的排序诊断](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/cpu_review_20261013/ranking_oracle_diagnostic.json)
- [20样例阶段结果](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/attempts/202e0b95fa24a619d34193686d58cf58ba9243fc/formal/raw/analysis/report.md)
- [实际尖峰汇总](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/attempts/202e0b95fa24a619d34193686d58cf58ba9243fc/formal/spike/analysis/report.md)
- [梯度分模块原始汇总](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/attempts/202e0b95fa24a619d34193686d58cf58ba9243fc/formal/spike/analysis/results.json)
- [随机128确认结果](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/attempts/913380eb26b6a81e11f0e8ff375e9ba010ef5a22/representative128_formal/analysis/results.json)
- [随机128提交收据](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/submissions/representative128_formal_913380eb26b6a81e11f0e8ff375e9ba010ef5a22.json)
- [已有输出的错误分解图](/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/cpu_review_20261013/retained_instance_errors.png)

原生SIU3R commit仍为8ea80166be76854f938e90521f1a5b688b755c87，官方评分和此前13列表格不变。
