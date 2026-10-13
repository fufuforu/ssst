# 理解瓶颈分析与实际尖峰捕获 v1

用户授权：先做理解瓶颈分析，再决定融合对照；尖峰诊断作为稳定性任务单独处理。
不改变64041主训练与64042依赖评测，不修改已有不可变快照、模型、loss、clip或资产。
核对历史后确认U3已试过query前三尺度VGGT残差融合，500步未显示收益；本轮不重复该训练。

## 1. 已有输出的独立248场景诊断

输入为64047～64049保存的原生official_states.pt中confirmation248的COCO RLE、类别及分数。
同一GT实例跨一个窗口的全部帧连接，保留原生ID；不同窗口重复出现的物体不算唯一物理实例。
A/B/C、两个原生视图全部GT RLE和类别逐条一致。1478窗口，输入5854、target5950个实例-窗口条目。
输出best-mask recall、同类best-mask recall、分数顺序一对一precision/recall，IoU阈值0.5/0.75。
这些是错误分析，不是新官方AP/PQ，也不替换13列表格。AP GT不使用PQ的预测void过滤。

已完成CPU结果：target同类IoU>=0.5 recall A49.48%、B49.06%、C49.19%；忽略类别后
为54.81%、54.24%、54.32%。因此只修正类别最多补回约5个百分点，仍有约46%对象找不到
IoU>=0.5的已保留mask。C比A多189个预测、少17个匹配，诊断FP增加206；不是重复框NMS问题
的证据，因为最终实例mask本来互斥，同一GT的多份高IoU匹配在该表示下受限。
C的大对象同类recall61.85%，小对象19.67%，极小对象1.84%。大小按窗口连接mask像素占比
>=1%、0.1～1%、<0.1%定义，不等于COCO尺寸或3D尺寸。

## 2. 冻结权重的原始query与实例组装检查

从C的四类失效和较好拟合窗口各选4个，共20场景/窗口，全部属于独立248。
这是刻意选择的诊断集，不代表随机验证集，不能用其分数宣布整体改善。
A/B/C各做同一20窗口的冻结前向，零优化器调用、生成只读context。
分解同一原生输出的四个阶段：

1. 全100个query的原始阈值mask，使用原alpha阈值。
2. 加原objectness/类别概率资格条件。
3. 原像素竞争之后、面积保留规则之前。
4. 原面积规则之后的最终导出mask。

保持原组装数学不变，逐帧与已有官方panoptic PNG核对。按GT保存每阶段best IoU，
判断资格过滤、像素竞争与面积过滤是否丢掉已有实例证据。
另在输入帧计算Fm×mask_embed(q_pre)的2D mask代理，只有类别无关含义，
不是预训练头的官方结果，也不能直接当作2D模型上限。
不永久保存features、100通道高分辨率mask或新增RGB/depth导出，避免挤占主训练checkpoint空间。
真实四卡smoke三组各1窗口，要求checkpoint/输入哈希及全部原生PNG重现，才正式跑20窗口。

判断后再选择下一步：如果好mask主要在过滤/竞争中丢失，先研究实例组装；
如果输入2D代理较好而GS原始mask明显差，优先检查feature lifting、query–GS绑定；
如果输入2D代理本身不足，继续检查视觉特征、query表示与mask特征入口。
这些比较只能缩小候选位置；不能只凭attention或代理mask证明因果根因。
暂不提交融合训练，避免重复已失败U3或同时改动多个模块。

## 3. 稳定性：实际范数触发的独立短重放

每组恢复自己的update0300模型、AdamW与四rank RNG，原数据流/global batch8/GC/LR/clip1不变。
每步在CPU暂存原生已有R/U梯度，实际安装全局梯度后用只读范数触发；
仅首次参考301及实际norm>=20时做分模块全局SUM/4梯度统计。
所有rank先核对触发一致，再进入可选collective，避免分布式分歧。
首次实际尖峰捕获后停止；最大141调用至441，若无尖峰明确记录NO_SPIKE。
阈值20只决定诊断采集，不决定是否更新、跳过、改变loss或中断主训练。
记录R/U/缩放U/余弦、重组误差、实际clip后范数和Adam参数变化。
保存实际批次IDs与更新前RNG；不写大量事件model/optimizer checkpoint，
因此RNG文件不是可独立恢复的事件checkpoint，不能声称未来可逐位重现该事件。

CPU合同已验证触发和未触发两种情况下，原生更新、AdamW矩、原日志及RNG逐位不变。
真实四卡smoke分别核对A/B/C第301步与原记录一致，再提交三个独立四卡任务。
捕获的是重放实际尖峰，不自动等同原第441步799；保留重放漂移信息。
只允许17/18号4090，排队不占主训练13号3090，不等待八卡。
任务使用提交推送后的固定SHA快照，preflight/formal及工程重试独立目录。

证据根目录：/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1。
CPU产物：cpu_review_20261013/results.json、objects.csv、detections.csv、windows.csv、raw_readout_cohort.json。
