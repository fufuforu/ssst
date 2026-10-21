# C起点四组读取短对照：终点评测与分析

四组训练、128场景官方评测及自动汇总均完成。三种干预均未给出重建与理解的综合改善；目前保留baseline作为联合任务参照。
anchor_visual有RGB重建正向信号，但语义与实例收益不成立；object_local_read有深度/PQ取舍；reliable_depth在本实现和强度下不宜继续放大。
这次没有证明encoder特征不足，也没有证明读取方式是唯一主瓶颈。

证据root：`/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04`。
原始自动产物`analysis/report.md/results.json/siu3r_comparison.csv`保持原样；补充CPU分析放在`completion_review_20261015`。

## 完成状态与比较合同

| 阶段 | Slurm | 终态 | 用时 |
|---|---:|---|---|
| object_local_read补已有导出评分 |64851|COMPLETED 0:0|2:01|
| baseline续跑、导出、评分 |64852|COMPLETED 0:0|44:53|
| anchor_visual续跑、导出、评分 |64853|COMPLETED 0:0|50:34|
| reliable_depth续跑、导出、评分 |64854|COMPLETED 0:0|56:44|
| 四组自动汇总 |64855|COMPLETED 0:0|5:38|

object_local_read训练/导出来自64717，训练已完成后发生的旧评分失败保留；64851仅补评分，没有重新训练或导出。
baseline/anchor_visual/reliable_depth续跑快照为649ef09；object_local_read训练/导出为7468268；实际GPU预检/20更新启动为410bd30。
18项训练数学文件/受保护函数/正式更新循环合同仍与预检相同，没有通过重新预检或重新训练覆盖旧记录。

每组300更新、2400曝光，8×3090、microbatch1/global8。四组同C起点、同可训练参数集合233188331参数、新增参数0，原20步checkpoint精确续跑到300。
固定2400不同训练窗口覆盖1075训练场景；固定128个不同留出场景各一窗口，与训练场景无交集。不是128场景反复训练，也不是完整312场景验证。
完整VGGT和MASt3R image encoder冻结/eval，全部BN保持eval；原重建与V3完整理解loss、matcher及辅助监督不变。
GS/VGGT memory adapter理解梯度路由均0.01，全局理解权重仍1，feedback峰值0.05，geometry policy仍monitor_v1。本轮Q缺席。

每组8个导出rank各16窗口；官方SIU3R commit为8ea80166be76854f938e90521f1a5b688b755c87，src无修改。
每rank官方CPU/GPU状态合并合同PASS；mAP/PQ/mIoU按官方状态全局合并，不平均rank AP。
原自动汇总实际校验跨组GT文件hash及训练流；补充审计独立校验输入hash/frame IDs、原生官方AP GT的RLE/类别和逐帧重建分数身份。
官方新视图列仍对应2输入+4额外帧的target集合；深度仍为官方scale/shift对齐。两个mIoUt空白。

补充CPU审计完成：32份训练日志共9600条记录，逐rank更新1–300连续且每步仅对应固定stream指定窗口；loss/梯度范数/观测均finite，全部冻结参数/encoder及BN buffer检查通过。
每一步全局梯度范数在8个rank间完全一致；四组resume参数hash均逐rank匹配其第20步启动证明。
四个终点checkpoint的SHA与导出回执一致；参数/optimizer/buffers均finite，1078个AdamW状态的step均为300，8份RNG和8337维采样计数完整，逐rank采样恰为300个不同窗口。
各组2400曝光均有2372条OK、28条monitor_v1质量WARNING；告警保留且未跳过窗口。不要把续跑monitor摘要中的280窗口误认为总共只训练280更新。
终点没有独立记录全部rank的参数hash，不能声称已重新测量第300步八rank权重逐位一致；上述同步检查和原训练DDP合同构成当前证据。

可复现CPU分析命令（不导入主模型，不使用GPU）：

```bash
/space/mawb/anaconda3/envs/tokengs/bin/python scripts/analyze_vggt_reading_controls_results_v1.py \
  --root /space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04 \
  --output /tmp/vggt_reading_review_new_directory --checkpoints
```

最初尝试在官方评分环境PyTorch2.4.1读取当前大体积压缩checkpoint时，PyTorchStreamReader报invalid header；失败日志保留在`completion_review_initial_checkpoint_reader_failure_20261015.log`。
改用原模型/导出环境PyTorch2.7.0后四个checkpoint均读取、SHA和内容审计成功，没有修改checkpoint、共享官方环境或训练代码。
这次额外CPU审计的读取失败不属于训练/导出/官方评分失败，也不构成重训理由。

## 唯一正式指标表

以下理解值为0–1；不与旧312场景表直接作差。

| 分组 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入mIoUs↑ | 输入mAP↑ | 输入PQ↑ | 输入mIoUt↑ | 新视图mIoUs↑ | 新视图mAP↑ | 新视图PQ↑ | 新视图mIoUt↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|baseline|0.08868|0.2156|22.28|0.7747|0.3722|0.5569|0.3126|0.6280||0.5507|0.2786|0.6031||
|object_local_read|0.08657|0.2115|22.07|0.7725|0.3745|0.5515|0.2975|0.6321||0.5490|0.2625|0.6072||
|anchor_visual|0.08668|0.2117|22.45|0.7775|0.3726|0.5511|0.3028|0.6343||0.5442|0.2708|0.5982||
|reliable_depth|0.08716|0.2130|22.17|0.7738|0.3741|0.5324|0.2933|0.6095||0.5262|0.2653|0.5855||

相对baseline，理解差值以下均为百分点：

- **object_local_read**：AbsRel −0.002108、RMSE −0.004060；PSNR −0.214270dB、SSIM −0.002175、LPIPS +0.002306。输入/新视图mAP −1.5134/−1.6173，mIoUs −0.5396/−0.1679，PQ +0.4190/+0.4129。更局部读取没有转化成综合收益。
- **anchor_visual**：AbsRel −0.002000、RMSE −0.003845；PSNR +0.171797dB、SSIM +0.002829，但LPIPS +0.000334。输入/新视图mAP −0.9794/−0.7853，mIoUs −0.5830/−0.6483；输入PQ +0.6385，新视图PQ −0.4885。已有Fm证据可以影响并改善部分重建，不能据此宣布理解特征瓶颈解决。
- **reliable_depth**：AbsRel −0.001526、RMSE −0.002549；PSNR −0.106784dB、SSIM −0.000888、LPIPS +0.001869。输入/新视图mIoUs −2.4463/−2.4479，mAP −1.9276/−1.3315，PQ −1.8465/−1.7563。深度误差的小改善伴随明显理解退步，本次沿射线深度偏置不适合作为默认增强。

不通过数“改善几项”评胜负，也不因为anchor_visual的PSNR最高就将其替换为联合任务默认模型。

逐场景配对重采样5000次（seed20261015，每场景一个窗口）的PSNR差95%区间：

| 干预−baseline | PSNR差/dB | 配对场景95%区间 | PSNR改善场景数 |
|---|---:|---:|---:|
|object_local_read|−0.21427|[−0.33440,−0.09307]|46/128|
|anchor_visual|+0.17180|[+0.03346,+0.31192]|83/128|
|reliable_depth|−0.10678|[−0.19438,−0.01884]|47/128|

三组AbsRel/RMSE改善的场景配对区间也均低于0。anchor_visual的LPIPS差区间[−0.000770,+0.001388]跨0，不能将其微小LPIPS均值退步解读成确定的普遍恶化。
这些区间仅衡量当前已训练终点在128场景上的采样不确定性，没有多seed训练重复、没有多重比较校正，也未对官方mAP/PQ伪造逐场景平均或区间。

## 干预确实生效，不能将负结果解释为路径没运行

128个留出窗口真实object-query→anchor CA观测，完整投影O输出的平均norm和逐head归一化熵如下。
这里熵先对每个head计算，再平均head/query，不是先平均attention heads后算熵。

| 分组 | L6 O norm | L8 O norm | L10 O norm | L12 O norm | L12归一化熵 |
|---|---:|---:|---:|---:|---:|
|baseline|0.26770|1.16610|1.04007|0.30677|0.92513|
|object_local_read|0.31273|1.13387|1.05968|0.33890|0.90964|
|anchor_visual|0.23927|1.18228|1.01876|0.33611|0.92544|
|reliable_depth|0.28713|1.17721|0.99003|0.32874|0.92489|

每个head真实V norm均已保存，非零且finite。不能再用mean(R)@原anchor的代理宣称完整V/O路径失效。
局部读取L12熵由0.92513降至0.90964，说明读取分布更集中，但这没有带来mAP改善；低熵本身不是正确读取的证明。

末次调用的局部支撑附加bias在乘λ前abs mean为1.46664，λ0.5后约0.73332；不是新增路径未打开。
anchor_visual有至少一个有效投影视图的anchor比例平均99.966%，最低99.316%，不是大量越界导致视觉证据全为0。
这只证明几何投影有效，不证明插值局部特征与对象语义对齐。
reliable_depth末次几何bias乘λ前abs mean为3.21387，乘0.1后约0.32139；reliability平均0.52693。
这些scalar是每窗口末次调用观测，不是全12层均值；几何干预作用于GS anchor→VGGT patch CA，以上O/熵属于下游对象CA，不能混为同一处attention。

## 分类与mask仍未一致

以下使用当前四组同口径的context原始mask、最终训练Hungarian匹配query诊断。好mask阈值IoU≥0.5；每组492个匹配GT。
它既不是最终官方预测/GT匹配，也不是将query拒绝直接等同于唯一GT漏检。

| 分组 | 好匹配mask | 好mask但no-object | 占好mask比例 | 匹配query为no-object | 匹配分类正确 |
|---|---:|---:|---:|---:|---:|
|baseline|331|49|14.80%|127|337|
|object_local_read|332|53|15.96%|133|334|
|anchor_visual|332|53|15.96%|124|340|
|reliable_depth|326|50|15.34%|129|333|

object_local_read和anchor_visual的好匹配mask只比baseline多1个，而冲突数均多4个，没有解决好mask/no-object一致性问题。
anchor_visual匹配分类正确多3个，但官方mAP仍下降，表明这项匹配分类统计无法替代部署候选质量/类别/置信排序/最终mask综合评估。
四组mask和候选都已改变，因此不能把mAP变化全部归因于排序，也不能把PQ变化全部归因于更好的分类。
旧feature诊断的好mask定义和计数不同，不跨口径比较396/76等历史数字。

独立读取32份官方保存的AP状态，用原生多视图RLE做最终保留候选分析，无新推理。
以下是诊断候选数/召回，不能替代官方全局mAP/PQ；同类最佳IoU按每个GT独立找候选，不是新的官方匹配规则。

| 分组 | 输入保留预测数 | 输入同类最佳IoU≥.5/.75的GT数 | 新视图保留预测数 | 新视图同类最佳IoU≥.5/.75的GT数 |
|---|---:|---:|---:|---:|
|baseline|489|253/170|493|247/165|
|object_local_read|479|253/168|479|244/159|
|anchor_visual|503|251/163|506|248/159|
|reliable_depth|488|250/163|494|247/160|

输入/新视图的GT数分别492/507，后者仍是官方6帧集合，不能要求两者GT数相同。
局部读取的候选数减少，IoU≥.5输入召回持平，但高IoU候选覆盖略降；PQ升而AP降可以并存，不能解释成实例质量全面提升。
局部视觉的候选数增加，但高IoU同类覆盖从170/165降到163/159，单纯增加候选没有解决实例质量。
可靠深度的IoU≥.5召回相近而高IoU覆盖/语义下降，亦不支持“只差一个更好的置信排序”的单因素解释。

作为另一口径的raw-query诊断，context类别无关最佳raw IoU≥.5/.75的GT数为baseline354/233、局部读取357/230、局部视觉354/233、可靠深度351/233；
平均最佳raw IoU分别0.64226/0.63871/0.64366/0.64056，缺少明显raw-mask整体改善。
这支持继续关注已有候选到最终同类实例输出的质量保持、类别/no-object竞争和组装过程；raw与官方最终候选计数也包含类别/重叠/面积处理差别，不能简单相减后把差额全归给某一个过滤器。

## 对瓶颈定位和交互层建议的影响

本轮将“局部对象读取”“已有Fm进入anchor”“可靠深度进入anchor→VGGT读取”分开检验，但三种机制仍会经训练/反馈改变后续表征和最终预测。
这不是冻结同一表征只换读取的静态因果分解，不能把全部终点差异指派给某个CA子步骤。

更有依据的更新是：

1. 当前简单的局部读取soft bias不是已经验证有效的理解改进，之前代理池化probe的优势没有自动转化为正式mAP收益。
2. 现有局部视觉信息并非完全没有用，anchor_visual的部分重建正向信号值得保留，但它没有解决理解性能不足。
3. confidence与深度可靠性相关，不等于由其构造的沿射线bias有联合任务收益；本次结果正说明两者必须区分。
4. 仍无法二选一断定“特征不丰富”或“query/anchor读取不足”为唯一主因。已有特征、对象竞争、分类/no-object和最终候选质量之间存在多个环节。

交互层位置仍是未测假设。这轮固定object-query/anchor双向交互在6/8/10/12，不能据此宣布这些层错误。
另外，GS anchor读取VGGT patch的重建CA原本就在每个decoder block执行；“改成全部层”需要明确是将对象query更新/feedback扩展到每层，而非补上原本缺失的VGGT读取。

若下一阶段优先研究交互时机，可讨论当前6/8/10/12与前移但仍4次的3/6/9/12短对照，先保持交互次数及模块数量一致。
但前移也改变query/对象几何初始化及scene normalization时刻，已有第6层权重迁到第3层也有分布变化，不能称为完全无其他变化的纯层号开关。
在时机对照有依据后，再讨论后半段6–12逐层交互；直接1–12全开同时改起始时机、次数、累计反馈和可能的独立模块容量，解释更困难。
需预先定义模块权重复用/初始化、起点复现、反馈累计预算和匹配训练预算，不能直接把新增层当作已有成熟层。

没有自动叠加三种干预、加Q、调整交互层或提交新的长期/GPU训练。300步单次训练、128场景结果只用于这次预算和实现；不能宣称长期上限，也不能由负结果证明必须新增encoder。
