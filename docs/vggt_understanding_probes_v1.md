# 完整训练池与独立理解：四组短对照

用户于 2026-10-13 授权启动。最终联合结构保留原双向：object query 在
6/8/10/12 层读取 anchor token，并通过原点积 feedback 写回 anchor 特征。
本轮先区分训练数据重复、固定特征上的理解学习能力和 query 的几何读取。
没有沿用旧四组训练后的参数，没有重新训练重建适配阶段。

## 共同保留

- 从原完整数据重建适配 epoch2 的确切边界初始化；SHA 为
  `a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c`。
  VGGT adapted asset 为 `c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace`。
- 保留 VGGT 与 MASt3R 架构、旧 GS decoder 权重、预训练视觉/理解权重、原查询数量、
  任务头、理解损失项和权重、GS 分割渲染/feature lifting、预测相机与
  `shared_context_depth_sim3_v2 / monitor_v1`，不输入 GT 几何生成预测。
- VGGT 整体冻结（DINO/AA/camera/depth）；聚合器 BF16，原 camera/depth/下游 FP32。
- 全局 batch8、四张 RTX4090，每卡两次 microbatch；500 次更新 /4000 窗口曝光。
  原 GPU AdamW、beta(.9,.95)、eps1e−8、原组学习率/decay、clip1，原200步 LR warmup
  和4172步 cosine 截取前500步；理解权重按新增曝光 min(exposure/200,1)。
- 完整训练池为1191场景/8337唯一窗口（原完整轮为8344次曝光，含7个padding）。
  本轮固定 seed42 permutation 的前4000唯一窗口，覆盖1177场景，无重复，
  四组使用相同窗口与顺序。没有宣称500步完整遍历全部训练数据。
- 验证仍为旧四组同一64场景/64六帧窗口，在0/100/300/500步评测全部13列，
  两处文本 mIoUt 空白。AP/PQ每个条件全局计算，不平均逐场景状态。

## 四组逐项差异

| 项目 | J：joint_full | U1：u_read | U2：u_no_anchor_read | U3：u_fusion |
|---|---|---|---|---|
| VGGT | 冻结 | 冻结 | 冻结 | 冻结 |
| VGGT memory adapter /GS decoder/activation | 原联合训练 | 冻结、eval | 同U1 | 同U1 |
| MASt3R encoder/ViT adapter/pixel decoder | 原联合训练范围 | 冻结、eval，固定视觉特征 | 同U1 | 同U1 |
| 预训练 query decoder/mask embedder | 原联合训练 | 训练 | 训练 | 训练 |
| panoptic读取/语义实例任务头 | 原联合训练 | 训练 | 训练，失活的anchor attention冻结 | 训练 |
| 6/8/10/12 anchor特征跨注意力残差 | 保留 | 保留 | 置零 | 保留 |
| query→anchor写回 | 原点积/原warmup保留 | 零，W_inject冻结 | 同U1 | 同U1 |
| query前新增VGGT融合 | 无 | 无 | 无 | 有，初始零残差 |
| 重建损失优化 | 保留 | 只计算监测，不反向 | 同U1 | 同U1 |
| 理解损失优化 | 原联合 | 原理解损失/权重 | 同U1 | 同U1 |

J 相对旧 `bidirectional` 只改变训练窗口流，从固定128场景循环换成完整训练池广泛抽样；
数学、初始化、原模块训练范围和500步预算一致。adapter和GS依旧接收
gR+0.01gU，理解与panoptic接收原gR+gU。

U1–U3固定两侧主干特征与重建，只让已有query decoder/mask embedder与panoptic
读取/分类/分割头学习原理解目标。冻结pixel decoder是为了固定送入query的视觉特征。
没有从零重置预训练头。重建损失仍按原代码计算监测，requires_grad必须为False；
优化仅使用原warmup后的理解梯度，不制造假的零重建损失或伪梯度。
原参数家族的AdamW组学习率保留：理解1e−5、panoptic1e−4。

冻结GS权重不能阻止变化的query通过写回改变GS前向。因此仅在U诊断中关闭写回，
并检查同一真实窗口在训练前后Gaussian张量SHA完全相同；官方评测汇总再验证
U各步RGB与原共同起点逐图SHA相同。GS正向几何、相机、分割支撑均为预测值，未换GT。
有效的读取/融合方案随后才接回原联合双向图。

U2只移除四个注册层的anchor attention **特征残差 ev**。保留原attention
权重及query几何状态更新、anchor初始化、图像读取、feature lifting、Gaussian
child features与分割支撑；它不是“完全没有几何”的视觉模型。
这组回答四层anchor特征读取对理解的局部贡献，不能外推为所有几何信息的贡献。

U3从冻结的VGGT memory adapter取四层混合后的1024维memory（同GS所用、2738个patch），
按两视图分别恢复37×37网格。共享LN/1024→256投影，逐视图双线性对齐到
query decoder的三层视觉特征；视觉LN后与几何拼接，经512→256→256 GELU残差融合。
最后投影初始化为零，初始三层特征和U1逐元素完全一致。没有混合视图、加入target
特征或改变相机。融合只作用于query的三尺度输入，原mask_features/Fm及lifting路径保持不变。
原anchor读取仍保留。新增模块在isolated seed314159中初始化，不推进原训练RNG。

## 能与不能回答的问题

- J与旧原双向同500步对比，检查减少小子集重复是否缓解验证退步；该干预同时
  改变场景/窗口多样性与重复频率，不进一步区分它们。
- U1证明固定特征/重建下理解能否提升。与J差异还包括主干冻结、无重建优化、
  无动态写回，故不能只凭U1胜出就认定某一种梯度冲突。
- U2对U1是anchor特征读取残差的消融。
- U3对U1是几何融合连同新增容量的筛选；若有收益，下一阶段补同参数量视觉
  融合对照及特征/空间对应破坏，确认是几何作用。单纯同时排列K/V不是有效对应破坏。
- 增加原边界、beta0.1的只读评测干预 `joint_full/initial_feedback_full`，
  与原生beta0起点及训练后饱和beta对比。它是未训练权重的显式前向干预，
  不是另外一个训练checkpoint，不改变训练调度；共17个正式评测条件。
- 保存前20步及100/300/500步的分模块、分任务梯度范数/余弦，来自已有autograd
  结果，只记录不影响gc、分布式平均、clip或optimizer。统计为rank/microbatch局部值，
  不能直接当全局合成梯度范数。
  当前按实际named_parameters前缀归组；anchor_decoder前缀不含先注册在
  enc_dec_backbone下的共享decoder block，不能把它解释为全部GS参数的范数。
  独立理解保留原warmup，首步理解权重0且重建不反向，因此首个optimizer调用无参数更新；
  20步保存状态中实际活跃参数Adam时钟为19，明确不把它记为20次非零参数变化。

## 工程检查与提交方式

CPU单元检查：零初始融合、逐视图对应、学习后几何梯度、不重复4000流、
只理解训练的原AdamW更新；这些明确不是实际模型/GPU证明。
实际单卡与四卡各对四组训练两次更新、原预检窗口（四卡含6923），验证有限loss/grad、
原参数身份、冻结参数/缓冲、跨rank权重一致、真实参数更新、U固定Gaussian输出，
并各完成一次官方真实窗口13列评分。所有预检成功后才提交正式作业。
预检与正式使用相同推送后的不可变SHA，失败创建新attempt，保留原错误。
默认四卡作业可等14/17/18的4090，训练J240G，U160G，自动评测四卡160G。
pin实际已验证的`gsplat_cuda_partial_1406/gsplat_cuda.so`（SHA
`82403af8b36fb4b64774afc7462ea6611896fcfe7a02785098bce9ff50d8d751`）。

证据：`/space/mawb/ssst/group_plus/vggt_understanding_probes_v1/attempts/attempt01`。
训练：`/space/mawb/ssst/workspace_group_plus/vggt_understanding_probes_v1/attempt01`。
启动确认20正式更新/160新增曝光后停止交互轮询；已预检的资源待分配组自行写
20/160确认。依赖评测与最终CPU状态审计无需用户再提交。
现有完整数据adapter0.01训练63552及评测63553不修改、不重提。

## 已执行的旧四组清理

按用户授权删除旧attempt02的19个冗余checkpoint，共105155906729字节（97.93GiB）。
保留旧原双向update500 checkpoint、原适配边界/ adapted VGGT、所有收据/config、
完整官方指标/PNG导出、曲线、日志、历史失败与不可变代码。清理前后保护文件
fingerprint一致，旧63630–63638/63647均COMPLETED0:0，活动63552/63553排除在外。
被删除的中间/非基线最终权重不再能直接恢复，历史评分依然有效。

清理脚本：[cleanup_completed_interaction_controls_v1.py](../scripts/cleanup_completed_interaction_controls_v1.py)。
清单与删除日志：
`/space/mawb/ssst/group_plus/vggt_interaction_controls_v1/attempts/attempt02/checkpoint_cleanup_20261013`。

进度查询：

```bash
/usr/bin/python3 scripts/show_vggt_understanding_probe_progress_v1.py
```

## 真实预检与正式提交

执行SHA：`d8361c6cf656094c197cc07cf56a04ba7f61caa4`。
单卡63651、四卡63667：四组均真实训练两次调用并完成原官方六帧窗口13列评分。
四卡证明四rank参数同步、原边界参数共同hash一致、冻结参数/缓冲未改、
U三组训练前后Gaussian SHA相同，融合输出层发生非零参数更新。
四卡GPU峰值原联合约19.42GiB，独立理解约6.87–6.95GiB。
CPU单元检查通过；native历史单卡对比初始化/数据相同，loss最大差7.45e−8、
preclip norm最大差3.18e−5，理解官方分数完全相同；没有声称跨GPU运行逐位重现。
历史比较原始差异与复核说明均保留在attempt01。

| 组 | 训练 | 自动官方评测 | GPU |
|---|---:|---:|---|
| joint_full |63694|63695|4 RTX4090|
| u_read |63696|63697|4 RTX4090|
| u_no_anchor_read |63698|63699|4 RTX4090|
| u_fusion |63700|63701|4 RTX4090|

汇总63702依赖四组评测；所有训练/评测使用同一不可变执行快照。
本段与进度显示名称的后续更新不修改该快照或训练数学。

独立CPU审计63707依赖四组训练完成，自动核验全部保存的20/160状态。
当前J在node17、U1在node18运行；两组20步/160曝光收据与可恢复checkpoint
已通过独立CPU核验（实际Adam时钟J为19/20、U1为19），采样、四rank RNG、
组学习率、冻结边界和共同参数身份通过。U2/U3仍等待四卡资源，已有同SHA真实预检，
分配后自动确认20/160。交互轮询已停止，不等待排队结束或500步完成。
证据：`attempt01/startup_validation_running_arms.json`、`POLLING_STOPPED.json`。

## 用户授权融合组转3090

用户随后询问63700是否可使用空闲3090。4090释放后被其他作业占用，63700仍未启动，
Slurm当时预测19:13开始。先hold63700避免重复启动，准备将融合组单独转node11四张3090。
原三组及63552/63553不修改。新增GPU/hostname许可只允许u_fusion；模型、数据、
初始化、BF16/FP32、冻结、loss/GC、AdamW、global batch8和所有训练调度源码保持不变。
启动前比对不可变原SHA的数值源码，并在3090真实四卡训练两次调用及官方窗口评分。
预检通过后才取消旧pending63700及其pending评测/汇总/审计，替换依赖链。
原formal_submission.json和所有旧作业记录保留，替换记录另存migration_3090.json。
正式融合评测仍申请四张4090，以保持四组推理/渲染硬件一致；汇总/审计明确保存
各组执行SHA。融合训练显卡不同，不能把微小差异全部归因于融合结构。

3090真实四卡预检63735已COMPLETED0:0，峰值每卡约6.95GiB；原共同参数hash、
训练cfg、16次真实曝光、四rank同步、冻结参数/缓冲、前后固定Gaussian与新融合
输出层真实更新和官方13列评分通过。执行SHA为
`5f342defdbc79c98c1337ab6f7d58718086f62fa`。对原数值源码的实际比较通过；
跨GPU有BF16运行差异，没有声称两种显卡逐位重现。
旧63700/63701/63702/63707均在PENDING状态取消；替代训练63740在node11四张3090运行，
正式4090评测63741、汇总63742、CPU审计63743已提交。已完成依赖通过sacct验证后
不附加可能被controller清除的afterok引用，仍在运行的依赖继续挂接。
新的独立CPU审计确认全部四组已保存20/160、原Adam/RNG/采样/冻结契约通过，
融合组实际活跃Adam时钟19。停止轮询，不等融合500步完成。
证据：migration_3090.json、startup_validation_after_migration.json、
POLLING_STOPPED_AFTER_MIGRATION.json；原提交与旧启动记录保留。

## 2026-10-13 最终结果

四组500步及全部官方评测、汇总、CPU审计已成功完成。用户要求的复核已读取
保存评分、四rank日志和SHA收据，未新增训练或修改原快照。
完整池联合仍回落；固定特征/重建并保留anchor读取的理解对照有语义收益，
当前几何融合无增益。额外只读干预显示开启feedback本身可降低PSNR，不能把
相对beta0起点的净差全部解释为过拟合或梯度冲突。
完整13列、局限和建议见
[结果复核](vggt_understanding_probes_v1_results_20261013.md)。
