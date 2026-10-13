# 固定权重β扫描与视觉特征／写回强度四组对照

用户已授权两阶段，2026-10-13执行。保留已有训练数学与encoder/任务头/loss；
只在本诊断中改变feedback幅度，以及MASt3R encoder/ViT adapter/pixel decoder是否冻结。
原VGGT、memory adapter、GS与query架构不变，不创建新的反馈参数或融合模块。

第一阶段固定原adapt2共同起点和J500参数，各评测β=0/0.025/0.05/0.1。
复用原起点β0/β0.1和J500β0.1，共新增5个条件。推理4090、原64场景六帧窗口，
官方SIU3R13列，两处文本mIoUt空白；不进行优化器调用。使用原capped反馈乘β/0.1，
不是通过修改参数模拟。实际GPU先重现已有同权重β0.1端点导出，再继续其余条件。
beta_selection.json保存完整多指标比较及选择理由，β0用于诊断，不据此直接删除写回。
小β候选为0.025/0.05，不仅按PSNR选方案；若无总体增益，明确选为机制探针而非胜者。

第二阶段的完整2×2：

| 条件 | 视觉特征模块 | feedback上限 |
|---|---|---|
| joint_full | 原联合训练范围 |0.1，复用原63694/63695|
| visual_fixed | 冻结MASt3R/ViT adapter/pixel decoder |0.1|
| feedback_small | 原联合训练范围 |第一阶段选定小β|
| visual_fixed_feedback_small | 冻结同上 |同一小β|

所有新组VGGT整体冻结，但memory adapter、GS、query decoder和panoptic继续原联合优化。
原anchor读取与点积写回、原损失、GC、AdamW/decay/clip1、200步LRwarmup及4172步cosine
截断到500保持。理解梯度系数adapter/GS均0.01；原理解损失外部曝光warmup保持。
小β采用原1000新增曝光warmup，只改变上限，实际feedback幅度独立记录，避免混淆原名义β。
从同一adapt2边界与四rank RNG启动，全局batch8、四卡、每卡两微批，500更新/4000曝光，
相同seed42完整8337窗口池的不重复4000流。仍在0/100/300/500保存并评测同64验证场景。
诊断验证集参与选β，后续必须在更广验证集确认，不宣称论文完整验证成绩或结构上限。

预检：CPU原生反馈等价/缩放反向/冻结与RNG合同；实际单卡和四卡各两更新，
含原6923窗口，有限loss/梯度/参数、冻结与四rank同步，以及真实官方13列评分。
只有全部通过才提交3个新正式训练及依赖评测、汇总；不重复原J训练。
20正式更新/160新增曝光保存参数/optimizer/RNG/采样状态，独立CPU核验后停止交互轮询。
运行计算节点使用已推送固定SHA，历史证据与正在运行63552/63553均保留。

证据root：`/space/mawb/ssst/group_plus/vggt_feedback_factorial_v1/attempts/attempt01`。
训练root：`/space/mawb/ssst/workspace_group_plus/vggt_feedback_factorial_v1/attempt01`。
