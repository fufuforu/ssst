# 分类读出因果对照：固定完整adapter0.01/feedback0.05核心

用户2026-10-14授权下一阶段小预算验证。目的为在已有完整权重上区分分类头适配不足、对象专属mask支撑不足、以及新增读出容量/场景上下文的普遍收益；不声称找到唯一错误模块。

共同起点：完整适配2epoch+联合4epoch的4172步联合终点，checkpoint SHA ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0。沿用原生strict official loader、适配VGGT资产、0.01 adapter系数及0.05 feedback，所有几何数学和monitor_v1不变。旧对照未重复；旧U3三尺度query前融合与本次分类末端mask支撑读回不同。

| 对照 | 训练参数 | 分类输入 | 目的 |
|---|---|---|---|
|A 固定参考|无|原q_final|复用保存输出，在同128场景重新评分|
|B head_only|原class_ln/class_head|原q_final|现有query特征能否经原分类头适配改善|
|C object_support|B及轻量读回模块|q_final加本query mask覆盖的3D特征残差|对象专属支撑是否有作用|
|D scene_support|与C完全相同参数量/初始化|q_final加全场景可见3D特征残差|排除新增容量/通用场景上下文作为唯一解释|

共同冻结VGGT、MASt3R/ViT adapter/pixel decoder、原query decoder、GS、memory adapter、mask embedder、panoptic其他参数。全部冻结模块保持eval，原6/8/10/12双向读取及点积feedback0.05仍执行，q_final与Gaussian mask生成保持原路径。原19类含no-object分类头与原loss不换。

3D feature使用现有context lifting与anchor-child混合特征（256维），不是新VGGT encoder或旧U3三尺度融合。C按本query的soft Gaussian membership乘可见mass加权平均；D仅按可见mass做同一全场景平均。输入均detach；无GT、目标图像或目标相机进入读回。相同LN/256→256→256 GELU投影，末层置零；C/D初始化参数逐位相同，起点分类结果与原版相同。仅分类q加残差，原mask query及feedback q不改。

三组300更新/global batch8/world4/micro1/accum2，2400不同窗口来自现有完整8337窗口池的固定共同stream，不在128验证场景训练。恢复原四rank RNG，fresh原生AdamW，原object_memory peak1e-4、betas/decay/clip1和200更新LR warmup，4172步cosine截断至300；原曝光33376续接，U=1及实际feedback0.05保持。A无训练。此为诊断预算，不能称作新完整训练终点。

使用原train_microbatch_window和完整set loss/Hungarian cost/系数，继续计算重建loss。由于训练参数只在分类末端，重建无可训练依赖，native understanding_only检查此条件并安装精确为零的R梯度；不删除或重加权理解项。匹配随类别分数原生更新，不人为固定GT分配；即使mask参数冻结，匹配改变也可能改变mask loss数值。

300步后自动评测同seed4242固定随机128场景，各一六帧窗口，原SIU3R evaluator和13列表，两处mIoUt空。RGB/depth与GT PNG逐帧核对原完整权重，只有完全相同时才重用原逐帧重建评分；理解重新官方评分，AP/PQ/IoU全局合并不平均分片。同步记录原生Hungarian的好mask/no-object及GT分数排序诊断；raw mask IoU与GT逐条对齐原完整复核。GT分数不可部署，不能写入实际13列。

因果解释限于干预效果：B有增益支持原分类读出尚有优化空间，但不证明query特征完美；C优于B和相同容量D，才支持对象专属mask支撑的作用。C/D均有收益则可能主要是场景上下文/读出容量。300步没有收益不能排除预算或优化因素。此轮固定重建和mask生成，之后再决定是否将有效改动接入联合训练，不直接归罪于原双向层或宣布结构上限。

先CPU合同/语法，再同不可变push SHA单卡与四卡各三组两更新真实smoke，包含4253/6923，核对初始分类与更新后RGB/depth原生重现、冻结版本、实际更新、四rank同步和原生13列评分。正式三组用四卡3090，B/D在11号、C在13号；preflight不计正式时钟。只保存分类参数补丁、AdamW/RNG与采样计数，不复制每组数GB核心权重，保留原源资产和历史失败。

证据root：`/space/mawb/ssst/group_plus/vggt_classification_readback_v1`；固定执行快照在snapshots，attempt01含preflight、reference、三组训练/自动评测/自动汇总。工程错误保留失败记录、修复验证后用新attempt续跑，不改正在运行的快照。启动正式20步/160曝光通过后停止主动轮询，让已提交训练和评测继续。
