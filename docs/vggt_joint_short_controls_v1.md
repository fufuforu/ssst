# AA 冻结／解冻与 GS decoder 初始化短对照

用户授权在等待原 B 期间执行两个小预算问题。三组共享一个控制，adapter 理解梯度系数均为0.01；不是把整个理解损失乘0.01。

| 组 | AA | GS 初始化 | 理解／panoptic |
|---|---|---|---|
| frozen_warm | 冻结 | 共同 VGGT epoch6 | 可训练 |
| aa_warm | 解冻24 frame＋24 global blocks | 同上 | 可训练 |
| frozen_scratch | 冻结 | 原模型构造函数的未加载权重状态 | 可训练 |

共同源为适配2轮＋冻结联合4轮epoch6，使用既有严格SHA和FP32 adapted VGGT资产。scratch只重置GS tokens、原decoder blocks、anchor decoder（含中心／半径／PE／refinement）和Gaussian head；保留VGGT、已训练adapter、理解与panoptic。构造时seed42，在任何迁移前捕获真实原生初始化，包括零残差头、共享decoder别名和固定初始化常数；不使用遍历reset_parameters来破坏专用初始化。该实验检验“在已经适配的特征空间中重置GS是否有短预算收益”，不能直接回答完整流程从第一轮开始训练新decoder的最终上限。

沿用之前已执行推理诊断的seed42选样：16个训练场景、另16个官方留出场景，各一个窗口，不按指标选择。队列窗口内容写入cohort JSON并锁定SHA。每组100正式更新，八卡microbatch1×accumulation2＝global16；每次使用全部16窗口的seed42+update排列，每窗口恰好100次曝光，共1600重复曝光。MASt3R及已有理解参数沿原trainable flags，LPIPS和DINO/camera/depth保持冻结。反馈beta0.1／理解权重1，原生重建及V3-Set全部损失保留；GS及AA理解梯度系数0.01，理解／panoptic保持原梯度规则，仅adapter系数由1改0.01。

复用原runtime的configure_joint_continuation、train_microbatch_window、CPUOffloadAdamW、数据provider、renderer和monitor_v1。AA FP32、DINO BF16/autocast、下游FP32；AA学习率1e-6、GS及理解1e-5、新object/memory1e-4，原no-decay、betas、epsilon、decay、clip1不变。和上一轮短诊断一样，20更新warmup后保持原峰值，不把它伪称为生产长cosine调度。每组fresh优化器；隔离smoke不计入正式时钟。

先三组配置分别真实单卡两更新，再分别真实八卡两更新；确认理解、decoder、adapter、panoptic及AA（仅解冻组）非零梯度和参数更新、全rank参数相同、冻结参数不变。每组还在真实留出窗口验证原生PNG/深度/语义实例导出及完整官方13列评分，再提交正式作业。八卡全部16训练窗口必须真实经过。工程错误保留旧attempt，修复验证提交推送后使用新的不可变快照。现有A/B与其自动评测不修改。

node17八张4090顺序执行三组训练，480G host memory；后续四卡评测160G。先检验GPU preflight再提交正式，不让失败的smoke产生正式训练。保存0/50推理快照及20/100含优化器／八rank RNG的可恢复checkpoint；真实保存后读回核验时钟、模型及AA权重。保留所有旧证据，不清理历史实验。空间提交门槛110GiB；新smoke不保存整模型checkpoint，用真实更新／同步收据和正式20更新的完整保存验证覆盖工程路径。

评测自动afterok接续，三个组0/50/100共9条件，每条件16留出窗口／96图像，合计144窗口／864图像。context-only generate、独立预测相机标定、原生all视图PNG/深度及语义实例导出；每张卡负责完整条件，直接调用固定SIU3R官方Evaluator，在16场景上全局计算mIoU/mAP/PQ，不能平均单窗口mAP/PQ。输出所有已实现指标的单行13列表格，两个文本mIoU留空，不另拆context/target-all/true-novel表。不同组初始状态中，AA/冻结旧decoder的PNG必须相同；各条件GT图像、深度、标签相同。评测无参数更新，明确是16场景小样本而非1860窗口官方完整结果。

判断AA对照使用frozen_warm与aa_warm，初始化对照使用frozen_warm与frozen_scratch；同时看全部11项，明确取舍。随机组要比较绝对终点和自身0→50→100曲线；短预算落后不能证明从零长训练更差，初始优势不能证明最终上限更好。单种子／小样本不作显著性结论。新的结果不能与上轮四窗口的24.66直接比较。

读取首个正式作业20更新／320曝光的真实startup与完整checkpoint核验后停止轮询；其他组仍可能等待依赖，使用同份已通过真实八卡验证的固定代码并各自自动写startup。训练和afterok评测继续，不能把排队组称为已经完成20更新。
