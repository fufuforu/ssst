# Adapter 理解梯度诊断与短对照

用户在三组四窗口拟合诊断完成后授权：“按你说的来查吧”。本次先进行已有 checkpoint 的零更新梯度分析，再进行两个300更新的受控短对照。不修改正在运行的正式 A/B 或其评测，不重启任何已有训练。

## 零更新梯度归因

沿用已固定的四训练窗口、共同适配2轮＋冻结联合4轮的epoch6源、FP32 AA/BF16 DINO和现有缓存重放。选择原联合组0/20/100/200/300更新、仅重建预测组100/200/300更新，共8个条件/32次窗口分析。所有实际文件校验SHA，20更新的额外checkpoint检查冻结状态，其余使用上轮完整校验记录；输入RGB与原训练proof匹配。当前源码默认前向、损失和梯度数学保持不变，新增可选instrumentation只暴露原生加权分类、thing BCE/Dice、stuff BCE/Dice的可微张量。

四GPU每卡一个固定窗口。分别计算重建、理解总项和五个原生组件的梯度；在GPU上按原生分桶通信准确平均参数梯度，host保存临时梯度，统计adapter、decoder、GS tokens、anchor几何及Gaussian head的范数、U/R比例和夹角。记录三个adapter系数1/0.01/0对应的静态组合梯度及clip系数，不能当成带Adam历史的更新预测。

分别保留最终重建Gaussian和理解readout Gaussian的输出梯度，检查这两次activation的数值完全相同。输出通道顺序沿用原始14通道：xyz0:3、opacity3:4、scale4:7、rotation7:11、RGB11:14。重建输出梯度只表示原生最终层及其loss权重，模型参数梯度仍包括所有监督层。记录region_mass概率边界及各理解项的像素梯度，并保留Hungarian匹配。Gaussian/像素输出属于各窗口，不能把不同场景的输出梯度逐元素平均解释为同一场景。

不创建optimizer，不更新参数，不保存features或梯度向量。五项损失之和、五项全局梯度之和分别验证与原生总理解目标和梯度一致；版本号及空parameter.grad验证零更新。这里只分析已有权重，不能宣称精确复现原训练第218次更新的瞬时尖峰。

## 两个受控短对照

| 条件 | Adapter 梯度 | GS 梯度 | 理解损失/反馈 |
|---|---|---|---|
| 原联合基线（复用） | g_rec + 1*g_under | g_rec + 0.01*g_under | 原损失；反馈beta0.1开启 |
| adapter_u001（新增） | g_rec + 0.01*g_under | g_rec + 0.01*g_under | 同上 |
| adapter_u0（新增） | g_rec | g_rec + 0.01*g_under | 同上 |

两组保留完整理解损失，包括分类、实例和stuff监督，只改变理解梯度进入VGGTMemoryAdapter的系数。修改native gc_combine/train_microbatch_window新增默认1的可选keyword；原生产入口不传新keyword，默认数学不变。训练诊断入口需要显式--gradient-routing-diagnostic，且只允许pred_joint和0/0.01/1，避免误用为生产配置。

和父实验相同：冻结全部VGGT、全部理解与panoptic/反馈参数、LPIPS；训练原adapter/GS tokens/GS decoder。反馈始终开启。预测相机和monitor_v1、模型及数据不变。每组八张4090，microbatch1×accumulation2×world8=global16，每次四窗口各重复4次。300更新/4800重复曝光，每窗口1200次。fresh原生FP32 CPU-offload AdamW；共同源参数/RNG策略、20更新warmup到原peak、GS1e-5/adapter1e-4、原clip1/no-decay/betas保持。保存0/20/100/200/300 checkpoint；原checkpoint和历史失败证据不修改。

CPU测试检验默认系数逐值一致、仅adapter理解梯度改变、原组件之和及梯度一致、有/无GT实例、八进程累积/平均/clip/Adam对全局平均参考、严格介入checkpoint标记与计数。两个新系数分别进行真实八卡两更新smoke，验证源初始化/像素/真实参数更新/冻结状态/保存和严格恢复。smoke为0正式更新，正式训练重新fresh初始化，不能把smoke时钟带入正式运行。

## 执行与评测

计算只使用已提交推送的固定快照，node17。证据root为`/space/mawb/ssst/group_plus/vggt_adapter_gradient_diagnostic_v1`，正式root为`/space/mawb/ssst/workspace_group_plus/vggt_adapter_gradient_diagnostic_v1`，每次工程尝试单独attempt。提交顺序：四卡probe→八卡双系数smoke；读取probe、确认smoke后提交八卡u001→八卡u0→四卡共同评测。Slurm afterok串联，不占用A/B卡，也不要求用户逐项批准工程修复。

共同评测复用上一轮exporter和固定SIU3R官方Evaluator，八checkpoint条件/32窗口/128张图像。新的checkpoint严格检查source、recipe、gradient_routing_variant、understanding_weight1、GS系数0.01和指定adapter系数。初始PNG与旧联合基线逐字节相等，各阶段GT PNG与旧基线逐字节相等，再与已完成旧joint/rec-only曲线配对比较。输出PSNR/SSIM/LPIPS、重建目标和理解探针，不拆分视图范围。只用于训练窗口拟合，不是完整13列表格；mAP/PQ保留情况仍需要后续“理解模块可训练＋留出窗口”的验证，不能以理解探针冒充。

提交命令：`scripts/submit_vggt_adapter_gradient_diagnostic_v1.py --stage {probe,preflight,formal}`；Slurm wrapper为`run_vggt_adapter_gradient_diagnostic_v1.sh`。提交回执记录job ID、固定SHA、目录、依赖和资源。真实两更新验证成功后才提交正式短对照。每组runtime在正式20次更新/320曝光自动验证并写startup_confirmation；读取首组正式20次启动确认后停止轮询。第二组仍按依赖等待，不能把它写成已经完成20次正式更新；它使用同一份已通过真实八卡验证的代码，启动后自动生成独立确认。两组及其后续评测继续执行。

下一阶段仅在短对照支持明确方向后，才以可训练理解分支和独立留出窗口同时验证重建、mAP/PQ/mIoU。当前不提交全数据长训练，不修改encoder或decoder初始化，不引入新loss/GT fallback/标定算法。
