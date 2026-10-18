# Unified 三组双向耦合短测

用户授权：实现特征回采、实例边界 RGB 监督及两者结合，三组各做短测。复用已完成的实例一致性基线67050，不重训基线，不自动转全量或官方128评测。

## 模型和梯度

三组均基于 `unified_instance_grounding`，原单序列超点 Query、12层共享 Decoder、Gaussian/分类/相似度 Mask、多对一监督、NMS 和 Panoptic 组装保持原实现。Omega/SAM 冻结，其余正常联合训练。旧执行快照及检查点不改。

1. `unified_instance_lift`：将共享 GFM 投影的2×32×32特征双线性升到256，与预测上下文相机、最终 Gaussian 一起送入 A 已有的 renderer adjoint。用实际 alpha-compositing 权重的转置回采特征，除以贡献质量 mass；不可见 Gaussian evidence 为0。`gate=mass/(mass+1)`，`child_new=child_old+0.1*gate*Linear(LayerNorm(evidence))`；再使用原 Query/Stuff 描述符计算 Mask，同时由新 Child 特征预测8维实例 embedding。分类器及 Query 本身不额外更新一次。新增4个张量、66,304参数（d256：LN512+Linear65,792）。回采的几何/相机权重 detach，仅支持图像特征的一阶梯度；原 Mask 和实例渲染损失继续对 Gaussian 几何求梯度，不是任务隔离。
2. `unified_instance_boundary`：不新增参数。从渲染8维实例特征的四邻域最大余弦距离得到边界响应，5×5扩张、阈值0.15/温度0.05 sigmoid。将**detach**后的权重用于归一化加权 Charbonnier RGB（epsilon0.001），只监督同相机、同256分辨率的两张输入图。无GT边界或alpha门控。`L=Lrec_old+lambda_bd*Lbd+u*(Lunder_old+0.01*Lground)`，`lambda_bd=0.02*min(1,u/0.05)`；短测20步ramp后为0.02，**不再乘理解系数0.05**。原实例一致性有效系数仍0.0005，保证只改变所声明的通路。
3. `unified_instance_bidirectional`：以上两项组合，无额外模块或损失。

新增模块在 fork_rng 中构造，不消耗后续训练RNG；预检及结束审计逐张量核对基线零步305个共同初始张量，保证新初始化而非接续256步终点。边界单独组的零步原生输出应与基线一致；回采组因新残差零步输出可不同。

这是受 InstanceSplat 启发的适配：论文边界来自光栅化前的像素实例特征，本实验来自渲染后的特征；特征回采是针对本项目的独立设计，不声称论文复现。论文边界消融的PSNR收益很小，短测不以大幅PSNR上涨为预期。

## 相同训练设置与工程门禁

同基线16训练场景/256更新、seed42、全局batch8、4rank每rank1场景并累积2次；LR1e-4、20步预热及原short cosine、理解u0.05/20步ramp、clip1。每32步检查点，初末各4train+4heldout固定探针。全新Adam，不继承smoke更新或训练终点。

每个Slurm作业：真实CUDA正负signed渲染检查→辅助loss梯度检查→新路径梯度契约检查→两步四rank DDP完整更新/逐步全参数SHA同步→严格smoke收据→独立torchrun重新初始化短训→原汇总/实例头审计/新通路审计。任何一步失败均停止写 `JOB_FAILED.json`。验收是数值有限、路径更新、同步正确，不要求loss下降。

新路径门禁要求：回采到投影梯度非零、可见Gaussian及残差非零、回采不对几何求导、新残差头确实变化；边界RGB梯度非零、实例源张量梯度不存在、权重detach。新增CPU测试覆盖线性渲染器的转置数值及反向、不可见Gaussian、边界对称/无环绕、损失归一化、参数/RNG兼容和边界独立权重。真实两rank Gloo分别验证三个版本、可变Query数、checkpoint Decoder、两次更新参数同步。

## 诊断与结论范围

保留基线已有PSNR、支持GT的最佳原始阈值Mask IoU、分类、输出实例数和实例一致性loss；新组额外记录实例特征对GT原型的紧致度、不同GT原型余弦、确定性打乱对应的紧致度，以及GT边界带/物体内部/背景的RGB MSE和预测边界权重。GT只用于预测后的损失/诊断，不进入初始化、回采或边界权重计算。

新诊断没有基线已存的完整实例特征，故不会伪造基线特征紧致度、边界权重。比较汇总报告前端点/Query计数是否相同；即使共同初始张量相同，也保留前端数值差异的限制。单seed/4个heldout探针不等于正式mAP/PQ或泛化结论；原始IoU只覆盖有来源支持的GT，最终输出实例数不等于正确实例数。

## 运行位置

- 基线：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt11_unified_instance_grounding_short_node13_4gpu`（67050，已完成，不重训）。
- 回采：同父目录 `attempt12_unified_instance_lift_short_node13_4gpu`。
- 边界：同父目录 `attempt13_unified_instance_boundary_short_node13_4gpu`。
- 双向：同父目录 `attempt14_unified_instance_bidirectional_short_node11_4gpu`。
- 三组设计/预检/提交/比较：同父目录 `instance_coupling_three_short_20261018`。
- 独立代码快照：`/space/mawb/ssst_unified_instance_coupling_short_code_20261018`。

选择13节点两组各4×3090、11节点一组4×3090，硬件型号及rank布局与基线相同；使用不同节点不是严格相同运行环境。每组16CPU/128G/4小时上限。基线实际34分26秒，边界组预计约35–50分钟，回采及双向组约45–90分钟（新增回采CUDA耗时尚未实测，排队另计）。三组成功后仅依赖CPU比较作业读取已有结果。

作业号/执行SHA/唯一调度核对以group的 `submission.json` / `SUBMISSION_STATUS_ONCE.json` 为准。提交后停止轮询，不等待CUDA smoke或训练结果，由用户通知完成后再核查。

## 已提交，停止轮询

2026-10-18提交，执行SHA `271c9ce1953910ff7e37e8e299e80872fc59d2f5`，上述独立快照保持不变。47项最终快照CPU检查通过，其中包括真实双rank Gloo覆盖三个版本；实际严格加载Omega/SAM，每组305个共同初始张量与67050零步状态逐值相同，同manifest/heldout SHA及16窗口，回采/双向各新增66,304参数，边界组不新增参数。日志和证据位于group的 `LOCAL_CHECKS.log`、`VALIDATION_EVIDENCE.json` 及各run的 `preflight.json`。

| 作业 | 内容 | 节点/资源 | 唯一一次调度核对 |
|---|---|---|---|
| 67128 | 回采→理解 | 13，4×3090，16CPU/128G | RUNNING |
| 67129 | 实例边界→RGB | 13，4×3090，16CPU/128G | RUNNING |
| 67130 | 两者结合 | 11，4×3090，16CPU/128G | RUNNING |
| 67131 | 读取已有基线及三组结果做CPU比较 | 13，2CPU/8G，无GPU | PENDING，afterok:67128:67129:67130 |

RUNNING仅代表调度运行；控制端**未查看或等待新GPU smoke、首步更新及训练进度**。由作业内部smoke门禁决定是否进入正式256步。未提交全量训练或官方128。用户通知完成后检查失败/完成收据、四rank日志/Adam/参数更新、新通路梯度与初末探针、COMPARISON.json；不重提基线或三组作业。
