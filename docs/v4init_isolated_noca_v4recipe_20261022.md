# V4init + No-Anchor-CA + 梯度隔离 + 早期V4训练配方（2026-10-22）

用户明确授权仅一组实验；确认采用L6/L12重建监督、理解loss权重1.0。此前两组70714–70719均已完成，不重提。

## 固定与改动

- 保留本轮V4init完整重建初始化：原FPS1024种子、精确来源像素、持久gs_tokens、XYZ/radius/RGB内容MLP、原始GFM2048的LN→Linear投影，65536GS。不是从已训练V4init epoch4续训。
- 直接复用V4init **零步checkpoint_initial.pt**，SHA256 `bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397`。来源updates0、空Adam、8rank RNG。原A4零步来源SHA `3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744`。
- 精确移除四个理解读出层Anchor CA的Q/K/V/O共32张量。ev=0，保留原norm(q+ev)；R=geometry_bias.softmax用于原c/s更新。FM CA、自注意力、FFN保留。理解读出仍L6/8/10/12。
- 显式隔离理解读取的重建tokens、Anchor mu/rho/radius、逐GS几何和mask渲染Gaussian参数。Write=0；无adapter。先运行重建，再以已有reconstruction_states API从detached状态运行理解，原重建state保持连通供RGB监督。
- 重建监督L6/L12、权重[1/3,2/3]；理解loss权重1.0，乘min(1,(update+1)/1043)。理解目标仍当前source_many_to_one / support_purity=.5，非早期V4的理解query/匹配架构。
- 仅core.anchor_decoder.refine_mu/refine_rho基础LR=1e-5，其余可训练参数1e-4。两组同warmup100、cosine乘数floor=.1；对应末端LR约1e-6/1e-5。没有修改Gaussian颜色/尺度等头的分组。
- AdamW betas(.9,.95)、eps1e-8；原weight decay规则（二维及以上且无_no_weight_decay标记为.05，其余0）；全参数统一clip norm=1，与早期V4一致。**全局clip仍可能通过总梯度范数耦合两分支，detach隔离不等于完全独立优化器轨迹。**
- 8337窗/1191场景、2context+2novel、256²、8卡×1/global8、4epoch/4172updates；原manifest、epoch顺序、冻结Omega/SAM身份不变。
- FM原生128²，GS lifting前按旧路径插值256；理解query仍原投影点云→超点256D。没有FMquery、FM256、额外loss或其它实验组。

这是组合配方测试，只能回答整套方案是否改善，不能分摊隔离、去CA、LR、监督层和理解权重各自贡献，也不是完整早期V4架构迁移。

## 验证

- 2项针对性CPU测试通过（11.144秒）：实际core的NoCA联合/隔离前向逐值一致；LU对全部重建参数零梯度、LR对理解/共享投影零梯度；9个初始化张量均获LR梯度；优化器覆盖和LR分组正确。
- CPU真实源审计：632张量逐值保留（含9个V4初始化张量）、移除32、无新增张量。可训练195961456参数。48个mu/rho头张量进入geometry组。
- 拒绝8类非法源：已训练、非空Adam、数据不符、rank RNG不全、额外张量、缺失保留张量、移除张量shape错误、原A4来源哈希错误。
- CPU只用合成8²渲染和6个实际anchor执行loss路由测试，不代表GPU通过。GPU单卡2步smoke使用真实256²、1024anchors/65536GS、原2+2视角，8次累积/global8；逐项核对几何LR、U权重、双向梯度隔离、9个初始化梯度、权重更新和FM128形状。

## 独立执行、依赖及完成收据

代码快照：`/space/mawb/ssst_v4init_isolated_noca_code_20261022_v1`。
运行根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1`。

依赖链仅一条：单卡2步smoke → afterok 8卡完整训练 → afterok 8卡官方1860评测及结果汇总。正式训练重新加载同一零步，绝不加载smoke权重或Adam。

节点池12/13/14/17/18，单节点3090/4090。GPU smoke失败阻断训练，训练失败阻断评测。提交后只核对一次调度接收状态并停止，不能为了首步、smoke或评测完成继续监控。

评测固定1860窗/312场景、target-all6（含context）、SIU3R commit `8ea80166be76854f938e90521f1a5b688b755c87`，不改SIU3R/src。验证导出输入与GT逐窗匹配旧A4，参数更新0；同时对比旧A4、本轮V4init joint、本轮旧A4 NoCA joint。

运行记录：根目录CPU_VALIDATION.json、CPU_TESTS.txt/json、SUBMITTED.json、STATUS_AFTER_SUBMIT.txt；smoke/SMOKE_PASSED.json、DIRECTIONAL_GRADIENTS.json；train/FIRST_UPDATE_VERIFIED.json、TRAINING_COMPLETE.json；eval/EVALUATION_VERIFIED.json、evaluation/metrics.json、comparison/vs_old_A4_and_V4init.csv。

GPU first-update verification is a training-side receipt, not a polling task. User notification is required before inspecting progress/results after the one acceptance check. Do not resubmit from stale RUNNING/PENDING records.
