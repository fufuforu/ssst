# 固定教师 Membership 引导几何：四卡初步对照

用户授权 G1/G2 各追加一轮，只用四卡；若有效，后续再另行从零加入机制训练。本轮没有从零训练组，也没有增加 IoU head。

|组|来源|可训练权重|目标|新增更新|
|---|---|---|---|---|
|G0|已完成 Baseline4|无|复用既有正式结果|0|
|G1|同一 Baseline4|48 个 refine_mu/refine_rho 张量|原重建目标|1043|
|G2|同一 Baseline4|同 G1|原重建目标 + λ Membership 渲染监督|1043|

来源 SHA256：694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc。原理解权重1.0、2Context+2Novel、epoch4/full_updates4172。G1/G2 均至累计 epoch5/full_updates5215，但标注为几何诊断续训，不能当作完整联合训练的 Baseline5。

固定项：原8337训练窗/1191场景、256²、epoch_order(epoch=4)、有效global batch8。四rank每rank两次微批累积，统一1e-6学习率、原AdamW betas(.9,.95)、eps1e-8、原weight decay规则，几何梯度clip norm1。完整源Adam先按原604状态恢复，再冻结非几何参数；几何step4172→5215，其他状态保持原样。源八rank RNG取前四rank，G1/G2一一相同；这是配对四卡试验，不声称与原八卡随机轨迹相同。模型eval，只有几何decoder设train以启用已有非重入activation checkpointing；其block没有随机dropout。

冻结教师在独立缓存阶段对原训练窗运行一次，不更新任何权重。用原训练 source_many_to_one/support_purity=.5 关系给query分组；同GT实例的Gaussian Membership取max。通道为stuff0、stuff1、受支持的GT实例、unmatched/unknown，并在Gaussian层面归一化；unmatched通道取所有未匹配thing query的max。GT只取训练窗两张Context，void、无支持GT实例像素忽略。缓存Membership为FP16，加载后以FP32重新归一化；相机、源Gaussian位置/尺度参考保留FP32。anchor_point_indices与Context RGB SHA强校验保障teacher/student Gaussian槽位对应。

新增分支将固定Membership作为Gaussian特征，用当前学生位置/尺度渲染到固定教师Context相机。新增分支的opacity、rotation、RGB显式detach；既有理解渲染的geometry detach不改。渲染结果是premultiplied alpha，残余透明度归入unknown通道，防止靠消失规避损失。L_mask=.5*(有效像素多类CE + 有GT正样本通道平均Dice)，Dice smoothing=1。G2 λ=.01，前100个新增更新按(step+1)/100线性引入；G1 λ=0。两组都执行同一新增前向以记录对照，不以G1的mask目标更新权重。教师不随学生更新。

冻结头的权重不变不保证RGB/opacity数值不变：上游几何可能改变重建attention和后续token。最终审计强制所有非几何权重及Adam moments/step逐位不变、几何实际更新且有限、四rank末态一致。日志记录CE/Dice、几何/尺度相对源位移和梯度范数。

评测：沿用原1860窗/312场景/2Context+4Novel、固定SIU3R 8ea80166be76854f938e90521f1a5b688b755c87，G1/G2重测全部11主指标及原生AP/AR；G0完整指标复用。三组均重测仅额外四张novel的PSNR/SSIM/LPIPS/AbsRel/RMSE，不混入Context。深度仍为官方逐图scale+shift协议，不主张metric尺度几何提升。附加GT实例边界7x7带上的RGB MAE/RMSE，Context/novel分开；仅作诊断，不用于选checkpoint。主因果比较G2-G1；G0为固定起点参考。单轮单种子试验不足以支持从零训练或统计显著性结论。

作业链：共享四卡教师缓存（首窗实际CUDA梯度门禁，不做optimizer更新）→G1/G2各四卡训练→各四卡正式导出→一张卡评分与汇总。所有依赖afterok，缓存梯度门禁失败则训练不能开始。首窗验证两几何头均获得非零有限梯度、Membership不求导、其他权重无梯度；训练首步检查累积/反传，末态审计DDP一致性。提交前CPU测试和真实source/data预检；GPU检查在作业内执行，提交时不得宣称通过。

独立运行目录和代码提交不可覆盖。提交后仅一次有界接收核对，停止轮询，等用户通知完成后核查。

## v2 SDPA工程修复重提

旧71349在首窗mask backward报LSE strideH，训练及评测依赖取消，原运行/快照保留。新独立v2_sdpafix目录仅修复attention后端：当grad enabled且additive bias.requires_grad时强制SDPBackend.MATH；无梯度教师/导出与无bias或固定bias的attention沿原路径。guard在SDPA入口，覆盖非重入checkpoint重计算。冻结Q/K/V、几何bias求导是重点回归情形；具体旧CUDA算子尚无anomaly定位，不把该数值等价绕过方案说成已证实上游根因。两组使用同一guard，配方/来源/数据/损失不变，允许浮点舍入差异。CPU新增5项测试与原7项一起运行，核对显式attention公式的输出和梯度、query+bias梯度、checkpoint、no_grad原路径、幂等性及math算子实际执行。首窗GPU门禁扩大为mask与原重建各自反传、anomaly trace、统计guard实际调用；没有optimizer更新，四rank通过后才启动训练。提交时GPU结果未知。math attention可能增加显存与耗时。DDP构造前后恢复RNG，保证四rank来源映射。
