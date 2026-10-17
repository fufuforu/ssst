# Unified Query：多对一实例监督短训练验证

用户确认pipeline并授权实现、提交作业，先验证Unified Query能否训练。本轮只提交一个新结构短验证，不新增旧一对一训练组、不恢复B、不修改66353，也不自动启动全量训练。

## 模型与损失

冻结官方VGGT-Omega预测相机/深度/点云/GFM；冻结已有SAM ViT-B生成区域，按ESAM MV的面积覆盖及未覆盖点XYZRGB KMeans补分组。有效区域至少一点，同区域压缩至点预算，三维邻域编码后超点池化成唯一Query序列。共享12层Decoder输出重建Gaussian与类别/跨Gaussian完整实例Mask。宽256、8头、上限512超点/4096融合点，每Query64个Child。无随机fallback、无MASt3R、无旧适配权重；超预算显式报错。

默认`matching_mode=source_many_to_one`。超点内GT实例占有效标注点比例≥0.5才可匹配；每Query选择占比最高的合格GT，恰好相同按GT顺序确定；一个GT可获得多个Query。所有正Query监督**完整GT实例Mask**，不是局部超点mask。未分配Query监督no-object，生成的Gaussian仍保留。没有支持的GT记录unsupported，不伪造正样本。

分类正样本在各GT内平均，随后跨GT归一；所有no-object Query先平均为单独背景组，以0.1相对权重参与归一。Mask BCE/Dice同样按GT先内后外平均。最终理解loss仍为0.1×(2CE+5thingBCE+5thingDice+5stuffBCE+5stuffDice)，重建沿用L6/8/10/12规范目标。共享层正常接收两任务梯度，无隔离。

推理：context两视图拼接Mask，阈值0.5及alpha>0.05；类别最大概率≥0.05且argmax非no-object才是候选。按类别概率×前景内Mask均值排序，同类IoU>0.7抑制；相同分数按Query索引稳定排序。选中身份复用于目标视图，不读取GT、不在目标视图重新选ID、不删除Gaussian。原官方组装接受该候选标志，其他阈值及评分器不变；旧模型没有该字段时沿用旧行为。这是新的模型推理定义，不能称全部后处理不变。

## 有界运行配方

- run：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu`。
- 快照：`/space/mawb/ssst_unified_esam_many_code_20261017`，与其他快照隔离。
- 18节点8×4090，32 CPU、192G内存、12小时上限。
- 原1191训练场景/8337窗口中复用既有确定性16场景短验证选择，每场景一个固定窗口；与128 heldout无场景重叠。选定列表及数据SHA见preflight.json。
- 短测256更新/2048场景曝光，相当于16窗口重复128遍，**不是完整数据128 epoch或原4epoch训练**；全局批8，每rank一场景。
- Omega/SAM冻结，14,615,492个其他参数全新初始化，AdamW空状态。共享13,912,881／重建296,576／理解406,035。容量不同于旧双流，不能把结果当单变量匹配消融。
- AdamW LR1e-4、20更新warmup、256更新短cosine、clip1；理解总权重0.05，前20更新ramp。该权重用于先检验联合路径，不复用旧梯度隔离。
- 每32步checkpoint，保留零步checkpoint、逐rank连续日志，检查非有限loss/梯度。首更新记录shared/reconstruction/understanding梯度norm、确认冻结模块无梯度，并核对八rank全部可训练参数SHA及实际变化。
- 训练前后同4个train＋4个heldout探针：重建/理解loss、PSNR、支持GT最佳Mask IoU、分类准确率、匹配GT/Query数量、去重输出及Query/点/Gaussian数量；保存紧凑RGB/语义/实例预测张量供复查。
- 结束同一作业内自动CPU汇总参数变化、逐rank256更新及前后探针。成功状态只代表可训练性运行完成，仍需人工分析质量，不强制loss下降，不自动启动全量训练。不是正式128场景/11指标评测。

## 提交前证据与限制

20项CPU测试通过，覆盖多对一四超点全正监督、同实例超点复制不放大loss、完整Mask梯度、同类NMS/异类及分离物体保留、组装候选传递、原监督/初始化/旧诊断回归；另1项真实双rank CPU Gloo测试通过，两个rank使用不同Query数量、启用checkpoint、两次Adam更新后全部参数SHA一致。合计21项。实际规范重建目标也通过假渲染器梯度检查。

真实CPU预检已加载并校验官方Omega和SAM权重，核对可训练分组全覆盖、16场景/128 heldout隔离、数据SHA及训练/验证真实RGB与标注读取；验证RGB256、2context与完整view首两张相同，heldout输入SHA匹配旧C。真实CUDA渲染、SAM提议预算和八卡GPU更新仍待本作业内执行，不把CPU或RUNNING称为GPU已通过。

作业号与执行SHA以submission.json为准。提交后只做一次有界状态核对并停止轮询，待用户通知完成再分析。

## 已提交并停止轮询

训练及内置探针/汇总作业 **66414**，2026-10-17 12:18:12（北京时间）提交；单次核对RUNNING、3dimage-18、8×4090、32CPU/192G。执行SHA`6aa16a0f7a81dd55d38a25ad5379fdba05cdb56f`，独立快照干净；源码中的LFS展示素材采用skip-smudge检出，训练权重仍使用已校验的外部实际文件。GPU首步、真实SAM区域预算、CUDA反向及八卡SHA尚未核实，不能把RUNNING视为训练通过。已停止轮询，用户通知后检查FIRST_UPDATE_VERIFIED.json、逐rank日志、TRAINING_COMPLETE.json、SHORT_VALIDATION_SUMMARY.json及初末探针。没有可靠的新架构实测步时，暂不给完成时间保证；作业会记录seconds和显存以供后续估算。
