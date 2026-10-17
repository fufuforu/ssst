# Learned-base＋SAM残差初始化隔离训练

用户于2026-10-17授权提交此单变量对照。以已完成V4完全隔离组为基准，从共同零步权重重新建立一个独立实验，不继承终点模型、不重放适配。

## 改动

原SAM槽：`q = LN(SAM_content_projection + geometry_encoding)`，完全替换learned base。

新SAM槽：

```
seed = SAM_content_projection + geometry_encoding
rms = sqrt(mean(stop_gradient(learned_base)^2)).clamp_min(1e-6)
q = LN(learned_base + sigmoid(gate_logit) * rms * LayerNorm_without_affine(seed))
```

全局共享可学习标量门控，初始sigmoid值0.1，新增参数1。残差先去均值/标准化，再校准到每槽learned base的RMS量级；初始残差norm约为base的10%，避免SAM表示的较大原始norm直接压过base。最终LayerNorm沿用原参数。门控可训练但不做额外warmup；gate与其他理解参数按原查询优化器规则训练（标量无weight decay）。

无SAM的fallback槽不变。V4几何池化、候选排序/去重/跨视图合并/80槽截断、局部c/s、L6/8/10/12交互、Write0、Hungarian匹配、类别/noobject与Mask损失及官方评测协议不变。保留局部c/s的已知风险，本轮不同时放宽，以便将变化限于query表示构造。

## 控制设置

- `attempt16_learned_base_sam_residual_isolated_node13_8gpu`，新独立不可变快照`/space/mawb/ssst_gfm_sam_residual_code_20261017`。
- 13节点8×3090，与V4隔离对照相同硬件；全局批8。
- 共同源为attempt11的`checkpoint_initial.pt`（0更新、空Adam、8rank RNG），所有649个共享张量严格复制；只有残差gate新增，不继承任何已训练理解权重。
- 官方Omega及SAM冻结；重建可训练，仅接收重建loss梯度；理解参数仅接收完整理解loss梯度。全局clip1仍有范数耦合。
- 完整1191场景/8337窗口，4epoch/4172更新，adapt/short0；重建/几何/理解LR=1e-4/1e-5/1e-4，100更新LR warmup＋cosine，理解loss首1043更新ramp。
- 使用原显式FP32 SUM/world梯度reducer，首GPU更新检查八rank所有可训练参数SHA一致。记录门控值、门控梯度及参数变化证据；CPU验证不替代GPU证明。
- 训练成功后自动官方128评测，11项有限、文本两列空；自动比较V4原隔离、V3原隔离与新残差组。

## 验证与判断

CPU测试覆盖base/seed/gate梯度、残差norm与尺度/偏移归一化、门控归零行为、候选数与c/s及fallback逐位不变、共享权重精确复制、新增参数范围、零步及配置锁定。真实源预检在CPU实体张量上核对共享值、空Adam及优化器分组/顺序，并检查数据SHA。导出按checkpoint配置重建残差模块，严格加载新gate，避免评测时意外回到旧初始化。

成功标准不是门控非零或loss下降本身：首先完整训练/官方评测，然后比较PSNR/mAP/PQ，并在需要时复用固定权重审计检查SAM正匹配、noobject和最终输出贡献。门控塌到零、SAM仍无正匹配也必须如实报告。若失败，不在本次训练中自动改匹配规则或重跑。

作业号与提交时间以run下submission.json为准。提交后仅单次核对调度状态并停止轮询，待用户通知结束后再检查，不在本轮持续等待GPU首步或训练结束。

验证记录：14项CPU单测通过，py_compile/bash -n/diff --check通过。真实CPU实体张量预检PASSED：649个共享张量值逐位一致，仅新增gate标量1参数；重建186881600、理解9128561参数，源SHA b9d7bd597cc2bd0d2439a1eae3771760f2be55241a2db1c4ef3879b08a75a03e，空Adam/8rank RNG及数据配置锁定。预检将meta模板实体化时保留原参数_no_weight_decay标记，以正确核对原优化器分组，不改变训练规则。GPU首步/同步未宣称通过，以作业收据为准。

## 已提交并停止轮询

训练66353：单次核对RUNNING、13节点8×3090；官方评测66354 afterok:66353；比较汇总66355 afterany:66354，二者当前Dependency。执行SHA767cbe87c8baabcb749b85a6e3121ca0e7a008fc，snapshot干净。GPU首步/门控梯度/八rank同步尚未核实，RUNNING仅证明资源分配，不作为训练数学通过证据。提交后不再轮询，待用户通知。按同节点此前隔离组估计训练4.5–5小时、评测10–20分钟，非完成时间保证。
