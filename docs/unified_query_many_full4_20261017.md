# Unified Query完整数据联合4epoch

用户在66414短验证结束后明确授权开始完整数据训练。本轮保持通过真实GPU验证的三维超点Unified Query结构、来源多对一完整Mask监督、context同类Mask NMS和全部重建Gaussian。唯一新训练组，不恢复B、不重复短验证、不影响66353。

## 起点与训练设置

- 独立run：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt02_esam_many_full_node11_8gpu`。
- 独立快照：`/space/mawb/ssst_unified_esam_full_code_20261017`。
- 当前18/14/17八卡均已占用，11节点八张3090空闲；采用11节点8×3090、32CPU/192G、24小时上限。硬件与短测18节点4090不同。
- 原锁定1191训练场景/8337窗口，4epoch，每epoch补齐至8344曝光/1043更新，合计4172更新/33376曝光（含padding）。全局批8，每rank一场景。
- **读取短测的零步`checkpoint_initial.pt`，不是256步终点。** 301个非冻结参数张量逐位复用，8rank RNG复用，Adam重新创建为空；不加载适配或已训练理解权重。显式SHA验证源文件，拒绝非0更新、非空Adam、结构/config、冻结资产、数据或rank数不兼容的输入。新run仅硬链接零步源为`checkpoint_untrained_initial.pt`，训练后不会覆盖它。
- 冻结官方Omega与SAM ViT-B；其他14,615,492参数训练。3D点编码/超点池化、12层共享Decoder与双任务头不变；无MASt3R、无随机fallback、无实例embedding分支。
- AdamW LR1e-4，100更新LR预热＋全4172步cosine，weight decay0.05（bias/LN等1D为0），clip1。
- 理解总权重0.05，首1043更新ramp；共享参数正常接收两任务梯度，无梯度隔离。相较短测仅将20步LR/理解预热换为完整训练计划的100/1043步。
- 首步检查各参数组梯度非零/有限、冻结模块无梯度、八rank所有可训练参数SHA一致且相对初始改变。每50步checkpoint，并保留4个epoch终点硬链接。
- 训练初始和每epoch末对同4train＋4heldout探针做固定权重前向，保存重建/理解指标、来源匹配、Mask及预测；不因loss未下降而自动中止或修改配方。

## 训练成功后的依赖评测

提交同节点1×3090评测，`afterok:training`且无效依赖取消。严格拒绝短验证checkpoint和未完成epoch/更新数，执行：

1. 独立全128固定heldout导出：context-only生成，独立目标相机标定，context上NMS选出的身份复用于目标视图。无参数更新，验证输入RGB SHA和旧C参考GT PNG完全一致。
2. 原SIU3R官方评分器（锁定commit8ea80166be76854f938e90521f1a5b688b755c87且src干净）评分。13列表格11个数值，文本两列空；非有限官方指标保留失败记录，不筛帧或替换算法。
3. 检查完整收据及CSV/native结果一致性；与已完成旧C、V4隔离、V4理解权重0.05组汇总比较。比较用于实际指标定位，容量、架构和训练配方不相同，不能宣称单变量匹配因果对照。

复用原导出、评分、结果检查和比较代码；仅为Unified Query新增训练后依赖wrapper，评分Markdown标题改为真实recipe；导出manifest明确记录新模型NMS，而不冒称全部后处理未变。

## 提交前验证

真实零步checkpoint预检通过：0更新/空Adam/8rank RNG/301张量值逐位一致；实际Omega/SAM文件SHA再次校验通过，数据SHA、1191/8337/128、完整计划4172更新、真实训练首末窗口和heldout样本读取、官方评测commit及src状态通过。此前66414已有真实八卡256步前后向和首步同步证据。

本轮21项CPU测试通过，包含拒绝已训练终点/非空Adam/错误数据或配置/RNG覆盖，以及既有初始化、梯度、多对一和NMS回归；语法、shell和diff检查通过。新全量作业的GPU首步仍需以本run收据为准，旧短测的成功不能代替它。

预估训练约4–6小时，评测另需约20–40分钟；依据短测4090约2.82秒/步及此前3090全量任务的量级估计，新数据Query数和节点负载会影响实际时间，非完成时间保证。

作业号及执行SHA以submission.json为准。提交后只做一次有界状态检查，之后停止轮询，等用户通知再检查。
