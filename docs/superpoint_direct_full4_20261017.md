# B共享Query直接Gaussian Mask：完整数据4 epoch

用户明确授权B也训练。独立run：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt06_unified_direct_masks_full_node18_8gpu`；不可变快照：`/space/mawb/ssst_superpoint_direct_full_code_20261017`。核实18节点空闲，选择8×4090、32CPU/192G/24小时上限，与13节点8×3090的A66506并行；不修改、重启或重复提交A或原Unified组。

## 结构和配方

保持66503已验证的B：冻结Omega/SAM；3D超点池化初始化一组共享Query，12层256维Decoder同时服务Gaussian重建与理解。Query生成非线性Mask函数参数，在全部Gaussian的几何/外观属性上求值，输出实例归属概率；另由Query读出Child Gaussian的20类语义概率。两者渲染成二维Mask，结合thing/no-object分类、context NMS及像素竞争形成panoptic。不是旧Gaussian特征与Query embedding相似度读出，不添加新的结构。约14.68M可训练参数。

保留短测0.1×直接语义CE（在理解loss内部），理解总权重0.05，正常联合反向。完整数据1191场景/8337窗口，4epoch4172更新/33376曝光、全局批8；AdamW LR1e-4、100步LR warmup及cosine、1043步理解ramp、weight decay0.05（1D为0）、clip1。初始与每epoch末相同4train/4heldout探针，每50步checkpoint并保留各epoch终点。没有额外2epoch重建适配。

使用B短测66503的零步checkpoint_initial，而非256步终点，空Adam/0更新。验证所有296张量逐位一致；和A一样严格核对源4rank RNG相同后显式复制到8rank，独立保存转换后的初始化文件和原/新SHA。训练入口严格零步/结构/资产/数据/world检查不放松。源硬件3090、新作业4090，不能声称数值轨迹完全相同。

## 评测和汇总

训练成功后afterok同节点1×4090评测：固定128场景、原官方评分器与11数值/两文本空列校验；自动比较旧C、已完成V4控制，以及届时已有COMPLETE.json的原Unified全量组。B完整训练与A数据/更新预算相同，但结构、参数量、额外语义监督和卡型不同，不是纯单变量对照。

另外提交无GPU的CPU汇总任务，afterok依赖A评测66507与B评测均成功；检查两组11指标及128场景RGB一致，再生成A/B/旧C比较表（如原Unified已完成则加入）。无需为了得到对照表再训练或再次GPU推理。不因A/B完成顺序不同而遗漏最终比较。

提交前运行B真实零步/冻结资产SHA/数据/官方评测版本预检，并复用受影响的初始化转换、完整探针及两结构回归测试；新全量GPU首步仍需以本run收据为准。预计训练约4–7小时、评测另计，仅依据4×3090短测32m59s和本次8×4090资源粗估；数据Query数量与节点负载影响实际时间。

作业号与执行SHA以submission.json为准。提交后仅一次有界状态核对并停止轮询，等用户通知完成后检查。

提交前验证完成：B的296张量与短测零步源逐位一致，源4rank RNG相同并显式扩展8rank，0更新/空Adam、真实资产SHA、1191/8337/128数据和官方评测版本预检通过；9项受影响CPU回归通过，shell语法、Python编译和diff检查通过。尚未核实新全量GPU首步。
