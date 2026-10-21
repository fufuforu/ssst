# 冻结重建、只训练理解：11节点八卡对照

用户提出11节点也有八卡，增加冻结重建对照。实际核对3dimage-11 IDLE、8×3090全部可分配；不修改/取消13节点warmup作业65661。

## 控制变量

来源与warmup组完全相同：checkpoint_adaptation_complete.pt，已完成适配2epoch/2086更新，joint_updates=0、short_updates=306，SHA256 2b07922b1b2c3e8f5dfa6c1591951e043bc418874f2cea379588003ddde6f3e3。不重跑适配、不使用已退化的joint4终点。

新增理解训练4epoch/4172更新，1191场景/8337窗口，8GPU×microbatch1×累积1=有效batch8。理解基础LR1e-4、原100步LR warmup/4epoch余弦、首1043更新理解loss beta线性升1，与65661一致；主要差异是重建完全冻结，而65661重建/几何LR为1e-5/1e-6。冻结组rec/geometry optimizer LR记录为0，参数requires_grad=False、grad=None，AdamW跳过，已有moments保留但不更新。

仅understanding（语义Adapter及mask投影）、query_initializer、panoptic（Object交互/分类/归属读出）可训练。重建Adapter、anchor内容/视觉投影、重建tokens/decoder、位置/半径细化及Gaussian头都固定。Omega/SAM仍冻结，Write=0，模型结构/损失公式/数据/官方评测不变，不增加文本分支。

4→8rank RNG映射与65661一致：rank0–3沿用来源状态，4–7为31415+80000+rank。DDP在冻结标记之后构建，避免reducer登记已冻结参数；构建前后保存/恢复RNG。optimizer在冻结前构造并加载完整源状态，保证param group映射和moments正确。

## 完整冻结与证据

1. 每次model.train后，重建子模块单独保持eval；理解子模块保持train。防止仅冻结参数却遗漏dropout/缓冲状态。
2. 冻结控制runtime flag仅在该对照生效；decoder每层detach重建tokens/mu/rho，去掉Write=0遗留的零值autograd连接，避免固定重建块挂在理解梯度图上。输出数值不改变，Object仍正常读取固定重建特征。Write非零时直接拒绝该控制。
3. 每次更新检查冻结参数无grad、版本不变；不以LR=0代替真正冻结。重建loss保留为诊断常量，实际梯度仅来自可训练理解路径。
4. rank0记录重建全部参数/持久buffer的初始state SHA256，终点复核完全相同；resume必须匹配原初始SHA，不能覆盖原证据。
5. 每次固定8训练＋8验证monitor额外记录Gaussian张量SHA，与初始同场景比较；若重建输出变化则显式失败，避免把隐性前向影响当作冻结成功。monitor beta=1的诊断口径沿用65661。
6. 最终评测加载checkpoint后还检查冻结完成收据及state SHA一致。固定权重导出no_grad，不更新参数。

CPU验证14项已通过：冻结权重/AdamW moments/模块模式/输出保持，buffer变化可被hash检测，Write非零拒绝，同一适配来源metadata校验；实际PointSeededAnchorDecoder的1024anchor控制验证理解参数变化不改变重建token/几何，但Object query仍有梯度；并保留warmup和batch8采样/缓存等检查。首轮toy测试用了2个anchor，触发原scene_normalization要求1024的断言，已将toy输入改为真实1024anchor约束后通过；没有改生产模型约束。真实源预检写preflight.json，编译、shell语法、diff检查通过。

## 运行与评测

新recipe omega_gfm_semantic_pool_v3_adapt2_frozen_recon_understanding4_uwarm1_v1。

运行目录：/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt09_frozen_recon_node11_8gpu

代码快照：/space/mawb/ssst_gfm_frozen_reconstruction_code_20261016

资源11节点8×3090、32CPU、192GiB、72小时。理解训练成功后afterok独立1GPU官方128评测；Evaluator与13列11值口径不变，NaN保留无效标记，text两列留空。评测完成后与65662（warmup评测）的afterany依赖CPU任务生成旧C/direct4/原adapt2+joint4/warmup/frozen五模型原生表，完整核对同128场景RGB身份；缺少完整原生结果则拒绝伪造比较。

比较目录：/space/mawb/ssst/group_plus/dual_stream_training_v1/comparison_joint_warmup_vs_frozen_recon_20261016。

作业号以attempt09/submission.json为准。提交后单次核对调度并停止轮询，等用户通知。RUNNING仅说明分配资源，不能称GPU首步或完整冻结输出检查已通过；后续看FROZEN_RECONSTRUCTION_INITIAL、FULL_STARTED、monitor Gaussian SHA和FROZEN_RECONSTRUCTION_COMPLETE。

## 已提交

训练65665；官方评测65666 afterok:65665；五模型比较65667 afterany:65662:65666。训练/评测代码SHA1ee41b0229ae53840b5ce84111bf6a5c4cd2a10d已推送；运行与比较目录submission.json保存了完整资源及依赖收据。

提交后单次核对：65665 RUNNING，11节点已分配8×3090、32CPU、192GiB；65666和65667 PENDING/Dependency。13节点65661没有被修改或中断。此时仅确认资源分配，未宣称GPU首步或冻结指纹已通过，随后停止轮询。不要重复提交或从short306重跑适配。
