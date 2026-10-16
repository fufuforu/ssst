# V4：仅缩放理解→重建梯度的单变量对照

用户授权：完成“理解参数接收完整理解梯度，传向重建参数的理解梯度乘0.05”对照。

## 科学合同

- 相同V4架构、相同0步未训练权重（SHA b9d7bd597cc2bd0d2439a1eae3771760f2be55241a2db1c4ef3879b08a75a03e）、空Adam及8rank RNG。只加载官方冻结Omega/SAM，不重复适配、不继承训练后模型。
- 4epoch/4172更新/global batch8，原1191场景8337窗口池及数据顺序不变；LR重建1e-4/几何1e-5/理解1e-4，首100步全局LR warmup+cosine。
- beta=min(1,(update+1)/1043)。重建参数梯度为dL_rec + 0.05*beta*dL_under；理解参数梯度为beta*dL_under。不是整体loss乘0.05，理解自身loss系数仍1。
- 梯度分组沿用隔离对照：understanding./query_initializer./panoptic.为理解，其余requires_grad参数为重建。Write=0、前向交互、初始化、membership、mask组装均不变。
- 沿用隔离对照autograd.grad与显式FP32 SUM/world分布式归约，保留全局unused→None及局部unused→0语义。仍为全局clip1，因此两参数组的实际更新尚有裁剪耦合；不能宣称完全独立。
- 新CLI --isolate-gradients --reconstruction-understanding-gradient-scale 0.05表示启用显式路由框架；并非完全隔离。配置recipe/gradient_isolation字符串、每步日志和首步同步收据均记录新路由。

## 执行与验证

- 新运行：/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt14_geometric_sam_routed005_node13_8gpu。
- 独立不可变快照：/space/mawb/ssst_gfm_v4_routed005_code_20261016。
- 使用13节点8×3090，与普通联合65761、隔离65831保持同型硬件。
- 真实源预检已通过：649个模型张量/196010160可训练参数，重建186881600、理解9128560；数据与优化器分组匹配。
- 梯度公式、共享非重入checkpoint、理解梯度不随路由系数改变、microbatch累积、真实双rank同步/局部与全局unused、旧隔离与加权策略回归：以实际测试结果及提交收据为准，不把RUNNING称GPU验证通过。
- 训练afterok接官方128评测（协议/阈值不变、11项有限要求、2项文本空白），比较原普通joint/隔离/整体0.05/冻结/旧C。
- 具体作业号以submission.json为准；只核对一次分配状态后停止轮询，等待用户通知。

## 判读

优先比较隔离65831（23.194PSNR、18.98%mAP、53.62%PQ）：能否保持重建并改善理解。再与整体0.05（21.910PSNR、19.85%mAP、58.79%PQ）比较。如果重建再次明显下降且理解无收益，优先保留隔离作为重建基线；不凭一个系数断言不存在有益语义约束，不把结果唯一归因于方向冲突（尚未实测同参数梯度cosine）。本轮只授权一组新训练。
