# GFM 双 Adapter v3：SAM 池化语义特征

用户明确要求改为语义Adapter之后池化，再重新运行。新模型 `LocusGSGFMDualStreamV3` / `siu3r_object_locus_gfm_dual_stream_v3`，配置 `train_siu3r_object_locus_gfm_dual_stream_v3`。

当前计算：

```
context RGB -> 冻结Omega GFM
                |-> 重建Adapter -> K/V -> 重建token -> Gaussian
                |-> 语义Adapter -> F_sem -> K/V -> 理解query
                                     |-> SAM Mask加权池化 -> query投影+几何编码 -> 初始query
                                     |-> 原逐Gaussian视觉支撑
```

重建anchor的点云位置/半径及对应原始GFM视觉内容初始化不变；仅理解query的池化来源改为同一个语义Adapter输出 `F_sem [B,2,256,128,128]`。其后独立query投影为LN256→Linear256→GELU→Linear256，再加几何编码及原query LN。SAM仍只生成自动候选区域，不提供GT提示或直接复制其latent token。

SAM Mask在原点云256网格计算去重/3D支持，以area覆盖权重映射到语义特征网格，先池化F_sem再投影。候选池化/合并得到的语义向量**不detach**，理解损失可以经初始化路径更新语义Adapter；只对离散跨视图关联的相似度判断detach。跨视图关联、最多80候选＋至少20回退槽位、100thing＋2stuff、L6/8/10/12、Mask读出、训练loss、Write=0均保留。重建与语义Adapter仍独立，TokenGS encoder仍移除。

## 修复上轮真实启动失败

65174最终FAILED / 1:0，2分49秒，未进入第一次训练更新，无FIRST_UPDATE或新checkpoint。rank0在迁移权重/optimizer之后写ARCHITECTURE_TRANSITION.json时，metadata和plan都包含 `retained_optimizer_states`，`dict(**metadata,**plan)`引发TypeError。旧日志和FAILED保留于attempt03_gfm_v2；不能称旧raw-GFM版本已训练或经过GPU反向验证。

本版以 `transition_receipt` 显式合并收据，允许且校验这一重复计数相等，出现矛盾值直接报错。已加入复现该错误的回归测试。源第50步文件完好，SHA仍为 `f711044d0241c6fd949353e45bb5a9b92630409dd4df259239084402bc84d3ea`。

## 迁移和重新验证

继续从原65167安全保存的50步状态迁移，而不是从失败65174恢复（它没有新训练状态）。共享189,530,347参数及对应AdamW状态沿用；新模块6,450,308参数新初始化，总可训练195,980,655。原始GFM池化的V2保留为历史代码路径，新V3独立注册，避免混淆。

新结构从迁移后的初始状态重新记录monitor，再做128/最多256次新结构短验证（沿用来源时钟时计数178/306）。门槛与全量4epoch策略保留；额外梯度检查单独记录 `semantic_query_projection`，不再错误标为raw GFM query projection。新投影的AdamW状态从0开始，其他共享参数保留来源50步moment；有效batch8不变。

新运行目录 `/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt04_semantic_pool`；固定18节点4张4090，每卡microbatch1、累积2。新启动脚本 `scripts/launch_gfm_semantic_pool_v3_node18.sbatch`。保留已有冻结SAM候选/标定缓存，不缓存可训练F_sem或query。

19项CPU检查通过：旧初始化6、训练策略与缓存4、GFM组件5、新semantic pooling/梯度/投影维度/迁移收据回归4。新模型与真实源50步checkpoint形状白名单核对通过，见新运行的 `preflight.json`。只有新运行出现FIRST_UPDATE且梯度/参数更新证据满足，才能称新版本GPU反向通过；尚无新模型质量结论。短验证通过才自动全量，失败保存现场停止。提交并确认启动后停止轮询。
