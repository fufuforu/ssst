# SAM 几何感知 Query 初始化 V4

用户本轮明确要求只修改初始化模块。实现为独立 V4：`siu3r_object_locus_gfm_dual_stream_v4`，配置 `train_siu3r_object_locus_gfm_dual_stream_v4`。V3 与正在运行的不可变训练快照不替换，不提交新作业。

## 实际计算

1. 使用已有 SAM 区域和冻结 Omega 预测点云，不使用 GT、target 视图、SAM latent 或 ESAM 权重。
2. 区域内只保留有限三维点，最多确定性等间隔取 256 个源像素；小区域保留全部有效点。按像素中心坐标双线性采样语义 Adapter 输出 F_sem，梯度可回到 Adapter。
3. 点坐标减区域均值，除以区域最大轴跨度（下限 1e-6）。只使用 detach 后几何。相对位置经 3→64→64 MLP 得到局部形状特征；区域 max 得到整体形状特征。
4. 拼接每点局部和整体形状，经 128→64→1 MLP 及 `2*sigmoid` 得到点权重。池化结果为 `mean(weight * semantic_feature) + Linear(global_shape)`，然后沿用 V3 query 投影、中心/尺度编码和 LayerNorm。

这一实现借鉴 [ESAM GeoAwarePooling](https://github.com/xuxw98/ESAM/blob/main/oneformer3d/geo_aware_pool.py)，以现有语义 GFM 替代其稀疏点云骨干输出，独立实现，无额外依赖。没有引入新的点云 encoder。新参数只有初始化模块中的 local_shape、point_weight、shape_projection。

候选分数排序、同视图 IoU 去重、跨视图关联规则及阈值、最多 80 proposal + 至少 20 learned fallback、100 query 总数不变。关联使用的池化向量已经改变，因此实际合并结果可能改变，不能声称候选身份严格相同。没有加入新关联头或监督。

重建初始化、双 Adapter、L6/8/10/12 交互、Write=0、Gaussian/Mask 输出、loss 与官方评测数学不变。保持原语义特征输入；不回到 raw GFM 池化。当前接口未传入几何 confidence，所以只剔除非有限几何，尚未加入置信度加权。均值/尺度仍由原 V3 规则生成。

## 入口及权重边界

模型注册与通用配置已接入。现有诊断训练脚本可用 `--architecture gfm_v4` 选择独立 recipe，记录 query_pooling=geometric_sam_v4 和新池化权重的梯度/更新证据；对应固定权重导出按该标识选择 V4。没有实际启动训练。

旧 V3 参数名称与形状全部保留，只增加初始化参数。不能直接把 V3 checkpoint 当 V4 resume，尤其不能按旧参数顺序加载 AdamW。脚本明确拒绝把旧 transition/staged/warmup 入口用于 V4。若后续从现有模型做初始化对照，应单独实现并审计按名称迁移共享权重与 optimizer moments 的边界，不重复训练已完成的适配；本次没有迁移任何 checkpoint。

## 验证边界

测试覆盖几何平移/正尺度不变性与形状敏感性、单像素区域、无效点、无候选 fallback、确定性有界采样、语义 Adapter/池化各分支的反向梯度、V3 原池化逐值相等、完整模型 meta 构造与参数差异白名单。

仅 CPU 单元验证与 meta 模型构造不能证明完整 GPU 前向/反向通过，也不能证明 mAP/PQ 提升。当前两组 V3 对照仍为原代码运行，不在本轮轮询。
