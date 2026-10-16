# V4 从头联合训练：损失梯度隔离对照

用户明确授权按建议验证梯度隔离。新attempt12_geometric_sam_isolated_node13_8gpu，13节点8×3090、32CPU、192GiB，独立不可变代码 /space/mawb/ssst_gfm_v4_isolated_code_20261016。与已完成65761/attempt11对照，前向结构、权重初始化、batch8、窗口/顺序、4epoch4172步、首1043步理解loss ramp、LR1e-4/1e-5/1e-4、100步全局LR warmup+cosine、AdamW、global clip_norm=1、Write=0及官方协议保持一致。

## 初始化

精确复用65761训练前的checkpoint_initial.pt（0 short/0 full/0 adaptation、空AdamW、8 rank RNG），不使用训练后的模型。预检源SHA b9d7bd597cc2bd0d2439a1eae3771760f2be55241a2db1c4ef3879b08a75a03e，源执行SHA5a3ff5fa70deb56c8d6bafe390d66f6171b50424。649张量及optimizer group注册/形状/学习率严格校验，初始权重与rank RNG复用。初始化monitor保留并恢复RNG，确保重新渲染初始结果不移动训练随机数起点。

这属于从同一个未训练初始状态重新实验，未继承任何适配或训练后的参数/moment。官方Omega/SAM仍冻结；重建186881600参数和理解9128560参数都保持可训练，无新增结构或参数。

## 唯一优化变量

- 重建参数（除understanding./query_initializer./panoptic.以外的全部可训练参数）仅接收 loss_recon 的偏导。
- 理解参数（以上三个prefix）仅接收 beta * loss_understanding 的偏导。
- 不detach前向token/特征，不改变Query读取重建信息，不引入新的loss项。

用autograd.grad分别求导。此路径不构造DDP，因为不能依赖DDP自动同步autograd.grad；显式按固定参数顺序、FP32 bucket执行SUM/world。局部unused在全局used时贡献0；全局unused保持grad=None，以避免AdamW无意义衰减/更新。梯度累积先本地累积、再每次optimizer更新平均一次。其它模式仍走原DDP及backward，不修改历史执行快照。

保留原全局clip_norm=1是为了控制变量；裁剪系数仍可能同时受两分支梯度范数影响，所以该实验隔离的是原始损失梯度的方向/参数归属，不意味着两分支优化完全独立。监测记录两类裁剪前范数，便于后续识别这一残余耦合。

首个实际训练更新检查两类梯度有限且非零、新几何池化梯度、并对所有可训练参数跨八rank计算完整SHA验证更新后副本完全一致。写ISOLATED_GRADIENT_SYNC_FIRST_UPDATE.json；缺失收据时不得声称GPU同步已通过。首次GPU反向若失败则停止，不以RUNNING代替验证结论。

## 测试与监测

解析梯度用例验证理解loss强度变化不改变原始重建梯度；验证microbatch等价、全局unused不衰减、共享nonreentrant checkpoint双反向、前向loss值不变。真实双进程CPU Gloo验证局部unused/全局unused、分桶平均、optimizer更新后副本一致。默认沙箱不能建立Gloo本地通信，测试在额外权限下仅绑定loopback执行。

额外诊断仅用于监测：固定8训练/8验证每100步记录深度非有限比例及GT有效位置上预期导出为0的深度比例（原normalized depth阈值7.5e-5，对应既有/0.15及16bit PNG四舍五入）。不改训练损失，不修改官方深度对齐或评分。正式评测仍为同一128场景、13列11值、text留空。

训练成功后自动官方评测，再比较普通V4从头joint、隔离V4从头joint、V4适配后冻结及旧C参考。作业ID见attempt12/submission.json。提交后单次核对分配、停止轮询，等待用户通知。

实际CPU检查28项通过（含真实双rank Gloo、共享checkpoint双反向与既有策略回归）；真实0步初始checkpoint预检通过，source649张量、186881600重建/9128560理解参数、空optimizer、8rank RNG。Python编译/shell语法/diff检查通过。
