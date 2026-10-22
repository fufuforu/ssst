# P1024iso C/D：关闭 Query→FM CA 的配对消融

用户于2026-10-22授权提交C/D两组。代码快照 `/space/mawb/ssst_point1024_nofmca_ablation_code_20261022_v1`，来自已完成No-CA快照 `046d08d54705d1a697edcc3495ce818a8726d014`。
运行目录 `/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_nofmca_ablation_20261022_v1/{C,D}/{smoke,train,eval}`。
不改其它快照，不重提旧作业，不提交LR扫描或几何LR实验。

|组|Query→Anchor CA|Query→FM CA|c/s几何权重|
|---|---|---|---|
|A，已完成70026/70042|开|开|学习式Anchor attention R|
|B，已完成70162/70183|关|开|纯几何softmax|
|C，本轮|开|关|学习式Anchor attention R|
|D，本轮|关|关|纯几何softmax|

## 实现与对照边界

DynamicObjectLayer 新增 `_read_image(q,image)` hook，默认仍执行原 `image_attention(q,image_ln(image))`，保持A/B行为。
新 C/D layer 的 hook 返回 `(zeros_like(q),None)`，**保留 `norms[1](q1+0)`**；不删除Query残差后的LayerNorm。
移除4层各自image_attention的Q/K/V/O共32个权重/偏置张量。image_ln仍保留原权重但不执行，无梯度；DDP沿用find_unused_parameters=True。没有重新初始化其它层。
C保留anchor_attention；D使用已有GeometricOnlyObjectLayer，R仍定义为geometry_bias.softmax(-1)，并按原函数更新c/s；不保留未定义R。
FM仍用于理解query初始化、stuff seed、逐Gaussian的FM lifting与融合。**这不是移除语义分支或全部FM信息。**
所有原模块先按原顺序构造，再移除禁用attention，以保持保留参数及新adapter的初始化RNG顺序。

## 冻结配方

- 从原A4零步checkpoint重新训练，SHA `3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744`，空Adam、8rank RNG；不使用A/B已训练终点，不使用smoke权重。
- 8337训练窗/1191场景、全局8（8卡×1）、4 epoch /4172 updates；2 context+4 novel、256²。
- 训练窗manifest SHA `8fe72589a2b8f1b52fa517cde39a0fce37d104adc2ee3a28585d1747292313f1`。
- 1024重建anchor /65536 GS；来源像素256维特征→1024；FM超点query初始化；理解256维；4个残差adapter64、隔离、Write=0。
- L6/L12 `[1/3,2/3]`；AdamW原参数与weight decay分组；所有参数LR 1e-4、warmup100、cosine→1e-5；理解权重0.05×min(1,(step+1)/1043)。**不恢复旧V4几何LR分组。**
- 全参数统一clip1、每50步checkpoint、epoch probes与epoch权重保留策略均沿用基线。
- 可调度节点池12/13/14/17/18（3090/4090），单节点；训练/评测8卡、smoke1卡。不使用节点列表错误地要求同时分配五台。

## 提交前实际验证

初验6项CPU测试通过，最终含C/D smoke收据身份与错误迁移拒绝的7项回归均通过（21.125秒）：No-FM image输入不变性/无该输入梯度、与原layer显式零readout参考完全相等、C/D真实core两项实际loss的方向隔离、B的几何权重及真实core回归、原零步迁移拒绝规则。
CPU全尺寸trainable core审计（冻结骨干用身份stub）：

|组|逐值继承原零步张量|精确移除张量|新增adapter张量|可训练参数量|
|---|---:|---:|---:|---:|
|C|625|68（36旧重建初始化器+32 FM CA）|16|192544624|
|D|593|100（36旧重建初始化器+32 Anchor CA+32 FM CA）|16|191491952|

C/D新adapter初始哈希相同，末层保持零，残差初始identity；保留原权重逐值相等。
重建core、点采样、FM初始化、损失、训练数据与学习率policy、官方评分脚本与已完成B快照逐字节一致。
CPU测试使用缩小渲染以验证真实梯度路径，不能替代GPU 1024anchor/65536GS验证。
实际CPU收据：运行根目录 `CPU_VALIDATION.json`，各stage `PREFLIGHT.json`。

## 自动依赖链与失败处理

每组独立 `smoke afterok→train afterok→eval`，无跨组等待训练依赖。
smoke是**两个optimizer step**，单窗口、每步累积8个microbatch；实际1024anchor/65536GS、2+4、256²。
复用专用smoke的方向反传隔离、分组梯度/四adapter梯度、形状/父索引别名、耗时/显存峰值、首步参数哈希与预算收据；不同架构与精确迁移张量数单独校验。
前期adapter隐藏层梯度为零是预期行为，不改零初始化；D早期读层adapter可以没有梯度，L12必须非零。
训练前严格重验 `SMOKE_PASSED.json` 与预算、代码commit；不会从smoke恢复权重/优化器。
任一smoke失败写 `JOB_FAILED.json`，afterok与kill-on-invalid-dep阻断下游，不静默重试。
训练保留既有I/O bootstrap策略：STARTED前的明确I/O超时可有界重试，STARTED后不自动重启训练；与A/B一致。

训练成功后冻结latest硬链接，核对4172 updates、8rank首步更新一致、训练数据/架构/CA标记、执行commit，才执行原导出器及固定SIU3R官方1860评分。
评测比较输入RGB与GT清单必须与A/B逐窗一致。每组写 `EVALUATION_VERIFIED.json`。
后完成的评测在自身 `comparison/` 自动生成A/B/C/D四组对比表；没有循环等待另一组，也不增加实验。

提交后只做一次有界squeue核对全部6个作业。RUNNING不代表GPU smoke通过。等待用户通知完成后再检查实际结果、逐类PQ/TP/FP/FN，不把smoke称作正式指标。

## 审查入口

- `tokengs/models/point1024_nofmca.py`：C/D模型、关闭FM read、保留/关闭Anchor CA；
- `tokengs/models/dynamic_panoptic_controller.py`：默认行为不变的image hook；
- `scripts/point1024_isoadapter_initial_state.py`：精确零步迁移与移除白名单；
- `scripts/train_unified_query_v1.py` / `scripts/export_unified_query_checkpoint.py`：新架构和身份断言；
- `scripts/prepare_point1024_nofmca.py` / `tests/test_point1024_nofmca.py`：真实CPU验证与独立运行目录；
- `scripts/run_point1024_nofmca.sh` / `scripts/point1024_nofmca_pipeline.py`：依赖门禁、原配方训练、原官方评测与对比；
- `scripts/submit_point1024_nofmca.py`：只提交本次C/D六个作业、仅一次状态核对。
