# V4 几何感知初始化：18 节点八卡冻结重建对照

用户确认 18 节点有八张 4090 并要求训练，选择 V4 几何感知 SAM 初始化 + 冻结重建、理解训练 4 epoch。对照为 65665/65666 的 V3 冻结重建组。两组起点、数据窗口、global batch8、理解 LR1e-4、首 1043 步理解 loss ramp、100 步全局 LR warmup/4 epoch cosine、Write=0 和官方评测协议相同。硬件 V3 为八张 3090，V4 为八张 4090；不能宣称逐位一致的训练轨迹。

同一 source 为 attempt06_adapt2_joint4_node17/checkpoint_adaptation_complete.pt，SHA256 2b07922b1b2c3e8f5dfa6c1591951e043bc418874f2cea379588003ddde6f3e3，旧 source execution SHA75f6ce182b4ee79e75485db541daf0ffc85f91a9。复用 short306 + adaptation2086，不重复训练适配，联合 full clock 从0开始。新增几何池化29505参数，理解Adapter/query/panoptic继续训练，其余参数与重建模式冻结。

## 迁移与验证

专用 `--geometric-from`/digest，拒绝旧 V3 resume 或按 optimizer 顺序直接加载。用真实 V3 meta 构造重建旧 optimizer 参数顺序与名称，严格核对所有旧 model key/shape、optimizer family/weight_decay/base_lr/Adam moment shape。旧参数和每个已有 moment/step 按名称迁移，新几何池化参数保持新初始化/空 moments。目标 LR 不由旧 optimizer 覆盖。

继承同一 4→8 rank RNG 规则：0–3恢复source，4–7使用31415+80000+rank。继承数据曝光与训练进度，但清空旧结构的梯度/更新证据，新模块必须在实际首步得到八rank梯度。每步冻结版本/grad检查、固定16窗口Gaussian一致性检查、重建参数与buffer最终指纹检查沿用冻结组。首步checkpoint和FULL_STARTED保存实际GPU证据；没有这些不能把RUNNING当反向成功。

CPU预检读取真实2.2GiB源checkpoint核对checksum、完整参数/Adam映射与新参数29505，并校验源重建指纹等于V3冻结组初始指纹。没有使用GT初始化或重算适配。CPU测试覆盖迁移保留、坏key/shape/group/LR拒绝、源阶段与配置拒绝、初始化梯度及旧warmup/冻结策略。

实际预检通过：637 个共享 tensor、637 组 AdamW 状态完整映射，仅新增 29505 参数；源重建指纹 c44561d98fe0262588293490c09aba39150acc4388d6c3329d42e47c3692822c，与 V3 冻结组初始指纹一致。23 项 CPU 检查通过；Python 编译、shell 语法与 diff 空白检查通过。

## 作业链

新目录 attempt10_geometric_sam_frozen_node18_8gpu；不可变代码 `/space/mawb/ssst_gfm_geometric_frozen_code_20261016`。训练32CPU/192GiB/八张4090，最多72小时，4172理解更新。训练成功后单GPU导出同一128场景并走原生官方评分，11项必需指标、两项text留空。深度若NaN仍显式无效并报错，不改官方协议。比较任务等待V3冻结评测65666与V4评测结束后汇总native结果。

具体job id、code SHA与提交请求以 attempt10/submission.json 为准。提交后只做一次分配状态核对，停止轮询，等待用户通知。不取消/重启65661或65665，不修改其执行快照。
