# 重建anchor Q+P：仅代码实现（2026-10-21）

本轮用户授权：实现“共享可学习Q × 当前场景超点P”的重建anchor扩展，理解query保留256维；**不提交作业**。本轮没有调度操作、没有GPU测试、没有改动此前运行快照或作业。代码独立位于 `/space/mawb/ssst_recon_qp_code_20261021_v1`。

## 建议阅读顺序

1. `tokengs/models/reconstruction_qp.py`：可学习Q bank与完整笛卡尔积展开，最核心实现。
2. `tokengs/models/reconstruction_qp_core.py`：只替换重建初始化，理解侧沿用当前256维基线。
3. `tokengs/models/superpoint_dual_core.py`：通用初始化hook，以及anchor/理解query数量解耦、Gaussian父索引。
4. `tokengs/models/superpoint_structural_variants.py`：新架构注册、K配置与计数日志；新架构默认L6+L12监督。
5. `scripts/recon_qp_initial_state.py`：显式原始零步迁移，只有Q bank是新增参数，其余张量严格恢复。
6. `scripts/train_unified_query_v1.py` / `scripts/export_unified_query_checkpoint.py`：训练、resume、checkpoint身份及导出接线。
7. `tests/test_reconstruction_qp.py`：展开、学习/持久化、非等量交互、原模型等价性及初始化拒绝检查。

## 实际公式和维度

设当前场景有N个超点：

- 现有重建初始化器输出 `[B,N,256]`，经原有 `reconstruction_seed_projection` 得到 **P `[B,N,1024]`**。
- 新增 **Q `nn.Parameter[K,1024]`**，默认K=20、正态初始化std=.02。Q在场景间共享，随optimizer持续学习，不在forward重新创建。
- **H[i,k] = P[i] + Q[k]**，直接相加，无额外LayerNorm/门控/池化/截断；输出 `[B,N*K,1024]`。
- 排序为超点优先：P0Q0,...,P0Q(K−1),P1Q0,...。中心和半径按此顺序重复K次，初始几何不缩小、不加随机偏移；后续沿原decoder逐层更新。
- 每个anchor仍生成64个Gaussian，因此 **anchor=N*K、Gaussian=64*N*K**。
- 理解初始化器仍只输出N个256维thing query，加2个原有stuff query；理解宽度、query数量、来源匹配、损失、decoder交互层均保持原有版本。无Q+P理解query扩展、无1024维理解分支、无IoU头混入。
- 理解query读取扩展后的anchor证据；结构和初始化参数保持不代表理解指标提前得到保证，仍需正式评测。

例：K=20、N=50 → 1000个重建anchor、64000个Gaussian；理解仍50个256维thing query+2个stuff query。N不固定为50，按每个场景实际超点数动态展开。原来的max_superpoints限制只约束N，不会暗中把N*K裁剪回512。

保留显式映射：`anchor_to_superpoint`（anchor→原超点）、`anchor_slot_ids`（anchor→Q序号）、`gaussian_parent_anchor`（GS→展开后anchor）、`gaussian_parent_query`（GS→未扩展的原超点/理解query）。理解loss仍使用原来的 `superpoints` 来源ID和N个query，不把新增anchor当成新的实例候选。

## 启用及checkpoint

架构CLI：`--architecture dual_superpoint_recon_qp`；bank大小：`--recon-qp-slots 20`。API为 `build_structural_model('dual_superpoint_recon_qp', opt, config, recon_qp_slots=20)`。

默认监督层L6/L12，权重1/3、2/3；显式RG_SUPERVISED仍可覆盖。Object交互仍在6/8/10/12，Write=0。训练预算尚未执行，由之后显式传入的epochs/max-updates决定；本轮没有为此创建run或训练权重。

从原始A4零步起步时必须显式给 `--initialize-recon-qp-from-legacy-zero`，并提供原zero路径及SHA。该路径要求：旧架构、资产/数据/config、global batch8匹配，0 updates、空Adam、原始8-rank RNG完整；若用1/2/4卡，会在收据中明确记录取前world个RNG。**所有旧张量键及形状必须完全匹配，仅允许新增 `core.reconstruction_qp.learned_q`**。正常restore没有放宽；不能把训练好的旧A8当零步无声重置Adam。

Checkpoint meta存储bank大小、监督层和损失权重；export按meta重建相同K并严格加载。Q参数属于reconstruction组，纳入原AdamW/clip/保存与恢复；首步单独检查其梯度非零。

日志新增 `num_superpoint_queries`、`num_reconstruction_anchors`、`num_object_queries`、`num_stuff_queries`、`num_gaussians`、`recon_qp_slots`。另外保留每个microbatch的 `microbatch_query_counts`，避免4卡/global batch8时只记录两个场景的均值，无法恢复个体数量分布。

## 本地验证

- 5项模型CPU测试通过，覆盖K=20,N=50完整1000展开及N=251不截断；Q梯度、跨场景持久化、optimizer更新与小state checkpoint往返；K=1/Q=0 seed等价；严格零步迁移；真实12层core在合成数据/8×8测试renderer下anchor数与object query数解耦、前反向、Q梯度、理解来源匹配及参数分组。
- 补充等价性回归通过：同一个模型的相同权重、K=1/Q=0，新路径与实际原始core.forward的Gaussian、membership、类别、semantic及region输出一致。
- 1项统计工具CPU测试通过：4卡每rank两个microbatch时，从逐场景数组恢复8个真实场景计数，拒绝仅用旧标量均值伪造分布。
- 真实core零步布局审计通过：原始zero SHA `3c7f1860...`，**693个旧张量全部逐值相同**，理解相关代码及初始化张量保留。唯一新增参数Q形状20×1024，20480参数；总可训练参数193819761。结果在 `docs/recon_qp_cpu_zero_audit.json`。
- 上述core审计冻结骨干使用身份stub，CPU前反向用合成数据及测试renderer；**未验证GPU、真实256×256渲染显存、训练收益或理解指标**。

## 历史超点/query数量统计

历史train_rankXX.jsonl已逐step记录 `num_thing_queries`。当前原生双分支的这个字段等于初始超点数/thing候选query数，不是真实物体数，不含2个stuff query；基线重建anchor数也等于它。

本次按读取时各文件字节长度做一次有界快照，取所有8 rank都有记录的完整iteration；所选运行均world8/global batch8，因此每rank每step恰好一个场景，可恢复真实每场景分布，没有把均值冒充计数。snapshot读取时间：2026-10-21T10:45:10.196711+08:00。

| 来源 | 完整iteration数 | 场景样本数 | 场景均值 | 中位数 | P5–P95 | 最小–最大 |
|---|---:|---:|---:|---:|---:|---:|
| A4（完整4轮） | 4172 | 33376 | 57.307 | 53.0 | 33.0–97.0 | 15–251 |
| 旧A4→A8（新增4轮） | 4172 | 33376 | 57.308 | 53.0 | 33.0–97.0 | 15–251 |
| 69254日志快照 | 10646 | 85168 | 57.297 | 53.0 | 33.0–97.0 | 15–251 |
| 69481日志快照 | 6913 | 55304 | 57.299 | 53.0 | 33.0–97.0 | 15–251 |

因此典型值约每场景57.3个、median53，P5–P95为33–97，观察范围15–251。8场景global iteration合计均值约458.4；不是把458个query放进同一个场景。

K=20时按这份数据：median为1060个anchor/67840个GS，P5–P95为660–1940个anchor，观察最大为5020个anchor/321280个GS；这是动态数量，不固定1000。

逐iteration八个rank的具体数、合计和均值在主工作区 `docs/query_count_stats_20261021/run*/per_iteration_counts.csv`；总汇在 `docs/query_count_stats_20261021/all_runs.json`。统计脚本为本快照 `scripts/summarize_query_counts.py`。新日志的microbatch数组亦受支持；老4卡只有均值的日志会明确拒绝恢复个体分布。

**本轮只交付代码与统计，未提交/取消/检查任何Slurm作业。现有作业沿此前交接继续，勿根据旧PENDING状态重提。**
