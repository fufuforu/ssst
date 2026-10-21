# Point1024 / source256 / isolated residual adapters：仅代码准备

用户确认先detach输入再加可学习adapter、重建种子按精确视图/像素来源采样，并进一步要求理解query改用fm采样到点云的特征。用户本轮明确**先不提交作业**。本快照独立于69787及其自动16epoch流水线；本轮未查询、更改、取消或提交任何Slurm作业。

代码：`/space/mawb/ssst_point1024_isoadapter_code_20261021_v1`。
模式：`dual_point1024_isoadapter`；预算和所有身份字段见 `point1024_isoadapter_experiment.json`。

## 初始化与预算

8337窗/1191场景，2 context+4 target，256×256。目标8卡×1/global8，4epoch=4172updates；LR1e-4、warmup100、单次cosine到1e-5；理解权重.05×min(1,(step+1)/1043)。w实际缩放理解梯度，并非仅用于日志；不影响隔离是否成立。

几何原样复用 `seed_anchors`：每视图保留置信度上80%的有效点，在48×48网格选置信度最大像素，合并候选后3D FPS1024；半径为采样种子间第6邻居距离，clamp[.01,.3]×候选点云尺度。点数不足时按原函数重复；没有增加新的三维去重操作。该函数对像素索引unique不等于跨视图XYZ去重。

初始化直接取种子返回的视图/像素索引，从原共享2048→256投影后的32×32特征图双线性采样256维向量，再用原 `reconstruction_seed_projection` Linear256→1024生成token。不是从融合点数组用原始像素索引硬gather。无显式XYZ/RGB种子内容，无可学习基底、无超点特征相加；原函数虽然返回RGB，但这里明确丢弃。

1024anchor、每anchor64GS，共65536GS。原12层重建decoder、GFM K/V memory、GS头、canonical重建loss保持；监督L6/L12权重1/3、2/3。与A4相同指数据和训练预算相同，不表示监督层或初始化架构相同。67952旧point1024包含可学习基底及几何/RGB内容，本版不能称为与其种子逐项相同。

理解沿用原融合点云几何、超点标签、初始化器结构与动态N个256维thing query+2stuff，不固定58。**初始化特征已改为理解fm**：复用 `cloud['source_indices']` 定位同一批源视图/像素，从256维semantic memory双线性采样，再用原 `source_to_point` 索引按点均值融合，得到[N,256]理解点特征，送入原理解初始化器完成邻域编码/超点聚合。**不对fm采样特征detach**，LU经query初始化路径训练fm；原重建 `cloud['features']` 不被覆盖，也不再进入理解初始化。4层object decoder仍按L6/8/10/12读取，image memory仍原256×256。

几何、融合索引和匹配关系保持，query特征数值会改变。此版不新增投影或理解点特征网络，只复用已有semantic memory。

## 完整梯度边界

每层独立：`x=tokens_L.detach(); tok_u=x+adapter_L(x)`，adapter=Linear1024→64/GELU/Linear64→1024，末层weight和bias均0，共528640参数。不能在残差相加后detach，否则adapter也被冻住。

为避免理解梯度经其他路径回流，四个读出层的mu/radii/ell、子GS特征构造使用的GS全部属性和mu/radii、mask renderer使用的GS全部属性均detach。panoptic.encode_token及其后的理解投影保持可学习。共享gfm_projection仅经重建种子路径获LR梯度；理解初始化改读独立semantic memory，其输入不detach。

原RGB/重建GS和12层state.tokens始终保留原计算图供LR使用，不能对它们全局覆盖detach。通过重建Write=0已有能力先计算12层reconstruction-only states，再顺序执行4个理解读出，结果挂回对应state；LR始终读取原重建token。

LR：更新共享投影、种子256→1024投影、reconstruction memory、12层blocks/几何头和GS头。
LU：更新understanding initializer、semantic memory、panoptic controller/分类/mask头、4个residual adapter。

零初始化adapter第一步只有末层可获得非零梯度，隐藏层第一步梯度为0属正常；原controller的零初始化anchor输出投影也可能让较早层adapter第一步无梯度。因此训练首步只要求L12 adapter有非零梯度，其余逐层记录。测试会打开原零读出投影来验证4个adapter在有效路径下均可学习，防止零初始化掩盖泄漏或断路。

## 代码与checkpoint

- `tokengs/models/point1024_isoadapter_model.py`：原始像素精确采样、fm按原索引采样融合、初始化、隔离mask渲染、新模式。
- `tokengs/models/point1024_isoadapter_core.py`：4个adapter、四层隔离读出、子GS隔离、原重建图保留。
- `scripts/point1024_isoadapter_initial_state.py`：显式原始零步迁移。严格资产/data/config/8rank真零步/空Adam；仅删除36个不再使用的重建初始化器张量，新增16个adapter张量，保留657个张量逐值恢复。正常restore不放宽。
- 训练入口支持 `--architecture dual_point1024_isoadapter --epochs 4`，首次原始零步迁移显式传 `--initialize-point1024-isoadapter-from-legacy-zero` 加原路径/SHA。其他epoch预算与short-validation拒绝，未来两步GPU smoke可用epochs4/max-updates2。
- 架构身份升级为 `OMEGA_POINT1024_SOURCE256_FMQUERY_ISOLATED_RESIDUAL_ADAPTER_V2`；checkpoint新增 `understanding_seed_feature_source=semantic_memory_fm_256`，export逐项检查并严格加载。旧V1即使张量形状相同也不会被当成新语义配方直接导出或resume。
- 旧的smoke→16epoch自动脚本属于另一个独立实验，本新架构没有提交脚本或作业。

## 本地验证边界

`tests/test_point1024_isoadapter.py` 覆盖adapter恒等初始化/输入无梯度/末层与隐藏层依次学习/小state保存恢复；精确像素采样及视图ID；真实1024种子路径；真实12层core小渲染前反向；理解初始化确实来自fm、query初始化本身能向fm反传且不流向重建投影、超点几何/匹配ID保持；分别对实际canonical LR与实际CE/BCE/Dice/stuff LU计算所有可训练参数的梯度，逐参数检查双向隔离并检查有效分支非零；4层adapter有效梯度；adapter=0与无adapter读出的前向等价；零步迁移与错误拒绝规则。

CPU渲染集成用6个anchor、合成冻结骨干与8×8 renderer；1024实际初始化独立执行。未验证真实65536GS GPU渲染、显存、多卡同步或指标。真实core零步CPU审计使用冻结骨干身份stub，见 `point1024_isoadapter_cpu_zero_audit.json`。

**没有提交GPU smoke、四轮训练或评测；未改动既有流水线。**

## 既有零步布局审计

旧V1的4项CPU结果见 `point1024_isoadapter_cpu_tests_final.log`；当前FM输入改版使用 `point1024_isoadapter_fmquery_cpu_tests.log`，不能用旧日志声称FM初始化已经验证。实际1024种子初始化已执行；完整12层core的双loss逐参数隔离检查在6anchor合成渲染上通过。零步审计通过：657旧张量逐值保留，精确删除36个旧重建初始化器张量，新增16个adapter张量/528640参数，总可训练参数193597296。没有进行GPU验证。

这是点种子/监督配置/梯度隔离/adapter组合臂，相比旧A4的结果不能单独归因于其中一项。预算相同不等于单变量架构对照。

## FM输入改版范围

只增加无参数的fm采样/融合与输入接线，不增减参数，不改变657个保留旧张量和528640个adapter参数的布局。当前测试额外覆盖来源视图、来源像素、融合点顺序和反向均值权重，以及仅对q_pre施加损失时fm梯度非零/重建投影梯度为零。当前5项CPU测试全部通过，包括完整两loss隔离回归，见 `point1024_isoadapter_fmquery_cpu_tests.log`。GPU smoke与真实65536GS渲染仍未做，本轮不提交作业。
