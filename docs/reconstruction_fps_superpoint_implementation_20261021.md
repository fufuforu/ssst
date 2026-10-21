# FPS1024 + learned embedding + superpoint feature（仅代码）

独立目录：`/space/mawb/ssst_recon_fps1024_sp_code_20261021_v1`。
来源：`ssst_recon_qp_code_20261021_v1`，保留原 Q+P 模式，新增独立模式 `dual_fps1024_superpoint`。
本次没有提交、取消、查询任何集群作业，没有修改正在使用的代码快照。

## 实际初始化

1. 使用当前两视图融合点云 `cloud['points']`（原有预处理，最多4096点），全局确定性 FPS 采样1024个位置，不按超点分配名额、不强制区域覆盖。
2. 起点取距点云均值最远的点；相同距离按原点索引打破平局。每个输入索引最多选一次；若不足1024点，按已选FPS顺序循环补足，不加位置噪声。记录 `anchor_point_indices`。
3. 从重建初始化器返回的 `point_superpoint_ids` 查每个采样点的超点ID。该ID已经压紧且与超点特征顺序一致，不用最近中心近似。
4. 原重建超点初始化器输出 `[1,N,256]`，经原投影得到 `[1,N,1024]`。
5. `h_i = learned_embedding[i] + projected_superpoint_feature[parent[i]]`，无额外XYZ/RGB特征拼接、无新位置MLP、无额外门控或归一化。
6. `learned_embedding` 是跨场景持久的 `nn.Parameter[1024,1024]`，标准差.02；纳入优化器、梯度检查、checkpoint及严格恢复。单独新增1048576参数。
7. 中心直接等于采样点XYZ；初始半径沿用对应超点原半径。后续位置编码、几何偏置、位置/半径细化使用原decoder。

原超点特征提取器本身已有邻域和区域空间编码，本版本不删除这些既有内容；“无额外位置MLP”指不再为采样点新增一条初始化特征路径。

固定1024个1024维重建anchor，每个产生64个GS，共65536个GS。
理解保持N个256维thing query +2 stuff，原初始化、匹配与loss不变；理解会读取新anchor，不能预先保证指标不变。默认重建监督L6/L12，显式RG_SUPERVISED仍可覆盖；交互层仍6/8/10/12，Write=0。

## 代码入口

- `tokengs/models/reconstruction_fps_superpoint.py`：FPS、embedding、超点特征查表相加、中心/半径、覆盖率。
- `tokengs/models/reconstruction_fps_superpoint_core.py`：重建分支接入。
- `tokengs/models/superpoint_structural_variants.py`：`LocusGSSuperpointDualFPSSuperpoint` 注册。
- `scripts/train_unified_query_v1.py`：`--architecture dual_fps1024_superpoint`。原始A4零步迁移需显式 `--initialize-fps-superpoint-from-legacy-zero` 和源路径/SHA；禁止训练后模型冒充零步。
- `scripts/recon_fps_superpoint_initial_state.py`：严格匹配旧模型资产、config、data、0update、空Adam和8-rank RNG，只允许多一个embedding，旧张量逐值恢复。
- `scripts/export_unified_query_checkpoint.py`：按保存的架构、固定1024身份与监督配方重建并严格载入。

输出保留anchor→point、anchor→superpoint、GS→anchor、GS→superpoint映射。逐microbatch日志保留原超点与anchor数量；标量日志另有 `num_sampled_superpoints` 和 `sampled_superpoint_coverage`。FPS覆盖率以当前保留的超点数为分母，不把FPS当作标签覆盖保证。

## 验证边界

CPU测试覆盖1024实际初始化、FPS顺序/唯一性/重复点/不足点、精确相加、采样坐标、梯度与更新、参数持久化、严格零步迁移；真实12层core用6个anchor和合成骨干/小渲染器检查理解维度、计数解耦、GS父索引、理解loss及双分支梯度。
真实core零步审计使用源checkpoint mmap与冻结骨干身份stub，结果见 `fps_superpoint_cpu_zero_audit.json`；测试日志见 `fps_superpoint_cpu_tests.log`。
未做GPU smoke、真实渲染显存测试、训练或正式评测。训练预算尚未指定。

## 本轮已完成结果

- 新模式4项CPU测试与原Q+P模式5项兼容性回归，共9项通过：`fps_superpoint_regression_tests.log`。
- 实际1024初始化和embedding保存恢复通过；真实12层core前反向用6个anchor的合成小渲染测试通过。
- 源零步SHA核对通过，693个旧张量逐值一致，唯一新增1048576参数；总可训练参数194847857。
- 理解初始化、controller、loss、交互decoder代码与原基线逐字节一致。
- 仅做CPU验证；GPU smoke与真实65536 GS渲染显存尚未验证。
