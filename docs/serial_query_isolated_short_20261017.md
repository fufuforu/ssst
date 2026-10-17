# 单组超点身份、先重建后理解：串行短验证

用户确认希望尝试：更新后的重建Token经过投影/归一化初始化理解状态，再经过独立理解Decoder。不是恢复已取消66380，也不是重新训练已完成原Unified/B；新独立架构`OMEGA_3D_SUPERPOINT_SERIAL_QUERY_ISOLATED_V1`，CLI`serial_query`。

## 实际前向

1. 冻结官方Omega与SAM。保留当前ESAM式区域标签→预测点云→三维邻域编码/超点池化的一套初始化；每个有效超点一个256维Token，没有新增独立Object Query表或learned fallback。
2. 复用原Unified的共享GFM特征融合/2048→256投影、12层256维重建Decoder、中心/尺度更新、重建Adapter与Gaussian头，先得到Q_rec和每Token64个Gaussian。L6/8/10/12重建监督保持原始计算图；本阶段没有语义交互。
3. `Q_sem_initial = Linear(LayerNorm(stopgrad(Q_rec)))`。LayerNorm稳定特征数值；Linear是256→256的可学习任务映射，不是维度必须变化。Linear初始为单位矩阵、bias0，保留归一化后的重建内容。保留原超点身份、顺序、数量与Gaussian parent关系，不再次SAM池化、不跨Token重排。
4. 相同冻结Omega四层经参数无关归一化/平均融合，再经过独立语义Adapter：LN2048→Linear256→GELU→LN256，形成两视图32×32的2048个语义memory token。没有新Encoder、RGB卷积分支或MASt3R权重。
5. Q_sem依次经过4层256维理解TransformerDecoder（8heads、FFN4D、自注意力+跨图像特征注意力、dropout0、norm-first）。不再更新或覆盖Q_rec、中心/半径或Gaussian参数。理解后的状态经原语义任务Adapter、分类头及Child/Query相似度Mask头；未使用B的非线性直接Mask头和额外语义CE。stuff仍由原读出生成，无新增独立Query初始化。
6. 同一组超点身份分别保存重建/理解两阶段状态。输出额外包含reconstruction_query、semantic_query_initial、understanding_query，便于后续审计。

## 梯度、损失及比较边界

第一版选择显式任务隔离：Q_rec→语义种子detach，Mask Child局部坐标使用detach的Gaussian/center/radius，理解光栅化也使用detach的Gaussian；仅切断seed不足以阻断后两条几何回路，因此三处均处理。重建初始化器/投影/12层Decoder/几何头/Gaussian头只受重建loss更新；语义memory/seed投影/4层Decoder/分类及Mask模块只受理解loss更新。重建分支仍训练，只有Omega/SAM冻结；这不是冻结重建。

沿用重建多层loss、来源多对一完整Mask监督、0.05理解总权重和context同类NMS0.7。两任务分别clip1，避免全局裁剪间接将理解梯度大小耦合到重建更新。旧模型默认全局clip行为保留，只有`isolated_task_gradients=True`的新架构进入分组裁剪分支。新架构没有共享训练参数组，训练首步诊断支持跳过空组；完整两个非空任务组仍要求梯度非零有限。相对原Unified同时增加了串行Decoder、独立语义memory和梯度隔离，不能称为仅串行顺序的纯单变量对照。

## 预算和资源

新run：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt07_serial_query_isolated_short_node18_4gpu`；新不可变快照：`/space/mawb/ssst_serial_query_short_code_20261017`。同原短验证固定16场景，256更新/2048曝光，全局批8，18节点4×4090，每rank梯度累积2。seed42新初始化、空Adam、不加载完整训练终点。AdamW LR1e-4、20步LR warmup/cosine、理解0.05/20步ramp、weight decay0.05（1D为0）、两任务各clip1。

初始/最终同4train+4heldout固定前向探针，首步四rank参数SHA检查，32步保存checkpoint，结束自动汇总新增语义memory/seed/Decoder参数变化。诊断不是官方128结果，也不会自动转全量。预计约20–35分钟，实际受Query数/节点负载影响。

## 提交前验证

25项CPU回归通过：同公共权重时重建Gaussian和Q_rec与原Unified逐位相同、超点身份保留、identity seed、原位模型不受改变；理解loss经真实step_loss与带几何依赖的模拟renderer无法给重建参数/Gaussian送梯度，语义memory/seed/4层Decoder实际有梯度；增加理解loss并分别clip后重建梯度与纯重建逐位相同；分组覆盖无遗漏/重复；既有初始化、完整探针、零步控制等回归。

真实2rank Gloo额外1项通过：两rank不同Query数量、连续2次Adam更新后全可训练参数SHA相同。真实冻结文件SHA、固定训练indices与实际train/heldout样本、meta模型参数分组预检通过。Python编译、shell语法和diff检查通过。GPU新结构首步仍未核实，CPU/Gloo成功不代表native CUDA已通过。

新实现：`tokengs/models/serial_query_core.py`，包装与factory在`superpoint_structural_variants.py`。现有export按架构factory可识别新变体；短测checkpoint仍禁止冒充完整训练官方导出。

提交收据与参数量见run/preflight.json、VALIDATION_EVIDENCE.json和submission.json。只做一次调度核对后停止轮询，等待用户通知完成，不恢复历史66380、不干预A/B。

## 已提交并停止轮询

短验证 **66603**，请求18节点4×4090、16CPU/128G，256更新；唯一一次状态核对 **PENDING (Priority)**。执行SHA **a57ce7d**、独立快照干净，383个可训练张量/19,424,708参数（重建14,209,457、理解5,215,251）。本次无正式128评测或全量续训依赖，初末探针和自动汇总在同一作业内执行。新GPU首步和4rank同步尚未核实，不能把CPU/Gloo成功当原生GPU验证。具体时间与收据见run/submission.json和SUBMISSION_STATUS_ONCE.json。已停止轮询，待用户通知完成后检查。
