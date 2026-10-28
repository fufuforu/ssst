# Frozen Baseline epoch4: updated-token IoU MLP

用户已选择直接使用预测 IoU 排序。本实验只训练一个外置质量回归头，回答冻结后的理解 token 是否足以预测最终实例 mask 的质量；不将其视作已经验证的几何互惠机制。

## 固定来源与预测头

- Baseline：U=1.0，原训练 2context+2novel，epoch4/update_in_epoch0/full_updates4172。
- 来源 checkpoint SHA256：`694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc`。
- 完整冻结原模型，eval 模式，不加载原训练优化器；所有原参数 requires_grad=False、grad=None，验证参数版本不变及 checkpoint 前后 SHA。
- 输入为最终 L12 更新后的 `unified_query` thing token，256 维；不是初始 `q_pre`。
- 新头：LayerNorm(256) → Linear(256,256) → GELU → Linear(256,1) → Sigmoid，共 66,561 个参数。与历史联合训练的 IoU head 无关。
- 缓存前向使用 `torch.no_grad()`。禁止 inference_mode：原模型 RendererTranspose 内部需要对临时变量局部求导。

## 数据与监督

从原始训练 manifest 的 8,337 窗/1,191 场景提取固定 token。原窗口字段及 index 保留，避免改变 build_batch 的确定性输入。按场景、seed31415 确定性划分头训练 1,072 场景、内部开发集 119 场景；与正式评测 312 场景严格互斥。原 backbone 曾训练于这 1,191 场景，内部开发集仅对新头留出。

监督来自每窗两张 Context GT：先使用原有 NMS、no-object/类别门槛、Membership、竞争及面积规则形成最终 Context panoptic mask，再取该 query mask 与所有 Context GT thing 实例的最大、类别无关 IoU。使用两张 Context 拼接的 mask，保持已完成 E2_rank Oracle 的定义。原有保留 query 全部参与，包括最终空 mask 和未匹配误检（目标为零）；已被原始筛选移除的 query 不用于监督和重新引入。

优化：MSE、AdamW、lr=1e-3、weight_decay=1e-4、batch1024、最多100 epoch、patience12，seed31415；只按内部开发集 MSE 选择 head_best。训练阶段不加载 backbone，优化器仅包含新头。记录初始化与训练后的 MSE/MAE、训练集均值常数基准、首步梯度、实际参数变化。即使训练后效果不佳，也保留真实结果，不自动追加实验。

## 正式推理与对照

正式 1,860 窗/312 场景，每窗只做一次冻结模型前向，同时导出原生 Baseline 和 IoU_MLP。IoU_MLP 只将 pred.json 中 score 替换为该 query 的预测 IoU，不乘类别分数。类别、候选集合、NMS、Membership、mask 竞争及最终 PNG 全部保持一致；两个评测臂预测 PNG 使用硬链接，原始 pred.json 独立保留。

正式 GT 不参与训练、模型选择、置信度计算和筛选。写完预测后才能计算 Context GT IoU，用于事后质量预测误差诊断。严格校验输入 RGB 和 GT 与既有正式数据一致；同前向原生 mask 严格核对，历史 Baseline 数值漂移完整记录。

沿用固定版本 SIU3R 评测两个臂，报告 Context/target mAP、AP50/AP75、AR、mIoU、PQ 及差值，target 为 all6（包含 Context）。mIoU/PQ 应完全一致；AP/AR 改善是假设，不预设成功。既有 E2_rank 是 Context GT 排序参考，不是新模型的结果，也不是严格的全局最优上界。重建五项指标沿用同一冻结来源的已核验结果并明确标记复用，不重新测量。

## 独立执行链与停止约定

独立运行目录：`/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline4_iou_mlp_20261023_v1`。

代码：`/space/mawb/ssst_frozen_iou_mlp_code_20261023_v1`，提交后按 code.sha 固定。全部 CPU 针对性测试通过后准备来源硬链接、分割及配置 SHA、不可重复提交标记，四个阶段先执行 sbatch --test-only：

1. cache：18 节点 6 GPU、24 CPU、192G，提取训练 token/GT IoU；时限12小时。
2. fit：afterok cache，0 GPU、8 CPU、32G，仅训练新头；时限2小时。
3. export：afterok fit，18 节点 6 GPU、24 CPU、192G，正式配对导出；时限12小时。
4. score：afterok export，1 GPU、8 CPU、96G，两臂正式评分与汇总；时限2小时。

卡数不要求八卡，GPU 阶段按分配的实际 world size 切片。以上为资源时限，不是预计耗时。只有一个新头实验，不增加分数乘积等消融，不重提既有 Oracle 或三条 epoch8 链。提交后只做一次有界接收状态核对，记录作业号/依赖并停止轮询；RUNNING 不等于 GPU 前向或多卡同步已核验。等用户通知完成后再检查。
