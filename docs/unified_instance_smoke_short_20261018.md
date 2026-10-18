# Unified 实例特征：GPU smoke通过后自动短训

用户授权：先做smoke，通过后16场景、256步短训。只运行 `unified_instance_grounding` 一组，不自动全量训练或正式128评测，不重跑原Unified基线。

## 流程与工程门禁

在同一个四卡作业中依次执行，任何阶段失败即停止并写 `JOB_FAILED.json`：

1. 真实Omega/SAM与真实训练场景，检查8通道signed特征的CUDA渲染：常数1渲染应等于alpha，交替正负1的渲染应保持符号及线性关系。
2. 仅辅助实例损失的CUDA梯度检查，分别测新增实例头、Gaussian几何及共享GFM投影；所有局部梯度有限，跨rank合计每组必须非零，不生成优化器更新。
3. 四rank DDP执行两次完整训练更新，使用不同场景、每rank两个microbatch、全局batch8。要求loss/梯度/参数有限、新实例头确实更新、冻结源无梯度、Adam状态为step2，且每次更新的全量可训练参数SHA跨rank一致。两个更新用原短训的warmup系数。
4. 生成并校验同代码SHA、同preflight SHA的 `SMOKE_PASSED.json` 后，启动独立torchrun进程重新初始化模型、seed42、空Adam，开始正式短训。smoke权重和Adam不写成短训来源。
5. 256步后执行已有短训汇总，再独立审计实例头变化、auxiliary loss日志、Adam step256，以及短训零步参数SHA与smoke零步SHA相同，证明没有继承smoke更新。

smoke验收关注工程正确性，不要求两步loss下降。最终短训也不设置必须loss下降的门槛，质量在用户通知完成后检查初末探针。

## 训练设置

- 节点13，4×3090、16CPU、128GiB内存，时限4小时。资源核对时13节点8卡空闲，18节点10卡已分配9卡。
- 同原Unified短测确定性选择的16个训练场景；1191场景/8337窗口manifest和固定128 heldout清单不变。
- seed42新初始化，只加载官方冻结Omega和SAM。共享Decoder、初始化器、读出与新增实例头均从本配方起点学习，不继承已训练模型。
- 全局batch8，每rank一个场景，累积两次；256更新。LR1e-4，20步warmup及原short cosine；理解权重0.05、前20步ramp，clip1。
- 实例辅助系数0.01位于理解loss内，warmup结束后的总loss有效系数为0.0005。新增特征/几何通路按已实现配方学习。
- 原4train+4heldout初末探针、compact预测、首更新参数同步检查、32步周期checkpoint及最终汇总。不是官方AP/PQ。

## 复现与运行位置

运行目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt11_unified_instance_grounding_short_node13_4gpu`。

独立快照：`/space/mawb/ssst_unified_instance_grounding_short_code_20261018`。作业号及实际执行SHA以运行目录的 `submission.json` 为准。

主要入口：

- `scripts/preflight_unified_instance_short.py`：CPU真实权重严格加载、数据与资产身份核对、训练参数组检查。
- `scripts/smoke_unified_instance_grounding.py`、`scripts/instance_grounding_smoke_policy.py`：真实CUDA烟测及严格收据。
- `scripts/run_unified_instance_short.sh`：smoke→收据门禁→全新短训→汇总。
- `scripts/summarize_unified_instance_short.py`：实例辅助路径及零步状态独立审计。

提交前完成CPU门禁测试、实例模型回归、真实双进程Gloo两次更新同步测试、真实资产/数据preflight和独立快照。CPU通过不代表CUDA通过，CUDA以作业内部收据为准。

本轮15项针对性CPU测试通过（新增门禁2项、真实双rank Gloo1项，另含实例模型11项及探针回归1项）；原模型实现阶段的54项回归结果仍保留。Shell语法和Python AST检查通过，未重复运行无关训练。

提交后只进行一次有界调度核对，随后停止轮询。后续状态以用户通知后检查为准。
