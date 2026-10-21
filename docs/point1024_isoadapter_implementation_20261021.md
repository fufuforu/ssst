> 当前目录已派生为2context+4novel数据版本；请先读 docs/training_2ctx4novel_implementation_20261021.md。下文2+2及69969节点记录属于父版本历史，不代表本目录已提交或完成GPU。

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

## 2026-10-21 独立审查修复与专用单卡 GPU smoke（最新）

本节替代上文“尚未准备GPU入口/本轮不提交”的历史状态；用户本轮仅授权修复和一次smoke，不授权全量训练或1860评测。

### P0-1：采用方案 a，明确两种证据的键与来源

`state.update(result)` 改为白名单 q/c/s/route/joint_delta/anchor_mask_logits/anchor_membership/ell。每层独立 `state['under_state']` 保存完整理解读出，其anchor_embedding/f_anchor由adapter后的h编码，供child语义特征**显式读取**。

顶层state.anchor_embedding/f_anchor由第一阶段**原始重建token**经原controller编码得到，不经过adapter，并在no_grad中作为只读诊断快照保存。它们不用于额外的LR/LU回传。child继续使用理解adapter证据，行为来源现在显式标明，避免理解结果污染重建字段；不是把child偷偷切换为未适配特征。

读出前保存第一阶段mu/radii对象及version；每次理解层完成检查对象身份及version，拒绝替换或原地修改。CPU回归主动注入对象层额外mu/radii键及原地修改，分别验证白名单保护和错误退出。

### P1核对

- P1-1：父类 `_render_understanding` 读取 `reconstruction` 做RGB/depth展示，读取gaussians+gaussian_membership重新feature render；region_mass/pixel_membership/semantic_scores都从这个feature render和p_class重新计算，没有复用generated中的同名结果。states只是返回引用，没有参与这些概率的计算。实际LU在unified_instance_loss中只用thing_logits、region_mass和无梯度source matching；RGB/depth展示与返回states不进入LU。因而只替换传给feature renderer的全部GS属性已隔离该路径，不对理解membership/fm再detach。完整CPU实际双loss逐参数检查继续通过；真实GPU定向检查由smoke执行。
- P1-2：采样入口要求 image_hw==fm.shape[-2:]，core也要求fm空间尺寸等于GS头配置图像尺寸；不一致ValueError。
- P1-3：构造时inspect.signature显式检查seed_anchors含return_source_indices，无静默fallback。
- P1-4：gaussian_parent_query与gaussian_parent_anchor返回同一张量，均表示GS→anchor；附带语义说明字符串，不表示GS→动态object query。不能用该别名去索引N个理解query。
- P1-5：如上，mu/radii对象身份与version保护，不只是检查数值是否近似相等。

### 本架构专用 GPU smoke 入口（不复用FPS脚本）

- `scripts/smoke_point1024_fmquery_isoadapter.py`：真实单窗/256×256/1024anchor/65536GS，单卡单rank，累积8次同窗保持global8，恰好2个optimizer更新。LR仍按4epoch/4172更新、warmup100轨迹；理解权重仍.05×min(1,(update+1)/1043)；同一AdamW/decay/clip1.0。只限制smoke更新数，不执行4epoch。
- `scripts/run_point1024_fmquery_smoke.sh`：离线资产环境、固定L6/L12、关闭旧额外实验开关、失败收据兜底。
- `scripts/verify_point1024_fmquery_smoke.py`：专用收据校验器；只有全部成功才SMOKE_PASSED。
- `scripts/submit_point1024_fmquery_smoke.py`：只提交一次单卡作业，然后一次15秒内有界状态核对；没有全量训练/评测/重试逻辑。

第一步首个microbatch单独LU.backward(retain_graph=True)，检查重建+shared参数.grad全部None/0；清空后单独LR.backward(retain_graph=True)，检查理解+adapter.grad全部None/0；再清空并按原总loss做累积训练。诊断反传不执行optimizer.step，不污染实际更新。每步记录reconstruction/shared/understanding/understanding_adapters梯度范数及L6/L8/L10/L12逐adapter范数，L12必须非零，早期层和隐藏层允许正常的首步零梯度。第一步时间包含定向诊断开销，不能拿它直接估算稳定吞吐。

首步哈希只验证1rank参数确实更新，不声称8rank同步；预算收据明确world1/global8/accumulation8/unique_window1/2updates/16microbatches。不写大模型权重，不做单窗导出或正式指标评测。本轮验收范围是用户指定的真实GPU训练与6项检查。

独立run（实际作业号/状态以其中收据为准）：
`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_isoadapter_gpu_smoke_20261021_v1`

证据文件：SUBMITTED.json、STATUS_AFTER_SUBMIT.txt、BUDGET.json、POINT1024_ISO_INITIALIZATION.json、DIRECTIONAL_GRADIENTS.json、STEP_RESULTS.json、FIRST_UPDATE_VERIFIED.json、SMOKE_PASSED.json或JOB_FAILED.json、slurm-<jobid>.out。

本次CPU验证：6项模型/采样/隔离/签名/污染保护测试通过（`point1024_isoadapter_review_fixes_cpu_final.log`）；2项收据通过/拒绝规则测试通过（tests/test_point1024_smoke_receipts.py）。这些是提交前CPU证据，不能作为GPU通过结论。提交后只做一次状态核对就停止，不等待GPU结果；若结果未就绪，6项实际数值均标注待核实，用户通知后再读取。

## 用户通知后检查69945：失败于smoke形状断言（2026-10-21）

Slurm终态FAILED，ExitCode1:0，3dimage-11单张3090，耗时00:05:54。JOB_FAILED.json标记stage=batch_load、completed_updates=0，错误Real window image shapes differ。没有SMOKE_PASSED/STEP_RESULTS/DIRECTIONAL_GRADIENTS/FIRST_UPDATE_VERIFIED；因此没有GPU前向/反向/更新或六项验收数值。

根因是新增smoke launcher第83行把images_all硬编码为[1,6,3,256,256]。原A4数据未改变：build_batch固定window.context+window.novel，均各2帧；_full_supervision明确2context+2novel监督，共4张图。所选scene0683_00，context[122,174]，novel[142,156]，target[122,174,142,156]。旧文档“2context+4target”应解释为2输入/4总监督视图（其中含2context），不是6张不同帧。这里是launcher断言错误，不是已确认的模型/显存/梯度失败。

成功完成源权重校验、真实GPU模型构建和零步迁移：657旧张量保留，36删除，16adapter张量新增；GPU型号RTX3090，总容量25430786048bytes（容量不等于峰值显存）。原始大权重仍硬链接。

下一步应只修smoke视图验证为原context+novel契约，并记录实际tensor shapes后再考虑重跑；不改数据、配方或模型。本次检查没有代码修复、没有自动重提、没有全量训练或评测。失败证据见run/JOB_FAILED.json和slurm-69945.out，结果汇总RESULT_REVIEW.json；CPU同窗数据回放已确认：images_input=[1,2,3,256,256]、images_all=[1,4,3,256,256]、images_output=[1,2,3,256,256]；证据FAILURE_DATA_SHAPE_DIAGNOSIS.json。

证据run目录：`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_isoadapter_gpu_smoke_20261021_v1`。

## 69945失败后的授权修正与node14重跑准备（2026-10-21）

用户检查失败后明确授权修改并重新提交单卡smoke到14。本次只修launcher数据形状契约，不改数据/模型/损失/预算：按原窗口2context+2novel验证images_all=[1,4,3,256,256]，images_input=[1,2,3,256,256]、images_output=[1,2,3,256,256]，并严格核frame_ids=context+novel。不能通过额外加载两张图来满足旧六帧断言。

新增BATCH_SHAPES.json：验证前先保存实际shape，验证成功保存实际frame_ids及context/novel/总监督视图数；BUDGET中同步记录该契约，最终verifier检查它。这样即使未来再次尺寸不符，失败记录也有观测值。

4项标准库测试通过，覆盖真实CPU回放对应的四帧契约、六帧拒绝、帧序拒绝，以及原收据接受/拒绝。模型文件相对cc72e05无变化，无需重复完整模型CPU测试。

此次新run：`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_isoadapter_gpu_smoke_20261021_v2_node14`；独立执行快照：`/space/mawb/ssst_point1024_fmquery_smoke_code_20261021_v2_node14`。旧69945目录/快照保留。提交资源明确3dimage-14/4090分区/1卡；仍单窗、global8按8次累积、2optimizer更新、4epoch4172步轨迹。没有全量训练或1860评测依赖。

作业号与唯一一次提交后状态核对见新run/SUBMITTED.json和STATUS_AFTER_SUBMIT.txt；GPU六项数值仍必须等实际SMOKE_PASSED或JOB_FAILED与STEP_RESULTS等收据，不能由本次CPU契约测试替代。提交后停止轮询。
