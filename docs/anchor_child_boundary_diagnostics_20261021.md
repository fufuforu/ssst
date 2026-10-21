# 固定 A4 / A4+4 的 Anchor–Child 与实例边界诊断

本轮不训练、不修改模型架构、原 checkpoint、损失或正式评测口径。独立代码目录：
`/space/mawb/ssst_anchor_child_boundary_diagnostics_code_20261021_v1`，基于旧双流评测快照 `f2d71dba78094c5922464d61146ef8eab5c54498`。
运行根目录：`/space/mawb/ssst/group_plus/anchor_child_boundary_diagnostics_20261021_v1`。

## 身份与固定窗口

|checkpoint|完成 epoch / updates|SHA256|
|---|---|---|
|旧 A4|4 / 4172|9d3f3f788de59927073cf6d93335623ff37653025258e8e53a0a232f2e01a534|
|旧 A4+4|8 / 8344|0ed6cd8f4fb740698c01ca345a0bbad9a2b1b96302acdbf106864874e5e282c9|

源文件路径及预检收据见运行目录 configuration.json / PREFLIGHT.json；预检实际读取两个 checkpoint，验证续训来源 SHA、keys/shapes、数据与冻结权重身份、训练进度一致。A4+4 的 continuation_sha256 等于上述 A4 SHA。
这两份均为 `OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1`：原有 L6/8/10/12、FM native128、非隔离、无 IoU head；不混用 single16 / P1024iso。

固定使用 representative128_cohort_20261014.json，SHA256
`9a310d764358ebd7cdd3d1844a48685264d25d48cced5d4263ef4da68c239a44`。
128 窗对应 128 个不同验证场景，复用训练脚本 cohorts/build_batch，校验每窗完整六视角输入 RGB SHA。
推理与诊断只使用前两个 Context：生成、lift、GT 归属、边界均不引入 target 信息；读取其余四帧只为复用锁定数据读取与原 RGB SHA 校验。
原 Context 输出 region_mass / RGB / GT 为 256²，FM native128 会在 core 中按原接口上采样；不擅自改成 128² 边界。
A4/A8 对齐校验包括场景、Context、RGB SHA、GT SHA、source_superpoint_ids SHA。

## 经用户批准的纯前向适配

原 lift_features 调用 RendererTranspose.forward，其中存在 autograd.grad，即便外部 no_grad 仍会进行反向。
用户已明确批准“数学等价的纯前向诊断适配”。只在诊断进程临时替换 RendererTranspose.apply；不修改 tokengs 模型文件。

令 W[p,g] 为完整 Gaussian 场景正常 alpha-compositing 的贡献权重。
每次对至多 64 个 GS 设置 one-hot 颜色通道，同时保留**全部 GS 的几何、opacity、深度顺序及遮挡**，背景为零；不是逐个 GS 单独渲染。
前向得到 W 的列块后立即累计 WᵀFM、Wᵀ1、WᵀGT，不保存完整 W。
保持原 evidence=(WᵀFM)/clamp(Wᵀ1,1e-6)、不可见置零、gate=mass/(mass+1)。

每个窗和两个 checkpoint 都必须满足：

- 所有列块之和与 renderer alpha 最大绝对误差 ≤2e-4；
- 四通道固定 probe 特征的 one-hot 重组与直接特征渲染误差 ≤2e-4；
- 独立 probe 的 `<Wf,u>=<f,Wᵀu>` 相对误差 ≤2e-5；
- 捕获实际 child_mlp / W_res 输出，重组最终 mask logits，与模型实际输出 assert_close(rtol=atol=2e-5)。

为了遵守零反向要求，不调用原反向 lift 作为参考：采用前向守恒、线性重组、转置内积恒等式验证代数等价；这是验证范围的明确限制。
GPU 运行期禁止 torch.autograd.grad / backward / Tensor.backward；model.eval().requires_grad_(False)，不构造优化器。
前后哈希全部 checkpoint model 键，检查所有参数/注册 buffer 的版本与全部 .grad=None；冻结资产身份与源 checkpoint SHA 另行核对。

## GT 归属与指标（运行前固定）

C[g,k]=Σ_context_pixels W[p,g]·GT_instance_k[p]，GT instance 以 (semantic,instance) 组合区分。
只有累计渲染贡献 mass≥1 alpha-pixel、thing 贡献至少占总贡献 50%、主实例纯度≥0.7 的 GS 才分配 GT；不满足者保留未归属，报告可见/归属数量。另报告纯度阈值 0.5/0.9 敏感性。
归属过程**不使用 GS 中心投影标签**。

一个 Anchor 若有至少两个实例各占其 thing 总贡献 ≥10% 且 ≥1 alpha-pixel，标记跨实例 Anchor。报告 /全部 Anchor 与 /可辨认 Anchor 的比例。

统计全部 Anchor 和跨实例 Anchor 两个 scope：

- 每子 GS residual / 实际 f_anchor 范数比，均值及 p10/p50/p90；同时记录绝对范数。
- 同一 Anchor 的 2016 个子 GS 对，按上述 GT 归属分同实例/异实例；分别记录 child=shared+residual、最终 FM 融合特征、residual-only 的 cosine。
- 对实际独立 sigmoid membership 同/异实例配对，记录 sigmoid L1、明确归一化为 categorical 后的 TV / JS。
- 同时记录 Bernoulli entropy、归一化 categorical entropy、sigmoid 饱和率，避免把独立 sigmoid 误当 softmax。
- 单独记录融合 FM、shared、residual、W_res 项的平均范数，不能仅凭 raw child 高 cosine 推断共享父特征是瓶颈。

per-window JSON 含分布及样本量、逐 Anchor 行。完整报告用128个场景作为配对 bootstrap 单位，4000 重采样、95% CI，不把同一场景的 GS 对当独立场景。
逐 Anchor 表沿锁定 source_superpoint 对齐；主比较是配对场景平均。
最终特征的 same-minus-different cosine gap 和 membership different-minus-same JS gap 均有正向 CI 才报告“一致改善证据”，否则保留混合/不确定结论。固定 checkpoint 比较是观察性证据，不能证明因果容量瓶颈。

## Boundary

沿用现有 class-aware NMS 和官方 assemble_panoptic，六图中的 Instance Mask 不改候选阈值/竞争/面积过滤。
对保留的 eligible thing queries 的 soft region_mass，按 query 向量求空间前向差分：
`raw=sqrt(sum_q(dx²+dy²)/2)`；不作逐图 min-max，不先二值化。
`soft=1-exp(-GaussianBlur(raw,sigma=1)/0.1)`；raw、soft 均保存原 float32 NPZ。
raw 图显示固定 [0,.2]，只影响六图显示，不影响统计。
另记录全 query（NMS 前）内部响应作为辅助参照，不替换主定义。

GT boundary 为相邻有效像素 GT thing instance 组合 ID 不同、且至少一侧为 thing 时的双侧边界；包含 thing/stuff 交界，忽略无效 GT。
记录连续 raw 对 GT 2px band 的 AP（正确处理 score ties）；soft≥.5 的 precision/recall/F1@1/2/3px。
物体内部定义为 GT thing 区域除去 GT 边界 3px 邻域，报告内部响应能量占比、平均响应、阈值激活率。
六图对每个 Context 保留 RGB、官方实例、原始响应、Soft Map、叠加、GT 边界。

下一步仅作为“小规模损失试验”的预先固定启发式门槛：BF1@2≥.70、场景均值 bootstrap 下界≥.65、AP≥.50、内部响应占比≤.20、内部阈值激活率≤.10。
不根据结果调阈值，不把门槛通过说成 Boundary-Aware RGB Loss 已有效。

## 执行、产物与失败处理

- CPU 定义验证：metrics 归属/歧义/空 GT、配对、正确/伪边界、tied AP、转置恒等式；ForwardLift 用已知矩阵 renderer 核对张量布局、GT 贡献、缓存与禁止反向 API。
- smoke：单卡，前两个固定窗 × 两 checkpoint，检查完整原模型与前向适配，成功生成 smoke/SMOKE_PASSED.json。
- full128：8 卡单节点，afterok smoke；额外核对成功收据、configuration SHA、执行 commit。8 rank 分摊相同的128窗，两 checkpoint 串行加载，不更新参数。
- 任一异常保存 JOB_FAILED，退出且不静默重试/跳窗/修改口径，依赖失败会阻断 full128。无训练作业。
- 可调度池：12/13 的3090、14/17/18 的4090；保留 float32 与关闭 TF32，相同指标，节点由 Slurm 决定。
- full/COMPLETE.json、paired_scene_statistics.csv、anchor_statistics.csv、conclusions.json、publication/REPORT.md 自动生成。
- 全部六图与紧凑 raw/soft NPZ 本地保留；仅前四个预定窗口的16幅六图及表格/收据/报告用于 GitHub，避免上传大量中间张量。无模型权重或数据集上传。
- 代码/协议先 push 新分支；GPU 结果完成后，用户通知再核对并 push 实际报告与示例。不在计算节点依赖代理凭证，不轮询等待。
- 提交后仅一次有界调度核对，随后停止；RUNNING 不代表 GPU smoke 已通过。

## 2026-10-22 实际完成记录

70234 单卡 smoke 4分57秒、70235 八卡128窗诊断11分27秒均通过；原先单卡耗时预估偏保守，实际记录取代预估。
[完整结果与固定样例](anchor_child_boundary_results_20261021/REPORT.md)包含正式1860训练对比、128窗配对统计、连续响应及数值等价/权重不变证据。
A4→A4+4 未显示同Anchor最终特征/membership区分度的一致显著改善；当前提取的Boundary不足以直接进入Boundary-Aware RGB Loss。
