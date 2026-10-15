# VGGT 重建与理解实验交接：2026-10-15

> **终点更新（2026-10-15）：四组300更新/2400曝光、128场景官方评测及64855汇总已全部成功完成。**
> 最新结果与补充审计见 [`vggt_reading_controls_results_20261015.md`](vggt_reading_controls_results_20261015.md)。
> 下文08:11运行/排队状态保留为历史现场，不能据此重新提交或恢复任何训练。当前没有授权新的交互层对照、长期训练或干预叠加。

这份文档用于新对话直接接手当前工作。**先读当前 worktree、运行记录和 Slurm；沿实际最新进度继续，不回退、不重新设计、不重复提交训练。** 当前已授权的阶段是四组 decoder 读取短对照，等待全部完成后检查、比较、分析。没有授权自动开始新的长期训练或叠加新结构。

## 1. 给新对话的开场指令

可以直接把下面这段发给新对话：

> 请接手执行 `/space/mawb/ssst_vggt_overfit_diagnostics_v1/docs/codex_handoff_vggt_reading_controls_20261015.md`。先核实指定 worktree 的最新代码、运行记录和 Slurm 状态，沿当前进度继续，不重新设计实验、不回退或重复训练。本轮 geometry_quality_policy=monitor_v1，普通工程问题自行定位、修复、验证、提交推送和精确续跑；保留历史失败。当前四组均已有正式 20 更新/160 曝光的真实启动证明，不需要持续轮询等训练结束。若已全部完成，审计并按 SIU3R 官方 13 列比较分析，两个 mIoUt 空白，不另拆三套视图表；综合重建与理解指标判断下一步，不只看 PSNR。

## 2. 当前仓库与权限

- 仓库：`https://github.com/fufuforu/ssst`。
- **当前 worktree：`/space/mawb/ssst_vggt_overfit_diagnostics_v1`**。
- **当前分支：`vggt-overfit-diagnostics-v1`**。
- 写交接前 HEAD：`03a4f45`，已推送；本文档会再产生一个仅文档提交。接手时重新运行 `git status`、`git log` 和远端核验。
- 当前 GPU 执行的不可变快照：`649ef09aef473128849539baf7453654033ce281`。worktree 后续文档提交不改变正在运行的代码。
- 不要在运行中的 snapshot 上修改文件或 checkout/fetch。
- 用户已授权普通工程修复、受影响验证、提交推送、重试及恢复，不逐项等批准。保持模型、数据、loss、训练数学锁定，不能跳过窗口、隐藏 nonfinite、引入 GT/target 生成输入或悄悄改 batch/精度来绕过问题。
- 用户原始附件：`/space/mawb/ssst/docs/codex_vggt_posefree_autonomous_recovery_prompt.md`。其中 monitor_v1 和自主工程恢复仍有效；**旧起点 8502d978、旧 worktree、旧冻结范围和最初八轮训练不是当前实验配置，已被后续明确授权的方案更新。不要照旧附件重启最初实验。**
- `monitor_v1`：重投影 median/p90、正 Z 参考阈值超标只记质量告警，不因此停训；数值/资产/输入契约等硬错误仍要定位。旧 strict 失败不得改写成成功。
- 用户反复要求确认真实启动后停止轮询。当前四组都已有正式 20 更新/160 曝光与八 rank 同步证明，这个要求已完成；无需在新对话重新持续守候。

## 3. 最新现场状态（2026-10-15 08:11，Asia/Shanghai）

以下是写文档时真实读取的状态，不是未来保证；接手应再次只读核实。

证据 root：

`/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04`

| 分组 | 最新有效作业 | 核实状态 | 正式进度 |
|---|---|---|---|
| baseline 原版联合训练 | 64852 | RUNNING，3dimage-13 八张 3090 | 98/300 更新，784/2400 曝光；从 20 精确恢复 |
| object_local_read 对象局部读取 | 64717 训练/导出；64851 补评分 | 训练及 128 场景官方评测均完成；64851 COMPLETED 0:0，用时 2:01 | 300/300，2400/2400 |
| anchor_visual anchor 局部视觉证据 | 64853 | PENDING，Resources | 已有 20/300、160 曝光，续跑尚未启动 |
| reliable_depth 可靠深度约束 | 64854 | PENDING，Priority | 已有 20/300、160 曝光，续跑尚未启动 |
| 四组自动汇总 | 64855 | PENDING，Dependency | 等待上述三组续跑/评测成功 |

64852/64853/64854 申请同型 8×3090、单节点、32 CPU、384G、4h，partition=3090，exclude=3dimage-12，因此可用 11 或 13，不再锁死 11。三组之间无无关串行训练依赖；它们共同的补评分前置 64851 已成功。当前 user 授权优先用空闲八卡，不能把本轮 microbatch1/global8 和此前 A/B 的 global16 混为一谈。

基线记录的约 3.92 秒/更新对应剩余纯训练约 13.2 分钟；**这不包括初始化、排队、checkpoint 保存、导出和评分，不能当成总完成 ETA**。排队组不能给确定结束时间。

### 已完成局部读取组的真实 13 列（仅当前 128 场景短对照）

| AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入 mIoUs↑ | 输入 mAP↑ | 输入 PQ↑ | 输入 mIoUt↑ | 新视图 mIoUs↑ | 新视图 mAP↑ | 新视图 PQ↑ | 新视图 mIoUt↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.08657|0.2115|22.07|0.7725|0.3745|0.5515|0.2975|0.6321||0.5490|0.2625|0.6072||

这是独立留出 128 场景各一窗口、300 步后的成绩，尚没有同预算 baseline 终点，**暂不能判断局部读取有没有增益，更不能拿它与完整 312 场景成绩直接作差**。

## 4. 当前四组究竟在做什么

完整设计及历史执行记录：`docs/vggt_reading_controls_v1.md`。以下是接手必需的固定设置。

### 共同起点 C

命名需辨清：历史也有 A/B/C 等字母。**本轮 C 是“完整 adapter0.01/feedback0.05 模型 + 分类对象支撑读回补丁”**，不是旧 feedback factorial 中某个也叫 C 的短对照。

完整底座 A：先重建适配 2 epoch，再冻结 VGGT 联合理解/重建 4 epoch，adapter 理解梯度路由系数 0.01、feedback 峰值 0.05。

```text
/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1/checkpoint_frozen_joint_epoch_04.pt
SHA256 ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0
```

C 分类对象支撑读回补丁（A 原核心冻结，仅分类 LN/head 及对象支撑读回训练 300 步得到）：

```text
/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/training/classification_patch_update0300.pt
SHA256 ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9
```

本轮四组都在同一 C 上进行真正的下游联合短训练。保留 C 读回原有 detach 行为，不暗中改分类梯度连接。

### 共同训练合同

- 每组 300 正式更新/2400 新窗口曝光；8×3090，microbatch1、accumulation1、global batch8。
- 使用完整 8337 窗口池中的固定 **2400 个不同训练窗口**，覆盖 1075 场景；不是在 128 场景反复训练。四组窗口顺序一致，update i/rank r 取 stream[i*8+r]。
- 训练池共 1191 场景。留出为固定 128 不同场景各一窗口，场景与训练无重叠。不是完整 312 场景/1860 窗口验证。
- **VGGT 整体和 MASt3R image encoder 都冻结/eval**；其余按 native 原 requires_grad + C 分类读回规则训练，所有 BatchNorm 保持 eval。不要误认为本轮只冻结 DINO。
- 原重建 loss + 完整 V3 set loss、matcher、辅助层 loss 均不变。理解总权重 1。
- GS 与 VGGT memory adapter 的理解梯度系数均为 0.01。这是模块梯度路由，**不是把全局理解 loss 乘 0.01**。
- feedback 峰值 0.05、双向结构及 6/8/10/12 交互保留。成熟源联合曝光时钟 35776，新训练计数另记 2400，不重启成熟任务的理解/feedback warmup。
- 短阶段起点 fresh native AdamW：原 decay 分组，beta .9/.95，eps1e-8，clip1；LR 原峰值 rec1e-5/理解1e-5/object_memory1e-4，warmup200，原 cosine4172 截断到300。
- 四组参数集合、容量、共同起点完全相同，新增可训练参数 **0**。
- 每组先保存 20 步 checkpoint，统一验证后 **恢复该 optimizer/RNG/采样计数** 到300；不是在续跑时重新 fresh，也不重复前20步。
- 三种干预强度前20次更新按 min(completed_new_updates/20,1) 开启，之后固定。零强度必须复现原 C。

### 三种独立干预及基线

1. **baseline**：保持原计算，只增加观测，不改输出。
2. **object_local_read**：只改变 object query→anchor CA 的对象读取偏置。L6 原样，L8/10/12 使用**上一交互层**预测 anchor soft mask（detach），增加 λ·log(0.1+0.9·mask/max)，λ峰值0.5；空/常量支撑及 stuff query 保持全局回退。原 anchor 内容、Q/K/V/O 和 feedback 方程不改，但后续 anchor 可经 query/feedback 间接受影响。没有 GT 或最终 mask 预知。
3. **anchor_visual**：保留原 CA 方程，把已有 Fm 在 anchor μ 投影到两个预测 context 相机后双线性采样，有效视图平均，无效为0。归一化残差在峰值不超过 anchor norm 的2.5%；visual λ峰值0.1。不加 encoder/投影参数。K/V 内容变了实际 attention 也可能变，不能称为固定 attention 权重的纯 V 实验。
4. **reliable_depth**：改变 **GS anchor→VGGT patch CA**，与第二组不是同一处 CA。保留点到无限射线的垂直距离偏置，额外加入 λ·reliability·clamp(-0.5·沿射线深度偏差²/radius²,-20,0)，λ峰值0.1。复用已有 predicted points/confidence，14×14池化与 patch rays 顺序对齐；每视图 q20/q80 归一化 confidence，退化常量为0。不是概率标定，不加载 point_head，不引入 GT depth 或新 loss。

强度预先固定，不在留出集调参。无增益只否定本次实现/强度下的收益，不能直接否定整个研究方向。

## 5. 当前结果怎样评测和判断

- 固定原 SIU3R 官方仓库 `/space/mawb/SIU3R`，commit `8ea80166be76854f938e90521f1a5b688b755c87`，不修改 evaluator 数学。
- 只用 SIU3R 表格 **13 个指标列**：5 个重建 + 输入4项 + 新视图4项；两个 mIoUt 空白。不要另做 context/target-all/true-novel 三套表。
- “新视图”沿用官方 target 集合（2输入帧+4额外帧），不能悄悄改成仅4额外帧。深度按官方 scale/shift 对齐。
- 每组训练300完成后自动八卡导出128窗口，再官方评分；mAP/PQ/mIoU 用官方状态全局合并，**不平均各 rank AP**。八rank分别验证真实 CPU/GPU 状态合并合同。
- 所有组相同 input/frame IDs/GT、相同训练流；summary 自动核验。记录实际每头 V norm、完整 O 输出 norm、每头 attention 熵，不能拿 mean(R)@原anchor 的代理宣称完整 CA 失效。
- 本轮 **不使用旧 Q**。结构变了输入分布也变；若后续加 Q，应各组等预算重新拟合，属于后续单独决定的工作。
- 综合重建、语义、实例、PQ及候选/匹配诊断判断，不能只看 PSNR 或数“多少项上涨”。不要把固定训练窗口拟合成绩当正式泛化结果。
- 若四组出现不同取舍，先报告作用位置和证据，再提出下一阶段；不未经讨论自动叠加干预或开4epoch。

最终自动产物在 attempt04：

```text
{arm}/training/progress.json
{arm}/training/checkpoint_latest.pt 和 checkpoint_latest.json
{arm}/training/startup_confirmation.json
{arm}/training/run_manifest.json 和 resume_manifest.json
{arm}/training/training_rank00.jsonl ... training_rank07.jsonl
{arm}/evaluation/metrics.json、siu3r_table1.csv、siu3r_table1.md、COMPLETE.json
{arm}/evaluation/matched_query_diagnostics.json
analysis/report.md、siu3r_comparison.csv、results.json
COMPLETE.json
```

## 6. 为什么走到这一步：已测量的主要结论

### 6.1 不能把指标差简单归因于训练不足

旧 adapter 系数1的适配2+冻结联合4：完整 PSNR19.37；再冻结联合2降到19.14；再解冻 AA 联合2为19.35，没有全面改善。后两组 global16/fresh optimizer 与源 global8 历史有差别，不能把变化全部归因训练时长。

AA 确实更新过，非冻结实现错误。低预算 AA 冻结/解冻对照也未得到清晰综合收益。新 GS 从零在100步预算明显更差，支持当前继续沿用旧 GS 初始化；**不能据此断言从第一阶段开始训练的新 decoder 长期上限更低**。

参考：`docs/vggt_b_and_joint_short_controls_results_20261013.md`。

### 6.2 adapter 理解梯度路由是已确认的重要改善点

固定窗口梯度诊断发现 adapter 理解梯度主导，主要来自 mask/Dice，部分时刻联合梯度与重建相反。系数1→0.01保留理解损失，改变 adapter 接收的理解梯度。

完整同预算配对后 PSNR 提升约 **3.02667dB** 到22.40。这支持此前19dB的大幅下降与路由设置密切相关；不能把全部 SIU3R 指标差距都归因于它。

固定四训练窗口24.5左右的拟合数不是完整验证成绩。相关报告：`docs/vggt_adapter_gradient_results_20261012.md`、`docs/vggt_adapter_full_and_feedback_factorial_results_20261013.md`。

### 6.3 feedback0.05 的完整收益很小，并非理解全面提升

完整数据适配2+冻结联合4，feedback0.1→0.05：PSNR22.40→22.42（约+0.0200dB），其他重建也略好；六个理解指标五降一升，不能称为全任务最优。短对照优势不等于长期终点同等收益。

当前保留0.05作为后续共同底座，不继续盲目搜 feedback，也不把理解偏弱主要归因于 feedback。

参考：`docs/vggt_feedback_b005_full_results_20261014.md`。

### 6.4 C 对象分类读回：小收益，未解决实例质量排序

在 A 的分类输入加入本 query 预测 mask×context可见性加权的对象支撑，原 mask/feedback query 不改。完整312场景六项理解有小收益，但未参与筛选的184场景 mAP 基本持平，增益有类别集中现象。

好 mask / no-object 冲突与实例质量排序不一致仍存在。验证时 GT Hungarian 匹配和部署时 no-object 过滤、最终官方预测/GT匹配是不同步骤；不能说预测query被删后训练matcher自动替它分配另一个query。

参考：`docs/vggt_classification_readback_full312_results_20261014.md`。

### 6.5 Q 只是外部质量读出，不参与主模型联合训练

Q 为 detached query 256→128→1 MLP，单独训练300步，预测最终同类 mask IoU；实例 score=s_native×sigmoid(Q)。保持分类/候选/no-object筛选/像素赢家/最终mask及RGB不变，只改变实例置信度排序。

固定 Q 完整312场景验证：输入 mAP26.2508→26.6855、新视图24.7191→25.1409（+0.4347/+0.4218个百分点）；新增184场景也为正。不是 object query 的简称，也不是用 Q 训练主模型4epoch。

### 6.6 表征不足还是读取不足：尚未定位单一主因

冻结诊断1024训练窗口/707场景、128独立留出：query 已有类别信息，交互后没有明显整体类别塌缩；q12相对q_pre条件类别准确率差的区间跨0，原分类头不逊于新线性probe。

现有对象 Gaussian 支撑、局部 Fm 对质量预测有互补；DINO类信息有小信号，但不足以宣称必须新增encoder。confidence分层与VGGT深度误差相关，但不是几何干预收益的因果证明。

用户提出 TokenSplat式渐进GFM融合、confidence几何偏置、DINO/point-map补充。当前优先分开检验读取局部性、已有局部视觉证据和可靠深度约束，正是正在跑的四组。多层大融合、point_head、AA解冻等不与本轮叠加。

参考：`docs/vggt_feature_reading_results_20261014.md`。其中“建议下一轮未提交”是历史时态；**本轮后来已获得授权且正在执行，不要再提交第二套**。

## 7. 已有完整验证参照：不要与128场景短对照混比

以下均为官方312场景/1860窗口。理解值为0–1，差值转成百分点时乘100。

| 模型 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入mIoUs↑ | 输入mAP↑ | 输入PQ↑ | 输入mIoUt↑ | 新视图mIoUs↑ | 新视图mAP↑ | 新视图PQ↑ | 新视图mIoUt↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A 完整adapter0.01/feedback0.05|0.08684|0.2237|22.42|0.7818|0.3636|0.5719|0.2614|0.6407||0.5676|0.2462|0.6231||
|C A+对象分类读回|0.08684|0.2237|22.42|0.7818|0.3636|0.5765|0.2625|0.6442||0.5720|0.2472|0.6261||
|C+固定Q|0.08684|0.2237|22.42|0.7818|0.3636|0.5765|0.2669|0.6442||0.5720|0.2514|0.6261||

真实完整表：`/space/mawb/ssst/group_plus/vggt_quality_full312_v1/attempts/attempt01/analysis/siu3r_comparison.md`。

## 8. 工程恢复历史：不能误判成还没训练或再次重训

- attempt01：观测 hook 访问临时移除的 reading_control；正式更新0，修复后保留旧记录。
- attempt02：native runtime bootstrap 导入顺序导致 fused_ssim_cuda 缺失；正式更新0。
- attempt03：把原生 CUDA atomic lift 的微小读出波动误要求逐位相等；正式更新0。后续引入原模型自身连续两次前向波动见证；上游 Gaussian/Fm/query/anchor 要求差0，logits/membership 的原生重复及零干预差均≤1e-5且finite。没有放宽训练数值合法性。
- attempt04 GPU预检64667：完整单卡/八卡、四组真实2步joint backward及官方真实窗口评分通过，用时2:25:10。
- 正式startup64668/64669/64670/64671：四组各20更新/160曝光通过；64672统一核验成功。证明文件 `ALL_STARTUPS_VERIFIED.json`。
- 共享存储紧张：只改checkpoint无损序列化，使用node-local临时torch archive→ZIP_DEFLATED level1→共享atomic替换，锁串行共享写；参数、optimizer、RNG、计数roundtrip严格相同，load不使用mmap。没有删优化器或改精度。
- 64717 object_local_read 实际已训练300并导出128窗口；失败发生在 **官方评分环境顶层误导入TokenGS，缺lpips**。不能写成训练失败或重新跑300步。
- 修复评分、reduce、summary不导入主模型，只在export延迟导入；649ef09是当前执行快照。64851只补已有导出评分并成功，官方8rank合并合同通过。
- 64716/64718在PENDING且RunTime0时取消替换为64852/64853；64719此前因64717失败依赖取消，恢复为64854。旧日志、失败及取消记录保留。
- 18项数学文件SHA/受保护函数AST/正式update-loop AST与实际预检快照410bd30一致，因此没有再重复2.4小时GPU预检。

关键回执：

```text
attempt04/submission.json
attempt04/schedule_scoring_recovery_20261015.json
attempt04/scoring_import_recovery_preflight_reuse.json
attempt04/serializer_only_preflight_reuse.json
attempt04/ALL_STARTUPS_VERIFIED.json
attempt04/slurm/
```

## 9. 环境与代码入口

训练/模型导出：`/space/mawb/anaconda3/envs/tokengs/bin/python`。

官方评分/状态合并：`/space/mawb/SIU3R/.venv_gpu_v4/bin/python`。

**官方环境没有完整TokenGS依赖及匹配的gsplat运行库，不要让score/reduce/summary导入模型。** 不要为了它修改共享官方环境；已经有正确的纯评分入口。native模型导入前需要原runtime兼容bootstrap。

资产：`/space/mawb/ssst_object_locus_frozen_vggt_posefree_v1_assets`，离线复用。现有launcher设置HF离线、VGGT源码PYTHONPATH、预编译gsplat、线程数和geometry evidence路径，优先复用。

主要入口（相对当前worktree）：

| 文件 | 作用 |
|---|---|
| `tokengs/models/object_locus_reading_controls_v1.py` | 三种干预与真实CA观测 |
| `scripts/train_vggt_reading_controls_v1.py` | smoke/train，--resume精确恢复 |
| `scripts/eval_vggt_reading_controls_v1.py` | export/score/reduce/summary |
| `scripts/run_vggt_reading_controls_v1.sh` | 固定快照Slurm入口；含score_existing |
| `scripts/submit_vggt_reading_controls_v1.py` | 最初完整链提交器，**已有本轮作业时不能直接重跑** |
| `scripts/show_vggt_reading_controls_progress_v1.py` | 标准库只读进度，含squeue |
| `scripts/vggt_reading_numerical_contract_v1.py` | 工程恢复时核验18项训练数学身份 |
| `scripts/vggt_lossless_checkpoint_v1.py` | checkpoint无损压缩/恢复 |
| `scripts/format_object_locus_posefree_siu3r_table.py` | 固定13列映射及空白文本指标 |
| `tests/test_vggt_reading_controls_v1.py` | 局部读取/几何采样/零强度等关键合同 |

临时恢复脚本 `/tmp/recover_vggt_reading_schedule_20261015.py` **已经执行，不可再次执行**；正式收据在共享attempt04，不依赖/tmp永久存在。

Slurm/网络命令在Codex受限环境通常需 `sandbox_permissions=require_escalated`。未升级直接执行squeue可能报socket权限错误并耗时；这不是集群故障。无需用长轮询等待。

Git代理现有旧127.0.0.1:7890不可用；已成功使用逐命令覆盖，不改全局配置：

```bash
git -c http.https://github.com/fufuforu/ssst.git.proxy=http://127.0.0.1:17899 push origin vggt-overfit-diagnostics-v1
```

## 10. 数据、源权重与清理边界

训练manifest：`/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json`，8337窗口/1191场景；实际key是`windows`。

固定流：当前repo `docs/vggt_understanding_probe_cohort_v1.json` 的 `train_source_indices` 前2400项。

固定留出：`/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json`；records包含name/source_index/frame_ids/input_rgb_sha256，没有scene字段。

完整官方验证：`/space/mawb/SIU3R/data/scannet/val_pair.json`，312场景/1860窗口。

**除A、C、VGGT资产外，下列legacy源即使看似“很旧”也不可删除**，当前构造恢复仍会用到：

```text
/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt
SHA256 68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a
```

Q补丁（未来复核需要，当前四组不加载）：

```text
/space/mawb/ssst/group_plus/vggt_readout_probe_v1/attempts/attempt03/training/head_checkpoint_update0300.pt
SHA256 61dd9a8fbd8915c1c8779e61c6e0a3997b2b3d15a2a9639e297b1c0946ffcee5
```

用户已明确授权清理无用旧实验。已执行26文件清理，释放 **78.21GiB**：旧feedback factorial三组15个checkpoint、已被epoch4替代的3个epoch1–3 checkpoint、8份已完成feature诊断features.pt缓存。清理当时可用91.34GiB，不代表现在实时可用量。

保留当前任务、必要source/资产、C/Q补丁、小型指标/分析/失败记录。**旧feedback短对照已没有可重跑权重，特征诊断已没有features.pt，不要承诺直接从这些缓存继续。** 小型系数/分析仍保留。

清理证据：`/space/mawb/ssst/group_plus/storage_cleanup_20261014_reading_controls/{manifest.json,deletions.jsonl,receipt.json}`。

## 11. 新对话应该如何继续

1. 读取本文和 `docs/vggt_reading_controls_v1.md`，核实git状态/当前执行SHA与Slurm；已有后续提交或作业则先查明，不回退。
2. 只读检查64852/64853/64854/64855及上述COMPLETE/进度/导出收据；识别“训练完成而评分失败”与“训练失败”的区别。
3. 尚在正常训练/排队时，汇报实际阶段即可，不再重复提交或长期轮询。四组20/160证据已齐。
4. 工程失败时先定位，保留失败日志，保持数学身份修复并做受影响验证，提交推送新不可变快照；尽量只补失败阶段。训练恢复使用实际latest checkpoint及其optimizer/RNG/采样计数。
5. 全部完成后，审计每组300更新/2400曝光、八rank采样/finite/冻结/恢复、128场景导出和官方合并合同、共同GT与起点、Q未启用、13列有限性及两空白。读取自动 `analysis/report.md/results.json`，再分析；自动汇总本身也可能有可修工程问题，不把它误报为训练失败。
6. 比较三干预相对同预算baseline，结合真实CA观测与对象匹配诊断解释因果责任及局限。需要时做合理的保存状态CPU分析，不自动启动新GPU实验。短对照结果不足以宣布长期上限；提出下一轮方案交用户决定。

可用只读命令（Slurm部分在支持集群访问的环境运行）：

```bash
cd /space/mawb/ssst_vggt_overfit_diagnostics_v1
git status --short
git log -5 --oneline
python3 scripts/show_vggt_reading_controls_progress_v1.py
squeue -j 64852,64853,64854,64855 -o '%.18i %.36j %.10T %.12M %.18R'
sacct -j 64851,64852,64853,64854,64855 --format=JobID,JobName%36,State,ExitCode,Elapsed,NodeList -P
```

文档所列历史报告中的“当前排队/下一步未实施”只代表该报告形成时刻；本交接第3节是本次最新观测，但接手时仍以实时证据为准。
