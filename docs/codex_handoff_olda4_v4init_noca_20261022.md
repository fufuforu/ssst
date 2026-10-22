# SSST 交接：旧 A4 的 V4 重建初始化 / No-Anchor-CA 实验（2026-10-22）

## 1. 接手先读与工作边界

用户要求先理解现状，再按明确指令行动。必须遵守 `/space/mawb/AGENTS.md`：提交前完成代码、可比性核对、针对性验证和独立目录；提交后只做一次有界状态核对，之后停止轮询。用户通知“跑完了 / 检查一下”再查结果。不要因为旧文档写着 RUNNING/PENDING 就重提、恢复或重复训练。

**本交接只整理已有证据，没有为写文档再次查询队列或训练进度。下述运行状态最后核实于北京时间2026-10-22 09:49，接手时不代表实时状态。**

本轮用户的最新研究决策：先聚焦结构，弄清 **重建初始化**、**理解 Query→重建 Anchor CA 是否必要**。监督层、提取 GS 特征的 FM 分辨率等暂时固定，不增加这些扫描；不要擅自加入 LR 扫描、隔离或 adapter 等新变量。

## 2. 现在应关注的两个实验

代码执行快照（不可修改 HEAD 或源文件）：
`/space/mawb/ssst_olda4_v4init_noca_code_20261022_v1`

执行 commit：`5335430b6f0e40fccb042162ae47f4723da32669`。

独立运行根目录：
`/space/mawb/ssst/group_plus/semantic_resolution_v1/oldA4_v4init_noca_ablation_20261022_v1`

|实验|单卡2步smoke|8卡训练|训练成功后的8卡1860评测|
|---|---:|---:|---:|
|V4init：旧 A4 + 完整 V4 重建初始化|70714|70715，afterok:70714|70716，afterok:70715|
|NoCA：旧 A4 去掉 Anchor CA|70717|70718，afterok:70717|70719，afterok:70718|

每组为独立链，不互相等待。smoke失败阻断训练，训练失败阻断评测。正式训练从原零步重新开始，不继承smoke权重或优化器。

### 最后一次核实状态（09:49）

- 70715：RUNNING，3dimage-13，8×3090，日志约150/4172 updates；最近100步平均6.036秒/步。
- 70718：RUNNING，3dimage-12，8×3090，日志约163/4172 updates；最近100步平均5.304秒/步。
- 两组 `SMOKE_PASSED.json` 存在；训练前的smoke gate已经放行。
- 两组 `FIRST_UPDATE_VERIFIED.json` 为 PASSED；已检查8rank均更新、首步权重哈希一致。因此此处有实际训练证据，不只是调度系统 RUNNING。
- 同次队列检查中70716/70719均为PENDING(Dependency)。不再额外轮询。
- 当时预计：V4init训练还需7–8小时（当天17–18点），NoCA还需6–7小时（当天16–17点）；包含checkpoint和阶段probe余量。正式1860评测额外约1–1.5小时，排队另计。这些是当时估计，不能作为后续实时进度。

节点池沿用12/13/14/17/18；单节点，3090或4090。不要用多节点nodelist误要求同时分配五台。

## 3. 已取消的上一批 C/D，不要恢复

用户认为没有必要继续，明确要求停止P1024iso C/D：

|内容|作业号|已核实状态|
|---|---|---|
|C：Anchor CA开、FM CA关训练|70692|CANCELLED|
|C依赖评测|70693|CANCELLED|
|D：Anchor CA关，几何R；FM CA关训练|70695|CANCELLED|
|D依赖评测|70696|CANCELLED|

已执行scancel并用一次sacct确认上述四项CANCELLED。
取消收据：`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_nofmca_ablation_20261022_v1/USER_CANCELLED.json`；各train/eval目录也有收据。
原单卡smoke为70691/70694；取消操作没有检查其最终状态，不要推断其结果。旧A4新实验70714–70719未被取消。

历史代码和运行目录保留用于审查，不需删权重、不需重新提交。

## 4. 两个新实验的精确定义

### 4.1 共同基线：旧 A4，不是 P1024iso

- 8337训练窗 /1191场景，原manifest与采样次序。
- **训练2 context +2 novel**，256×256。此前用户要求P1024iso改2+4，这次明确回到旧A4，不能继承那个改动。
- 8卡×1、全局batch8，4epoch /4172 updates。
- AdamW betas=(.9,.95)、eps=1e-8，原weight-decay分组，全参数统一clip=1。
- **全部参数基础LR=1e-4**，warmup100、cosine→1e-5；理解权重 `.05*min(1,(step+1)/1043)`。
- 重建监督 **L6/8/10/12，权重[.1,.2,.3,.4]**。不是此前sup612配方。
- 理解query由原投影点云特征→超点初始化，256D，四层理解decoder。
- SemanticMemory原生 **128×128**；GS feature lifting前依照旧路径插值到256。不是P1024iso的原生FM256，也不是FM初始化理解query。
- **联合梯度、无隔离、无adapter**；Write=0。不加IoU/depth/boundary loss。
- 对比用同一个旧A4零步权重重新训练，不能拿epoch4接着训练。

原零步：
`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt`

SHA256：`3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744`。

训练manifest：`/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json`
SHA256：`87e3cde298569c4e7735cf75ee596868eadf01e9a6a95438c6be51e42f55a523`。

### 4.2 V4init 组：用户明确选择完整初始化迁移

用户曾被问“保留旧A4动态超点数量，只改特征；还是完整迁移V4固定1024初始化”，回答选择后者。

流程：

```text
VGGT points/confidence + context RGB
    → 原 seed_anchors(count=1024, return_source_indices=True)
    → mu[1,1024,3]、radius[1,1024]、RGB、精确来源像素索引
原始 GFM：四层归一化特征均值，2048D、32×32
    → 按种子来源像素双线性采样 → visual[1,1024,2048]
content = concat((mu-origin)/ell, radius/ell, RGB)    # 7D
初始 token = 持久可学习 gs_tokens[1024,1024]
           + MLP(7→1024→1024)(content)
           + (LayerNorm(2048)→Linear(2048→1024))(visual)
    → 原12层重建decoder、原GS头，每Anchor64个GS，共65536
```

这里的视觉分支是 **LN后Linear** 的整体投影，与另两项相加；不是把LN输出直接与1024维token相加。

`gs_tokens`用原V4的gs_token_std初始化，作为模型参数跨场景保存和更新；position/radius/RGB内容编码及原始GFM视觉编码与V4公式相同。原seed函数、GFM构造/采样函数不改。

RGB只是token内容输入，没有直接初始化Gaussian颜色头。不使用旧2048→256→1024的重建特征瓶颈。

只替换重建初始化；理解超点query来源、初始化权重、decoder、FM、匹配策略和损失不改。无用的重建超点初始化器及旧256→1024投影被精确移除。

该实验同时改变Anchor/GS数量、持久基底、内容编码和视觉投影；**只能检验整套初始化是否有效，不能单独证明“容量有效”或“可学习基底有效”。** 也不是整套V4模型或隔离训练的迁移。

父索引：`gaussian_parent_anchor`和兼容名`gaussian_parent_query`都表示重建Anchor索引；显式`gaussian_parent_query_semantics=legacy_alias_of_anchor_index_not_object_query`。移除不再成立的`anchor_to_superpoint`输出；理解匹配不使用这些父索引。

### 4.3 NoCA 组：仅移除旧 A4 的 Object→Anchor 学习式CA

- 重建Anchor仍是原超点初始化，数量动态、每Anchor64GS。
- 四个理解读出层移除Anchor Attention Q/K/V/O，读出ev=0，保留`norms[0](q+ev)`。
- `R=geometry_bias(mu,c,s).softmax(-1)[:,None]`，传入原`update_geometry`更新c/s；stuff行均匀。不会出现未定义R。
- Object→FM CA、自注意力、FFN、共享初始化和逐GS特征融合均保留。
- 没有新增detach。LU仍可以通过GS几何、child mask特征、共享特征投影等路径影响重建。**去掉一个CA不等于梯度隔离。**

## 5. 当前训练预测 mask 与 GT 怎样配对

两组配置完全一致：`matching_mode=source_many_to_one`、`support_matching=True`、`support_purity=.5`。**当前没有使用匈牙利一对一匹配，也没有根据预测mask IoU来配对。**

1. 仅用两个Context视角的GT，将相同实例ID在两视角的mask组成一个监督目标；GT类别用该实例有效像素类别众数。
2. 每个理解query对应一个超点。通过保留的原视图/像素source_indices，读取该超点采样来源的GT标签。
3. support(q,k)=超点q中属于GT实例k的有效来源点数 /超点q的所有有效来源点数。分母包括有效stuff点；无效/void与thing实例ID缺失的点忽略。
4. 至少50%才允许配对；多个候选取support最高。多个query可以匹配同一GT。
5. 配对后，每个query的逐GS membership经过渲染及alpha归一化，得到Soft region_mass，与GT在两个Context视角的**完整实例mask**计算BCE+Dice，并监督类别。来源是局部超点，但目标不是局部超点mask。
6. 损失先在同一GT对应的query内平均，再对GT平均，避免超点多的GT占更大权重。
7. 未匹配query作为no-object，参与低权重分类，无正实例BCE/Dice；无足够纯超点支持的GT可能无匹配，会记录unsupported_GT_count。
8. 两个stuff query直接监督wall/floor，不参与实例配对。

1024是V4init的重建Anchor数量，不是匹配GT的理解query数量。理解loss仍仅监督Context；RGB重建用原2+2四视角。

入口：`tokengs/models/unified_query_loss.py`的`support_relation`和`unified_instance_loss`；GT构建在`object_locus_v3_set_loss.py:build_context_instance_targets`；渲染在`unified_query_v1.py:_render_understanding`。

## 6. 实现入口、验证和运行收据

执行快照内：

- `tokengs/models/olda4_initialization_ablation.py`：两组模型/core、V4初始化、几何NoCA。
- `tokengs/models/superpoint_dual_core.py`：默认行为不变的branch/anchor初始化hook。
- `tokengs/models/superpoint_structural_variants.py`：注册`dual_olda4_v4init` /`dual_olda4_noca`。
- `scripts/olda4_ablation_initial_state.py`：严格零步迁移与recipe_identity。
- `scripts/train_unified_query_v1.py` /`export_unified_query_checkpoint.py`：新架构训练与评测身份核对。
- `tests/test_olda4_ablation.py`：五项针对性CPU测试。
- `scripts/prepare_olda4_ablation.py`：真实源零步审计、独立目录与配置准备。
- `scripts/smoke_olda4_ablation.py`：真实GPU单窗2步，全局8按8次累积，原2+2、256²；V4init真实1024/65536，NoCA原动态数量。
- `scripts/olda4_ablation_pipeline.py`：smoke/训练完成门禁、冻结checkpoint、对齐1860输入与GT、对比旧A4。
- `scripts/run_olda4_ablation.sh` /`submit_olda4_ablation.py`：依赖链及唯一有界状态核对。

旧A4 decoder、seed/GFM/超点模块、SemanticMemory、unified_query_v1、unified_query_loss、训练调度策略等8个源文件与旧A4原文件逐字节相同；为了恢复旧A4，部分文件相对P1024分支是显式回退，不能误当作给运行中的P1024改代码。

CPU测试5项通过，耗时16.148秒；真实零步迁移：

|组|逐值保留原张量|精确移除张量|新增张量|可训练参数数|
|---|---:|---:|---:|---:|
|V4init|655|38（重建超点初始化36＋旧投影2）|9|197014128|
|NoCA|661|32（四层Anchor CA）|0|192746609|

每组验证拒绝7类非法迁移：已训练权重、非空Adam、数据不符、rank RNG不全、额外张量、缺失保留张量、移除张量shape错误。

CPU只缩小渲染用于实际loss路由测试，不能代替GPU；09:49已另确认smoke gate和8rank首步更新通过。

运行收据路径（相对运行根目录）：

- `CPU_VALIDATION.json`、`SUBMITTED.json`、`STATUS_AFTER_SUBMIT.txt`（后者只代表最初提交状态）。
- 每组`smoke/SMOKE_PASSED.json`、`STEP_RESULTS.json`、`DIRECTIONAL_GRADIENTS.json`、`FIRST_UPDATE_VERIFIED.json`、`BUDGET.json`、`BATCH_SHAPES.json`。
- 每组`train/STARTED.json`、`OLDA4_INITIALIZATION.json`、`SMOKE_GATE_VERIFIED.json`、`FIRST_UPDATE_VERIFIED.json`、`FIRST_GRADIENTS_rank*.json`、`train_rank*.jsonl`。
- 完成后应有`train/TRAINING_COMPLETE.json`，progress必须epoch4/update_in_epoch0/full_updates4172；每epoch probe不是正式指标。
- 失败看`JOB_FAILED.json`与Slurm日志；不要静默重复训练。已有I/O bootstrap重试只允许STARTED前的明确文件系统超时，STARTED后不自动重新训练。
- 评测完成后看`eval/EVALUATION_VERIFIED.json`、`evaluation/metrics.json`、`checkpoint_identity.json`及`comparison/vs_old_A4.csv`。

## 7. 评测口径与已有对照

固定官方1860窗 /312场景，cohort：
`/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json`
SHA256：`0e4cec01a6fd17dd5a89ba0c89128af85f18017a1509dd4050c2315546dab951`。

SIU3R固定commit：`8ea80166be76854f938e90521f1a5b688b755c87`。不改`/space/mawb/SIU3R/src`。

target保持既有**target-all6（含context）**口径，不是单独novel4。生成只读Context；GT仅在监督/评分阶段使用。正式评测固定权重，不训练。

已完成的官方1860（百分数列为百分比）：

|模型|PSNR|AbsRel|RMSE|Context mIoU|Context PQ|Target mIoU|Target PQ|
|---|---:|---:|---:|---:|---:|---:|---:|
|旧A4|18.20926|.186947|.410051|55.94314|65.35344|55.23170|63.14721|
|P1024iso|18.50274|.206897|.451476|54.17143|62.01948|53.35396|59.05575|
|P1024iso No-Anchor-CA|18.68690|.211182|.460095|54.97380|63.44598|54.18155|60.52644|

旧A4对照目录：`/space/mawb/ssst/group_plus/semantic_resolution_v1/old_A4_official1860_20261021_node12_8gpu`。
P1024iso评测：`.../point1024_fmquery_isoadapter_official1860_after70026_20261021_v1`。
P1024iso NoCA评测：`.../point1024_isoadapter_noca_official1860_after70162_20261021_v1`。

P1024iso训练70026 /评测70042、NoCA训练70162 /评测70183此前均已检查COMPLETED，不能重提。相对旧A4有多项配方变化，不能据这些结果单独断言“梯度隔离导致理解退化”。P1024 NoCA小幅改善也不能直接推断旧A4去CA一定有效，所以现在才有旧A4 NoCA实验。

PQ是20类等权平均，单种子2–3pt以内需看逐类及TP/FP/FN；PQ数组是things-first（语义3..20、1、2），mIoU是语义1..20，不能按wall-first误贴PQ类别。

## 8. 用户当前关于初始化的判断与解释边界

用户认为此前unified query表现不佳，可能因为token不是TokenGS那种持久可学习向量，只是模型学了初始化方式。这个假设值得验证，但还没有得到因果结论。

准确区别：

```text
超点初始化：q_s = f_theta(场景s的超点特征、几何)
持久基底：  q_s,i = e_i + 场景内容编码
```

前者每个场景重新生成query，优化器保存/更新的是f_theta、decoder等参数，不保存该场景激活值。**但token在decoder中会更新，损失也通过token反传训练初始化器，并非“初始化之后不学习”。** 后者的e_i是跨迭代持久的模型参数，能够学习稳定槽位起点与跨场景生成先验。

理解query由超点出发有明确区域支持；重建token可能更需要稳定可学习基底，二者不必使用相同初始化。旧unified query还承担重建/理解双任务，因此之前效果差也可能涉及任务耦合，不可全部归因于初始化。

新V4init恰好检验重建侧改用持久基底+内容/视觉编码、理解侧保留超点初始化的方向；但它同时改变容量和投影，结论只能针对整套方案。

用户已明确：监督层/FM分辨率后续可以研究，**目前不安排**。我们认可其当前优先级较低，但监督层会改变梯度、FM分辨率可能影响边界/小物体，并非已证实无关紧要。

## 9. 结果出来后的判断顺序（尚未授权新增提交）

|结果|建议|
|---|---|
|仅V4init稳定有效|优先保留完整初始化方案|
|仅NoCA稳定有效|优先保留NoCA|
|两者各自有效|可向用户建议V4init+NoCA组合，检验交互；当前未提交该第三组|
|都无明显收益|回到旧A4，重新定位瓶颈|

同时看重建、深度和理解指标，不只看PSNR；小幅PQ变化看per-class和TP/FP/FN。先检查源checkpoint、预算、输入与GT一致性、正式评测完成收据，再解释性能。

此前V4隔离在相同128窗可到23.194dB，旧A4约18.500，P1024iso匹配同128约18.6994、NoCA约18.9582。差距不是换评测cohort即可解释。V4还采用geometry基础LR1e-5而旧A4/P1024统一1e-4，并有不同初始化；不能声称隔离本身就应得到23dB。**本轮不擅自加入geometry LR分组。**

更早“监督层已用sup612裁决”的结论属于另一实验族；本轮为了旧A4可比性，明确保持其原四层监督，不能自动改sup612。

## 10. 其它已完成工作和历史资料

旧A4/A4+4固定128窗的共享Anchor/Child特征与Boundary纯前向诊断已完成（70234 smoke、70235 full），代码/报告已发布，不能重跑：
GitHub分支 `diagnostics/anchor-child-boundary-20261021`，报告 `docs/anchor_child_boundary_results_20261021/REPORT.md`。
本地运行根目录：`/space/mawb/ssst/group_plus/anchor_child_boundary_diagnostics_20261021_v1`。

结论：A4→A8的同Anchor跨实例cosine间隔、membership JS间隔未见统计上明确改善；原始共享Anchor特征占主导，但FM融合后已有区分，不能据此证明因果压制。Boundary BF1@2约.323→.402，仍有明显物体内部伪边界，暂不足以直接进入当前Boundary-aware RGB loss方案。没有新增Boundary Loss。

历史主交接：
`/space/mawb/ssst_vggt_overfit_diagnostics_v1/docs/codex_handoff_a_reconstruction_capacity_20261020.md`
与同目录 `codex_handoff_supplement_official1860_iouhead_20261020.md`。
这些文件中的RUNNING/PENDING可能早已过期。旧69254等训练已不继续，IoU epoch8只有probe无找到备份权重，用户选择找不到则暂不评测，不要自行重训。

过去失败的容量×16、单纯延长训练、匈牙利一对一、纯NMS去重不应原样重复。用户曾指出旧容量实验预算和监督层不一致，这是对旧结论适用范围的质疑，不等于授权不加控制地重跑。

## 11. GitHub与文档发布

仓库：`https://github.com/fufuforu/ssst`。
本轮分支：`experiments/old-a4-v4init-noca-20261022`。
实验规格：`docs/oldA4_v4init_noca_20261022.md`。
运行记录：`docs/oldA4_v4init_noca_records_20261022/`。

执行快照HEAD封存；补文档使用独立publication worktree：
`/space/mawb/ssst_olda4_v4init_noca_publication_20261022_v1`。
本交接也复制到公共交接目录，便于新对话从路径读取。不要在执行快照中提交文档导致训练meta/code_sha改变。

用户此前明确授权代码、报告推送GitHub及使用其代理；不推checkpoint、数据集或大量中间张量。SSH443+HTTP代理7890可用，避免输出凭据或改全局git配置。

最后磁盘核实 `/space` 为193T总量、34T可用、83%；历史“222GB剩余”已过期，但仍应使用硬链接checkpoint、避免冗余大文件。此处不是写文档时重新测得的磁盘状态。
