> **2026-10-23 当前状态更新**：三组4epoch训练/评测成功；71031/71033/71035因存储耗尽失败，71032/71034/71036取消，无8epoch正式结果。详见[本次检查与恢复状态](codex_handoff_v4init_results_and_extend8_failed_20261023.md)。本文下方PENDING为历史状态，不重提。

# 最新交接：三组V4init隔离实验已提交epoch4→8续训与评测（2026-10-22）

遵守 `/space/mawb/AGENTS.md`。六项新作业已提交，不重提。提交后唯一调度核对已完成，已停止轮询。下述状态只代表提交时，收到用户“跑完了/检查一下”再查。

## 新作业与依赖

北京时间 **2026-10-22 21:13:04** 提交，唯一核对71031–71036均PENDING(Dependency)，尚未分配节点。

|组|理解权重|训练视角|源4epoch训练|源4epoch评测|新续训到8epoch|新8epoch评测|
|---|---:|---|---:|---:|---:|---:|
|Baseline|1.0|2context+2novel|71001|71002|71031|71032|
|U005|0.05|2context+2novel|71009|71010|71033|71034|
|Views2ctx4|1.0|2context+4novel|71012|71013|71035|71036|

依赖分别71001→71031→71032、71009→71033→71034、71012→71035→71036，均afterok。只新增3续训＋3评测；原4epoch训练、评测和71014汇总不改、不重提。三组互不等待，8epoch续训不等待epoch4评测。

每项新作业申请单节点8卡3090/4090、32CPU/192G，允许12/13/14/17/18节点，不同时要求五台。新GPU源加载、前反向、首次新增更新和八卡同步均尚未核实；之后RUNNING也不能等同这些检查通过。没有为写交接再次查询队列或训练日志。

粗估追加四轮约6–8小时，2+4可能更久；每项1860评测约1–1.5小时，源任务剩余和排队另计。依据是此前8×3090 V4init四轮6h50、1860约1h04，未额外监测新组吞吐。

## 封存、来源与审计

- 执行快照：`/space/mawb/ssst_v4init_extend8_code_20261022_v1`
- 执行commit：`ce8446d102ac1d9a7347c8525ec38935d4922de3`，不要改HEAD或运行源码。
- 运行根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_20261022_v1`，Baseline/U005/Views2ctx4各含train和eval。
- publication worktree：`/space/mawb/ssst_v4init_extend8_publication_20261022_v1`
- GitHub分支：`experiments/v4init-three-extend8-20261022`
- 规格：`docs/v4init_three_extend8_20261022.md`；记录：`docs/v4init_extend8_records_20261022/`。

提交前对每组已有checkpoint只观察一次，固定该次硬链接完成实体审计，再清除临时观察链接；不是重复读取训练进度。三组各632模型张量、604份已有Adam状态，名称/顺序/shape/dtype/有限性及8rank RNG覆盖正确；实际模型和完整Adam state_dict恢复逐值一致，重设base_lr不改变moments/step。

观察到的中间进度只是审计来源，绝不用于新续训。最终epoch4 SHA必须在父训练afterok后由门禁冻结；提交configuration.json中的continuation_sha256=null是刻意等待最终源，并非漏配。源最终模型/optimizer布局必须与SOURCE_LAYOUT.json一致，且TRAINING_COMPLETE必须epoch4/4172、父8rank首更新收据通过。

源模型代码：Baseline为310d49d04b6ac1feda55439bb21b6c88a8fc5882；U005/Views2ctx4为ef98a6f9d8ffdba3c39c5cb4660eb65726bd1f2e；不能改成新续训SHA。9个关键模型/loss/decoder/options/采样文件与源快照逐字节相同，见MODEL_SOURCE_IDENTITY.json。

真实epoch5首窗index746已读，2+2与2+4分别4/6个重建监督视角，context始终2个。模型、Adam状态、8rank RNG、数据清单、原视角协议均按组保留。

9项CPU测试通过，8.495秒；包括恢复Adam后下一步与连续参考逐值相同、来源/optimizer非法变化拒绝、100步升温/余弦LR/几何0.1比、理解warmup不重启、实际文件冻结门禁与重复拒绝、epoch5采样/RNG、真实core前向与双向隔离回归。CPU不代表GPU续训已经通过。

源审计、数据核对、独立目录、测试/编译/shell语法/diff检查、Slurm test-only与代码封存均在实际提交前完成。

## 用户通知完成后的检查入口

1. 71031/71033/71035训练与71032/71034/71036评测，以及各JOB_FAILED.json和Slurm日志；不要因旧PENDING而重提。
2. train/CONTINUATION_SOURCE.json、CONTINUATION_LOADED_rank00–07、FIRST_GRADIENTS_rank*.json、FIRST_UPDATE_VERIFIED.json。必须源epoch4/4172、Adam与RNG保留，实际LR比.1、理解不重新warmup。
3. train/TRAINING_COMPLETE.json必须epoch8/update_in_epoch0/full_updates8344；新增各rank日志4172条（4173..8344）。最终参数/Adam有限且有更新；不机械要求Write0未消费参数都变化。
4. checkpoint_initial.pt是epoch4的真实训练起点，不是零步。eval/FINAL_CHECKPOINT_FROZEN.json、EVALUATION_VERIFIED.json、evaluation/metrics.json、checkpoint_identity.json与comparison/。

原4epoch评测若届时尚未完成，8epoch仍评分并在comparison.json标epoch4_reference_pending=true；无等待或重评。用户之后检查时可补读取三组结果统一分析。本轮没有新增独立跨三组汇总job。

SIU3R固定commit8ea80166be76854f938e90521f1a5b688b755c87、1860窗/312场景、target-all6含context，不改SIU3R/src。

STARTED后的训练失败不自动重跑。旧I/O bootstrap只允许STARTED前明确文件系统超时。不自动继续8→16，须用户之后另行决定。

前置交接：[权重/视角两组](codex_handoff_v4init_weight_views_20261022.md)、[V4配方基线](codex_handoff_v4init_isolated_noca_20261022.md)。70714–70719已完成，不重提；取消C/D不恢复。

已停止轮询，等用户通知完成后再检查。

---

以下为提交前完整实验规格，便于脱离对话复核：

# 三组V4init隔离实验：epoch4→8续训与官方1860评测（2026-10-22）

用户明确授权三个续训任务和三个各自epoch8评测；不安排16epoch或额外实验。原epoch4训练/评测/汇总保持不变。

|组|源4epoch训练|源4epoch评测|理解权重|训练视角|
|---|---:|---:|---:|---|
|Baseline|71001|71002|1.0|2context+2novel|
|U005|71009|71010|0.05|2context+2novel|
|Views2ctx4|71012|71013|1.0|2context+4novel|

## 明确的续训策略

- 每组从自己的最终epoch4/4172更新继续到epoch8/8344更新，只追加4172更新/33376窗口曝光。沿用epoch索引4–7的原确定性采样，不重播0–3。
- 主LR首步取源checkpoint的实际末端值（约1e-5），100个新增更新线性升至3e-5，之后在新增4轮内cosine降至3e-6。refine_mu/refine_rho实际LR始终为主LR0.1倍：约1e-6→3e-6→3e-7。
- 不将旧4轮曲线追溯拉长至8轮；这是追加训练＋显式LR重启，不是单纯epoch数量单变量。
- 保留模型全部张量、Adam一二阶动量/step、全部8rank Python/NumPy/Torch/CUDA RNG。只更改续训LR调度所需的optimizer base_lr和训练metadata，不重置Adam。
- 理解权重保持1.0/0.05/1.0，1043步理解warmup的累计时钟不重启（续训时已完成）。Write0、全局clip1、AdamW/betas/eps/weight decay、隔离/CA/FM/query/匹配、监督层和loss保持源组。
- 完整V4初始化1024Anchor/65536GS，L6/L12[1/3,2/3]，FM128，源支持多对一匹配.5，understanding loss仍只监督两个context。global8=8×1，数据8337窗/1191场景。

## 提交前审计与源最终门禁

提交前对每组已有checkpoint**只读一次**并用硬链接固定该次观察，验证632模型张量及604份已有Adam状态的名称/顺序/shape/dtype/有限性、rank RNG覆盖、源架构/代码/数据及对应配方。CPU实体模型恢复后632张量逐值相同，Adam完整state_dict恢复相同，LR metadata重设不改变moments/step。审计只是准备证据，不会把中间checkpoint用于续训。

三组源epoch4仍可能在训练，因此最终权重SHA在提交前不能预先声称已确认。新续训afterok依赖对应源训练。启动门禁必须看到TRAINING_COMPLETE、epoch4/4172完整进度、8rank首更新收据，才冻结**checkpoint_epoch4.pt**到新运行目录并核验最终source SHA、模型布局及Adam。没有中间checkpoint回退；源失败时续训被阻断。

每rank训练启动再次核验实际模型、优化器与冻结源，严格比较允许变化之外的全部metadata。模型参数加载哈希与源一致；记录CONTINUATION_LOADED_rank00–07及新增首步FIRST_UPDATE_VERIFIED。首步在累计update4173，TensorBoard继续使用累计步数，写train/psnr等原指标。

真实epoch5首窗index746，2+2/2+4实际batch分别4/6个重建监督视角；context输入始终2个。9个关键模型、loss、decoder、options和采样文件与原执行快照逐字节相同。源证据见CPU_SOURCE_VALIDATION.json、MODEL_SOURCE_IDENTITY.json及各组PREFLIGHT.json/SOURCE_LAYOUT.json。

针对性测试覆盖：LR边界/100步升温/末端/0.1分组比、理解warmup不重启、Adam+模型恢复后下一步与连续参考一致、拒绝部分epoch/重置Adam/时钟错误/丢rank RNG/配方改变、拒绝optimizer名称顺序/shape/NaN变化、真实afterok冻结门禁拒绝重复启动、累计epoch5采样和RNG恢复；模型CPU回归覆盖双向梯度隔离与前向逐值一致。CPU验证不等于新GPU续训通过。

## 六个作业的独立依赖

每组：已有4epoch训练 → afterok 新8epoch续训 → afterok 新epoch8正式评测。只新增这六个任务，三组不互相等待。原epoch4评测和71014汇总不改动。续训不依赖epoch4评测完成，所以可独立推进。

新epoch8评测固定1860窗/312场景、target-all6含context、SIU3R commit `8ea80166be76854f938e90521f1a5b688b755c87`。只读最终epoch8权重，评测更新0；输入和GT逐窗匹配既有旧A4。每组在自身epoch4正式评测可用时输出comparison/epoch4_vs_epoch8.csv和差值；若其尚未完成，保留epoch8正式结果并记录epoch4_reference_pending=true，不等待或重评。

## 路径与停止轮询

执行快照：`/space/mawb/ssst_v4init_extend8_code_20261022_v1`。
运行根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_20261022_v1`，Baseline/U005/Views2ctx4各含train/eval。
GitHub分支：`experiments/v4init-three-extend8-20261022`。

单节点8卡3090/4090，节点池12/13/14/17/18。提交后只一次调度接收核对，然后记录与交接，不为等待源完成、续训首步或评测而轮询。用户通知后再检查。

完成核验：train/CONTINUATION_SOURCE.json、CONTINUATION_LOADED_rank*.json、FIRST_UPDATE_VERIFIED.json、TRAINING_COMPLETE.json（epoch8/update_in_epoch0/full_updates8344）、新增日志应4173..8344；checkpoint_initial.pt是epoch4训练起点，不是未训练零步。eval/EVALUATION_VERIFIED.json、FINAL_CHECKPOINT_FROZEN.json、evaluation/metrics.json、checkpoint_identity.json及comparison/。

不存在自动继续epoch8→16；那需要用户之后另行决定。STARTED后的训练失败不自动重跑，失败保留JOB_FAILED.json与Slurm日志。
