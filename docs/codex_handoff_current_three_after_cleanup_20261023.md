# 接手总交接：当前三组 V4init 续训与清理后的工作区（2026-10-23）

本文件是新对话的优先入口，汇总截至本对话结束的已知事实。历史交接的“最新”“RUNNING/PENDING”都可能过期；作业事实必须沿本文件指定的实际三条链核实。**本次仅写文档，没有查询队列或新训练日志，也没有新增训练、评测、汇总或监控任务。**

## 1. 用户意图与工作约定

先读 `/space/mawb/AGENTS.md`，遵守提交后停止轮询。用户反复强调不要重复提交作业。

当前只做三个既定实验，已授权各自续训到累计8epoch及正式8epoch评测。用户曾表达可能继续16epoch，但没有授权提交8→16；不要自行延长或增加实验组。用户通知“跑完了/检查一下”后，再检查当前作业、日志、完成收据、实际权重和正式指标；失败先诊断，不自动从头训练。

已授权清理不影响当前三组的旧checkpoint，后续又明确允许清理旧MASt3R实验权重；这些清理已完成。不要因主动退役的旧权重缺失而重新训练。配置/日志/指标/导出数据/代码保留。

新对话接手如需确认尚在运行，可做一次有界状态核对后停止，不等待首步或结束，不后台监控。用户明确要求检查完成结果时，则完成该次必要核验。

## 2. 目前唯一应跟进的三条作业链

|组|理解权重|训练输入|当前训练|afterok评测|恢复点→终点|
|---|---:|---|---:|---:|---|
|Baseline|1.0|2context+2novel|71192|71193|5500→8344更新|
|U005|0.05|2context+2novel|71194|71195|4172→8344更新|
|Views2ctx4|1.0|2context+4novel|71188|71189|4172→8344更新|

清理时最后一次有界队列观察：71192和71188 RUNNING，71194 PENDING，三个评测等待依赖。这是**历史调度状态，不是此刻实时状态**。本次写交接未刷新。Baseline提交核对时在3dimage-12（8×3090）；其他当前节点不在本次重新确认。每个训练/评测单节点8卡，32CPU/192G，允许节点池12/13/14/17/18、3090/4090。RUNNING不能替代GPU恢复、前反向、DDP同步和首步新增更新验证。

不要跟进为“待重提”的旧链：71031/71033/71035因存储耗尽失败，其评测71032/71034/71036已取消；71184/71186因JSON元数据betas类型误判失败，其评测71185/71187已取消。现用71192/71194替代前两条；71188/71189一直保留。旧收据和目录不覆盖。

无新8epoch summary作业；各评测已包含本组epoch4→8比较。历史4epoch汇总71014已完成，不重提。

## 3. 运行目录与冻结执行代码

公共根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/`。

- Baseline：`v4init_isolated_extend8_recovery_20261023_v2_jsonfix/Baseline/{train,eval}`。
- U005：`v4init_isolated_extend8_recovery_20261023_v2_jsonfix/U005/{train,eval}`。
- Views2ctx4：`v4init_isolated_extend8_recovery_20261023_v1/Views2ctx4/{train,eval}`。

Baseline/U005执行快照：`/space/mawb/ssst_v4init_extend8_recovery_code_20261023_v2_jsonfix`，SHA **a959b9168f4a3049688171079ac63e52d616273c**。

Views2ctx4执行快照：`/space/mawb/ssst_v4init_extend8_recovery_code_20261023_v1`，SHA **9c9ffcca5580a1b05003cc31d112177dd5000354**。

旧部分续训来源代码：`/space/mawb/ssst_v4init_extend8_code_20261022_v1`，SHA **ce8446d102ac1d9a7347c8525ec38935d4922de3**。保留用于来源核对。

不要修改执行快照源码、HEAD或目录。文档/后续开发使用独立publication：`/space/mawb/ssst_v4init_extend8_publication_20261022_v1`；远程分支 `experiments/v4init-extend8-recovery-20261023`。其文档HEAD与执行SHA不同是刻意隔离。当前清理归档提交为 `9654a20`；本交接会在其后单独提交。

最新恢复细节：[JSON修复后两组交接](codex_handoff_v4init_jsonfix_retry_two_20261023.md)；Views2ctx4来源及公共门禁：[三组恢复交接](codex_handoff_v4init_extend8_recovery_20261023.md)。两文档和全部提交记录也在publication的docs下。

## 4. 实验固定项与学习率，避免接手误改

当前三组共同模型：完整V4重建初始化（FPS1024 anchors、65536GS、持久gs_tokens、XYZ/radius/RGB内容MLP、rawGFM2048的LN→Linear投影）；去掉的是**理解读出中的Anchor CA**，FM CA保留；显式双向梯度隔离，Write0、无adapter。理解query仍当前A4投影点云→动态超点，FM原生128、source_many_to_one/support_purity=.5；理解读出L6/8/10/12，理解监督仅两个context。重建监督L6/L12，权重1/3、2/3。

不能把这套方案描述成“只改理解初始化和去CA”，或“完整迁回早期V4架构”：隔离、几何LR、重建监督层、理解权重等也属于组合变化，当前query/匹配仍为本轮架构。去CA也不等于自动无梯度耦合，显式detach仍保留；统一global clip norm=1会通过总梯度范数间接耦合两分支。

原4epoch主基础LR=1e-4；**仅refine_mu/refine_rho基础LR=1e-5**（48个头张量），不是所有Gaussian属性头一律1e-5。warmup100、cosine floor倍率.1；AdamW betas(.9,.95)、eps1e-8、既有weight decay；理解权重乘min(1,(update+1)/1043)。

4→8采用此前讨论确定的追加训练配方：从本组源4epoch实际末端主LR约1e-5出发，100个新增更新升至3e-5，再cosine降至累计8344时3e-6；几何组始终主LR的0.1倍，即约1e-6→3e-6→3e-7。保持完整模型、Adam moments/step和8rank RNG，不重置优化器，也不把旧4轮曲线追溯拉长。

当前是**恢复上述续训曲线**：Baseline从5500继续原曲线下降（下一主LR约2.437852545768077e-5）；不得再升温。U005/Views2ctx4从4172开始执行既定4→8曲线。1043步理解warmup沿累计时钟，不重启。

数据8337窗/1191场景、256²、global8=8×1；4epoch4172更新，8epoch8344更新。2+4只增加重建监督novel视角，输入context与理解监督不改；2+2重建监督4视角、2+4重建监督6视角。相同更新数不等于相同渲染计算量。

共同V4init零步来源：`oldA4_v4init_noca_ablation_20261022_v1/V4init/train/checkpoint_initial.pt`，SHA **bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397**，已保护。旧A4原始零步SHA **3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744**及共享引用也保留。

## 5. 恢复来源与故障修复的事实边界

Baseline恢复旧71031可读latest：progress=5/285/5500，SHA **13232437521ff24eaf7b808f344ea20a0a8be9294423ff743d175ec0689fe70c**。旧日志到5550但5501–5550未保存，当前从5500状态精确重算；不要把日志5550当源checkpoint进度。

U005恢复旧71033初始化后的latest：progress=4/0/4172，SHA **7bad9b6390c3d613bac13c167548878947539510702d8459c6487356aac7447f**，模型与其原4epoch最终权重相同，但完整续训metadata/Adam/RNG保留。

Views2ctx4由自身原4epoch最终权重开始：progress=4/0/4172，SHA **3a5954de0ea26ff6d70dd1a31c1f2aa073be17b44ffb83af01951cade6268fc5**。

71184/71186失败原因是JSON序列化把optimizer_groups.betas的tuple变为list，严格恢复元数据比较误判。修复只规范该容器类型，数值/配方差异仍拒绝。13项CPU测试通过，并从实际落盘source_epoch4_meta.json读回、重建runtime identity进行实体预检：632模型状态、604份Adam、8rank RNG逐值恢复一致。CPU通过不声称新GPU运行通过。

不再每epoch保留单独权重：当前使用`--no-epoch-checkpoints`，保留起点、latest、原子临时文件和final评测硬链接，epoch probes/TensorBoard保留。**缺checkpoint_epoch8.pt本身不是失败。**

## 6. 已核验的正式4epoch结果，完整11项

三组4epoch训练/评测：Baseline71001/71002、U00571009/71010、Views2ctx4 71012/71013，均已完成并核验；汇总71014完成。共同起点、各自4172更新，模型/Adam有限且实际更新，八rank同步收据通过。

固定官方1860窗/312场景，SIU3R SHA `8ea80166be76854f938e90521f1a5b688b755c87`；**target-all6包含context**，不是novel-only。输入/GT已与旧A4逐窗核对一致，评测更新0。深度指标有scale+shift拟合，不直接视作metric尺度几何。

|组|PSNR|SSIM|LPIPS|AbsRel|RMSE|ctx mIoU|ctx PQ|ctx mAP|tgt mIoU|tgt PQ|tgt mAP|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Baseline_U1_2ctx2|23.051962|0.768544|0.385600|0.052728|0.176459|58.165844|68.419362|24.535201|57.782599|66.064329|23.135531|
|U005_2ctx2|23.197397|0.771784|0.379379|0.052387|0.174517|57.260959|67.754723|23.697737|56.975650|65.585867|22.227938|
|U1_2ctx4|22.978475|0.763999|0.391338|0.051869|0.173334|57.447499|68.710445|24.500170|57.225543|66.315398|23.054257|

理解指标按百分数。原始精度CSV/JSON：`/space/mawb/ssst/group_plus/obsolete_experiment_retirement_20261023_v1/current_three_epoch4_11metrics.{csv,json}`，publication归档也有副本。

目前证据：降理解权重PSNR约+0.145dB，但六项理解指标均下降；尚不支持“1.0明显过大”。2+4未显示整体优势。单seed与小差值不能证明稳定排序。组合隔离配方恢复重建并改善理解，但不能把贡献唯一归因于隔离、embedding、去CA或几何LR的某一项。旧“隔离最好16epoch”PSNR23.294来自128窗，与本轮1860窗不可直接比较。更详细诊断见[codex_handoff_v4init_results_and_extend8_failed_20261023.md](codex_handoff_v4init_results_and_extend8_failed_20261023.md)。

截至上次核验，**没有确认可报告的正式8epoch指标**；本次写交接没有检查是否刚完成，不把4epoch指标说成8epoch。接手检查真实结果后再判断续训收益及是否值得16epoch。

4epoch原运行根：Baseline在`v4init_isolated_noca_v4recipe_20261022_v1/{train,eval}`；U005/Views2ctx4在`v4init_isolated_weight_views_20261022_v1/{U005,Views2ctx4}/{train,eval}`，均相对公共根。

## 7. 已完成清理与必须保留的依赖

最近两批主动清理：102个旧checkpoint路径/84份独立文件释放68.00GiB；随后90个旧MASt3R权重或补丁路径/89份独立文件释放342.8934GiB。更早另有中间权重和取消C/D分支清理，勿把这些数字当同一批重复加算。最近一次df：可用约484G（完成收据约483.08GiB），使用率69%；这是清理后观察，不是永久容量承诺。

审计目录：

- `/space/mawb/ssst/group_plus/obsolete_experiment_retirement_20261023_v1`：102条旧权重清单、删除日志/收据、保护清单及当前三组11项指标。
- `/space/mawb/ssst/group_plus/mast3r_checkpoint_retirement_20261023_v1`：MASt3R最终清单、逐项日志、完成收据、额外保留核验及脚本。

各受影响目录有`CHECKPOINTS_RETIRED_20261023.json`或`CHECKPOINTS_RETIRED_MAST3R_20261023.json`。旧TRAINING_COMPLETE代表历史完成，不保证主动退役后的权重存在；不要因此重提实验。清理脚本完成后拒绝重复执行。

MASt3R权重按实际state keys或实验文档来源确认，不按名字含VGGT一概删除；40份官方评测状态/特征缓存保留且校验未变，32项未确认MASt3R来源模型未选入本次清单。全局MASt3R/SIU3R预训练文件没有删除。

当前三组完整恢复链、初始化、epoch4/final/latest及共享输入均保留。活动/相邻`/space/mawb/ssst/group_plus/spatial_pair_20261023_v1`及其输入保留；历史spatial作业71165/71166/71167不是本三组，勿擅自取消、恢复或清理，最新终态本次未检查。

重要评分参考也必须保留：

- `old_A4_official1860_20261021_node12_8gpu/evaluation/{export_complete.json,gt_reference_manifest.jsonl,metrics.json,checkpoint_identity.json}`，相对公共根。这些是当前正式评测对齐参考；旧权重被退役不代表可以删除参考。
- `/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv`：当前`score_dual_stream_v1.py`仍读取旧MASt3R对照CSV；模型权重已清理，CSV仍必需。
- `/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt`：当前重建预训练资产，未删。
- 所有代码worktree、冻结执行快照、数据集、配置、日志、完成收据、TensorBoard、正式CSV/JSON及RGB/深度/分割导出均保留。

详见[旧checkpoint清理记录](codex_handoff_obsolete_checkpoint_cleanup_completed_20261023.md)与[MASt3R清理记录](codex_handoff_mast3r_checkpoint_cleanup_20261023.md)。最近一次MASt3R删除前后验证39项当前依赖文件身份未变。

## 8. 新对话下一次检查的顺序

1. 先核实71192/71194/71188与71193/71195/71189的实际终态、失败收据及对应日志。不要依靠历史状态重提。若仍运行且用户没有要求持续监控，报告一次后停止。
2. Baseline/U005读新运行的`train/RECOVERY_SOURCE.json`、`RECOVERY_LOADED_rank00–07`；Views2ctx4读`CONTINUATION_SOURCE.json`、`CONTINUATION_LOADED_rank00–07`。确认本轮源SHA/进度5500或4172、完整Adam/RNG和八rank参数哈希一致，`FIRST_UPDATE_VERIFIED.json`必须本次新收据。
3. 完成时Baseline各rank新增日志应2844条/累计5501–8344，另两组各4172条/累计4173–8344；`TRAINING_COMPLETE.json`为epoch8/update_in_epoch0/full_updates8344。权重及Adam有限且实际有更新；Write0未消费状态无须改变。
4. eval从已完成final latest冻结到`source_checkpoint.pt`；核实`FINAL_CHECKPOINT_FROZEN.json`、`EVALUATION_VERIFIED.json`、`evaluation/metrics.json`及`checkpoint_identity.json`。核实1860窗/312场景、SIU3R版本、更新0、target-all6含context、输入/GT与参考一致。
5. 读取各`comparison/epoch4_vs_epoch8.csv`，并统一汇总11项：PSNR、SSIM、LPIPS、AbsRel、RMSE、ctx mIoU/PQ/mAP、tgt mIoU/PQ/mAP；报告与本组4epoch的变化，注明指标方向/口径和单seed限制。
6. 若失败，区分工程I/O、恢复身份、优化/数值等原因，并沿真实落盘checkpoint进度提出最小修复；不把CPU预检当GPU成功，不从日志进度虚构可恢复状态，不自动重启或覆盖旧目录。
7. 8→16尚待结果及用户明确指令，不自行提交。TensorBoard若需继续排查，先读真实event tags和当前日志，不重启训练去补历史标量。

接手时不需要重新跑smoke、重做已完成消融或重复评测；只有新故障或新变更才触发相应针对性验证。已停止轮询，等用户通知完成后再检查。
