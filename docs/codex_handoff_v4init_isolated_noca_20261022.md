# 交接：V4init + NoCA + 梯度隔离 + 早期V4配方（2026-10-22）

先遵守 `/space/mawb/AGENTS.md`。本交接只记录提交时唯一有界状态核对，不代表之后的实时进度。**不要因旧RUNNING/PENDING重提任何作业；等用户通知完成后再检查。**

## 最新授权与实验

用户在本轮V4init基础上明确选择：移除Object→Anchor CA，显式隔离理解到重建的梯度，refine_mu/refine_rho基础LR改1e-5；进一步确认完整采用早期V4的L6/L12监督[1/3,2/3]和理解loss权重1.0。

仅这一组，没有扫描或额外对照训练。仍是当前A4理解query、source_many_to_one/support_purity=.5、原FM128；FM CA保留。数据维持8337窗/1191场景、2context+2novel、256²、global8、4epoch4172updates，官方1860窗/312场景target-all6（含context）评测。不是完整早期V4架构迁移。

重建初始化保留上一组V4init的1024 Anchor/65536GS及全部9个初始化张量。**使用V4init零步checkpoint_initial.pt，不使用epoch4训练权重。** 源SHA `bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397`；源meta原A4零步来源SHA仍为`3c7f1860...b25744`。新组不新增adapter，不改变seed/GFM函数。

主LR1e-4、geometry heads1e-5，同warmup100/cosine乘数floor.1（末端分别约1e-5/1e-6）；理解权重1.0×min(1,(update+1)/1043)。AdamW原betas/eps/weight-decay规则，全参数统一clip1沿用早期V4，因此裁剪比例仍可能跨任务耦合，不能把detach说成完全独立优化器轨迹。

## 代码与运行目录（封存后不可改执行HEAD）

- 执行快照：`/space/mawb/ssst_v4init_isolated_noca_code_20261022_v1`
- 执行commit：`310d49d04b6ac1feda55439bb21b6c88a8fc5882`
- 运行根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1`
- 文档publication worktree：`/space/mawb/ssst_v4init_isolated_noca_publication_20261022_v1`
- GitHub分支：`experiments/v4init-isolated-noca-v4recipe-20261022`
- 实验规格：`docs/v4init_isolated_noca_v4recipe_20261022.md`

## 提交收据与最后一次核对

北京时间2026-10-22 **17:44** 提交，依赖链为：

|阶段|作业|依赖|资源|该次核对状态|
|---|---:|---|---|---|
|真实GPU单窗2步smoke|71000|无|单卡3090，12节点|RUNNING|
|完整4epoch训练|71001|afterok:71000|单节点8卡3090/4090，节点待分配|PENDING(Dependency)|
|正式1860评测及结果汇总|71002|afterok:71001|单节点8卡3090/4090，节点待分配|PENDING(Dependency)|

训练/评测节点池12/13/14/17/18；没有要求同时占用五台。smoke失败阻断训练，训练失败阻断评测。正式训练重新加载零步及原8rank RNG，绝不继承smoke权重/Adam。

**GPU执行、smoke是否通过、8rank首步更新均尚未检查；RUNNING仅是调度运行。** 后续没有为了补交接再次查询队列、读取smoke/训练日志或等待首步。耗时仅参考此前V4init 3090训练6h50、正式评测1h04；本组不同隔离和LR，粗估训练6–8小时、评测1–1.5小时，排队另计。

## 提交前验证完成

- 2项针对性CPU测试11.144秒通过：NoCA joint与显式隔离core前向逐值一致；LU→重建梯度精确0、LR→理解/共享投影梯度精确0；9个初始化张量均可获LR梯度；优化器覆盖、分组LR和schedule ratio正确。
- 真实源零步审计：632张量逐值保留、精确移除32个Anchor CA张量、无新增。195961456可训练参数。48个mu/rho头张量使用geometry LR。
- 严格拒绝8种非法迁移，详见运行根CPU_VALIDATION.json；测试日志CPU_TESTS.txt/json。
- CPU采用合成8²渲染/6个anchor验证真实core和loss，不能代替实际GPU。已提交的smoke使用真实1024/65536、256²、2+2、8次累积/global8、真实梯度隔离与初始化梯度验证。
- 源模型/源数据可比性、独立目录、shell语法、Python编译、提交参数test-only、代码封存均在实际提交前完成。没有把准备留到停止轮询之后。

## 结果检查入口

收到用户“跑完了/检查一下”后一次核对71000/71001/71002及：

- smoke/SMOKE_PASSED.json、DIRECTIONAL_GRADIENTS.json、STEP_RESULTS.json：必须双向隔离，真实LR ratio和U warmup正确，9个初始化梯度非零。
- train/FIRST_UPDATE_VERIFIED.json、V4INIT_ZERO_INITIALIZATION.json、TRAINING_COMPLETE.json、8rank日志；进度必须epoch4/update_in_epoch0/full_updates4172。源码正常记录anchor位移、半径变化、geometry/main实际LR和clip前norm，便于分析几何。
- eval/FINAL_CHECKPOINT_FROZEN.json、EVALUATION_VERIFIED.json、evaluation/metrics.json、checkpoint_identity.json；固定1860窗输入/GT匹配旧A4、评测更新0。
- eval/comparison/vs_old_A4_and_V4init.csv 自动对比旧A4、本轮V4init joint、旧A4 NoCA joint、新组；汇总在评测71002内部执行，无额外汇总job。
- 失败看JOB_FAILED.json及对应Slurm日志，不静默重训。旧I/O bootstrap重试只适用于STARTED前明确文件系统超时，不会自动恢复STARTED后的训练。

组合配方是否有效可以比较；不能把结果单独归因于去CA、隔离、LR、监督层或理解loss权重中的任一项。

## 上一轮已完成结果（不是仍运行）

70714–70719全部COMPLETED 0:0；V4init训练70715/评测70716在13节点8×3090，NoCA训练70718/评测70719在12节点8×3090。均4epoch4172步、8rank更新、最终模型有限、Adam step4172、权重与评测冻结hardlink及SHA一致。输入/GT匹配1860窗/312场景，评测参数更新0。

|模型|PSNR|AbsRel|RMSE|Context PQ|Target mIoU|Target PQ|Target mAP|
|---|---:|---:|---:|---:|---:|---:|---:|
|旧A4|18.20926|.186947|.410051|65.35344|55.23170|63.14721|19.10869|
|V4init joint|18.13261|.213299|.460933|61.75372|51.39058|57.14629|14.51325|
|旧A4 NoCA joint|18.12626|.189256|.416349|66.15839|55.13307|63.85902|18.94069|

V4init的Target PQ -6.00点，19/20类下降；NoCA +.712点，14/20类提高，但mIoU/mAP及重建略降。原评测未保存TP/FP/FN累计状态，不能归因误检/漏检。导出metadata unique_scenes误填1860窗口数；按official_pair.scene去重确为312，输入和GT一致性不受影响。

本地完整结果：旧运行根RESULTS_20261022.md/csv及RESULT_CHECKS_20261022.json。本publication docs/v4init_isolated_noca_records_20261022/也复制了PREVIOUS_结果与核验文件。

早期V4隔离确曾在相同128窗达到23.194dB；当前实验不把这个128窗分数和1860窗分数直接混减。早期23dB说明隔离在该架构有效，不代表P1024iso的隔离单独已经失败，也不证明本轮初始化单独足以救回重建。

其它取消/历史边界沿用 [上轮交接](codex_handoff_olda4_v4init_noca_20261022.md)。此前C/D70692/70693/70695/70696已取消，不恢复；不擅自安排新组或恢复旧69254。

已停止轮询，等用户通知完成后再检查。
