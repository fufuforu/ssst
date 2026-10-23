> **最新：JSON修复后已重提两条链**：71192→71193（Baseline）、71194→71195（U005）；现有71188→71189（2+4）不动。见[最新两组恢复交接](codex_handoff_v4init_jsonfix_retry_two_20261023.md)。已停止轮询，不重复提交。

# 最新检查：2+4正常运行，两恢复组在JSON元数据校验前失败（2026-10-23 08:47）

用户本次要求“检查一下正在运行的作业”。只做一次有界作业/日志检查，随后定位并修复可逆代码缺陷；没有新增、恢复或取消任何作业，未修改正在执行的冻结快照。已停止轮询。

|组|训练状态|节点/卡数|本次已观察进度|评测|
|---|---|---|---|---|
|Baseline U1 2+2|71184 FAILED，exit1，4m44s|3dimage-12，8×3090|恢复校验前失败，未STARTED，新增0步；5500步源保留|71185 CANCELLED|
|U005 2+2|71186 FAILED，exit1，4m29s|3dimage-13，8×3090|恢复校验前失败，未STARTED，新增0步；4172步源保留|71187 CANCELLED|
|U1 2+4|71188 RUNNING，核对时36m24s|3dimage-13，8×3090|累计4455/8344，新增283步|71189 PENDING(Dependency)|

表中为本次单次观察，之后没有重复读取进展；不是持续监控。另有spatial71165排队(Resources)，71166/71167依赖等待，本次未进一步检查或改动其运行。

## 2+4的实际GPU证据

8rank的CONTINUATION_LOADED均PASSED，源progress4/0/4172、604份Adam(step4172)、RNG恢复、模型参数逐值哈希一致；FIRST_UPDATE_VERIFIED为8rank同一更新后哈希且changed=true。各rank日志均283条、4173–4455，loss/PSNR/grad_norm均有限。理解权重1.0，末条主LR2.9867067839827323e-5、几何0.1倍，单步约5.6秒。

rank0最近100个训练样本平均PSNR23.0769dB，这是训练即时指标，不是官方1860评测，也不代表验证集提高。还剩3889更新，按观察到的步速加保存/probes约6–7小时，之后正式评测约1–1.5小时；估计受I/O等影响。

## 两组失败原因和修复

异常在v4init_recovery_policy.recovery_meta→v4init_extend8_policy.transition_meta的current!=expected检查，报`Continuation changed model/assets/data/task recipe`。未写STARTED，未产生RECOVERY_LOADED或训练日志，故没有新增模型更新；恢复源硬链接保留。检查时磁盘仍约35GiB可用，日志没有空间耗尽报错。

真实源Torch metadata与source_epoch4_meta.json逐项比较，两组都只有四个optimizer_groups[i].betas类型不同：Torch保存为tuple(0.9,0.95)，JSON往返为list[0.9,0.95]。新恢复入口把JSON metadata直接交给严格Python相等比较，误判配方变化。这是本次新增恢复代码的缺陷；先前CPU预检传的是内存中的Torch原对象，漏测落盘后读回的路径。2+4走原continue分支，不触发该恢复入口。

修复仅在独立publication worktree：

1. recovery_meta复制输入metadata，仅将optimizer_groups中的JSON betas列表恢复成tuple；所有数值与其余模型/资产/数据/配方比较保持严格，不跳过校验。
2. prepare_v4init_recovery真实预检改为读取已落盘source_epoch4_meta.json，与GPU入口走同一路径。
3. 补充JSON往返回归：先在原代码复现同样异常，再验证4172和5500恢复均通过；另验证实际beta数值改变仍被拒绝。完整13项CPU测试通过（11.207秒测试正文），包含原恢复下一步Adam一致性、LR/RNG及评测收据门禁。

尚未用GPU验证这份修复，也没有重提两组失败任务。执行快照/space/mawb/ssst_v4init_extend8_recovery_code_20261023_v1仍为9c9ffcca5580a1b05003cc31d112177dd5000354且无脚本/model/test改动，保证71188不受影响。修复源码在/space/mawb/ssst_v4init_extend8_publication_20261022_v1，推送到experiments/v4init-extend8-recovery-20261023。

后续若授权重新提交，只修复Baseline/U005两条链、各建新目录及封存快照，仍从5500/4172恢复；不得重提正在运行的71188/71189。准备脚本现有RUN常量指向已提交目录，会主动拒绝，需要新运行版本目录；不要绕过拒绝覆盖旧目录。原4epoch以及旧5500/4172恢复源、审计和失败日志均保留。

本次结构化观察、测试记录保存在原运行根inspection_20261023_0847/；文档归档见v4init_recovery_status_20261023_records/。

已停止轮询，等用户通知后再检查。
