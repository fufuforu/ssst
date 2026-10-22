# 三组V4init检查结果：4epoch成功，续训因空间耗尽失败（2026-10-23）

本次按用户“作业都完成了 请检查”只读检查作业、日志、完成收据、实际checkpoint和正式评测。不重提任何训练或评测，不删除文件，不修改执行快照。

## 当前真实状态

|组|4epoch训练/评测|续训作业|节点/卡数|续训实际进度|8epoch评测|
|---|---|---|---|---|---|
|Baseline，U1，2+2|71001/71002，均COMPLETED|71031 FAILED，2:05:04|3dimage-18，8×4090|日志到5550，完整checkpoint到5500|71032 CANCELLED|
|U005，2+2|71009/71010，均COMPLETED|71033 FAILED，0:05:57|3dimage-12，8×3090|加载4172成功，新增更新0|71034 CANCELLED|
|U1，2+4|71012/71013，均COMPLETED|71035 FAILED，0:00:55|3dimage-12，8×3090|冻结源文件前失败，未STARTED|71036 CANCELLED|

4epoch汇总71014 COMPLETED。三个8epoch评测为afterok依赖取消，未执行，没有8epoch正式指标。

### 故障证据

- U005在`train_unified_query_v1.py:242`创建checkpoint_initial硬链接时，明确`OSError: [Errno 28] No space left on device`。
- Views2ctx4在`v4init_extend8_pipeline.py:19`冻结checkpoint_source_epoch4硬链接时，同样Errno28。
- Baseline完成5550更新后，在定期保存checkpoint_latest时，`PytorchStreamWriter failed writing file data/299: file write failed`，随后`unexpected pos 488119232 vs 488119120`。结合相邻作业Errno28及磁盘检查，归因为空间耗尽引发的保存失败；无证据表明本次由LR、梯度爆炸或NaN触发。
- 检查时`df -h /space/mawb`显示/space总量1.5T、使用率100%、可用449M；单份模型+Adam checkpoint约2.344GB，原子替换还需要临时完整文件空间。硬链接自身也需要元数据空间。df -i未提供可用inode统计，不声称已排除inode/配额细分问题。

### 可恢复状态与实体审计

- Baseline完整latest为epoch=5、update_in_epoch=285、full_updates=5500（epoch为零起算，约5.27轮）；Adam604份，step全部5500，8rank RNG齐全，模型632个state条目及Adam张量均有限。epoch5整轮checkpoint也可读，progress=5/0/5215。
- Baseline八rank源恢复收据和首更新同步通过；八rank日志各1378条，4173–5550，loss/PSNR/grad_norm均有限。实际首步主LR1.000001339e-5、几何0.1倍、理解权重1.0。恢复latest需要重算最后未落盘50步；不得把日志5550当作checkpoint进度。
- U005的latest可读，仍为4/0/4172，与本组源模型逐值相同；604份Adam、step4172、8rank RNG齐全。八rank恢复收据存在，尚无首次新增更新。
- Views2ctx4未生成新运行的源冻结checkpoint/STARTED/恢复收据，使用其原4epoch最终checkpoint作为后续恢复来源。
- 三组原4epoch最终checkpoint均为4/0/4172；每份632条模型状态、604份Adam(step4172)、8rank RNG均完整有限。相较共同零步，均604条模型状态变化，其中48条refine_mu/refine_rho及持久gs_tokens有变化；未消费Write0等状态不要求改变。
- 三组原训练八rank日志均各4172条、1–4172，检查loss/PSNR/grad_norm均有限；首更新八rank哈希同步收据通过。

详细实体审计：[v4init_completion_audit_20261023.json](v4init_completion_audit_20261023.json)。不自动恢复STARTED后失败，不从头重训。恢复前应先解决存储容量并为余下checkpoint、缓存和三组评测输出预留空间；本次没有清理、迁移或重提。

## 已完成的4epoch正式结果

固定官方1860窗/312场景，SIU3R commit 8ea80166be76854f938e90521f1a5b688b755c87；target-all6包含context，并非novel-only。三组评测完成收据通过，本次重新读取export记录和GT清单，与旧A4的输入及GT逐窗完全一致，评测更新0。共同零步SHA为bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397。

|组|PSNR|AbsRel|RMSE|ctx mIoU|ctx PQ|ctx mAP|tgt mIoU|tgt PQ|tgt mAP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|U1，2+2|23.052|0.05273|0.17646|58.17|68.42|24.54|57.78|66.06|23.14|
|U0.05，2+2|23.197|0.05239|0.17452|57.26|67.75|23.70|56.98|65.59|22.23|
|U1，2+4|22.978|0.05187|0.17333|57.45|68.71|24.50|57.23|66.32|23.05|

理解指标按百分数展示。

- 降理解权重：PSNR +0.145dB，ctx/tgt mIoU -0.905/-0.807个百分点、PQ -0.665/-0.478、mAP -0.837/-0.908。降低权重得到小幅重建收益，但理解六项均下降，当前证据不支持“1.0明显过大”。隔离梯度后仍存在全局norm clip=1的间接耦合，不能假设权重完全不影响重建。
- 增加训练视角：PSNR -0.073dB，ctx/tgt PQ +0.291/+0.251，mIoU -0.718/-0.557，mAP基本持平略降；深度数值有改善，但该评测的深度有scale+shift拟合，不能直接等同metric几何。单seed不能断言小差值稳定。
- U1、2+2当前综合更均衡，U0.05为PSNR最高；2+4没有明确的整体优势。
- 对同1860评测的旧A4，基线PSNR从18.209升至23.052（+4.843dB），ctx mIoU/PQ/mAP +2.223/+3.066/+3.583，tgt +2.551/+2.917/+4.027。这说明本套隔离+早期V4配方恢复了重建并改善理解，但多个因素同时变化，不能仅据此锁定某一项为唯一原因。之前128窗的“隔离最好16epoch”23.294与此1860窗不是同一口径，不能直接判定胜负。
- 8epoch续训尚未完成，因此无法判断100步LR重升温及继续到8epoch的收益，更无法据此决策16epoch。

## 接手入口

执行快照仍冻结ce8446d102ac1d9a7347c8525ec38935d4922de3，路径/space/mawb/ssst_v4init_extend8_code_20261022_v1。原运行根/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_20261022_v1；失败日志及源状态均保留。旧交接中的PENDING仅为历史提交状态，以本文为当前检查结论；不得据旧状态重提。

已完成本次检查，停止轮询；无新增作业。
