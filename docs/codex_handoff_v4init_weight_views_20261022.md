# 交接：V4init隔离版的理解权重 / 2+4两组消融（2026-10-22）

遵守 `/space/mawb/AGENTS.md`。**新组已经提交，不重复提交，不恢复旧取消作业。提交后唯一一次状态核对已完成，等用户通知完成再查。** 本交接不含实时进度；未为写文档再次查队列或训练日志。

## 最新授权

用户明确要求在“现在的版本”基础上新增两组：①调整理解权重；②训练视角改2+4。用户进一步确认第一组采用**0.05**。两组分别相对已有基线单轴变化，不把它们合并，也没有新增其它实验组。

|实验|理解loss基础权重|训练视角|关键变化|
|---|---:|---|---|
|已有基线71001|1.0|2context+2novel|不重提、不改动|
|U005|0.05|2context+2novel|仅理解权重|
|Views2ctx4|1.0|2context+4novel|仅重建训练视角|

共同：完整V4重建初始化1024Anchor/65536GS，原持久基底+7D内容MLP+rawGFM2048视觉投影；去Anchor CA、保留FM CA；显式双向梯度隔离，Write0、无adapter；当前A4投影点云→超点理解query、FM128、来源多对一支持匹配.5，理解监督仅context；L6/L12监督[1/3,2/3]；主LR1e-4、refine_mu/refine_rho1e-5、warmup100/cosine乘数floor.1；理解权重均warmup1043；AdamW及全局clip1保持基线，范数裁剪仍可能跨任务耦合。

8337窗/1191场景、256²、global8=8×1、4epoch4172更新、原采样顺序及8rank RNG。三组从同一V4init零步checkpoint_initial.pt出发，SHA `bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397`，空Adam、updates0；不接着已训练权重续训。

## 执行封存与目录

- 执行快照：`/space/mawb/ssst_v4init_weight_views_code_20261022_v1`
- 执行commit：`ef98a6f9d8ffdba3c39c5cb4660eb65726bd1f2e`；不要改HEAD或运行源码。
- 运行根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_weight_views_20261022_v1`
- 两组独立子目录U005、Views2ctx4，各含smoke/train/eval。
- publication worktree：`/space/mawb/ssst_v4init_weight_views_publication_20261022_v1`
- GitHub分支：`experiments/v4init-isolated-weight-views-20261022`
- 规格：`docs/v4init_isolated_weight_views_20261022.md`；提交/CPU记录：`docs/v4init_weight_views_records_20261022/`。

## 已提交作业及唯一状态核对

北京时间2026-10-22 **18:03:58** 提交；该次核对所有新作业被调度接受，均PENDING，尚未分配节点：

|实验|单卡2步smoke|8卡完整训练|8卡正式1860评测|
|---|---:|---:|---:|
|U005|71008|71009，afterok:71008|71010，afterok:71009|
|Views2ctx4|71011|71012，afterok:71011|71013，afterok:71012|

smoke原因None，训练/评测原因Dependency。每组独立链，不互相等待，也不需要等71001才能开始。正式训练从共同零步重载，不继承smoke权重/Adam。

**CPU汇总71014**：afterok依赖**71002、71010、71013**，该次PENDING(Dependency)。只申请4CPU/16G，不申请GPU；核验三组正式结果与输入/GT/预算/source权重SHA，生成对基线CSV、逐类PQ及报告。汇总不启动训练、推理或重复评测。若基线失败，汇总被依赖阻断，但两组新评测可独立完成。

GPU阶段节点池12/13/14/17/18，3090/4090；单节点smoke1卡，训练/评测8卡，不要求同时占五台。GPU执行、smoke通过与首步更新均**未核实**。不可将PENDING或后续RUNNING当GPU验证。

耗时只参考之前V4init 8×3090训练6h50、评测1h04；低权重组粗估6–8h，2+4新增渲染/冻结几何处理，可能更久，未测本组步时；正式1860每组约1–1.5h，排队另计。

## 2+4数据和提交前验证

- 复用已有冻结2+4清单，独立复制到运行根`training_windows_2ctx4novel.json`；SHA和全8337父窗一一关系已验证。上下文和原novel不变，只增加原context区间内两个不同novel，没有重采样context。
- loader协议沿用`point1024_2context_4novel_v1`，这是历史清单协议名，不代表新模型换回P1024iso架构。
- 真实CPU首训练窗：2+2帧[122,174,142,156]；2+4帧[122,174,142,156,160,166]。原四视角RGB、语义/实例GT与mask逐值保留，生成输入images_input逐值相同。
- 实际batch：images_all分别[1,4,3,256,256]/[1,6,3,256,256]；images_output分别2/4个novel。生成仅两context，RGB重建监督4/6视角；理解仍仅context，新增novel GT不进入理解loss。
- 同4172更新意味着窗口曝光33376不变，重建视角曝光133504→200256；不是同计算量实验，loss沿原方式按视角取均值。
- 每组632张量逐值保留、删除32个CA张量、无新增，195961456可训练参数；48个mu/rho头张量使用geometry LR；各拒绝8类非法零步迁移。
- 与基线8个核心模型/loss/decoder/epoch策略/窗口扩展源文件逐字节相同，记录在BASELINE_SOURCE_IDENTITY.json。只增加模型身份及训练参数/数据协议接入。
- **9项CPU测试11.211秒通过**：真实core前向一致和双向隔离、实际六视角RGB监督、context-only理解GT、理解权重缩放（clip前重建梯度不变）、LR比、单轴身份、窗口和batch契约；CPU不代表GPU。
- Python编译、shell语法、diff检查、真实源与数据审计、独立目录、scheduler test-only及代码封存在提交前完成。

## 完成后检查

用户说“跑完了/检查一下”后再查71008–71014。不要凭本交接历史PENDING重提：

1. 各组smoke/SMOKE_PASSED.json、DIRECTIONAL_GRADIENTS.json、STEP_RESULTS.json、BATCH_SHAPES.json。真实256²/1024/65536/对应2+2或2+4/global8累积；实际geometry/main LR、理解warmup、双向零泄漏、9初始化梯度均有门禁。
2. 各组train/FIRST_UPDATE_VERIFIED.json、V4INIT_ZERO_INITIALIZATION.json、TRAINING_COMPLETE.json，进度epoch4/update_in_epoch0/full_updates4172；8rank日志和最终权重更新/有限性。
3. 各组eval/EVALUATION_VERIFIED.json、FINAL_CHECKPOINT_FROZEN.json、evaluation/metrics.json、checkpoint_identity.json。固定1860窗/312场景，target-all6含context，与旧A4输入/GT一致，评测更新0。
4. 总comparison/SUMMARY_VERIFIED.json、REPORT.md、vs_isolated_baseline.csv、per_class_pq.csv。三组比较：U005 vs baseline检验理解权重；Views2ctx4 vs baseline检验视角。不能据此解释两变量组合的交互，单seed不等于稳定增益。
5. 失败看JOB_FAILED.json/SUMMARY_FAILED.json和对应Slurm日志，不静默重训。I/O bootstrap只允许STARTED前明确文件系统超时，STARTED后无自动重训。

SIU3R固定commit `8ea80166be76854f938e90521f1a5b688b755c87`，不改SIU3R/src。

## 基线与历史边界

本轮没有查询71000–71002的当前进度。它们的最后已知状态和完整配方见[codex_handoff_v4init_isolated_noca_20261022.md](codex_handoff_v4init_isolated_noca_20261022.md)，其中17:44状态仅为历史。基线执行快照仍310d49d，不能改为本轮新SHA。

更早70714–70719已完成且核验，不重提。V4init joint相对旧A4正式1860重建无收益、Target PQ -6点；旧A4 NoCA joint Target PQ +.712点但mIoU/mAP略降。旧C/D取消组不恢复。旧attempt16的23.294是128窗，与本轮1860不得直接混减。

已停止轮询，等用户通知完成后再检查。


## 2026-10-22 21:13:04 后续：三组已提交4→8续训

Baseline 71001→71031→71032，U005 71009→71033→71034，Views2ctx4 71012→71035→71036。只新增三个续训和三个8epoch评测，原任务不改。唯一核对新六项均PENDING(Dependency)。最新交接见[codex_handoff_v4init_extend8_20261022.md](codex_handoff_v4init_extend8_20261022.md)。不要重提，已停止轮询。
