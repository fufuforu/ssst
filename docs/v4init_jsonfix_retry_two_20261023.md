# JSON修复后的两组续训重提（2026-10-23）

用户明确授权“修复之后提交续训作业吧”。只重提因metadata JSON类型误判在STARTED前失败的Baseline和U005两条链；现有Views2ctx4训练71188和评测71189不改、不取消、不重提。空间分支71165/71166/71167也不改。

新执行快照：/space/mawb/ssst_v4init_extend8_recovery_code_20261023_v2_jsonfix。
新运行根：/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_recovery_20261023_v2_jsonfix。
旧失败目录及9c9ffcc执行快照保留。

|组|理解权重|训练视角|恢复累计步|终点|剩余更新|
|---|---:|---|---:|---:|---:|
|Baseline|1.0|2+2|5500|8344/8epoch|2844|
|U005|0.05|2+2|4172|8344/8epoch|4172|

源仍为旧ce8446d续训目录中完整checkpoint_latest，SHA分别13232437521ff24eaf7b808f344ea20a0a8be9294423ff743d175ec0689fe70c及7bad9b6390c3d613bac13c167548878947539510702d8459c6487356aac7447f。不是从失败的71184/71186寻找不存在的新checkpoint。

恢复模型、完整Adam moments/step和8rank RNG；原4172→8344的100步升温/余弦曲线按累计时钟继续，不重启理解warmup。理解权重、L6/L12、隔离/NoCA、FM128、2+2、global8和所有模型/loss保持原组。几何LR始终主LR0.1倍。Baseline重算旧未落盘5501–5550，不回退到epoch4。

修复仅恢复JSON序列化造成的optimizer_groups.betas list→tuple类型，保持原数值和其他字段严格校验。实际Adam中的betas也仍受layout校验约束。13项CPU测试通过，其中新增JSON往返用例在原代码复现失败，修复后4172/5500均通过，真实beta变化仍被拒绝。

本轮真实CPU预检从已落盘source_epoch4_meta.json读取，再以实际模型配置和optimizer groups重建runtime metadata，走同一recovery_meta；不再用原metadata对象代替JSON读回。两组632条模型状态、604份Adam、8rank RNG逐值恢复一致，源进度及SHA匹配。真实下一rank0训练窗口分别1041/746，batch均2context、4个监督视角。模型、原调度/数据/评分十个关键文件与ce8446d一致；trainer、launcher和eval gate与当前71188执行快照9c9ffcc逐字节一致，执行修复只在recovery policy。

两个独立train及各自afterok eval，共四个新作业。禁止创建Views2ctx4目录、源审计arms必须恰好Baseline/U005；新训练不依赖已FAILED的71184/71186。每项单节点8卡3090/4090、32CPU/192G；双分区排除11，允许池12/13/14/17/18，不把五个节点作为必选nodelist。Slurm test-only先完成。

保留起点、latest和final评测硬链接，不保留中间epoch权重，epoch probes和累计TensorBoard保留。容量门限28GiB涵盖两条新链checkpoint峰值/正式导出，并为现有2+4后续输出留余量；其他并行写入未独占预留。提交前再次核对可用空间。

评测仍1860窗/312场景、target-all6含context、SIU3R8ea80166be76854f938e90521f1a5b688b755c87、更新0。完成收据8344后从final latest冻结；RECOVERY_LOADED八rank须分别5500/4172并验证首更新同步，再与各自4epoch比较。

CPU通过不代表新GPU加载、前反向或首次同步已验证。提交后仅一次有界调度接收核对，记录交接及授权推送后停止轮询，不等待首步。没有额外GPU smoke、summary或8→16。
