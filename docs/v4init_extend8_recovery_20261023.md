# V4init三组续训恢复（2026-10-23）

用户在空间清理后明确要求“先提交续训作业吧”。恢复原三组到epoch8，并保持此前授权的各自官方1860评测依赖，不扩展实验或epoch16。

旧71031/71033/71035因空间耗尽失败；71032/71034/71036取消。旧运行目录与ce8446d执行快照不修改。新运行根为 `/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_extend8_recovery_20261023_v1`。

|组|理解权重|视角|恢复源|累计起点|还需更新|
|---|---:|---|---|---:|---:|
|Baseline|1.0|2+2|旧71031最后完整latest|5500|2844|
|U005|0.05|2+2|旧71033已初始化的continuation latest|4172|4172|
|Views2ctx4|1.0|2+4|本组原epoch4最终权重|4172|4172|

终点均8344更新/epoch8。Baseline旧日志5501–5550未落盘，会按5500状态精确重算；没有回退到epoch4重新训练。CPU预检下一rank0窗口1041，与旧未保存5501步的记录一致，下一主LR2.437852545768077e-5、几何0.1倍。U005模型与本组epoch4最终模型逐值相同，但保留已经建立的continuation metadata/Adam；源step4172，没有新增更新。

## 训练与恢复约束

模型、L6/L12监督、隔离/NoCA、FM128、query、数据、loss、理解权重、Adam与RNG不变；新的代码只补独立恢复校验、收据和评测门禁。十个关键模型/数据/调度/评分文件与ce8446d快照逐字节相同。

原epoch4→8曲线不重启：在累计4172–4271间从原末端约1e-5升到3e-5，再cosine到最终3e-6；geometry为0.1倍。Baseline直接在累计5500处继续下降，不重新100步升温。理解warmup沿用全局时钟，不重新启动。

恢复入口在全新目录中加载显式SHA冻结源。按原4epoch metadata和实际新runtime重新构造完整continuation identity，仅允许执行code SHA变化；模型keys/shape、Adam名称顺序及moment shape/dtype/finite、step与进度关系、LR时钟、8rank RNG均检查。加载后全部可训练参数哈希必须等于恢复源，八rank各写RECOVERY_LOADED，首更新重新核验同步；Views2ctx4继续使用CONTINUATION_LOADED。恢复组跳过额外initial probes，保持恢复RNG时钟；epoch probes仍保留。

CPU实体预检对三组均验证632状态条目、604份Adam及step、8rank RNG，恢复模型及完整Adam逐值一致，读取下一真实训练batch确认2context及4/6个监督视角。11项CPU测试通过，覆盖5500及4172恢复后的下一步与连续参考完全一致，错误LR/时钟/配置/NaN/RNG/Adam拒绝、学习率不重启、新八rank恢复收据门禁和final latest冻结。CPU检查不等于GPU前反向或DDP已验证。

## 容量与独立任务

新任务使用 `--no-epoch-checkpoints`，保留起点、latest、原子保存临时文件和评测冻结的最终硬链接，不保留epoch5/6/7权重；仍输出各epoch probes和TensorBoard累计步数。预留新增容量预算32GiB（含三组checkpoint峰值和正式导出），准备和提交前要求可用空间超过32GiB。清理后约46GiB可用，其他并行任务空间没有独占预留，不承诺完全免疫外部磁盘消耗。

三条独立链：各恢复train → afterok 各epoch8 eval。没有对已失败71031/71033/71035设置afterok；三项恢复训练可独立调度。每项单节点8卡3090/4090，32CPU/192G；3090/4090分区并排除11节点，当前可用节点池12/13/14/17/18，绝不将五个节点一起作为必选nodelist。

评测固定1860窗/312场景、SIU3R 8ea80166be76854f938e90521f1a5b688b755c87，target-all6含context。新门禁先核验8344完成收据、新恢复八rank收据与首次更新，然后从final latest冻结；原评分和与各自epoch4对比逻辑保持不变。

只提交这3训练+3评测，无额外GPU smoke、summary或8→16。提交后仅一次有界接收核对；记录交接后停止轮询。STARTED后失败不自动重跑。
