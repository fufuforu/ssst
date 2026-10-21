# I/O 超时诊断、70008 重提与 IoU epoch8 权重核对

用户授权：69254/69481不续训；诊断超时及对未来训练影响；70008工程修复后重新提交。IoU epoch8若无备份则不评测，不能用latest替代。

直接证据：69254在13节点、69481在12节点，均在provider初始化枚举当前scene/depth目录时Errno110；70008在13节点import torch.testing元数据目录时Errno110，torchrun尚未启动训练worker。/space挂载类型为LeoFS；路径既覆盖数据也覆盖Python环境，且横跨两节点。因此直接故障是共享文件系统访问超时，而非梯度路由、LR、CUDA OOM或数值发散。应用traceback无法进一步确认存储服务/网络/元数据负载哪个是底层起因；直接SSH节点检查因publickey认证不通，未改权限或联系他人。

更正此前报告：ObjectLocusV1Provider显式subset=[scene]确实传入dataset；因此每batch枚举的是当前场景depth目录，不是全训练库。反复扫描同一场景仍会制造不必要的元数据请求，但尚不能证明它导致后端超时。

工程修复不宣称修复了LeoFS服务：

- scripts/io_resilience.py：成功的完整有序scene帧列表按完整路径做LRU2048缓存，返回tuple，provider复制为list；不改变帧集合或采样，失败结果不缓存。冻结数据训练期间不修改目录，缓存成立。
- scripts/train_dual_stream_v1.py：单个固定batch对ETIMEDOUT/ESTALE/EIO/ECONNRESET最多3次，等待1/3秒；每次从同一CPU Python/NumPy/Torch seed重新构建，最终恢复调用前RNG。不是跳样本、改变augmentation或重做optimizer step。任何其他错误（缺文件/标签/形状/数值/CUDA）直接失败。
- scripts/run_with_io_bootstrap_retry.py：标准库启动器流式保留日志，只有Errno110 TimeoutError且无STARTED/checkpoint/首步更新收据时，允许最多3次torchrun启动。已有STARTED后绝不自动重新启动训练，不会静默从零重跑；结果记BOOTSTRAP_ATTEMPTS.json。
- 永久服务中断仍可能导致训练失败；有限重试只缓解瞬时I/O故障。没有修改旧快照/旧训练，没有自建后台监控，没有复制Python环境或训练图像到共享盘。

验证：test_io_resilience.py四项覆盖重试次数、永久错误拒绝、缓存完整帧序/不缓存失败、训练开始后拒绝启动重试与真实子进程首启失败后成功；原2+4/收据七项测试复跑。真实CPU审计脚本audit_io_resilience_cpu.py比较未缓存原生glob、缓存、注入已消耗随机数的TimeoutError后重试，检查所有batch Tensor逐值相同以及外部三套CPU RNG保持。具体结果见io_resilience_cpu_audit_20261021.json/log；提交器必须见PASSED收据才允许提交。

新运行目录：`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_2ctx4novel_full4_20261021_v2_iofix_node13_8gpu`，新代码快照`/space/mawb/ssst_point1024_fmquery_2ctx4novel_full4_code_20261021_v2_iofix_node13`。仍8×3090@13，global8，2+4，8337窗，4epoch/4172updates，LR1e-4/warmup100/floor1e-5、LU.05/1043、L6/L12、Point1024/FMquery/残差隔离adapter，原零步hardlink。模型和优化器无改动。来源smoke69992，不拿smoke训练后权重。实际job与唯一有界squeue结果记SUBMITTED.json/STATUS_AFTER_SUBMIT.txt。

IoU权重核对：train16/configuration明确checkpoint_retention=initial+latest；训练runner使用no-epoch-checkpoints。检查正式包train16与smoke、早期iouhead_16ep与iouhead_smoke，各仅initial/latest/untrained及少量probe张量，没有checkpoint_epoch8备份。八个epoch8 probe只是少量渲染/理解输出，不能重建8344步模型。最新checkpoint是10050步，不等于epoch8；按用户新答复暂不评测，不恢复旧训练，也不补训复刻epoch8。之前single16 epoch8与旧A8的1860评分保持原样。
