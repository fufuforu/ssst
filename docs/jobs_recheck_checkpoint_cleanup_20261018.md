# 作业复核及近期 checkpoint 清理：2026-10-18

用户明确授权检查已完成作业，并核对引用后删除无用 checkpoint。本轮没有提交或重跑训练、推理、评分。

## 作业与结果

单次 Slurm 核对：本人无运行或排队作业。最新已提交实验仍为 A 续训66640/评测66641–42/汇总66643、Unified续训66648/评测66649–50/汇总66651、串行身份残差66653，均 COMPLETED/0:0。新代码入口 `dual_bounded` 与 `unified_ray_bias` 未提交训练，没有新版结果可汇报。

本轮重新检查三组训练完成收据、20份 rank 日志：A/Unified 各8rank新增4172步，4173→8344连续；残差4rank各256步，1→256连续，loss/grad norm均有限。四份正式评测的 checkpoint identity 与 epoch6/8对应；同128个场景、帧及RGB SHA，13列中11项有限值、两项文本为空，评测参数更新0。checkpoint路径按训练节点 `/datapool/...` 与控制节点 `/space/...` 的同一实验目录映射核实。

| 模型 | PSNR | AbsRel ↓ | Target mAP | Target PQ |
|---|---:|---:|---:|---:|
| 旧 VGGT+MASt3R C | 22.5966 | 0.08730 | 27.3756% | 60.9790% |
| A4 | 18.5003 | 0.18980 | 22.7075% | 62.7599% |
| A8 | 19.3692 | 0.16636 | 24.8446% | 61.5404% |
| Unified4 | 18.0855 | 0.08880 | 16.5192% | 45.0114% |
| Unified8 | 18.4225 | 0.08470 | 15.5542% | 48.2612% |

完整精度与11指标以[此前完成分析](continuation_and_serial_residual_results_20261018.md)和原生CSV为准。A追加训练提高重建和AP，但PQ未同步提高；Unified追加训练未改善AP。串行残差验证最终最佳IoU约0.0404→0.0386，Recall@0.5仍0，保留原报告的重复前向告警，不将其称为最终Mask改善。

## 引用审计与删除范围

范围限定为 `unified_query_training_v1` 中近期 A/B/Unified/串行实验，未清理旧C、旧GFM模型、官方Omega/SAM权重及其他研究项目。

删除前枚举46个训练根目录checkpoint文件，按device/inode去重为38份独立内容。38份内容全部经CPU `torch.load(mmap=True)`加载，并检查ZIP目录，记录46个路径的进度、架构、模型张量数、优化器状态数及meta；这不等于逐字节重新验证每个权重或ZIP数据块CRC。

引用审计包括近期训练目录已完成的文本扫描、项目docs/scripts、历史group_plus最多3层的JSON/JSONL/Markdown/Python/shell/YAML、46个checkpoint的内部meta，以及正式评测identity。识别 `/datapool`、`/space` 和相对路径，检查是否作为续训/零步初始化/推理/评测来源。删除清单与文件名相同的历史命中均指向其他实验，没有目标文件的依赖引用。无关大型历史图片导出的全树扫描已停止；本结论限于核对过的本地记录，不声称检查过外部私人笔记。

| 运行目录 | 已删除 checkpoint | 数量 |
|---|---|---:|
| attempt02_esam_many_full_node11_8gpu | epoch1、2、3 | 3 |
| attempt05_legacy_dual_superpoint_full_node13_8gpu | epoch1、2、3 | 3 |
| attempt06_unified_direct_masks_full_node18_8gpu | epoch1、2、3 | 3 |
| attempt08_legacy_dual_superpoint_extend8_node13_8gpu | epoch5、7 | 2 |
| attempt09_unified_query_extend8_node11_8gpu | epoch5、7 | 2 |

以上均为 `checkpoint_epochN.pt`，各文件nlink=1，训练已完成且最终checkpoint有效，不属于正式评测端点。所有短测终点、初始/准备零步状态、A/Unified的epoch4/6/8、B的epoch4、source别名及最终latest均保留。保留的33个路径在删除前后逐一核实device/inode/大小/mtime/nlink一致；日志、评测、可视化和源码快照保留。

实际删除13文件，逻辑大小 **13,042,111,149字节 = 12.1464GiB**，分配空间13,042,107,904字节。文件系统可用空间观察到增加13,042,102,272字节；共享文件系统数值可能受其他活动影响。未采用移动到同盘回收站来冒充释放空间。这些中间轮次的精确状态已不可直接恢复；完整训练终点与初始化仍可使用。

## 审计收据

目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/maintenance_20261018/`

- `completed_jobs_recheck.json`：本轮训练日志与正式评测复核。
- `checkpoint_root_inventory_before.json`、`checkpoint_metadata_verified.json`：删除前文件/硬链接和内部元数据。
- `checkpoint_basename_reference_hits.txt`：有界引用搜索命中，保留原行。
- `checkpoint_cleanup_plan.json`：13项删除计划、33项保留清单、删除理由、元数据pickle SHA256及文件指纹。
- `checkpoint_cleanup_receipt.json`：逐文件删除收据、保留项验证及空间统计，状态COMPLETE。
- `audit_recent_checkpoint_retention.py`：本轮审计/执行程序归档。只应用显式计划，不是未来任务的自动清理服务。

后续不要把已删中间epoch当作新的续训源；也不要把本轮复核结果误标为两个尚未训练的新架构的结果。
