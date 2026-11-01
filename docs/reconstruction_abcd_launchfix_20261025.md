# ABCD启动参数修复与原定实验恢复

前批smoke71693–71696均FAILED/2:0，耗时26–41秒。日志一致为：`run.py: error: ambiguous option: --run could match --run-path, --run_path`。torchrun解析即退出，未启动worker，四组smoke/train目录无模型checkpoint，正式训练71697/71700/71703/71706及导出、评分、汇总均因依赖失败取消，实际新增更新0。

根因是启动脚本未隔开torchrun与任务参数，且任务参数`--run`与launcher选项前缀冲突。独立v2快照把任务参数改为`--run-root`（保留Python内部dest=run），并在smoke/train/export的torchrun flags后、任务脚本前加入显式`--`。

用已安装torch2.7的真实get_args_parser解析shell文件中的实际argv，覆盖smoke/train/export，并复现旧参数失败。模型、loss、初始化、Adam迁移、数据、步数与资源约束均未改变，沿用cbae1b1上已通过的14项科学测试及完整四组CPU迁移审计；本次另测launcher而不重复低价值整套测试。原v1执行快照/失败日志保留，v2独立run及代码SHA重新归档。

仍是四组smoke全部通过→各组训练→导出→Native/纯Novel评分→统一汇总。ABC各追加1epoch（助手默认），D零训4epoch；不得把本次启动重试当额外实验或声称GPU验证已通过。新提交后一次有界接收核对即停止，等用户通知再检查。
