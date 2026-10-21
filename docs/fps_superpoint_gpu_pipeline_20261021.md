# FPS1024 + embedding + 超点特征：GPU smoke → 条件提交16epoch

用户授权：不加入颜色先验；先GPU smoke，通过后提交八卡单次16epoch。
代码基于5209c0c，模型仅新增日志维度读数；FPS初始化、理解分支及颜色头与CPU测试通过的模型逐字节一致。

独立运行目录：
- smoke：`/space/mawb/ssst/group_plus/semantic_resolution_v1/fps1024_superpoint_smoke_20261021_v1`
- full：`/space/mawb/ssst/group_plus/semantic_resolution_v1/fps1024_superpoint_single16_20261021_v1`

资源均单节点八卡，3090/4090分区，只允许13/14/17/18。Smoke 90分钟时限，正式训练48小时时限；时限不代表预计耗时。

smoke使用与正式训练相同的16epoch/16688步cosine轨迹，但max-updates=2。训练前从原始A4真零步迁移：693张量严格保留，仅新增embedding；源权重用硬链接，无重复大文件复制。理解256维、L6/L12、global8、LR1e-4、理解权重.05/warmup1043、Write=0，无IoU头/Q+P bank/颜色先验。

两步完成后单进程执行真实单窗export，严格校验8rank各2步、有限loss/梯度、1024anchor/65536GS/理解256维、embedding梯度非零且权重变化、首步8rank参数哈希一致、所有旧初始张量逐值一致、导出成功及checkpoint代码身份，然后写SMOKE_PASSED。

只有verifier成功，smoke作业才调用submit_fps_superpoint_after_smoke.py提交正式八卡作业。子作业另有afterok:smoke依赖，确保整个父作业成功后才调度；启动时再次核对通过收据与代码身份。独占SUBMISSION_REQUESTED标记避免重复提交；提交后仅一次15秒内有界squeue接收核对，无重试/轮询。失败不提交。

正式训练从同一原始零步重新开始，并非从两步checkpoint继续；完整16epoch/16688更新。保留初始与latest，禁用每epoch完整checkpoint，仍做原initial/epoch/final小探针和TensorBoard。未自行增加正式评测作业。

入口：scripts/run_fps_superpoint.sh、scripts/verify_fps_superpoint_smoke.py、scripts/submit_fps_superpoint_after_smoke.py。
关键记录：PREFLIGHT.json、SUBMITTED.json、FPS_SUPERPOINT_INITIALIZATION.json、FIRST_GRADIENTS_rank*.json、FIRST_UPDATE_VERIFIED.json、LIMIT_REACHED.json、SMOKE_PASSED.json；失败见JOB_FAILED.json；正式训练作业号在full/SUBMITTED.json，完成见TRAINING_COMPLETE.json。

通过/拒绝与防重提交2项CPU测试通过，原模型9项CPU测试沿用；提交前再次执行真实core零步CPU审计。所有CPU审计均不等于GPU验证。
提交smoke后主会话仅核对一次状态即停止；不守候smoke/训练结果。用户通知后再检查，不从历史PENDING/RUNNING重提。
